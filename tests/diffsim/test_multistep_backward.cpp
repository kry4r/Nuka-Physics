// ---------------------------------------------------------------------------
// p02-B/C -- multi-step differentiable rollout: tape + checkpoint + recompute +
// backward runner. Validates against the p02-A single-step spine + central-FD.
// ---------------------------------------------------------------------------
//
// Acceptance gates (each isolates one property):
//   1. REPLAY BIT-EXACT (R2/D1): forward N steps recording per-step state;
//      restore checkpoint 0 + replay to N; EXPECT_EQ (bit-exact) the state.
//      Validated on a FIXED-base small chain AND on go2_float (contact-free,
//      near-fixed-orientation trajectory).
//   2. CHECKPOINT-vs-FULL-TAPE: backward with K=10 vs full-tape (a checkpoint
//      every step). The grad_actions + grad_mass are BIT-IDENTICAL by
//      construction (shared inner reverse loop + bit-exact replay). Reported as
//      max diff; any nonzero is a determinism bug, not a tolerance.
//   3. MULTI-STEP GRAD vs FD: fixed-base rollout (p02-A spine exact);
//      grad_action[k] + grad_mass vs central-difference on a scalar loss
//      (< 1e-3 rel away from clamps).
//   4. D1 TWO-RUN (R6): the whole backward twice -> bit-exact grads.
//   5. MEMORY/COST: a 1000-step rollout with K=100 fits single-GPU; reports the
//      reverse/forward time ratio.
//
// CONTACT-FREE: the differentiable forward is the contact-free PD step the p02-A
// adjoint reverses (explicit damping; drive -> ABA -> integrate; NO contact
// solve). The fixed-base chain is where the p02-A grad spine is EXACT; go2_float
// is used only for replay-determinism (not grad-through-base-reorientation,
// which is deferred from p02-A).
// ---------------------------------------------------------------------------

#include "diffsim/backward_runner.hpp"
#include "diffsim/checkpoint.hpp"
#include "diffsim/recompute_orchestrator.hpp"
#include "diffsim/step_backward.hpp"
#include "diffsim/tape.hpp"

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooked_blob.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace diffsim = nuka::diffsim;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// ---------------------------------------------------------------------------
// A fixed-base 3-link revolute chain (mirrors the p02-A FD test's model: nonzero
// COM offsets / damping / armature / q so every adjoint path is exercised).
// ---------------------------------------------------------------------------
struct ChainModel {
    articulation::ArticulationHostState host;
    std::vector<diffsim::LinkMassParams> mass_params;
    std::vector<uint32_t> dof_links;
    std::vector<float> stiffness, damping, force_limits;
    uint32_t link_count = 0u;
};

ChainModel BuildFixedChain() {
    articulation::ArticulationCookedTopology topo;
    topo.root_body = 0u;
    topo.link_bodies = {0u, 1u, 2u, 3u};
    topo.parent_links = {kInvalidLink, 0u, 1u, 2u};
    topo.joint_types = {articulation::ArticulationJointType::Fixed,
                        articulation::ArticulationJointType::Revolute,
                        articulation::ArticulationJointType::Revolute,
                        articulation::ArticulationJointType::Revolute};
    topo.joint_axes = {Vec3::UnitZ(), Vec3::UnitZ(), Vec3::UnitY(), Vec3::UnitX()};
    topo.parent_frames = {Transform::Identity(),
                          Transform{Vec3{0.30f, 0.0f, 0.0f}, Quat::Identity()},
                          Transform{Vec3{0.25f, 0.0f, 0.0f}, Quat::Identity()},
                          Transform{Vec3{0.20f, 0.0f, 0.0f}, Quat::Identity()}};
    topo.child_frames = {Transform::Identity(), Transform::Identity(),
                         Transform::Identity(), Transform::Identity()};
    topo.local_poses = {Transform::Identity(), Transform::Identity(),
                        Transform::Identity(), Transform::Identity()};
    topo.inertial_frames = {
        Transform{Vec3{0.05f, 0.02f, -0.01f}, Quat::Identity()},
        Transform{Vec3{0.12f, -0.03f, 0.04f}, Quat::Identity()},
        Transform{Vec3{0.10f, 0.05f, -0.02f}, Quat::Identity()},
        Transform{Vec3{0.08f, -0.04f, 0.03f}, Quat::Identity()}};
    topo.initial_positions = {0.0f, 0.30f, -0.45f, 0.20f};
    topo.joint_dampings = {0.0f, 0.15f, 0.10f, 0.08f};
    topo.joint_armatures = {0.0f, 0.03f, 0.02f, 0.015f};
    const std::vector<float> masses = {3.0f, 1.2f, 0.8f, 0.5f};
    const std::vector<Vec3> inertias = {
        Vec3{0.06f, 0.05f, 0.04f}, Vec3{0.012f, 0.010f, 0.008f},
        Vec3{0.008f, 0.007f, 0.005f}, Vec3{0.005f, 0.004f, 0.003f}};
    topo.masses = masses;
    topo.inertias = inertias;

    nuka::scene::CookedBodyTable bodies;
    bodies.poses = {Transform::Identity(), Transform::Identity(),
                    Transform::Identity(), Transform::Identity()};
    bodies.local_poses = bodies.poses;
    bodies.inertial_frames = topo.inertial_frames;
    bodies.masses = masses;
    bodies.inertias = inertias;
    bodies.is_static = {1u, 0u, 0u, 0u};

    ChainModel m;
    m.host = articulation::BuildArticulationHostState({topo}, bodies);
    m.link_count = m.host.TotalLinkCount();
    for (uint32_t i = 0u; i < m.link_count; ++i) {
        diffsim::LinkMassParams lp;
        lp.mass = masses[i];
        lp.diagonal_inertia = inertias[i];
        lp.inertial_frame = topo.inertial_frames[i];
        m.mass_params.push_back(lp);
    }
    m.dof_links = {1u, 2u, 3u};
    // PD gains: nonzero on the DOFs, wide force limits (unsaturated).
    m.stiffness = {0.0f, 25.0f, 20.0f, 15.0f};
    m.damping = {0.0f, 1.5f, 1.2f, 0.9f};
    m.force_limits = {0.0f, 200.0f, 200.0f, 200.0f};
    return m;
}

// ---------------------------------------------------------------------------
// A FLOATING-base chain (6-DOF root + 2 revolute links). Mirrors the p02-A FD
// test's floating model. Used for the multi-step floating grad-vs-FD test that
// validates the cross-step grad_link_velocity carry + the v_root_pre snapshot.
// ---------------------------------------------------------------------------
ChainModel BuildFloatingChain() {
    articulation::ArticulationCookedTopology topo;
    topo.root_body = 0u;
    topo.link_bodies = {0u, 1u, 2u};
    topo.parent_links = {kInvalidLink, 0u, 1u};
    topo.joint_types = {articulation::ArticulationJointType::FloatingBase,
                        articulation::ArticulationJointType::Revolute,
                        articulation::ArticulationJointType::Revolute};
    topo.joint_axes = {Vec3::UnitZ(), Vec3::UnitX(), Vec3::UnitY()};
    topo.parent_frames = {Transform::Identity(),
                          Transform{Vec3{0.30f, 0.0f, 0.0f}, Quat::Identity()},
                          Transform{Vec3{0.25f, 0.0f, 0.0f}, Quat::Identity()}};
    topo.child_frames = {Transform::Identity(), Transform::Identity(),
                         Transform::Identity()};
    topo.local_poses = {Transform::Identity(), Transform::Identity(),
                        Transform::Identity()};
    topo.inertial_frames = {
        Transform{Vec3{0.04f, 0.03f, -0.02f}, Quat::Identity()},
        Transform{Vec3{0.12f, -0.03f, 0.04f}, Quat::Identity()},
        Transform{Vec3{0.10f, 0.05f, -0.02f}, Quat::Identity()}};
    topo.initial_positions = {0.0f, 0.30f, -0.45f};
    topo.joint_dampings = {0.0f, 0.15f, 0.10f};
    topo.joint_armatures = {0.0f, 0.03f, 0.02f};
    const std::vector<float> masses = {4.0f, 1.2f, 0.8f};
    const std::vector<Vec3> inertias = {
        Vec3{0.08f, 0.07f, 0.05f}, Vec3{0.012f, 0.010f, 0.008f},
        Vec3{0.008f, 0.007f, 0.005f}};
    topo.masses = masses;
    topo.inertias = inertias;

    nuka::scene::CookedBodyTable bodies;
    bodies.poses = {Transform::Identity(), Transform::Identity(), Transform::Identity()};
    bodies.local_poses = bodies.poses;
    bodies.inertial_frames = topo.inertial_frames;
    bodies.masses = masses;
    bodies.inertias = inertias;
    bodies.is_static = {0u, 0u, 0u};  // movable root -> FloatingBase

    ChainModel m;
    m.host = articulation::BuildArticulationHostState({topo}, bodies);
    m.link_count = m.host.TotalLinkCount();
    for (uint32_t i = 0u; i < m.link_count; ++i) {
        diffsim::LinkMassParams lp;
        lp.mass = masses[i];
        lp.diagonal_inertia = inertias[i];
        lp.inertial_frame = topo.inertial_frames[i];
        m.mass_params.push_back(lp);
    }
    m.dof_links = {1u, 2u};  // revolute DOFs (root is FloatingBase)
    m.stiffness = {0.0f, 20.0f, 15.0f};
    m.damping = {0.0f, 1.2f, 0.9f};
    m.force_limits = {0.0f, 200.0f, 200.0f};
    return m;
}

// ---------------------------------------------------------------------------
// Device harness for a contact-free differentiable rollout on a small chain.
// Owns the device state + orchestrator + tape + checkpoint manager + runner.
// ---------------------------------------------------------------------------
class RolloutHarness {
public:
    RolloutHarness(const nuka::phi::DeviceContext& ctx, const ChainModel& model,
                   float dt, float gravity_z, const diffsim::TapeDesc& desc)
        : ctx_(ctx), model_(model) {
        device_ = articulation::UploadArticulationState(ctx_, model.host);
        n_ = model.link_count;
        diffsim::RolloutParams rp;
        rp.dt = dt;
        rp.gravity_z = gravity_z;
        orch_ = std::make_unique<diffsim::RecomputeOrchestrator>(
            ctx_, device_.View(), rp, model.stiffness, model.damping,
            model.force_limits, model.mass_params);
        tape_ = std::make_unique<diffsim::Tape>(ctx_, desc, n_);
        cm_ = std::make_unique<diffsim::CheckpointManager>(
            ctx_, desc.max_checkpoints, device_.View().articulation_count, n_,
            /*lambda_width=*/0u);
        runner_ = std::make_unique<diffsim::BackwardRunner>(ctx_, *orch_);
        desc_ = desc;
    }

    uint32_t N() const { return n_; }
    diffsim::Tape& Tape() { return *tape_; }
    diffsim::CheckpointManager& Checkpoints() { return *cm_; }
    diffsim::BackwardRunner& Runner() { return *runner_; }
    diffsim::RecomputeOrchestrator& Orch() { return *orch_; }
    articulation::ArticulationDeviceState State() { return device_.View(); }

    void SetState(const std::vector<float>& q, const std::vector<float>& qd) {
        Upload(State().q, q);
        Upload(State().qdot, qd);
    }
    std::vector<float> GetQ() { return Download(State().q, n_); }
    std::vector<float> GetQdot() { return Download(State().qdot, n_); }

    // Floating-base root spatial velocity (6 floats at link_velocity[0]).
    void SetRootVel(const std::array<float, 6>& v) {
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        cudaMemcpy(State().link_velocity, v.data(), 6u * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
    std::array<float, 6> GetRootVel() {
        std::array<float, 6> v{};
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        cudaMemcpy(v.data(), State().link_velocity, 6u * sizeof(float),
                   cudaMemcpyDeviceToHost);
        return v;
    }

    // Drive a recorded rollout: at each step, upload the step's action into a
    // device buffer, capture a checkpoint on the K-cadence, record, StepOnce.
    void Record(const std::vector<std::vector<float>>& actions) {
        tape_->Reset();
        cm_->Reset();
        steps_since_cp_ = 0u;
        action_buf_ = nuka::phi::Buffer(n_ * sizeof(float),
                                        nuka::phi::MemoryKind::Device);
        const bool full_tape = (desc_.recompute_on_backward == 0u);
        for (uint32_t s = 0u; s < actions.size(); ++s) {
            Upload(static_cast<float*>(action_buf_.Data()), actions[s]);
            const bool cap = full_tape || (s == 0u) ||
                             (steps_since_cp_ >= desc_.checkpoint_interval);
            uint32_t slot = 0u, has = 0u;
            if (cap) {
                slot = cm_->Capture(State(), s, nullptr);
                has = 1u;
                steps_since_cp_ = 0u;
            }
            tape_->RecordStep(static_cast<const float*>(action_buf_.Data()), has,
                              slot);
            orch_->StepOnce(static_cast<const float*>(action_buf_.Data()));
            steps_since_cp_ += 1u;
        }
        Sync();
    }

    // Snapshot the authoritative state (base_pose / link_velocity / q / qdot).
    struct StateSnapshot {
        std::vector<Transform> base_pose;
        std::vector<articulation::LinkSpatialVel> link_velocity;
        std::vector<float> q, qdot;
    };
    StateSnapshot Snapshot() {
        StateSnapshot s;
        const auto st = State();
        s.base_pose.resize(st.articulation_count);
        s.link_velocity.resize(n_);
        s.q.resize(n_);
        s.qdot.resize(n_);
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        cudaMemcpy(s.base_pose.data(), st.base_pose,
                   st.articulation_count * sizeof(Transform),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(s.link_velocity.data(), st.link_velocity,
                   n_ * sizeof(articulation::LinkSpatialVel),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(s.q.data(), st.q, n_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(s.qdot.data(), st.qdot, n_ * sizeof(float),
                   cudaMemcpyDeviceToHost);
        return s;
    }

    // Restore checkpoint 0 and replay forward to `to_step` (executes steps 0..to_step-1).
    void ReplayFromCheckpoint0(uint32_t to_step) {
        cm_->Restore(0u, State(), nullptr);
        for (uint32_t s = 0u; s < to_step; ++s) {
            orch_->StepOnce(tape_->ActionSlice(s));
        }
        Sync();
    }

    nuka::phi::Buffer MakeOutputActions(uint32_t step_count) {
        nuka::phi::Buffer b(static_cast<size_t>(step_count) * n_ * sizeof(float),
                            nuka::phi::MemoryKind::Device);
        return b;
    }
    nuka::phi::Buffer MakeOutputMass() {
        return nuka::phi::Buffer(n_ * sizeof(float), nuka::phi::MemoryKind::Device);
    }

    void Sync() { ctx_.stream.Synchronize(); }

    void Upload(float* dst, const std::vector<float>& v) {
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        cudaMemcpy(dst, v.data(), v.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
    std::vector<float> Download(const float* src, uint32_t count) {
        std::vector<float> v(count, 0.0f);
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        cudaMemcpy(v.data(), src, count * sizeof(float), cudaMemcpyDeviceToHost);
        return v;
    }
    std::vector<float> DownloadBuf(const nuka::phi::Buffer& b) {
        std::vector<float> v(b.Size() / sizeof(float), 0.0f);
        b.CopyToHost(v.data(), b.Size());
        return v;
    }

private:
    const nuka::phi::DeviceContext& ctx_;
    const ChainModel& model_;
    articulation::ArticulationDeviceBuffers device_;
    uint32_t n_ = 0u;
    diffsim::TapeDesc desc_;
    uint32_t steps_since_cp_ = 0u;
    nuka::phi::Buffer action_buf_;
    std::unique_ptr<diffsim::RecomputeOrchestrator> orch_;
    std::unique_ptr<diffsim::Tape> tape_;
    std::unique_ptr<diffsim::CheckpointManager> cm_;
    std::unique_ptr<diffsim::BackwardRunner> runner_;
};

// A mild per-step action sequence around the rest pose (unsaturated, contact-free).
std::vector<std::vector<float>> MildActions(uint32_t n_steps, uint32_t n_links,
                                            const std::vector<uint32_t>& dofs,
                                            const std::vector<float>& base) {
    std::vector<std::vector<float>> acts(n_steps, std::vector<float>(n_links, 0.0f));
    for (uint32_t s = 0u; s < n_steps; ++s) {
        std::vector<float> a(n_links, 0.0f);
        const float phase = 0.05f * static_cast<float>(s);
        for (size_t i = 0u; i < dofs.size(); ++i) {
            const uint32_t k = dofs[i];
            a[k] = base[k] + 0.10f * std::sin(phase + 0.7f * static_cast<float>(i));
        }
        acts[s] = a;
    }
    return acts;
}

bool BitEqual(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0u; i < a.size(); ++i) {
        uint32_t x, y;
        std::memcpy(&x, &a[i], 4);
        std::memcpy(&y, &b[i], 4);
        if (x != y) return false;
    }
    return true;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float w = 0.0f;
    for (size_t i = 0u; i < a.size(); ++i)
        w = std::max(w, std::fabs(a[i] - b[i]));
    return w;
}

}  // namespace

// ===========================================================================
// 1. REPLAY BIT-EXACT (fixed base). Forward N steps; snapshot; restore ckpt 0;
//    replay to N; the authoritative state is bit-exact.
// ===========================================================================
TEST(MultiStepBackward, ReplayBitExact_FixedBase) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    const float dt = 0.005f, g = -9.81f;
    const uint32_t N = 50u;
    diffsim::TapeDesc desc;
    desc.checkpoint_interval = 10u;
    desc.max_tape_entries = 256u;
    desc.max_checkpoints = 64u;
    desc.recompute_on_backward = 1u;

    RolloutHarness h(ctx, model, dt, g, desc);
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f, 0.20f};
    const std::vector<float> qd0(model.link_count, 0.0f);
    h.SetState(q0, qd0);
    const auto actions = MildActions(N, model.link_count, model.dof_links,
                                     model.stiffness.empty() ? q0 : q0);
    h.Record(actions);
    const auto orig = h.Snapshot();

    // Restore + replay to N.
    h.SetState(q0, qd0);  // (replay overwrites anyway; reset for cleanliness)
    h.ReplayFromCheckpoint0(N);
    const auto replayed = h.Snapshot();

    EXPECT_TRUE(BitEqual(orig.q, replayed.q)) << "q not bit-exact after replay";
    EXPECT_TRUE(BitEqual(orig.qdot, replayed.qdot))
        << "qdot not bit-exact after replay";
    std::vector<float> obp, rbp, ov, rv;
    for (const auto& t : orig.base_pose) {
        obp.insert(obp.end(), {t.position.x, t.position.y, t.position.z,
                               t.rotation.x, t.rotation.y, t.rotation.z,
                               t.rotation.w});
    }
    for (const auto& t : replayed.base_pose) {
        rbp.insert(rbp.end(), {t.position.x, t.position.y, t.position.z,
                               t.rotation.x, t.rotation.y, t.rotation.z,
                               t.rotation.w});
    }
    for (const auto& v : orig.link_velocity)
        ov.insert(ov.end(), v.v, v.v + 6);
    for (const auto& v : replayed.link_velocity)
        rv.insert(rv.end(), v.v, v.v + 6);
    EXPECT_TRUE(BitEqual(obp, rbp)) << "base_pose not bit-exact";
    EXPECT_TRUE(BitEqual(ov, rv)) << "link_velocity not bit-exact";
    printf("[replay fixed-base] N=%u state bit-exact after restore+replay\n", N);
}

// ===========================================================================
// 2. CHECKPOINT-vs-FULL-TAPE backward agreement. Same rollout, two reverse
//    passes: K=10 (recompute) and full-tape (a checkpoint every step). The
//    grad_actions + grad_mass are BIT-IDENTICAL.
// ===========================================================================
TEST(MultiStepBackward, CheckpointVsFullTape_BitIdentical) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    const float dt = 0.005f, g = -9.81f;
    const uint32_t N = 40u;
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f, 0.20f};
    const std::vector<float> qd0(model.link_count, 0.0f);
    const auto actions = MildActions(N, model.link_count, model.dof_links, q0);

    auto run = [&](uint32_t recompute) {
        diffsim::TapeDesc desc;
        desc.checkpoint_interval = 10u;
        desc.max_tape_entries = 256u;
        desc.max_checkpoints = N + 4u;
        desc.recompute_on_backward = recompute;
        RolloutHarness h(ctx, model, dt, g, desc);
        h.SetState(q0, qd0);
        h.Record(actions);
        auto ga = h.MakeOutputActions(N);
        auto gm = h.MakeOutputMass();
        diffsim::BackwardSeeds seeds;  // null seeds; we seed via grad_q/grad_qdot
        // Seed a nontrivial loss on the final state.
        std::vector<float> sq(model.link_count, 0.0f), sqd(model.link_count, 0.0f);
        for (uint32_t k : model.dof_links) { sq[k] = 0.6f; sqd[k] = -0.3f; }
        nuka::phi::Buffer bsq(model.link_count * sizeof(float),
                              nuka::phi::MemoryKind::Device);
        nuka::phi::Buffer bsqd(model.link_count * sizeof(float),
                               nuka::phi::MemoryKind::Device);
        bsq.CopyFromHost(sq.data(), sq.size() * sizeof(float));
        bsqd.CopyFromHost(sqd.data(), sqd.size() * sizeof(float));
        seeds.grad_q_final = static_cast<const float*>(bsq.Data());
        seeds.grad_qdot_final = static_cast<const float*>(bsqd.Data());
        diffsim::BackwardOutputs out;
        out.grad_actions = static_cast<float*>(ga.Data());
        out.grad_mass = static_cast<float*>(gm.Data());
        h.Runner().Run(h.Tape(), h.Checkpoints(), seeds, out);
        std::vector<float> result = h.DownloadBuf(ga);
        const std::vector<float> mass = h.DownloadBuf(gm);
        result.insert(result.end(), mass.begin(), mass.end());
        return result;
    };

    const std::vector<float> ckpt = run(/*recompute=*/1u);
    const std::vector<float> full = run(/*recompute=*/0u);
    ASSERT_EQ(ckpt.size(), full.size());
    const float max_diff = MaxAbsDiff(ckpt, full);
    printf("[ckpt vs full-tape] N=%u max |diff| = %.3e (bit-identical=%d)\n", N,
           max_diff, BitEqual(ckpt, full) ? 1 : 0);
    EXPECT_TRUE(BitEqual(ckpt, full))
        << "checkpoint and full-tape grads must be bit-identical; max diff="
        << max_diff;
}

// ===========================================================================
// 3. MULTI-STEP GRAD vs FD (fixed base). The checkpointed backward's
//    grad_action[k] and grad_mass vs central-difference on a scalar loss over a
//    short rollout. The p02-A spine is exact for the fixed base.
// ===========================================================================
TEST(MultiStepBackward, MultiStepGradVsFD_FixedBase) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    const float dt = 0.005f, g = -9.81f;
    const uint32_t N = 6u;  // short horizon -> FD truncation stays clean
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f, 0.20f};
    const std::vector<float> qd0(model.link_count, 0.0f);
    const auto& dofs = model.dof_links;

    // Fixed (step-independent) action so FD on action[k] of one step is clean.
    std::vector<std::vector<float>> actions(N, std::vector<float>(model.link_count, 0.0f));
    const std::vector<float> base_action = {0.0f, 0.45f, -0.30f, 0.35f};
    for (uint32_t s = 0u; s < N; ++s) actions[s] = base_action;

    // Scalar loss on the FINAL state: L = sum_k wq*q'_k + wqd*qdot'_k.
    std::vector<float> wq(model.link_count, 0.0f), wqd(model.link_count, 0.0f);
    wq[1] = 0.6f; wq[2] = -0.9f; wq[3] = 0.4f;
    wqd[1] = -0.3f; wqd[2] = 0.7f; wqd[3] = -0.5f;

    diffsim::TapeDesc desc;
    desc.checkpoint_interval = 3u;  // exercise the recompute path (K < N)
    desc.max_tape_entries = 64u;
    desc.max_checkpoints = 32u;
    desc.recompute_on_backward = 1u;

    // -- Analytic grads via the backward runner. --
    RolloutHarness h(ctx, model, dt, g, desc);
    h.SetState(q0, qd0);
    h.Record(actions);
    auto ga = h.MakeOutputActions(N);
    auto gm = h.MakeOutputMass();
    nuka::phi::Buffer bsq(model.link_count * sizeof(float), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer bsqd(model.link_count * sizeof(float), nuka::phi::MemoryKind::Device);
    bsq.CopyFromHost(wq.data(), wq.size() * sizeof(float));
    bsqd.CopyFromHost(wqd.data(), wqd.size() * sizeof(float));
    diffsim::BackwardSeeds seeds;
    seeds.grad_q_final = static_cast<const float*>(bsq.Data());
    seeds.grad_qdot_final = static_cast<const float*>(bsqd.Data());
    diffsim::BackwardOutputs out;
    out.grad_actions = static_cast<float*>(ga.Data());
    out.grad_mass = static_cast<float*>(gm.Data());
    h.Runner().Run(h.Tape(), h.Checkpoints(), seeds, out);
    const std::vector<float> grad_actions = h.DownloadBuf(ga);  // [N*n]
    const std::vector<float> grad_mass = h.DownloadBuf(gm);     // [n]

    // -- Central FD: a pure forward roll + loss, with perturbations. The mass
    //    perturbation must change the ACTUAL forward physics, i.e. the uploaded
    //    spatial inertia state.link_inertia[link] (the orchestrator's mass_params
    //    only build dI/dmass). So perturb the host's link_inertia via the SAME
    //    MakeSpatialInertia representation dI/dmass slopes (mass <-> spatial
    //    inertia consistency, exactly as the p02-A FD test's SetLinkMass).
    auto forward_loss = [&](std::vector<std::vector<float>> acts,
                            int mass_link, float mass_delta) {
        ChainModel m = model;
        if (mass_link >= 0) {
            const diffsim::LinkMassParams& lp = model.mass_params[mass_link];
            m.host.link_inertia[mass_link] = articulation::MakeSpatialInertia(
                lp.mass + mass_delta, lp.diagonal_inertia, lp.inertial_frame);
        }
        diffsim::TapeDesc d2 = desc;
        d2.recompute_on_backward = 0u;  // irrelevant; we only forward-roll
        RolloutHarness hh(ctx, m, dt, g, d2);
        hh.SetState(q0, qd0);
        // Forward roll via StepOnce on uploaded actions.
        nuka::phi::Buffer ab(model.link_count * sizeof(float),
                             nuka::phi::MemoryKind::Device);
        for (uint32_t s = 0u; s < N; ++s) {
            hh.Upload(static_cast<float*>(ab.Data()), acts[s]);
            hh.Orch().StepOnce(static_cast<const float*>(ab.Data()));
        }
        hh.Sync();
        const std::vector<float> q = hh.GetQ();
        const std::vector<float> qd = hh.GetQdot();
        float L = 0.0f;
        for (uint32_t k : dofs) L += wq[k] * q[k] + wqd[k] * qd[k];
        return L;
    };

    const float eps = 1e-3f;
    // grad_action[k] for each step s, each DOF k.
    float worst_action_rel = 0.0f;
    for (uint32_t s = 0u; s < N; ++s) {
        for (uint32_t k : dofs) {
            auto perturb = [&](float d) {
                auto acts = actions;
                acts[s][k] += d;
                return forward_loss(acts, -1, 0.0f);
            };
            const float fd = (perturb(eps) - perturb(-eps)) / (2.0f * eps);
            const float an = grad_actions[s * model.link_count + k];
            const float denom = std::max(1e-3f, std::fabs(an) + std::fabs(fd));
            const float rel = std::fabs(an - fd) / denom;
            worst_action_rel = std::max(worst_action_rel, rel);
        }
    }
    // grad_mass for each link. Combined absolute-OR-relative metric: a component
    // passes within abs_tol OR rel_tol (the mass gradient here is small-magnitude
    // -- the loss is weighted on q'/qdot', not directly on mass -- so FD
    // truncation noise on a near-zero gradient must not be flagged as a relative
    // miss; same convention as the p02-A floating mass check).
    const float mass_abs_tol = 5e-3f, mass_rel_tol = 5e-3f;
    float worst_mass_score = 0.0f;
    for (uint32_t l = 0u; l < model.link_count; ++l) {
        auto perturb = [&](float d) { return forward_loss(actions, (int)l, d); };
        const float fd = (perturb(eps) - perturb(-eps)) / (2.0f * eps);
        const float an = grad_mass[l];
        const float abs_err = std::fabs(an - fd);
        const float denom = std::max(1e-6f, std::fabs(an) + std::fabs(fd));
        const float rel_err = abs_err / denom;
        const float score = std::min(abs_err / mass_abs_tol, rel_err / mass_rel_tol);
        worst_mass_score = std::max(worst_mass_score, score);
        printf("  [mass l=%u] analytic=%.5e fd=%.5e abs_err=%.2e rel_err=%.2e\n",
               l, an, fd, abs_err, rel_err);
    }
    printf("[multistep grad vs FD] N=%u worst_action_rel=%.3e worst_mass_score=%.3e\n",
           N, worst_action_rel, worst_mass_score);
    EXPECT_LT(worst_action_rel, 1e-3f);
    EXPECT_LT(worst_mass_score, 1.0f);  // < 1 == every link within abs OR rel tol
}

// ===========================================================================
// 3b. MULTI-STEP GRAD vs FD (FLOATING base). Validates the cross-step
//     grad_link_velocity carry + the per-step v_root_pre snapshot over a real
//     multi-step rollout: grad_action[k] + grad_mass + d/d(v_root_0) vs
//     central-difference on a scalar loss over the joint qdot' AND the root
//     spatial velocity v_root'.
//
//     gravity_z = 0 (CRITICAL): the base-orientation channel (a_grav =
//     R(base_rot)^T g + the quaternion pose integrator) is DEFERRED from p02-A
//     and NOT differentiated. With gravity on, base tilt would make
//     d(a_grav)/d(base_rot) a real forward-physics term the analytic grad omits
//     -> a phantom FD mismatch unrelated to this fix. With g=0, a_grav == 0
//     identically, so the deferred channel contributes nothing and the rollout
//     isolates the velocity channel (gyroscopic ForceCross(v, I*v), the LDLT
//     solve, and the v_root_pre linearization point). A nonzero INITIAL root
//     angular velocity exercises the gyroscopic term across every step.
// ===========================================================================
TEST(MultiStepBackward, MultiStepGradVsFD_Floating) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFloatingChain();
    const float dt = 0.01f, g = 0.0f;  // g=0 -> deferred a_grav channel is inert
    const uint32_t N = 5u;             // short horizon -> FD truncation stays clean
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f};
    const std::vector<float> qd0(model.link_count, 0.0f);
    // Nonzero root spatial velocity, AMPLIFIED angular part (first 3 comps) so the
    // root gyroscopic bias is exercised each step.
    const std::array<float, 6> rootv0 = {1.5f, -1.2f, 1.8f, 0.20f, -0.15f, 0.12f};
    const auto& dofs = model.dof_links;

    // Fixed (step-independent) action so FD on action[k] of one step is clean.
    std::vector<std::vector<float>> actions(N, std::vector<float>(model.link_count, 0.0f));
    const std::vector<float> base_action = {0.0f, 0.40f, -0.30f};
    for (uint32_t s = 0u; s < N; ++s) actions[s] = base_action;

    // Scalar loss on the FINAL state: joints qdot' + root spatial velocity v_root'.
    std::vector<float> wqd(model.link_count, 0.0f);
    wqd[1] = -0.3f; wqd[2] = 0.7f;
    const std::array<float, 6> wrv = {0.3f, -0.4f, 0.6f, -0.2f, 0.5f, -0.35f};

    diffsim::TapeDesc desc;
    desc.checkpoint_interval = 2u;  // exercise the recompute path (K < N)
    desc.max_tape_entries = 64u;
    desc.max_checkpoints = 32u;
    desc.recompute_on_backward = 1u;

    // -- Analytic grads via the backward runner. --
    RolloutHarness h(ctx, model, dt, g, desc);
    h.SetState(q0, qd0);
    h.SetRootVel(rootv0);
    h.Record(actions);
    auto ga = h.MakeOutputActions(N);
    auto gm = h.MakeOutputMass();
    nuka::phi::Buffer bsqd(model.link_count * sizeof(float), nuka::phi::MemoryKind::Device);
    bsqd.CopyFromHost(wqd.data(), wqd.size() * sizeof(float));
    // grad_link_velocity_final: seed wrv on the root 6-vector (zero elsewhere).
    std::vector<float> glv(model.link_count * 6u, 0.0f);
    for (uint32_t i = 0u; i < 6u; ++i) glv[i] = wrv[i];
    nuka::phi::Buffer bglv(glv.size() * sizeof(float), nuka::phi::MemoryKind::Device);
    bglv.CopyFromHost(glv.data(), glv.size() * sizeof(float));
    diffsim::BackwardSeeds seeds;
    seeds.grad_qdot_final = static_cast<const float*>(bsqd.Data());
    seeds.grad_link_velocity_final = static_cast<const float*>(bglv.Data());
    diffsim::BackwardOutputs out;
    out.grad_actions = static_cast<float*>(ga.Data());
    out.grad_mass = static_cast<float*>(gm.Data());
    h.Runner().Run(h.Tape(), h.Checkpoints(), seeds, out);
    const std::vector<float> grad_actions = h.DownloadBuf(ga);  // [N*n]
    const std::vector<float> grad_mass = h.DownloadBuf(gm);     // [n]

    // -- Central FD: a pure forward roll + loss, with perturbations on action,
    //    mass, and the initial root velocity. --
    auto forward_loss = [&](const std::vector<std::vector<float>>& acts,
                            int mass_link, float mass_delta,
                            int rootv_comp, float rootv_delta) {
        ChainModel m = model;
        if (mass_link >= 0) {
            const diffsim::LinkMassParams& lp = model.mass_params[mass_link];
            m.host.link_inertia[mass_link] = articulation::MakeSpatialInertia(
                lp.mass + mass_delta, lp.diagonal_inertia, lp.inertial_frame);
        }
        diffsim::TapeDesc d2 = desc;
        d2.recompute_on_backward = 0u;
        RolloutHarness hh(ctx, m, dt, g, d2);
        hh.SetState(q0, qd0);
        std::array<float, 6> rv = rootv0;
        if (rootv_comp >= 0) rv[rootv_comp] += rootv_delta;
        hh.SetRootVel(rv);
        nuka::phi::Buffer ab(model.link_count * sizeof(float),
                             nuka::phi::MemoryKind::Device);
        for (uint32_t s = 0u; s < N; ++s) {
            hh.Upload(static_cast<float*>(ab.Data()), acts[s]);
            hh.Orch().StepOnce(static_cast<const float*>(ab.Data()));
        }
        hh.Sync();
        const std::vector<float> qd = hh.GetQdot();
        const std::array<float, 6> rvf = hh.GetRootVel();
        float L = 0.0f;
        for (uint32_t k : dofs) L += wqd[k] * qd[k];
        for (uint32_t i = 0u; i < 6u; ++i) L += wrv[i] * rvf[i];
        return L;
    };

    const float eps = 1e-3f;
    // grad_action[k] for each step s, each DOF k.
    float worst_action_rel = 0.0f;
    for (uint32_t s = 0u; s < N; ++s) {
        for (uint32_t k : dofs) {
            auto perturb = [&](float d) {
                auto acts = actions;
                acts[s][k] += d;
                return forward_loss(acts, -1, 0.0f, -1, 0.0f);
            };
            const float fd = (perturb(eps) - perturb(-eps)) / (2.0f * eps);
            const float an = grad_actions[s * model.link_count + k];
            const float denom = std::max(1e-3f, std::fabs(an) + std::fabs(fd));
            const float rel = std::fabs(an - fd) / denom;
            worst_action_rel = std::max(worst_action_rel, rel);
        }
    }
    // grad_mass for each link (small-magnitude -> abs-OR-rel, same as fixed-base).
    // NOTE: grad_action + grad_mass are seeded with grad_link_velocity_final=wrv on
    // the root and the FD loss includes v_root', so they flow back THROUGH every
    // step's v_root_pre snapshot + the cross-step grad_link_velocity carry. A wrong
    // snapshot at any step would show here -> this IS the floating carry validation
    // (the dedicated linearization-point GUARD is the large-dt twin-variant test).
    const float mass_abs_tol = 5e-3f, mass_rel_tol = 1e-2f;
    float worst_mass_score = 0.0f;
    for (uint32_t l = 0u; l < model.link_count; ++l) {
        auto perturb = [&](float d) { return forward_loss(actions, (int)l, d, -1, 0.0f); };
        const float fd = (perturb(eps) - perturb(-eps)) / (2.0f * eps);
        const float an = grad_mass[l];
        const float abs_err = std::fabs(an - fd);
        const float denom = std::max(1e-6f, std::fabs(an) + std::fabs(fd));
        const float rel_err = abs_err / denom;
        const float score = std::min(abs_err / mass_abs_tol, rel_err / mass_rel_tol);
        worst_mass_score = std::max(worst_mass_score, score);
        printf("  [float mass l=%u] analytic=%.5e fd=%.5e abs=%.2e rel=%.2e\n",
               l, an, fd, abs_err, rel_err);
    }
    printf("[multistep grad vs FD FLOATING] N=%u worst_action_rel=%.3e "
           "worst_mass_score=%.3e\n",
           N, worst_action_rel, worst_mass_score);
    EXPECT_LT(worst_action_rel, 3e-3f);
    EXPECT_LT(worst_mass_score, 1.0f);
}

// ===========================================================================
// 4. D1 TWO-RUN: the whole backward twice -> bit-exact grad_actions + grad_mass.
// ===========================================================================
TEST(MultiStepBackward, D1_TwoRunBitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    const float dt = 0.005f, g = -9.81f;
    const uint32_t N = 30u;
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f, 0.20f};
    const std::vector<float> qd0(model.link_count, 0.0f);
    const auto actions = MildActions(N, model.link_count, model.dof_links, q0);

    diffsim::TapeDesc desc;
    desc.checkpoint_interval = 7u;
    desc.max_tape_entries = 64u;
    desc.max_checkpoints = 32u;
    desc.recompute_on_backward = 1u;

    auto run = [&]() {
        RolloutHarness h(ctx, model, dt, g, desc);
        h.SetState(q0, qd0);
        h.Record(actions);
        auto ga = h.MakeOutputActions(N);
        auto gm = h.MakeOutputMass();
        std::vector<float> wq(model.link_count, 0.0f), wqd(model.link_count, 0.0f);
        for (uint32_t k : model.dof_links) { wq[k] = 0.5f; wqd[k] = -0.4f; }
        nuka::phi::Buffer bsq(model.link_count * sizeof(float), nuka::phi::MemoryKind::Device);
        nuka::phi::Buffer bsqd(model.link_count * sizeof(float), nuka::phi::MemoryKind::Device);
        bsq.CopyFromHost(wq.data(), wq.size() * sizeof(float));
        bsqd.CopyFromHost(wqd.data(), wqd.size() * sizeof(float));
        diffsim::BackwardSeeds seeds;
        seeds.grad_q_final = static_cast<const float*>(bsq.Data());
        seeds.grad_qdot_final = static_cast<const float*>(bsqd.Data());
        diffsim::BackwardOutputs out;
        out.grad_actions = static_cast<float*>(ga.Data());
        out.grad_mass = static_cast<float*>(gm.Data());
        h.Runner().Run(h.Tape(), h.Checkpoints(), seeds, out);
        std::vector<float> r = h.DownloadBuf(ga);
        const std::vector<float> m = h.DownloadBuf(gm);
        r.insert(r.end(), m.begin(), m.end());
        return r;
    };
    const std::vector<float> r1 = run();
    const std::vector<float> r2 = run();
    EXPECT_TRUE(BitEqual(r1, r2)) << "backward not bit-exact across two runs";
    printf("[D1 two-run] N=%u backward bit-exact across two runs\n", N);
}

// ===========================================================================
// 5. MEMORY/COST: 1000-step rollout with K=100 fits single-GPU; report the
//    reverse/forward time ratio.
// ===========================================================================
TEST(MultiStepBackward, MemoryCost_1000Step_K100) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    const float dt = 0.002f, g = -9.81f;  // benign dt so the explicit-damping roll is stable
    const uint32_t N = 1000u;
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f, 0.20f};
    const std::vector<float> qd0(model.link_count, 0.0f);
    // Small actions so the chain stays near rest (contact-free, no blow-up).
    std::vector<std::vector<float>> actions(N, std::vector<float>(model.link_count, 0.0f));
    for (uint32_t s = 0u; s < N; ++s) {
        for (uint32_t k : model.dof_links)
            actions[s][k] = q0[k] + 0.05f * std::sin(0.01f * static_cast<float>(s));
    }

    diffsim::TapeDesc desc;
    desc.checkpoint_interval = 100u;
    desc.max_tape_entries = N + 4u;
    desc.max_checkpoints = (N / 100u) + 8u;
    desc.recompute_on_backward = 1u;

    RolloutHarness h(ctx, model, dt, g, desc);
    h.SetState(q0, qd0);

    auto t0 = std::chrono::high_resolution_clock::now();
    h.Record(actions);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto ga = h.MakeOutputActions(N);
    auto gm = h.MakeOutputMass();
    std::vector<float> wq(model.link_count, 0.0f);
    for (uint32_t k : model.dof_links) wq[k] = 0.3f;
    nuka::phi::Buffer bsq(model.link_count * sizeof(float), nuka::phi::MemoryKind::Device);
    bsq.CopyFromHost(wq.data(), wq.size() * sizeof(float));
    diffsim::BackwardSeeds seeds;
    seeds.grad_q_final = static_cast<const float*>(bsq.Data());
    diffsim::BackwardOutputs out;
    out.grad_actions = static_cast<float*>(ga.Data());
    out.grad_mass = static_cast<float*>(gm.Data());

    auto t2 = std::chrono::high_resolution_clock::now();
    h.Runner().Run(h.Tape(), h.Checkpoints(), seeds, out);
    auto t3 = std::chrono::high_resolution_clock::now();

    const double fwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double rev_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const std::vector<float> grad_mass = h.DownloadBuf(gm);
    // Sanity: grads are finite (the rollout did not NaN).
    bool finite = true;
    for (float v : grad_mass) finite = finite && std::isfinite(v);
    const std::vector<float> grad_actions = h.DownloadBuf(ga);
    for (float v : grad_actions) finite = finite && std::isfinite(v);
    EXPECT_TRUE(finite) << "1000-step backward produced non-finite grads";
    printf("[1000-step K=100] forward=%.1f ms, reverse=%.1f ms, reverse/forward=%.2fx, "
           "checkpoints=%u (finite=%d)\n",
           fwd_ms, rev_ms, rev_ms / std::max(1e-3, fwd_ms),
           h.Checkpoints().Count(), finite ? 1 : 0);
}

// ===========================================================================
// 1b. REPLAY BIT-EXACT on go2_float (contact-free, near-fixed-orientation). Only
//     the replay-DETERMINISM is asserted (NOT grad through base reorientation,
//     which is deferred from p02-A).
// ===========================================================================
TEST(MultiStepBackward, ReplayBitExact_Go2Float) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    articulation::ArticulationHostState host =
        articulation::BuildArticulationHostState(world.template_view.articulations,
                                                 world.template_view.body_table);
    const uint32_t n = host.TotalLinkCount();
    ASSERT_GT(n, 0u);

    // Build mass params from the host topology (same order as the engine).
    std::vector<diffsim::LinkMassParams> mass_params;
    for (const auto& topo : host.articulations) {
        for (uint32_t local = 0u; local < topo.link_bodies.size(); ++local) {
            diffsim::LinkMassParams lp;
            lp.mass = local < topo.masses.size() ? topo.masses[local] : 0.0f;
            lp.diagonal_inertia = local < topo.inertias.size()
                                      ? topo.inertias[local]
                                      : Vec3{0.0f, 0.0f, 0.0f};
            lp.inertial_frame = local < topo.inertial_frames.size()
                                    ? topo.inertial_frames[local]
                                    : Transform::Identity();
            mass_params.push_back(lp);
        }
    }
    // PD gains: moderate stiffness on the actuated revolute joints, hold the rest
    // pose (drive_target = q). Near-fixed-orientation: gravity OFF so the base
    // does not reorient over the horizon (replay determinism only).
    std::vector<float> stiffness(n, 0.0f), damping(n, 0.0f), force_limits(n, 0.0f);
    for (uint32_t l = 0u; l < n; ++l) {
        if (host.joint_type[l] == articulation::ArticulationJointType::Revolute) {
            stiffness[l] = 40.0f;
            damping[l] = 2.0f;
            force_limits[l] = 24.0f;
        }
    }

    auto device = articulation::UploadArticulationState(ctx, host);

    diffsim::RolloutParams rp;
    rp.dt = 0.005f;
    // Gravity ON: the base free-falls (StepOnce has NO contact pipeline, so the
    // rollout stays contact-free regardless of ground) -> nonzero base_pose
    // translation + growing link_velocity, so the checkpoint capture/restore of
    // those floating-base buffers is GENUINELY exercised (a static rollout would
    // hold them ~0 and pass even if the float-buffer restore were broken). Short
    // horizon (40 * 0.005s) -> near-fixed base ORIENTATION (deferred from p02-A),
    // no NaN.
    rp.gravity_z = -9.81f;
    diffsim::TapeDesc desc;
    desc.checkpoint_interval = 8u;
    desc.max_tape_entries = 128u;
    desc.max_checkpoints = 32u;
    desc.recompute_on_backward = 1u;

    diffsim::RecomputeOrchestrator orch(ctx, device.View(), rp, stiffness, damping,
                                        force_limits, mass_params);
    diffsim::Tape tape(ctx, desc, n);
    diffsim::CheckpointManager cm(ctx, desc.max_checkpoints,
                                  device.View().articulation_count, n, 0u);

    const uint32_t N = 40u;
    // Action = hold the rest pose (drive_target = q0).
    std::vector<float> hold = host.q;
    nuka::phi::Buffer ab(n * sizeof(float), nuka::phi::MemoryKind::Device);
    ab.CopyFromHost(hold.data(), n * sizeof(float));

    uint32_t since = 0u;
    for (uint32_t s = 0u; s < N; ++s) {
        uint32_t slot = 0u, has = 0u;
        if (s == 0u || since >= desc.checkpoint_interval) {
            slot = cm.Capture(orch.State(), s, nullptr);
            has = 1u;
            since = 0u;
        }
        tape.RecordStep(static_cast<const float*>(ab.Data()), has, slot);
        orch.StepOnce(static_cast<const float*>(ab.Data()));
        since += 1u;
    }
    ctx.stream.Synchronize();

    auto snap = [&]() {
        std::vector<float> q(n), qd(n);
        std::vector<articulation::LinkSpatialVel> lv(n);
        nuka::phi::ScopedDeviceGuard guard(ctx.device_id);
        cudaMemcpy(q.data(), orch.State().q, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(qd.data(), orch.State().qdot, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(lv.data(), orch.State().link_velocity,
                   n * sizeof(articulation::LinkSpatialVel), cudaMemcpyDeviceToHost);
        std::vector<float> out = q;
        out.insert(out.end(), qd.begin(), qd.end());
        for (const auto& v : lv) out.insert(out.end(), v.v, v.v + 6);
        return out;
    };
    const std::vector<float> orig = snap();

    // Restore checkpoint 0, replay to N.
    cm.Restore(0u, orch.State(), nullptr);
    for (uint32_t s = 0u; s < N; ++s)
        orch.StepOnce(tape.ActionSlice(s));
    ctx.stream.Synchronize();
    const std::vector<float> replayed = snap();

    EXPECT_TRUE(BitEqual(orig, replayed))
        << "go2_float replay not bit-exact (max diff="
        << MaxAbsDiff(orig, replayed) << ")";
    printf("[replay go2_float] N=%u links=%u state bit-exact after restore+replay "
           "(max diff=%.3e)\n",
           N, n, MaxAbsDiff(orig, replayed));
}
