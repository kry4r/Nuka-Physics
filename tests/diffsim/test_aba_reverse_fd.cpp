// ---------------------------------------------------------------------------
// p02-A -- FD-validated single-step ABA reverse (contact-free PD spine)
// ---------------------------------------------------------------------------
//
// Validates nuka::diffsim::StepBackward, the hand-written reverse-mode adjoint of
// ONE contact-free step (drive -> ABA -> integrate), against central-difference
// finite differences on the REAL engine forward kernels, plus an EXACT M^-1
// cross-check (no FD truncation noise) and a D1 two-run bit-exact check.
//
// The FD ladder (each rung gates the next, to localize bugs):
//   (a) Fixed-base ABA  d(qddot)/d(tau, qdot)            reverse vs central-FD
//       + EXACT M^-1 cross-check  d(qddot)/d(tau) = M^-1 (forward oracle, ~1e-6)
//       + d(qddot)/d(q) X-path (q-channel)               reverse vs central-FD
//   (b) + mass          d(qddot)/d(mass)                 reverse vs central-FD
//   (d) + integration   d(state')/d(tau, qdot)           reverse vs central-FD
//   (e) + drive         d(loss(state'))/d(action), d/d(mass)  reverse vs central-FD
//
// Test model: a FIXED-base 3-link revolute chain with NONZERO COM offsets (so
// dI/dmass exercises the full parallel-axis + COM coupling -- the system-ID bug a
// COM-at-origin model cannot catch), NONZERO joint damping (d jf/d qdot), NONZERO
// armature (the D diagonal), and NONZERO qdot (the gyroscopic / velocity-bias
// path). gravity is on for the FD rungs and OFF only for the M^-1 isolation.
// ---------------------------------------------------------------------------

#include "diffsim/step_backward.hpp"

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "scene/cooked_blob.hpp"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace diffsim = nuka::diffsim;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;

// Test-only RAII device buffer over the phi v2 opaque Buffer*. Mirrors the legacy
// phi::Buffer surface (Data/Size/CopyFromHost/CopyToHost, move-only) so the test
// bodies stay byte-identical after the buffer sweep. Allocated from the
// device-level DEFAULT (stream-0) type; CopyFromHost/CopyToHost run on stream 0
// exactly like the legacy synchronous cudaMemcpy this replaces.
class OwnedDeviceBuffer {
public:
    OwnedDeviceBuffer() = default;
    explicit OwnedDeviceBuffer(size_t bytes) : bytes_(bytes) {
        buf_ = nuka::phi::BufferAlloc(
            nuka::phi::DeviceBufferType(nuka::phi::InitBestDevice()), bytes);
    }
    ~OwnedDeviceBuffer() { if (buf_ != nullptr) nuka::phi::BufferFree(buf_); }
    OwnedDeviceBuffer(OwnedDeviceBuffer&& o) noexcept
        : buf_(o.buf_), bytes_(o.bytes_) { o.buf_ = nullptr; o.bytes_ = 0u; }
    OwnedDeviceBuffer& operator=(OwnedDeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) nuka::phi::BufferFree(buf_);
            buf_ = o.buf_; bytes_ = o.bytes_; o.buf_ = nullptr; o.bytes_ = 0u;
        }
        return *this;
    }
    OwnedDeviceBuffer(const OwnedDeviceBuffer&) = delete;
    OwnedDeviceBuffer& operator=(const OwnedDeviceBuffer&) = delete;
    void* Data() { return buf_ != nullptr ? nuka::phi::BufferBase(buf_) : nullptr; }
    const void* Data() const { return buf_ != nullptr ? nuka::phi::BufferBase(buf_) : nullptr; }
    size_t Size() const { return bytes_; }
    void CopyFromHost(const void* src, size_t bytes) {
        nuka::phi::BufferUpload(buf_, src, 0, bytes);
    }
    void CopyToHost(void* dst, size_t bytes) const {
        nuka::phi::BufferDownload(buf_, dst, 0, bytes);
    }
private:
    nuka::phi::Buffer* buf_ = nullptr;
    size_t bytes_ = 0u;
};

// ---------------------------------------------------------------------------
// A fixed-base 3-link revolute chain. Link 0 is a FIXED root (anchor, qddot=0);
// links 1,2,3 are revolute DOFs. Nonzero COM offsets / damping / armature so the
// FD ladder exercises every adjoint path (see file header).
// ---------------------------------------------------------------------------
struct ChainModel {
    articulation::ArticulationHostState host;
    std::vector<diffsim::LinkMassParams> mass_params;  // per global link
    std::vector<uint32_t> dof_links;                   // links with a real DOF
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
    // NONZERO COM offsets (inertial_frame.position) -> dI/dmass has the full
    // parallel-axis + off-diagonal coupling, not just the lower-right block.
    topo.inertial_frames = {
        Transform{Vec3{0.05f, 0.02f, -0.01f}, Quat::Identity()},
        Transform{Vec3{0.12f, -0.03f, 0.04f}, Quat::Identity()},
        Transform{Vec3{0.10f, 0.05f, -0.02f}, Quat::Identity()},
        Transform{Vec3{0.08f, -0.04f, 0.03f}, Quat::Identity()}};
    topo.initial_positions = {0.0f, 0.30f, -0.45f, 0.20f};   // nonzero q
    topo.joint_dampings = {0.0f, 0.15f, 0.10f, 0.08f};       // nonzero Kd_joint
    topo.joint_armatures = {0.0f, 0.03f, 0.02f, 0.015f};     // nonzero armature
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
    bodies.is_static = {1u, 0u, 0u, 0u};  // root static -> Fixed root

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
    m.dof_links = {1u, 2u, 3u};  // revolute DOFs (root link 0 is Fixed)
    return m;
}

// ---------------------------------------------------------------------------
// A FLOATING-base model: 6-DOF root + 2 revolute links, nonzero COM offsets so
// dI/dmass exercises the full coupling on the base too. Mirrors the fixed chain's
// physical-data layout (rung c).
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
    m.dof_links = {1u, 2u};  // revolute DOFs (root is FloatingBase, no scalar DOF)
    return m;
}

// ---------------------------------------------------------------------------
// Device harness: holds the uploaded buffers + scratch grad buffers, runs the
// forward composition and the reverse, with helpers to set/get per-link arrays.
// ---------------------------------------------------------------------------
class Harness {
public:
    Harness(const nuka::phi::DeviceContext& ctx, const ChainModel& model)
        : ctx_(ctx), model_(model) {
        device_ = articulation::UploadArticulationState(ctx_, model.host);
        n_ = model.link_count;
        // dI/dmass (host -> device).
        const std::vector<float> dIdm =
            diffsim::BuildSpatialInertiaMassJacobian(model.mass_params);
        dI_dmass_ = MakeDeviceFloat(dIdm);
        // grad scratch buffers (total_link_count each / *6 for link velocity).
        grad_q_ = MakeDeviceFloat(std::vector<float>(n_, 0.0f));
        grad_qdot_ = MakeDeviceFloat(std::vector<float>(n_, 0.0f));
        grad_target_ = MakeDeviceFloat(std::vector<float>(n_, 0.0f));
        grad_mass_ = MakeDeviceFloat(std::vector<float>(n_, 0.0f));
        grad_tau_ = MakeDeviceFloat(std::vector<float>(n_, 0.0f));
        grad_qddot_seed_ = MakeDeviceFloat(std::vector<float>(n_, 0.0f));
        grad_link_vel_ = MakeDeviceFloat(std::vector<float>(n_ * 6u, 0.0f));
        q_pre_ = MakeDeviceFloat(std::vector<float>(n_, 0.0f));
        qdot_pre_ = MakeDeviceFloat(std::vector<float>(n_, 0.0f));
        v_root_pre_ = MakeDeviceFloat(std::vector<float>(n_ * 6u, 0.0f));
        // p08b orientation channel: per-articulation base-pose snapshot + grad.
        art_count_ = model.host.articulation_link_offset.size();
        base_pose_pre_ = OwnedDeviceBuffer(art_count_ * sizeof(Transform));
        // grad_base_pose_out: 7 floats per articulation (3 pos + 4 quat).
        grad_base_pose_ = MakeDeviceFloat(std::vector<float>(art_count_ * 7u, 0.0f));
    }

    articulation::ArticulationDeviceState View() { return device_.View(); }
    uint32_t N() const { return n_; }

    // -- per-link device array I/O --
    void SetQ(const std::vector<float>& q) { Upload(View().q, q); }
    void SetQdot(const std::vector<float>& qd) { Upload(View().qdot, qd); }
    void SetTau(const std::vector<float>& t) { Upload(View().tau, t); }
    std::vector<float> GetQ() { return Download(View().q); }
    std::vector<float> GetQdot() { return Download(View().qdot); }
    std::vector<float> GetQddot() { return Download(View().qddot); }

    // Floating-base root spatial velocity (6 floats at link_velocity[0]).
    void SetRootVel(const std::array<float, 6>& v) {
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        ASSERT_EQ(cudaMemcpy(View().link_velocity, v.data(), 6u * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);
    }
    std::array<float, 6> GetRootVel() {
        std::array<float, 6> v{};
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        EXPECT_EQ(cudaMemcpy(v.data(), View().link_velocity, 6u * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        return v;
    }
    // Floating-base forward: ABA + floating-base velocity integrate + joint vel +
    // joint pos integrate (the contact-free floating step, matching StepFloating).
    void ForwardFloating(float gravity_z, float dt) {
        articulation::FeatherstoneAba::ComputeAccelerations(ctx_, View(), gravity_z);
        // Snapshot the PRE-INTEGRATION root spatial velocity: the next kernel
        // (IntegrateFloatingBaseVelocity) overwrites link_velocity[root] in place
        // (v_pre -> v_post). The root gyroscopic adjoint must linearize at v_pre,
        // so StepBackward reads this snapshot (in.v_root_pre), not the live state.
        SnapshotVRootPre();
        articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(ctx_, View(), dt,
                                                                     gravity_z);
        articulation::FeatherstoneAba::IntegrateVelocity(ctx_, View(), dt);
        articulation::FeatherstoneAba::IntegratePosition(ctx_, View(), dt);
        // p08b: snapshot the PRE-POSE-INTEGRATION base orientation, then integrate
        // the quaternion pose. The orientation adjoint linearizes at base_pose_pre
        // (the pose kernel overwrites base_pose in place, pre->post), mirroring
        // v_root_pre. Snapshot must precede the pose kernel.
        SnapshotBasePosePre();
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(ctx_, View(), dt);
        Sync();
    }
    // Copy the live link_velocity buffer (LinkSpatialVel == float[6], no padding)
    // into the v_root_pre scratch the reverse reads.
    void SnapshotVRootPre() {
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        ASSERT_EQ(cudaMemcpy(v_root_pre_.Data(), View().link_velocity,
                             n_ * sizeof(articulation::LinkSpatialVel),
                             cudaMemcpyDeviceToDevice),
                  cudaSuccess);
    }
    void SeedGradRootVel(const std::array<float, 6>& s) {
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        ASSERT_EQ(cudaMemcpy(grad_link_vel_.Data(), s.data(), 6u * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);
    }
    // dL/d(v_root_pre): the back-propagated root spatial-velocity gradient (the
    // base-DOF state->state hand-off Jacobian p02-B's tape chains).
    std::array<float, 6> GetGradRootVel() {
        std::array<float, 6> v{};
        grad_link_vel_.CopyToHost(v.data(), 6u * sizeof(float));
        return v;
    }

    // -- p08b orientation-channel helpers (articulation 0) --
    // Set the live base_pose[0] (position + rotation) for the floating root.
    void SetBasePose(const Transform& pose) {
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        ASSERT_EQ(cudaMemcpy(View().base_pose, &pose, sizeof(Transform),
                             cudaMemcpyHostToDevice), cudaSuccess);
    }
    Transform GetBasePose() {
        Transform pose;
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        EXPECT_EQ(cudaMemcpy(&pose, View().base_pose, sizeof(Transform),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        return pose;
    }
    // Copy the live base_pose buffer into the base_pose_pre snapshot (the reverse
    // reads in.base_pose_pre); called by ForwardFloating BEFORE the pose kernel.
    void SnapshotBasePosePre() {
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        ASSERT_EQ(cudaMemcpy(base_pose_pre_.Data(), View().base_pose,
                             art_count_ * sizeof(Transform),
                             cudaMemcpyDeviceToDevice),
                  cudaSuccess);
    }
    // Seed grad_base_pose_out[0] = [dL/d position'(3), dL/d rotation'(4)].
    void SeedGradBasePose(const std::array<float, 7>& s) {
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        ASSERT_EQ(cudaMemcpy(grad_base_pose_.Data(), s.data(), 7u * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);
    }
    // dL/d base_pose_pre (3 pos + 4 quat) for articulation 0, after the reverse.
    std::array<float, 7> GetGradBasePose() {
        std::array<float, 7> v{};
        grad_base_pose_.CopyToHost(v.data(), 7u * sizeof(float));
        return v;
    }

    // Internal mass-set helper (the FD-test mass perturbation): rebuild the
    // link's spatial inertia from MakeSpatialInertia(mass, diag, frame) and write
    // it to device link_inertia[link] -- the SAME representation dI/dmass slopes.
    void SetLinkMass(uint32_t link, float mass) {
        diffsim::LinkMassParams lp = model_.mass_params[link];
        const articulation::LinkSpatialInertia I =
            articulation::MakeSpatialInertia(mass, lp.diagonal_inertia,
                                             lp.inertial_frame);
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        articulation::LinkSpatialInertia* dst = View().link_inertia + link;
        ASSERT_EQ(cudaMemcpy(dst, &I, sizeof(I), cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    // -- forward composition (contact-free, fixed base) --
    // ABA only (rungs a/b/c + M^-1): tau already set, ComputeAccelerations.
    void ForwardAbaOnly(float gravity_z) {
        articulation::FeatherstoneAba::ComputeAccelerations(ctx_, View(), gravity_z);
        Sync();
    }
    // ABA + integrate (rung d): tau set, then accel + vel + pos integrate.
    void ForwardAbaIntegrate(float gravity_z, float dt) {
        articulation::FeatherstoneAba::ComputeAccelerations(ctx_, View(), gravity_z);
        articulation::FeatherstoneAba::IntegrateVelocity(ctx_, View(), dt);
        articulation::FeatherstoneAba::IntegratePosition(ctx_, View(), dt);
        Sync();
    }
    // drive + ABA + integrate (rung e): the full single step.
    void ForwardFullStep(const std::vector<float>& targets,
                         const std::vector<float>& kp,
                         const std::vector<float>& kd,
                         const std::vector<float>& flim,
                         float gravity_z, float dt) {
        OwnedDeviceBuffer bt = MakeDeviceFloat(targets);
        OwnedDeviceBuffer bp = MakeDeviceFloat(kp);
        OwnedDeviceBuffer bd = MakeDeviceFloat(kd);
        OwnedDeviceBuffer bl = MakeDeviceFloat(flim);
        articulation::FeatherstoneAba::ApplyPositionDrives(
            ctx_, View(), static_cast<const float*>(bt.Data()),
            static_cast<const float*>(bp.Data()), static_cast<const float*>(bd.Data()),
            static_cast<const float*>(bl.Data()), /*defer_velocity_damping=*/false);
        articulation::FeatherstoneAba::ComputeAccelerations(ctx_, View(), gravity_z);
        articulation::FeatherstoneAba::IntegrateVelocity(ctx_, View(), dt);
        articulation::FeatherstoneAba::IntegratePosition(ctx_, View(), dt);
        Sync();
    }

    // -- reverse --
    diffsim::StepBackwardGrads MakeGrads() {
        diffsim::StepBackwardGrads g;
        g.grad_q_out = static_cast<float*>(grad_q_.Data());
        g.grad_qdot_out = static_cast<float*>(grad_qdot_.Data());
        g.grad_target_out = static_cast<float*>(grad_target_.Data());
        g.grad_mass_out = static_cast<float*>(grad_mass_.Data());
        g.grad_tau_out = static_cast<float*>(grad_tau_.Data());
        g.grad_link_velocity_out = static_cast<float*>(grad_link_vel_.Data());
        g.grad_base_pose_out = static_cast<float*>(grad_base_pose_.Data());
        return g;
    }
    diffsim::StepBackwardInputs MakeInputs(float dt, float gravity_z) {
        diffsim::StepBackwardInputs in;
        in.q_pre = static_cast<const float*>(q_pre_.Data());
        in.qdot_pre = static_cast<const float*>(qdot_pre_.Data());
        in.v_root_pre = static_cast<const float*>(v_root_pre_.Data());
        in.base_pose_pre =
            static_cast<const Transform*>(base_pose_pre_.Data());
        in.dI_dmass = static_cast<const float*>(dI_dmass_.Data());
        in.dt = dt;
        in.gravity_z = gravity_z;
        return in;
    }
    void SnapshotPre(const std::vector<float>& q, const std::vector<float>& qd) {
        Upload(static_cast<float*>(q_pre_.Data()), q);
        Upload(static_cast<float*>(qdot_pre_.Data()), qd);
    }
    void ZeroGradSeeds() {
        const std::vector<float> z(n_, 0.0f);
        Upload(static_cast<float*>(grad_q_.Data()), z);
        Upload(static_cast<float*>(grad_qdot_.Data()), z);
        Upload(static_cast<float*>(grad_target_.Data()), z);
        Upload(static_cast<float*>(grad_mass_.Data()), z);
        Upload(static_cast<float*>(grad_tau_.Data()), z);
        Upload(static_cast<float*>(grad_qddot_seed_.Data()), z);
        Upload(static_cast<float*>(grad_link_vel_.Data()), std::vector<float>(n_ * 6u, 0.0f));
        Upload(static_cast<float*>(grad_base_pose_.Data()),
               std::vector<float>(art_count_ * 7u, 0.0f));
    }
    void SeedQddot(const std::vector<float>& s) {
        Upload(static_cast<float*>(grad_qddot_seed_.Data()), s);
    }
    void SeedGradQ(const std::vector<float>& s) { Upload(static_cast<float*>(grad_q_.Data()), s); }
    void SeedGradQdot(const std::vector<float>& s) { Upload(static_cast<float*>(grad_qdot_.Data()), s); }
    float* GradQddotSeedPtr() { return static_cast<float*>(grad_qddot_seed_.Data()); }

    std::vector<float> GetGradQ() { return DownloadBuf(grad_q_); }
    std::vector<float> GetGradQdot() { return DownloadBuf(grad_qdot_); }
    std::vector<float> GetGradTarget() { return DownloadBuf(grad_target_); }
    std::vector<float> GetGradMass() { return DownloadBuf(grad_mass_); }
    std::vector<float> GetGradTau() { return DownloadBuf(grad_tau_); }

    void RunBackward(const diffsim::StepBackwardInputs& in,
                     const diffsim::StepBackwardGrads& g) {
        diffsim::StepBackward(ctx_, View(), in, g);
        Sync();
    }
    void Sync() { ctx_.stream.Synchronize(); }

private:
    OwnedDeviceBuffer MakeDeviceFloat(const std::vector<float>& v) {
        OwnedDeviceBuffer b(v.size() * sizeof(float));
        if (!v.empty()) b.CopyFromHost(v.data(), v.size() * sizeof(float));
        return b;
    }
    void Upload(float* dst, const std::vector<float>& v) {
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        ASSERT_EQ(cudaMemcpy(dst, v.data(), v.size() * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);
    }
    std::vector<float> Download(const float* src) {
        std::vector<float> v(n_, 0.0f);
        nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
        EXPECT_EQ(cudaMemcpy(v.data(), src, n_ * sizeof(float),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        return v;
    }
    std::vector<float> DownloadBuf(const OwnedDeviceBuffer& b) {
        std::vector<float> v(b.Size() / sizeof(float), 0.0f);
        b.CopyToHost(v.data(), b.Size());
        return v;
    }

    const nuka::phi::DeviceContext& ctx_;
    const ChainModel& model_;
    articulation::ArticulationDeviceBuffers device_;
    uint32_t n_ = 0u;
    size_t art_count_ = 0u;
    OwnedDeviceBuffer dI_dmass_, grad_q_, grad_qdot_, grad_target_, grad_mass_,
        grad_tau_, grad_qddot_seed_, grad_link_vel_, q_pre_, qdot_pre_, v_root_pre_,
        base_pose_pre_, grad_base_pose_;
};

float MaxRelErr(const std::vector<float>& a, const std::vector<float>& b,
                const std::vector<uint32_t>& idx) {
    float worst = 0.0f;
    for (uint32_t i : idx) {
        const float denom = std::max(1e-4f, std::fabs(a[i]) + std::fabs(b[i]));
        const float rel = std::fabs(a[i] - b[i]) / denom;
        worst = std::max(worst, rel);
    }
    return worst;
}

// Combined absolute-OR-relative error: a component passes if it is within abs_tol
// (absolute) OR rel_tol (relative). Standard for FD validation where some grad
// components are near zero (their relative error is dominated by FD truncation
// noise on a tiny magnitude). Returns the max over components of
// min(abs_err / abs_tol, rel_err / rel_tol) -- < 1 means all pass.
float MaxAbsOrRel(const std::vector<float>& a, const std::vector<float>& b,
                  const std::vector<uint32_t>& idx, float abs_tol, float rel_tol) {
    float worst = 0.0f;
    for (uint32_t i : idx) {
        const float abs_err = std::fabs(a[i] - b[i]);
        const float denom = std::max(1e-6f, std::fabs(a[i]) + std::fabs(b[i]));
        const float rel_err = abs_err / denom;
        const float score = std::min(abs_err / abs_tol, rel_err / rel_tol);
        worst = std::max(worst, score);
    }
    return worst;
}

// Baseline q/qdot/tau for the FD configs (nonzero, away from the clamp).
const std::vector<float> kBaseQ = {0.0f, 0.30f, -0.45f, 0.20f};
const std::vector<float> kBaseQdot = {0.0f, 0.40f, -0.25f, 0.55f};
const std::vector<float> kBaseTau = {0.0f, 0.50f, -0.30f, 0.20f};

}  // namespace

// ===========================================================================
// EXACT M^-1 cross-check: d(qddot)/d(tau) = M^-1 (qdot=0, g=0). Forward oracle
// column j (tau=e_j -> qddot=col j) vs reverse row i (seed qddot=e_i ->
// grad_tau=row i). Symmetric to ~1e-6, no FD truncation noise.
// ===========================================================================
TEST(AbaReverse, ExactMInverseCrossCheck) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;
    const float g0 = 0.0f;

    const std::vector<float> zeros(h.N(), 0.0f);

    // Forward oracle: M^-1 columns.
    std::vector<std::vector<float>> minv_cols(h.N(), std::vector<float>(h.N(), 0.0f));
    for (uint32_t j : dofs) {
        h.SetQ(kBaseQ);
        h.SetQdot(zeros);
        std::vector<float> tau(h.N(), 0.0f);
        tau[j] = 1.0f;
        h.SetTau(tau);
        h.ForwardAbaOnly(g0);
        minv_cols[j] = h.GetQddot();
    }

    // Reverse rows: seed qddot=e_i, read grad_tau (== row i of M^-1).
    std::vector<std::vector<float>> minv_rows(h.N(), std::vector<float>(h.N(), 0.0f));
    for (uint32_t i : dofs) {
        // Re-run forward at the baseline (qdot=0) so the checkpoints are valid.
        h.SetQ(kBaseQ);
        h.SetQdot(zeros);
        h.SetTau(zeros);
        h.ForwardAbaOnly(g0);
        h.SnapshotPre(kBaseQ, zeros);
        h.ZeroGradSeeds();
        std::vector<float> seed(h.N(), 0.0f);
        seed[i] = 1.0f;
        h.SeedQddot(seed);
        auto in = h.MakeInputs(/*dt=*/0.0f, g0);
        in.has_drive = false;
        in.has_integrate = false;
        in.enable_q_channel = false;  // tau-channel only
        in.grad_qddot_seed = h.GradQddotSeedPtr();
        h.RunBackward(in, h.MakeGrads());
        minv_rows[i] = h.GetGradTau();
    }

    float worst = 0.0f;
    for (uint32_t i : dofs)
        for (uint32_t j : dofs)
            worst = std::max(worst, std::fabs(minv_rows[i][j] - minv_cols[j][i]));
    printf("[M^-1 cross-check] max |row_i[j] - col_j[i]| = %.3e\n", worst);
    EXPECT_LT(worst, 1e-5f);
    // Sanity: M^-1 is non-trivial (not all zeros).
    float mag = 0.0f;
    for (uint32_t j : dofs) for (uint32_t i : dofs) mag += std::fabs(minv_cols[j][i]);
    EXPECT_GT(mag, 1e-3f);
}

// ===========================================================================
// Rung (a): d(qddot)/d(tau) and d(qddot)/d(qdot) -- reverse vs central FD.
// gravity ON, nonzero qdot. Scalar loss L = sum_k w_k * qddot_k.
// ===========================================================================
TEST(AbaReverse, RungA_dQddot_dTau_dQdot) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;
    const float g0 = 9.81f;
    // loss weights on qddot (one per DOF).
    std::vector<float> w(h.N(), 0.0f);
    w[1] = 0.7f; w[2] = -1.1f; w[3] = 0.5f;

    auto loss_from_qddot = [&](const std::vector<float>& qddot) {
        float s = 0.0f; for (uint32_t k : dofs) s += w[k] * qddot[k]; return s;
    };

    // Reverse: seed grad_qddot = w, read grad_tau, grad_qdot.
    h.SetQ(kBaseQ); h.SetQdot(kBaseQdot); h.SetTau(kBaseTau);
    h.ForwardAbaOnly(g0);
    h.SnapshotPre(kBaseQ, kBaseQdot);
    h.ZeroGradSeeds();
    h.SeedQddot(w);
    auto in = h.MakeInputs(0.0f, g0);
    in.has_drive = false; in.has_integrate = false; in.enable_q_channel = false;
    in.grad_qddot_seed = h.GradQddotSeedPtr();
    h.RunBackward(in, h.MakeGrads());
    const std::vector<float> grad_tau = h.GetGradTau();
    const std::vector<float> grad_qdot = h.GetGradQdot();

    // Central FD on tau and qdot.
    const float eps = 2e-3f;
    std::vector<float> fd_tau(h.N(), 0.0f), fd_qdot(h.N(), 0.0f);
    for (uint32_t k : dofs) {
        auto perturb_tau = [&](float d) {
            std::vector<float> t = kBaseTau; t[k] += d;
            h.SetQ(kBaseQ); h.SetQdot(kBaseQdot); h.SetTau(t);
            h.ForwardAbaOnly(g0); return loss_from_qddot(h.GetQddot());
        };
        fd_tau[k] = (perturb_tau(eps) - perturb_tau(-eps)) / (2.0f * eps);
        auto perturb_qdot = [&](float d) {
            std::vector<float> qd = kBaseQdot; qd[k] += d;
            h.SetQ(kBaseQ); h.SetQdot(qd); h.SetTau(kBaseTau);
            h.ForwardAbaOnly(g0); return loss_from_qddot(h.GetQddot());
        };
        fd_qdot[k] = (perturb_qdot(eps) - perturb_qdot(-eps)) / (2.0f * eps);
    }
    const float e_tau = MaxRelErr(grad_tau, fd_tau, dofs);
    const float e_qdot = MaxRelErr(grad_qdot, fd_qdot, dofs);
    printf("[rung a] max_rel_err d/dtau = %.3e, d/dqdot = %.3e\n", e_tau, e_qdot);
    EXPECT_LT(e_tau, 1e-3f);
    EXPECT_LT(e_qdot, 1e-3f);
}

// ===========================================================================
// Rung (a) q-channel: d(qddot)/d(q) -- the link_xup = JointTransform(q) X-path.
// gravity ON, qdot ON. Reverse (enable_q_channel) vs central FD on q.
// ===========================================================================
TEST(AbaReverse, RungA_dQddot_dQ_Xpath) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;
    const float g0 = 9.81f;
    std::vector<float> w(h.N(), 0.0f);
    w[1] = 0.7f; w[2] = -1.1f; w[3] = 0.5f;
    auto loss_from_qddot = [&](const std::vector<float>& qddot) {
        float s = 0.0f; for (uint32_t k : dofs) s += w[k] * qddot[k]; return s;
    };

    h.SetQ(kBaseQ); h.SetQdot(kBaseQdot); h.SetTau(kBaseTau);
    h.ForwardAbaOnly(g0);
    h.SnapshotPre(kBaseQ, kBaseQdot);
    h.ZeroGradSeeds();
    h.SeedQddot(w);
    auto in = h.MakeInputs(0.0f, g0);
    in.has_drive = false; in.has_integrate = false; in.enable_q_channel = true;
    in.grad_qddot_seed = h.GradQddotSeedPtr();
    h.RunBackward(in, h.MakeGrads());
    const std::vector<float> grad_q = h.GetGradQ();

    const float eps = 2e-3f;
    std::vector<float> fd_q(h.N(), 0.0f);
    for (uint32_t k : dofs) {
        auto perturb_q = [&](float d) {
            std::vector<float> q = kBaseQ; q[k] += d;
            h.SetQ(q); h.SetQdot(kBaseQdot); h.SetTau(kBaseTau);
            h.ForwardAbaOnly(g0); return loss_from_qddot(h.GetQddot());
        };
        fd_q[k] = (perturb_q(eps) - perturb_q(-eps)) / (2.0f * eps);
    }
    const float e_q = MaxRelErr(grad_q, fd_q, dofs);
    printf("[rung a q-channel] max_rel_err d/dq = %.3e\n", e_q);
    EXPECT_LT(e_q, 2e-3f);
}

// ===========================================================================
// Rung (b): d(qddot)/d(mass) for each link -- reverse vs central FD. The FD
// perturbs the SAME spatial-inertia representation (SetLinkMass calls
// MakeSpatialInertia) that dI/dmass slopes. NONZERO COM -> full coupling.
// ===========================================================================
TEST(AbaReverse, RungB_dQddot_dMass) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;
    const float g0 = 9.81f;
    std::vector<float> w(h.N(), 0.0f);
    w[1] = 0.7f; w[2] = -1.1f; w[3] = 0.5f;
    auto loss_from_qddot = [&](const std::vector<float>& qddot) {
        float s = 0.0f; for (uint32_t k : dofs) s += w[k] * qddot[k]; return s;
    };

    h.SetQ(kBaseQ); h.SetQdot(kBaseQdot); h.SetTau(kBaseTau);
    h.ForwardAbaOnly(g0);
    h.SnapshotPre(kBaseQ, kBaseQdot);
    h.ZeroGradSeeds();
    h.SeedQddot(w);
    auto in = h.MakeInputs(0.0f, g0);
    in.has_drive = false; in.has_integrate = false; in.enable_q_channel = false;
    in.grad_qddot_seed = h.GradQddotSeedPtr();
    h.RunBackward(in, h.MakeGrads());
    const std::vector<float> grad_mass = h.GetGradMass();

    // Central FD on each link's mass (perturb spatial inertia consistently).
    const float eps = 2e-3f;
    std::vector<float> fd_mass(h.N(), 0.0f);
    std::vector<uint32_t> all_links;
    for (uint32_t l = 0u; l < h.N(); ++l) all_links.push_back(l);
    for (uint32_t l : all_links) {
        const float m0 = model.mass_params[l].mass;
        auto perturb_mass = [&](float d) {
            h.SetLinkMass(l, m0 + d);
            h.SetQ(kBaseQ); h.SetQdot(kBaseQdot); h.SetTau(kBaseTau);
            h.ForwardAbaOnly(g0);
            const float lo = loss_from_qddot(h.GetQddot());
            h.SetLinkMass(l, m0);  // restore
            return lo;
        };
        fd_mass[l] = (perturb_mass(eps) - perturb_mass(-eps)) / (2.0f * eps);
    }
    const float e_mass = MaxRelErr(grad_mass, fd_mass, all_links);
    printf("[rung b] max_rel_err d/dmass = %.3e  (grad=[%.4f %.4f %.4f %.4f] fd=[%.4f %.4f %.4f %.4f])\n",
           e_mass, grad_mass[0], grad_mass[1], grad_mass[2], grad_mass[3],
           fd_mass[0], fd_mass[1], fd_mass[2], fd_mass[3]);
    EXPECT_LT(e_mass, 5e-3f);
}

// ===========================================================================
// Rung (d): d(state')/d(tau, qdot) through IntegrateVelocity + IntegratePosition.
// loss on post-step q' and qdot'. Reverse (has_integrate=true) vs central FD.
// ===========================================================================
TEST(AbaReverse, RungD_Integrate_dTau_dQdot) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;
    const float g0 = 9.81f, dt = 0.01f;
    std::vector<float> wq(h.N(), 0.0f), wqd(h.N(), 0.0f);
    wq[1] = 0.6f; wq[2] = -0.9f; wq[3] = 0.4f;
    wqd[1] = -0.3f; wqd[2] = 0.7f; wqd[3] = -0.5f;
    auto loss_post = [&]() {
        const std::vector<float> q = h.GetQ();
        const std::vector<float> qd = h.GetQdot();
        float s = 0.0f;
        for (uint32_t k : dofs) s += wq[k] * q[k] + wqd[k] * qd[k];
        return s;
    };

    h.SetQ(kBaseQ); h.SetQdot(kBaseQdot); h.SetTau(kBaseTau);
    h.ForwardAbaIntegrate(g0, dt);
    h.SnapshotPre(kBaseQ, kBaseQdot);
    h.ZeroGradSeeds();
    h.SeedGradQ(wq);
    h.SeedGradQdot(wqd);
    auto in = h.MakeInputs(dt, g0);
    in.has_drive = false; in.has_integrate = true; in.enable_q_channel = false;
    h.RunBackward(in, h.MakeGrads());
    const std::vector<float> grad_tau = h.GetGradTau();
    const std::vector<float> grad_qdot = h.GetGradQdot();

    const float eps = 2e-3f;
    std::vector<float> fd_tau(h.N(), 0.0f), fd_qdot(h.N(), 0.0f);
    for (uint32_t k : dofs) {
        auto p_tau = [&](float d) {
            std::vector<float> t = kBaseTau; t[k] += d;
            h.SetQ(kBaseQ); h.SetQdot(kBaseQdot); h.SetTau(t);
            h.ForwardAbaIntegrate(g0, dt); return loss_post();
        };
        fd_tau[k] = (p_tau(eps) - p_tau(-eps)) / (2.0f * eps);
        auto p_qd = [&](float d) {
            std::vector<float> qd = kBaseQdot; qd[k] += d;
            h.SetQ(kBaseQ); h.SetQdot(qd); h.SetTau(kBaseTau);
            h.ForwardAbaIntegrate(g0, dt); return loss_post();
        };
        fd_qdot[k] = (p_qd(eps) - p_qd(-eps)) / (2.0f * eps);
    }
    const float e_tau = MaxRelErr(grad_tau, fd_tau, dofs);
    const float e_qdot = MaxRelErr(grad_qdot, fd_qdot, dofs);
    printf("[rung d] max_rel_err d/dtau = %.3e, d/dqdot = %.3e\n", e_tau, e_qdot);
    EXPECT_LT(e_tau, 2e-3f);
    EXPECT_LT(e_qdot, 2e-3f);
}

// ===========================================================================
// Rung (e): full single step. d(loss(state'))/d(action) and d(loss)/d(mass).
// action = drive_targets. drive + ABA + integrate, gravity on. Reverse vs FD.
// ===========================================================================
TEST(AbaReverse, RungE_FullStep_dAction_dMass) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;
    const float g0 = 9.81f, dt = 0.01f;
    const std::vector<float> kp = {0.0f, 25.0f, 20.0f, 15.0f};
    const std::vector<float> kd = {0.0f, 1.5f, 1.2f, 0.9f};
    const std::vector<float> flim = {0.0f, 200.0f, 200.0f, 200.0f};  // wide -> unsaturated
    const std::vector<float> targets = {0.0f, 0.45f, -0.30f, 0.35f};
    std::vector<float> wq(h.N(), 0.0f), wqd(h.N(), 0.0f);
    wq[1] = 0.6f; wq[2] = -0.9f; wq[3] = 0.4f;
    wqd[1] = -0.3f; wqd[2] = 0.7f; wqd[3] = -0.5f;
    auto loss_post = [&]() {
        const std::vector<float> q = h.GetQ();
        const std::vector<float> qd = h.GetQdot();
        float s = 0.0f;
        for (uint32_t k : dofs) s += wq[k] * q[k] + wqd[k] * qd[k];
        return s;
    };

    h.SetQ(kBaseQ); h.SetQdot(kBaseQdot);
    h.ForwardFullStep(targets, kp, kd, flim, g0, dt);
    h.SnapshotPre(kBaseQ, kBaseQdot);
    h.ZeroGradSeeds();
    h.SeedGradQ(wq);
    h.SeedGradQdot(wqd);
    auto in = h.MakeInputs(dt, g0);
    in.has_drive = true; in.has_integrate = true; in.enable_q_channel = true;
    // drive descriptors for the drive reverse.
    OwnedDeviceBuffer bt(targets.size() * sizeof(float));
    bt.CopyFromHost(targets.data(), targets.size() * sizeof(float));
    OwnedDeviceBuffer bp(kp.size() * sizeof(float));
    bp.CopyFromHost(kp.data(), kp.size() * sizeof(float));
    OwnedDeviceBuffer bd(kd.size() * sizeof(float));
    bd.CopyFromHost(kd.data(), kd.size() * sizeof(float));
    OwnedDeviceBuffer bl(flim.size() * sizeof(float));
    bl.CopyFromHost(flim.data(), flim.size() * sizeof(float));
    in.drive_targets = static_cast<const float*>(bt.Data());
    in.drive_stiffness = static_cast<const float*>(bp.Data());
    in.drive_damping = static_cast<const float*>(bd.Data());
    in.drive_force_limits = static_cast<const float*>(bl.Data());
    h.RunBackward(in, h.MakeGrads());
    const std::vector<float> grad_action = h.GetGradTarget();
    const std::vector<float> grad_mass = h.GetGradMass();

    const float eps = 2e-3f;
    std::vector<float> fd_action(h.N(), 0.0f);
    for (uint32_t k : dofs) {
        auto p_act = [&](float d) {
            std::vector<float> t = targets; t[k] += d;
            h.SetQ(kBaseQ); h.SetQdot(kBaseQdot);
            h.ForwardFullStep(t, kp, kd, flim, g0, dt); return loss_post();
        };
        fd_action[k] = (p_act(eps) - p_act(-eps)) / (2.0f * eps);
    }
    std::vector<uint32_t> all_links;
    for (uint32_t l = 0u; l < h.N(); ++l) all_links.push_back(l);
    std::vector<float> fd_mass(h.N(), 0.0f);
    for (uint32_t l : all_links) {
        const float m0 = model.mass_params[l].mass;
        auto p_mass = [&](float d) {
            h.SetLinkMass(l, m0 + d);
            h.SetQ(kBaseQ); h.SetQdot(kBaseQdot);
            h.ForwardFullStep(targets, kp, kd, flim, g0, dt);
            const float lo = loss_post();
            h.SetLinkMass(l, m0);
            return lo;
        };
        fd_mass[l] = (p_mass(eps) - p_mass(-eps)) / (2.0f * eps);
    }
    const float e_action = MaxRelErr(grad_action, fd_action, dofs);
    const float e_mass = MaxRelErr(grad_mass, fd_mass, all_links);
    printf("[rung e] max_rel_err d/daction = %.3e, d/dmass = %.3e\n", e_action, e_mass);
    EXPECT_LT(e_action, 3e-3f);
    EXPECT_LT(e_mass, 1e-2f);
}

// ===========================================================================
// Rung (c): FLOATING base. d(state')/d(tau, mass) where state' = post-step joint
// qdot' (joints) + root spatial velocity v_root'. Tests the LDLT-solve adjoint at
// the root + the floating-base velocity-integrator reverse + the root gyroscopic
// pass-1 path. Reverse vs central FD. gravity ON, nonzero root velocity + qdot.
// ===========================================================================
TEST(AbaReverse, RungC_Floating_dTau_dMass) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFloatingChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;  // {1,2}
    const float g0 = 9.81f, dt = 0.01f;
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f};
    const std::vector<float> qd0 = {0.0f, 0.40f, -0.25f};
    const std::vector<float> tau0 = {0.0f, 0.50f, -0.30f};
    const std::array<float, 6> rootv0 = {0.10f, -0.05f, 0.08f, 0.20f, -0.15f, 0.12f};
    // p08b: IntegrateFloatingBasePose mutates base_pose in place; reset the
    // orientation to its construction-time value before every forward so the FD
    // loops do not drift the gravity-coupling linearization point (see the LargeDt
    // variant for the detailed rationale).
    const Transform bp0 = h.GetBasePose();

    // loss = sum_k wqd[k]*qdot'_k + sum_i wrv[i]*v_root'_i
    std::vector<float> wqd(h.N(), 0.0f);
    wqd[1] = 0.7f; wqd[2] = -0.5f;
    const std::array<float, 6> wrv = {0.3f, -0.4f, 0.6f, -0.2f, 0.5f, -0.35f};
    auto loss_post = [&]() {
        const std::vector<float> qd = h.GetQdot();
        const std::array<float, 6> rv = h.GetRootVel();
        float s = 0.0f;
        for (uint32_t k : dofs) s += wqd[k] * qd[k];
        for (uint32_t i = 0u; i < 6u; ++i) s += wrv[i] * rv[i];
        return s;
    };

    auto set_state = [&]() {
        h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rootv0);
        h.SetBasePose(bp0);
    };

    // Reverse: seed grad_qdot' = wqd, grad v_root' = wrv.
    set_state();
    h.ForwardFloating(g0, dt);
    h.SnapshotPre(q0, qd0);
    h.ZeroGradSeeds();
    h.SeedGradQdot(wqd);
    h.SeedGradRootVel(wrv);
    auto in = h.MakeInputs(dt, g0);
    in.has_drive = false; in.has_integrate = true; in.enable_q_channel = false;
    h.RunBackward(in, h.MakeGrads());
    const std::vector<float> grad_tau = h.GetGradTau();
    const std::vector<float> grad_mass = h.GetGradMass();
    // dL/d(v_root_pre): the base-DOF state->state hand-off Jacobian (p02-B chains
    // step t's v_root' into step t+1's v_root_pre).
    const std::array<float, 6> grad_rootv = h.GetGradRootVel();

    const float eps = 2e-3f;
    std::vector<float> fd_tau(h.N(), 0.0f);
    for (uint32_t k : dofs) {
        auto p_tau = [&](float d) {
            std::vector<float> t = tau0; t[k] += d;
            h.SetQ(q0); h.SetQdot(qd0); h.SetTau(t); h.SetRootVel(rootv0);
            h.SetBasePose(bp0);
            h.ForwardFloating(g0, dt); return loss_post();
        };
        fd_tau[k] = (p_tau(eps) - p_tau(-eps)) / (2.0f * eps);
    }
    // Central FD on each of the 6 root spatial-velocity components.
    std::array<float, 6> fd_rootv{};
    for (uint32_t c = 0u; c < 6u; ++c) {
        auto p_rv = [&](float d) {
            std::array<float, 6> rv = rootv0; rv[c] += d;
            h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rv);
            h.SetBasePose(bp0);
            h.ForwardFloating(g0, dt); return loss_post();
        };
        fd_rootv[c] = (p_rv(eps) - p_rv(-eps)) / (2.0f * eps);
    }
    std::vector<uint32_t> all_links;
    for (uint32_t l = 0u; l < h.N(); ++l) all_links.push_back(l);
    std::vector<float> fd_mass(h.N(), 0.0f);
    for (uint32_t l : all_links) {
        const float m0 = model.mass_params[l].mass;
        auto p_mass = [&](float d) {
            h.SetLinkMass(l, m0 + d);
            h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rootv0);
            h.SetBasePose(bp0);
            h.ForwardFloating(g0, dt);
            const float lo = loss_post();
            h.SetLinkMass(l, m0);
            return lo;
        };
        fd_mass[l] = (p_mass(eps) - p_mass(-eps)) / (2.0f * eps);
    }
    const float e_tau = MaxRelErr(grad_tau, fd_tau, dofs);
    // Mass grads on the floating base are small-magnitude (loss weighted on the
    // velocities, not directly on mass); use abs-OR-rel so FD truncation noise on
    // a ~5e-3 gradient is not flagged as a relative miss. abs 5e-4 OR rel 1e-2.
    const float e_mass = MaxAbsOrRel(grad_mass, fd_mass, all_links,
                                     /*abs_tol=*/5e-4f, /*rel_tol=*/1e-2f);
    // d/d(v_root) is O(1) -> strict relative metric.
    std::vector<float> grv(grad_rootv.begin(), grad_rootv.end());
    std::vector<float> frv(fd_rootv.begin(), fd_rootv.end());
    std::vector<uint32_t> rv_idx = {0u, 1u, 2u, 3u, 4u, 5u};
    const float e_rootv = MaxRelErr(grv, frv, rv_idx);
    printf("[rung c floating] max_rel_err d/dtau = %.3e, d/dmass score = %.3e, "
           "d/dv_root = %.3e  (gm=[%.4f %.4f %.4f] fd=[%.4f %.4f %.4f])\n",
           e_tau, e_mass, e_rootv, grad_mass[0], grad_mass[1], grad_mass[2],
           fd_mass[0], fd_mass[1], fd_mass[2]);
    EXPECT_LT(e_tau, 3e-3f);
    EXPECT_LT(e_mass, 1.0f);  // score < 1 == every component within abs OR rel tol
    EXPECT_LT(e_rootv, 2e-3f);
}

// ===========================================================================
// Rung (c) LARGE-(a_free - a_grav): the LINEARIZATION-POINT regression guard.
// The root gyroscopic bias p[root] = ForceCross(v, I*v) is QUADRATIC in the root
// spatial velocity v, so its adjoint (and the grad_I[root] -> grad_mass[root]
// path) depends on WHERE it is linearized. The forward integrator overwrites
// link_velocity[root] in place (v_pre -> v_post = v_pre + (a_free-a_grav)*dt),
// so the reverse MUST read the PRE-integration v_pre (in.v_root_pre), not the
// live post value. Here we AMPLIFY the root angular velocity and use a LARGE
// dt=0.05 so |v_post - v_pre| ~ 1-2: if the reverse mistakenly linearized at
// v_post, d/d(v_root) vs central-FD would be ~4.7e-2 (~8%). With v_root_pre it
// is the FD noise floor (~1e-3). The error scales O((a_free-a_grav)*dt) -> tiny
// on the near-free-fall demo (which masked the bug) but large once p03 contacts
// make (a_free-a_grav) big. This test FAILS if the linearization point regresses.
// ===========================================================================
TEST(AbaReverse, RungC_Floating_LargeDt_LinearizationPoint) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFloatingChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;  // {1,2}
    const float g0 = 9.81f, dt = 0.05f;  // LARGE dt -> big |v_post - v_pre|
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f};
    const std::vector<float> qd0 = {0.0f, 0.40f, -0.25f};
    const std::vector<float> tau0 = {0.0f, 0.50f, -0.30f};
    // AMPLIFIED root angular velocity (first 3 comps) -> the gyroscopic term
    // ForceCross(v, I*v) is large and strongly v-dependent.
    const std::array<float, 6> rootv0 = {3.0f, -2.5f, 4.0f, 0.20f, -0.15f, 0.12f};
    // p08b: IntegrateFloatingBasePose now mutates base_pose IN PLACE every forward.
    // The FD loops below re-run ForwardFloating repeatedly without managing the
    // orientation, so base_pose would DRIFT across iterations (gravity coupling
    // a_grav = R(base_rot)^T g changes -> contaminated FD reference). Capture the
    // construction-time orientation and reset it before EVERY forward (mirrors the
    // q/qdot/tau/rootvel resets), keeping the linearization point fixed.
    const Transform bp0 = h.GetBasePose();

    std::vector<float> wqd(h.N(), 0.0f);
    wqd[1] = 0.7f; wqd[2] = -0.5f;
    const std::array<float, 6> wrv = {0.3f, -0.4f, 0.6f, -0.2f, 0.5f, -0.35f};
    auto loss_post = [&]() {
        const std::vector<float> qd = h.GetQdot();
        const std::array<float, 6> rv = h.GetRootVel();
        float s = 0.0f;
        for (uint32_t k : dofs) s += wqd[k] * qd[k];
        for (uint32_t i = 0u; i < 6u; ++i) s += wrv[i] * rv[i];
        return s;
    };
    auto set_state = [&]() {
        h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rootv0);
        h.SetBasePose(bp0);
    };

    // Sanity: confirm the perturbation regime really is large -- |v_post - v_pre|
    // on the root must be O(1) for this to discriminate the linearization point.
    set_state();
    h.ForwardFloating(g0, dt);
    const std::array<float, 6> vpost = h.GetRootVel();
    float dv = 0.0f;
    for (uint32_t i = 0u; i < 6u; ++i) dv += std::fabs(vpost[i] - rootv0[i]);
    EXPECT_GT(dv, 0.5f) << "regime not large enough to discriminate v_pre vs v_post";

    // -- Reverse, run TWICE: with the v_pre snapshot (the FIX), and with
    //    v_root_pre=nullptr (the kernel then falls back to the live POST-
    //    integration state -- i.e. the BUG). The fix must (a) match FD to the
    //    noise floor and (b) beat the bug variant by a wide margin. This directly
    //    encodes the reviewer's "the two variants diverge -> confirms the
    //    linearization point is the cause", and is immune to FD-floor calibration.
    auto run_reverse = [&](bool use_v_pre) {
        set_state();
        h.ForwardFloating(g0, dt);
        h.SnapshotPre(q0, qd0);
        h.ZeroGradSeeds();
        h.SeedGradQdot(wqd);
        h.SeedGradRootVel(wrv);
        auto in = h.MakeInputs(dt, g0);
        in.has_drive = false; in.has_integrate = true; in.enable_q_channel = false;
        if (!use_v_pre) in.v_root_pre = nullptr;  // -> live (v_post) fallback = bug
        h.RunBackward(in, h.MakeGrads());
        struct R { std::array<float, 6> rootv; std::vector<float> mass; } r;
        r.rootv = h.GetGradRootVel();
        r.mass = h.GetGradMass();
        return r;
    };
    const auto fix = run_reverse(/*use_v_pre=*/true);
    const auto bug = run_reverse(/*use_v_pre=*/false);

    // Central FD on each root spatial-velocity component (the channel that reads
    // v through the gyroscopic ForceCross(v, I*v) bias).
    const float eps = 2e-3f;
    std::array<float, 6> fd_rootv{};
    for (uint32_t c = 0u; c < 6u; ++c) {
        auto p_rv = [&](float d) {
            std::array<float, 6> rv = rootv0; rv[c] += d;
            h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rv);
            h.SetBasePose(bp0);
            h.ForwardFloating(g0, dt); return loss_post();
        };
        fd_rootv[c] = (p_rv(eps) - p_rv(-eps)) / (2.0f * eps);
    }
    // Central FD on the floating-base ROOT mass (the grad_I[root]->grad_mass[root]
    // path also reads v via w = I*v in the gyroscopic outer product).
    const float m0 = model.mass_params[0].mass;
    auto p_mass = [&](float d) {
        h.SetLinkMass(0u, m0 + d);
        h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rootv0);
        h.SetBasePose(bp0);
        h.ForwardFloating(g0, dt);
        const float lo = loss_post();
        h.SetLinkMass(0u, m0);
        return lo;
    };
    const float fd_mass_root = (p_mass(eps) - p_mass(-eps)) / (2.0f * eps);

    std::vector<float> frv(fd_rootv.begin(), fd_rootv.end());
    std::vector<uint32_t> rv_idx = {0u, 1u, 2u, 3u, 4u, 5u};
    std::vector<float> grv_fix(fix.rootv.begin(), fix.rootv.end());
    std::vector<float> grv_bug(bug.rootv.begin(), bug.rootv.end());
    const float e_rootv_fix = MaxRelErr(grv_fix, frv, rv_idx);
    const float e_rootv_bug = MaxRelErr(grv_bug, frv, rv_idx);
    // Mass[root] is small-magnitude here (loss weighted on velocities, not mass),
    // so use ABSOLUTE error vs FD -- relative error is dominated by FD truncation
    // (same convention as RungC's MaxAbsOrRel). The v_pre variant must be much
    // closer to FD than the v_post (bug) variant.
    const float em_fix = std::fabs(fix.mass[0] - fd_mass_root);
    const float em_bug = std::fabs(bug.mass[0] - fd_mass_root);
    printf("[rung c large-dt] |dv_root|=%.3f\n"
           "  d/dv_root  rel-err: v_pre(fix)=%.3e  v_post(bug)=%.3e\n"
           "  d/dmass[0] abs-err: v_pre(fix)=%.3e  v_post(bug)=%.3e  "
           "(fd=%.4e gm_fix=%.4e gm_bug=%.4e)\n",
           dv, e_rootv_fix, e_rootv_bug, em_fix, em_bug,
           fd_mass_root, fix.mass[0], bug.mass[0]);

    // (a) The v_pre (fix) reverse matches FD to the noise floor on the O(1)
    //     velocity channel.
    EXPECT_LT(e_rootv_fix, 5e-3f);
    // (b) The fix BEATS the v_post (bug) variant by a wide margin on BOTH the
    //     velocity channel and the gyroscopic mass[root] path -- the divergence
    //     IS the linearization-point error (~4.7e-2 on d/dv_root if unfixed).
    EXPECT_GT(e_rootv_bug, 5e-3f) << "regime did not exercise the v-dependence";
    EXPECT_LT(e_rootv_fix * 4.0f, e_rootv_bug)
        << "v_pre fix must beat the v_post linearization by a wide margin";
    EXPECT_LT(em_fix * 2.0f, em_bug)
        << "v_pre fix must beat v_post on grad_mass[root] gyroscopic path";
}

// ===========================================================================
// p08b -- ORIENTATION channel. The floating-base backward now differentiates the
// base ORIENTATION end-to-end: (1) the gravity-rotation a_grav = R(base_rot)^T g
// in the velocity integrator, and (2) the quaternion POSE integrator (position'
// = pos + R(base_rot)*v_lin*dt ; rotation' = normalize(QuatMul(base_rot, dq))).
// We seed a loss on base_pose' and FD-validate d(loss)/d{base_rot_pre (4 quat
// comps), v_root_pre angular (3), tau}. The base is REORIENTING: non-identity
// base_rot_pre + nonzero root angular velocity, so the orientation genuinely
// changes over the step and the channel is exercised. gravity ON.
// ===========================================================================
TEST(AbaReverse, RungF_Floating_OrientationChannel) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFloatingChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;  // {1,2}
    const float g0 = 9.81f, dt = 0.02f;
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f};
    const std::vector<float> qd0 = {0.0f, 0.40f, -0.25f};
    const std::vector<float> tau0 = {0.0f, 0.50f, -0.30f};
    // Reorienting root: nonzero angular (first 3) + linear (last 3) velocity.
    const std::array<float, 6> rootv0 = {0.6f, -0.4f, 0.5f, 0.20f, -0.15f, 0.12f};
    // NON-IDENTITY base orientation (tilt about a generic axis) so the a_grav ->
    // rotation grad is not in the g||body-z null and the QuatMul path is full-rank.
    const Transform base_pose0{Vec3{0.5f, -0.3f, 0.2f},
                               Quat::FromAxisAngle(Vec3{0.3f, 0.6f, -0.2f}, 0.7f)};

    // loss = sum_k wqd[k]*qdot'_k + sum_p wpos[p]*position'_p + sum_r wrot[r]*rotation'_r
    std::vector<float> wqd(h.N(), 0.0f);
    wqd[1] = 0.3f; wqd[2] = -0.2f;
    const std::array<float, 3> wpos = {0.7f, -0.5f, 0.4f};
    const std::array<float, 4> wrot = {0.6f, -0.45f, 0.35f, -0.55f};
    auto loss_post = [&]() {
        const std::vector<float> qd = h.GetQdot();
        const Transform pose = h.GetBasePose();
        float s = 0.0f;
        for (uint32_t k : dofs) s += wqd[k] * qd[k];
        s += wpos[0] * pose.position.x + wpos[1] * pose.position.y +
             wpos[2] * pose.position.z;
        s += wrot[0] * pose.rotation.w + wrot[1] * pose.rotation.x +
             wrot[2] * pose.rotation.y + wrot[3] * pose.rotation.z;
        return s;
    };
    auto set_state = [&](const Transform& bp) {
        h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rootv0);
        h.SetBasePose(bp);
    };

    // -- Reverse: seed grad_qdot' = wqd, grad base_pose' = [wpos, wrot]. --
    set_state(base_pose0);
    h.ForwardFloating(g0, dt);
    h.SnapshotPre(q0, qd0);
    h.ZeroGradSeeds();
    h.SeedGradQdot(wqd);
    h.SeedGradBasePose({wpos[0], wpos[1], wpos[2], wrot[0], wrot[1], wrot[2], wrot[3]});
    auto in = h.MakeInputs(dt, g0);
    in.has_drive = false; in.has_integrate = true; in.enable_q_channel = false;
    h.RunBackward(in, h.MakeGrads());
    const std::vector<float> grad_tau = h.GetGradTau();
    const std::array<float, 7> grad_bp = h.GetGradBasePose();  // [pos(3), quat(4)]
    const std::array<float, 6> grad_rootv = h.GetGradRootVel();

    const float eps = 1e-3f;
    // FD d/d(base_rot_pre): perturb each of the 4 quaternion components of the
    // PRE-step base orientation and re-run the forward (the forward reads this as
    // base_pose[0].rotation for BOTH the a_grav rotate and the pose integrator).
    std::array<float, 4> fd_brot{};
    for (uint32_t c = 0u; c < 4u; ++c) {
        auto p_brot = [&](float d) {
            Transform bp = base_pose0;
            float* qc = &bp.rotation.w;  // w,x,y,z contiguous in math::Quat
            qc[c] += d;
            set_state(bp);
            h.ForwardFloating(g0, dt);
            return loss_post();
        };
        fd_brot[c] = (p_brot(eps) - p_brot(-eps)) / (2.0f * eps);
    }
    // FD d/d(v_root_pre): all 6 components (angular feeds dq + a_grav-coupled
    // velocity; linear feeds v_lin_world + the velocity channel).
    std::array<float, 6> fd_rootv{};
    for (uint32_t c = 0u; c < 6u; ++c) {
        auto p_rv = [&](float d) {
            std::array<float, 6> rv = rootv0; rv[c] += d;
            h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rv);
            h.SetBasePose(base_pose0);
            h.ForwardFloating(g0, dt);
            return loss_post();
        };
        fd_rootv[c] = (p_rv(eps) - p_rv(-eps)) / (2.0f * eps);
    }
    // FD d/d(tau).
    std::vector<float> fd_tau(h.N(), 0.0f);
    for (uint32_t k : dofs) {
        auto p_tau = [&](float d) {
            std::vector<float> t = tau0; t[k] += d;
            h.SetQ(q0); h.SetQdot(qd0); h.SetTau(t); h.SetRootVel(rootv0);
            h.SetBasePose(base_pose0);
            h.ForwardFloating(g0, dt);
            return loss_post();
        };
        fd_tau[k] = (p_tau(eps) - p_tau(-eps)) / (2.0f * eps);
    }

    // Compare. The base-rotation grad (4 quat comps) is the headline channel.
    std::vector<float> gbr(grad_bp.begin() + 3, grad_bp.end());  // quat part
    std::vector<float> fbr(fd_brot.begin(), fd_brot.end());
    std::vector<uint32_t> q_idx = {0u, 1u, 2u, 3u};
    const float e_brot = MaxAbsOrRel(gbr, fbr, q_idx, /*abs_tol=*/2e-3f,
                                     /*rel_tol=*/1e-3f);
    std::vector<float> grv(grad_rootv.begin(), grad_rootv.end());
    std::vector<float> frv(fd_rootv.begin(), fd_rootv.end());
    std::vector<uint32_t> rv_idx = {0u, 1u, 2u, 3u, 4u, 5u};
    const float e_rootv = MaxAbsOrRel(grv, frv, rv_idx, /*abs_tol=*/2e-3f,
                                      /*rel_tol=*/1e-3f);
    const float e_tau = MaxRelErr(grad_tau, fd_tau, dofs);
    // Position grad is an identity passthrough of the seed (loss-linear) -> exact.
    const float pos_err = std::fabs(grad_bp[0] - wpos[0]) +
                          std::fabs(grad_bp[1] - wpos[1]) +
                          std::fabs(grad_bp[2] - wpos[2]);
    printf("[rung f orientation] d/d base_rot score=%.3e  d/d v_root score=%.3e  "
           "d/d tau rel=%.3e  pos passthrough err=%.3e\n"
           "  grad_brot=[%.4f %.4f %.4f %.4f] fd=[%.4f %.4f %.4f %.4f]\n"
           "  grad_rootv=[%.4f %.4f %.4f %.4f %.4f %.4f]\n"
           "  fd_rootv  =[%.4f %.4f %.4f %.4f %.4f %.4f]\n",
           e_brot, e_rootv, e_tau, pos_err,
           grad_bp[3], grad_bp[4], grad_bp[5], grad_bp[6],
           fd_brot[0], fd_brot[1], fd_brot[2], fd_brot[3],
           grad_rootv[0], grad_rootv[1], grad_rootv[2], grad_rootv[3], grad_rootv[4], grad_rootv[5],
           fd_rootv[0], fd_rootv[1], fd_rootv[2], fd_rootv[3], fd_rootv[4], fd_rootv[5]);
    EXPECT_LT(e_brot, 1.0f) << "base-orientation adjoint vs FD (abs OR rel)";
    EXPECT_LT(e_rootv, 1.0f) << "v_root adjoint vs FD (orientation-coupled)";
    EXPECT_LT(e_tau, 3e-3f);
    EXPECT_LT(pos_err, 1e-6f) << "position grad must be an exact identity passthrough";
}

// ===========================================================================
// p08b -- ORIENTATION LINEARIZATION-POINT guard (twin variant). All three
// orientation reads (a_grav rotate, v_lin_world rotate, QuatMul) use the PRE-step
// base orientation; the pose integrator overwrites base_pose in place (pre->post),
// so the reverse MUST read in.base_pose_pre, NOT the live (post) base_pose. With a
// LARGE dt + large angular velocity the post orientation differs from pre by an
// O(1) rotation, so linearizing at the wrong point misses FD by a wide margin.
// The fix (snapshot) must match FD AND beat the bug (live-post) variant. Mirrors
// RungC_Floating_LargeDt_LinearizationPoint for the orientation channel.
// ===========================================================================
TEST(AbaReverse, RungF_Floating_OrientationLinearizationPoint) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFloatingChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;
    const float g0 = 9.81f, dt = 0.08f;  // LARGE dt -> big |base_rot_post - pre|
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f};
    const std::vector<float> qd0 = {0.0f, 0.40f, -0.25f};
    const std::vector<float> tau0 = {0.0f, 0.50f, -0.30f};
    // AMPLIFIED root angular velocity -> the quaternion rotates substantially.
    const std::array<float, 6> rootv0 = {5.0f, -4.0f, 6.0f, 0.20f, -0.15f, 0.12f};
    const Transform base_pose0{Vec3{0.5f, -0.3f, 0.2f},
                               Quat::FromAxisAngle(Vec3{0.3f, 0.6f, -0.2f}, 0.7f)};

    std::vector<float> wqd(h.N(), 0.0f);
    wqd[1] = 0.3f; wqd[2] = -0.2f;
    const std::array<float, 3> wpos = {0.7f, -0.5f, 0.4f};
    const std::array<float, 4> wrot = {0.6f, -0.45f, 0.35f, -0.55f};
    auto loss_post = [&]() {
        const std::vector<float> qd = h.GetQdot();
        const Transform pose = h.GetBasePose();
        float s = 0.0f;
        for (uint32_t k : dofs) s += wqd[k] * qd[k];
        s += wpos[0] * pose.position.x + wpos[1] * pose.position.y +
             wpos[2] * pose.position.z;
        s += wrot[0] * pose.rotation.w + wrot[1] * pose.rotation.x +
             wrot[2] * pose.rotation.y + wrot[3] * pose.rotation.z;
        return s;
    };
    auto set_state = [&]() {
        h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rootv0);
        h.SetBasePose(base_pose0);
    };

    // Sanity: confirm the post orientation really differs from pre by O(1).
    set_state();
    h.ForwardFloating(g0, dt);
    const Transform post = h.GetBasePose();
    float dq_mag = std::fabs(post.rotation.w - base_pose0.rotation.w) +
                   std::fabs(post.rotation.x - base_pose0.rotation.x) +
                   std::fabs(post.rotation.y - base_pose0.rotation.y) +
                   std::fabs(post.rotation.z - base_pose0.rotation.z);
    EXPECT_GT(dq_mag, 0.15f) << "regime not large enough to discriminate pre vs post";

    // -- Reverse twice: with the pre snapshot (FIX), and with base_pose_pre
    //    pointed at the LIVE (post) base_pose (the BUG). --
    auto run_reverse = [&](bool use_pre) {
        set_state();
        h.ForwardFloating(g0, dt);
        h.SnapshotPre(q0, qd0);
        h.ZeroGradSeeds();
        h.SeedGradQdot(wqd);
        h.SeedGradBasePose({wpos[0], wpos[1], wpos[2], wrot[0], wrot[1], wrot[2], wrot[3]});
        auto in = h.MakeInputs(dt, g0);
        in.has_drive = false; in.has_integrate = true; in.enable_q_channel = false;
        if (!use_pre) {
            // BUG: linearize at the live POST base_pose instead of the snapshot.
            in.base_pose_pre =
                static_cast<const Transform*>(h.View().base_pose);
        }
        h.RunBackward(in, h.MakeGrads());
        return h.GetGradBasePose();
    };
    const auto fix = run_reverse(/*use_pre=*/true);
    const auto bug = run_reverse(/*use_pre=*/false);

    // FD on the 4 quaternion components of the PRE-step base orientation.
    const float eps = 1e-3f;
    std::array<float, 4> fd_brot{};
    for (uint32_t c = 0u; c < 4u; ++c) {
        auto p_brot = [&](float d) {
            Transform bp = base_pose0;
            float* qc = &bp.rotation.w;
            qc[c] += d;
            h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rootv0);
            h.SetBasePose(bp);
            h.ForwardFloating(g0, dt);
            return loss_post();
        };
        fd_brot[c] = (p_brot(eps) - p_brot(-eps)) / (2.0f * eps);
    }
    std::vector<float> fbr(fd_brot.begin(), fd_brot.end());
    std::vector<float> gfix(fix.begin() + 3, fix.end());
    std::vector<float> gbug(bug.begin() + 3, bug.end());
    std::vector<uint32_t> q_idx = {0u, 1u, 2u, 3u};
    // Use abs-OR-rel (the SAME tolerances as the sibling RungF_OrientationChannel
    // test on this identical quantity): the quaternion w-derivative here is a
    // near-zero component (~3e-3) whose central-FD value jitters at the float32
    // truncation floor (~1e-4 at eps=1e-3), so a bare relative metric flags FD
    // noise, not a gradient error. The reverse value is stable and lands dead in
    // the middle of the FD across eps -- the genuine residual is on the O(0.1-0.7)
    // components, which match to ~5 digits. The post-linearization BUG variant
    // still corrupts those large components by ~10% (dq_mag=0.609), far above the
    // abs floor, so the fix-beats-bug discrimination is preserved.
    const float e_fix = MaxAbsOrRel(gfix, fbr, q_idx, /*abs_tol=*/2e-3f,
                                    /*rel_tol=*/1e-3f);
    const float e_bug = MaxAbsOrRel(gbug, fbr, q_idx, /*abs_tol=*/2e-3f,
                                    /*rel_tol=*/1e-3f);
    printf("[rung f lin-pt] |dquat|=%.3f  d/d base_rot score (abs-or-rel): "
           "pre(fix)=%.3e  post(bug)=%.3e\n", dq_mag, e_fix, e_bug);
    EXPECT_LT(e_fix, 1.0f) << "pre-snapshot orientation adjoint must match FD";
    EXPECT_GT(e_bug, 1.0f) << "regime did not exercise the orientation lin-point";
    EXPECT_LT(e_fix * 3.0f, e_bug)
        << "pre snapshot must beat the live-post linearization by a wide margin";
}

// ===========================================================================
// p08b -- ZERO-SEED ORIENTATION INVARIANT. The orientation channel must be a
// PURE ADDITION gated by the base_pose' cotangent: with base_pose_pre NON-NULL
// (orientation_on path live) but the grad_base_pose_out seed = 0, the reverse
// MUST produce grad_v_root / grad_qdot / grad_tau / grad_mass that are
// BYTE-IDENTICAL to the legacy base_pose_pre = NULL path. This permanently
// protects the v0.5-shipped velocity channel and forces out any seed-independent
// structural leak in the orientation reverse. Seeds the velocity channel (wqd +
// wrv) so the legacy path produces nonzero output to compare against.
// ===========================================================================
TEST(AbaReverse, RungF_Floating_ZeroSeedOrientationInvariant) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFloatingChain();
    Harness h(ctx, model);
    const float g0 = 9.81f, dt = 0.05f;  // large dt -> exercises orientation paths
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f};
    const std::vector<float> qd0 = {0.0f, 0.40f, -0.25f};
    const std::vector<float> tau0 = {0.0f, 0.50f, -0.30f};
    const std::array<float, 6> rootv0 = {0.6f, -0.4f, 0.5f, 0.20f, -0.15f, 0.12f};
    const Transform base_pose0{Vec3{0.5f, -0.3f, 0.2f},
                               Quat::FromAxisAngle(Vec3{0.3f, 0.6f, -0.2f}, 0.7f)};
    std::vector<float> wqd(h.N(), 0.0f);
    wqd[1] = 0.7f; wqd[2] = -0.5f;
    const std::array<float, 6> wrv = {0.3f, -0.4f, 0.6f, -0.2f, 0.5f, -0.35f};

    // Run the reverse with the velocity channel seeded (wqd + wrv) and the
    // base_pose' cotangent left at ZERO. `orient` selects whether base_pose_pre is
    // passed (orientation_on live) or nulled (legacy velocity-only path).
    auto run = [&](bool orient) {
        h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rootv0);
        h.SetBasePose(base_pose0);
        h.ForwardFloating(g0, dt);
        h.SnapshotPre(q0, qd0);
        h.ZeroGradSeeds();              // grad_base_pose_out seed = 0
        h.SeedGradQdot(wqd);
        h.SeedGradRootVel(wrv);
        auto in = h.MakeInputs(dt, g0);
        in.has_drive = false; in.has_integrate = true; in.enable_q_channel = false;
        if (!orient) in.base_pose_pre = nullptr;  // legacy: orientation channel OFF
        auto g = h.MakeGrads();
        if (!orient) g.grad_base_pose_out = nullptr;
        h.RunBackward(in, g);
        std::vector<float> out = h.GetGradTau();
        const std::vector<float> gm = h.GetGradMass();
        out.insert(out.end(), gm.begin(), gm.end());
        const std::vector<float> gqd = h.GetGradQdot();
        out.insert(out.end(), gqd.begin(), gqd.end());
        const std::array<float, 6> grv = h.GetGradRootVel();
        out.insert(out.end(), grv.begin(), grv.end());
        return out;
    };
    const std::vector<float> legacy = run(/*orient=*/false);
    const std::vector<float> zeroseed = run(/*orient=*/true);
    ASSERT_EQ(legacy.size(), zeroseed.size());
    // Byte-identical: the zero-seed orientation path must add EXACTLY nothing to
    // the velocity channel (grad_v_root / grad_qdot / grad_tau / grad_mass).
    for (size_t i = 0u; i < legacy.size(); ++i) {
        uint32_t b1 = 0u, b2 = 0u;
        std::memcpy(&b1, &legacy[i], sizeof(uint32_t));
        std::memcpy(&b2, &zeroseed[i], sizeof(uint32_t));
        EXPECT_EQ(b1, b2) << "zero-seed orientation leaked into velocity channel "
                             "at flat index " << i;
    }
}

// ===========================================================================
// D1 (floating base): the floating-base reverse (LDLT + new stages) is bit-exact
// across two runs. Compares grad_tau + grad_mass + grad_link_velocity[0:6].
// ===========================================================================
TEST(AbaReverse, D1_Floating_TwoRunBitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFloatingChain();
    Harness h(ctx, model);
    const std::vector<uint32_t>& dofs = model.dof_links;
    const float g0 = 9.81f, dt = 0.01f;
    const std::vector<float> q0 = {0.0f, 0.30f, -0.45f};
    const std::vector<float> qd0 = {0.0f, 0.40f, -0.25f};
    const std::vector<float> tau0 = {0.0f, 0.50f, -0.30f};
    const std::array<float, 6> rootv0 = {0.10f, -0.05f, 0.08f, 0.20f, -0.15f, 0.12f};
    std::vector<float> wqd(h.N(), 0.0f);
    wqd[1] = 0.7f; wqd[2] = -0.5f;
    const std::array<float, 6> wrv = {0.3f, -0.4f, 0.6f, -0.2f, 0.5f, -0.35f};
    // p08b: reset the in-place-mutated base_pose so both runs start identically.
    const Transform bp0 = h.GetBasePose();

    auto run = [&]() {
        h.SetQ(q0); h.SetQdot(qd0); h.SetTau(tau0); h.SetRootVel(rootv0);
        h.SetBasePose(bp0);
        h.ForwardFloating(g0, dt);
        h.SnapshotPre(q0, qd0);
        h.ZeroGradSeeds();
        h.SeedGradQdot(wqd);
        h.SeedGradRootVel(wrv);
        auto in = h.MakeInputs(dt, g0);
        in.has_drive = false; in.has_integrate = true; in.enable_q_channel = false;
        h.RunBackward(in, h.MakeGrads());
        std::vector<float> out = h.GetGradTau();
        const std::vector<float> gm = h.GetGradMass();
        out.insert(out.end(), gm.begin(), gm.end());
        const std::array<float, 6> grv = h.GetGradRootVel();
        out.insert(out.end(), grv.begin(), grv.end());
        (void)dofs;
        return out;
    };
    const std::vector<float> r1 = run();
    const std::vector<float> r2 = run();
    ASSERT_EQ(r1.size(), r2.size());
    for (size_t i = 0u; i < r1.size(); ++i) {
        uint32_t b1 = 0u, b2 = 0u;
        std::memcpy(&b1, &r1[i], sizeof(uint32_t));
        std::memcpy(&b2, &r2[i], sizeof(uint32_t));
        EXPECT_EQ(b1, b2) << "floating byte mismatch at " << i;
    }
}

// ===========================================================================
// D1: the backward is bit-exact across two runs on the same inputs (no float
// atomics, fixed order). Compare grad_action + grad_mass byte-for-byte.
// ===========================================================================
TEST(AbaReverse, D1_TwoRunBitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    Harness h(ctx, model);
    const float g0 = 9.81f, dt = 0.01f;
    const std::vector<float> kp = {0.0f, 25.0f, 20.0f, 15.0f};
    const std::vector<float> kd = {0.0f, 1.5f, 1.2f, 0.9f};
    const std::vector<float> flim = {0.0f, 200.0f, 200.0f, 200.0f};
    const std::vector<float> targets = {0.0f, 0.45f, -0.30f, 0.35f};
    std::vector<float> wq(h.N(), 0.0f), wqd(h.N(), 0.0f);
    wq[1] = 0.6f; wq[2] = -0.9f; wq[3] = 0.4f;
    wqd[1] = -0.3f; wqd[2] = 0.7f; wqd[3] = -0.5f;

    auto run = [&]() {
        h.SetQ(kBaseQ); h.SetQdot(kBaseQdot);
        h.ForwardFullStep(targets, kp, kd, flim, g0, dt);
        h.SnapshotPre(kBaseQ, kBaseQdot);
        h.ZeroGradSeeds();
        h.SeedGradQ(wq); h.SeedGradQdot(wqd);
        auto in = h.MakeInputs(dt, g0);
        in.has_drive = true; in.has_integrate = true; in.enable_q_channel = true;
        OwnedDeviceBuffer bt(targets.size() * sizeof(float));
        bt.CopyFromHost(targets.data(), targets.size() * sizeof(float));
        OwnedDeviceBuffer bp(kp.size() * sizeof(float));
        bp.CopyFromHost(kp.data(), kp.size() * sizeof(float));
        OwnedDeviceBuffer bd(kd.size() * sizeof(float));
        bd.CopyFromHost(kd.data(), kd.size() * sizeof(float));
        OwnedDeviceBuffer bl(flim.size() * sizeof(float));
        bl.CopyFromHost(flim.data(), flim.size() * sizeof(float));
        in.drive_targets = static_cast<const float*>(bt.Data());
        in.drive_stiffness = static_cast<const float*>(bp.Data());
        in.drive_damping = static_cast<const float*>(bd.Data());
        in.drive_force_limits = static_cast<const float*>(bl.Data());
        h.RunBackward(in, h.MakeGrads());
        std::vector<float> out = h.GetGradTarget();
        const std::vector<float> gm = h.GetGradMass();
        out.insert(out.end(), gm.begin(), gm.end());
        return out;
    };

    const std::vector<float> r1 = run();
    const std::vector<float> r2 = run();
    ASSERT_EQ(r1.size(), r2.size());
    for (size_t i = 0u; i < r1.size(); ++i) {
        uint32_t b1 = 0u, b2 = 0u;
        std::memcpy(&b1, &r1[i], sizeof(uint32_t));
        std::memcpy(&b2, &r2[i], sizeof(uint32_t));
        EXPECT_EQ(b1, b2) << "byte mismatch at " << i;
    }
}

// ===========================================================================
// M3b nk-arena seam smoke: the SAME single-step adjoint (rung-a configuration)
// run on nk::World ARENA-backed state through
// MakeArticulationDeviceStateFromViews — the diffsim pointer-source seam the
// M3 plan row mandates ("指针源改 arena diff 字段，算法不动"). The forward
// kernels AND StepBackward run UNTOUCHED; only the pointers change. Gate:
// qddot + grad_tau + grad_qdot BIT-EXACT vs the legacy-buffer harness.
// (The full c_abi repoint — WorldRecord holding an nk::World, Tape segments in
// the arena Tape buffer, backward op family in NkOp — is the M9 switch.)
// ===========================================================================

#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"

TEST(AbaReverse, NkArenaSeamForwardReverseBitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    ChainModel model = BuildFixedChain();
    const float g0 = 9.81f;
    const uint32_t n = model.link_count;
    std::vector<float> w(n, 0.0f);
    w[1] = 0.7f; w[2] = -1.1f; w[3] = 0.5f;

    // --- legacy-buffer reference (the rung-a reverse, verbatim). -------------
    Harness h(ctx, model);
    h.SetQ(kBaseQ); h.SetQdot(kBaseQdot); h.SetTau(kBaseTau);
    h.ForwardAbaOnly(g0);
    const std::vector<float> ref_qddot = h.GetQddot();
    h.SnapshotPre(kBaseQ, kBaseQdot);
    h.ZeroGradSeeds();
    h.SeedQddot(w);
    auto in_ref = h.MakeInputs(0.0f, g0);
    in_ref.has_drive = false; in_ref.has_integrate = false;
    in_ref.enable_q_channel = false;
    in_ref.grad_qddot_seed = h.GradQddotSeedPtr();
    h.RunBackward(in_ref, h.MakeGrads());
    const std::vector<float> ref_grad_tau = h.GetGradTau();
    const std::vector<float> ref_grad_qdot = h.GetGradQdot();

    // --- nk::World arena-backed run. -----------------------------------------
    nuka::phi::Device* dev = nuka::phi::InitBestDevice();
    ASSERT_NE(dev, nullptr);
    nuka::phi::Backend* backend = nuka::phi::DeviceInitBackend(dev, nullptr);
    ASSERT_NE(backend, nullptr);

    // Transcribe the SAME host state into an nk::Model (the CookToModel
    // articulation transcription, programmatic).
    nuka::nk::Model nk_model;
    {
        auto& cap = nk_model.capacities;
        cap.env_count = 1;
        cap.links_per_env = n;
        cap.dofs_per_env = 3;          // three revolute DOFs
        cap.max_contacts_per_env = 4;  // arena sizing only (no contacts here)
        cap.max_rows_per_env = 12;
        auto& a = nk_model.articulation;
        const auto& host = model.host;
        a.link_count = n;
        a.parent_link = host.parent_link;
        a.joint_axis = host.joint_axis;
        a.parent_offset = host.parent_offset;
        a.link_local_pose = host.link_local_pose;
        a.link_inertial_frame = host.link_inertial_frame;
        a.joint_damping = host.joint_damping;
        a.joint_armature = host.joint_armature;
        a.initial_q = host.q;
        a.initial_link_pose = host.link_pose;
        a.base_pose = host.base_pose.front();
        a.link_body = host.link_body;
        for (auto jt : host.joint_type) {
            a.joint_type.push_back(static_cast<uint8_t>(jt));
        }
        a.link_inertia_spatial.resize(static_cast<size_t>(n) * 36u);
        for (uint32_t l = 0; l < n; ++l) {
            for (uint32_t k = 0; k < 36u; ++k) {
                a.link_inertia_spatial[static_cast<size_t>(l) * 36u + k] =
                    host.link_inertia[l].I[k];
            }
        }
        a.dof_count = 3;
    }
    nuka::nk::World world(std::move(nk_model), 1u, dev, backend);
    ASSERT_TRUE(world.Ready());

    // THE SEAM: legacy kernel-parameter view over the nk arena.
    articulation::ArticulationDeviceState state =
        articulation::MakeArticulationDeviceStateFromViews(
            world.ModelViewRef(), world.DataViewRef(), n, 1u);

    auto upload = [&](float* dst, const std::vector<float>& v) {
        ASSERT_EQ(cudaMemcpy(dst, v.data(), v.size() * sizeof(float),
                             cudaMemcpyHostToDevice), cudaSuccess);
    };
    upload(state.q, kBaseQ);
    upload(state.qdot, kBaseQdot);
    upload(state.tau, kBaseTau);

    // Forward (the UNTOUCHED legacy launcher on arena-backed state).
    articulation::FeatherstoneAba::ComputeAccelerations(ctx, state, g0);
    ctx.stream.Synchronize();
    std::vector<float> nk_qddot(n, 0.0f);
    ASSERT_EQ(cudaMemcpy(nk_qddot.data(), state.qddot, n * sizeof(float),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    // Reverse (StepBackward UNTOUCHED) with its own scratch grad buffers.
    auto make_buf = [&](size_t count) {
        OwnedDeviceBuffer b(count * sizeof(float));
        std::vector<float> z(count, 0.0f);
        b.CopyFromHost(z.data(), z.size() * sizeof(float));
        return b;
    };
    OwnedDeviceBuffer q_pre = make_buf(n), qdot_pre = make_buf(n);
    OwnedDeviceBuffer v_root_pre = make_buf(static_cast<size_t>(n) * 6u);
    OwnedDeviceBuffer base_pose_pre(sizeof(Transform));
    OwnedDeviceBuffer grad_q = make_buf(n), grad_qdot = make_buf(n);
    OwnedDeviceBuffer grad_target = make_buf(n), grad_mass = make_buf(n);
    OwnedDeviceBuffer grad_tau = make_buf(n);
    OwnedDeviceBuffer grad_seed = make_buf(n);
    OwnedDeviceBuffer grad_link_vel = make_buf(static_cast<size_t>(n) * 6u);
    OwnedDeviceBuffer grad_base_pose = make_buf(7);
    const std::vector<float> dIdm =
        diffsim::BuildSpatialInertiaMassJacobian(model.mass_params);
    OwnedDeviceBuffer dI_dmass(dIdm.size() * sizeof(float));
    dI_dmass.CopyFromHost(dIdm.data(), dIdm.size() * sizeof(float));
    q_pre.CopyFromHost(kBaseQ.data(), n * sizeof(float));
    qdot_pre.CopyFromHost(kBaseQdot.data(), n * sizeof(float));
    grad_seed.CopyFromHost(w.data(), n * sizeof(float));

    diffsim::StepBackwardInputs in;
    in.q_pre = static_cast<const float*>(q_pre.Data());
    in.qdot_pre = static_cast<const float*>(qdot_pre.Data());
    in.v_root_pre = static_cast<const float*>(v_root_pre.Data());
    in.base_pose_pre = static_cast<const Transform*>(base_pose_pre.Data());
    in.dI_dmass = static_cast<const float*>(dI_dmass.Data());
    in.dt = 0.0f;
    in.gravity_z = g0;
    in.has_drive = false; in.has_integrate = false; in.enable_q_channel = false;
    in.grad_qddot_seed = static_cast<const float*>(grad_seed.Data());

    diffsim::StepBackwardGrads g;
    g.grad_q_out = static_cast<float*>(grad_q.Data());
    g.grad_qdot_out = static_cast<float*>(grad_qdot.Data());
    g.grad_target_out = static_cast<float*>(grad_target.Data());
    g.grad_mass_out = static_cast<float*>(grad_mass.Data());
    g.grad_tau_out = static_cast<float*>(grad_tau.Data());
    g.grad_link_velocity_out = static_cast<float*>(grad_link_vel.Data());
    g.grad_base_pose_out = static_cast<float*>(grad_base_pose.Data());

    diffsim::StepBackward(ctx, state, in, g);
    ctx.stream.Synchronize();

    std::vector<float> nk_grad_tau(n, 0.0f), nk_grad_qdot(n, 0.0f);
    grad_tau.CopyToHost(nk_grad_tau.data(), n * sizeof(float));
    grad_qdot.CopyToHost(nk_grad_qdot.data(), n * sizeof(float));

    // BIT-EXACT: the arena pointer source must not change a single float.
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(nk_qddot[i], ref_qddot[i]) << "qddot[" << i << "]";
        EXPECT_EQ(nk_grad_tau[i], ref_grad_tau[i]) << "grad_tau[" << i << "]";
        EXPECT_EQ(nk_grad_qdot[i], ref_grad_qdot[i]) << "grad_qdot[" << i << "]";
    }

    nuka::phi::BackendFree(backend);
}
