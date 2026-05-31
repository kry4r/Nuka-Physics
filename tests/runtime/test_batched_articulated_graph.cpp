// ---------------------------------------------------------------------------
// p01-W3 -- CUDA-graph step capture/replay BIT-IDENTICAL gate.
//
// The graph step path (gpu::BatchedArticulatedWorld::StepGraph) captures the
// batched articulated Step() kernel sequence into a CUDA graph once and replays
// the instantiated graph thereafter to amortize per-kernel launch overhead. It
// is a D1-PRESERVING perf optimization: it replays the EXACT same kernels in the
// EXACT same order, so it MUST be BIT-FOR-BIT identical to the non-graph Step().
//
// THE GATE (non-negotiable): run N>=200 steps on a multi-env Go2 world TWICE from
// the SAME initial state -- once via the normal Step(), once via the graph
// StepGraph() -- and assert the resulting q / qdot / base_pose (position +
// rotation) / root link_velocity / persistent lambda buffers are BYTE-IDENTICAL
// (exact float equality, not a tolerance). Any non-zero mismatch is a FAILURE:
// a graph that differs even slightly has regressed D1.
//
// The world / drive / ground setup mirrors test_go2_pd_standing.cpp (the proven
// free-floating Go2 standing PD config) so the captured step exercises the full
// production pipeline incl. an ACTIVE contact solve (feet on the ground), not a
// contacts-inactive no-op.
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <array>
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
namespace gpu = nuka::runtime::gpu;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kKp = 60.0f;
constexpr float kKd = 4.0f;
constexpr float kForceLimit = 24.0f;
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 1.0f / 240.0f;
constexpr float kPoseHip = 0.1f;
constexpr float kPoseThigh = 0.8f;
constexpr float kPoseCalf = -1.5f;
constexpr std::array<float, 4> kHipSign = {1.0f, -1.0f, 1.0f, -1.0f};

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// Cooked free-floating Go2 + foot shapes + standing PD drives. Mirrors
// CookGo2Float() in test_go2_pd_standing.cpp (the proven-stable holdable crouch).
struct CookedFloat {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;
    std::vector<float> drive_targets;
    std::vector<float> drive_stiffness;
    std::vector<float> drive_damping;
    std::vector<float> drive_force_limits;
    std::vector<uint32_t> actuated_links;
};

CookedFloat CookGo2Float() {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);

    CookedFloat result;
    result.host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);
    const uint32_t link_count = result.host.TotalLinkCount();

    const auto& shapes = world.template_view.shape_table;
    for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
        if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) {
            continue;
        }
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = kInvalidLink;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (result.host.link_body[link] == body) {
                calf_link = link;
                break;
            }
        }
        if (calf_link == kInvalidLink) {
            continue;
        }
        articulation::FootShape foot;
        foot.calf_local_link = calf_link;
        foot.local_offset = shape < shapes.local_transforms.size()
                                ? shapes.local_transforms[shape].position
                                : Vec3::Zero();
        foot.radius = shape < shapes.radii.size() ? shapes.radii[shape] : 0.0f;
        result.feet.push_back(foot);
    }

    result.drive_targets = result.host.q;
    result.drive_stiffness.assign(link_count, 0.0f);
    result.drive_damping.assign(link_count, 0.0f);
    result.drive_force_limits.assign(link_count, 0.0f);
    const auto& actuators = world.template_view.actuator_table;
    const auto& joints = world.template_view.joint_table;
    for (uint32_t actuator = 0u;
         actuator < world.template_view.actuator_count; ++actuator) {
        if (actuator >= actuators.joint_ids.size() ||
            actuator >= actuators.types.size() ||
            actuators.types[actuator] != nuka::scene::ActuatorType::Position) {
            continue;
        }
        const auto joint = actuators.joint_ids[actuator];
        if (joint >= joints.child_bodies.size()) {
            continue;
        }
        const auto child_body = joints.child_bodies[joint];
        uint32_t link = kInvalidLink;
        for (uint32_t l = 0u; l < link_count; ++l) {
            if (result.host.link_body[l] == child_body) {
                link = l;
                break;
            }
        }
        if (link == kInvalidLink ||
            result.host.joint_type[link] ==
                articulation::ArticulationJointType::Fixed ||
            result.host.joint_type[link] ==
                articulation::ArticulationJointType::FloatingBase) {
            continue;
        }
        result.drive_stiffness[link] = kKp;
        result.drive_damping[link] = kKd;
        result.drive_force_limits[link] = kForceLimit;
        result.actuated_links.push_back(link);
    }

    for (uint32_t leg = 0u; leg < 4u && leg * 3u + 2u < result.actuated_links.size();
         ++leg) {
        const uint32_t hip_link = result.actuated_links[leg * 3u + 0u];
        const uint32_t thigh_link = result.actuated_links[leg * 3u + 1u];
        const uint32_t calf_link = result.actuated_links[leg * 3u + 2u];
        const float hip_target = kPoseHip * kHipSign[leg];
        result.drive_targets[hip_link] = hip_target;
        result.drive_targets[thigh_link] = kPoseThigh;
        result.drive_targets[calf_link] = kPoseCalf;
        result.host.q[hip_link] = hip_target;
        result.host.q[thigh_link] = kPoseThigh;
        result.host.q[calf_link] = kPoseCalf;
    }
    return result;
}

nuka::phi::Buffer UploadReplicatedDrive(const std::vector<float>& base,
                                        uint32_t env_count) {
    std::vector<float> all(base.size() * env_count);
    for (uint32_t e = 0u; e < env_count; ++e) {
        for (size_t i = 0u; i < base.size(); ++i) {
            all[e * base.size() + i] = base[i];
        }
    }
    nuka::phi::Buffer b(all.size() * sizeof(float), nuka::phi::MemoryKind::Device);
    b.CopyFromHost(all.data(), all.size() * sizeof(float));
    return b;
}

// Seats the ground a hair above the rest foot bottoms (mirrors test_go2_pd_*).
float SeatGround(const nuka::phi::DeviceContext& context, CookedFloat& cooked) {
    auto& host = cooked.host;
    const uint32_t link_count = host.TotalLinkCount();
    host.base_pose[0].rotation = Quat::Identity();
    auto probe = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, probe.View(),
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> rest_pose(link_count);
    pose_buf.CopyToHost(rest_pose.data(), rest_pose.size() * sizeof(Transform));
    float min_bottom = std::numeric_limits<float>::infinity();
    for (const auto& foot : cooked.feet) {
        const Transform calf = rest_pose[foot.calf_local_link];
        const Vec3 center = calf.position + calf.rotation.Rotate(foot.local_offset);
        min_bottom = std::min(min_bottom, center.z - foot.radius);
    }
    constexpr float kRestPenetration = 0.002f;
    return min_bottom + kRestPenetration;
}

gpu::BatchedArticulatedStepParams MakeStepParams(const nuka::phi::Buffer& targets,
                                                 const nuka::phi::Buffer& stiffness,
                                                 const nuka::phi::Buffer& damping,
                                                 const nuka::phi::Buffer& limits) {
    gpu::BatchedArticulatedStepParams params;
    params.drive_targets = static_cast<const float*>(targets.Data());
    params.drive_stiffness = static_cast<const float*>(stiffness.Data());
    params.drive_damping = static_cast<const float*>(damping.Data());
    params.drive_force_limits = static_cast<const float*>(limits.Data());
    params.gravity_z = kGravityZ;
    params.dt = kDt;
    params.baumgarte_max_velocity = 3.0f;
    return params;
}

}  // namespace

// ---------------------------------------------------------------------------
// THE GATE: graph replay is BYTE-IDENTICAL to the non-graph step over N>=200
// steps on a multi-env world from the same initial state.
// ---------------------------------------------------------------------------
TEST(BatchedArticulatedGraph, GraphStepByteIdenticalToNonGraph) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    const uint32_t root = host.articulation_link_offset[0];
    const uint32_t link_count = host.TotalLinkCount();
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);
    const float ground_height = SeatGround(context, cooked);

    constexpr uint32_t kEnvCount = 32u;  // multi-env (proves replication holds too).
    constexpr uint32_t kPhase = 125u;    // per phase; 2*125 = 250 >= 200 mandated.

    // A SECOND drive-target set: the crouch targets nudged. Written into the SAME
    // device buffer mid-run (phase B) to prove the graph reads drive-target buffer
    // CONTENTS at replay time -- NOT a snapshot baked at capture. This is the RL
    // use case (per-step-varying actions through a stable buffer); if the graph
    // ignored the phase-B write, phase B would diverge from the non-graph path and
    // this test would catch it. Same buffer pointer => StepParamsMatch stays true,
    // so phase B genuinely exercises REPLAY-with-new-contents, not a re-capture.
    std::vector<float> targets_b = cooked.drive_targets;
    for (uint32_t link : cooked.actuated_links) {
        targets_b[link] += 0.05f;  // small, holdable nudge off the crouch.
    }

    // run() advances 2*kPhase steps via either Step() or StepGraph() from the SAME
    // cooked initial state; between the two phases it rewrites the drive-target
    // buffer (same pointer) to targets_b. Returns the full downloaded state +
    // lambda. The two worlds are constructed identically and never share buffers.
    auto run = [&](bool use_graph) {
        auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
        gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                        ground_height);
        auto targets = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
        auto stiffness = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
        auto damping = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
        auto limits = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
        auto params = MakeStepParams(targets, stiffness, damping, limits);

        auto step = [&]() {
            if (use_graph) {
                bw.StepGraph(params);
            } else {
                bw.Step(params);
            }
        };
        auto sync = [&]() {
            // Sync the stream the chosen path ran on BEFORE touching the buffer
            // (graph runs on the PRIVATE stream; CopyFromHost is a synchronous
            // default-stream memcpy that does NOT wait on the private stream, so
            // without this sync the rewrite would race the in-flight replay).
            if (use_graph) bw.GraphStreamSynchronize();
            context.stream.Synchronize();
        };

        // Phase A: original crouch targets.
        for (uint32_t s = 0u; s < kPhase; ++s) step();
        sync();
        // Rewrite the SAME drive-target buffer in place to the nudged set.
        {
            std::vector<float> all(targets_b.size() * kEnvCount);
            for (uint32_t e = 0u; e < kEnvCount; ++e) {
                for (size_t i = 0u; i < targets_b.size(); ++i) {
                    all[e * targets_b.size() + i] = targets_b[i];
                }
            }
            targets.CopyFromHost(all.data(), all.size() * sizeof(float));
        }
        // Phase B: replay/step with the NEW buffer contents (pointer unchanged).
        for (uint32_t s = 0u; s < kPhase; ++s) step();
        sync();

        articulation::ArticulationHostState out = batched;
        bw.Download(&out);
        return std::make_pair(out, bw.DownloadLambda());
    };

    const auto [ref_state, ref_lambda] = run(/*use_graph=*/false);
    const auto [gph_state, gph_lambda] = run(/*use_graph=*/true);

    // ---- BYTE-IDENTICAL comparisons (exact float equality, no tolerance). ----
    ASSERT_EQ(ref_state.q.size(), gph_state.q.size());
    ASSERT_EQ(ref_state.qdot.size(), gph_state.qdot.size());
    ASSERT_EQ(ref_lambda.size(), gph_lambda.size());

    size_t q_mis = 0u, qdot_mis = 0u;
    for (size_t i = 0u; i < ref_state.q.size(); ++i) {
        if (ref_state.q[i] != gph_state.q[i]) ++q_mis;
        if (ref_state.qdot[i] != gph_state.qdot[i]) ++qdot_mis;
    }

    // base_pose: byte-compare position (3 floats) + rotation quaternion (4 floats)
    // for every env. memcmp catches any bit difference incl. signed-zero / NaN.
    size_t base_pose_mis = 0u;
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        if (std::memcmp(&ref_state.base_pose[env], &gph_state.base_pose[env],
                        sizeof(Transform)) != 0) {
            ++base_pose_mis;
        }
    }

    // root link_velocity (the floating base's 6-DOF spatial velocity) per env.
    size_t vroot_mis = 0u;
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        const uint32_t r = env * link_count;
        for (uint32_t c = 0u; c < 6u; ++c) {
            if (ref_state.link_velocity[r].v[c] != gph_state.link_velocity[r].v[c]) {
                ++vroot_mis;
            }
        }
    }
    (void)root;

    size_t lambda_mis = 0u;
    for (size_t i = 0u; i < ref_lambda.size(); ++i) {
        if (ref_lambda[i] != gph_lambda[i]) ++lambda_mis;
    }

    std::printf("[graph-gate] envs=%u steps=%u (2 phases, drive targets varied "
                "mid-run) mismatches: q=%zu qdot=%zu base_pose=%zu v_root=%zu "
                "lambda=%zu\n",
                kEnvCount, 2u * kPhase, q_mis, qdot_mis, base_pose_mis, vroot_mis,
                lambda_mis);

    EXPECT_EQ(q_mis, 0u) << "graph q differs from non-graph (D1 regression)";
    EXPECT_EQ(qdot_mis, 0u) << "graph qdot differs from non-graph (D1 regression)";
    EXPECT_EQ(base_pose_mis, 0u)
        << "graph base_pose differs from non-graph (D1 regression)";
    EXPECT_EQ(vroot_mis, 0u)
        << "graph root link_velocity differs from non-graph (D1 regression)";
    EXPECT_EQ(lambda_mis, 0u)
        << "graph lambda differs from non-graph (D1 regression)";

    // Sanity: the run is non-trivial (state actually evolved + contacts active).
    EXPECT_TRUE(std::isfinite(gph_state.base_pose[0].position.z));
    double sum_lambda_n = 0.0;
    for (uint32_t slot = 0u; slot < articulation::kMaxFootContactsPerEnv; ++slot) {
        sum_lambda_n += gph_lambda[static_cast<size_t>(slot) * 3u + 0u];
    }
    EXPECT_GT(sum_lambda_n, 0.0)
        << "no contact impulse -- the captured step did not exercise the solve";
}
