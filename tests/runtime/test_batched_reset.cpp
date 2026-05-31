// ---------------------------------------------------------------------------
// [p03] Engine-side per-env RESET primitive (RL autoreset) -- authority +
// isolation + D1 determinism.
//
// A trained-RL vectorized env must reset individual envs to a known initial
// pose when they terminate (autoreset). The probe that motivated this task
// proved a Python-side write through ARTICULATION_LINK_POSE is NON-authoritative
// for a floating base: the integrator overwrites the base pose from its INTERNAL
// base_pose[articulation] buffer after one step. So the reset must write the
// engine's authoritative state. BatchedArticulatedWorld::Reset / ResetEnvs do
// that (snapshot captured at construction -> restored on reset).
//
// Gates (asserted together so no single tautology false-passes):
//   (1) AUTHORITY + ISOLATION across a step. Diverge N>=4 envs (PD-drive a
//       free-floating Go2 so they evolve away from the seated init), snapshot
//       the diverged state, reset_envs([2]):
//         (1a) IMMEDIATELY after reset (no step): env 2's base_pose / q / qdot
//              are BIT-EXACT to the creation-time initial snapshot, and envs
//              0/1/3 are BIT-EXACT to their pre-reset diverged values (isolation).
//         (1b) After ONE more Step(): env 2's base pose is still near the seated
//              init (loose tol) -- it did NOT snap back to the diverged/fallen
//              pose (the bug this fixes: a non-authoritative write would be
//              overwritten by the integrator within the step).
//   (2) RESET ALL. After divergence, reset() returns EVERY env to the initial
//       snapshot (bit-exact base_pose / q / qdot, all envs).
//   (3) D1. Two identical (diverge -> reset_envs -> step) runs are bit-identical
//       on q / qdot / base velocity. The reset is INSIDE the compared sequence,
//       so this proves the reset itself is deterministic.
//
// Construction (cook go2_float, foot derivation, ground seating, the replicated
// PD drive) MIRRORS tests/runtime/test_go2_pd_standing.cpp.
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
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

bool TransformBitEqual(const Transform& a, const Transform& b) {
    return a.position.x == b.position.x && a.position.y == b.position.y &&
           a.position.z == b.position.z && a.rotation.w == b.rotation.w &&
           a.rotation.x == b.rotation.x && a.rotation.y == b.rotation.y &&
           a.rotation.z == b.rotation.z;
}

}  // namespace

// ---------------------------------------------------------------------------
// (1) AUTHORITY + ISOLATION across a step, and (2) RESET ALL.
// ---------------------------------------------------------------------------
TEST(BatchedReset, AuthorityIsolationAndResetAll) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    const uint32_t link_count = host.TotalLinkCount();      // base_link_count.
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);
    const float ground_height = SeatGround(context, cooked);

    constexpr uint32_t kEnvCount = 8u;  // N >= 4.
    auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
    gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                    ground_height);
    auto targets = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
    auto stiffness = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
    auto damping = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
    auto limits = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
    auto params = MakeStepParams(targets, stiffness, damping, limits);

    // The creation-time initial snapshot (env 0 == every env: identical replicas).
    articulation::ArticulationHostState init = batched;
    bw.Download(&init);  // pre-step == cooked init (qdot 0, base at seat, q crouch).

    // Diverge: drive away from the seated init for many steps. With the firm PD
    // the envs settle to the crouch, well away from qdot==0 and the un-stepped
    // base pose, which is plenty of divergence to detect a snap-back.
    const uint32_t kDivergeSteps = 200u;
    for (uint32_t s = 0u; s < kDivergeSteps; ++s) {
        bw.Step(params);
    }
    context.stream.Synchronize();
    articulation::ArticulationHostState diverged = batched;
    bw.Download(&diverged);

    // Sanity: the dynamics actually moved (otherwise the test is vacuous).
    bool moved = false;
    for (uint32_t l = 0u; l < link_count; ++l) {
        if (diverged.q[l] != init.q[l] || diverged.qdot[l] != init.qdot[l]) {
            moved = true;
            break;
        }
    }
    ASSERT_TRUE(moved) << "envs did not diverge from the init -- test is vacuous";

    // --- reset_envs([2]) -----------------------------------------------------
    const uint32_t reset_env = 2u;
    const uint32_t ids[1] = {reset_env};
    bw.ResetEnvs(ids, 1u);
    context.stream.Synchronize();
    articulation::ArticulationHostState after_reset = batched;
    bw.Download(&after_reset);

    // (1a-i) env 2 BIT-EXACT to the initial snapshot (base_pose, q, qdot).
    EXPECT_TRUE(TransformBitEqual(after_reset.base_pose[reset_env],
                                  init.base_pose[reset_env]))
        << "reset env base_pose not bit-exact to the initial snapshot";
    size_t q_diff = 0u, qdot_diff = 0u;
    for (uint32_t l = 0u; l < link_count; ++l) {
        const uint32_t link = reset_env * link_count + l;
        if (after_reset.q[link] != init.q[link]) ++q_diff;
        if (after_reset.qdot[link] != init.qdot[link]) ++qdot_diff;
    }
    EXPECT_EQ(q_diff, 0u) << "reset env q not bit-exact to the initial snapshot";
    EXPECT_EQ(qdot_diff, 0u) << "reset env qdot not bit-exact to the initial snapshot";
    // base spatial velocity restored (== init, which is zero).
    size_t vroot_diff = 0u;
    for (uint32_t c = 0u; c < 6u; ++c) {
        if (after_reset.link_velocity[reset_env * link_count].v[c] !=
            init.link_velocity[reset_env * link_count].v[c]) {
            ++vroot_diff;
        }
    }
    EXPECT_EQ(vroot_diff, 0u) << "reset env base velocity not restored to init";

    // (1a-ii) ISOLATION: every other env BIT-EXACT to its diverged value.
    size_t iso_diff = 0u;
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        if (env == reset_env) continue;
        if (!TransformBitEqual(after_reset.base_pose[env], diverged.base_pose[env])) {
            ++iso_diff;
        }
        for (uint32_t l = 0u; l < link_count; ++l) {
            const uint32_t link = env * link_count + l;
            if (after_reset.q[link] != diverged.q[link]) ++iso_diff;
            if (after_reset.qdot[link] != diverged.qdot[link]) ++iso_diff;
        }
        for (uint32_t c = 0u; c < 6u; ++c) {
            if (after_reset.link_velocity[env * link_count].v[c] !=
                diverged.link_velocity[env * link_count].v[c]) {
                ++iso_diff;
            }
        }
    }
    EXPECT_EQ(iso_diff, 0u)
        << "reset_envs perturbed an un-listed env -- isolation broken";

    // (1b) AUTHORITY ACROSS A STEP: one more Step; env 2's base pose must NOT
    // snap back to the diverged pose -- it stays near the seated init. (A non-
    // authoritative write would be overwritten by the integrator this step.)
    bw.Step(params);
    context.stream.Synchronize();
    articulation::ArticulationHostState stepped = batched;
    bw.Download(&stepped);

    const float init_z = init.base_pose[reset_env].position.z;
    const float diverged_z = diverged.base_pose[reset_env].position.z;
    const float stepped_z = stepped.base_pose[reset_env].position.z;
    // One step from the seated init moves the base by < a few mm; the seated
    // init differs from the diverged crouch by >> that. So "near init, far from
    // diverged" cleanly proves the reset stuck across the step.
    EXPECT_NEAR(stepped_z, init_z, 5.0e-3f)
        << "reset env base z snapped away after a step (reset is non-authoritative)";
    EXPECT_GT(std::fabs(stepped_z - diverged_z), 5.0e-3f)
        << "reset env base z is still at the diverged pose -- reset did not take";

    // (2) RESET ALL: re-diverge, then reset() returns EVERY env to the init.
    for (uint32_t s = 0u; s < kDivergeSteps; ++s) {
        bw.Step(params);
    }
    context.stream.Synchronize();
    bw.Reset();
    context.stream.Synchronize();
    articulation::ArticulationHostState after_all = batched;
    bw.Download(&after_all);

    size_t all_diff = 0u;
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        if (!TransformBitEqual(after_all.base_pose[env], init.base_pose[env])) {
            ++all_diff;
        }
        for (uint32_t l = 0u; l < link_count; ++l) {
            const uint32_t link = env * link_count + l;
            if (after_all.q[link] != init.q[link]) ++all_diff;
            if (after_all.qdot[link] != init.qdot[link]) ++all_diff;
        }
        for (uint32_t c = 0u; c < 6u; ++c) {
            if (after_all.link_velocity[env * link_count].v[c] !=
                init.link_velocity[env * link_count].v[c]) {
                ++all_diff;
            }
        }
    }
    EXPECT_EQ(all_diff, 0u)
        << "reset() did not return every env to the initial snapshot";
}

// ---------------------------------------------------------------------------
// (3) D1: two identical (diverge -> reset_envs -> step) runs are bit-identical.
// The reset is INSIDE the compared sequence -> proves the reset is deterministic.
// ---------------------------------------------------------------------------
TEST(BatchedReset, ResetIsDeterministic) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    const uint32_t link_count = host.TotalLinkCount();
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);
    const float ground_height = SeatGround(context, cooked);

    constexpr uint32_t kEnvCount = 16u;
    const uint32_t kDivergeSteps = 120u;
    const uint32_t kPostSteps = 40u;
    // Reset a fixed subset (the masked path) inside the compared sequence.
    const std::array<uint32_t, 3> reset_ids = {1u, 4u, 9u};

    auto run = [&]() {
        auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
        gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                        ground_height);
        auto targets = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
        auto stiffness = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
        auto damping = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
        auto limits = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
        auto params = MakeStepParams(targets, stiffness, damping, limits);
        for (uint32_t s = 0u; s < kDivergeSteps; ++s) bw.Step(params);
        bw.ResetEnvs(reset_ids.data(), static_cast<uint32_t>(reset_ids.size()));
        for (uint32_t s = 0u; s < kPostSteps; ++s) bw.Step(params);
        context.stream.Synchronize();
        articulation::ArticulationHostState out = batched;
        bw.Download(&out);
        return out;
    };

    const auto a = run();
    const auto b = run();

    size_t q_mis = 0u, qdot_mis = 0u, vroot_mis = 0u, base_mis = 0u;
    for (size_t i = 0u; i < a.q.size(); ++i) {
        if (a.q[i] != b.q[i]) ++q_mis;
        if (a.qdot[i] != b.qdot[i]) ++qdot_mis;
    }
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        const uint32_t r = env * link_count;
        for (uint32_t c = 0u; c < 6u; ++c) {
            if (a.link_velocity[r].v[c] != b.link_velocity[r].v[c]) ++vroot_mis;
        }
        if (!TransformBitEqual(a.base_pose[env], b.base_pose[env])) ++base_mis;
    }
    EXPECT_EQ(q_mis, 0u);
    EXPECT_EQ(qdot_mis, 0u);
    EXPECT_EQ(vroot_mis, 0u);
    EXPECT_EQ(base_mis, 0u);
}
