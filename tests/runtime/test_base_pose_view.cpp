// ---------------------------------------------------------------------------
// [p03 episode-boundary fix] Authoritative un-lagged base-pose view: correct
// IMMEDIATELY after reset_envs (no step), while the FK-lagged link_pose root
// slot is unchanged.
//
// A vectorized RL env must return the POST-reset observation for a terminated
// env. The base-orientation term (projected_gravity) reads the root world quat.
// ARTICULATION_LINK_POSE's root slot is FK-lagged one step (stage-4 FK from the
// pre-integrate base; q/base advanced at stage 11 with no later FK), so right
// after reset_envs it still shows the pre-reset (fallen) orientation -- wrong for
// the just-reset env. The engine's authoritative base_pose[articulation] buffer
// (the one the integrator advances and reset restores) has no such lag.
//
// This test exercises base_pose via BatchedArticulatedWorld::View().base_pose --
// the SAME buffer NUKA_FIELD_BASE_POSE aliases through the C ABI -- proving:
//   (1) IMMEDIATELY after reset_envs([r]) (NO step): View().base_pose[r] is
//       BIT-EXACT to the creation-time snapshot (upright init), so the un-lagged
//       view is correct with no step.
//   (2) In the SAME state, the FK-lagged link_pose ROOT slot for env r still
//       holds the DIVERGED (fallen) pose -- i.e. the "old" lagged view behaviour
//       is unchanged by the reset (reset does not refresh link_pose). The
//       contrast between (1) and (2) is the whole point: base_pose is the source
//       a post-reset obs must read.
//   (3) ISOLATION: base_pose of every non-reset env is BIT-EXACT to its diverged
//       value (reset_envs touched only env r).
//
// Construction MIRRORS tests/runtime/test_batched_reset.cpp (cook go2_float,
// foot derivation, ground seating, replicated PD drive).
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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
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

// Device->host copy of the per-env root link_pose (the FK-lagged view's root
// slot, env-major at stride base_link_count). Reads View().link_pose directly
// (the same device buffer NUKA_FIELD_ARTICULATION_LINK_POSE aliases).
std::vector<Transform> DownloadRootLinkPose(
    const articulation::ArticulationDeviceState& state, uint32_t env_count,
    uint32_t base_link_count) {
    std::vector<Transform> all(static_cast<size_t>(env_count) * base_link_count);
    EXPECT_EQ(cudaMemcpy(all.data(), state.link_pose,
                         all.size() * sizeof(Transform), cudaMemcpyDeviceToHost),
              cudaSuccess);
    std::vector<Transform> roots(env_count);
    for (uint32_t e = 0u; e < env_count; ++e) {
        roots[e] = all[static_cast<size_t>(e) * base_link_count];
    }
    return roots;
}

std::vector<Transform> DownloadBasePose(
    const articulation::ArticulationDeviceState& state, uint32_t env_count) {
    std::vector<Transform> roots(env_count);
    EXPECT_EQ(cudaMemcpy(roots.data(), state.base_pose,
                         roots.size() * sizeof(Transform), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return roots;
}

}  // namespace

// ---------------------------------------------------------------------------
// base_pose un-lagged immediately after reset_envs; lagged link_pose unchanged.
// ---------------------------------------------------------------------------
TEST(BasePoseView, AuthoritativeUnlaggedAfterResetEnvs) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    const uint32_t base_link_count = host.TotalLinkCount();
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);
    const float ground_height = SeatGround(context, cooked);

    constexpr uint32_t kEnvCount = 8u;
    auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
    gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                    ground_height);
    auto targets = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
    auto stiffness = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
    auto damping = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
    auto limits = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
    auto params = MakeStepParams(targets, stiffness, damping, limits);

    const articulation::ArticulationDeviceState state = bw.View();

    // Snapshot the creation-time (upright init) base_pose BEFORE any step.
    context.stream.Synchronize();
    const std::vector<Transform> init_base = DownloadBasePose(state, kEnvCount);

    // Diverge: drive the free-floating Go2 far from the seated init.
    const uint32_t kDivergeSteps = 200u;
    for (uint32_t s = 0u; s < kDivergeSteps; ++s) {
        bw.Step(params);
    }
    context.stream.Synchronize();
    const std::vector<Transform> diverged_base = DownloadBasePose(state, kEnvCount);
    const std::vector<Transform> diverged_root_link =
        DownloadRootLinkPose(state, kEnvCount, base_link_count);

    // Sanity: the base actually moved away from init (else the test is vacuous).
    bool moved = false;
    for (uint32_t e = 0u; e < kEnvCount; ++e) {
        if (!TransformBitEqual(diverged_base[e], init_base[e])) {
            moved = true;
            break;
        }
    }
    ASSERT_TRUE(moved) << "base_pose did not diverge from init -- test is vacuous";

    // --- reset_envs([2]), NO step --------------------------------------------
    const uint32_t reset_env = 2u;
    const uint32_t ids[1] = {reset_env};
    bw.ResetEnvs(ids, 1u);
    context.stream.Synchronize();

    const std::vector<Transform> after_base = DownloadBasePose(state, kEnvCount);
    const std::vector<Transform> after_root_link =
        DownloadRootLinkPose(state, kEnvCount, base_link_count);

    // (1) base_pose[reset_env] is BIT-EXACT to the upright init -- no step needed.
    EXPECT_TRUE(TransformBitEqual(after_base[reset_env], init_base[reset_env]))
        << "authoritative base_pose not restored to the upright init by reset_envs";

    // (2) the FK-lagged link_pose ROOT slot for reset_env is UNCHANGED by reset
    // (it still holds the diverged/fallen pose -- reset does not refresh FK). This
    // is exactly the staleness the un-lagged base_pose view exists to bypass.
    EXPECT_TRUE(TransformBitEqual(after_root_link[reset_env],
                                  diverged_root_link[reset_env]))
        << "link_pose root slot changed across reset_envs -- expected it to stay "
           "the lagged (diverged) pose";
    // And the two views genuinely DIFFER for the reset env (the upright init quat
    // vs the diverged quat) -- so reading base_pose vs link_pose is observable.
    EXPECT_FALSE(TransformBitEqual(after_base[reset_env], after_root_link[reset_env]))
        << "base_pose and lagged link_pose coincide for the reset env -- the test "
           "cannot distinguish un-lagged from lagged";

    // (3) ISOLATION: base_pose of every non-reset env unchanged by the reset.
    size_t iso_diff = 0u;
    for (uint32_t e = 0u; e < kEnvCount; ++e) {
        if (e == reset_env) continue;
        if (!TransformBitEqual(after_base[e], diverged_base[e])) ++iso_diff;
    }
    EXPECT_EQ(iso_diff, 0u)
        << "reset_envs perturbed a non-reset env's base_pose -- isolation broken";
}
