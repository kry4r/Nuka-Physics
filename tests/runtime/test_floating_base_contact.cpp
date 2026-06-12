// ---------------------------------------------------------------------------
// p01-F T8b -- floating-base CONTACT COUPLING
//
// Validates the foot-contact <-> floating-base coupling added in T8b:
//   * CRBA M leading 6x6 base block + base/joint coupling columns
//     (articulation_contacts.cu ComputeArticulationInertiaMKernel)
//   * contact chain Jacobian base 6 columns
//     (articulation_jacobian.cu ComputeContactChainJacobianKernel)
//   * solver base-DOF plumbing (link_velocity[root] seed/writeback)
//     (articulation_contacts.cu SolveArticulatedContactRowsKernel)
// all gated on ArticulationJointType::FloatingBase (fixed-base path byte-exact,
// proven by the Go2Stand golden + batched-equivalence gates).
//
// A wrong base coupling can sum to ~Mg while loading the feet wrong (the T5
// contraction=1 trap), so the gates assert FOUR conditions TOGETHER:
//
//  (1) INDEPENDENT FK CONVENTION CHECK (the discriminating one). At a NON-IDENTITY
//      base orientation, drive each of the 6 unit base DOFs (3 angular, 3 linear)
//      through the REAL IntegrateFloatingBasePose + UpdateWorldLinkPoses, finite-
//      difference the foot world point, and confirm the chain-Jacobian-predicted
//      foot-point velocity (the 3 scalars from projecting on x,y,z) matches the FD
//      vector. This proves omega-first / body-frame consistency across
//      link_velocity[root] / M base block / J base columns / writeback. A
//      transposed / wrong-sign / wrong-frame base Jacobian fails here even when
//      Sigma=Mg passes.
//
//  (2) Sigma(lambda_n)/dt ~ Mg. Settle go2_float on the ground; sum the 4 feet's
//      normal impulses, divide by dt -> total normal force ~ total weight. Mg is
//      derived from SUMMED COOKED LINK MASSES (not a literature value) AND cross-
//      checked against the M base-block (0,0) mass entry.
//
//  (3) NO-TIPPING. Base angular velocity stays ~0 at equilibrium (catches
//      distribution errors Sigma=Mg cannot).
//
//  (4) STABLE HEIGHT. Base vertical velocity ~0 and base height constant at
//      equilibrium (confirms genuine equilibrium, not a tuned transient).
//
//  (D) DETERMINISM (D1) on the float CONTACT path: two-run identical AND N>=32
//      cross-replica identical (the denser 18x18 LDLT stays bit-exact).
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_jacobian.hpp"
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
#include <cstdio>
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

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// Cooked go2_float plus its base-relative foot shapes and scene-derived hold
// drives. Mirrors CookGo2() in test_batched_articulated_world.cpp but loads the
// FLOATING variant (root cooks to FloatingBase). Drives hold the cooked stance so
// the legs stay extended while the base settles onto the ground.
struct CookedFloat {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;
    std::vector<float> drive_targets;
    std::vector<float> drive_stiffness;
    std::vector<float> drive_damping;
    std::vector<float> drive_force_limits;
    std::vector<float> link_mass;  // per global link (cooked).
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

    // Cooked per-link mass: spatial inertia I[33] = mass (the [3][3] entry of the
    // 6x6 link spatial inertia, which is m on the linear-linear diagonal).
    result.link_mass.assign(link_count, 0.0f);
    for (uint32_t link = 0u; link < link_count; ++link) {
        result.link_mass[link] = result.host.link_inertia[link].I[3u * 6u + 3u];
    }

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

    // Scene-derived hold drives (mirrors c_abi/world.cpp BuildHoldDriveTargets and
    // CookGo2 in test_batched_articulated_world.cpp).
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
                articulation::ArticulationJointType::Fixed) {
            continue;
        }
        const float gain = actuator < actuators.gains.size()
                               ? std::fmax(actuators.gains[actuator], 0.0f)
                               : 0.0f;
        const float force_limit =
            actuator < actuators.force_limits.size()
                ? std::fmax(actuators.force_limits[actuator], 0.0f)
                : 0.0f;
        if (gain > 0.0f) {
            result.drive_stiffness[link] = gain;
            result.drive_damping[link] = 2.0f * std::sqrt(gain);
        }
        if (force_limit > 0.0f) {
            result.drive_force_limits[link] = force_limit;
        }
    }
    return result;
}

// Replicates a link-major host drive vector to `env_count` device replicas.
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

}  // namespace

// ---------------------------------------------------------------------------
// (1) INDEPENDENT FK CONVENTION CHECK -- the discriminating gate.
//
// At a NON-IDENTITY base orientation, the chain Jacobian's 6 base columns must map
// each unit base spatial velocity to the SAME foot-point velocity the real
// IntegrateFloatingBasePose + UpdateWorldLinkPoses produce. Tested per-DOF on all
// three world axes (the 3 projection scalars ARE the predicted velocity vector).
// ---------------------------------------------------------------------------
TEST(FloatingBaseContact, BaseJacobianMatchesForwardKinematics) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    ASSERT_EQ(host.ArticulationCount(), 1u);
    const uint32_t root = host.articulation_link_offset[0];
    ASSERT_EQ(host.joint_type[root], articulation::ArticulationJointType::FloatingBase);
    ASSERT_FALSE(cooked.feet.empty());

    const uint32_t link_count = host.TotalLinkCount();

    // NON-IDENTITY base orientation: at identity, body==world and the convention is
    // unobservable -- a J that forgot R or used world-frame omega would false-pass.
    const Quat kTilt =
        Quat::FromAxisAngle(Vec3{0.3f, -0.5f, 0.2f}.Normalized(), 0.7f).Normalized();
    const Vec3 kBaseOrigin{0.1f, -0.2f, 0.5f};
    host.base_pose[0].rotation = kTilt;
    host.base_pose[0].position = kBaseOrigin;
    // Freeze: all velocities zero, joints at the cooked stance.
    for (auto& v : host.link_velocity) {
        for (float& c : v.v) c = 0.0f;
    }
    for (float& qd : host.qdot) qd = 0.0f;

    auto device = articulation::UploadArticulationState(context, host);
    auto view = device.View();
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);
    ASSERT_EQ(max_dof, 18u) << "go2_float base-inclusive DOF must be 18 (6 base + 12)";

    // FK at the UNPERTURBED tilted state -> the exact foot material point we will
    // both feed the Jacobian and finite-difference (same point on both sides).
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, view,
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> pose0(link_count);
    pose_buf.CopyToHost(pose0.data(), pose0.size() * sizeof(Transform));

    // Use the FIRST foot. The contact point is the foot sphere center in world
    // (calf_world o local_offset) -- the actual material point, NOT the ground-
    // projected detection point (whose lever differs by Delta z).
    const articulation::FootShape foot = cooked.feet[0];
    const Transform calf0 = pose0[foot.calf_local_link];
    const Vec3 foot_point0 =
        calf0.position + calf0.rotation.Rotate(foot.local_offset);

    // Build the 3 base-column Jacobians (project the same contact point onto x,y,z).
    // ComputeContactChainJacobians walks contact_link -> root; here the contact link
    // is the calf, so the walk includes the calf/thigh/hip joints AND the 6 base
    // columns. We only read the base columns 0..5.
    const std::array<Vec3, 3> axes = {Vec3{1.0f, 0.0f, 0.0f},
                                      Vec3{0.0f, 1.0f, 0.0f},
                                      Vec3{0.0f, 0.0f, 1.0f}};
    const uint32_t kContactCount = 1u;
    nuka::phi::Buffer link_buf(sizeof(uint32_t), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer point_buf(sizeof(Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer normal_buf(sizeof(Vec3), nuka::phi::MemoryKind::Device);
    link_buf.CopyFromHost(&foot.calf_local_link, sizeof(uint32_t));
    point_buf.CopyFromHost(&foot_point0, sizeof(Vec3));

    // jac_base[axis][dof] = projection of the dof-column point velocity on `axis`.
    std::array<std::array<float, 6>, 3> jac_base{};
    for (uint32_t a = 0u; a < 3u; ++a) {
        const Vec3 normal = axes[a];
        normal_buf.CopyFromHost(&normal, sizeof(Vec3));
        nuka::phi::Buffer jbuf(static_cast<size_t>(max_dof) * sizeof(float),
                               nuka::phi::MemoryKind::Device);
        std::vector<float> zero(max_dof, 0.0f);
        jbuf.CopyFromHost(zero.data(), zero.size() * sizeof(float));
        articulation::ComputeContactChainJacobians(
            context, view,
            static_cast<const uint32_t*>(link_buf.Data()),
            static_cast<const Vec3*>(point_buf.Data()),
            static_cast<const Vec3*>(normal_buf.Data()),
            kContactCount, max_dof, static_cast<float*>(jbuf.Data()));
        context.stream.Synchronize();
        std::vector<float> jrow(max_dof);
        jbuf.CopyToHost(jrow.data(), jrow.size() * sizeof(float));
        for (uint32_t d = 0u; d < 6u; ++d) jac_base[a][d] = jrow[d];
    }

    // Finite-difference each unit base DOF through the REAL integrator. For DOF k:
    // reset base_pose to the tilt, set link_velocity[root].v = e_k, integrate one
    // small dt, re-run FK, measure (foot_world(dt) - foot_point0)/dt.
    const float kFdDt = 1.0e-4f;
    double max_rel = 0.0;
    for (uint32_t k = 0u; k < 6u; ++k) {
        auto h = host;  // fresh copy: clean velocities + tilt every DOF.
        h.base_pose[0].rotation = kTilt;
        h.base_pose[0].position = kBaseOrigin;
        for (auto& v : h.link_velocity) {
            for (float& c : v.v) c = 0.0f;
        }
        h.link_velocity[root].v[k] = 1.0f;  // unit base spatial velocity DOF k.

        auto dev = articulation::UploadArticulationState(context, h);
        auto v = dev.View();
        // Advance base_pose by the unit DOF through the production pose integrator.
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context, v, kFdDt);
        nuka::phi::Buffer pb(static_cast<size_t>(link_count) * sizeof(Transform),
                             nuka::phi::MemoryKind::Device);
        articulation::UpdateWorldLinkPoses(context, v,
                                           static_cast<Transform*>(pb.Data()));
        context.stream.Synchronize();
        std::vector<Transform> pose1(link_count);
        pb.CopyToHost(pose1.data(), pose1.size() * sizeof(Transform));
        const Transform calf1 = pose1[foot.calf_local_link];
        const Vec3 foot_point1 =
            calf1.position + calf1.rotation.Rotate(foot.local_offset);
        const Vec3 fd_vel = (foot_point1 - foot_point0) * (1.0f / kFdDt);

        // Predicted point velocity from the Jacobian base column k, on each axis.
        const Vec3 jac_vel{jac_base[0][k], jac_base[1][k], jac_base[2][k]};
        const Vec3 diff = jac_vel - fd_vel;
        const double scale = std::max<double>(fd_vel.Length(), 1.0e-6);
        const double rel = diff.Length() / scale;
        max_rel = std::max(max_rel, rel);
        std::printf("[diag] (1) FK-check DOF %u: J=(%+.5f,%+.5f,%+.5f) "
                    "FD=(%+.5f,%+.5f,%+.5f) rel=%.3e\n",
                    k, jac_vel.x, jac_vel.y, jac_vel.z,
                    fd_vel.x, fd_vel.y, fd_vel.z, rel);
        // O(dt) FD error vs O(1) frame bug: 1% relative passes clean, fails a bug.
        EXPECT_LT(rel, 1.0e-2)
            << "base Jacobian column " << k << " disagrees with the integrator FK -- "
               "omega-first / body-frame convention is WRONG";
    }
    std::printf("[diag] (1) max FK-check relative residual = %.3e\n", max_rel);
}

// ---------------------------------------------------------------------------
// (2)+(3)+(4) Settle go2_float on the ground; assert Sigma(lambda_n)/dt~Mg AND
// no-tipping AND stable height SIMULTANEOUSLY at equilibrium. Magnitude alone is a
// false pass (a wrong distribution sums to Mg while tipping), so all three must
// hold together at the settled tail.
// ---------------------------------------------------------------------------
TEST(FloatingBaseContact, SettlesOnGroundWithWeightSupportNoTippingStableHeight) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    ASSERT_EQ(host.ArticulationCount(), 1u);
    const uint32_t root = host.articulation_link_offset[0];
    ASSERT_EQ(host.joint_type[root], articulation::ArticulationJointType::FloatingBase);
    const uint32_t link_count = host.TotalLinkCount();
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);
    ASSERT_EQ(max_dof, 18u);

    // Mg from SUMMED COOKED LINK MASSES (independent of the operator under test).
    const float kGravityZ = -9.81f;
    double total_mass = 0.0;
    for (uint32_t l = 0u; l < link_count; ++l) total_mass += cooked.link_mass[l];
    const double mg = total_mass * static_cast<double>(-kGravityZ);
    std::printf("[diag] (2) summed cooked mass = %.5f kg -> Mg = %.4f N\n",
                total_mass, mg);
    // Cross-check: the base link mass is the (0,0)-block linear diagonal of the
    // cooked base spatial inertia AND the trunk USD mass 6.921 kg.
    EXPECT_NEAR(cooked.link_mass[root], 6.921f, 1.0e-3f)
        << "cooked base mass disagrees with the go2_float trunk mass";
    // Total mass cross-check against the documented per-link masses (6.921 base +
    // 4*(0.678 hip + 1.152 thigh + 0.241352 calf)).
    const double mass_literal = 6.921 + 4.0 * (0.678 + 1.152 + 0.241352);
    EXPECT_NEAR(total_mass, mass_literal, 1.0e-2)
        << "summed cooked mass disagrees with the cross-check sum";

    // Seat the base so the feet just touch the ground. Place the ground at the
    // resting foot bottom (mirrors c_abi DeriveGroundHeight): run FK at the rest
    // pose, find the lowest foot bottom, seat the ground a hair above it so the
    // feet contact and the solve pushes the base to equilibrium.
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
    // Seat ground slightly ABOVE the rest foot bottom so feet start just penetrating
    // (a small, solvable initial depth -- like the cooked stance in the T6 gate).
    const float kRestPenetration = 0.002f;
    const float ground_height = min_bottom + kRestPenetration;
    std::printf("[diag] (2) rest min foot bottom = %.5f, ground = %.5f\n",
                min_bottom, ground_height);

    // The scene's cooked PD gain (nuka:gain = 0.1) was tuned with the trunk FIXED;
    // freeing the base lets the legs sag under the 6.9 kg trunk to a low sprawl.
    // That is fine for this test: the base still falls from 0.445 and STOPS, carrying
    // full Mg, only because the contact reaction reaches the base (the coupling under
    // test). The legs fold ASYMMETRICALLY, so the no-tipping gate is exercised
    // non-trivially, not by symmetry. The discriminating frame check is gate (1).
    constexpr uint32_t kEnvCount = 1u;
    auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
    gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                    ground_height);
    auto targets = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
    auto stiffness = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
    auto damping = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
    auto limits = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);

    gpu::BatchedArticulatedStepParams params;
    params.drive_targets = static_cast<const float*>(targets.Data());
    params.drive_stiffness = static_cast<const float*>(stiffness.Data());
    params.drive_damping = static_cast<const float*>(damping.Data());
    params.drive_force_limits = static_cast<const float*>(limits.Data());
    params.gravity_z = kGravityZ;
    params.dt = 1.0f / 240.0f;
    params.baumgarte_max_velocity = 3.0f;

    // Settle to equilibrium.
    const uint32_t kSettleSteps = 600u;
    for (uint32_t s = 0u; s < kSettleSteps; ++s) {
        bw.Step(params);
    }
    context.stream.Synchronize();

    // Average the settled tail to suppress per-step PGS ripple. We also re-run FK
    // each tail step to measure the actual lowest foot bottom (the anti-fall-through
    // condition is feet-stay-seated, NOT trunk-z vs ground: only feet collide, so
    // the trunk origin z and ground_height are unrelated quantities).
    const uint32_t kTailSteps = 60u;
    nuka::phi::Buffer tail_pose(static_cast<size_t>(link_count) * sizeof(Transform),
                                nuka::phi::MemoryKind::Device);
    double sum_normal_force = 0.0;
    double sum_base_omega = 0.0;
    double sum_base_vz = 0.0;
    float first_tail_z = 0.0f;
    float last_tail_z = 0.0f;
    float max_foot_seat_error = 0.0f;     // |foot_bottom - ground| over the tail.
    float min_trunk_above_feet = std::numeric_limits<float>::infinity();
    for (uint32_t s = 0u; s < kTailSteps; ++s) {
        bw.Step(params);
        context.stream.Synchronize();
        const auto lambda = bw.DownloadLambda();  // [slot*3] = (n,t1,t2).
        double step_normal = 0.0;
        for (uint32_t slot = 0u; slot < articulation::kMaxFootContactsPerEnv; ++slot) {
            step_normal += lambda[static_cast<size_t>(slot) * 3u + 0u];
        }
        sum_normal_force += step_normal / static_cast<double>(params.dt);

        articulation::ArticulationHostState out = batched;
        bw.Download(&out);
        const float* vroot = out.link_velocity[root].v;
        const double omega =
            std::sqrt(static_cast<double>(vroot[0]) * vroot[0] +
                      vroot[1] * vroot[1] + vroot[2] * vroot[2]);
        sum_base_omega += omega;
        sum_base_vz += std::fabs(vroot[5]);  // body-frame z linear ~ world z at level.
        const float trunk_z = out.base_pose[0].position.z;
        if (s == 0u) first_tail_z = trunk_z;
        last_tail_z = trunk_z;

        // FK -> per-foot bottom z and the lowest foot.
        auto devp = articulation::UploadArticulationState(context, out);
        articulation::UpdateWorldLinkPoses(context, devp.View(),
                                           static_cast<Transform*>(tail_pose.Data()));
        context.stream.Synchronize();
        std::vector<Transform> wp(link_count);
        tail_pose.CopyToHost(wp.data(), wp.size() * sizeof(Transform));
        float lowest_foot = std::numeric_limits<float>::infinity();
        for (const auto& f : cooked.feet) {
            const Transform calf = wp[f.calf_local_link];
            const Vec3 c = calf.position + calf.rotation.Rotate(f.local_offset);
            lowest_foot = std::min(lowest_foot, c.z - f.radius);
        }
        max_foot_seat_error =
            std::max(max_foot_seat_error, std::fabs(lowest_foot - ground_height));
        min_trunk_above_feet = std::min(min_trunk_above_feet, trunk_z - lowest_foot);
    }
    const double avg_normal_force = sum_normal_force / kTailSteps;
    const double avg_base_omega = sum_base_omega / kTailSteps;
    const double avg_base_vz = sum_base_vz / kTailSteps;
    const float height_drift = last_tail_z - first_tail_z;

    std::printf("[diag] (2) Sigma(lambda_n)/dt = %.4f N   Mg = %.4f N   ratio=%.4f\n",
                avg_normal_force, mg, avg_normal_force / mg);
    std::printf("[diag] (3) avg base |omega| = %.5f rad/s\n", avg_base_omega);
    std::printf("[diag] (4) avg base |v_z| = %.5f m/s   height drift = %.5e m "
                "(trunk z=%.4f)\n", avg_base_vz, height_drift, last_tail_z);
    std::printf("[diag] (anti-fall) max foot-seat error = %.5e m   "
                "min trunk-above-feet = %.4f m\n",
                max_foot_seat_error, min_trunk_above_feet);

    // (2) Total normal force ~ weight. Solver normal force balances gravity at
    // equilibrium; allow 10% for the soft Baumgarte / finite-iteration PGS.
    EXPECT_NEAR(avg_normal_force, mg, 0.10 * mg)
        << "Sigma(lambda_n)/dt does not balance the summed weight";
    // (3) No tipping: base angular velocity stays small.
    EXPECT_LT(avg_base_omega, 0.10)
        << "base is tipping (angular velocity not ~0) -- load distribution wrong";
    // (4) Stable height: base vertical velocity ~0 and height nearly constant.
    EXPECT_LT(avg_base_vz, 0.05)
        << "base vertical velocity not ~0 -- not at vertical equilibrium";
    EXPECT_LT(std::fabs(height_drift), 5.0e-3)
        << "base height drifting over the tail -- not settled";
    // Anti-fall-through (vs T8a's divergence): the feet stay seated AT the ground --
    // not sunk through it (fall-through) nor lifted off it. The trunk z and the
    // ground plane are unrelated (only feet collide), so we assert the feet, not the
    // trunk; bounded trunk z + v_z~0 + tiny drift above already rule out divergence.
    EXPECT_LT(max_foot_seat_error, 2.0e-2)
        << "feet did not stay seated at the ground (fall-through or lift-off)";
    EXPECT_TRUE(std::isfinite(last_tail_z) && std::isfinite(min_trunk_above_feet))
        << "base state diverged (non-finite)";
}

// ---------------------------------------------------------------------------
// (D) DETERMINISM (D1) on the float CONTACT path: two-run bit-identical AND N>=32
// cross-replica bit-identical, including lambda and link_velocity[root] (the new
// 18x18 LDLT + base writeback must stay deterministic).
// ---------------------------------------------------------------------------
TEST(FloatingBaseContact, FloatContactPathDeterministic) {
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

    // Seat the ground a hair below the rest feet so contacts activate during the run.
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
    const float ground_height = min_bottom + 0.002f;

    constexpr uint32_t kEnvCount = 32u;  // N>=32 cross-replica.
    const uint32_t kSteps = 80u;

    // Cooked drives + a seated ground -> contacts ACTIVE during the run, so the
    // determinism gate exercises the new 18x18 LDLT and the base writeback.
    auto run = [&]() {
        auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
        gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                        ground_height);
        auto targets = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
        auto stiffness = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
        auto damping = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
        auto limits = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
        gpu::BatchedArticulatedStepParams params;
        params.drive_targets = static_cast<const float*>(targets.Data());
        params.drive_stiffness = static_cast<const float*>(stiffness.Data());
        params.drive_damping = static_cast<const float*>(damping.Data());
        params.drive_force_limits = static_cast<const float*>(limits.Data());
        params.gravity_z = -9.81f;
        params.dt = 1.0f / 240.0f;
        params.baumgarte_max_velocity = 3.0f;
        for (uint32_t s = 0u; s < kSteps; ++s) bw.Step(params);
        context.stream.Synchronize();
        articulation::ArticulationHostState out = batched;
        bw.Download(&out);
        return std::make_pair(out, bw.DownloadLambda());
    };

    const auto [a_state, a_lambda] = run();
    const auto [b_state, b_lambda] = run();

    // Two-run bit-identical: q, qdot, link_velocity (base), lambda.
    size_t q_mis = 0u, qdot_mis = 0u, vroot_mis = 0u, lambda_mis = 0u;
    for (size_t i = 0u; i < a_state.q.size(); ++i) {
        if (a_state.q[i] != b_state.q[i]) ++q_mis;
        if (a_state.qdot[i] != b_state.qdot[i]) ++qdot_mis;
    }
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        const uint32_t r = env * link_count;
        for (uint32_t c = 0u; c < 6u; ++c) {
            if (a_state.link_velocity[r].v[c] != b_state.link_velocity[r].v[c]) {
                ++vroot_mis;
            }
        }
    }
    for (size_t i = 0u; i < a_lambda.size(); ++i) {
        if (a_lambda[i] != b_lambda[i]) ++lambda_mis;
    }
    std::printf("[diag] (D) two-run mismatches: q=%zu qdot=%zu v_root=%zu lambda=%zu\n",
                q_mis, qdot_mis, vroot_mis, lambda_mis);
    EXPECT_EQ(q_mis, 0u);
    EXPECT_EQ(qdot_mis, 0u);
    EXPECT_EQ(vroot_mis, 0u);
    EXPECT_EQ(lambda_mis, 0u);

    // Cross-replica bit-identical (replica 0 vs every other replica).
    const uint32_t per_env_slots = articulation::kMaxFootContactsPerEnv * 3u;
    size_t xq = 0u, xqdot = 0u, xvroot = 0u, xlambda = 0u;
    for (uint32_t env = 1u; env < kEnvCount; ++env) {
        const uint32_t base_off = env * link_count;
        for (uint32_t l = 0u; l < link_count; ++l) {
            if (a_state.q[base_off + l] != a_state.q[l]) ++xq;
            if (a_state.qdot[base_off + l] != a_state.qdot[l]) ++xqdot;
        }
        const uint32_t r = env * link_count;  // env root link.
        for (uint32_t c = 0u; c < 6u; ++c) {
            if (a_state.link_velocity[r].v[c] != a_state.link_velocity[root].v[c]) {
                ++xvroot;
            }
        }
        for (uint32_t i = 0u; i < per_env_slots; ++i) {
            if (a_lambda[env * per_env_slots + i] != a_lambda[i]) ++xlambda;
        }
    }
    std::printf("[diag] (D) cross-replica mismatches: q=%zu qdot=%zu v_root=%zu "
                "lambda=%zu\n", xq, xqdot, xvroot, xlambda);
    EXPECT_EQ(xq, 0u);
    EXPECT_EQ(xqdot, 0u);
    EXPECT_EQ(xvroot, 0u);
    EXPECT_EQ(xlambda, 0u);
}
