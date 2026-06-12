// ---------------------------------------------------------------------------
// p14a (v0.7) -- per-link contact wrench (NUKA_FIELD_LINK_CONTACT_WRENCH) tests.
//
// Three gates:
//   (1) SYNTHETIC kernel unit test -- the DISCRIMINATING case. Calls
//       nuka::sensor::ComputeContactWrench directly with hand-built per-slot
//       inputs chosen so the friction (t1*lambda_t1 + t2*lambda_t2) is non-zero
//       and NOT parallel to the normal, AND the torque arm r=(point-origin) is NOT
//       parallel to F, so tau=r x F is non-trivial. Proves the friction
//       reconstruction, the cross product, AND the /dt scaling -- none of which the
//       normal-only stance test exercises.
//   (2) Go2 foot-stance INTEGRATION test -- a 64-env free-floating Go2 settles into
//       a stance; assert Sigma over an env's links of F_z ~ total_mass*g, and each
//       foot link's net F_z is positive (upward). Sourced mass+g from the cooked
//       model + the engine's gravity. Mirrors test_go2_pd_standing.cpp.
//   (3) D1 two-run byte-exact -- the stance scenario run twice yields a byte-
//       identical wrench buffer AND per-slot contact-force buffer.
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/gpu/batched_articulated_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "sensor/contact_wrench.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;
namespace gpu = nuka::runtime::gpu;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kInvalidLink = ~0u;
constexpr float kGravityZ = -9.81f;
constexpr float kDt = 1.0f / 240.0f;
constexpr float kKp = 60.0f;
constexpr float kKd = 4.0f;
constexpr float kForceLimit = 24.0f;
constexpr float kPoseHip = 0.1f;
constexpr float kPoseThigh = 0.8f;
constexpr float kPoseCalf = -1.5f;
constexpr std::array<float, 4> kHipSign = {1.0f, -1.0f, 1.0f, -1.0f};

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

}  // namespace

// ---------------------------------------------------------------------------
// (1) SYNTHETIC kernel unit test: friction non-parallel-to-normal + non-trivial
// r x F + /dt scaling, all checked against hand-computed values.
// ---------------------------------------------------------------------------
TEST(LinkContactWrench, SyntheticFrictionAndTorqueExact) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    // One env, base_link_count = 3 (so total_link_count = 3), 4 slots/env.
    constexpr uint32_t kEnvCount = 1u;
    constexpr uint32_t kBlc = 3u;
    constexpr uint32_t kMaxFoot = articulation::kMaxFootContactsPerEnv;  // 4
    constexpr uint32_t kSlotCount = kEnvCount * kMaxFoot;
    constexpr uint32_t kTotalLinks = kEnvCount * kBlc;
    const float dt = 0.01f;
    const float inv_dt = 1.0f / dt;

    // Two contacts on link g=1, one on link g=2, slot 3 inactive. Pick directions
    // so n / t1 / t2 are an orthonormal-ish but NOT axis-aligned basis and the
    // friction is genuinely off-normal; pick points so r x F is non-trivial.
    std::vector<float> lambda(kSlotCount * 3u, 0.0f);
    std::vector<Vec3> point(kSlotCount, Vec3::Zero());
    std::vector<Vec3> normal(kSlotCount, Vec3::Zero());
    std::vector<Vec3> t1(kSlotCount, Vec3::Zero());
    std::vector<Vec3> t2(kSlotCount, Vec3::Zero());
    std::vector<uint32_t> link(kSlotCount, kInvalidLink);
    std::vector<Transform> pose(kTotalLinks, Transform::Identity());

    // Link origins (the torque reference points).
    pose[0].position = Vec3{0.0f, 0.0f, 0.0f};
    pose[1].position = Vec3{1.0f, 2.0f, 3.0f};
    pose[2].position = Vec3{-1.0f, 0.5f, 2.0f};

    auto set_slot = [&](uint32_t s, uint32_t g, Vec3 p, Vec3 n, Vec3 ta1, Vec3 ta2,
                        float ln, float lt1, float lt2) {
        link[s] = g;
        point[s] = p;
        normal[s] = n;
        t1[s] = ta1;
        t2[s] = ta2;
        lambda[s * 3u + 0u] = ln;
        lambda[s * 3u + 1u] = lt1;
        lambda[s * 3u + 2u] = lt2;
    };

    // Use an orthonormal basis NOT aligned to world axes:
    //   n = +Z, t1 = +X, t2 = +Y is too axis-aligned; rotate t1/t2 in the XY plane.
    const float c = std::cos(0.7f), sn = std::sin(0.7f);
    const Vec3 n0{0.0f, 0.0f, 1.0f};
    const Vec3 t1a{c, sn, 0.0f};
    const Vec3 t2a{-sn, c, 0.0f};
    // Slot 0 -> link 1: point offset from link1 origin; nonzero friction.
    set_slot(0u, 1u, Vec3{1.2f, 2.0f, 3.0f}, n0, t1a, t2a, 5.0f, 2.0f, -1.0f);
    // Slot 1 -> link 1: a second contact, different point + friction.
    set_slot(1u, 1u, Vec3{1.0f, 2.3f, 3.1f}, n0, t1a, t2a, 3.0f, -1.5f, 0.5f);
    // Slot 2 -> link 2: a third basis (tilted normal) to exercise non-Z F.
    const Vec3 n1 = Vec3{0.3f, 0.0f, 0.9539392f};  // ~unit-ish, off +Z
    const Vec3 t1b = Vec3{0.0f, 1.0f, 0.0f};
    const Vec3 t2b = n1.Cross(t1b);
    set_slot(2u, 2u, Vec3{-0.5f, 0.5f, 2.4f}, n1, t1b, t2b, 4.0f, 1.0f, 2.0f);
    // Slot 3 inactive (link == ~0u, lambda 0).

    // Hand-compute the expected F and tau for links 1 and 2.
    auto force_slot = [&](uint32_t s) -> Vec3 {
        return (normal[s] * lambda[s * 3u + 0u] + t1[s] * lambda[s * 3u + 1u] +
                t2[s] * lambda[s * 3u + 2u]) * inv_dt;
    };
    Vec3 expF[kTotalLinks];
    Vec3 expT[kTotalLinks];
    for (uint32_t g = 0u; g < kTotalLinks; ++g) {
        expF[g] = Vec3::Zero();
        expT[g] = Vec3::Zero();
    }
    for (uint32_t s = 0u; s < kSlotCount; ++s) {
        if (link[s] == kInvalidLink) continue;
        const uint32_t g = link[s];
        const Vec3 fs = force_slot(s);
        const Vec3 r = point[s] - pose[g].position;
        expF[g] += fs;
        expT[g] += r.Cross(fs);
    }
    // Sanity on the test design: friction must be off-normal and r not parallel F.
    {
        const Vec3 fric = (t1[0] * lambda[1] + t2[0] * lambda[2]);
        ASSERT_GT(fric.Length(), 1.0e-3f) << "test bug: friction is zero";
        ASSERT_GT(std::fabs(fric.Dot(normal[0])) /
                      (fric.Length() * normal[0].Length() + 1e-9f),
                  -1.0f);  // trivially true; documents intent
        ASSERT_GT(expT[1].Length(), 1.0e-3f) << "test bug: torque on link1 is zero";
        ASSERT_GT(expT[2].Length(), 1.0e-3f) << "test bug: torque on link2 is zero";
    }

    // Upload, run the free function, read back.
    nuka::phi::Buffer d_lambda(lambda.size() * sizeof(float));
    nuka::phi::Buffer d_point(point.size() * sizeof(Vec3));
    nuka::phi::Buffer d_normal(normal.size() * sizeof(Vec3));
    nuka::phi::Buffer d_t1(t1.size() * sizeof(Vec3));
    nuka::phi::Buffer d_t2(t2.size() * sizeof(Vec3));
    nuka::phi::Buffer d_link(link.size() * sizeof(uint32_t));
    nuka::phi::Buffer d_pose(pose.size() * sizeof(Transform));
    nuka::phi::Buffer d_force(kSlotCount * 3u * sizeof(float));
    nuka::phi::Buffer d_wrench(kTotalLinks * 6u * sizeof(float));
    d_lambda.CopyFromHost(lambda.data(), lambda.size() * sizeof(float));
    d_point.CopyFromHost(point.data(), point.size() * sizeof(Vec3));
    d_normal.CopyFromHost(normal.data(), normal.size() * sizeof(Vec3));
    d_t1.CopyFromHost(t1.data(), t1.size() * sizeof(Vec3));
    d_t2.CopyFromHost(t2.data(), t2.size() * sizeof(Vec3));
    d_link.CopyFromHost(link.data(), link.size() * sizeof(uint32_t));
    d_pose.CopyFromHost(pose.data(), pose.size() * sizeof(Transform));

    nuka::sensor::ComputeContactWrench(
        context, static_cast<const float*>(d_lambda.Data()),
        static_cast<const Vec3*>(d_point.Data()),
        static_cast<const Vec3*>(d_normal.Data()),
        static_cast<const Vec3*>(d_t1.Data()),
        static_cast<const Vec3*>(d_t2.Data()),
        static_cast<const uint32_t*>(d_link.Data()),
        static_cast<const Transform*>(d_pose.Data()),
        kEnvCount, kBlc, kMaxFoot, dt,
        static_cast<float*>(d_force.Data()),
        static_cast<float*>(d_wrench.Data()));
    context.stream.Synchronize();

    std::vector<float> wrench(kTotalLinks * 6u);
    std::vector<float> force(kSlotCount * 3u);
    d_wrench.CopyToHost(wrench.data(), wrench.size() * sizeof(float));
    d_force.CopyToHost(force.data(), force.size() * sizeof(float));

    // Per-slot contact force == lambda/dt componentwise.
    for (uint32_t s = 0u; s < kSlotCount; ++s) {
        for (uint32_t c = 0u; c < 3u; ++c) {
            EXPECT_NEAR(force[s * 3u + c], lambda[s * 3u + c] * inv_dt, 1.0e-3f)
                << "slot " << s << " comp " << c;
        }
    }
    // Per-link wrench == hand-computed F and tau.
    const float tol = 1.0e-2f;  // forces ~ O(100..1000); relative-ish abs tol.
    for (uint32_t g = 0u; g < kTotalLinks; ++g) {
        EXPECT_NEAR(wrench[g * 6u + 0u], expF[g].x, tol) << "F.x link " << g;
        EXPECT_NEAR(wrench[g * 6u + 1u], expF[g].y, tol) << "F.y link " << g;
        EXPECT_NEAR(wrench[g * 6u + 2u], expF[g].z, tol) << "F.z link " << g;
        EXPECT_NEAR(wrench[g * 6u + 3u], expT[g].x, tol) << "tau.x link " << g;
        EXPECT_NEAR(wrench[g * 6u + 4u], expT[g].y, tol) << "tau.y link " << g;
        EXPECT_NEAR(wrench[g * 6u + 5u], expT[g].z, tol) << "tau.z link " << g;
    }
    // Link 0 has no contact -> all zero.
    for (uint32_t c = 0u; c < 6u; ++c) {
        EXPECT_EQ(wrench[0u * 6u + c], 0.0f) << "link0 should be untouched, comp " << c;
    }
    std::printf("[diag] synthetic: link1 F=(%.2f,%.2f,%.2f) tau=(%.2f,%.2f,%.2f)\n",
                wrench[6], wrench[7], wrench[8], wrench[9], wrench[10], wrench[11]);
}

// ---------------------------------------------------------------------------
// Shared Go2 stance setup (a trimmed copy of test_go2_pd_standing's CookGo2Float
// + SeatGround, just the pieces this test needs).
// ---------------------------------------------------------------------------
namespace {

struct CookedFloat {
    articulation::ArticulationHostState host;
    std::vector<articulation::FootShape> feet;
    std::vector<float> drive_targets;
    std::vector<float> drive_stiffness;
    std::vector<float> drive_damping;
    std::vector<float> drive_force_limits;
    std::vector<float> link_mass;
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

    result.link_mass.assign(link_count, 0.0f);
    for (uint32_t link = 0u; link < link_count; ++link) {
        result.link_mass[link] = result.host.link_inertia[link].I[3u * 6u + 3u];
    }

    const auto& shapes = world.template_view.shape_table;
    for (uint32_t shape = 0u; shape < shapes.types.size(); ++shape) {
        if (shapes.types[shape] != nuka::scene::ShapeType::Sphere) continue;
        const uint32_t body = shapes.body_ids[shape];
        uint32_t calf_link = kInvalidLink;
        for (uint32_t link = 0u; link < link_count; ++link) {
            if (result.host.link_body[link] == body) { calf_link = link; break; }
        }
        if (calf_link == kInvalidLink) continue;
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
        if (joint >= joints.child_bodies.size()) continue;
        const auto child_body = joints.child_bodies[joint];
        uint32_t link = kInvalidLink;
        for (uint32_t l = 0u; l < link_count; ++l) {
            if (result.host.link_body[l] == child_body) { link = l; break; }
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
    for (uint32_t e = 0u; e < env_count; ++e)
        for (size_t i = 0u; i < base.size(); ++i) all[e * base.size() + i] = base[i];
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
    return min_bottom + 0.002f;
}

gpu::BatchedArticulatedStepParams MakeStepParams(const nuka::phi::Buffer& t,
                                                 const nuka::phi::Buffer& s,
                                                 const nuka::phi::Buffer& d,
                                                 const nuka::phi::Buffer& l) {
    gpu::BatchedArticulatedStepParams p;
    p.drive_targets = static_cast<const float*>(t.Data());
    p.drive_stiffness = static_cast<const float*>(s.Data());
    p.drive_damping = static_cast<const float*>(d.Data());
    p.drive_force_limits = static_cast<const float*>(l.Data());
    p.gravity_z = kGravityZ;
    p.dt = kDt;
    p.baumgarte_max_velocity = 3.0f;
    return p;
}

// Copies a device Transform* range into a phi::Buffer (D2D) for host readback of
// the engine's internal link_pose (== the world_pose_ the wrench kernel reads as
// its torque-origin source; D2D-copied at step stage 4).
void CudaPoseCopy(const nuka::phi::DeviceContext& context,
                  const Transform* src, nuka::phi::Buffer& dst, uint32_t count) {
    nuka::phi::ScopedDeviceGuard guard(context.device_id);
    cudaMemcpyAsync(dst.Data(), src, static_cast<size_t>(count) * sizeof(Transform),
                    cudaMemcpyDeviceToDevice, context.stream.Native());
    context.stream.Synchronize();
}

}  // namespace

// ---------------------------------------------------------------------------
// (2) Go2 stance integration: Sigma over an env's links of F_z ~ Mg, foot F_z > 0.
// ---------------------------------------------------------------------------
TEST(LinkContactWrench, Go2StanceBalancesWeight) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    const uint32_t link_count = host.TotalLinkCount();
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);

    double total_mass = 0.0;
    for (uint32_t l = 0u; l < link_count; ++l) total_mass += cooked.link_mass[l];
    const double mg = total_mass * static_cast<double>(-kGravityZ);
    std::printf("[diag] summed cooked mass = %.5f kg -> Mg = %.4f N\n",
                total_mass, mg);

    const float ground_height = SeatGround(context, cooked);
    constexpr uint32_t kEnvCount = 64u;
    auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
    gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                    ground_height);
    auto targets = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
    auto stiffness = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
    auto damping = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
    auto limits = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
    auto params = MakeStepParams(targets, stiffness, damping, limits);

    const uint32_t kSettle = 1500u;
    for (uint32_t s = 0u; s < kSettle; ++s) bw.Step(params);
    context.stream.Synchronize();

    ASSERT_EQ(bw.TotalLinkCount(), kEnvCount * bw.BaseLinkCount());
    const uint32_t blc = bw.BaseLinkCount();

    // Average the wrench over a short settled tail (suppress PGS ripple).
    const uint32_t kTail = 120u;
    std::vector<double> avg_env0_link_Fz(blc, 0.0);
    double avg_sum_Fz = 0.0;
    // Identify env-0 foot links (the calf links from cooked.feet).
    std::array<uint32_t, articulation::kMaxFootContactsPerEnv> foot_links{};
    for (uint32_t i = 0u; i < cooked.feet.size() && i < foot_links.size(); ++i) {
        foot_links[i] = cooked.feet[i].calf_local_link;  // env-0 global == local
    }
    for (uint32_t step = 0u; step < kTail; ++step) {
        bw.Step(params);
        context.stream.Synchronize();
        const auto wrench = bw.DownloadLinkContactWrench();
        double sum_fz = 0.0;
        for (uint32_t l = 0u; l < blc; ++l) {
            const double fz = wrench[(0u * blc + l) * 6u + 2u];  // env 0
            avg_env0_link_Fz[l] += fz;
            sum_fz += fz;
        }
        avg_sum_Fz += sum_fz;
    }
    avg_sum_Fz /= kTail;
    for (auto& v : avg_env0_link_Fz) v /= kTail;

    std::printf("[diag] env0 Sigma F_z = %.4f N  Mg = %.4f N  ratio = %.4f\n",
                avg_sum_Fz, mg, avg_sum_Fz / mg);
    std::printf("[diag] env0 per-foot-link F_z:");
    for (uint32_t i = 0u; i < cooked.feet.size(); ++i) {
        std::printf(" link%u=%.3f", foot_links[i], avg_env0_link_Fz[foot_links[i]]);
    }
    std::printf("\n");

    // Sigma F_z ~ Mg (allow 10% for soft Baumgarte / finite PGS).
    EXPECT_NEAR(avg_sum_Fz, mg, 0.10 * mg)
        << "summed per-link contact F_z does not balance the weight";
    // Each foot link carries positive upward F_z (the ground pushes up). If this is
    // negative, the lambda sign / normal direction is wrong.
    for (uint32_t i = 0u; i < cooked.feet.size(); ++i) {
        EXPECT_GT(avg_env0_link_Fz[foot_links[i]], 0.0)
            << "foot link " << foot_links[i] << " net F_z not positive (upward)";
    }

    // -- WIRING / ARG-ORDER cross-check (advisor #4). The kernel-direct synthetic
    // test does NOT exercise ComputeContactWrenchOutputs's argument list; the
    // Sigma-F_z and D1 gates are blind to a transposed geometry arg (swapping
    // contact_point_<->world_pose_ corrupts only torque; F_z and determinism are
    // unaffected). Here we recompute env-0's per-link contact wrench ON THE HOST
    // from the ENGINE-downloaded per-slot buffers (point/normal/force/link) plus
    // the engine's link-frame origins (state.link_pose == the SAME world_pose_ the
    // kernel reads, D2D-copied at stage 4), and require it to match the engine's
    // own wrench buffer. A point/origin swap shows up as a wrong torque arm; the
    // CONTACT_FORCE == DownloadLambda()/dt check pins the impulse->force scaling. --
    {
        bw.Step(params);
        context.stream.Synchronize();
        const auto wrench = bw.DownloadLinkContactWrench();
        const auto force = bw.DownloadContactForce();   // {Fn,Ft1,Ft2}/dt per slot
        const auto lambda = bw.DownloadLambda();         // {ln,lt1,lt2} per slot
        const auto link = bw.DownloadContactLink();      // global link / slot
        const uint32_t slot_count = bw.SlotCount();
        std::vector<Vec3> point(slot_count), normal(slot_count);
        bw.ContactPointBuffer().CopyToHost(point.data(), point.size() * sizeof(Vec3));
        bw.ContactNormalBuffer().CopyToHost(normal.data(), normal.size() * sizeof(Vec3));
        std::vector<Transform> link_pose(bw.TotalLinkCount());
        // state.link_pose is the engine-internal copy of world_pose_ (the kernel's
        // origin source); read it back as the host origin reference.
        nuka::phi::Buffer pose_view(bw.TotalLinkCount() * sizeof(Transform),
                                    nuka::phi::MemoryKind::Device);
        CudaPoseCopy(context, bw.View().link_pose, pose_view, bw.TotalLinkCount());
        pose_view.CopyToHost(link_pose.data(), link_pose.size() * sizeof(Transform));

        // (a) CONTACT_FORCE == lambda/dt componentwise (the impulse->force scale).
        size_t force_mismatch = 0u;
        for (size_t i = 0u; i < force.size(); ++i) {
            if (std::fabs(force[i] - lambda[i] / kDt) > 1.0e-2f) ++force_mismatch;
        }
        EXPECT_EQ(force_mismatch, 0u)
            << "CONTACT_FORCE[slot] != DownloadLambda()[slot]/dt";

        // (a2) Per-contact ENUMERATION identity (the mj-parity readback surface).
        // CONTACT_LINK aliases ContactLinkBuffer (the C-ABI returns its Data()
        // pointer directly), so DownloadContactLink() IS that buffer; assert the
        // active (link != ~0u) slots are real links < total_link_count, and that
        // CONTACT_NORMAL on an active slot is the world +Z ground normal (the rigid
        // foot-vs-ground detector). Inactive slots carry link == ~0u and zero data.
        size_t active_slots = 0u;
        for (uint32_t s = 0u; s < slot_count; ++s) {
            if (link[s] == kInvalidLink) continue;
            ++active_slots;
            EXPECT_LT(link[s], bw.TotalLinkCount())
                << "CONTACT_LINK[" << s << "] out of range";
            EXPECT_NEAR(normal[s].z, 1.0f, 1.0e-4f)
                << "CONTACT_NORMAL[" << s << "].z not +Z (ground normal)";
            EXPECT_NEAR(normal[s].x, 0.0f, 1.0e-4f);
            EXPECT_NEAR(normal[s].y, 0.0f, 1.0e-4f);
        }
        EXPECT_GT(active_slots, 0u) << "no active contacts in a settled stance";

        // (b) Host recompute of env-0 per-link wrench from downloaded per-slot data.
        // The NORMAL-channel world force is normal*Fn (Fn == force[base+0]); the
        // FRICTION world direction needs the tangents (not exposed), so for the
        // F_z and torque cross-check we use the dominant normal channel: in stance
        // the normal is +Z so F_z == Sum Fn and the normal-force torque arm is
        // (point-origin) x (n*Fn). A point/origin swap corrupts THIS arm.
        const uint32_t kMaxFoot = articulation::kMaxFootContactsPerEnv;
        for (uint32_t g = 0u; g < blc; ++g) {  // env 0 global link == local
            Vec3 host_F = Vec3::Zero();
            Vec3 host_Tn = Vec3::Zero();  // NORMAL-channel torque (point-origin)x(n*Fn)
            float fn_sum = 0.0f;          // for the friction-magnitude guard below
            const Vec3 origin = link_pose[g].position;
            for (uint32_t slot = 0u; slot < kMaxFoot; ++slot) {  // env 0 slots
                if (link[slot] != g) continue;
                const float Fn = force[slot * 3u + 0u];
                const Vec3 fn_world = normal[slot] * Fn;  // normal channel only
                host_F += fn_world;
                host_Tn += (point[slot] - origin).Cross(fn_world);
                fn_sum += std::fabs(Fn);
            }
            // F_z is friction-independent (normal == +Z), so the engine wrench F_z
            // must equal the host normal-channel recompute exactly (loose tol for
            // PGS ripple).
            const double eFz = wrench[g * 6u + 2u];
            EXPECT_NEAR(eFz, host_F.z, 0.05 * std::fabs(eFz) + 1.0e-3)
                << "engine F_z vs host normal recompute mismatch, link " << g;
            // Torque arm cross-check (the point/origin-swap detector): the engine
            // torque includes a SMALL friction contribution we cannot recompute
            // (tangents unexposed), so compare the engine tau against the
            // normal-channel host tau with a tolerance scaled by the contact's
            // force magnitude * a generous arm. A contact_point_<->world_pose_ swap
            // would put the arm ~|origin| off -> tau wrong by ~Fn*|origin| >> tol.
            if (fn_sum > 1.0e-3f) {
                const double tau_tol = 0.20 * fn_sum + 1.0e-2;  // N*m, friction slack
                for (uint32_t c = 0u; c < 3u; ++c) {
                    EXPECT_NEAR(wrench[g * 6u + 3u + c],
                                (c == 0 ? host_Tn.x : c == 1 ? host_Tn.y : host_Tn.z),
                                tau_tol)
                        << "engine tau vs host normal-channel recompute, link " << g
                        << " comp " << c << " (point/origin arm swap?)";
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (3) D1: two runs of the stance scenario yield byte-identical wrench + force.
// ---------------------------------------------------------------------------
TEST(LinkContactWrench, D1ByteExactTwoRuns) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);
    const float ground_height = SeatGround(context, cooked);
    constexpr uint32_t kEnvCount = 32u;
    const uint32_t kSteps = 200u;

    auto run = [&]() {
        auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
        gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                        ground_height);
        auto t = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
        auto s = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
        auto d = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
        auto l = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
        auto params = MakeStepParams(t, s, d, l);
        for (uint32_t step = 0u; step < kSteps; ++step) bw.Step(params);
        context.stream.Synchronize();
        return std::make_pair(bw.DownloadLinkContactWrench(),
                              bw.DownloadContactForce());
    };

    const auto [a_w, a_f] = run();
    const auto [b_w, b_f] = run();
    ASSERT_EQ(a_w.size(), b_w.size());
    ASSERT_EQ(a_f.size(), b_f.size());
    size_t w_mis = 0u, f_mis = 0u;
    for (size_t i = 0u; i < a_w.size(); ++i)
        if (std::memcmp(&a_w[i], &b_w[i], sizeof(float)) != 0) ++w_mis;
    for (size_t i = 0u; i < a_f.size(); ++i)
        if (std::memcmp(&a_f[i], &b_f[i], sizeof(float)) != 0) ++f_mis;
    std::printf("[diag] D1 two-run mismatches: wrench=%zu force=%zu\n", w_mis, f_mis);
    EXPECT_EQ(w_mis, 0u);
    EXPECT_EQ(f_mis, 0u);
}

// ---------------------------------------------------------------------------
// (4) StepGraph (RL graph-replay) path produces the SAME wrench + force as
// Step(). The eager readout is launched at the END of StepKernels() (the graph
// twin) through the `ctx` parameter so the launches are CAPTURED into the graph;
// this test proves that path actually runs AND is byte-identical to the non-graph
// Step() readout (dt is baked into the captured kernel arg; StepParamsMatch
// compares dt so a replay always divides by the correct dt). RL trains through
// StepGraph, so this is the production-critical surface.
// ---------------------------------------------------------------------------
TEST(LinkContactWrench, StepGraphMatchesStepWrench) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto cooked = CookGo2Float();
    auto& host = cooked.host;
    const uint32_t max_dof = articulation::ArticulationDofCount(host, 0u);
    const float ground_height = SeatGround(context, cooked);
    constexpr uint32_t kEnvCount = 32u;
    const uint32_t kSteps = 200u;

    auto run = [&](bool use_graph) {
        auto batched = articulation::ReplicateArticulationHostState(host, kEnvCount);
        gpu::BatchedArticulatedWorld bw(context, batched, cooked.feet, max_dof,
                                        ground_height);
        auto t = UploadReplicatedDrive(cooked.drive_targets, kEnvCount);
        auto s = UploadReplicatedDrive(cooked.drive_stiffness, kEnvCount);
        auto d = UploadReplicatedDrive(cooked.drive_damping, kEnvCount);
        auto l = UploadReplicatedDrive(cooked.drive_force_limits, kEnvCount);
        auto params = MakeStepParams(t, s, d, l);
        for (uint32_t step = 0u; step < kSteps; ++step) {
            if (use_graph) bw.StepGraph(params); else bw.Step(params);
        }
        if (use_graph) bw.GraphStreamSynchronize();
        context.stream.Synchronize();
        return std::make_pair(bw.DownloadLinkContactWrench(),
                              bw.DownloadContactForce());
    };

    const auto [step_w, step_f] = run(false);
    const auto [graph_w, graph_f] = run(true);
    ASSERT_EQ(step_w.size(), graph_w.size());
    ASSERT_EQ(step_f.size(), graph_f.size());
    size_t w_mis = 0u, f_mis = 0u;
    for (size_t i = 0u; i < step_w.size(); ++i)
        if (std::memcmp(&step_w[i], &graph_w[i], sizeof(float)) != 0) ++w_mis;
    for (size_t i = 0u; i < step_f.size(); ++i)
        if (std::memcmp(&step_f[i], &graph_f[i], sizeof(float)) != 0) ++f_mis;
    std::printf("[diag] StepGraph-vs-Step mismatches: wrench=%zu force=%zu\n",
                w_mis, f_mis);
    // Sanity: the graph path actually produced a non-trivial wrench (not all zero).
    double graph_abs_sum = 0.0;
    for (float v : graph_w) graph_abs_sum += std::fabs(v);
    EXPECT_GT(graph_abs_sum, 1.0) << "StepGraph produced an all-zero wrench -- the "
                                     "eager readout did not run under graph replay";
    EXPECT_EQ(w_mis, 0u);
    EXPECT_EQ(f_mis, 0u);
}
