// ---------------------------------------------------------------------------
// v0.8 G2 -- the PRODUCTION union-scene-factory EQUALITY gate.
// ---------------------------------------------------------------------------
// src/runtime/coresident/h1_union_scene_factory.cpp REPRODUCES the test-side
// G1c/G1d union scene authoring (tests/coresident/h1_union_scene_shared.hpp +
// h1_demo_shared.hpp) verbatim, because the C-ABI union world cannot include a
// tests/ header. A drifted copy would silently train python policies on an
// UNVALIDATED scene -- the exact failure class the A2 grasp factory promotion
// guarded against. THIS gate pins the promotion: the factory's
// BatchedSceneTemplate must be BYTE-IDENTICAL (every float bit) to the
// shared-header authoring on the same context, INCLUDING the 200-step oracle
// settle pre-roll both sides run independently (D1 makes the two settles
// byte-equal iff the authoring code is byte-equal -- so this is a genuinely
// discriminative copy-drift tripwire).
//
// Plus the factory-only metadata sanity (dof maps / grip columns / limits /
// drive tables) the python substrate consumes.
// ---------------------------------------------------------------------------

#include "phi/device_context.hpp"
#include "runtime/coresident/h1_union_scene_factory.hpp"

#include "h1_union_scene_shared.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

namespace coresident = nuka::runtime::coresident;
namespace h1u = nuka::test::h1union;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

// Exact (bit-level) float equality via EXPECT_EQ -- the gate is BYTE identity,
// not closeness. Helpers compare composite math types field-by-field (struct
// memcmp would also test padding bytes, which are unspecified).
void ExpectVec3Eq(const Vec3& a, const Vec3& b, const char* what) {
    EXPECT_EQ(a.x, b.x) << what;
    EXPECT_EQ(a.y, b.y) << what;
    EXPECT_EQ(a.z, b.z) << what;
}
void ExpectQuatEq(const Quat& a, const Quat& b, const char* what) {
    EXPECT_EQ(a.w, b.w) << what;
    EXPECT_EQ(a.x, b.x) << what;
    EXPECT_EQ(a.y, b.y) << what;
    EXPECT_EQ(a.z, b.z) << what;
}
void ExpectTransformEq(const Transform& a, const Transform& b, const char* what) {
    ExpectVec3Eq(a.position, b.position, what);
    ExpectQuatEq(a.rotation, b.rotation, what);
}

void ExpectBodyStateEq(const nuka::runtime::rigid::BodyState& a,
                       const nuka::runtime::rigid::BodyState& b, const char* what) {
    ExpectVec3Eq(a.position, b.position, what);
    ExpectQuatEq(a.orientation, b.orientation, what);
    ExpectVec3Eq(a.linear_velocity, b.linear_velocity, what);
    ExpectVec3Eq(a.angular_velocity, b.angular_velocity, what);
    EXPECT_EQ(a.inv_mass, b.inv_mass) << what;
    ExpectVec3Eq(a.inv_inertia, b.inv_inertia, what);
    ExpectVec3Eq(a.force, b.force, what);
    ExpectVec3Eq(a.torque, b.torque, what);
}

template <typename T>
void ExpectFlatVecEq(const std::vector<T>& a, const std::vector<T>& b,
                     const char* what) {
    ASSERT_EQ(a.size(), b.size()) << what << " size";
    if (!a.empty()) {
        EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(T)))
            << what << " bytes";
    }
}

}  // namespace

TEST(H1UnionSceneFactory, TemplateByteIdenticalToSharedHeaderAuthoring) {
    if (!h1u::AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    // ----- the TEST-SIDE proven authoring (the G1d gates' exact path) ----------
    h1u::UnionScene sc = h1u::BuildGraspStandTableScene(context);
    ASSERT_TRUE(sc.place.found) << "shared-header placement search failed";
    const coresident::BatchedSceneTemplate ref = h1u::MakeUnionTemplate(sc);

    // ----- the PRODUCTION factory (what the C-ABI / python construct) ----------
    const coresident::H1UnionScene scene = coresident::BuildH1UnionScene(context);
    const coresident::BatchedSceneTemplate& t = scene.tmpl;

    // --- scalars / flags -------------------------------------------------------
    EXPECT_EQ(t.has_grasp, ref.has_grasp);
    EXPECT_EQ(t.has_ground, ref.has_ground);
    EXPECT_EQ(t.has_feet, ref.has_feet);
    EXPECT_EQ(t.has_table, ref.has_table);
    EXPECT_EQ(t.cup_local_index, ref.cup_local_index);
    EXPECT_EQ(t.friction_mu, ref.friction_mu);
    EXPECT_EQ(t.condim, ref.condim);
    EXPECT_EQ(t.foot_mu, ref.foot_mu);
    EXPECT_EQ(t.ground.height, ref.ground.height);
    EXPECT_EQ(t.ground.broadphase_id, ref.ground.broadphase_id);
    EXPECT_EQ(t.table_height, ref.table_height);
    EXPECT_EQ(t.table_mu, ref.table_mu);
    EXPECT_EQ(t.table_broadphase_id, ref.table_broadphase_id);
    ExpectVec3Eq(t.cup_table_proxy_half, ref.cup_table_proxy_half, "proxy_half");
    ExpectVec3Eq(t.cup_table_proxy_offset, ref.cup_table_proxy_offset, "proxy_off");
    EXPECT_EQ(t.cup_table_proxy_id, ref.cup_table_proxy_id);
    EXPECT_EQ(t.reset_jitter_x, ref.reset_jitter_x);
    EXPECT_EQ(t.reset_jitter_y, ref.reset_jitter_y);

    // --- the cup body + hull ----------------------------------------------------
    ASSERT_EQ(t.bodies_per_env.size(), ref.bodies_per_env.size());
    for (size_t i = 0u; i < t.bodies_per_env.size(); ++i)
        ExpectBodyStateEq(t.bodies_per_env[i], ref.bodies_per_env[i], "cup body");
    EXPECT_EQ(t.cup.broadphase_body_id, ref.cup.broadphase_body_id);
    ExpectFlatVecEq(t.cup.hull_verts, ref.cup.hull_verts, "cup hull verts");

    // --- fingertips + feet -------------------------------------------------------
    ASSERT_EQ(t.fingertips.size(), ref.fingertips.size());
    for (size_t i = 0u; i < t.fingertips.size(); ++i) {
        EXPECT_EQ(t.fingertips[i].link, ref.fingertips[i].link) << i;
        EXPECT_EQ(t.fingertips[i].broadphase_handle, ref.fingertips[i].broadphase_handle) << i;
        ExpectVec3Eq(t.fingertips[i].local_offset, ref.fingertips[i].local_offset, "ft off");
        EXPECT_EQ(t.fingertips[i].radius, ref.fingertips[i].radius) << i;
    }
    ASSERT_EQ(t.feet.size(), ref.feet.size());
    for (size_t i = 0u; i < t.feet.size(); ++i) {
        EXPECT_EQ(t.feet[i].link, ref.feet[i].link) << i;
        EXPECT_EQ(t.feet[i].broadphase_handle, ref.feet[i].broadphase_handle) << i;
        ExpectVec3Eq(t.feet[i].local_offset, ref.feet[i].local_offset, "foot off");
        EXPECT_EQ(t.feet[i].radius, ref.feet[i].radius) << i;
    }

    // --- the drive seeds ----------------------------------------------------------
    ExpectFlatVecEq(t.grip_torque, ref.grip_torque, "grip_torque");
    ExpectFlatVecEq(t.drive_force_limits, ref.drive_force_limits, "drive_force_limits");

    // --- the SETTLED gripper proto (the heart of the gate: the two 200-step
    // settles ran independently -- byte-equal iff the authoring is byte-equal) ---
    const auto& p = t.gripper_proto;
    const auto& rp = ref.gripper_proto;
    ASSERT_EQ(p.TotalLinkCount(), rp.TotalLinkCount());
    ExpectFlatVecEq(p.q, rp.q, "proto q (settled)");
    ExpectFlatVecEq(p.qdot, rp.qdot, "proto qdot (settled)");
    ExpectFlatVecEq(p.link_velocity, rp.link_velocity, "proto link_velocity (settled)");
    ASSERT_EQ(p.base_pose.size(), rp.base_pose.size());
    for (size_t i = 0u; i < p.base_pose.size(); ++i)
        ExpectTransformEq(p.base_pose[i], rp.base_pose[i], "proto base_pose (settled)");
    ExpectFlatVecEq(p.joint_damping, rp.joint_damping, "proto joint_damping");
    ExpectFlatVecEq(p.joint_armature, rp.joint_armature, "proto joint_armature");
    ExpectFlatVecEq(p.link_body, rp.link_body, "proto link_body");
    ASSERT_EQ(p.joint_type.size(), rp.joint_type.size());
    for (size_t i = 0u; i < p.joint_type.size(); ++i)
        EXPECT_EQ(static_cast<int>(p.joint_type[i]), static_cast<int>(rp.joint_type[i])) << i;
    ASSERT_EQ(p.link_inertia.size(), rp.link_inertia.size());
    for (size_t i = 0u; i < p.link_inertia.size(); ++i)
        EXPECT_EQ(0, std::memcmp(p.link_inertia[i].I, rp.link_inertia[i].I,
                                 sizeof(p.link_inertia[i].I))) << "link_inertia " << i;

    // --- factory metadata mirrors the shared-header scene -------------------------
    EXPECT_EQ(scene.dof_stride, sc.dof_stride);
    EXPECT_EQ(scene.base_dof, sc.base_dof);
    EXPECT_EQ(scene.root_link, sc.root_link);
    EXPECT_EQ(scene.link_count, sc.link_count);
    EXPECT_EQ(scene.poly_cx, sc.poly_cx);
    EXPECT_EQ(scene.total_mass, sc.total_mass);
    ExpectTransformEq(scene.seat_pose, sc.seat_pose, "seat_pose");
    EXPECT_EQ(scene.table_height, sc.sg.table_height);
    ExpectFlatVecEq(scene.grip_links, sc.grip_links, "grip_links");

    std::printf("[G2 FACTORY EQ] template byte-identical: dof=%u links=%u "
                "fingertips=%zu feet=%zu table_z=%.4f cup0_z=%.4f\n",
                scene.dof_stride, scene.link_count, t.fingertips.size(),
                t.feet.size(), t.table_height, t.bodies_per_env[0].position.z);
}

TEST(H1UnionSceneFactory, MetadataShapesAndLimits) {
    if (!h1u::AssetsAvailable()) GTEST_SKIP() << "h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const coresident::H1UnionScene scene = coresident::BuildH1UnionScene(context);

    EXPECT_EQ(scene.dof_stride, 51u);
    EXPECT_EQ(scene.base_dof, 6u);
    ASSERT_EQ(scene.grip_links.size(), 12u);
    ASSERT_EQ(scene.grip_dofs.size(), 12u);
    for (uint32_t d : scene.grip_dofs) {
        EXPECT_GE(d, scene.base_dof);  // never a base column.
        EXPECT_LT(d, scene.dof_stride);
    }

    // Per-DOF limits: base columns dead (0); every joint column positive; the
    // known envelope values present (knee 300 / hip 200 / ankle 40 / wrist 6).
    ASSERT_EQ(scene.dof_torque_limit.size(), scene.dof_stride);
    for (uint32_t d = 0u; d < scene.base_dof; ++d)
        EXPECT_EQ(scene.dof_torque_limit[d], 0.0f) << "base col " << d;
    uint32_t n300 = 0u, n200 = 0u, n40 = 0u, n6 = 0u, npos = 0u;
    for (uint32_t d = scene.base_dof; d < scene.dof_stride; ++d) {
        const float v = scene.dof_torque_limit[d];
        EXPECT_GT(v, 0.0f) << "joint col " << d << " has no limit";
        if (v == 300.0f) ++n300;
        if (v == 200.0f) ++n200;
        if (v == 40.0f) ++n40;
        if (v == 6.0f) ++n6;
        if (v > 0.0f) ++npos;
    }
    EXPECT_EQ(n300, 2u);   // two knees.
    EXPECT_GE(n200, 6u);   // six hip joints (+ torso at 200).
    EXPECT_GE(n40, 2u);    // two ankles (+ shoulders at 40).
    EXPECT_GE(n6, 1u);     // wrist(s).
    EXPECT_EQ(npos, scene.dof_stride - scene.base_dof);

    // Drive tables: same length, ankle entries appended, grip flags only on the
    // close columns, every dof in range, hold/close grip targets differ by the
    // close offset.
    ASSERT_EQ(scene.drive_hold.size(), scene.drive_rest.size());
    ASSERT_EQ(scene.drive_hold.size(), scene.drive_close.size());
    ASSERT_GT(scene.drive_hold.size(), 12u);
    uint32_t grip_entries = 0u;
    for (size_t i = 0u; i < scene.drive_hold.size(); ++i) {
        const auto& h = scene.drive_hold[i];
        const auto& c = scene.drive_close[i];
        ASSERT_LT(h.dof, scene.dof_stride);
        EXPECT_EQ(h.dof, c.dof);
        EXPECT_EQ(h.grip, c.grip);
        if (h.grip != 0u) {
            ++grip_entries;
            EXPECT_FLOAT_EQ(c.target - h.target, coresident::kH1UnionCloseOffset);
            EXPECT_FLOAT_EQ(scene.drive_rest[i].target - h.target,
                            coresident::kH1UnionRestBackOffset);
        }
    }
    EXPECT_EQ(grip_entries, 12u);

    std::printf("[G2 FACTORY META] grip_dofs={");
    for (uint32_t d : scene.grip_dofs) std::printf("%u,", d);
    std::printf("} drive_table=%zu entries ankle_tau=(%.3f, %.3f)\n",
                scene.drive_hold.size(), scene.ankle_settle_tau[0],
                scene.ankle_settle_tau[1]);
}
