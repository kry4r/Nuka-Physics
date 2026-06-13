// ---------------------------------------------------------------------------
// M7 T4 TRANSITIONAL — deleted in T6 with the factory; the permanent gate is
// h1_grasp_lift (T5).
// ---------------------------------------------------------------------------
// THE CRUX GATE. Asserts CookSceneToUnionTemplate(Load("h1_cup_table.nks"))
// reproduces BuildH1UnionScene(ctx) — i.e. the .nks-reading cook (src/scene/cook/
// union_cook) produces the SAME union scene template the 1026-line
// h1_union_scene_factory authored via FK / placement / a 200-step settle.
//
// EQUALITY POLICY (per the M7 T4 spec):
//   * STRUCTURAL fields EXACTLY (EXPECT_EQ): bodies_per_env, has_*, condim, all
//     broadphase ids, link / dof counts, dof_stride, grip_dofs (the 12), the
//     fingertip / foot counts (30 / 4), drive-table lengths, slot emission order.
//   * GEOMETRIC / FLOAT fields within a TIGHT tolerance (EXPECT_NEAR, tol 1e-5):
//     the settled q / base pose, sphere local offsets + radii, cup hull verts,
//     masses, μ, table_height, drive targets / gains / limits, dof_torque_limit,
//     total_mass, poly_cx. Byte-identity is NOT required — the factory derives
//     via FK while the cook reads baked data; a tight tolerance proves
//     equivalence. (Where a field IS byte-identical we still EXPECT_NEAR with a
//     0 tol observed; many are exact because the .nks stored the factory's value
//     with %.9g binary32-lossless and the cook copies it verbatim.)
//
// PLUS: BuildNkUnionModel(cooked.tmpl) is exercised (build/smoke) — the bridge
// the parity oracle / perf gate run, proving the cooked template maps to a
// well-formed UnionCsr nk::Model.
//
// NOTE on the settle VELOCITIES: the factory's gripper_proto carried the
// residual settle qdot / link_velocity; the .nks initial_state stores only the
// settled POSE (q + root), so the cook zeroes them. This gate therefore does NOT
// compare qdot / link_velocity (the spec's float-field list is q + base_pose).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "phi/device_context.hpp"
#include "runtime/coresident/h1_union_nk_model.hpp"
#include "runtime/coresident/h1_union_scene_factory.hpp"
#include "scene/cook/union_cook.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace coresident = nuka::runtime::coresident;
namespace cook = nuka::scene::cook;
namespace nk = nuka::nk;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr const char* kNksPath = "examples/scenes/h1_cup_table.nks";
constexpr float kTol = 1.0e-5f;

bool AssetsAvailable() {
    return std::filesystem::exists(kNksPath) &&
           std::filesystem::exists(coresident::kH1UnionMjcfDefault) &&
           std::filesystem::exists(coresident::kH1UnionCupDefault);
}

// Track the worst absolute deviation across every EXPECT_NEAR so the report can
// state the max deviation (the spec asks for the per-field comparison result).
double g_max_dev = 0.0;
const char* g_max_dev_what = "(none)";

void Near(float a, float b, const char* what) {
    const double d = std::abs(static_cast<double>(a) - static_cast<double>(b));
    if (d > g_max_dev) { g_max_dev = d; g_max_dev_what = what; }
    EXPECT_NEAR(a, b, kTol) << what;
}
void NearD(double a, double b, const char* what) {
    const double d = std::abs(a - b);
    if (d > g_max_dev) { g_max_dev = d; g_max_dev_what = what; }
    EXPECT_NEAR(a, b, static_cast<double>(kTol)) << what;
}
void NearVec3(const Vec3& a, const Vec3& b, const char* what) {
    Near(a.x, b.x, what); Near(a.y, b.y, what); Near(a.z, b.z, what);
}
void NearQuat(const Quat& a, const Quat& b, const char* what) {
    Near(a.w, b.w, what); Near(a.x, b.x, what);
    Near(a.y, b.y, what); Near(a.z, b.z, what);
}
void NearTransform(const Transform& a, const Transform& b, const char* what) {
    NearVec3(a.position, b.position, what);
    NearQuat(a.rotation, b.rotation, what);
}

}  // namespace

TEST(UnionCookMatchesFactory, CookReproducesFactoryTemplate) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / h1_with_hand / cup not present";
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    // ----- the FACTORY oracle (cook -> stance -> seat -> curl -> placement ->
    // 200-step settle -> carry -> table) -------------------------------------
    const coresident::H1UnionScene fac = coresident::BuildH1UnionScene(context);
    ASSERT_TRUE(fac.place_found);
    const coresident::BatchedSceneTemplate& ft = fac.tmpl;

    // ----- the COOK under test (read the baked .nks) -------------------------
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    const cook::CookedUnionScene cooked = cook::CookSceneToUnionTemplate(scene, 1);
    const coresident::BatchedSceneTemplate& ct = cooked.tmpl;

    // ===== STRUCTURAL (EXACT) ================================================
    // shape metadata.
    EXPECT_EQ(cooked.dof_stride, fac.dof_stride);
    EXPECT_EQ(cooked.base_dof, fac.base_dof);
    EXPECT_EQ(cooked.root_link, fac.root_link);
    EXPECT_EQ(cooked.link_count, fac.link_count);
    EXPECT_EQ(cooked.dof_stride, 51u);
    EXPECT_EQ(cooked.base_dof, 6u);

    // flags / counts.
    EXPECT_EQ(ct.has_grasp, ft.has_grasp);
    EXPECT_EQ(ct.has_ground, ft.has_ground);
    EXPECT_EQ(ct.has_feet, ft.has_feet);
    EXPECT_EQ(ct.has_table, ft.has_table);
    EXPECT_EQ(ct.condim, ft.condim);
    EXPECT_EQ(ct.cup_local_index, ft.cup_local_index);
    EXPECT_EQ(ct.bodies_per_env.size(), ft.bodies_per_env.size());
    EXPECT_EQ(ct.fingertips.size(), ft.fingertips.size());
    EXPECT_EQ(ct.feet.size(), ft.feet.size());
    EXPECT_EQ(ct.fingertips.size(), 30u);
    EXPECT_EQ(ct.feet.size(), 4u);
    EXPECT_EQ(ct.grip_torque.size(), ft.grip_torque.size());
    EXPECT_EQ(ct.drive_force_limits.size(), ft.drive_force_limits.size());

    // broadphase ids (all of them).
    EXPECT_EQ(ct.cup.broadphase_body_id, ft.cup.broadphase_body_id);
    EXPECT_EQ(ct.ground.broadphase_id, ft.ground.broadphase_id);
    EXPECT_EQ(ct.table_broadphase_id, ft.table_broadphase_id);
    EXPECT_EQ(ct.cup_table_proxy_id, ft.cup_table_proxy_id);

    // grip dofs (the 12) + grip links — EXACT (resolved against the same cook).
    ASSERT_EQ(cooked.grip_dofs.size(), fac.grip_dofs.size());
    ASSERT_EQ(cooked.grip_dofs.size(), 12u);
    for (size_t i = 0u; i < cooked.grip_dofs.size(); ++i)
        EXPECT_EQ(cooked.grip_dofs[i], fac.grip_dofs[i]) << "grip_dof " << i;
    ASSERT_EQ(cooked.grip_links.size(), fac.grip_links.size());
    for (size_t i = 0u; i < cooked.grip_links.size(); ++i)
        EXPECT_EQ(cooked.grip_links[i], fac.grip_links[i]) << "grip_link " << i;

    // drive-table lengths.
    EXPECT_EQ(cooked.drive_hold.size(), fac.drive_hold.size());
    EXPECT_EQ(cooked.drive_rest.size(), fac.drive_rest.size());
    EXPECT_EQ(cooked.drive_close.size(), fac.drive_close.size());
    EXPECT_EQ(cooked.dof_torque_limit.size(), fac.dof_torque_limit.size());

    // SLOT EMISSION ORDER: fingertips + feet must align link-for-link with the
    // factory (the union slot order feet -> fingers -> table is built off these
    // by BuildNkUnionModel; identical link order == identical slot order).
    ASSERT_EQ(ct.fingertips.size(), ft.fingertips.size());
    for (size_t i = 0u; i < ct.fingertips.size(); ++i) {
        EXPECT_EQ(ct.fingertips[i].link, ft.fingertips[i].link) << "ft link " << i;
        EXPECT_EQ(ct.fingertips[i].broadphase_handle,
                  ft.fingertips[i].broadphase_handle) << "ft bp " << i;
    }
    ASSERT_EQ(ct.feet.size(), ft.feet.size());
    for (size_t i = 0u; i < ct.feet.size(); ++i) {
        EXPECT_EQ(ct.feet[i].link, ft.feet[i].link) << "foot link " << i;
        EXPECT_EQ(ct.feet[i].broadphase_handle,
                  ft.feet[i].broadphase_handle) << "foot bp " << i;
    }

    // ===== GEOMETRIC / FLOAT (TIGHT tol) =====================================
    // scene scalars.
    NearD(cooked.total_mass, fac.total_mass, "total_mass");
    Near(cooked.cup_mass, fac.cup_mass, "cup_mass");
    Near(cooked.poly_cx, fac.poly_cx, "poly_cx");
    Near(cooked.table_height, fac.table_height, "table_height");
    Near(cooked.ankle_settle_tau[0], fac.ankle_settle_tau[0], "ankle_tau0");
    Near(cooked.ankle_settle_tau[1], fac.ankle_settle_tau[1], "ankle_tau1");

    // friction / table scalars.
    Near(ct.friction_mu, ft.friction_mu, "friction_mu");
    Near(ct.foot_mu, ft.foot_mu, "foot_mu");
    Near(ct.table_mu, ft.table_mu, "table_mu");
    Near(ct.table_height, ft.table_height, "tmpl.table_height");
    Near(ct.ground.height, ft.ground.height, "ground.height");
    NearVec3(ct.cup_table_proxy_half, ft.cup_table_proxy_half, "proxy_half");
    NearVec3(ct.cup_table_proxy_offset, ft.cup_table_proxy_offset, "proxy_off");

    // the cup BodyState (settled pose + scaled-bbox inertia + mass).
    ASSERT_EQ(ct.bodies_per_env.size(), 1u);
    ASSERT_EQ(ft.bodies_per_env.size(), 1u);
    {
        const auto& c = ct.bodies_per_env[0];
        const auto& f = ft.bodies_per_env[0];
        NearVec3(c.position, f.position, "cup pos");
        NearQuat(c.orientation, f.orientation, "cup quat");
        Near(c.inv_mass, f.inv_mass, "cup inv_mass");
        NearVec3(c.inv_inertia, f.inv_inertia, "cup inv_inertia");
    }

    // cup hull verts.
    ASSERT_EQ(ct.cup.hull_verts.size(), ft.cup.hull_verts.size());
    for (size_t i = 0u; i < ct.cup.hull_verts.size(); ++i)
        Near(ct.cup.hull_verts[i], ft.cup.hull_verts[i], "cup hull vert");

    // fingertip + foot geometry (offset + radius).
    for (size_t i = 0u; i < ct.fingertips.size(); ++i) {
        NearVec3(ct.fingertips[i].local_offset, ft.fingertips[i].local_offset,
                 "ft offset");
        Near(ct.fingertips[i].radius, ft.fingertips[i].radius, "ft radius");
    }
    for (size_t i = 0u; i < ct.feet.size(); ++i) {
        NearVec3(ct.feet[i].local_offset, ft.feet[i].local_offset, "foot offset");
        Near(ct.feet[i].radius, ft.feet[i].radius, "foot radius");
    }

    // the SETTLED gripper proto (q + base pose — the heart of the equivalence:
    // the factory's 200-step settle product vs the .nks baked IC).
    const auto& cp = ct.gripper_proto;
    const auto& fp = ft.gripper_proto;
    ASSERT_EQ(cp.TotalLinkCount(), fp.TotalLinkCount());
    ASSERT_EQ(cp.q.size(), fp.q.size());
    for (size_t i = 0u; i < cp.q.size(); ++i) Near(cp.q[i], fp.q[i], "proto q");
    ASSERT_EQ(cp.base_pose.size(), fp.base_pose.size());
    for (size_t i = 0u; i < cp.base_pose.size(); ++i)
        NearTransform(cp.base_pose[i], fp.base_pose[i], "proto base_pose");
    // topology / inertia / damping (the cook + factory cook the SAME H1 the same
    // way; these are byte-equal — tight tol still validates).
    ASSERT_EQ(cp.joint_damping.size(), fp.joint_damping.size());
    for (size_t i = 0u; i < cp.joint_damping.size(); ++i)
        Near(cp.joint_damping[i], fp.joint_damping[i], "joint_damping");
    ASSERT_EQ(cp.joint_armature.size(), fp.joint_armature.size());
    for (size_t i = 0u; i < cp.joint_armature.size(); ++i)
        Near(cp.joint_armature[i], fp.joint_armature[i], "joint_armature");
    ASSERT_EQ(cp.link_body.size(), fp.link_body.size());
    for (size_t i = 0u; i < cp.link_body.size(); ++i)
        EXPECT_EQ(cp.link_body[i], fp.link_body[i]) << "link_body " << i;

    // the three drive tables: dof + grip flag EXACT; target / kp / kd / tlim tol.
    auto cmp_table = [](const std::vector<coresident::H1UnionDriveEntry>& c,
                        const std::vector<coresident::H1UnionDriveEntry>& f,
                        const char* tag) {
        ASSERT_EQ(c.size(), f.size()) << tag << " size";
        for (size_t i = 0u; i < c.size(); ++i) {
            EXPECT_EQ(c[i].link, f[i].link) << tag << " link " << i;
            EXPECT_EQ(c[i].dof, f[i].dof) << tag << " dof " << i;
            EXPECT_EQ(c[i].grip, f[i].grip) << tag << " grip " << i;
            Near(c[i].target, f[i].target, tag);
            Near(c[i].kp, f[i].kp, tag);
            Near(c[i].kd, f[i].kd, tag);
            Near(c[i].tlim, f[i].tlim, tag);
        }
    };
    cmp_table(cooked.drive_hold, fac.drive_hold, "drive_hold");
    cmp_table(cooked.drive_rest, fac.drive_rest, "drive_rest");
    cmp_table(cooked.drive_close, fac.drive_close, "drive_close");

    // per-DOF torque limits.
    ASSERT_EQ(cooked.dof_torque_limit.size(), fac.dof_torque_limit.size());
    for (size_t i = 0u; i < cooked.dof_torque_limit.size(); ++i)
        Near(cooked.dof_torque_limit[i], fac.dof_torque_limit[i], "dof_torque_limit");

    std::printf("[T4 COOK==FACTORY] dof=%u links=%u fingertips=%zu feet=%zu "
                "table_z=%.5f cup0_z=%.5f | max float dev=%.3e on %s\n",
                cooked.dof_stride, cooked.link_count, ct.fingertips.size(),
                ct.feet.size(), cooked.table_height, ct.bodies_per_env[0].position.z,
                g_max_dev, g_max_dev_what);
}

// BuildNkUnionModel(cooked.tmpl) build/smoke — the bridge the parity oracle +
// perf gate run. Proves the cooked template maps to a well-formed UnionCsr Model
// (the slot order feet -> fingers -> table; the capacities; the cup body init).
TEST(UnionCookMatchesFactory, CookedTemplateBuildsNkUnionModel) {
    if (!AssetsAvailable())
        GTEST_SKIP() << "h1_cup_table.nks / h1_with_hand / cup not present";

    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    const cook::CookedUnionScene cooked = cook::CookSceneToUnionTemplate(scene, 1);

    const nk::Model model = coresident::BuildNkUnionModel(cooked.tmpl, 1u);
    EXPECT_EQ(model.contact_family, nk::ContactFamily::UnionCsr);
    // feet (4 FootSpherePlane) + fingers (30 FingerSphereHull) + table (1
    // BodyBoxPlane) == 35 union slots, in that order.
    ASSERT_EQ(model.union_slots.size(), 35u);
    for (uint32_t i = 0u; i < 4u; ++i)
        EXPECT_EQ(model.union_slots[i].cls, nk::UnionSlot::kFootSpherePlane) << i;
    for (uint32_t i = 4u; i < 34u; ++i)
        EXPECT_EQ(model.union_slots[i].cls, nk::UnionSlot::kFingerSphereHull) << i;
    EXPECT_EQ(model.union_slots[34].cls, nk::UnionSlot::kBodyBoxPlane);
    EXPECT_EQ(model.capacities.dofs_per_env, cooked.dof_stride);
    EXPECT_EQ(model.capacities.links_per_env, cooked.link_count);
    EXPECT_EQ(model.capacities.bodies_per_env, 1u);
    EXPECT_TRUE(model.table_enabled_default);
    EXPECT_EQ(model.body_init.size(), 1u);

    std::printf("[T4 NK BRIDGE] cooked.tmpl -> UnionCsr Model: slots=%zu "
                "rows/env=%u hull_verts=%u\n",
                model.union_slots.size(), model.capacities.max_rows_per_env,
                model.capacities.max_hull_verts);
}
