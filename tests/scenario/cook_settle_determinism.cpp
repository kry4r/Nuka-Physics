// ---------------------------------------------------------------------------
// M7 T2 — cook_settle_determinism: the milestone gate for the two reusable cook
// primitives (src/scene/cook/settle.* + placement.*).
//
//   (1) SETTLE DETERMINISM — the plan's "settle two runs byte-identical" gate.
//       Build a SMALL GENERIC scene (a tiny CookToModel articulation + an
//       injected movable cup body), settle it TWICE with the same SettleSpec on
//       the deterministic device path, and assert the two InitialStateComponent
//       qpos vectors + the settled body poses are BYTE-IDENTICAL (memcmp). The
//       settle PD-holds the articulation at its seeded q (a Hold over the link
//       path) while the cup comes to rest under gravity; T1's snapshot_body_*
//       coverage makes the cup pose round-trip through SnapshotState.
//
//   (2) REST PLACEMENT — FindRestPlacement seats a cup on a table so the cup
//       bottom rests at table_top_z + clearance (EXPECT_NEAR, 1e-4). Pure
//       geometry, asset-light (boxes synthesized in-code).
//
// Asset-light: no .nuka-assets dependency (the scene is built programmatically),
// so the gate runs anywhere a CUDA backend is present.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/device_context.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/cook/placement.hpp"
#include "scene/cook/settle.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
using nuka::math::Transform;
using nuka::math::Vec3;
using namespace nuka::scene;

constexpr float kDt = 1.0f / 240.0f;
constexpr float kGravityZ = -9.81f;

// --- shared backend (outlives the World dtors; freed at process exit) ----------
struct Backend {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
};
nuka::phi::DeviceContext& ProcessContext() {
    static nuka::phi::DeviceContext ctx = nuka::phi::MakeDefaultDeviceContext();
    return ctx;
}
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        ProcessContext();
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig SettleConfig() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = kDt;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = kGravityZ;
    cfg.vel_iters = 16;
    cfg.pos_iters = 0;
    cfg.fold_drive_damping = 0;
    return cfg;
}

// A tiny GENERIC articulation: a static base + one revolute child link (a box) +
// a Position actuator. Cooks to a single-articulation Model (link 0 = base/Fixed,
// link 1 = the revolute link) with a non-empty initial_q + a hold drive.
SceneIR BuildTinyArticulation() {
    SceneIR s;
    RigidBodyRecord base;
    base.name = "robot_base";
    base.is_static = true;
    const BodyId base_id = s.AddRigidBody(base);

    RigidBodyRecord link;
    link.name = "robot_link";
    link.parent_id = base_id;
    link.local_transform.position = {0.0f, 0.0f, 0.5f};
    link.mass = 1.0f;
    link.inertia = {0.01f, 0.01f, 0.01f};
    const BodyId link_id = s.AddRigidBody(link);

    CollisionShapeRecord shape;
    shape.name = "robot_link_box";
    shape.body_id = link_id;
    shape.type = ShapeType::Box;
    shape.half_extents = {0.05f, 0.05f, 0.05f};
    s.AddCollisionShape(shape);

    JointRecord joint;
    joint.name = "robot_hinge";
    joint.parent_body = base_id;
    joint.child_body = link_id;
    joint.type = JointType::Revolute;
    joint.axis = {0.0f, 1.0f, 0.0f};
    joint.initial_position = 0.2f;  // a non-zero seeded q (the hold target).
    const JointId joint_id = s.AddJoint(joint);

    ActuatorRecord act;
    act.name = "robot_motor";
    act.joint_id = joint_id;
    act.type = ActuatorType::Position;
    act.gain = 40.0f;
    act.force_limit = 50.0f;
    s.AddActuator(act);
    return s;
}

// The settle product captured for the determinism comparison.
struct SettleSnapshot {
    std::vector<float> qpos;
    std::vector<Transform> body_poses;
    bool ok = false;
};

// Cook the tiny articulation, INJECT one movable cup rigid body (a box above a
// floor) into the cooked Model (the generic CookToModel does not yet author
// movable body_init — that is the T4 wiring; here the gate supplies a movable
// cup directly so the body-pose snapshot/download arm is exercised), build the
// World, settle once, and capture the settled IC + body poses.
SettleSnapshot RunSettleOnce() {
    Backend b = GetBackend();
    SceneIR scene = BuildTinyArticulation();
    cook::CookToModelResult cooked = cook::CookToModel(scene, /*env_count=*/1);
    nk::Model& model = cooked.model;

    // Inject a movable cup as the LAST body row. The articulation link-bodies
    // (rows 0..body_count-1) get inv_mass 0 (frozen at Identity — the body arm
    // skips inv_mass<=0), and the appended cup row gets a real pose + mass so it
    // falls under gravity and round-trips through the body snapshot.
    nk::ModelCapacities& cap = model.capacities;
    const uint32_t art_bodies = cap.bodies_per_env;
    const uint32_t cup_row = art_bodies;
    cap.bodies_per_env = art_bodies + 1u;
    model.body_init.assign(cap.bodies_per_env, nk::Model::BodyInit{});
    nk::Model::BodyInit& cup = model.body_init[cup_row];
    cup.pose = Transform::Identity();
    cup.pose.position = Vec3{0.0f, 0.0f, 0.30f};  // a few cm above the origin.
    cup.inv_mass = 1.0f / 0.2f;                    // 0.2 kg cup (movable).
    const float ii = 1.0f / (0.2f * (0.08f * 0.08f) / 6.0f);
    cup.inv_inertia = Vec3{ii, ii, ii};

    nk::World world(std::move(model), 1u, b.dev, b.backend, SettleConfig());
    EXPECT_TRUE(world.Ready());

    // Settle: hold every articulation link at its seeded q (path glob "*"),
    // 120 deterministic device steps.
    cook::SettleSpec spec;
    spec.steps = 120;
    spec.dt = kDt;
    spec.holds.push_back({/*dof_pattern=*/"*", cook::SettleSpec::Hold::Mode::PD,
                          /*kp=*/50.0f, /*kd=*/4.0f});

    const cook::SettleResult r = cook::Settle(world, scene, cooked.scene_map, spec);

    SettleSnapshot out;
    out.ok = r.ok;
    out.qpos = r.initial_state.qpos;
    out.body_poses.reserve(r.body_poses.size());
    for (const auto& [entity, pose] : r.body_poses) {
        (void)entity;
        out.body_poses.push_back(pose);
    }
    return out;
}

}  // namespace

// ===========================================================================
// (1) SETTLE DETERMINISM — two runs byte-identical.
// ===========================================================================
TEST(CookSettleDeterminism, TwoRunsByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    const SettleSnapshot a = RunSettleOnce();
    const SettleSnapshot b = RunSettleOnce();
    ASSERT_TRUE(a.ok) << "settle run A failed (world not Ready / device op failed)";
    ASSERT_TRUE(b.ok) << "settle run B failed";

    // The articulation IC (q per link) is non-empty and byte-identical.
    ASSERT_FALSE(a.qpos.empty()) << "settle produced no articulation q";
    ASSERT_EQ(a.qpos.size(), b.qpos.size());
    EXPECT_EQ(std::memcmp(a.qpos.data(), b.qpos.data(),
                          a.qpos.size() * sizeof(float)),
              0)
        << "two settle runs produced different InitialStateComponent qpos (D1)";

    // The settled movable-body poses (incl. the cup) are byte-identical.
    ASSERT_FALSE(a.body_poses.empty()) << "settle produced no body poses";
    ASSERT_EQ(a.body_poses.size(), b.body_poses.size());
    EXPECT_EQ(std::memcmp(a.body_poses.data(), b.body_poses.data(),
                          a.body_poses.size() * sizeof(Transform)),
              0)
        << "two settle runs produced different settled body poses (D1)";

    // The cup must actually have SETTLED (moved down under gravity from its
    // injected 0.30 m start) — proves the settle stepped + the body snapshot
    // captured the moved pose (not just a frozen Identity).
    const Transform& cup = a.body_poses.back();
    std::printf("[settle] q[last]=%.6f  cup z=%.6f (start 0.30)\n",
                a.qpos.back(), cup.position.z);
    EXPECT_LT(cup.position.z, 0.30f)
        << "the cup did not move under gravity — the settle did not step / the "
           "body-pose snapshot did not capture the settled pose";
}

// ===========================================================================
// (1b) SETTLE-TO-REST — a MOVABLE rigid cup dropped onto a plane settles to
// rest and the two runs are byte-identical. This is the headline cook use case
// (a cup coming to rest on a surface): a bodies-only union Model (no articulation
// needed — the generic CookToModel does not author movable body_init / box-plane
// contact yet; that is T4, so the gate supplies the union slot directly) so the
// settle ACTUALLY rests the body. Proves the settle + body snapshot/download on a
// genuinely resting movable body, deterministically.
// ===========================================================================
namespace {

// A free rigid box (the "cup") a few mm above a static z=0 plane (kBodyBoxPlane).
nk::Model BuildCupOnPlaneModel() {
    nk::Model model;
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1;
    cap.bodies_per_env = 1;

    const float box_he = 0.04f;
    const float floor_z = 0.0f;
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = Vec3{0.0f, 0.0f, floor_z + box_he + 0.005f};  // 5 mm drop.
    bi.inv_mass = 1.0f / 0.2f;  // 0.2 kg cup.
    const float ii = 1.0f / (0.2f * (2 * box_he) * (2 * box_he) / 6.0f);
    bi.inv_inertia = Vec3{ii, ii, ii};
    model.body_init.push_back(bi);

    nk::UnionSlot s;
    s.cls = nk::UnionSlot::kBodyBoxPlane;
    s.body = 0u;
    s.condim = 1;
    s.box_half = Vec3{box_he, box_he, box_he};
    s.plane_height = floor_z;
    s.mu = 0.0f;
    model.union_slots.push_back(s);

    model.contact_family = nk::ContactFamily::UnionCsr;
    cap.max_contacts_per_env = static_cast<uint32_t>(model.union_slots.size());
    uint32_t rows = 0;
    for (const nk::UnionSlot& u : model.union_slots) rows += u.MaxRows();
    cap.max_rows_per_env = rows;
    model.union_solref[0] = 0.02f; model.union_solref[1] = 1.0f;
    model.union_solimp[0] = 0.9f;  model.union_solimp[1] = 0.95f;
    model.union_solimp[2] = 0.001f; model.union_solimp[3] = 0.5f;
    model.union_solimp[4] = 2.0f;
    return model;
}

// Settle the cup-on-plane Model once (no holds — bodies-only). Returns the
// settled cup pose.
Transform RestCupOnce(uint32_t steps) {
    Backend b = GetBackend();
    nk::World world(BuildCupOnPlaneModel(), 1u, b.dev, b.backend, SettleConfig());
    EXPECT_TRUE(world.Ready());
    // An empty SceneIR + SceneMap (bodies-only settle resolves no holds; the
    // body-pose download path is what we exercise here).
    SceneIR empty_scene;
    SceneMap empty_map;
    cook::SettleSpec spec;
    spec.steps = static_cast<int>(steps);
    spec.dt = kDt;
    const cook::SettleResult r =
        cook::Settle(world, empty_scene, empty_map, spec);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.body_poses.size(), 1u);
    return r.body_poses.empty() ? Transform::Identity()
                                : r.body_poses.front().second;
}

}  // namespace

TEST(CookSettleDeterminism, MovableCupSettlesToRestDeterministic) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    const Transform a = RestCupOnce(/*steps=*/200u);
    const Transform b = RestCupOnce(/*steps=*/200u);

    // Byte-identical settled pose across two runs (D1).
    EXPECT_EQ(std::memcmp(&a, &b, sizeof(Transform)), 0)
        << "two cup-on-plane settle runs produced different rest poses (D1)";

    // The cup rests on the plane: its bottom (centre z - half 0.04) sits at ~0
    // (the plane), NOT free-fallen through it. A 5 mm drop settles to rest.
    const float box_he = 0.04f;
    const float cup_bottom = a.position.z - box_he;
    std::printf("[rest] cup centre z=%.6f  bottom z=%.6f (plane 0)\n",
                a.position.z, cup_bottom);
    EXPECT_NEAR(cup_bottom, 0.0f, 0.01f)
        << "the cup did not settle to rest on the plane (bottom z=" << cup_bottom
        << ")";
}

// ===========================================================================
// (2) REST PLACEMENT — cup seated on a table at top + clearance.
// ===========================================================================
namespace {

// A static table (a box body) + a movable cup (a box body, parked anywhere).
// FindRestPlacement should seat the cup bottom at the table top + clearance.
SceneIR BuildCupOnTable(BodyId* out_table, BodyId* out_cup) {
    SceneIR s;

    RigidBodyRecord table;
    table.name = "table";
    table.is_static = true;
    table.local_transform.position = {0.2f, -0.1f, 0.0f};  // table at some pose.
    const BodyId table_id = s.AddRigidBody(table);
    CollisionShapeRecord table_top;
    table_top.name = "table_top";
    table_top.body_id = table_id;
    table_top.type = ShapeType::Box;
    table_top.half_extents = {0.4f, 0.4f, 0.02f};  // 4 cm thick top.
    table_top.local_transform.position = {0.0f, 0.0f, 0.75f};  // top centre @ 0.75.
    s.AddCollisionShape(table_top);

    RigidBodyRecord cup;
    cup.name = "cup";
    cup.local_transform.position = {5.0f, 5.0f, 5.0f};  // parked far (irrelevant).
    const BodyId cup_id = s.AddRigidBody(cup);
    CollisionShapeRecord cup_shape;
    cup_shape.name = "cup_body";
    cup_shape.body_id = cup_id;
    cup_shape.type = ShapeType::Box;
    cup_shape.half_extents = {0.05f, 0.05f, 0.06f};  // 12 cm tall cup.
    s.AddCollisionShape(cup_shape);

    if (out_table) *out_table = table_id;
    if (out_cup) *out_cup = cup_id;
    return s;
}

}  // namespace

TEST(CookSettleDeterminism, FindRestPlacementSeatsCupOnTable) {
    BodyId table_id = kInvalidBody, cup_id = kInvalidBody;
    SceneIR scene = BuildCupOnTable(&table_id, &cup_id);

    const float clearance = 1.0e-3f;
    const float table_top_z = cook::SupportBodyTopZ(scene, table_id);
    // Table top centre @ z=0.75, half-thickness 0.02 -> top z = 0.77 (in world,
    // table body at z=0).
    EXPECT_NEAR(table_top_z, 0.77f, 1e-5f);

    // Seat the cup. The cup is symmetric so its root-relative min-z is -0.06.
    const Transform placement =
        cook::FindRestPlacement(scene, cup_id, table_id, clearance);

    // Apply the placement to the cup record and recompute its bottom z.
    scene.GetBodyMut(cup_id).local_transform = placement;
    const Transform cup_world = cook::BodyWorldTransform(scene, cup_id);
    const float cup_rel_min_z = cook::ObjectRootRelativeMinZ(scene, cup_id);
    const float cup_bottom_z =
        cup_world.TransformPoint(Vec3{0.0f, 0.0f, cup_rel_min_z}).z;

    std::printf("[placement] table_top_z=%.6f cup_bottom_z=%.6f (want %.6f)\n",
                table_top_z, cup_bottom_z, table_top_z + clearance);
    EXPECT_NEAR(cup_bottom_z, table_top_z + clearance, 1e-4f)
        << "cup bottom not seated at table top + clearance";
    EXPECT_GE(cup_bottom_z, table_top_z - 1e-4f)
        << "cup penetrates the table top";
}
