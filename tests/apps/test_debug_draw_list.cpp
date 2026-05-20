// ---------------------------------------------------------------------------
// Tests for nuka::app::DebugDrawList
// ---------------------------------------------------------------------------

#include "apps/debug_shell/debug_draw.hpp"
#include "apps/debug_shell/debug_visualization.hpp"
#include "constraint/constraint_block.hpp"
#include "constraint/contact_manifold.hpp"
#include "scene/scene_pipeline.hpp"

#include <gtest/gtest.h>

#include <span>
#include <vector>

using namespace nuka::app;
using namespace nuka::math;
using namespace nuka::collision;

namespace {

nuka::scene::SceneIR BuildVisualizationScene() {
    nuka::scene::SceneIR scene;

    nuka::scene::RigidBodyRecord base;
    base.name = "base";
    base.is_static = true;
    base.local_transform.position = {1.0f, 0.0f, 0.0f};
    const auto base_id = scene.AddRigidBody(base);

    nuka::scene::RigidBodyRecord link;
    link.name = "link";
    link.parent_id = base_id;
    link.mass = 2.0f;
    link.local_transform.position = {0.0f, 0.0f, 2.0f};
    const auto link_id = scene.AddRigidBody(link);

    nuka::scene::CollisionShapeRecord box;
    box.name = "base_box";
    box.body_id = base_id;
    box.type = nuka::scene::ShapeType::Box;
    box.half_extents = {0.2f, 0.3f, 0.4f};
    scene.AddCollisionShape(box);

    nuka::scene::CollisionShapeRecord sphere;
    sphere.name = "link_sphere";
    sphere.body_id = link_id;
    sphere.type = nuka::scene::ShapeType::Sphere;
    sphere.radius = 0.25f;
    sphere.local_transform.position = {0.0f, 0.0f, 0.5f};
    scene.AddCollisionShape(sphere);

    nuka::scene::CollisionShapeRecord capsule;
    capsule.name = "link_capsule";
    capsule.body_id = link_id;
    capsule.type = nuka::scene::ShapeType::Capsule;
    capsule.radius = 0.1f;
    capsule.half_height = 0.75f;
    capsule.local_transform.position = {0.5f, 0.0f, 0.0f};
    scene.AddCollisionShape(capsule);

    nuka::scene::JointRecord joint;
    joint.name = "hinge";
    joint.parent_body = base_id;
    joint.child_body = link_id;
    joint.axis = {0.0f, 1.0f, 0.0f};
    scene.AddJoint(joint);

    return scene;
}

size_t CountCommands(const DebugDrawList& list, DrawCommandType type) {
    size_t count = 0;
    for (const auto& command : list.Commands()) {
        if (command.type == type) {
            ++count;
        }
    }
    return count;
}

const DrawCommand* FirstCommand(const DebugDrawList& list, DrawCommandType type) {
    for (const auto& command : list.Commands()) {
        if (command.type == type) {
            return &command;
        }
    }
    return nullptr;
}

void ExpectVec3Eq(const Vec3& actual, const Vec3& expected) {
    EXPECT_FLOAT_EQ(actual.x, expected.x);
    EXPECT_FLOAT_EQ(actual.y, expected.y);
    EXPECT_FLOAT_EQ(actual.z, expected.z);
}

DebugVisualizationInput MakeVisualizationInput(const nuka::scene::CompiledScene& compiled) {
    DebugVisualizationInput input;
    input.render_scene = &compiled.render;
    input.scene_graph = &compiled.graph;
    input.physics_world = &compiled.physics;
    return input;
}

DebugVisualizationOptions DisabledVisualizationOptions() {
    DebugVisualizationOptions options;
    options.draw_collision_shapes = false;
    options.draw_shape_aabbs = false;
    options.draw_joint_axes = false;
    options.draw_centers_of_mass = false;
    options.draw_contact_points = false;
    options.draw_constraint_errors = false;
    return options;
}

} // namespace

// ---------------------------------------------------------------------------
// Contact overlay command count
// ---------------------------------------------------------------------------

TEST(DebugDraw, ProducesContactOverlayCommands) {
    const auto commands = BuildContactOverlayCommandCount(4);
    EXPECT_EQ(commands, 4u);
}

// ---------------------------------------------------------------------------
// AddLine produces a line command
// ---------------------------------------------------------------------------

TEST(DebugDraw, AddLineCommand) {
    DebugDrawList list;
    list.AddLine({0, 0, 0}, {1, 1, 1});
    ASSERT_EQ(list.CommandCount(), 1u);
    EXPECT_EQ(list.Commands()[0].type, DrawCommandType::Line);
}

// ---------------------------------------------------------------------------
// AddSphere produces a sphere command
// ---------------------------------------------------------------------------

TEST(DebugDraw, AddSphereCommand) {
    DebugDrawList list;
    list.AddSphere({1, 2, 3}, 0.5f);
    ASSERT_EQ(list.CommandCount(), 1u);
    EXPECT_EQ(list.Commands()[0].type, DrawCommandType::Sphere);
    EXPECT_FLOAT_EQ(list.Commands()[0].radius, 0.5f);
}

// ---------------------------------------------------------------------------
// AddCapsule produces a capsule command
// ---------------------------------------------------------------------------

TEST(DebugDraw, AddCapsuleCommand) {
    DebugDrawList list;
    list.AddCapsule({1, 2, 3}, Vec3::UnitZ(), 0.25f, 0.75f);
    ASSERT_EQ(list.CommandCount(), 1u);
    EXPECT_EQ(list.Commands()[0].type, DrawCommandType::Capsule);
    ExpectVec3Eq(list.Commands()[0].position, {1, 2, 3});
    ExpectVec3Eq(list.Commands()[0].end, Vec3::UnitZ());
    EXPECT_FLOAT_EQ(list.Commands()[0].radius, 0.25f);
    EXPECT_FLOAT_EQ(list.Commands()[0].half_height, 0.75f);
}

// ---------------------------------------------------------------------------
// AddAABB produces an AABB command
// ---------------------------------------------------------------------------

TEST(DebugDraw, AddAABBCommand) {
    AABB box;
    box.min = {-1, -1, -1};
    box.max = { 1,  1,  1};

    DebugDrawList list;
    list.AddAABB(box);
    ASSERT_EQ(list.CommandCount(), 1u);
    EXPECT_EQ(list.Commands()[0].type, DrawCommandType::AABB);
}

// ---------------------------------------------------------------------------
// AddFrame produces 3 line commands (X, Y, Z axes)
// ---------------------------------------------------------------------------

TEST(DebugDraw, AddFrameProducesThreeLines) {
    DebugDrawList list;
    list.AddFrame(Transform::Identity(), 1.0f);
    EXPECT_EQ(list.CommandCount(), 3u);
    for (const auto& cmd : list.Commands()) {
        EXPECT_EQ(cmd.type, DrawCommandType::Line);
    }
}

// ---------------------------------------------------------------------------
// Clear resets the command list
// ---------------------------------------------------------------------------

TEST(DebugDraw, ClearResetsCommands) {
    DebugDrawList list;
    list.AddLine({0, 0, 0}, {1, 0, 0});
    list.AddSphere({0, 0, 0}, 1.0f);
    ASSERT_EQ(list.CommandCount(), 2u);

    list.Clear();
    EXPECT_EQ(list.CommandCount(), 0u);
    EXPECT_TRUE(list.Commands().empty());
}

// ---------------------------------------------------------------------------
// BuildDebugVisualization emits collision body proxies and shape AABBs
// ---------------------------------------------------------------------------

TEST(DebugVisualization, EmitsCollisionBodyPrimitivesAndAabbs) {
    const auto compiled = nuka::scene::BuildCompiledScene(BuildVisualizationScene());

    auto options = DisabledVisualizationOptions();
    options.draw_collision_shapes = true;
    options.draw_shape_aabbs = true;

    const auto list = BuildDebugVisualization(MakeVisualizationInput(compiled), options);

    EXPECT_EQ(CountCommands(list, DrawCommandType::Box), 1u);
    EXPECT_EQ(CountCommands(list, DrawCommandType::Sphere), 1u);
    EXPECT_EQ(CountCommands(list, DrawCommandType::Capsule), 1u);
    EXPECT_EQ(CountCommands(list, DrawCommandType::AABB), 3u);

    const auto* capsule = FirstCommand(list, DrawCommandType::Capsule);
    ASSERT_NE(capsule, nullptr);
    ExpectVec3Eq(capsule->position, {1.5f, 0.0f, 2.0f});
    EXPECT_FLOAT_EQ(capsule->radius, 0.1f);
    EXPECT_FLOAT_EQ(capsule->half_height, 0.75f);
}

// ---------------------------------------------------------------------------
// BuildDebugVisualization emits joint axes and body centers of mass
// ---------------------------------------------------------------------------

TEST(DebugVisualization, EmitsJointAxesAndCentersOfMass) {
    const auto compiled = nuka::scene::BuildCompiledScene(BuildVisualizationScene());

    auto options = DisabledVisualizationOptions();
    options.draw_joint_axes = true;
    options.draw_centers_of_mass = true;
    options.joint_axis_length = 0.5f;

    const auto list = BuildDebugVisualization(MakeVisualizationInput(compiled), options);

    EXPECT_EQ(CountCommands(list, DrawCommandType::Sphere), 2u);
    ASSERT_EQ(CountCommands(list, DrawCommandType::Line), 1u);

    const auto* axis = FirstCommand(list, DrawCommandType::Line);
    ASSERT_NE(axis, nullptr);
    ExpectVec3Eq(axis->position, {1.0f, 0.0f, 2.0f});
    ExpectVec3Eq(axis->end, {1.0f, 0.5f, 2.0f});
}

// ---------------------------------------------------------------------------
// BuildDebugVisualization emits contact normals and constraint error vectors
// ---------------------------------------------------------------------------

TEST(DebugVisualization, EmitsContactsAndConstraintErrors) {
    const auto compiled = nuka::scene::BuildCompiledScene(BuildVisualizationScene());

    nuka::constraint::ContactManifold manifold;
    nuka::constraint::ContactPoint point;
    point.position = {2.0f, 3.0f, 4.0f};
    point.normal = {0.0f, 1.0f, 0.0f};
    point.penetration = 0.02f;
    manifold.AddPoint(point);

    nuka::constraint::ConstraintBlock constraint;
    constraint.type = nuka::constraint::ConstraintType::Joint;
    constraint.body_a = 1u;
    constraint.row_count = 1u;
    constraint.jacobian_linear_a[0] = Vec3::UnitX();
    constraint.rhs[0] = 0.25f;

    std::vector<nuka::constraint::ContactManifold> contacts = {manifold};
    std::vector<nuka::constraint::ConstraintBlock> constraints = {constraint};

    auto input = MakeVisualizationInput(compiled);
    input.contact_manifolds = std::span<const nuka::constraint::ContactManifold>(
        contacts.data(), contacts.size());
    input.constraint_blocks = std::span<const nuka::constraint::ConstraintBlock>(
        constraints.data(), constraints.size());

    auto options = DisabledVisualizationOptions();
    options.draw_contact_points = true;
    options.draw_constraint_errors = true;

    const auto list = BuildDebugVisualization(input, options);

    ASSERT_EQ(CountCommands(list, DrawCommandType::ContactPoint), 1u);
    ASSERT_EQ(CountCommands(list, DrawCommandType::Line), 1u);

    const auto* contact = FirstCommand(list, DrawCommandType::ContactPoint);
    ASSERT_NE(contact, nullptr);
    ExpectVec3Eq(contact->position, {2.0f, 3.0f, 4.0f});
    ExpectVec3Eq(contact->end, {0.0f, 1.0f, 0.0f});
    EXPECT_FLOAT_EQ(contact->radius, 0.02f);

    const auto* error = FirstCommand(list, DrawCommandType::Line);
    ASSERT_NE(error, nullptr);
    ExpectVec3Eq(error->position, {1.0f, 0.0f, 2.0f});
    ExpectVec3Eq(error->end, {1.25f, 0.0f, 2.0f});
}
