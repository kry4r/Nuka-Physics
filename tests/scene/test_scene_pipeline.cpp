// ---------------------------------------------------------------------------
// Tests for SceneIR -> SceneGraph / PhysicsWorld / RenderScene conversion
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "import/usd_importer.hpp"
#include "scene/scene_pipeline.hpp"

#include <gtest/gtest.h>

namespace {

nuka::scene::SceneIR BuildProgrammaticRobotScene() {
    nuka::scene::SceneIR scene;

    nuka::scene::MaterialRecord material;
    material.name = "programmatic_mat";
    material.base_color = {0.2f, 0.4f, 0.8f};
    const auto material_id = scene.AddMaterial(std::move(material));

    nuka::scene::RigidBodyRecord base;
    base.name = "base";
    base.is_static = true;
    const auto base_id = scene.AddRigidBody(std::move(base));

    nuka::scene::RigidBodyRecord link;
    link.name = "link";
    link.parent_id = base_id;
    link.local_transform.position = {0.0f, 0.0f, 1.0f};
    link.mass = 2.0f;
    const auto link_id = scene.AddRigidBody(std::move(link));

    nuka::scene::CollisionShapeRecord shape;
    shape.name = "link_box";
    shape.body_id = link_id;
    shape.material_id = material_id;
    shape.type = nuka::scene::ShapeType::Box;
    shape.half_extents = {0.1f, 0.2f, 0.3f};
    scene.AddCollisionShape(std::move(shape));

    nuka::scene::JointRecord joint;
    joint.name = "hinge";
    joint.parent_body = base_id;
    joint.child_body = link_id;
    joint.axis = {0.0f, 0.0f, 1.0f};
    const auto joint_id = scene.AddJoint(std::move(joint));

    nuka::scene::ActuatorRecord actuator;
    actuator.name = "motor";
    actuator.joint_id = joint_id;
    actuator.type = nuka::scene::ActuatorType::Velocity;
    actuator.gain = 5.0f;
    scene.AddActuator(std::move(actuator));

    nuka::scene::SensorRecord sensor;
    sensor.name = "imu";
    sensor.attached_body = link_id;
    sensor.type = nuka::scene::SensorType::Imu;
    sensor.sample_rate_hz = 240.0f;
    scene.AddSensor(std::move(sensor));

    nuka::scene::CameraRecord camera;
    camera.name = "camera";
    camera.attached_body = link_id;
    scene.AddCamera(std::move(camera));

    nuka::scene::LightRecord light;
    light.name = "sun";
    light.type = nuka::scene::LightType::Directional;
    scene.AddLight(std::move(light));

    return scene;
}

} // namespace

TEST(ScenePipeline, ProgrammaticSceneBuildsAllInternalViews) {
    const auto compiled = nuka::scene::BuildCompiledScene(BuildProgrammaticRobotScene());

    EXPECT_EQ(compiled.graph.NodeCount(), 2u);
    EXPECT_EQ(compiled.physics.body_count, 2u);
    EXPECT_EQ(compiled.physics.joint_count, 1u);
    EXPECT_EQ(compiled.physics.shape_count, 1u);
    EXPECT_EQ(compiled.physics.actuator_count, 1u);
    EXPECT_EQ(compiled.physics.sensor_count, 1u);
    EXPECT_EQ(compiled.render.mesh_instance_count, 1u);
    EXPECT_EQ(compiled.render.material_count, 1u);
    EXPECT_EQ(compiled.render.camera_count, 1u);
    EXPECT_EQ(compiled.render.light_count, 1u);
    EXPECT_EQ(compiled.render.debug_proxy_count, 1u);
}

TEST(ScenePipeline, ComputesWorldTransformsFromBodyHierarchy) {
    const auto compiled = nuka::scene::BuildCompiledScene(BuildProgrammaticRobotScene());

    ASSERT_EQ(compiled.graph.NodeCount(), 2u);
    EXPECT_EQ(compiled.graph.GetNode(1).parent, 0u);
    EXPECT_FLOAT_EQ(compiled.graph.GetNode(1).world_transform.position.z, 1.0f);
}

TEST(ScenePipeline, ImportedMjcfSceneBuildsPhysicsAndRenderViews) {
    const auto scene = nuka::import::LoadMjcf("tests/data/complete_robot.xml");
    const auto compiled = nuka::scene::BuildCompiledScene(scene);

    EXPECT_EQ(compiled.graph.NodeCount(), 2u);
    EXPECT_EQ(compiled.physics.body_count, 2u);
    EXPECT_EQ(compiled.physics.joint_count, 1u);
    EXPECT_EQ(compiled.physics.actuator_count, 1u);
    EXPECT_EQ(compiled.physics.sensor_count, 1u);
    EXPECT_EQ(compiled.render.mesh_instance_count, 2u);
    EXPECT_EQ(compiled.render.material_count, 1u);
    EXPECT_EQ(compiled.render.camera_count, 1u);
    EXPECT_EQ(compiled.render.light_count, 1u);
}

TEST(ScenePipeline, ImportedUsdSceneBuildsPhysicsAndRenderViews) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");
    const auto compiled = nuka::scene::BuildCompiledScene(scene);

    EXPECT_EQ(compiled.graph.NodeCount(), 2u);
    EXPECT_EQ(compiled.physics.body_count, 2u);
    EXPECT_EQ(compiled.physics.joint_count, 1u);
    EXPECT_EQ(compiled.physics.actuator_count, 1u);
    EXPECT_EQ(compiled.physics.sensor_count, 1u);
    EXPECT_EQ(compiled.render.mesh_instance_count, 2u);
    EXPECT_EQ(compiled.render.material_count, 1u);
    EXPECT_EQ(compiled.render.camera_count, 1u);
    EXPECT_EQ(compiled.render.light_count, 1u);
}

TEST(ScenePipeline, RenderSceneKeepsShapeMaterialAndBodyBindings) {
    const auto compiled = nuka::scene::BuildCompiledScene(BuildProgrammaticRobotScene());

    ASSERT_EQ(compiled.render.mesh_instance_count, 1u);
    EXPECT_EQ(compiled.render.mesh_instances[0].body_id, 1u);
    EXPECT_EQ(compiled.render.mesh_instances[0].material_id, 0u);
    EXPECT_EQ(compiled.render.mesh_instances[0].shape_type, nuka::scene::ShapeType::Box);
}
