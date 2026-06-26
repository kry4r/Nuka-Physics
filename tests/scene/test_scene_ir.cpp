// ---------------------------------------------------------------------------
// Tests for nuka::scene::SceneIR
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

using namespace nuka::scene;

// ---------------------------------------------------------------------------
// Basic body + joint
// ---------------------------------------------------------------------------

TEST(SceneIr, SupportsRigidBodyAndJointRecords) {
    SceneIR scene;
    const auto body  = scene.AddRigidBody("base_link");
    const auto joint = scene.AddJoint("joint0", body, body);
    EXPECT_EQ(scene.RigidBodyCount(), 1u);
    EXPECT_EQ(scene.JointCount(), 1u);
    EXPECT_EQ(body, 0u);
    EXPECT_EQ(joint, 0u);
}

// ---------------------------------------------------------------------------
// Collision shapes
// ---------------------------------------------------------------------------

TEST(SceneIr, AddCollisionShapeSimple) {
    SceneIR scene;
    const auto body  = scene.AddRigidBody("box_body");
    const auto shape = scene.AddCollisionShape(body, ShapeType::Box);
    EXPECT_EQ(scene.ShapeCount(), 1u);
    EXPECT_EQ(scene.GetShape(shape).body_id, body);
    EXPECT_EQ(scene.GetShape(shape).type, ShapeType::Box);
}

TEST(SceneIr, AddCollisionShapeWithRecord) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("sphere_body");

    CollisionShapeRecord rec;
    rec.body_id = body;
    rec.type    = ShapeType::Sphere;
    rec.radius  = 1.5f;
    const auto shape = scene.AddCollisionShape(std::move(rec));

    EXPECT_EQ(scene.ShapeCount(), 1u);
    EXPECT_FLOAT_EQ(scene.GetShape(shape).radius, 1.5f);
    EXPECT_EQ(scene.GetShape(shape).type, ShapeType::Sphere);
}

// ---------------------------------------------------------------------------
// Sensors
// ---------------------------------------------------------------------------

TEST(SceneIr, AddSensor) {
    SceneIR scene;
    const auto body   = scene.AddRigidBody("sensor_body");
    const auto sensor = scene.AddSensor("imu0", body);
    EXPECT_EQ(scene.SensorCount(), 1u);
    EXPECT_EQ(scene.GetSensor(sensor).mount, MountFrame::Body);
    EXPECT_EQ(scene.GetSensor(sensor).mount_index, body);
    EXPECT_EQ(scene.GetSensor(sensor).name, "imu0");
    EXPECT_EQ(scene.GetSensor(sensor).type, SensorType::Imu);
}

TEST(SceneIr, SupportsRenderAndControlRecords) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("robot");
    const auto child = scene.AddRigidBody("link");
    const auto joint = scene.AddJoint("hinge", body, child);

    MaterialRecord material;
    material.name = "rubber";
    material.base_color = {0.1f, 0.2f, 0.3f};
    material.roughness = 0.8f;
    const auto material_id = scene.AddMaterial(std::move(material));

    CameraRecord camera;
    camera.name = "wrist_camera";
    camera.attached_body = child;
    camera.vertical_fov_degrees = 60.0f;
    const auto camera_id = scene.AddCamera(std::move(camera));

    LightRecord light;
    light.name = "key_light";
    light.type = LightType::Directional;
    light.intensity = 400.0f;
    const auto light_id = scene.AddLight(std::move(light));

    ActuatorRecord actuator;
    actuator.name = "hinge_motor";
    actuator.joint_id = joint;
    actuator.type = ActuatorType::Position;
    actuator.gain = 25.0f;
    const auto actuator_id = scene.AddActuator(std::move(actuator));

    EXPECT_EQ(scene.MaterialCount(), 1u);
    EXPECT_EQ(scene.CameraCount(), 1u);
    EXPECT_EQ(scene.LightCount(), 1u);
    EXPECT_EQ(scene.ActuatorCount(), 1u);
    EXPECT_EQ(scene.GetMaterial(material_id).name, "rubber");
    EXPECT_FLOAT_EQ(scene.GetMaterial(material_id).base_color.y, 0.2f);
    EXPECT_EQ(scene.GetCamera(camera_id).attached_body, child);
    EXPECT_EQ(scene.GetLight(light_id).type, LightType::Directional);
    EXPECT_EQ(scene.GetActuator(actuator_id).joint_id, joint);
    EXPECT_FLOAT_EQ(scene.GetActuator(actuator_id).gain, 25.0f);
}

// ---------------------------------------------------------------------------
// Record field access
// ---------------------------------------------------------------------------

TEST(SceneIr, BodyRecordFieldDefaults) {
    SceneIR scene;
    const auto id = scene.AddRigidBody("link");
    const auto& body = scene.GetBody(id);
    EXPECT_EQ(body.name, "link");
    EXPECT_EQ(body.parent_id, kInvalidBody);
    EXPECT_FLOAT_EQ(body.mass, 1.0f);
    EXPECT_FALSE(body.is_static);
}

TEST(SceneIr, BodyRecordWithCustomFields) {
    SceneIR scene;
    RigidBodyRecord rec;
    rec.name      = "ground";
    rec.mass      = 0.0f;
    rec.is_static = true;
    rec.inertia   = {0.0f, 0.0f, 0.0f};
    const auto id = scene.AddRigidBody(std::move(rec));

    const auto& body = scene.GetBody(id);
    EXPECT_TRUE(body.is_static);
    EXPECT_FLOAT_EQ(body.mass, 0.0f);
}

TEST(SceneIr, JointRecordFieldAccess) {
    SceneIR scene;
    const auto b0 = scene.AddRigidBody("parent");
    const auto b1 = scene.AddRigidBody("child");
    const auto jid = scene.AddJoint("hinge", b0, b1);

    const auto& j = scene.GetJoint(jid);
    EXPECT_EQ(j.name, "hinge");
    EXPECT_EQ(j.parent_body, b0);
    EXPECT_EQ(j.child_body, b1);
    EXPECT_EQ(j.type, JointType::Revolute);
}

// ---------------------------------------------------------------------------
// Multiple bodies
// ---------------------------------------------------------------------------

TEST(SceneIr, MultipleBodies) {
    SceneIR scene;
    const auto a = scene.AddRigidBody("a");
    const auto b = scene.AddRigidBody("b");
    const auto c = scene.AddRigidBody("c");

    EXPECT_EQ(scene.RigidBodyCount(), 3u);
    EXPECT_EQ(a, 0u);
    EXPECT_EQ(b, 1u);
    EXPECT_EQ(c, 2u);

    EXPECT_EQ(scene.GetBody(a).name, "a");
    EXPECT_EQ(scene.GetBody(b).name, "b");
    EXPECT_EQ(scene.GetBody(c).name, "c");
}

// ---------------------------------------------------------------------------
// Bulk accessors
// ---------------------------------------------------------------------------

TEST(SceneIr, BulkAccessors) {
    SceneIR scene;
    scene.AddRigidBody("r0");
    scene.AddRigidBody("r1");
    scene.AddJoint("j0", 0, 1);
    scene.AddCollisionShape(0, ShapeType::Sphere);
    scene.AddSensor("s0", 0);
    scene.AddMaterial(MaterialRecord{"mat"});
    scene.AddCamera(CameraRecord{"cam"});
    scene.AddLight(LightRecord{"light"});
    scene.AddActuator(ActuatorRecord{"motor"});

    EXPECT_EQ(scene.Bodies().size(), 2u);
    EXPECT_EQ(scene.Joints().size(), 1u);
    EXPECT_EQ(scene.Shapes().size(), 1u);
    EXPECT_EQ(scene.Sensors().size(), 1u);
    EXPECT_EQ(scene.Materials().size(), 1u);
    EXPECT_EQ(scene.Cameras().size(), 1u);
    EXPECT_EQ(scene.Lights().size(), 1u);
    EXPECT_EQ(scene.Actuators().size(), 1u);
}

// ---------------------------------------------------------------------------
// Out-of-range access throws
// ---------------------------------------------------------------------------

TEST(SceneIr, OutOfRangeThrows) {
    SceneIR scene;
    EXPECT_THROW(scene.GetBody(0), std::out_of_range);
    EXPECT_THROW(scene.GetJoint(0), std::out_of_range);
    EXPECT_THROW(scene.GetShape(0), std::out_of_range);
    EXPECT_THROW(scene.GetSensor(0), std::out_of_range);
    EXPECT_THROW(scene.GetMaterial(0), std::out_of_range);
    EXPECT_THROW(scene.GetCamera(0), std::out_of_range);
    EXPECT_THROW(scene.GetLight(0), std::out_of_range);
    EXPECT_THROW(scene.GetActuator(0), std::out_of_range);
    EXPECT_THROW(scene.GetMedia(0), std::out_of_range);
}

// ---------------------------------------------------------------------------
// Media records: add / round-trip / projection / copy
// ---------------------------------------------------------------------------

TEST(SceneIr, AddMediaReturnsValidIdAndRoundTripsFields) {
    SceneIR scene;

    MediaRecord rec;
    rec.name   = "pool";
    rec.kind   = MediaRecord::Kind::Fluid;
    rec.method = MediaRecord::Method::Pbf;
    rec.fluid_box.min     = {-0.5f, -0.5f, 0.0f};
    rec.fluid_box.max     = {0.5f, 0.5f, 0.4f};
    rec.fluid_box.spacing = 0.02f;
    rec.pbf.rest_density  = 1000.0f;
    rec.pbf.support_scale = 2.0f;
    rec.pbf.iters         = 6u;
    rec.pbf.walls_enabled   = true;
    rec.pbf.walls_min       = {-0.5f, -0.5f, 0.0f};
    rec.pbf.walls_max       = {0.5f, 0.5f, 1.0f};
    rec.pbf.floor_z         = 0.0f;
    rec.pbf.boundary_layers = 2u;
    rec.render_skin.normal_offset = 0.004f;
    rec.render_skin.smooth_iters  = 3u;
    rec.baked = ParseAssetRef("pool.nka#TETM/0");
    rec.render_material_id = 7u;

    const auto id = scene.AddMedia(rec);
    EXPECT_EQ(id, 0u);
    EXPECT_NE(id, kInvalidMedia);
    EXPECT_EQ(scene.MediaCount(), 1u);
    ASSERT_EQ(scene.Media().size(), 1u);

    const auto& back = scene.Media()[0];
    EXPECT_EQ(back.id, id);                      // AddMedia stamps the dense id
    EXPECT_EQ(back.name, "pool");
    EXPECT_EQ(back.kind, MediaRecord::Kind::Fluid);
    EXPECT_EQ(back.method, MediaRecord::Method::Pbf);
    EXPECT_FLOAT_EQ(back.fluid_box.spacing, 0.02f);
    EXPECT_FLOAT_EQ(back.fluid_box.max.z, 0.4f);
    EXPECT_FLOAT_EQ(back.pbf.rest_density, 1000.0f);
    EXPECT_EQ(back.pbf.iters, 6u);
    EXPECT_TRUE(back.pbf.walls_enabled);
    EXPECT_EQ(back.pbf.boundary_layers, 2u);
    EXPECT_FLOAT_EQ(back.render_skin.normal_offset, 0.004f);
    EXPECT_EQ(back.render_skin.smooth_iters, 3u);
    EXPECT_EQ(back.baked, ParseAssetRef("pool.nka#TETM/0"));
    EXPECT_EQ(back.render_material_id, 7u);
    EXPECT_EQ(scene.GetMedia(id).name, "pool");   // GetMedia mirrors the bulk accessor
}

TEST(SceneIr, ProjectMediaClothWritesSystemKindAndSoftBody) {
    SceneIR scene;
    const auto body_entity = scene.EntityOfBody(scene.AddRigidBody("anchor"));

    MediaRecord rec;
    rec.name   = "drape";
    rec.kind   = MediaRecord::Kind::Cloth;
    rec.method = MediaRecord::Method::Xpbd;
    rec.cloth_grid.nx = 8u;
    rec.cloth_grid.ny = 8u;
    rec.xpbd.distance_alpha = 1.0e-5f;
    const auto id = scene.AddMedia(rec);

    const EntityId media = scene.EntityOfMedia(id);
    ASSERT_NE(media, kInvalidEntity);
    EXPECT_NE(media, body_entity);                // a fresh entity, not the body
    EXPECT_TRUE(scene.Ecs().Alive(media));

    const auto* sk = scene.Ecs().Get<SystemKindComponent>(media);
    ASSERT_NE(sk, nullptr);
    EXPECT_EQ(sk->kind, SystemKindComponent::Cloth);

    const auto* sb = scene.Ecs().Get<SoftBodyComponent>(media);
    ASSERT_NE(sb, nullptr);
    EXPECT_EQ(sb->sim_method, SoftBodyComponent::SimMethod::Xpbd);
    EXPECT_FLOAT_EQ(sb->xpbd_compliance, 1.0e-5f);
    EXPECT_EQ(scene.Ecs().Get<FluidComponent>(media), nullptr);
}

TEST(SceneIr, ProjectMediaTetMpmWritesSoftKindAndMpmMethod) {
    SceneIR scene;

    MediaRecord rec;
    rec.name   = "jelly";
    rec.kind   = MediaRecord::Kind::SoftTet;
    rec.method = MediaRecord::Method::MlsMpm;
    rec.mpm.youngs  = 5.0e5f;
    rec.mpm.poisson = 0.3f;
    const auto id = scene.AddMedia(rec);

    const EntityId media = scene.EntityOfMedia(id);
    ASSERT_NE(media, kInvalidEntity);
    const auto* sk = scene.Ecs().Get<SystemKindComponent>(media);
    ASSERT_NE(sk, nullptr);
    EXPECT_EQ(sk->kind, SystemKindComponent::Soft);

    const auto* sb = scene.Ecs().Get<SoftBodyComponent>(media);
    ASSERT_NE(sb, nullptr);
    EXPECT_EQ(sb->sim_method, SoftBodyComponent::SimMethod::MlsMpm);
    EXPECT_FLOAT_EQ(sb->young, 5.0e5f);
    EXPECT_FLOAT_EQ(sb->poisson, 0.3f);
}

TEST(SceneIr, ProjectMediaFluidWritesFluidComponent) {
    SceneIR scene;

    MediaRecord rec;
    rec.name   = "water";
    rec.kind   = MediaRecord::Kind::Fluid;
    rec.method = MediaRecord::Method::Pbf;
    rec.fluid_box.spacing = 0.025f;
    rec.pbf.rest_density  = 998.0f;
    const auto id = scene.AddMedia(rec);

    const EntityId media = scene.EntityOfMedia(id);
    ASSERT_NE(media, kInvalidEntity);
    const auto* sk = scene.Ecs().Get<SystemKindComponent>(media);
    ASSERT_NE(sk, nullptr);
    EXPECT_EQ(sk->kind, SystemKindComponent::Fluid);

    const auto* fl = scene.Ecs().Get<FluidComponent>(media);
    ASSERT_NE(fl, nullptr);
    EXPECT_FLOAT_EQ(fl->particle_size, 0.025f);
    EXPECT_FLOAT_EQ(fl->rho0, 998.0f);
    EXPECT_EQ(scene.Ecs().Get<SoftBodyComponent>(media), nullptr);
}

TEST(SceneIr, CopyCtorPreservesMediaAndReprojects) {
    SceneIR scene;
    MediaRecord rec;
    rec.name   = "drape";
    rec.kind   = MediaRecord::Kind::Cloth;
    rec.method = MediaRecord::Method::Xpbd;
    const auto id = scene.AddMedia(rec);

    SceneIR copy(scene);
    ASSERT_EQ(copy.MediaCount(), 1u);
    EXPECT_EQ(copy.GetMedia(id).name, "drape");
    EXPECT_EQ(copy.GetMedia(id).kind, MediaRecord::Kind::Cloth);

    // The deep copy rebuilds the facade, so the media entity re-projects.
    const EntityId media = copy.EntityOfMedia(id);
    ASSERT_NE(media, kInvalidEntity);
    const auto* sk = copy.Ecs().Get<SystemKindComponent>(media);
    ASSERT_NE(sk, nullptr);
    EXPECT_EQ(sk->kind, SystemKindComponent::Cloth);
    ASSERT_NE(copy.Ecs().Get<SoftBodyComponent>(media), nullptr);
}

TEST(SceneIr, ZeroMediaSceneProjectsNoMediaEntity) {
    SceneIR scene;
    scene.AddRigidBody("base");
    EXPECT_EQ(scene.MediaCount(), 0u);
    EXPECT_TRUE(scene.Media().empty());
    // No media id is valid, so no media entity exists (byte-identity guard).
    EXPECT_EQ(scene.EntityOfMedia(0), kInvalidEntity);
    size_t fluids = 0, softs = 0;
    scene.Ecs().ForEach<FluidComponent>([&](EntityId, const FluidComponent&) { ++fluids; });
    scene.Ecs().ForEach<SoftBodyComponent>([&](EntityId, const SoftBodyComponent&) { ++softs; });
    EXPECT_EQ(fluids, 0u);
    EXPECT_EQ(softs, 0u);
}
