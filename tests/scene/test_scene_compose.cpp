// ---------------------------------------------------------------------------
// Tests for nuka::scene::Compose – format-agnostic SceneIR merge layer
// (USD <-> MJCF coexistence core; v0.7 must-do)
// ---------------------------------------------------------------------------

#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

using namespace nuka::scene;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

namespace {

// Build a small but cross-referenced addon SceneIR:
//   body 0 = root (no parent), with a non-identity local transform
//   body 1 = child of body 0
//   material 0
//   shape 0 on body 1, material 0
//   shape 1 on body 0, kInvalidMaterial
//   joint 0: parent body 0 -> child body 1
//   actuator 0 on joint 0
//   sensor 0 attached to body 1
//   camera 0 attached to body 0
//   light 0 attached to body 1
SceneIR MakeAddon() {
    SceneIR s;

    RigidBodyRecord root;
    root.name = "root";
    root.parent_id = kInvalidBody;
    root.local_transform = Transform{Vec3{10.0f, 20.0f, 30.0f},
                                     Quat::FromAxisAngle(Vec3::UnitZ(), 0.5f)};
    const BodyId b0 = s.AddRigidBody(std::move(root));

    RigidBodyRecord child;
    child.name = "child";
    child.parent_id = b0;
    child.local_transform = Transform{Vec3{1.0f, 0.0f, 0.0f}, Quat::Identity()};
    const BodyId b1 = s.AddRigidBody(std::move(child));

    MaterialRecord mat;
    mat.name = "addon_mat";
    const MaterialId m0 = s.AddMaterial(std::move(mat));

    CollisionShapeRecord shape0;
    shape0.name = "shape_on_child";
    shape0.body_id = b1;
    shape0.material_id = m0;
    shape0.type = ShapeType::Sphere;
    s.AddCollisionShape(std::move(shape0));

    CollisionShapeRecord shape1;
    shape1.name = "shape_no_mat";
    shape1.body_id = b0;
    shape1.material_id = kInvalidMaterial;
    shape1.type = ShapeType::Box;
    s.AddCollisionShape(std::move(shape1));

    JointRecord joint;
    joint.name = "addon_joint";
    joint.parent_body = b0;
    joint.child_body = b1;
    const JointId j0 = s.AddJoint(std::move(joint));

    ActuatorRecord act;
    act.name = "addon_act";
    act.joint_id = j0;
    s.AddActuator(std::move(act));

    SensorDesc sensor;
    sensor.name = "addon_sensor";
    sensor.mount = MountFrame::Body;
    sensor.mount_index = b1;
    s.AddSensor(std::move(sensor));

    CameraRecord cam;
    cam.name = "addon_cam";
    cam.attached_body = b0;
    s.AddCamera(std::move(cam));

    LightRecord light;
    light.name = "addon_light";
    light.attached_body = b1;
    s.AddLight(std::move(light));

    return s;
}

// Build a base scene of arbitrary nonzero counts so offsets are exercised.
//   2 bodies, 1 material, 1 shape, 1 joint, 1 actuator, 1 sensor, 1 camera, 1 light
SceneIR MakeBase() {
    SceneIR s;
    const BodyId bb0 = s.AddRigidBody("base_root");
    const BodyId bb1 = s.AddRigidBody("base_b1");

    MaterialRecord mat;
    mat.name = "base_mat";
    s.AddMaterial(std::move(mat));

    CollisionShapeRecord shape;
    shape.body_id = bb1;
    shape.material_id = 0;
    s.AddCollisionShape(std::move(shape));

    s.AddJoint("base_joint", bb0, bb1);

    ActuatorRecord act;
    act.name = "base_act";
    act.joint_id = 0;
    s.AddActuator(std::move(act));

    SensorDesc sensor;
    sensor.name = "base_sensor";
    sensor.mount = MountFrame::Body;
    sensor.mount_index = bb0;
    s.AddSensor(std::move(sensor));

    CameraRecord cam;
    cam.name = "base_cam";
    cam.attached_body = bb0;
    s.AddCamera(std::move(cam));

    LightRecord light;
    light.name = "base_light";
    light.attached_body = bb1;
    s.AddLight(std::move(light));

    return s;
}

bool TransformsNear(const Transform& a, const Transform& b, float eps = 1e-5f) {
    return std::abs(a.position.x - b.position.x) < eps &&
           std::abs(a.position.y - b.position.y) < eps &&
           std::abs(a.position.z - b.position.z) < eps &&
           std::abs(a.rotation.w - b.rotation.w) < eps &&
           std::abs(a.rotation.x - b.rotation.x) < eps &&
           std::abs(a.rotation.y - b.rotation.y) < eps &&
           std::abs(a.rotation.z - b.rotation.z) < eps;
}

}  // namespace

// ---------------------------------------------------------------------------
// Counts add
// ---------------------------------------------------------------------------

TEST(SceneCompose, CountsAdd) {
    const SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();
    const Transform placement = Transform::Identity();

    const SceneIR out = Compose(base, addon, placement);

    EXPECT_EQ(out.RigidBodyCount(), base.RigidBodyCount() + addon.RigidBodyCount());
    EXPECT_EQ(out.JointCount(), base.JointCount() + addon.JointCount());
    EXPECT_EQ(out.ShapeCount(), base.ShapeCount() + addon.ShapeCount());
    EXPECT_EQ(out.MaterialCount(), base.MaterialCount() + addon.MaterialCount());
    EXPECT_EQ(out.ActuatorCount(), base.ActuatorCount() + addon.ActuatorCount());
    EXPECT_EQ(out.SensorCount(), base.SensorCount() + addon.SensorCount());
    EXPECT_EQ(out.CameraCount(), base.CameraCount() + addon.CameraCount());
    EXPECT_EQ(out.LightCount(), base.LightCount() + addon.LightCount());
}

// ---------------------------------------------------------------------------
// Body id offset + parent remap
// ---------------------------------------------------------------------------

TEST(SceneCompose, BodyParentRemapAndRootStaysRoot) {
    const SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();
    const Transform placement = Transform::Identity();

    const SceneIR out = Compose(base, addon, placement);
    const size_t boff = base.RigidBodyCount();

    // addon root (index 0) -> base+0, parent stays kInvalidBody (root).
    const RigidBodyRecord& root = out.GetBody(static_cast<BodyId>(boff + 0));
    EXPECT_EQ(root.parent_id, kInvalidBody);

    // addon child (index 1) -> base+1, parent_id offset to base+0.
    const RigidBodyRecord& child = out.GetBody(static_cast<BodyId>(boff + 1));
    EXPECT_EQ(child.parent_id, static_cast<BodyId>(boff + 0));
}

TEST(SceneCompose, RootGetsPlacementComposedTransform) {
    const SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();

    const Transform placement{Vec3{100.0f, 0.0f, 0.0f},
                              Quat::FromAxisAngle(Vec3::UnitY(), 1.0f)};

    const SceneIR out = Compose(base, addon, placement);
    const size_t boff = base.RigidBodyCount();

    const Transform original = addon.GetBody(0).local_transform;
    const Transform expected = placement * original;  // pins composition order

    const Transform& got = out.GetBody(static_cast<BodyId>(boff + 0)).local_transform;
    EXPECT_TRUE(TransformsNear(got, expected))
        << "root transform must equal placement * original_local";

    // Non-root child keeps its own local transform unchanged.
    const Transform childOrig = addon.GetBody(1).local_transform;
    const Transform& childGot = out.GetBody(static_cast<BodyId>(boff + 1)).local_transform;
    EXPECT_TRUE(TransformsNear(childGot, childOrig))
        << "non-root addon body local transform must be unchanged";
}

// ---------------------------------------------------------------------------
// Shape remap (body_id + material_id), sentinel material preserved
// ---------------------------------------------------------------------------

TEST(SceneCompose, ShapeBodyAndMaterialRemap) {
    const SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();
    const SceneIR out = Compose(base, addon, Transform::Identity());

    const size_t boff = base.RigidBodyCount();
    const size_t moff = base.MaterialCount();
    const size_t soff = base.ShapeCount();

    // addon shape 0 was on body 1, material 0.
    const CollisionShapeRecord& s0 = out.GetShape(static_cast<ShapeId>(soff + 0));
    EXPECT_EQ(s0.body_id, static_cast<BodyId>(boff + 1));
    EXPECT_EQ(s0.material_id, static_cast<MaterialId>(moff + 0));

    // addon shape 1 was on body 0, kInvalidMaterial -> stays kInvalidMaterial.
    const CollisionShapeRecord& s1 = out.GetShape(static_cast<ShapeId>(soff + 1));
    EXPECT_EQ(s1.body_id, static_cast<BodyId>(boff + 0));
    EXPECT_EQ(s1.material_id, kInvalidMaterial);
}

// ---------------------------------------------------------------------------
// Joint remap (parent_body + child_body)
// ---------------------------------------------------------------------------

TEST(SceneCompose, JointBodyRemap) {
    const SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();
    const SceneIR out = Compose(base, addon, Transform::Identity());

    const size_t boff = base.RigidBodyCount();
    const size_t joff = base.JointCount();

    const JointRecord& j = out.GetJoint(static_cast<JointId>(joff + 0));
    EXPECT_EQ(j.parent_body, static_cast<BodyId>(boff + 0));
    EXPECT_EQ(j.child_body, static_cast<BodyId>(boff + 1));
}

// ---------------------------------------------------------------------------
// Actuator / sensor / camera / light cross-ref remap
// ---------------------------------------------------------------------------

TEST(SceneCompose, ActuatorJointRemap) {
    const SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();
    const SceneIR out = Compose(base, addon, Transform::Identity());

    const size_t joff = base.JointCount();
    const size_t aoff = base.ActuatorCount();

    const ActuatorRecord& a = out.GetActuator(static_cast<ActuatorId>(aoff + 0));
    EXPECT_EQ(a.joint_id, static_cast<JointId>(joff + 0));
}

TEST(SceneCompose, SensorCameraLightBodyRemap) {
    const SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();
    const SceneIR out = Compose(base, addon, Transform::Identity());

    const size_t boff = base.RigidBodyCount();
    const size_t senoff = base.SensorCount();
    const size_t camoff = base.CameraCount();
    const size_t ligoff = base.LightCount();

    EXPECT_EQ(out.GetSensor(static_cast<SensorId>(senoff + 0)).mount_index,
              static_cast<BodyId>(boff + 1));
    EXPECT_EQ(out.GetCamera(static_cast<CameraId>(camoff + 0)).attached_body,
              static_cast<BodyId>(boff + 0));
    EXPECT_EQ(out.GetLight(static_cast<LightId>(ligoff + 0)).attached_body,
              static_cast<BodyId>(boff + 1));
}

// A light/sensor/camera with a kInvalid attached_body stays kInvalid.
TEST(SceneCompose, InvalidAttachedBodyStaysInvalid) {
    const SceneIR base = MakeBase();

    SceneIR addon;
    LightRecord light;
    light.name = "free_light";
    light.attached_body = kInvalidBody;
    addon.AddLight(std::move(light));

    const SceneIR out = Compose(base, addon, Transform::Identity());
    const size_t ligoff = base.LightCount();
    EXPECT_EQ(out.GetLight(static_cast<LightId>(ligoff + 0)).attached_body,
              kInvalidBody);
}

// ---------------------------------------------------------------------------
// Base preserved (Compose is pure; returns a new SceneIR)
// ---------------------------------------------------------------------------

TEST(SceneCompose, BaseUnchanged) {
    SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();

    const size_t b = base.RigidBodyCount();
    const size_t j = base.JointCount();
    const size_t s = base.ShapeCount();
    const size_t m = base.MaterialCount();
    const size_t a = base.ActuatorCount();
    const size_t sen = base.SensorCount();
    const size_t cam = base.CameraCount();
    const size_t lig = base.LightCount();

    const SceneIR out = Compose(base, addon, Transform::Identity());
    (void)out;

    EXPECT_EQ(base.RigidBodyCount(), b);
    EXPECT_EQ(base.JointCount(), j);
    EXPECT_EQ(base.ShapeCount(), s);
    EXPECT_EQ(base.MaterialCount(), m);
    EXPECT_EQ(base.ActuatorCount(), a);
    EXPECT_EQ(base.SensorCount(), sen);
    EXPECT_EQ(base.CameraCount(), cam);
    EXPECT_EQ(base.LightCount(), lig);
}

// ---------------------------------------------------------------------------
// Name prefix
// ---------------------------------------------------------------------------

TEST(SceneCompose, NamePrefixApplied) {
    const SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();
    const std::string prefix = "cup/";

    const SceneIR out = Compose(base, addon, Transform::Identity(), prefix);
    const size_t boff = base.RigidBodyCount();

    EXPECT_EQ(out.GetBody(static_cast<BodyId>(boff + 0)).name, prefix + "root");
    EXPECT_EQ(out.GetBody(static_cast<BodyId>(boff + 1)).name, prefix + "child");
}

TEST(SceneCompose, NoPrefixLeavesNamesUnchanged) {
    const SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();

    const SceneIR out = Compose(base, addon, Transform::Identity());
    const size_t boff = base.RigidBodyCount();

    EXPECT_EQ(out.GetBody(static_cast<BodyId>(boff + 0)).name, "root");
    EXPECT_EQ(out.GetBody(static_cast<BodyId>(boff + 1)).name, "child");

    // base names untouched as well.
    EXPECT_EQ(out.GetBody(0).name, "base_root");
}

// ---------------------------------------------------------------------------
// Determinism (D1): compose twice -> identical results
// ---------------------------------------------------------------------------

TEST(SceneCompose, DeterministicAcrossRuns) {
    const SceneIR base = MakeBase();
    const SceneIR addon = MakeAddon();
    const Transform placement{Vec3{3.0f, -2.0f, 7.0f},
                              Quat::FromAxisAngle(Vec3::UnitX(), 0.75f)};

    const SceneIR out1 = Compose(base, addon, placement, "p/");
    const SceneIR out2 = Compose(base, addon, placement, "p/");

    ASSERT_EQ(out1.RigidBodyCount(), out2.RigidBodyCount());
    ASSERT_EQ(out1.JointCount(), out2.JointCount());

    for (size_t i = 0; i < out1.RigidBodyCount(); ++i) {
        const RigidBodyRecord& a = out1.GetBody(static_cast<BodyId>(i));
        const RigidBodyRecord& b = out2.GetBody(static_cast<BodyId>(i));
        EXPECT_EQ(a.name, b.name);
        EXPECT_EQ(a.parent_id, b.parent_id);
        // bitwise-identical transform (no reassociation).
        EXPECT_EQ(a.local_transform.position.x, b.local_transform.position.x);
        EXPECT_EQ(a.local_transform.position.y, b.local_transform.position.y);
        EXPECT_EQ(a.local_transform.position.z, b.local_transform.position.z);
        EXPECT_EQ(a.local_transform.rotation.w, b.local_transform.rotation.w);
        EXPECT_EQ(a.local_transform.rotation.x, b.local_transform.rotation.x);
        EXPECT_EQ(a.local_transform.rotation.y, b.local_transform.rotation.y);
        EXPECT_EQ(a.local_transform.rotation.z, b.local_transform.rotation.z);
    }
    for (size_t i = 0; i < out1.JointCount(); ++i) {
        EXPECT_EQ(out1.GetJoint(static_cast<JointId>(i)).parent_body,
                  out2.GetJoint(static_cast<JointId>(i)).parent_body);
        EXPECT_EQ(out1.GetJoint(static_cast<JointId>(i)).child_body,
                  out2.GetJoint(static_cast<JointId>(i)).child_body);
    }
}
