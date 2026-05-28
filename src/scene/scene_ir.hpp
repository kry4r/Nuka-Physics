#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::SceneIR – Canonical intermediate representation for scenes
// ---------------------------------------------------------------------------

#include "scene/canonical_types.hpp"
#include "math/transform.hpp"

#include <string>
#include <vector>

namespace nuka::scene {

// ---------------------------------------------------------------------------
// Record types
// ---------------------------------------------------------------------------

struct RigidBodyRecord {
    std::string name;
    BodyId id                              = kInvalidBody;
    BodyId parent_id                       = kInvalidBody;
    math::Transform local_transform        = math::Transform::Identity();
    math::Transform inertial_transform     = math::Transform::Identity();
    float mass                             = 1.0f;
    math::Vec3 inertia                     = {1.0f, 1.0f, 1.0f};
    bool is_static                         = false;
};

struct CollisionShapeRecord {
    ShapeId id                             = 0;
    BodyId body_id                         = kInvalidBody;
    MaterialId material_id                 = kInvalidMaterial;
    std::string name;
    ShapeType type                         = ShapeType::Box;
    math::Transform local_transform        = math::Transform::Identity();
    math::Vec3 half_extents                = {0.5f, 0.5f, 0.5f};
    float radius                           = 0.5f;
    float half_height                      = 0.5f;
};

struct JointRecord {
    std::string name;
    JointId id                             = kInvalidJoint;
    BodyId parent_body                     = kInvalidBody;
    BodyId child_body                      = kInvalidBody;
    JointType type                         = JointType::Revolute;
    math::Vec3 axis                        = {0.0f, 0.0f, 1.0f};
    math::Transform parent_frame           = math::Transform::Identity();
    math::Transform child_frame            = math::Transform::Identity();
    float lower_limit                      = -3.14159f;
    float upper_limit                      =  3.14159f;
    float damping                          = 0.0f;
    float armature                         = 0.0f;
    float stiffness                        = 0.0f;
};

struct SensorRecord {
    std::string name;
    SensorId id                            = 0;
    BodyId attached_body                   = kInvalidBody;
    SensorType type                        = SensorType::Imu;
    math::Transform local_transform        = math::Transform::Identity();
    float sample_rate_hz                   = 0.0f;
};

struct MaterialRecord {
    std::string name;
    MaterialId id                          = kInvalidMaterial;
    math::Vec3 base_color                  = {1.0f, 1.0f, 1.0f};
    float alpha                            = 1.0f;
    float roughness                        = 0.5f;
    float metallic                         = 0.0f;
};

struct CameraRecord {
    std::string name;
    CameraId id                            = 0;
    BodyId attached_body                   = kInvalidBody;
    math::Transform local_transform        = math::Transform::Identity();
    float vertical_fov_degrees             = 45.0f;
    float near_clip                        = 0.01f;
    float far_clip                         = 1000.0f;
};

struct LightRecord {
    std::string name;
    LightId id                             = 0;
    LightType type                         = LightType::Point;
    BodyId attached_body                   = kInvalidBody;
    math::Transform local_transform        = math::Transform::Identity();
    math::Vec3 color                       = {1.0f, 1.0f, 1.0f};
    float intensity                        = 1.0f;
};

struct ActuatorRecord {
    std::string name;
    ActuatorId id                          = 0;
    JointId joint_id                       = kInvalidJoint;
    ActuatorType type                      = ActuatorType::Motor;
    float gain                             = 1.0f;
    float force_limit                      = 0.0f;
};

// ---------------------------------------------------------------------------
// SceneIR
// ---------------------------------------------------------------------------

class SceneIR {
public:
    // -- mutators -----------------------------------------------------------
    BodyId   AddRigidBody(std::string name);
    BodyId   AddRigidBody(RigidBodyRecord record);

    ShapeId  AddCollisionShape(BodyId body, ShapeType type);
    ShapeId  AddCollisionShape(CollisionShapeRecord record);

    JointId  AddJoint(std::string name, BodyId parent, BodyId child);
    JointId  AddJoint(JointRecord record);

    SensorId AddSensor(std::string name, BodyId body);
    SensorId AddSensor(SensorRecord record);
    MaterialId AddMaterial(MaterialRecord record);
    CameraId AddCamera(CameraRecord record);
    LightId AddLight(LightRecord record);
    ActuatorId AddActuator(ActuatorRecord record);

    // -- counts -------------------------------------------------------------
    size_t RigidBodyCount() const;
    size_t JointCount()     const;
    size_t ShapeCount()     const;
    size_t SensorCount()    const;
    size_t MaterialCount()  const;
    size_t CameraCount()    const;
    size_t LightCount()     const;
    size_t ActuatorCount()  const;

    // -- accessors ----------------------------------------------------------
    const RigidBodyRecord&      GetBody(BodyId id)    const;
    const JointRecord&          GetJoint(JointId id)  const;
    const CollisionShapeRecord& GetShape(ShapeId id)  const;
    const SensorRecord&         GetSensor(SensorId id) const;
    const MaterialRecord&       GetMaterial(MaterialId id) const;
    const CameraRecord&         GetCamera(CameraId id) const;
    const LightRecord&          GetLight(LightId id) const;
    const ActuatorRecord&       GetActuator(ActuatorId id) const;

    RigidBodyRecord&      GetBodyMut(BodyId id);
    JointRecord&          GetJointMut(JointId id);
    CollisionShapeRecord& GetShapeMut(ShapeId id);
    SensorRecord&         GetSensorMut(SensorId id);
    MaterialRecord&       GetMaterialMut(MaterialId id);
    CameraRecord&         GetCameraMut(CameraId id);
    LightRecord&          GetLightMut(LightId id);
    ActuatorRecord&       GetActuatorMut(ActuatorId id);

    // -- bulk accessors -----------------------------------------------------
    const std::vector<RigidBodyRecord>&      Bodies()  const;
    const std::vector<JointRecord>&          Joints()  const;
    const std::vector<CollisionShapeRecord>& Shapes()  const;
    const std::vector<SensorRecord>&         Sensors() const;
    const std::vector<MaterialRecord>&       Materials() const;
    const std::vector<CameraRecord>&         Cameras() const;
    const std::vector<LightRecord>&          Lights() const;
    const std::vector<ActuatorRecord>&       Actuators() const;

private:
    std::vector<RigidBodyRecord>      bodies_;
    std::vector<JointRecord>          joints_;
    std::vector<CollisionShapeRecord> shapes_;
    std::vector<SensorRecord>         sensors_;
    std::vector<MaterialRecord>       materials_;
    std::vector<CameraRecord>         cameras_;
    std::vector<LightRecord>          lights_;
    std::vector<ActuatorRecord>       actuators_;
};

} // namespace nuka::scene
