// ---------------------------------------------------------------------------
// nuka::scene::SceneIR implementation
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

#include <stdexcept>
#include <utility>

namespace nuka::scene {

BodyId SceneIR::AddRigidBody(std::string name) {
    RigidBodyRecord rec;
    rec.name = std::move(name);
    return AddRigidBody(std::move(rec));
}

BodyId SceneIR::AddRigidBody(RigidBodyRecord record) {
    const auto id = static_cast<BodyId>(bodies_.size());
    record.id = id;
    bodies_.push_back(std::move(record));
    return id;
}

ShapeId SceneIR::AddCollisionShape(BodyId body, ShapeType type) {
    CollisionShapeRecord rec;
    rec.body_id = body;
    rec.type = type;
    return AddCollisionShape(std::move(rec));
}

ShapeId SceneIR::AddCollisionShape(CollisionShapeRecord record) {
    const auto id = static_cast<ShapeId>(shapes_.size());
    record.id = id;
    shapes_.push_back(std::move(record));
    return id;
}

JointId SceneIR::AddJoint(std::string name, BodyId parent, BodyId child) {
    JointRecord rec;
    rec.name = std::move(name);
    rec.parent_body = parent;
    rec.child_body = child;
    return AddJoint(std::move(rec));
}

JointId SceneIR::AddJoint(JointRecord record) {
    const auto id = static_cast<JointId>(joints_.size());
    record.id = id;
    joints_.push_back(std::move(record));
    return id;
}

SensorId SceneIR::AddSensor(std::string name, BodyId body) {
    SensorRecord rec;
    rec.name = std::move(name);
    rec.attached_body = body;
    return AddSensor(std::move(rec));
}

SensorId SceneIR::AddSensor(SensorRecord record) {
    const auto id = static_cast<SensorId>(sensors_.size());
    record.id = id;
    sensors_.push_back(std::move(record));
    return id;
}

MaterialId SceneIR::AddMaterial(MaterialRecord record) {
    const auto id = static_cast<MaterialId>(materials_.size());
    record.id = id;
    materials_.push_back(std::move(record));
    return id;
}

CameraId SceneIR::AddCamera(CameraRecord record) {
    const auto id = static_cast<CameraId>(cameras_.size());
    record.id = id;
    cameras_.push_back(std::move(record));
    return id;
}

LightId SceneIR::AddLight(LightRecord record) {
    const auto id = static_cast<LightId>(lights_.size());
    record.id = id;
    lights_.push_back(std::move(record));
    return id;
}

ActuatorId SceneIR::AddActuator(ActuatorRecord record) {
    const auto id = static_cast<ActuatorId>(actuators_.size());
    record.id = id;
    actuators_.push_back(std::move(record));
    return id;
}

size_t SceneIR::RigidBodyCount() const { return bodies_.size(); }
size_t SceneIR::JointCount() const { return joints_.size(); }
size_t SceneIR::ShapeCount() const { return shapes_.size(); }
size_t SceneIR::SensorCount() const { return sensors_.size(); }
size_t SceneIR::MaterialCount() const { return materials_.size(); }
size_t SceneIR::CameraCount() const { return cameras_.size(); }
size_t SceneIR::LightCount() const { return lights_.size(); }
size_t SceneIR::ActuatorCount() const { return actuators_.size(); }

const RigidBodyRecord& SceneIR::GetBody(BodyId id) const {
    if (id >= bodies_.size()) {
        throw std::out_of_range("SceneIR::GetBody - invalid BodyId");
    }
    return bodies_[id];
}

const JointRecord& SceneIR::GetJoint(JointId id) const {
    if (id >= joints_.size()) {
        throw std::out_of_range("SceneIR::GetJoint - invalid JointId");
    }
    return joints_[id];
}

const CollisionShapeRecord& SceneIR::GetShape(ShapeId id) const {
    if (id >= shapes_.size()) {
        throw std::out_of_range("SceneIR::GetShape - invalid ShapeId");
    }
    return shapes_[id];
}

const SensorRecord& SceneIR::GetSensor(SensorId id) const {
    if (id >= sensors_.size()) {
        throw std::out_of_range("SceneIR::GetSensor - invalid SensorId");
    }
    return sensors_[id];
}

const MaterialRecord& SceneIR::GetMaterial(MaterialId id) const {
    if (id >= materials_.size()) {
        throw std::out_of_range("SceneIR::GetMaterial - invalid MaterialId");
    }
    return materials_[id];
}

const CameraRecord& SceneIR::GetCamera(CameraId id) const {
    if (id >= cameras_.size()) {
        throw std::out_of_range("SceneIR::GetCamera - invalid CameraId");
    }
    return cameras_[id];
}

const LightRecord& SceneIR::GetLight(LightId id) const {
    if (id >= lights_.size()) {
        throw std::out_of_range("SceneIR::GetLight - invalid LightId");
    }
    return lights_[id];
}

const ActuatorRecord& SceneIR::GetActuator(ActuatorId id) const {
    if (id >= actuators_.size()) {
        throw std::out_of_range("SceneIR::GetActuator - invalid ActuatorId");
    }
    return actuators_[id];
}

RigidBodyRecord& SceneIR::GetBodyMut(BodyId id) {
    if (id >= bodies_.size()) {
        throw std::out_of_range("SceneIR::GetBodyMut - invalid BodyId");
    }
    return bodies_[id];
}

JointRecord& SceneIR::GetJointMut(JointId id) {
    if (id >= joints_.size()) {
        throw std::out_of_range("SceneIR::GetJointMut - invalid JointId");
    }
    return joints_[id];
}

CollisionShapeRecord& SceneIR::GetShapeMut(ShapeId id) {
    if (id >= shapes_.size()) {
        throw std::out_of_range("SceneIR::GetShapeMut - invalid ShapeId");
    }
    return shapes_[id];
}

SensorRecord& SceneIR::GetSensorMut(SensorId id) {
    if (id >= sensors_.size()) {
        throw std::out_of_range("SceneIR::GetSensorMut - invalid SensorId");
    }
    return sensors_[id];
}

MaterialRecord& SceneIR::GetMaterialMut(MaterialId id) {
    if (id >= materials_.size()) {
        throw std::out_of_range("SceneIR::GetMaterialMut - invalid MaterialId");
    }
    return materials_[id];
}

CameraRecord& SceneIR::GetCameraMut(CameraId id) {
    if (id >= cameras_.size()) {
        throw std::out_of_range("SceneIR::GetCameraMut - invalid CameraId");
    }
    return cameras_[id];
}

LightRecord& SceneIR::GetLightMut(LightId id) {
    if (id >= lights_.size()) {
        throw std::out_of_range("SceneIR::GetLightMut - invalid LightId");
    }
    return lights_[id];
}

ActuatorRecord& SceneIR::GetActuatorMut(ActuatorId id) {
    if (id >= actuators_.size()) {
        throw std::out_of_range("SceneIR::GetActuatorMut - invalid ActuatorId");
    }
    return actuators_[id];
}

const std::vector<RigidBodyRecord>& SceneIR::Bodies() const { return bodies_; }
const std::vector<JointRecord>& SceneIR::Joints() const { return joints_; }
const std::vector<CollisionShapeRecord>& SceneIR::Shapes() const { return shapes_; }
const std::vector<SensorRecord>& SceneIR::Sensors() const { return sensors_; }
const std::vector<MaterialRecord>& SceneIR::Materials() const { return materials_; }
const std::vector<CameraRecord>& SceneIR::Cameras() const { return cameras_; }
const std::vector<LightRecord>& SceneIR::Lights() const { return lights_; }
const std::vector<ActuatorRecord>& SceneIR::Actuators() const { return actuators_; }

} // namespace nuka::scene
