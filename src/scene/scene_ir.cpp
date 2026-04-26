// ---------------------------------------------------------------------------
// nuka::scene::SceneIR – implementation
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

#include <stdexcept>
#include <utility>

namespace nuka::scene {

// ---------------------------------------------------------------------------
// AddRigidBody
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// AddCollisionShape
// ---------------------------------------------------------------------------

ShapeId SceneIR::AddCollisionShape(BodyId body, ShapeType type) {
    CollisionShapeRecord rec;
    rec.body_id = body;
    rec.type    = type;
    return AddCollisionShape(std::move(rec));
}

ShapeId SceneIR::AddCollisionShape(CollisionShapeRecord record) {
    const auto id = static_cast<ShapeId>(shapes_.size());
    record.id = id;
    shapes_.push_back(std::move(record));
    return id;
}

// ---------------------------------------------------------------------------
// AddJoint
// ---------------------------------------------------------------------------

JointId SceneIR::AddJoint(std::string name, BodyId parent, BodyId child) {
    JointRecord rec;
    rec.name        = std::move(name);
    rec.parent_body = parent;
    rec.child_body  = child;
    return AddJoint(std::move(rec));
}

JointId SceneIR::AddJoint(JointRecord record) {
    const auto id = static_cast<JointId>(joints_.size());
    record.id = id;
    joints_.push_back(std::move(record));
    return id;
}

// ---------------------------------------------------------------------------
// AddSensor
// ---------------------------------------------------------------------------

SensorId SceneIR::AddSensor(std::string name, BodyId body) {
    SensorRecord rec;
    rec.name          = std::move(name);
    rec.attached_body = body;
    const auto id     = static_cast<SensorId>(sensors_.size());
    rec.id            = id;
    sensors_.push_back(std::move(rec));
    return id;
}

// ---------------------------------------------------------------------------
// Counts
// ---------------------------------------------------------------------------

size_t SceneIR::RigidBodyCount() const { return bodies_.size(); }
size_t SceneIR::JointCount()     const { return joints_.size(); }
size_t SceneIR::ShapeCount()     const { return shapes_.size(); }
size_t SceneIR::SensorCount()    const { return sensors_.size(); }

// ---------------------------------------------------------------------------
// Accessors (const)
// ---------------------------------------------------------------------------

const RigidBodyRecord& SceneIR::GetBody(BodyId id) const {
    if (id >= bodies_.size())
        throw std::out_of_range("SceneIR::GetBody – invalid BodyId");
    return bodies_[id];
}

const JointRecord& SceneIR::GetJoint(JointId id) const {
    if (id >= joints_.size())
        throw std::out_of_range("SceneIR::GetJoint – invalid JointId");
    return joints_[id];
}

const CollisionShapeRecord& SceneIR::GetShape(ShapeId id) const {
    if (id >= shapes_.size())
        throw std::out_of_range("SceneIR::GetShape – invalid ShapeId");
    return shapes_[id];
}

const SensorRecord& SceneIR::GetSensor(SensorId id) const {
    if (id >= sensors_.size())
        throw std::out_of_range("SceneIR::GetSensor – invalid SensorId");
    return sensors_[id];
}

// ---------------------------------------------------------------------------
// Accessors (mutable)
// ---------------------------------------------------------------------------

RigidBodyRecord& SceneIR::GetBodyMut(BodyId id) {
    if (id >= bodies_.size())
        throw std::out_of_range("SceneIR::GetBodyMut – invalid BodyId");
    return bodies_[id];
}

JointRecord& SceneIR::GetJointMut(JointId id) {
    if (id >= joints_.size())
        throw std::out_of_range("SceneIR::GetJointMut – invalid JointId");
    return joints_[id];
}

CollisionShapeRecord& SceneIR::GetShapeMut(ShapeId id) {
    if (id >= shapes_.size())
        throw std::out_of_range("SceneIR::GetShapeMut – invalid ShapeId");
    return shapes_[id];
}

// ---------------------------------------------------------------------------
// Bulk accessors
// ---------------------------------------------------------------------------

const std::vector<RigidBodyRecord>&      SceneIR::Bodies()  const { return bodies_; }
const std::vector<JointRecord>&          SceneIR::Joints()  const { return joints_; }
const std::vector<CollisionShapeRecord>& SceneIR::Shapes()  const { return shapes_; }
const std::vector<SensorRecord>&         SceneIR::Sensors() const { return sensors_; }

} // namespace nuka::scene
