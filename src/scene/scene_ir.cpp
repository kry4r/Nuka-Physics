// ---------------------------------------------------------------------------
// nuka::scene::SceneIR implementation
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

#include <cstddef>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nuka::scene {

// ---------------------------------------------------------------------------
// lifecycle / copy semantics
// ---------------------------------------------------------------------------
//
// Copying SceneIR must produce a fully INDEPENDENT structural world: a plain
// shared_ptr copy of tree_ would alias the same SceneNode objects across both
// instances, and the copy's ecs_ backrefs (BindNode weak_ptrs) would point into
// the source's tree. Rebuilding tree_ + ecs_ from the (already deep-copied)
// record vectors via the same write-through path Add* uses is the simplest
// correct deep copy -- it is deterministic (records carry dense ids in append
// order) and reuses one code path, so the copy is structurally identical to a
// fresh import of the same records. This is what gives Compose() its purity.

SceneIR::SceneIR() = default;

SceneIR::SceneIR(const SceneIR& other)
    : bodies_(other.bodies_),
      joints_(other.joints_),
      shapes_(other.shapes_),
      sensors_(other.sensors_),
      materials_(other.materials_),
      cameras_(other.cameras_),
      lights_(other.lights_),
      actuators_(other.actuators_),
      media_(other.media_),
      scripts_(other.scripts_),
      exclude_pairs_(other.exclude_pairs_),
      contact_pairs_(other.contact_pairs_),
      initial_state_(other.initial_state_),
      settle_(other.settle_),
      terrain_(other.terrain_) {
    // initial_state_ / settle_ / terrain_ are authored metadata, NOT projected
    // from records, so they must be copied explicitly (RebuildFacade only
    // rebuilds the tree/ECS from records and would otherwise drop them).
    RebuildFacade();
}

SceneIR& SceneIR::operator=(const SceneIR& other) {
    if (this == &other) {
        return *this;
    }
    bodies_        = other.bodies_;
    joints_        = other.joints_;
    shapes_        = other.shapes_;
    sensors_       = other.sensors_;
    materials_     = other.materials_;
    cameras_       = other.cameras_;
    lights_        = other.lights_;
    actuators_     = other.actuators_;
    media_         = other.media_;
    scripts_       = other.scripts_;
    exclude_pairs_ = other.exclude_pairs_;
    contact_pairs_ = other.contact_pairs_;
    initial_state_ = other.initial_state_;   // authored metadata (not from records)
    settle_        = other.settle_;
    terrain_       = other.terrain_;
    RebuildFacade();
    return *this;
}

// Replay the record vectors through the projection helpers to rebuild a fresh
// tree_ + ecs_. The records are projected in the SAME order the importers /
// compose append them in (materials, then bodies, shapes, joints, then the
// referencing records), which is the order needed so a body's parent node and a
// shape/joint's body node already exist when projected. The record-id <-> entity
// maps and the BodyId -> node map are rebuilt alongside.
void SceneIR::RebuildFacade() {
    facade_dirty_ = false;  // the rebuilt facade reflects the records
    tree_ = SceneGraph();
    ecs_  = Registry();
    body_entity_.clear();
    shape_entity_.clear();
    joint_entity_.clear();
    media_entity_.clear();
    script_entity_.clear();
    body_node_.clear();
    material_ids_.clear();

    for (const MaterialRecord& rec : materials_) {
        ProjectMaterial(rec);
    }
    for (const RigidBodyRecord& rec : bodies_) {
        ProjectBody(rec);
    }
    for (const CollisionShapeRecord& rec : shapes_) {
        ProjectShape(rec);
    }
    for (const JointRecord& rec : joints_) {
        ProjectJoint(rec);
    }
    for (const CameraRecord& rec : cameras_) {
        ProjectCamera(rec);
    }
    for (const LightRecord& rec : lights_) {
        ProjectLight(rec);
    }
    for (const ActuatorRecord& rec : actuators_) {
        ProjectActuator(rec);
    }
    // Media hangs free under root (no body dependency); projected after bodies so
    // a copied/rebuilt scene keeps its media entities.
    for (const MediaRecord& rec : media_) {
        ProjectMedia(rec);
    }
    // Scripts project last: parent-path resolution may reference any earlier node.
    for (const ScriptRecord& rec : scripts_) {
        ProjectScript(rec);
    }
}

// ---------------------------------------------------------------------------
// Add* mutators (record store + facade write-through)
// ---------------------------------------------------------------------------

BodyId SceneIR::AddRigidBody(std::string name) {
    RigidBodyRecord rec;
    rec.name = std::move(name);
    return AddRigidBody(std::move(rec));
}

BodyId SceneIR::AddRigidBody(RigidBodyRecord record) {
    const auto id = static_cast<BodyId>(bodies_.size());
    record.id = id;
    EnsureFacade();  // incremental projection needs a current facade
    bodies_.push_back(std::move(record));
    ProjectBody(bodies_.back());
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
    EnsureFacade();  // incremental projection needs a current facade
    shapes_.push_back(std::move(record));
    ProjectShape(shapes_.back());
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
    EnsureFacade();  // incremental projection needs a current facade
    joints_.push_back(std::move(record));
    ProjectJoint(joints_.back());
    return id;
}

SensorId SceneIR::AddSensor(std::string name, BodyId body) {
    SensorDesc rec;
    rec.name = std::move(name);
    rec.mount = MountFrame::Body;
    rec.mount_index = body;
    return AddSensor(std::move(rec));
}

SensorId SceneIR::AddSensor(SensorDesc record) {
    // Sensors stay record-only (no component projection); the mount is resolved
    // by the runtime FK world-pose pass, not the scene-tree facade.
    const auto id = static_cast<SensorId>(sensors_.size());
    record.id = id;
    sensors_.push_back(std::move(record));
    return id;
}

MaterialId SceneIR::AddMaterial(MaterialRecord record) {
    const auto id = static_cast<MaterialId>(materials_.size());
    record.id = id;
    EnsureFacade();  // incremental projection needs a current facade
    materials_.push_back(std::move(record));
    ProjectMaterial(materials_.back());
    return id;
}

CameraId SceneIR::AddCamera(CameraRecord record) {
    const auto id = static_cast<CameraId>(cameras_.size());
    record.id = id;
    EnsureFacade();  // incremental projection needs a current facade
    cameras_.push_back(std::move(record));
    ProjectCamera(cameras_.back());
    return id;
}

LightId SceneIR::AddLight(LightRecord record) {
    const auto id = static_cast<LightId>(lights_.size());
    record.id = id;
    EnsureFacade();  // incremental projection needs a current facade
    lights_.push_back(std::move(record));
    ProjectLight(lights_.back());
    return id;
}

ActuatorId SceneIR::AddActuator(ActuatorRecord record) {
    const auto id = static_cast<ActuatorId>(actuators_.size());
    record.id = id;
    EnsureFacade();  // incremental projection needs a current facade
    actuators_.push_back(std::move(record));
    ProjectActuator(actuators_.back());
    return id;
}

MediaId SceneIR::AddMedia(MediaRecord record) {
    const auto id = static_cast<MediaId>(media_.size());
    record.id = id;
    EnsureFacade();  // incremental projection needs a current facade
    media_.push_back(std::move(record));
    ProjectMedia(media_.back());
    return id;
}

TerrainId SceneIR::AddTerrain(TerrainRecord record) {
    const auto id = static_cast<TerrainId>(terrain_.size());
    terrain_.push_back(std::move(record));  // authored metadata: no facade projection
    return id;
}

ScriptId SceneIR::AddScript(ScriptRecord record) {
    const auto id = static_cast<ScriptId>(scripts_.size());
    record.id = id;
    EnsureFacade();  // incremental projection needs a current facade
    scripts_.push_back(std::move(record));
    ProjectScript(scripts_.back());
    return id;
}

void SceneIR::AddExcludePair(BodyId a, BodyId b) {
    // Canonicalize as (min,max) so (a,b) and (b,a) store identically. No dedup
    // here — that (and the filter policy) is C1c.
    if (b < a) {
        std::swap(a, b);
    }
    exclude_pairs_.emplace_back(a, b);
}

void SceneIR::AddContactPair(ContactPairOverride pair) {
    // Stored verbatim in authoring order; no dedup / merge here (that is C1c).
    contact_pairs_.push_back(pair);
}

bool SceneIR::RemoveBodySubtree(BodyId root) {
    if (root >= bodies_.size()) {
        return false;
    }

    // Deleted set: the root + every transitive parent_id descendant.
    std::vector<bool> del(bodies_.size(), false);
    del[root] = true;
    for (bool more = true; more;) {
        more = false;
        for (size_t i = 0; i < bodies_.size(); ++i) {
            const BodyId p = bodies_[i].parent_id;
            if (!del[i] && p != kInvalidBody && p < del.size() && del[p]) {
                del[i] = true;
                more = true;
            }
        }
    }

    // Deleted body names, captured before compaction (the IC-key match below).
    std::vector<std::string> deleted_names;
    for (size_t i = 0; i < bodies_.size(); ++i) {
        if (del[i]) deleted_names.push_back(bodies_[i].name);
    }

    // Compact bodies; old->new id map (kInvalidBody for deleted).
    std::vector<BodyId> body_map(bodies_.size(), kInvalidBody);
    {
        std::vector<RigidBodyRecord> kept;
        kept.reserve(bodies_.size());
        for (size_t i = 0; i < bodies_.size(); ++i) {
            if (del[i]) continue;
            body_map[i] = static_cast<BodyId>(kept.size());
            kept.push_back(std::move(bodies_[i]));
        }
        for (size_t i = 0; i < kept.size(); ++i) {
            kept[i].id = static_cast<BodyId>(i);
            if (kept[i].parent_id != kInvalidBody) {
                kept[i].parent_id = body_map[kept[i].parent_id];
            }
        }
        bodies_ = std::move(kept);
    }
    const auto body_deleted = [&](BodyId b) {
        return b != kInvalidBody && (b >= body_map.size() || body_map[b] == kInvalidBody);
    };
    const auto remap_body = [&](BodyId b) {
        return (b == kInvalidBody) ? kInvalidBody : body_map[b];
    };

    // Compact shapes (drop those on deleted bodies); old->new shape id map.
    std::vector<ShapeId> shape_map(shapes_.size(), kInvalidShape);
    {
        std::vector<CollisionShapeRecord> kept;
        kept.reserve(shapes_.size());
        for (size_t i = 0; i < shapes_.size(); ++i) {
            if (body_deleted(shapes_[i].body_id)) continue;
            shape_map[i] = static_cast<ShapeId>(kept.size());
            kept.push_back(std::move(shapes_[i]));
        }
        for (size_t i = 0; i < kept.size(); ++i) {
            kept[i].id = static_cast<ShapeId>(i);
            kept[i].body_id = remap_body(kept[i].body_id);
        }
        shapes_ = std::move(kept);
    }

    // Compact joints (drop those touching a deleted body); old->new joint id map.
    std::vector<JointId> joint_map(joints_.size(), kInvalidJoint);
    {
        std::vector<JointRecord> kept;
        kept.reserve(joints_.size());
        for (size_t i = 0; i < joints_.size(); ++i) {
            if (body_deleted(joints_[i].parent_body) ||
                body_deleted(joints_[i].child_body)) {
                continue;
            }
            joint_map[i] = static_cast<JointId>(kept.size());
            kept.push_back(std::move(joints_[i]));
        }
        for (size_t i = 0; i < kept.size(); ++i) {
            kept[i].id = static_cast<JointId>(i);
            kept[i].parent_body = remap_body(kept[i].parent_body);
            kept[i].child_body = remap_body(kept[i].child_body);
        }
        joints_ = std::move(kept);
    }

    // Actuators on a removed joint drop; survivors remap the joint reference.
    {
        std::vector<ActuatorRecord> kept;
        kept.reserve(actuators_.size());
        for (auto& a : actuators_) {
            if (a.joint_id != kInvalidJoint &&
                (a.joint_id >= joint_map.size() || joint_map[a.joint_id] == kInvalidJoint)) {
                continue;
            }
            if (a.joint_id != kInvalidJoint) a.joint_id = joint_map[a.joint_id];
            a.id = static_cast<ActuatorId>(kept.size());
            kept.push_back(std::move(a));
        }
        actuators_ = std::move(kept);
    }

    // Body-mounted sensors on a deleted body drop; survivors remap. Link/Base
    // mounts are articulation-indexed (a cook product) and pass through.
    {
        std::vector<SensorDesc> kept;
        kept.reserve(sensors_.size());
        for (auto& s : sensors_) {
            if (s.mount == MountFrame::Body && body_deleted(s.mount_index)) continue;
            if (s.mount == MountFrame::Body && s.mount_index != kInvalidBody) {
                s.mount_index = body_map[s.mount_index];
            }
            s.id = static_cast<SensorId>(kept.size());
            kept.push_back(std::move(s));
        }
        sensors_ = std::move(kept);
    }

    // A camera / light attached to a deleted body belongs to the subtree: drop.
    {
        std::vector<CameraRecord> kept;
        for (auto& c : cameras_) {
            if (body_deleted(c.attached_body)) continue;
            c.attached_body = remap_body(c.attached_body);
            c.id = static_cast<CameraId>(kept.size());
            kept.push_back(std::move(c));
        }
        cameras_ = std::move(kept);
    }
    {
        std::vector<LightRecord> kept;
        for (auto& l : lights_) {
            if (body_deleted(l.attached_body)) continue;
            l.attached_body = remap_body(l.attached_body);
            l.id = static_cast<LightId>(kept.size());
            kept.push_back(std::move(l));
        }
        lights_ = std::move(kept);
    }

    // Exclude pairs touching a deleted body drop; survivors remap+re-canonicalize.
    {
        std::vector<std::pair<BodyId, BodyId>> kept;
        for (const auto& p : exclude_pairs_) {
            if (body_deleted(p.first) || body_deleted(p.second)) continue;
            BodyId a = remap_body(p.first), b = remap_body(p.second);
            if (b < a) std::swap(a, b);
            kept.emplace_back(a, b);
        }
        exclude_pairs_ = std::move(kept);
    }

    // Contact-pair overrides touching a deleted shape drop; survivors remap.
    {
        const auto shape_deleted = [&](ShapeId s) {
            return s != kInvalidShape &&
                   (s >= shape_map.size() || shape_map[s] == kInvalidShape);
        };
        std::vector<ContactPairOverride> kept;
        for (auto p : contact_pairs_) {
            if (shape_deleted(p.geom1) || shape_deleted(p.geom2)) continue;
            if (p.geom1 != kInvalidShape) p.geom1 = shape_map[p.geom1];
            if (p.geom2 != kInvalidShape) p.geom2 = shape_map[p.geom2];
            kept.push_back(p);
        }
        contact_pairs_ = std::move(kept);
    }

    // Drop the settled IC of any articulation the removal touched (its key is a
    // path-prefix of a deleted body's record name); untouched entries persist.
    for (auto it = initial_state_.begin(); it != initial_state_.end();) {
        bool touched = false;
        for (const std::string& n : deleted_names) {
            if (n == it->first ||
                (n.size() > it->first.size() &&
                 n.compare(0, it->first.size(), it->first) == 0 &&
                 n[it->first.size()] == '/')) {
                touched = true;
                break;
            }
        }
        it = touched ? initial_state_.erase(it) : std::next(it);
    }

    // Materials / media / terrain / settle / scripts are body-independent and
    // persist verbatim (a script's dangling parent_path re-projects at root).
    RebuildFacade();
    return true;
}

bool SceneIR::RemoveScript(ScriptId id) {
    if (id >= scripts_.size()) {
        return false;
    }
    scripts_.erase(scripts_.begin() + static_cast<std::ptrdiff_t>(id));
    for (size_t i = 0; i < scripts_.size(); ++i) {
        scripts_[i].id = static_cast<ScriptId>(i);
    }
    RebuildFacade();
    return true;
}

size_t SceneIR::RigidBodyCount() const { return bodies_.size(); }
size_t SceneIR::JointCount() const { return joints_.size(); }
size_t SceneIR::ShapeCount() const { return shapes_.size(); }
size_t SceneIR::SensorCount() const { return sensors_.size(); }
size_t SceneIR::MaterialCount() const { return materials_.size(); }
size_t SceneIR::CameraCount() const { return cameras_.size(); }
size_t SceneIR::LightCount() const { return lights_.size(); }
size_t SceneIR::ActuatorCount() const { return actuators_.size(); }
size_t SceneIR::MediaCount() const { return media_.size(); }
size_t SceneIR::TerrainCount() const { return terrain_.size(); }
size_t SceneIR::ScriptCount() const { return scripts_.size(); }

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

const SensorDesc& SceneIR::GetSensor(SensorId id) const {
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

const MediaRecord& SceneIR::GetMedia(MediaId id) const {
    if (id >= media_.size()) {
        throw std::out_of_range("SceneIR::GetMedia - invalid MediaId");
    }
    return media_[id];
}

const TerrainRecord& SceneIR::GetTerrain(TerrainId id) const {
    if (id >= terrain_.size()) {
        throw std::out_of_range("SceneIR::GetTerrain - invalid TerrainId");
    }
    return terrain_[id];
}

const ScriptRecord& SceneIR::GetScript(ScriptId id) const {
    if (id >= scripts_.size()) {
        throw std::out_of_range("SceneIR::GetScript - invalid ScriptId");
    }
    return scripts_[id];
}

RigidBodyRecord& SceneIR::GetBodyMut(BodyId id) {
    if (id >= bodies_.size()) {
        throw std::out_of_range("SceneIR::GetBodyMut - invalid BodyId");
    }
    facade_dirty_ = true;  // record mutation bypasses write-through
    return bodies_[id];
}

JointRecord& SceneIR::GetJointMut(JointId id) {
    if (id >= joints_.size()) {
        throw std::out_of_range("SceneIR::GetJointMut - invalid JointId");
    }
    facade_dirty_ = true;  // record mutation bypasses write-through
    return joints_[id];
}

CollisionShapeRecord& SceneIR::GetShapeMut(ShapeId id) {
    if (id >= shapes_.size()) {
        throw std::out_of_range("SceneIR::GetShapeMut - invalid ShapeId");
    }
    facade_dirty_ = true;  // record mutation bypasses write-through
    return shapes_[id];
}

SensorDesc& SceneIR::GetSensorMut(SensorId id) {
    if (id >= sensors_.size()) {
        throw std::out_of_range("SceneIR::GetSensorMut - invalid SensorId");
    }
    facade_dirty_ = true;  // record mutation bypasses write-through
    return sensors_[id];
}

MaterialRecord& SceneIR::GetMaterialMut(MaterialId id) {
    if (id >= materials_.size()) {
        throw std::out_of_range("SceneIR::GetMaterialMut - invalid MaterialId");
    }
    facade_dirty_ = true;  // record mutation bypasses write-through
    return materials_[id];
}

CameraRecord& SceneIR::GetCameraMut(CameraId id) {
    if (id >= cameras_.size()) {
        throw std::out_of_range("SceneIR::GetCameraMut - invalid CameraId");
    }
    facade_dirty_ = true;  // record mutation bypasses write-through
    return cameras_[id];
}

LightRecord& SceneIR::GetLightMut(LightId id) {
    if (id >= lights_.size()) {
        throw std::out_of_range("SceneIR::GetLightMut - invalid LightId");
    }
    facade_dirty_ = true;  // record mutation bypasses write-through
    return lights_[id];
}

ActuatorRecord& SceneIR::GetActuatorMut(ActuatorId id) {
    if (id >= actuators_.size()) {
        throw std::out_of_range("SceneIR::GetActuatorMut - invalid ActuatorId");
    }
    facade_dirty_ = true;  // record mutation bypasses write-through
    return actuators_[id];
}

MediaRecord& SceneIR::GetMediaMut(MediaId id) {
    if (id >= media_.size()) {
        throw std::out_of_range("SceneIR::GetMediaMut - invalid MediaId");
    }
    facade_dirty_ = true;  // record mutation bypasses write-through
    return media_[id];
}

TerrainRecord& SceneIR::GetTerrainMut(TerrainId id) {
    if (id >= terrain_.size()) {
        throw std::out_of_range("SceneIR::GetTerrainMut - invalid TerrainId");
    }
    return terrain_[id];  // authored metadata: no facade dependency
}

ScriptRecord& SceneIR::GetScriptMut(ScriptId id) {
    if (id >= scripts_.size()) {
        throw std::out_of_range("SceneIR::GetScriptMut - invalid ScriptId");
    }
    facade_dirty_ = true;  // record mutation bypasses write-through
    return scripts_[id];
}

const std::vector<RigidBodyRecord>& SceneIR::Bodies() const { return bodies_; }
const std::vector<JointRecord>& SceneIR::Joints() const { return joints_; }
const std::vector<CollisionShapeRecord>& SceneIR::Shapes() const { return shapes_; }
const std::vector<SensorDesc>& SceneIR::Sensors() const { return sensors_; }
const std::vector<MaterialRecord>& SceneIR::Materials() const { return materials_; }
const std::vector<CameraRecord>& SceneIR::Cameras() const { return cameras_; }
const std::vector<LightRecord>& SceneIR::Lights() const { return lights_; }
const std::vector<ActuatorRecord>& SceneIR::Actuators() const { return actuators_; }
const std::vector<MediaRecord>& SceneIR::Media() const { return media_; }
const std::vector<TerrainRecord>& SceneIR::Terrain() const { return terrain_; }
const std::vector<ScriptRecord>& SceneIR::Scripts() const { return scripts_; }
const std::vector<std::pair<BodyId, BodyId>>& SceneIR::ExcludePairs() const {
    return exclude_pairs_;
}
const std::vector<ContactPairOverride>& SceneIR::ContactPairs() const {
    return contact_pairs_;
}

// ---------------------------------------------------------------------------
// reverse maps (record-id -> projected entity)
// ---------------------------------------------------------------------------

EntityId SceneIR::EntityOfBody(BodyId id) const {
    EnsureFacade();
    return id < body_entity_.size() ? body_entity_[id] : kInvalidEntity;
}
EntityId SceneIR::EntityOfShape(ShapeId id) const {
    EnsureFacade();
    return id < shape_entity_.size() ? shape_entity_[id] : kInvalidEntity;
}
EntityId SceneIR::EntityOfJoint(JointId id) const {
    EnsureFacade();
    return id < joint_entity_.size() ? joint_entity_[id] : kInvalidEntity;
}
EntityId SceneIR::EntityOfMedia(MediaId id) const {
    EnsureFacade();
    return id < media_entity_.size() ? media_entity_[id] : kInvalidEntity;
}
EntityId SceneIR::EntityOfScript(ScriptId id) const {
    EnsureFacade();
    return id < script_entity_.size() ? script_entity_[id] : kInvalidEntity;
}

// Lazy facade resync. Get*Mut hands out mutable record references that bypass
// the Add* write-through; it marks the facade dirty and the next facade read
// lands here. Re-projection is a pure function of the records, so this is
// exactly the copy-constructor rebuild, done in place. (const_cast: the facade
// is a derived cache of the records; rebuilding it does not mutate the
// logical, record-level state of the scene.)
void SceneIR::EnsureFacade() const {
    if (!facade_dirty_) {
        return;
    }
    SceneIR* self = const_cast<SceneIR*>(this);
    self->facade_dirty_ = false;
    self->RebuildFacade();
}

// ---------------------------------------------------------------------------
// write-through projection helpers (records -> tree + ECS)
// ---------------------------------------------------------------------------

namespace {

// Map a record ShapeType to the closed component Kind set. TriMesh / HeightField
// have no exact component Kind yet; a TriMesh projects to a ConvexHull collision
// shape (its cooked hull lands in .nka in M2c) and HeightField falls back to a
// Plane. This is the per-shape projection only; the record keeps the exact type.
CollisionShapeComponent::Kind KindFromShapeType(ShapeType type) {
    switch (type) {
        case ShapeType::Sphere:      return CollisionShapeComponent::Kind::Sphere;
        case ShapeType::Capsule:     return CollisionShapeComponent::Kind::Capsule;
        case ShapeType::Box:         return CollisionShapeComponent::Kind::Box;
        case ShapeType::Plane:       return CollisionShapeComponent::Kind::Plane;
        case ShapeType::ConvexHull:  return CollisionShapeComponent::Kind::ConvexHull;
        case ShapeType::TriMesh:     return CollisionShapeComponent::Kind::ConvexHull;
        case ShapeType::HeightField: return CollisionShapeComponent::Kind::Plane;
    }
    return CollisionShapeComponent::Kind::Box;
}

// Pack a shape's primitive sizing into the component's params[4]. The layout is
// per-kind: Sphere = (radius); Capsule = (radius, half_height); Box / Plane =
// half-extents (x,y,z). ConvexHull / SdfMesh carry no params (geometry lives in
// the record / the cooked .nka).
void FillShapeParams(const CollisionShapeRecord& rec, float (&params)[4]) {
    params[0] = params[1] = params[2] = params[3] = 0.0f;
    switch (rec.type) {
        case ShapeType::Sphere:
            params[0] = rec.radius;
            break;
        case ShapeType::Capsule:
            params[0] = rec.radius;
            params[1] = rec.half_height;
            break;
        case ShapeType::Box:
        case ShapeType::Plane:
            params[0] = rec.half_extents.x;
            params[1] = rec.half_extents.y;
            params[2] = rec.half_extents.z;
            break;
        default:
            break;  // mesh kinds: no inline params
    }
}

JointComponent::Kind KindFromJointType(JointType type) {
    switch (type) {
        case JointType::Revolute:  return JointComponent::Revolute;
        case JointType::Prismatic: return JointComponent::Prismatic;
        case JointType::Fixed:     return JointComponent::Fixed;
        case JointType::Spherical: return JointComponent::Spherical;
        case JointType::Free:      return JointComponent::Free;
    }
    return JointComponent::Fixed;
}

ActuatorComponent::Mode ModeFromActuatorType(ActuatorType type) {
    switch (type) {
        case ActuatorType::Position: return ActuatorComponent::PD;
        case ActuatorType::Velocity: return ActuatorComponent::Velocity;
        case ActuatorType::Motor:
        case ActuatorType::Force:    return ActuatorComponent::Torque;
    }
    return ActuatorComponent::Torque;
}

LightComponent::Type LightCompTypeFromLightType(LightType type) {
    switch (type) {
        case LightType::Point:       return LightComponent::Type::Point;
        case LightType::Directional: return LightComponent::Type::Directional;
        case LightType::Spot:        return LightComponent::Type::Spot;
        case LightType::Area:        return LightComponent::Type::Area;
    }
    return LightComponent::Type::Point;
}

SystemKindComponent::K SystemKindFromMediaKind(MediaRecord::Kind kind) {
    switch (kind) {
        case MediaRecord::Kind::Cloth:   return SystemKindComponent::Cloth;
        case MediaRecord::Kind::SoftTet: return SystemKindComponent::Soft;
        case MediaRecord::Kind::Fluid:   return SystemKindComponent::Fluid;
        case MediaRecord::Kind::Granular: return SystemKindComponent::Soft;
    }
    return SystemKindComponent::Soft;
}

SoftBodyComponent::SimMethod SimMethodFromMediaMethod(MediaRecord::Method method) {
    return method == MediaRecord::Method::MlsMpm ? SoftBodyComponent::SimMethod::MlsMpm
                                                 : SoftBodyComponent::SimMethod::Xpbd;
}

// Stable node name for an UNNAMED record: derive from the RECORD identity
// (kind + dense record id), NOT from the SceneGraph's global node-id counter.
// The graph counter walks differently when the same records are replayed in a
// different projection order (interleaved import vs grouped rebuild/Load), so
// counter-derived autonames are not stable scene properties; record-derived
// ones are (record ids round-trip positionally through .nks). The roundtrip
// gate asserts EXACT name equality on the strength of this.
std::string StableAutoName(const char* kind, uint32_t record_id,
                           const std::string& record_name) {
    if (!record_name.empty()) {
        return record_name;
    }
    return std::string(kind) + "_" + std::to_string(record_id);
}

}  // namespace

void SceneIR::ProjectMaterial(const MaterialRecord& rec) {
    // Split one authored MaterialRecord into BOTH asset tables under the same
    // name. The MaterialId -> {phys,render} mapping lets shape projection bind a
    // shape's material to the right asset ids.
    RenderMaterial rm;
    rm.base_color[0] = rec.base_color.x;
    rm.base_color[1] = rec.base_color.y;
    rm.base_color[2] = rec.base_color.z;
    rm.base_color[3] = rec.alpha;
    rm.opacity   = rec.alpha;
    rm.roughness = rec.roughness;
    rm.metallic  = rec.metallic;
    rm.emissive[0] = rec.emissive.x;
    rm.emissive[1] = rec.emissive.y;
    rm.emissive[2] = rec.emissive.z;
    rm.sheen        = rec.sheen;
    rm.transmission = rec.transmission;
    rm.ior          = rec.ior;
    rm.absorption[0] = rec.absorption.x;
    rm.absorption[1] = rec.absorption.y;
    rm.absorption[2] = rec.absorption.z;

    PhysicsMaterial pm;
    if (rec.friction_mu >= 0.0f) {
        pm.static_friction  = rec.friction_mu;
        pm.dynamic_friction = rec.friction_mu;
    }

    const uint32_t render_id = ecs_.AddRenderMaterial(rec.name, rm);
    const uint32_t phys_id   = ecs_.AddPhysicsMaterial(rec.name, pm);

    if (rec.id >= material_ids_.size()) {
        material_ids_.resize(rec.id + 1);
    }
    material_ids_[rec.id] = MaterialIds{phys_id, render_id};
}

void SceneIR::ProjectBody(const RigidBodyRecord& rec) {
    if (rec.id >= body_entity_.size()) {
        body_entity_.resize(rec.id + 1, kInvalidEntity);
        body_node_.resize(rec.id + 1);
    }

    // The record name is interpreted as a PATH ("h1/pelvis"): intermediate
    // segments become reusable plain group nodes (each its own entity carrying
    // only a NameComponent), and the final segment is the body node. The body's
    // PARENT NODE is its parent body's node when parent_id is valid (kinematic
    // hierarchy native in the tree); otherwise the path-prefix group chain hangs
    // at root level.
    std::shared_ptr<SceneNode> parent_node;
    if (rec.parent_id != kInvalidBody && rec.parent_id < body_node_.size()) {
        parent_node = body_node_[rec.parent_id];
    }
    if (!parent_node) {
        parent_node = tree_.Root();
    }

    // Split the name on '/'. All but the last segment are group nodes.
    std::vector<std::string> segments;
    {
        std::string segment;
        std::istringstream ss(rec.name);
        while (std::getline(ss, segment, '/')) {
            if (!segment.empty()) {
                segments.push_back(segment);
            }
        }
        if (segments.empty()) {
            // Empty/'/'-only name: stable record-derived autoname (see
            // StableAutoName).
            segments.push_back(StableAutoName("body", rec.id, rec.name));
        }
    }

    for (size_t i = 0; i + 1 < segments.size(); ++i) {
        // Reuse an existing group child with this name, else create one.
        std::shared_ptr<SceneNode> existing;
        for (auto child = parent_node->first_child; child; child = child->next_sibling) {
            if (child->name == segments[i]) {
                existing = child;
                break;
            }
        }
        if (existing) {
            parent_node = existing;
            continue;
        }
        const EntityId group = ecs_.Create();
        ecs_.Add(group, NameComponent{segments[i]});
        auto node = tree_.AddEntity(group, parent_node, segments[i]);
        ecs_.BindNode(group, node);
        parent_node = node;
    }

    const std::string& leaf = segments.back();
    const EntityId entity = ecs_.Create();
    auto node = tree_.AddEntity(entity, parent_node, leaf);
    ecs_.BindNode(entity, node);

    // NameComponent mirrors the (post-dedup) node name, not the raw path.
    ecs_.Add(entity, NameComponent{node->name});
    ecs_.Add(entity, TransformComponent{rec.local_transform, math::Vec3{1.0f, 1.0f, 1.0f}});

    RigidBodyComponent rb;
    rb.mass           = rec.mass;
    rb.inertia_diag   = rec.inertia;
    rb.inertial_frame = rec.inertial_transform;
    rb.kinematic      = rec.is_static;
    ecs_.Add(entity, std::move(rb));

    // Articulated routing is decided at cook (M3); keep Rigid for all for now.
    ecs_.Add(entity, SystemKindComponent{SystemKindComponent::Rigid});

    body_entity_[rec.id] = entity;
    body_node_[rec.id]   = node;
}

void SceneIR::ProjectShape(const CollisionShapeRecord& rec) {
    if (rec.id >= shape_entity_.size()) {
        shape_entity_.resize(rec.id + 1, kInvalidEntity);
    }

    std::shared_ptr<SceneNode> body_node;
    if (rec.body_id != kInvalidBody && rec.body_id < body_node_.size()) {
        body_node = body_node_[rec.body_id];
    }
    if (!body_node) {
        body_node = tree_.Root();  // orphan shape (no body): hang at root
    }

    const EntityId entity = ecs_.Create();
    auto node = tree_.AddEntity(entity, body_node,
                                StableAutoName("geom", rec.id, rec.name));
    ecs_.BindNode(entity, node);
    ecs_.Add(entity, NameComponent{node->name});
    ecs_.Add(entity, TransformComponent{rec.local_transform, math::Vec3{1.0f, 1.0f, 1.0f}});

    // Resolve the shape's material into the asset-table ids (if any).
    uint32_t phys_id   = Registry::kNoSlot;
    uint32_t render_id = Registry::kNoSlot;
    if (rec.material_id != kInvalidMaterial && rec.material_id < material_ids_.size()) {
        phys_id   = material_ids_[rec.material_id].phys;
        render_id = material_ids_[rec.material_id].render;
    }

    // Projection: a non-colliding geom (contype==0 && conaffinity==0) is a
    // VISUAL-only mesh (the h1 finger pattern); everything else is a collision
    // shape. M8.5 T5 (the visual-mesh cook): when the record carries a resolved
    // MESH AssetRef (set by nks Load from the visual_mesh node's "mesh" ref), bind
    // it onto VisualMeshComponent.mesh so the render consumer (render_world.cpp)
    // decodes real triangles instead of a placeholder box. Empty ref => the
    // consumer keeps the prior placeholder-box behavior (no regression).
    if (rec.contype == 0 && rec.conaffinity == 0) {
        VisualMeshComponent vis;
        vis.render_material_id = render_id;
        if (!rec.visual_mesh_ref.empty()) {
            vis.mesh = ParseAssetRef(rec.visual_mesh_ref);
        } else {
            // No MESH asset: record the PRIMITIVE so render_world can tessellate it
            // (the kitchen counters/walls/floor are box/cylinder VISUAL geoms with
            // no mesh). cylinder is imported as Capsule; hull/trimesh have no inline
            // params and stay None (their triangles come from a later .nka pass).
            switch (rec.type) {
                case ShapeType::Box:     vis.prim_kind = VisualMeshComponent::PrimKind::Box;     break;
                case ShapeType::Plane:   vis.prim_kind = VisualMeshComponent::PrimKind::Plane;   break;
                case ShapeType::Sphere:  vis.prim_kind = VisualMeshComponent::PrimKind::Sphere;  break;
                case ShapeType::Capsule: vis.prim_kind = VisualMeshComponent::PrimKind::Capsule; break;
                default:                 vis.prim_kind = VisualMeshComponent::PrimKind::None;    break;
            }
            if (vis.prim_kind != VisualMeshComponent::PrimKind::None) {
                FillShapeParams(rec, vis.prim_params);
            }
        }
        ecs_.Add(entity, std::move(vis));
    } else {
        CollisionShapeComponent cs;
        cs.kind = KindFromShapeType(rec.type);
        FillShapeParams(rec, cs.params);
        cs.physics_material_id = phys_id;
        cs.group = rec.contype;
        cs.mask  = rec.conaffinity;
        ecs_.Add(entity, std::move(cs));
    }

    shape_entity_[rec.id] = entity;
}

void SceneIR::ProjectJoint(const JointRecord& rec) {
    if (rec.id >= joint_entity_.size()) {
        joint_entity_.resize(rec.id + 1, kInvalidEntity);
    }

    // A joint lives ON THE CHILD BODY's entity (MuJoCo/URDF: the joint connects a
    // child to its parent). If the child already carries a JointComponent (a rare
    // multi-joint body), the extra joint gets its OWN auxiliary child entity+node
    // named after the joint, under the child body node, so neither joint is lost.
    EntityId child_entity = kInvalidEntity;
    std::shared_ptr<SceneNode> child_node;
    if (rec.child_body != kInvalidBody && rec.child_body < body_entity_.size()) {
        child_entity = body_entity_[rec.child_body];
        child_node   = body_node_[rec.child_body];
    }

    EntityId target = child_entity;
    if (child_entity == kInvalidEntity) {
        // No child body resolved: park the joint on its own node at root.
        target = ecs_.Create();
        auto node = tree_.AddEntity(target, tree_.Root(),
                                    StableAutoName("joint", rec.id, rec.name));
        ecs_.BindNode(target, node);
        ecs_.Add(target, NameComponent{node->name});
    } else if (ecs_.Has<JointComponent>(child_entity)) {
        // Multi-joint body: auxiliary joint entity under the child body node.
        target = ecs_.Create();
        auto node = tree_.AddEntity(target, child_node,
                                    StableAutoName("joint", rec.id, rec.name));
        ecs_.BindNode(target, node);
        ecs_.Add(target, NameComponent{node->name});
    }

    JointComponent jc;
    jc.kind = KindFromJointType(rec.type);
    if (rec.parent_body != kInvalidBody && rec.parent_body < body_entity_.size()) {
        jc.parent_body = body_entity_[rec.parent_body];
    }
    jc.child_body = child_entity;
    jc.axis       = rec.axis;
    jc.limit_lo   = rec.lower_limit;
    jc.limit_hi   = rec.upper_limit;
    jc.damping    = rec.damping;
    jc.armature   = rec.armature;
    ecs_.Add(target, std::move(jc));

    joint_entity_[rec.id] = target;
}

void SceneIR::ProjectCamera(const CameraRecord& rec) {
    // Under the attached body's node if it references one, else under root.
    std::shared_ptr<SceneNode> parent_node = tree_.Root();
    if (rec.attached_body != kInvalidBody && rec.attached_body < body_node_.size() &&
        body_node_[rec.attached_body]) {
        parent_node = body_node_[rec.attached_body];
    }
    const EntityId entity = ecs_.Create();
    auto node = tree_.AddEntity(entity, parent_node,
                                StableAutoName("camera", rec.id, rec.name));
    ecs_.BindNode(entity, node);
    ecs_.Add(entity, NameComponent{node->name});

    CameraComponent cam;
    cam.local_transform      = rec.local_transform;
    cam.vertical_fov_degrees = rec.vertical_fov_degrees;
    cam.near_clip            = rec.near_clip;
    cam.far_clip             = rec.far_clip;
    ecs_.Add(entity, std::move(cam));
}

void SceneIR::ProjectLight(const LightRecord& rec) {
    std::shared_ptr<SceneNode> parent_node = tree_.Root();
    if (rec.attached_body != kInvalidBody && rec.attached_body < body_node_.size() &&
        body_node_[rec.attached_body]) {
        parent_node = body_node_[rec.attached_body];
    }
    const EntityId entity = ecs_.Create();
    auto node = tree_.AddEntity(entity, parent_node,
                                StableAutoName("light", rec.id, rec.name));
    ecs_.BindNode(entity, node);
    ecs_.Add(entity, NameComponent{node->name});

    LightComponent light;
    light.type            = LightCompTypeFromLightType(rec.type);
    light.local_transform = rec.local_transform;
    light.color           = rec.color;
    light.intensity       = rec.intensity;
    ecs_.Add(entity, std::move(light));
}

void SceneIR::ProjectActuator(const ActuatorRecord& rec) {
    // An actuator targets a joint; place its component on the joint's projected
    // (child-body or auxiliary) entity, best-effort. Record-only if unresolved.
    if (rec.joint_id == kInvalidJoint || rec.joint_id >= joint_entity_.size()) {
        return;
    }
    const EntityId target = joint_entity_[rec.joint_id];
    if (target == kInvalidEntity) {
        return;
    }
    ActuatorComponent ac;
    ac.mode         = ModeFromActuatorType(rec.type);
    ac.stiffness    = rec.gain;
    ac.effort_limit = rec.force_limit;
    ecs_.Add(target, std::move(ac));
}

void SceneIR::ProjectMedia(const MediaRecord& rec) {
    if (rec.id >= media_entity_.size()) {
        media_entity_.resize(rec.id + 1, kInvalidEntity);
    }

    // A medium is a free scene object: its own entity + node under root carrying
    // the solver-routing kind and a per-kind soft/fluid constitutive component.
    const EntityId entity = ecs_.Create();
    auto node = tree_.AddEntity(entity, tree_.Root(),
                                StableAutoName("media", rec.id, rec.name));
    ecs_.BindNode(entity, node);
    ecs_.Add(entity, NameComponent{node->name});
    ecs_.Add(entity, SystemKindComponent{SystemKindFromMediaKind(rec.kind)});

    if (rec.kind == MediaRecord::Kind::Fluid) {
        FluidComponent fl;
        fl.particle_size = rec.fluid_box.spacing;
        fl.rho0 = (rec.method == MediaRecord::Method::MlsMpm) ? rec.mpm.density
                                                              : rec.pbf.rest_density;
        ecs_.Add(entity, std::move(fl));
    } else {
        SoftBodyComponent sb;
        sb.sim_method        = SimMethodFromMediaMethod(rec.method);
        sb.tet_or_cloth_mesh = rec.baked;
        if (rec.method == MediaRecord::Method::MlsMpm) {
            sb.young   = rec.mpm.youngs;
            sb.poisson = rec.mpm.poisson;
        } else {
            sb.xpbd_compliance = rec.xpbd.distance_alpha;
        }
        ecs_.Add(entity, std::move(sb));
    }

    media_entity_[rec.id] = entity;
}

void SceneIR::ProjectScript(const ScriptRecord& rec) {
    if (rec.id >= script_entity_.size()) {
        script_entity_.resize(rec.id + 1, kInvalidEntity);
    }
    // A /script node: its own entity + node hung under the authored parent path
    // (root when empty / unresolved), carrying the inline source + stable id.
    std::shared_ptr<SceneNode> parent_node = tree_.Root();
    if (!rec.parent_path.empty()) {
        if (auto p = tree_.NodeOf(rec.parent_path)) parent_node = p;
    }
    const EntityId entity = ecs_.Create();
    auto node = tree_.AddEntity(entity, parent_node,
                                StableAutoName("script", rec.id, rec.name));
    ecs_.BindNode(entity, node);
    ecs_.Add(entity, NameComponent{node->name});
    ScriptComponent sc;
    sc.source    = rec.source;
    sc.stable_id = rec.stable_id;
    ecs_.Add(entity, std::move(sc));
    script_entity_[rec.id] = entity;
}

} // namespace nuka::scene
