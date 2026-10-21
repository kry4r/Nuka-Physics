// ---------------------------------------------------------------------------
// The editor's ONE general scene-edit seam (see editor_edits.hpp).
//
// HOST-ONLY / zero-CUDA-token.
// ---------------------------------------------------------------------------

#include "runtime/app/viewer/editor_edits.hpp"

#include "runtime/app/viewer/editor_import.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

namespace {

namespace render = nuka::render;

// The authored local transform on an entity (identity for a group / transform-less
// node so the chain walk treats it as a passthrough frame).
math::Transform LocalOf(const scene::Registry& ecs, scene::EntityId e) {
    if (const auto* t = ecs.Get<scene::TransformComponent>(e)) return t->local;
    return math::Transform::Identity();
}

}  // namespace

void EntityRecordIndex::Build(const scene::SceneIR& scene) {
    body.clear();
    shape.clear();
    media.clear();
    for (scene::BodyId i = 0; i < scene.RigidBodyCount(); ++i) {
        const scene::EntityId e = scene.EntityOfBody(i);
        if (e != scene::kInvalidEntity) body.emplace(e, i);
    }
    for (scene::ShapeId i = 0; i < scene.ShapeCount(); ++i) {
        const scene::EntityId e = scene.EntityOfShape(i);
        if (e != scene::kInvalidEntity) shape.emplace(e, i);
    }
    for (scene::MediaId i = 0; i < scene.MediaCount(); ++i) {
        const scene::EntityId e = scene.EntityOfMedia(i);
        if (e != scene::kInvalidEntity) media.emplace(e, i);
    }
}

const render::RenderInstance* InstanceOfEntity(const render::RenderWorld& world,
                                               scene::EntityId entity) {
    for (const render::RenderInstance& inst : world.instances) {
        if (inst.entity == entity) return &inst;
    }
    return nullptr;
}

MaterialBinding ResolveMaterial(const EditorScene& es, const EntityRecordIndex& idx,
                                scene::EntityId entity) {
    MaterialBinding bind;
    const auto sit = idx.shape.find(entity);
    if (sit != idx.shape.end()) {
        const scene::CollisionShapeRecord& s = es.scene.GetShape(sit->second);
        if (s.material_id != scene::kInvalidMaterial) bind.material_id = s.material_id;
    }
    if (const render::RenderInstance* inst = InstanceOfEntity(es.sim->GetRenderWorld(), entity)) {
        bind.render_slot = inst->render_material_id;
    }
    bind.valid = bind.render_slot != render::kNoId &&
                 bind.render_slot < es.sim->GetRenderWorld().MaterialCount();
    return bind;
}

bool IsMovableInstance(const render::RenderInstance& inst, const EditorScene& es) {
    return IsMovableSource(inst.pose_source, es.Layout());
}

math::Transform AuthoredWorldTransform(const scene::SceneIR& scene, scene::EntityId entity) {
    const scene::Registry& ecs = scene.Ecs();
    const scene::SceneGraph& tree = scene.Tree();
    std::shared_ptr<scene::SceneNode> node = ecs.NodeOf(entity);
    if (!node) return math::Transform::Identity();
    // Collect the node chain to the root, then compose root-down.
    std::vector<scene::EntityId> chain;
    for (auto cur = node; cur && cur != tree.Root(); cur = tree.ParentOf(cur)) {
        chain.push_back(cur->entity);
    }
    math::Transform world = math::Transform::Identity();
    for (size_t i = chain.size(); i-- > 0;) {
        world = world * LocalOf(ecs, chain[i]);
    }
    return world;
}

namespace {

// A movable instance queues a teleport for the physics thread; a static visual has no
// physics row and moves in place.
void MoveInstance(EditorScene& es, render::RenderInstance& inst, const math::Transform& world) {
    PoseWrite write;
    if (ResolvePoseWrite(es.sim->GetRenderWorld(), es.Layout(), inst.entity, world, &write)) {
        es.sim->QueuePoseWrite(write);
    } else {
        inst.world_xform = world;
        es.sim->MarkStaticMoved();
    }
}

// The nearest body node at or above `e`, or kInvalidEntity.
scene::EntityId CarrierOf(const scene::SceneIR& s, scene::EntityId e) {
    const scene::Registry& ecs = s.Ecs();
    for (auto n = ecs.NodeOf(e); n; n = n->parent.lock())
        if (n->entity != scene::kInvalidEntity && ecs.Has<scene::RigidBodyComponent>(n->entity))
            return n->entity;
    return scene::kInvalidEntity;
}

}  // namespace

void ApplyTransformEdit(EditorScene& es, const EntityRecordIndex& idx,
                        scene::EntityId entity, const math::Transform& world, bool commit) {
    // Live: the entity's own instance moves, and each visual a body node carries follows it.
    const math::Transform to_entity = AuthoredWorldTransform(es.scene, entity).Inverse();
    for (render::RenderInstance& ri : es.sim->GetRenderWorld().instances) {
        if (ri.entity == entity)
            MoveInstance(es, ri, world);
        else if (idx.IsBody(entity) && CarrierOf(es.scene, ri.entity) == entity)
            MoveInstance(es, ri, world * to_entity * AuthoredWorldTransform(es.scene, ri.entity));
    }

    if (!commit) return;
    // Persist: re-express the world pose in the parent frame and write the record's
    // local transform (the authoring authority Save reads).
    math::Transform parent_world = math::Transform::Identity();
    if (const auto node = es.scene.Ecs().NodeOf(entity)) {
        if (const auto parent = es.scene.Tree().ParentOf(node)) {
            if (parent != es.scene.Tree().Root()) {
                parent_world = AuthoredWorldTransform(es.scene, parent->entity);
            }
        }
    }
    const math::Transform local = parent_world.Inverse() * world;
    const auto bit = idx.body.find(entity);
    if (bit != idx.body.end()) {
        es.scene.GetBodyMut(bit->second).local_transform = local;
        return;
    }
    const auto sit = idx.shape.find(entity);
    if (sit != idx.shape.end()) {
        es.scene.GetShapeMut(sit->second).local_transform = local;
    }
}

const render::RenderInstance* PoseInstance(const EditorScene& es, const EntityRecordIndex& idx,
                                           scene::EntityId entity) {
    const bool body = idx.IsBody(entity);
    const render::RenderInstance* carried = nullptr;
    for (const render::RenderInstance& ri : es.sim->GetRenderWorld().instances) {
        if (ri.entity == entity) return &ri;
        if (body && carried == nullptr && CarrierOf(es.scene, ri.entity) == entity) carried = &ri;
    }
    return carried;
}

math::Transform LiveWorldTransform(const scene::SceneIR& scene, scene::EntityId entity,
                                   const render::RenderInstance& inst) {
    if (inst.entity == entity) return inst.world_xform;
    // A carried visual keeps its authored offset from the body, as ApplyTransformEdit moves it.
    return inst.world_xform * AuthoredWorldTransform(scene, inst.entity).Inverse() *
           AuthoredWorldTransform(scene, entity);
}

void ApplyMaterialEdit(EditorScene& es, const MaterialBinding& bind,
                       const scene::RenderMaterial& mat, bool commit) {
    render::RenderWorld& rw = es.sim->GetRenderWorld();
    if (bind.render_slot != render::kNoId && bind.render_slot < rw.MaterialCount()) {
        rw.irradiance_volume.reset();
        rw.materials[bind.render_slot] = mat;
        // A map the world has not drawn yet decodes once here.
        render::DecodeMaterialTextures(rw);
    }
    if (!commit || bind.material_id == scene::kInvalidMaterial) return;
    WriteMaterialRecord(mat, &es.scene.GetMaterialMut(bind.material_id));
}

void WriteMaterialRecord(const scene::RenderMaterial& mat, scene::MaterialRecord* rec) {
    rec->base_color = math::Vec3{mat.base_color[0], mat.base_color[1], mat.base_color[2]};
    rec->alpha       = mat.base_color[3];
    rec->roughness   = mat.roughness;
    rec->metallic    = mat.metallic;
    rec->emissive    = math::Vec3{mat.emissive[0], mat.emissive[1], mat.emissive[2]};
    rec->sheen       = mat.sheen;
    rec->transmission = mat.transmission;
    rec->ior         = mat.ior;
    rec->absorption  = math::Vec3{mat.absorption[0], mat.absorption[1], mat.absorption[2]};
    rec->albedo_map    = mat.albedo_map;
    rec->roughness_map = mat.roughness_map;
    rec->normal_map    = mat.normal_map;
    rec->metallic_map  = mat.metallic_map;
    rec->emissive_map  = mat.emissive_map;
    rec->roughness_channel = mat.roughness_channel;
    rec->metallic_channel  = mat.metallic_channel;
    rec->uv_scale  = mat.uv_scale;
    rec->triplanar = mat.triplanar;
}

// -- scene-tree structural edits --------------------------------------------

namespace {

// The path of `entity`'s node in the current (possibly just re-projected) facade.
std::string PathOfEntity(const scene::SceneIR& s, scene::EntityId e) {
    return s.Tree().PathOf(s.Ecs().NodeOf(e));
}

// The BodyId an entity backs, or kInvalidBody. Linear over a handful of bodies.
scene::BodyId BodyOfEntity(const scene::SceneIR& s, scene::EntityId e) {
    for (scene::BodyId i = 0; i < s.RigidBodyCount(); ++i)
        if (s.EntityOfBody(i) == e) return i;
    return scene::kInvalidBody;
}

}  // namespace

std::string RenameNode(EditorScene& es, const std::string& path,
                       const std::string& new_leaf) {
    if (new_leaf.empty() || new_leaf.find('/') != std::string::npos) return "";
    const auto node = es.scene.Tree().NodeOf(path);
    if (!node) return "";
    const scene::EntityId e = node->entity;

    for (scene::BodyId i = 0; i < es.scene.RigidBodyCount(); ++i) {
        if (es.scene.EntityOfBody(i) != e) continue;
        // Replace the record name's last '/'-segment, preserving any group prefix.
        std::string nm = es.scene.GetBody(i).name;
        const size_t slash = nm.rfind('/');
        es.scene.GetBodyMut(i).name =
            (slash == std::string::npos) ? new_leaf : nm.substr(0, slash + 1) + new_leaf;
        return PathOfEntity(es.scene, es.scene.EntityOfBody(i));
    }
    for (scene::ShapeId i = 0; i < es.scene.ShapeCount(); ++i) {
        if (es.scene.EntityOfShape(i) != e) continue;
        es.scene.GetShapeMut(i).name = new_leaf;
        return PathOfEntity(es.scene, es.scene.EntityOfShape(i));
    }
    for (scene::MediaId i = 0; i < es.scene.MediaCount(); ++i) {
        if (es.scene.EntityOfMedia(i) != e) continue;
        es.scene.GetMediaMut(i).name = new_leaf;
        return PathOfEntity(es.scene, es.scene.EntityOfMedia(i));
    }
    // A pure group renames the prefix of every body name under it, unless the new path is taken.
    const size_t slash = path.rfind('/');
    const std::string renamed = (slash == std::string::npos ? std::string() : path.substr(0, slash + 1)) + new_leaf;
    const auto under = [](const std::string& name, const std::string& group) {
        return name == group || name.rfind(group + "/", 0) == 0;
    };
    for (const scene::RigidBodyRecord& r : es.scene.Bodies())
        if (under(r.name, renamed)) return "";
    bool any = false;
    for (scene::BodyId i = 0; i < es.scene.RigidBodyCount(); ++i) {
        if (!under(es.scene.GetBody(i).name, path)) continue;
        es.scene.GetBodyMut(i).name = renamed + es.scene.GetBody(i).name.substr(path.size());
        any = true;
    }
    return any ? renamed : "";
}

namespace {

// Per-kind spawn geometry + dynamics. The ONE switch in the spawn path: only the
// geometry and whether the kind is a (massless) static plane vary; everything
// downstream (records, projection, cook, render) is general.
struct SpawnSpec {
    scene::ShapeType type;
    math::Vec3       half_extents;   // box / plane
    float            radius;         // sphere / capsule / cylinder
    float            half_height;    // capsule / cylinder
    bool             is_static;      // plane: a ground-like body has no mass
    const char*      tag;            // node + material base name
};

SpawnSpec SpecOf(PrimitiveKind kind) {
    switch (kind) {
        case PrimitiveKind::Sphere:
            return {scene::ShapeType::Sphere,  {0.25f, 0.25f, 0.25f}, 0.25f, 0.25f, false, "sphere"};
        case PrimitiveKind::Capsule:
            return {scene::ShapeType::Capsule, {0.20f, 0.20f, 0.45f}, 0.20f, 0.25f, false, "capsule"};
        case PrimitiveKind::Cylinder:
            return {scene::ShapeType::Cylinder, {0.20f, 0.20f, 0.25f}, 0.20f, 0.25f, false, "cylinder"};
        case PrimitiveKind::Plane:
            return {scene::ShapeType::Plane,   {5.0f, 5.0f, 0.05f},   0.50f, 0.50f, true,  "plane"};
        case PrimitiveKind::Box:
        default:
            return {scene::ShapeType::Box,     {0.25f, 0.25f, 0.25f}, 0.25f, 0.25f, false, "box"};
    }
}

}  // namespace

std::string SpawnPrimitive(EditorScene& es, PrimitiveKind kind,
                           const math::Transform& placement,
                           const std::string& parent_path) {
    const SpawnSpec spec = SpecOf(kind);

    scene::BodyId parent_body = scene::kInvalidBody;
    if (const auto parent = es.scene.Tree().NodeOf(parent_path))
        parent_body = BodyOfEntity(es.scene, parent->entity);

    // Re-express the requested WORLD placement in the parent's frame (identity for a
    // group / root parent), matching ReparentNode's pose preservation.
    math::Transform parent_world = math::Transform::Identity();
    if (parent_body != scene::kInvalidBody)
        parent_world = AuthoredWorldTransform(es.scene, es.scene.EntityOfBody(parent_body));

    scene::RigidBodyRecord b;
    b.is_static = spec.is_static;
    b.mass = spec.is_static ? 0.0f : 1.0f;
    // Under a parent body the name is the bare leaf; under a group / root the group
    // prefix path carries it there (the tree dedupes duplicate sibling names).
    b.parent_id = parent_body;
    b.name = (parent_body != scene::kInvalidBody || parent_path.empty())
                 ? std::string(spec.tag)
                 : parent_path + "/" + spec.tag;
    // Body names stay unique: a rebuild carries state over by body name.
    const auto taken = [&](const std::string& n) {
        for (const scene::RigidBodyRecord& r : es.scene.Bodies())
            if (r.name == n) return true;
        return false;
    };
    const std::string base = b.name;
    for (int i = 1; taken(b.name); ++i) b.name = base + "_" + std::to_string(i);
    b.local_transform = parent_world.Inverse() * placement;
    const scene::BodyId nb = es.scene.AddRigidBody(b);

    scene::MaterialRecord m;
    m.name = std::string(spec.tag) + "_mat";
    m.base_color = math::Vec3{0.72f, 0.72f, 0.75f};
    const scene::MaterialId mid = es.scene.AddMaterial(m);

    // The colliding geom drives physics: the cook makes the body's first colliding
    // shape its collidable row.
    scene::CollisionShapeRecord cs;
    cs.body_id = nb;
    cs.type = spec.type;
    cs.half_extents = spec.half_extents;
    cs.radius = spec.radius;
    cs.half_height = spec.half_height;
    cs.material_id = mid;
    cs.name = std::string(spec.tag) + "_geom";
    es.scene.AddCollisionShape(cs);

    // A visual-only twin (contype/conaffinity 0) projects to a VisualMeshComponent so
    // the body renders unconditionally; the cook keeps it non-colliding (no extra row).
    scene::CollisionShapeRecord vis = cs;
    vis.name = std::string(spec.tag) + "_visual";
    vis.contype = 0;
    vis.conaffinity = 0;
    es.scene.AddCollisionShape(vis);

    return PathOfEntity(es.scene, es.scene.EntityOfBody(nb));
}

std::string AddBoxChild(EditorScene& es, const std::string& parent_path) {
    math::Transform placement;
    placement.position = math::Vec3{0.0f, 0.0f, 0.5f};
    return SpawnPrimitive(es, PrimitiveKind::Box, placement, parent_path);
}

std::string DeleteSubtree(EditorScene& es, const std::string& path) {
    const auto node = es.scene.Tree().NodeOf(path);
    if (!node) return "";
    const std::string parent_path =
        node->parent.lock() ? es.scene.Tree().PathOf(node->parent.lock()) : std::string();

    // A /script node deletes as its one record (never cooked -> no physics change).
    for (size_t i = 0; i < es.scene.ScriptCount(); ++i) {
        const auto sid = static_cast<scene::ScriptId>(i);
        if (es.scene.EntityOfScript(sid) == node->entity) {
            return es.scene.RemoveScript(sid) ? parent_path : "";
        }
    }
    // A shape node, a collider or a visual, deletes as its one record.
    for (size_t i = 0; i < es.scene.ShapeCount(); ++i) {
        const auto sid = static_cast<scene::ShapeId>(i);
        if (es.scene.EntityOfShape(sid) == node->entity) return es.scene.RemoveShape(sid) ? parent_path : "";
    }

    // A body subtree deletes through the ONE general SceneIR remover: joints /
    // actuators / sensors / pairs / touched ICs go with it, everything else persists.
    const scene::BodyId root = BodyOfEntity(es.scene, node->entity);
    if (root != scene::kInvalidBody) return es.scene.RemoveBodySubtree(root) ? parent_path : "";
    // A group deletes each body subtree its path prefixes; a removal re-densifies the ids, so each scan restarts.
    const std::string prefix = path + "/";
    bool removed = false;
    for (bool more = true; more; removed = removed || more) {
        more = false;
        for (scene::BodyId b = 0; b < es.scene.RigidBodyCount() && !more; ++b) {
            const scene::RigidBodyRecord& r = es.scene.GetBody(b);
            if (r.parent_id == scene::kInvalidBody && r.name.rfind(prefix, 0) == 0)
                more = es.scene.RemoveBodySubtree(b);
        }
    }
    return removed ? parent_path : "";
}

std::string ReparentNode(EditorScene& es, const std::string& path,
                         const std::string& new_parent_path) {
    const auto node = es.scene.Tree().NodeOf(path);
    if (!node) return "";
    const scene::BodyId b = BodyOfEntity(es.scene, node->entity);
    if (b == scene::kInvalidBody) return "";  // only body subtrees reparent

    // Jointed / settled bodies carry articulation structure a reparent would
    // reshape; decline rather than silently miswire it (same guard as delete).
    for (size_t j = 0; j < es.scene.JointCount(); ++j) {
        const scene::JointRecord& jr = es.scene.GetJoint(static_cast<scene::JointId>(j));
        if (jr.parent_body == b || jr.child_body == b) return "";
    }
    if (!es.scene.InitialState().empty()) return "";

    // The new parent body (kInvalidBody when reparenting under a group / root).
    scene::BodyId new_parent = scene::kInvalidBody;
    std::string group_prefix;
    if (!new_parent_path.empty()) {
        const auto pnode = es.scene.Tree().NodeOf(new_parent_path);
        if (!pnode) return "";
        new_parent = BodyOfEntity(es.scene, pnode->entity);
        if (new_parent == b) return "";  // no self-parenting
        // Only a body or a PURE group can adopt: a shape/media/script node's name
        // as a group prefix would mint a same-named phantom group. Decline.
        if (new_parent == scene::kInvalidBody) {
            const scene::EntityId pe = pnode->entity;
            for (scene::ShapeId i = 0; i < es.scene.ShapeCount(); ++i)
                if (es.scene.EntityOfShape(i) == pe) return "";
            for (scene::MediaId i = 0; i < es.scene.MediaCount(); ++i)
                if (es.scene.EntityOfMedia(i) == pe) return "";
            for (size_t i = 0; i < es.scene.ScriptCount(); ++i)
                if (es.scene.EntityOfScript(static_cast<scene::ScriptId>(i)) == pe) return "";
            for (size_t i = 0; i < es.scene.JointCount(); ++i)
                if (es.scene.EntityOfJoint(static_cast<scene::JointId>(i)) == pe) return "";
            group_prefix = new_parent_path;
        }
    }

    // Reject a cycle: the new parent must not be b or one of b's descendants.
    if (new_parent != scene::kInvalidBody) {
        std::vector<bool> desc(es.scene.RigidBodyCount(), false);
        desc[b] = true;
        for (bool more = true; more;) {
            more = false;
            for (scene::BodyId i = 0; i < es.scene.RigidBodyCount(); ++i) {
                const scene::BodyId p = es.scene.GetBody(i).parent_id;
                if (!desc[i] && p != scene::kInvalidBody && p < desc.size() && desc[p]) {
                    desc[i] = true;
                    more = true;
                }
            }
        }
        if (new_parent < desc.size() && desc[new_parent]) return "";
    }

    // Preserve the world pose: re-express it in the new parent body's frame (or world
    // for a group / root parent). The leaf name moves under the new group prefix.
    const math::Transform b_world =
        AuthoredWorldTransform(es.scene, es.scene.EntityOfBody(b));
    math::Transform parent_world = math::Transform::Identity();
    if (new_parent != scene::kInvalidBody)
        parent_world = AuthoredWorldTransform(es.scene, es.scene.EntityOfBody(new_parent));

    std::string nm = es.scene.GetBody(b).name;
    const size_t slash = nm.rfind('/');
    const std::string leaf = (slash == std::string::npos) ? nm : nm.substr(slash + 1);
    scene::RigidBodyRecord& rec = es.scene.GetBodyMut(b);
    rec.parent_id = new_parent;
    rec.name = group_prefix.empty() ? leaf : group_prefix + "/" + leaf;
    rec.local_transform = parent_world.Inverse() * b_world;
    return PathOfEntity(es.scene, es.scene.EntityOfBody(b));
}

std::vector<scene::BodyId> JointSubtree(const scene::SceneIR& scene, scene::BodyId root) {
    std::vector<scene::BodyId> bodies{root};
    for (size_t i = 0; i < bodies.size(); ++i)
        for (const scene::JointRecord& j : scene.Joints())
            if (j.parent_body == bodies[i]) bodies.push_back(j.child_body);
    return bodies;
}

std::string AttachBody(scene::SceneIR& scene, scene::BodyId child, scene::BodyId parent, scene::BodyId mount,
                       scene::JointType type, const math::Vec3& axis, const math::Transform* frame, std::string name) {
    const math::Transform at = BodyWorld(scene, mount);
    const math::Transform in_child = BodyWorld(scene, child).Inverse() * at;
    const math::Transform place = frame != nullptr ? *frame : BodyWorld(scene, parent).Inverse() * at;
    if (name.empty()) {
        const std::string& n = scene.GetBody(child).name;
        name = n.substr(n.rfind('/') + 1u) + "_mount";
    }
    const auto taken = [&](const std::string& n) {
        for (const scene::JointRecord& j : scene.Joints())
            if (j.name == n) return true;
        return false;
    };
    const std::string base = name;
    for (int i = 2; taken(name); ++i) name = base + "_" + std::to_string(i);
    scene::JointRecord joint;
    joint.name = name;
    joint.parent_body = parent;
    joint.child_body = child;
    joint.type = type;
    joint.axis = axis.Normalized();
    joint.parent_frame = place * in_child.Inverse();
    scene.InsertJoint(std::move(joint));
    // A link's pose is its joint's frame, as an imported robot's links have it.
    scene::RigidBodyRecord& body = scene.GetBodyMut(child);
    body.parent_id = scene::kInvalidBody;
    body.local_transform = math::Transform::Identity();
    body.is_static = false;
    return name;
}

// -- euler<->quat (intrinsic XYZ, degrees) ----------------------------------

math::Quat QuatFromEulerDeg(const math::Vec3& deg) {
    constexpr float kDeg = 3.14159265358979323846f / 180.0f;
    const math::Quat qx = math::Quat::FromAxisAngle({1.0f, 0.0f, 0.0f}, deg.x * kDeg);
    const math::Quat qy = math::Quat::FromAxisAngle({0.0f, 1.0f, 0.0f}, deg.y * kDeg);
    const math::Quat qz = math::Quat::FromAxisAngle({0.0f, 0.0f, 1.0f}, deg.z * kDeg);
    return (qx * qy * qz).Normalized();
}

math::Vec3 EulerDegFromQuat(const math::Quat& q) {
    const math::Quat n = q.Normalized();
    const float w = n.w, x = n.x, y = n.y, z = n.z;
    // Rotation-matrix entries for R = Rx*Ry*Rz; extract X (a), Y (b), Z (c).
    const float m00 = 1.0f - 2.0f * (y * y + z * z);
    const float m01 = 2.0f * (x * y - w * z);
    const float m02 = 2.0f * (x * z + w * y);
    const float m12 = 2.0f * (y * z - w * x);
    const float m22 = 1.0f - 2.0f * (x * x + y * y);
    const float m11 = 1.0f - 2.0f * (x * x + z * z);
    const float m21 = 2.0f * (y * z + w * x);
    const float sb = std::min(1.0f, std::max(-1.0f, m02));
    const float b = std::asin(sb);
    float a, c;
    if (std::fabs(m02) < 0.9999f) {
        a = std::atan2(-m12, m22);
        c = std::atan2(-m01, m00);
    } else {  // gimbal: fold the dependent angle into X
        a = std::atan2(m21, m11);
        c = 0.0f;
    }
    constexpr float kRad = 180.0f / 3.14159265358979323846f;
    return math::Vec3{a * kRad, b * kRad, c * kRad};
}

}  // namespace nuka::runtime::app::viewer
