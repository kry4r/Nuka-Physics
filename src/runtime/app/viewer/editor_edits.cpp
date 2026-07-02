// ---------------------------------------------------------------------------
// The editor's ONE general scene-edit seam (see editor_edits.hpp).
//
// HOST-ONLY / zero-CUDA-token.
// ---------------------------------------------------------------------------

#include "runtime/app/viewer/editor_edits.hpp"

#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "runtime/app/command_queue.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"

#include <cmath>
#include <vector>

namespace nuka::runtime::app::viewer {

namespace {

namespace render = nuka::render;

// The cooked Model's FloatingBase joint type value (mirror of systems.cpp's
// constant; the canonical enum header pulls in CUDA, off-limits here).
constexpr uint8_t kFloatingBaseJointType = 3u;

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
    using K = render::PoseSource::Kind;
    if (inst.pose_source.kind == K::Body || inst.pose_source.kind == K::Base) return true;
    if (inst.pose_source.kind != K::Link) return false;
    const nk::ModelArticulation& art = es.world->GetModel().articulation;
    const uint32_t root = (art.root_link != ~uint32_t(0)) ? art.root_link : 0u;
    return inst.pose_source.row == root && root < art.joint_type.size() &&
           art.joint_type[root] == kFloatingBaseJointType;
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

void ApplyTransformEdit(EditorScene& es, const EntityRecordIndex& idx,
                        scene::EntityId entity, const math::Transform& world, bool commit) {
    render::RenderWorld& rw = es.sim->GetRenderWorld();
    render::RenderInstance* inst = nullptr;
    for (render::RenderInstance& ri : rw.instances) {
        if (ri.entity == entity) { inst = &ri; break; }
    }

    // Live: teleport a movable body through the general MoveEntity path; otherwise
    // move the render instance in place (static visuals have no physics row).
    if (inst != nullptr) {
        if (IsMovableInstance(*inst, es)) {
            es.sim->Commands().Push(Command::MakeMoveEntity(entity, world));
        } else {
            inst->world_xform = world;
        }
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

void ApplyMaterialEdit(EditorScene& es, const MaterialBinding& bind,
                       const scene::RenderMaterial& mat, bool commit) {
    render::RenderWorld& rw = es.sim->GetRenderWorld();
    if (bind.render_slot != render::kNoId && bind.render_slot < rw.MaterialCount()) {
        rw.materials[bind.render_slot] = mat;
    }
    if (!commit || bind.material_id == scene::kInvalidMaterial) return;
    scene::MaterialRecord& rec = es.scene.GetMaterialMut(bind.material_id);
    rec.base_color = math::Vec3{mat.base_color[0], mat.base_color[1], mat.base_color[2]};
    rec.alpha       = mat.base_color[3];
    rec.roughness   = mat.roughness;
    rec.metallic    = mat.metallic;
    rec.emissive    = math::Vec3{mat.emissive[0], mat.emissive[1], mat.emissive[2]};
    rec.sheen       = mat.sheen;
    rec.transmission = mat.transmission;
    rec.ior         = mat.ior;
    rec.absorption  = math::Vec3{mat.absorption[0], mat.absorption[1], mat.absorption[2]};
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
    return "";  // pure group / not record-backed
}

namespace {

// Per-kind spawn geometry + dynamics. The ONE switch in the spawn path: only the
// geometry and whether the kind is a (massless) static plane vary; everything
// downstream (records, projection, cook, render) is general.
struct SpawnSpec {
    scene::ShapeType type;
    math::Vec3       half_extents;   // box / plane
    float            radius;         // sphere / capsule
    float            half_height;    // capsule
    bool             is_static;      // plane: a ground-like body has no mass
    const char*      tag;            // node + material base name
};

SpawnSpec SpecOf(PrimitiveKind kind) {
    switch (kind) {
        case PrimitiveKind::Sphere:
            return {scene::ShapeType::Sphere,  {0.25f, 0.25f, 0.25f}, 0.25f, 0.25f, false, "sphere"};
        case PrimitiveKind::Capsule:
            return {scene::ShapeType::Capsule, {0.20f, 0.20f, 0.45f}, 0.20f, 0.25f, false, "capsule"};
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

    // A body subtree deletes through the ONE general SceneIR remover: joints /
    // actuators / sensors / pairs / touched ICs go with it, everything else persists.
    const scene::BodyId root = BodyOfEntity(es.scene, node->entity);
    if (root == scene::kInvalidBody) return "";
    return es.scene.RemoveBodySubtree(root) ? parent_path : "";
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
