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
