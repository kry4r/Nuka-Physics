#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer -- the editor's ONE general scene-edit seam.
//
// Every property edit (inspector, gizmo, tree op) routes through here so there is
// a single mutation path, never a per-type poke. An edit updates BOTH the live
// runtime (the CommandQueue/MoveEntity physics seam for movable bodies, the
// RenderWorld for render state) AND, on commit, the authoritative SceneIR record
// nks::Save persists. Operates on the general SceneIR / RenderWorld / ECS, with no
// scene-specific code.
//
// HOST-ONLY / zero-CUDA-token: the only device touch is nk::Data via the queued
// MoveEntity (applied in Simulation::FramePublish behind the phi vtable).
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "render/render_world.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "scene/ecs/components.hpp"
#include "scene/ecs/entity.hpp"

#include <unordered_map>

namespace nuka::runtime::app::viewer {

// Reverse index: a selected entity -> the SceneIR record id it carries. Rebuilt
// whenever the scene's facade is (re)projected so an edit can fetch the
// authoritative record behind a node. Bodies / shapes / media cover the editable
// node kinds; other nodes resolve to none.
struct EntityRecordIndex {
    std::unordered_map<scene::EntityId, scene::BodyId, scene::EntityIdHash>  body;
    std::unordered_map<scene::EntityId, scene::ShapeId, scene::EntityIdHash> shape;
    std::unordered_map<scene::EntityId, scene::MediaId, scene::EntityIdHash> media;

    void Build(const scene::SceneIR& scene);
    bool IsBody(scene::EntityId e) const { return body.count(e) != 0; }
    bool IsShape(scene::EntityId e) const { return shape.count(e) != 0; }
    bool IsMedia(scene::EntityId e) const { return media.count(e) != 0; }
};

// The render + authoring material behind a selected entity. `render_slot` is the
// RenderWorld.materials index (live preview); `material_id` is the authoring
// record (Save). valid==false when the entity carries no editable material.
struct MaterialBinding {
    bool              valid       = false;
    scene::MaterialId material_id = scene::kInvalidMaterial;
    uint32_t          render_slot = render::kNoId;
};

// The RenderInstance a selected entity owns (nullptr if none).
const render::RenderInstance* InstanceOfEntity(const render::RenderWorld& world,
                                               scene::EntityId entity);

// Resolve which render-material slot + authoring MaterialRecord a selected entity
// edits (via its shape's material). Falls back to the shared default render slot
// when the shape carries no material id.
MaterialBinding ResolveMaterial(const EditorScene& es, const EntityRecordIndex& idx,
                                scene::EntityId entity);

// True iff the instance's pose is a live, teleportable physics row (free Body,
// articulation Base, or the floating root Link) -- i.e. a MoveEntity reaches it.
bool IsMovableInstance(const render::RenderInstance& inst, const EditorScene& es);

// The authored WORLD transform of an entity, composing the TransformComponent
// chain from the root down (the bind pose Save persists; not the live physics
// pose). Identity for an entity with no node / transform.
math::Transform AuthoredWorldTransform(const scene::SceneIR& scene,
                                       scene::EntityId entity);

// Apply a transform edit. `world` is the new WORLD pose of the selected instance.
// Live: a movable body teleports via the CommandQueue (MoveEntity -> ApplyMoveEntity);
// a static instance updates its RenderInstance in place. When `commit`, the
// authoring record's local transform is rewritten (world re-expressed in the
// parent frame) so Save persists it.
void ApplyTransformEdit(EditorScene& es, const EntityRecordIndex& idx,
                        scene::EntityId entity, const math::Transform& world,
                        bool commit);

// Apply a material edit. Live: RenderWorld.materials[render_slot] = `mat`. When
// `commit`, the authoring MaterialRecord is rewritten so Save persists it.
void ApplyMaterialEdit(EditorScene& es, const MaterialBinding& bind,
                       const scene::RenderMaterial& mat, bool commit);

// Euler<->quat (intrinsic XYZ, degrees) for the inspector's rotation rows.
math::Quat QuatFromEulerDeg(const math::Vec3& deg);
math::Vec3 EulerDegFromQuat(const math::Quat& q);

// -- scene-tree structural edits --------------------------------------------
// These mutate es.scene's RECORDS (the authority); the derived tree / ECS / entity
// ids re-project afterward, so the viewer rebuilds its record index + re-resolves
// selection by PATH. Add / Delete need a re-cook (RecookEditorScene); Rename does
// not (a node name does not reach the cooked model). They take + return tree PATHS
// because a path is the only identity stable across the re-projection.

// Replace the leaf segment of the record-backed node at `path` with `new_leaf` (one
// segment, no '/'). A body keeps any group prefix in its record name; a shape / media
// sets its record name. Returns the node's new path, or "" if `path` is a pure group
// (derived, not authored), is not record-backed, or `new_leaf` is empty/invalid.
std::string RenameNode(EditorScene& es, const std::string& path,
                       const std::string& new_leaf);

// Add a movable unit-box body (+ box shape + neutral material) under the body or
// group at `parent_path` ("" -> scene root). Returns the new node's path, or "".
std::string AddBoxChild(EditorScene& es, const std::string& parent_path);

// Remove the body subtree rooted at `path` (its bodies + shapes + the joints that
// touch them) by rebuilding the authority without them. Returns the parent path to
// reselect, or "" when `path` is not a body node.
std::string DeleteSubtree(EditorScene& es, const std::string& path);

// Move the body subtree at `path` under the body / group at `new_parent_path` ("" ->
// scene root), preserving its world pose. Returns the node's new path, or "" when
// `path` is not a body, the move is a cycle, or the body is jointed / settled.
std::string ReparentNode(EditorScene& es, const std::string& path,
                         const std::string& new_parent_path);

}  // namespace nuka::runtime::app::viewer
