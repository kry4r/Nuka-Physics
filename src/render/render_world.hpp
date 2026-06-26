#pragma once
// ---------------------------------------------------------------------------
// nuka::render::RenderWorld - the host-side, render-backend-agnostic scene the
// renderer consumes (M8 manifest #1).
//
// RenderWorld is a PURE DATA PRODUCT built ONCE from the ECS Registry + the
// cook's SceneMap (BuildRenderWorld). It carries everything a renderer needs to
// draw a frame -- per-instance world transforms, deduplicated host triangle
// geometry (MeshLibrary), a PBR material table, and camera/light tables -- and
// NOTHING renderer-specific. There is NO Vulkan, NO CUDA, NO GPU type anywhere
// in this header so that BOTH downstream consumers can be built on top of it:
//   * the M8 Vulkan forward PBR raster renderer (src/render/raster/...), and
//   * the M11 software CUDA path-tracer via RenderWorldToTwoLevelScene
//     (src/rt/... + rt_adapter).
// Keeping RenderWorld a backend-agnostic data product is the explicit M11
// RT-consumer design constraint (recon brief §4.6 / Risk R-design): the same
// RenderWorld feeds raster AND the path-tracer with no renderer types leaking
// in.
//
// The frame loop's TransformSyncSystem (M8 T4) overwrites each instance's
// `world_xform` every frame as `downloaded_pose ∘ cached_visual_local`, where
// `downloaded_pose` comes from Data::DownloadField(pose_source.field, row) for
// the SELECTED env (host-download path, NO CUDA in this layer). Resolving the
// pose source (which FieldId + which row) is done ONCE at BuildRenderWorld time
// and stored on the instance so the per-frame publisher never re-walks SceneMap
// (recon §2 Tier C #1 + Decision D1).
//
// HOST-ONLY: this file (and render_world.cpp) must contain no CUDA tokens (no
// triple-chevron kernel launch, no cuda-runtime call, no cuda_runtime include,
// no phi backend_cuda include) -- enforced by the zero-CUDA-token lint red-line
// over src/render/** (recon §4.5).
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "scene/asset/asset_ref.hpp"
#include "scene/ecs/components.hpp"
#include "scene/ecs/entity.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nuka::scene {
class Registry;
class SceneMap;
}  // namespace nuka::scene

namespace nuka::render {

// Sentinel "no row / no id" matching SceneMap::kNoRow / Registry::kNoSlot.
inline constexpr uint32_t kNoId = ~uint32_t(0);

// ---------------------------------------------------------------------------
// MeshGeometry - host triangle geometry for ONE renderable mesh.
//
// Interleaving-free, render-backend-agnostic SoA-ish streams (positions are
// mandatory; normals/uvs may be empty -> the renderer synthesizes them).
// Sourced either from a decoded .nka MESH chunk (DecodeMesh) or, when no MESH
// chunk exists for an instance, from a collision-primitive tessellation so that
// EVERY RenderInstance always has renderable triangles (Decision D3 fallback).
// ---------------------------------------------------------------------------
struct MeshGeometry {
    std::vector<float>    positions;  // x,y,z per vertex (3 floats/vertex)
    std::vector<float>    normals;    // x,y,z per vertex (empty => renderer derives)
    std::vector<float>    uvs;        // u,v   per vertex (empty => none)
    std::vector<uint32_t> indices;    // triangle indices (3 per triangle)

    uint32_t VertexCount() const { return static_cast<uint32_t>(positions.size() / 3); }
    uint32_t TriangleCount() const { return static_cast<uint32_t>(indices.size() / 3); }
    bool     Empty() const { return positions.empty() || indices.empty(); }
};

// How a mesh in the library was produced (informational; lets a consumer or
// the D3 report distinguish asset-backed fidelity from primitive fallback).
enum class MeshSource : uint8_t {
    NkaMesh,           // real triangles decoded from a .nka MESH chunk
    PrimitiveFallback  // tessellated collision primitive (box/sphere/capsule/plane)
};

// ---------------------------------------------------------------------------
// MeshLibrary - deduplicated store of host triangle geometry.
//
// Meshes are deduped so identical .nka MESH references (same AssetRef) share one
// entry; primitive-fallback meshes are deduped by a synthetic key so identical
// primitives (same kind + params) also share one entry. RenderInstance::mesh_id
// indexes into `meshes`.
// ---------------------------------------------------------------------------
class MeshLibrary {
public:
    // Number of distinct meshes held.
    uint32_t Count() const { return static_cast<uint32_t>(meshes_.size()); }

    // Geometry / provenance for a mesh id (id < Count()).
    const MeshGeometry& Geometry(uint32_t mesh_id) const { return meshes_[mesh_id]; }
    MeshSource          Source(uint32_t mesh_id) const { return sources_[mesh_id]; }

    // Replace an interned mesh's geometry in place (a deforming particle surface
    // updates its ONE mesh each frame; id/key stay stable so the table never grows).
    void ReplaceGeometry(uint32_t mesh_id, MeshGeometry geometry) {
        meshes_[mesh_id] = std::move(geometry);
    }

    // Intern a mesh decoded from a .nka MESH chunk keyed by its AssetRef. The
    // first call with a given ref loads + decodes (via `loader`) and stores it;
    // subsequent calls with an equal ref return the same id. `loader` returns the
    // decoded geometry; it runs at most once per distinct ref.
    template <class Loader>
    uint32_t InternNkaMesh(const scene::AssetRef& ref, Loader&& loader) {
        const std::string key = "nka:" + scene::ToString(ref);
        auto it = key_to_id_.find(key);
        if (it != key_to_id_.end()) {
            return it->second;
        }
        const uint32_t id = static_cast<uint32_t>(meshes_.size());
        meshes_.push_back(loader());
        sources_.push_back(MeshSource::NkaMesh);
        key_to_id_.emplace(key, id);
        return id;
    }

    // Intern a primitive-fallback mesh keyed by a caller-supplied stable key
    // (e.g. "prim:box:0.5,0.5,0.5"). `builder` runs at most once per distinct
    // key and returns the tessellated geometry.
    template <class Builder>
    uint32_t InternPrimitive(const std::string& key, Builder&& builder) {
        auto it = key_to_id_.find(key);
        if (it != key_to_id_.end()) {
            return it->second;
        }
        const uint32_t id = static_cast<uint32_t>(meshes_.size());
        meshes_.push_back(builder());
        sources_.push_back(MeshSource::PrimitiveFallback);
        key_to_id_.emplace(key, id);
        return id;
    }

private:
    std::vector<MeshGeometry>                  meshes_;
    std::vector<MeshSource>                     sources_;
    std::unordered_map<std::string, uint32_t>   key_to_id_;
};

// ---------------------------------------------------------------------------
// PoseSource - resolved ONCE at build time: which Data field + which row drives
// an instance's pose. The publisher reads exactly this (no SceneMap re-walk).
//
//   Link  -> Data field LinkPose, indexed by `row` = CookedRef::link_index
//   Body  -> Data field BodyPose, indexed by `row` = CookedRef::body_row
//   Base  -> Data field BasePose, indexed by `row` = env's base slot
//   Static-> no physics pose; the instance stays at its bind-pose world_xform
//
// `field` deliberately does NOT name nk::FieldId here so render_world.hpp keeps
// no dependency on the nk core; the publisher (runtime/app) maps PoseSource::Kind
// -> FieldId. This keeps RenderWorld a pure data product (M11 RT-consumer safe).
// ---------------------------------------------------------------------------
struct PoseSource {
    enum class Kind : uint8_t { Static, Link, Body, Base };
    Kind     kind = Kind::Static;
    uint32_t row  = kNoId;  // link_index / body_row / base slot (kNoId for Static)
};

// ---------------------------------------------------------------------------
// RenderInstance - one renderable, per-ENTITY.
//
// `world_xform` is the live frame transform the renderer draws with. It is
// initialized to the entity's REST/BIND world pose at build time and overwritten
// each frame by TransformSync as `downloaded_pose ∘ cached_visual_local`.
//
// `cached_visual_local` is the composed transform from the PHYSICS node frame
// (the link/body whose Data pose drives this instance, identified by
// `pose_source`) DOWN to the visual node -- computed ONCE at build time by
// walking the SceneGraph node chain (Decision D1). Where the visual node IS the
// physics node, this is identity (the geom sits directly on its body) or the
// geom's own local offset.
// ---------------------------------------------------------------------------
struct RenderInstance {
    scene::EntityId entity            = scene::kInvalidEntity;
    uint32_t        mesh_id           = kNoId;        // index into MeshLibrary
    uint32_t        render_material_id = kNoId;        // index into materials table
    math::Transform world_xform       = math::Transform::Identity();  // live frame pose
    math::Transform cached_visual_local = math::Transform::Identity(); // physics-frame -> visual-frame
    PoseSource      pose_source;                       // where to read the live pose
};

// Camera / light entries are flat copies of the corresponding components with
// their resolved bind-pose world transforms, indexed independently of instances.
struct RenderCamera {
    scene::EntityId entity              = scene::kInvalidEntity;
    math::Transform world_xform         = math::Transform::Identity();
    float           vertical_fov_degrees = 45.0f;
    float           near_clip            = 0.01f;
    float           far_clip            = 1000.0f;
    PoseSource      pose_source;  // a camera may be attached to a moving body
    math::Transform cached_visual_local = math::Transform::Identity();
};

struct RenderLight {
    scene::EntityId entity      = scene::kInvalidEntity;
    scene::LightComponent::Type type = scene::LightComponent::Type::Point;
    math::Transform world_xform = math::Transform::Identity();
    math::Vec3      color       = {1.0f, 1.0f, 1.0f};
    float           intensity   = 1.0f;
    PoseSource      pose_source;
    math::Transform cached_visual_local = math::Transform::Identity();
};

// ---------------------------------------------------------------------------
// RenderWorld - the full backend-agnostic render scene.
// ---------------------------------------------------------------------------
struct RenderWorld {
    std::vector<RenderInstance>        instances;
    MeshLibrary                        meshes;
    std::vector<scene::RenderMaterial> materials;  // indexed by render_material_id
    std::vector<RenderCamera>          cameras;
    std::vector<RenderLight>           lights;

    // The default render material applied when an instance references no material
    // (render_material_id == kNoId). Lives at the end of `materials` after build.
    uint32_t default_material_id = kNoId;

    uint32_t InstanceCount() const { return static_cast<uint32_t>(instances.size()); }
    uint32_t MaterialCount() const { return static_cast<uint32_t>(materials.size()); }
    uint32_t CameraCount() const { return static_cast<uint32_t>(cameras.size()); }
    uint32_t LightCount() const { return static_cast<uint32_t>(lights.size()); }
};

// ---------------------------------------------------------------------------
// BuildRenderWorld - construct the RenderWorld ONCE from the ECS Registry and
// the cook's SceneMap.
//
//   * Walks the Registry for every entity with a VisualMeshComponent -> one
//     RenderInstance each.
//   * Resolves the instance's mesh: the VisualMeshComponent.mesh AssetRef loads
//     real triangles from its .nka MESH chunk where present; otherwise falls back
//     to a collision-primitive tessellation (the entity's CollisionShapeComponent
//     if it has one, else a unit box) so the instance ALWAYS has triangles
//     (Decision D3). Meshes are deduped into the MeshLibrary.
//   * Resolves the material (VisualMeshComponent.render_material_id -> the
//     Registry render-material table; a shared default otherwise).
//   * Resolves cached_visual_local by composing the TransformComponent.local
//     chain from the entity's nearest physics-bound ancestor node down to the
//     visual node (Decision D1).
//   * Resolves pose_source by reading the nearest physics-bound CookedRef
//     (link_index -> Link, else body_row -> Body, else Static) so the publisher
//     never re-walks SceneMap.
//   * world_xform is initialized to the bind-pose world transform (the composed
//     node chain from the root), to be overwritten by TransformSync each frame.
//   * Cameras / lights are gathered from CameraComponent / LightComponent.
//
// The Registry's entity->SceneNode backref (Registry::NodeOf) supplies the node
// chain, so BuildRenderWorld needs ONLY (Registry, SceneMap) per the M8 manifest
// signature -- no separate SceneGraph argument.
// ---------------------------------------------------------------------------
RenderWorld BuildRenderWorld(const scene::Registry& registry, const scene::SceneMap& map);

}  // namespace nuka::render
