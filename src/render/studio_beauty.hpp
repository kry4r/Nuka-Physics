#pragma once
// ---------------------------------------------------------------------------
// nuka::render -- the shared studio render scene for the live robot+particle look.
//
// The ONE render setup the go2_cloth_drape demo AND the live-world render bridge
// (the C-ABI render-beauty verb) draw: a RenderWorld of the robot's FK-posed link
// visuals + a deforming particle-surface instance + a studio ground floor, with a
// premium material palette and cinematic RasterOptions. Built ONCE per world, then
// refreshed each frame from the live link poses + particle positions. The media is
// data (which links exist, which surface triangles) -- there is no per-scene fork.
//
// HOST-ONLY, ZERO CUDA: the offline path-tracer is reached only through the
// render::RtBackendI interface (CreateCudaRtBackend), so this compiles into
// nuka_render with no CUDA token; a consumer that wants a live RT backend links
// nuka_phi2_rt for the strong factory (the weak fallback reports no backend).
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"  // RasterOptions, VulkanOffscreenReport
#include "render/render_world.hpp"
#include "runtime/soft/particle_surface.hpp"  // SurfaceTopology

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace nuka::scene {
class Registry;
class SceneMap;
}  // namespace nuka::scene

namespace nuka::render {

// The studio render scene: a RenderWorld (robot link instances + a ground floor +
// one deforming particle-surface instance PER cooked medium) plus the per-instance
// link rebind it is refreshed from and the cinematic RasterOptions both the demo and
// the bridge draw. The media is data (which surfaces, which triangles), not a fork.
struct StudioScene {
    // One cooked medium's deforming render surface: its triangle topology + render
    // skin, the stable mesh it is rebuilt into each frame, and its RenderInstance.
    struct DeformingSurface {
        runtime::soft::SurfaceTopology topology;     // connectivity + inflation/relax skin.
        uint32_t    material_id = kNoId;             // the surface's render material.
        uint32_t    mesh_id = kNoId;                 // the (stable) deforming-surface mesh id.
        std::size_t instance = ~std::size_t(0);      // its RenderInstance index.
    };

    // A surface-less particle medium's render skin: every particle in [first,
    // first+count) (count 0 = the whole field) bakes into ONE instanced-sphere
    // mesh rebuilt each frame (granular grains; also an MPM-fluid debug view).
    struct ParticleSkin {
        float       radius = 0.0f;
        uint32_t    material_id = kNoId;         // resolved render material slot.
        uint32_t    first = 0u, count = 0u;
        uint32_t    mesh_id = kNoId;
        std::size_t instance = ~std::size_t(0);
        // Grain look: analytic spheres vs octahedra, plus deterministic per-grain
        // radius/albedo jitter (all 0/false => today's uniform octahedra).
        bool        round = false;
        float       radius_jitter = 0.0f;
        float       tint_jitter = 0.0f;
    };

    RenderWorld world;
    RasterOptions options;  // studio lighting/sky/floor; the caller sets camera_* each frame.
    std::vector<DeformingSurface> surfaces;      // one per cooked particle medium (empty => none).
    std::vector<ParticleSkin> particle_skins;    // instanced-sphere media (empty => none).
    std::vector<uint32_t> link_of_instance;      // per link-posed instance -> link index.
    std::vector<math::Transform> visual_local;   // per link-posed instance -> physics->visual offset.
    uint32_t link_instance_count = 0u;           // [0, link_instance_count) follow a link pose.

    // Each build-time instance's AUTHORED scene material id (kNoId = none). The
    // studio palette overrides these by default; UseAuthoredSceneMaterials rebinds.
    std::vector<uint32_t> authored_material_of_instance;
};

// Build the studio scene from a cooked scene's ECS + SceneMap. `surface_topologies`
// is each particle medium's triangle connectivity (empty => robot + floor only); each
// renders as its own deforming instance. `width`/`height` seed options; the caller
// sets options.camera_* before each render.
StudioScene BuildStudioScene(const scene::Registry& registry,
                             const scene::SceneMap& map,
                             const std::vector<runtime::soft::SurfaceTopology>& surface_topologies,
                             uint32_t width, uint32_t height);

// Single-surface convenience: wraps one (possibly empty) topology into the list form.
StudioScene BuildStudioScene(const scene::Registry& registry,
                             const scene::SceneMap& map,
                             const runtime::soft::SurfaceTopology& surface_topology,
                             uint32_t width, uint32_t height);

// Bind medium `surface_index`'s authored scene render material (a Registry material
// id) over the studio default. Call after BuildStudioScene, before the first Publish.
void SetStudioSurfaceMaterial(StudioScene& scene, const scene::Registry& registry,
                              std::size_t surface_index, uint32_t scene_material_id);

// Rebind every instance that carries an AUTHORED scene material back to it (the
// studio hero palette stays only where the scene authored nothing). Call after
// BuildStudioScene when the scene opts in (EnvironmentRecord.use_scene_materials).
void UseAuthoredSceneMaterials(StudioScene& scene);

// Register a per-particle instanced-sphere skin (a surface-less particle medium).
// `scene_material_id` kNoId/invalid falls back to a neutral granular material;
// `count` 0 covers the whole particle field. `round` swaps the octahedron grains for
// analytic spheres; `radius_jitter`/`tint_jitter` add deterministic per-grain scatter
// (all default => today's uniform octahedra, byte-identical).
void AddStudioParticleSkin(StudioScene& scene, const scene::Registry& registry,
                           uint32_t scene_material_id, float radius,
                           uint32_t first, uint32_t count,
                           bool round = false, float radius_jitter = 0.0f,
                           float tint_jitter = 0.0f);

// Refresh per-frame: each link-posed instance's world_xform = link_pose[link] o
// visual_local (a free-BODY instance poses from body_pose[row] instead); every
// particle surface is rebuilt from `particle_pos` in place (the stable per-surface
// mesh, so the mesh table never grows across frames); every particle skin re-bakes
// its instanced-sphere mesh from the live positions.
void PublishStudioScene(StudioScene& scene,
                        const std::vector<math::Transform>& link_pose,
                        const std::vector<math::Vec3>& particle_pos,
                        const std::vector<math::Transform>& body_pose = {});

// The offline CUDA path-tracer over a StudioScene's RenderWorld. Rebuilds the BLAS
// each frame (the surface deforms) and traces to a host RGBA8 report. ok() is false
// when no CUDA RT backend is linked/available (the render lib's weak fallback).
class StudioRtRenderer {
public:
    StudioRtRenderer();
    ~StudioRtRenderer();
    StudioRtRenderer(const StudioRtRenderer&) = delete;
    StudioRtRenderer& operator=(const StudioRtRenderer&) = delete;

    bool ok() const;
    void SetBeauty(bool on, uint32_t samples);  // soft shadows/AO/GI vs flat RT.
    VulkanOffscreenReport Render(const RenderWorld& world, const RasterOptions& options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nuka::render
