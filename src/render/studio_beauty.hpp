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
// one deforming particle-surface instance) plus the per-instance link rebind it is
// refreshed from and the cinematic RasterOptions both the demo and the bridge draw.
struct StudioScene {
    RenderWorld world;
    RasterOptions options;  // studio lighting/sky/floor; the caller sets camera_* each frame.
    runtime::soft::SurfaceTopology surface_topology;  // particle-surface connectivity (empty => none).
    std::vector<uint32_t> link_of_instance;      // per link-posed instance -> link index.
    std::vector<math::Transform> visual_local;   // per link-posed instance -> physics->visual offset.
    uint32_t link_instance_count = 0u;           // [0, link_instance_count) follow a link pose.
    uint32_t surface_material_id = kNoId;         // the fabric material for the particle surface.
    uint32_t surface_mesh_id = kNoId;             // the (stable) deforming-surface mesh id.
    std::size_t surface_instance = ~std::size_t(0);  // the particle-surface RenderInstance.
};

// Build the studio scene from a cooked scene's ECS + SceneMap. `surface_topology`
// is the particle media's triangle connectivity (empty => robot + floor only).
// `width`/`height` seed options; the caller sets options.camera_* before each render.
StudioScene BuildStudioScene(const scene::Registry& registry,
                             const scene::SceneMap& map,
                             const runtime::soft::SurfaceTopology& surface_topology,
                             uint32_t width, uint32_t height);

// Refresh per-frame: each link-posed instance's world_xform = link_pose[link] o
// visual_local; the particle surface (when present) is rebuilt from `particle_pos`
// in place (the stable surface mesh, so the mesh table never grows across frames).
void PublishStudioScene(StudioScene& scene,
                        const std::vector<math::Transform>& link_pose,
                        const std::vector<math::Vec3>& particle_pos);

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
