// ---------------------------------------------------------------------------
// nuka::runtime::app -- the frame-loop systems (M8 manifest #9).
//
// SimSystem (nk::World step) and RenderSystem (offscreen raster). InputSystem +
// TransformSyncSystem are header-only (trivial bodies in systems.hpp).
//
// RenderSystem's draw call is the ONLY place this library touches nuka_render's
// VulkanRasterRenderer::Render symbol; it is gated behind NK_BUILD_VULKAN_VALIDATION
// (Decision D5) so a Vulkan-less build still links -- Run is then a no-op and the
// caller (Simulation) never attaches a renderer in that configuration anyway.
//
// HOST-ONLY / zero-CUDA-token (src/runtime/app/** lint red-line): the physics
// step goes through nk::World (whose device work is behind the phi v2 vtable);
// no kernel launch, no cuda-runtime call, no cuda_runtime / phi backend_cuda
// include appears here.
// ---------------------------------------------------------------------------

#include "runtime/app/systems.hpp"

#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"

namespace nuka::runtime::app {

bool SimSystem::Run(nk::World& world, bool planned) {
    if (planned) {
        return world.StepPlanned() == phi::Status::Ok;
    }
    return world.Step().AllOk();
}

bool RenderSystem::Run(render::VulkanRasterRenderer* renderer,
                       const render::RenderWorld& world,
                       const render::RasterOptions& options,
                       render::VulkanOffscreenReport* out_report) {
    if (renderer == nullptr || out_report == nullptr) {
        return false;
    }
#ifdef NK_BUILD_VULKAN_VALIDATION
    *out_report = renderer->Render(world, options);
    return true;
#else
    // Built without the Vulkan validation toolchain: the renderer symbol is not
    // linked into this library. RenderSystem is inert; the render-off frame loop
    // (Simulation with render disabled) never reaches here, and a render-on build
    // turns this on by defining NK_BUILD_VULKAN_VALIDATION.
    (void)renderer;
    (void)world;
    (void)options;
    return false;
#endif
}

}  // namespace nuka::runtime::app
