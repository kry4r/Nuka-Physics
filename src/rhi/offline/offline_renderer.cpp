// ---------------------------------------------------------------------------
// nuka::rhi::offline::OfflineRenderer -- implementation. Orchestrates the
// offline render: RenderWorld -> tracer scene (adapter) -> RenderGraph passes
// driven on the selected PHI backend -> the profile's host output.
//
// HOST-ONLY / CUDA-FREE: drives the tracer through render::RtBackendI; the only
// device work happens inside that backend (the CUDA RT TU).
// ---------------------------------------------------------------------------

#include "rhi/offline/offline_renderer.hpp"

#include "render/rt_adapter.hpp"     // render::RenderWorldToTwoLevelScene
#include "rhi/offline/render_graph.hpp"
#include "rhi/offline/render_pass.hpp"
#include "rt/two_level_render.hpp"   // rt::TwoLevelScene

namespace nuka::rhi::offline {

// The RT backend is vended through PHI: CreateCudaRtBackend() binds to the global
// active architecture (the device + stream physics uses) -- the only impl today.
OfflineRenderer::OfflineRenderer()
    : rt_backend_(render::CreateCudaRtBackend()) {}

OfflineRenderer::~OfflineRenderer() = default;

rt::Framebuffer OfflineRenderer::Render(const render::RenderWorld& world,
                                        const rt::PinholeCamera& camera,
                                        const RenderProfile& profile) {
    rt::Framebuffer frame;
    if (rt_backend_ == nullptr) {
        return frame;
    }

    // Re-express the world as the tracer scene-desc and build the per-mesh BLAS.
    const rt::TwoLevelScene scene = render::RenderWorldToTwoLevelScene(world);
    render::RtSceneHandle* handle = rt_backend_->BuildScene(scene);
    if (handle == nullptr) {
        return frame;
    }

    PassContext ctx;
    ctx.backend = rt_backend_.get();
    ctx.handle = handle;
    ctx.scene = &scene;
    ctx.camera = &camera;
    ctx.beauty = &profile.beauty;
    ctx.frame = &frame;

    const RenderGraph graph = RenderGraph::FromProfile(profile);
    graph.Execute(ctx);

    rt_backend_->FreeScene(handle);
    return frame;
}

}  // namespace nuka::rhi::offline
