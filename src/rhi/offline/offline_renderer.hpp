#pragma once
// ---------------------------------------------------------------------------
// nuka::rhi::offline::OfflineRenderer -- the host orchestrator of the offline
// render pipeline. It is a CLIENT of PHI: it binds to the global active
// architecture (the device + stream physics uses), drives the self-written ray
// tracer on that backend, and returns the profile's output.
//
// The render pipeline is expressed as a RenderGraph of passes selected by a
// RenderProfile (ONE general path: replay vs sensor is config, not a fork). The
// renderer builds the tracer scene from a RenderWorld via the existing
// CUDA-free adapter (render::RenderWorldToTwoLevelScene) and runs the graph.
//
// HOST-ONLY / CUDA-FREE: names only phi::Backend* / render:: / rt:: opaque
// types -- the same zero-CUDA red-line src/render holds. The RT backend is
// obtained through PHI; CreateCudaRtBackend() is the only impl today (a
// backend-neutral vending entry arrives with the second backend).
// ---------------------------------------------------------------------------

#include "render/render_world.hpp"   // render::RenderWorld
#include "render/rt_backend.hpp"     // render::RtBackendI / RtSceneHandle
#include "rhi/offline/render_profile.hpp"
#include "rt/camera.hpp"             // rt::PinholeCamera
#include "rt/framebuffer.hpp"        // rt::Framebuffer

#include <memory>

namespace nuka::rhi::offline {

// The offline render facade. One instance per render context; holds the RT
// backend it vends. Render a RenderWorld through a profile.
class OfflineRenderer {
public:
    // Bind to the global active architecture and obtain the RT backend through PHI;
    // usable() is false when no device is available (phi "unavailable -> null").
    OfflineRenderer();
    ~OfflineRenderer();

    OfflineRenderer(const OfflineRenderer&) = delete;
    OfflineRenderer& operator=(const OfflineRenderer&) = delete;

    // True iff a live RT backend was obtained (a render will produce an image).
    bool usable() const { return rt_backend_ != nullptr; }

    // Render the world through a Beauty profile and return the host framebuffer
    // (exactly what the cinematic replay path produces today). Builds the tracer
    // scene from `world` via the CUDA-free adapter, builds the pass list from the
    // profile, drives the tracer on the selected backend, and returns the result.
    // An empty framebuffer is returned when the renderer is not usable.
    rt::Framebuffer Render(const render::RenderWorld& world,
                           const rt::PinholeCamera& camera,
                           const RenderProfile& profile);

private:
    std::unique_ptr<render::RtBackendI>   rt_backend_;
};

}  // namespace nuka::rhi::offline
