#pragma once
// ---------------------------------------------------------------------------
// nuka::rhi::offline -- the render-pass abstraction. A render is an ordered list
// of passes (render_graph) selected by the RenderProfile; each pass is one stage
// over the shared geometry/traversal core. This is the render analogue of the
// physics op table -- a SEPARATE pass registry, never folded into the physics
// NkOp enum.
//
// One real pass is implemented today: the beauty geometry+shade pass that drives
// the self-written tracer through render::RtBackendI. The interface is declared
// so later passes (lidar ray-gen, motion-vector, denoise) slot in as additional
// RenderPassKind values with their own Execute body -- no empty stub passes.
//
// HOST-ONLY / CUDA-FREE: names only render::RtBackendI + rt:: PODs (opaque),
// never a CUDA token.
// ---------------------------------------------------------------------------

#include "render/rt_backend.hpp"     // render::RtBackendI / RtSceneHandle / RtAovBuffers
#include "rt/camera.hpp"             // rt::PinholeCamera
#include "rt/framebuffer.hpp"        // rt::Framebuffer (host output)
#include "rt/two_level_render.hpp"   // rt::TwoLevelScene / rt::BeautyOptions

#include <cstdint>

namespace nuka::rhi::offline {

// The kinds of pass the pipeline knows. A profile maps to an ordered subset of
// these (render_graph). Distinct from the physics op enum on purpose.
enum class RenderPassKind : uint8_t {
    BeautyShade,  // cinematic shade over the closest-hit geometry -> color image
};

// Everything a pass reads/writes for one frame: the backend driving the tracer,
// the per-mesh scene handle (BLAS built once), the live scene (current
// transforms/materials/light), the camera, and the host framebuffer the pass
// writes its result into. A pass mutates only `frame`.
struct PassContext {
    render::RtBackendI*       backend = nullptr;
    render::RtSceneHandle*    handle  = nullptr;
    const rt::TwoLevelScene*  scene   = nullptr;
    const rt::PinholeCamera*  camera  = nullptr;
    const rt::BeautyOptions*  beauty  = nullptr;  // cinematic controls for shade passes
    rt::Framebuffer*          frame   = nullptr;  // host output written by the pass
};

// One render-pipeline stage. Implementations name their RenderPassKind and run
// one dispatch (or a small fixed sequence) over the shared core via ctx.backend.
class RenderPass {
public:
    virtual ~RenderPass() = default;
    virtual RenderPassKind Kind() const = 0;
    virtual void Execute(const PassContext& ctx) = 0;
};

// The beauty geometry+shade pass: drives the tracer's stochastic cinematic trace
// (MSAA + soft shadows + AO + one-bounce GI + sky/fog) into the host framebuffer.
// The pixel writes are the tracer's; this pass adds no arithmetic.
class BeautyShadePass final : public RenderPass {
public:
    RenderPassKind Kind() const override { return RenderPassKind::BeautyShade; }

    void Execute(const PassContext& ctx) override {
        *ctx.frame = ctx.backend->TraceBeautyToHost(ctx.handle, *ctx.scene,
                                                    *ctx.camera, *ctx.beauty);
    }
};

}  // namespace nuka::rhi::offline
