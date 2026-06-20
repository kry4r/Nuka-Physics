#pragma once
// ---------------------------------------------------------------------------
// nuka::rhi::offline::RenderProfile -- the DATA that selects what the offline
// renderer produces. The render pipeline is a sequence of passes (render_graph)
// chosen by this profile, NOT a code fork: replay and sensor are the same
// geometry/traversal core under different profiles (ONE general path).
//
// The cinematic controls (sun / sky / fog / MSAA spp) are NOT redeclared here --
// they live on the tracer's rt::BeautyOptions and are carried through by value,
// so there is a single source of truth for the shading parameters.
//
// HOST-ONLY / CUDA-FREE: this header names only rt:: PODs (no phi backend, no
// CUDA token), so it compiles into the host RHI lib.
// ---------------------------------------------------------------------------

#include "rt/two_level_render.hpp"  // rt::BeautyOptions (cinematic sun/sky/fog/spp)

#include <cstdint>

namespace nuka::rhi::offline {

// What the renderer produces. Beauty is the cinematic replay image (stochastic
// MSAA + soft shadows + AO + one-bounce GI + sky/fog). Sensor is the in-loop
// batched AOV path; it is a DECLARED seam filled by the batched sensor work --
// SelectFor rejects it today so an unimplemented kind never silently no-ops.
enum class RenderOutput : uint8_t {
    Beauty,  // cinematic replay RGB image -> host rt::Framebuffer
    Sensor,  // batched per-env AOV tensor (declared; the sensor path fills it)
};

// How deterministic the output must be. Golden = exact, reproducible bytes
// (no temporal reuse): the regression / golden / diff-sim path. The cinematic
// beauty trace is exact given its fixed rt::BeautyOptions::seed, so it is Golden.
enum class Determinism : uint8_t {
    Golden,  // bit-reproducible run-to-run within a backend/device (D1)
};

// The render configuration. `image_width`/`image_height` size the camera's
// framebuffer; `beauty` carries the cinematic shading controls (sun/sky/fog/spp)
// straight to the tracer. A Beauty profile runs the beauty shade pass; the
// Sensor seam is declared for the batched AOV path.
struct RenderProfile {
    RenderOutput output      = RenderOutput::Beauty;
    Determinism  determinism = Determinism::Golden;
    uint32_t     image_width  = 0u;
    uint32_t     image_height = 0u;

    // Cinematic controls reused verbatim from the tracer (single source of truth).
    rt::BeautyOptions beauty{};

    // A Beauty profile at default determinism producing an image of the given size.
    static RenderProfile BeautyReplay(uint32_t width, uint32_t height) {
        RenderProfile p;
        p.output = RenderOutput::Beauty;
        p.determinism = Determinism::Golden;
        p.image_width = width;
        p.image_height = height;
        return p;
    }

    // A Sensor profile producing the batched per-env AOV tensor at the given
    // per-sensor resolution. The cheap sensor shade (1spp, one hard shadow ray, no
    // AO, no GI) is the FP32 path the batched trace runs; the controls are set here
    // as the single source of truth for what that profile means.
    static RenderProfile Sensor(uint32_t width, uint32_t height) {
        RenderProfile p;
        p.output = RenderOutput::Sensor;
        p.determinism = Determinism::Golden;
        p.image_width = width;
        p.image_height = height;
        p.beauty.samples = 1u;       // 1spp
        p.beauty.shadow_rays = 1u;   // one hard shadow ray
        p.beauty.ao_samples = 0u;    // no ambient occlusion
        p.beauty.gi_bounces = 0u;    // no indirect bounce
        return p;
    }
};

}  // namespace nuka::rhi::offline
