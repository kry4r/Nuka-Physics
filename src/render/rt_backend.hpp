#pragma once
// ---------------------------------------------------------------------------
// nuka::render::RtBackendI -- the backend-agnostic interface to the self-written
// (NO OptiX / NO closed SDK) two-level (TLAS/BLAS) ray tracer (M11 §3.10 RT row).
//
// §3.10 splits the RT subsystem into an ENGINE-LAYER interface (this header, in
// src/render -- ZERO CUDA token, lint red-line) and a CUDA-BACKEND impl
// (src/phi/backend_cuda/rt/rt_backend_cuda.cpp, the only place the tracer's
// device entry points are reached). The concrete backend WRAPS the existing
// rt::BuildTwoLevelScene / rt::RenderFrame -- it does NOT reimplement the tracer.
//
// CONTRACT (RT-1 / OD-12):
//   * BuildScene(scene)  -- build the per-mesh BLAS + upload local prim buffers
//                           ONCE (rigid -> never refit); returns an opaque,
//                           backend-owned scene handle (RtSceneHandle).
//   * Trace(handle, scene, camera, AovBuffers) -- rebuild the per-frame TLAS over
//                           the CURRENT instance transforms, dispatch the nested-
//                           traversal kernel, and WRITE the 6 AOVs into the
//                           CALLER-PROVIDED phi v2 Buffer* outputs (no host copy).
//   * TraceToHost(...)   -- thin convenience that allocates+downloads into a host
//                           rt::Framebuffer (for tests / host-download recorders).
//   * FreeScene(handle)  -- release the backend scene handle.
//
// The 6 AOVs are the rt::Framebuffer channels: color (3*float/px), depth
// (1*float/px), normal (3*float/px), albedo (3*float/px), uv (2*float/px),
// prim (1*uint32/px). The Buffer* outputs are phi v2 device buffers the caller
// allocates from the SAME BufferType the backend uses (RtBackendI::AovBufferType()
// surfaces it so render-layer callers never name a CUDA type).
//
// FACTORY / FALLBACK (OD-12): CreateCudaRtBackend() returns a heap-owned backend
// when a device is available, else nullptr (the same "unavailable -> null"
// contract phi's registry uses). It is DEFINED in the CUDA backend TU
// (src/phi/backend_cuda/rt/rt_backend_cuda.cpp, part of nuka_phi2_rt); a consumer
// that wants a live RT backend links nuka_phi2_rt. The redline-protected render
// lib (src/render) only USES the interface (RtBackendI) + the adapter and never
// references the factory, so nuka_render links standalone with no CUDA dep.
// RtBackendAvailable() is the cheap probe.
//
// ZERO CUDA: this header names ONLY phi v2 opaque Buffer*/BufferType* (NOT CUDA
// tokens) and the CUDA-free rt scene-desc PODs. It compiles under g++.
// ---------------------------------------------------------------------------

#include "phi/buffer.hpp"          // nuka::phi::Buffer / BufferType (opaque, CUDA-free)
#include "rt/camera.hpp"           // rt::PinholeCamera (CUDA-free POD)
#include "rt/framebuffer.hpp"      // rt::Framebuffer (host download convenience)
#include "rt/two_level_render.hpp" // rt::TwoLevelScene (CUDA-free scene-desc POD)

#include <cstdint>
#include <memory>

namespace nuka::render {

// Opaque handle to a backend-built two-level scene (per-mesh BLAS + uploaded
// local prim buffers). Owned by the RtBackendI that produced it; freed via
// RtBackendI::FreeScene. The concrete type lives in the CUDA backend TU.
struct RtSceneHandle;

// Caller-provided phi v2 device buffers, one per AOV, that Trace writes into.
// Each must be sized for the camera's width*height (see byte sizes below). The
// caller owns/frees them. nullptr for an AOV means "do not write this channel"
// (the backend skips that download/copy).
//
//   color  : width*height*3 * sizeof(float)
//   depth  : width*height   * sizeof(float)
//   normal : width*height*3 * sizeof(float)
//   albedo : width*height*3 * sizeof(float)
//   uv     : width*height*2 * sizeof(float)
//   prim   : width*height   * sizeof(uint32_t)
struct RtAovBuffers {
    phi::Buffer* color  = nullptr;
    phi::Buffer* depth  = nullptr;
    phi::Buffer* normal = nullptr;
    phi::Buffer* albedo = nullptr;
    phi::Buffer* uv     = nullptr;
    phi::Buffer* prim   = nullptr;
};

// The backend-agnostic RT interface. One instance per render context; holds the
// device/backend. Build a scene once, trace it many frames, free it.
class RtBackendI {
public:
    virtual ~RtBackendI() = default;

    // The phi v2 BufferType the AOV outputs must be allocated from (the backend's
    // device buffer type). render-layer callers BufferAlloc against this without
    // naming any CUDA type.
    virtual phi::BufferType* AovBufferType() = 0;

    // Build the per-mesh BLAS + upload local prim buffers ONCE. Returns an opaque
    // handle owned by this backend (free with FreeScene). nullptr on failure.
    virtual RtSceneHandle* BuildScene(const rt::TwoLevelScene& scene) = 0;

    // Rebuild the per-frame TLAS over scene.instances' CURRENT transforms,
    // dispatch the nested-traversal kernel, and write the requested AOVs into the
    // caller-provided device buffers. `handle` must have been built (BuildScene)
    // from a scene with the SAME meshes; transforms / materials / light are read
    // fresh from `scene` each call (moving an instance tracks).
    virtual void Trace(RtSceneHandle* handle,
                       const rt::TwoLevelScene& scene,
                       const rt::PinholeCamera& camera,
                       const RtAovBuffers& aov) = 0;

    // Convenience: Trace into freshly-downloaded host AOVs (rt::Framebuffer).
    // Allocates the AOV buffers, traces, downloads, frees -- for tests and the
    // host-download recorder path. Same pixels as Trace (the device path is the
    // single source of truth).
    virtual rt::Framebuffer TraceToHost(RtSceneHandle* handle,
                                        const rt::TwoLevelScene& scene,
                                        const rt::PinholeCamera& camera) = 0;

    // Release a scene handle built by this backend.
    virtual void FreeScene(RtSceneHandle* handle) = 0;
};

// Factory: a heap-owned CUDA RT backend when the CUDA RT backend TU is linked AND
// a device initializes, else nullptr. The weak fallback (no CUDA backend linked)
// returns nullptr so the render lib links standalone; the strong definition in
// src/phi/backend_cuda/rt/rt_backend_cuda.cpp wins when present.
std::unique_ptr<RtBackendI> CreateCudaRtBackend();

// Cheap probe: true iff CreateCudaRtBackend() would return a usable backend.
bool RtBackendAvailable();

}  // namespace nuka::render
