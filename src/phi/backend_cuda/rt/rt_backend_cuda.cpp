// ---------------------------------------------------------------------------
// nuka::render::CudaRtBackend -- the CONCRETE CUDA implementation of the
// backend-agnostic render::RtBackendI (M11 §3.10 RT row, RT-1/RT-3).
//
// This is the ONLY place outside the rt device TUs that reaches the self-written
// two-level tracer's host entry points (rt::BuildTwoLevelScene / rt::RenderFrame).
// It WRAPS them -- it does NOT reimplement the tracer (kernel bodies are
// byte-identical; this TU adds NO arithmetic). The device trace is the single
// source of truth; the BufferI-output path is the same kernel writing the caller's
// device buffers in place (no host round-trip), byte-identical to the D1 goldens.
//
// Lives in src/phi/backend_cuda/rt/ (the CUDA backend layer). It is HOST C++ (no
// kernel launch here) but is homed in backend_cuda because it is the CUDA-backed
// concrete; the redline-protected engine layer (src/render) only sees the
// interface (render/rt_backend.hpp). Compiled into nuka_phi2_rt (the dedicated
// --fmad=false RT lib), so linking that lib provides the STRONG
// CreateCudaRtBackend definition that wins over the render-lib weak fallback.
// ---------------------------------------------------------------------------

#include "render/rt_backend.hpp"

#include "phi/backend.hpp"   // ActiveBackend / InitBestDevice / BackendDeviceBufferType
#include "phi/buffer.hpp"    // BufferAlloc / BufferFree
#include "rt/two_level_render.hpp"

#include <memory>

namespace nuka::render {

// The opaque scene handle is just the device two-level scene (per-mesh BLAS +
// uploaded local prim buffers). render::RtSceneHandle is forward-declared in the
// interface; here it IS rt::TwoLevelSceneDevice (move-only, owns the BLAS).
struct RtSceneHandle {
    rt::TwoLevelSceneDevice device;
};

namespace {

// Map the caller's RtAovBuffers (one phi::Buffer* per AOV) to the device-resident
// target the tracer writes in place (null channels are skipped by the kernel).
rt::RtDeviceAovs ToDeviceAovs(const RtAovBuffers& aov) {
    rt::RtDeviceAovs out;
    out.color = aov.color;
    out.depth = aov.depth;
    out.normal = aov.normal;
    out.albedo = aov.albedo;
    out.uv = aov.uv;
    out.prim = aov.prim;
    return out;
}

// The CUDA RT backend: holds the selected phi backend (its device + main stream +
// AOV buffer type) and wraps the two-level tracer, which runs on that device/stream.
class CudaRtBackend final : public RtBackendI {
public:
    explicit CudaRtBackend(phi::Backend* backend) : backend_(backend) {}

    phi::BufferType* AovBufferType() override {
        return phi::BackendDeviceBufferType(backend_);
    }

    RtSceneHandle* BuildScene(const rt::TwoLevelScene& scene) override {
        auto handle = std::make_unique<RtSceneHandle>();
        // The persistent texture-env cache survives FreeScene, so a per-frame
        // rebuild over unchanged textures reuses the device residency (no re-upload).
        handle->device = rt::BuildTwoLevelScene(scene, backend_, &tex_env_cache_);
        return handle.release();
    }

    void Trace(RtSceneHandle* handle,
               const rt::TwoLevelScene& scene,
               const rt::PinholeCamera& camera,
               const RtAovBuffers& aov) override {
        // The kernel writes the caller's device AOV buffers in place on the selected
        // stream (no host round-trip); identical pixels to TraceToHost.
        rt::RenderFrameToAovs(handle->device, scene, camera, ToDeviceAovs(aov), backend_);
    }

    rt::Framebuffer TraceToHost(RtSceneHandle* handle,
                                const rt::TwoLevelScene& scene,
                                const rt::PinholeCamera& camera) override {
        return rt::RenderFrame(handle->device, scene, camera, backend_);
    }

    rt::Framebuffer TraceBeautyToHost(RtSceneHandle* handle,
                                      const rt::TwoLevelScene& scene,
                                      const rt::PinholeCamera& camera,
                                      const rt::BeautyOptions& opt) override {
        return rt::RenderBeauty(handle->device, scene, camera, opt, backend_);
    }

    void FreeScene(RtSceneHandle* handle) override { delete handle; }

private:
    phi::Backend* backend_ = nullptr;
    // Device-resident textures/env shared across every BuildScene on this backend.
    rt::TextureEnvCache tex_env_cache_;
};

}  // namespace

// Bind to the global active architecture so render shares the device + stream
// physics uses. Null when no device is available (phi "unavailable -> null").
std::unique_ptr<RtBackendI> CreateCudaRtBackend() {
    phi::Backend* backend = phi::ActiveBackend();
    if (backend == nullptr) {
        return nullptr;
    }
    return std::make_unique<CudaRtBackend>(backend);
}

bool RtBackendAvailable() { return phi::InitBestDevice() != nullptr; }

}  // namespace nuka::render
