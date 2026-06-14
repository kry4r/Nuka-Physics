// ---------------------------------------------------------------------------
// nuka::render::CudaRtBackend -- the CONCRETE CUDA implementation of the
// backend-agnostic render::RtBackendI (M11 §3.10 RT row, RT-1/RT-3).
//
// This is the ONLY place outside the rt device TUs that reaches the self-written
// two-level tracer's host entry points (rt::BuildTwoLevelScene / rt::RenderFrame).
// It WRAPS them -- it does NOT reimplement the tracer (R5: kernel bodies are
// byte-identical through the M11 home; this TU adds NO arithmetic). The device
// trace (RenderFrame) remains the single source of truth, so every RT D1 golden
// is preserved unchanged; the BufferI-output path is a post-trace upload of the
// already-D1 host AOVs into the caller's device buffers.
//
// Lives in src/phi/backend_cuda/rt/ (the CUDA backend layer). It is HOST C++ (no
// kernel launch here) but is homed in backend_cuda because it is the CUDA-backed
// concrete; the redline-protected engine layer (src/render) only sees the
// interface (render/rt_backend.hpp). Compiled into nuka_phi2_rt (the dedicated
// --fmad=false RT lib), so linking that lib provides the STRONG
// CreateCudaRtBackend definition that wins over the render-lib weak fallback.
// ---------------------------------------------------------------------------

#include "render/rt_backend.hpp"

#include "phi/backend.hpp"   // InitBestDevice / DeviceBufferType
#include "phi/buffer.hpp"    // BufferAlloc / BufferUpload / BufferFree
#include "rt/two_level_render.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace nuka::render {

// The opaque scene handle is just the device two-level scene (per-mesh BLAS +
// uploaded local prim buffers). render::RtSceneHandle is forward-declared in the
// interface; here it IS rt::TwoLevelSceneDevice (move-only, owns the BLAS).
struct RtSceneHandle {
    rt::TwoLevelSceneDevice device;
};

namespace {

// Upload one host AOV channel into a caller-provided device Buffer* (skip if the
// caller passed nullptr for that channel or the host channel is empty).
template <class T>
void UploadAov(phi::Buffer* dst, const std::vector<T>& host) {
    if (dst == nullptr || host.empty()) {
        return;
    }
    phi::BufferUpload(dst, host.data(), 0, host.size() * sizeof(T));
}

// The CUDA RT backend: holds the phi device (for the AOV BufferType) and wraps
// the host two-level tracer entry points.
class CudaRtBackend final : public RtBackendI {
public:
    explicit CudaRtBackend(phi::Device* device) : device_(device) {}

    phi::BufferType* AovBufferType() override {
        return phi::DeviceBufferType(device_);
    }

    RtSceneHandle* BuildScene(const rt::TwoLevelScene& scene) override {
        auto handle = std::make_unique<RtSceneHandle>();
        handle->device = rt::BuildTwoLevelScene(scene);
        return handle.release();
    }

    void Trace(RtSceneHandle* handle,
               const rt::TwoLevelScene& scene,
               const rt::PinholeCamera& camera,
               const RtAovBuffers& aov) override {
        // The device trace (RenderFrame) is the single source of truth and is
        // byte-identical to the rt D1 goldens. Run it, then upload the resulting
        // (already-D1) host AOVs into the caller's device buffers. This keeps the
        // BufferI output contract (OD-12) without perturbing the tracer.
        const rt::Framebuffer frame =
            rt::RenderFrame(handle->device, scene, camera);
        UploadAov(aov.color, frame.color);
        UploadAov(aov.depth, frame.depth);
        UploadAov(aov.normal, frame.normal);
        UploadAov(aov.albedo, frame.albedo);
        UploadAov(aov.uv, frame.uv);
        UploadAov(aov.prim, frame.prim);
    }

    rt::Framebuffer TraceToHost(RtSceneHandle* handle,
                                const rt::TwoLevelScene& scene,
                                const rt::PinholeCamera& camera) override {
        return rt::RenderFrame(handle->device, scene, camera);
    }

    void FreeScene(RtSceneHandle* handle) override { delete handle; }

private:
    phi::Device* device_ = nullptr;
};

}  // namespace

// STRONG definitions (win over the render-lib weak fallback when nuka_phi2_rt is
// linked). Returns nullptr when no device initializes (the phi "unavailable ->
// null" contract).
std::unique_ptr<RtBackendI> CreateCudaRtBackend() {
    phi::Device* device = phi::InitBestDevice();
    if (device == nullptr) {
        return nullptr;
    }
    return std::make_unique<CudaRtBackend>(device);
}

bool RtBackendAvailable() { return phi::InitBestDevice() != nullptr; }

}  // namespace nuka::render
