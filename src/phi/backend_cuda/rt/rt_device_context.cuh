#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- the render device context: the selected PHI device + main stream +
// AOV/scratch buffer type, resolved from a phi::Backend*. ALL RT device work
// (TLAS/BLAS build, uploads, kernel launch, transfers) shares this device +
// stream so the explicit stream sync is meaningful and transfers serialize
// behind the kernel. This is the single home for that resolution (no device-0 /
// default-stream hardcode anywhere on the RT path).
// ---------------------------------------------------------------------------

#include "phi/backend.hpp"                     // Backend / Device / InitBestDevice
#include "phi/buffer.hpp"                       // BufferType / BufferAlloc / ...
#include "phi/buffer_transfer_v2.hpp"           // UploadVectorV2
#include "phi/backend_cuda/cuda_internal.cuh"   // CudaBackend / CudaBackendMainStream

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace nuka::rt {

struct RtContext {
    int device_id = 0;
    cudaStream_t stream = nullptr;
    phi::BufferType* device_bt = nullptr;
};

// Resolve from a phi backend. A non-null backend uses its device id + main stream
// + device buffer type. A null backend falls back to the registry's first device,
// initialized once to a process-lifetime backend so that path also runs on a real
// device id + main stream. Returns a zeroed context (device_bt null) when no
// device is available.
inline RtContext ResolveRtContext(phi::Backend* backend) {
    if (backend == nullptr) {
        static phi::Backend* s_fallback = []() -> phi::Backend* {
            phi::Device* dev = phi::InitBestDevice();
            return dev != nullptr ? phi::DeviceInitBackend(dev, nullptr) : nullptr;
        }();
        backend = s_fallback;
    }
    RtContext ctx;
    if (backend == nullptr) {
        return ctx;
    }
    phi::CudaBackend* cb = reinterpret_cast<phi::CudaBackend*>(backend);
    ctx.device_id = cb->device_id;
    ctx.stream = phi::CudaBackendMainStream(cb);
    ctx.device_bt = phi::BackendDeviceBufferType(backend);
    return ctx;
}

// File-local RAII over the phi-v2 opaque Buffer*, allocated from the resolved
// context's buffer type so transfers + the kernel share one stream. Move-only;
// mirrors the legacy phi::Buffer surface the render bodies use.
class OwnedBuffer {
public:
    OwnedBuffer() = default;
    OwnedBuffer(phi::BufferType* bt, std::size_t bytes) {
        buf_ = phi::BufferAlloc(bt, bytes);
    }
    explicit OwnedBuffer(phi::Buffer* buf) : buf_(buf) {}
    ~OwnedBuffer() { if (buf_ != nullptr) phi::BufferFree(buf_); }
    OwnedBuffer(OwnedBuffer&& o) noexcept : buf_(o.buf_) { o.buf_ = nullptr; }
    OwnedBuffer& operator=(OwnedBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) phi::BufferFree(buf_);
            buf_ = o.buf_; o.buf_ = nullptr;
        }
        return *this;
    }
    OwnedBuffer(const OwnedBuffer&) = delete;
    OwnedBuffer& operator=(const OwnedBuffer&) = delete;
    void* Data() const { return buf_ != nullptr ? phi::BufferBase(buf_) : nullptr; }
    void CopyFromHost(const void* src, std::size_t bytes) const {
        if (buf_ != nullptr && bytes > 0u) phi::BufferUpload(buf_, src, 0, bytes);
    }
    void CopyToHost(void* dst, std::size_t bytes) const {
        if (buf_ != nullptr && bytes > 0u) phi::BufferDownload(buf_, dst, 0, bytes);
    }
private:
    phi::Buffer* buf_ = nullptr;
};

template <typename T>
inline OwnedBuffer UploadOwned(phi::BufferType* bt, const std::vector<T>& values) {
    return OwnedBuffer(phi::UploadVectorV2(bt, values));
}

}  // namespace nuka::rt
