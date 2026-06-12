#pragma once
// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — INTERNAL shared definitions.
//
// Concrete opaque-handle layouts shared between cuda_backend.cu (Backend /
// Device / RegistryEntry / Plan / Event) and cuda_buffer.cu (BufferType /
// Buffer). NOT a public header — consumers use phi/backend.hpp + phi/buffer.hpp
// and the vtable wrappers there; they never see CUDA types or these structs.
//
// Each concrete struct begins with its `const <Iface>* iface;` so the public
// wrappers can dispatch (the ggml opaque-handle contract).
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "phi/backend.hpp"
#include "phi/buffer.hpp"

namespace nuka::phi {

// Forward: the backend owns the streams every buffer op runs on.
struct CudaBackend;

// The two buffer-type vtables (device + pinned host) live in cuda_buffer.cu.
extern const BufferTypeI kCudaDeviceBufferTypeI;
extern const BufferTypeI kCudaHostBufferTypeI;

// ---------------------------------------------------------------------------
// Concrete BufferType: a memory class bound to a backend (for the stream) and
// a flag selecting device vs pinned-host allocation.
// ---------------------------------------------------------------------------
struct CudaBufferType {
    const BufferTypeI* iface;   // MUST be first (opaque-handle contract)
    CudaBackend*       backend; // for the async stream; null => default stream
    bool               is_host;
};

// ---------------------------------------------------------------------------
// Concrete Buffer: a single allocation.
// ---------------------------------------------------------------------------
struct CudaBuffer {
    const BufferI* iface;   // MUST be first
    CudaBackend*   backend; // owning backend (stream source)
    void*          ptr;     // device (or pinned-host) base address
    size_t         bytes;
    bool           is_host;
};

// ---------------------------------------------------------------------------
// Concrete Backend: holds a main stream + a dedicated capture stream, the two
// buffer types, and the device id. The capture stream is used ONLY for
// CUDA-graph stream-capture in plan_create; everything else runs on `main`.
// ---------------------------------------------------------------------------
struct CudaBackend {
    const BackendI* iface;   // MUST be first
    int             device_id;
    cudaStream_t    main;
    cudaStream_t    capture;
    CudaBufferType  device_bt;
    CudaBufferType  host_bt;
};

// Stream accessor used by cuda_buffer.cu (a null backend => stream 0).
cudaStream_t CudaBackendMainStream(CudaBackend* b);

// Factory for a CudaBuffer (defined in cuda_buffer.cu). Returns nullptr on OOM.
Buffer* CudaBufferAlloc(CudaBackend* backend, size_t bytes, bool is_host);

// One-shot explicit registration of the CUDA RegistryEntry into the global
// registry. Called from registry.cpp::InitBestDevice() so the static
// nuka_phi2 lib's entry is never dropped by the linker (the self-registration
// trap de-risk). Idempotent.
void RegisterCudaBackendEntry();

} // namespace nuka::phi
