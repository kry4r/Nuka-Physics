// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — BufferType / Buffer implementations.
//
// THE single home of allocation in the v2 layer (the legacy
// phi/backend_cuda/buffer.cu serves nuka_phi and is untouched). Two buffer
// types: device (cudaMalloc) and pinned host (cudaMallocHost). upload /
// download / memset are async on the owning backend's main stream; copy_from is
// a device-to-device async copy. Alignment is 256 (CUDA's standard min
// allocation alignment), matching the contract.
//
// Allocation here is legitimate (this is the allocator); the lint
// hot_path_cuda_malloc rule is scoped to phi/backend_cuda/ops/** specifically so
// it does NOT cover this file.
// ---------------------------------------------------------------------------

#include "phi/backend_cuda/cuda_internal.cuh"

#include <cstdint>

namespace nuka::phi {

namespace {

CudaBuffer* AsBuffer(Buffer* b) { return reinterpret_cast<CudaBuffer*>(b); }
CudaBufferType* AsBufferType(BufferType* t) { return reinterpret_cast<CudaBufferType*>(t); }

// --- Buffer vtable -------------------------------------------------------

void BufferFreeImpl(Buffer* b) {
    CudaBuffer* cb = AsBuffer(b);
    if (cb == nullptr) {
        return;
    }
    if (cb->ptr != nullptr) {
        if (cb->is_host) {
            (void)cudaFreeHost(cb->ptr);
        } else {
            (void)cudaFree(cb->ptr);
        }
    }
    delete cb;
}

void* BufferBaseImpl(Buffer* b) { return AsBuffer(b)->ptr; }

void BufferUploadImpl(Buffer* b, const void* src, size_t off, size_t n) {
    CudaBuffer* cb = AsBuffer(b);
    cudaStream_t s = CudaBackendMainStream(cb->backend);
    (void)cudaMemcpyAsync(static_cast<uint8_t*>(cb->ptr) + off, src, n,
                          cudaMemcpyHostToDevice, s);
}

void BufferDownloadImpl(Buffer* b, void* dst, size_t off, size_t n) {
    CudaBuffer* cb = AsBuffer(b);
    cudaStream_t s = CudaBackendMainStream(cb->backend);
    (void)cudaMemcpyAsync(dst, static_cast<const uint8_t*>(cb->ptr) + off, n,
                          cudaMemcpyDeviceToHost, s);
    // download is observable host-side; make it synchronous from the caller's
    // view by syncing the stream the copy was issued on.
    (void)cudaStreamSynchronize(s);
}

void BufferMemsetImpl(Buffer* b, uint8_t v, size_t off, size_t n) {
    CudaBuffer* cb = AsBuffer(b);
    cudaStream_t s = CudaBackendMainStream(cb->backend);
    (void)cudaMemsetAsync(static_cast<uint8_t*>(cb->ptr) + off, v, n, s);
}

void BufferCopyFromImpl(Buffer* dst, Buffer* src, size_t doff, size_t soff, size_t n) {
    CudaBuffer* cd = AsBuffer(dst);
    CudaBuffer* cs = AsBuffer(src);
    cudaStream_t s = CudaBackendMainStream(cd->backend);
    (void)cudaMemcpyAsync(static_cast<uint8_t*>(cd->ptr) + doff,
                          static_cast<const uint8_t*>(cs->ptr) + soff, n,
                          cudaMemcpyDeviceToDevice, s);
}

const BufferI kCudaBufferI = {
    /*free      =*/ &BufferFreeImpl,
    /*base      =*/ &BufferBaseImpl,
    /*upload    =*/ &BufferUploadImpl,
    /*download  =*/ &BufferDownloadImpl,
    /*memset    =*/ &BufferMemsetImpl,
    /*copy_from =*/ &BufferCopyFromImpl,
};

// --- BufferType vtable ---------------------------------------------------

const char* DeviceBufferTypeName(BufferType*) { return "cuda_device"; }
const char* HostBufferTypeName(BufferType*) { return "cuda_pinned_host"; }

Buffer* BufferTypeAllocImpl(BufferType* t, size_t bytes) {
    CudaBufferType* ct = AsBufferType(t);
    return CudaBufferAlloc(ct->backend, bytes, ct->is_host);
}

size_t BufferTypeAlignmentImpl(BufferType*) { return 256; }

bool DeviceBufferTypeIsHost(BufferType*) { return false; }
bool HostBufferTypeIsHost(BufferType*) { return true; }

} // namespace

// Exposed to cuda_backend.cu so it can wire the two buffer types onto a backend.
const BufferTypeI kCudaDeviceBufferTypeI = {
    /*get_name  =*/ &DeviceBufferTypeName,
    /*alloc     =*/ &BufferTypeAllocImpl,
    /*alignment =*/ &BufferTypeAlignmentImpl,
    /*is_host   =*/ &DeviceBufferTypeIsHost,
};

const BufferTypeI kCudaHostBufferTypeI = {
    /*get_name  =*/ &HostBufferTypeName,
    /*alloc     =*/ &BufferTypeAllocImpl,
    /*alignment =*/ &BufferTypeAlignmentImpl,
    /*is_host   =*/ &HostBufferTypeIsHost,
};

Buffer* CudaBufferAlloc(CudaBackend* backend, size_t bytes, bool is_host) {
    void* ptr = nullptr;
    if (bytes > 0) {
        cudaError_t err = is_host ? cudaMallocHost(&ptr, bytes)
                                  : cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            (void)cudaGetLastError();  // clear the sticky error
            return nullptr;
        }
    }
    CudaBuffer* cb = new CudaBuffer{};
    cb->iface   = &kCudaBufferI;
    cb->backend = backend;
    cb->ptr     = ptr;
    cb->bytes   = bytes;
    cb->is_host = is_host;
    return reinterpret_cast<Buffer*>(cb);
}

} // namespace nuka::phi
