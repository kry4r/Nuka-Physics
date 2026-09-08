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
#include <new>
#include <cstring>

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
        (void)cudaSetDevice(cb->device_id);
        if (cb->is_host) {
            (void)cudaFreeHost(cb->ptr);
        } else {
            (void)cudaFree(cb->ptr);
        }
    }
    delete cb;
}

void* BufferBaseImpl(Buffer* b) { return AsBuffer(b)->ptr; }

Status BufferUploadImpl(Buffer* b, const void* src, size_t off, size_t n) {
    CudaBuffer* cb = AsBuffer(b);
    if (off > cb->bytes || n > cb->bytes - off || (n != 0u && src == nullptr)) return Status::InvalidArgument;
    if (n == 0u) return Status::Ok;
    if (const auto status = cudaSetDevice(cb->device_id); status != cudaSuccess) return CudaStatus(status);
    cudaStream_t s = CudaBackendMainStream(cb->backend);
    if (cb->is_host) {
        const auto status = cudaStreamSynchronize(s);
        if (status != cudaSuccess) return CudaStatus(status);
        std::memcpy(static_cast<uint8_t*>(cb->ptr) + off, src, n);
        return Status::Ok;
    }
    return CudaStatus(cudaMemcpyAsync(static_cast<uint8_t*>(cb->ptr) + off, src, n,
                                      cudaMemcpyHostToDevice, s));
}

Status BufferDownloadImpl(Buffer* b, void* dst, size_t off, size_t n) {
    CudaBuffer* cb = AsBuffer(b);
    if (off > cb->bytes || n > cb->bytes - off || (n != 0u && dst == nullptr)) return Status::InvalidArgument;
    if (n == 0u) return Status::Ok;
    if (const auto status = cudaSetDevice(cb->device_id); status != cudaSuccess) return CudaStatus(status);
    cudaStream_t s = CudaBackendMainStream(cb->backend);
    if (cb->is_host) {
        const auto status = cudaStreamSynchronize(s);
        if (status != cudaSuccess) return CudaStatus(status);
        std::memcpy(dst, static_cast<const uint8_t*>(cb->ptr) + off, n);
        return Status::Ok;
    }
    const auto status = cudaMemcpyAsync(dst, static_cast<const uint8_t*>(cb->ptr) + off, n,
                                        cudaMemcpyDeviceToHost, s);
    if (status != cudaSuccess) return CudaStatus(status);
    // download is observable host-side; make it synchronous from the caller's
    // view by syncing the stream the copy was issued on.
    return CudaStatus(cudaStreamSynchronize(s));
}

Status BufferMemsetImpl(Buffer* b, uint8_t v, size_t off, size_t n) {
    CudaBuffer* cb = AsBuffer(b);
    if (off > cb->bytes || n > cb->bytes - off) return Status::InvalidArgument;
    if (n == 0u) return Status::Ok;
    if (const auto status = cudaSetDevice(cb->device_id); status != cudaSuccess) return CudaStatus(status);
    cudaStream_t s = CudaBackendMainStream(cb->backend);
    if (cb->is_host) {
        const auto status = cudaStreamSynchronize(s);
        if (status != cudaSuccess) return CudaStatus(status);
        std::memset(static_cast<uint8_t*>(cb->ptr) + off, v, n);
        return Status::Ok;
    }
    return CudaStatus(cudaMemsetAsync(static_cast<uint8_t*>(cb->ptr) + off, v, n, s));
}

Status BufferCopyFromImpl(Buffer* dst, Buffer* src, size_t doff, size_t soff, size_t n) {
    if (IfaceOf(src) != IfaceOf(dst)) return Status::Unsupported;
    CudaBuffer* cd = AsBuffer(dst);
    CudaBuffer* cs = AsBuffer(src);
    if (doff > cd->bytes || n > cd->bytes - doff || soff > cs->bytes || n > cs->bytes - soff)
        return Status::InvalidArgument;
    if (n == 0u) return Status::Ok;
    if (cd->device_id != cs->device_id || cd->backend != cs->backend) return Status::Unsupported;
    if (const auto status = cudaSetDevice(cd->device_id); status != cudaSuccess) return CudaStatus(status);
    cudaStream_t s = CudaBackendMainStream(cd->backend);
    if (cd->is_host && cs->is_host) {
        const auto status = cudaStreamSynchronize(s);
        if (status != cudaSuccess) return CudaStatus(status);
        std::memcpy(static_cast<uint8_t*>(cd->ptr) + doff, static_cast<const uint8_t*>(cs->ptr) + soff, n);
        return Status::Ok;
    }
    return CudaStatus(cudaMemcpyAsync(static_cast<uint8_t*>(cd->ptr) + doff,
                          static_cast<const uint8_t*>(cs->ptr) + soff, n,
                          cudaMemcpyDefault, s));
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

Buffer* BufferTypeAllocImpl(BufferType* t, size_t bytes, Status* status) {
    CudaBufferType* ct = AsBufferType(t);
    if (ct->device_id >= 0) {
        const auto selected = cudaSetDevice(ct->device_id);
        if (selected != cudaSuccess) { if (status) *status = CudaStatus(selected); return nullptr; }
    }
    return CudaBufferAlloc(ct->backend, bytes, ct->is_host, status);
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

Buffer* CudaBufferAlloc(CudaBackend* backend, size_t bytes, bool is_host, Status* status) {
    if (status) *status = Status::Ok;
    int device_id = 0;
    const auto selected = backend ? cudaSetDevice(backend->device_id) : cudaGetDevice(&device_id);
    if (selected != cudaSuccess) { if (status) *status = CudaStatus(selected); return nullptr; }
    if (backend) device_id = backend->device_id;
    void* ptr = nullptr;
    if (bytes > 0) {
        cudaError_t err = is_host ? cudaMallocHost(&ptr, bytes)
                                  : cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            if (status) *status = CudaStatus(err);
            (void)cudaGetLastError();  // clear the sticky error
            return nullptr;
        }
    }
    CudaBuffer* cb = new (std::nothrow) CudaBuffer{};
    if (!cb) {
        if (ptr) { if (is_host) cudaFreeHost(ptr); else cudaFree(ptr); }
        if (status) *status = Status::OutOfMemory;
        return nullptr;
    }
    cb->iface   = &kCudaBufferI;
    cb->backend = backend;
    cb->ptr     = ptr;
    cb->bytes   = bytes;
    cb->is_host = is_host;
    cb->device_id = device_id;
    return reinterpret_cast<Buffer*>(cb);
}

} // namespace nuka::phi
