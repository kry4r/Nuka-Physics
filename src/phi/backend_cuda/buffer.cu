// ---------------------------------------------------------------------------
// PHI CUDA backend - Buffer implementation
// ---------------------------------------------------------------------------

#include "phi/buffer_legacy.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace nuka::phi {

// ===========================================================================
// Buffer
// ===========================================================================

Buffer::Buffer(size_t size_bytes, MemoryKind kind)
    : size_(size_bytes), kind_(kind) {
    if (size_bytes == 0) return;

    cudaError_t err = cudaSuccess;
    switch (kind) {
        case MemoryKind::Device:
            err = cudaMalloc(&data_, size_bytes);
            break;
        case MemoryKind::Host:
            err = cudaMallocHost(&data_, size_bytes);
            break;
        case MemoryKind::Managed:
            err = cudaMallocManaged(&data_, size_bytes);
            break;
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("PHI Buffer allocation failed: ") +
                                 cudaGetErrorString(err));
    }
}

Buffer::~Buffer() {
    if (data_) {
        switch (kind_) {
            case MemoryKind::Device:
            case MemoryKind::Managed:
                cudaFree(data_);
                break;
            case MemoryKind::Host:
                cudaFreeHost(data_);
                break;
        }
    }
}

Buffer::Buffer(Buffer&& other) noexcept
    : data_(other.data_), size_(other.size_), kind_(other.kind_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        // Free current allocation
        if (data_) {
            switch (kind_) {
                case MemoryKind::Device:
                case MemoryKind::Managed:
                    cudaFree(data_);
                    break;
                case MemoryKind::Host:
                    cudaFreeHost(data_);
                    break;
            }
        }
        data_ = other.data_;
        size_ = other.size_;
        kind_ = other.kind_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void* Buffer::Data() { return data_; }
const void* Buffer::Data() const { return data_; }
size_t Buffer::Size() const { return size_; }
MemoryKind Buffer::Kind() const { return kind_; }

void Buffer::CopyFromHost(const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(data_, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CopyFromHost failed: ") +
                                 cudaGetErrorString(err));
    }
}

void Buffer::CopyToHost(void* dst, size_t bytes) const {
    cudaError_t err = cudaMemcpy(dst, data_, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CopyToHost failed: ") +
                                 cudaGetErrorString(err));
    }
}

} // namespace nuka::phi
