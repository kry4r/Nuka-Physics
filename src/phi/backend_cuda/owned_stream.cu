// ---------------------------------------------------------------------------
// PHI CUDA backend - owning stream implementation
// ---------------------------------------------------------------------------

#include "phi/owned_stream.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace nuka::phi {

namespace {

void CheckCuda(cudaError_t error, const char* label) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(label) + " failed: " +
                                 cudaGetErrorString(error));
    }
}

} // namespace

OwnedStream::OwnedStream() {
    CheckCuda(cudaStreamCreate(&handle_), "cudaStreamCreate");
}

OwnedStream::~OwnedStream() {
    if (handle_) {
        (void)cudaStreamDestroy(handle_);
    }
}

OwnedStream::OwnedStream(OwnedStream&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
}

OwnedStream& OwnedStream::operator=(OwnedStream&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            (void)cudaStreamDestroy(handle_);
        }
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

void OwnedStream::Synchronize() const {
    View().Synchronize();
}

} // namespace nuka::phi
