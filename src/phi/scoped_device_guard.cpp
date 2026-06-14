// ---------------------------------------------------------------------------
// PHI - scoped CUDA active-device guard
// ---------------------------------------------------------------------------

#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::phi {

namespace {

void CheckCuda(cudaError_t error, const char* label) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(label) + " failed: " +
                                 cudaGetErrorString(error));
    }
}

} // namespace

ScopedDeviceGuard::ScopedDeviceGuard(int device_id) {
    CheckCuda(cudaGetDevice(&prior_device_), "cudaGetDevice");
    if (prior_device_ != device_id) {
        CheckCuda(cudaSetDevice(device_id), "cudaSetDevice");
        restore_ = true;
    }
}

ScopedDeviceGuard::~ScopedDeviceGuard() {
    if (restore_) {
        (void)cudaSetDevice(prior_device_);
    }
}

} // namespace nuka::phi
