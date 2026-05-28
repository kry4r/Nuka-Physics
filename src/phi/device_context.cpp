// ---------------------------------------------------------------------------
// PHI - per-handle CUDA device context
// ---------------------------------------------------------------------------

#include "phi/device_context.hpp"

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

void StreamView::Synchronize() const {
    CheckCuda(cudaStreamSynchronize(handle_), "cudaStreamSynchronize");
}

DeviceContext MakeDeviceContext(int device_id,
                                cudaStream_t external_stream,
                                const PlatformContract& contract) {
    DeviceContext context;
    context.device_id = device_id;
    context.stream = StreamView{external_stream};
    context.contract = contract;
    context.contract.cuda_device_id = device_id;
    return context;
}

DeviceContext MakeDefaultDeviceContext() {
    return MakeDeviceContext(0, nullptr, PlatformContract{});
}

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
