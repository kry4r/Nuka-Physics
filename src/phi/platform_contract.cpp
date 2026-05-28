// ---------------------------------------------------------------------------
// nuka::phi::PlatformContract - backend selection policy
// ---------------------------------------------------------------------------

#include "phi/platform_contract.hpp"

#include "phi/device.hpp"

#include <stdexcept>

namespace nuka::phi {

BackendSelectionResult ResolvePhysicsBackend(const BackendSelectionRequest& request,
                                             const PlatformContract& contract) {
    BackendSelectionResult result;
    if (request.policy == BackendSelectionPolicy::ForceCpuReference) {
        result.selected_backend = PhysicsBackend::CpuReference;
        result.production_backend = false;
        result.reference_backend = true;
        result.cuda_device_id = -1;
        return result;
    }

    const int requested_device = request.cuda_device_id >= 0
        ? request.cuda_device_id
        : contract.cuda_device_id;
    const int device_count = GetDeviceCount();
    if (device_count <= 0) {
        throw std::runtime_error("CUDA backend requested, but no CUDA device is available");
    }

    if (requested_device < 0 || requested_device >= device_count) {
        throw std::runtime_error("Requested CUDA device id is outside the available device range");
    }

    result.selected_backend = PhysicsBackend::Cuda;
    result.production_backend = true;
    result.reference_backend = false;
    result.cuda_device_id = requested_device;
    return result;
}

BackendSelectionResult ResolvePhysicsBackend(const BackendSelectionRequest& request) {
    return ResolvePhysicsBackend(request, PlatformContract{});
}

} // namespace nuka::phi
