// ---------------------------------------------------------------------------
// PHI CUDA backend – device management & capability query
// ---------------------------------------------------------------------------

#include "phi/device.hpp"
#include "phi/capabilities.hpp"
#include "phi/platform_contract.hpp"

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

namespace nuka::phi {

PlatformContract GetPlatformContract() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDevice failed: ") +
                                 cudaGetErrorString(err));
    }

    PlatformContract contract;
    contract.physics_backend = PhysicsBackend::Cuda;
    contract.reference_backend = PhysicsBackend::CpuReference;
    contract.default_backend_policy = BackendSelectionPolicy::PreferCuda;
    contract.requires_gpu_simulation = true;
    contract.cpu_reference_only = true;
    contract.backend_selection_layer_enabled = true;
    contract.cuda_device_id = device;
    return contract;
}

BackendSelectionResult ResolvePhysicsBackend(const BackendSelectionRequest& request) {
    BackendSelectionResult result;
    if (request.policy == BackendSelectionPolicy::ForceCpuReference) {
        result.selected_backend = PhysicsBackend::CpuReference;
        result.production_backend = false;
        result.reference_backend = true;
        result.cuda_device_id = -1;
        return result;
    }

    const int device_count = GetDeviceCount();
    if (device_count <= 0) {
        throw std::runtime_error("CUDA backend requested, but no CUDA device is available");
    }

    if (request.cuda_device_id < 0 || request.cuda_device_id >= device_count) {
        throw std::runtime_error("Requested CUDA device id is outside the available device range");
    }

    result.selected_backend = PhysicsBackend::Cuda;
    result.production_backend = true;
    result.reference_backend = false;
    result.cuda_device_id = request.cuda_device_id;
    return result;
}

// ---- Capabilities --------------------------------------------------------

Capabilities QueryCapabilities() {
    cudaDeviceProp prop{};
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDevice failed: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") +
                                 cudaGetErrorString(err));
    }

    Capabilities caps{};
    caps.warp_size                = prop.warpSize;
    caps.max_threads_per_block    = prop.maxThreadsPerBlock;
    caps.max_shared_memory_per_block = static_cast<int>(prop.sharedMemPerBlock);
    caps.multiprocessor_count     = prop.multiProcessorCount;
    caps.compute_capability_major = prop.major;
    caps.compute_capability_minor = prop.minor;
    // async copy requires compute capability >= 8.0
    caps.supports_async_copy  = (prop.major > 8) || (prop.major == 8 && prop.minor >= 0);
    // CUDA graphs require compute capability >= 6.0
    caps.supports_graph_launch = (prop.major >= 6);
    return caps;
}

// ---- Device management ---------------------------------------------------

int GetDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") +
                                 cudaGetErrorString(err));
    }
    return count;
}

DeviceInfo GetDeviceInfo(int device_id) {
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") +
                                 cudaGetErrorString(err));
    }

    DeviceInfo info{};
    info.device_id    = device_id;
    info.total_memory = prop.totalGlobalMem;
    std::strncpy(info.name, prop.name, sizeof(info.name) - 1);
    info.name[sizeof(info.name) - 1] = '\0';
    return info;
}

void SetDevice(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                                 cudaGetErrorString(err));
    }
}

} // namespace nuka::phi
