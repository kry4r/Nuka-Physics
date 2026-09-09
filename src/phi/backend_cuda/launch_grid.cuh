#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

namespace nuka::phi {

inline cudaError_t RequireCooperativeLaunch() {
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    static thread_local std::vector<int> supported_devices;
    for (int supported : supported_devices) {
        if (supported == device) return cudaSuccess;
    }
    int supported = 0;
    status = cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device);
    if (status != cudaSuccess) return status;
    if (supported == 0) return cudaErrorNotSupported;
    supported_devices.push_back(device);
    return cudaSuccess;
}

// A fixed resident grid consumes device-side work counts through grid-stride loops.
template <typename Kernel>
cudaError_t ResidentGridSize(Kernel kernel, uint32_t block_size, size_t shared_bytes,
                            uint32_t work_bound, uint32_t* grid_size) {
    if (!grid_size || block_size == 0u || block_size > std::numeric_limits<int>::max() ||
        shared_bytes > std::numeric_limits<int>::max()) return cudaErrorInvalidValue;
    if (work_bound == 0u) { *grid_size = 0u; return cudaSuccess; }
    int device = -1;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    struct Capacity {
        Kernel kernel;
        int device;
        uint32_t block_size;
        size_t shared_bytes;
        uint32_t blocks;
    };
    static thread_local std::vector<Capacity> capacities;
    for (const auto& capacity : capacities) {
        if (capacity.kernel == kernel && capacity.device == device &&
            capacity.block_size == block_size && capacity.shared_bytes == shared_bytes) {
            *grid_size = std::min(work_bound, capacity.blocks);
            return cudaSuccess;
        }
    }
    cudaFuncAttributes attributes{};
    status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) return status;
    if (shared_bytes > static_cast<size_t>(attributes.maxDynamicSharedSizeBytes)) {
        int limit = 0;
        status = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        if (status != cudaSuccess) return status;
        if (shared_bytes + attributes.sharedSizeBytes > static_cast<size_t>(limit))
            return cudaErrorInvalidConfiguration;
        status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      limit - static_cast<int>(attributes.sharedSizeBytes));
        if (status != cudaSuccess) return status;
    }
    int blocks_per_sm = 0, sm_count = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel,
                                                         static_cast<int>(block_size), shared_bytes);
    if (status != cudaSuccess) return status;
    status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) return status;
    if (blocks_per_sm <= 0 || sm_count <= 0) return cudaErrorInvalidConfiguration;
    const auto blocks = static_cast<uint32_t>(std::min<uint64_t>(
        uint64_t{static_cast<uint32_t>(blocks_per_sm)} * static_cast<uint32_t>(sm_count),
        std::numeric_limits<uint32_t>::max()));
    capacities.push_back({kernel, device, block_size, shared_bytes, blocks});
    *grid_size = std::min(work_bound, blocks);
    return cudaSuccess;
}

}  // namespace nuka::phi
