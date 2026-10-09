#pragma once
// Kernel launches share an explicit stream for eager execution and graph capture.

#include <array>
#include <cuda_runtime.h>

namespace nuka::phi {

inline thread_local bool g_cuda_programmatic_launch = false;

struct CudaLaunchScope {
    bool previous;
    explicit CudaLaunchScope(bool allowed) : previous(g_cuda_programmatic_launch) {
        g_cuda_programmatic_launch = allowed;
    }
    ~CudaLaunchScope() { g_cuda_programmatic_launch = previous; }
};

__device__ inline void BeginProgrammaticKernel() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
#endif
}

template <bool dependent = false, typename Kernel, typename... Args>
inline void LaunchCuda(Kernel kernel,
                       dim3 grid,
                       dim3 block,
                       size_t shared_mem,
                       cudaStream_t stream,
                       Args... args) {
    if constexpr (dependent) {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = g_cuda_programmatic_launch;
        cudaLaunchConfig_t config{};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = shared_mem;
        config.stream = stream;
        config.attrs = &attribute;
        config.numAttrs = 1u;
        (void)cudaLaunchKernelEx(&config, kernel, args...);
    } else {
        kernel<<<grid, block, shared_mem, stream>>>(args...);
    }
}

template <typename Kernel, typename... Args>
inline cudaError_t LaunchCooperativeCuda(Kernel kernel,
                                         dim3 grid,
                                         dim3 block,
                                         size_t shared_mem,
                                         cudaStream_t stream,
                                         Args... args) {
    std::array<void*, sizeof...(Args)> arguments{static_cast<void*>(&args)...};
    return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel),
                                       grid, block, arguments.data(), shared_mem, stream);
}

} // namespace nuka::phi
