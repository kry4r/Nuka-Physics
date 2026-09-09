#pragma once
// Kernel launches share an explicit stream for eager execution and graph capture.

#include <array>
#include <cuda_runtime.h>

namespace nuka::phi {

template <typename Kernel, typename... Args>
inline void LaunchCuda(Kernel kernel,
                       dim3 grid,
                       dim3 block,
                       size_t shared_mem,
                       cudaStream_t stream,
                       Args... args) {
    kernel<<<grid, block, shared_mem, stream>>>(args...);
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
