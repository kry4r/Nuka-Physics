// ---------------------------------------------------------------------------
// nuka::phi interop-scatter WEAK fallback (M11 §3.E INT-1/INT-6, OD-1=(A)).
//
// The CudaVulkanInteropPublisher (runtime/app, zero-CUDA) calls
// phi::CreateCudaVkScatter() / phi::CudaVkScatterAvailable() through the
// CUDA-FREE seam (phi/interop_scatter.hpp). The STRONG definitions live in the
// CUDA backend TU (phi/backend_cuda/interop/cuda_vk_scatter.cu, compiled into
// nuka_phi2_interop) and WIN at link time whenever that lib is linked (the
// windowed viewer exe). This WEAK fallback lets the app/render libs LINK STANDALONE
// (e.g. the default build-cuda128 host tools, or a Vulkan-less toolchain) with no
// CUDA dependency: it returns "no interop available", so the publisher gracefully
// degrades to HostDownloadPublisher.
//
// Same pattern intent as the RT facet's CreateCudaRtBackend (strong in the CUDA
// lib), but with an explicit weak fallback because the app layer REFERENCES the
// factory directly (unlike nuka_render, which never names the RT factory). HOST,
// zero-CUDA token (this TU is in src/runtime/app/** -- the lint red-line); it names
// only the CUDA-free interface.
// ---------------------------------------------------------------------------

#include "phi/interop_scatter.hpp"

namespace nuka::phi {

// __attribute__((weak)): overridden by the strong symbol in
// phi/backend_cuda/interop/cuda_vk_scatter.cu when nuka_phi2_interop is linked.
__attribute__((weak)) std::unique_ptr<InteropScatterI> CreateCudaVkScatter() {
    return nullptr;
}

__attribute__((weak)) bool CudaVkScatterAvailable() { return false; }

}  // namespace nuka::phi
