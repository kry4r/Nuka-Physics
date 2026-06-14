#pragma once
// ---------------------------------------------------------------------------
// nuka::render::InteropTransformSsbo -- the OPT-IN, default-OFF exportable
// per-instance transform SSBO for the CUDA<->Vulkan zero-copy render path (M11
// INT-4, OD-2=(A)).
//
// Allocates a DEVICE-LOCAL VkBuffer + VkDeviceMemory of `instance_capacity` mat4
// rows (16 floats each), with the memory made EXPORTABLE via VK_KHR_external_memory
// (VkExportMemoryAllocateInfo, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) +
// VkMemoryDedicatedAllocateInfo (NVIDIA wants/needs dedicated for an imported
// buffer), and exports an OPAQUE_FD via vkGetMemoryFdKHR. The CUDA scatter
// (phi/backend_cuda/interop) imports that fd with cudaImportExternalMemory and
// writes the composed object->world matrices straight into this buffer -- NO
// device->host copy. The renderer's instanced vertex shader (mesh_instanced.vert)
// then reads each instance's matrix from this SSBO by gl_InstanceIndex.
//
// GRACEFUL DEGRADE (the only path runnable on this lavapipe/CPU-Vulkan box):
// vkGetMemoryFdKHR is looked up via vkGetDeviceProcAddr -- on an ICD that does not
// support VK_KHR_external_memory_fd it is null, so Create() reports Supported()==
// false and exports NO fd. Likewise any allocation/export failure leaves
// Supported()==false. The caller (the publisher/viewer) then keeps the default
// per-draw push-constant pipeline (D1 oracle path untouched) and falls back to
// HostDownloadPublisher. NOTHING about the default offscreen render path changes
// when this helper is never created (interop default-OFF -> every D1 gate is
// byte-identical).
//
// ZERO CUDA TOKEN: this is pure C++/Vulkan (src/render/** red-line). The exported
// fd is surfaced as a plain `int` (ExportedFd()) -- no CUDA type. The publisher
// forwards it through the CUDA-free phi::ExternalMemoryDesc seam.
// ---------------------------------------------------------------------------

#include <vulkan/vulkan.h>

#include <cstdint>

namespace nuka::render {

class InteropTransformSsbo {
public:
    InteropTransformSsbo() = default;
    ~InteropTransformSsbo();

    InteropTransformSsbo(const InteropTransformSsbo&) = delete;
    InteropTransformSsbo& operator=(const InteropTransformSsbo&) = delete;

    // Allocate the exportable device-local SSBO + a descriptor set (set 1, binding
    // 0, STORAGE_BUFFER) bound over the whole buffer, and export an OPAQUE_FD.
    // Returns true iff the buffer was allocated, the fd exported, AND the device
    // supports VK_KHR_external_memory_fd (vkGetMemoryFdKHR present). On false the
    // helper is inert (Supported()==false) and the caller must NOT use the
    // instanced path. `instance_capacity` mat4 rows are reserved (>=1).
    bool Create(VkDevice device, VkPhysicalDevice physical,
                uint32_t instance_capacity);

    bool Supported() const { return supported_; }

    // The exported OPAQUE_FD (vkGetMemoryFdKHR). The caller (CUDA importer) OWNS the
    // returned fd lifetime per the Vulkan fd-export contract: cudaImportExternalMemory
    // dup()s it, but the EXPORTER must close it after import. We hand ownership to the
    // caller exactly once via TakeExportedFd(); a second call returns -1.
    int TakeExportedFd();

    // The full VkDeviceMemory size (POSIX fd import needs the whole allocation size).
    uint64_t SizeBytes() const { return size_bytes_; }
    bool     Dedicated() const { return dedicated_; }
    uint32_t Capacity()  const { return capacity_; }

    // Vulkan handles for the instanced pipeline (set 1 = per-instance transforms).
    VkDescriptorSetLayout SetLayout() const { return set_layout_; }
    VkDescriptorSet       DescriptorSet() const { return descriptor_set_; }
    VkBuffer              Buffer() const { return buffer_; }

private:
    void Destroy();

    VkDevice              device_         = VK_NULL_HANDLE;  // borrowed (not owned)
    VkBuffer              buffer_         = VK_NULL_HANDLE;
    VkDeviceMemory        memory_         = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_     = VK_NULL_HANDLE;
    VkDescriptorPool      pool_           = VK_NULL_HANDLE;
    VkDescriptorSet       descriptor_set_ = VK_NULL_HANDLE;
    uint64_t              size_bytes_     = 0;
    uint32_t              capacity_       = 0;
    int                   exported_fd_    = -1;
    bool                  dedicated_      = false;
    bool                  supported_      = false;
};

}  // namespace nuka::render
