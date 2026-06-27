// ---------------------------------------------------------------------------
// nuka::render::InteropTransformSsbo (M11 INT-4, OD-2=(A)).
//
// The exportable device-local per-instance transform SSBO + its descriptor set,
// plus an OPAQUE_FD export via vkGetMemoryFdKHR. See the header for the contract.
// Pure C++/Vulkan, zero CUDA token (src/render/** red-line). Every Vulkan call is
// defensively checked; ANY failure (incl. the lavapipe "no external-memory-fd"
// case) leaves the helper inert (Supported()==false) so the caller degrades.
// ---------------------------------------------------------------------------

#include "render/raster/interop_transform_ssbo.hpp"

// close() for the exported fd is POSIX-only; the OPAQUE_FD path is inert on
// Windows (no vkGetMemoryFdKHR), so the fd is never taken there.
#ifndef _WIN32
#include <unistd.h>  // close()
#endif

namespace nuka::render {

namespace {

uint32_t FindMemoryTypeIndex(VkPhysicalDevice physical, uint32_t type_filter,
                             VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        const bool type_ok = (type_filter & (1u << i)) != 0u;
        const bool flags_ok =
            (props.memoryTypes[i].propertyFlags & required) == required;
        if (type_ok && flags_ok) return i;
    }
    return UINT32_MAX;
}

}  // namespace

InteropTransformSsbo::~InteropTransformSsbo() { Destroy(); }

bool InteropTransformSsbo::Create(VkDevice device, VkPhysicalDevice physical,
                                  uint32_t instance_capacity) {
    Destroy();
    device_ = device;
    capacity_ = (instance_capacity == 0u) ? 1u : instance_capacity;
    // 16 floats (column-major mat4) per instance row.
    size_bytes_ = static_cast<uint64_t>(capacity_) * 16u * sizeof(float);

    // The export-fd entry point is an extension function; resolve it dynamically.
    // On an ICD without VK_KHR_external_memory_fd (lavapipe) this is null -> we
    // gracefully report unsupported and allocate nothing exportable.
    auto vkGetMemoryFdKHR_fn = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
    if (vkGetMemoryFdKHR_fn == nullptr) {
        // No external-memory-fd support -> inert. The caller keeps the default
        // per-draw pipeline + HostDownloadPublisher.
        return false;
    }

    // 1. The buffer, declared as externally-shareable (OPAQUE_FD) + a storage buffer.
    VkExternalMemoryBufferCreateInfo ext_buf{};
    ext_buf.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext_buf.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo buf{};
    buf.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf.pNext = &ext_buf;
    buf.size = size_bytes_;
    buf.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &buf, nullptr, &buffer_) != VK_SUCCESS) {
        buffer_ = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, buffer_, &req);
    const uint32_t mem_type =
        FindMemoryTypeIndex(physical, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type == UINT32_MAX) { Destroy(); return false; }

    // 2. Exportable + dedicated allocation (NVIDIA recommends dedicated for an
    //    imported buffer; defensive on every ICD).
    VkMemoryDedicatedAllocateInfo dedicated{};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.buffer = buffer_;

    VkExportMemoryAllocateInfo export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_info.pNext = &dedicated;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &export_info;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = mem_type;
    if (vkAllocateMemory(device_, &alloc, nullptr, &memory_) != VK_SUCCESS) {
        memory_ = VK_NULL_HANDLE;
        Destroy();
        return false;
    }
    size_bytes_ = req.size;        // the actual allocation size (fd import needs it)
    dedicated_ = true;
    if (vkBindBufferMemory(device_, buffer_, memory_, 0) != VK_SUCCESS) {
        Destroy();
        return false;
    }

    // 3. Export the OPAQUE_FD.
    VkMemoryGetFdInfoKHR get_fd{};
    get_fd.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    get_fd.memory = memory_;
    get_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (vkGetMemoryFdKHR_fn(device_, &get_fd, &exported_fd_) != VK_SUCCESS ||
        exported_fd_ < 0) {
        exported_fd_ = -1;
        Destroy();
        return false;
    }

    // 4. Descriptor set layout / pool / set for set 1 binding 0 (STORAGE_BUFFER,
    //    vertex stage -> the instanced vertex shader reads it).
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0u;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1u;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1u;
    layout_info.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &set_layout_) !=
        VK_SUCCESS) {
        set_layout_ = VK_NULL_HANDLE;
        Destroy();
        return false;
    }

    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u};
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1u;
    pool_info.poolSizeCount = 1u;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool_) != VK_SUCCESS) {
        pool_ = VK_NULL_HANDLE;
        Destroy();
        return false;
    }

    VkDescriptorSetAllocateInfo set_alloc{};
    set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc.descriptorPool = pool_;
    set_alloc.descriptorSetCount = 1u;
    set_alloc.pSetLayouts = &set_layout_;
    if (vkAllocateDescriptorSets(device_, &set_alloc, &descriptor_set_) != VK_SUCCESS) {
        descriptor_set_ = VK_NULL_HANDLE;
        Destroy();
        return false;
    }

    VkDescriptorBufferInfo dbuf{};
    dbuf.buffer = buffer_;
    dbuf.offset = 0u;
    dbuf.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set_;
    write.dstBinding = 0u;
    write.descriptorCount = 1u;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &dbuf;
    vkUpdateDescriptorSets(device_, 1u, &write, 0u, nullptr);

    supported_ = true;
    return true;
}

int InteropTransformSsbo::TakeExportedFd() {
    const int fd = exported_fd_;
    exported_fd_ = -1;  // ownership transferred to the caller (CUDA importer)
    return fd;
}

void InteropTransformSsbo::Destroy() {
    if (device_ != VK_NULL_HANDLE) {
        if (descriptor_set_ != VK_NULL_HANDLE && pool_ != VK_NULL_HANDLE) {
            // freed with the pool below
        }
        if (pool_ != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device_, pool_, nullptr); pool_ = VK_NULL_HANDLE; }
        if (set_layout_ != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr); set_layout_ = VK_NULL_HANDLE; }
        if (buffer_ != VK_NULL_HANDLE) { vkDestroyBuffer(device_, buffer_, nullptr); buffer_ = VK_NULL_HANDLE; }
        if (memory_ != VK_NULL_HANDLE) { vkFreeMemory(device_, memory_, nullptr); memory_ = VK_NULL_HANDLE; }
    }
#ifndef _WIN32
    if (exported_fd_ >= 0) { ::close(exported_fd_); exported_fd_ = -1; }
#endif
    descriptor_set_ = VK_NULL_HANDLE;
    size_bytes_ = 0;
    capacity_ = 0;
    dedicated_ = false;
    supported_ = false;
}

}  // namespace nuka::render
