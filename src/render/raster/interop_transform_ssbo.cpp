#include "render/raster/interop_transform_ssbo.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nuka::render {
namespace {

#ifdef _WIN32
constexpr VkExternalMemoryHandleTypeFlagBits kMemoryHandle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits kSemaphoreHandle =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
constexpr VkExternalMemoryHandleTypeFlagBits kMemoryHandle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits kSemaphoreHandle = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

void Check(VkResult result, const char* what) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string("pose slots: ") + what + " failed (VkResult " +
                                 std::to_string(static_cast<int>(result)) + ")");
}

uint32_t DeviceLocalType(VkPhysicalDevice physical, uint32_t type_bits) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        if ((type_bits & (1u << i)) != 0u &&
            (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0u)
            return i;
    throw std::runtime_error("pose slots: no device-local memory type");
}

template <class Fn>
Fn DeviceProc(VkDevice device, const char* name) {
    const auto fn = reinterpret_cast<Fn>(vkGetDeviceProcAddr(device, name));
    if (fn == nullptr) throw std::runtime_error(std::string("pose slots: ") + name + " unavailable");
    return fn;
}

void CloseExport(InteropExport& e) {
#ifdef _WIN32
    if (e.memory_handle != nullptr) CloseHandle(e.memory_handle);
    if (e.semaphore_handle != nullptr) CloseHandle(e.semaphore_handle);
#else
    if (e.memory_fd >= 0) close(e.memory_fd);
    if (e.semaphore_fd >= 0) close(e.semaphore_fd);
#endif
    e = InteropExport{};
}

}  // namespace

InteropTransformSsbo::~InteropTransformSsbo() { Destroy(); }

void InteropTransformSsbo::Create(VkDevice device, VkPhysicalDevice physical, uint32_t rows,
                                  uint32_t slot_count) {
    Destroy();
    device_ = device;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical, &props);
    const VkDeviceSize align = std::max<VkDeviceSize>(props.limits.minStorageBufferOffsetAlignment, 16u);
    const VkDeviceSize used = static_cast<VkDeviceSize>(std::max(rows, 1u)) * 16u * sizeof(float);
    const VkDeviceSize slot_bytes = (used + align - 1u) / align * align;

    VkExternalMemoryBufferCreateInfo external{};
    external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external.handleTypes = kMemoryHandle;
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = &external;
    buffer_info.size = slot_bytes * slot_count;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Check(vkCreateBuffer(device_, &buffer_info, nullptr, &buffer_), "vkCreateBuffer");

    // NVIDIA wants a dedicated allocation for memory CUDA imports.
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, buffer_, &req);
    VkMemoryDedicatedAllocateInfo dedicated{};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.buffer = buffer_;
    VkExportMemoryAllocateInfo export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_info.pNext = &dedicated;
    export_info.handleTypes = kMemoryHandle;
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &export_info;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = DeviceLocalType(physical, req.memoryTypeBits);
    Check(vkAllocateMemory(device_, &alloc, nullptr, &memory_), "vkAllocateMemory");
    Check(vkBindBufferMemory(device_, buffer_, memory_, 0u), "vkBindBufferMemory");
    export_.memory_bytes = req.size;
    export_.slot_bytes = slot_bytes;

    VkExportSemaphoreCreateInfo export_sem{};
    export_sem.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    export_sem.handleTypes = kSemaphoreHandle;
    VkSemaphoreTypeCreateInfo type{};
    type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type.pNext = &export_sem;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = 0u;
    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &type;
    Check(vkCreateSemaphore(device_, &sem_info, nullptr, &timeline_), "vkCreateSemaphore(timeline)");

#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR mem_get{};
    mem_get.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    mem_get.memory = memory_;
    mem_get.handleType = kMemoryHandle;
    HANDLE memory_handle = nullptr;
    Check(DeviceProc<PFN_vkGetMemoryWin32HandleKHR>(device_, "vkGetMemoryWin32HandleKHR")(device_, &mem_get,
                                                                                         &memory_handle),
          "vkGetMemoryWin32HandleKHR");
    export_.memory_handle = memory_handle;
    VkSemaphoreGetWin32HandleInfoKHR sem_get{};
    sem_get.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    sem_get.semaphore = timeline_;
    sem_get.handleType = kSemaphoreHandle;
    HANDLE semaphore_handle = nullptr;
    Check(DeviceProc<PFN_vkGetSemaphoreWin32HandleKHR>(device_, "vkGetSemaphoreWin32HandleKHR")(
              device_, &sem_get, &semaphore_handle),
          "vkGetSemaphoreWin32HandleKHR");
    export_.semaphore_handle = semaphore_handle;
#else
    VkMemoryGetFdInfoKHR mem_get{};
    mem_get.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    mem_get.memory = memory_;
    mem_get.handleType = kMemoryHandle;
    Check(DeviceProc<PFN_vkGetMemoryFdKHR>(device_, "vkGetMemoryFdKHR")(device_, &mem_get, &export_.memory_fd),
          "vkGetMemoryFdKHR");
    VkSemaphoreGetFdInfoKHR sem_get{};
    sem_get.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    sem_get.semaphore = timeline_;
    sem_get.handleType = kSemaphoreHandle;
    Check(DeviceProc<PFN_vkGetSemaphoreFdKHR>(device_, "vkGetSemaphoreFdKHR")(device_, &sem_get,
                                                                             &export_.semaphore_fd),
          "vkGetSemaphoreFdKHR");
#endif

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0u;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1u;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1u;
    layout_info.pBindings = &binding;
    Check(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &set_layout_), "vkCreateDescriptorSetLayout");

    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, slot_count};
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = slot_count;
    pool_info.poolSizeCount = 1u;
    pool_info.pPoolSizes = &pool_size;
    Check(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool_), "vkCreateDescriptorPool");

    const std::vector<VkDescriptorSetLayout> layouts(slot_count, set_layout_);
    sets_.assign(slot_count, VK_NULL_HANDLE);
    VkDescriptorSetAllocateInfo set_alloc{};
    set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_alloc.descriptorPool = pool_;
    set_alloc.descriptorSetCount = slot_count;
    set_alloc.pSetLayouts = layouts.data();
    Check(vkAllocateDescriptorSets(device_, &set_alloc, sets_.data()), "vkAllocateDescriptorSets");

    std::vector<VkDescriptorBufferInfo> infos(slot_count);
    std::vector<VkWriteDescriptorSet> writes(slot_count);
    for (uint32_t s = 0; s < slot_count; ++s) {
        infos[s].buffer = buffer_;
        infos[s].offset = slot_bytes * s;
        infos[s].range = used;
        writes[s].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[s].dstSet = sets_[s];
        writes[s].dstBinding = 0u;
        writes[s].descriptorCount = 1u;
        writes[s].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[s].pBufferInfo = &infos[s];
    }
    vkUpdateDescriptorSets(device_, slot_count, writes.data(), 0u, nullptr);
}

InteropExport InteropTransformSsbo::TakeExport() {
    const InteropExport out = export_;
    export_.memory_handle = export_.semaphore_handle = nullptr;
    export_.memory_fd = export_.semaphore_fd = -1;
    return out;
}

void InteropTransformSsbo::Destroy() {
    CloseExport(export_);
    if (device_ != VK_NULL_HANDLE) {
        if (pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, pool_, nullptr);
        if (set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
        if (timeline_ != VK_NULL_HANDLE) vkDestroySemaphore(device_, timeline_, nullptr);
        if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
        if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);
    }
    pool_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    timeline_ = VK_NULL_HANDLE;
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    sets_.clear();
}

}  // namespace nuka::render
