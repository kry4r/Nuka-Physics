#pragma once
// Exportable pose slots: one buffer of `slot_count` regions of per-instance mat4s, a
// descriptor set per region (set 1, binding 0) and a timeline semaphore CUDA signals.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace nuka::render {

// The OS handles CUDA imports. A Win32 handle stays the importer's to close; a file
// descriptor passes to CUDA when its import succeeds.
struct InteropExport {
    void*    memory_handle    = nullptr;
    int      memory_fd        = -1;
    uint64_t memory_bytes     = 0;
    void*    semaphore_handle = nullptr;
    int      semaphore_fd     = -1;
    uint64_t slot_bytes       = 0;
};

class InteropTransformSsbo {
public:
    InteropTransformSsbo() = default;
    ~InteropTransformSsbo();
    InteropTransformSsbo(const InteropTransformSsbo&) = delete;
    InteropTransformSsbo& operator=(const InteropTransformSsbo&) = delete;

    // Throws std::runtime_error naming the Vulkan step that failed.
    void Create(VkDevice device, VkPhysicalDevice physical, uint32_t rows, uint32_t slot_count);
    // The exported handles; ownership passes to the caller.
    InteropExport TakeExport();

    VkDescriptorSetLayout SetLayout() const { return set_layout_; }
    VkDescriptorSet Slot(uint32_t slot) const { return sets_[slot]; }
    VkSemaphore Timeline() const { return timeline_; }
    uint32_t SlotCount() const { return static_cast<uint32_t>(sets_.size()); }

private:
    void Destroy();

    VkDevice device_ = VK_NULL_HANDLE;  // borrowed
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets_;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    InteropExport export_;
};

}  // namespace nuka::render
