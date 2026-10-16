#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace nuka::render::detail {

inline VkExtent2D ChooseSwapchainExtent(const VkSurfaceCapabilitiesKHR& caps,
                                       uint32_t width, uint32_t height) {
    if (width == 0u || height == 0u) return {0u, 0u};
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return caps.currentExtent;
    }
    return {std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width),
            std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height)};
}

inline VkCompositeAlphaFlagBitsKHR ChooseCompositeAlpha(VkCompositeAlphaFlagsKHR supported) {
    constexpr std::array choices{VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                                 VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                                 VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                                 VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
    for (const auto choice : choices) {
        if ((supported & choice) != 0u) return choice;
    }
    throw std::runtime_error("PresentRenderer: surface advertises no composite alpha mode");
}

inline VkImageUsageFlags SwapchainImageUsage(VkImageUsageFlags supported) {
    return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (supported & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}

inline bool CanCaptureSwapchain(VkImageUsageFlags usage, VkFormat format) {
    const bool rgba8 = format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB ||
                       format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
    return rgba8 && (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0u;
}

}  // namespace nuka::render::detail
