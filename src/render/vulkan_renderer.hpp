#pragma once
// ---------------------------------------------------------------------------
// nuka::render::vulkan_renderer -- production renderer backend probe
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>

namespace nuka::render {

enum class RenderBackend {
    Vulkan
};

struct VulkanRendererReport {
    RenderBackend backend = RenderBackend::Vulkan;
    bool production_backend = true;
    uint32_t instance_api_version_major = 0;
    uint32_t instance_api_version_minor = 0;
    uint32_t instance_api_version_patch = 0;
    uint32_t physical_device_count = 0;
    std::string selected_device_name;
};

VulkanRendererReport ProbeVulkanRenderer();

} // namespace nuka::render
