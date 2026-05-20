// ---------------------------------------------------------------------------
// nuka::render::vulkan_renderer implementation
// ---------------------------------------------------------------------------

#include "render/vulkan_renderer.hpp"

#include <vulkan/vulkan.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::render {

namespace {

void CheckVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

} // namespace

VulkanRendererReport ProbeVulkanRenderer() {
    uint32_t loader_api_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion(&loader_api_version) != VK_SUCCESS) {
        loader_api_version = VK_API_VERSION_1_0;
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "nuka-physics";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "nuka";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkInstance instance = VK_NULL_HANDLE;
    CheckVk(vkCreateInstance(&create_info, nullptr, &instance), "vkCreateInstance");

    uint32_t device_count = 0;
    CheckVk(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
            "vkEnumeratePhysicalDevices(count)");

    std::vector<VkPhysicalDevice> devices(device_count);
    if (device_count > 0) {
        CheckVk(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()),
                "vkEnumeratePhysicalDevices(list)");
    }

    VulkanRendererReport report;
    report.backend = RenderBackend::Vulkan;
    report.production_backend = true;
    report.instance_api_version_major = VK_VERSION_MAJOR(loader_api_version);
    report.instance_api_version_minor = VK_VERSION_MINOR(loader_api_version);
    report.instance_api_version_patch = VK_VERSION_PATCH(loader_api_version);
    report.physical_device_count = device_count;

    if (!devices.empty()) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(devices.front(), &properties);
        report.selected_device_name = properties.deviceName;
    }

    vkDestroyInstance(instance, nullptr);
    return report;
}

} // namespace nuka::render
