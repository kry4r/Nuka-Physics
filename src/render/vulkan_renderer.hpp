#pragma once
// ---------------------------------------------------------------------------
// nuka::render::vulkan_renderer -- production renderer backend probe
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "render/render_scene.hpp"

#include <cstdint>
#include <string>
#include <vector>

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

enum class VulkanDebugDrawCommandType : uint32_t {
    Line = 0,
    Sphere = 1,
    Capsule = 2,
    Box = 3,
    AABB = 4,
    Frame = 5,
    ContactPoint = 6
};

struct VulkanDebugDrawCommand {
    VulkanDebugDrawCommandType type = VulkanDebugDrawCommandType::Line;
    math::Vec3 position = math::Vec3::Zero();
    math::Vec3 end = math::Vec3::Zero();
    math::Vec3 size = math::Vec3::Zero();
    float radius = 0.0f;
    float half_height = 0.0f;
    uint32_t color = 0xFFFFFFFFu;
};

struct VulkanRgba8 {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;
};

struct VulkanOffscreenOptions {
    uint32_t width = 640;
    uint32_t height = 360;
    float view_scale = 180.0f;
    math::Vec3 view_center = {0.0f, 0.0f, 0.25f};
    VulkanRgba8 background = {10, 12, 16, 255};
};

struct VulkanOffscreenReport {
    RenderBackend backend = RenderBackend::Vulkan;
    bool production_backend = true;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t command_count = 0;
    uint32_t physical_device_count = 0;
    std::string selected_device_name;
    size_t non_background_pixel_count = 0;
    std::vector<VulkanRgba8> pixels;
};

VulkanRendererReport ProbeVulkanRenderer();

VulkanOffscreenReport RenderDebugDrawListVulkan(
    const std::vector<VulkanDebugDrawCommand>& commands,
    const VulkanOffscreenOptions& options = {});

VulkanOffscreenReport RenderSceneVulkan(
    const RenderScene& scene,
    const VulkanOffscreenOptions& options = {});

} // namespace nuka::render
