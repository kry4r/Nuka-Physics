#pragma once
// ---------------------------------------------------------------------------
// nuka::render::vulkan_renderer -- production renderer backend probe
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "render/render_scene.hpp"
#include "render/vulkan_offscreen_types.hpp"  // RenderBackend, VulkanRgba8, VulkanOffscreenReport

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::render {

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

struct VulkanOffscreenOptions {
    uint32_t width = 640;
    uint32_t height = 360;
    float view_scale = 180.0f;
    math::Vec3 view_center = {0.0f, 0.0f, 0.25f};
    VulkanRgba8 background = {10, 12, 16, 255};
};

VulkanRendererReport ProbeVulkanRenderer();

VulkanOffscreenReport RenderDebugDrawListVulkan(
    const std::vector<VulkanDebugDrawCommand>& commands,
    const VulkanOffscreenOptions& options = {});

VulkanOffscreenReport RenderSceneVulkan(
    const RenderScene& scene,
    const VulkanOffscreenOptions& options = {});

} // namespace nuka::render
