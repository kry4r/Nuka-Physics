// ---------------------------------------------------------------------------
// Vulkan renderer backend contract tests.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "render/vulkan_renderer.hpp"

TEST(VulkanRenderer, CreatesInstanceAndEnumeratesPhysicalDevices) {
    const auto report = nuka::render::ProbeVulkanRenderer();
    EXPECT_GE(report.instance_api_version_major, 1u);
    EXPECT_GE(report.physical_device_count, 1u);
    EXPECT_FALSE(report.selected_device_name.empty());
}

TEST(VulkanRenderer, DeclaresVulkanAsProductionBackend) {
    const auto report = nuka::render::ProbeVulkanRenderer();
    EXPECT_EQ(report.backend, nuka::render::RenderBackend::Vulkan);
    EXPECT_TRUE(report.production_backend);
}

TEST(VulkanRenderer, RendersDebugCommandsToOffscreenImage) {
    std::vector<nuka::render::VulkanDebugDrawCommand> commands;

    nuka::render::VulkanDebugDrawCommand line;
    line.type = nuka::render::VulkanDebugDrawCommandType::Line;
    line.position = {-0.4f, -0.2f, 0.0f};
    line.end = {0.4f, 0.2f, 0.0f};
    line.color = 0xFF30D5C8u;
    commands.push_back(line);

    nuka::render::VulkanDebugDrawCommand sphere;
    sphere.type = nuka::render::VulkanDebugDrawCommandType::Sphere;
    sphere.position = {0.0f, 0.0f, 0.0f};
    sphere.radius = 0.12f;
    sphere.color = 0xFFFFC857u;
    commands.push_back(sphere);

    nuka::render::VulkanOffscreenOptions options;
    options.width = 128;
    options.height = 96;
    options.view_scale = 90.0f;
    options.view_center = {0.0f, 0.0f, 0.0f};

    const auto result = nuka::render::RenderDebugDrawListVulkan(commands, options);

    EXPECT_EQ(result.backend, nuka::render::RenderBackend::Vulkan);
    EXPECT_TRUE(result.production_backend);
    EXPECT_EQ(result.width, options.width);
    EXPECT_EQ(result.height, options.height);
    EXPECT_EQ(result.command_count, commands.size());
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_FALSE(result.pixels.empty());
}
