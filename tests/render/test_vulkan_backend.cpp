// ---------------------------------------------------------------------------
// Vulkan renderer backend contract tests.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "render/vulkan_renderer.hpp"

#include <algorithm>

namespace {

bool HasPixel(const std::vector<nuka::render::VulkanRgba8>& pixels,
              nuka::render::VulkanRgba8 expected) {
    return std::any_of(pixels.begin(), pixels.end(), [&](const auto& pixel) {
        return pixel.r == expected.r && pixel.g == expected.g &&
            pixel.b == expected.b && pixel.a == expected.a;
    });
}

} // namespace

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

// M11 retarget: the former RendersRenderSceneMaterialsToOffscreenImage test fed
// a render::RenderScene (deleted) through the RenderSceneVulkan wrapper. The
// only live overlay path is now the command list, so the test builds the same
// colored shapes directly as VulkanDebugDrawCommands. Colors are RGBA8 packed
// 0xRRGGBBAA (matching the compute shader's pixel writes) so the per-shape
// material color -> rendered pixel coverage is preserved.
TEST(VulkanRenderer, RendersColoredShapeCommandsToOffscreenImage) {
    std::vector<nuka::render::VulkanDebugDrawCommand> commands;

    nuka::render::VulkanDebugDrawCommand sphere;
    sphere.type = nuka::render::VulkanDebugDrawCommandType::Sphere;
    sphere.position = {-0.35f, 0.0f, 0.0f};
    sphere.radius = 0.15f;
    sphere.color = 0xFF0000FFu;  // red, RGBA8 packed
    commands.push_back(sphere);

    nuka::render::VulkanDebugDrawCommand box;
    box.type = nuka::render::VulkanDebugDrawCommandType::Box;
    box.position = {0.35f, 0.0f, 0.0f};
    box.size = {0.12f, 0.12f, 0.12f};
    box.color = 0x00FF00FFu;  // green, RGBA8 packed
    commands.push_back(box);

    nuka::render::VulkanOffscreenOptions options;
    options.width = 160;
    options.height = 120;
    options.view_scale = 120.0f;
    options.view_center = {0.0f, 0.0f, 0.0f};

    const auto result = nuka::render::RenderDebugDrawListVulkan(commands, options);

    EXPECT_EQ(result.backend, nuka::render::RenderBackend::Vulkan);
    EXPECT_TRUE(result.production_backend);
    EXPECT_EQ(result.width, options.width);
    EXPECT_EQ(result.height, options.height);
    EXPECT_EQ(result.command_count, commands.size());
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_TRUE(HasPixel(result.pixels, {255, 0, 0, 255}));
    EXPECT_TRUE(HasPixel(result.pixels, {0, 255, 0, 255}));
}
