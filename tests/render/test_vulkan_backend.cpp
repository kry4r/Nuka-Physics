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
