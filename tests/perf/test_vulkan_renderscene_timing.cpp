// ---------------------------------------------------------------------------
// Vulkan RenderScene timing coverage
// ---------------------------------------------------------------------------

#include "render/render_scene.hpp"
#include "render/vulkan_renderer.hpp"

#include <gtest/gtest.h>

#include <chrono>

TEST(VulkanRenderSceneTiming, RenderSceneMaterialPassUnderOneSecond) {
    nuka::render::RenderScene scene;

    nuka::render::RenderMaterial red;
    red.material_id = 0u;
    red.base_color = {1.0f, 0.05f, 0.02f};
    scene.materials.push_back(red);

    nuka::render::RenderMaterial blue;
    blue.material_id = 1u;
    blue.base_color = {0.05f, 0.25f, 1.0f};
    scene.materials.push_back(blue);

    constexpr uint32_t kInstanceCount = 128u;
    scene.mesh_instances.reserve(kInstanceCount);
    for (uint32_t index = 0; index < kInstanceCount; ++index) {
        nuka::render::RenderMeshInstance mesh;
        mesh.material_id = index % 2u;
        mesh.shape_type = (index % 3u == 0u)
            ? nuka::scene::ShapeType::Sphere
            : nuka::scene::ShapeType::Box;
        const float column = static_cast<float>(index % 16u);
        const float row = static_cast<float>(index / 16u);
        mesh.world_transform.position = {
            (column - 7.5f) * 0.16f,
            (row - 3.5f) * 0.16f,
            0.0f};
        mesh.radius = 0.045f;
        mesh.half_extents = {0.045f, 0.045f, 0.045f};
        scene.mesh_instances.push_back(mesh);
    }

    scene.material_count = static_cast<uint32_t>(scene.materials.size());
    scene.mesh_instance_count = static_cast<uint32_t>(scene.mesh_instances.size());

    nuka::render::VulkanOffscreenOptions options;
    options.width = 320u;
    options.height = 180u;
    options.view_scale = 120.0f;
    options.view_center = {0.0f, 0.0f, 0.0f};

    const auto start = std::chrono::steady_clock::now();
    const auto result = nuka::render::RenderSceneVulkan(scene, options);
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(result.backend, nuka::render::RenderBackend::Vulkan);
    EXPECT_TRUE(result.production_backend);
    EXPECT_EQ(result.width, options.width);
    EXPECT_EQ(result.height, options.height);
    EXPECT_EQ(result.command_count, kInstanceCount);
    EXPECT_GT(result.physical_device_count, 0u);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_LT(elapsed_ms, 1000);
}
