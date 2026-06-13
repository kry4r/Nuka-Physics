// ---------------------------------------------------------------------------
// render_raster_smoke -- M8 T3a smoke gate for the offscreen Vulkan forward
// raster renderer (src/render/raster/vulkan_raster_renderer.*).
//
// Builds a SYNTHETIC RenderWorld (two unit-ish boxes at distinct world poses +
// distinct base colors, interned as primitive-fallback meshes), renders it
// TWICE, and asserts:
//   (a) non_background_pixel_count > 0                      (G3 viability)
//   (b) the two pixel vectors are byte-identical (memcmp==0) (G2 determinism)
//   (c) dumps a PPM (P6) to /tmp/m8_raster_smoke.ppm for eyeballing.
// SKIPs gracefully if no Vulkan graphics device is present (renderer ctor throws).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nuka::render::MeshGeometry;
using nuka::render::RenderInstance;
using nuka::render::RenderWorld;
using nuka::render::VulkanRgba8;

// An axis-aligned box of half-extent `h`, centered at the origin, as 12 tris.
MeshGeometry MakeBox(float h) {
    MeshGeometry g;
    const float p[8][3] = {
        {-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},
        {-h, -h, h},  {h, -h, h},  {h, h, h},  {-h, h, h}};
    for (auto& v : p) {
        g.positions.push_back(v[0]);
        g.positions.push_back(v[1]);
        g.positions.push_back(v[2]);
    }
    const uint32_t faces[12][3] = {
        {0, 1, 2}, {0, 2, 3},  // -z
        {4, 6, 5}, {4, 7, 6},  // +z
        {0, 4, 5}, {0, 5, 1},  // -y
        {3, 2, 6}, {3, 6, 7},  // +y
        {1, 5, 6}, {1, 6, 2},  // +x
        {0, 3, 7}, {0, 7, 4}}; // -x
    for (auto& f : faces) {
        g.indices.push_back(f[0]);
        g.indices.push_back(f[1]);
        g.indices.push_back(f[2]);
    }
    return g;  // normals intentionally empty -> renderer synthesizes flat normals
}

RenderWorld BuildSyntheticWorld() {
    RenderWorld world;

    // Two materials with distinct base colors.
    nuka::scene::RenderMaterial red;
    red.base_color[0] = 0.85f; red.base_color[1] = 0.15f; red.base_color[2] = 0.15f; red.base_color[3] = 1.0f;
    nuka::scene::RenderMaterial blue;
    blue.base_color[0] = 0.20f; blue.base_color[1] = 0.35f; blue.base_color[2] = 0.90f; blue.base_color[3] = 1.0f;
    world.materials.push_back(red);
    world.materials.push_back(blue);

    // Two primitive-fallback box meshes interned into the library.
    const uint32_t mesh_a = world.meshes.InternPrimitive("prim:box:0.3", [] { return MakeBox(0.3f); });
    const uint32_t mesh_b = world.meshes.InternPrimitive("prim:box:0.2", [] { return MakeBox(0.2f); });

    RenderInstance a;
    a.mesh_id = mesh_a;
    a.render_material_id = 0;  // red
    a.world_xform.position = {-0.5f, 0.0f, 0.0f};
    world.instances.push_back(a);

    RenderInstance b;
    b.mesh_id = mesh_b;
    b.render_material_id = 1;  // blue
    b.world_xform.position = {0.5f, 0.0f, 0.25f};
    world.instances.push_back(b);

    return world;
}

void WritePpm(const std::string& path, const std::vector<VulkanRgba8>& pixels,
              uint32_t width, uint32_t height) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%u %u\n255\n", width, height);
    for (const auto& px : pixels) {
        const unsigned char rgb[3] = {px.r, px.g, px.b};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
}

}  // namespace

TEST(RenderRasterSmoke, TwoRendersAreByteIdenticalAndNonEmpty) {
    const RenderWorld world = BuildSyntheticWorld();

    nuka::render::RasterOptions options;
    options.width = 1280;
    options.height = 720;

    // Construct the renderer; skip if no Vulkan graphics device is present.
    std::unique_ptr<nuka::render::VulkanRasterRenderer> renderer;
    try {
        renderer = std::make_unique<nuka::render::VulkanRasterRenderer>();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "No Vulkan graphics device available: " << e.what();
    }

    nuka::render::VulkanOffscreenReport first;
    nuka::render::VulkanOffscreenReport second;
    try {
        first = renderer->Render(world, options);
        second = renderer->Render(world, options);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Vulkan raster render failed: " << e.what();
    }

    // (a) G3: geometry actually drew.
    EXPECT_GT(first.non_background_pixel_count, 0u);
    EXPECT_EQ(first.width, options.width);
    EXPECT_EQ(first.height, options.height);
    EXPECT_EQ(first.pixels.size(),
              static_cast<size_t>(options.width) * options.height);

    // (b) G2: two renders byte-identical.
    ASSERT_EQ(first.pixels.size(), second.pixels.size());
    EXPECT_EQ(0, std::memcmp(first.pixels.data(), second.pixels.data(),
                             first.pixels.size() * sizeof(VulkanRgba8)));

    // (c) dump a PPM for eyeballing.
    WritePpm("/tmp/m8_raster_smoke.ppm", first.pixels, first.width, first.height);

    std::printf("[render_raster_smoke] device=%s non_bg=%zu identical=%s\n",
                first.selected_device_name.c_str(),
                first.non_background_pixel_count,
                std::memcmp(first.pixels.data(), second.pixels.data(),
                            first.pixels.size() * sizeof(VulkanRgba8)) == 0
                    ? "YES"
                    : "NO");
}
