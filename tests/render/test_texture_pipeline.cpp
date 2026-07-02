// ---------------------------------------------------------------------------
// Host gates for the beauty texture pipeline: image decode (stb), the
// RenderWorld texture library, and the rt_adapter mapping. Asserts the
// UNTEXTURED path is a strict data no-op (an old material maps to the same
// rt::Material fields it always did) and the textured path plumbs faithfully.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "render/render_world.hpp"
#include "render/rt_adapter.hpp"
#include "render/rt_framebuffer_to_report.hpp"
#include "render/texture_image.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using nuka::render::FramebufferToReport;
using nuka::render::LoadEnvironment;
using nuka::render::LoadTexture;
using nuka::render::MeshGeometry;
using nuka::render::RenderWorld;
using nuka::render::TextureLibrary;

std::string WriteTestPng(const char* name, int w, int h,
                         const std::vector<uint8_t>& rgb) {
    const std::string path = std::string(::testing::TempDir()) + name;
    EXPECT_EQ(stbi_write_png(path.c_str(), w, h, 3, rgb.data(), w * 3), 1);
    return path;
}

TEST(TexturePipeline, DecodesPngWithColorspaceFlag) {
    // 2x2 RGB: red, green, blue, mid-grey (raw sRGB bytes; the sampler linearises).
    const std::vector<uint8_t> rgb = {255, 0, 0, 0, 255, 0,
                                      0, 0, 255, 128, 128, 128};
    const std::string path = WriteTestPng("tex_rgb.png", 2, 2, rgb);
    const nuka::rt::Texture tex = LoadTexture(path, /*srgb=*/true);
    ASSERT_FALSE(tex.Empty());
    EXPECT_EQ(tex.width, 2u);
    EXPECT_EQ(tex.height, 2u);
    EXPECT_EQ(tex.channels, 3u);
    EXPECT_EQ(tex.srgb, 1u);
    ASSERT_EQ(tex.texels.size(), 12u);
    EXPECT_NEAR(tex.texels[0], 1.0f, 1e-6f);           // R of texel 0
    EXPECT_NEAR(tex.texels[9], 128.0f / 255.0f, 1e-6f);  // grey stays raw here

    const nuka::rt::Texture lin = LoadTexture(path, /*srgb=*/false);
    EXPECT_EQ(lin.srgb, 0u);
    EXPECT_TRUE(LoadTexture("/nonexistent/nope.png", true).Empty());
}

TEST(TexturePipeline, DecodesTwoChannelGreyAlphaAsTwoChannels) {
    // stb req_comp=0 returns 2 channels for a grey+alpha image; the loader stores
    // them faithfully (channel 1 is a decoy the device sampler must NOT read as G/B).
    const std::vector<uint8_t> ga = {200, 40, 200, 40};  // 2x1: ch0=grey, ch1=alpha
    const std::string path = std::string(::testing::TempDir()) + "tex_grey_alpha.png";
    ASSERT_EQ(stbi_write_png(path.c_str(), 2, 1, 2, ga.data(), 2 * 2), 1);
    const nuka::rt::Texture tex = LoadTexture(path, /*srgb=*/false);
    ASSERT_FALSE(tex.Empty());
    EXPECT_EQ(tex.channels, 2u);
    ASSERT_EQ(tex.texels.size(), 4u);
    EXPECT_NEAR(tex.texels[0], 200.0f / 255.0f, 1e-6f);  // ch0 grey
    EXPECT_NEAR(tex.texels[1], 40.0f / 255.0f, 1e-6f);   // ch1 alpha decoy
}

TEST(TexturePipeline, LibraryDedupesByPathAndColorspace) {
    const std::vector<uint8_t> rgb(2 * 2 * 3, 200);
    const std::string path = WriteTestPng("tex_dedupe.png", 2, 2, rgb);
    TextureLibrary lib;
    const uint32_t a = lib.Intern(path, true, &LoadTexture);
    const uint32_t b = lib.Intern(path, true, &LoadTexture);
    const uint32_t c = lib.Intern(path, false, &LoadTexture);
    EXPECT_EQ(a, b);                       // same path+space -> one image
    EXPECT_NE(a, c);                       // colorspace splits the entry
    EXPECT_EQ(lib.Count(), 2u);
    EXPECT_EQ(lib.Resolve(path, true), static_cast<int>(a));
    EXPECT_EQ(lib.Resolve("", true), -1);
    EXPECT_EQ(lib.Resolve("missing.png", true), -1);
    EXPECT_EQ(lib.Intern("", true, &LoadTexture), nuka::render::kNoId);
}

TEST(TexturePipeline, UntexturedMaterialMapsAsAlways) {
    // An old (map-free) material must produce the identical rt::Material data:
    // texture indices unbound, no scene textures, environment disabled.
    RenderWorld world;
    nuka::scene::RenderMaterial m;
    m.base_color[0] = 0.25f; m.base_color[1] = 0.5f; m.base_color[2] = 0.75f;
    m.roughness = 0.4f;
    m.metallic = 0.9f;
    world.materials.push_back(m);
    nuka::render::DecodeMaterialTextures(world);
    EXPECT_EQ(world.textures.Count(), 0u);

    const nuka::rt::TwoLevelScene scene =
        nuka::render::RenderWorldToTwoLevelScene(world);
    ASSERT_EQ(scene.materials.size(), 2u);  // authored + appended default
    const nuka::rt::Material& rt0 = scene.materials[0];
    EXPECT_EQ(rt0.albedo_tex, -1);
    EXPECT_EQ(rt0.roughness_tex, -1);
    EXPECT_EQ(rt0.normal_tex, -1);
    EXPECT_FLOAT_EQ(rt0.uv_scale, 1.0f);
    EXPECT_EQ(rt0.triplanar, 1u);
    EXPECT_FLOAT_EQ(rt0.albedo.x, 0.25f);
    EXPECT_FLOAT_EQ(rt0.roughness, 0.4f);
    EXPECT_FLOAT_EQ(rt0.metallic, 0.9f);
    EXPECT_TRUE(scene.textures && scene.textures->empty());
    EXPECT_FALSE(scene.environment.Enabled());
}

TEST(TexturePipeline, TexturedMaterialResolvesAndUvsRide) {
    const std::vector<uint8_t> rgb(4 * 4 * 3, 90);
    const std::string albedo = WriteTestPng("tex_albedo.png", 4, 4, rgb);
    const std::string normal = WriteTestPng("tex_normal.png", 4, 4, rgb);

    RenderWorld world;
    nuka::scene::RenderMaterial m;
    m.albedo_map = albedo;
    m.normal_map = normal;
    m.uv_scale = 3.0f;
    m.triplanar = false;
    world.materials.push_back(m);
    nuka::render::DecodeMaterialTextures(world);
    EXPECT_EQ(world.textures.Count(), 2u);

    // One triangle with authored UVs -> tri_uvs ride 1:1 into the BLAS mesh.
    MeshGeometry geo;
    geo.positions = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    geo.uvs = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    geo.indices = {0, 1, 2};
    world.meshes.InternPrimitive("t", [&] { return geo; });

    const nuka::rt::TwoLevelScene scene =
        nuka::render::RenderWorldToTwoLevelScene(world);
    const nuka::rt::Material& rt0 = scene.materials[0];
    EXPECT_GE(rt0.albedo_tex, 0);
    EXPECT_GE(rt0.normal_tex, 0);
    EXPECT_EQ(rt0.roughness_tex, -1);
    EXPECT_FLOAT_EQ(rt0.uv_scale, 3.0f);
    EXPECT_EQ(rt0.triplanar, 0u);
    ASSERT_TRUE(scene.textures);
    ASSERT_EQ(scene.textures->size(), 2u);
    EXPECT_EQ((*scene.textures)[rt0.albedo_tex].srgb, 1u);   // albedo decodes sRGB
    EXPECT_EQ((*scene.textures)[rt0.normal_tex].srgb, 0u);   // normal stays linear
    ASSERT_EQ(scene.meshes.size(), 1u);
    ASSERT_EQ(scene.meshes[0].tri_uvs.size(), 1u);
    EXPECT_FLOAT_EQ(scene.meshes[0].tri_uvs[0].uv1.x, 1.0f);
    EXPECT_FLOAT_EQ(scene.meshes[0].tri_uvs[0].uv2.y, 1.0f);
}

TEST(TexturePipeline, EnvironmentLoadsHdrOrDisables) {
    EXPECT_FALSE(LoadEnvironment("", 0.0f, 1.0f).Enabled());
    EXPECT_FALSE(LoadEnvironment("/nonexistent/sky.hdr", 0.0f, 1.0f).Enabled());
    // An LDR image is accepted and linearised (equirect fallback path).
    const std::vector<uint8_t> rgb(8 * 4 * 3, 255);
    const std::string path = WriteTestPng("env_ldr.png", 8, 4, rgb);
    const nuka::rt::EnvironmentMap env = LoadEnvironment(path, 0.5f, 2.0f);
    ASSERT_TRUE(env.Enabled());
    EXPECT_EQ(env.width, 8u);
    EXPECT_EQ(env.height, 4u);
    EXPECT_FLOAT_EQ(env.yaw, 0.5f);
    EXPECT_FLOAT_EQ(env.intensity, 2.0f);
    EXPECT_NEAR(env.texels[0], 1.0f, 1e-5f);  // white linearises to 1
}

TEST(TexturePipeline, NksRoundTripsTextureAndEnvironmentFields) {
    nuka::scene::SceneIR scene;
    nuka::scene::RigidBodyRecord body;
    body.name = "block";
    body.is_static = true;
    const nuka::scene::BodyId bid = scene.AddRigidBody(std::move(body));

    nuka::scene::MaterialRecord mat;
    mat.name = "wood";
    mat.albedo_map = "tex/albedo.jpg";
    mat.roughness_map = "tex/rough.jpg";
    mat.normal_map = "tex/normal.jpg";
    mat.uv_scale = 2.5f;
    mat.triplanar = false;
    const nuka::scene::MaterialId mid = scene.AddMaterial(std::move(mat));

    nuka::scene::CollisionShapeRecord shape;
    shape.body_id = bid;
    shape.name = "block_shape";
    shape.type = nuka::scene::ShapeType::Box;
    shape.material_id = mid;
    scene.AddCollisionShape(std::move(shape));

    scene.EnvironmentMut() =
        nuka::scene::EnvironmentRecord{"sky/env_2k.hdr", 33.0f, 1.5f, true};

    const std::string nks1 = std::string(::testing::TempDir()) + "texrt.nks";
    const std::string nks2 = std::string(::testing::TempDir()) + "texrt2.nks";
    nuka::scene::nks::Save(scene, nks1);
    const nuka::scene::SceneIR loaded = nuka::scene::nks::Load(nks1);

    ASSERT_EQ(loaded.MaterialCount(), 1u);
    const nuka::scene::MaterialRecord& lm = loaded.GetMaterial(0);
    EXPECT_EQ(lm.albedo_map, "tex/albedo.jpg");
    EXPECT_EQ(lm.roughness_map, "tex/rough.jpg");
    EXPECT_EQ(lm.normal_map, "tex/normal.jpg");
    EXPECT_FLOAT_EQ(lm.uv_scale, 2.5f);
    EXPECT_FALSE(lm.triplanar);
    EXPECT_EQ(loaded.Environment().hdri, "sky/env_2k.hdr");
    EXPECT_FLOAT_EQ(loaded.Environment().yaw_deg, 33.0f);
    EXPECT_FLOAT_EQ(loaded.Environment().intensity, 1.5f);
    EXPECT_TRUE(loaded.Environment().use_scene_materials);

    // Second save is byte-identical (the .nks determinism contract).
    nuka::scene::nks::Save(loaded, nks2);
    auto read = [](const std::string& p) {
        FILE* f = fopen(p.c_str(), "rb");
        EXPECT_NE(f, nullptr);
        std::string out;
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
        fclose(f);
        return out;
    };
    EXPECT_EQ(read(nks1), read(nks2));
}

TEST(TexturePipeline, ReportKeepsMissColorOnlyWhenAsked) {
    nuka::rt::Framebuffer fb;
    fb.width = 2; fb.height = 1;
    fb.color = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    fb.prim = {0u, 0xFFFFFFFFu};  // pixel 0 hit, pixel 1 miss
    const nuka::render::VulkanRgba8 bg{7, 8, 9, 255};

    const auto legacy = FramebufferToReport(fb, bg);
    EXPECT_EQ(legacy.pixels[1].r, 7);  // miss painted with the background
    EXPECT_EQ(legacy.non_background_pixel_count, 1u);

    const auto env = FramebufferToReport(fb, bg, /*keep_miss_color=*/true);
    EXPECT_GT(env.pixels[1].g, 200);   // miss keeps the traced (tonemapped) green
    EXPECT_EQ(env.pixels[0].r, legacy.pixels[0].r);  // hit pixels unchanged
    EXPECT_EQ(env.non_background_pixel_count, 1u);   // count semantics unchanged
}

}  // namespace
