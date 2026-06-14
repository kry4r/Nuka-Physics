// ---------------------------------------------------------------------------
// H1 cup demo PBR-style renderer feasibility smoke.
//
// This is the first gate for the final H1 cup video path: it proves the existing
// two-level RT renderer can render shaded geometry for a hero-style robot/cup/
// table composition without using the Go2 stick/debug renderer.
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "rt/camera.hpp"
#include "rt/framebuffer.hpp"
#include "phi/backend_cuda/rt/ray_box.cuh"
#include "rt/two_level_render.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;

rt::Material Material(math::Vec3 albedo, float metallic, float roughness) {
    rt::Material material;
    material.albedo = albedo;
    material.metallic = metallic;
    material.roughness = roughness;
    return material;
}

void AddFace(rt::BlasMesh* mesh,
             math::Vec3 v0,
             math::Vec3 v1,
             math::Vec3 v2,
             math::Vec3 v3) {
    mesh->triangles.push_back({v0, v1, v2, 0u});
    mesh->triangles.push_back({v0, v2, v3, 0u});
}

rt::BlasMesh MakeBoxMesh(math::Vec3 half_extents) {
    const float x = half_extents.x;
    const float y = half_extents.y;
    const float z = half_extents.z;
    const std::array<math::Vec3, 8> v{{
        {-x, -y, -z}, { x, -y, -z}, { x,  y, -z}, {-x,  y, -z},
        {-x, -y,  z}, { x, -y,  z}, { x,  y,  z}, {-x,  y,  z},
    }};

    rt::BlasMesh mesh;
    AddFace(&mesh, v[4], v[5], v[6], v[7]); // +Z
    AddFace(&mesh, v[0], v[3], v[2], v[1]); // -Z
    AddFace(&mesh, v[1], v[2], v[6], v[5]); // +X
    AddFace(&mesh, v[0], v[4], v[7], v[3]); // -X
    AddFace(&mesh, v[2], v[3], v[7], v[6]); // +Y
    AddFace(&mesh, v[0], v[1], v[5], v[4]); // -Y
    return mesh;
}

rt::BlasMesh MakeSphereMesh(float radius) {
    rt::BlasMesh mesh;
    mesh.spheres.push_back({{0.0f, 0.0f, 0.0f}, radius, 0u});
    return mesh;
}

void AddInstance(rt::TwoLevelScene* scene,
                 uint32_t mesh,
                 math::Vec3 position,
                 uint32_t material,
                 math::Quat rotation = math::Quat::Identity()) {
    scene->instances.push_back({mesh, {position, rotation}, material});
}

rt::TwoLevelScene BuildHeroScene() {
    rt::TwoLevelScene scene;

    const uint32_t floor_mesh = static_cast<uint32_t>(scene.meshes.size());
    scene.meshes.push_back(MakeBoxMesh({4.5f, 3.0f, 0.04f}));
    const uint32_t table_mesh = static_cast<uint32_t>(scene.meshes.size());
    scene.meshes.push_back(MakeBoxMesh({0.95f, 0.55f, 0.07f}));
    const uint32_t torso_mesh = static_cast<uint32_t>(scene.meshes.size());
    scene.meshes.push_back(MakeBoxMesh({0.20f, 0.13f, 0.34f}));
    const uint32_t limb_mesh = static_cast<uint32_t>(scene.meshes.size());
    scene.meshes.push_back(MakeBoxMesh({0.055f, 0.055f, 0.28f}));
    const uint32_t head_mesh = static_cast<uint32_t>(scene.meshes.size());
    scene.meshes.push_back(MakeSphereMesh(0.11f));
    const uint32_t joint_mesh = static_cast<uint32_t>(scene.meshes.size());
    scene.meshes.push_back(MakeSphereMesh(0.045f));
    const uint32_t cup_mesh = static_cast<uint32_t>(scene.meshes.size());
    scene.meshes.push_back(MakeSphereMesh(0.095f));

    scene.materials = {
        Material({0.055f, 0.060f, 0.070f}, 0.0f, 0.72f), // dark studio floor
        Material({0.42f, 0.30f, 0.20f}, 0.0f, 0.62f),    // matte table
        Material({0.24f, 0.27f, 0.32f}, 0.35f, 0.36f),   // H1 metal/plastic
        Material({0.08f, 0.11f, 0.16f}, 0.2f, 0.45f),    // darker limbs
        Material({0.92f, 0.88f, 0.78f}, 0.0f, 0.28f),    // cup ceramic
    };

    AddInstance(&scene, floor_mesh, {0.0f, 0.0f, -0.04f}, 0u);
    AddInstance(&scene, table_mesh, {0.58f, 0.02f, 0.39f}, 1u);

    AddInstance(&scene, torso_mesh, {-0.28f, 0.0f, 1.18f}, 2u);
    AddInstance(&scene, head_mesh, {-0.28f, 0.0f, 1.62f}, 2u);

    const auto leg_pitch = math::Quat::FromAxisAngle({1.0f, 0.0f, 0.0f}, 0.12f);
    AddInstance(&scene, limb_mesh, {-0.38f, -0.10f, 0.70f}, 3u, leg_pitch);
    AddInstance(&scene, limb_mesh, {-0.18f, -0.10f, 0.70f}, 3u, leg_pitch);
    AddInstance(&scene, limb_mesh, {-0.38f,  0.10f, 0.70f}, 3u, leg_pitch);
    AddInstance(&scene, limb_mesh, {-0.18f,  0.10f, 0.70f}, 3u, leg_pitch);

    const auto arm_reach = math::Quat::FromAxisAngle({0.0f, 1.0f, 0.0f}, 1.05f);
    AddInstance(&scene, limb_mesh, {-0.04f, -0.18f, 1.25f}, 3u, arm_reach);
    AddInstance(&scene, limb_mesh, { 0.22f, -0.18f, 1.02f}, 3u, arm_reach);
    AddInstance(&scene, joint_mesh, {0.42f, -0.17f, 0.88f}, 2u);

    AddInstance(&scene, cup_mesh, {0.56f, -0.12f, 0.58f}, 4u);

    scene.light.directional = false;
    scene.light.position = {-1.8f, -2.2f, 3.0f};
    scene.light.color = {1.0f, 0.94f, 0.86f};
    scene.light.intensity = 7.5f;
    scene.ambient.color = {0.045f, 0.050f, 0.060f};
    return scene;
}

struct FrameStats {
    size_t hit_pixels = 0;
    size_t floor_pixels = 0;
    size_t table_pixels = 0;
    size_t robot_pixels = 0;
    size_t cup_pixels = 0;
    size_t dark_floor_pixels = 0;
    size_t lit_floor_pixels = 0;
    size_t center_region_hits = 0;
    size_t table_region_pixels = 0;
    size_t robot_region_pixels = 0;
    size_t cup_region_pixels = 0;
    float min_luma = 1.0e30f;
    float max_luma = -1.0e30f;
};

FrameStats CollectStats(const rt::Framebuffer& framebuffer) {
    FrameStats stats;
    for (size_t pixel = 0; pixel < framebuffer.Pixels(); ++pixel) {
        if (framebuffer.prim[pixel] == rt::kNoPrim) {
            continue;
        }
        ++stats.hit_pixels;
        const float r = framebuffer.color[pixel * 3u + 0u];
        const float g = framebuffer.color[pixel * 3u + 1u];
        const float b = framebuffer.color[pixel * 3u + 2u];
        const float luma = r + g + b;
        stats.min_luma = std::min(stats.min_luma, luma);
        stats.max_luma = std::max(stats.max_luma, luma);

        const uint32_t x = static_cast<uint32_t>(pixel % framebuffer.width);
        const uint32_t y = static_cast<uint32_t>(pixel / framebuffer.width);
        if (x >= framebuffer.width * 3u / 8u && x < framebuffer.width * 5u / 8u &&
            y >= framebuffer.height * 3u / 8u && y < framebuffer.height * 5u / 8u) {
            ++stats.center_region_hits;
        }
        const auto albedo = framebuffer.AlbedoAt(x, y);
        if (albedo.x < 0.07f && albedo.y < 0.08f && albedo.z < 0.09f) {
            ++stats.floor_pixels;
            if (luma < 0.06f) {
                ++stats.dark_floor_pixels;
            }
            if (luma > 0.10f) {
                ++stats.lit_floor_pixels;
            }
        } else if (albedo.x > 0.35f && albedo.x < 0.50f &&
                   albedo.y > 0.24f && albedo.y < 0.36f &&
                   albedo.z > 0.15f && albedo.z < 0.25f) {
            ++stats.table_pixels;
            if (x >= framebuffer.width / 3u && x < framebuffer.width * 7u / 8u &&
                y >= framebuffer.height / 3u && y < framebuffer.height * 7u / 8u) {
                ++stats.table_region_pixels;
            }
        } else if (albedo.x > 0.86f && albedo.y > 0.80f && albedo.z > 0.70f) {
            ++stats.cup_pixels;
            if (x >= framebuffer.width / 3u && x < framebuffer.width * 7u / 8u &&
                y >= framebuffer.height / 4u && y < framebuffer.height * 3u / 4u) {
                ++stats.cup_region_pixels;
            }
        } else {
            ++stats.robot_pixels;
            if (x >= framebuffer.width / 4u && x < framebuffer.width * 3u / 4u &&
                y >= framebuffer.height / 5u && y < framebuffer.height * 4u / 5u) {
                ++stats.robot_region_pixels;
            }
        }
    }
    return stats;
}

} // namespace

TEST(H1CupPbrRender, SmokeHeroScene) {
    rt::TwoLevelScene scene = BuildHeroScene();
    auto camera = rt::BuildPinhole({-1.55f, -2.25f, 1.18f},
                                   {0.18f, -0.02f, 0.82f},
                                   {0.0f, 0.0f, 1.0f},
                                   0.33f * kPi,
                                   160u,
                                   90u);

    auto device_scene = rt::BuildTwoLevelScene(scene);
    const rt::Framebuffer frame = rt::RenderFrame(device_scene, scene, camera);
    const FrameStats stats = CollectStats(frame);

    std::printf("[h1-cup-pbr-smoke] hits=%zu floor=%zu table=%zu robot=%zu cup=%zu "
                "dark_floor=%zu lit_floor=%zu center_region=%zu "
                "regions(table=%zu robot=%zu cup=%zu) luma=[%.6f, %.6f]\n",
                stats.hit_pixels, stats.floor_pixels, stats.table_pixels,
                stats.robot_pixels, stats.cup_pixels, stats.dark_floor_pixels,
                stats.lit_floor_pixels, stats.center_region_hits,
                stats.table_region_pixels, stats.robot_region_pixels,
                stats.cup_region_pixels, stats.min_luma, stats.max_luma);

    ASSERT_EQ(frame.width, 160u);
    ASSERT_EQ(frame.height, 90u);
    ASSERT_EQ(frame.color.size(), frame.Pixels() * 3u);
    ASSERT_EQ(frame.depth.size(), frame.Pixels());
    ASSERT_EQ(frame.normal.size(), frame.Pixels() * 3u);
    ASSERT_EQ(frame.albedo.size(), frame.Pixels() * 3u);
    ASSERT_EQ(frame.prim.size(), frame.Pixels());

    EXPECT_GT(stats.hit_pixels, frame.Pixels() / 6u) << "hero camera sees too little geometry";
    EXPECT_GT(stats.floor_pixels, 100u) << "studio floor is not visible";
    EXPECT_GT(stats.table_pixels, 40u) << "table is not visible";
    EXPECT_GT(stats.robot_pixels, 120u) << "H1 proxy is not visible";
    EXPECT_GT(stats.cup_pixels, 8u) << "cup proxy is not visible";
    EXPECT_GT(stats.table_region_pixels, 20u) << "table is outside the hero camera region";
    EXPECT_GT(stats.robot_region_pixels, 80u) << "H1 proxy is outside the hero camera region";
    EXPECT_GT(stats.cup_region_pixels, 4u) << "cup proxy is outside the hero camera region";
    EXPECT_GT(stats.max_luma - stats.min_luma, 0.08f) << "image lacks shaded contrast";
    EXPECT_GT(stats.dark_floor_pixels, 8u) << "no dark floor/contact-shadow-like pixels";
    EXPECT_GT(stats.lit_floor_pixels, 8u) << "floor has no lit contrast against dark pixels";

    EXPECT_GT(stats.center_region_hits, 20u) << "hero composition center region is blank";
}
