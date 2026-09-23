#include <gtest/gtest.h>

#ifdef NK_BUILD_VULKAN_VALIDATION

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "nk/pipeline/world.hpp"
#include "render/scene_asset.hpp"
#include "render/studio_beauty.hpp"
#include "render/rt_adapter.hpp"
#include "scene/format/nks.hpp"
#include "runtime/app/pose_publisher.hpp"
#include "runtime/app/simulation.hpp"
#include "../../tools/perf/robot_cloth_fluid_scene.hpp"

namespace {
namespace nk = nuka::nk;
namespace phi = nuka::phi;
namespace app = nuka::runtime::app;
namespace render = nuka::render;
namespace fixture = nuka::perf::fixture;
using nuka::math::Transform;
using nuka::math::Vec3;

void CheckPublishedPoses(nk::World& physics, uint32_t env,
                         const render::RenderWorld& scene) {
    const auto& caps = physics.GetModel().capacities;
    uint32_t links = 0u, bodies = 0u;
    for (const auto& instance : scene.instances) {
        nk::FieldId field;
        uint64_t row = env;
        switch (instance.pose_source.kind) {
        case render::PoseSource::Kind::Static: continue;
        case render::PoseSource::Kind::Link:
            ASSERT_LT(instance.pose_source.row, caps.links_per_env);
            field = nk::FieldId::LinkPose;
            row = uint64_t{env} * caps.links_per_env + instance.pose_source.row;
            ++links;
            break;
        case render::PoseSource::Kind::Body:
            ASSERT_LT(instance.pose_source.row, caps.bodies_per_env);
            field = nk::FieldId::BodyPose;
            row = uint64_t{env} * caps.bodies_per_env + instance.pose_source.row;
            ++bodies;
            break;
        case render::PoseSource::Kind::Base:
            field = nk::FieldId::BasePose;
            break;
        }
        Transform pose;
        ASSERT_TRUE(physics.GetData().DownloadField(field, &pose, sizeof(pose), row * sizeof(pose)));
        const Transform expected = pose * instance.cached_visual_local;
        EXPECT_EQ(std::memcmp(&instance.world_xform, &expected, sizeof(expected)), 0)
            << "pose row " << instance.pose_source.row << ", environment " << env;
    }
    EXPECT_GT(links, 0u);
    EXPECT_GT(bodies, 0u);
}
}  // namespace

TEST(RenderPhysicsParity, PhysicsRenderParityRenderDeterminismAndCoverage) {
    const auto path = std::filesystem::path(NUKA_SOURCE_DIR) / "examples/scenes/go2_stand.usda";
    ASSERT_TRUE(std::filesystem::exists(path));
    auto* device = phi::InitBestDevice();
    if (!device) GTEST_SKIP() << "no physics device";
    auto* backend = phi::DeviceInitBackend(device, nullptr);
    ASSERT_NE(backend, nullptr);

    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    try {
        renderer = std::make_unique<render::VulkanRasterRenderer>();
    } catch (const std::exception& error) {
        GTEST_SKIP() << "no Vulkan graphics device: " << error.what();
    }

    const auto prepared = fixture::Prepare(path, device, backend, fixture::Cfg());
    fixture::SceneVisuals visuals;
    auto model = fixture::CookMpmPrepared(prepared, 2u, &visuals);
    auto config = fixture::Cfg();
    config.dt = 0.001f;
    config.vel_iters = 128u;
    nk::World physics(std::move(model), 2u, device, backend, config);
    ASSERT_TRUE(physics.Ready()) << physics.CreationError();
    const auto& caps = physics.GetModel().capacities;
    constexpr uint32_t selected_env = 1u;

    uint32_t moved_body = 0u;
    while (moved_body < caps.bodies_per_env &&
           physics.GetModel().body_init[moved_body].inv_mass == 0.0f) ++moved_body;
    ASSERT_LT(moved_body, caps.bodies_per_env);
    const uint64_t moved_offset = (uint64_t{selected_env} * caps.bodies_per_env + moved_body) * sizeof(Transform);
    Transform displaced;
    ASSERT_TRUE(physics.GetData().DownloadField(nk::FieldId::BodyPose, &displaced, sizeof(displaced), moved_offset));
    displaced.position.y += 0.005f;
    ASSERT_TRUE(physics.GetData().UploadField(nk::FieldId::BodyPose, &displaced, sizeof(displaced), moved_offset));

    app::HostDownloadPublisher publisher;
    app::Simulation simulation(physics, publisher,
        render::BuildRenderWorld(visuals.scene.Ecs(), visuals.scene_map));
    simulation.SetEnvIndex(selected_env + caps.env_count);
    render::RasterOptions options;
    options.width = 640u;
    options.height = 400u;
    const auto asset_path = std::filesystem::path(NUKA_SOURCE_DIR) / "examples/assets/nuka_lab/gripper.nks";
    auto asset_scene = nuka::scene::nks::Load(asset_path.string());
    const auto asset = render::BuildSceneRenderAsset(asset_scene, asset_path.parent_path().string());
    ASSERT_FALSE(asset.world.instances.empty());
    for (uint32_t mesh = 0u; mesh < asset.world.meshes.Count(); ++mesh)
        ASSERT_EQ(asset.world.meshes.Source(mesh), render::MeshSource::NkaMesh);
    render::ApplySceneCamera(options, asset, "three-quarter");
    simulation.EnableRendering(renderer.get(), options);
    for (uint32_t frame = 0u; frame < 4u; ++frame) {
        ASSERT_TRUE(simulation.Frame()) << "frame " << frame;
        ASSERT_TRUE(simulation.LastFrameRendered());
    }
    CheckPublishedPoses(physics, selected_env, simulation.GetRenderWorld());

    std::vector<Transform> links(caps.links_per_env), bodies(caps.bodies_per_env);
    std::vector<Vec3> positions(caps.particles_per_env);
    const auto download = [&](nk::FieldId field, auto& values) {
        const uint64_t bytes = values.size() * sizeof(values[0]);
        return physics.GetData().DownloadField(field, values.data(), bytes, selected_env * bytes);
    };
    ASSERT_TRUE(download(nk::FieldId::LinkPose, links));
    ASSERT_TRUE(download(nk::FieldId::BodyPose, bodies));
    ASSERT_TRUE(download(nk::FieldId::ParticlePos, positions));

    std::vector<nuka::runtime::soft::SurfaceTopology> topologies;
    for (const auto& info : visuals.material_surfaces) {
        nuka::runtime::soft::SurfaceTopology topology;
        topology.triangles = info.triangles;
        topologies.push_back(std::move(topology));
    }
    const auto& particles = physics.GetModel().particles;
    for (const auto& info : particles.surface_info) {
        nuka::runtime::soft::SurfaceTopology topology;
        const auto first = particles.surface_triangles.begin() + 3u * info.triangle_offset;
        topology.triangles.assign(first, first + 3u * info.triangle_count);
        for (auto& vertex : topology.triangles) vertex += info.vertex_offset;
        topologies.push_back(std::move(topology));
    }
    ASSERT_GE(topologies.size(), 2u);
    auto studio = render::BuildStudioScene(visuals.scene.Ecs(), visuals.scene_map,
        topologies, options.width, options.height, false);
    render::PublishStudioScene(studio, links, positions, bodies);
    render::RenderAssetBinding binding;
    render::SetSceneRenderAsset(studio.world, binding, asset);
    render::ApplySceneLighting(studio.options, asset);
    EXPECT_FLOAT_EQ(studio.options.shadow_bias, 0.0002f);
    EXPECT_FLOAT_EQ(studio.options.shadow_filter_radius, 1.0f);
    render::ApplySceneCamera(studio.options, asset, "close");
    EXPECT_FLOAT_EQ(studio.options.shadow_radius, 0.21f);
    render::ApplySceneCamera(studio.options, asset, "three-quarter");
    EXPECT_FLOAT_EQ(studio.options.shadow_radius, asset.environment.shadow.radius);
    const auto instance_count = studio.world.InstanceCount();
    const auto mesh_count = studio.world.meshes.Count();
    const auto material_count = studio.world.materials.size();
    const auto camera_count = studio.world.CameraCount();
    const auto light_count = studio.world.LightCount();
    const auto original = renderer->Render(studio.world, studio.options);

    const auto avocado = asset.materials.at("avocado");
    asset_scene.GetMaterialMut(avocado).base_color = {0.70f, 0.08f, 0.04f};
    asset_scene.GetBodyMut(0u).local_transform.position.x += 0.08f;
    const auto output = std::filesystem::path(NUKA_SOURCE_DIR) / ".nuka-runs/render_asset_roundtrip";
    std::filesystem::create_directories(output / "first");
    std::filesystem::create_directories(output / "second");
    {
        std::ofstream(output / "cycle_a.nks") << R"({"imports":[{"file":"cycle_b.nks"}]})";
        std::ofstream(output / "cycle_b.nks") << R"({"imports":[{"file":"cycle_a.nks"}]})";
        EXPECT_THROW(nuka::scene::nks::Load((output / "cycle_a.nks").string()), std::runtime_error);
    }
    const auto first_path = output / "first/environment.nks";
    const auto second_path = output / "second/environment.nks";
    nuka::scene::nks::Save(asset_scene, first_path.string());
    const auto reloaded = nuka::scene::nks::Load(first_path.string());
    nuka::scene::nks::Save(reloaded, second_path.string());
    const auto read = [](const std::filesystem::path& file) {
        std::ifstream stream(file, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    };
    EXPECT_EQ(read(first_path), read(second_path));
    EXPECT_EQ(read(output / "first/environment.nka"), read(output / "second/environment.nka"));
    ASSERT_EQ(reloaded.ShapeCount(), asset_scene.ShapeCount());
    EXPECT_TRUE(reloaded.Environment().sky.enabled);
    EXPECT_FLOAT_EQ(reloaded.Environment().sky.fill, asset.environment.sky.fill);
    EXPECT_FLOAT_EQ(reloaded.GetMaterial(avocado).base_color.x, 0.70f);
    EXPECT_FLOAT_EQ(reloaded.GetBody(0u).local_transform.position.x, 0.43f);
    const auto edited = render::BuildSceneRenderAsset(reloaded, first_path.parent_path().string());
    render::SetSceneRenderAsset(studio.world, binding, edited);
    const auto changed = renderer->Render(studio.world, studio.options);
    ASSERT_EQ(changed.pixels.size(), original.pixels.size());
    EXPECT_NE(std::memcmp(changed.pixels.data(), original.pixels.data(),
        changed.pixels.size() * sizeof(render::VulkanRgba8)), 0);
    render::SetSceneRenderAsset(studio.world, binding, asset);
    render::PublishStudioScene(studio, links, positions, bodies);
    EXPECT_EQ(studio.world.InstanceCount(), instance_count);
    EXPECT_EQ(studio.world.meshes.Count(), mesh_count);
    EXPECT_EQ(studio.world.materials.size(), material_count);
    EXPECT_EQ(studio.world.CameraCount(), camera_count);
    EXPECT_EQ(studio.world.LightCount(), light_count);
    const auto rt_scene = render::RenderWorldToTwoLevelScene(asset.world);
    ASSERT_TRUE(rt_scene.light.directional);
    EXPECT_LT(rt_scene.light.direction.z, 0.0f);
    EXPECT_FLOAT_EQ(rt_scene.light.color.x, 3.4f);
    CheckPublishedPoses(physics, selected_env, studio.world);

    for (const auto& surface : studio.surfaces) {
        const auto& mesh = studio.world.meshes.Geometry(surface.mesh_id);
        EXPECT_EQ(mesh.indices, surface.topology.triangles);
        ASSERT_EQ(mesh.positions.size(), positions.size() * 3u);
        for (uint32_t vertex : mesh.indices) {
            ASSERT_LT(vertex, positions.size());
            EXPECT_EQ(mesh.positions[3u * vertex], positions[vertex].x);
            EXPECT_EQ(mesh.positions[3u * vertex + 1u], positions[vertex].y);
            EXPECT_EQ(mesh.positions[3u * vertex + 2u], positions[vertex].z);
        }
    }

    const auto first = renderer->Render(studio.world, studio.options);
    const auto second = renderer->Render(studio.world, studio.options);
    ASSERT_FALSE(first.pixels.empty());
    ASSERT_EQ(first.pixels.size(), second.pixels.size());
    EXPECT_EQ(std::memcmp(first.pixels.data(), second.pixels.data(),
        first.pixels.size() * sizeof(render::VulkanRgba8)), 0);
    EXPECT_EQ(std::memcmp(first.pixels.data(), original.pixels.data(),
        first.pixels.size() * sizeof(render::VulkanRgba8)), 0);
    EXPECT_GT(first.non_background_pixel_count, 0u);
    std::printf("[RENDER-PARITY] instances=%u surfaces=%zu env=%u non_bg=%zu ICD=\"%s\"\n",
        instance_count, studio.surfaces.size(), selected_env,
        first.non_background_pixel_count, renderer->DeviceName().c_str());
}

#else

TEST(RenderPhysicsParity, RequiresVulkanValidationBuild) {
    GTEST_SKIP() << "requires -DNK_BUILD_VULKAN_VALIDATION";
}

#endif
