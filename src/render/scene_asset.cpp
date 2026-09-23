#include "render/scene_asset.hpp"

#include "render/texture_image.hpp"
#include "scene/scene_map.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace nuka::render {
namespace {

void CopyColor(float* destination, math::Vec3 color) {
    destination[0] = color.x;
    destination[1] = color.y;
    destination[2] = color.z;
}

std::string Resolve(const std::string& path, const std::string& directory) {
    if (path.empty()) return {};
    return (std::filesystem::path(directory) / path).lexically_normal().string();
}

void ValidatePlacement(const math::Transform& placement) {
    if (!std::isfinite(placement.position.LengthSq()) ||
        !std::isfinite(placement.rotation.Norm()) ||
        std::fabs(placement.rotation.Norm() - 1.0f) > 1e-4f)
        throw std::invalid_argument("Render asset placement requires a finite rigid transform");
}

}  // namespace

SceneRenderAsset BuildSceneRenderAsset(const scene::SceneIR& scene,
                                      const std::string& asset_directory) {
    for (const auto& body : scene.Bodies()) {
        if (!body.is_static)
            throw std::invalid_argument("Dynamic render assets must be loaded through the physics scene");
    }
    for (const auto& shape : scene.Shapes()) {
        if (shape.contype != 0u || shape.conaffinity != 0u)
            throw std::invalid_argument("Colliding render assets must be loaded through the physics scene");
    }
    if (!scene.Joints().empty() || !scene.Media().empty() || !scene.Terrain().empty())
        throw std::invalid_argument("Simulated render assets must be loaded through the physics scene");

    SceneRenderAsset asset;
    asset.world = BuildRenderWorld(scene.Ecs(), scene::SceneMap{});
    asset.environment = scene.Environment();
    for (const auto& material : scene.Materials()) {
        if (!asset.materials.emplace(material.name, material.id).second)
            throw std::invalid_argument("Render asset has duplicate material name: " + material.name);
    }
    for (uint32_t i = 0u; i < asset.world.cameras.size(); ++i) {
        const auto node = scene.Ecs().NodeOf(asset.world.cameras[i].entity);
        const std::string path = scene.Tree().PathOf(node);
        asset.cameras.emplace(path, i);
    }
    for (auto& material : asset.world.materials) {
        material.albedo_map = Resolve(material.albedo_map, asset_directory);
        material.roughness_map = Resolve(material.roughness_map, asset_directory);
        material.normal_map = Resolve(material.normal_map, asset_directory);
    }
    DecodeMaterialTextures(asset.world);
    if (!asset.environment.hdri.empty()) {
        constexpr float to_radians = 3.14159265358979323846f / 180.0f;
        asset.world.environment = LoadEnvironment(Resolve(asset.environment.hdri, asset_directory),
            asset.environment.yaw_deg * to_radians, asset.environment.intensity);
        if (!asset.world.environment.Enabled())
            throw std::runtime_error("Cannot load environment: " + asset.environment.hdri);
    }
    return asset;
}

void SetSceneRenderAsset(RenderWorld& world, RenderAssetBinding& binding,
                         const SceneRenderAsset& asset, const math::Transform& placement) {
    ValidatePlacement(placement);
    const auto& source = asset.world;
    if (binding.initialized &&
        (binding.meshes.size() != source.meshes.Count() ||
         binding.materials.size() != source.materials.size() ||
         binding.instances.size() != source.instances.size() ||
         binding.cameras.size() != source.cameras.size() ||
         binding.lights.size() != source.lights.size()))
        throw std::invalid_argument("Render asset topology changed; rebuild the render scene");

    if (!binding.initialized) {
        for (uint32_t i = 0u; i < source.meshes.Count(); ++i)
            binding.meshes.push_back(world.meshes.AppendGeometry(source.meshes.Geometry(i), source.meshes.Source(i)));
        for (const auto& material : source.materials) {
            binding.materials.push_back(static_cast<uint32_t>(world.materials.size()));
            world.materials.push_back(material);
        }
        const auto allocate = [](auto& slots, auto& destination, std::size_t count) {
            for (std::size_t i = 0u; i < count; ++i) {
                slots.push_back(destination.size());
                destination.emplace_back();
            }
        };
        allocate(binding.instances, world.instances, source.instances.size());
        allocate(binding.cameras, world.cameras, source.cameras.size());
        allocate(binding.lights, world.lights, source.lights.size());
        binding.initialized = true;
    } else {
        for (uint32_t i = 0u; i < source.meshes.Count(); ++i)
            world.meshes.ReplaceGeometry(binding.meshes.at(i), source.meshes.Geometry(i));
        for (std::size_t i = 0u; i < source.materials.size(); ++i)
            world.materials.at(binding.materials.at(i)) = source.materials[i];
    }
    for (std::size_t i = 0u; i < source.instances.size(); ++i) {
        auto instance = source.instances[i];
        instance.entity = scene::kInvalidEntity;
        instance.mesh_id = binding.meshes.at(instance.mesh_id);
        instance.render_material_id = binding.materials.at(instance.render_material_id);
        instance.world_xform = placement * instance.world_xform;
        world.instances.at(binding.instances.at(i)) = instance;
    }
    for (std::size_t i = 0u; i < source.cameras.size(); ++i) {
        auto camera = source.cameras[i];
        camera.entity = scene::kInvalidEntity;
        camera.world_xform = placement * camera.world_xform;
        world.cameras.at(binding.cameras.at(i)) = camera;
    }
    for (std::size_t i = 0u; i < source.lights.size(); ++i) {
        auto light = source.lights[i];
        light.entity = scene::kInvalidEntity;
        light.world_xform = placement * light.world_xform;
        world.lights.at(binding.lights.at(i)) = light;
    }
    if (asset.environment.Authored()) world.environment = source.environment;
    DecodeMaterialTextures(world);
}

void ApplySceneEnvironment(RasterOptions& options, const scene::EnvironmentRecord& environment,
                           const math::Transform& placement) {
    ValidatePlacement(placement);
    options.beauty_exposure_ev = environment.exposure_ev;
    options.beauty_grade = environment.grade;
    options.beauty_specular_env = environment.specular_env;
    options.beauty_sky_fill = environment.ibl_full_fill && !environment.hdri.empty() ?
        environment.intensity : 0.30f;
    if (environment.sky.enabled) {
        const auto& sky = environment.sky;
        for (const auto color : {sky.top, sky.bottom, sky.ground, sky.ambient_sky,
                                  sky.ambient_ground, sky.background}) {
            if (!std::isfinite(color.LengthSq()))
                throw std::invalid_argument("Scene sky colors must be finite");
        }
        if (!std::isfinite(sky.fill) || sky.fill < 0.0f)
            throw std::invalid_argument("Scene sky fill must be finite and nonnegative");
        options.sky_gradient = true;
        CopyColor(options.sky_top, sky.top);
        CopyColor(options.sky_bottom, sky.bottom);
        CopyColor(options.ground_color, sky.ground);
        CopyColor(options.sun_ambient_sky, sky.ambient_sky);
        CopyColor(options.sun_ambient_ground, sky.ambient_ground);
        options.beauty_sky_fill = sky.fill;
        const auto byte = [](float value) {
            return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
        };
        options.background = {byte(sky.background.x), byte(sky.background.y), byte(sky.background.z), 255u};
    }
    for (std::size_t i = 0u; i < 3u; ++i)
        options.beauty_sun_disc[i] = options.sun_color[i] * environment.sun_disc;
    if (environment.shadow.enabled) {
        if (!(environment.shadow.radius > 0.0f) || !std::isfinite(environment.shadow.radius) ||
            !std::isfinite(environment.shadow.center.LengthSq()))
            throw std::invalid_argument("Scene shadow bounds must be finite with a positive radius");
        options.shadow_center = placement.TransformPoint(environment.shadow.center);
        options.shadow_radius = environment.shadow.radius;
        const auto& shadow = environment.shadow;
        if (shadow.map_size) options.shadow_map_size = *shadow.map_size;
        if (shadow.strength) options.shadow_strength = *shadow.strength;
        if (shadow.bias) options.shadow_bias = *shadow.bias;
        if (shadow.filter_radius) options.shadow_filter_radius = *shadow.filter_radius;
        if (options.shadow_map_size == 0u || !std::isfinite(options.shadow_strength) ||
            options.shadow_strength < 0.0f || options.shadow_strength > 1.0f ||
            !std::isfinite(options.shadow_bias) || options.shadow_bias < 0.0f ||
            !std::isfinite(options.shadow_filter_radius) || options.shadow_filter_radius < 0.0f)
            throw std::invalid_argument("Scene shadow sampling parameters are invalid");
    }
}

void ApplySceneLighting(RasterOptions& options, const SceneRenderAsset& asset,
                        const math::Transform& placement) {
    ValidatePlacement(placement);
    options.use_sun_light = false;
    for (const auto& light : asset.world.lights) {
        if (light.type != scene::LightComponent::Type::Directional) continue;
        const auto pose = placement * light.world_xform;
        CopyColor(options.sun_direction, pose.TransformDirection({0.0f, 0.0f, 1.0f}).Normalized());
        CopyColor(options.sun_color, light.color * light.intensity);
        options.use_sun_light = true;
        break;
    }
    options.draw_ground = false;
    ApplySceneEnvironment(options, asset.environment, placement);
}

void ApplySceneCamera(RasterOptions& options, const SceneRenderAsset& asset,
                      const std::string& name, const math::Transform& placement) {
    ValidatePlacement(placement);
    const auto found = asset.cameras.find(name);
    if (found == asset.cameras.end()) throw std::invalid_argument("Unknown scene camera: " + name);
    const auto& camera = asset.world.cameras.at(found->second);
    const auto pose = placement * camera.world_xform;
    ValidatePlacement(pose);
    if (!(camera.vertical_fov_degrees > 0.0f && camera.vertical_fov_degrees < 180.0f) ||
        !(camera.near_clip > 0.0f && camera.far_clip > camera.near_clip) ||
        !std::isfinite(camera.far_clip) || !(camera.focus_distance > 0.0f) ||
        !std::isfinite(camera.focus_distance) || camera.shadow_radius < 0.0f ||
        !std::isfinite(camera.shadow_radius))
        throw std::invalid_argument("Scene camera requires a valid field of view and clipping range");
    options.use_camera_override = true;
    options.camera_eye = pose.position;
    options.camera_target = pose.TransformPoint({0.0f, 0.0f, -camera.focus_distance});
    options.camera_up = pose.TransformDirection({0.0f, 1.0f, 0.0f});
    options.camera_fov_degrees = camera.vertical_fov_degrees;
    options.camera_near = camera.near_clip;
    options.camera_far = camera.far_clip;
    options.shadow_center = asset.environment.shadow.enabled ?
        placement.TransformPoint(asset.environment.shadow.center) : math::Vec3{};
    options.shadow_radius = asset.environment.shadow.enabled ? asset.environment.shadow.radius : 0.0f;
    if (camera.shadow_radius > 0.0f) {
        options.shadow_center = options.camera_target;
        options.shadow_radius = camera.shadow_radius;
    }
}

uint32_t SceneAssetMaterial(const SceneRenderAsset& asset, const RenderAssetBinding& binding,
                            const std::string& material) {
    const auto found = asset.materials.find(material);
    if (found == asset.materials.end()) throw std::invalid_argument("Unknown scene material: " + material);
    return binding.materials.at(found->second);
}

}  // namespace nuka::render
