#pragma once

#include "render/raster/vulkan_raster_renderer.hpp"
#include "scene/scene_ir.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace nuka::render {

struct SceneRenderAsset {
    RenderWorld world;
    scene::EnvironmentRecord environment;
    std::unordered_map<std::string, uint32_t> materials;
    std::unordered_map<std::string, uint32_t> cameras;
};

struct RenderAssetBinding {
    bool initialized = false;
    std::vector<uint32_t> meshes, materials;
    std::vector<std::size_t> instances, cameras, lights;
};

// Render assets contain static visuals. Simulated objects belong to the cooked scene.
SceneRenderAsset BuildSceneRenderAsset(const scene::SceneIR& scene,
                                      const std::string& asset_directory = {});

// Update geometry, materials and placement in stable slots; topology changes require a rebuild.
void SetSceneRenderAsset(RenderWorld& world, RenderAssetBinding& binding,
                         const SceneRenderAsset& asset,
                         const math::Transform& placement = math::Transform::Identity());

void ApplySceneEnvironment(RasterOptions& options, const scene::EnvironmentRecord& environment,
                           const math::Transform& placement = math::Transform::Identity());
void ApplySceneLighting(RasterOptions& options, const SceneRenderAsset& asset,
                        const math::Transform& placement = math::Transform::Identity());
void ApplySceneCamera(RasterOptions& options, const SceneRenderAsset& asset,
                      const std::string& camera,
                      const math::Transform& placement = math::Transform::Identity());
uint32_t SceneAssetMaterial(const SceneRenderAsset& asset, const RenderAssetBinding& binding,
                            const std::string& material);

}  // namespace nuka::render
