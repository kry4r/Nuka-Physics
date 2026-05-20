#pragma once
// ---------------------------------------------------------------------------
// nuka::render::RenderScene - render-facing compiled scene view
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "scene/canonical_types.hpp"
#include "scene/scene_ir.hpp"

#include <cstdint>
#include <vector>

namespace nuka::render {

struct RenderMeshInstance {
    scene::ShapeId shape_id = 0;
    scene::BodyId body_id = scene::kInvalidBody;
    scene::MaterialId material_id = scene::kInvalidMaterial;
    scene::ShapeType shape_type = scene::ShapeType::Box;
    math::Transform local_transform = math::Transform::Identity();
    math::Transform world_transform = math::Transform::Identity();
    math::Vec3 half_extents = {0.5f, 0.5f, 0.5f};
    float radius = 0.5f;
    float half_height = 0.5f;
};

struct RenderMaterial {
    scene::MaterialId material_id = scene::kInvalidMaterial;
    math::Vec3 base_color = {1.0f, 1.0f, 1.0f};
    float alpha = 1.0f;
    float roughness = 0.5f;
    float metallic = 0.0f;
};

struct RenderCamera {
    scene::CameraId camera_id = 0;
    scene::BodyId attached_body = scene::kInvalidBody;
    math::Transform local_transform = math::Transform::Identity();
    math::Transform world_transform = math::Transform::Identity();
    float vertical_fov_degrees = 45.0f;
    float near_clip = 0.01f;
    float far_clip = 1000.0f;
};

struct RenderLight {
    scene::LightId light_id = 0;
    scene::LightType type = scene::LightType::Point;
    scene::BodyId attached_body = scene::kInvalidBody;
    math::Transform local_transform = math::Transform::Identity();
    math::Transform world_transform = math::Transform::Identity();
    math::Vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
};

struct DebugProxy {
    scene::BodyId body_id = scene::kInvalidBody;
    scene::ShapeId shape_id = 0;
    scene::ShapeType shape_type = scene::ShapeType::Box;
    math::Transform local_transform = math::Transform::Identity();
    math::Transform world_transform = math::Transform::Identity();
    math::Vec3 half_extents = {0.5f, 0.5f, 0.5f};
    float radius = 0.5f;
    float half_height = 0.5f;
};

struct RenderScene {
    uint32_t mesh_instance_count = 0;
    uint32_t material_count = 0;
    uint32_t camera_count = 0;
    uint32_t light_count = 0;
    uint32_t debug_proxy_count = 0;

    std::vector<RenderMeshInstance> mesh_instances;
    std::vector<RenderMaterial> materials;
    std::vector<RenderCamera> cameras;
    std::vector<RenderLight> lights;
    std::vector<DebugProxy> debug_proxies;
};

RenderScene BuildRenderScene(const scene::SceneIR& scene);

} // namespace nuka::render
