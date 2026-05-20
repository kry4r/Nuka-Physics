// ---------------------------------------------------------------------------
// nuka::render::RenderScene implementation
// ---------------------------------------------------------------------------

#include "render/render_scene.hpp"

namespace nuka::render {

namespace {

math::Transform BodyWorldTransform(const scene::SceneIR& scene, scene::BodyId body_id) {
    if (body_id == scene::kInvalidBody || body_id >= scene.RigidBodyCount()) {
        return math::Transform::Identity();
    }

    const auto& body = scene.GetBody(body_id);
    if (body.parent_id == scene::kInvalidBody) {
        return body.local_transform;
    }
    return BodyWorldTransform(scene, body.parent_id) * body.local_transform;
}

} // namespace

RenderScene BuildRenderScene(const scene::SceneIR& scene) {
    RenderScene render;

    render.materials.reserve(scene.MaterialCount());
    for (const auto& material : scene.Materials()) {
        RenderMaterial out;
        out.material_id = material.id;
        out.base_color = material.base_color;
        out.alpha = material.alpha;
        out.roughness = material.roughness;
        out.metallic = material.metallic;
        render.materials.push_back(out);
    }

    render.mesh_instances.reserve(scene.ShapeCount());
    render.debug_proxies.reserve(scene.ShapeCount());
    for (const auto& shape : scene.Shapes()) {
        const math::Transform body_world = BodyWorldTransform(scene, shape.body_id);
        const math::Transform shape_world = body_world * shape.local_transform;

        RenderMeshInstance mesh;
        mesh.shape_id = shape.id;
        mesh.body_id = shape.body_id;
        mesh.material_id = shape.material_id;
        mesh.shape_type = shape.type;
        mesh.world_transform = shape_world;
        mesh.half_extents = shape.half_extents;
        mesh.radius = shape.radius;
        render.mesh_instances.push_back(mesh);

        DebugProxy proxy;
        proxy.body_id = shape.body_id;
        proxy.shape_id = shape.id;
        proxy.shape_type = shape.type;
        proxy.world_transform = shape_world;
        render.debug_proxies.push_back(proxy);
    }

    render.cameras.reserve(scene.CameraCount());
    for (const auto& camera : scene.Cameras()) {
        RenderCamera out;
        out.camera_id = camera.id;
        out.attached_body = camera.attached_body;
        out.world_transform = BodyWorldTransform(scene, camera.attached_body) * camera.local_transform;
        out.vertical_fov_degrees = camera.vertical_fov_degrees;
        out.near_clip = camera.near_clip;
        out.far_clip = camera.far_clip;
        render.cameras.push_back(out);
    }

    render.lights.reserve(scene.LightCount());
    for (const auto& light : scene.Lights()) {
        RenderLight out;
        out.light_id = light.id;
        out.type = light.type;
        out.attached_body = light.attached_body;
        out.world_transform = BodyWorldTransform(scene, light.attached_body) * light.local_transform;
        out.color = light.color;
        out.intensity = light.intensity;
        render.lights.push_back(out);
    }

    render.mesh_instance_count = static_cast<uint32_t>(render.mesh_instances.size());
    render.material_count = static_cast<uint32_t>(render.materials.size());
    render.camera_count = static_cast<uint32_t>(render.cameras.size());
    render.light_count = static_cast<uint32_t>(render.lights.size());
    render.debug_proxy_count = static_cast<uint32_t>(render.debug_proxies.size());
    return render;
}

} // namespace nuka::render
