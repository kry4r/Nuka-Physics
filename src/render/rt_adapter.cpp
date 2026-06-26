// ---------------------------------------------------------------------------
// nuka::render::RenderWorldToTwoLevelScene -- impl (M11 RT-4). HOST, CUDA-FREE.
// See rt_adapter.hpp for the contract. Compiles into nuka_render (zero CUDA).
// ---------------------------------------------------------------------------

#include "render/rt_adapter.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/ecs/components.hpp"  // scene::RenderMaterial

#include <cmath>
#include <cstdint>

namespace nuka::render {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// One MeshLibrary geometry -> one BLAS of LOCAL-space triangles (the instance
// transform places it; the two-level tracer transforms the ray to local). An
// empty / non-triangulated mesh yields an empty (valid, no-hit) BLAS so blas_id
// stays 1:1 with the MeshLibrary id.
rt::BlasMesh MeshToBlas(const MeshGeometry& geo) {
    rt::BlasMesh blas;
    const uint32_t tri_count = geo.TriangleCount();
    blas.triangles.reserve(tri_count);
    // Emit per-vertex normals into the BLAS only when the geometry ships them; an
    // empty tri_normals leaves the beauty path on the flat-normal byte-exact fallback.
    const bool has_normals = geo.normals.size() == geo.positions.size() &&
                             !geo.normals.empty();
    if (has_normals) {
        blas.tri_normals.reserve(tri_count);
    }
    for (uint32_t t = 0; t < tri_count; ++t) {
        const uint32_t i0 = geo.indices[t * 3u + 0u];
        const uint32_t i1 = geo.indices[t * 3u + 1u];
        const uint32_t i2 = geo.indices[t * 3u + 2u];
        const auto vert = [&geo](uint32_t i) -> math::Vec3 {
            return {geo.positions[i * 3u + 0u],
                    geo.positions[i * 3u + 1u],
                    geo.positions[i * 3u + 2u]};
        };
        // material_id on the prim is UNUSED in G1-A (material is per-instance);
        // 0 matches the in-tree BlasMesh builders.
        blas.triangles.push_back({vert(i0), vert(i1), vert(i2), 0u});
        if (has_normals) {
            const auto nrm = [&geo](uint32_t i) -> math::Vec3 {
                return {geo.normals[i * 3u + 0u],
                        geo.normals[i * 3u + 1u],
                        geo.normals[i * 3u + 2u]};
            };
            blas.tri_normals.push_back({nrm(i0), nrm(i1), nrm(i2)});
        }
    }
    return blas;
}

// scene::RenderMaterial -> rt::Material (Lambert+GGX). base_color.rgb -> albedo,
// metallic/roughness pass through, emissive -> emission.
rt::Material RenderMaterialToRt(const scene::RenderMaterial& m) {
    rt::Material out;
    out.albedo = {m.base_color[0], m.base_color[1], m.base_color[2]};
    out.metallic = m.metallic;
    out.roughness = m.roughness;
    out.emission = {m.emissive[0], m.emissive[1], m.emissive[2]};
    out.transmission = m.transmission;
    out.ior = m.ior;
    out.absorption = {m.absorption[0], m.absorption[1], m.absorption[2]};
    out.sheen = m.sheen;
    return out;
}

}  // namespace

rt::TwoLevelScene RenderWorldToTwoLevelScene(const RenderWorld& world) {
    rt::TwoLevelScene scene;

    // Meshes: 1:1 with the MeshLibrary (same ids).
    const uint32_t mesh_count = world.meshes.Count();
    scene.meshes.reserve(mesh_count);
    for (uint32_t m = 0; m < mesh_count; ++m) {
        scene.meshes.push_back(MeshToBlas(world.meshes.Geometry(m)));
    }

    // Materials: one rt::Material per RenderWorld material (in order, so
    // render_material_id maps directly), plus an appended default for instances
    // that reference no / out-of-range material.
    scene.materials.reserve(world.materials.size() + 1u);
    for (const scene::RenderMaterial& m : world.materials) {
        scene.materials.push_back(RenderMaterialToRt(m));
    }
    const uint32_t default_mat_id = static_cast<uint32_t>(scene.materials.size());
    scene.materials.push_back(rt::Material{});  // neutral default

    // Instances: 1:1 with RenderWorld instances using each instance's LIVE
    // world_xform. blas_id == mesh_id (the 1:1 mesh map above). Resolve material.
    scene.instances.reserve(world.instances.size());
    for (const RenderInstance& inst : world.instances) {
        rt::Instance out;
        out.blas_id = inst.mesh_id;
        out.transform = inst.world_xform;
        out.material_id = (inst.render_material_id < world.materials.size())
                              ? inst.render_material_id
                              : default_mat_id;
        scene.instances.push_back(out);
    }

    // Light: the bounded sensor renderer takes ONE analytic light. Use the first
    // RenderLight (if any) as a point light at its world position; else leave the
    // rt::Light default (directional). Extra lights are ignored (documented).
    if (!world.lights.empty()) {
        const RenderLight& l = world.lights.front();
        scene.light.directional = false;
        scene.light.position = l.world_xform.position;
        scene.light.color = l.color;
        scene.light.intensity = l.intensity;
    }

    return scene;
}

rt::PinholeCamera RenderCameraToPinhole(const RenderCamera& camera,
                                        uint32_t width,
                                        uint32_t height) {
    // Standard camera basis from the resolved world transform: looks down local
    // -Z, local +Y up. eye = world position; target = eye + forward.
    const math::Vec3 eye = camera.world_xform.position;
    const math::Vec3 forward = camera.world_xform.TransformDirection({0.0f, 0.0f, -1.0f});
    const math::Vec3 world_up = camera.world_xform.TransformDirection({0.0f, 1.0f, 0.0f});
    const float vfov_radians = camera.vertical_fov_degrees * (kPi / 180.0f);
    return rt::BuildPinhole(eye, eye + forward, world_up, vfov_radians, width, height);
}

}  // namespace nuka::render
