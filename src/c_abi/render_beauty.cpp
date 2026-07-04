// ---------------------------------------------------------------------------
// nuka::c_abi -- the offline beauty render of the LIVE world (host RGB image).
//
// Renders the world's CURRENT state with the self-written offline CUDA path-tracer
// (NOT the gated realtime Vulkan present): the robot link visuals FK-posed from the
// live link poses, any particle media surfaced from the live particle field, and a
// studio floor -- via the SHARED studio render scene (render::studio_beauty), the
// exact setup the go2_cloth_drape demo draws. The render scene is built ONCE from
// the world's retained SceneIR + the cooked particle-surface topology and refreshed
// from the live state each call. ONE render path: the media is data, never a fork.
// ---------------------------------------------------------------------------

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "render/studio_beauty.hpp"
#include "render/texture_image.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace nuka::c_abi {

// The BeautyRender ctor/dtor are out-of-line here so the unique_ptr<StudioScene> /
// unique_ptr<StudioRtRenderer> members are created/destroyed where those render
// types are complete (internal.hpp only forward-declares them).
BeautyRender::BeautyRender() = default;
BeautyRender::~BeautyRender() = default;

namespace {

namespace cook = nuka::scene::cook;

// Resolve an authored (possibly relative) asset path against the scene file's
// directory; empty stays empty, absolute passes through, no-dir uses the cwd.
std::string ResolveScenePath(const std::string& path, const std::string& scene_dir) {
    if (path.empty() || scene_dir.empty()) return path;
    const std::filesystem::path p(path);
    if (p.is_absolute()) return path;
    return (std::filesystem::path(scene_dir) / p).string();
}

// Build the world's beauty render bridge: cook the retained scene to recover the
// EntityId<->row SceneMap, build the shared studio scene (with the world's retained
// per-medium particle-surface topologies + their authored render skins), and create
// the offline RT beauty tracer. Returns the result code; on OK `record->beauty` holds
// the built bridge.
nuka_result_t EnsureBeautyBridge(WorldRecord* record, uint32_t width, uint32_t height) {
    if (record->beauty) {
        return NUKA_RESULT_OK;
    }
    // Cook the render scene at one env (the single-env visual template, env 0); the
    // live link/particle state is downloaded for env 0 each render.
    const cook::CookToModelResult cooked = cook::CookToModel(*record->scene, 1);

    // Each cooked medium's render surface (triangles + skin) drives one deforming
    // instance; the skin params ride the medium's authored MediaRenderSkin. A
    // surface-less record (particle_radius > 0) becomes an instanced-sphere skin.
    std::vector<nuka::runtime::soft::SurfaceTopology> topologies;
    std::vector<uint32_t> topology_material;   // authored id per pushed topology.
    topologies.reserve(record->particle_surfaces.size());
    for (const cook::MediaRenderSurface& s : record->particle_surfaces) {
        if (s.triangles.empty()) continue;
        nuka::runtime::soft::SurfaceTopology topo;
        topo.triangles = s.triangles;
        topo.normal_offset = s.normal_offset;
        topo.smooth_iters = s.smooth_iters;
        topo.smooth_lambda = s.smooth_lambda;
        topologies.push_back(std::move(topo));
        topology_material.push_back(s.render_material_id);
    }

    auto bridge = std::make_unique<BeautyRender>();
    bridge->scene = std::make_unique<nuka::render::StudioScene>(
        nuka::render::BuildStudioScene(record->scene->Ecs(), cooked.scene_map,
                                       topologies, width, height));
    // A medium's authored render_material_id, when valid, overrides the studio
    // default on its deforming surface / instanced-sphere skin.
    for (std::size_t i = 0; i < topology_material.size(); ++i) {
        if (topology_material[i] != 0xFFFFFFFFu) {
            nuka::render::SetStudioSurfaceMaterial(*bridge->scene,
                                                   record->scene->Ecs(), i,
                                                   topology_material[i]);
        }
    }
    for (const cook::MediaRenderSurface& s : record->particle_surfaces) {
        if (!s.triangles.empty() || s.particle_radius <= 0.0f) continue;
        nuka::render::AddStudioParticleSkin(*bridge->scene, record->scene->Ecs(),
                                            s.render_material_id, s.particle_radius,
                                            s.particle_first, s.particle_count);
    }
    if (bridge->scene->world.instances.empty()) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no renderable visual geometry.
    }

    // The scene's authored render appearance: the material policy, each material's
    // image maps (paths resolved against the scene dir), and the HDR environment.
    const nuka::scene::EnvironmentRecord& env = record->scene->Environment();
    if (env.use_scene_materials) {
        nuka::render::UseAuthoredSceneMaterials(*bridge->scene);
    }
    for (nuka::scene::RenderMaterial& m : bridge->scene->world.materials) {
        m.albedo_map = ResolveScenePath(m.albedo_map, record->scene_dir);
        m.roughness_map = ResolveScenePath(m.roughness_map, record->scene_dir);
        m.normal_map = ResolveScenePath(m.normal_map, record->scene_dir);
    }
    nuka::render::DecodeMaterialTextures(bridge->scene->world);
    if (!env.hdri.empty()) {
        constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
        bridge->scene->world.environment = nuka::render::LoadEnvironment(
            ResolveScenePath(env.hdri, record->scene_dir), env.yaw_deg * kDegToRad,
            env.intensity);
    }
    bridge->renderer = std::make_unique<nuka::render::StudioRtRenderer>();
    if (!bridge->renderer->ok()) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no offline CUDA RT backend available.
    }
    record->beauty = std::move(bridge);
    return NUKA_RESULT_OK;
}

}  // namespace

}  // namespace nuka::c_abi

extern "C" {

nuka_result_t nuka_world_render_beauty(nuka_world_handle world,
                                       const nuka_beauty_camera_t* camera,
                                       uint32_t width, uint32_t height,
                                       uint32_t spp, uint8_t dtype,
                                       void* out_rgb, size_t out_capacity,
                                       size_t* out_pixel_count) {
    if (width == 0u || height == 0u) {
        return NUKA_RESULT_INVALID_ARG;
    }
    const size_t need = static_cast<size_t>(width) * height * 3u;

    // SIZE QUERY: a null output buffer just reports the scalar count to allocate.
    if (out_rgb == nullptr) {
        if (out_pixel_count != nullptr) *out_pixel_count = need;
        return NUKA_RESULT_OK;
    }
    if (camera == nullptr || (dtype != 0u && dtype != 1u)) {
        return NUKA_RESULT_INVALID_ARG;
    }
    if (out_capacity < need) {
        return NUKA_RESULT_INVALID_ARG;
    }

    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world || !record->scene) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }

    try {
        const nuka_result_t built =
            nuka::c_abi::EnsureBeautyBridge(record, width, height);
        if (built != NUKA_RESULT_OK) {
            return built;
        }
        nuka::render::StudioScene& studio = *record->beauty->scene;

        // Download the live env-0 state: the link poses (FK visuals) + the particle
        // positions (the deforming surface). A field with no storage is skipped.
        const nuka::nk::Model& model = record->world->GetModel();
        const uint32_t link_count = model.capacities.links_per_env;
        const uint32_t particle_count = model.capacities.particles_per_env;
        const nuka::nk::Data& data = record->world->GetData();

        std::vector<nuka::math::Transform> link_pose(link_count,
                                                     nuka::math::Transform::Identity());
        if (link_count > 0u &&
            record->world->FieldPtr(nuka::nk::FieldId::LinkPose) != nullptr) {
            data.DownloadField(nuka::nk::FieldId::LinkPose, link_pose.data(),
                               static_cast<uint64_t>(link_count) *
                                   sizeof(nuka::math::Transform));
        }
        std::vector<nuka::math::Vec3> particle_pos(particle_count,
                                                   nuka::math::Vec3::Zero());
        if (particle_count > 0u &&
            (!studio.surfaces.empty() || !studio.particle_skins.empty()) &&
            record->world->FieldPtr(nuka::nk::FieldId::ParticlePos) != nullptr) {
            data.DownloadField(nuka::nk::FieldId::ParticlePos, particle_pos.data(),
                               static_cast<uint64_t>(particle_count) *
                                   sizeof(nuka::math::Vec3));
        }

        const uint32_t body_count = model.capacities.bodies_per_env;
        std::vector<nuka::math::Transform> body_pose(
            body_count, nuka::math::Transform::Identity());
        if (body_count > 0u &&
            record->world->FieldPtr(nuka::nk::FieldId::BodyPose) != nullptr) {
            data.DownloadField(nuka::nk::FieldId::BodyPose, body_pose.data(),
                               static_cast<uint64_t>(body_count) *
                                   sizeof(nuka::math::Transform));
        }

        nuka::render::PublishStudioScene(studio, link_pose, particle_pos, body_pose);

        // Drive the camera + image size; trace the offline beauty frame to host.
        nuka::render::RasterOptions& o = studio.options;
        o.width = width;
        o.height = height;
        o.use_camera_override = true;
        o.camera_eye = {camera->eye[0], camera->eye[1], camera->eye[2]};
        o.camera_target = {camera->look[0], camera->look[1], camera->look[2]};
        o.camera_up = {camera->up[0], camera->up[1], camera->up[2]};
        o.camera_fov_degrees = camera->fov_deg;

        // Author-driven beauty look levers; neutral env values reproduce the studio
        // defaults exactly (sun_disc keys off the key colour, direction unchanged).
        const nuka::scene::EnvironmentRecord& env = record->scene->Environment();
        o.beauty_sky_fill =
            (env.ibl_full_fill && !env.hdri.empty()) ? env.intensity : 0.30f;
        o.beauty_sun_disc[0] = env.sun_disc * o.sun_color[0];
        o.beauty_sun_disc[1] = env.sun_disc * o.sun_color[1];
        o.beauty_sun_disc[2] = env.sun_disc * o.sun_color[2];
        o.beauty_exposure_ev = env.exposure_ev;
        o.beauty_grade = env.grade;
        record->beauty->renderer->SetBeauty(true, spp != 0u ? spp : 16u);
        const nuka::render::VulkanOffscreenReport report =
            record->beauty->renderer->Render(studio.world, o);

        // Pack RGBA8 -> the caller's RGB buffer (top row first, R,G,B per pixel).
        const size_t pixels = static_cast<size_t>(width) * height;
        const size_t n = (report.pixels.size() < pixels) ? report.pixels.size() : pixels;
        if (dtype == 0u) {
            auto* out = static_cast<uint8_t*>(out_rgb);
            for (size_t i = 0; i < n; ++i) {
                const nuka::render::VulkanRgba8& p = report.pixels[i];
                out[i * 3 + 0] = p.r; out[i * 3 + 1] = p.g; out[i * 3 + 2] = p.b;
            }
        } else {
            auto* out = static_cast<float*>(out_rgb);
            for (size_t i = 0; i < n; ++i) {
                const nuka::render::VulkanRgba8& p = report.pixels[i];
                out[i * 3 + 0] = static_cast<float>(p.r) / 255.0f;
                out[i * 3 + 1] = static_cast<float>(p.g) / 255.0f;
                out[i * 3 + 2] = static_cast<float>(p.b) / 255.0f;
            }
        }
        if (out_pixel_count != nullptr) *out_pixel_count = need;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

}  // extern "C"
