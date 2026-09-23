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
#include "render/scene_asset.hpp"
#include "render/texture_image.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"

#include <cstdint>
#include <cmath>
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
        if (s.surface_spacing > 0.0f) {
            nuka::render::AddStudioDensitySurface(*bridge->scene, record->scene->Ecs(),
                s.render_material_id, s.surface_spacing, s.particle_first, s.particle_count);
            continue;
        }
        nuka::render::AddStudioParticleSkin(*bridge->scene, record->scene->Ecs(),
                                            s.render_material_id, s.particle_radius,
                                            s.particle_first, s.particle_count,
                                            s.grain_round != 0u, s.grain_radius_jitter,
                                            s.grain_tint_jitter);
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

namespace {

nuka_result_t RenderBeauty(nuka_world_handle world,
                                       const nuka_beauty_camera_t* camera,
                                       uint32_t width, uint32_t height,
                                       uint32_t spp, uint8_t dtype,
                                       void* out_rgb, size_t out_capacity,
                                       size_t* out_pixel_count,
                                       const nuka_scene_camera_t* authored = nullptr) {
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
            (!studio.surfaces.empty() || !studio.particle_skins.empty() ||
             !studio.density_surfaces.empty()) &&
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
        nuka::render::RasterOptions o = studio.options;
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
        nuka::render::ApplySceneEnvironment(o, env);
        o.camera_near = authored != nullptr ? authored->near_clip : 0.05f;
        o.camera_far = authored != nullptr ? authored->far_clip : 1000.0f;
        if (authored != nullptr && authored->shadow_radius > 0.0f) {
            o.shadow_center = o.camera_target;
            o.shadow_radius = authored->shadow_radius;
        }
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

}  // namespace

extern "C" {

nuka_result_t nuka_world_get_scene_camera(nuka_world_handle world, const char* name,
                                         uint32_t env_index, nuka_scene_camera_t* out_camera) {
    if (name == nullptr || name[0] == '\0' || out_camera == nullptr) return NUKA_RESULT_INVALID_ARG;
    const auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world || !record->scene) return NUKA_RESULT_NOT_SUPPORTED;
    try {
        const auto& model = record->world->GetModel();
        if (env_index >= model.capacities.env_count) return NUKA_RESULT_INVALID_ARG;
        const nuka::scene::CameraRecord* selected = nullptr;
        const std::string requested(name), suffix = "/" + requested;
        for (const auto& camera : record->scene->Cameras()) {
            if (camera.name == requested) { selected = &camera; break; }
        }
        if (selected == nullptr) {
            for (const auto& camera : record->scene->Cameras()) {
                if (camera.name.size() < suffix.size() ||
                    camera.name.compare(camera.name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
                if (selected != nullptr) return NUKA_RESULT_INVALID_ARG;
                selected = &camera;
            }
        }
        if (selected == nullptr) return NUKA_RESULT_INVALID_ARG;
        const auto& camera = *selected;
        const auto& local = camera.local_transform;
        if (!std::isfinite(local.position.LengthSq()) || !std::isfinite(local.rotation.Norm()) ||
            std::fabs(local.rotation.Norm() - 1.0f) > 1e-4f ||
            !(camera.vertical_fov_degrees > 0.0f && camera.vertical_fov_degrees < 180.0f) ||
            !(camera.near_clip > 0.0f && camera.far_clip > camera.near_clip) ||
            !std::isfinite(camera.far_clip) || !(camera.focus_distance > 0.0f) ||
            !std::isfinite(camera.focus_distance) || !(camera.shadow_radius >= 0.0f) ||
            !std::isfinite(camera.shadow_radius)) return NUKA_RESULT_INVALID_ARG;
        nuka_scene_camera_t result{};
        result.mount = NUKA_SENSOR_MOUNT_WORLD;
        result.local_offset[0] = local.position.x;
        result.local_offset[1] = local.position.y;
        result.local_offset[2] = local.position.z;
        result.local_offset[3] = local.rotation.w;
        result.local_offset[4] = local.rotation.x;
        result.local_offset[5] = local.rotation.y;
        result.local_offset[6] = local.rotation.z;
        auto pose = local;
        if (camera.attached_body != nuka::scene::kInvalidBody) {
            const auto body = camera.attached_body;
            if (body >= record->scene->RigidBodyCount() || body >= model.capacities.bodies_per_env)
                return NUKA_RESULT_INVALID_ARG;
            const bool link = body < model.body_to_link.size() && model.body_to_link[body] != ~uint32_t{0};
            result.mount = link ? NUKA_SENSOR_MOUNT_LINK : NUKA_SENSOR_MOUNT_BODY;
            result.mount_index = link ? model.body_to_link[body] : body;
            const auto stride = link ? model.capacities.links_per_env : model.capacities.bodies_per_env;
            const auto field = link ? nuka::nk::FieldId::LinkPose : nuka::nk::FieldId::BodyPose;
            nuka::math::Transform parent;
            const auto offset = (uint64_t{env_index} * stride + result.mount_index) * sizeof(parent);
            if (!record->world->GetData().DownloadField(field, &parent, sizeof(parent), offset))
                return NUKA_RESULT_INTERNAL;
            pose = parent * local;
        }
        const auto look = pose.TransformPoint({0.0f, 0.0f, -camera.focus_distance});
        const auto up = pose.TransformDirection({0.0f, 1.0f, 0.0f});
        const auto copy = [](float* output, nuka::math::Vec3 value) {
            output[0] = value.x; output[1] = value.y; output[2] = value.z;
        };
        copy(result.view.eye, pose.position);
        copy(result.view.look, look);
        copy(result.view.up, up);
        result.view.fov_deg = camera.vertical_fov_degrees;
        result.near_clip = camera.near_clip;
        result.far_clip = camera.far_clip;
        result.focus_distance = camera.focus_distance;
        result.shadow_radius = camera.shadow_radius;
        *out_camera = result;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_render_beauty(nuka_world_handle world, const nuka_beauty_camera_t* camera,
                                       uint32_t width, uint32_t height, uint32_t spp, uint8_t dtype,
                                       void* out_rgb, size_t out_capacity, size_t* out_pixel_count) {
    return RenderBeauty(world, camera, width, height, spp, dtype, out_rgb, out_capacity, out_pixel_count);
}

nuka_result_t nuka_world_render_scene_camera(nuka_world_handle world, const char* name,
                                            uint32_t width, uint32_t height, uint32_t spp, uint8_t dtype,
                                            void* out_rgb, size_t out_capacity, size_t* out_pixel_count) {
    nuka_scene_camera_t camera{};
    const auto status = nuka_world_get_scene_camera(world, name, 0u, &camera);
    if (status != NUKA_RESULT_OK) return status;
    return RenderBeauty(world, &camera.view, width, height, spp, dtype, out_rgb, out_capacity,
                        out_pixel_count, &camera);
}

}  // extern "C"
