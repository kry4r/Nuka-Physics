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
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"

#include <cstdint>
#include <exception>
#include <vector>

namespace nuka::c_abi {

// The BeautyRender ctor/dtor are out-of-line here so the unique_ptr<StudioScene> /
// unique_ptr<StudioRtRenderer> members are created/destroyed where those render
// types are complete (internal.hpp only forward-declares them).
BeautyRender::BeautyRender() = default;
BeautyRender::~BeautyRender() = default;

namespace {

namespace cook = nuka::scene::cook;

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
    // instance; the skin params ride the medium's authored MediaRenderSkin.
    std::vector<nuka::runtime::soft::SurfaceTopology> topologies;
    topologies.reserve(record->particle_surfaces.size());
    for (const cook::MediaRenderSurface& s : record->particle_surfaces) {
        if (s.triangles.empty()) continue;
        nuka::runtime::soft::SurfaceTopology topo;
        topo.triangles = s.triangles;
        topo.normal_offset = s.normal_offset;
        topo.smooth_iters = s.smooth_iters;
        topo.smooth_lambda = s.smooth_lambda;
        topologies.push_back(std::move(topo));
    }

    auto bridge = std::make_unique<BeautyRender>();
    bridge->scene = std::make_unique<nuka::render::StudioScene>(
        nuka::render::BuildStudioScene(record->scene->Ecs(), cooked.scene_map,
                                       topologies, width, height));
    if (bridge->scene->world.instances.empty()) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no renderable visual geometry.
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
        if (particle_count > 0u && !studio.surfaces.empty() &&
            record->world->FieldPtr(nuka::nk::FieldId::ParticlePos) != nullptr) {
            data.DownloadField(nuka::nk::FieldId::ParticlePos, particle_pos.data(),
                               static_cast<uint64_t>(particle_count) *
                                   sizeof(nuka::math::Vec3));
        }

        nuka::render::PublishStudioScene(studio, link_pose, particle_pos);

        // Drive the camera + image size; trace the offline beauty frame to host.
        nuka::render::RasterOptions& o = studio.options;
        o.width = width;
        o.height = height;
        o.use_camera_override = true;
        o.camera_eye = {camera->eye[0], camera->eye[1], camera->eye[2]};
        o.camera_target = {camera->look[0], camera->look[1], camera->look[2]};
        o.camera_up = {camera->up[0], camera->up[1], camera->up[2]};
        o.camera_fov_degrees = camera->fov_deg;
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
