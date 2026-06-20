// ---------------------------------------------------------------------------
// nuka::c_abi -- the device-resident batched camera-sensor surface.
//
// S cameras per env (each attach appends one mount) rendered into a single
// (env_count, sensors_per_env, height, width, channels) device AOV tensor via the
// SAME RT tracer at a cheap sensor profile -- the in-the-loop obs the RL stack
// reads zero-copy (nuka_world_get_sensor_view + torch.from_dlpack). No host download.
//
// THE COOK BRIDGE: the per-env visual binding (which mesh/material each visual
// instance is + the link/body it follows + its physics->visual offset) is built
// from the SAME cooked scene the physics ran -- the WorldRecord's retained
// SceneIR -> BuildRenderWorld over its ECS + the cook's SceneMap ->
// RenderWorldToTwoLevelScene. That is exactly the path go2_terrain_demo uses to
// recover the shape->link binding; here it feeds the batched sensor backend. The
// scene is the per-env template (co-residence replicas included), replicated
// across envs by the batched render. ONE scene/cook definition for physics +
// render -- no second authoring path.
// ---------------------------------------------------------------------------

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "nk/pipeline/world.hpp"
#include "render/render_world.hpp"
#include "render/rt_adapter.hpp"
#include "render/sensor_backend.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"

#include <exception>
#include <vector>

namespace nuka::c_abi {

// The SensorAttachment ctor/dtor are out-of-line here (internal.hpp holds a vector
// of the incomplete scene::SensorDesc). The dtor frees the backend-owned scene
// handle THROUGH the backend before the backend drops.
SensorAttachment::SensorAttachment() = default;
SensorAttachment::~SensorAttachment() {
    if (backend && handle != nullptr) {
        backend->FreeSensorScene(handle);
        handle = nullptr;
    }
}

namespace {

namespace cook = nuka::scene::cook;

// Lower the world's cooked scene to the env-shared sensor binding: RenderWorld ->
// TwoLevelScene + per-instance rows/blas_id/material_id. False if no geometry.
bool BuildSensorSceneDesc(const nuka::scene::SceneIR& scene, uint32_t env_count,
                          nuka::render::SensorSceneDesc* out) {
    // Cook to recover the EntityId<->row SceneMap (the binding render needs), at
    // the world's env_count so pose-source rows index the live arena.
    const cook::CookToModelResult cooked =
        cook::CookToModel(scene, static_cast<int>(env_count));
    const nuka::render::RenderWorld rw =
        nuka::render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);
    if (rw.instances.empty()) {
        return false;
    }

    out->scene = nuka::render::RenderWorldToTwoLevelScene(rw);
    // TwoLevelScene appends a neutral default after rw.materials; an out-of-range
    // render_material_id resolves to that last slot (same as the adapter's).
    const uint32_t mat_count = static_cast<uint32_t>(rw.materials.size());
    const uint32_t default_mat_id =
        static_cast<uint32_t>(out->scene.materials.size()) - 1u;

    out->rows.reserve(rw.instances.size());
    out->blas_id.reserve(rw.instances.size());
    out->material_id.reserve(rw.instances.size());
    for (const nuka::render::RenderInstance& inst : rw.instances) {
        nuka::phi::InstanceScatterRow r;
        // PoseSource::Kind {Static,Link,Body,Base} == InstanceScatterRow::kind
        // {0,1,2,3} by construction; row is the cooked link/body/base index.
        r.kind = static_cast<uint32_t>(inst.pose_source.kind);
        r.row = inst.pose_source.row;
        r.cached_visual_local[0] = inst.cached_visual_local.position.x;
        r.cached_visual_local[1] = inst.cached_visual_local.position.y;
        r.cached_visual_local[2] = inst.cached_visual_local.position.z;
        r.cached_visual_local[3] = inst.cached_visual_local.rotation.w;
        r.cached_visual_local[4] = inst.cached_visual_local.rotation.x;
        r.cached_visual_local[5] = inst.cached_visual_local.rotation.y;
        r.cached_visual_local[6] = inst.cached_visual_local.rotation.z;
        out->rows.push_back(r);
        out->blas_id.push_back(inst.mesh_id);  // blas_id == mesh_id (1:1 mesh map).
        out->material_id.push_back((inst.render_material_id < mat_count)
                                       ? inst.render_material_id
                                       : default_mat_id);
    }
    return true;
}

// Lower the c_abi mount enum + offset to one camera's SensorDesc (one row of the
// mount table); attach appends these into the per-env S-camera list.
nuka::scene::SensorDesc MakeCameraSensorDesc(nuka_sensor_mount_t mount_frame,
                                             uint32_t mount_index,
                                             const float local_offset[7],
                                             float vfov_deg, uint32_t width,
                                             uint32_t height) {
    nuka::scene::SensorDesc s;
    s.type = nuka::scene::SensorType::Camera;
    s.mount = static_cast<nuka::scene::MountFrame>(mount_frame);  // 0=Link,1=Body,2=Base.
    s.mount_index = mount_index;
    if (local_offset != nullptr) {
        s.local_offset.position.x = local_offset[0];
        s.local_offset.position.y = local_offset[1];
        s.local_offset.position.z = local_offset[2];
        s.local_offset.rotation.w = local_offset[3];
        s.local_offset.rotation.x = local_offset[4];
        s.local_offset.rotation.y = local_offset[5];
        s.local_offset.rotation.z = local_offset[6];
    }
    s.cam.width = static_cast<uint16_t>(width);
    s.cam.height = static_cast<uint16_t>(height);
    s.cam.vfov_degrees = vfov_deg;
    return s;
}

}  // namespace

}  // namespace nuka::c_abi

extern "C" {

nuka_result_t nuka_world_attach_camera_sensor(nuka_world_handle world,
                                              nuka_sensor_mount_t mount_frame,
                                              uint32_t mount_index,
                                              const float local_offset[7],
                                              float vfov_deg,
                                              uint32_t width,
                                              uint32_t height) {
    if (mount_frame > NUKA_SENSOR_MOUNT_BASE) {
        return NUKA_RESULT_INVALID_ARG;
    }
    if (width == 0u || height == 0u || vfov_deg <= 0.0f) {
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
        // The sensor backend binds the device's ACTIVE phi backend (the same
        // device/stream physics runs on); null when no RT backend/device.
        std::unique_ptr<nuka::render::SensorBackendI> backend =
            nuka::render::CreateCudaSensorBackend();
        if (!backend) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }

        nuka::render::SensorSceneDesc desc;
        if (!nuka::c_abi::BuildSensorSceneDesc(*record->scene, record->world->EnvCount(),
                                               &desc)) {
            return NUKA_RESULT_NOT_SUPPORTED;  // scene has no renderable geometry.
        }

        // Append this camera to the world's mount list (S cameras per env, one
        // (E,S,H,W) block). A re-attach at the SAME w/h adds another sensor; a
        // different w/h would make the tensor ragged, so it RESETS to a fresh
        // single-camera set (the documented re-callable contract).
        const bool same_size = record->sensor && record->sensor->width == width &&
                               record->sensor->height == height;
        if (same_size) {
            desc.sensors = record->sensor->sensors;
        }
        desc.sensors.push_back(nuka::c_abi::MakeCameraSensorDesc(
            mount_frame, mount_index, local_offset, vfov_deg, width, height));

        nuka::render::SensorSceneHandle* handle = backend->BuildSensorScene(desc);
        if (handle == nullptr) {
            return NUKA_RESULT_INTERNAL;
        }

        // Carry a previously-set render-DR config across the rebuild so a re-attach
        // keeps the per-env appearance (the fresh tables get the same randomization).
        nuka::rt::RenderDrConfig carry_dr;
        nuka::rt::SensorFidelityConfig carry_fid;
        if (record->sensor) {
            carry_dr = record->sensor->render_dr;
            carry_fid = record->sensor->fidelity;
        }

        // Install the rebuilt attachment (the old handle frees through its backend
        // in the SensorAttachment dtor when `record->sensor` is overwritten).
        auto attach = std::make_unique<nuka::c_abi::SensorAttachment>();
        attach->backend = std::move(backend);
        attach->handle = handle;
        attach->sensors = std::move(desc.sensors);
        attach->width = width;
        attach->height = height;
        attach->rendered = false;
        attach->render_dr = carry_dr;
        attach->fidelity = carry_fid;
        if (carry_dr.enabled) {
            attach->backend->SetRenderDr(attach->handle, carry_dr,
                                         record->world->EnvCount());
        }
        if (carry_fid.Enabled()) {
            attach->backend->SetSensorFidelity(attach->handle, carry_fid);
        }
        record->sensor = std::move(attach);
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_render_sensors(nuka_world_handle world) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world || !record->sensor || record->sensor->handle == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }

    try {
        nuka::nk::World& w = *record->world;
        const nuka::nk::ModelCapacities& cap = w.GetModel().capacities;

        // The LIVE env-major LinkPose/BodyPose/BasePose arena buffers (the SAME
        // ptrs DownloadField reads); the kernel derives env, so env_index is 0.
        nuka::phi::ScatterFkSource fk;
        fk.link_pose = w.FieldPtr(nuka::nk::FieldId::LinkPose);
        fk.body_pose = w.FieldPtr(nuka::nk::FieldId::BodyPose);
        fk.base_pose = w.FieldPtr(nuka::nk::FieldId::BasePose);
        fk.env_index = 0u;
        fk.links_per_env = cap.links_per_env;
        fk.bodies_per_env = cap.bodies_per_env;
        // Cross-stream RAW ordering: make the sensor render wait on the World's FK
        // ops so it never reads torn/stale poses.
        fk.world_backend = w.Backend();

        nuka::c_abi::SensorAttachment& s = *record->sensor;
        s.backend->RenderSensors(s.handle, fk, w.EnvCount(), s.width, s.height);
        s.rendered = true;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_get_sensor_view(nuka_world_handle world,
                                         nuka_sensor_channel_t channel,
                                         nuka_buffer_view_t* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    out->device_ptr = nullptr;
    out->element_count = 0u;
    out->element_stride_bytes = 0u;
    out->dtype = 0u;
    if (channel > NUKA_SENSOR_CHANNEL_PRIM) {
        return NUKA_RESULT_INVALID_ARG;
    }

    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->sensor || record->sensor->handle == nullptr ||
        !record->sensor->rendered) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no render yet -> no device tensor.
    }

    try {
        nuka::c_abi::SensorAttachment& s = *record->sensor;
        const nuka::render::SensorAovShape shape = s.backend->AovShape(s.handle);
        const uint64_t pixels = static_cast<uint64_t>(shape.env_count) *
                                shape.sensors_per_env * shape.height * shape.width;

        // The view mirrors nuka_world_get_buffer_view (flat device base +
        // element_count + stride/dtype); the caller reshapes to (N,H,W,ch).
        const void* ptr = nullptr;
        uint64_t element_count = 0u;
        uint32_t stride = sizeof(float);
        uint8_t dtype = 0u;
        switch (channel) {
            case NUKA_SENSOR_CHANNEL_COLOR:
                ptr = s.backend->SensorColorDevice(s.handle);
                element_count = pixels * 3u;
                break;
            case NUKA_SENSOR_CHANNEL_DEPTH:
                ptr = s.backend->SensorDepthDevice(s.handle);
                element_count = pixels;
                break;
            case NUKA_SENSOR_CHANNEL_NORMAL:
                ptr = s.backend->SensorNormalDevice(s.handle);
                element_count = pixels * 3u;
                break;
            case NUKA_SENSOR_CHANNEL_ALBEDO:
                ptr = s.backend->SensorAlbedoDevice(s.handle);
                element_count = pixels * 3u;
                break;
            case NUKA_SENSOR_CHANNEL_PRIM:
                ptr = s.backend->SensorPrimDevice(s.handle);
                element_count = pixels;
                stride = sizeof(uint32_t);
                dtype = 1u;  // uint32 plane (the per-pixel instance/prim id).
                break;
        }
        if (ptr == nullptr) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }

        out->device_ptr = const_cast<void*>(ptr);
        out->element_count = element_count;
        out->element_stride_bytes = stride;
        out->dtype = dtype;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_get_sensor_dims(nuka_world_handle world,
                                         uint32_t* env_count,
                                         uint32_t* sensors_per_env,
                                         uint32_t* height,
                                         uint32_t* width,
                                         uint32_t* channels) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world || !record->sensor || record->sensor->handle == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no camera attached.
    }
    // The AOV tensor is (env_count, sensors_per_env, H, W). `channels` is the
    // per-pixel count of the color/normal/albedo planes (depth/prim are 1); the
    // caller derives a plane's exact ch from its view element_count if it differs.
    const nuka::c_abi::SensorAttachment& s = *record->sensor;
    if (env_count != nullptr) *env_count = record->world->EnvCount();
    if (sensors_per_env != nullptr) {
        *sensors_per_env = static_cast<uint32_t>(s.sensors.size());
    }
    if (height != nullptr) *height = s.height;
    if (width != nullptr) *width = s.width;
    if (channels != nullptr) *channels = 3u;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_set_render_randomization(
    nuka_world_handle world, const nuka_render_dr_desc_t* desc) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world || !record->sensor || record->sensor->handle == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no camera attached -> no tables to fill.
    }

    // A NULL desc disables DR (restores base replicas -> cross-env byte-identical).
    nuka::rt::RenderDrConfig cfg;
    if (desc != nullptr) {
        cfg.color_jitter = desc->color_jitter;
        cfg.roughness_jitter = desc->roughness_jitter;
        cfg.metallic_jitter = desc->metallic_jitter;
        cfg.light_dir_jitter = desc->light_dir_jitter;
        cfg.light_intensity_jitter = desc->light_intensity_jitter;
        cfg.light_color_jitter = desc->light_color_jitter;
        cfg.ambient_intensity_jitter = desc->ambient_intensity_jitter;
        cfg.seed = desc->seed;
        cfg.enabled = desc->enabled != 0;
    }

    try {
        nuka::c_abi::SensorAttachment& s = *record->sensor;
        s.render_dr = cfg;
        s.backend->SetRenderDr(s.handle, cfg, record->world->EnvCount());
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_set_sensor_fidelity(
    nuka_world_handle world, const nuka_sensor_fidelity_desc_t* desc) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world || !record->sensor || record->sensor->handle == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no camera attached -> no scene to set.
    }

    // A NULL desc restores the cheap shade (a byte no-op). Sample counts are capped
    // here so an over-cap profile is rejected INVALID_ARG (not a hung launch).
    nuka::rt::SensorFidelityConfig cfg;  // default == cheap shade
    if (desc != nullptr) {
        if (desc->spp > 256u || desc->shadow_samples > 256u ||
            desc->ao_samples > 256u) {
            return NUKA_RESULT_INVALID_ARG;
        }
        cfg.spp = desc->spp < 1u ? 1u : desc->spp;
        cfg.shadow_samples = desc->shadow_samples;
        if (desc->sun_angular_radius > 0.0f) cfg.sun_angular_radius = desc->sun_angular_radius;
        cfg.ao_enabled = desc->ao_enabled != 0;
        if (desc->ao_samples > 0u) cfg.ao_samples = desc->ao_samples;
        if (desc->ao_radius > 0.0f) cfg.ao_radius = desc->ao_radius;
        cfg.gi_enabled = desc->gi_enabled != 0;
        cfg.tonemap_enabled = desc->tonemap_enabled != 0;
        if (desc->sky_intensity > 0.0f) cfg.sky_intensity = desc->sky_intensity;
        cfg.fog_density = desc->fog_density;
        cfg.seed = desc->seed;
    }

    try {
        nuka::c_abi::SensorAttachment& s = *record->sensor;
        s.fidelity = cfg;
        s.backend->SetSensorFidelity(s.handle, cfg);
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
