#pragma once
// Persistent batched sensor output with backend-independent scene and particle bindings.
// Surface reconstruction may stage through host memory; image and range views remain device-resident.

#include "phi/interop_scatter.hpp"  // phi::ScatterFkSource / InstanceScatterRow (CUDA-free)
#include "rt/render_dr.hpp"         // rt::RenderDrConfig (CUDA-free per-env DR POD)
#include "rt/sensor_fidelity.hpp"   // rt::SensorFidelityConfig (CUDA-free shade POD)
#include "rt/sensor_state.hpp"
#include "rt/particle_surface.hpp"
#include "rt/two_level_render.hpp"  // rt::TwoLevelScene (CUDA-free scene-desc POD)
#include "scene/scene_ir.hpp"       // scene::SensorDesc (CUDA-free)

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace nuka::render {

// Opaque handle to a backend-built batched sensor scene (BLAS once + the env-shared
// binding + the per-env TLAS scratch + the device AOV tensor + the mount table +
// the device camera buffer). Owned by the SensorBackendI that produced it; freed
// via FreeSensorScene. The concrete type lives in the CUDA backend TU.
struct SensorSceneHandle;

// The CUDA-FREE batched sensor scene-desc the interface takes (the mirror of the
// CUDA-side rt::BatchedSensorSceneDesc, which pulls device tokens). `scene` is the
// env-shared cooked scene (one cluster of meshes); rows/blas_id/material_id bind
// the instances_per_env instances (env-invariant); `sensors` is the mount table.
struct SensorSceneDesc {
    rt::TwoLevelScene                     scene;
    std::vector<phi::InstanceScatterRow>  rows;
    std::vector<uint32_t>                 blas_id;
    std::vector<uint32_t>                 material_id;
    std::vector<scene::SensorDesc>        sensors;  // mount/intrinsics per sensor
    rt::ParticlePositionSource           particles;
    std::vector<rt::ParticleSurfaceBinding> particle_surfaces;
};

inline void AppendParticleSurface(SensorSceneDesc& desc, rt::ParticleSurfaceBinding surface,
                                   uint32_t material_id) {
    surface.mesh_id = static_cast<uint32_t>(desc.scene.meshes.size());
    desc.scene.meshes.emplace_back();
    rt::Instance instance;
    instance.blas_id = surface.mesh_id;
    instance.material_id = material_id;
    desc.scene.instances.push_back(instance);
    desc.rows.emplace_back();
    desc.blas_id.push_back(surface.mesh_id);
    desc.material_id.push_back(material_id);
    desc.particle_surfaces.push_back(std::move(surface));
}

// The device AOV tensor's logical shape after a render: an (env_count,
// sensors_per_env, height, width, channels) view (S cameras per env, env-major).
// `channels` is 3 for color/normal/albedo and 1 for depth/prim (the accessors
// below return the matching base pointer). The pixel/tile count is
// env_count*sensors_per_env*height*width.
struct SensorAovShape {
    uint32_t env_count        = 0u;
    uint32_t sensors_per_env  = 0u;
    uint32_t height           = 0u;
    uint32_t width            = 0u;
};

// The device range tensor's logical shape after a lidar render: an (env_count,
// sensors_per_env, az_count, el_count) view (S lidars per env, env-major). The
// cell count is env_count*sensors_per_env*az_count*el_count.
struct SensorRangeShape {
    uint32_t env_count        = 0u;
    uint32_t sensors_per_env  = 0u;
    uint32_t az_count         = 0u;
    uint32_t el_count         = 0u;
};

// The backend-agnostic batched sensor interface. One instance per render context;
// holds the device/backend. Build a scene once, render it per step, free it.
class SensorBackendI {
public:
    virtual ~SensorBackendI() = default;

    // Build the persistent batched sensor scene (BLAS once + binding + mount table).
    // Returns an opaque handle owned by this backend (free with FreeSensorScene).
    // nullptr on failure.
    virtual SensorSceneHandle* BuildSensorScene(const SensorSceneDesc& desc) = 0;

    // Refresh live geometry and mounted cameras, then trace into persistent [E*S*H*W] device views.
    // The selected backend owns reconstruction, transfer and tracing synchronization.
    virtual void RenderSensors(SensorSceneHandle* handle,
                               const phi::ScatterFkSource& fk,
                               uint32_t env_count,
                               uint32_t width,
                               uint32_t height) = 0;

    // Device pointers into the persistent AOV tensor after RenderSensors:
    // color/normal/albedo [E*S*H*W*3], depth/prim [E*S*H*W]. Null until first render.
    virtual const float*    SensorColorDevice(const SensorSceneHandle* handle) const = 0;
    virtual const float*    SensorDepthDevice(const SensorSceneHandle* handle) const = 0;
    virtual const float*    SensorNormalDevice(const SensorSceneHandle* handle) const = 0;
    virtual const float*    SensorAlbedoDevice(const SensorSceneHandle* handle) const = 0;
    virtual const uint32_t* SensorPrimDevice(const SensorSceneHandle* handle) const = 0;

    // The (env_count, sensors_per_env, height, width) shape of the tensor the
    // accessors point into, as of the last RenderSensors (all-zero before the first).
    virtual SensorAovShape AovShape(const SensorSceneHandle* handle) const = 0;

    // Refresh the same live geometry for mounted lidar fans and trace into persistent [E*S*az*el] views.
    // A scene with no lidar mounts performs no range trace.
    virtual void RenderLidars(SensorSceneHandle* handle,
                              const phi::ScatterFkSource& fk,
                              uint32_t env_count) = 0;

    // Device pointer into the persistent range tensor after RenderLidars:
    // [E*S*az*el] floats. Null until the first lidar render.
    virtual const float* SensorRangeDevice(const SensorSceneHandle* handle) const = 0;

    // The (env_count, sensors_per_env, az_count, el_count) shape of the range tensor
    // the accessor points into, as of the last RenderLidars (all-zero before it).
    virtual SensorRangeShape RangeShape(const SensorSceneHandle* handle) const = 0;

    // Record the per-env render-DR config and (re)fill the per-env appearance
    // tables for `env_count` envs (cfg.enabled==false -> exact base replicas, so
    // the cross-env tiles stay byte-identical). Idempotent for a fixed seed.
    virtual void SetRenderDr(SensorSceneHandle* handle, const rt::RenderDrConfig& cfg,
                             uint32_t env_count) = 0;

    // Record the default high-quality sensor profile (textured materials + MSAA +
    // soft shadow + AO/GI + ACES + sRGB). Caps spp/samples LOUDLY.
    virtual void SetSensorFidelity(SensorSceneHandle* handle,
                                   const rt::SensorFidelityConfig& cfg) = 0;

    // Select camera AOV outputs. mask==0 means the legacy all-AOV profile;
    // bit positions match COLOR..PRIM in nuka_sensor_channel_t.
    virtual void SetSensorAovMask(SensorSceneHandle* handle, uint32_t mask) = 0;

    virtual void SetCameraResponse(SensorSceneHandle* handle, uint32_t id,
                                    const sensor::CameraResponse& config) = 0;
    virtual void SetRangeResponse(SensorSceneHandle* handle, bool lidar, uint32_t id,
                                   const sensor::RangeResponse& config) = 0;
    virtual void SetSensorSampleTimes(SensorSceneHandle* handle, const std::vector<double>& times) = 0;
    virtual sensor::ImagingStamp ImagingStamp(const SensorSceneHandle* handle,
                                               bool lidar, uint32_t sensor, uint32_t env) const = 0;
    virtual void ResetSensorState(SensorSceneHandle* handle, const std::vector<uint32_t>& env_ids) = 0;
    virtual rt::SensorStateSnapshot CaptureSensorState(const SensorSceneHandle* handle) const = 0;
    virtual void RestoreSensorState(SensorSceneHandle* handle, const rt::SensorStateSnapshot& snapshot) = 0;

    // Release a sensor scene handle built by this backend.
    virtual void FreeSensorScene(SensorSceneHandle* handle) = 0;
};

// Factory: a heap-owned CUDA sensor backend when the CUDA RT backend TU is linked
// AND a device initializes, else nullptr. The weak fallback (no CUDA backend
// linked) returns nullptr so the render/RHI libs link standalone; the strong
// definition in src/phi/backend_cuda/rt/sensor_backend_cuda.cpp wins when present.
std::unique_ptr<SensorBackendI> CreateCudaSensorBackend();

// Cheap probe: true iff CreateCudaSensorBackend() would return a usable backend.
bool SensorBackendAvailable();

}  // namespace nuka::render
