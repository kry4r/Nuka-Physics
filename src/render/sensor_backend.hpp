#pragma once
// ---------------------------------------------------------------------------
// nuka::render::SensorBackendI -- the backend-agnostic interface to the BATCHED
// device-resident sensor render (N per-env TLASes built in one LBVH launch + one
// flat [E*H*W] trace into a single (N,H,W,ch) device AOV tensor). It is the
// SIBLING of render::RtBackendI: a SEPARATE interface because the sensor path has
// a different shape -- a PERSISTENT scene (BLAS + per-env TLAS scratch + the AOV
// tensor reused across steps), per-step DEVICE poses + mounted cameras, and a
// DEVICE-RESIDENT tensor output (no host download). Folding those onto RtBackendI
// (single camera -> host rt::Framebuffer) would burden its clean single-camera
// contract with an unrelated lifecycle, so this stays its own one-responsibility
// interface that mirrors the same factory / weak-fallback pattern.
//
// ENGINE-LAYER (this header, src/render -- ZERO CUDA token) + CUDA-BACKEND impl
// (src/phi/backend_cuda/rt/sensor_backend_cuda.cpp, the only place outside the rt
// device TUs that reaches BuildBatchedSensorScene / RenderSensorsBatched /
// ScatterEnvCameras). The concrete backend WRAPS those entry points -- it does
// NOT reimplement the batched render; this TU adds NO arithmetic.
//
// CONTRACT:
//   * BuildSensorScene(desc) -- BLAS built once + the env-shared per-instance
//                               binding uploaded + the mount table lowered; returns
//                               an opaque, backend-owned handle. nullptr on failure.
//   * RenderSensors(handle, fk, E, W, H) -- scatter the per-env cameras from the
//                               handle's mount table + `fk` (the device pose source),
//                               then ONE batched build/refit + ONE flat trace into
//                               the persistent device AOV tensor (no host round-trip).
//   * Sensor*Device(handle) -- device pointers into that tensor after a render.
//   * FreeSensorScene(handle) -- release the handle.
//
// FACTORY / FALLBACK: CreateCudaSensorBackend() binds phi::ActiveBackend() and
// returns a heap-owned backend when a device is available, else nullptr (the same
// "unavailable -> null" contract RtBackendI uses). DEFINED in the CUDA backend TU;
// a consumer that wants a live sensor backend links nuka_phi2_rt.
//
// ZERO CUDA: names ONLY CUDA-free rt:: / phi:: / scene:: PODs. Compiles under g++.
// ---------------------------------------------------------------------------

#include "phi/interop_scatter.hpp"  // phi::ScatterFkSource / InstanceScatterRow (CUDA-free)
#include "rt/render_dr.hpp"         // rt::RenderDrConfig (CUDA-free per-env DR POD)
#include "rt/sensor_fidelity.hpp"   // rt::SensorFidelityConfig (CUDA-free shade POD)
#include "rt/two_level_render.hpp"  // rt::TwoLevelScene (CUDA-free scene-desc POD)
#include "scene/scene_ir.hpp"       // scene::SensorDesc (CUDA-free)

#include <cstdint>
#include <memory>
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
};

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

    // Scatter the per-env cameras from the handle's mount table + `fk` (the device
    // LinkPose/BodyPose/BasePose source), then ONE batched build/refit + ONE flat
    // trace over [E*H*W] rays into the persistent device AOV tensor IN PLACE on the
    // selected backend's stream (no host round-trip). `width`/`height` size each
    // sensor image; `env_count` is the number of envs (cameras = env_count*sensors).
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

    // Scatter the per-env lidars from the handle's lidar mount table + `fk`, then
    // ONE batched build/refit + ONE range trace over [E*S*az*el] rays into the
    // persistent device range tensor IN PLACE (no host round-trip). A no-op if the
    // handle carries no lidar mounts. The az/el fan comes from the lidar pattern.
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

    // Record the opt-in shading-fidelity profile (spp / soft shadow / AO / GI /
    // tonemap). The DEFAULT config (Enabled()==false) keeps the cheap-shade
    // arithmetic -> the AOV bytes are unchanged. Caps spp/samples LOUDLY.
    virtual void SetSensorFidelity(SensorSceneHandle* handle,
                                   const rt::SensorFidelityConfig& cfg) = 0;

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
