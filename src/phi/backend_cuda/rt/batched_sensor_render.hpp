#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- the BATCHED device-resident sensor render: N per-env TLASes built
// in ONE LBVH launch + ONE flat trace launch over [E*H*W] rays into a single
// (N,H,W,ch) device AOV tensor. The structural lever for near-real-time N-env
// RGB/depth sensors: it fills the GPU with all envs' rays and removes the
// N-serial per-env launches the naive loop pays.
//
// It REUSES the committed pieces verbatim: BuildTwoLevelScene (BLAS once), the
// batched LBVH build/refit (collision/lbvh_batched.cuh), the device scatter
// (sensor_scatter.cu), and the templated traversal nest ClosestHit/ReconstructHit
// (two_level_render_kernels.cuh) instantiated Real=float. The scene + materials +
// per-instance binding are env-shared (replicated scene); only the per-env poses +
// cameras vary. The single-camera FP64 golden path is untouched -- this is an
// additive FP32 cheap-shade profile.
//
// This is the CUDA-FREE host-facing API (the device tokens live in the .cu via the
// BatchedSensorSceneDevice pimpl, mirroring rt/two_level_render.hpp); it names only
// CUDA-free rt:: / phi:: / scene:: PODs so a host backend TU can drive it.
// ---------------------------------------------------------------------------

#include "phi/backend.hpp"
#include "phi/interop_scatter.hpp"  // ScatterFkSource / InstanceScatterRow
#include "rt/camera.hpp"
#include "rt/material.hpp"
#include "rt/two_level_render.hpp"  // TwoLevelScene / TwoLevelSceneDevice
#include "scene/scene_ir.hpp"       // scene::SensorDesc (mount table)

#include <cstdint>
#include <memory>
#include <vector>

namespace nuka::rt {

// Float AOV channels written per pixel (the prim1 uint32 plane is separate):
// color3 + depth1 + normal3 + albedo3. The obs reads color+depth+prim.
inline constexpr uint32_t kSensorFloatChannels = 10u;

// Persistent device state for an N-env replicated-scene sensor render. BLAS once;
// the TLAS nodes + scatter buffers + AOV tensor are sized lazily, reused per step.
class BatchedSensorSceneDevice {
public:
    BatchedSensorSceneDevice();
    ~BatchedSensorSceneDevice();
    BatchedSensorSceneDevice(BatchedSensorSceneDevice&&) noexcept;
    BatchedSensorSceneDevice& operator=(BatchedSensorSceneDevice&&) noexcept;
    BatchedSensorSceneDevice(const BatchedSensorSceneDevice&) = delete;
    BatchedSensorSceneDevice& operator=(const BatchedSensorSceneDevice&) = delete;

    struct Impl;
    Impl* GetImpl() { return impl_.get(); }

private:
    std::unique_ptr<Impl> impl_;
};

// The env-shared cooked scene + per-instance binding the batched render consumes;
// `rows`/`blas_id`/`material_id` bind the instances_per_env instances (env-invariant).
struct BatchedSensorSceneDesc {
    TwoLevelScene scene;
    std::vector<phi::InstanceScatterRow> rows;
    std::vector<uint32_t> blas_id;
    std::vector<uint32_t> material_id;
};

// Build the persistent batched sensor scene: BLAS once + the env-shared binding
// tables + the SensorBlasRef table. `backend` selects device + stream.
BatchedSensorSceneDevice BuildBatchedSensorScene(const BatchedSensorSceneDesc& desc,
                                                 phi::Backend* backend = nullptr);

// ONE build + ONE trace per step into the persistent (N,H,W,ch) device tensor:
// scatter -> batched LBVH build/refit -> flat trace over [E*H*W]. `cameras_device`
// is a device array of E PinholeCameras; outputs via the accessors below (no D2H).
void RenderSensorsBatched(BatchedSensorSceneDevice& device,
                          const phi::ScatterFkSource& fk,
                          const PinholeCamera* cameras_device,
                          uint32_t env_count,
                          uint32_t width,
                          uint32_t height,
                          phi::Backend* backend = nullptr);

// Upload the env-invariant sensor mount table into the persistent scene (lowered
// from scene::SensorDesc via BuildSensorMountRows inside the .cu). RenderSensorsMounted
// scatters the per-env cameras from it each step; call once after BuildBatchedSensorScene.
void SetSensorMounts(BatchedSensorSceneDevice& device,
                     const std::vector<scene::SensorDesc>& sensors);

// ONE step driven by the stored mount table: scatter cameras (fk * local_offset
// for every env x sensor) into the persistent camera buffer, then RenderSensorsBatched
// into the (N,H,W,ch) tensor. `width`/`height` size each sensor image; the camera
// count is env_count * sensors_per_env. SetSensorMounts must have been called.
void RenderSensorsMounted(BatchedSensorSceneDevice& device,
                          const phi::ScatterFkSource& fk,
                          uint32_t env_count,
                          uint32_t width,
                          uint32_t height,
                          phi::Backend* backend = nullptr);

// Device pointers into the persistent AOV tensor after RenderSensorsBatched:
// color/normal/albedo [E*H*W*3], depth/prim [E*H*W]. Null until the first render.
const float* SensorColorDevice(const BatchedSensorSceneDevice& device);
const float* SensorDepthDevice(const BatchedSensorSceneDevice& device);
const float* SensorNormalDevice(const BatchedSensorSceneDevice& device);
const float* SensorAlbedoDevice(const BatchedSensorSceneDevice& device);
const uint32_t* SensorPrimDevice(const BatchedSensorSceneDevice& device);

}  // namespace nuka::rt
