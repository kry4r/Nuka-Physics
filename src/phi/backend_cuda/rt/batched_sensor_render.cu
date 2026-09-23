// Batched cameras and lidars share traversal over environment-local instance trees.
// Live poses and particle surfaces feed device-resident observation tensors.

#include "phi/backend_cuda/rt/batched_sensor_render.hpp"

#include "collision/aabb.hpp"
#include "collision/lbvh_batched.cuh"
#include "collision/lbvh_node.cuh"
#include "math/vec3.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/rt/prim_id.cuh"
#include "phi/backend_cuda/rt/particle_surface.cuh"
#include "phi/backend_cuda/rt/reconstructed_surface.cuh"
#include "phi/backend_cuda/rt/ray_box.cuh"
#include "phi/backend_cuda/rt/rt_device_context.cuh"  // RtContext / OwnedBuffer
#include "phi/backend_cuda/rt/sensor_scatter.hpp"
#include "phi/backend_cuda/rt/shading.cuh"
#include "phi/backend_cuda/rt/two_level_render_kernels.cuh"
#include "phi/scoped_device_guard.hpp"
#include "rt/render_dr.hpp"
#include "rt/sensor_fidelity.hpp"   // SensorFidelityConfig (opt-in beauty shade)
#include "sensor/noise/philox.cuh"  // Philox4x32 host/device pure RNG
#include "sensor/noise/image_formation.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::rt {

namespace {

constexpr uint32_t kBlockSize = 128u;

// Full TLAS rebuild cadence (refit handles the frames between); same idea + value
// as the single-camera RtRenderContext. Correctness is identical either way.
constexpr uint32_t kTlasRebuildPeriod = 32u;

using ::nuka::collision::AABB;
using ::nuka::collision::gpu::LbvhNode;
using ::nuka::math::Vec3;

__device__ inline float SensorReflectance(const Material& material, float incidence) {
    const float transmission = fminf(1.0f, fmaxf(0.0f, material.transmission));
    const float luminance = fmaxf(0.0f, 0.2126f * material.albedo.x +
        0.7152f * material.albedo.y + 0.0722f * material.albedo.z);
    const float eta = fmaxf(material.ior, 1.0e-3f);
    const float ratio = (eta - 1.0f) / (eta + 1.0f);
    const float f0 = ratio * ratio;
    const float grazing = 1.0f - fminf(1.0f, fmaxf(0.0f, incidence));
    const float grazing2 = grazing * grazing;
    const float fresnel = f0 + (1.0f - f0) * grazing2 * grazing2 * grazing;
    return luminance * (1.0f - transmission) + fresnel * transmission;
}

__global__ void AdvanceImagingCounters(sensor::ImagingStamp* stamps, uint32_t count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        ++stamps[index].acquisitions;
        stamps[index].valid = 1u;
    }
}

void CheckCuda(cudaError_t result, const char* op) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(op) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// Rebase the scatter's GLOBAL DevInstance.instance_id (env*M+local) to env-LOCAL
// (0..M-1): the prim_id pack uses 12 bits for the instance, and the env is already
// known from the tile -> env-local ids keep the pack valid + tiles comparable.
__global__ void RebaseInstanceIdsKernel(DevInstance* __restrict__ instances,
                                        uint32_t env_count,
                                        uint32_t instances_per_env) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * instances_per_env;
    if (i >= total) return;
    instances[i].instance_id = i % instances_per_env;
}

__global__ void PrepareImagingKeys(const sensor::CameraResponse* cameras, const sensor::RangeResponse* ranges,
    uint32_t count, uint32_t sensors_per_env, uint32_t channel,
    sensor::noise::CameraSampleKeys* camera_keys, sensor::noise::RangeSampleKeys* range_keys) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t sensor_id = i % sensors_per_env;
    const uint32_t env = i / sensors_per_env;
    if (cameras) camera_keys[i] = sensor::noise::MakeCameraSampleKeys(cameras[sensor_id].seed, env, sensor_id);
    range_keys[i] = sensor::noise::MakeRangeSampleKeys(ranges[sensor_id].seed, env, sensor_id, channel);
}

__global__ void ReplicateSensorBlasRefsKernel(const SensorBlasRef* shared, uint32_t meshes,
                                             uint32_t count, SensorBlasRef* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = shared[i % meshes];
}

// A symmetric draw in [-1, 1], a PURE function of (seed, env, axis): one uniform
// from a distinct Philox lane mapped from (0,1] to [-1,1]. The light/ambient axes
// pass material_slot==0 (one term per env); the material axes pass the slot, and
// the seq packs (slot, axis) injectively so distinct (env, slot, axis) never alias.
__device__ inline float DrSym(uint64_t seed, uint32_t env, uint32_t material_slot,
                              rt::RenderDrAxis axis) {
    const uint32_t kAxisStride = 16u;  // > the axis enum count (lanes per slot)
    const uint64_t seq =
        static_cast<uint64_t>(material_slot) * kAxisStride +
        static_cast<uint32_t>(axis);
    const sensor::noise::Philox4x32Counter out = sensor::noise::Philox4x32_10(
        sensor::noise::MakeCounter(env, seq), sensor::noise::SplitSeed(seed));
    return 2.0f * sensor::noise::Uint32ToUniform01(out.v[0]) - 1.0f;  // (-1, 1]
}

__device__ inline float DrClamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Device mirror of the quality flags. Sky and sample parameters travel in
// BeautyParams; spp/samples are host-capped before launch.
struct FidelityParams {
    uint32_t spp;
    uint32_t tonemap;
    uint32_t srgb;
    uint64_t seed;
};

// Per-sample RNG: a PCG stream seeded from one Philox draw keyed by (seed, ray,
// sample) -> distinct streams, no mutable state, so fidelity-on is two-run exact.
struct PhiloxSeededRng {
    uint32_t s;
    __device__ inline float NextF() {
        s = s * 747796405u + 2891336453u;
        const uint32_t word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
        const uint32_t r = (word >> 22u) ^ word;
        return static_cast<float>(r >> 8) * (1.0f / 16777216.0f);  // [0,1)
    }
};

__device__ inline PhiloxSeededRng MakeFidelityRng(uint64_t seed, uint32_t ray,
                                                  uint32_t sample) {
    const sensor::noise::Philox4x32Counter out = sensor::noise::Philox4x32_10(
        sensor::noise::MakeCounter(ray, sample), sensor::noise::SplitSeed(seed));
    return PhiloxSeededRng{out.v[0]};
}

// Fill the per-env material table [E*M] from the base materials as a pure function
// of (seed, env, axis). DR off -> an exact base replica (byte-identical tiles).
__global__ void FillEnvMaterialsKernel(const Material* __restrict__ base,
                                       uint32_t material_count,
                                       rt::RenderDrConfig cfg, uint32_t env_count,
                                       Material* __restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * material_count;
    if (i >= total) return;
    const uint32_t env = i / material_count;
    const uint32_t slot = i % material_count;
    Material m = base[slot];
    if (cfg.enabled) {
        m.albedo.x = DrClamp(m.albedo.x + cfg.color_jitter *
                             DrSym(cfg.seed, env, slot, rt::RenderDrAxis::MaterialColorR), 0.0f, 1.0f);
        m.albedo.y = DrClamp(m.albedo.y + cfg.color_jitter *
                             DrSym(cfg.seed, env, slot, rt::RenderDrAxis::MaterialColorG), 0.0f, 1.0f);
        m.albedo.z = DrClamp(m.albedo.z + cfg.color_jitter *
                             DrSym(cfg.seed, env, slot, rt::RenderDrAxis::MaterialColorB), 0.0f, 1.0f);
        m.roughness = DrClamp(m.roughness + cfg.roughness_jitter *
                             DrSym(cfg.seed, env, slot, rt::RenderDrAxis::MaterialRoughness), 1.0e-3f, 1.0f);
        m.metallic = DrClamp(m.metallic + cfg.metallic_jitter *
                             DrSym(cfg.seed, env, slot, rt::RenderDrAxis::MaterialMetallic), 0.0f, 1.0f);
    }
    out[i] = m;
}

// Fill the per-env light [E] + ambient [E] from the base as a pure function of
// (seed, env, axis). DR off -> an exact base replica.
__global__ void FillEnvLightsKernel(Light base_light, AmbientTerm base_ambient,
                                    rt::RenderDrConfig cfg, uint32_t env_count,
                                    Light* __restrict__ out_light,
                                    AmbientTerm* __restrict__ out_ambient) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= env_count) return;
    Light l = base_light;
    AmbientTerm a = base_ambient;
    if (cfg.enabled) {
        l.direction.x += cfg.light_dir_jitter * DrSym(cfg.seed, env, 0u, rt::RenderDrAxis::LightDirX);
        l.direction.y += cfg.light_dir_jitter * DrSym(cfg.seed, env, 0u, rt::RenderDrAxis::LightDirY);
        l.direction.z += cfg.light_dir_jitter * DrSym(cfg.seed, env, 0u, rt::RenderDrAxis::LightDirZ);
        l.intensity = fmaxf(0.0f, l.intensity * (1.0f + cfg.light_intensity_jitter *
                            DrSym(cfg.seed, env, 0u, rt::RenderDrAxis::LightIntensity)));
        l.color.x = fmaxf(0.0f, l.color.x * (1.0f + cfg.light_color_jitter *
                          DrSym(cfg.seed, env, 0u, rt::RenderDrAxis::LightColorR)));
        l.color.y = fmaxf(0.0f, l.color.y * (1.0f + cfg.light_color_jitter *
                          DrSym(cfg.seed, env, 0u, rt::RenderDrAxis::LightColorG)));
        l.color.z = fmaxf(0.0f, l.color.z * (1.0f + cfg.light_color_jitter *
                          DrSym(cfg.seed, env, 0u, rt::RenderDrAxis::LightColorB)));
        const float am = fmaxf(0.0f, 1.0f + cfg.ambient_intensity_jitter *
                              DrSym(cfg.seed, env, 0u, rt::RenderDrAxis::AmbientIntensity));
        a.color.x *= am;
        a.color.y *= am;
        a.color.z *= am;
    }
    out_light[env] = l;
    out_ambient[env] = a;
}

__device__ inline uint32_t SensorPixelIndex(uint32_t work, uint32_t width,
                                             uint32_t height) {
    if ((width & 7u) != 0u || (height & 7u) != 0u) return work;
    const uint32_t tile_width = width / 8u;
    const uint32_t tile = work / 64u;
    const uint32_t lane = work & 63u;
    const uint32_t tile_x = tile % tile_width;
    const uint32_t tile_y = tile / tile_width;
    const uint32_t px = tile_x * 8u + (lane & 7u);
    const uint32_t py = tile_y * 8u + (lane >> 3u);
    return py * width + px;
}

// Center rays supply geometric AOVs; RGB accumulates sampled linear illumination.
// Camera, instance and material tables use environment-major indexing.
__global__ void BatchedSensorTraceKernel(const PinholeCamera* __restrict__ cameras,
                                         const LbvhNode* __restrict__ tlas_nodes,
                                         uint32_t leaves_per_env,
                                         const DevInstance* __restrict__ instances,
                                         const Material* __restrict__ materials,
                                         const DevTexture* __restrict__ textures,
                                         uint32_t material_count,
                                         const Light* __restrict__ lights,
                                         const AmbientTerm* __restrict__ ambients,
                                         uint32_t num_cameras,
                                         uint32_t sensors_per_env,
                                         uint32_t width,
                                         uint32_t height,
                                         FidelityParams fid,
                                         BeautyParams sky,
                                         uint32_t aov_mask,
                                         const sensor::CameraResponse* __restrict__ camera_responses,
                                         const sensor::RangeResponse* __restrict__ range_responses,
                                         const sensor::noise::CameraSampleKeys* __restrict__ camera_keys,
                                         const sensor::noise::RangeSampleKeys* __restrict__ range_keys,
                                         const sensor::ImagingStamp* __restrict__ stamps,
                                         float* __restrict__ out_color,
                                         float* __restrict__ out_depth,
                                         float* __restrict__ out_normal,
                                         float* __restrict__ out_albedo,
                                         uint32_t* __restrict__ out_prim) {
    const uint32_t work = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pix_per_cam = width * height;
    const uint64_t total = static_cast<uint64_t>(num_cameras) * pix_per_cam;
    if (static_cast<uint64_t>(work) >= total) return;

    const uint32_t cam = work / pix_per_cam;
    const uint32_t p = work % pix_per_cam;
    const uint32_t local_p = SensorPixelIndex(p, width, height);
    const uint32_t px = local_p % width;
    const uint32_t py = local_p / width;
    const uint32_t gid = cam * pix_per_cam + local_p;
    const uint32_t env = cam / sensors_per_env;
    const uint32_t sensor_id = cam % sensors_per_env;
    const bool measured_depth = range_responses && range_responses[sensor_id].enabled &&
        (aov_mask & kSensorAovDepth) != 0u;

    const PinholeCamera camera = cameras[cam];
    const Ray ray = camera.GenerateRay(px, py);
    // Headlight anchor: a body-attached light rides the RENDERING camera's live
    // origin (resolved per pixel below), so it tracks the mount every frame.
    const Vec3 cam_origin = camera.origin;

    // Env-offset TLAS node slice + env-offset instance slice (TLAS leaf .left is
    // env-local, so the env instance slice resolves it directly).
    const LbvhNode* env_nodes = tlas_nodes + static_cast<uint64_t>(env) * (2u * leaves_per_env - 1u);
    const DevInstance* env_inst = instances + static_cast<uint64_t>(env) * leaves_per_env;

    float best_t;
    uint32_t best_prim;
    ClosestHit<float>(env_nodes, leaves_per_env, env_inst, ray.origin, ray.dir, 0.0f,
                      &best_t, &best_prim);

    // Depth clip on the center ray (drives every AOV): a hit outside the camera's
    // [near,far] reads as a miss. Default clip is wide-open => no pixel changes.
    if (best_prim != kNoPrim &&
        (best_t < camera.near_clip || best_t > camera.far_clip)) {
        best_prim = kNoPrim;
    }

    Vec3 color{0.0f, 0.0f, 0.0f};
    float depth = RtMissDepth();
    Vec3 normal{0.0f, 0.0f, 0.0f};
    Vec3 albedo{0.0f, 0.0f, 0.0f};
    uint32_t prim_id = kNoPrim;

    // DEPTH/PRIM need only the primary closest hit. Hit reconstruction is needed
    // for NORMAL/ALBEDO/COLOR; material + lighting tables are read only by the
    // channels that consume them. The default all-AOV mask follows the exact
    // legacy arithmetic below.
    const bool need_details =
        (aov_mask & (kSensorAovColor | kSensorAovNormal | kSensorAovAlbedo)) != 0u || measured_depth;
    const bool need_material =
        (aov_mask & (kSensorAovColor | kSensorAovAlbedo)) != 0u || measured_depth;
    const bool need_color = (aov_mask & kSensorAovColor) != 0u;
    const Material* env_mats = nullptr;
    Light light{};
    if (need_material) {
        env_mats = materials + static_cast<uint64_t>(env) * material_count;
    }
    if (need_color) {
        light = lights[env];
    }

    if (best_prim != kNoPrim) {
        depth = best_t;
        prim_id = best_prim;

        if (need_details) {
            uint32_t inst, local_prim;
            UnpackPrimId(best_prim, &inst, &local_prim);

            Vec3 n;
            float uv_u, uv_v;
            ReconstructHit<float>(env_inst, best_prim, ray.origin, ray.dir, &n, &uv_u, &uv_v);
            normal = n;
            Material mat{};
            if (need_material) {
                const uint32_t material_id = env_inst[inst].material_id;
                mat = env_mats[material_id];
                if (need_color || (aov_mask & kSensorAovAlbedo) != 0u || measured_depth) {
                    const Vec3 hit{ray.origin.x + best_t * ray.dir.x,
                                   ray.origin.y + best_t * ray.dir.y,
                                   ray.origin.z + best_t * ray.dir.z};
                    ApplyMaterialTextures(env_inst, textures, best_prim, hit, uv_u, uv_v,
                                          n, &mat, true);
                }
                albedo = mat.albedo;
            }

            if (measured_depth) {
                const float incidence = fabsf(n.Dot(ray.dir));
                const float reflectance = SensorReflectance(mat, incidence);
                depth = sensor::noise::FormRange(depth, incidence, reflectance,
                    camera.near_clip, camera.far_clip, RtMissDepth(), range_responses[sensor_id],
                    range_keys[cam], local_p, stamps[cam].acquisitions);
            }

            // RGB is evaluated below by the default quality path. Keeping the
            // center-ray reconstruction above makes non-color AOVs inexpensive.
        }
    }

    // The default quality path accumulates spp jittered samples through the shared
    // beauty shade. Misses sample the sky dome; textured materials are sampled in
    // the same path. AOVs above always come from the center ray.
    if (need_color) {
        const uint32_t S = fid.spp < 1u ? 1u : fid.spp;
        Vec3 accum{0.0f, 0.0f, 0.0f};
        for (uint32_t s = 0; s < S; ++s) {
            PhiloxSeededRng rng = MakeFidelityRng(fid.seed, gid, s);
            const float jx = (s == 0u) ? 0.0f : (rng.NextF() - 0.5f);
            const float jy = (s == 0u) ? 0.0f : (rng.NextF() - 0.5f);
            const Ray r = camera.GenerateRayJitter(px, py, jx, jy);
            float bt; uint32_t bp;
            ClosestHit<float>(env_nodes, leaves_per_env, env_inst, r.origin, r.dir, 0.0f,
                              &bt, &bp);
            if (bp == kNoPrim) {
                const Vec3 miss = SkyColor(r.dir, sky);
                accum.x += miss.x; accum.y += miss.y; accum.z += miss.z;
                continue;
            }
            Vec3 sn; float su, sv;
            ReconstructHit<float>(env_inst, bp, r.origin, r.dir, &sn, &su, &sv);
            const float snv = sn.x * (-r.dir.x) + sn.y * (-r.dir.y) + sn.z * (-r.dir.z);
            const Vec3 snf = (snv < 0.0f) ? Vec3{-sn.x, -sn.y, -sn.z} : sn;
            uint32_t si, slp; UnpackPrimId(bp, &si, &slp);
            const Material smat = env_mats[env_inst[si].material_id];
            const Vec3 shit{r.origin.x + bt * r.dir.x, r.origin.y + bt * r.dir.y,
                            r.origin.z + bt * r.dir.z};
            const Vec3 sV = RtNormalize<float>(Vec3{-r.dir.x, -r.dir.y, -r.dir.z});
            Material sample_mat = smat;
            Vec3 shade_normal = snf;
            if (smat.transmission > 0.0f || sky.smooth_normals != 0u) {
                const Vec3 smooth = SmoothWorldNormal(env_inst, bp, su, sv, sn);
                shade_normal = smooth.Dot(sV) < 0.0f ? smooth * -1.0f : smooth;
            }
            Vec3 col;
            if (smat.transmission > 0.0f) {
                col = ShadeTransmissive(env_nodes, leaves_per_env, env_inst, env_mats,
                                        light, sky, shit, shade_normal, sV, sample_mat,
                                        &rng, textures);
            } else {
                shade_normal = ApplyMaterialTextures(env_inst, textures, bp, shit, su, sv,
                                                      shade_normal, &sample_mat, true);
                col = ShadeBeauty<PhiloxSeededRng>(env_nodes, leaves_per_env, env_inst,
                                                  env_mats, light, sky, shit, shade_normal,
                                                  sV, sample_mat, &rng, textures, &cam_origin);
            }
            if (sky.fog_density > 0.0f) {
                const float f = 1.0f - expf(-sky.fog_density * bt);
                col.x += (sky.fog_color.x - col.x) * f;
                col.y += (sky.fog_color.y - col.y) * f;
                col.z += (sky.fog_color.z - col.z) * f;
            }
            accum.x += col.x; accum.y += col.y; accum.z += col.z;
        }
        const float inv_s = 1.0f / static_cast<float>(S);
        color = Vec3{accum.x * inv_s, accum.y * inv_s, accum.z * inv_s};
        if (camera_responses && camera_responses[sensor_id].enabled) {
            const auto& response = camera_responses[sensor_id];
            const auto& keys = camera_keys[cam];
            const uint64_t sequence = stamps[cam].acquisitions;
            color.x = sensor::noise::FormCameraChannel(color.x, response, keys, local_p, py, 0u, sequence);
            color.y = sensor::noise::FormCameraChannel(color.y, response, keys, local_p, py, 1u, sequence);
            color.z = sensor::noise::FormCameraChannel(color.z, response, keys, local_p, py, 2u, sequence);
        }
        if (fid.tonemap != 0u) {
            color = Vec3{TonemapAces(color.x), TonemapAces(color.y), TonemapAces(color.z)};
        }
        if (fid.srgb != 0u) {
            color = Vec3{LinearToSrgb(color.x), LinearToSrgb(color.y), LinearToSrgb(color.z)};
        }
    }

    if ((aov_mask & kSensorAovColor) != 0u) {
        out_color[gid * 3u + 0u] = color.x;
        out_color[gid * 3u + 1u] = color.y;
        out_color[gid * 3u + 2u] = color.z;
    }
    if ((aov_mask & kSensorAovDepth) != 0u) out_depth[gid] = depth;
    if ((aov_mask & kSensorAovNormal) != 0u) {
        out_normal[gid * 3u + 0u] = normal.x;
        out_normal[gid * 3u + 1u] = normal.y;
        out_normal[gid * 3u + 2u] = normal.z;
    }
    if ((aov_mask & kSensorAovAlbedo) != 0u) {
        out_albedo[gid * 3u + 0u] = albedo.x;
        out_albedo[gid * 3u + 1u] = albedo.y;
        out_albedo[gid * 3u + 2u] = albedo.z;
    }
    if ((aov_mask & kSensorAovPrim) != 0u) out_prim[gid] = prim_id;
}

// Dedicated primary-hit camera path. Keeping this as a separate kernel is
// intentional: a runtime branch inside BatchedSensorTraceKernel still carries the
// full shading register footprint and throttles occupancy. DEPTH|PRIM therefore
// compiles to only ray generation + closest hit + clip + selected stores.
__global__ void BatchedSensorPrimaryHitKernel(
    const PinholeCamera* __restrict__ cameras,
    const LbvhNode* __restrict__ tlas_nodes,
    uint32_t leaves_per_env,
    const DevInstance* __restrict__ instances,
    uint32_t num_cameras,
    uint32_t sensors_per_env,
    uint32_t width,
    uint32_t height,
    uint32_t aov_mask,
    float* __restrict__ out_depth,
    uint32_t* __restrict__ out_prim) {
    const uint32_t work = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pix_per_cam = width * height;
    const uint64_t total = static_cast<uint64_t>(num_cameras) * pix_per_cam;
    if (static_cast<uint64_t>(work) >= total) return;

    const uint32_t cam = work / pix_per_cam;
    const uint32_t p = work % pix_per_cam;
    const uint32_t local_p = SensorPixelIndex(p, width, height);
    const uint32_t px = local_p % width;
    const uint32_t py = local_p / width;
    const uint32_t gid = cam * pix_per_cam + local_p;
    const uint32_t env = cam / sensors_per_env;

    const PinholeCamera camera = cameras[cam];
    const Ray ray = camera.GenerateRay(px, py);

    const LbvhNode* env_nodes =
        tlas_nodes + static_cast<uint64_t>(env) * (2u * leaves_per_env - 1u);
    const DevInstance* env_inst =
        instances + static_cast<uint64_t>(env) * leaves_per_env;

    float depth;
    uint32_t prim_id;
    ClosestHit<float>(env_nodes, leaves_per_env, env_inst, ray.origin, ray.dir,
                      0.0f, &depth, &prim_id);
    if (prim_id != kNoPrim &&
        (depth < camera.near_clip || depth > camera.far_clip)) {
        prim_id = kNoPrim;
    }
    if (prim_id == kNoPrim) depth = RtMissDepth();

    if ((aov_mask & kSensorAovDepth) != 0u) out_depth[gid] = depth;
    if ((aov_mask & kSensorAovPrim) != 0u) out_prim[gid] = prim_id;
}

// One thread per GLOBAL lidar ray over [num_lidars*az*el]: flat gid -> (lidar,
// az_index, el_index). Lidars are env-major (lidars[env*S+s]); the env owning a
// lidar is lidar/S, and the TLAS node + instance slices index by env (the SAME
// env-shared trees the camera trace reads). The (az,el) ray is generated in the
// sensor-local frame then rotated to world by the resolved mount rotation, and the
// SAME ClosestHit returns the world distance -- the range. miss => max_range;
// otherwise clamp(t, min_range, max_range). One writer per cell, no atomics ->
// FP32-deterministic. Strictly cheaper than the camera kernel (no shade/AOVs).
__global__ void BatchedLidarTraceKernel(const LidarSensor* __restrict__ lidars,
                                        const LbvhNode* __restrict__ tlas_nodes,
                                        uint32_t leaves_per_env,
                                        const DevInstance* __restrict__ instances,
                                        uint32_t num_lidars,
                                        uint32_t sensors_per_env,
                                        uint32_t az_count,
                                        uint32_t el_count,
                                        const Material* __restrict__ materials,
                                        const DevTexture* __restrict__ textures,
                                        uint32_t material_count,
                                        const sensor::RangeResponse* __restrict__ responses,
                                        const sensor::noise::RangeSampleKeys* __restrict__ keys,
                                        const sensor::ImagingStamp* __restrict__ stamps,
                                        float* __restrict__ out_range) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t rays_per_lidar = az_count * el_count;
    const uint64_t total = static_cast<uint64_t>(num_lidars) * rays_per_lidar;
    if (static_cast<uint64_t>(gid) >= total) return;

    const uint32_t lidar = gid / rays_per_lidar;
    const uint32_t env = lidar / sensors_per_env;
    const uint32_t r = gid % rays_per_lidar;
    const uint32_t el_index = r % el_count;
    const uint32_t az_index = r / el_count;

    const LidarSensor s = lidars[lidar];
    const Vec3 d_local = LidarRayDirLocal(az_index, el_index, az_count, el_count,
                                          s.az_min, s.az_max, s.el_min, s.el_max);
    // Default-construct + assign (the Quat 4-arg ctor is host-only); QuatRotate is
    // the SAME HD rotate the instance + camera scatter use.
    math::Quat rot;
    rot.w = s.rotation[0];
    rot.x = s.rotation[1];
    rot.y = s.rotation[2];
    rot.z = s.rotation[3];
    const Vec3 dir = QuatRotate(rot, d_local);
    const Vec3 origin{s.origin[0], s.origin[1], s.origin[2]};

    const LbvhNode* env_nodes = tlas_nodes + static_cast<uint64_t>(env) * (2u * leaves_per_env - 1u);
    const DevInstance* env_inst = instances + static_cast<uint64_t>(env) * leaves_per_env;

    float best_t;
    uint32_t best_prim;
    ClosestHit<float>(env_nodes, leaves_per_env, env_inst, origin, dir, 0.0f, &best_t,
                      &best_prim);

    float range = s.max_range;
    if (best_prim != kNoPrim) {
        range = best_t < s.min_range ? s.min_range
                                     : (best_t > s.max_range ? s.max_range : best_t);
        const uint32_t sensor_id = lidar % sensors_per_env;
        if (responses && responses[sensor_id].enabled) {
            Vec3 n;
            float u, v;
            ReconstructHit<float>(env_inst, best_prim, origin, dir, &n, &u, &v);
            const float incidence = fabsf(n.Dot(dir));
            uint32_t instance, primitive;
            UnpackPrimId(best_prim, &instance, &primitive);
            Material mat = materials[size_t{env} * material_count + env_inst[instance].material_id];
            const Vec3 hit = origin + dir * best_t;
            ApplyMaterialTextures(env_inst, textures, best_prim, hit, u, v, n, &mat, true);
            const float reflectance = SensorReflectance(mat, incidence);
            range = sensor::noise::FormRange(best_t, incidence, reflectance, s.min_range, s.max_range,
                s.max_range, responses[sensor_id], keys[lidar], r, stamps[lidar].acquisitions);
        }
    }
    out_range[gid] = range;
}

}  // namespace

// Persistent device state: BLAS-once handle + the env-shared instance binding
// tables + per-env TLAS node array + batched-build scratch + scatter outputs +
// the (N,H,W,ch) AOV tensor. Sized lazily, reused across steps.
struct BatchedSensorSceneDevice::Impl {
    TwoLevelSceneDevice blas;  // BLAS built once (reused from the shared path)
    Light light;               // base (env 0) light; per-env table derived from it
    AmbientTerm ambient;       // base ambient; per-env table derived from it

    // Env-shared per-instance binding (uploaded once). d_materials holds the BASE
    // material set [M]; the per-env table d_materials_env [E*M] is derived from it.
    OwnedBuffer d_rows, d_blas_id, d_material_id, d_blas_refs, d_materials;
    std::vector<OwnedBuffer> d_texture_texels;
    OwnedBuffer d_textures;
    uint32_t texture_count = 0u;
    uint32_t instances_per_env = 0u;
    uint32_t material_count = 0u;

    ParticlePositionSource particles;
    std::vector<particle_surface_detail::SurfaceCache> particle_surfaces;
    std::vector<particle_surface_detail::ReconstructedSurfaceCache> reconstructed_surfaces;
    OwnedBuffer d_env_blas_refs;
    size_t env_blas_refs_bytes = 0u;
    uint32_t mesh_count = 0u, blas_ref_envs = 0u;

    // Per-env appearance tables (the ONE thing the trace reads): materials [E*M],
    // light/ambient [E]. DR off -> exact base replicas (byte-identical tiles).
    OwnedBuffer d_materials_env, d_light_env, d_ambient_env;
    std::size_t matenv_b = 0u, lightenv_b = 0u, ambenv_b = 0u;
    uint32_t dr_env_count = 0u;  // env count the per-env tables are filled for
    rt::RenderDrConfig dr_cfg;   // disabled by default -> replicas

    // Default high-quality shading configuration used by every camera render.
    rt::SensorFidelityConfig fid_cfg;

    sensor::ImagingState imaging;
    OwnedBuffer d_camera_responses, d_depth_responses, d_lidar_responses;
    OwnedBuffer d_camera_stamps, d_lidar_stamps;
    OwnedBuffer d_camera_keys, d_depth_keys, d_lidar_keys;
    size_t camera_key_bytes = 0u, depth_key_bytes = 0u, lidar_key_bytes = 0u;
    bool camera_keys_dirty = true, lidar_keys_dirty = true;
    size_t camera_response_bytes = 0u, depth_response_bytes = 0u, lidar_response_bytes = 0u;
    size_t camera_stamp_bytes = 0u, lidar_stamp_bytes = 0u;
    bool responses_dirty = true;
    bool camera_stamps_dirty = true, lidar_stamps_dirty = true;
    uint32_t image_width = 0u, image_height = 0u;

    // Selected camera outputs. Default is the legacy all-AOV tensor; setters
    // normalize public mask==0 to this value.
    uint32_t aov_mask = kSensorAovAll;

    // Scatter outputs + batched TLAS (sized to E*M / E*(2M-1)); *_b track each
    // allocation's byte size so growth-only realloc is honest across steps.
    OwnedBuffer d_instances, d_world_aabbs, d_tlas_nodes;
    OwnedBuffer d_morton, d_index, d_sortkey, d_visit;
    OwnedBuffer d_lbvh_workspace;
    std::size_t lbvh_workspace_bytes = 0u;
    std::size_t inst_b = 0u, aabb_b = 0u, node_b = 0u;
    std::size_t mort_b = 0u, idx_b = 0u, key_b = 0u, vis_b = 0u;
    uint32_t env_count = 0u;
    bool topology_built = false;
    uint32_t frames_since_rebuild = 0u;

    // (E,S,H,W,ch) AOV tensor (color3 + depth1 + normal3 + albedo3 + prim1).
    OwnedBuffer d_color, d_depth, d_normal, d_albedo, d_prim;
    std::size_t col_b = 0u, dep_b = 0u, nrm_b = 0u, alb_b = 0u, prim_b = 0u;
    uint64_t aov_rays = 0u;

    // Env-invariant mount table (uploaded once) + the E*S env-major camera buffer
    // the scatter writes (sized lazily). RenderSensorsMounted scatters then renders.
    OwnedBuffer d_mounts, d_cameras;
    uint32_t sensors_per_env = 0u;
    std::size_t cam_b = 0u;

    // Env-invariant lidar mount table + the E*S env-major LidarSensor buffer the
    // lidar scatter writes + the (E,S,az,el) range tensor (all sized lazily). The
    // fan dims are uniform across the lidar set (every row's pattern, capped LOUD).
    OwnedBuffer d_lidar_mounts, d_lidars, d_range;
    uint32_t lidars_per_env = 0u;
    uint32_t lidar_az = 0u, lidar_el = 0u;
    std::size_t lidar_mount_b = 0u, lidars_b = 0u, range_b = 0u;
};

BatchedSensorSceneDevice::BatchedSensorSceneDevice()
    : impl_(std::make_unique<Impl>()) {}
BatchedSensorSceneDevice::~BatchedSensorSceneDevice() = default;
BatchedSensorSceneDevice::BatchedSensorSceneDevice(BatchedSensorSceneDevice&&) noexcept = default;
BatchedSensorSceneDevice& BatchedSensorSceneDevice::operator=(
    BatchedSensorSceneDevice&&) noexcept = default;

BatchedSensorSceneDevice BuildBatchedSensorScene(const BatchedSensorSceneDesc& desc,
                                                 phi::Backend* backend) {
    if (desc.rows.size() != desc.blas_id.size() ||
        desc.rows.size() != desc.material_id.size()) {
        throw std::runtime_error(
            "BuildBatchedSensorScene: rows/blas_id/material_id size mismatch");
    }
    const uint32_t m = static_cast<uint32_t>(desc.rows.size());
    if (m == 0u) {
        throw std::runtime_error("BuildBatchedSensorScene: zero instances per env");
    }
    if (m > kMaxInstances) {
        throw std::runtime_error(
            "BuildBatchedSensorScene: instances_per_env exceeds the 12-bit prim_id "
            "instance cap (kMaxInstances=4096)");
    }

    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);

    BatchedSensorSceneDevice out;
    BatchedSensorSceneDevice::Impl* impl = out.GetImpl();

    // BLAS built once via the SHARED path; surface its per-mesh SensorBlasRefs.
    impl->blas = BuildTwoLevelScene(desc.scene, backend);
    std::vector<SensorBlasRef> refs;
    CollectSensorBlasRefs(impl->blas, &refs);
    impl->mesh_count = static_cast<uint32_t>(refs.size());
    for (uint32_t b : desc.blas_id) {
        if (b >= refs.size()) {
            throw std::runtime_error("BuildBatchedSensorScene: blas_id out of range");
        }
    }

    impl->particles = desc.particles;
    std::vector<uint8_t> bound_meshes(refs.size(), 0u);
    for (const auto& surface : desc.particle_surfaces) {
        if (surface.mesh_id >= refs.size() || bound_meshes[surface.mesh_id])
            throw std::invalid_argument("invalid or duplicate particle surface mesh binding");
        if (!desc.particles.positions || !desc.particles.env_count)
            throw std::invalid_argument("particle surface requires live particle positions");
        bound_meshes[surface.mesh_id] = 1u;
        if (surface.kind == ParticleSurfaceBinding::Kind::Triangles)
            impl->particle_surfaces.emplace_back(surface, desc.particles.particles_per_env, ctx);
        else
            impl->reconstructed_surfaces.emplace_back(surface, desc.particles.particles_per_env);
    }

    impl->light = desc.scene.light;
    impl->ambient = desc.scene.ambient;
    impl->instances_per_env = m;

    const std::vector<Material> mats =
        desc.scene.materials.empty() ? std::vector<Material>{Material{}} : desc.scene.materials;
    impl->material_count = static_cast<uint32_t>(mats.size());

    impl->d_rows = UploadOwned(ctx.device_bt, desc.rows);
    impl->d_blas_id = UploadOwned(ctx.device_bt, desc.blas_id);
    impl->d_material_id = UploadOwned(ctx.device_bt, desc.material_id);
    impl->d_blas_refs = UploadOwned(ctx.device_bt, refs);
    impl->d_materials = UploadOwned(ctx.device_bt, mats);
    if (desc.scene.textures != nullptr) {
        const std::vector<Texture>& textures = *desc.scene.textures;
        std::vector<DevTexture> device_textures(textures.size());
        impl->d_texture_texels.reserve(textures.size());
        for (std::size_t i = 0; i < textures.size(); ++i) {
            const Texture& texture = textures[i];
            if (texture.Empty()) continue;
            impl->d_texture_texels.push_back(UploadOwned(ctx.device_bt, texture.texels));
            DevTexture& device_texture = device_textures[i];
            device_texture.texels = static_cast<const float*>(
                impl->d_texture_texels.back().Data());
            device_texture.width = texture.width;
            device_texture.height = texture.height;
            device_texture.channels = texture.channels;
            device_texture.srgb = texture.srgb;
        }
        if (!device_textures.empty()) {
            impl->d_textures = UploadOwned(ctx.device_bt, device_textures);
            impl->texture_count = static_cast<uint32_t>(device_textures.size());
        }
    }
    cudaStreamSynchronize(ctx.stream);
    return out;
}

namespace {

// (Re)allocate an OwnedBuffer to `bytes`, only on growth/first use.
void EnsureBytes(OwnedBuffer& buf, std::size_t& cur_bytes, phi::BufferType* bt,
                 std::size_t bytes) {
    if (bytes != cur_bytes || buf.Data() == nullptr) {
        buf = OwnedBuffer(bt, bytes);
        cur_bytes = bytes;
    }
}

void SizeImagingState(BatchedSensorSceneDevice::Impl* impl, uint32_t env_count) {
    auto& state = impl->imaging;
    if (state.env_count != env_count) {
        state.env_count = env_count;
        state.sample_times.assign(env_count, 0.0);
        state.camera_stamps.clear();
        state.lidar_stamps.clear();
        impl->camera_stamps_dirty = impl->lidar_stamps_dirty = true;
        impl->camera_keys_dirty = impl->lidar_keys_dirty = true;
    }
    state.camera_stamps.resize(size_t{env_count} * state.cameras.size());
    state.lidar_stamps.resize(size_t{env_count} * state.lidars.size());
}

template <typename T>
void UploadImagingVector(OwnedBuffer& buffer, size_t& allocated, const std::vector<T>& values,
                          const RtContext& ctx) {
    if (values.empty()) return;
    const size_t bytes = values.size() * sizeof(T);
    EnsureBytes(buffer, allocated, ctx.device_bt, bytes);
    buffer.CopyFromHost(values.data(), bytes);
}

void PrepareImaging(BatchedSensorSceneDevice::Impl* impl, const RtContext& ctx,
                     uint32_t env_count, uint32_t sensors_per_env, bool lidar, bool device_stamps) {
    auto& state = impl->imaging;
    if ((lidar && state.lidars.size() != sensors_per_env) ||
        (!lidar && (state.cameras.size() != sensors_per_env || state.depths.size() != sensors_per_env)))
        impl->responses_dirty = true;
    if (lidar) state.lidars.resize(sensors_per_env);
    else {
        state.cameras.resize(sensors_per_env);
        state.depths.resize(sensors_per_env);
    }
    SizeImagingState(impl, env_count);
    if (impl->responses_dirty) {
        UploadImagingVector(impl->d_camera_responses, impl->camera_response_bytes, state.cameras, ctx);
        UploadImagingVector(impl->d_depth_responses, impl->depth_response_bytes, state.depths, ctx);
        UploadImagingVector(impl->d_lidar_responses, impl->lidar_response_bytes, state.lidars, ctx);
        impl->responses_dirty = false;
        impl->camera_keys_dirty = impl->lidar_keys_dirty = true;
    }
    auto& stamps = lidar ? state.lidar_stamps : state.camera_stamps;
    auto& keys_dirty = lidar ? impl->lidar_keys_dirty : impl->camera_keys_dirty;
    if (device_stamps && keys_dirty) {
        auto& range_keys = lidar ? impl->d_lidar_keys : impl->d_depth_keys;
        auto& range_bytes = lidar ? impl->lidar_key_bytes : impl->depth_key_bytes;
        EnsureBytes(range_keys, range_bytes, ctx.device_bt, stamps.size() * sizeof(sensor::noise::RangeSampleKeys));
        if (!lidar) EnsureBytes(impl->d_camera_keys, impl->camera_key_bytes, ctx.device_bt,
            stamps.size() * sizeof(sensor::noise::CameraSampleKeys));
        phi::LaunchCuda(PrepareImagingKeys, dim3((stamps.size() + kBlockSize - 1u) / kBlockSize),
            dim3(kBlockSize), 0u, ctx.stream,
            lidar ? nullptr : static_cast<const sensor::CameraResponse*>(impl->d_camera_responses.Data()),
            static_cast<const sensor::RangeResponse*>((lidar ? impl->d_lidar_responses : impl->d_depth_responses).Data()),
            static_cast<uint32_t>(stamps.size()), sensors_per_env, lidar ? 4u : 3u,
            lidar ? nullptr : static_cast<sensor::noise::CameraSampleKeys*>(impl->d_camera_keys.Data()),
            static_cast<sensor::noise::RangeSampleKeys*>(range_keys.Data()));
        CheckCuda(cudaGetLastError(), "prepare imaging random keys");
        keys_dirty = false;
    }
    for (auto& stamp : stamps) if (stamp.acquisitions == UINT64_MAX)
        throw std::overflow_error("imaging acquisition counter exhausted");
    auto& dirty = lidar ? impl->lidar_stamps_dirty : impl->camera_stamps_dirty;
    auto& buffer = lidar ? impl->d_lidar_stamps : impl->d_camera_stamps;
    auto& allocated = lidar ? impl->lidar_stamp_bytes : impl->camera_stamp_bytes;
    if (device_stamps && (dirty || allocated != stamps.size() * sizeof(sensor::ImagingStamp))) {
        UploadImagingVector(buffer, allocated, stamps, ctx);
        CheckCuda(cudaStreamSynchronize(ctx.stream), "initialize imaging counters");
        dirty = false;
    }
    for (uint32_t env = 0u; env < env_count; ++env) for (uint32_t s = 0u; s < sensors_per_env; ++s) {
        auto& stamp = stamps[size_t{env} * sensors_per_env + s];
        ++stamp.acquisitions;
        stamp.sample_time = state.sample_times[env];
        stamp.valid = 1u;
    }
    if (device_stamps) {
        phi::LaunchCuda(AdvanceImagingCounters, dim3((stamps.size() + kBlockSize - 1u) / kBlockSize),
            dim3(kBlockSize), 0u, ctx.stream, static_cast<sensor::ImagingStamp*>(buffer.Data()),
            static_cast<uint32_t>(stamps.size()));
        CheckCuda(cudaGetLastError(), "advance imaging counters");
    } else dirty = true;
}

void ClearImagingSensor(BatchedSensorSceneDevice::Impl* impl, const RtContext& ctx,
                         bool lidar, uint32_t sensor_id, uint32_t env) {
    auto& state = impl->imaging;
    const size_t count = lidar ? state.lidars.size() : state.cameras.size();
    if (sensor_id >= count || env >= state.env_count) throw std::invalid_argument("imaging sensor index out of range");
    auto& stamps = lidar ? state.lidar_stamps : state.camera_stamps;
    stamps[size_t{env} * count + sensor_id] = {};
    auto& device_stamps = lidar ? impl->d_lidar_stamps : impl->d_camera_stamps;
    const size_t stamp_bytes = lidar ? impl->lidar_stamp_bytes : impl->camera_stamp_bytes;
    const size_t stamp_offset = (size_t{env} * count + sensor_id) * sizeof(sensor::ImagingStamp);
    if (device_stamps.Data() && stamp_offset + sizeof(sensor::ImagingStamp) <= stamp_bytes)
        CheckCuda(cudaMemsetAsync(static_cast<uint8_t*>(device_stamps.Data()) + stamp_offset,
            0, sizeof(sensor::ImagingStamp), ctx.stream), "reset imaging counter");
    const size_t elements = lidar ? size_t{impl->lidar_az} * impl->lidar_el
        : size_t{impl->image_width} * impl->image_height;
    const size_t tile = size_t{env} * count + sensor_id;
    const auto clear = [&](OwnedBuffer& buffer, size_t bytes, size_t components) {
        const size_t tile_bytes = elements * components * sizeof(float);
        if (tile_bytes && buffer.Data() && (tile + 1u) * tile_bytes <= bytes)
            CheckCuda(cudaMemsetAsync(static_cast<uint8_t*>(buffer.Data()) + tile * tile_bytes,
                0, tile_bytes, ctx.stream), "clear imaging tile");
    };
    if (lidar) clear(impl->d_range, impl->range_b, 1u);
    else {
        clear(impl->d_color, impl->col_b, 3u);
        clear(impl->d_depth, impl->dep_b, 1u);
        clear(impl->d_normal, impl->nrm_b, 3u);
        clear(impl->d_albedo, impl->alb_b, 3u);
        clear(impl->d_prim, impl->prim_b, 1u);
    }
}

// Per-env appearance table cap. A vision DR run is thousands of envs; the table
// stays trivial (a 4-material set at 65536 envs is ~8 MB), but cap it LOUDLY so a
// nonsense env_count fails fast instead of overflowing the allocation.
constexpr uint32_t kMaxRenderDrEnvs = 1u << 20;  // 1,048,576 envs

// Fidelity sample caps: each pixel runs spp * (shadow_samples + ao_samples) trace
// rays, so a runaway profile would stall the launch. Cap LOUDLY so a nonsense
// profile fails fast instead of hanging the device.
constexpr uint32_t kMaxFidelitySpp = 256u;
constexpr uint32_t kMaxFidelitySamples = 256u;  // per soft-shadow / AO dimension

// Lower the stored quality config to the kernel's device params and assert the
// sample caps before launch.
void BuildFidelityParams(const rt::SensorFidelityConfig& cfg, FidelityParams* fp,
                         BeautyParams* sky) {
    if (cfg.spp > kMaxFidelitySpp) {
        throw std::runtime_error(
            "SetSensorFidelity: spp exceeds the per-pixel sample cap "
            "(kMaxFidelitySpp=256)");
    }
    if (cfg.shadow_samples > kMaxFidelitySamples ||
        cfg.ao_samples > kMaxFidelitySamples || cfg.transmit_bounces > kMaxFidelitySamples) {
        throw std::runtime_error(
            "SetSensorFidelity: shadow, AO or transmission samples exceed the sample cap "
            "(kMaxFidelitySamples=256)");
    }
    fp->spp = cfg.spp < 1u ? 1u : cfg.spp;
    fp->tonemap = cfg.tonemap_enabled ? 1u : 0u;
    fp->srgb = cfg.srgb_enabled ? 1u : 0u;
    fp->seed = cfg.seed;
    // 0 = shadow rays OFF (the shader skips the visibility cone; MuJoCo's
    // castshadow="false"), 1+ = hard/soft shadow rays.
    sky->shadow_rays = cfg.shadow_samples;
    sky->sun_angular_radius = cfg.sun_angular_radius;
    // AO off -> the shared shade's indirect term must vanish: zero sky ambient +
    // no GI bounce, with a single (zero-contribution) AO ray (the primary-ray miss
    // still shows the sky background, which is gated separately).
    sky->ao_samples = cfg.ao_enabled ? (cfg.ao_samples < 1u ? 1u : cfg.ao_samples) : 1u;
    sky->ao_radius = cfg.ao_radius;
    sky->gi_bounces = (cfg.ao_enabled && cfg.gi_enabled) ? 1u : 0u;
    sky->sky_intensity = cfg.ao_enabled ? cfg.sky_intensity : 0.0f;
    sky->sky_top = cfg.sky_top;
    sky->sky_bottom = cfg.sky_bottom;
    sky->sky_ground = cfg.sky_ground;
    sky->fog_color = cfg.fog_color;
    sky->fog_density = cfg.fog_density;
    sky->transmit_bounces = cfg.transmit_bounces;
    sky->smooth_normals = cfg.smooth_normals ? 1u : 0u;
    // Batched sensors carry no sun disc -> the sky stays byte-identical.
    sky->sun_dir = Vec3{0.0f, 0.0f, 0.0f};
    sky->sun_radiance = Vec3{0.0f, 0.0f, 0.0f};
    sky->sun_cos_radius = 0.0f;
}

// (Re)fill the per-env material/light/ambient tables from the base set + the
// stored DR config, for `env_count` envs. DR off -> exact base replicas, so the
// trace's cross-env tiles stay byte-identical. Refills only on an env-count or
// config change; the fill is a pure function of (seed, env, axis).
void EnsureEnvAppearanceTables(BatchedSensorSceneDevice::Impl* impl,
                               const RtContext& ctx, uint32_t env_count, bool force) {
    if (env_count == 0u) return;
    if (env_count > kMaxRenderDrEnvs) {
        throw std::runtime_error(
            "RenderDr: env_count exceeds the per-env appearance-table cap "
            "(kMaxRenderDrEnvs=1048576)");
    }
    const uint32_t mc = impl->material_count;
    const bool resized = env_count != impl->dr_env_count;
    EnsureBytes(impl->d_materials_env, impl->matenv_b, ctx.device_bt,
                static_cast<std::size_t>(env_count) * mc * sizeof(Material));
    EnsureBytes(impl->d_light_env, impl->lightenv_b, ctx.device_bt,
                static_cast<std::size_t>(env_count) * sizeof(Light));
    EnsureBytes(impl->d_ambient_env, impl->ambenv_b, ctx.device_bt,
                static_cast<std::size_t>(env_count) * sizeof(AmbientTerm));
    if (!resized && !force) return;

    const uint32_t kBlock = 128u;
    const uint32_t mat_total = env_count * mc;
    const uint32_t mat_grid = (mat_total + kBlock - 1u) / kBlock;
    phi::LaunchCuda(FillEnvMaterialsKernel, dim3(mat_grid), dim3(kBlock), 0u,
                    ctx.stream, static_cast<const Material*>(impl->d_materials.Data()),
                    mc, impl->dr_cfg, env_count,
                    static_cast<Material*>(impl->d_materials_env.Data()));
    const uint32_t light_grid = (env_count + kBlock - 1u) / kBlock;
    phi::LaunchCuda(FillEnvLightsKernel, dim3(light_grid), dim3(kBlock), 0u,
                    ctx.stream, impl->light, impl->ambient, impl->dr_cfg, env_count,
                    static_cast<Light*>(impl->d_light_env.Data()),
                    static_cast<AmbientTerm*>(impl->d_ambient_env.Data()));
    impl->dr_env_count = env_count;
}

// Lidar fan + range-tensor cap. az_count*el_count*E*S rays must fit a launch + the
// range allocation; cap each dimension LOUDLY so a nonsense pattern fails fast
// instead of overflowing the buffer or stalling the device. 16384 per axis is far
// above any real lidar (a 128-beam x 2048-az sweep is 2048<<16384) yet finite.
constexpr uint32_t kMaxLidarAxis = 1u << 14;  // 16,384 az or el samples

// Scatter fk*cvl -> the per-env DevInstance table + world-AABBs, rebase the
// instance ids to env-local, then build/refit the N per-env TLASes. The ONE
// topology path the camera AND lidar traces both consume (same trees, same poses).
// Sizes the scatter/LBVH buffers (growth-only) and honors the rebuild cadence.
void EnsureEnvTopology(BatchedSensorSceneDevice::Impl* impl, const RtContext& ctx,
                       const phi::ScatterFkSource& fk, uint32_t env_count) {
    phi::BufferType* bt = ctx.device_bt;
    const uint32_t m = impl->instances_per_env;
    const uint64_t total_inst = static_cast<uint64_t>(env_count) * m;
    const uint64_t node_count = static_cast<uint64_t>(env_count) * (2u * m - 1u);

    if (env_count != impl->env_count || !impl->topology_built) {
        size_t bytes = 0u;
        const auto status = collision::gpu::QueryLbvhWorkspaceBytes(env_count, m, &bytes);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        EnsureBytes(impl->d_lbvh_workspace, impl->lbvh_workspace_bytes, bt, bytes);
    }

    EnsureBytes(impl->d_instances, impl->inst_b, bt, total_inst * sizeof(DevInstance));
    EnsureBytes(impl->d_world_aabbs, impl->aabb_b, bt, total_inst * sizeof(AABB));
    EnsureBytes(impl->d_tlas_nodes, impl->node_b, bt, node_count * sizeof(LbvhNode));
    EnsureBytes(impl->d_morton, impl->mort_b, bt, total_inst * sizeof(uint32_t));
    EnsureBytes(impl->d_index, impl->idx_b, bt, total_inst * sizeof(uint32_t));
    EnsureBytes(impl->d_sortkey, impl->key_b, bt, total_inst * sizeof(uint64_t));
    EnsureBytes(impl->d_visit, impl->vis_b, bt, total_inst * sizeof(uint32_t));

    const bool need_rebuild = !impl->topology_built || env_count != impl->env_count ||
                              impl->frames_since_rebuild >= kTlasRebuildPeriod;

    auto* d_instances = static_cast<DevInstance*>(impl->d_instances.Data());
    auto* d_world_aabbs = static_cast<AABB*>(impl->d_world_aabbs.Data());
    auto* d_nodes = static_cast<LbvhNode*>(impl->d_tlas_nodes.Data());

    const auto* blas_refs = static_cast<const SensorBlasRef*>(impl->d_blas_refs.Data());
    uint32_t blas_refs_per_env = 0u;
    if (!impl->particle_surfaces.empty() || !impl->reconstructed_surfaces.empty()) {
        const uint64_t count = uint64_t{env_count} * impl->mesh_count;
        if (count > UINT32_MAX - kBlockSize)
            throw std::invalid_argument("sensor mesh table exceeds device index capacity");
        EnsureBytes(impl->d_env_blas_refs, impl->env_blas_refs_bytes, bt, count * sizeof(SensorBlasRef));
        auto* live_refs = static_cast<SensorBlasRef*>(impl->d_env_blas_refs.Data());
        if (impl->blas_ref_envs != env_count) {
            phi::LaunchCuda(ReplicateSensorBlasRefsKernel, dim3((count + kBlockSize - 1u) / kBlockSize),
                dim3(kBlockSize), 0u, ctx.stream, blas_refs, impl->mesh_count,
                static_cast<uint32_t>(count), live_refs);
            impl->blas_ref_envs = env_count;
        }
        for (auto& surface : impl->particle_surfaces)
            surface.Update(ctx, impl->particles, env_count, live_refs, impl->mesh_count);
        for (auto& surface : impl->reconstructed_surfaces)
            surface.Update(ctx, impl->particles, env_count, live_refs, impl->mesh_count);
        blas_refs = live_refs;
        blas_refs_per_env = impl->mesh_count;
    }

    ScatterEnvInstances(ctx.stream, fk,
                        static_cast<const phi::InstanceScatterRow*>(impl->d_rows.Data()),
                        static_cast<const uint32_t*>(impl->d_blas_id.Data()),
                        static_cast<const uint32_t*>(impl->d_material_id.Data()),
                        blas_refs, env_count, m, d_instances, d_world_aabbs, blas_refs_per_env);
    {
        const uint32_t grid = (static_cast<uint32_t>(total_inst) + kBlockSize - 1u) / kBlockSize;
        phi::LaunchCuda(RebaseInstanceIdsKernel, dim3(grid), dim3(kBlockSize), 0u,
                        ctx.stream, d_instances, env_count, m);
    }
    if (need_rebuild) {
        const auto status = collision::gpu::BuildLbvhBatchedNodes(
            ctx.stream, ctx.device_id, d_world_aabbs, env_count, m, d_nodes,
            static_cast<uint32_t*>(impl->d_morton.Data()),
            static_cast<uint32_t*>(impl->d_index.Data()),
            static_cast<uint64_t*>(impl->d_sortkey.Data()),
            static_cast<uint32_t*>(impl->d_visit.Data()),
            impl->d_lbvh_workspace.Data(), impl->lbvh_workspace_bytes);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        impl->env_count = env_count;
        impl->topology_built = true;
        impl->frames_since_rebuild = 0u;
    } else {
        collision::gpu::RefitLbvhBatched(
            ctx.stream, ctx.device_id, d_nodes, d_world_aabbs, env_count, m,
            static_cast<uint32_t*>(impl->d_visit.Data()));
        ++impl->frames_since_rebuild;
    }
}

}  // namespace

void RenderSensorsBatched(BatchedSensorSceneDevice& device,
                          const phi::ScatterFkSource& fk,
                          const PinholeCamera* cameras_device,
                          uint32_t env_count,
                          uint32_t sensors_per_env,
                          uint32_t width,
                          uint32_t height,
                          phi::Backend* backend) {
    BatchedSensorSceneDevice::Impl* impl = device.GetImpl();
    const uint32_t m = impl->instances_per_env;
    const uint32_t s = sensors_per_env == 0u ? 1u : sensors_per_env;
    if (env_count == 0u || width == 0u || height == 0u || m == 0u) {
        return;
    }

    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);
    phi::BufferType* bt = ctx.device_bt;

    // Scene is env-shared (E trees of M instances); cameras fan out E*S env-major.
    const uint64_t num_cameras = static_cast<uint64_t>(env_count) * s;
    if (num_cameras > UINT32_MAX || uint64_t{width} * height > UINT32_MAX / num_cameras)
        throw std::invalid_argument("camera pixel count exceeds the supported index range");
    const uint64_t rays = num_cameras * width * height;

    const uint32_t aov_mask = impl->aov_mask;
    const bool camera_response = (aov_mask & kSensorAovColor) != 0u &&
        std::any_of(impl->imaging.cameras.begin(), impl->imaging.cameras.end(),
        [](const auto& config) { return config.enabled != 0u; });
    const bool depth_response = (aov_mask & kSensorAovDepth) != 0u &&
        std::any_of(impl->imaging.depths.begin(), impl->imaging.depths.end(),
        [](const auto& config) { return config.enabled != 0u; });
    PrepareImaging(impl, ctx, env_count, s, false, camera_response || depth_response);

    if ((aov_mask & kSensorAovColor) != 0u) {
        EnsureBytes(impl->d_color, impl->col_b, bt, rays * 3u * sizeof(float));
    }
    if ((aov_mask & kSensorAovDepth) != 0u) {
        EnsureBytes(impl->d_depth, impl->dep_b, bt, rays * sizeof(float));
    }
    if ((aov_mask & kSensorAovNormal) != 0u) {
        EnsureBytes(impl->d_normal, impl->nrm_b, bt, rays * 3u * sizeof(float));
    }
    if ((aov_mask & kSensorAovAlbedo) != 0u) {
        EnsureBytes(impl->d_albedo, impl->alb_b, bt, rays * 3u * sizeof(float));
    }
    if ((aov_mask & kSensorAovPrim) != 0u) {
        EnsureBytes(impl->d_prim, impl->prim_b, bt, rays * sizeof(uint32_t));
    }

    // Per-env material/light/ambient tables (refilled on an env-count change; DR
    // off -> base replicas). The trace reads these BY ENV -- the ONE path.
    const bool need_appearance =
        (aov_mask & (kSensorAovColor | kSensorAovAlbedo)) != 0u || depth_response;
    if (need_appearance) {
        EnsureEnvAppearanceTables(impl, ctx, env_count, false);
    }

    // 1+2) Scatter + batched LBVH build/refit (the ONE topology path lidar reuses).
    EnsureEnvTopology(impl, ctx, fk, env_count);
    auto* d_instances = static_cast<DevInstance*>(impl->d_instances.Data());
    auto* d_nodes = static_cast<LbvhNode*>(impl->d_tlas_nodes.Data());

    // 3) ONE flat trace over [E*S*H*W] into the persistent (E,S,H,W,ch) AOV tensor.
    // DEPTH|PRIM selects a separately compiled primary-hit kernel so it carries no
    // shading register footprint. Any appearance AOV uses the general shade path.
    const uint32_t grid = static_cast<uint32_t>((rays + kBlockSize - 1u) / kBlockSize);
    const bool primary_only =
        (aov_mask & ~(kSensorAovDepth | kSensorAovPrim)) == 0u && !depth_response;
    if (primary_only) {
        phi::LaunchCuda(BatchedSensorPrimaryHitKernel, dim3(grid),
                        dim3(kBlockSize), 0u, ctx.stream, cameras_device, d_nodes,
                        m, d_instances, static_cast<uint32_t>(num_cameras), s,
                        width, height, aov_mask,
                        static_cast<float*>(impl->d_depth.Data()),
                        static_cast<uint32_t*>(impl->d_prim.Data()));
        CheckCuda(cudaGetLastError(), "BatchedSensorPrimaryHitKernel launch");
    } else {
        FidelityParams fid;
        BeautyParams sky;
        BuildFidelityParams(impl->fid_cfg, &fid, &sky);
        phi::LaunchCuda(BatchedSensorTraceKernel, dim3(grid), dim3(kBlockSize), 0u,
                        ctx.stream, cameras_device, d_nodes, m, d_instances,
                        static_cast<const Material*>(impl->d_materials_env.Data()),
                        static_cast<const DevTexture*>(impl->d_textures.Data()),
                        impl->material_count,
                        static_cast<const Light*>(impl->d_light_env.Data()),
                        static_cast<const AmbientTerm*>(impl->d_ambient_env.Data()),
                        static_cast<uint32_t>(num_cameras), s,
                        width, height, fid, sky, aov_mask,
                        camera_response ? static_cast<const sensor::CameraResponse*>(impl->d_camera_responses.Data()) : nullptr,
                        depth_response ? static_cast<const sensor::RangeResponse*>(impl->d_depth_responses.Data()) : nullptr,
                        static_cast<const sensor::noise::CameraSampleKeys*>(impl->d_camera_keys.Data()),
                        static_cast<const sensor::noise::RangeSampleKeys*>(impl->d_depth_keys.Data()),
                        static_cast<const sensor::ImagingStamp*>(impl->d_camera_stamps.Data()),
                        static_cast<float*>(impl->d_color.Data()),
                        static_cast<float*>(impl->d_depth.Data()),
                        static_cast<float*>(impl->d_normal.Data()),
                        static_cast<float*>(impl->d_albedo.Data()),
                        static_cast<uint32_t*>(impl->d_prim.Data()));
        CheckCuda(cudaGetLastError(), "BatchedSensorTraceKernel launch");
    }
    impl->aov_rays = rays;
    impl->image_width = width;
    impl->image_height = height;
}

void SetSensorMounts(BatchedSensorSceneDevice& device,
                     const std::vector<scene::SensorDesc>& sensors) {
    if (sensors.empty()) {
        throw std::runtime_error("SetSensorMounts: empty mount table");
    }
    const std::vector<SensorMountRow> mounts = BuildSensorMountRows(sensors);
    BatchedSensorSceneDevice::Impl* impl = device.GetImpl();
    const RtContext ctx = ResolveRtContext(nullptr);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);
    impl->d_mounts = UploadOwned(ctx.device_bt, mounts);
    impl->sensors_per_env = static_cast<uint32_t>(mounts.size());
    impl->imaging.cameras.resize(mounts.size());
    impl->imaging.depths.resize(mounts.size());
    impl->imaging.camera_stamps.assign(size_t{impl->imaging.env_count} * mounts.size(), {});
    impl->camera_stamps_dirty = true;
    impl->responses_dirty = true;
    cudaStreamSynchronize(ctx.stream);
}

void SetRenderDr(BatchedSensorSceneDevice& device, const RenderDrConfig& cfg,
                 uint32_t env_count, phi::Backend* backend) {
    BatchedSensorSceneDevice::Impl* impl = device.GetImpl();
    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);
    // Store the config and force a refill of the per-env tables (a fixed seed makes
    // this idempotent; disabling restores exact base replicas -> byte-identical).
    impl->dr_cfg = cfg;
    EnsureEnvAppearanceTables(impl, ctx, env_count, true);
    cudaStreamSynchronize(ctx.stream);
}

void SetSensorFidelity(BatchedSensorSceneDevice& device,
                       const SensorFidelityConfig& cfg) {
    BatchedSensorSceneDevice::Impl* impl = device.GetImpl();
    // Record the quality config; the next render always uses the shared beauty
    // path. Validation remains here so a bad profile fails before launch.
    FidelityParams fp;
    BeautyParams sky;
    BuildFidelityParams(cfg, &fp, &sky);
    impl->fid_cfg = cfg;
}

void SetSensorAovMask(BatchedSensorSceneDevice& device, uint32_t mask) {
    BatchedSensorSceneDevice::Impl* impl = device.GetImpl();
    const uint32_t effective = mask == 0u ? kSensorAovAll : mask;
    if ((effective & ~kSensorAovAll) != 0u) {
        throw std::runtime_error(
            "SetSensorAovMask: mask contains bits outside COLOR..PRIM");
    }
    impl->aov_mask = effective;
}

void SetCameraResponse(BatchedSensorSceneDevice& device, uint32_t id,
                       const sensor::CameraResponse& config, phi::Backend* backend) {
    auto* impl = device.GetImpl();
    if (id >= impl->imaging.cameras.size() || !sensor::ValidCameraResponse(config))
        throw std::invalid_argument("invalid camera response or sensor index");
    const auto ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    for (uint32_t env = 0u; env < impl->imaging.env_count; ++env)
        ClearImagingSensor(impl, ctx, false, id, env);
    CheckCuda(cudaStreamSynchronize(ctx.stream), "clear camera response history");
    impl->imaging.cameras[id] = config;
    impl->responses_dirty = true;
}

void SetRangeResponse(BatchedSensorSceneDevice& device, bool lidar, uint32_t id,
                      const sensor::RangeResponse& config, phi::Backend* backend) {
    auto* impl = device.GetImpl();
    auto& configs = lidar ? impl->imaging.lidars : impl->imaging.depths;
    if (id >= configs.size() || !sensor::ValidRangeResponse(config))
        throw std::invalid_argument("invalid range response or sensor index");
    const auto ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    for (uint32_t env = 0u; env < impl->imaging.env_count; ++env)
        ClearImagingSensor(impl, ctx, lidar, id, env);
    CheckCuda(cudaStreamSynchronize(ctx.stream), "clear range response history");
    configs[id] = config;
    impl->responses_dirty = true;
}

void SetSensorSampleTimes(BatchedSensorSceneDevice& device, const std::vector<double>& times) {
    if (times.empty() || times.size() > UINT32_MAX) throw std::invalid_argument("invalid imaging clock count");
    for (double time : times) if (!std::isfinite(time) || time < 0.0)
        throw std::invalid_argument("invalid imaging sample time");
    auto* impl = device.GetImpl();
    SizeImagingState(impl, static_cast<uint32_t>(times.size()));
    impl->imaging.sample_times = times;
}

sensor::ImagingStamp ImagingStamp(const BatchedSensorSceneDevice& device,
                                  bool lidar, uint32_t id, uint32_t env) {
    const auto& state = const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->imaging;
    const size_t count = lidar ? state.lidars.size() : state.cameras.size();
    if (id >= count || (state.env_count && env >= state.env_count))
        throw std::invalid_argument("imaging stamp index out of range");
    if (!state.env_count) return {};
    const auto& stamps = lidar ? state.lidar_stamps : state.camera_stamps;
    return stamps[size_t{env} * count + id];
}

void ResetSensorState(BatchedSensorSceneDevice& device, const std::vector<uint32_t>& ids,
                      phi::Backend* backend) {
    auto* impl = device.GetImpl();
    auto& state = impl->imaging;
    if (!state.env_count) return;
    for (uint32_t env : ids) if (env >= state.env_count)
        throw std::invalid_argument("imaging reset environment out of range");
    const auto ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    const auto reset = [&](uint32_t env) {
        for (uint32_t s = 0u; s < state.cameras.size(); ++s) ClearImagingSensor(impl, ctx, false, s, env);
        for (uint32_t s = 0u; s < state.lidars.size(); ++s) ClearImagingSensor(impl, ctx, true, s, env);
        state.sample_times[env] = 0.0;
    };
    if (ids.empty()) for (uint32_t env = 0u; env < state.env_count; ++env) reset(env);
    else for (uint32_t env : ids) reset(env);
    CheckCuda(cudaStreamSynchronize(ctx.stream), "reset imaging state");
}

SensorStateSnapshot CaptureSensorState(const BatchedSensorSceneDevice& device, phi::Backend* backend) {
    auto* impl = const_cast<BatchedSensorSceneDevice&>(device).GetImpl();
    const auto ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    CheckCuda(cudaStreamSynchronize(ctx.stream), "capture imaging state");
    SensorStateSnapshot snapshot;
    snapshot.imaging = impl->imaging;
    snapshot.render_dr = impl->dr_cfg;
    snapshot.fidelity = impl->fid_cfg;
    snapshot.aov_mask = impl->aov_mask;
    snapshot.width = impl->image_width;
    snapshot.height = impl->image_height;
    snapshot.lidar_az = impl->lidar_az;
    snapshot.lidar_el = impl->lidar_el;
    const auto copy = [](const OwnedBuffer& buffer, size_t bytes, auto& values) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        if (bytes && buffer.Data()) {
            values.resize(bytes / sizeof(Value));
            buffer.CopyToHost(values.data(), bytes);
        }
    };
    if (snapshot.width) {
        if (impl->aov_mask & kSensorAovColor) copy(impl->d_color, impl->col_b, snapshot.color);
        if (impl->aov_mask & kSensorAovDepth) copy(impl->d_depth, impl->dep_b, snapshot.depth);
        if (impl->aov_mask & kSensorAovNormal) copy(impl->d_normal, impl->nrm_b, snapshot.normal);
        if (impl->aov_mask & kSensorAovAlbedo) copy(impl->d_albedo, impl->alb_b, snapshot.albedo);
        if (impl->aov_mask & kSensorAovPrim) copy(impl->d_prim, impl->prim_b, snapshot.prim);
    }
    copy(impl->d_range, impl->range_b, snapshot.range);
    CheckCuda(cudaStreamSynchronize(ctx.stream), "download imaging state");
    return snapshot;
}

void RestoreSensorState(BatchedSensorSceneDevice& device, const SensorStateSnapshot& snapshot,
                         phi::Backend* backend) {
    auto* impl = device.GetImpl();
    const auto& state = snapshot.imaging;
    if (state.cameras.size() != impl->imaging.cameras.size() || state.depths.size() != state.cameras.size() ||
        state.lidars.size() != impl->imaging.lidars.size() || state.sample_times.size() != state.env_count ||
        state.camera_stamps.size() != size_t{state.env_count} * state.cameras.size() ||
        state.lidar_stamps.size() != size_t{state.env_count} * state.lidars.size())
        throw std::invalid_argument("imaging checkpoint layout mismatch");
    for (const auto& config : state.cameras) if (!sensor::ValidCameraResponse(config))
        throw std::invalid_argument("invalid camera checkpoint response");
    for (const auto* configs : {&state.depths, &state.lidars})
        for (const auto& config : *configs) if (!sensor::ValidRangeResponse(config))
            throw std::invalid_argument("invalid range checkpoint response");
    const uint64_t pixels = uint64_t{state.env_count} * state.cameras.size() * snapshot.width * snapshot.height;
    const uint64_t rays = uint64_t{state.env_count} * state.lidars.size() * snapshot.lidar_az * snapshot.lidar_el;
    const auto valid_size = [](const auto& values, uint64_t expected) { return values.empty() || values.size() == expected; };
    if (pixels > UINT32_MAX || rays > UINT32_MAX || !valid_size(snapshot.color, pixels * 3u) ||
        !valid_size(snapshot.depth, pixels) || !valid_size(snapshot.normal, pixels * 3u) ||
        !valid_size(snapshot.albedo, pixels * 3u) || !valid_size(snapshot.prim, pixels) ||
        !valid_size(snapshot.range, rays) || (snapshot.aov_mask & ~kSensorAovAll))
        throw std::invalid_argument("invalid imaging checkpoint buffers");
    const auto ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    const auto restore = [&](OwnedBuffer& buffer, size_t& bytes, const auto& values) {
        if (values.empty()) {
            if (buffer.Data()) CheckCuda(cudaMemsetAsync(buffer.Data(), 0, bytes, ctx.stream), "clear imaging checkpoint view");
        } else UploadImagingVector(buffer, bytes, values, ctx);
    };
    restore(impl->d_color, impl->col_b, snapshot.color);
    restore(impl->d_depth, impl->dep_b, snapshot.depth);
    restore(impl->d_normal, impl->nrm_b, snapshot.normal);
    restore(impl->d_albedo, impl->alb_b, snapshot.albedo);
    restore(impl->d_prim, impl->prim_b, snapshot.prim);
    restore(impl->d_range, impl->range_b, snapshot.range);
    impl->imaging = state;
    impl->responses_dirty = true;
    impl->camera_stamps_dirty = impl->lidar_stamps_dirty = true;
    impl->image_width = snapshot.width;
    impl->image_height = snapshot.height;
    impl->aov_rays = pixels;
    impl->aov_mask = snapshot.aov_mask;
    impl->fid_cfg = snapshot.fidelity;
    impl->dr_cfg = snapshot.render_dr;
    impl->dr_env_count = 0u;
    impl->lidar_az = snapshot.lidar_az;
    impl->lidar_el = snapshot.lidar_el;
    impl->topology_built = false;
    CheckCuda(cudaStreamSynchronize(ctx.stream), "restore imaging state");
}

void RenderSensorsMounted(BatchedSensorSceneDevice& device,
                          const phi::ScatterFkSource& fk,
                          uint32_t env_count,
                          uint32_t width,
                          uint32_t height,
                          phi::Backend* backend) {
    BatchedSensorSceneDevice::Impl* impl = device.GetImpl();
    if (impl->sensors_per_env == 0u || impl->d_mounts.Data() == nullptr) {
        throw std::runtime_error(
            "RenderSensorsMounted: SetSensorMounts must be called first");
    }
    if (env_count == 0u || width == 0u || height == 0u) {
        return;
    }

    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);

    // Scatter cam_world = fk * local_offset per (env x sensor) into the persistent
    // E*S env-major camera buffer, then drive the SAME batched build/refit + flat
    // trace. With sensors_per_env==1 the camera index collapses to the env index.
    const uint64_t num_cameras =
        static_cast<uint64_t>(env_count) * impl->sensors_per_env;
    EnsureBytes(impl->d_cameras, impl->cam_b, ctx.device_bt,
                static_cast<std::size_t>(num_cameras) * sizeof(PinholeCamera));
    auto* d_cams = static_cast<PinholeCamera*>(impl->d_cameras.Data());
    ScatterEnvCameras(ctx.stream, fk,
                      static_cast<const SensorMountRow*>(impl->d_mounts.Data()),
                      env_count, impl->sensors_per_env, d_cams);
    RenderSensorsBatched(device, fk, d_cams, env_count, impl->sensors_per_env, width,
                         height, backend);
}

void RenderLidarsBatched(BatchedSensorSceneDevice& device,
                         const phi::ScatterFkSource& fk,
                         const LidarSensor* lidars_device,
                         uint32_t env_count,
                         uint32_t sensors_per_env,
                         uint32_t az_count,
                         uint32_t el_count,
                         phi::Backend* backend) {
    BatchedSensorSceneDevice::Impl* impl = device.GetImpl();
    const uint32_t m = impl->instances_per_env;
    const uint32_t s = sensors_per_env == 0u ? 1u : sensors_per_env;
    if (env_count == 0u || az_count == 0u || el_count == 0u || m == 0u) {
        return;
    }
    if (az_count > kMaxLidarAxis || el_count > kMaxLidarAxis) {
        throw std::runtime_error(
            "RenderLidarsBatched: az_count/el_count exceeds the lidar fan cap "
            "(kMaxLidarAxis=16384)");
    }

    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);
    phi::BufferType* bt = ctx.device_bt;

    const uint64_t num_lidars = static_cast<uint64_t>(env_count) * s;
    if (num_lidars > UINT32_MAX || uint64_t{az_count} * el_count > UINT32_MAX / num_lidars)
        throw std::invalid_argument("lidar ray count exceeds the supported index range");
    const uint64_t rays = num_lidars * az_count * el_count;
    const bool response = std::any_of(impl->imaging.lidars.begin(), impl->imaging.lidars.end(),
        [](const auto& config) { return config.enabled != 0u; });
    PrepareImaging(impl, ctx, env_count, s, true, response);
    if (response) EnsureEnvAppearanceTables(impl, ctx, env_count, false);
    EnsureBytes(impl->d_range, impl->range_b, bt, rays * sizeof(float));

    // Scatter + batched LBVH build/refit (the SAME topology the camera trace uses),
    // then the range-only trace over [E*S*az*el] on those per-env TLASes.
    EnsureEnvTopology(impl, ctx, fk, env_count);
    auto* d_instances = static_cast<DevInstance*>(impl->d_instances.Data());
    auto* d_nodes = static_cast<LbvhNode*>(impl->d_tlas_nodes.Data());

    const uint32_t grid = static_cast<uint32_t>((rays + kBlockSize - 1u) / kBlockSize);
    phi::LaunchCuda(BatchedLidarTraceKernel, dim3(grid), dim3(kBlockSize), 0u,
                    ctx.stream, lidars_device, d_nodes, m, d_instances,
                    static_cast<uint32_t>(num_lidars), s, az_count, el_count,
                    static_cast<const Material*>(impl->d_materials_env.Data()),
                    static_cast<const DevTexture*>(impl->d_textures.Data()), impl->material_count,
                    response ? static_cast<const sensor::RangeResponse*>(impl->d_lidar_responses.Data()) : nullptr,
                    static_cast<const sensor::noise::RangeSampleKeys*>(impl->d_lidar_keys.Data()),
                    static_cast<const sensor::ImagingStamp*>(impl->d_lidar_stamps.Data()),
                    static_cast<float*>(impl->d_range.Data()));
    CheckCuda(cudaGetLastError(), "BatchedLidarTraceKernel launch");
    impl->lidar_az = az_count;
    impl->lidar_el = el_count;
}

void SetLidarMounts(BatchedSensorSceneDevice& device,
                    const std::vector<scene::SensorDesc>& sensors) {
    if (sensors.empty()) {
        throw std::runtime_error("SetLidarMounts: empty mount table");
    }
    const std::vector<SensorMountRow> mounts = BuildSensorMountRows(sensors);
    // The range tensor is rectangular, so every lidar in the set shares one (az,el)
    // fan. Reject a ragged set + a nonsense pattern LOUDLY (no silent truncation).
    const uint32_t az = mounts.front().az_count;
    const uint32_t el = mounts.front().el_count;
    if (az == 0u || el == 0u) {
        throw std::runtime_error(
            "SetLidarMounts: lidar pattern has az_count/el_count == 0");
    }
    if (az > kMaxLidarAxis || el > kMaxLidarAxis) {
        throw std::runtime_error(
            "SetLidarMounts: az_count/el_count exceeds the lidar fan cap "
            "(kMaxLidarAxis=16384)");
    }
    for (const SensorMountRow& r : mounts) {
        if (r.az_count != az || r.el_count != el) {
            throw std::runtime_error(
                "SetLidarMounts: all lidars in a set must share one (az,el) fan");
        }
    }
    BatchedSensorSceneDevice::Impl* impl = device.GetImpl();
    const RtContext ctx = ResolveRtContext(nullptr);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);
    impl->d_lidar_mounts = UploadOwned(ctx.device_bt, mounts);
    impl->lidars_per_env = static_cast<uint32_t>(mounts.size());
    impl->imaging.lidars.resize(mounts.size());
    impl->imaging.lidar_stamps.assign(size_t{impl->imaging.env_count} * mounts.size(), {});
    impl->lidar_stamps_dirty = true;
    impl->responses_dirty = true;
    impl->lidar_az = az;
    impl->lidar_el = el;
    cudaStreamSynchronize(ctx.stream);
}

void RenderLidarsMounted(BatchedSensorSceneDevice& device,
                         const phi::ScatterFkSource& fk,
                         uint32_t env_count,
                         phi::Backend* backend) {
    BatchedSensorSceneDevice::Impl* impl = device.GetImpl();
    if (impl->lidars_per_env == 0u || impl->d_lidar_mounts.Data() == nullptr) {
        throw std::runtime_error(
            "RenderLidarsMounted: SetLidarMounts must be called first");
    }
    if (env_count == 0u) {
        return;
    }

    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);

    // Scatter origin/rotation = fk * local_offset per (env x lidar) into the
    // persistent E*S env-major LidarSensor buffer, then drive the range trace.
    const uint64_t num_lidars =
        static_cast<uint64_t>(env_count) * impl->lidars_per_env;
    EnsureBytes(impl->d_lidars, impl->lidars_b, ctx.device_bt,
                static_cast<std::size_t>(num_lidars) * sizeof(LidarSensor));
    auto* d_lidars = static_cast<LidarSensor*>(impl->d_lidars.Data());
    ScatterEnvLidars(ctx.stream, fk,
                     static_cast<const SensorMountRow*>(impl->d_lidar_mounts.Data()),
                     env_count, impl->lidars_per_env, d_lidars);
    RenderLidarsBatched(device, fk, d_lidars, env_count, impl->lidars_per_env,
                        impl->lidar_az, impl->lidar_el, backend);
}

const float* SensorColorDevice(const BatchedSensorSceneDevice& device) {
    auto* impl = const_cast<BatchedSensorSceneDevice&>(device).GetImpl();
    return (impl->aov_mask & kSensorAovColor) != 0u
               ? static_cast<const float*>(impl->d_color.Data())
               : nullptr;
}
const float* SensorDepthDevice(const BatchedSensorSceneDevice& device) {
    auto* impl = const_cast<BatchedSensorSceneDevice&>(device).GetImpl();
    return (impl->aov_mask & kSensorAovDepth) != 0u
               ? static_cast<const float*>(impl->d_depth.Data())
               : nullptr;
}
const float* SensorNormalDevice(const BatchedSensorSceneDevice& device) {
    auto* impl = const_cast<BatchedSensorSceneDevice&>(device).GetImpl();
    return (impl->aov_mask & kSensorAovNormal) != 0u
               ? static_cast<const float*>(impl->d_normal.Data())
               : nullptr;
}
const float* SensorAlbedoDevice(const BatchedSensorSceneDevice& device) {
    auto* impl = const_cast<BatchedSensorSceneDevice&>(device).GetImpl();
    return (impl->aov_mask & kSensorAovAlbedo) != 0u
               ? static_cast<const float*>(impl->d_albedo.Data())
               : nullptr;
}
const uint32_t* SensorPrimDevice(const BatchedSensorSceneDevice& device) {
    auto* impl = const_cast<BatchedSensorSceneDevice&>(device).GetImpl();
    return (impl->aov_mask & kSensorAovPrim) != 0u
               ? static_cast<const uint32_t*>(impl->d_prim.Data())
               : nullptr;
}

uint32_t SensorsPerEnv(const BatchedSensorSceneDevice& device) {
    return const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->sensors_per_env;
}

const float* SensorRangeDevice(const BatchedSensorSceneDevice& device) {
    return static_cast<const float*>(
        const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->d_range.Data());
}

uint32_t LidarsPerEnv(const BatchedSensorSceneDevice& device) {
    return const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->lidars_per_env;
}
uint32_t LidarAzCount(const BatchedSensorSceneDevice& device) {
    return const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->lidar_az;
}
uint32_t LidarElCount(const BatchedSensorSceneDevice& device) {
    return const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->lidar_el;
}

}  // namespace nuka::rt
