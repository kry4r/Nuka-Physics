// ---------------------------------------------------------------------------
// nuka::rt -- the BATCHED device-resident sensor render. ONE batched LBVH build
// over N per-env instance world-AABBs (collision/lbvh_batched.cuh) + ONE flat
// trace launch over [E*S*H*W] rays into a single (E,S,H,W,ch) device AOV tensor
// (S cameras per env, each on its own mount; S==1 collapses to (E,H,W,ch)).
//
// THE KERNEL is RenderFrameKernel (two_level_render.cu) generalized to a flat
// global ray index over cameras: cam = gid/(H*W), env = cam/S; it rebases the
// TLAS node slice (env*(2M-1)) + the DevInstance slice (env*M) per env (the scene
// is env-shared) and reuses the SAME ClosestHit / ReconstructHit / ShadeDirect
// nest, instantiated Real=float (the FP32 sensor cost). The single-camera FP64
// golden path is untouched -- this is an additive cheap-shade profile, NOT a branch.
//
// The TLAS leaf `.left` is the env-LOCAL instance index (the batched build's leaf
// payload), so `instances + env*M` resolves it directly; the prim_id pack uses an
// env-LOCAL instance id (rebased from the scatter's global id) so it fits the
// 12-bit instance field AND every env tile is byte-identical to a standalone
// single-env render of that env's scene + camera.
// ---------------------------------------------------------------------------

#include "phi/backend_cuda/rt/batched_sensor_render.hpp"

#include "collision/aabb.hpp"
#include "collision/lbvh_batched.cuh"
#include "collision/lbvh_node.cuh"
#include "math/vec3.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/rt/prim_id.cuh"
#include "phi/backend_cuda/rt/ray_box.cuh"
#include "phi/backend_cuda/rt/rt_device_context.cuh"  // RtContext / OwnedBuffer
#include "phi/backend_cuda/rt/sensor_scatter.hpp"
#include "phi/backend_cuda/rt/shading.cuh"
#include "phi/backend_cuda/rt/two_level_render_kernels.cuh"
#include "phi/scoped_device_guard.hpp"
#include "rt/render_dr.hpp"
#include "rt/sensor_fidelity.hpp"   // SensorFidelityConfig (opt-in beauty shade)
#include "sensor/noise/philox.cuh"  // Philox4x32 host/device pure RNG

#include <cuda_runtime.h>

#include <cstdint>
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

// Device mirror of the fidelity flags the trace branches on (sky/sample params
// travel in BeautyParams; spp/samples are host-capped before launch).
struct FidelityParams {
    uint32_t spp;
    uint32_t enabled;  // 0 -> the exact cheap-shade arithmetic (byte no-op)
    uint32_t tonemap;
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

// One thread per GLOBAL ray over [num_cameras*H*W]: flat gid -> (camera, px, py).
// Cameras are env-major (cameras[env*S+s]); the env owning a camera is cam/S, and
// the TLAS node + instance + material slices index by env (the scene is env-shared
// -- only cameras fan out). One writer per pixel, no atomics -> FP32-deterministic
// byte-exact per tile. S==1 collapses cam==env, byte-identical to one cam per env.
//
// fid.enabled==0 takes the EXACT cheap-shade arithmetic (1 spp, 1 hard shadow,
// ShadeDirect) -> the AOV bytes are unchanged. fid.enabled!=0 accumulates spp
// jittered samples through the shared beauty shade (soft shadow / AO / GI per the
// sky params) and averages into color; depth/normal/albedo/prim ALWAYS come from
// the center ray, so only color changes under fidelity.
__global__ void BatchedSensorTraceKernel(const PinholeCamera* __restrict__ cameras,
                                         const LbvhNode* __restrict__ tlas_nodes,
                                         uint32_t leaves_per_env,
                                         const DevInstance* __restrict__ instances,
                                         const Material* __restrict__ materials,
                                         uint32_t material_count,
                                         const Light* __restrict__ lights,
                                         const AmbientTerm* __restrict__ ambients,
                                         uint32_t num_cameras,
                                         uint32_t sensors_per_env,
                                         uint32_t width,
                                         uint32_t height,
                                         FidelityParams fid,
                                         BeautyParams sky,
                                         float* __restrict__ out_color,
                                         float* __restrict__ out_depth,
                                         float* __restrict__ out_normal,
                                         float* __restrict__ out_albedo,
                                         uint32_t* __restrict__ out_prim) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pix_per_cam = width * height;
    const uint64_t total = static_cast<uint64_t>(num_cameras) * pix_per_cam;
    if (static_cast<uint64_t>(gid) >= total) return;

    const uint32_t cam = gid / pix_per_cam;
    const uint32_t env = cam / sensors_per_env;
    const uint32_t p = gid % pix_per_cam;
    const uint32_t px = p % width;
    const uint32_t py = p / width;

    const PinholeCamera camera = cameras[cam];
    const Ray ray = camera.GenerateRay(px, py);

    // Per-env appearance: light/ambient by env, materials env-offset by env*M.
    // DR off -> these tables are base replicas, so the bytes match the env-shared
    // look exactly (ONE path: always read BY ENV).
    const Light light = lights[env];
    const AmbientTerm ambient = ambients[env];
    const Material* env_mats = materials + static_cast<uint64_t>(env) * material_count;

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

    if (best_prim != kNoPrim) {
        depth = best_t;
        prim_id = best_prim;

        uint32_t inst, local_prim;
        UnpackPrimId(best_prim, &inst, &local_prim);
        const uint32_t material_id = env_inst[inst].material_id;

        Vec3 n;
        float uv_u, uv_v;
        ReconstructHit<float>(env_inst, best_prim, ray.origin, ray.dir, &n, &uv_u, &uv_v);
        normal = n;
        const Material mat = env_mats[material_id];
        albedo = mat.albedo;

        const Vec3 hit{ray.origin.x + best_t * ray.dir.x,
                       ray.origin.y + best_t * ray.dir.y,
                       ray.origin.z + best_t * ray.dir.z};
        const Vec3 V = RtNormalize<float>(Vec3{-ray.dir.x, -ray.dir.y, -ray.dir.z});
        Vec3 L;
        float light_dist;
        if (light.directional) {
            L = RtNormalize<float>(Vec3{-light.direction.x, -light.direction.y, -light.direction.z});
            light_dist = RtMissDepth();
        } else {
            const Vec3 to{light.position.x - hit.x, light.position.y - hit.y,
                          light.position.z - hit.z};
            L = RtNormalize<float>(to);
            light_dist = sqrtf(to.x * to.x + to.y * to.y + to.z * to.z);
        }

        // One hard shadow ray through the SAME nest (sensor profile: 1 shadow,
        // no AO/GI/MSAA). Offset along the viewer-faced normal to avoid acne.
        Vec3 ns = n;
        const float nv = n.x * V.x + n.y * V.y + n.z * V.z;
        if (nv < 0.0f) {
            ns = Vec3{-n.x, -n.y, -n.z};
        }
        const float shadow_eps = 1.0e-3f;
        const Vec3 sorigin{hit.x + ns.x * shadow_eps, hit.y + ns.y * shadow_eps,
                           hit.z + ns.z * shadow_eps};
        int lit = 1;
        float st;
        uint32_t sp;
        ClosestHit<float>(env_nodes, leaves_per_env, env_inst, sorigin, L, shadow_eps,
                          &st, &sp);
        if (sp != kNoPrim && st < light_dist - shadow_eps) {
            lit = 0;
        }

        const Vec3 light_col{light.color.x * light.intensity,
                             light.color.y * light.intensity,
                             light.color.z * light.intensity};
        color = ShadeDirect(n, V, L, light_col, mat, lit, ambient.color);
    }

    // Opt-in beauty shade: accumulate spp jittered samples through the SHARED
    // beauty shade (the single-camera look) and average. The center sample's AOVs
    // above are kept; only color is replaced. Seeded per (ray, sample) -> two-run
    // byte-exact. Misses sample the sky dome instead of staying black.
    if (fid.enabled != 0u) {
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
            Vec3 col = ShadeBeauty<PhiloxSeededRng>(env_nodes, leaves_per_env, env_inst,
                                                    env_mats, light, sky, shit, snf, sV,
                                                    smat, &rng);
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
        if (fid.tonemap != 0u) {
            color = Vec3{TonemapAces(color.x), TonemapAces(color.y), TonemapAces(color.z)};
        }
    }

    out_color[gid * 3u + 0u] = color.x;
    out_color[gid * 3u + 1u] = color.y;
    out_color[gid * 3u + 2u] = color.z;
    out_depth[gid] = depth;
    out_normal[gid * 3u + 0u] = normal.x;
    out_normal[gid * 3u + 1u] = normal.y;
    out_normal[gid * 3u + 2u] = normal.z;
    out_albedo[gid * 3u + 0u] = albedo.x;
    out_albedo[gid * 3u + 1u] = albedo.y;
    out_albedo[gid * 3u + 2u] = albedo.z;
    out_prim[gid] = prim_id;
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
    uint32_t instances_per_env = 0u;
    uint32_t material_count = 0u;

    // Per-env appearance tables (the ONE thing the trace reads): materials [E*M],
    // light/ambient [E]. DR off -> exact base replicas (byte-identical tiles).
    OwnedBuffer d_materials_env, d_light_env, d_ambient_env;
    std::size_t matenv_b = 0u, lightenv_b = 0u, ambenv_b = 0u;
    uint32_t dr_env_count = 0u;  // env count the per-env tables are filled for
    rt::RenderDrConfig dr_cfg;   // disabled by default -> replicas

    // Opt-in shading fidelity. Default (Enabled()==false) -> the trace takes the
    // exact cheap-shade arithmetic (byte-identical to today).
    rt::SensorFidelityConfig fid_cfg;

    // Scatter outputs + batched TLAS (sized to E*M / E*(2M-1)); *_b track each
    // allocation's byte size so growth-only realloc is honest across steps.
    OwnedBuffer d_instances, d_world_aabbs, d_tlas_nodes;
    OwnedBuffer d_morton, d_index, d_sortkey, d_visit;
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
    for (uint32_t b : desc.blas_id) {
        if (b >= refs.size()) {
            throw std::runtime_error("BuildBatchedSensorScene: blas_id out of range");
        }
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

// Per-env appearance table cap. A vision DR run is thousands of envs; the table
// stays trivial (a 4-material set at 65536 envs is ~8 MB), but cap it LOUDLY so a
// nonsense env_count fails fast instead of overflowing the allocation.
constexpr uint32_t kMaxRenderDrEnvs = 1u << 20;  // 1,048,576 envs

// Fidelity sample caps: each pixel runs spp * (shadow_samples + ao_samples) trace
// rays, so a runaway profile would stall the launch. Cap LOUDLY so a nonsense
// profile fails fast instead of hanging the device.
constexpr uint32_t kMaxFidelitySpp = 256u;
constexpr uint32_t kMaxFidelitySamples = 256u;  // per soft-shadow / AO dimension

// Lower the stored fidelity config to the kernel's device params, asserting the
// sample caps. enabled==false -> the kernel takes the exact cheap-shade branch.
void BuildFidelityParams(const rt::SensorFidelityConfig& cfg, FidelityParams* fp,
                         BeautyParams* sky) {
    if (cfg.spp > kMaxFidelitySpp) {
        throw std::runtime_error(
            "SetSensorFidelity: spp exceeds the per-pixel sample cap "
            "(kMaxFidelitySpp=256)");
    }
    if (cfg.shadow_samples > kMaxFidelitySamples ||
        cfg.ao_samples > kMaxFidelitySamples) {
        throw std::runtime_error(
            "SetSensorFidelity: shadow_samples/ao_samples exceed the sample cap "
            "(kMaxFidelitySamples=256)");
    }
    fp->spp = cfg.spp < 1u ? 1u : cfg.spp;
    fp->enabled = cfg.Enabled() ? 1u : 0u;
    fp->tonemap = cfg.tonemap_enabled ? 1u : 0u;
    fp->seed = cfg.seed;
    sky->shadow_rays = cfg.shadow_samples < 1u ? 1u : cfg.shadow_samples;
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

    ScatterEnvInstances(ctx.stream, fk,
                        static_cast<const phi::InstanceScatterRow*>(impl->d_rows.Data()),
                        static_cast<const uint32_t*>(impl->d_blas_id.Data()),
                        static_cast<const uint32_t*>(impl->d_material_id.Data()),
                        static_cast<const SensorBlasRef*>(impl->d_blas_refs.Data()),
                        env_count, m, d_instances, d_world_aabbs);
    {
        const uint32_t grid = (static_cast<uint32_t>(total_inst) + kBlockSize - 1u) / kBlockSize;
        phi::LaunchCuda(RebaseInstanceIdsKernel, dim3(grid), dim3(kBlockSize), 0u,
                        ctx.stream, d_instances, env_count, m);
    }
    if (need_rebuild) {
        collision::gpu::BuildLbvhBatchedNodes(
            ctx.stream, ctx.device_id, d_world_aabbs, env_count, m, d_nodes,
            static_cast<uint32_t*>(impl->d_morton.Data()),
            static_cast<uint32_t*>(impl->d_index.Data()),
            static_cast<uint64_t*>(impl->d_sortkey.Data()),
            static_cast<uint32_t*>(impl->d_visit.Data()));
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
    if (m < 2u) {
        throw std::runtime_error(
            "RenderSensorsBatched: the batched LBVH build needs >=2 instances/env");
    }

    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);
    phi::BufferType* bt = ctx.device_bt;

    // Scene is env-shared (E trees of M instances); cameras fan out E*S env-major.
    const uint64_t num_cameras = static_cast<uint64_t>(env_count) * s;
    const uint64_t rays = num_cameras * width * height;

    EnsureBytes(impl->d_color, impl->col_b, bt, rays * 3u * sizeof(float));
    EnsureBytes(impl->d_depth, impl->dep_b, bt, rays * sizeof(float));
    EnsureBytes(impl->d_normal, impl->nrm_b, bt, rays * 3u * sizeof(float));
    EnsureBytes(impl->d_albedo, impl->alb_b, bt, rays * 3u * sizeof(float));
    EnsureBytes(impl->d_prim, impl->prim_b, bt, rays * sizeof(uint32_t));

    // Per-env material/light/ambient tables (refilled on an env-count change; DR
    // off -> base replicas). The trace reads these BY ENV -- the ONE path.
    EnsureEnvAppearanceTables(impl, ctx, env_count, false);

    // 1+2) Scatter + batched LBVH build/refit (the ONE topology path lidar reuses).
    EnsureEnvTopology(impl, ctx, fk, env_count);
    auto* d_instances = static_cast<DevInstance*>(impl->d_instances.Data());
    auto* d_nodes = static_cast<LbvhNode*>(impl->d_tlas_nodes.Data());

    // 3) ONE flat trace over [E*S*H*W] into the persistent (E,S,H,W,ch) AOV tensor.
    //    Materials/light/ambient resolved BY ENV from the per-env tables. The
    //    fidelity params gate the shade: disabled -> the exact cheap arithmetic.
    FidelityParams fid;
    BeautyParams sky;
    BuildFidelityParams(impl->fid_cfg, &fid, &sky);
    const uint32_t grid = static_cast<uint32_t>((rays + kBlockSize - 1u) / kBlockSize);
    phi::LaunchCuda(BatchedSensorTraceKernel, dim3(grid), dim3(kBlockSize), 0u,
                    ctx.stream, cameras_device, d_nodes, m, d_instances,
                    static_cast<const Material*>(impl->d_materials_env.Data()),
                    impl->material_count,
                    static_cast<const Light*>(impl->d_light_env.Data()),
                    static_cast<const AmbientTerm*>(impl->d_ambient_env.Data()),
                    static_cast<uint32_t>(num_cameras), s,
                    width, height, fid, sky,
                    static_cast<float*>(impl->d_color.Data()),
                    static_cast<float*>(impl->d_depth.Data()),
                    static_cast<float*>(impl->d_normal.Data()),
                    static_cast<float*>(impl->d_albedo.Data()),
                    static_cast<uint32_t*>(impl->d_prim.Data()));
    CheckCuda(cudaGetLastError(), "BatchedSensorTraceKernel launch");
    impl->aov_rays = rays;
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
    // Validate the caps here too so a bad config is rejected at set time, not only
    // at the next render. The stored config drives the trace's shade branch.
    FidelityParams fp;
    BeautyParams sky;
    BuildFidelityParams(cfg, &fp, &sky);
    impl->fid_cfg = cfg;
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
    if (m < 2u) {
        throw std::runtime_error(
            "RenderLidarsBatched: the batched LBVH build needs >=2 instances/env");
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
    const uint64_t rays = num_lidars * az_count * el_count;
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
    return static_cast<const float*>(
        const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->d_color.Data());
}
const float* SensorDepthDevice(const BatchedSensorSceneDevice& device) {
    return static_cast<const float*>(
        const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->d_depth.Data());
}
const float* SensorNormalDevice(const BatchedSensorSceneDevice& device) {
    return static_cast<const float*>(
        const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->d_normal.Data());
}
const float* SensorAlbedoDevice(const BatchedSensorSceneDevice& device) {
    return static_cast<const float*>(
        const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->d_albedo.Data());
}
const uint32_t* SensorPrimDevice(const BatchedSensorSceneDevice& device) {
    return static_cast<const uint32_t*>(
        const_cast<BatchedSensorSceneDevice&>(device).GetImpl()->d_prim.Data());
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
