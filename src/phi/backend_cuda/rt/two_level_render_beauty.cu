// ---------------------------------------------------------------------------
// nuka::rt -- the FP32 BEAUTY translation unit of the two-level tracer. Compiled
// --fmad=true (FMA on) + instantiated Real=float so the dominant trace cost runs
// at full FP32+FMA rate, NOT the golden's FP64/1-64-rate path. The GOLDEN kernel
// (two_level_render.cu) stays FP64/--fmad=false and byte-exact; this path is
// stochastic + NON-golden, so beauty pixels may differ from the old FP64 beauty.
//
// Shares the SAME nested-traversal device fns (two_level_render_kernels.cuh)
// the golden uses -- only the Real template parameter + the TU compile flag
// differ. ONE general path; no per-scene branch.
// ---------------------------------------------------------------------------

#include "phi/backend_cuda/rt/two_level_render_kernels.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace nuka::rt {

namespace {

constexpr uint32_t kBeautyBlockThreads = 256u;

// Bound register use to leave resident warps for secondary-ray latency hiding.
constexpr uint32_t kBeautyMinBlocksPerSm = 3u;

// The beauty path runs every secondary ray + the slab/intersection math in FP32.
using Real = float;

// PCG32-ish hash RNG: cheap, well-distributed, deterministic from (seed, counter)
// so a fixed BeautyOptions::seed makes the whole frame reproducible run-to-run.
__device__ __forceinline__ uint32_t PcgHash(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct BeautyRng {
    uint32_t s;
    __device__ __forceinline__ float NextF() {
        s = PcgHash(s);
        // 24-bit mantissa float in [0,1).
        return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
    }
};

__device__ __forceinline__ Vec3 SumPixelLanes(Vec3 value, uint32_t mask, uint32_t lanes) {
    for (uint32_t offset = lanes / 2u; offset != 0u; offset /= 2u) {
        value.x += __shfl_down_sync(mask, value.x, offset, lanes);
        value.y += __shfl_down_sync(mask, value.y, offset, lanes);
        value.z += __shfl_down_sync(mask, value.z, offset, lanes);
    }
    return value;
}

uint32_t PixelSampleLanes(uint32_t samples) {
    uint32_t lanes = 32u;
    if (const char* text = std::getenv("NUKA_RT_SAMPLE_LANES")) {
        char* end = nullptr;
        const unsigned long value = std::strtoul(text, &end, 10);
        if (end != text && *end == '\0' && value >= 1u && value <= 32u &&
            (value & (value - 1u)) == 0u)
            lanes = static_cast<uint32_t>(value);
    }
    while (lanes > samples && lanes > 1u) lanes /= 2u;
    return lanes;
}

// OrthoBasis / CosineHemisphere / SampleCone / SkyColor / ShadeBeauty are SHARED
// (two_level_render_kernels.cuh), so the batched sensor TU shades identically.

// Color, albedo and shading normal share the jittered pixel filter.
// The center sample fills depth, UV and primitive identity.
__global__ void __launch_bounds__(kBeautyBlockThreads, kBeautyMinBlocksPerSm)
RenderBeautyKernel(PinholeCamera camera,
                                   const LbvhNode* __restrict__ tlas_nodes,
                                   uint32_t tlas_leaf_count,
                                   const DevInstance* __restrict__ instances,
                                   const Material* __restrict__ materials,
                                   const DevTexture* __restrict__ textures,
                                   Light light, BeautyParams sky, uint32_t samples,
                                   uint32_t base_seed, uint32_t sample_lanes,
                                   float* __restrict__ out_color,
                                   float* __restrict__ out_depth,
                                   float* __restrict__ out_normal,
                                   float* __restrict__ out_albedo,
                                   float* __restrict__ out_uv,
                                   uint32_t* __restrict__ out_prim) {
    const uint64_t thread = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t pixel = thread / sample_lanes;
    const uint32_t lane = static_cast<uint32_t>(thread % sample_lanes);
    const bool valid = pixel < static_cast<uint64_t>(camera.width) * camera.height;
    const uint32_t member_mask = __ballot_sync(0xffffffffu, valid);
    if (!valid) return;
    const uint32_t px = static_cast<uint32_t>(pixel % camera.width);
    const uint32_t py = static_cast<uint32_t>(pixel / camera.width);

    const uint32_t pixel_seed = PcgHash(base_seed ^ (static_cast<uint32_t>(pixel) * 2654435761u));
    const uint32_t S = samples < 1u ? 1u : samples;
    BeautyRng pixel_rng{PcgHash(pixel_seed ^ 0x68bc21ebu)};
    const float shift_x = pixel_rng.NextF(), shift_y = pixel_rng.NextF();
    const float inv_jittered = 1.0f / static_cast<float>(S > 1u ? S - 1u : 1u);

    Vec3 accum{0.0f, 0.0f, 0.0f};
    float c_depth = RtMissDepth();
    Vec3 sum_normal{0.0f, 0.0f, 0.0f}, sum_albedo{0.0f, 0.0f, 0.0f};
    float c_u = 0.0f, c_v = 0.0f;
    uint32_t c_prim = kNoPrim;

    for (uint64_t sample = lane; sample < S; sample += sample_lanes) {
        const uint32_t s = static_cast<uint32_t>(sample);
        // Sample identity, rather than lane scheduling, determines its random stream.
        BeautyRng rng{PcgHash(pixel_seed ^ (s * 0x9e3779b9u))};
        // A per-pixel rotated Hammersley set covers the pixel without random clumps.
        // The center sample retains the depth, UV and primitive selection contract.
        float jx = 0.0f, jy = 0.0f;
        if (s != 0u) {
            const float u = (static_cast<float>(s - 1u) + 0.5f) * inv_jittered + shift_x;
            const float v = static_cast<float>(__brev(s - 1u) >> 8u) *
                                (1.0f / 16777216.0f) + shift_y;
            jx = u - floorf(u) - 0.5f;
            jy = v - floorf(v) - 0.5f;
        }
        const Ray ray = camera.GenerateRayJitter(px, py, jx, jy);

        float bt; uint32_t bp;
        ClosestHit<Real>(tlas_nodes, tlas_leaf_count, instances, ray.origin, ray.dir, 0.0f,
                         &bt, &bp);
        if (bp == kNoPrim) {
            const Vec3 miss = SkyColor(ray.dir, sky);
            accum.x += miss.x; accum.y += miss.y; accum.z += miss.z;
            continue;
        }
        Vec3 n; float u, v;
        ReconstructHit<Real>(instances, bp, ray.origin, ray.dir, &n, &u, &v);
        const float nv = n.x * (-ray.dir.x) + n.y * (-ray.dir.y) + n.z * (-ray.dir.z);
        const Vec3 Nf = (nv < 0.0f) ? Vec3{-n.x, -n.y, -n.z} : n;
        uint32_t inst, lp; UnpackPrimId(bp, &inst, &lp);
        Material mat = materials[instances[inst].material_id];
        // Per-grain scatter tint: a hit sphere carrying a baked color multiplies the
        // material albedo. No-op unless the BLAS ships sphere colors (default null).
        if (instances[inst].blas.sph_color != nullptr) {
            const DevPrim gp = instances[inst].blas.prims[lp];
            if (gp.kind == static_cast<uint32_t>(PrimKind::Sphere)) {
                const Vec3 gc = instances[inst].blas.sph_color[gp.sub];
                mat.albedo = Vec3{mat.albedo.x * gc.x, mat.albedo.y * gc.y,
                                  mat.albedo.z * gc.z};
            }
        }
        const Vec3 hit{ray.origin.x + bt * ray.dir.x, ray.origin.y + bt * ray.dir.y,
                       ray.origin.z + bt * ray.dir.z};
        const Vec3 Vv = RtNormalize<Real>(Vec3{-ray.dir.x, -ray.dir.y, -ray.dir.z});
        Vec3 col, f_albedo, f_normal;
        if (mat.transmission > 0.0f) {
            // Dielectric arm: smooth normal + Fresnel reflect/refract + Beer; the
            // __noinline__ shader keeps the opaque (transmission==0) frame byte-exact.
            const Vec3 sn = SmoothWorldNormal(instances, bp, u, v, n);
            col = ShadeTransmissive(tlas_nodes, tlas_leaf_count, instances, materials,
                                    light, sky, hit, sn, Vv, mat, &rng, textures);
            f_albedo = Vec3{1.0f, 1.0f, 1.0f};
            f_normal = sn;
        } else {
            // Opaque arm: optional smooth normal, then the material's image maps
            // (albedo/roughness/normal) enrich the shade at this hit.
            Vec3 Nsh = Nf;
            if (sky.smooth_normals != 0u) {
                const Vec3 sn = SmoothWorldNormal(instances, bp, u, v, n);
                const float snv = sn.x * (-ray.dir.x) + sn.y * (-ray.dir.y) +
                                  sn.z * (-ray.dir.z);
                Nsh = (snv < 0.0f) ? Vec3{-sn.x, -sn.y, -sn.z} : sn;
            }
            Nsh = ApplyMaterialTextures(instances, textures, bp, hit, u, v, Nsh,
                                        &mat, true);
            WeaveHit weave;
            const bool woven = ApplyWeave(instances, bp, u, v, Nsh, &mat, &weave);
            col = ShadeBeauty(tlas_nodes, tlas_leaf_count, instances, materials,
                              light, sky, hit, Nsh, Vv, mat, &rng, textures, nullptr,
                              woven ? &weave : nullptr);
            f_albedo = mat.albedo;
            f_normal = woven ? weave.n_p : Nsh;
        }
        // Height/distance fog toward the sky-horizon: blend by 1-exp(-density*t).
        if (sky.fog_density > 0.0f) {
            const float f = 1.0f - expf(-sky.fog_density * bt);
            col.x += (sky.fog_color.x - col.x) * f;
            col.y += (sky.fog_color.y - col.y) * f;
            col.z += (sky.fog_color.z - col.z) * f;
        }
        accum.x += col.x; accum.y += col.y; accum.z += col.z;
        sum_albedo = sum_albedo + f_albedo;
        sum_normal = sum_normal + f_normal;

        if (s == 0u) {
            c_depth = bt; c_u = u; c_v = v; c_prim = bp;
        }
    }

    accum = SumPixelLanes(accum, member_mask, sample_lanes);
    sum_albedo = SumPixelLanes(sum_albedo, member_mask, sample_lanes);
    sum_normal = SumPixelLanes(sum_normal, member_mask, sample_lanes);
    if (lane != 0u) return;
    const float inv_s = 1.0f / static_cast<float>(S);
    if (out_color != nullptr) {
        out_color[pixel * 3u + 0u] = accum.x * inv_s;
        out_color[pixel * 3u + 1u] = accum.y * inv_s;
        out_color[pixel * 3u + 2u] = accum.z * inv_s;
    }
    if (out_depth != nullptr) out_depth[pixel] = c_depth;
    if (out_normal != nullptr) {
        out_normal[pixel * 3u + 0u] = sum_normal.x * inv_s;
        out_normal[pixel * 3u + 1u] = sum_normal.y * inv_s;
        out_normal[pixel * 3u + 2u] = sum_normal.z * inv_s;
    }
    if (out_albedo != nullptr) {
        out_albedo[pixel * 3u + 0u] = sum_albedo.x * inv_s;
        out_albedo[pixel * 3u + 1u] = sum_albedo.y * inv_s;
        out_albedo[pixel * 3u + 2u] = sum_albedo.z * inv_s;
    }
    if (out_uv != nullptr) {
        out_uv[pixel * 2u + 0u] = c_u;
        out_uv[pixel * 2u + 1u] = c_v;
    }
    if (out_prim != nullptr) out_prim[pixel] = c_prim;
}

}  // namespace

// Launch the FP32 beauty kernel on the resolved stream. The host build path
// (BuildFrameTlas + buffers + download) stays in two_level_render.cu.
void LaunchBeautyKernel(const PinholeCamera& camera,
                        const LbvhNode* tlas_nodes,
                        uint32_t tlas_leaf_count,
                        const DevInstance* instances,
                        const Material* materials,
                        const DevTexture* textures,
                        const Light& light,
                        const BeautyParams& sky,
                        uint32_t samples,
                        uint32_t base_seed,
                        const AovTarget& dst,
                        cudaStream_t stream) {
    const uint32_t lanes = PixelSampleLanes(samples);
    const dim3 block(kBeautyBlockThreads);
    const uint64_t threads = static_cast<uint64_t>(camera.width) * camera.height * lanes;
    const dim3 grid(static_cast<uint32_t>((threads + kBeautyBlockThreads - 1u) / kBeautyBlockThreads));
    RenderBeautyKernel<<<grid, block, 0, stream>>>(
        camera, tlas_nodes, tlas_leaf_count, instances, materials, textures, light,
        sky, samples, base_seed, lanes, dst.color, dst.depth, dst.normal, dst.albedo,
        dst.uv, dst.prim);
}

}  // namespace nuka::rt
