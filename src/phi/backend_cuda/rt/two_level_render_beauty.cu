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

namespace nuka::rt {

namespace {

constexpr uint32_t kBlockDim = 16u;
constexpr uint32_t kBeautyBlockThreads = kBlockDim * kBlockDim;

// Target resident blocks per SM for the beauty kernel. At the un-bounded 220
// registers only one 256-thread block fits an sm_86 SM (~17% occupancy); bounding
// to two blocks caps the kernel at 128 registers (a small, measured spill) and the
// doubled occupancy hides the incoherent secondary-ray latency -> ~1.4x measured.
constexpr uint32_t kBeautyMinBlocksPerSm = 2u;

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

// OrthoBasis / CosineHemisphere / SampleCone / SkyColor / ShadeBeauty are SHARED
// (two_level_render_kernels.cuh), so the batched sensor TU shades identically.

// One thread per pixel: accumulate S jittered sub-pixel samples through the
// stochastic shade. The CENTER sample also fills depth/normal/albedo/uv/prim so
// the host RGBA8 conversion's prim-based hit test still classifies background.
__global__ void __launch_bounds__(kBeautyBlockThreads, kBeautyMinBlocksPerSm)
RenderBeautyKernel(PinholeCamera camera,
                                   const LbvhNode* __restrict__ tlas_nodes,
                                   uint32_t tlas_leaf_count,
                                   const DevInstance* __restrict__ instances,
                                   const Material* __restrict__ materials,
                                   const DevTexture* __restrict__ textures,
                                   Light light, BeautyParams sky, uint32_t samples,
                                   uint32_t base_seed,
                                   float* __restrict__ out_color,
                                   float* __restrict__ out_depth,
                                   float* __restrict__ out_normal,
                                   float* __restrict__ out_albedo,
                                   float* __restrict__ out_uv,
                                   uint32_t* __restrict__ out_prim) {
    const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= camera.width || py >= camera.height) return;
    const uint32_t pixel = py * camera.width + px;

    BeautyRng rng{PcgHash(base_seed ^ (pixel * 2654435761u))};
    const uint32_t S = samples < 1u ? 1u : samples;

    Vec3 accum{0.0f, 0.0f, 0.0f};
    float c_depth = RtMissDepth();
    Vec3 c_normal{0.0f, 0.0f, 0.0f}, c_albedo{0.0f, 0.0f, 0.0f};
    float c_u = 0.0f, c_v = 0.0f;
    uint32_t c_prim = kNoPrim;

    for (uint32_t s = 0; s < S; ++s) {
        // Sub-pixel jitter in [-0.5,0.5]; sample 0 uses center for the AOV fill.
        const float jx = (s == 0u) ? 0.0f : (rng.NextF() - 0.5f);
        const float jy = (s == 0u) ? 0.0f : (rng.NextF() - 0.5f);
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
        const Vec3 hit{ray.origin.x + bt * ray.dir.x, ray.origin.y + bt * ray.dir.y,
                       ray.origin.z + bt * ray.dir.z};
        const Vec3 Vv = RtNormalize<Real>(Vec3{-ray.dir.x, -ray.dir.y, -ray.dir.z});
        Vec3 col;
        if (mat.transmission > 0.0f) {
            // Dielectric arm: smooth normal + Fresnel reflect/refract + Beer; the
            // __noinline__ shader keeps the opaque (transmission==0) frame byte-exact.
            const Vec3 sn = SmoothWorldNormal(instances, bp, u, v, n);
            const float tv = sn.x * (-ray.dir.x) + sn.y * (-ray.dir.y) + sn.z * (-ray.dir.z);
            const Vec3 Nt = (tv < 0.0f) ? Vec3{-sn.x, -sn.y, -sn.z} : sn;
            col = ShadeTransmissive(tlas_nodes, tlas_leaf_count, instances, materials,
                                    light, sky, hit, Nt, Vv, mat, &rng, textures);
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
            col = ShadeBeauty(tlas_nodes, tlas_leaf_count, instances, materials,
                              light, sky, hit, Nsh, Vv, mat, &rng, textures);
        }
        // Height/distance fog toward the sky-horizon: blend by 1-exp(-density*t).
        if (sky.fog_density > 0.0f) {
            const float f = 1.0f - expf(-sky.fog_density * bt);
            col.x += (sky.fog_color.x - col.x) * f;
            col.y += (sky.fog_color.y - col.y) * f;
            col.z += (sky.fog_color.z - col.z) * f;
        }
        accum.x += col.x; accum.y += col.y; accum.z += col.z;

        if (s == 0u) {
            c_depth = bt; c_normal = n; c_albedo = mat.albedo;
            c_u = u; c_v = v; c_prim = bp;
        }
    }

    const float inv_s = 1.0f / static_cast<float>(S);
    if (out_color != nullptr) {
        out_color[pixel * 3u + 0u] = accum.x * inv_s;
        out_color[pixel * 3u + 1u] = accum.y * inv_s;
        out_color[pixel * 3u + 2u] = accum.z * inv_s;
    }
    if (out_depth != nullptr) out_depth[pixel] = c_depth;
    if (out_normal != nullptr) {
        out_normal[pixel * 3u + 0u] = c_normal.x;
        out_normal[pixel * 3u + 1u] = c_normal.y;
        out_normal[pixel * 3u + 2u] = c_normal.z;
    }
    if (out_albedo != nullptr) {
        out_albedo[pixel * 3u + 0u] = c_albedo.x;
        out_albedo[pixel * 3u + 1u] = c_albedo.y;
        out_albedo[pixel * 3u + 2u] = c_albedo.z;
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
    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid((camera.width + kBlockDim - 1u) / kBlockDim,
                    (camera.height + kBlockDim - 1u) / kBlockDim);
    RenderBeautyKernel<<<grid, block, 0, stream>>>(
        camera, tlas_nodes, tlas_leaf_count, instances, materials, textures, light,
        sky, samples, base_seed, dst.color, dst.depth, dst.normal, dst.albedo,
        dst.uv, dst.prim);
}

}  // namespace nuka::rt
