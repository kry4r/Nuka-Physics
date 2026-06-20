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

// Orthonormal basis around a unit normal (Duff et al. 2017, branchless).
__device__ __forceinline__ void OrthoBasis(const Vec3& n, Vec3* t, Vec3* b) {
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    *t = Vec3{1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x};
    *b = Vec3{c, sign + n.y * n.y * a, -n.y};
}

// Cosine-weighted hemisphere sample about `n` (Malley's method).
__device__ __forceinline__ Vec3 CosineHemisphere(const Vec3& n, float u1, float u2) {
    const float r = sqrtf(u1);
    const float phi = 6.2831853071795864769f * u2;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    Vec3 t, b;
    OrthoBasis(n, &t, &b);
    return RtNormalize<Real>(Vec3{t.x * x + b.x * y + n.x * z,
                                  t.y * x + b.y * y + n.y * z,
                                  t.z * x + b.z * y + n.z * z});
}

// Sample a cone of half-angle `ang` about the unit axis `L` (soft sun disc).
__device__ __forceinline__ Vec3 SampleCone(const Vec3& L, float ang, float u1, float u2) {
    const float cos_max = cosf(ang);
    const float cos_t = 1.0f - u1 * (1.0f - cos_max);
    const float sin_t = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
    const float phi = 6.2831853071795864769f * u2;
    Vec3 t, b;
    OrthoBasis(L, &t, &b);
    const float x = sin_t * cosf(phi);
    const float y = sin_t * sinf(phi);
    return RtNormalize<Real>(Vec3{t.x * x + b.x * y + L.x * cos_t,
                                  t.y * x + b.y * y + L.y * cos_t,
                                  t.z * x + b.z * y + L.z * cos_t});
}

// Procedural sky + ground dome along a ray direction (the miss shader). Smooth
// horizon blend; below-horizon fades to a neutral ground fill so bounce rays that
// escape downward still pick up plausible fill instead of pure black.
__device__ __forceinline__ Vec3 SkyColor(const Vec3& dir, const BeautyParams& sky) {
    if (dir.z >= 0.0f) {
        const float up = dir.z;                        // 0 horizon .. 1 zenith
        const float w = up * up;                       // bias color toward horizon
        return Vec3{sky.sky_bottom.x + (sky.sky_top.x - sky.sky_bottom.x) * w,
                    sky.sky_bottom.y + (sky.sky_top.y - sky.sky_bottom.y) * w,
                    sky.sky_bottom.z + (sky.sky_top.z - sky.sky_bottom.z) * w};
    }
    const float w = fminf(1.0f, -dir.z * 1.5f);
    return Vec3{sky.sky_bottom.x + (sky.sky_ground.x - sky.sky_bottom.x) * w,
                sky.sky_bottom.y + (sky.sky_ground.y - sky.sky_bottom.y) * w,
                sky.sky_bottom.z + (sky.sky_ground.z - sky.sky_bottom.z) * w};
}

// Per-pixel hit at world point `hit` with viewer-faced normal `Nf`: K-ray soft
// sun shadow + cosine AO/GI bounces. Returns the lit RGB. Visibility rays use the
// any-hit traversal; the AO/GI bounce uses a closest-hit pruned to ao_radius.
__device__ __forceinline__ Vec3 ShadeBeauty(const LbvhNode* __restrict__ tlas_nodes,
                                            uint32_t tlas_leaf_count,
                                            const DevInstance* __restrict__ instances,
                                            const Material* __restrict__ materials,
                                            Light light, BeautyParams sky,
                                            const Vec3& hit, const Vec3& Nf, const Vec3& V,
                                            const Material& mat, BeautyRng* rng) {
    // Sun direction (toward the light) + a finite angular size -> penumbra.
    Vec3 Ls;
    if (light.directional) {
        Ls = RtNormalize<Real>(Vec3{-light.direction.x, -light.direction.y, -light.direction.z});
    } else {
        Ls = RtNormalize<Real>(Vec3{light.position.x - hit.x, light.position.y - hit.y,
                                    light.position.z - hit.z});
    }
    const Vec3 light_col{light.color.x * light.intensity,
                         light.color.y * light.intensity,
                         light.color.z * light.intensity};
    const float eps = 1.0e-3f;
    const Vec3 sorigin{hit.x + Nf.x * eps, hit.y + Nf.y * eps, hit.z + Nf.z * eps};

    // Soft shadow: average visibility over K cone-sampled sun directions (any-hit
    // visibility -- boolean occlusion, first hit wins).
    float vis = 0.0f;
    const uint32_t K = sky.shadow_rays < 1u ? 1u : sky.shadow_rays;
    for (uint32_t k = 0; k < K; ++k) {
        const Vec3 Lk = SampleCone(Ls, sky.sun_angular_radius, rng->NextF(), rng->NextF());
        if (Lk.x * Nf.x + Lk.y * Nf.y + Lk.z * Nf.z <= 0.0f) continue;
        if (!AnyOccluder<Real>(tlas_nodes, tlas_leaf_count, instances, sorigin, Lk, eps,
                               RtMissDepth())) {
            vis += 1.0f;
        }
    }
    vis /= static_cast<float>(K);

    const float NoL = fmaxf(0.0f, Nf.x * Ls.x + Nf.y * Ls.y + Nf.z * Ls.z);
    // Half vector for a GGX-ish specular highlight so the shell catches the sun.
    Vec3 H = RtNormalize<Real>(Vec3{V.x + Ls.x, V.y + Ls.y, V.z + Ls.z});
    const float NoH = fmaxf(0.0f, Nf.x * H.x + Nf.y * H.y + Nf.z * H.z);
    const float alpha = fmaxf(1.0e-3f, mat.roughness * mat.roughness);
    const float a2 = alpha * alpha;
    const float dterm = NoH * NoH * (a2 - 1.0f) + 1.0f;
    const float spec = (a2 / (3.14159265f * dterm * dterm)) * (0.04f + mat.metallic);

    const float kd = (1.0f - mat.metallic) * (1.0f / 3.14159265f);
    Vec3 direct{
        light_col.x * (mat.albedo.x * kd + spec) * NoL * vis,
        light_col.y * (mat.albedo.y * kd + spec) * NoL * vis,
        light_col.z * (mat.albedo.z * kd + spec) * NoL * vis};

    // AO + one-bounce GI: cosine-weighted hemisphere rays. A ray that escapes the
    // AO radius (or misses) samples the sky dome; a near hit darkens the crease
    // and (for GI) picks up the bounce albedo*shade of the hit surface. The
    // closest-hit is pruned to ao_radius (far subtrees skipped).
    Vec3 indirect{0.0f, 0.0f, 0.0f};
    const uint32_t M = sky.ao_samples < 1u ? 1u : sky.ao_samples;
    for (uint32_t m = 0; m < M; ++m) {
        const Vec3 d = CosineHemisphere(Nf, rng->NextF(), rng->NextF());
        const Vec3 ro{hit.x + Nf.x * eps, hit.y + Nf.y * eps, hit.z + Nf.z * eps};
        float bt; uint32_t bp;
        ClosestHit<Real>(tlas_nodes, tlas_leaf_count, instances, ro, d, eps, &bt, &bp,
                         sky.ao_radius);
        if (bp == kNoPrim || bt >= sky.ao_radius) {
            // Open sky above -> sky-dome ambient (cosine already in the sample pdf).
            const Vec3 s = SkyColor(d, sky);
            indirect.x += s.x * sky.sky_intensity;
            indirect.y += s.y * sky.sky_intensity;
            indirect.z += s.z * sky.sky_intensity;
            continue;
        }
        if (sky.gi_bounces == 0u) continue;  // AO only: occluded -> no light
        // One bounce: re-shade the hit surface with a direct sun term (no further
        // recursion) and add its albedo-weighted contribution.
        Vec3 bn; float bu, bv;
        ReconstructHit<Real>(instances, bp, ro, d, &bn, &bu, &bv);
        const float bnv = bn.x * (-d.x) + bn.y * (-d.y) + bn.z * (-d.z);
        Vec3 bnf = (bnv < 0.0f) ? Vec3{-bn.x, -bn.y, -bn.z} : bn;
        uint32_t bi, blp; UnpackPrimId(bp, &bi, &blp);
        const Material bmat = materials[instances[bi].material_id];
        const Vec3 bhit{ro.x + bt * d.x, ro.y + bt * d.y, ro.z + bt * d.z};
        const Vec3 bso{bhit.x + bnf.x * eps, bhit.y + bnf.y * eps, bhit.z + bnf.z * eps};
        const bool bshadow =
            AnyOccluder<Real>(tlas_nodes, tlas_leaf_count, instances, bso, Ls, eps,
                              RtMissDepth());
        const float bNoL = fmaxf(0.0f, bnf.x * Ls.x + bnf.y * Ls.y + bnf.z * Ls.z);
        const float bvis = bshadow ? 0.0f : 1.0f;
        const float bfac = bNoL * bvis * (1.0f / 3.14159265f);
        indirect.x += bmat.albedo.x * light_col.x * bfac;
        indirect.y += bmat.albedo.y * light_col.y * bfac;
        indirect.z += bmat.albedo.z * light_col.z * bfac;
    }
    const float inv_m = 1.0f / static_cast<float>(M);
    indirect.x *= inv_m; indirect.y *= inv_m; indirect.z *= inv_m;

    // Indirect modulates the surface albedo (Lambert response to ambient/bounce).
    return Vec3{direct.x + mat.albedo.x * indirect.x,
                direct.y + mat.albedo.y * indirect.y,
                direct.z + mat.albedo.z * indirect.z};
}

// One thread per pixel: accumulate S jittered sub-pixel samples through the
// stochastic shade. The CENTER sample also fills depth/normal/albedo/uv/prim so
// the host RGBA8 conversion's prim-based hit test still classifies background.
__global__ void RenderBeautyKernel(PinholeCamera camera,
                                   const LbvhNode* __restrict__ tlas_nodes,
                                   uint32_t tlas_leaf_count,
                                   const DevInstance* __restrict__ instances,
                                   const Material* __restrict__ materials,
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
        const Material mat = materials[instances[inst].material_id];
        const Vec3 hit{ray.origin.x + bt * ray.dir.x, ray.origin.y + bt * ray.dir.y,
                       ray.origin.z + bt * ray.dir.z};
        const Vec3 Vv = RtNormalize<Real>(Vec3{-ray.dir.x, -ray.dir.y, -ray.dir.z});
        Vec3 col = ShadeBeauty(tlas_nodes, tlas_leaf_count, instances, materials,
                               light, sky, hit, Nf, Vv, mat, &rng);
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
        camera, tlas_nodes, tlas_leaf_count, instances, materials, light, sky,
        samples, base_seed, dst.color, dst.depth, dst.normal, dst.albedo, dst.uv,
        dst.prim);
}

}  // namespace nuka::rt
