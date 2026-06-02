#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::sdf – sparse narrow-band SDF device/host sample API (v0.7 p07)
// ---------------------------------------------------------------------------
// The runtime side of the cooked sparse SDF. The cooker (src/import/cooker/
// sparse_sdf_cooker.*) produces, per convex-geometry piece, a narrow band of
// signed-distance + gradient cells keyed by packed (i,j,k). This header defines
//
//   * the (i,j,k) <-> packed-uint64 cell-key codec (ONE definition shared by the
//     cooker, the host test mirror, and the device sampler),
//   * SparseSdfDevice  -- a flat, raw-pointer view that p08 (Newton-summed-SDF
//     contact) consumes on the GPU, and
//   * sparse_sdf_sample(...) -- the trilinear-interp sampler.
//
// HOST + DEVICE: this is a plain header compiled by BOTH g++ (cook-time tests
// sample the cooked CPU blob through it) and nvcc (p08 device kernels). It is
// therefore CUDA-FREE: it uses nuka::math::Vec3 (already __host__ __device__)
// rather than CUDA's float3, and carries NO `#error compile-with-nvcc` guard.
// p08's Newton loop transforms world points with math::Transform (which yields
// Vec3), so a Vec3-typed sampler is the natural — and CUDA-header-free —
// interface for the p07->p08 handoff.
//
// DETERMINISM: the sampler performs a binary search over an ASCENDING-sorted
// cell-key array and a fixed 8-corner trilinear blend in a fixed accumulation
// order. It contains no atomics (clean under physics_path float_atomic_add lint)
// and no thread-order-dependent reduction, so two runs are bit-identical.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_SDF_HD __host__ __device__
#else
#define NUKA_SDF_HD
#endif

namespace nuka::runtime::sdf {

// Number of bits per axis in a packed cell key. 21 bits -> [0, 2^21) per axis,
// which covers a 2M-voxel span on each axis (far beyond any cooked band) while
// fitting three axes in 63 bits. The origin is placed at the band's lower
// corner so every band index is >= 0 (unsigned), making the packed key
// monotonic in (i, then j, then k) — required for the binary search below.
inline constexpr uint32_t kSdfKeyBitsPerAxis = 21u;
inline constexpr uint32_t kSdfKeyAxisMask    = (1u << kSdfKeyBitsPerAxis) - 1u;

// Pack a non-negative voxel index (i,j,k) into a sortable 64-bit key. The pack
// is monotonic: ordering keys ascending orders cells by i, then j, then k.
NUKA_SDF_HD inline uint64_t PackSdfCellKey(uint32_t i, uint32_t j, uint32_t k) {
    return (static_cast<uint64_t>(i) << (2u * kSdfKeyBitsPerAxis)) |
           (static_cast<uint64_t>(j) << (1u * kSdfKeyBitsPerAxis)) |
           (static_cast<uint64_t>(k));
}

NUKA_SDF_HD inline void UnpackSdfCellKey(uint64_t key, uint32_t& i, uint32_t& j, uint32_t& k) {
    i = static_cast<uint32_t>((key >> (2u * kSdfKeyBitsPerAxis)) & kSdfKeyAxisMask);
    j = static_cast<uint32_t>((key >> (1u * kSdfKeyBitsPerAxis)) & kSdfKeyAxisMask);
    k = static_cast<uint32_t>(key & kSdfKeyAxisMask);
}

// Flat, raw-pointer view of one cooked SDF. On the host this points into the
// CookedSdfTable vectors (.data()); on the device into uploaded buffers. Same
// struct, same sampler. cell_keys is ASCENDING-sorted; cell_values[n] /
// cell_gradients[n] correspond to cell_keys[n].
struct SparseSdfDevice {
    math::Vec3      origin{};            // world/local-space corner of voxel (0,0,0)
    float           voxel_size = 0.0f;   // edge length of one voxel
    uint32_t        dims[3]    = {0, 0, 0};
    const uint64_t* cell_keys     = nullptr;  // sorted; PackSdfCellKey(i,j,k)
    const float*    cell_values   = nullptr;  // signed distance at cell
    const math::Vec3* cell_gradients = nullptr; // precomputed grad (may be null)
    uint32_t        cell_count    = 0;

    // Distance returned when a query lacks band coverage. Large + positive
    // signals "far outside this collider"; p08 treats it as no-contact.
    static constexpr float kOutsideBand = 1.0e30f;
};

// Binary search for `key` in the ascending cell_keys array. Returns the index,
// or cell_count if absent. Deterministic (pure comparison search).
NUKA_SDF_HD inline uint32_t FindSdfCell(const SparseSdfDevice& sdf, uint64_t key) {
    uint32_t lo = 0;
    uint32_t hi = sdf.cell_count;
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo) >> 1u);
        const uint64_t mk = sdf.cell_keys[mid];
        if (mk < key) {
            lo = mid + 1u;
        } else if (mk > key) {
            hi = mid;
        } else {
            return mid;
        }
    }
    return sdf.cell_count;
}

// Floor division helper that returns the integer grid coordinate of point p
// along one axis (p already shifted by origin and divided by voxel_size).
NUKA_SDF_HD inline int32_t SdfFloorInt(float v) {
    const int32_t iv = static_cast<int32_t>(v);
    return (v < static_cast<float>(iv)) ? (iv - 1) : iv;
}

// Sample the SDF (+ gradient) at point p (in the SDF's own local frame). Returns
// the trilinearly-interpolated signed distance. If any of the 8 surrounding band
// corners is missing (query is outside narrow-band coverage), returns
// kOutsideBand and leaves out_grad at the gradient of whatever corners exist
// (zeroed if none) — p08 reads the large distance, not the grad, in that case.
//
// Fixed-order trilinear blend: corner accumulation order is the canonical
// (di,dj,dk) lexicographic loop, weights are the standard products — bit-stable.
NUKA_SDF_HD inline float sparse_sdf_sample(const SparseSdfDevice& sdf,
                                           math::Vec3 p,
                                           math::Vec3& out_grad) {
    out_grad = math::Vec3{0.0f, 0.0f, 0.0f};
    if (sdf.cell_count == 0u || sdf.voxel_size <= 0.0f) {
        return SparseSdfDevice::kOutsideBand;
    }

    const float inv_h = 1.0f / sdf.voxel_size;
    const float gx = (p.x - sdf.origin.x) * inv_h;
    const float gy = (p.y - sdf.origin.y) * inv_h;
    const float gz = (p.z - sdf.origin.z) * inv_h;

    const int32_t i0 = SdfFloorInt(gx);
    const int32_t j0 = SdfFloorInt(gy);
    const int32_t k0 = SdfFloorInt(gz);

    const float fx = gx - static_cast<float>(i0);
    const float fy = gy - static_cast<float>(j0);
    const float fz = gz - static_cast<float>(k0);

    // Gather the 8 corner values + gradients. Bail to kOutsideBand if any corner
    // is not in the band (so we never blend a band value with a fabricated one).
    float    val[2][2][2];
    math::Vec3 grad[2][2][2];
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                const int32_t ci = i0 + di;
                const int32_t cj = j0 + dj;
                const int32_t ck = k0 + dk;
                if (ci < 0 || cj < 0 || ck < 0) {
                    return SparseSdfDevice::kOutsideBand;
                }
                const uint64_t key = PackSdfCellKey(static_cast<uint32_t>(ci),
                                                    static_cast<uint32_t>(cj),
                                                    static_cast<uint32_t>(ck));
                const uint32_t idx = FindSdfCell(sdf, key);
                if (idx >= sdf.cell_count) {
                    return SparseSdfDevice::kOutsideBand;
                }
                val[di][dj][dk] = sdf.cell_values[idx];
                grad[di][dj][dk] = (sdf.cell_gradients != nullptr)
                                       ? sdf.cell_gradients[idx]
                                       : math::Vec3{0.0f, 0.0f, 0.0f};
            }
        }
    }

    // Trilinear weights, fixed accumulation order.
    const float wx[2] = {1.0f - fx, fx};
    const float wy[2] = {1.0f - fy, fy};
    const float wz[2] = {1.0f - fz, fz};

    float    phi = 0.0f;
    math::Vec3 g{0.0f, 0.0f, 0.0f};
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                const float w = wx[di] * wy[dj] * wz[dk];
                phi += w * val[di][dj][dk];
                g.x += w * grad[di][dj][dk].x;
                g.y += w * grad[di][dj][dk].y;
                g.z += w * grad[di][dj][dk].z;
            }
        }
    }
    out_grad = g;
    return phi;
}

} // namespace nuka::runtime::sdf
