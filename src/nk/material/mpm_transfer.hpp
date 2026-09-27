#pragma once

#include <cmath>
#include <cstdint>

#include "nk/material/mpm_material.hpp"

#if defined(__CUDACC__)
#define NUKA_MPM_TRANSFER_HD __host__ __device__
#else
#define NUKA_MPM_TRANSFER_HD
#endif

namespace nuka::nk {

// Each lattice carries half of every particle's mass, momentum and force.
inline constexpr float kMpmLatticeShare = 1.0f / static_cast<float>(kMpmLattices);

// Lattice l has nodes at grid coordinate i + offset(l), a quarter cell either side of the cells.
NUKA_MPM_TRANSFER_HD inline float MpmLatticeOffset(uint32_t lattice) {
    return lattice == 0u ? -0.25f : 0.25f;
}

// The C2 compact kernel 1 - r + sin(2 pi r) / (2 pi) on r in [0, 1]; K(r) + K(1 - r) = 1.
NUKA_MPM_TRANSFER_HD inline float MpmCompactKernel(float r) {
    constexpr float kTwoPi = 6.28318530717958647692f;
#if defined(__CUDA_ARCH__)
    return 1.0f - r + sinpif(2.0f * r) / kTwoPi;
#else
    return 1.0f - r + std::sin(kTwoPi * r) / kTwoPi;
#endif
}

// Floor division that keeps negative half-cells on the lower cell.
NUKA_MPM_TRANSFER_HD inline int64_t MpmFloorHalf(int64_t value) {
    return value >= 0 ? value / 2 : -((1 - value) / 2);
}

// Half-cell h = floor(2 g + 1/2) fixes the cell on both lattices: floor(h / 2) and
// floor((h - 1) / 2). Non-finite coordinates land far outside every grid.
NUKA_MPM_TRANSFER_HD inline int64_t MpmHalfCell(float coordinate) {
    return std::isfinite(coordinate) && fabsf(coordinate) < 0x1p60f
        ? static_cast<int64_t>(floorf(2.0f * coordinate + 0.5f)) : -0x40000000;
}

NUKA_MPM_TRANSFER_HD inline int64_t MpmLatticeBase(int64_t half_cell, uint32_t lattice) {
    return MpmFloorHalf(half_cell - static_cast<int64_t>(lattice));
}

// One dual-stencil axis: per-lattice weights sum to one, offsets are node minus particle in
// cells, moment is the diagonal APIC D entry and first the mean offset forming D off-diagonal.
struct MpmCompactAxis {
    int64_t base[kMpmLattices];
    float w[kMpmLattices][kMpmStencilWidth];
    float offset[kMpmLattices][kMpmStencilWidth];
    float first[kMpmLattices];
    float moment;
};

NUKA_MPM_TRANSFER_HD inline MpmCompactAxis MpmCompactWeights(float coordinate, int64_t half_cell) {
    MpmCompactAxis axis;
    axis.moment = 0.0f;
    for (uint32_t lattice = 0u; lattice < kMpmLattices; ++lattice) {
        axis.base[lattice] = MpmLatticeBase(half_cell, lattice);
        const float local = coordinate - MpmLatticeOffset(lattice) -
                            static_cast<float>(axis.base[lattice]);
        const float f = fminf(fmaxf(local, 0.0f), 1.0f);
        axis.w[lattice][0] = MpmCompactKernel(f);
        axis.w[lattice][1] = MpmCompactKernel(1.0f - f);
        axis.offset[lattice][0] = -f;
        axis.offset[lattice][1] = 1.0f - f;
        axis.first[lattice] = axis.w[lattice][1] - f;
        axis.moment += kMpmLatticeShare *
            (axis.w[lattice][0] * f * f + axis.w[lattice][1] * (1.0f - f) * (1.0f - f));
    }
    return axis;
}

NUKA_MPM_TRANSFER_HD inline MpmCompactAxis MpmCompactWeights(float coordinate) {
    return MpmCompactWeights(coordinate, MpmHalfCell(coordinate));
}

// D^-1 of APIC in 1/length^2 as xx, yy, zz, xy, xz, yz. Weights factor per axis on each
// lattice, so D_ab = sum_l share * first_a[l] * first_b[l] off the diagonal.
NUKA_MPM_TRANSFER_HD inline bool MpmApicInverse(const MpmCompactAxis (&axes)[3], float inv_dx,
                                                float (&out)[6]) {
    float off[3] = {0.0f, 0.0f, 0.0f};
    for (uint32_t lattice = 0u; lattice < kMpmLattices; ++lattice) {
        off[0] += kMpmLatticeShare * axes[0].first[lattice] * axes[1].first[lattice];
        off[1] += kMpmLatticeShare * axes[0].first[lattice] * axes[2].first[lattice];
        off[2] += kMpmLatticeShare * axes[1].first[lattice] * axes[2].first[lattice];
    }
    const float xx = axes[0].moment, yy = axes[1].moment, zz = axes[2].moment;
    const float xy = off[0], xz = off[1], yz = off[2];
    const float cxx = yy * zz - yz * yz, cyy = xx * zz - xz * xz, czz = xx * yy - xy * xy;
    const float cxy = xz * yz - xy * zz, cxz = xy * yz - xz * yy, cyz = xy * xz - xx * yz;
    const float determinant = xx * cxx + xy * cxy + xz * cxz;
    if (!(determinant > 0.0f) || !std::isfinite(determinant)) return false;
    const float scale = inv_dx * inv_dx / determinant;
    out[0] = cxx * scale; out[1] = cyy * scale; out[2] = czz * scale;
    out[3] = cxy * scale; out[4] = cxz * scale; out[5] = cyz * scale;
    return true;
}

// Nodes of one lattice follow the dims layout; lattice 1 follows lattice 0 within the env.
// Signed coordinates keep an escaping stencil from wrapping into another environment.
NUKA_MPM_TRANSFER_HD inline int64_t MpmNodeIndex(uint32_t env, uint32_t lattice, int64_t x,
    int64_t y, int64_t z, const uint32_t dims[3], uint32_t nodes_per_env) {
    if (x < 0 || y < 0 || z < 0 || x >= int64_t{dims[0]} ||
        y >= int64_t{dims[1]} || z >= int64_t{dims[2]}) return -1;
    const uint32_t lattice_nodes = nodes_per_env / kMpmLattices;
    const uint32_t local = static_cast<uint32_t>((z * dims[1] + y) * dims[0] + x);
    return int64_t{env} * nodes_per_env + int64_t{lattice} * lattice_nodes + local;
}

}  // namespace nuka::nk

#undef NUKA_MPM_TRANSFER_HD
