#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_MPM_TRANSFER_HD __host__ __device__
#else
#define NUKA_MPM_TRANSFER_HD
#endif

namespace nuka::nk {

struct MpmQuadraticBasis {
    int64_t base;
    float w[3];
};

// Transfer and contact interpolation use the same quadratic partition of unity.
NUKA_MPM_TRANSFER_HD inline MpmQuadraticBasis MpmQuadraticWeights(float coordinate) {
    MpmQuadraticBasis basis;
    basis.base = std::isfinite(coordinate) && fabsf(coordinate) < 0x1p62f
        ? static_cast<int64_t>(floorf(coordinate - 0.5f)) : -0x40000000;
    const float offset = coordinate - static_cast<float>(basis.base);
    basis.w[0] = 0.5f * (1.5f - offset) * (1.5f - offset);
    const float centered = offset - 1.0f;
    basis.w[1] = 0.75f - centered * centered;
    basis.w[2] = 0.5f * (offset - 0.5f) * (offset - 0.5f);
    return basis;
}

// Signed coordinates keep an escaping stencil from wrapping into another environment.
NUKA_MPM_TRANSFER_HD inline int64_t MpmNodeIndex(uint32_t env, int64_t x, int64_t y,
    int64_t z, const uint32_t dims[3], uint32_t nodes_per_env) {
    if (x < 0 || y < 0 || z < 0 || x >= int64_t{dims[0]} ||
        y >= int64_t{dims[1]} || z >= int64_t{dims[2]}) return -1;
    const uint32_t local = static_cast<uint32_t>((z * dims[1] + y) * dims[0] + x);
    return int64_t{env} * nodes_per_env + local;
}

}  // namespace nuka::nk

#undef NUKA_MPM_TRANSFER_HD
