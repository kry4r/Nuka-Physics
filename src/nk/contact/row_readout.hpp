#pragma once

#include "nk/solve/nk_row.hpp"

#if defined(__CUDACC__)
#define NUKA_CONTACT_HD __host__ __device__
#else
#define NUKA_CONTACT_HD
#endif

namespace nuka::nk {

NUKA_CONTACT_HD inline uint32_t ContactPointAxes(const NkRow& row) {
    return row.flags & nk_row_flags::kPyramidal ? 4u : 3u;
}

NUKA_CONTACT_HD inline math::Vec3 ContactPointImpulse(
    const NkRow* rows, const float* lambda, uint32_t first, uint32_t stride) {
    const NkRow& row = rows[first];
    if (row.flags & nk_row_flags::kPyramidal) {
        const float a = lambda[first], b = lambda[first + stride];
        const float c = lambda[first + 2u * stride], d = lambda[first + 3u * stride];
        return {a + b + c + d, row.mu * (a - b), row.friction_secondary * (c - d)};
    }
    return {lambda[first], lambda[first + stride], lambda[first + 2u * stride]};
}

NUKA_CONTACT_HD inline math::Vec3 ContactPointNormal(
    const NkRow* rows, uint32_t first, uint32_t stride, bool side_b) {
    const auto n = side_b ? rows[first].b.jlin : rows[first].a.jlin;
    if (rows[first].flags & nk_row_flags::kPyramidal) {
        const auto other = side_b ? rows[first + stride].b.jlin : rows[first + stride].a.jlin;
        return (n + other) * 0.5f;
    }
    return n;
}

NUKA_CONTACT_HD inline math::Vec3 ContactPointWorldImpulse(
    const NkRow* rows, const float* lambda, uint32_t first, uint32_t stride) {
    if ((rows[first].flags & nk_row_flags::kActive) == 0u) return {};
    math::Vec3 impulse{};
    for (uint32_t axis = 0u; axis < ContactPointAxes(rows[first]); ++axis) {
        const uint32_t index = first + axis * stride;
        impulse += rows[index].a.jlin * lambda[index];
    }
    return impulse;
}

} // namespace nuka::nk

#undef NUKA_CONTACT_HD
