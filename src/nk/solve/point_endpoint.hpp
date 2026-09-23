#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "math/symmetric_mat3.hpp"
#include "nk/material/mpm_material.hpp"
#include "nk/solve/nk_row.hpp"

#if defined(__CUDACC__)
#define NUKA_ENDPOINT_HD __host__ __device__
#else
#define NUKA_ENDPOINT_HD
#endif

namespace nuka::nk {

inline constexpr uint32_t kTriangleEndpointTerms = 3u;

// Terms are sorted by (kind, index), with duplicate mass degrees of freedom combined.
struct PointEndpointRange {
    uint32_t first = 0u;
    uint32_t count = 0u;
};

// Columns map a point-mass velocity to the velocity of an interpolated endpoint.
struct PointEndpointTerm {
    uint32_t kind = kNkSideParticle;
    uint32_t index = ~0u;
    math::Vec3 column[3]{};

    NUKA_ENDPOINT_HD uint64_t Key() const {
        return (uint64_t{kind} << 32u) | index;
    }

    NUKA_ENDPOINT_HD math::Vec3 TransposeMultiply(math::Vec3 direction) const {
        return {column[0].Dot(direction), column[1].Dot(direction), column[2].Dot(direction)};
    }
    NUKA_ENDPOINT_HD math::Vec3 Multiply(math::Vec3 velocity) const {
        return column[0] * velocity.x + column[1] * velocity.y + column[2] * velocity.z;
    }
};

static_assert(sizeof(PointEndpointRange) == 2u * sizeof(uint32_t));
static_assert(sizeof(PointEndpointTerm) == 2u * sizeof(uint32_t) + 9u * sizeof(float));

NUKA_ENDPOINT_HD inline PointEndpointTerm WeightedPointEndpointTerm(
    uint32_t kind, uint32_t index, float weight) {
    return {kind, index, {{weight, 0, 0}, {0, weight, 0}, {0, 0, weight}}};
}

NUKA_ENDPOINT_HD inline uint32_t CanonicalizePointEndpointTerms(PointEndpointTerm* terms, uint32_t count) {
    for (uint32_t i = 1u; i < count; ++i) {
        const auto term = terms[i];
        uint32_t j = i;
        while (j > 0u && terms[j - 1u].Key() > term.Key()) {
            terms[j] = terms[j - 1u];
            --j;
        }
        terms[j] = term;
    }
    uint32_t retained = 0u;
    for (uint32_t i = 0u; i < count; ++i) {
        if (retained > 0u && terms[retained - 1u].Key() == terms[i].Key()) {
            for (uint32_t axis = 0u; axis < 3u; ++axis)
                terms[retained - 1u].column[axis] += terms[i].column[axis];
        } else {
            terms[retained++] = terms[i];
        }
    }
    return retained;
}

inline constexpr uint64_t MpmPointEndpointCount(uint32_t particles, uint32_t surface_contacts) {
    return uint64_t{particles} + surface_contacts;
}

inline constexpr uint64_t MpmPointEndpointTermCount(uint32_t particles, uint32_t surface_contacts) {
    return uint64_t{particles} * kMpmStencilNodes + uint64_t{surface_contacts} * kTriangleEndpointTerms;
}

// The barycentric point follows the vertices; its offset follows their best-fit angular velocity.
// This reproduces rigid motion at the contact point and preserves impulse work and both momenta.
NUKA_ENDPOINT_HD inline bool BuildTrianglePointEndpoint(
    const uint32_t* indices, const math::Vec3* vertices, math::Vec3 barycentric,
    math::Vec3 contact_point, PointEndpointTerm* terms) {
    using math::Vec3;
    const Vec3 centroid = (vertices[0] + vertices[1] + vertices[2]) / 3.0f;
    const Vec3 attached = vertices[0] * barycentric.x + vertices[1] * barycentric.y +
                          vertices[2] * barycentric.z;
    const Vec3 offset = contact_point - attached;
    const Vec3 relative[3] = {vertices[0] - centroid, vertices[1] - centroid,
                              vertices[2] - centroid};
    math::SymmetricMat3 tensor;
    for (uint32_t i = 0u; i < kTriangleEndpointTerms; ++i) {
        const Vec3 r = relative[i];
        tensor.xx += r.y * r.y + r.z * r.z;
        tensor.yy += r.x * r.x + r.z * r.z;
        tensor.zz += r.x * r.x + r.y * r.y;
        tensor.xy -= r.x * r.y;
        tensor.xz -= r.x * r.z;
        tensor.yz -= r.y * r.z;
    }
    const double scale = fmaxf(tensor.xx, fmaxf(tensor.yy, tensor.zz));
    if (!(scale > 0.0) || !(scale <= FLT_MAX)) return false;
    const double xx = tensor.xx / scale, yy = tensor.yy / scale, zz = tensor.zz / scale;
    const double xy = tensor.xy / scale, xz = tensor.xz / scale, yz = tensor.yz / scale;
    const double cxx = yy * zz - yz * yz, cyy = xx * zz - xz * xz, czz = xx * yy - xy * xy;
    const double cxy = xz * yz - xy * zz, cxz = xy * yz - xz * yy, cyz = xy * xz - xx * yz;
    const double determinant = xx * cxx + xy * cxy + xz * cxz;
    if (!(determinant > 1.0e-12)) return false;
    const double inverse_scale = 1.0 / (determinant * scale);
    const math::SymmetricMat3 inverse{float(cxx * inverse_scale), float(cyy * inverse_scale),
        float(czz * inverse_scale), float(cxy * inverse_scale), float(cxz * inverse_scale),
        float(cyz * inverse_scale)};
    const Vec3 basis[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const float weights[3] = {barycentric.x, barycentric.y, barycentric.z};
    for (uint32_t i = 0u; i < kTriangleEndpointTerms; ++i) {
        terms[i].kind = kNkSideParticle;
        terms[i].index = indices[i];
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            const Vec3 omega = inverse.Multiply(relative[i].Cross(basis[axis]));
            terms[i].column[axis] = basis[axis] * weights[i] + omega.Cross(offset);
        }
    }
    return true;
}

}  // namespace nuka::nk

#undef NUKA_ENDPOINT_HD
