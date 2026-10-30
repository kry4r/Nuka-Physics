#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "math/symmetric_mat3.hpp"
#include "math/finite.hpp"
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
        return {static_cast<float>(double(column[0].x)*direction.x + double(column[0].y)*direction.y + double(column[0].z)*direction.z),
                static_cast<float>(double(column[1].x)*direction.x + double(column[1].y)*direction.y + double(column[1].z)*direction.z),
                static_cast<float>(double(column[2].x)*direction.x + double(column[2].y)*direction.y + double(column[2].z)*direction.z)};
    }
    NUKA_ENDPOINT_HD math::Vec3 Multiply(math::Vec3 velocity) const {
        return {static_cast<float>(double(column[0].x)*velocity.x + double(column[1].x)*velocity.y + double(column[2].x)*velocity.z),
                static_cast<float>(double(column[0].y)*velocity.x + double(column[1].y)*velocity.y + double(column[2].y)*velocity.z),
                static_cast<float>(double(column[0].z)*velocity.x + double(column[1].z)*velocity.y + double(column[2].z)*velocity.z)};
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

// Per env: material points, then surface contacts, then stress cells.
inline constexpr uint64_t MpmPointEndpointCount(uint32_t particles, uint32_t surface_contacts,
                                                uint32_t stress_cells) {
    return uint64_t{particles} + surface_contacts + stress_cells;
}

inline constexpr uint64_t MpmPointEndpointTermCount(uint32_t particles, uint32_t surface_contacts,
                                                    uint32_t stress_cells) {
    return uint64_t{particles} * kMpmStencilNodes + uint64_t{surface_contacts} * kTriangleEndpointTerms +
           uint64_t{stress_cells} * kMpmCellStencilNodes;
}

// The barycentric point follows the vertices; its offset follows their best-fit angular velocity.
// This reproduces rigid motion at the contact point and preserves impulse work and both momenta.
NUKA_ENDPOINT_HD inline bool BuildTrianglePointEndpoint(
    const uint32_t* indices, const math::Vec3* vertices, math::Vec3 barycentric,
    math::Vec3 contact_point, PointEndpointTerm* terms) {
    const double total = double{barycentric.x} + barycentric.y + barycentric.z;
    if (!math::IsFinite(total) || total == 0.0) return false;
    const double weights[3] = {barycentric.x / total, barycentric.y / total, barycentric.z / total};
    const double position[3][3] = {{vertices[0].x, vertices[0].y, vertices[0].z},
        {vertices[1].x, vertices[1].y, vertices[1].z}, {vertices[2].x, vertices[2].y, vertices[2].z}};
    const double point[3] = {contact_point.x, contact_point.y, contact_point.z};
    double relative[3][3], offset[3];
    // Local edge coordinates keep world translation out of the rotational fit.
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        const double e1 = position[1][axis] - position[0][axis];
        const double e2 = position[2][axis] - position[0][axis];
        relative[0][axis] = -(e1 + e2) / 3.0;
        relative[1][axis] = relative[0][axis] + e1;
        relative[2][axis] = relative[0][axis] + e2;
        offset[axis] = (point[axis] - position[0][axis]) - weights[1] * e1 - weights[2] * e2;
    }
    double xx = 0.0, yy = 0.0, zz = 0.0, xy = 0.0, xz = 0.0, yz = 0.0;
    for (uint32_t i = 0u; i < kTriangleEndpointTerms; ++i) {
        const double x = relative[i][0], y = relative[i][1], z = relative[i][2];
        xx += y * y + z * z; yy += x * x + z * z; zz += x * x + y * y;
        xy -= x * y; xz -= x * z; yz -= y * z;
    }
    const double scale = fmax(xx, fmax(yy, zz));
    if (!(scale > 0.0) || !(scale <= FLT_MAX)) return false;
    xx /= scale; yy /= scale; zz /= scale; xy /= scale; xz /= scale; yz /= scale;
    const double cxx = yy * zz - yz * yz, cyy = xx * zz - xz * xz, czz = xx * yy - xy * xy;
    const double cxy = xz * yz - xy * zz, cxz = xy * yz - xz * yy, cyz = xy * xz - xx * yz;
    const double determinant = xx * cxx + xy * cxy + xz * cxz;
    if (!(determinant > 1.0e-12)) return false;
    const double inverse_scale = 1.0 / (determinant * scale);
    const double inverse[3][3] = {{cxx * inverse_scale, cxy * inverse_scale, cxz * inverse_scale},
        {cxy * inverse_scale, cyy * inverse_scale, cyz * inverse_scale},
        {cxz * inverse_scale, cyz * inverse_scale, czz * inverse_scale}};
    float columns[kTriangleEndpointTerms][3][3];
    for (uint32_t i = 0u; i < kTriangleEndpointTerms; ++i) {
        terms[i].kind = kNkSideParticle;
        terms[i].index = indices[i];
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            double cross[3], omega[3]{};
            for (uint32_t c = 0u; c < 3u; ++c) {
                const uint32_t j = (c + 1u) % 3u, k = (c + 2u) % 3u;
                cross[c] = relative[i][j] * double(k == axis) - relative[i][k] * double(j == axis);
            }
            for (uint32_t c = 0u; c < 3u; ++c)
                for (uint32_t j = 0u; j < 3u; ++j) omega[c] += inverse[c][j] * cross[j];
            for (uint32_t c = 0u; c < 3u; ++c) {
                const uint32_t j = (c + 1u) % 3u, k = (c + 2u) % 3u;
                const double value = double(c == axis) * weights[i] + omega[j] * offset[k] - omega[k] * offset[j];
                if (!(value >= -FLT_MAX && value <= FLT_MAX)) return false;
                columns[i][axis][c] = static_cast<float>(value);
            }
        }
    }
    // Preserve the unit partition when coefficients round to single precision.
    // The smallest coefficient absorbs the residual with the finest available spacing.
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        for (uint32_t c = 0u; c < 3u; ++c) {
            uint32_t pivot = 0u;
            for (uint32_t i = 1u; i < kTriangleEndpointTerms; ++i)
                if (fabs(columns[i][axis][c]) < fabs(columns[pivot][axis][c])) pivot = i;
            double residual = double(c == axis);
            for (uint32_t i = 0u; i < kTriangleEndpointTerms; ++i)
                if (i != pivot) residual -= columns[i][axis][c];
            if (!(residual >= -FLT_MAX && residual <= FLT_MAX)) return false;
            columns[pivot][axis][c] = static_cast<float>(residual);
        }
        for (uint32_t i = 0u; i < kTriangleEndpointTerms; ++i)
            terms[i].column[axis] = {columns[i][axis][0], columns[i][axis][1], columns[i][axis][2]};
    }
    return true;
}

}  // namespace nuka::nk

#undef NUKA_ENDPOINT_HD
