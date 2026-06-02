// ---------------------------------------------------------------------------
// nuka::runtime::soft -- cook-time cloth XPBD topology builder (v0.7 p09-B)
// ---------------------------------------------------------------------------
// HOST cook-time code (no device physics): builds the stretch + bend constraint
// list once from a triangle mesh. The runtime iterates it on the GPU.
// ---------------------------------------------------------------------------

#include "runtime/soft/cloth_topology.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace nuka::runtime::soft {

namespace {

// Half-edge key for an undirected edge (sorted vertex pair).
using EdgeKey = std::pair<uint32_t, uint32_t>;

EdgeKey MakeEdgeKey(uint32_t a, uint32_t b) {
    return (a < b) ? EdgeKey{a, b} : EdgeKey{b, a};
}

} // namespace

bool ComputeIsometricBendStencil(const math::Vec3& shared_a,
                                 const math::Vec3& shared_b,
                                 const math::Vec3& apex0,
                                 const math::Vec3& apex1,
                                 float k_out[4]) {
    // The stencil k is the unique (up to scale) affine null-vector of the flat
    // flap: the null vector of the 3x4 matrix M whose columns are (1, u_j, v_j),
    // where (u_j, v_j) are the flap vertices' coordinates in an in-plane 2D basis.
    // Its components are the signed 3x3 minors:  k_j = (-1)^j det(M without col j)
    // (the generalized-cross-product / cofactor identity gives sum_j k_j m_j = 0,
    // i.e. sum_j k_j = 0 AND sum_j k_j (u_j,v_j) = 0). Mapping back to 3D, since
    // (u_j,v_j) are an isometric (rigid) image of the planar 3D points, the same k
    // also satisfies sum_j k_j x_j = 0 in 3D.
    const std::array<math::Vec3, 4> verts = {shared_a, shared_b, apex0, apex1};

    // Build an orthonormal in-plane basis from the flap normal.
    const math::Vec3 edge = shared_b - shared_a;
    const math::Vec3 diag = apex0 - shared_a;
    const math::Vec3 normal = edge.Cross(diag);
    const float normal_len = normal.Length();
    if (normal_len <= 0.0f) {
        return false;  // collinear shared-edge/apex0: degenerate flap.
    }
    math::Vec3 e1 = edge;
    const float e1_len = e1.Length();
    if (e1_len <= 0.0f) {
        return false;  // coincident shared-edge endpoints.
    }
    e1 = e1 / e1_len;
    const math::Vec3 e2 = (normal / normal_len).Cross(e1);  // unit, in-plane

    // Project the four flap vertices to 2D (u, v) relative to shared_a.
    std::array<float, 4> u{};
    std::array<float, 4> v{};
    for (int j = 0; j < 4; ++j) {
        const math::Vec3 d = verts[static_cast<std::size_t>(j)] - shared_a;
        u[static_cast<std::size_t>(j)] = d.Dot(e1);
        v[static_cast<std::size_t>(j)] = d.Dot(e2);
    }

    // 3x3 minor det of columns (1,u,v) with column `skip` removed.
    auto minor = [&](int skip) -> float {
        int c[3];
        int n = 0;
        for (int j = 0; j < 4; ++j) {
            if (j != skip) {
                c[n++] = j;
            }
        }
        const auto col = [&](int idx, float& a0, float& a1, float& a2) {
            a0 = 1.0f;
            a1 = u[static_cast<std::size_t>(c[idx])];
            a2 = v[static_cast<std::size_t>(c[idx])];
        };
        float a0, a1, a2, b0, b1, b2, d0, d1, d2;
        col(0, a0, a1, a2);
        col(1, b0, b1, b2);
        col(2, d0, d1, d2);
        // det of 3x3 with rows [a0 b0 d0; a1 b1 d1; a2 b2 d2] (columns are the
        // three kept flap vertices' (1,u,v)).
        return a0 * (b1 * d2 - b2 * d1) - b0 * (a1 * d2 - a2 * d1) +
               d0 * (a1 * b2 - a2 * b1);
    };

    float k[4];
    float max_abs = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const float sign = (j % 2 == 0) ? 1.0f : -1.0f;
        k[j] = sign * minor(j);
        max_abs = std::max(max_abs, std::fabs(k[j]));
    }
    if (max_abs <= 0.0f) {
        return false;  // fully degenerate flap (no 1-D null space).
    }
    const float inv = 1.0f / max_abs;
    for (int j = 0; j < 4; ++j) {
        k_out[j] = k[j] * inv;  // normalized so max|k| == 1 (scale folds to alpha).
    }
    return true;
}

void BuildClothConstraints(const std::vector<math::Vec3>& rest_positions,
                           const std::vector<ClothTriangle>& triangles,
                           const ClothTopologyOptions& options,
                           XpbdConstraintSet& out) {
    // Pass 1: stretch constraints over de-duplicated edges + collect, for each
    // undirected edge, the OPPOSITE (apex) vertices of every incident triangle.
    // A deterministic ordered map keeps the emission order reproducible (D1).
    std::map<EdgeKey, std::vector<uint32_t>> edge_to_apex;
    for (const ClothTriangle& tri : triangles) {
        for (int e = 0; e < 3; ++e) {
            const uint32_t a = tri.v[e];
            const uint32_t b = tri.v[(e + 1) % 3];
            const uint32_t apex = tri.v[(e + 2) % 3];
            edge_to_apex[MakeEdgeKey(a, b)].push_back(apex);
        }
    }

    if (options.emit_distance_constraints) {
        for (const auto& entry : edge_to_apex) {
            XpbdDistanceConstraint dc;
            dc.particle_a = entry.first.first;
            dc.particle_b = entry.first.second;
            dc.rest_length =
                (rest_positions[dc.particle_a] - rest_positions[dc.particle_b])
                    .Length();
            dc.compliance_alpha = options.distance_compliance_alpha;
            out.distance.push_back(dc);
        }
    }

    // Pass 2: one bend constraint per INTERIOR edge (exactly two incident
    // triangles -> two apex vertices). The flap is {shared_a, shared_b, apex0,
    // apex1}; the stored gradients are K_i = k_i * n_hat_rest.
    if (options.emit_bend_constraints) {
        for (const auto& entry : edge_to_apex) {
            if (entry.second.size() != 2u) {
                continue;  // boundary edge (1 tri) or non-manifold (>2): no bend.
            }
            const uint32_t sa = entry.first.first;
            const uint32_t sb = entry.first.second;
            const uint32_t a0 = entry.second[0];
            const uint32_t a1 = entry.second[1];

            float k[4];
            if (!ComputeIsometricBendStencil(rest_positions[sa], rest_positions[sb],
                                             rest_positions[a0], rest_positions[a1],
                                             k)) {
                continue;  // degenerate flap: skip the bend constraint.
            }

            // Unit rest flap normal (well-defined: ComputeIsometricBendStencil
            // already rejected the zero-area case). C = sum_i K_i . p_i with
            // K_i = k_i * n_hat is == 0 at the flat rest state.
            const math::Vec3 edge = rest_positions[sb] - rest_positions[sa];
            const math::Vec3 diag = rest_positions[a0] - rest_positions[sa];
            const math::Vec3 normal = edge.Cross(diag);
            const math::Vec3 n_hat = normal / normal.Length();

            XpbdBendConstraint bc;
            const uint32_t ids[4] = {sa, sb, a0, a1};
            for (int j = 0; j < 4; ++j) {
                bc.particle[j] = ids[j];
                bc.k[j] = n_hat * k[j];
            }
            bc.compliance_alpha = options.bend_compliance_alpha;
            out.bend.push_back(bc);
        }
    }
}

} // namespace nuka::runtime::soft
