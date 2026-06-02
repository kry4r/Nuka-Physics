#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::soft -- COOK-TIME triangle-mesh (cloth) XPBD topology builder
// ---------------------------------------------------------------------------
//
// v0.7 p09-B. Given a triangle mesh (3 vertex indices per face) and the rest
// positions, builds the XPBD constraint list a cloth patch needs:
//   - one DISTANCE (stretch) constraint per UNIQUE edge, rest length = the rest
//     edge length;
//   - one BEND constraint per INTERIOR edge (an edge shared by exactly two
//     triangles), using the Bergou et al. 2006 ISOMETRIC bending stencil.
//
// Bergou stencil construction (flat-rest, the configuration a cloth patch starts
// from): the flap is the 4 vertices {shared-edge a, shared-edge b, apex of tri 1,
// apex of tri 2}. The isometric scalar stencil k = [k0 k1 k2 k3] is the LINEAR-
// PRECISION weight set with sum_i k_i = 0 and sum_i k_i * x_i_rest = 0. For a
// non-degenerate flat (coplanar) flap, those 3 conditions on 4 unknowns PIN k up
// to scale, so we compute k DIRECTLY as the unique affine null-vector of the rest
// flap (== the cotangent isometric stencil up to a scalar -- the scalar folds
// into compliance_alpha and the sign is irrelevant since the projection drives
// C->0 either way). The stored per-particle gradient is the CONSTANT vector
//     K_i = k_i * n_hat_rest        (n_hat_rest = unit rest flap normal)
// so the row's scalar constraint C = sum_i K_i . p_i is == 0 at flat rest and has
// grad_{p_i} C = K_i (constant, no per-step recompute). Curved-rest cotangent
// weights for general developable rest shapes are deferred to p09-C/p15.
//
// Cook-time HOST builder (the per-step SIM stays GPU-only). Deterministic: edges
// and bend flaps are emitted in a fixed sorted order.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "runtime/soft/xpbd_world.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::soft {

// One triangle: three particle indices (CCW winding not required).
struct ClothTriangle {
    uint32_t v[3] = {0u, 0u, 0u};
};

struct ClothTopologyOptions {
    float distance_compliance_alpha = 0.0f;  // 0 == inextensible (rigid stretch).
    float bend_compliance_alpha = 0.0f;      // 0 == rigid bend; larger == floppier.
    bool emit_distance_constraints = true;
    bool emit_bend_constraints = true;
};

// Append stretch (distance) + bend constraints for `triangles` (rest geometry =
// `rest_positions`) onto `out`. A degenerate flap (collinear / coincident flap
// vertices, zero-area triangle) is SKIPPED for the bend constraint (its stencil
// is undefined); the distance constraints are still emitted.
void BuildClothConstraints(const std::vector<math::Vec3>& rest_positions,
                           const std::vector<ClothTriangle>& triangles,
                           const ClothTopologyOptions& options,
                           XpbdConstraintSet& out);

// Compute the Bergou isometric bend stencil for a flat flap. `shared_a`,
// `shared_b` are the two shared-edge vertices; `apex0`, `apex1` are the two
// opposite vertices. Returns the four scalar weights k[0..3] (ordered
// {shared_a, shared_b, apex0, apex1}) -- the unique affine null-vector of the
// flap, normalized so the largest |k| == 1. Returns false (k untouched) when the
// flap is degenerate (rank-deficient beyond the expected 1-D null space, i.e. the
// flap collapses to a line/point). Exposed for the cook unit tests.
bool ComputeIsometricBendStencil(const math::Vec3& shared_a,
                                 const math::Vec3& shared_b,
                                 const math::Vec3& apex0,
                                 const math::Vec3& apex1,
                                 float k_out[4]);

} // namespace nuka::runtime::soft
