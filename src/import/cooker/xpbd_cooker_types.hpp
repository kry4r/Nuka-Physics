#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::soft -- CUDA-FREE XPBD param / constraint structs
// ---------------------------------------------------------------------------
//
// M9 T11 soft/fluid Phase 2b: the legacy XPBD soft-body stepper has been
// DELETED -- the forward sim is op-ified onto the unified nk core
// (phi/backend_cuda/ops/particles.cu, run by nk::World). The plain data structs
// the cook-time topology builders (cloth_topology / tetmesh_topology), the XPBD
// COOKER, and the re-pointed XPBD tests still need are relocated here, into a
// header with NO GPU/World dependency (no phi::Buffer, no device context, no .cu).
//
// These are PURE host PODs that describe an XPBD soft-body BEFORE upload: the
// particle set, the four constraint families (distance / bend / volume /
// shape-match), and the constraint-set aggregate. The constraint MATH and the
// per-step solve live on the nk path now; this header only carries the cook-time
// description the topology builders emit and the nk cook transcribes.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime::soft {

// Host-side description of an XPBD particle set.
struct XpbdParticleSet {
    std::vector<math::Vec3> positions;
    std::vector<math::Vec3> velocities;
    std::vector<float> inv_masses;  // 0 == pinned (kinematic) particle.
};

// Host-side description of one distance constraint between two particles.
struct XpbdDistanceConstraint {
    uint32_t particle_a = 0u;
    uint32_t particle_b = 0u;
    float rest_length = 0.0f;
    float compliance_alpha = 0.0f;  // XPBD compliance (1/stiffness); 0 == rigid.
};

// Host-side description of one BEND constraint over four particles (id 7). The
// Bergou isometric stencil is stored as four CONSTANT gradient vectors K_i =
// k_i*n_hat_rest (grad_{p_i} C), built once at cook time (cloth_topology). The
// row's scalar constraint is C = sum_i K[i] . p[particle[i]], == 0 at flat rest.
struct XpbdBendConstraint {
    uint32_t particle[4] = {0u, 0u, 0u, 0u};
    math::Vec3 k[4] = {math::Vec3{0.0f, 0.0f, 0.0f}, math::Vec3{0.0f, 0.0f, 0.0f},
                       math::Vec3{0.0f, 0.0f, 0.0f}, math::Vec3{0.0f, 0.0f, 0.0f}};
    float compliance_alpha = 0.0f;  // XPBD compliance (1/bend-stiffness).
};

// Host-side description of one VOLUME constraint over the four particles of a tet
// (id 8). rest_volume_times6 is 6*V_rest = the rest triple product, computed at
// cook time with the IDENTICAL det expression the solver uses (so C == 0 at rest
// regardless of vertex winding -- this is what avoids the sign-convention bug).
struct XpbdVolumeConstraint {
    uint32_t particle[4] = {0u, 0u, 0u, 0u};
    float rest_volume_times6 = 0.0f;  // det([p2-p1,p3-p1,p4-p1]) at rest.
    float compliance_alpha = 0.0f;    // XPBD compliance (1/volume-stiffness).
};

// Host-side description of one SHAPE-MATCH cluster over a VARIABLE number of
// particles (id 9; Mueller et al. 2005 meshless shape matching). The cluster is
// pulled toward the best rigid transform of its REST shape. rest_positions holds
// the cluster's rest coordinates x_i^0 (one per cluster particle, same order as
// `particle`); rest data (rest centroid c0, the per-particle q_i = x_i^0 - c0,
// the mass-weighted total) is cooked once at upload from rest_positions +
// cluster_mass. cluster_mass[i] is the shape-match WEIGHT m_i used in the
// centroid / covariance sums (defaults to the particle's mass = 1/inv_mass when
// finite; a pinned particle -- inv_mass 0 -- still contributes to the cluster
// shape with a chosen weight). stiffness in [0,1] is the per-step fraction of the
// goal correction applied (XPBD shape-match stiffness; from compliance_alpha).
struct XpbdShapeMatchCluster {
    std::vector<uint32_t> particle;          // cluster particle indices (size n>=1)
    std::vector<math::Vec3> rest_positions;  // x_i^0, size n (same order)
    std::vector<float> cluster_mass;         // m_i shape-match weight, size n (>0)
    float stiffness = 1.0f;                  // goal-pull fraction in [0,1].
};

struct XpbdConstraintSet {
    std::vector<XpbdDistanceConstraint> distance;
    std::vector<XpbdBendConstraint> bend;
    std::vector<XpbdVolumeConstraint> volume;
    std::vector<XpbdShapeMatchCluster> shape_match;
};

} // namespace nuka::runtime::soft
