#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::soft::XpbdWorld -- GPU-resident XPBD soft-body world (p09-A/B)
// ---------------------------------------------------------------------------
//
// v0.7 p09-A: the FOUNDATIONAL vertical slice of XPBD (Extended Position-Based
// Dynamics, Macklin et al. 2016) soft-body support. p09-B adds the BEND (id 7)
// and VOLUME (id 8) constraint families on top. This module owns the GPU particle
// state (positions / prev-positions / velocities / inverse masses) and a set of
// DISTANCE + BEND + VOLUME constraints, and runs the XPBD integration loop
// ENTIRELY on the GPU (master plan SS5.6 GPU-only hard constraint -- no CPU
// physics in the production path):
//
//     predict :  prev = p ;  p += v*dt + gravity*dt^2                  (per particle)
//     solve   :  iterate distance, then bend, then volume constraints  (per step)
//                (each a single-thread FIXED-ORDER Gauss-Seidel sweep)
//     correct :  v = (p - prev) / dt                                   (per particle)
//
// All three constraint families share the XPBD generalized projection (one
// constraint over its particles i with inverse masses w_i and per-particle
// gradient g_i = grad_{p_i} C):
//     a~  = compliance_alpha / dt^2
//     dl  = (-C - a~*lambda) / (sum_i w_i*|g_i|^2 + a~)
//     p_i += w_i*g_i*dl ;  lambda += dl
// Distance is the |g|=1 special case (g_a = n, g_b = -n, denom = w_a+w_b+a~).
// Bend (Bergou isometric, 4 particles): g_i = K_i, a CONSTANT vector per
//   constraint (K_i = k_i*n_hat_rest), C = sum_i K_i . p_i, == 0 at flat rest.
// Volume (tet, 4 particles): C = det([p2-p1,p3-p1,p4-p1]) - 6*V_rest, with
//   g_2 = (p3-p1)x(p4-p1), g_3 = (p4-p1)x(p2-p1), g_4 = (p2-p1)x(p3-p1),
//   g_1 = -(g_2+g_3+g_4) (each a cross product; sum_i g_i = 0 conserves momentum).
//
// D1 STRONG DETERMINISM: every device buffer is allocated ONCE at upload (no
// per-step cudaMalloc); the predict/correct kernels write distinct per-particle
// outputs (no atomics); each constraint solve is a single-thread FIXED-ORDER
// Gauss-Seidel sweep (constraints visited in index order, no float atomics, no
// thread-order dependence). Two runs of Step() with identical inputs produce
// byte-identical position/velocity buffers.
//
// Parallel / colored Gauss-Seidel (for many-constraint scaling) is DEFERRED
// (task #25; named downstream consumer = the perf budget / p10). The serial sweep
// is D1-correct and sufficient for the small cloth/tet scenes the tests exercise.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"

#include <cstddef>
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
    std::vector<uint32_t> particle;       // cluster particle indices (size n>=1)
    std::vector<math::Vec3> rest_positions;  // x_i^0, size n (same order)
    std::vector<float> cluster_mass;      // m_i shape-match weight, size n (>0)
    float stiffness = 1.0f;               // goal-pull fraction in [0,1].
};

struct XpbdConstraintSet {
    std::vector<XpbdDistanceConstraint> distance;
    std::vector<XpbdBendConstraint> bend;
    std::vector<XpbdVolumeConstraint> volume;
    std::vector<XpbdShapeMatchCluster> shape_match;
};

// Host snapshot downloaded from the device (state read-back for tests / oracles).
struct XpbdWorldState {
    std::vector<math::Vec3> positions;
    std::vector<math::Vec3> velocities;
};

struct XpbdStepOptions {
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    float dt = 1.0f / 60.0f;
    uint32_t step_count = 1u;
    // XPBD solver iterations per step. A single constraint reaches its within-step
    // fixed point in ONE iteration; multiple iterations matter for the COUPLED
    // constraint systems (cloth/tet) the bend/volume rows live in. >= 1.
    uint32_t solver_iterations = 1u;
};

struct XpbdStepReport {
    uint32_t particle_count = 0u;
    uint32_t distance_constraint_count = 0u;
    uint32_t bend_constraint_count = 0u;
    uint32_t volume_constraint_count = 0u;
    uint32_t shape_match_cluster_count = 0u;
    uint32_t simulated_step_count = 0u;
    uint32_t kernel_launch_count = 0u;
};

// GPU-resident XPBD world. All device buffers are allocated once on upload; Step
// launches only pre-existing kernels over them (no hot-path allocation).
class XpbdWorld {
public:
    XpbdWorld() = default;
    XpbdWorld(uint32_t particle_count,
              uint32_t distance_constraint_count,
              uint32_t bend_constraint_count,
              uint32_t volume_constraint_count,
              uint32_t shape_match_cluster_count,
              phi::Buffer positions,
              phi::Buffer prev_positions,
              phi::Buffer velocities,
              phi::Buffer inv_masses,
              phi::Buffer distance_particle_a,
              phi::Buffer distance_particle_b,
              phi::Buffer distance_rest_length,
              phi::Buffer distance_compliance_alpha,
              phi::Buffer distance_lambda,
              phi::Buffer bend_particles,
              phi::Buffer bend_gradients,
              phi::Buffer bend_compliance_alpha,
              phi::Buffer bend_lambda,
              phi::Buffer volume_particles,
              phi::Buffer volume_rest_times6,
              phi::Buffer volume_compliance_alpha,
              phi::Buffer volume_lambda,
              phi::Buffer shape_match_cluster_offset,
              phi::Buffer shape_match_cluster_count_buf,
              phi::Buffer shape_match_stiffness,
              phi::Buffer shape_match_rest_centroid,
              phi::Buffer shape_match_particles,
              phi::Buffer shape_match_rest_q,
              phi::Buffer shape_match_mass);

    XpbdWorld(const XpbdWorld&) = delete;
    XpbdWorld& operator=(const XpbdWorld&) = delete;
    XpbdWorld(XpbdWorld&&) noexcept = default;
    XpbdWorld& operator=(XpbdWorld&&) noexcept = default;

    uint32_t ParticleCount() const { return particle_count_; }
    uint32_t DistanceConstraintCount() const { return distance_constraint_count_; }
    uint32_t BendConstraintCount() const { return bend_constraint_count_; }
    uint32_t VolumeConstraintCount() const { return volume_constraint_count_; }
    uint32_t ShapeMatchClusterCount() const { return shape_match_cluster_count_; }
    std::size_t DeviceBytes() const;
    bool HasUploadedState() const;
    XpbdWorldState DownloadState() const;

    math::Vec3* DevicePositions();
    const math::Vec3* DevicePositions() const;
    math::Vec3* DevicePrevPositions();
    math::Vec3* DeviceVelocities();
    const math::Vec3* DeviceVelocities() const;
    const float* DeviceInvMasses() const;

    const uint32_t* DeviceDistanceParticleA() const;
    const uint32_t* DeviceDistanceParticleB() const;
    const float* DeviceDistanceRestLength() const;
    const float* DeviceDistanceComplianceAlpha() const;
    float* DeviceDistanceLambda();

    // Bend: particles are 4-per-constraint (uint32_t[4]); gradients are the four
    // constant K_i vectors (math::Vec3[4]) per constraint, flattened row-major.
    const uint32_t* DeviceBendParticles() const;
    const math::Vec3* DeviceBendGradients() const;
    const float* DeviceBendComplianceAlpha() const;
    float* DeviceBendLambda();

    // Volume: 4-per-constraint particle indices + per-constraint rest triple
    // product (6*V_rest) + compliance.
    const uint32_t* DeviceVolumeParticles() const;
    const float* DeviceVolumeRestTimes6() const;
    const float* DeviceVolumeComplianceAlpha() const;
    float* DeviceVolumeLambda();

    // Shape-match: per-cluster offset/count into the flat particle/q/mass arrays,
    // per-cluster stiffness + rest centroid c0, and the flattened cluster
    // particle indices, rest offsets q_i = x_i^0 - c0, and weights m_i.
    const uint32_t* DeviceShapeMatchClusterOffset() const;
    const uint32_t* DeviceShapeMatchClusterCount() const;
    const float* DeviceShapeMatchStiffness() const;
    const math::Vec3* DeviceShapeMatchRestCentroid() const;
    const uint32_t* DeviceShapeMatchParticles() const;
    const math::Vec3* DeviceShapeMatchRestQ() const;
    const float* DeviceShapeMatchMass() const;

private:
    uint32_t particle_count_ = 0u;
    uint32_t distance_constraint_count_ = 0u;
    uint32_t bend_constraint_count_ = 0u;
    uint32_t volume_constraint_count_ = 0u;
    uint32_t shape_match_cluster_count_ = 0u;
    phi::Buffer positions_;
    phi::Buffer prev_positions_;
    phi::Buffer velocities_;
    phi::Buffer inv_masses_;
    phi::Buffer distance_particle_a_;
    phi::Buffer distance_particle_b_;
    phi::Buffer distance_rest_length_;
    phi::Buffer distance_compliance_alpha_;
    phi::Buffer distance_lambda_;
    phi::Buffer bend_particles_;        // 4 * bend_count uint32_t
    phi::Buffer bend_gradients_;        // 4 * bend_count math::Vec3
    phi::Buffer bend_compliance_alpha_;
    phi::Buffer bend_lambda_;
    phi::Buffer volume_particles_;      // 4 * volume_count uint32_t
    phi::Buffer volume_rest_times6_;
    phi::Buffer volume_compliance_alpha_;
    phi::Buffer volume_lambda_;
    phi::Buffer shape_match_cluster_offset_;   // cluster_count uint32_t
    phi::Buffer shape_match_cluster_count_buf_; // cluster_count uint32_t (n_c)
    phi::Buffer shape_match_stiffness_;        // cluster_count float
    phi::Buffer shape_match_rest_centroid_;    // cluster_count math::Vec3 (c0)
    phi::Buffer shape_match_particles_;        // sum(n_c) uint32_t
    phi::Buffer shape_match_rest_q_;           // sum(n_c) math::Vec3 (x_i^0 - c0)
    phi::Buffer shape_match_mass_;             // sum(n_c) float (m_i)
};

XpbdWorld UploadXpbdWorld(const phi::DeviceContext& context,
                          const XpbdParticleSet& particles,
                          const XpbdConstraintSet& constraints);
XpbdWorld UploadXpbdWorld(const XpbdParticleSet& particles,
                          const XpbdConstraintSet& constraints);

XpbdStepReport StepXpbdWorld(const phi::DeviceContext& context,
                             XpbdWorld& world,
                             const XpbdStepOptions& options = {});
XpbdStepReport StepXpbdWorld(XpbdWorld& world,
                             const XpbdStepOptions& options = {});

} // namespace nuka::runtime::soft
