#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::soft::XpbdWorld -- GPU-resident XPBD soft-body world (p09-A)
// ---------------------------------------------------------------------------
//
// v0.7 p09-A: the FOUNDATIONAL vertical slice of XPBD (Extended Position-Based
// Dynamics, Macklin et al. 2016) soft-body support. This module owns the GPU
// particle state (positions / prev-positions / velocities / inverse masses) and
// a set of DISTANCE constraints (id 6 = kXpbdDistanceRowClassId), and runs the
// XPBD integration loop ENTIRELY on the GPU (master plan SS5.6 GPU-only hard
// constraint -- no CPU physics in the production path):
//
//     predict :  prev = p ;  p += v*dt + (gravity + f_ext)*dt^2          (per particle)
//     solve   :  iterate the distance constraints (fixed-order GS sweep)  (per step)
//     correct :  v = (p - prev) / dt                                      (per particle)
//
// The distance projection is the XPBD scheme (one constraint between particles
// a, b with inverse masses w_a, w_b):
//     C   = |p_a - p_b| - rest_length
//     n   = (p_a - p_b) / |p_a - p_b|
//     a~  = compliance_alpha / dt^2
//     dl  = (-C - a~*lambda) / (w_a + w_b + a~)
//     p_a += w_a*n*dl ;  p_b -= w_b*n*dl ;  lambda += dl
//
// D1 STRONG DETERMINISM: every device buffer is allocated ONCE at upload (no
// per-step cudaMalloc); the predict/correct kernels write distinct per-particle
// outputs (no atomics); the constraint solve is a single-thread FIXED-ORDER
// Gauss-Seidel sweep (constraints visited in index order, no float atomics, no
// thread-order dependence). Two runs of Step() with identical inputs produce
// byte-identical position/velocity buffers.
//
// Parallel / colored Gauss-Seidel (for many-constraint scaling) is DEFERRED to
// p09-B (named downstream consumer of this single-thread serial sweep).
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

struct XpbdConstraintSet {
    std::vector<XpbdDistanceConstraint> distance;
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
    // fixed point in ONE iteration; multiple iterations matter only for coupled
    // constraint systems. >= 1.
    uint32_t solver_iterations = 1u;
};

struct XpbdStepReport {
    uint32_t particle_count = 0u;
    uint32_t distance_constraint_count = 0u;
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
              phi::Buffer positions,
              phi::Buffer prev_positions,
              phi::Buffer velocities,
              phi::Buffer inv_masses,
              phi::Buffer distance_particle_a,
              phi::Buffer distance_particle_b,
              phi::Buffer distance_rest_length,
              phi::Buffer distance_compliance_alpha,
              phi::Buffer distance_lambda);

    XpbdWorld(const XpbdWorld&) = delete;
    XpbdWorld& operator=(const XpbdWorld&) = delete;
    XpbdWorld(XpbdWorld&&) noexcept = default;
    XpbdWorld& operator=(XpbdWorld&&) noexcept = default;

    uint32_t ParticleCount() const { return particle_count_; }
    uint32_t DistanceConstraintCount() const { return distance_constraint_count_; }
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

private:
    uint32_t particle_count_ = 0u;
    uint32_t distance_constraint_count_ = 0u;
    phi::Buffer positions_;
    phi::Buffer prev_positions_;
    phi::Buffer velocities_;
    phi::Buffer inv_masses_;
    phi::Buffer distance_particle_a_;
    phi::Buffer distance_particle_b_;
    phi::Buffer distance_rest_length_;
    phi::Buffer distance_compliance_alpha_;
    phi::Buffer distance_lambda_;
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
