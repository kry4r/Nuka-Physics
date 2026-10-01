#pragma once
// Vertex blocks descend inertial and elastic energy through the current local curvature.
// Rows use the step's frozen response W; their accumulated impulses stay fixed during each descent.

#include <cstdint>

#include "math/symmetric_mat3.hpp"
#include "math/vec3.hpp"
#include "nk/solve/vertex_block.hpp"
#include "phi/backend_cuda/ops/union_types.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi::nkops {

struct VertexBlockView {
    const nk::VbdElement* elements = nullptr;
    const uint32_t* offsets = nullptr;
    const uint32_t* incidence = nullptr;
    const uint32_t* color_vertices = nullptr;
    const uint32_t* color_segments = nullptr;
    const math::Vec3* start = nullptr;  // interval-start positions
    const math::Vec3* target = nullptr;
    const float* inertia = nullptr;
    const float* effective_step = nullptr;
    const uint32_t* row_first = nullptr;
    uint32_t* env_status = nullptr;
    VertexBlockLayout layout{};
    float dt = 0.0f;

    __device__ uint32_t Particle(uint32_t env, uint32_t vertex) const {
        return env * layout.particles_per_env + layout.begin + vertex;
    }
    __device__ uint32_t Slot(uint32_t env, uint32_t vertex) const {
        return env * layout.vertices + vertex;
    }
};

// Whether a point side moves a vertex-block particle.
__device__ inline bool TouchesVertexBlock(const NkRowSide& side, const PointMassView& points,
                                          const VertexBlockLayout& l) {
    if (l.vertices == 0u || !PointMassView::IsPointSide(side.kind)) return false;
    for (uint32_t i = 0u; i < points.Count(side); ++i) {
        const auto term = points.At(side, i);
        if (term.kind != kNkSideParticle) continue;
        const uint32_t local = term.index % l.particles_per_env;
        if (local >= l.begin && local - l.begin < l.vertices) return true;
    }
    return false;
}

// Interval-start positions and relative in-step displacements reach the elements separately.
__device__ inline nk::vbd::ElementGeometry ElementStepGeometry(const VertexBlockView& b,
                                                              const math::Vec3* velocity,
                                                              const nk::VbdElement& element,
                                                              uint32_t env) {
    nk::vbd::ElementGeometry geometry;
    const math::Vec3 rate = velocity[b.Particle(env, element.vertex[0])];
    for (uint32_t j = 0u; j < nk::VbdElementVertexCount(element.kind); ++j) {
        const uint32_t particle = b.Particle(env, element.vertex[j]);
        geometry.start[j] = b.start[particle];
        geometry.delta[j] = (velocity[particle] - rate) * b.dt;
    }
    return geometry;
}

// Rayleigh damping contributes d H u to the gradient.
__device__ inline void GatherVertexBlock(const VertexBlockView& b, const math::Vec3* velocity,
                                         uint32_t env, uint32_t vertex, uint32_t lane,
                                         uint32_t width, math::Vec3& gradient,
                                         math::SymmetricMat3& hessian) {
    gradient = {};
    hessian = {};
    const math::Vec3 own = velocity[b.Particle(env, vertex)];
    for (uint32_t i = b.offsets[vertex] + lane; i < b.offsets[vertex + 1u]; i += width) {
        const uint32_t packed = b.incidence[i];
        const nk::VbdElement element = b.elements[nk::VbdIncidenceElement(packed)];
        const nk::vbd::ElementGeometry geometry = ElementStepGeometry(b, velocity, element, env);
        math::Vec3 g;
        math::SymmetricMat3 h;
        nk::vbd::ElementBlock(element, nk::VbdIncidenceLocal(packed), geometry, &g, &h);
        if (element.damping > 0.0f) {
            g = g + h.Multiply(own) * element.damping;
            h = nk::vbd::Scaled(h, 1.0f + element.damping / b.dt);
        }
        gradient = gradient + g;
        nk::vbd::AddTo(hessian, h);
    }
}

// Backtracking covers the significand resolution of a single-precision step.
constexpr uint32_t kVertexStepHalvings = 24u;
// Fraction of the first-order decrease a vertex step must realize.
constexpr float kVertexStepDecrease = 1.0e-4f;
// Trial steps start over-relaxed to carry smooth error across stiff membranes in fewer sweeps;
// Armijo backtracking still enforces descent. Row-coupled blocks start at their dominating-metric step.
constexpr float kVertexStepRelaxation = 1.9f;

// Change of the elastic and Rayleigh energy at a vertex whose rate moves from `from` by `move`.
// Rayleigh damping is the dissipation (d h / 2) u^T H u with H at the current iterate.
__device__ inline float VertexEnergyChange(const VertexBlockView& b, const math::Vec3* velocity,
                                           uint32_t env, uint32_t vertex, math::Vec3 from,
                                           math::Vec3 move, uint32_t lane, uint32_t width) {
    float change = 0.0f;
    for (uint32_t i = b.offsets[vertex] + lane; i < b.offsets[vertex + 1u]; i += width) {
        const uint32_t packed = b.incidence[i];
        const nk::VbdElement element = b.elements[nk::VbdIncidenceElement(packed)];
        const uint32_t local = nk::VbdIncidenceLocal(packed);
        nk::vbd::ElementGeometry geometry = ElementStepGeometry(b, velocity, element, env);
        geometry.delta[local] = (from - velocity[b.Particle(env, element.vertex[0])]) * b.dt;
        change += nk::vbd::ElementEnergyChange(element, local, geometry, move * b.dt);
        if (element.damping > 0.0f) {
            math::Vec3 g;
            math::SymmetricMat3 h;
            nk::vbd::ElementBlock(element, local, geometry, &g, &h);
            change += 0.5f * element.damping * b.dt * move.Dot(h.Multiply(from * 2.0f + move));
        }
    }
    return change;
}

__device__ inline void VertexBlockEquationWarp(const VertexBlockView& b, PointMassView points,
                                               const math::Vec3* particle_error, uint32_t env,
                                               uint32_t vertex, uint32_t lane, math::Vec3& u,
                                               math::Vec3& force, math::SymmetricMat3& hessian) {
    const uint32_t particle = b.Particle(env, vertex);
    math::Vec3 gradient;
    GatherVertexBlock(b, points.particle_velocity, env, vertex, lane, warpSize, gradient, hessian);
    // The sums reach every lane, so every lane forms the same step and search.
    gradient = {WarpSum(gradient.x), WarpSum(gradient.y), WarpSum(gradient.z)};
    hessian = {WarpSum(hessian.xx), WarpSum(hessian.yy), WarpSum(hessian.zz),
               WarpSum(hessian.xy), WarpSum(hessian.xz), WarpSum(hessian.yz)};
    const uint32_t slot = b.Slot(env, vertex);
    u = points.particle_velocity[particle];
    if (particle_error != nullptr) u = u - particle_error[particle];
    const math::Vec3 impulse = points.particle_row_impulse[particle];
    const float inertia = b.inertia[slot];
    const float h = b.dt;
    const math::Vec3 target = b.target[slot];
    force = (u * h - target) * -inertia - gradient + impulse / h;
    nk::vbd::AddIdentity(hessian, inertia);
}

// A completed sweep must also satisfy the vertex equations after all neighboring row updates.
__device__ inline bool VertexBlockStationaryWarp(const VertexBlockView& b, PointMassView points,
                                                 const math::Vec3* particle_error, uint32_t env,
                                                 uint32_t vertex, uint32_t lane, float tolerance) {
    const uint32_t particle = b.Particle(env, vertex);
    if (!(points.particle_inv_mass[particle] > 0.0f)) return true;
    math::Vec3 u, force;
    math::SymmetricMat3 hessian, inverse;
    VertexBlockEquationWarp(b, points, particle_error, env, vertex, lane, u, force, hessian);
    const float inertia = b.inertia[b.Slot(env, vertex)];
    if (!(fabsf(force.x) <= FLT_MAX && fabsf(force.y) <= FLT_MAX && fabsf(force.z) <= FLT_MAX)) {
        if (lane == 0u) atomicOr(b.env_status + env, kEnvStatusSolverFailure);
        return false;
    }
    if (!(tolerance > 0.0f)) return force.x == 0.0f && force.y == 0.0f && force.z == 0.0f;
    if (!nk::vbd::Invert(hessian, 1.0e-6f * inertia * inertia * inertia, &inverse)) return false;
    const math::Vec3 correction = inverse.Multiply(force) / b.dt;
    const math::Vec3 momentum_error = nk::vbd::MomentumResidualVelocity(
        force, points.particle_inv_mass[particle], b.effective_step[b.Slot(env, vertex)]);
    return fabsf(correction.x) <= tolerance && fabsf(correction.y) <= tolerance &&
           fabsf(correction.z) <= tolerance && fabsf(momentum_error.x) <= tolerance &&
           fabsf(momentum_error.y) <= tolerance && fabsf(momentum_error.z) <= tolerance;
}

// Descend the incremental potential through local curvature, holding accumulated row impulses fixed.
// Convergence bounds both the Newton correction and the physical momentum-equation defect.
__device__ inline bool SolveVertexBlockWarp(const VertexBlockView& b, PointMassView points,
                                            math::Vec3* particle_error, uint32_t env,
                                            uint32_t vertex, uint32_t lane, float tolerance) {
    const uint32_t particle = b.Particle(env, vertex);
    if (!(points.particle_inv_mass[particle] > 0.0f)) return false;
    math::Vec3 u, force;
    math::SymmetricMat3 hessian;
    VertexBlockEquationWarp(b, points, particle_error, env, vertex, lane, u, force, hessian);
    const uint32_t slot = b.Slot(env, vertex);
    const math::Vec3 impulse = points.particle_row_impulse[particle];
    const float inertia = b.inertia[slot], h = b.dt;
    const math::Vec3 target = b.target[slot];
    if (!(fabsf(force.x) <= FLT_MAX && fabsf(force.y) <= FLT_MAX && fabsf(force.z) <= FLT_MAX)) {
        if (lane == 0u) atomicOr(b.env_status + env, kEnvStatusSolverFailure);
        return true;
    }
    math::SymmetricMat3 inverse;
    const bool invertible = nk::vbd::Invert(hessian, 1.0e-6f * inertia * inertia * inertia, &inverse);
    const math::Vec3 correction = invertible ? inverse.Multiply(force) / h : math::Vec3{};
    const math::Vec3 momentum_error = nk::vbd::MomentumResidualVelocity(
        force, points.particle_inv_mass[particle], b.effective_step[slot]);
    // Row-coupled blocks add the least share of the frozen row metric that makes the search metric
    // dominate this block and the row response; gradient and stationarity correction are unchanged.
    math::SymmetricMat3 search_inverse = inverse;
    bool search_invertible = invertible;
    // Only rows that have pushed this block in the step couple it; an idle speculative row exchanges nothing.
    const bool rows = b.row_first != nullptr && b.row_first[particle] != ~0u &&
                      (impulse.x != 0.0f || impulse.y != 0.0f || impulse.z != 0.0f);
    if (rows) {
        const math::SymmetricMat3 response = points.particle_response[particle];
        math::SymmetricMat3 frozen_metric;
        const bool frozen_invertible = nk::vbd::Invert(response, 0.0f, &frozen_metric);
        if (frozen_invertible) {
            const double ratio =
                nk::vbd::SmallestRelativeEigenvalue(hessian, nk::vbd::Scaled(response, h * h));
            const float share = float(fmin(fmax(1.0 - ratio, 0.0), 1.0));
            nk::vbd::AddTo(hessian, nk::vbd::Scaled(frozen_metric, share / (h * h)));
        }
        search_invertible = frozen_invertible && nk::vbd::Invert(hessian, 0.0f, &search_inverse);
    }
    if (!search_invertible) {
        if (lane == 0u) atomicOr(b.env_status + env, kEnvStatusSolverFailure);
        return true;
    }
    const math::Vec3 step = search_invertible ? search_inverse.Multiply(force) / h : math::Vec3{};
    const float slope = -h * force.Dot(step);
    math::Vec3 next = u;
    float scale = rows ? 1.0f : kVertexStepRelaxation;
    for (uint32_t halving = 0u; halving <= kVertexStepHalvings && slope < 0.0f; ++halving) {
        const math::Vec3 move = step * scale;
        const float elastic = WarpSum(VertexEnergyChange(b, points.particle_velocity, env, vertex,
                                                         u, move, lane, warpSize));
        const float change = elastic - impulse.Dot(move) +
                             0.5f * inertia * h * move.Dot((u * 2.0f + move) * h - target * 2.0f);
        if (change <= kVertexStepDecrease * scale * slope) {
            next = u + move;
            break;
        }
        scale *= 0.5f;
    }
    // Every lane has read the vertex state before lane 0 overwrites it.
    __syncwarp();
    if (lane == 0u) {
        points.particle_velocity[particle] = next;
        if (particle_error != nullptr) particle_error[particle] = {};
    }
    if (!invertible || !search_invertible) return force.x != 0.0f || force.y != 0.0f || force.z != 0.0f;
    return tolerance > 0.0f
        ? !(fabsf(correction.x) <= tolerance && fabsf(correction.y) <= tolerance &&
            fabsf(correction.z) <= tolerance && fabsf(momentum_error.x) <= tolerance &&
            fabsf(momentum_error.y) <= tolerance && fabsf(momentum_error.z) <= tolerance)
        : (force.x != 0.0f || force.y != 0.0f || force.z != 0.0f);
}

}  // namespace nuka::phi::nkops
