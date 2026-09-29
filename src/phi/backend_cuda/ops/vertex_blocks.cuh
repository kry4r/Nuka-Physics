#pragma once
// Vertex blocks: each dynamic vertex takes a 3x3 Newton step on its inertial and elastic energy
// with the impulses rows applied to it held fixed; rows see it through a frozen response W.

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
    math::Vec3* row_impulse = nullptr;
    math::Vec3* written = nullptr;
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

// Elastic force and Hessian at a vertex, with lanes striding its elements; positions are x = start + h u.
// Rayleigh damping adds (d / h) H to the block and d H u to the gradient.
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
        math::Vec3 x[4];
        const uint32_t count = nk::VbdElementVertexCount(element.kind);
        for (uint32_t j = 0u; j < count; ++j) {
            const uint32_t particle = b.Particle(env, element.vertex[j]);
            x[j] = b.start[particle] + velocity[particle] * b.dt;
        }
        math::Vec3 g;
        math::SymmetricMat3 h;
        nk::vbd::ElementBlock(element, nk::VbdIncidenceLocal(packed), x, &g, &h);
        if (element.damping > 0.0f) {
            g = g + h.Multiply(own) * element.damping;
            nk::vbd::AddTo(hessian, nk::vbd::Scaled(h, 1.0f + element.damping / b.dt));
        } else {
            nk::vbd::AddTo(hessian, h);
        }
        gradient = gradient + g;
    }
}

// Newton step of one vertex. Row impulses since its last write are recovered through W^-1.
// Returns whether the velocity moved past `tolerance` (any change when it is 0).
__device__ inline bool SolveVertexBlockWarp(const VertexBlockView& b, PointMassView points,
                                            math::Vec3* particle_error, uint32_t env,
                                            uint32_t vertex, uint32_t lane, float tolerance) {
    math::Vec3 gradient;
    math::SymmetricMat3 hessian;
    GatherVertexBlock(b, points.particle_velocity, env, vertex, lane, warpSize, gradient, hessian);
    gradient = {WarpSum(gradient.x), WarpSum(gradient.y), WarpSum(gradient.z)};
    hessian = {WarpSum(hessian.xx), WarpSum(hessian.yy), WarpSum(hessian.zz),
               WarpSum(hessian.xy), WarpSum(hessian.xz), WarpSum(hessian.yz)};
    bool significant = false;
    if (lane == 0u) {
        const uint32_t particle = b.Particle(env, vertex);
        const uint32_t slot = b.Slot(env, vertex);
        math::Vec3 u = points.particle_velocity[particle];
        if (particle_error != nullptr) u = u - particle_error[particle];
        math::SymmetricMat3 stiffness;
        math::Vec3 impulse = b.row_impulse[slot];
        if (nk::vbd::Invert(points.particle_response[particle], 0.0f, &stiffness))
            impulse = impulse + stiffness.Multiply(u - b.written[slot]);
        const float inertia = b.inertia[slot];
        const float h = b.dt;
        const math::Vec3 force = (u * h - b.target[slot]) * -inertia - gradient + impulse / h;
        nk::vbd::AddIdentity(hessian, inertia);
        math::SymmetricMat3 inverse;
        math::Vec3 next = u;
        // A block whose determinant falls below the inertial floor keeps its velocity this sweep.
        if (nk::vbd::Invert(hessian, 1.0e-6f * inertia * inertia * inertia, &inverse))
            next = u + inverse.Multiply(force) / h;
        points.particle_velocity[particle] = next;
        if (particle_error != nullptr) particle_error[particle] = {};
        b.written[slot] = next;
        b.row_impulse[slot] = impulse;
        const math::Vec3 change = next - u;
        significant = tolerance > 0.0f
            ? fmaxf(fabsf(change.x), fmaxf(fabsf(change.y), fabsf(change.z))) > tolerance
            : (change.x != 0.0f || change.y != 0.0f || change.z != 0.0f);
    }
    return __shfl_sync(0xffffffffu, significant ? 1u : 0u, 0u) != 0u;
}

}  // namespace nuka::phi::nkops
