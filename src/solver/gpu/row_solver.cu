// ---------------------------------------------------------------------------
// nuka::solver::gpu::row_solver implementation
// ---------------------------------------------------------------------------

#include "solver/gpu/row_solver.cuh"

#include "constraint/contact_row_sides.hpp"  // v0.8 C5a: ContactRowSides (POD)
#include "constraint/reaction_provider.hpp"  // v0.8 C5a: ReactionProvider math cores
#include "constraint/row_articulation_refs.hpp"  // v0.8 C5b: RowArticulationRefs (POD)
#include "math/cuda_vec_ops.cuh"
#include "phi/buffer_legacy.hpp"
#include "phi/buffer_transfer.hpp"
#include "solver/gpu/row_scheduler.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::solver::gpu {

namespace {

constexpr uint32_t kRowSolverBlockSize = 128u;

struct DeviceRowBuffers {
    constraint::Row* rows = nullptr;
    uint32_t* body_indices = nullptr;
    constraint::RowJacobian6* jacobian_data = nullptr;
    constraint::RowMaterial* materials = nullptr;
    constraint::RowAnchor* anchors = nullptr;
    // v0.8 C5a: per-row CollidableRef sides (one per row, from
    // EmitCompliantContactRows). Read ONLY on the compliant branch
    // (row.flags & Compliant). Legacy SolveConstraints passes nullptr/0 and the
    // legacy branch NEVER touches this -> legacy solve byte-identical.
    const constraint::ContactRowSides* sides = nullptr;
    uint32_t sides_count = 0u;
    // v0.8 C5b-core: per-row articulation indirection + global articulation
    // reaction buffers (chain-J rows, M^-1 tiles, live qdot). Read ONLY on the
    // compliant branch for a side whose art_index != kInvalidArtIndex. Legacy /
    // C5a callers leave these null/0 -> the articulation arms never fire ->
    // byte-identical. See constraint/row_articulation_refs.hpp.
    const constraint::RowArticulationRefs* art_refs = nullptr;
    uint32_t art_refs_count = 0u;
    const float* chain_jacobians = nullptr;   // [slot * dof_stride] per (row,side)
    const float* inertia_m_inv = nullptr;     // [art_index * dof_stride^2]
    float* qdot = nullptr;                     // [art_index * dof_stride] (MUTABLE)
    uint32_t dof_stride = 0u;
    // v0.8 C6b: per-particle reaction SoA (UnifiedSolve coupling rows only). A
    // ParticleInvMass contact side resolves its scalar inverse mass + linear
    // velocity by ref.handle DIRECTLY into these arrays (no per-row indirection
    // stream, unlike articulation -- a particle is a single point DOF). velocities
    // is MUTABLE (the apply writes it; downloaded after the solve). Legacy / C5a /
    // C5b callers leave these null/0 -> the particle arm never fires -> the rigid
    // and articulation paths stay byte-identical.
    const float* particle_inv_mass = nullptr;   // [particle_count]
    math::Vec3* particle_velocities = nullptr;  // [particle_count] (MUTABLE)
    uint32_t particle_count = 0u;
    uint32_t row_count = 0u;
    uint32_t body_index_count = 0u;
    uint32_t jacobian_data_count = 0u;
};

// Small-vector / quaternion primitives now come from the shared device math
// library (math/cuda_vec_ops.cuh). Bodies are bit-identical to the former local
// copies. The former local quaternion `Normalize` maps to
// QuatNormalizeSqrtLe(., 1e-12f); `Multiply` -> QuatMul; `Rotate` -> RotateShort
// (renamed at call sites). The former forward-declared local `Dot` is dropped;
// the shared Dot is used instead. UploadVector comes from
// phi/buffer_transfer.hpp.
namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Cross;
using mg::Dot;
using mg::Length;
using mg::MakeQuat;
using mg::MakeVec3;
using mg::QuatMul;
using mg::RotateShort;
using mg::Scale;
using mg::Sub;

__device__ void ApplyAngularPositionCorrection(runtime::rigid::BodyState& body,
                                               math::Vec3 angular_jacobian,
                                               float position_impulse) {
    const math::Vec3 angular_delta =
        MakeVec3(angular_jacobian.x * body.inv_inertia.x * position_impulse,
                 angular_jacobian.y * body.inv_inertia.y * position_impulse,
                 angular_jacobian.z * body.inv_inertia.z * position_impulse);
    const float angle = Length(angular_delta);
    if (angle <= 1.0e-8f) {
        return;
    }

    const math::Vec3 axis = Scale(angular_delta, 1.0f / angle);
    const float half_angle = 0.5f * angle;
    const float sine = sinf(half_angle);
    const math::Quat dq =
        MakeQuat(cosf(half_angle), axis.x * sine, axis.y * sine, axis.z * sine);
    body.orientation = mg::QuatNormalizeSqrtLe(QuatMul(dq, body.orientation), 1.0e-12f);
}

__device__ uint32_t UMin(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

__device__ uint32_t UMax(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

__device__ bool ValidBody(uint32_t body, uint32_t body_count) {
    return body != constraint::kInvalidBodyIndex && body < body_count;
}

__device__ constraint::RowJacobian6 JacobianForRowBody(const DeviceRowBuffers& rows,
                                                       const constraint::Row& row,
                                                       uint32_t local_body_index) {
    constraint::RowJacobian6 result;
    result.linear = MakeVec3(0.0f, 0.0f, 0.0f);
    result.angular = MakeVec3(0.0f, 0.0f, 0.0f);
    if (local_body_index >= row.body_count) {
        return result;
    }
    const uint32_t index = row.jacobian_offset + local_body_index;
    if (index >= rows.jacobian_data_count) {
        return result;
    }
    return rows.jacobian_data[index];
}

__device__ uint32_t BodyForRowBody(const DeviceRowBuffers& rows,
                                   const constraint::Row& row,
                                   uint32_t local_body_index) {
    if (local_body_index >= row.body_count) {
        return constraint::kInvalidBodyIndex;
    }
    const uint32_t index = row.body_list_offset + local_body_index;
    if (index >= rows.body_index_count) {
        return constraint::kInvalidBodyIndex;
    }
    return rows.body_indices[index];
}

__device__ float ComputeJv(const DeviceRowBuffers& rows,
                           uint32_t row_index,
                           const runtime::rigid::BodyState* bodies,
                           uint32_t body_count) {
    const constraint::Row& row = rows.rows[row_index];
    float jv = 0.0f;
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, body_count)) {
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        const auto& body = bodies[body_index];
        jv += Dot(jacobian.linear, body.linear_velocity);
        jv += Dot(jacobian.angular, body.angular_velocity);
    }
    return jv;
}

__device__ float ComputeEffectiveMass(const DeviceRowBuffers& rows,
                                      uint32_t row_index,
                                      const runtime::rigid::BodyState* bodies,
                                      uint32_t body_count) {
    const constraint::Row& row = rows.rows[row_index];
    float diagonal = 0.0f;
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, body_count)) {
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        const auto& body = bodies[body_index];
        diagonal += body.inv_mass * Dot(jacobian.linear, jacobian.linear);
        diagonal += jacobian.angular.x * jacobian.angular.x * body.inv_inertia.x;
        diagonal += jacobian.angular.y * jacobian.angular.y * body.inv_inertia.y;
        diagonal += jacobian.angular.z * jacobian.angular.z * body.inv_inertia.z;
    }
    return diagonal > 1.0e-12f ? 1.0f / diagonal : 0.0f;
}

__device__ void ApplyVelocityImpulse(const DeviceRowBuffers& rows,
                                     uint32_t row_index,
                                     float delta_impulse,
                                     runtime::rigid::BodyState* bodies,
                                     uint32_t body_count) {
    const constraint::Row& row = rows.rows[row_index];
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, body_count)) {
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        auto& body = bodies[body_index];
        if (body.inv_mass <= 0.0f) {
            continue;
        }
        body.linear_velocity = Add(
            body.linear_velocity,
            Scale(jacobian.linear, body.inv_mass * delta_impulse));
        body.angular_velocity.x += jacobian.angular.x * body.inv_inertia.x * delta_impulse;
        body.angular_velocity.y += jacobian.angular.y * body.inv_inertia.y * delta_impulse;
        body.angular_velocity.z += jacobian.angular.z * body.inv_inertia.z * delta_impulse;
    }
}

// ===========================================================================
// v0.8 C5a -- COMPLIANT BRANCH device helpers (gated on row.flags & Compliant)
// ===========================================================================
// These run ONLY for compliant rows. They thread the per-row ContactRowSides
// (rows.sides[row_index]) and dispatch each side by side.react. For C5a only the
// RigidInvMass arm is exercised (the box stack is all-rigid). The dispatch switch
// is PRESENT for all kinds so C5b adds the articulation arm without restructuring;
// ParticleInvMass / ArticulationChainJ / StaticNull are in-place but UNEXERCISED
// (see the per-arm comments). This is the spec's "inline RigidInvMass" option:
// we do NOT route a device-resident ReactionProviderViews array for C5a; instead
// we build a RigidReactionState from bodies[idx] on the spot, which reuses the
// SAME math cores (RigidEffectiveInvMass / RigidApplyImpulse) as constraint::
// ReactionProvider so the wiring is behavior-faithful to C4c.

// Per-side CollidableRef for local body 0 (== side.a) / local body 1 (== side.b),
// matching EmitCompliantContactRows (push_back({manifold.a, manifold.b}) paired
// with AddRow({a.handle, b.handle}, jac_a, jac_b)).
__device__ constraint::CollidableRef SideForRowBody(const DeviceRowBuffers& rows,
                                                    uint32_t row_index,
                                                    uint32_t local_body_index) {
    if (rows.sides == nullptr || row_index >= rows.sides_count) {
        return constraint::CollidableRef{};
    }
    return local_body_index == 0u ? rows.sides[row_index].a
                                  : rows.sides[row_index].b;
}

// v0.8 C5b-core: the per-row articulation indirection for local body 0 (== side.a)
// / local body 1 (== side.b). Returns the not-an-articulation sentinel if no
// art_refs stream is present or the row is out of range -> the articulation arm
// stays inert (the side resolves via its C5a per-kind dispatch instead).
__device__ constraint::RowArticulationSide ArtSideForRowBody(
    const DeviceRowBuffers& rows,
    uint32_t row_index,
    uint32_t local_body_index) {
    if (rows.art_refs == nullptr || row_index >= rows.art_refs_count) {
        return constraint::RowArticulationSide{};  // art_index == kInvalidArtIndex
    }
    return local_body_index == 0u ? rows.art_refs[row_index].a
                                  : rows.art_refs[row_index].b;
}

// v0.8 C5b-core: build a C4c ArticulationReactionState from a side's
// RowArticulationSide indirection into the global buffers. The chain-J row is
// keyed by chain_jac_slot (per-contact), the M^-1 tile + the live qdot slice by
// art_index (per-articulation) -- DISTINCT indices (see row_articulation_refs.hpp).
// qdot is the MUTABLE live joint-velocity slice ArticulationApplyImpulse writes.
__device__ constraint::ArticulationReactionState BuildArticulationState(
    const DeviceRowBuffers& rows,
    const constraint::RowArticulationSide& art_side) {
    constraint::ArticulationReactionState state;
    if (art_side.art_index == constraint::kInvalidArtIndex ||
        rows.chain_jacobians == nullptr || rows.inertia_m_inv == nullptr ||
        rows.qdot == nullptr || rows.dof_stride == 0u) {
        return state;  // null pointers -> the C4c helpers no-op (return 0 / skip).
    }
    const size_t stride = static_cast<size_t>(rows.dof_stride);
    state.chain_jacobian =
        rows.chain_jacobians + static_cast<size_t>(art_side.chain_jac_slot) * stride;
    state.inertia_M_inv =
        rows.inertia_m_inv + static_cast<size_t>(art_side.art_index) * stride * stride;
    state.qdot = rows.qdot + static_cast<size_t>(art_side.art_index) * stride;
    state.dof_stride = rows.dof_stride;
    return state;
}

// v0.8 C6b: build a C4c ParticleReactionState for a ParticleInvMass contact side.
// The particle is indexed DIRECTLY by side.handle (== particle id) into the
// per-particle SoA -- no per-row indirection stream (a particle is one point DOF,
// unlike an articulation chain). A non-particle side / null SoA / out-of-range
// handle -> an empty state (inv_mass 0 + null velocity) the C4c helpers no-op on.
// The velocity pointer aliases the MUTABLE device array so ParticleApplyImpulse
// writes through it directly (downloaded after the solve).
__device__ constraint::ParticleReactionState BuildParticleState(
    const DeviceRowBuffers& rows,
    const constraint::CollidableRef& side) {
    constraint::ParticleReactionState state;
    if (side.react != constraint::ReactionProviderKind::ParticleInvMass ||
        rows.particle_inv_mass == nullptr || rows.particle_velocities == nullptr ||
        side.handle >= rows.particle_count) {
        return state;  // null pointers -> the C4c helpers no-op (0 / skip).
    }
    state.inv_mass = rows.particle_inv_mass[side.handle];
    state.velocity = &rows.particle_velocities[side.handle];
    return state;
}

// One side's J M^-1 J^T contribution via the C4c provider math, dispatched by
// side.react. Returns the RAW additive diagonal term (NOT the reciprocal);
// the caller sums both sides + compliance_alpha and reciprocates once.
//
// v0.8 C5b-core: `art_state` is the resolved ArticulationReactionState for this
// side (null pointers if the side is not an articulation); the ArticulationChainJ
// arm uses it directly (the side's RowJacobian6 is IGNORED -- the chain-J row is
// the reduced-coordinate Jacobian).
__device__ float CompliantSideEffectiveInvMass(
    const constraint::CollidableRef& side,
    const constraint::RowJacobian6& j,
    uint32_t body_index,
    const runtime::rigid::BodyState* bodies,
    uint32_t body_count,
    const constraint::ArticulationReactionState& art_state,
    const constraint::ParticleReactionState& part_state) {
    switch (side.react) {
        case constraint::ReactionProviderKind::RigidInvMass: {
            if (!ValidBody(body_index, body_count)) {
                return 0.0f;  // static/invalid rigid side: zero reaction.
            }
            const auto& body = bodies[body_index];
            constraint::RigidReactionState state;
            state.inv_mass = body.inv_mass;
            state.inv_inertia = body.inv_inertia;
            return constraint::RigidEffectiveInvMass(state, j);
        }
        case constraint::ReactionProviderKind::ArticulationChainJ:
            // v0.8 C5b-core: reduced-coordinate chain-J reaction = J M^-1 J^T over
            // the per-contact chain Jacobian + the articulation's M^-1 tile. The
            // C4c helper returns 0 if art_state's pointers are null (not wired).
            return constraint::ArticulationEffectiveInvMass(art_state);
        case constraint::ReactionProviderKind::ParticleInvMass:
            // v0.8 C6b: particle reaction J M^-1 J^T = inv_mass * |J_linear|^2
            // (M^-1 = inv_mass*I3, no angular DOF). The C4c helper returns 0 if
            // part_state is empty (a non-particle side / null SoA / out-of-range).
            return constraint::ParticleEffectiveInvMass(part_state, j);
        case constraint::ReactionProviderKind::StaticNull:
        default:
            return 0.0f;  // immovable world collider: zero reaction.
    }
}

// Apply a solved impulse delta to one side's DOFs via the C4c provider math,
// dispatched by side.react. Mutates bodies[idx] for the RigidInvMass arm.
//
// v0.8 C5b-core: `art_state` is the resolved ArticulationReactionState (with the
// MUTABLE qdot slice) for an articulation side; the ArticulationChainJ arm writes
// qdot += M^-1 J^T delta through it. Null pointers -> the C4c helper no-ops.
__device__ void CompliantSideApplyImpulse(
    const constraint::CollidableRef& side,
    const constraint::RowJacobian6& j,
    float delta,
    uint32_t body_index,
    runtime::rigid::BodyState* bodies,
    uint32_t body_count,
    const constraint::ArticulationReactionState& art_state,
    const constraint::ParticleReactionState& part_state) {
    switch (side.react) {
        case constraint::ReactionProviderKind::RigidInvMass: {
            if (!ValidBody(body_index, body_count)) {
                return;
            }
            auto& body = bodies[body_index];
            if (body.inv_mass <= 0.0f) {
                return;  // immovable rigid: no-op (mirrors RigidApplyImpulse).
            }
            // BYTE-FOR-BYTE constraint::RigidApplyImpulse on the live body state.
            body.linear_velocity = Add(
                body.linear_velocity, Scale(j.linear, body.inv_mass * delta));
            body.angular_velocity.x += j.angular.x * body.inv_inertia.x * delta;
            body.angular_velocity.y += j.angular.y * body.inv_inertia.y * delta;
            body.angular_velocity.z += j.angular.z * body.inv_inertia.z * delta;
            return;
        }
        case constraint::ReactionProviderKind::ArticulationChainJ:
            // v0.8 C5b-core: qdot += M^-1 J^T delta via the C4c apply (BYTE-FOR-BYTE
            // SolveArticulatedContactRowsKernel's flat-qdot loop). For a FIXED-ROOT
            // chain (v0.8 go2/h1) the flat-qdot apply is exact; floating-base base-
            // DOF routing to link_velocity is a C5b-coresidence/later gap.
            constraint::ArticulationApplyImpulse(art_state, delta);
            return;
        case constraint::ReactionProviderKind::ParticleInvMass:
            // v0.8 C6b: dv = delta * inv_mass * J_linear (no angular). Writes the
            // MUTABLE particle velocity through part_state.velocity (the C4c helper
            // no-ops on inv_mass<=0 or a null velocity pointer).
            constraint::ParticleApplyImpulse(part_state, j, delta);
            return;
        case constraint::ReactionProviderKind::StaticNull:
        default:
            return;  // immovable: no-op.
    }
}

// v0.8 C5b-core ★ THE THIRD ARM: one side's constraint velocity contribution
// Jv = (this side's row of J) . (this side's DOF velocities), dispatched by
// side.react. This REPLACES the shared ComputeJv on the compliant path because
// ComputeJv loops row BODIES and `continue`s on !ValidBody -> an articulation
// side (body index >= body_count) is silently DROPPED, so the foot's real
// approach velocity (chain_J . qdot) would vanish. (That hides at equilibrium --
// a standing chain has foot Jv ~= 0 -- but breaks transients + two-way.) The
// per-arm math mirrors each provider:
//   RigidInvMass    : dot(Jlin, v) + dot(Jang, w)  (== ComputeJv's per-body term).
//   ArticulationChainJ: sum_r chain_jacobian[r] * qdot[r] (the reduced-coord Jv).
//   ParticleInvMass : dot(Jlin, v) (angular ignored) -- C6 wires the particle SoA;
//                     the rigid bodies[] array has no particle velocities, so this
//                     stays 0 here (unexercised, named C6 gap).
//   StaticNull      : 0 (immovable world: no constraint velocity).
__device__ float CompliantSideConstraintVelocity(
    const constraint::CollidableRef& side,
    const constraint::RowJacobian6& j,
    uint32_t body_index,
    const runtime::rigid::BodyState* bodies,
    uint32_t body_count,
    const constraint::ArticulationReactionState& art_state,
    const constraint::ParticleReactionState& part_state) {
    switch (side.react) {
        case constraint::ReactionProviderKind::RigidInvMass: {
            if (!ValidBody(body_index, body_count)) {
                return 0.0f;  // static/invalid rigid side: no constraint velocity.
            }
            const auto& body = bodies[body_index];
            // BYTE-FOR-BYTE ComputeJv's per-body accumulation (Jlin.v + Jang.w),
            // so an all-rigid compliant row's jv is numerically identical to the
            // shared ComputeJv -> C5a tests do not move.
            return Dot(j.linear, body.linear_velocity) +
                   Dot(j.angular, body.angular_velocity);
        }
        case constraint::ReactionProviderKind::ArticulationChainJ: {
            // Reduced-coordinate Jv = sum_r chain_jacobian[r] * qdot[r]. Null
            // pointers (not wired) -> 0. Padding cols of J are zeroed by the
            // chain-J builder, so iterating the full stride is safe.
            if (art_state.chain_jacobian == nullptr || art_state.qdot == nullptr) {
                return 0.0f;
            }
            float jv = 0.0f;
            for (uint32_t r = 0u; r < art_state.dof_stride; ++r) {
                jv += art_state.chain_jacobian[r] * art_state.qdot[r];
            }
            return jv;
        }
        case constraint::ReactionProviderKind::ParticleInvMass:
            // v0.8 C6b ★ THE PARTICLE Jv LEG. Jv = dot(J_linear, v_particle)
            // (angular ignored -- a particle has no rotational DOF). Reading the
            // particle's REAL velocity here is load-bearing: the compliant solve is
            // lambda = m_eff*(aref*dt - Jv - R*lambda), and dropping this term
            // (returning 0) would silently use Jv = 0 for the particle side -> a
            // WRONG fixed point that still passes a rest/momentum-only gate but
            // breaks transients + two-way (the exact C5b-core failure mode). Null
            // velocity (empty part_state) -> 0.
            return (part_state.velocity != nullptr)
                       ? Dot(j.linear, *part_state.velocity)
                       : 0.0f;
        case constraint::ReactionProviderKind::StaticNull:
        default:
            return 0.0f;
    }
}

// v0.8 C5c co-residence: the per-row world contact point (rides ContactRowSides).
// Returns the origin if no sides stream is present (legacy / non-contact paths).
__device__ math::Vec3 ContactPointForRow(const DeviceRowBuffers& rows,
                                         uint32_t row_index) {
    if (rows.sides == nullptr || row_index >= rows.sides_count) {
        return MakeVec3(0.0f, 0.0f, 0.0f);
    }
    return rows.sides[row_index].contact_point;
}

// v0.8 C5c co-residence ★ THE ANGULAR FIX. The compliant emitter builds a
// LINEAR-ONLY contact row (RowJacobian6.angular == 0). For a RIGID side, fill the
// angular contact Jacobian r x n IN-KERNEL from the per-row contact point and the
// body COM (body.position) -- mirroring the legacy MAXIMAL solver
// (batched_device_world.cu: jacobian_angular = Cross(r, dir)). Using r x j.linear
// gives the CORRECT sign for BOTH sides because j.linear already carries +n (side a)
// / -n (side b); for a friction spoke j.linear is +/-tangent, so r x tangent is the
// matching friction torque. Articulation / static / particle sides ignore j.angular
// (their reaction does not read it), so this is a no-op for them. WITHOUT it an
// off-COM contact gives a rigid body NO torque -> it cannot rotate (the grasp blocker).
__device__ constraint::RowJacobian6 AugmentRigidContactAngular(
    const constraint::CollidableRef& side,
    constraint::RowJacobian6 j,
    uint32_t body_index,
    const runtime::rigid::BodyState* bodies,
    uint32_t body_count,
    math::Vec3 contact_point) {
    if (side.react == constraint::ReactionProviderKind::RigidInvMass &&
        ValidBody(body_index, body_count)) {
        const math::Vec3 r = Sub(contact_point, bodies[body_index].position);
        j.angular = Cross(r, j.linear);
    }
    return j;
}

// Effective mass for a compliant row: 1 / (eff_inv_a + eff_inv_b + R), where R is
// Row.compliance_alpha (the dual regularizer -- the headline C5a wiring). Floored
// like the legacy ComputeEffectiveMass (>1e-12 -> reciprocal, else 0).
__device__ float ComputeCompliantEffectiveMass(const DeviceRowBuffers& rows,
                                               uint32_t row_index,
                                               const runtime::rigid::BodyState* bodies,
                                               uint32_t body_count) {
    const constraint::Row& row = rows.rows[row_index];
    const math::Vec3 contact_point = ContactPointForRow(rows, row_index);
    float diagonal = 0.0f;
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const constraint::CollidableRef side =
            SideForRowBody(rows, row_index, local);
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        const constraint::RowJacobian6 jacobian = AugmentRigidContactAngular(
            side, JacobianForRowBody(rows, row, local), body_index, bodies,
            body_count, contact_point);
        const constraint::RowArticulationSide art_side =
            ArtSideForRowBody(rows, row_index, local);
        const constraint::ArticulationReactionState art_state =
            BuildArticulationState(rows, art_side);
        const constraint::ParticleReactionState part_state =
            BuildParticleState(rows, side);
        diagonal += CompliantSideEffectiveInvMass(side, jacobian, body_index,
                                                  bodies, body_count, art_state,
                                                  part_state);
    }
    diagonal += row.compliance_alpha;  // + R (dual regularizer)
    return diagonal > 1.0e-12f ? 1.0f / diagonal : 0.0f;
}

// v0.8 C5b-core: the compliant row's Jv, summed over its sides via the per-side
// dispatch (NOT the shared ComputeJv -- see CompliantSideConstraintVelocity).
__device__ float ComputeCompliantJv(const DeviceRowBuffers& rows,
                                    uint32_t row_index,
                                    const runtime::rigid::BodyState* bodies,
                                    uint32_t body_count) {
    const constraint::Row& row = rows.rows[row_index];
    const math::Vec3 contact_point = ContactPointForRow(rows, row_index);
    float jv = 0.0f;
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const constraint::CollidableRef side =
            SideForRowBody(rows, row_index, local);
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        const constraint::RowJacobian6 jacobian = AugmentRigidContactAngular(
            side, JacobianForRowBody(rows, row, local), body_index, bodies,
            body_count, contact_point);
        const constraint::RowArticulationSide art_side =
            ArtSideForRowBody(rows, row_index, local);
        const constraint::ArticulationReactionState art_state =
            BuildArticulationState(rows, art_side);
        const constraint::ParticleReactionState part_state =
            BuildParticleState(rows, side);
        jv += CompliantSideConstraintVelocity(side, jacobian, body_index,
                                              bodies, body_count, art_state,
                                              part_state);
    }
    return jv;
}

// Apply a compliant row's solved impulse delta across both sides via the provider
// dispatch (NOT the legacy ApplyVelocityImpulse, which is hard-wired to the rigid
// BodyState arm and ignores side.react).
__device__ void ApplyCompliantVelocityImpulse(const DeviceRowBuffers& rows,
                                              uint32_t row_index,
                                              float delta,
                                              runtime::rigid::BodyState* bodies,
                                              uint32_t body_count) {
    const constraint::Row& row = rows.rows[row_index];
    const math::Vec3 contact_point = ContactPointForRow(rows, row_index);
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const constraint::CollidableRef side =
            SideForRowBody(rows, row_index, local);
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        const constraint::RowJacobian6 jacobian = AugmentRigidContactAngular(
            side, JacobianForRowBody(rows, row, local), body_index, bodies,
            body_count, contact_point);
        const constraint::RowArticulationSide art_side =
            ArtSideForRowBody(rows, row_index, local);
        const constraint::ArticulationReactionState art_state =
            BuildArticulationState(rows, art_side);
        const constraint::ParticleReactionState part_state =
            BuildParticleState(rows, side);
        CompliantSideApplyImpulse(side, jacobian, delta, body_index,
                                  bodies, body_count, art_state, part_state);
    }
}

__device__ bool IsFrictionRow(const DeviceRowBuffers& rows, uint32_t row_index) {
    // v0.8 C5a: a compliant friction row must NEVER take the legacy bilateral
    // [-mu*lambda_n, +mu*lambda_n] override (it uses unilateral coupled-pyramid
    // bounds via IsCompliantFrictionRow). Gate it out here; legacy friction rows
    // (no Compliant flag) keep the original predicate verbatim.
    if (rows.rows[row_index].flags & constraint::row_flags::Compliant) {
        return false;
    }
    const auto& material = rows.materials[row_index];
    return material.kind == constraint::RowKind::Contact &&
           material.friction_row_count > 0u &&
           row_index >= material.first_friction_row &&
           row_index < material.first_friction_row + material.friction_row_count;
}

// v0.8 C5a: a COMPLIANT friction spoke (unilateral coupled-pyramid edge). True
// only for rows the new emitter tagged Compliant|Friction. Its bound is the NEW
// one-sided [0, mu*TotalNormalLambda(group)] (NOT the bilateral IsFrictionRow
// override). Legacy rows never carry Compliant, so this is always false for them.
__device__ bool IsCompliantFrictionRow(const DeviceRowBuffers& rows,
                                       uint32_t row_index) {
    const uint16_t flags = rows.rows[row_index].flags;
    return (flags & constraint::row_flags::Compliant) &&
           (flags & constraint::row_flags::Friction);
}

__device__ bool IsContactNormalRow(const DeviceRowBuffers& rows, uint32_t row_index) {
    const auto& material = rows.materials[row_index];
    return material.kind == constraint::RowKind::Contact &&
           row_index >= material.group_id &&
           row_index < material.group_id + UMax(material.normal_row_count, 1u);
}

__device__ float TotalNormalLambda(const DeviceRowBuffers& rows, uint32_t row_index) {
    const auto& material = rows.materials[row_index];
    if (material.kind != constraint::RowKind::Contact ||
        material.normal_row_count == 0u) {
        return 0.0f;
    }
    float total = 0.0f;
    const uint32_t end = UMin(material.group_id + material.normal_row_count,
                              rows.row_count);
    for (uint32_t row = material.group_id; row < end; ++row) {
        total += fmaxf(rows.rows[row].lambda, 0.0f);
    }
    return total;
}

__device__ void PrepareVelocityTargetRow(DeviceRowBuffers rows,
                                         uint32_t row_index,
                                         const runtime::rigid::BodyState* bodies,
                                         uint32_t body_count) {
    if (row_index >= rows.row_count || !IsContactNormalRow(rows, row_index)) {
        return;
    }
    // v0.8 C5a: compliant normal rows already carry the velocity bias in rhs
    // (=aref, which folds in the signed approach velocity). Do NOT let the legacy
    // restitution path overwrite that bias. Legacy rows (no Compliant flag) keep
    // the original restitution-rhs logic verbatim.
    if (rows.rows[row_index].flags & constraint::row_flags::Compliant) {
        return;
    }

    const auto& material = rows.materials[row_index];
    if (material.restitution <= 0.0f) {
        return;
    }

    const float jv = ComputeJv(rows, row_index, bodies, body_count);
    if (jv < 0.0f) {
        rows.rows[row_index].rhs = fmaxf(rows.rows[row_index].rhs,
                                         -material.restitution * jv);
    }
}

// v0.8 C5a: the COMPLIANT velocity solve. Effective mass uses the per-side
// ReactionProvider math + Row.compliance_alpha (R); the impulse accumulation form
// mirrors the legacy solve (clamp the ACCUMULATED impulse, apply the delta), but
// bounds + apply are compliant-specific:
//   normal spoke   (Unilateral): [0, FLT_MAX]            (rhs = aref already set)
//   friction spoke (Friction)  : [0, mu*TotalNormalLambda] one-sided pyramid edge
__device__ void SolveCompliantVelocityRow(DeviceRowBuffers rows,
                                          uint32_t row_index,
                                          runtime::rigid::BodyState* bodies,
                                          uint32_t body_count,
                                          float dt) {
    constraint::Row& row = rows.rows[row_index];
    const float effective_mass =
        ComputeCompliantEffectiveMass(rows, row_index, bodies, body_count);
    // v0.8 C5b-core: jv via the per-side dispatch (ComputeCompliantJv), NOT the
    // shared ComputeJv -- the shared one drops articulation sides (body index >=
    // body_count -> !ValidBody -> continue). For an all-rigid compliant row the
    // per-side sum is numerically identical to ComputeJv (the rigid arm reuses
    // ComputeJv's per-body Jlin.v + Jang.w term), so C5a tests do not move. The
    // shared ComputeJv stays BYTE-UNTOUCHED for the legacy path.
    const float jv = ComputeCompliantJv(rows, row_index, bodies, body_count);
    const float old_impulse = row.lambda;
    // C5a: row.rhs holds the MuJoCo reference ACCELERATION (aref). The velocity-
    // impulse PGS needs a velocity-target bias, so scale by dt: the impulse
    // lambda = f*dt satisfies (A+R)lambda = aref*dt - Jv (R is already in 1/mass
    // units, so it is NOT scaled -- it stays in the denominator as-is). Computed
    // locally (no buffer mutation) so the caller's rows are reusable.
    const float rhs_v = row.rhs * dt;
    // v0.8 C5c-2 FIX (wrong-fixed-point consistency): the compliant denominator is
    // (A_ii + R) (ComputeCompliantEffectiveMass folds R = compliance_alpha into the
    // diagonal), so the regularized Gauss-Seidel update MUST also carry the matching
    // -R*lambda_i feedback in the NUMERATOR. ComputeCompliantJv returns Jv =
    // sum_j A_ij*lambda_j (the FULL row, diagonal included, since qdot already
    // reflects every applied impulse), so rhs_v - jv = b_i - sum_j A_ij*lambda_j.
    // WITHOUT the -R*old_impulse term the increment's fixed point is jv == rhs_v
    // i.e. A*lambda = b (R cancels): the kernel converges (flat across iters) to a
    // fixed point ~R/A above the exact solve of its OWN assembled (A+R)lambda=b
    // system (measured 0.57% == R/A on the 4-foot go2 sample). Adding -R*old_impulse
    // makes the increment lambda_new = (b_i - sum_{j!=i}A_ij*lambda_j) / (A_ii + R) --
    // EXACTLY the assembled-system Gauss-Seidel update (matches the C5c-2 exact-solve
    // oracle to PGS precision). LEGACY/RIGID PATH IS UNTOUCHED: R == 0 for every
    // non-compliant row AND for compliant friction rows (compliance_alpha = 0; the
    // emitter only writes R on compliant normal rows), so this term is identically
    // zero there -> byte-identical fixed-root / rigid solve.
    const float lambda =
        effective_mass * (rhs_v - jv - row.compliance_alpha * old_impulse);
    float lower = row.lower;
    float upper = row.upper;
    if (IsCompliantFrictionRow(rows, row_index)) {
        // NEW unilateral coupled-pyramid edge: lambda >= 0, capped at
        // mu*TotalNormalLambda(group). Each +/- spoke is an INDEPENDENT one-sided
        // row (this is why the legacy bilateral IsFrictionRow override is gated off
        // for compliant rows -- it would double the per-axis capacity).
        const float friction_limit =
            fmaxf(rows.materials[row_index].friction, 0.0f) *
            TotalNormalLambda(rows, row_index);
        lower = 0.0f;
        upper = friction_limit;
    }

    const float new_impulse = fminf(fmaxf(old_impulse + lambda, lower), upper);
    row.lambda = new_impulse;
    const float delta = new_impulse - old_impulse;
    if (fabsf(delta) > 1.0e-12f) {
        ApplyCompliantVelocityImpulse(rows, row_index, delta, bodies, body_count);
    }
}

__device__ void SolveVelocityRow(DeviceRowBuffers rows,
                                 uint32_t row_index,
                                 runtime::rigid::BodyState* bodies,
                                 uint32_t body_count,
                                 float dt) {
    // v0.8 C5a: compliant rows take the new branch; legacy rows (no Compliant
    // flag) fall through to the VERBATIM legacy solve below -> byte-identical.
    // dt is used ONLY by the compliant branch (aref->velocity-target scale).
    if (rows.rows[row_index].flags & constraint::row_flags::Compliant) {
        SolveCompliantVelocityRow(rows, row_index, bodies, body_count, dt);
        return;
    }
    constraint::Row& row = rows.rows[row_index];
    const float effective_mass = ComputeEffectiveMass(rows, row_index, bodies, body_count);
    const float jv = ComputeJv(rows, row_index, bodies, body_count);
    const float lambda = effective_mass * (row.rhs - jv);
    const float old_impulse = row.lambda;
    float lower = row.lower;
    float upper = row.upper;
    if (IsFrictionRow(rows, row_index)) {
        const float friction_limit =
            fmaxf(rows.materials[row_index].friction, 0.0f) *
            TotalNormalLambda(rows, row_index);
        lower = -friction_limit;
        upper = friction_limit;
    }

    const float new_impulse = fminf(fmaxf(old_impulse + lambda, lower), upper);
    row.lambda = new_impulse;
    const float delta = new_impulse - old_impulse;
    if (fabsf(delta) > 1.0e-12f) {
        ApplyVelocityImpulse(rows, row_index, delta, bodies, body_count);
    }
}

__device__ float SolvePositionRow(DeviceRowBuffers rows,
                                  uint32_t row_index,
                                  runtime::rigid::BodyState* bodies,
                                  uint32_t body_count,
                                  float slop,
                                  float baumgarte) {
    // v0.8 C5a: compliant contacts self-correct via the aref velocity bias (the
    // spring-damper acceleration), so they SKIP Baumgarte position projection
    // entirely. Returning 0.0 contributes no position error for these rows.
    // Legacy + joint rows (no Compliant flag) keep Baumgarte verbatim below.
    if (rows.rows[row_index].flags & constraint::row_flags::Compliant) {
        return 0.0f;
    }
    const constraint::Row& row = rows.rows[row_index];
    const auto& material = rows.materials[row_index];
    float error = 0.0f;
    bool apply_angular = false;
    math::Vec3 angular_a = MakeVec3(0.0f, 0.0f, 0.0f);
    math::Vec3 angular_b = MakeVec3(0.0f, 0.0f, 0.0f);

    if (IsContactNormalRow(rows, row_index)) {
        error = fmaxf(material.position_error, 0.0f);
    } else if (material.kind == constraint::RowKind::Joint) {
        const uint32_t body_a = BodyForRowBody(rows, row, 0u);
        const uint32_t body_b = BodyForRowBody(rows, row, 1u);
        const auto jacobian_a = JacobianForRowBody(rows, row, 0u);
        const auto jacobian_b = JacobianForRowBody(rows, row, 1u);
        math::Vec3 axis = jacobian_b.linear;
        if (Length(axis) <= 1.0e-8f) {
            axis = Scale(jacobian_a.linear, -1.0f);
        }
        if (Length(axis) <= 1.0e-8f) {
            return 0.0f;
        }

        math::Vec3 position_a = MakeVec3(0.0f, 0.0f, 0.0f);
        math::Vec3 position_b = MakeVec3(0.0f, 0.0f, 0.0f);
        math::Quat orientation_a = MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
        math::Quat orientation_b = MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
        if (ValidBody(body_a, body_count)) {
            position_a = bodies[body_a].position;
            orientation_a = bodies[body_a].orientation;
        }
        if (ValidBody(body_b, body_count)) {
            position_b = bodies[body_b].position;
            orientation_b = bodies[body_b].orientation;
        }
        const auto& anchor = rows.anchors[row_index];
        const math::Vec3 r_a = RotateShort(orientation_a, anchor.local_a);
        const math::Vec3 r_b = RotateShort(orientation_b, anchor.local_b);
        error = Dot(Sub(Add(position_a, r_a), Add(position_b, r_b)), axis);
        angular_a = Scale(Cross(r_a, axis), -1.0f);
        angular_b = Cross(r_b, axis);
        apply_angular = true;
    } else {
        return 0.0f;
    }

    const float correction = baumgarte *
        (error >= 0.0f ? fmaxf(error - slop, 0.0f) : fminf(error + slop, 0.0f));
    if (fabsf(correction) <= 1.0e-8f) {
        return fabsf(error);
    }

    const float effective_mass = ComputeEffectiveMass(rows, row_index, bodies, body_count);
    const float position_impulse = correction * effective_mass;
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        if (!ValidBody(body_index, body_count)) {
            continue;
        }
        auto& body = bodies[body_index];
        if (body.inv_mass <= 0.0f) {
            if (apply_angular) {
                const math::Vec3 angular = local == 0u ? angular_a : angular_b;
                ApplyAngularPositionCorrection(body, angular, position_impulse);
            }
            continue;
        }
        const auto jacobian = JacobianForRowBody(rows, row, local);
        body.position = Add(body.position,
                            Scale(jacobian.linear,
                                  body.inv_mass * position_impulse));
        if (apply_angular) {
            const math::Vec3 angular = local == 0u ? angular_a : angular_b;
            ApplyAngularPositionCorrection(body, angular, position_impulse);
        }
    }

    return fabsf(error);
}

// =============================================================================
// G1(d) throughput increment: ONE CUDA BLOCK PER CONNECTED COMPONENT.
// =============================================================================
// The former kernel ran the whole row set on a SINGLE block (<<<1, 128>>>): all
// velocity_iterations x colors x rows serialized onto one SM, so the union-scene
// wall time grew linearly with env count (the measured flat ~50-74 env-steps/s).
// Rows in different CONNECTED COMPONENTS of the shared-mutable-state relation
// (bodies / qdot tiles / particle velocities / lambda groups -- see
// BuildRowComponentPartitions) touch disjoint memory, so their relative timing
// cannot change any float. This kernel therefore runs one block per component;
// each block walks ITS component's rows in ascending global-color order
// (segments), with __syncthreads() between segments -- the EXACT operation
// sequence the single-block sweep produced for that component's data (same ops,
// same order, same operands), executed concurrently across components. Per-env
// results are byte-identical BY CONSTRUCTION, for every caller (batched union
// world AND the legacy/N=1 consumers).
//
// max_error_out: the old kernel reported max(lane-0's running position-solve
// error, final-state recompute over all rows). Per-block we reduce ALL lanes'
// running errors (a superset of lane 0's -- still a lower-bound-honest max) and
// the final-state recompute over the block's rows, then combine across blocks
// with atomicMax on the float's bit pattern (exact for the non-negative finite
// errors produced here; max is order-independent, so D1 holds). No test or
// production caller consumes this value beyond >= 0.0f sanity.
__global__ void SolveRowsComponentSweepKernel(DeviceRowBuffers rows,
                                              DeviceRowComponentPartitions comps,
                                              runtime::rigid::BodyState* bodies,
                                              uint32_t body_count,
                                              uint32_t velocity_iterations,
                                              uint32_t position_iterations,
                                              float slop,
                                              float baumgarte,
                                              float dt,
                                              float* max_error_out) {
    const uint32_t comp_index = blockIdx.x;
    if (comp_index >= comps.component_count) {
        return;
    }
    const RowComponentRange comp = comps.components[comp_index];
    const uint32_t lane = threadIdx.x;

    for (uint32_t local = lane; local < comp.row_count; local += blockDim.x) {
        PrepareVelocityTargetRow(rows,
                                 comps.row_indices[comp.row_offset + local],
                                 bodies,
                                 body_count);
    }
    __syncthreads();

    for (uint32_t iter = 0u; iter < velocity_iterations; ++iter) {
        for (uint32_t seg = 0u; seg < comp.segment_count; ++seg) {
            const RowColorRange range = comps.segments[comp.segment_offset + seg];
            for (uint32_t local = lane;
                 local < range.row_count;
                 local += blockDim.x) {
                SolveVelocityRow(rows,
                                 comps.row_indices[range.row_offset + local],
                                 bodies,
                                 body_count,
                                 dt);
            }
            __syncthreads();
        }
    }

    float local_error = 0.0f;
    for (uint32_t iter = 0u; iter < position_iterations; ++iter) {
        for (uint32_t seg = 0u; seg < comp.segment_count; ++seg) {
            const RowColorRange range = comps.segments[comp.segment_offset + seg];
            for (uint32_t local = lane;
                 local < range.row_count;
                 local += blockDim.x) {
                local_error = fmaxf(
                    local_error,
                    SolvePositionRow(rows,
                                     comps.row_indices[range.row_offset + local],
                                     bodies,
                                     body_count,
                                     slop,
                                     baumgarte));
            }
            __syncthreads();
        }
    }

    __shared__ float error_lanes[kRowSolverBlockSize];
    error_lanes[lane] = local_error;
    __syncthreads();

    if (lane == 0u) {
        float max_error = 0.0f;
        for (uint32_t t = 0u; t < blockDim.x; ++t) {
            max_error = fmaxf(max_error, error_lanes[t]);
        }
        // Final-state recompute over THIS component's rows (the old kernel's
        // position_iterations-gated recompute loop body, verbatim; looping it
        // position_iterations times recomputed the identical max, so one pass
        // gated on position_iterations > 0 reproduces the same value).
        if (position_iterations > 0u) {
            for (uint32_t local = 0u; local < comp.row_count; ++local) {
                const uint32_t row_index =
                    comps.row_indices[comp.row_offset + local];
                const auto& row = rows.rows[row_index];
                const auto& material = rows.materials[row_index];
                if (IsContactNormalRow(rows, row_index)) {
                    max_error = fmaxf(max_error,
                                      fmaxf(material.position_error, 0.0f));
                } else if (material.kind == constraint::RowKind::Joint) {
                    const uint32_t body_a = BodyForRowBody(rows, row, 0u);
                    const uint32_t body_b = BodyForRowBody(rows, row, 1u);
                    const auto jacobian_a = JacobianForRowBody(rows, row, 0u);
                    const auto jacobian_b = JacobianForRowBody(rows, row, 1u);
                    math::Vec3 axis = jacobian_b.linear;
                    if (Length(axis) <= 1.0e-8f) {
                        axis = Scale(jacobian_a.linear, -1.0f);
                    }
                    if (Length(axis) <= 1.0e-8f) {
                        continue;
                    }
                    math::Vec3 position_a = MakeVec3(0.0f, 0.0f, 0.0f);
                    math::Vec3 position_b = MakeVec3(0.0f, 0.0f, 0.0f);
                    math::Quat orientation_a =
                        MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
                    math::Quat orientation_b =
                        MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
                    if (ValidBody(body_a, body_count)) {
                        position_a = bodies[body_a].position;
                        orientation_a = bodies[body_a].orientation;
                    }
                    if (ValidBody(body_b, body_count)) {
                        position_b = bodies[body_b].position;
                        orientation_b = bodies[body_b].orientation;
                    }
                    const auto& anchor = rows.anchors[row_index];
                    const math::Vec3 r_a =
                        RotateShort(orientation_a, anchor.local_a);
                    const math::Vec3 r_b =
                        RotateShort(orientation_b, anchor.local_b);
                    max_error = fmaxf(
                        max_error,
                        fabsf(Dot(Sub(Add(position_a, r_a),
                                     Add(position_b, r_b)),
                                  axis)));
                }
            }
        }
        // Non-negative finite floats order identically as their int bit
        // patterns -> atomicMax on the bits is an exact float max.
        atomicMax(reinterpret_cast<int*>(max_error_out),
                  __float_as_int(max_error));
    }
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) +
                                 " failed: " +
                                 cudaGetErrorString(result));
    }
}

// UploadVector now comes from the shared host buffer-transfer header
// (phi/buffer_transfer.hpp); the former local copy was byte-identical.
// (The separate UploadToScratch helper below is out of scope and stays local.)
using ::nuka::phi::UploadVector;

struct RowSolverScratch {
    int device_id = -1;
    phi::Buffer rows;
    phi::Buffer body_indices;
    phi::Buffer jacobians;
    phi::Buffer materials;
    phi::Buffer anchors;
    phi::Buffer comp_rows;
    phi::Buffer comp_segments;
    phi::Buffer comp_ranges;
    phi::Buffer bodies;
    phi::Buffer max_error;
    phi::Buffer sides;  // v0.8 C5a: per-row ContactRowSides (UnifiedSolve only)
    // v0.8 C5b-core: articulation reaction buffers (UnifiedSolve only).
    phi::Buffer art_refs;
    phi::Buffer chain_jacobians;
    phi::Buffer inertia_m_inv;
    phi::Buffer qdot;
    // v0.8 C6b: per-particle reaction SoA (UnifiedSolve coupling rows only).
    phi::Buffer particle_inv_mass;
    phi::Buffer particle_velocities;  // MUTABLE: downloaded after the solve
    size_t rows_bytes = 0u;
    size_t body_indices_bytes = 0u;
    size_t jacobians_bytes = 0u;
    size_t materials_bytes = 0u;
    size_t anchors_bytes = 0u;
    size_t comp_rows_bytes = 0u;
    size_t comp_segments_bytes = 0u;
    size_t comp_ranges_bytes = 0u;
    size_t bodies_bytes = 0u;
    size_t max_error_bytes = 0u;
    size_t sides_bytes = 0u;  // v0.8 C5a
    // v0.8 C5b-core
    size_t art_refs_bytes = 0u;
    size_t chain_jacobians_bytes = 0u;
    size_t inertia_m_inv_bytes = 0u;
    size_t qdot_bytes = 0u;
    // v0.8 C6b
    size_t particle_inv_mass_bytes = 0u;
    size_t particle_velocities_bytes = 0u;
};

RowSolverScratch& ThreadScratchForDevice(int device_id) {
    thread_local RowSolverScratch scratch;
    if (scratch.device_id != device_id) {
        scratch = RowSolverScratch{};
        scratch.device_id = device_id;
    }
    return scratch;
}

void EnsureScratchBuffer(phi::Buffer& buffer,
                         size_t& capacity_bytes,
                         size_t required_bytes) {
    if (capacity_bytes >= required_bytes) {
        return;
    }
    buffer = phi::Buffer(required_bytes, phi::MemoryKind::Device);
    capacity_bytes = required_bytes;
}

template <typename T>
void UploadToScratch(phi::Buffer& buffer,
                     size_t& capacity_bytes,
                     const std::vector<T>& values) {
    const size_t required_bytes = values.size() * sizeof(T);
    EnsureScratchBuffer(buffer, capacity_bytes, required_bytes);
    if (required_bytes > 0u) {
        buffer.CopyFromHost(values.data(), required_bytes);
    }
}

runtime::gpu::CudaConstraintRowSchedulerReport MakeSchedulerReport(
    uint32_t row_count,
    uint32_t color_count,
    const RowSolveConfig& config) {
    runtime::gpu::CudaConstraintRowBufferView view;
    view.kind = runtime::gpu::CudaConstraintRowBufferKind::UniversalRowCsr;
    view.layout = runtime::gpu::CudaConstraintRowLayout::UniversalRowCsr;
    view.schedule_mode =
        runtime::gpu::CudaConstraintRowScheduleMode::IslandColoredSweep;
    view.row_count = row_count;
    view.owner_count = color_count;
    view.rows_per_owner = 0u;
    view.row_stride_bytes = sizeof(constraint::Row);

    runtime::gpu::CudaConstraintRowSchedulerConfig scheduler_config;
    scheduler_config.iterations = config.velocity_iterations;
    return runtime::gpu::MakeCudaConstraintRowSchedulerReport(view,
                                                              scheduler_config);
}

} // namespace

RowSolveReport SolveRows(const phi::DeviceContext& context,
                         constraint::RowBuffers& rows,
                         std::vector<runtime::rigid::BodyState>& bodies,
                         const RowSolveConfig& config) {
    return SolveRows(context,
                     rows,
                     bodies.data(),
                     static_cast<uint32_t>(bodies.size()),
                     config);
}

RowSolveReport SolveRows(const phi::DeviceContext& context,
                         constraint::RowBuffers& rows,
                         runtime::rigid::BodyState* bodies,
                         uint32_t body_count,
                         const RowSolveConfig& config) {
    // v0.8 C5a: the legacy entry == the shared impl with NO sides. With sides ==
    // nullptr the kernel's compliant branch is never entered for a legacy row
    // (legacy rows carry no Compliant flag), so this is byte-identical.
    return SolveRowsWithSides(context, rows, bodies, body_count,
                              /*sides=*/nullptr, /*sides_count=*/0u, config);
}

RowSolveReport SolveRowsWithSides(const phi::DeviceContext& context,
                                  constraint::RowBuffers& rows,
                                  runtime::rigid::BodyState* bodies,
                                  uint32_t body_count,
                                  const constraint::ContactRowSides* sides,
                                  uint32_t sides_count,
                                  const RowSolveConfig& config,
                                  const RowArticulationData& articulation,
                                  const RowParticleData& particles) {
    RowSolveReport report;
    report.row_count = rows.RowCount();
    report.velocity_iterations = config.velocity_iterations;
    report.position_iterations = config.position_iterations;
    // v0.8 C5c-1: lift the body_count==0 early-return for the articulation-only
    // case (go2 foot-ground is link<->static: no rigid bodies, but the chain-J
    // reaction arm must still run). Proceed when there are rows AND at least one
    // reacting side class is present: rigid bodies OR articulation data. The rigid
    // apply path stays inert with body_count==0 because ValidBody(body, 0) is
    // always false, so the kernel never dereferences a null `bodies`. Legacy /
    // C5a / C5b rigid callers always pass body_count>0 + non-null bodies, so their
    // guard behaviour is unchanged (they never hit the new have_articulation leg).
    const bool have_rigid = body_count > 0u && bodies != nullptr;
    const bool have_articulation_early =
        articulation.art_refs != nullptr && articulation.art_refs_count > 0u &&
        articulation.dof_stride > 0u;
    // v0.8 C6b: a particle-only coupling solve (e.g. particle<->particle, or
    // particle<->static) has no rigid bodies and no articulation, but the particle
    // arm must still run. Legacy/C5a/C5b callers pass empty particles -> unchanged.
    const bool have_particles_early =
        particles.inv_mass != nullptr && particles.velocities != nullptr &&
        particles.count > 0u;
    if (rows.RowCount() == 0u ||
        (!have_rigid && !have_articulation_early && !have_particles_early)) {
        return report;
    }

    // G1(d) mechanism probe (env-gated, default OFF, zero hot-path cost): with
    // NUKA_ROW_SOLVER_TIMING set, prints the per-call host-schedule vs kernel
    // wall split + the launch geometry. This is the instrumentation that
    // confirmed the union-scene bottleneck was the SINGLE-BLOCK sweep kernel
    // (grid_blocks=1 at ~600 ms/call, N=32) and that measures the
    // one-block-per-component replacement.
    static const bool kTimingEnabled =
        std::getenv("NUKA_ROW_SOLVER_TIMING") != nullptr;
    const auto t_color_start = std::chrono::steady_clock::now();

    phi::ScopedDeviceGuard guard(context.device_id);
    const RowColorPartitions partitions = BuildRowColorPartitions(rows);
    if (!ValidateNoSharedBodiesPerColor(rows, partitions)) {
        throw std::runtime_error("row scheduler produced a conflicting color partition");
    }

    // G1(d) throughput increment: repartition the colored rows into connected
    // components of the shared-mutable-state relation so the kernel can run one
    // block per component (byte-identical schedule -- see row_scheduler.hpp).
    const RowComponentPartitions components = BuildRowComponentPartitions(
        rows, partitions, sides, sides_count, articulation.art_refs,
        articulation.art_refs_count);

    const auto t_color_end = std::chrono::steady_clock::now();

    report.color_count = partitions.ColorCount();
    report.row_scheduler_report =
        MakeSchedulerReport(rows.RowCount(), partitions.ColorCount(), config);

    auto& scratch = ThreadScratchForDevice(context.device_id);
    UploadToScratch(scratch.rows, scratch.rows_bytes, rows.rows);
    UploadToScratch(scratch.body_indices,
                    scratch.body_indices_bytes,
                    rows.body_indices);
    UploadToScratch(scratch.jacobians, scratch.jacobians_bytes, rows.jacobian_data);
    UploadToScratch(scratch.materials, scratch.materials_bytes, rows.materials);
    UploadToScratch(scratch.anchors, scratch.anchors_bytes, rows.anchors);
    UploadToScratch(scratch.comp_rows,
                    scratch.comp_rows_bytes,
                    components.row_indices);
    UploadToScratch(scratch.comp_segments,
                    scratch.comp_segments_bytes,
                    components.segments);
    UploadToScratch(scratch.comp_ranges,
                    scratch.comp_ranges_bytes,
                    components.components);
    // v0.8 C5c-1: only upload the rigid body SoA when there ARE rigid bodies
    // (body_count==0 + bodies==nullptr is the legitimate articulation-only case --
    // do not push a null pointer through a 0-byte memcpy).
    const size_t bodies_bytes =
        body_count * sizeof(runtime::rigid::BodyState);
    if (have_rigid) {
        EnsureScratchBuffer(scratch.bodies, scratch.bodies_bytes, bodies_bytes);
        scratch.bodies.CopyFromHost(bodies, bodies_bytes);
    }
    EnsureScratchBuffer(scratch.max_error,
                        scratch.max_error_bytes,
                        sizeof(float));
    float zero = 0.0f;
    scratch.max_error.CopyFromHost(&zero, sizeof(float));

    // v0.8 C5a: upload the per-row ContactRowSides stream (UnifiedSolve only).
    // Legacy SolveRows passes sides == nullptr -> nothing uploaded, device_view.
    // sides stays null, and the kernel never reads it for legacy rows.
    const bool have_sides = sides != nullptr && sides_count > 0u;
    if (have_sides) {
        const size_t sides_bytes =
            static_cast<size_t>(sides_count) * sizeof(constraint::ContactRowSides);
        EnsureScratchBuffer(scratch.sides, scratch.sides_bytes, sides_bytes);
        scratch.sides.CopyFromHost(sides, sides_bytes);
    }

    // v0.8 C5b-core: upload the articulation reaction buffers (UnifiedSolve only).
    // qdot is MUTABLE -- it is downloaded after the solve (the apply writes it).
    // Legacy / C5a callers pass an empty RowArticulationData -> nothing uploaded,
    // the device_view articulation pointers stay null, and the kernel never enters
    // the articulation arm (every side's art_index is the sentinel).
    const bool have_articulation =
        articulation.art_refs != nullptr && articulation.art_refs_count > 0u &&
        articulation.dof_stride > 0u;
    if (have_articulation) {
        const size_t art_refs_bytes = static_cast<size_t>(articulation.art_refs_count) *
                                      sizeof(constraint::RowArticulationRefs);
        EnsureScratchBuffer(scratch.art_refs, scratch.art_refs_bytes, art_refs_bytes);
        scratch.art_refs.CopyFromHost(articulation.art_refs, art_refs_bytes);
        if (articulation.chain_jacobians != nullptr &&
            articulation.chain_jacobians_count > 0u) {
            const size_t bytes =
                static_cast<size_t>(articulation.chain_jacobians_count) * sizeof(float);
            EnsureScratchBuffer(scratch.chain_jacobians,
                                scratch.chain_jacobians_bytes, bytes);
            scratch.chain_jacobians.CopyFromHost(articulation.chain_jacobians, bytes);
        }
        if (articulation.inertia_m_inv != nullptr &&
            articulation.inertia_m_inv_count > 0u) {
            const size_t bytes =
                static_cast<size_t>(articulation.inertia_m_inv_count) * sizeof(float);
            EnsureScratchBuffer(scratch.inertia_m_inv,
                                scratch.inertia_m_inv_bytes, bytes);
            scratch.inertia_m_inv.CopyFromHost(articulation.inertia_m_inv, bytes);
        }
        if (articulation.qdot != nullptr && articulation.qdot_count > 0u) {
            const size_t bytes =
                static_cast<size_t>(articulation.qdot_count) * sizeof(float);
            EnsureScratchBuffer(scratch.qdot, scratch.qdot_bytes, bytes);
            scratch.qdot.CopyFromHost(articulation.qdot, bytes);
        }
    }

    // v0.8 C6b: upload the per-particle reaction SoA (UnifiedSolve coupling rows
    // only). velocities is MUTABLE -- downloaded after the solve (the apply writes
    // it). Legacy / C5a / C5b callers pass empty particles -> nothing uploaded, the
    // device_view particle pointers stay null, and the particle arm never fires.
    const bool have_particles = have_particles_early;
    if (have_particles) {
        const size_t inv_mass_bytes =
            static_cast<size_t>(particles.count) * sizeof(float);
        EnsureScratchBuffer(scratch.particle_inv_mass,
                            scratch.particle_inv_mass_bytes, inv_mass_bytes);
        scratch.particle_inv_mass.CopyFromHost(particles.inv_mass, inv_mass_bytes);
        const size_t vel_bytes =
            static_cast<size_t>(particles.count) * sizeof(math::Vec3);
        EnsureScratchBuffer(scratch.particle_velocities,
                            scratch.particle_velocities_bytes, vel_bytes);
        scratch.particle_velocities.CopyFromHost(particles.velocities, vel_bytes);
    }

    DeviceRowBuffers device_view;
    device_view.rows = static_cast<constraint::Row*>(scratch.rows.Data());
    device_view.body_indices =
        static_cast<uint32_t*>(scratch.body_indices.Data());
    device_view.jacobian_data =
        static_cast<constraint::RowJacobian6*>(scratch.jacobians.Data());
    device_view.materials =
        static_cast<constraint::RowMaterial*>(scratch.materials.Data());
    device_view.anchors =
        static_cast<constraint::RowAnchor*>(scratch.anchors.Data());
    device_view.sides =
        have_sides
            ? static_cast<const constraint::ContactRowSides*>(scratch.sides.Data())
            : nullptr;  // v0.8 C5a
    device_view.sides_count = have_sides ? sides_count : 0u;
    // v0.8 C5b-core: articulation device pointers (null unless have_articulation).
    device_view.art_refs =
        have_articulation
            ? static_cast<const constraint::RowArticulationRefs*>(scratch.art_refs.Data())
            : nullptr;
    device_view.art_refs_count = have_articulation ? articulation.art_refs_count : 0u;
    device_view.chain_jacobians =
        (have_articulation && articulation.chain_jacobians != nullptr &&
         articulation.chain_jacobians_count > 0u)
            ? static_cast<const float*>(scratch.chain_jacobians.Data())
            : nullptr;
    device_view.inertia_m_inv =
        (have_articulation && articulation.inertia_m_inv != nullptr &&
         articulation.inertia_m_inv_count > 0u)
            ? static_cast<const float*>(scratch.inertia_m_inv.Data())
            : nullptr;
    device_view.qdot =
        (have_articulation && articulation.qdot != nullptr &&
         articulation.qdot_count > 0u)
            ? static_cast<float*>(scratch.qdot.Data())
            : nullptr;
    device_view.dof_stride = have_articulation ? articulation.dof_stride : 0u;
    // v0.8 C6b: particle device pointers (null unless have_particles).
    device_view.particle_inv_mass =
        have_particles ? static_cast<const float*>(scratch.particle_inv_mass.Data())
                       : nullptr;
    device_view.particle_velocities =
        have_particles
            ? static_cast<math::Vec3*>(scratch.particle_velocities.Data())
            : nullptr;
    device_view.particle_count = have_particles ? particles.count : 0u;
    device_view.row_count = rows.RowCount();
    device_view.body_index_count = rows.BodyIndexCount();
    device_view.jacobian_data_count = rows.JacobianDataCount();

    DeviceRowComponentPartitions device_components;
    device_components.row_indices =
        static_cast<const uint32_t*>(scratch.comp_rows.Data());
    device_components.segments =
        static_cast<const RowColorRange*>(scratch.comp_segments.Data());
    device_components.components =
        static_cast<const RowComponentRange*>(scratch.comp_ranges.Data());
    device_components.component_count = components.ComponentCount();

    const auto t_upload_end = std::chrono::steady_clock::now();

    constexpr uint32_t kBlockSize = kRowSolverBlockSize;
    const cudaStream_t stream = context.stream.Native();
    SolveRowsComponentSweepKernel<<<components.ComponentCount(), kBlockSize, 0,
                                    stream>>>(
        device_view,
        device_components,
        static_cast<runtime::rigid::BodyState*>(scratch.bodies.Data()),
        body_count,
        config.velocity_iterations,
        config.position_iterations,
        config.slop,
        config.baumgarte,
        config.dt,
        static_cast<float*>(scratch.max_error.Data()));
    CheckCuda(cudaGetLastError(), "SolveRowsComponentSweepKernel launch");
    ++report.row_scheduler_report.solver_launch_count;

    CheckCuda(cudaStreamSynchronize(stream), "RowSolver stream synchronize");

    if (kTimingEnabled) {
        const auto t_kernel_end = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::fprintf(stderr,
                     "[ROW-SOLVER-TIMING] rows=%u colors=%u grid_blocks=%u "
                     "block_threads=%u launches=1 coloring_ms=%.3f upload_ms=%.3f "
                     "kernel_ms=%.3f\n",
                     rows.RowCount(), partitions.ColorCount(),
                     components.ComponentCount(), kBlockSize,
                     ms(t_color_start, t_color_end), ms(t_color_end, t_upload_end),
                     ms(t_upload_end, t_kernel_end));
    }
    // v0.8 C5c-1: download the mutated rigid bodies only when they exist (the
    // articulation-only path has no rigid SoA to copy back).
    if (have_rigid) {
        scratch.bodies.CopyToHost(bodies, bodies_bytes);
    }
    scratch.rows.CopyToHost(rows.rows.data(),
                            rows.rows.size() * sizeof(constraint::Row));
    // v0.8 C5b-core: download the MUTATED articulation qdot (the apply wrote it).
    if (have_articulation && articulation.qdot != nullptr &&
        articulation.qdot_count > 0u) {
        scratch.qdot.CopyToHost(
            articulation.qdot,
            static_cast<size_t>(articulation.qdot_count) * sizeof(float));
    }
    // v0.8 C6b: download the MUTATED particle velocities (the apply wrote them).
    if (have_particles) {
        scratch.particle_velocities.CopyToHost(
            particles.velocities,
            static_cast<size_t>(particles.count) * sizeof(math::Vec3));
    }
    scratch.max_error.CopyToHost(&report.max_position_error, sizeof(float));
    report.row_scheduler_report.executed_iterations = config.velocity_iterations;
    report.row_scheduler_report.active_row_count = rows.RowCount();
    report.row_scheduler_report.normal_impulse_count = rows.RowCount();
    return report;
}

} // namespace nuka::solver::gpu
