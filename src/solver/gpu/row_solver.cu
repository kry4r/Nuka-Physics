// ---------------------------------------------------------------------------
// nuka::solver::gpu::row_solver implementation
// ---------------------------------------------------------------------------

#include "solver/gpu/row_solver.cuh"

#include "constraint/contact_row_sides.hpp"  // v0.8 C5a: ContactRowSides (POD)
#include "constraint/reaction_provider.hpp"  // v0.8 C5a: ReactionProvider math cores
#include "math/cuda_vec_ops.cuh"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"
#include "solver/gpu/row_scheduler.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

// One side's J M^-1 J^T contribution via the C4c provider math, dispatched by
// side.react. Returns the RAW additive diagonal term (NOT the reciprocal);
// the caller sums both sides + compliance_alpha and reciprocates once.
__device__ float CompliantSideEffectiveInvMass(const constraint::CollidableRef& side,
                                               const constraint::RowJacobian6& j,
                                               uint32_t body_index,
                                               const runtime::rigid::BodyState* bodies,
                                               uint32_t body_count) {
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
            // C5b GAP: the reduced-coordinate chain-J reaction needs the per-contact
            // chain Jacobian + M^-1 tile resolved at world build (not bodies[idx]).
            // Present for dispatch; unexercised in C5a (all-rigid box stack).
            return 0.0f;
        case constraint::ReactionProviderKind::ParticleInvMass:
            // C6 GAP: particle reaction needs the particle SoA inv_mass/velocity
            // views, not the rigid BodyState array. Present for dispatch; unexercised.
            return 0.0f;
        case constraint::ReactionProviderKind::StaticNull:
        default:
            return 0.0f;  // immovable world collider: zero reaction.
    }
}

// Apply a solved impulse delta to one side's DOFs via the C4c provider math,
// dispatched by side.react. Mutates bodies[idx] for the RigidInvMass arm.
__device__ void CompliantSideApplyImpulse(const constraint::CollidableRef& side,
                                          const constraint::RowJacobian6& j,
                                          float delta,
                                          uint32_t body_index,
                                          runtime::rigid::BodyState* bodies,
                                          uint32_t body_count) {
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
            // C5b GAP (see CompliantSideEffectiveInvMass). Present for dispatch.
            return;
        case constraint::ReactionProviderKind::ParticleInvMass:
            // C6 GAP (see CompliantSideEffectiveInvMass). Present for dispatch.
            return;
        case constraint::ReactionProviderKind::StaticNull:
        default:
            return;  // immovable: no-op.
    }
}

// Effective mass for a compliant row: 1 / (eff_inv_a + eff_inv_b + R), where R is
// Row.compliance_alpha (the dual regularizer -- the headline C5a wiring). Floored
// like the legacy ComputeEffectiveMass (>1e-12 -> reciprocal, else 0).
__device__ float ComputeCompliantEffectiveMass(const DeviceRowBuffers& rows,
                                               uint32_t row_index,
                                               const runtime::rigid::BodyState* bodies,
                                               uint32_t body_count) {
    const constraint::Row& row = rows.rows[row_index];
    float diagonal = 0.0f;
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const constraint::CollidableRef side =
            SideForRowBody(rows, row_index, local);
        const auto jacobian = JacobianForRowBody(rows, row, local);
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        diagonal += CompliantSideEffectiveInvMass(side, jacobian, body_index,
                                                  bodies, body_count);
    }
    diagonal += row.compliance_alpha;  // + R (dual regularizer)
    return diagonal > 1.0e-12f ? 1.0f / diagonal : 0.0f;
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
    for (uint32_t local = 0; local < row.body_count; ++local) {
        const constraint::CollidableRef side =
            SideForRowBody(rows, row_index, local);
        const auto jacobian = JacobianForRowBody(rows, row, local);
        const uint32_t body_index = BodyForRowBody(rows, row, local);
        CompliantSideApplyImpulse(side, jacobian, delta, body_index,
                                  bodies, body_count);
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
    const float jv = ComputeJv(rows, row_index, bodies, body_count);
    // C5a: row.rhs holds the MuJoCo reference ACCELERATION (aref). The velocity-
    // impulse PGS needs a velocity-target bias, so scale by dt: the impulse
    // lambda = f*dt satisfies (A+R)lambda = aref*dt - Jv (R is already in 1/mass
    // units, so it is NOT scaled -- it stays in the denominator as-is). Computed
    // locally (no buffer mutation) so the caller's rows are reusable.
    const float rhs_v = row.rhs * dt;
    const float lambda = effective_mass * (rhs_v - jv);
    const float old_impulse = row.lambda;
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

__global__ void SolveRowsSweepKernel(DeviceRowBuffers rows,
                                     DeviceRowColorPartitions partitions,
                                     runtime::rigid::BodyState* bodies,
                                     uint32_t body_count,
                                     uint32_t velocity_iterations,
                                     uint32_t position_iterations,
                                     float slop,
                                     float baumgarte,
                                     float dt,
                                     float* max_error_out) {
    const uint32_t lane = threadIdx.x;

    for (uint32_t row_index = lane;
         row_index < rows.row_count;
         row_index += blockDim.x) {
        PrepareVelocityTargetRow(rows, row_index, bodies, body_count);
    }
    __syncthreads();

    for (uint32_t iter = 0u; iter < velocity_iterations; ++iter) {
        for (uint32_t color = 0u; color < partitions.color_count; ++color) {
            const RowColorRange range = partitions.color_ranges[color];
            for (uint32_t local = lane;
                 local < range.row_count;
                 local += blockDim.x) {
                SolveVelocityRow(rows,
                                 partitions.row_indices[range.row_offset + local],
                                 bodies,
                                 body_count,
                                 dt);
            }
            __syncthreads();
        }
    }

    float local_error = 0.0f;
    for (uint32_t iter = 0u; iter < position_iterations; ++iter) {
        for (uint32_t color = 0u; color < partitions.color_count; ++color) {
            const RowColorRange range = partitions.color_ranges[color];
            for (uint32_t local = lane;
                 local < range.row_count;
                 local += blockDim.x) {
                local_error = fmaxf(
                    local_error,
                    SolvePositionRow(rows,
                                     partitions.row_indices[range.row_offset + local],
                                     bodies,
                                     body_count,
                                     slop,
                                     baumgarte));
            }
            __syncthreads();
        }
    }

    if (lane == 0u) {
        float max_error = local_error;
        for (uint32_t iter = 0u; iter < position_iterations; ++iter) {
            for (uint32_t color = 0u; color < partitions.color_count; ++color) {
                const RowColorRange range = partitions.color_ranges[color];
                for (uint32_t local = 0u; local < range.row_count; ++local) {
                    const uint32_t row_index =
                        partitions.row_indices[range.row_offset + local];
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
        }
        *max_error_out = max_error;
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
    phi::Buffer color_rows;
    phi::Buffer color_ranges;
    phi::Buffer bodies;
    phi::Buffer max_error;
    phi::Buffer sides;  // v0.8 C5a: per-row ContactRowSides (UnifiedSolve only)
    size_t rows_bytes = 0u;
    size_t body_indices_bytes = 0u;
    size_t jacobians_bytes = 0u;
    size_t materials_bytes = 0u;
    size_t anchors_bytes = 0u;
    size_t color_rows_bytes = 0u;
    size_t color_ranges_bytes = 0u;
    size_t bodies_bytes = 0u;
    size_t max_error_bytes = 0u;
    size_t sides_bytes = 0u;  // v0.8 C5a
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
                                  const RowSolveConfig& config) {
    RowSolveReport report;
    report.row_count = rows.RowCount();
    report.velocity_iterations = config.velocity_iterations;
    report.position_iterations = config.position_iterations;
    if (rows.RowCount() == 0u || body_count == 0u || bodies == nullptr) {
        return report;
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const RowColorPartitions partitions = BuildRowColorPartitions(rows);
    if (!ValidateNoSharedBodiesPerColor(rows, partitions)) {
        throw std::runtime_error("row scheduler produced a conflicting color partition");
    }

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
    UploadToScratch(scratch.color_rows,
                    scratch.color_rows_bytes,
                    partitions.row_indices);
    UploadToScratch(scratch.color_ranges,
                    scratch.color_ranges_bytes,
                    partitions.color_ranges);
    const size_t bodies_bytes =
        body_count * sizeof(runtime::rigid::BodyState);
    EnsureScratchBuffer(scratch.bodies, scratch.bodies_bytes, bodies_bytes);
    scratch.bodies.CopyFromHost(bodies, bodies_bytes);
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
    device_view.row_count = rows.RowCount();
    device_view.body_index_count = rows.BodyIndexCount();
    device_view.jacobian_data_count = rows.JacobianDataCount();

    DeviceRowColorPartitions device_partitions;
    device_partitions.row_indices =
        static_cast<const uint32_t*>(scratch.color_rows.Data());
    device_partitions.color_ranges =
        static_cast<const RowColorRange*>(scratch.color_ranges.Data());
    device_partitions.color_count = partitions.ColorCount();

    constexpr uint32_t kBlockSize = kRowSolverBlockSize;
    const cudaStream_t stream = context.stream.Native();
    SolveRowsSweepKernel<<<1u, kBlockSize, 0, stream>>>(
        device_view,
        device_partitions,
        static_cast<runtime::rigid::BodyState*>(scratch.bodies.Data()),
        body_count,
        config.velocity_iterations,
        config.position_iterations,
        config.slop,
        config.baumgarte,
        config.dt,
        static_cast<float*>(scratch.max_error.Data()));
    CheckCuda(cudaGetLastError(), "SolveRowsSweepKernel launch");
    ++report.row_scheduler_report.solver_launch_count;

    CheckCuda(cudaStreamSynchronize(stream), "RowSolver stream synchronize");
    scratch.bodies.CopyToHost(bodies, bodies_bytes);
    scratch.rows.CopyToHost(rows.rows.data(),
                            rows.rows.size() * sizeof(constraint::Row));
    scratch.max_error.CopyToHost(&report.max_position_error, sizeof(float));
    report.row_scheduler_report.executed_iterations = config.velocity_iterations;
    report.row_scheduler_report.active_row_count = rows.RowCount();
    report.row_scheduler_report.normal_impulse_count = rows.RowCount();
    return report;
}

} // namespace nuka::solver::gpu
