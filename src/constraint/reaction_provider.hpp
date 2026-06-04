#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint -- ReactionProvider (v0.8 C4c) -- impulse -> DOF delta
// ---------------------------------------------------------------------------
// VALIDATED, NOT WIRED. C4c implements the §0.5 reaction-provider interface:
// given a unit impulse along a row Jacobian on collidable `ref`, return that
// side's effective inverse mass (J M^-1 J^T) and apply a solved impulse to its
// DOFs. C5 (UnifiedSolve) consumes this so a single contact row between any two
// CollidableTypes resolves each side class-blind (the two-way reaction crux).
//
// THREE v0.8 reaction models, keyed by ReactionProviderKind (the §0.4 seam,
// read off ref.react which the broadphase copies from
// GetCollidableTypeInfo(type).react). Each MIRRORS an existing, already-D1
// production math path BYTE-FOR-BYTE so the C5 wiring is behavior-preserving:
//
//   RigidInvMass       -> row_solver.cu ComputeEffectiveMass / ApplyVelocityImpulse
//                         (maximal 6-DOF: invM = inv_mass I3 (+) inv_inertia
//                          diagonal applied to the world-frame angular Jacobian).
//   ArticulationChainJ -> articulation_contacts.cu ComputeContactEffectiveMass
//                         (m_eff = 1/(J M^-1 J^T)) + the SolveArticulatedContactRows
//                         qdot += M^-1 J^T dlambda apply loop. J is the PRE-BUILT
//                         dof-wide chain row from ComputeContactChainJacobians.
//   ParticleInvMass    -> the xpbd/pbf scalar inverse mass (particles are purely
//                         translational: invM = inv_mass |J_linear|^2).
//   StaticNull         -> immovable world collider: invM = 0, apply = no-op (so a
//                         link<->static or rigid<->static row has a well-defined
//                         zero-reaction side).
//
// DELIBERATE FIDELITY NOTE (rigid angular term). §C4c prose writes the rigid
// inverse mass as `R I_body^-1 R^T`, the general world-frame inverse inertia.
// The PRODUCTION maximal path (row_solver.cu) instead applies the stored Vec3
// `inv_inertia` componentwise against the world-frame angular Jacobian (a
// diagonal contraction; orientation never re-enters at the solve). C4c MUST
// match row_solver EXACTLY (behavior-preservation gate), so RigidReactionState
// carries the diagonal inv_inertia and contracts it diagonally -- NOT R I^-1 R^T.
// Promoting to the full tensor is a future row_solver change, out of C4c scope.
//
// HD-NESS. The MATH cores below are __host__ __device__ pure functions over
// plain-old-data binding structs (RigidReactionState / ArticulationReactionState
// / ParticleReactionState), so C5 can call them verbatim inside a device kernel.
// The ReactionProvider dispatcher that resolves a CollidableRef -> its binding
// is HD too; what it CANNOT be is constexpr-data -- the buffer views it closes
// over (body velocities, the per-articulation M^-1 tiles, the per-contact chain
// Jacobian rows, particle velocities) are resolved against real world buffers at
// WORLD BUILD. That is the §0.4 device-fn-ptr seam, mirrored from the collidable
// AABB-provider seam in collidable.hpp; see "C5 DEVICE-WIRING SEAM" at the end.
// ---------------------------------------------------------------------------

#include "constraint/collidable.hpp"  // CollidableRef, ReactionProviderKind
#include "constraint/row.hpp"         // RowJacobian6
#include "math/vec3.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_REACTION_HD __host__ __device__
#else
#define NUKA_REACTION_HD
#endif

namespace nuka::constraint {

// Denominator floor shared with the production paths. row_solver.cu treats a
// J M^-1 J^T below 1e-12 as "no reaction" (effective mass 0); the articulation
// path clamps the reciprocal at 1e-9. We expose BOTH so each reaction model
// reproduces its own production convention exactly (do not unify them).
inline constexpr float kRigidEffMassDenomEpsilon = 1.0e-12f;        // row_solver.cu:154
inline constexpr float kArticulationEffMassDenomEpsilon = 1.0e-9f;  // kEffectiveMassDenomEpsilon

// ===========================================================================
// RigidInvMass -- maximal 6-DOF rigid body (mirror of row_solver.cu)
// ===========================================================================
// One side's view of a single rigid body. `inv_mass` and `inv_inertia` mirror
// runtime::rigid::BodyState's fields verbatim (inv_inertia is the DIAGONAL,
// world-frame, applied componentwise -- see the fidelity note above). `velocity`
// /`angular_velocity` point at the body's live velocity so apply_impulse mutates
// it in place exactly as ApplyVelocityImpulse does. For effective_inv_mass only
// the scalars are read (the pointers may be null).
struct RigidReactionState {
    float inv_mass = 0.0f;
    math::Vec3 inv_inertia = math::Vec3::Zero();
    math::Vec3* velocity = nullptr;          // body.linear_velocity
    math::Vec3* angular_velocity = nullptr;  // body.angular_velocity
};

// effective_inv_mass = J M^-1 J^T for the maximal diagonal M^-1. BYTE-FOR-BYTE
// the per-body accumulation in row_solver.cu's ComputeEffectiveMass (:154):
//   diag += inv_mass * dot(Jlin,Jlin)
//         + Jang.x^2*invI.x + Jang.y^2*invI.y + Jang.z^2*invI.z
// NOTE: this returns the raw diagonal CONTRIBUTION (the J M^-1 J^T term), NOT
// the reciprocal. row_solver reciprocates the SUMMED diagonal over all of a
// row's bodies; C5 sums the two sides' contributions then reciprocates once, so
// each provider must hand back its additive term, not 1/term. (The standalone
// "effective mass" m_eff = 1/sum is formed by the caller; see the test.)
NUKA_REACTION_HD inline float RigidEffectiveInvMass(const RigidReactionState& s,
                                                    const RowJacobian6& j) {
    float diagonal = 0.0f;
    diagonal += s.inv_mass * j.linear.Dot(j.linear);
    diagonal += j.angular.x * j.angular.x * s.inv_inertia.x;
    diagonal += j.angular.y * j.angular.y * s.inv_inertia.y;
    diagonal += j.angular.z * j.angular.z * s.inv_inertia.z;
    return diagonal;
}

// apply_impulse: dv = dlambda * M^-1 J^T. BYTE-FOR-BYTE ApplyVelocityImpulse:
//   v   += Jlin * (inv_mass * dlambda)
//   w.k += Jang.k * invI.k * dlambda
// Static/immovable rigids (inv_mass <= 0) are skipped, mirroring row_solver.
NUKA_REACTION_HD inline void RigidApplyImpulse(const RigidReactionState& s,
                                               const RowJacobian6& j,
                                               float dlambda) {
    if (s.inv_mass <= 0.0f || s.velocity == nullptr || s.angular_velocity == nullptr) {
        return;
    }
    *s.velocity += j.linear * (s.inv_mass * dlambda);
    s.angular_velocity->x += j.angular.x * s.inv_inertia.x * dlambda;
    s.angular_velocity->y += j.angular.y * s.inv_inertia.y * dlambda;
    s.angular_velocity->z += j.angular.z * s.inv_inertia.z * dlambda;
}

// ===========================================================================
// ArticulationChainJ -- Featherstone chain (mirror of articulation_contacts.cu)
// ===========================================================================
// The articulation side does NOT operate on the raw RowJacobian6 {n, r x n}: the
// reduced-coordinate reaction needs the FULL dof-wide chain Jacobian J (the
// projection of the contact row onto every ancestor joint DOF), which the proven
// ComputeContactChainJacobians builder produces from the contact point + normal.
// THIS IS THE C4c ARBITRARY-CONTACT-POINT GENERALIZATION: the builder already
// takes a per-contact world point and forms lever = point - link_pose[link]
// for the contact link AND every ancestor, so an arbitrary manifold point (not
// just the foot-sphere center the foot-ground path fed it) "just works" -- C4c
// resolves `chain_jacobian` from any manifold point. (See the dedicated test.)
//
// `chain_jacobian` is the dof_stride-wide row for THIS contact; `inertia_M_inv`
// is THIS articulation's dof_stride x dof_stride symmetric inverse tile (both
// resolved at world build); `qdot` is the articulation's live joint-velocity
// slice apply_impulse mutates. Padding columns of J are zero (the builder's
// guarantee), so iterating the full stride is safe.
struct ArticulationReactionState {
    const float* chain_jacobian = nullptr;  // [dof_stride] this contact's J row
    const float* inertia_M_inv = nullptr;   // [dof_stride*dof_stride] M^-1 tile
    float* qdot = nullptr;                   // [dof_stride] live joint velocities
    uint32_t dof_stride = 0u;
};

// effective_inv_mass = J M^-1 J^T. BYTE-FOR-BYTE ComputeContactEffectiveMass's
// denom accumulation (articulation_contacts.cu:550-560): the SAME nested
// (for r: row = sum_c Minv[r*stride+c]*J[c]; denom += J[r]*row) order. Returns
// the raw J M^-1 J^T term (NOT 1/term) for the same two-sides-summed reason as
// RigidEffectiveInvMass; the production kernel reciprocates 1/max(denom,eps),
// which the standalone-m_eff helper / test reproduces.
NUKA_REACTION_HD inline float ArticulationEffectiveInvMass(
    const ArticulationReactionState& s) {
    if (s.chain_jacobian == nullptr || s.inertia_M_inv == nullptr) {
        return 0.0f;
    }
    const float* const Minv = s.inertia_M_inv;
    const float* const J = s.chain_jacobian;
    const uint32_t n = s.dof_stride;
    float denom = 0.0f;
    for (uint32_t r = 0u; r < n; ++r) {
        float row = 0.0f;
        for (uint32_t c = 0u; c < n; ++c) {
            row += Minv[static_cast<size_t>(r) * n + c] * J[c];
        }
        denom += J[r] * row;
    }
    return denom;
}

// apply_impulse: dqdot = dlambda * M^-1 J^T. BYTE-FOR-BYTE the apply loop in
// SolveArticulatedContactRowsKernel (articulation_contacts.cu ~858-867):
//   for r: acc = sum_c Minv[r*stride+c] * J[c]; qdot[r] += acc * dlambda
//
// ★ C5-WIRING WARNING (C4-review IMPORTANT-1) — flat-qdot vs scatter-by-link.
// This writes a CONTIGUOUS qdot[0..dof_stride). Production
// SolveArticulatedContactRowsKernel SCATTERS each logical DOF to either
// state.qdot[link] (scalar joints) OR link_velocity[root].v[component] (a
// FloatingBase base's 6 DOFs). For a FIXED-ROOT chain (v0.8's covered case) the
// logical-DOF order maps monotonically to consecutive qdot slots, so a private
// dof_stride-wide qdot slice is byte-faithful (padding cols are zeroed by the
// chain-J builder). BUT for a FLOATING-BASE articulation the base-DOF reactions
// belong in link_velocity, NOT qdot — a naive flat-qdot apply SILENTLY DROPS the
// contact's push on the floating base. C5 MUST route base DOFs to link_velocity
// when wiring floating-base articulations into the unified solve. (Named consumer
// = C5; do not assume flat-qdot for floating bases.)
NUKA_REACTION_HD inline void ArticulationApplyImpulse(
    const ArticulationReactionState& s, float dlambda) {
    if (s.chain_jacobian == nullptr || s.inertia_M_inv == nullptr ||
        s.qdot == nullptr) {
        return;
    }
    const float* const Minv = s.inertia_M_inv;
    const float* const J = s.chain_jacobian;
    const uint32_t n = s.dof_stride;
    for (uint32_t r = 0u; r < n; ++r) {
        float acc = 0.0f;
        const float* const minv_row = Minv + static_cast<size_t>(r) * n;
        for (uint32_t c = 0u; c < n; ++c) {
            acc += minv_row[c] * J[c];
        }
        s.qdot[r] += acc * dlambda;
    }
}

// ===========================================================================
// ParticleInvMass -- translational point mass (mirror of the xpbd/pbf path)
// ===========================================================================
// A particle has only a scalar inverse mass and a linear velocity -- no angular
// DOF. `inv_mass` mirrors CudaParticleSet::inv_masses[i]; `velocity` points at
// CudaParticleSet::velocities[i]. The row's angular Jacobian is IGNORED (a
// particle cannot absorb torque).
struct ParticleReactionState {
    float inv_mass = 0.0f;
    math::Vec3* velocity = nullptr;  // particle linear velocity
};

// effective_inv_mass = inv_mass * |J_linear|^2 (J M^-1 J^T for M^-1 = inv_mass I3).
NUKA_REACTION_HD inline float ParticleEffectiveInvMass(const ParticleReactionState& s,
                                                       const RowJacobian6& j) {
    return s.inv_mass * j.linear.Dot(j.linear);
}

// apply_impulse: dv = dlambda * inv_mass * J_linear.
NUKA_REACTION_HD inline void ParticleApplyImpulse(const ParticleReactionState& s,
                                                  const RowJacobian6& j,
                                                  float dlambda) {
    if (s.inv_mass <= 0.0f || s.velocity == nullptr) {
        return;
    }
    *s.velocity += j.linear * (s.inv_mass * dlambda);
}

// ===========================================================================
// StaticNull -- immovable world collider
// ===========================================================================
// invM = 0, apply = no-op. No state. Provided as free helpers so the dispatch
// has a uniform shape across all four kinds.
NUKA_REACTION_HD inline float StaticEffectiveInvMass() { return 0.0f; }
NUKA_REACTION_HD inline void StaticApplyImpulse() { /* immovable: no-op */ }

// ===========================================================================
// §0.5 INTERFACE -- ReactionProvider dispatcher (keyed by ref.react, the §0.4 seam)
// ===========================================================================
// One object resolving the four reaction models. It does NOT own per-instance
// state; it carries the per-type RESOLVED buffer views (filled at world build,
// the seam) and, given a CollidableRef, gathers that instance's binding and
// dispatches on ref.react. For C4c (validated-not-wired) the binding-resolution
// is left to the caller via the explicit overloads below + the per-kind cores
// above (so the host test drives each path directly); C5 fills the views and
// uses the dispatch. The dispatch SIGNATURE matches §0.5 exactly:
//   float effective_inv_mass(const CollidableRef&, const RowJacobian6&) const;
//   void  apply_impulse(const CollidableRef&, const RowJacobian6&, float) const;
//
// The bindings a built world resolves per CollidableType. RigidInvMass /
// ParticleInvMass index by ref.handle into a flat per-instance array; the
// articulation binding additionally needs the per-contact chain-J row and the
// per-articulation M^-1 tile, so C5 resolves it from the contact (not ref.handle
// alone) -- documented in the seam at the bottom. For C4c these stay simple
// pointers the test/world-build fills.
struct ReactionProviderViews {
    // RigidInvMass: per-rigid-body state, indexed by ref.handle.
    RigidReactionState* rigid = nullptr;
    uint32_t rigid_count = 0u;
    // ParticleInvMass: per-particle state, indexed by ref.handle.
    ParticleReactionState* particle = nullptr;
    uint32_t particle_count = 0u;
    // ArticulationChainJ: per-link (ref.handle == global link) resolved binding.
    // C5 fills chain_jacobian / inertia_M_inv / qdot / dof_stride per CONTACT
    // (the chain-J row is per-contact, not per-link), so this is an array the
    // contact-assembly indexes; for C4c the test supplies one directly.
    ArticulationReactionState* articulation = nullptr;
    uint32_t articulation_count = 0u;
};

struct ReactionProvider {
    ReactionProviderViews views{};

    // --- §0.5 dispatch: effective inverse mass (the J M^-1 J^T term) ---------
    NUKA_REACTION_HD float effective_inv_mass(const CollidableRef& ref,
                                              const RowJacobian6& j) const {
        switch (ref.react) {
            case ReactionProviderKind::RigidInvMass:
                if (views.rigid != nullptr && ref.handle < views.rigid_count) {
                    return RigidEffectiveInvMass(views.rigid[ref.handle], j);
                }
                return 0.0f;
            case ReactionProviderKind::ArticulationChainJ:
                if (views.articulation != nullptr &&
                    ref.handle < views.articulation_count) {
                    return ArticulationEffectiveInvMass(views.articulation[ref.handle]);
                }
                return 0.0f;
            case ReactionProviderKind::ParticleInvMass:
                if (views.particle != nullptr && ref.handle < views.particle_count) {
                    return ParticleEffectiveInvMass(views.particle[ref.handle], j);
                }
                return 0.0f;
            case ReactionProviderKind::StaticNull:
            default:
                return StaticEffectiveInvMass();
        }
    }

    // --- §0.5 dispatch: apply a solved impulse to the side's DOFs ------------
    NUKA_REACTION_HD void apply_impulse(const CollidableRef& ref,
                                        const RowJacobian6& j,
                                        float dlambda) const {
        switch (ref.react) {
            case ReactionProviderKind::RigidInvMass:
                if (views.rigid != nullptr && ref.handle < views.rigid_count) {
                    RigidApplyImpulse(views.rigid[ref.handle], j, dlambda);
                }
                return;
            case ReactionProviderKind::ArticulationChainJ:
                if (views.articulation != nullptr &&
                    ref.handle < views.articulation_count) {
                    ArticulationApplyImpulse(views.articulation[ref.handle], dlambda);
                }
                return;
            case ReactionProviderKind::ParticleInvMass:
                if (views.particle != nullptr && ref.handle < views.particle_count) {
                    ParticleApplyImpulse(views.particle[ref.handle], j, dlambda);
                }
                return;
            case ReactionProviderKind::StaticNull:
            default:
                StaticApplyImpulse();
                return;
        }
    }
};

// ---------------------------------------------------------------------------
// C5 DEVICE-WIRING SEAM (how the providers become device-callable at world build)
// ---------------------------------------------------------------------------
// MIRRORS the collidable AABB-provider seam (collidable.hpp §0.4): the per-kind
// MATH above is fixed device code; only the BUFFER VIEWS are runtime. At world
// build C5 fills a ReactionProviderViews whose pointers reference the SAME device
// buffers the production paths already own:
//   - rigid       -> a device array of RigidReactionState built from the rigid
//                    BodyState SoA (inv_mass, inv_inertia, &linear_velocity[i],
//                    &angular_velocity[i]); ref.handle == maximal body id.
//   - particle    -> a device array of ParticleReactionState from the particle
//                    world (inv_masses[i], &velocities[i]); ref.handle == particle id.
//   - articulation-> resolved PER CONTACT during row assembly: ComputeContactChain
//                    Jacobians fills chain_jacobian (any manifold point -- the C4c
//                    generalization), FactorArticulationInertiaM the M^-1 tile,
//                    state.qdot the joint-velocity slice. ref.handle == global link.
// Because ReactionProvider holds raw pointers + a switch (no virtuals, no host
// std::function), the SAME object is usable inside a __global__: pass it by value,
// the dispatch compiles to a device switch over the 4 kinds. C5 wires this into
// UnifiedSolve; C4c only validates the math + dispatch standalone.
// ---------------------------------------------------------------------------

} // namespace nuka::constraint

#undef NUKA_REACTION_HD
