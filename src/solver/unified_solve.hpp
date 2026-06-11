#pragma once
// ---------------------------------------------------------------------------
// nuka::solver -- UnifiedSolve (v0.8 C5a). The class-blind contact-solve entry.
// ---------------------------------------------------------------------------
// VALIDATED, NOT WIRED. C5a is the first standalone-callable entry that drives
// the COMPLIANT contact path end-to-end on the GPU: it consumes the C4b
// EmitCompliantContactRows output (compliant normal rows carrying R/aref + the
// linearized pyramid friction spokes + the per-row ContactRowSides) and runs the
// (now compliant-aware) production graph-colored PGS kernel in row_solver.cu.
//
// It is a NEW entry alongside the legacy solver::SolveConstraints; it does NOT
// flip the production world stepper. The legacy GPU solve stays byte-identical
// (UnifiedSolve adds a separate code path; the legacy path passes sides=nullptr
// and the kernel never reads sides for a non-Compliant row).
//
// WHAT IS / ISN'T WIRED in C5a:
//   - The compliant EFFECTIVE MASS adds Row.compliance_alpha (= R, the dual
//     regularizer) to the summed two-sided J M^-1 J^T -- the headline C5a wiring.
//   - Per-side reaction dispatch is by ContactRowSides::{a,b}.react. The
//     RigidInvMass arm is CORRECT (built inline from the rigid BodyState, reusing
//     the C4c RigidEffectiveInvMass / RigidApplyImpulse math). The other arms
//     (ArticulationChainJ -> C5b, ParticleInvMass -> C6, StaticNull -> zero) are
//     dispatched but UNEXERCISED in C5a's all-rigid scope (see row_solver.cu).
//   - Friction uses the NEW unilateral coupled-pyramid bounds [0, mu*sum lambda_n]
//     (NOT the legacy bilateral override).
//   - Compliant rows SKIP Baumgarte position projection (they self-correct via
//     the aref velocity bias). UnifiedSolve is therefore VELOCITY-only for the
//     compliant rows it drives; the CALLER owns gravity + integration.
//
// DETERMINISM (D1): the graph-colored schedule now runs ONE CUDA BLOCK PER
// CONNECTED COMPONENT of the shared-state relation (G1d throughput increment,
// row_scheduler.hpp); per-component arithmetic is byte-identical to the former
// single-block sweep (same ops, same order, same operands), components share no
// memory, and the only cross-block combine is an order-independent atomicMax on
// the (unconsumed beyond >=0) max_position_error. Fixed SolverConfig iteration
// counts, fixed-order scans.
// ---------------------------------------------------------------------------

#include "constraint/reaction_provider.hpp"      // ReactionProvider
#include "constraint/row_articulation_refs.hpp"  // RowArticulationRefs (C5b)
#include "constraint/row_buffers.hpp"            // RowBuffers
#include "constraint/row_builder.hpp"            // ContactRowSides
#include "math/vec3.hpp"                         // math::Vec3 (C6b particles)
#include "runtime/rigid/body_state.hpp"          // BodyState
#include "solver/rigid_solver.hpp"               // SolverConfig

#include <vector>

namespace nuka::solver {

// Everything the unified solve needs for ONE velocity-solve pass over a set of
// (mostly compliant) contact rows. `rows`/`state` are mutated in place (the rows'
// lambda warm-start + the bodies' velocities). `sides` is the per-row side tag
// stream from EmitCompliantContactRows (same order as rows.rows); `reactions` is
// the §0.5 provider dispatcher (carried for the C5b/C6 device-resident wiring --
// C5a dispatches the RigidInvMass arm inline from `state`, so a default-empty
// provider is accepted). `dt` is the substep dt (informational for C5a).
// v0.8 C5b-core: the articulation reaction inputs (HOST-provided; UnifiedSolve
// uploads them and downloads `qdot` back -- it is mutated by the apply). All
// optional: a SolveContext with `art_refs == nullptr` (the default) drives the
// pure-rigid C5a path byte-identically (the articulation arms never fire). For
// C5b-core the data is built by HAND in the test; the foot-ground emitter fills
// it in production (C5c), which may also add a device-ptr fast path (the
// BatchedArticulatedWorld chain-J / M^-1 / qdot are already on the GPU, so an
// upload from host vectors is redundant there -- named C5c gap).
//   art_refs       : one RowArticulationRefs per row (same order as rows->rows).
//   chain_jacobians: flat per-(row,side) chain-J rows, dof_stride floats / slot.
//   inertia_m_inv  : flat per-articulation M^-1 tiles, dof_stride^2 floats / art.
//   qdot           : flat per-articulation joint velocities, dof_stride / art
//                    (MUTABLE -- the solved impulse writes qdot += M^-1 J^T dl;
//                    UnifiedSolve copies the result back into this vector).
//   dof_stride     : uniform max DOF (env-major, matching BatchedArticulatedWorld).
struct SolveArticulationContext {
    const std::vector<constraint::RowArticulationRefs>* art_refs = nullptr;
    const std::vector<float>* chain_jacobians = nullptr;
    const std::vector<float>* inertia_m_inv = nullptr;
    std::vector<float>* qdot = nullptr;  // MUTABLE: downloaded after the solve
    uint32_t dof_stride = 0u;
};

// v0.8 C6b: optional per-particle reaction state for a ParticleInvMass coupling
// side. A particle is resolved DIRECTLY by ContactRowSides::{a,b}.handle (==
// particle id) into these flat arrays -- no per-row indirection stream (unlike
// articulation). All optional: a SolveContext with `inv_mass == nullptr` (the
// default) drives the rigid/articulation path byte-identically (the particle arm
// never fires). The coupling framework (C6b) builds it by hand in the test; the
// PBD co-step (C6a) binds it to the live particle world velocity buffer.
//   inv_mass   : per particle, 1 float each (1/particle_mass).
//   velocities : per particle, 1 Vec3 each (MUTABLE -- the apply writes
//                v += inv_mass * J_linear * dlambda; downloaded after the solve).
struct SolveParticleContext {
    const std::vector<float>* inv_mass = nullptr;
    std::vector<math::Vec3>* velocities = nullptr;  // MUTABLE: downloaded
};

struct SolveContext {
    constraint::RowBuffers* rows = nullptr;
    std::vector<runtime::rigid::BodyState>* state = nullptr;
    const std::vector<constraint::ContactRowSides>* sides = nullptr;
    const constraint::ReactionProvider* reactions = nullptr;
    float dt = 0.0f;
    // v0.8 C5b-core: optional reduced-coordinate articulation reaction data.
    SolveArticulationContext articulation{};
    // v0.8 C6b: optional per-particle reaction data (cross-system coupling rows).
    SolveParticleContext particles{};
};

// Upload rows + bodies + sides -> launch the compliant-aware row_solver kernel ->
// download. A no-op if rows/state are empty or null.
void UnifiedSolve(const SolveContext& ctx, const solver::SolverConfig& config);

} // namespace nuka::solver
