#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::coupling -- the PBD<->rigid unified-contact CO-STEP bridge
// (v0.8 C6a). VALIDATED, NOT WIRED.
// ---------------------------------------------------------------------------
// The reusable HOST bridge that runs the Genesis-style "couple" phase for a
// homogeneous PBD particle system (XPBD soft/cloth, PBF fluid) against rigid
// bodies, through the ONE unified compliant contact solve -- but with the
// coupling done as a solver-resolved compliant ROW (the (A+R) PGS), NOT
// Genesis's explicit -dmv/dt force. It wires the REAL v0.8 front-end together:
//
//   pre_coupling (CALLER): predict particle positions (gravity*dt + integrate).
//   couple (THIS BRIDGE):
//     rigid LBVH over the box AABB(s)               (collision::BuildLbvhForQuery)
//       -> BuildParticleRigidCandidatePairs          (C2c front-end, upload pos)
//         -> DownloadPairs                            (host CandidatePair stream)
//           -> BuildContactManifolds (sphere x box)   (C5 contact_stream_driver)
//             -> EmitCouplingRows (re-class to id13)   (C6b coupling framework)
//               -> UnifiedSolve (state=bodies, particles=SoA, sides)  (C6b solver)
//   post_coupling (CALLER): integrate positions from the CORRECTED velocities.
//
// It MUTATES (two-way velocity correction):
//   - the host particle velocity SoA in place (UnifiedSolve downloads the solved
//     particle velocities back), and
//   - the host rigid BodyState vector in place (the box reaction).
// It does NOT touch positions (pre/post_coupling own integration) and is
// DECOUPLED from XpbdWorld/PbfWorld -- it takes raw host SoAs (positions /
// velocities / inv_masses), exactly what those worlds already expose, so the
// later production flip (C7) is a binding, not a redesign.
//
// SCOPE: particle<->RIGID only. particle<->particle is K3 (id10), untouched.
// The boxes resolve as analytical Box prims (sphere x box -> the analytical
// narrowphase tier); no SDF is cooked (the SDF tier is validated separately).
//
// HANDLE / BODY-INDEX CONTRACT (load-bearing -- see unified_costep.cpp):
//   * the particle side's CollidableRef.handle == the particle SoA index (the
//     UnifiedSolve ParticleInvMass arm indexes inv_mass/velocities by handle).
//   * a box's broadphase BODY id MUST equal its index in `bodies` (the rigid
//     arm indexes the BodyState vector by the row's body index, which the
//     emitter sets from the manifold's rigid-side handle). The bridge therefore
//     uses body id == box index 0..box_count-1 for the rigid AABBs.
// ---------------------------------------------------------------------------

#include "constraint/row_builder.hpp"            // ContactRowComplianceInputs
#include "math/vec3.hpp"
#include "phi/device_context.hpp"                // phi::DeviceContext
#include "runtime/rigid/body_state.hpp"          // BodyState
#include "scene/cooked_blob.hpp"                 // SystemKind
#include "solver/solver_config.hpp"              // SolverConfig

#include <cstdint>
#include <vector>

namespace nuka::runtime::coupling {

// What the couple phase did this step (diagnostics for the caller / the test).
struct UnifiedCoStepReport {
    uint32_t candidate_pairs = 0u;  // broadphase (particle, box) candidates found
    uint32_t manifolds       = 0u;  // narrowphase manifolds with >=1 contact point
    uint32_t rows            = 0u;  // compliant coupling rows emitted to the solve
    // # emitted rows whose ContactRowSides carries a ParticleInvMass side -- proves
    // the front-end tagged the particle side's reaction provider (so UnifiedSolve
    // dispatches the particle arm, not a vacuous rigid-only solve).
    uint32_t particle_arm_rows = 0u;
};

// One box collider for the couple phase: an axis-aligned cube (the bridge ships
// the analytical Box prim; non-axis-aligned boxes are a trivial frame extension
// the resolver could carry, deferred). `center` is its world COM; the matching
// BodyState in the caller's `bodies` vector (SAME index) carries inv_mass /
// velocity / inertia for the two-way reaction.
struct CoupleBox {
    math::Vec3 center{0.0f, 0.0f, 0.0f};
    float      half_extent = 0.5f;
};

// Run the COUPLE phase for a homogeneous particle system against rigid boxes.
//
//   context              : the device context (LBVH build / particle upload).
//   particle_positions   : host SoA, `particle_count` PREDICTED world positions
//                          (the caller's pre_coupling output) -- read only.
//   particle_velocities  : host SoA, `particle_count` velocities, MUTABLE -- the
//                          solver downloads the corrected particle velocities here.
//   particle_inv_mass    : host SoA, `particle_count` 1/mass (0 == pinned).
//   particle_count       : the particle SoA length.
//   particle_radius      : the per-particle contact sphere radius (the sphere
//                          prim radius + the broadphase query half-extent).
//   system               : the SystemKind (Soft/Cloth/Fluid) -- the matrix gate.
//   boxes                : the rigid box colliders (1:1 with `bodies` by index).
//   bodies               : host rigid BodyState vector, MUTABLE -- the box
//                          reaction is downloaded here (index i <-> boxes[i]).
//   compliance           : the shared C4a compliance context (vel/invweight/dt/
//                          condim/refsafe). `dt` should be the substep dt; set
//                          invweight ~ (inv_mass_particle + inv_mass_box) so the
//                          R/aref schedule is tuned for the actual masses.
//   dt                   : the substep dt (passed to UnifiedSolve, informational).
//   config               : the SolverConfig (velocity_iterations etc.). Use a
//                          MULTI-iteration config (the bridge rebuilds rows every
//                          step -> NO cross-step warm-start -> 1 iter under-
//                          resolves and particles tunnel).
//
// Empty inputs -> a zero report, no mutation. Deterministic (D1): the stream is
// thrust::stable_sort'd, the driver is a fixed sequential walk, the solver is the
// graph-colored PGS -- no new atomics.
UnifiedCoStepReport StepUnifiedParticleRigidCoupling(
    const phi::DeviceContext& context,
    const math::Vec3* particle_positions,
    std::vector<math::Vec3>* particle_velocities,
    const std::vector<float>& particle_inv_mass,
    uint32_t particle_count,
    float particle_radius,
    scene::SystemKind system,
    const std::vector<CoupleBox>& boxes,
    std::vector<runtime::rigid::BodyState>* bodies,
    const constraint::ContactRowComplianceInputs& compliance,
    float dt,
    const solver::SolverConfig& config);

}  // namespace nuka::runtime::coupling
