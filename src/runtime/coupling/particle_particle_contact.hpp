#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::coupling::ParticleParticleContact -- the K3 cross-system
// particle-particle contact CO-STEP (v0.7 p11).
// ---------------------------------------------------------------------------
//
// The v0.7 INFRASTRUCTURE deliverable for cross-system coupling (master plan SS7
// lines 319-323/336/350: "a row with a Jacobian spanning two systems; the
// scheduler does not distinguish"). It couples the two PARTICLE subsystems already
// built -- XPBD soft bodies (src/runtime/soft/xpbd_world, ids 6-9) and PBF fluid
// (src/runtime/fluid/pbf_world) -- with a SINGLE class-blind unilateral
// non-penetration constraint (row class id 10, ParticleParticleContactRow). The
// row does NOT care which subsystem each particle belongs to, so this one co-step
// covers soft<->soft (incl. self-collision) and soft<->fluid ("towel absorbs
// water"). (K2 particle<->rigid-SDF coupling is a SEPARATE, DEFERRED v1.0 row.)
//
// CO-STEP SEMANTICS: this runs as an ADDITIVE coupling pass AFTER each subsystem's
// own internal solve, operating directly on both worlds' POSITION buffers. For
// each penetrating pair (i,j) with C = |p_i - p_j| - d_min < 0 and contact normal
// n = (p_i - p_j)/|p_i - p_j| it applies the mass-weighted symmetric position
// projection (the XPBD particle-collision constraint; a~ = 0, lambda warm-start 0):
//     dp_i = +(w_i/(w_i+w_j))*(-C)*n ,  dp_j = -(w_j/(w_i+w_j))*(-C)*n
// (w = inverse mass). This conserves the MASS-WEIGHTED centroid (m_i*dp_i +
// m_j*dp_j = 0 since m*w = 1) and is consistent with BOTH subsystems (both are
// position-based). p11 does NOT re-derive velocity from the correction (the
// position change propagates to velocity on the NEXT subsystem step's finalize);
// the co-step is a position-space pass. v1.0 wires the working-coupling backward.
//
// CROSS-SYSTEM INDEXING: positions + inverse masses are concatenated into a UNION
// scratch [ xpbd particles | pbf particles ]; a global index g < n_x refers to
// xpbd[g], else pbf[g - n_x]. The broadphase reuses the p05 cross-system uniform
// particle grid (BuildParticleUniformGrid) over the UNION; the grid's CSR neighbor
// lists are sorted ASCENDING by neighbor index (the D1 source). PBF inverse mass is
// the uniform 1/particle_mass (0 if mass is non-positive -> pinned).
//
// D1 STRONG DETERMINISM (HARD): a particle in MULTIPLE pairs accumulates the SUM of
// every pair's half-correction. We do this with the SAME Jacobi double-buffer
// pattern p10-A used for PBF -- NO float atomicAdd (master plan SS336-337). Each
// thread (one per union particle) gathers its OWN half over its ascending neighbor
// list into a per-particle delta scratch (fixed-order __fadd_rn reduction reading
// only the consistent pre-correction positions), THEN a separate own-index apply
// pass writes the corrected positions. The launch boundary between the two passes
// is the barrier that makes the gather race-free + D1 (a fused in-place update
// would have thread k read p_i while thread i writes it -- the exact race p10-A
// fixed). Two runs over N steps produce byte-identical positions + velocities.
//
// MOMENTUM CONSERVATION relies on SYMMETRIC neighbor detection (if i lists j then j
// lists i). The grid keeps each particle's 32 LOWEST-index neighbors; under
// truncation the relation can be asymmetric (j lists i but not vice versa), so j's
// half would apply without i's -- a momentum leak. The coupling step surfaces
// truncated_particle_count in its report; the oracles assert it is 0.
//
// INERT-WHEN-NO-PAIRS BYTE-IDENTITY: when the union grid reports zero neighbors
// within range, NO correction kernel launches (host-side early-out -- the same
// "skip the launch, don't scale-by-0" posture as the PBF polish passes). Running
// XPBD + PBF independently is then byte-identical to attaching this co-step.
//
// PLACEMENT: src/runtime/coupling/** is in the lint physics_path scope (the
// no-float-atomic ban is ENFORCED here) but NOT hot_path -- so the per-step union
// scratch + grid build are lint-allowed, the same perf posture as fluid (a named
// perf deferral for the 1M-particle pass). The generated row-class _forward.cu /
// _adjoint.cu (id 10) are the registry/dispatch + reverse-mode artifacts; the
// actual forward solve is the hand-written kernels in the .cu (exactly as the XPBD
// forward kernels coexist with the hand-written xpbd_world.cu sweeps).
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/fluid/pbf_world.hpp"
#include "runtime/soft/xpbd_world.hpp"

#include <cstdint>

namespace nuka::runtime::coupling {

// Fixed parameters of the K3 particle-particle contact co-step.
struct ParticleParticleContactParams {
    // Minimum contact distance d_min == the sum of the two contact radii. p11
    // treats every particle as carrying the SAME contact radius, so d_min is the
    // single uniform value (2 * contact_radius). A pair (i,j) is penetrating iff
    // |p_i - p_j| < d_min.
    //
    // DEFERRED (named v1.0 consumer = the working-coupling pass): PER-PARTICLE
    // contact radii (d_min_ij = r_i + r_j) for heterogeneous species. The
    // extension is trivial -- a per-union-particle radius array + d_min computed
    // in-kernel as r_i + r_j -- and matters most for soft<->fluid where the two
    // species' radii differ. v0.7 ships the uniform-radius infrastructure; the
    // per-particle radius rides the same v1.0 working-coupling deliverable that
    // wires the K2/K3 backward.
    float contact_distance_d_min = 0.1f;
    // XPBD compliance alpha (1/stiffness) for the contact projection; 0 == rigid
    // (the full -C correction). Kept for parity with the row's compliance support;
    // the position-based co-step uses alpha~ = compliance_alpha / dt^2.
    float compliance_alpha = 0.0f;
    // Coupling iterations per call (each is a full Jacobi gather+apply sweep over
    // the SAME per-call neighbor grid). >= 1.
    uint32_t solver_iterations = 1u;
};

// Report of one coupling co-step (diagnostics; the momentum-conservation oracle
// asserts truncated_particle_count == 0 so the symmetric-detection assumption
// holds).
struct ParticleParticleContactReport {
    uint32_t xpbd_particle_count = 0u;
    uint32_t pbf_particle_count = 0u;
    uint32_t union_particle_count = 0u;
    uint32_t total_candidate_neighbors = 0u;   // union grid TotalNeighbors().
    uint32_t truncated_particle_count = 0u;     // union grid TruncatedParticleCount().
    uint32_t kernel_launch_count = 0u;          // 0 when inert (no cross pairs).
    bool inert = false;                         // true => no correction kernel ran.
};

// Run the K3 cross-system particle-particle contact co-step over the attached XPBD
// soft world + PBF fluid world. Operates on both worlds' DEVICE position buffers
// in place. EITHER world may be empty (the union is then just the other's
// particles -- soft self-collision or fluid self-collision still apply). When the
// combined particle count is 0, or no pairs are within d_min, the call is INERT
// (no correction kernel launches; report.inert == true) and both worlds' state is
// byte-identical to not calling this.
ParticleParticleContactReport StepParticleParticleCoupling(
    const phi::DeviceContext& context,
    soft::XpbdWorld& xpbd_world,
    fluid::PbfWorld& pbf_world,
    const ParticleParticleContactParams& params);

ParticleParticleContactReport StepParticleParticleCoupling(
    soft::XpbdWorld& xpbd_world,
    fluid::PbfWorld& pbf_world,
    const ParticleParticleContactParams& params);

} // namespace nuka::runtime::coupling
