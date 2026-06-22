#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::coupling -- the GENERIC cross-system coupling-row framework
// (v0.8 C6b, architecture 0.7). VALIDATED, NOT WIRED.
// ---------------------------------------------------------------------------
// The K2 framework, un-deferred: emit solver-resolved CONTACT rows for a pair
// whose two CollidableRef sides may be ANY two CollidableTypes (rigid /
// articulation-link / particle / static). This is the cross-system
// generalization that replaces Genesis's O(N^2) static per-pair coupling
// whitelist (research-genesis.md:96) and its conditionally-stable EXPLICIT
// -dmv/dt force (legacy_coupler.py:284-347) with ONE implicit COMPLIANT row
// carried by the unified regularized PGS (the (A+R) solve the R-sweep guard
// 0e39ace locked R-independent across the box/arm mass ratios coupling hits).
// The solref/solimp compliant schedule (C4a, 0.2) replaces Genesis's
// influence = exp(-d/eps) soft gate.
//
// ★ DESIGN: a coupling row IS a compliant contact row -- a unilateral
// non-penetration normal row + pyramid friction spokes, with a ContactRowSides
// tag carrying each side's CollidableRef so the unified solver dispatches the
// per-side ReactionProvider (RigidInvMass / ArticulationChainJ / ParticleInvMass
// / StaticNull) at solve time. So EmitCouplingRows REUSES EmitCompliantContactRows
// (C4b) verbatim and only RE-CLASSES the emitted rows to RowClassId
// CouplingContactRow (id13). The SOLVE dispatches on the Compliant flag + the
// per-side react, NOT on row_class_id, so the re-class is a pure classification
// label (the framework contract: cross-system coupling rows are id13, distinct
// from the same-system MaximalContactRow id0 and the particle-particle id10).
// This is the deliberate non-duplication: there is exactly ONE compliant-contact
// emission path, and coupling is a labeling of it.
//
// SHIPS the FRAMEWORK + the K2 particle<->rigid-SDF proving pair
// (test_coupling_framework). Specific production pairs (rigid<->MPM, cloth<->rigid)
// are v0.9 R8 -- each becomes a CandidatePair source feeding this same emitter,
// NOT a new hand-written O(N^2) kernel (the win over Genesis).
// Scope: ROW-coupled media only. Grid-transfer media (MPM) couple through a
// background grid as a SECOND CouplingProvider, not a contact-row source.
//
// Co-step ordering (Genesis simulator.py, research-genesis.md:18,99):
// preprocess -> pre_coupling (each system advances) -> couple (emit + solve these
// rows) -> post_coupling. v0.8 ships the framework; C6a wires the PBD co-step
// bridge (XPBD/PBF consume unified manifolds -> these rows).
// ---------------------------------------------------------------------------

#include "codegen/generated/row_class_registry.hpp"  // kCouplingContactRowId
#include "collision/candidate_pair.hpp"              // CandidatePair
#include "constraint/contact_manifold.hpp"           // ContactManifold
#include "constraint/row_builder.hpp"                // EmitCompliantContactRows
#include "constraint/contact_row_sides.hpp"          // ContactRowSides

#include <algorithm>
#include <span>
#include <vector>

namespace nuka::runtime::coupling {

// A cross-system coupling pair is structurally a CandidatePair (the C2 broadphase
// output): two CollidableRef sides (type/react/handle) + a canonical stable_key.
// The framework consumes the broadphase's candidate stream directly -- a coupling
// pair IS a candidate pair the narrowphase produced a manifold for.
using CouplingPair = collision::CandidatePair;

// Emit cross-system coupling rows for `pairs` (one manifold per pair, parallel
// order). Each manifold[i] must correspond to pairs[i] (the broadphase ->
// narrowphase 1:1 contract; the manifold ALSO carries the two CollidableRef sides
// the narrowphase stamped, which must agree with the pair's sides). For each
// manifold the framework emits compliant contact rows (normal + pyramid friction,
// C4b) + their ContactRowSides, then re-classes those rows to CouplingContactRow
// (id13). Empty / null out args -> no-op. `inputs` is the shared compliance
// context (vel/invweight/dt/condim/refsafe) -- the caller supplies the per-pair
// reaction inv-weight from the C4c providers (sum of both sides' J M^-1 J^T).
inline void EmitCouplingRows(std::span<const CouplingPair> pairs,
                             std::span<const constraint::ContactManifold> manifolds,
                             const constraint::ContactRowComplianceInputs& inputs,
                             constraint::RowBuffers* out_rows,
                             std::vector<constraint::ContactRowSides>* out_sides) {
    if (out_rows == nullptr || out_sides == nullptr) {
        return;
    }
    const size_t count = std::min(pairs.size(), manifolds.size());
    for (size_t i = 0; i < count; ++i) {
        const uint32_t row_begin = out_rows->RowCount();
        // Emit the compliant contact rows + sides for this one manifold (C4b). The
        // manifold's a/b CollidableRefs become the ContactRowSides, so the unified
        // solver resolves each side's ReactionProvider by side.react at solve time.
        constraint::EmitCompliantContactRows(
            std::span<const constraint::ContactManifold>(&manifolds[i], 1), inputs,
            out_rows, out_sides);
        // Re-class the just-emitted rows as the cross-system coupling class (id13).
        // Pure classification: the compliant solve reads the Compliant flag + the
        // per-side react, never row_class_id -> dynamics are identical to a
        // same-system compliant contact, only the registry label differs.
        for (uint32_t r = row_begin; r < out_rows->RowCount(); ++r) {
            out_rows->rows[r].row_class_id =
                solver::generated::kCouplingContactRowId;
        }
    }
}

} // namespace nuka::runtime::coupling
