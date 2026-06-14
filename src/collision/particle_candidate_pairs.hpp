#pragma once
// ---------------------------------------------------------------------------
// nuka::collision -- particle<->rigid + particle<->particle candidate-pair
// builders, gated by the C1c SystemPairMatrix (v0.8 C2c)
// ---------------------------------------------------------------------------
// Two NEW, VALIDATED-NOT-FORCED broadphase paths that REUSE the existing query
// infra (cross_system_query.hpp particle->rigid LBVH query + particle_uniform_grid
// .hpp particle->particle grid) and reframe their CSR output into the C2a unified
// CandidatePairStream, gated by the SystemPairMatrix. These are NOT wired into the
// production world (the PBD co-step wiring is C6a; the legacy soft/fluid steppers
// were op-ified onto nk). EXCLUDED here: articulation-link<->anything (needs per-link AABB
// that does not yet exist -- the C5b-prereq long-pole).
//
// THE REFRAME (NO append atomic, NO scan):
//   Both source queries are already D1 CSR (count->scan->fill, lowest-index
//   truncation, integer-only). Because we reframe EVERY CSR entry (no filtering),
//   the CSR offsets ARE the deterministic output slots: entry k of particle p
//   writes to out[offsets[p] + k]. Every slot is written exactly once -> no scan,
//   no append/global atomic, no float atomic. We then download and call the C2a
//   BuildCandidatePairStream (stamps the canonical stable_key + thrust::stable_sort
//   -> byte-exact), then collapse adjacent identical pairs (same dedup as C2b).
//
// THE HANDLE MODEL (same as C2b, controller-ratified):
//   * particle side: handle = particle id (the CSR row index).
//   * rigid side:    handle = BODY id. The cross-system query's per-particle
//     candidate index is a RIGID SHAPE index (LBVH leaf / AABB-input order); we
//     map shape s -> rigid_shape_body_ids[s] to get the body handle. A MULTI-shape
//     body maps two shapes -> the SAME body -> the SAME body-pair candidate; the
//     post-sort adjacent dedup collapses those (C3 narrowphase re-expands shapes).
//   * particle<->particle: both sides are particle ids; canonicalized (min,max) in
//     the emit kernel so the grid's double-listing (i->j AND j->i) collapses to one
//     i<j pair after the sort+dedup.
//
// THE SYSTEMKIND GATE SEAM (documented):
//   v0.8 particles are ALL CollidableType::Particle (one collidable type). The
//   caller passes a SystemKind (Soft/Cloth/Fluid) PURELY so the SystemPairMatrix
//   gate is meaningful (e.g. disable Fluid<->Rigid). The per-particle SystemKind
//   tagging for MIXED particle systems is DEFERRED to the C6a co-step wiring; here
//   one builder call = one homogeneous particle system of a single SystemKind.
// ---------------------------------------------------------------------------

#include "collision/broadphase_lbvh.hpp"        // LbvhBroadphaseResult
#include "collision/candidate_pair.hpp"          // CandidatePairStream
#include "collision/particle_uniform_grid.hpp"   // ParticleGridConfig
#include "math/vec3.hpp"
#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>
#include "scene/cooked_blob.hpp"                  // SystemKind, SystemPairMatrix

#include <cstdint>

namespace nuka::collision {

using scene::SystemKind;
using scene::SystemPairMatrix;

// --- particle <-> rigid -----------------------------------------------------
// Query `particle_count` DEVICE-resident particle positions against the rigid
// LBVH `rigid_tree` (which MUST retain its node tree -- build it via
// BuildLbvhForQuery). Reframe each (particle p, rigid shape s) candidate into a
// CandidatePair{ a=(Particle, p), b=(RigidBody, rigid_shape_body_ids[s]) } and
// emit a sorted, deduplicated D1 CandidatePairStream.
//
//   particle_positions  : `particle_count` math::Vec3 in DEVICE memory.
//   particle_system     : the SystemKind of these particles (Soft/Cloth/Fluid),
//                         used only for the matrix gate (see seam doc above).
//   rigid_tree          : LbvhBroadphaseResult from BuildLbvhForQuery (HasNodes()).
//   rigid_shape_body_ids: HOST array, length = the rigid shape count fed to the
//                         LBVH (== rigid_tree.LeafCount()); maps shape -> body id.
//   query_radius        : per-particle AABB half-extent for the overlap query.
//   matrix              : C1c SystemPairMatrix; if !IsSystemPairEnabled(matrix,
//                         particle_system, Rigid) -> EMPTY stream (the gate).
//
// `particle_count == 0` or the gate-off -> empty stream (no kernel launch).
CandidatePairStream BuildParticleRigidCandidatePairs(
    cudaStream_t stream, int device_id,
    const math::Vec3* particle_positions, uint32_t particle_count,
    SystemKind particle_system,
    const gpu::LbvhBroadphaseResult& rigid_tree,
    const uint32_t* rigid_shape_body_ids,
    float query_radius,
    const SystemPairMatrix& matrix);

// Default-context overload.
CandidatePairStream BuildParticleRigidCandidatePairs(
    const math::Vec3* particle_positions, uint32_t particle_count,
    SystemKind particle_system,
    const gpu::LbvhBroadphaseResult& rigid_tree,
    const uint32_t* rigid_shape_body_ids,
    float query_radius,
    const SystemPairMatrix& matrix);

// --- particle <-> particle --------------------------------------------------
// Build the uniform grid over `particle_count` DEVICE-resident positions, then
// reframe each neighbor pair (i, j) into a canonical CandidatePair{ a=(Particle,
// min(i,j)), b=(Particle, max(i,j)) }. The grid double-lists (i->j AND j->i); the
// canonical emit + post-sort adjacent dedup collapses each to one i<j pair. Gated
// on IsSystemPairEnabled(matrix, particle_system, particle_system) (e.g. F<->F).
//
//   config       : ParticleGridConfig (cell_size >= query_radius -- see the grid
//                  header precondition; use MakeParticleGridConfig for the common
//                  case).
//   query_radius : neighbor search radius.
//
// `particle_count == 0` or gate-off -> empty stream (no kernel launch).
CandidatePairStream BuildParticleParticleCandidatePairs(
    cudaStream_t stream, int device_id,
    const math::Vec3* particle_positions, uint32_t particle_count,
    SystemKind particle_system,
    const gpu::ParticleGridConfig& config,
    float query_radius,
    const SystemPairMatrix& matrix);

// Default-context overload.
CandidatePairStream BuildParticleParticleCandidatePairs(
    const math::Vec3* particle_positions, uint32_t particle_count,
    SystemKind particle_system,
    const gpu::ParticleGridConfig& config,
    float query_radius,
    const SystemPairMatrix& matrix);

} // namespace nuka::collision
