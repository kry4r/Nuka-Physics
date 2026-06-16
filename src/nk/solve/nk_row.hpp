#pragma once
// ---------------------------------------------------------------------------
// nk::NkRow — the union-family (CSR compliant) constraint-row record (M4).
//
// ONE NkRow per row slot of the `urows` Data field (32 f32 / 128 B; the field
// aliases this struct — static_assert'd). Written by the AssembleRows op,
// consumed by SolveRowsBlockIsland. The semantics transcribe the legacy
// unified row solve 1:1:
//   * rhs               == Row.rhs (aref, the solref/solimp reference accel)
//   * compliance_alpha  == Row.compliance_alpha (R, the dual regularizer)
//   * lower/upper       == Row bounds (normal: [0, FLT_MAX]; friction spokes
//                          get the in-kernel [0, mu*TotalNormalLambda] bound,
//                          exactly row_solver.cu IsCompliantFrictionRow)
//   * mu                == RowMaterial.friction (the per-class stamp)
//   * group_first/group_normal_count == RowMaterial group_id/normal_row_count
//                          re-keyed to FIXED worst-case row slots (inactive
//                          normal slots carry lambda == 0, so the
//                          TotalNormalLambda sum over the fixed span equals the
//                          legacy sum over the compacted emission)
//   * sides a/b         == ContactRowSides + RowArticulationRefs folded into
//                          one record: per-side reaction kind + index + the
//                          6-dof contact Jacobian (jlin = the emitted row
//                          direction; jang = the C5c r x jlin rigid augment,
//                          assembled once — the legacy kernel recomputed it
//                          per iteration from the SAME constant inputs).
//
// The per-row articulation chain Jacobian J and its PRECOMPUTED companion
// w = M^-1 J^T live in the chain_jacobian / row_minv_jt fields (row-major
// [slot * dof_stride + k]); the precomputed effective mass in row_meff.
//
// PURE C++ POD (no CUDA tokens) so the schedule builder, tests, and the CUDA
// ops all share the one definition.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "math/vec3.hpp"

namespace nuka::nk {

// Reaction-side kinds — the constraint::ReactionProviderKind dispatch values,
// mirrored as plain u32 codes (the row solver's side dispatch semantics).
inline constexpr uint32_t kNkSideRigid    = 0u;  // RigidInvMass
inline constexpr uint32_t kNkSideArtic    = 1u;  // ArticulationChainJ
inline constexpr uint32_t kNkSideParticle = 2u;  // ParticleInvMass
inline constexpr uint32_t kNkSideStatic   = 3u;  // StaticNull

namespace nk_row_flags {
inline constexpr uint32_t kActive   = 1u << 0;  // slot live this step (watermark)
inline constexpr uint32_t kFriction = 1u << 1;  // pyramid spoke (vs normal row)
}  // namespace nk_row_flags

// General contact pipeline (PairDriven family, Phase 1B): the FIXED per-candidate-
// slot row layout the assembly (EmitPairDrivenRowsKernel) emits and the cook sizes
// max_rows_per_env by. Each candidate slot expands to kPdPtsPerSlot manifold points
// (cvx caps at 4), each point -> 1 normal row + kPdSpokesPerPt friction spokes
// (+t0,-t0,+t1,-t1; condim-3 pyramid). rows_per_slot == pts*(1+spokes). Shared
// between the cook (host) and the assembly (device) so the schedule, the assembly,
// and the row budget all agree on the layout.
inline constexpr uint32_t kPairDrivenPtsPerSlot = 4u;
inline constexpr uint32_t kPairDrivenSpokesPerPt = 4u;
inline constexpr uint32_t kPairDrivenRowsPerSlot =
    kPairDrivenPtsPerSlot * (1u + kPairDrivenSpokesPerPt);  // 20

struct NkRowSide {
    uint32_t   kind  = kNkSideStatic;  // kNkSide* dispatch code
    uint32_t   index = ~0u;            // rigid: GLOBAL body row; artic: art index
    math::Vec3 jlin{};                 // contact Jacobian, linear part
    math::Vec3 jang{};                 // rigid: r x jlin (others: unused, 0)
};

struct NkRow {
    uint32_t flags = 0u;               // nk_row_flags
    uint32_t group_first = 0u;         // GLOBAL row slot of the group's normals
    uint32_t group_normal_count = 0u;  // worst-case normal slots in the group
    uint32_t env = 0u;                 // owning env
    float    rhs = 0.0f;               // aref
    float    compliance_alpha = 0.0f;  // R
    float    lower = 0.0f;
    float    upper = 0.0f;
    float    mu = 0.0f;                // friction coefficient (cone bound)
    float    reserved[7] = {};
    NkRowSide a;
    NkRowSide b;
};

static_assert(sizeof(NkRowSide) == 8 * sizeof(float),
              "NkRowSide must pack to 8 scalars");
static_assert(sizeof(NkRow) == 32 * sizeof(float),
              "urows field elem:32 must match NkRow");

}  // namespace nuka::nk
