#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::RowArticulationRefs -- per-row articulation side descriptors
// ---------------------------------------------------------------------------
// v0.8 C5b-core: the linchpin that lets the UnifiedSolve kernel resolve an
// ArticulationChainJ contact side (a reduced-coordinate Featherstone chain) the
// SAME way the rigid arm resolves a rigid body. C5a wired the rigid/static arms
// inline from the rigid BodyState SoA; the articulation arm needs DATA that does
// NOT live in BodyState -- the per-contact chain Jacobian row, the
// per-articulation M^-1 tile, and the live joint-velocity (qdot) slice. This
// header carries the per-row INDIRECTION into those global device buffers.
//
// Split out as its own tiny POD header (mirrors contact_row_sides.hpp, v0.8 C5a)
// so BOTH the GPU row solver (a C++17 CUDA TU) and the host UnifiedSolve adapter
// (which puts these in SolveContext) can consume it WITHOUT a heavy include. No
// dependency beyond <cstdint>.
//
// LAYOUT (global buffers, uniform dof_stride = max DOF; env-major, matching
// BatchedArticulatedWorld). For a row SIDE whose `art_index != kInvalidArtIndex`:
//   chain_jacobian row  -> chain_jacobians + chain_jac_slot * dof_stride   [dof_stride]
//   M^-1 tile           -> inertia_m_inv   + art_index    * dof_stride^2   [dof_stride^2]
//   live qdot slice     -> qdot            + art_index    * dof_stride     [dof_stride] (MUTABLE)
// `art_index` keys the per-articulation tiles (M^-1, qdot); `chain_jac_slot` keys
// the per-contact J row. They are DISTINCT: two contacts on the SAME articulation
// share one `art_index` but have DIFFERENT `chain_jac_slot`s.
//
// The sentinel `kInvalidArtIndex` (~0u) means "this side is NOT an articulation"
// (it is rigid / particle / static); the kernel then leaves the articulation arm
// inert for that side and resolves it via the C5a per-kind dispatch instead.
//
// Production never reads this (legacy SolveRows passes none -> the articulation
// arms never fire -> legacy solve byte-identical). For C5b-core the emitter that
// fills these is the TEST (built by hand); the foot-ground emitter that fills
// them in production is C5c.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::constraint {

// Sentinel art_index meaning "this side is not an articulation side".
inline constexpr uint32_t kInvalidArtIndex = ~0u;

// One contact side's articulation indirection. Defaults to the not-an-
// articulation sentinel so a side left unset is inert on the articulation arm.
struct RowArticulationSide {
    uint32_t art_index = kInvalidArtIndex;  // keys M^-1 tile + qdot slice
    uint32_t chain_jac_slot = 0u;           // keys the per-contact chain-J row
};

// Per-row pair of articulation side descriptors, in the SAME (a, b) order as
// ContactRowSides and the row's local body 0 / 1.
struct RowArticulationRefs {
    RowArticulationSide a;
    RowArticulationSide b;
};

}  // namespace nuka::constraint
