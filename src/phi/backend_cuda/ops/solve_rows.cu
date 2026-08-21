// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — SolveRowsBlockIsland (the spec device-resident
// unified row solve). TWO contact families behind the ONE op (params->family):
//
// (: the legacy FUSED block-per-articulation PGS path was DELETED here —
// `SolveArticulatedContactRowsKernel` + its dispatch branch are gone. The ONE
// general island solver below now serves both remaining families.)
//
// UNION (kContactFamilyUnionCsr) — THE the spec KERNEL: grid = total_islands
// (= N_env x islands/env from the build-time SolveSchedule), block = 256;
// per block:
// for it in vel_iters:
// for c in colors(island): // device-resident segment table
// rows of color c in parallel (thread/row) -> __syncthreads
// Per-row math = row_solver.cu's COMPLIANT branch, preserved numerically:
// * jv = per-side dispatch (CompliantSideConstraintVelocity):
// rigid: dot(jlin,v)+dot(jang,w); artic: serial
// sum_r J[r]*qdot[r] (ascending r, the legacy order);
// particle: dot(jlin,v); static: 0. Side a then b.
// * update = lambda_new = clamp(lambda + meff*(rhs*dt - jv
// - R*lambda), bounds) — incl. the C5c-2 regularizer
// feedback (-R*lambda) the legacy kernel carries.
// * friction = the unilateral coupled-pyramid bound
// [0, mu * TotalNormalLambda(group)] (IsCompliantFriction-
// Row semantics; the group sum over the FIXED normal slots
// equals the legacy compacted sum — inactive slots carry
// lambda == 0).
// * apply = per-side dispatch: rigid v += jlin*(im*dl), w += jang*
// invI*dl (skip im<=0); artic qdot[r] += w[r]*dl with the
// PRECOMPUTED w = M^-1 J^T (the ArticulationApplyImpulse
// inner product hoisted to AssembleRows — same products,
// same order); particle v += jlin*(im*dl). |dl| > 1e-12
// gate, side a then b. meff is the assembly-hoisted
// ComputeCompliantEffectiveMass (constant across iters).
// The articulation qdot tile lives in SHARED memory for the island's env
// (loaded from qdot_flat before the sweep, scattered back to link_velocity /
// qdot through the cooked dof maps after — the legacy pack/scatter 1:1; a
// contact-free env scatters its own unchanged values, a no-op). 51-DOF
// M^-1 J^T is the per-thread register loop over the row's coalesced
// chain_jacobian / row_minv_jt segments (fields.yaml layout note).
// Watermark early-exit: an inactive row slot (flags bit0 clear) returns
// immediately — max grid + early exit keeps the kernel graph-capturable
// (the design capacity policy).
// pos_iters is accepted but unused: every union row is COMPLIANT, and the
// legacy compliant path skips Baumgarte position projection entirely
// (SolvePositionRow returns 0 for Compliant rows; the legacy union runs
// position_iterations = 0) — preserved semantics, not an omission.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

namespace mg = ::nuka::math::gpu;

__forceinline__ __device__ float Dot3(math::Vec3 a, math::Vec3 b) {
    return mg::Dot(a, b);
}

// ===========================================================================
// UNION-family island kernel (NEW — the spec kernel).
// ===========================================================================

// Island record: the schedule's per-island quad. The flat schedule array is a
// run of these (schedule.cpp builds it as raw {seg_off, seg_cnt, flags, env}
// stride-4 u32). FLAG: schedule.cpp:256-323 + fields.yaml encode the same quad
// with raw `* 4u + k` arithmetic — they should adopt THIS struct so a field
// addition is one edit, not a silent stride drift across three sites.
struct IslandRecord {
    uint32_t seg_off;   // first color segment of this island
    uint32_t seg_cnt;   // color segment count
    uint32_t flags;     // bit0 == has-articulation
    uint32_t env;       // owning env
};
static_assert(sizeof(IslandRecord) == 4u * sizeof(uint32_t),
              "IslandRecord must pack exactly the stride-4 schedule quad");

// Union's cooked color segments can expose two independent rows at once, so it
// keeps two warps. Dynamic PairDriven islands deliberately preserve serial GS by
// giving each active row its own segment; a second warp can never receive work and
// only doubles the surplus-block traffic. One warp also lets the per-row ordering
// fence use __syncwarp instead of a CTA barrier without changing any arithmetic.
constexpr uint32_t kUnionIslandBlockSize = 64u;
constexpr uint32_t kPairDrivenIslandBlockSize = 32u;
constexpr uint32_t kScalarIslandBlockSize = 32u;
constexpr uint32_t kScalarIslandGridBlocks = 64u;

// Slim per-row SOLVE record (16 f32 = 64 B), built IN-KERNEL at launch from
// the NkRow records. Rationale (MEASURED): the sweep re-reads its row record
// every iteration; broadcasting the full 128 B NkRow to every lane spilled to
// local memory and dominated the bookkeeping cost. The slim record carries
// exactly what the velocity update needs: flags / group (re-based env-LOCAL) /
// compliance terms / the side dispatch folded to {artic bits + at most ONE
// dynamic (rigid|particle) side}. A row with TWO dynamic sides (the
// particle x particle class — never emitted by the assembly) sets the
// FALLBACK bit and the row solve reads the full NkRow from global (slow but
// correct path; no silent drop).
struct SlimRow {
    uint32_t flags;
    uint32_t group_local;   // group_first - env_row_base
    uint32_t group_cnt;
    uint32_t code;          // kSlim* bits
    uint32_t dyn_index;
    float rhs, R, lower, upper, mu;
    float jl[3];            // the dynamic side's linear jacobian
    float ja[3];            // ... angular (rigid r x j augment)
    // the ENV-LOCAL articulation tile
    // index for each artic side (global_artic - env*artics_per_env) so the row
    // reads/writes the CORRECT per-articulation qdot tile in shared memory. At
    // K==1 both collapse to 0 (one tile/env) -> byte-identical to the single-tile
    // path. ~0u sentinel when the side is not articulation.
    uint32_t a_tile;        // side-A artic env-local tile index (or ~0u).
    uint32_t b_tile;        // side-B artic env-local tile index (or ~0u).
};
static_assert(sizeof(SlimRow) == 18 * sizeof(float), "SlimRow must be 72 B");
constexpr uint32_t kSlimAArt = 1u << 0;
constexpr uint32_t kSlimBArt = 1u << 1;
constexpr uint32_t kSlimHasDyn = 1u << 2;
constexpr uint32_t kSlimDynIsB = 1u << 3;
constexpr uint32_t kSlimDynParticle = 1u << 4;
constexpr uint32_t kSlimFallback = 1u << 5;  // two dynamic sides: read NkRow

// The island kernel's DYNAMIC shared working-set size for a given env slice
// (see the kernel's layout carve). Union: qdot tile + lambda + meff + slim records
// + J + w (R x D each) + order + segments. PairDriven: ONLY the qdot tile (every
// row-scaled array lives in global). The 51-DOF / 190-row union scene
// needs ~92 KB — within sm_8x+'s opt-in dynamic shared (set once via
// cudaFuncAttributeMaxDynamicSharedMemorySize). A model whose slice exceeds
// the device limit fails LOUDLY at the op entry (Model capacity property;
// the broadphase-cooked sizing revisits it).
// the qdot region holds ALL of the env's articulation tiles
// (qdot_tiles * dof_stride floats) so a PairDriven mixed island can service every
// co-resident articulation it touches. For the Union/Fused single-artic families
// qdot_tiles == kMaxArticulationDof / dof_stride is unused; they pass
// qdot_tiles == 1 BUT the legacy carve reserved kMaxArticulationDof floats — so
// to keep the union footprint UNCHANGED (the H1 golden's measured ~92 KB) the op
// passes the legacy reservation for the non-PairDriven families (with_b_arm == 0)
// and the compact K-tile reservation for PairDriven. The B-arm J_b/w_b region
// is present ONLY when with_b_arm (the PairDriven family) — the union path
// never allocates it, so its shared footprint is byte-for-byte the legacy carve.
// cache_jw: the Union family caches the per-row chain-J (J) + M^-1 J^T (w) rows in
// SHARED (the measured latency core of its dense 190-row sweep). The PairDriven
// family does NOT (with_b_arm == true): its worst-case per-env island can hold
// HUNDREDS of rows x ~18 DOF x 4 arrays (J,w,J_b,w_b) which would blow the ~99 KB
// dynamic-shared limit, so it reads J/w/J_b/w_b straight from GLOBAL each iteration
// (bounded shared; correctness-first). The shared carve excludes those regions.
// The PairDriven family ALSO keeps lambda / meff / order / segments in GLOBAL (all
// scale with rows_per_env); only the qdot tile region stays in shared. The Union
// family stages every row-scaled array in shared (its carve is unchanged).
inline size_t IslandSharedBytes(uint32_t rows_per_env, uint32_t dof_stride,
                                uint32_t qdot_floats, bool cache_jw,
                                bool pos_pass, uint32_t k_tiles) {
    if (!cache_jw) {
        // PairDriven: only the qdot tile(s) live in shared; lambda/meff/order/seg
        // are read from GLOBAL, so the carve does not scale with rows_per_env. The
        // split-impulse position pass adds a parallel pseudo qdot tile (same size).
        // The dynamic-island path also carves a per-component tile present/list pair
        // (k_tiles u32 each) so the load/scatter touch ONLY this island's tiles.
        return sizeof(float) * qdot_floats * (pos_pass ? 2u : 1u) +
               sizeof(uint32_t) * 2ull * k_tiles;
    }
    const uint64_t jw = 2ull * rows_per_env * dof_stride;
    const uint64_t slim = sizeof(SlimRow) * rows_per_env;
    return sizeof(float) *
               (qdot_floats +                                          // qdot tile(s)
                2ull * rows_per_env +                                   // lambda+meff
                jw) +                                                   // J + w (union only)
           slim +                                                       // slim (union only)
           sizeof(uint32_t) * 3ull * rows_per_env;                      // order+segs
}

// Build the compact solve record from the full NkRow. The Union family stages
// these in shared once per launch; the PairDriven family builds them inline in
// the sweep (reading the NkRow from global), keeping the per-row shared footprint
// out of the rows_per_env-sized carve so a many-contact env fits the block.
__device__ inline SlimRow MakeSlimRow(const NkRow& row, uint32_t env_row_base,
                                      uint32_t env_artic_base) {
    SlimRow sr;
    sr.flags = row.flags;
    sr.group_local = row.group_first - env_row_base;
    sr.group_cnt = row.group_normal_count;
    sr.rhs = row.rhs;
    sr.R = row.compliance_alpha;
    sr.lower = row.lower;
    sr.upper = row.upper;
    sr.mu = row.mu;
    uint32_t code = 0u;
    sr.a_tile = ~0u;
    sr.b_tile = ~0u;
    if (row.a.kind == kNkSideArtic) {
        code |= kSlimAArt;
        sr.a_tile = (row.a.index >= env_artic_base) ? (row.a.index - env_artic_base) : 0u;
    }
    if (row.b.kind == kNkSideArtic) {
        code |= kSlimBArt;
        sr.b_tile = (row.b.index >= env_artic_base) ? (row.b.index - env_artic_base) : 0u;
    }
    const bool a_dyn = row.a.kind == kNkSideRigid || row.a.kind == kNkSideParticle;
    const bool b_dyn = row.b.kind == kNkSideRigid || row.b.kind == kNkSideParticle;
    sr.dyn_index = 0u;
    sr.jl[0] = sr.jl[1] = sr.jl[2] = 0.0f;
    sr.ja[0] = sr.ja[1] = sr.ja[2] = 0.0f;
    if (a_dyn) {
        code |= kSlimHasDyn;
        if (row.a.kind == kNkSideParticle) code |= kSlimDynParticle;
        sr.dyn_index = row.a.index;
        sr.jl[0] = row.a.jlin.x; sr.jl[1] = row.a.jlin.y; sr.jl[2] = row.a.jlin.z;
        sr.ja[0] = row.a.jang.x; sr.ja[1] = row.a.jang.y; sr.ja[2] = row.a.jang.z;
        if (b_dyn) code |= kSlimFallback;
    } else if (b_dyn) {
        code |= kSlimHasDyn | kSlimDynIsB;
        if (row.b.kind == kNkSideParticle) code |= kSlimDynParticle;
        sr.dyn_index = row.b.index;
        sr.jl[0] = row.b.jlin.x; sr.jl[1] = row.b.jlin.y; sr.jl[2] = row.b.jlin.z;
        sr.ja[0] = row.b.jang.x; sr.ja[1] = row.b.jang.y; sr.ja[2] = row.b.jang.z;
    }
    sr.code = code;
    return sr;
}

// One row's velocity update — row_solver.cu SolveCompliantVelocityRow,
// preserved numerically (see the file header), executed by ONE WARP:
// * the Jv reduction stays the LEGACY SERIAL ascending-r loop on lane 0
// over the SHARED J slice (a tree/warp reduction would change the
// summation order),
// * the lambda update (bounds, clamp, regularizer feedback) is lane 0,
// * the M^-1 J^T apply is warp-PARALLEL over the dof elements (each
// qdot[r] += w[r]*delta is one independent multiply-add — identical
// rounding regardless of which lane executes it; w is shared-cached).
__device__ void ApplySlimImpulse(
    const SlimRow& sr, uint32_t j_row, uint32_t wlane, float delta,
    uint32_t env_artic_base, float* qdot_sh,
    const NkRow* __restrict__ urows,
    const float* J_sh, const float* w_sh, const float* J_b_sh,
    const float* w_b_sh, math::Vec3* body_lin_vel,
    math::Vec3* body_ang_vel, const float* body_inv_mass,
    const math::Vec3* body_inv_inertia, const float* particle_inv_mass,
    math::Vec3* particle_vel, uint32_t dof_stride) {
    const uint32_t code = sr.code;
    const uint32_t a_tile = (sr.a_tile == ~0u) ? 0u : sr.a_tile;
    const uint32_t b_tile = (sr.b_tile == ~0u) ? 0u : sr.b_tile;
    for (int side = 0; side < 2; ++side) {
        const bool art = side == 0 ? (code & kSlimAArt) != 0u
                                   : (code & kSlimBArt) != 0u;
        const bool dyn = (code & kSlimHasDyn) &&
                         ((side == 1) == ((code & kSlimDynIsB) != 0u));
        if (art) {
            const uint32_t tile = (side == 0) ? a_tile : b_tile;
            const float* w = side == 0
                                 ? w_sh + static_cast<size_t>(j_row) * dof_stride
                                 : ((w_b_sh != nullptr) ? w_b_sh : w_sh) +
                                       static_cast<size_t>(j_row) * dof_stride;
            float* const qd = qdot_sh + static_cast<size_t>(tile) * dof_stride;
            for (uint32_t r = wlane; r < dof_stride; r += 32u)
                qd[r] += w[r] * delta;
        } else if (dyn && wlane == 0u) {
            if (code & kSlimDynParticle) {
                if (particle_vel != nullptr && particle_inv_mass != nullptr) {
                    const float im = particle_inv_mass[sr.dyn_index];
                    if (im > 0.0f) {
                        math::Vec3& v = particle_vel[sr.dyn_index];
                        v.x += sr.jl[0] * (im * delta);
                        v.y += sr.jl[1] * (im * delta);
                        v.z += sr.jl[2] * (im * delta);
                    }
                }
            } else {
                const float im = body_inv_mass[sr.dyn_index];
                if (im > 0.0f) {
                    const math::Vec3 ii = body_inv_inertia[sr.dyn_index];
                    math::Vec3& v = body_lin_vel[sr.dyn_index];
                    math::Vec3& w = body_ang_vel[sr.dyn_index];
                    v.x += sr.jl[0] * (im * delta);
                    v.y += sr.jl[1] * (im * delta);
                    v.z += sr.jl[2] * (im * delta);
                    w.x += sr.ja[0] * ii.x * delta;
                    w.y += sr.ja[1] * ii.y * delta;
                    w.z += sr.ja[2] * ii.z * delta;
                }
            }
        } else if ((code & kSlimFallback) && side == 1 && wlane == 0u) {
            const NkRow row = urows[j_row];
            if (row.b.kind == kNkSideParticle && particle_vel != nullptr &&
                particle_inv_mass != nullptr) {
                const float im = particle_inv_mass[row.b.index];
                if (im > 0.0f) {
                    math::Vec3& v = particle_vel[row.b.index];
                    v.x += row.b.jlin.x * (im * delta);
                    v.y += row.b.jlin.y * (im * delta);
                    v.z += row.b.jlin.z * (im * delta);
                }
            } else if (row.b.kind == kNkSideRigid) {
                const float im = body_inv_mass[row.b.index];
                if (im > 0.0f) {
                    const math::Vec3 ii = body_inv_inertia[row.b.index];
                    math::Vec3& v = body_lin_vel[row.b.index];
                    math::Vec3& w = body_ang_vel[row.b.index];
                    v.x += row.b.jlin.x * (im * delta);
                    v.y += row.b.jlin.y * (im * delta);
                    v.z += row.b.jlin.z * (im * delta);
                    w.x += row.b.jang.x * ii.x * delta;
                    w.y += row.b.jang.y * ii.y * delta;
                    w.z += row.b.jang.z * ii.z * delta;
                }
            }
        }
    }
}

__device__ float ComputeSlimRowVelocity(
    const SlimRow& sr, uint32_t gslot, uint32_t j_row,
    const float* J, const float* J_b, const float* qdot,
    const NkRow* __restrict__ urows,
    const math::Vec3* __restrict__ body_lin_vel,
    const math::Vec3* __restrict__ body_ang_vel,
    const math::Vec3* __restrict__ particle_vel, uint32_t dof_stride) {
    const uint32_t code = sr.code;
    const uint32_t a_tile = sr.a_tile == ~0u ? 0u : sr.a_tile;
    const uint32_t b_tile = sr.b_tile == ~0u ? 0u : sr.b_tile;
    float art_jv_a = 0.0f;
    if (code & kSlimAArt) {
        const float* const row_j = J + static_cast<size_t>(j_row) * dof_stride;
        const float* const qd = qdot + static_cast<size_t>(a_tile) * dof_stride;
        for (uint32_t r = 0u; r < dof_stride; ++r) art_jv_a += row_j[r] * qd[r];
    }
    float art_jv_b = 0.0f;
    if (code & kSlimBArt) {
        const float* const row_j = (J_b != nullptr ? J_b : J) +
                                   static_cast<size_t>(j_row) * dof_stride;
        const float* const qd = qdot + static_cast<size_t>(b_tile) * dof_stride;
        for (uint32_t r = 0u; r < dof_stride; ++r) art_jv_b += row_j[r] * qd[r];
    }
    float dyn_jv = 0.0f;
    if (code & kSlimHasDyn) {
        const math::Vec3 jl{sr.jl[0], sr.jl[1], sr.jl[2]};
        if (code & kSlimDynParticle) {
            dyn_jv = particle_vel != nullptr ? Dot3(jl, particle_vel[sr.dyn_index])
                                             : 0.0f;
        } else {
            const math::Vec3 ja{sr.ja[0], sr.ja[1], sr.ja[2]};
            dyn_jv = Dot3(jl, body_lin_vel[sr.dyn_index]) +
                     Dot3(ja, body_ang_vel[sr.dyn_index]);
        }
    }
    float jv = 0.0f;
    if (code & kSlimAArt) jv += art_jv_a;
    else if ((code & kSlimHasDyn) && !(code & kSlimDynIsB)) jv += dyn_jv;
    if (code & kSlimBArt) jv += art_jv_b;
    else if ((code & kSlimHasDyn) && (code & kSlimDynIsB)) jv += dyn_jv;
    if (code & kSlimFallback) {
        const NkRow row = urows[gslot];
        if (row.b.kind == kNkSideParticle && particle_vel != nullptr) {
            jv += Dot3(row.b.jlin, particle_vel[row.b.index]);
        } else if (row.b.kind == kNkSideRigid) {
            jv += Dot3(row.b.jlin, body_lin_vel[row.b.index]) +
                  Dot3(row.b.jang, body_ang_vel[row.b.index]);
        }
    }
    return jv;
}

__device__ void ProjectContactBlock(const NkRow& normal, float* lambda_n,
                                    float* lambda_t1, float* lambda_t2) {
    *lambda_n = fmaxf(*lambda_n, 0.0f);
    const float mu1 = fmaxf(normal.mu, 0.0f);
    const float mu2 = fmaxf(normal.reserved[0], 0.0f);
    if (*lambda_n <= 0.0f || mu1 <= 1.0e-12f || mu2 <= 1.0e-12f) {
        *lambda_t1 = 0.0f;
        *lambda_t2 = 0.0f;
        return;
    }
    const float q1 = *lambda_t1 / mu1;
    const float q2 = *lambda_t2 / mu2;
    const float radius = sqrtf(q1 * q1 + q2 * q2);
    if (radius > *lambda_n && radius > 1.0e-12f) {
        const float scale = *lambda_n / radius;
        *lambda_t1 *= scale;
        *lambda_t2 *= scale;
    }
}

__device__ void SolveUnionRowWarp(uint32_t ls,            // env-local slot
                                  uint32_t gslot,         // global slot
                                  uint32_t env_row_base,  // env's first global slot
                                  uint32_t env_artic_base,// env's first global artic
                                  uint32_t j_row,         // row index into J/w arrays
                                  uint32_t wlane,
                                  const SlimRow* slim_sh, // null => build inline
                                  float* lambda_sh,       // null => use global lambda
                                  const float* meff_sh,   // null => use global row_meff
                                  float* __restrict__ lambda,    // global lambda
                                  const float* __restrict__ row_meff,  // global meff
                                  const float* J_sh,      // shared (union) OR global (PD)
                                  const float* w_sh,
                                  const float* J_b_sh,    // side-B chain-J (or null)
                                  const float* w_b_sh,    // side-B M^-1 J^T (or null)
                                  float* qdot_sh,         // K tiles, [tile*dof+r]
                                  const NkRow* __restrict__ urows,
                                  math::Vec3* __restrict__ body_lin_vel,
                                  math::Vec3* __restrict__ body_ang_vel,
                                  const float* __restrict__ body_inv_mass,
                                  const math::Vec3* __restrict__ body_inv_inertia,
                                  const float* __restrict__ particle_inv_mass,
                                  math::Vec3* __restrict__ particle_vel,
                                  uint32_t dof_stride,
                                  float dt,
                                  bool apply_cached_impulse) {
    // Union stages slim in shared (slim_sh != null); PairDriven builds it inline
    // from the global NkRow (slim_sh == null) so the shared carve stays small.
    SlimRow sr_local;
    const SlimRow* sr;
    if (slim_sh != nullptr) {
        sr = slim_sh + ls;
    } else {
        sr_local = MakeSlimRow(urows[gslot], env_row_base, env_artic_base);
        sr = &sr_local;
    }
    const uint32_t flags = sr->flags;
    if (!(flags & nk::nk_row_flags::kActive)) {
        return;  // watermark early-exit (inactive slot).
    }
    const uint32_t code = sr->code;
    // each artic side reads/writes its OWN per-articulation tile in qdot_sh.
    // At K==1 a_tile == b_tile == 0 (one tile/env) -> the legacy single-tile path.
    const uint32_t a_tile = (sr->a_tile == ~0u) ? 0u : sr->a_tile;
    const uint32_t b_tile = (sr->b_tile == ~0u) ? 0u : sr->b_tile;
    const bool block_row =
        (flags & (nk::nk_row_flags::kBlockNormal |
                  nk::nk_row_flags::kBlockTangent)) != 0u;
    uint32_t block_normal_slot = 0u;
    uint32_t block_tangent1_slot = 0u;
    uint32_t block_tangent2_slot = 0u;
    float block_old_normal = 0.0f;
    float block_old_tangent1 = 0.0f;
    float block_old_tangent2 = 0.0f;
    float block_delta_normal = 0.0f;
    float block_delta_tangent1 = 0.0f;
    float block_delta_tangent2 = 0.0f;
    if (block_row) {
        const NkRow& block = urows[gslot];
        const uint32_t count = block.group_normal_count;
        const uint32_t point = (gslot - block.group_first) % count;
        block_normal_slot = block.group_first + point;
        block_tangent1_slot = block.group_first + count + point;
        block_tangent2_slot = block.group_first + 2u * count + point;
        block_old_normal = lambda[block_normal_slot];
        block_old_tangent1 = lambda[block_tangent1_slot];
        block_old_tangent2 = lambda[block_tangent2_slot];
    }
    if (block_row && gslot != block_normal_slot) return;

    float delta = 0.0f;
    if (wlane == 0u) {
        float jv = ComputeSlimRowVelocity(
            *sr, gslot, j_row, J_sh, J_b_sh, qdot_sh, urows, body_lin_vel,
            body_ang_vel, particle_vel, dof_stride);
        float block_jv_tangent1 = 0.0f;
        float block_jv_tangent2 = 0.0f;
        if (block_row && !apply_cached_impulse) {
            const SlimRow tangent1 = MakeSlimRow(urows[block_tangent1_slot],
                                                 env_row_base, env_artic_base);
            const SlimRow tangent2 = MakeSlimRow(urows[block_tangent2_slot],
                                                 env_row_base, env_artic_base);
            block_jv_tangent1 = ComputeSlimRowVelocity(
                tangent1, block_tangent1_slot, block_tangent1_slot, J_sh, J_b_sh,
                qdot_sh, urows, body_lin_vel, body_ang_vel, particle_vel,
                dof_stride);
            block_jv_tangent2 = ComputeSlimRowVelocity(
                tangent2, block_tangent2_slot, block_tangent2_slot, J_sh, J_b_sh,
                qdot_sh, urows, body_lin_vel, body_ang_vel, particle_vel,
                dof_stride);
        }

        // lambda / meff source: SHARED slice (Union) or GLOBAL (PairDriven, where
        // the row-scaled staging stays out of shared). The global env-local slot of
        // a group member g is env_row_base + group_local + g == its global slot.
        const float effective_mass = meff_sh != nullptr ? meff_sh[ls]
                                                        : row_meff[gslot];
        const float old_impulse = lambda_sh != nullptr ? lambda_sh[ls]
                                                       : lambda[gslot];
        if (apply_cached_impulse) {
            if (block_row && gslot == block_normal_slot) {
                block_delta_normal = block_old_normal;
                block_delta_tangent1 = block_old_tangent1;
                block_delta_tangent2 = block_old_tangent2;
            } else if (!block_row) {
                delta = (fabsf(old_impulse) > 1.0e-12f) ? old_impulse : 0.0f;
            }
        } else if (block_row) {
            const NkRow& normal = urows[block_normal_slot];
            const NkRow& tangent1 = urows[block_tangent1_slot];
            const NkRow& tangent2 = urows[block_tangent2_slot];
            const float residual_n = normal.rhs * dt - jv -
                                     normal.compliance_alpha * block_old_normal;
            const float residual_t1 = tangent1.rhs * dt - block_jv_tangent1 -
                                      tangent1.compliance_alpha * block_old_tangent1;
            const float residual_t2 = tangent2.rhs * dt - block_jv_tangent2 -
                                      tangent2.compliance_alpha * block_old_tangent2;
            float new_normal = block_old_normal +
                normal.reserved[1] * residual_n +
                normal.reserved[2] * residual_t1 +
                normal.reserved[3] * residual_t2;
            float new_tangent1 = block_old_tangent1 +
                normal.reserved[2] * residual_n +
                normal.reserved[4] * residual_t1 +
                normal.reserved[5] * residual_t2;
            float new_tangent2 = block_old_tangent2 +
                normal.reserved[3] * residual_n +
                normal.reserved[5] * residual_t1 +
                normal.reserved[6] * residual_t2;
            ProjectContactBlock(normal, &new_normal, &new_tangent1, &new_tangent2);
            lambda[block_normal_slot] = new_normal;
            lambda[block_tangent1_slot] = new_tangent1;
            lambda[block_tangent2_slot] = new_tangent2;
            block_delta_normal = new_normal - block_old_normal;
            block_delta_tangent1 = new_tangent1 - block_old_tangent1;
            block_delta_tangent2 = new_tangent2 - block_old_tangent2;
        } else {
            const float rhs_v = sr->rhs * dt;
            const float lambda_inc =
                effective_mass * (rhs_v - jv - sr->R * old_impulse);
            float lower = sr->lower;
            float upper = sr->upper;
            if (flags & nk::nk_row_flags::kFriction) {
                float total = 0.0f;
                for (uint32_t g = 0u; g < sr->group_cnt; ++g) {
                    const float lg = lambda_sh != nullptr
                                         ? lambda_sh[sr->group_local + g]
                                         : lambda[env_row_base + sr->group_local + g];
                    total += fmaxf(lg, 0.0f);
                }
                lower = 0.0f;
                upper = fmaxf(sr->mu, 0.0f) * total;
            }
            const float new_impulse =
                fminf(fmaxf(old_impulse + lambda_inc, lower), upper);
            if (lambda_sh != nullptr) lambda_sh[ls] = new_impulse;
            else lambda[gslot] = new_impulse;
            const float d = new_impulse - old_impulse;
            delta = fabsf(d) > 1.0e-12f ? d : 0.0f;
        }
    }
    delta = __shfl_sync(0xffffffffu, delta, 0);
    if (block_row) {
        block_delta_normal = __shfl_sync(0xffffffffu, block_delta_normal, 0);
        block_delta_tangent1 = __shfl_sync(0xffffffffu, block_delta_tangent1, 0);
        block_delta_tangent2 = __shfl_sync(0xffffffffu, block_delta_tangent2, 0);
        if (fabsf(block_delta_normal) > 1.0e-12f) {
            const SlimRow sr_block = MakeSlimRow(urows[block_normal_slot],
                                                  env_row_base, env_artic_base);
            ApplySlimImpulse(sr_block, block_normal_slot, wlane, block_delta_normal,
                             env_artic_base, qdot_sh, urows, J_sh, w_sh, J_b_sh,
                             w_b_sh, body_lin_vel, body_ang_vel, body_inv_mass,
                             body_inv_inertia, particle_inv_mass, particle_vel,
                             dof_stride);
        }
        if (fabsf(block_delta_tangent1) > 1.0e-12f) {
            const SlimRow sr_block = MakeSlimRow(urows[block_tangent1_slot],
                                                  env_row_base, env_artic_base);
            ApplySlimImpulse(sr_block, block_tangent1_slot, wlane,
                             block_delta_tangent1, env_artic_base, qdot_sh, urows,
                             J_sh, w_sh, J_b_sh, w_b_sh, body_lin_vel,
                             body_ang_vel, body_inv_mass, body_inv_inertia,
                             particle_inv_mass, particle_vel, dof_stride);
        }
        if (fabsf(block_delta_tangent2) > 1.0e-12f) {
            const SlimRow sr_block = MakeSlimRow(urows[block_tangent2_slot],
                                                  env_row_base, env_artic_base);
            ApplySlimImpulse(sr_block, block_tangent2_slot, wlane,
                             block_delta_tangent2, env_artic_base, qdot_sh, urows,
                             J_sh, w_sh, J_b_sh, w_b_sh, body_lin_vel,
                             body_ang_vel, body_inv_mass, body_inv_inertia,
                             particle_inv_mass, particle_vel, dof_stride);
        }
    }
    if (!block_row && delta != 0.0f) {
        // side a then b (the legacy apply order). The artic arm is the
        // dof-wide element-independent apply — warp-parallel; the rigid /
        // particle arms touch a handful of scalars — lane 0.
        for (int side = 0; side < 2; ++side) {
            const bool art = side == 0 ? (code & kSlimAArt) != 0u
                                       : (code & kSlimBArt) != 0u;
            const bool dyn = (code & kSlimHasDyn) &&
                             ((side == 1) == ((code & kSlimDynIsB) != 0u));
            if (art) {
                // qdot_side += w_side * delta with the PRECOMPUTED w = M^-1 J^T,
                // into the side's OWN per-articulation tile — element-
                // independent multiply-add (bit-equal whatever lane executes it).
                const uint32_t tile = (side == 0) ? a_tile : b_tile;
                const float* w;
                if (side == 0) {
                    w = w_sh + static_cast<size_t>(j_row) * dof_stride;
                } else {
                    w = ((w_b_sh != nullptr) ? w_b_sh : w_sh) +
                        static_cast<size_t>(j_row) * dof_stride;
                }
                float* const qd = qdot_sh + static_cast<size_t>(tile) * dof_stride;
                for (uint32_t r = wlane; r < dof_stride; r += 32u) {
                    qd[r] += w[r] * delta;
                }
            } else if (dyn && wlane == 0u) {
                if (code & kSlimDynParticle) {
                    if (particle_vel != nullptr && particle_inv_mass != nullptr) {
                        const float im = particle_inv_mass[sr->dyn_index];
                        if (im > 0.0f) {
                            math::Vec3& v = particle_vel[sr->dyn_index];
                            v.x += sr->jl[0] * (im * delta);
                            v.y += sr->jl[1] * (im * delta);
                            v.z += sr->jl[2] * (im * delta);
                        }
                    }
                } else {
                    const float im = body_inv_mass[sr->dyn_index];
                    if (im > 0.0f) {  // immovable rigid: no-op.
                        const math::Vec3 ii = body_inv_inertia[sr->dyn_index];
                        math::Vec3& v = body_lin_vel[sr->dyn_index];
                        math::Vec3& w = body_ang_vel[sr->dyn_index];
                        v.x += sr->jl[0] * (im * delta);
                        v.y += sr->jl[1] * (im * delta);
                        v.z += sr->jl[2] * (im * delta);
                        w.x += sr->ja[0] * ii.x * delta;
                        w.y += sr->ja[1] * ii.y * delta;
                        w.z += sr->ja[2] * ii.z * delta;
                    }
                }
            } else if ((code & kSlimFallback) && side == 1 && wlane == 0u) {
                // Two-dynamic-side fallback: apply side b from the full record.
                const NkRow row = urows[gslot];
                if (row.b.kind == kNkSideParticle && particle_vel != nullptr &&
                    particle_inv_mass != nullptr) {
                    const float im = particle_inv_mass[row.b.index];
                    if (im > 0.0f) {
                        math::Vec3& v = particle_vel[row.b.index];
                        v.x += row.b.jlin.x * (im * delta);
                        v.y += row.b.jlin.y * (im * delta);
                        v.z += row.b.jlin.z * (im * delta);
                    }
                } else if (row.b.kind == kNkSideRigid) {
                    const float im = body_inv_mass[row.b.index];
                    if (im > 0.0f) {
                        const math::Vec3 ii = body_inv_inertia[row.b.index];
                        math::Vec3& v = body_lin_vel[row.b.index];
                        math::Vec3& w = body_ang_vel[row.b.index];
                        v.x += row.b.jlin.x * (im * delta);
                        v.y += row.b.jlin.y * (im * delta);
                        v.z += row.b.jlin.z * (im * delta);
                        w.x += row.b.jang.x * ii.x * delta;
                        w.y += row.b.jang.y * ii.y * delta;
                        w.z += row.b.jang.z * ii.z * delta;
                    }
                }
            }
        }
    }
}

// One NORMAL row's split-impulse position update, executed by ONE WARP. A
// GEOMETRIC projection (no -R*lambda compliance feedback): it drives a SEPARATE
// pseudo velocity from the contact penetration depth so position can close the
// overlap WITHOUT injecting energy into the persisted velocity. Mirrors the
// velocity warp's machinery (same J, same w = M^-1 J^T, same meff, same side
// dispatch, same fixed sweep order) but reads/writes ONLY the pseudo
// accumulators. Friction rows are skipped (one-sided non-penetration only);
// the pseudo impulse is clamped >= 0.
__device__ void SolvePositionRowWarp(uint32_t gslot,
                                     uint32_t env_row_base,
                                     uint32_t env_artic_base,
                                     uint32_t j_row,
                                     uint32_t wlane,
                                     const float* __restrict__ row_meff,
                                     const float* __restrict__ row_penetration,
                                     float* __restrict__ row_pseudo_lambda,
                                     const float* J_sh,
                                     const float* w_sh,
                                     const float* J_b_sh,
                                     const float* w_b_sh,
                                     float* qdot_pseudo_sh,
                                     const NkRow* __restrict__ urows,
                                     math::Vec3* __restrict__ body_pseudo_lin,
                                     math::Vec3* __restrict__ body_pseudo_ang,
                                     const float* __restrict__ body_inv_mass,
                                     const math::Vec3* __restrict__ body_inv_inertia,
                                     const float* __restrict__ particle_inv_mass,
                                     math::Vec3* __restrict__ particle_pseudo_vel,
                                     uint32_t dof_stride,
                                     float beta, float slop, float dt,
                                     float baumgarte_max_velocity) {
    const SlimRow sr = MakeSlimRow(urows[gslot], env_row_base, env_artic_base);
    const uint32_t flags = sr.flags;
    if (!(flags & nk::nk_row_flags::kActive) ||
        (flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kBlockTangent))) {
        return;  // inactive or tangential row: no position correction.
    }
    const uint32_t code = sr.code;
    const uint32_t a_tile = (sr.a_tile == ~0u) ? 0u : sr.a_tile;
    const uint32_t b_tile = (sr.b_tile == ~0u) ? 0u : sr.b_tile;

    float delta = 0.0f;
    if (wlane == 0u) {
        // Pseudo separating velocity from the SAME J over the pseudo qdot tile
        // (artic) and the pseudo body/particle velocities (rigid/particle).
        float art_jv_a = 0.0f;
        if (code & kSlimAArt) {
            const float* const J = J_sh + static_cast<size_t>(j_row) * dof_stride;
            const float* const qd =
                qdot_pseudo_sh + static_cast<size_t>(a_tile) * dof_stride;
            for (uint32_t r = 0u; r < dof_stride; ++r) art_jv_a += J[r] * qd[r];
        }
        float art_jv_b = 0.0f;
        if (code & kSlimBArt) {
            const float* const J =
                ((J_b_sh != nullptr) ? J_b_sh : J_sh) +
                static_cast<size_t>(j_row) * dof_stride;
            const float* const qd =
                qdot_pseudo_sh + static_cast<size_t>(b_tile) * dof_stride;
            for (uint32_t r = 0u; r < dof_stride; ++r) art_jv_b += J[r] * qd[r];
        }
        float dyn_jv = 0.0f;
        if (code & kSlimHasDyn) {
            const math::Vec3 jl{sr.jl[0], sr.jl[1], sr.jl[2]};
            if (code & kSlimDynParticle) {
                dyn_jv = (particle_pseudo_vel != nullptr)
                             ? Dot3(jl, particle_pseudo_vel[sr.dyn_index])
                             : 0.0f;
            } else {
                const math::Vec3 ja{sr.ja[0], sr.ja[1], sr.ja[2]};
                dyn_jv = Dot3(jl, body_pseudo_lin[sr.dyn_index]) +
                         Dot3(ja, body_pseudo_ang[sr.dyn_index]);
            }
        }
        float jv = 0.0f;
        if (code & kSlimAArt) jv += art_jv_a;
        else if ((code & kSlimHasDyn) && !(code & kSlimDynIsB)) jv += dyn_jv;
        if (code & kSlimBArt) jv += art_jv_b;
        else if ((code & kSlimHasDyn) && (code & kSlimDynIsB)) jv += dyn_jv;

        // Pseudo separating velocity from the penetration, capped at
        // baumgarte_max_velocity (+inf default => byte-identical) for bounded push-out.
        const float depth = row_penetration[gslot];
        const float bias =
            fminf(beta * fmaxf(depth - slop, 0.0f) / dt, baumgarte_max_velocity);
        const float effective_mass = row_meff[gslot];
        const float old_imp = row_pseudo_lambda[gslot];
        // GEOMETRIC projection: no -R*lambda compliance term (this is position,
        // not a compliant force). One-sided (pseudo impulse >= 0).
        const float new_imp = fmaxf(old_imp + effective_mass * (bias - jv), 0.0f);
        row_pseudo_lambda[gslot] = new_imp;
        const float d = new_imp - old_imp;
        delta = (fabsf(d) > 1.0e-12f) ? d : 0.0f;
    }
    delta = __shfl_sync(0xffffffffu, delta, 0);
    if (delta != 0.0f) {
        for (int side = 0; side < 2; ++side) {
            const bool art = side == 0 ? (code & kSlimAArt) != 0u
                                       : (code & kSlimBArt) != 0u;
            const bool dyn = (code & kSlimHasDyn) &&
                             ((side == 1) == ((code & kSlimDynIsB) != 0u));
            if (art) {
                const uint32_t tile = (side == 0) ? a_tile : b_tile;
                const float* w = ((side == 0) ? w_sh
                                              : ((w_b_sh != nullptr) ? w_b_sh : w_sh)) +
                                 static_cast<size_t>(j_row) * dof_stride;
                float* const qd =
                    qdot_pseudo_sh + static_cast<size_t>(tile) * dof_stride;
                for (uint32_t r = wlane; r < dof_stride; r += 32u) {
                    qd[r] += w[r] * delta;
                }
            } else if (dyn && wlane == 0u) {
                // Particle pseudo-vel feeds the body side's split-impulse; the particle's
                // own position is owned by its medium (XPBD/PBF), not integrated here.
                if (code & kSlimDynParticle) {
                    if (particle_pseudo_vel != nullptr &&
                        particle_inv_mass != nullptr) {
                        const float im = particle_inv_mass[sr.dyn_index];
                        if (im > 0.0f) {
                            math::Vec3& v = particle_pseudo_vel[sr.dyn_index];
                            v.x += sr.jl[0] * (im * delta);
                            v.y += sr.jl[1] * (im * delta);
                            v.z += sr.jl[2] * (im * delta);
                        }
                    }
                } else {
                    const float im = body_inv_mass[sr.dyn_index];
                    if (im > 0.0f) {
                        const math::Vec3 ii = body_inv_inertia[sr.dyn_index];
                        math::Vec3& v = body_pseudo_lin[sr.dyn_index];
                        math::Vec3& w = body_pseudo_ang[sr.dyn_index];
                        v.x += sr.jl[0] * (im * delta);
                        v.y += sr.jl[1] * (im * delta);
                        v.z += sr.jl[2] * (im * delta);
                        w.x += sr.ja[0] * ii.x * delta;
                        w.y += sr.ja[1] * ii.y * delta;
                        w.z += sr.ja[2] * ii.z * delta;
                    }
                }
            }
        }
    }
}

__device__ void ApplyDynamicImpulseScalar(
    uint32_t gslot, float delta, const NkRow* __restrict__ urows,
    math::Vec3* __restrict__ body_lin_vel,
    math::Vec3* __restrict__ body_ang_vel,
    const float* __restrict__ body_inv_mass,
    const math::Vec3* __restrict__ body_inv_inertia,
    const float* __restrict__ particle_inv_mass,
    math::Vec3* __restrict__ particle_vel) {
    if (fabsf(delta) <= 1.0e-12f) return;
    const NkRow row = urows[gslot];
    for (int side = 0; side < 2; ++side) {
        const NkRowSide& sd = side == 0 ? row.a : row.b;
        if (sd.kind == kNkSideParticle && particle_vel != nullptr &&
            particle_inv_mass != nullptr) {
            const float im = particle_inv_mass[sd.index];
            if (im > 0.0f) {
                math::Vec3& v = particle_vel[sd.index];
                v.x += sd.jlin.x * (im * delta);
                v.y += sd.jlin.y * (im * delta);
                v.z += sd.jlin.z * (im * delta);
            }
        } else if (sd.kind == kNkSideRigid) {
            const float im = body_inv_mass[sd.index];
            if (im > 0.0f) {
                const math::Vec3 ii = body_inv_inertia[sd.index];
                math::Vec3& v = body_lin_vel[sd.index];
                math::Vec3& w = body_ang_vel[sd.index];
                v.x += sd.jlin.x * (im * delta);
                v.y += sd.jlin.y * (im * delta);
                v.z += sd.jlin.z * (im * delta);
                w.x += sd.jang.x * ii.x * delta;
                w.y += sd.jang.y * ii.y * delta;
                w.z += sd.jang.z * ii.z * delta;
            }
        }
    }
}

// A component with no articulation side has no reduced-coordinate vector to fan
// across a warp: every row update is scalar rigid/particle work. Dynamic islanding
// proves distinct components share no mutable side or friction group, so one CUDA
// thread may run each component's original ascending GS sweep independently. The
// arithmetic below is the no-artic subset of SolveUnionRowWarp, kept statement-for-
// statement so only execution width changes.
__device__ void SolveDynamicRowScalar(
    uint32_t gslot, uint32_t env_row_base, uint32_t env_artic_base,
    const NkRow* __restrict__ urows, float* __restrict__ lambda,
    const float* __restrict__ row_meff,
    math::Vec3* __restrict__ body_lin_vel,
    math::Vec3* __restrict__ body_ang_vel,
    const float* __restrict__ body_inv_mass,
    const math::Vec3* __restrict__ body_inv_inertia,
    const float* __restrict__ particle_inv_mass,
    math::Vec3* __restrict__ particle_vel, float dt,
    bool apply_cached_impulse) {
    const SlimRow sr = MakeSlimRow(urows[gslot], env_row_base, env_artic_base);
    const uint32_t flags = sr.flags;
    if (!(flags & nk::nk_row_flags::kActive)) return;
    const bool block_row =
        (flags & (nk::nk_row_flags::kBlockNormal |
                  nk::nk_row_flags::kBlockTangent)) != 0u;
    uint32_t block_normal_slot = 0u;
    uint32_t block_tangent1_slot = 0u;
    uint32_t block_tangent2_slot = 0u;
    float block_old_normal = 0.0f;
    float block_old_tangent1 = 0.0f;
    float block_old_tangent2 = 0.0f;
    if (block_row) {
        const NkRow& block = urows[gslot];
        const uint32_t count = block.group_normal_count;
        const uint32_t point = (gslot - block.group_first) % count;
        block_normal_slot = block.group_first + point;
        block_tangent1_slot = block.group_first + count + point;
        block_tangent2_slot = block.group_first + 2u * count + point;
        block_old_normal = lambda[block_normal_slot];
        block_old_tangent1 = lambda[block_tangent1_slot];
        block_old_tangent2 = lambda[block_tangent2_slot];
    }
    if (block_row && gslot != block_normal_slot) return;

    const float jv = ComputeSlimRowVelocity(
        sr, gslot, gslot, nullptr, nullptr, nullptr, urows, body_lin_vel,
        body_ang_vel, particle_vel, 0u);
    float block_jv_tangent1 = 0.0f;
    float block_jv_tangent2 = 0.0f;
    if (block_row && !apply_cached_impulse) {
        const SlimRow tangent1 = MakeSlimRow(urows[block_tangent1_slot],
                                             env_row_base, env_artic_base);
        const SlimRow tangent2 = MakeSlimRow(urows[block_tangent2_slot],
                                             env_row_base, env_artic_base);
        block_jv_tangent1 = ComputeSlimRowVelocity(
            tangent1, block_tangent1_slot, block_tangent1_slot, nullptr, nullptr,
            nullptr, urows, body_lin_vel, body_ang_vel, particle_vel, 0u);
        block_jv_tangent2 = ComputeSlimRowVelocity(
            tangent2, block_tangent2_slot, block_tangent2_slot, nullptr, nullptr,
            nullptr, urows, body_lin_vel, body_ang_vel, particle_vel, 0u);
    }

    const float effective_mass = row_meff[gslot];
    const float old_impulse = lambda[gslot];
    float delta = 0.0f;
    float block_delta_normal = 0.0f;
    float block_delta_tangent1 = 0.0f;
    float block_delta_tangent2 = 0.0f;
    if (apply_cached_impulse) {
        if (block_row && gslot == block_normal_slot) {
            block_delta_normal = block_old_normal;
            block_delta_tangent1 = block_old_tangent1;
            block_delta_tangent2 = block_old_tangent2;
        } else if (!block_row) {
            delta = (fabsf(old_impulse) > 1.0e-12f) ? old_impulse : 0.0f;
        }
    } else if (block_row) {
        const NkRow& normal = urows[block_normal_slot];
        const NkRow& tangent1 = urows[block_tangent1_slot];
        const NkRow& tangent2 = urows[block_tangent2_slot];
        const float residual_n = normal.rhs * dt - jv -
                                 normal.compliance_alpha * block_old_normal;
        const float residual_t1 = tangent1.rhs * dt - block_jv_tangent1 -
                                  tangent1.compliance_alpha * block_old_tangent1;
        const float residual_t2 = tangent2.rhs * dt - block_jv_tangent2 -
                                  tangent2.compliance_alpha * block_old_tangent2;
        float new_normal = block_old_normal +
            normal.reserved[1] * residual_n +
            normal.reserved[2] * residual_t1 +
            normal.reserved[3] * residual_t2;
        float new_tangent1 = block_old_tangent1 +
            normal.reserved[2] * residual_n +
            normal.reserved[4] * residual_t1 +
            normal.reserved[5] * residual_t2;
        float new_tangent2 = block_old_tangent2 +
            normal.reserved[3] * residual_n +
            normal.reserved[5] * residual_t1 +
            normal.reserved[6] * residual_t2;
        ProjectContactBlock(normal, &new_normal, &new_tangent1, &new_tangent2);
        lambda[block_normal_slot] = new_normal;
        lambda[block_tangent1_slot] = new_tangent1;
        lambda[block_tangent2_slot] = new_tangent2;
        block_delta_normal = new_normal - block_old_normal;
        block_delta_tangent1 = new_tangent1 - block_old_tangent1;
        block_delta_tangent2 = new_tangent2 - block_old_tangent2;
    } else {
        const float rhs_v = sr.rhs * dt;
        const float lambda_inc =
            effective_mass * (rhs_v - jv - sr.R * old_impulse);
        float lower = sr.lower;
        float upper = sr.upper;
        if (flags & nk::nk_row_flags::kFriction) {
            float total = 0.0f;
            for (uint32_t g = 0u; g < sr.group_cnt; ++g) {
                total += fmaxf(lambda[env_row_base + sr.group_local + g], 0.0f);
            }
            lower = 0.0f;
            upper = fmaxf(sr.mu, 0.0f) * total;
        }
        const float new_impulse =
            fminf(fmaxf(old_impulse + lambda_inc, lower), upper);
        lambda[gslot] = new_impulse;
        const float d = new_impulse - old_impulse;
        delta = fabsf(d) > 1.0e-12f ? d : 0.0f;
    }
    if (block_row) {
        ApplyDynamicImpulseScalar(block_normal_slot, block_delta_normal, urows,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_inv_inertia, particle_inv_mass, particle_vel);
        ApplyDynamicImpulseScalar(block_tangent1_slot, block_delta_tangent1, urows,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_inv_inertia, particle_inv_mass, particle_vel);
        ApplyDynamicImpulseScalar(block_tangent2_slot, block_delta_tangent2, urows,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_inv_inertia, particle_inv_mass, particle_vel);
    } else {
        ApplyDynamicImpulseScalar(gslot, delta, urows, body_lin_vel, body_ang_vel,
                                  body_inv_mass, body_inv_inertia,
                                  particle_inv_mass, particle_vel);
    }
}

// Exact no-artic subset of SolvePositionRowWarp. As in the existing path, the
// split pass acts only on SlimRow's selected dynamic side; the two-dynamic fallback
// remains velocity-only, preserving current semantics byte-for-byte.
__device__ void SolvePositionRowScalar(
    uint32_t gslot, uint32_t env_row_base, uint32_t env_artic_base,
    const float* __restrict__ row_meff,
    const float* __restrict__ row_penetration,
    float* __restrict__ row_pseudo_lambda,
    const NkRow* __restrict__ urows,
    math::Vec3* __restrict__ body_pseudo_lin,
    math::Vec3* __restrict__ body_pseudo_ang,
    const float* __restrict__ body_inv_mass,
    const math::Vec3* __restrict__ body_inv_inertia,
    const float* __restrict__ particle_inv_mass,
    math::Vec3* __restrict__ particle_pseudo_vel,
    float beta, float slop, float dt, float baumgarte_max_velocity) {
    const SlimRow sr = MakeSlimRow(urows[gslot], env_row_base, env_artic_base);
    const uint32_t flags = sr.flags;
    if (!(flags & nk::nk_row_flags::kActive) ||
        (flags & (nk::nk_row_flags::kFriction | nk::nk_row_flags::kBlockTangent))) return;
    const uint32_t code = sr.code;
    float dyn_jv = 0.0f;
    if (code & kSlimHasDyn) {
        const math::Vec3 jl{sr.jl[0], sr.jl[1], sr.jl[2]};
        if (code & kSlimDynParticle) {
            dyn_jv = (particle_pseudo_vel != nullptr)
                         ? Dot3(jl, particle_pseudo_vel[sr.dyn_index]) : 0.0f;
        } else {
            const math::Vec3 ja{sr.ja[0], sr.ja[1], sr.ja[2]};
            dyn_jv = Dot3(jl, body_pseudo_lin[sr.dyn_index]) +
                     Dot3(ja, body_pseudo_ang[sr.dyn_index]);
        }
    }
    float jv = 0.0f;
    if ((code & kSlimHasDyn) && !(code & kSlimDynIsB)) jv += dyn_jv;
    if ((code & kSlimHasDyn) && (code & kSlimDynIsB)) jv += dyn_jv;
    const float depth = row_penetration[gslot];
    const float bias =
        fminf(beta * fmaxf(depth - slop, 0.0f) / dt, baumgarte_max_velocity);
    const float effective_mass = row_meff[gslot];
    const float old_imp = row_pseudo_lambda[gslot];
    const float new_imp = fmaxf(old_imp + effective_mass * (bias - jv), 0.0f);
    row_pseudo_lambda[gslot] = new_imp;
    const float d = new_imp - old_imp;
    const float delta = (fabsf(d) > 1.0e-12f) ? d : 0.0f;
    if (delta == 0.0f) return;

    for (int side = 0; side < 2; ++side) {
        const bool dyn = (code & kSlimHasDyn) &&
                         ((side == 1) == ((code & kSlimDynIsB) != 0u));
        if (!dyn) continue;
        if (code & kSlimDynParticle) {
            if (particle_pseudo_vel != nullptr && particle_inv_mass != nullptr) {
                const float im = particle_inv_mass[sr.dyn_index];
                if (im > 0.0f) {
                    math::Vec3& v = particle_pseudo_vel[sr.dyn_index];
                    v.x += sr.jl[0] * (im * delta);
                    v.y += sr.jl[1] * (im * delta);
                    v.z += sr.jl[2] * (im * delta);
                }
            }
        } else {
            const float im = body_inv_mass[sr.dyn_index];
            if (im > 0.0f) {
                const math::Vec3 ii = body_inv_inertia[sr.dyn_index];
                math::Vec3& v = body_pseudo_lin[sr.dyn_index];
                math::Vec3& w = body_pseudo_ang[sr.dyn_index];
                v.x += sr.jl[0] * (im * delta);
                v.y += sr.jl[1] * (im * delta);
                v.z += sr.jl[2] * (im * delta);
                w.x += sr.ja[0] * ii.x * delta;
                w.y += sr.ja[1] * ii.y * delta;
                w.z += sr.ja[2] * ii.z * delta;
            }
        }
    }
}

__global__ void SolveRowsScalarIslandsKernel(
    const NkRow* __restrict__ urows, float* __restrict__ lambda,
    const float* __restrict__ row_meff,
    math::Vec3* __restrict__ body_lin_vel,
    math::Vec3* __restrict__ body_ang_vel,
    const float* __restrict__ body_inv_mass,
    const math::Vec3* __restrict__ body_inv_inertia,
    const float* __restrict__ particle_inv_mass,
    math::Vec3* __restrict__ particle_vel,
    const uint32_t* __restrict__ islands,
    const uint32_t* __restrict__ row_order,
    const float* __restrict__ row_penetration,
    float* __restrict__ row_pseudo_lambda,
    math::Vec3* __restrict__ body_pseudo_lin,
    math::Vec3* __restrict__ body_pseudo_ang,
    math::Vec3* __restrict__ particle_pseudo_vel,
    const uint32_t* __restrict__ island_count_dev,
    uint32_t rows_per_env, uint32_t artics_per_env,
    uint32_t vel_iters, uint32_t pos_iters,
    float pos_beta, float pos_slop, float dt,
    float baumgarte_max_velocity) {
    const uint32_t live_islands = *island_count_dev;
    // BuildSolveIslands packs live components into a prefix. Interleave that prefix
    // across blocks before filling the next lane so a modest fixed grid spreads
    // serial GS components over the device instead of concentrating them in the
    // first few CTAs. Each component remains owned by exactly one thread and keeps
    // its original ascending row/iteration order.
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t island = blockIdx.x + threadIdx.x * gridDim.x;
         island < live_islands; island += stride) {
        const IslandRecord rec =
            reinterpret_cast<const IslandRecord*>(islands)[island];
        if (rec.flags & 1u) continue;  // reduced-coordinate component: warp path.
        const uint32_t env_row_base = rec.env * rows_per_env;
        const uint32_t env_artic_base =
            rec.env * (artics_per_env == 0u ? 1u : artics_per_env);
        for (uint32_t r = 0u; r < rec.seg_cnt; ++r) {
            SolveDynamicRowScalar(row_order[rec.seg_off + r], env_row_base,
                                  env_artic_base, urows, lambda, row_meff,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_inv_inertia, particle_inv_mass,
                                  particle_vel, dt, true);
        }
        for (uint32_t it = 0u; it < vel_iters; ++it) {
            for (uint32_t r = 0u; r < rec.seg_cnt; ++r) {
                SolveDynamicRowScalar(row_order[rec.seg_off + r], env_row_base,
                                      env_artic_base, urows, lambda, row_meff,
                                      body_lin_vel, body_ang_vel, body_inv_mass,
                                      body_inv_inertia, particle_inv_mass,
                                      particle_vel, dt, false);
            }
        }
        if (pos_iters == 0u) continue;
        for (uint32_t r = 0u; r < rec.seg_cnt; ++r)
            row_pseudo_lambda[row_order[rec.seg_off + r]] = 0.0f;
        for (uint32_t it = 0u; it < pos_iters; ++it) {
            for (uint32_t r = 0u; r < rec.seg_cnt; ++r) {
                SolvePositionRowScalar(
                    row_order[rec.seg_off + r], env_row_base, env_artic_base,
                    row_meff, row_penetration, row_pseudo_lambda, urows,
                    body_pseudo_lin, body_pseudo_ang, body_inv_mass,
                    body_inv_inertia, particle_inv_mass, particle_pseudo_vel,
                    pos_beta, pos_slop, dt, baumgarte_max_velocity);
            }
        }
    }
}

// The per-DOF scatter target: the global link row + spatial component the cooked
// dof maps route this articulation tile's DOF to (comp==~0u => a 1-DOF joint qdot).
__device__ __forceinline__ void ArticDofTarget(
    uint32_t env, uint32_t a, uint32_t k, uint32_t dof_stride,
    uint32_t links_per_dog, uint32_t base_link_count,
    const uint32_t* __restrict__ dof_to_link,
    const uint32_t* __restrict__ dof_to_component,
    uint32_t& comp, size_t& gl) {
    const size_t flat = static_cast<size_t>(env) * dof_stride + k;
    comp = dof_to_component[flat];
    const uint32_t link = dof_to_link[flat] + a * links_per_dog;
    gl = static_cast<size_t>(env) * base_link_count + link;
}

__global__ void SolveRowsBlockIslandKernel(
    const NkRow* __restrict__ urows,
    float* __restrict__ lambda,
    const float* __restrict__ chain_jacobian,
    const float* __restrict__ row_minv_jt,
    const float* __restrict__ chain_jacobian_b,  // side-B (or null)
    const float* __restrict__ row_minv_jt_b,     // side-B (or null)
    const float* __restrict__ row_meff,
    float* __restrict__ qdot_flat,
    Spatial6* __restrict__ link_velocity,
    float* __restrict__ qdot,
    math::Vec3* __restrict__ body_lin_vel,
    math::Vec3* __restrict__ body_ang_vel,
    const float* __restrict__ body_inv_mass,
    const math::Vec3* __restrict__ body_inv_inertia,
    const float* __restrict__ particle_inv_mass,
    math::Vec3* __restrict__ particle_vel,
    const uint32_t* __restrict__ islands,
    const uint32_t* __restrict__ segments,
    const uint32_t* __restrict__ row_order,
    const uint32_t* __restrict__ dof_to_link,
    const uint32_t* __restrict__ dof_to_component,
    uint32_t* __restrict__ pd_solve_scratch,  // PairDriven GLOBAL order+seg scratch
    const float* __restrict__ row_penetration,    // split-impulse depth (or null)
    float* __restrict__ row_pseudo_lambda,        // split-impulse accumulator (or null)
    float* __restrict__ qdot_pseudo,              // per-link pseudo joint vel (or null)
    Spatial6* __restrict__ link_velocity_pseudo,  // per-link pseudo spatial vel (or null)
    float* __restrict__ qdot_pseudo_flat,         // per-artic pseudo tile (or null)
    math::Vec3* __restrict__ body_pseudo_lin_vel, // per-body pseudo lin vel (or null)
    math::Vec3* __restrict__ body_pseudo_ang_vel, // per-body pseudo ang vel (or null)
    math::Vec3* __restrict__ particle_pseudo_vel, // per-particle pseudo vel (or null)
    const uint32_t* __restrict__ island_count_dev, // dynamic island count (or null)
    uint32_t total_islands,
    uint32_t rows_per_env,
    uint32_t dof_stride,
    uint32_t base_link_count,
    uint32_t artics_per_env,    // co-resident artic tiles per env (1 at K==1)
    uint32_t with_b_arm,        // PairDriven carves J_b/w_b shared regions
    uint32_t vel_iters,
    uint32_t pos_iters,
    float pos_beta, float pos_slop,
    float dt,
    float baumgarte_max_velocity) {
    const uint32_t island = blockIdx.x;
    // Dynamic islanding (PairDriven): the live component count is a DEVICE scalar
    // (BuildSolveIslands writes it each step) and the grid is the static max-island
    // bound, so blocks past the live count early-exit. Static schedule: total_islands.
    const bool dynamic = (island_count_dev != nullptr);
    const uint32_t live_islands = dynamic ? *island_count_dev : total_islands;
    if (island >= live_islands) {
        return;
    }
    // Read the island's quad via the named record (same memory as the raw
    // stride-4 layout; reinterpret keeps the schedule array untouched). For the
    // dynamic schedule the quad is {row_off, row_cnt, flags, env} indexing the
    // pre-grouped island_rows (each row its OWN color); static: {seg_off, seg_cnt}.
    const IslandRecord rec =
        reinterpret_cast<const IslandRecord*>(islands)[island];
    const uint32_t seg_off = rec.seg_off;
    const uint32_t seg_cnt = rec.seg_cnt;
    const uint32_t flags = rec.flags;
    const uint32_t env = rec.env;
    // Dynamic rigid/particle-only components are solved by the scalar-island
    // kernel. Static schedules and any component touching an articulation retain
    // this warp path unchanged.
    if (dynamic && !(flags & 1u)) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t env_row_base = env * rows_per_env;
    const uint32_t k_tiles = artics_per_env == 0u ? 1u : artics_per_env;
    const uint32_t env_artic_base = env * k_tiles;  // first global artic of this env

    // DYNAMIC SHARED working set for the 64-iteration sweep (sized by the op
    // entry). Union caches the ENV SLICE\'s lambdas / effective masses / slim
    // records / chain-J rows / M^-1 J^T rows + the island\'s segment + order
    // tables here — all re-read EVERY iteration, the measured latency core of the
    // sweep. PairDriven keeps every row-scaled array in global (its per-env island
    // is too large for shared); only the qdot tile lives here. Loading the WHOLE
    // env slice is read-safe under multiple islands per env; only THIS island\'s
    // rows are written back (its lambdas), so concurrent blocks never collide.
    extern __shared__ unsigned char dyn_sh[];
    // The Union/Fused families CACHE the per-row J/w in shared (with_b_arm == 0 ->
    // cache_jw == true). PairDriven does NOT (its per-env worst-case island can be
    // hundreds of rows; caching J,w,J_b,w_b would blow the ~99 KB shared limit) --
    // it reads J/w/J_b/w_b from GLOBAL each iteration (bounded shared).
    const bool cache_jw = (with_b_arm == 0u);
    // the qdot region. Union: the legacy kMaxArticulationDof reservation (shared
    // footprint byte-for-byte the H1-golden carve). PairDriven: compact K-tile.
    const uint32_t qdot_floats =
        (with_b_arm != 0u) ? (k_tiles * dof_stride) : kMaxArticulationDof;
    float* const qdot_sh = reinterpret_cast<float*>(dyn_sh);
    // Union stages lambda/meff/slim/J/w/order/seg in shared; PairDriven keeps every
    // row-scaled array in GLOBAL (its per-env island can be thousands of rows),
    // leaving ONLY the qdot tile(s) in shared. lambda_sh / meff_sh == null select
    // the global lambda / row_meff source in SolveUnionRowWarp; order_sh / seg_sh
    // point into this env's disjoint slice of the global pd_solve_scratch buffer
    // (rows_per_env order u32 followed by 2*rows_per_env segment u32 per env).
    // Split-impulse position pass (PairDriven only; pos_iters>0). The parallel
    // pseudo qdot tile follows the real qdot tile in shared, sized identically.
    const bool pos_pass = (pos_iters > 0u) && !cache_jw;
    float* const qdot_pseudo_sh = pos_pass ? (qdot_sh + qdot_floats) : nullptr;
    float* lambda_sh = nullptr;
    float* meff_sh = nullptr;
    float* J_sh = nullptr;   // shared J/w cache (Union only).
    float* w_sh = nullptr;
    SlimRow* slim_sh = nullptr;
    uint32_t* order_sh = nullptr;
    uint32_t* seg_sh = nullptr;
    // Dynamic-island per-component tile working set (PairDriven): a present-bitmap +
    // compacted list of THIS component's env-local artic tiles, carved after the
    // qdot region(s). The load/scatter touch ONLY these tiles so concurrent
    // same-env components never race on qdot_flat. Unused by the Union path.
    uint32_t* tile_present = nullptr;
    uint32_t* tile_list = nullptr;
    if (cache_jw) {
        lambda_sh = qdot_sh + qdot_floats;
        meff_sh = lambda_sh + rows_per_env;
        J_sh = meff_sh + rows_per_env;
        w_sh = J_sh + static_cast<size_t>(rows_per_env) * dof_stride;
        slim_sh = reinterpret_cast<SlimRow*>(
            w_sh + static_cast<size_t>(rows_per_env) * dof_stride);
        order_sh = reinterpret_cast<uint32_t*>(slim_sh + rows_per_env);
        seg_sh = order_sh + rows_per_env;                        // 2R u32
    } else {
        float* const tile_base =
            qdot_sh + static_cast<size_t>(qdot_floats) * (pos_pass ? 2u : 1u);
        tile_present = reinterpret_cast<uint32_t*>(tile_base);
        tile_list = tile_present + k_tiles;
        // Static PairDriven fallback (dynamic == false): the legacy GLOBAL
        // order/segment scratch (one block per env). The dynamic path reads the
        // pre-grouped row_order directly and never touches this scratch.
        order_sh = pd_solve_scratch + static_cast<size_t>(3u) * env_row_base;
        seg_sh = order_sh + rows_per_env;                        // 2R u32
    }
    __shared__ uint32_t tile_cnt_sh;

    const bool has_artic = (flags & 1u) != 0u && dof_stride > 0u;
    // Dynamic path: collect THIS component's distinct env-local artic tiles into
    // tile_list (present-bitmap then compact). The component's rows all live in this
    // block (the union-find merged every shared tile), so its tile set is closed.
    if (dynamic && has_artic) {
        for (uint32_t i = lane; i < k_tiles; i += blockDim.x) tile_present[i] = 0u;
        if (lane == 0u) tile_cnt_sh = 0u;
        __syncthreads();
        for (uint32_t r = lane; r < seg_cnt; r += blockDim.x) {
            const NkRow& row = urows[row_order[seg_off + r]];
            if (row.a.kind == kNkSideArtic && row.a.index >= env_artic_base) {
                atomicOr(&tile_present[row.a.index - env_artic_base], 1u);
            }
            if (row.b.kind == kNkSideArtic && row.b.index >= env_artic_base) {
                atomicOr(&tile_present[row.b.index - env_artic_base], 1u);
            }
        }
        __syncthreads();
        for (uint32_t t = lane; t < k_tiles; t += blockDim.x) {
            if (tile_present[t] != 0u) tile_list[atomicAdd(&tile_cnt_sh, 1u)] = t;
        }
        __syncthreads();
    }
    if (has_artic) {
        if (dynamic) {
            // load ONLY this component's tiles (into their env-local qdot_sh slots);
            // the unloaded slots are never read (no component row touches them).
            const uint32_t tc = tile_cnt_sh;
            for (uint32_t u = 0u; u < tc; ++u) {
                const uint32_t tile = tile_list[u];
                for (uint32_t k = lane; k < dof_stride; k += blockDim.x) {
                    qdot_sh[tile * dof_stride + k] = qdot_flat[
                        static_cast<size_t>(env_artic_base + tile) * dof_stride + k];
                }
            }
        } else {
            // load EVERY co-resident articulation tile of this env (K tiles). At
            // K==1 this is the single legacy tile (qdot_flat[env*dof_stride]).
            for (uint32_t i = lane; i < k_tiles * dof_stride; i += blockDim.x) {
                qdot_sh[i] = qdot_flat[static_cast<size_t>(env_artic_base) * dof_stride + i];
            }
        }
    }
    // Union stages lambda / meff / slim records in shared; PairDriven reads lambda /
    // row_meff straight from global and builds slim inline in the sweep (its carve
    // has none of these regions).
    if (cache_jw) {
        for (uint32_t i = lane; i < rows_per_env; i += blockDim.x) {
            const size_t g = static_cast<size_t>(env_row_base) + i;
            lambda_sh[i] = lambda[g];
            meff_sh[i] = row_meff[g];
            slim_sh[i] = MakeSlimRow(urows[g], env_row_base, env_artic_base);
        }
    }
    // Union: cache the per-row J/w into shared (the dense-sweep latency core).
    // PairDriven reads J/w/J_b/w_b from global in the sweep (no shared cache).
    if (cache_jw) {
        for (size_t i = lane; i < static_cast<size_t>(rows_per_env) * dof_stride;
             i += blockDim.x) {
            const size_t g = static_cast<size_t>(env_row_base) * dof_stride + i;
            J_sh[i] = chain_jacobian[g];
            w_sh[i] = row_minv_jt[g];
        }
    }
    // The island\'s rows are CONTIGUOUS in row_order (per-component spans).
    // COMPACT the schedule to the ACTIVE rows once (lane 0): the active set is
    // FIXED for the whole sweep (row flags never change inside the solve), so
    // dropping inactive rows / empty segments here only removes NO-OP visits —
    // the surviving execution order is the schedule\'s order, unchanged.
    __shared__ uint32_t live_seg_cnt_sh;
    if (dynamic) {
        // The dynamic schedule already grouped the component's ACTIVE rows in
        // row_order[seg_off .. seg_off+seg_cnt), each its OWN color (serial GS, the
        // bit-identical single-island order). No compaction / scratch needed.
        if (lane == 0u) live_seg_cnt_sh = seg_cnt;
    } else if (lane == 0u) {
        uint32_t out_rows = 0u;
        uint32_t out_segs = 0u;
        const uint32_t span_start =
            seg_cnt > 0u ? segments[static_cast<size_t>(seg_off) * 2u + 0u] : 0u;
        for (uint32_t s = 0u; s < seg_cnt; ++s) {
            const uint32_t off = segments[static_cast<size_t>(seg_off + s) * 2u + 0u];
            const uint32_t cnt = segments[static_cast<size_t>(seg_off + s) * 2u + 1u];
            const uint32_t seg_begin = out_rows;
            for (uint32_t i = 0u; i < cnt; ++i) {
                const uint32_t gslot = row_order[off + i];
                // slim_sh is not yet visible (no barrier) — read the flag from
                // the GLOBAL record (one u32 per row, once per launch).
                if (urows[gslot].flags & nk::nk_row_flags::kActive) {
                    order_sh[out_rows++] = gslot;
                }
            }
            if (out_rows > seg_begin) {
                seg_sh[2u * out_segs + 0u] = seg_begin;
                seg_sh[2u * out_segs + 1u] = out_rows - seg_begin;
                ++out_segs;
            }
        }
        live_seg_cnt_sh = out_segs;
        (void)span_start;
    }
    __syncthreads();
    const uint32_t live_seg_cnt = live_seg_cnt_sh;

    const uint32_t warp = lane >> 5u;
    const uint32_t wlane = lane & 31u;
    const uint32_t nwarps = blockDim.x >> 5u;
    if (with_b_arm != 0u) {
        for (uint32_t s = 0u; s < live_seg_cnt; ++s) {
            const uint32_t off = dynamic ? (seg_off + s) : seg_sh[2u * s + 0u];
            const uint32_t cnt = dynamic ? 1u : seg_sh[2u * s + 1u];
            const uint32_t* const obase = dynamic ? row_order : order_sh;
            for (uint32_t idx = warp; idx < cnt; idx += nwarps) {
                const uint32_t gslot = obase[off + idx];
                SolveUnionRowWarp(gslot - env_row_base, gslot, env_row_base,
                                  env_artic_base, gslot, wlane, nullptr,
                                  nullptr, nullptr, lambda, row_meff,
                                  chain_jacobian, row_minv_jt,
                                  chain_jacobian_b, row_minv_jt_b,
                                  qdot_sh, urows, body_lin_vel, body_ang_vel,
                                  body_inv_mass, body_inv_inertia,
                                  particle_inv_mass, particle_vel,
                                  dof_stride, dt, true);
            }
            if (dynamic) __syncwarp(); else __syncthreads();
        }
    }
    for (uint32_t it = 0u; it < vel_iters; ++it) {
        for (uint32_t s = 0u; s < live_seg_cnt; ++s) {
            // Dynamic: the s-th unit is one component row at row_order[seg_off+s]
            // (its own color). Static: the s-th compacted color segment in seg_sh.
            const uint32_t off = dynamic ? (seg_off + s) : seg_sh[2u * s + 0u];
            const uint32_t cnt = dynamic ? 1u : seg_sh[2u * s + 1u];
            const uint32_t* const obase = dynamic ? row_order : order_sh;
            // One WARP per row of the color (rows of one color share no
            // mutable state — warp-level parallel is exactly the old
            // thread-level parallel, with the row\'s inner work distributed
            // across the warp\'s lanes).
            for (uint32_t idx = warp; idx < cnt; idx += nwarps) {
                const uint32_t gslot = obase[off + idx];
                // Union: shared J/w cache indexed by env-local slot. PairDriven:
                // GLOBAL J/w/J_b/w_b indexed by the global slot (j_row == gslot).
                const float* Ja = cache_jw ? J_sh : chain_jacobian;
                const float* Wa = cache_jw ? w_sh : row_minv_jt;
                const float* Jb = cache_jw ? nullptr : chain_jacobian_b;
                const float* Wb = cache_jw ? nullptr : row_minv_jt_b;
                const uint32_t j_row = cache_jw ? (gslot - env_row_base) : gslot;
                SolveUnionRowWarp(gslot - env_row_base, gslot, env_row_base,
                                  env_artic_base, j_row, wlane,
                                  cache_jw ? slim_sh : nullptr,
                                  lambda_sh, meff_sh, lambda, row_meff,
                                  Ja, Wa, Jb, Wb,
                                  qdot_sh, urows,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_inv_inertia, particle_inv_mass,
                                  particle_vel, dof_stride, dt, false);
            }
            if (dynamic) __syncwarp(); else __syncthreads();
        }
    }

    // Split-impulse position pass (PairDriven, pos_iters>0): drive a SEPARATE
    // pseudo velocity from the geometric penetration into qdot_pseudo_sh / the
    // pseudo body+particle velocities, never the persisted velocity. Same fixed
    // sweep order (D1). The pseudo qdot tile starts at ZERO (pseudo velocity is
    // built from depth, not carried from the velocity solve).
    if (pos_pass) {
        if (has_artic) {
            for (uint32_t i = lane; i < k_tiles * dof_stride; i += blockDim.x) {
                qdot_pseudo_sh[i] = 0.0f;
            }
        }
        // Zero this island's per-row pseudo accumulators (global, race-free per
        // island). Dynamic: the component rows in row_order[seg_off..); static: the
        // compacted ACTIVE rows in order_sh.
        if (dynamic) {
            for (uint32_t i = lane; i < seg_cnt; i += blockDim.x) {
                row_pseudo_lambda[row_order[seg_off + i]] = 0.0f;
            }
        } else if (live_seg_cnt > 0u) {
            const uint32_t last_off = seg_sh[2u * (live_seg_cnt - 1u) + 0u];
            const uint32_t last_cnt = seg_sh[2u * (live_seg_cnt - 1u) + 1u];
            const uint32_t island_rows = last_off + last_cnt;
            for (uint32_t i = lane; i < island_rows; i += blockDim.x) {
                row_pseudo_lambda[order_sh[i]] = 0.0f;
            }
        }
        __syncthreads();
        for (uint32_t it = 0u; it < pos_iters; ++it) {
            for (uint32_t s = 0u; s < live_seg_cnt; ++s) {
                const uint32_t off = dynamic ? (seg_off + s) : seg_sh[2u * s + 0u];
                const uint32_t cnt = dynamic ? 1u : seg_sh[2u * s + 1u];
                const uint32_t* const obase = dynamic ? row_order : order_sh;
                for (uint32_t idx = warp; idx < cnt; idx += nwarps) {
                    const uint32_t gslot = obase[off + idx];
                    SolvePositionRowWarp(
                        gslot, env_row_base, env_artic_base, gslot, wlane,
                        row_meff, row_penetration, row_pseudo_lambda,
                        chain_jacobian, row_minv_jt,
                        chain_jacobian_b, row_minv_jt_b,
                        qdot_pseudo_sh, urows,
                        body_pseudo_lin_vel, body_pseudo_ang_vel,
                        body_inv_mass, body_inv_inertia,
                        particle_inv_mass, particle_pseudo_vel,
                        dof_stride, pos_beta, pos_slop, dt,
                        baumgarte_max_velocity);
                }
                if (dynamic) __syncwarp(); else __syncthreads();
            }
        }
    }

    // Write back THIS island's lambdas (the persistent warm-start/readout
    // field) — walk the island's own ACTIVE rows (the only lambdas the sweep
    // can change; inactive slots keep the assembled 0). Multi-island-per-env
    // safe: islands never share rows. Union stages lambda in shared, so it copies
    // the slice back here; PairDriven writes lambda[gslot] in-place during the
    // sweep (no shared slice), so its writeback is already done.
    if (cache_jw && live_seg_cnt > 0u) {
        const uint32_t last_off = seg_sh[2u * (live_seg_cnt - 1u) + 0u];
        const uint32_t last_cnt = seg_sh[2u * (live_seg_cnt - 1u) + 1u];
        const uint32_t island_rows = last_off + last_cnt;
        for (uint32_t i = lane; i < island_rows; i += blockDim.x) {
            const uint32_t gslot = order_sh[i];
            lambda[gslot] = lambda_sh[gslot - env_row_base];
        }
    }

    // Scatter the post-solve qdot tile(s) back through the cooked dof maps.
    // with_b_arm == 0 (Union/Fused): the LEGACY single-tile scatter, byte-exact
    // (env tile 0 via the per:dof dof_to_link/component maps). with_b_arm != 0
    // (PairDriven): scatter ALL K co-resident articulation tiles. The cooked
    // dof_to_link/component are per:dof (ONE dog's map); for tile a > 0 the same
    // template-local (link, component) applies to articulation (env_artic_base+a),
    // whose links live at global articulation_link_offset[that artic] + local. At
    // K==1 (single tile) PairDriven and the legacy path land on the SAME slots.
    if (has_artic) {
        if (with_b_arm == 0u) {
            for (uint32_t k = lane; k < dof_stride; k += blockDim.x) {
                const size_t flat = static_cast<size_t>(env) * dof_stride + k;
                const uint32_t link = dof_to_link[flat];
                const uint32_t comp = dof_to_component[flat];
                const size_t gl = static_cast<size_t>(env) * base_link_count + link;
                const float v = qdot_sh[k];
                if (comp != ~0u) {
                    link_velocity[gl].v[comp] = v;
                } else {
                    qdot[gl] = v;
                }
                qdot_flat[flat] = v;
            }
        } else {
            // PairDriven multi-tile scatter. dof_to_link/component are the per:dog
            // template map; the global link of tile a is env*base_link + (link +
            // a*base_link/k_tiles) ONLY if each dog occupies a contiguous link
            // block of base_link/k_tiles. The multi-dog cook lays out K dogs'
            // links contiguously per env (link_to_articulation groups them), so
            // tile a's links are offset by a * links_per_dog. links_per_dog =
            // base_link_count / k_tiles.
            const uint32_t links_per_dog =
                (k_tiles > 0u) ? (base_link_count / k_tiles) : base_link_count;
            // Dynamic: scatter ONLY this component's tiles (tile_list[0..tile_cnt)),
            // so concurrent same-env components never write the same qdot_flat slot.
            // Static: ALL k_tiles. Both use the per-tile template map (a*links_per_dog).
            const uint32_t scatter_tiles = dynamic ? tile_cnt_sh : k_tiles;
            for (uint32_t i = lane; i < scatter_tiles * dof_stride; i += blockDim.x) {
                const uint32_t u = i / dof_stride;       // tile slot in the iteration
                const uint32_t k = i - u * dof_stride;   // DOF within tile
                const uint32_t a = dynamic ? tile_list[u] : u;  // env-local tile
                uint32_t comp; size_t gl;
                ArticDofTarget(env, a, k, dof_stride, links_per_dog, base_link_count,
                               dof_to_link, dof_to_component, comp, gl);
                const float v = qdot_sh[static_cast<size_t>(a) * dof_stride + k];
                if (comp != ~0u) {
                    link_velocity[gl].v[comp] = v;
                } else {
                    qdot[gl] = v;
                }
                qdot_flat[static_cast<size_t>(env_artic_base + a) * dof_stride + k] = v;
                // Split-impulse: scatter the pseudo qdot tile to its OWN buffers,
                // mirroring the real scatter (so IntegratePosition reads it
                // additively without touching the persisted velocity).
                if (pos_pass) {
                    const float vp = qdot_pseudo_sh[static_cast<size_t>(a) * dof_stride + k];
                    if (comp != ~0u) {
                        link_velocity_pseudo[gl].v[comp] = vp;
                    } else {
                        qdot_pseudo[gl] = vp;
                    }
                    qdot_pseudo_flat[static_cast<size_t>(env_artic_base + a) *
                                         dof_stride + k] = vp;
                }
            }
        }
    }
}

// Flush qdot_flat -> link_velocity for an articulation no active row claimed
// (cc_artic_first==sentinel): the static all-tiles scatter did this every step.
__global__ void FlushOrphanArticKernel(
    const uint32_t* __restrict__ cc_artic_first, uint32_t artic_count,
    uint32_t artics_per_env, uint32_t dof_stride, uint32_t base_link_count,
    const float* __restrict__ qdot_flat, Spatial6* __restrict__ link_velocity,
    float* __restrict__ qdot, const uint32_t* __restrict__ dof_to_link,
    const uint32_t* __restrict__ dof_to_component) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= artic_count * dof_stride) return;
    const uint32_t ag = t / dof_stride;
    if (cc_artic_first[ag] != ~0u) return;  // a contact row already solved this tile.
    const uint32_t k = t - ag * dof_stride;
    const uint32_t ape = (artics_per_env == 0u) ? 1u : artics_per_env;
    const uint32_t env = ag / ape;
    const uint32_t a = ag - env * ape;       // env-local tile (env_artic_base + a == ag).
    const uint32_t links_per_dog = (ape > 0u) ? (base_link_count / ape) : base_link_count;
    uint32_t comp; size_t gl;
    ArticDofTarget(env, a, k, dof_stride, links_per_dog, base_link_count,
                   dof_to_link, dof_to_component, comp, gl);
    const float v = qdot_flat[static_cast<size_t>(ag) * dof_stride + k];
    if (comp != ~0u) {
        link_velocity[gl].v[comp] = v;
    } else {
        qdot[gl] = v;
    }
}

// Expose lower/upper impulses independently from contact and actuator telemetry.
__global__ void WriteJointLimitImpulseKernel(
    const float* __restrict__ lambda, uint32_t total_link_count,
    uint32_t env_count, uint32_t base_link_count, uint32_t rows_per_env,
    uint32_t contact_rows_per_env, float* __restrict__ limit_impulse) {
    const uint32_t link = blockIdx.x * blockDim.x + threadIdx.x;
    if (link >= total_link_count) return;
    const uint32_t env = link / base_link_count;
    if (env >= env_count) return;
    const uint32_t local_link = link - env * base_link_count;
    const uint32_t row_base = env * rows_per_env + contact_rows_per_env +
                              local_link * 2u;
    limit_impulse[static_cast<size_t>(link) * 2u] = lambda[row_base];
    limit_impulse[static_cast<size_t>(link) * 2u + 1u] = lambda[row_base + 1u];
}

// --- op entry point ---------------------------------------------------------

Status OpSolveRowsBlockIsland(const ModelView& model, const DataView& data,
                              const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const SolveRowsBlockIslandParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }

    if (p->family == kContactFamilyUnionCsr ||
        p->family == kContactFamilyPairDriven) {
        if (p->total_islands == 0u) {
            return Status::Ok;
        }
        if (p->max_dof > kMaxArticulationDof) {
            return Status::Failed;  // shared qdot tile capacity (legacy cap).
        }
        const bool with_b_arm = (p->family == kContactFamilyPairDriven);
        const uint32_t artics_per_env =
            (p->articulation_count > 0u && p->env_count > 0u)
                ? (p->articulation_count / p->env_count) : 1u;
        // The dynamic CC pass runs for PairDriven; the validation hook forces the
        // cook-time static schedule (the byte-identity reference) instead.
        const bool run_dynamic = with_b_arm && (p->force_static_islands == 0u);
        // Dynamic islanding (PairDriven): the grid is the static MAX-island bound —
        // each component consumes >=1 distinct dynamic entity (the broadphase drops
        // static-static pairs), so #components <= artics + bodies + particles. The
        // kernel reads the LIVE component count from data.island_count and early-exits
        // the surplus blocks. The static schedule keeps its cook-time island count.
        const uint32_t max_island_bound =
            p->articulation_count + p->total_body_count + p->total_particle_count;
        const uint32_t grid_islands = run_dynamic ? max_island_bound : p->total_islands;
        if (grid_islands == 0u) {
            return Status::Ok;
        }
        // qdot region: legacy kMaxArticulationDof reservation for UnionCsr (the
        // H1-golden footprint), the compact K-tile reservation for PairDriven .
        const uint32_t qdot_floats =
            with_b_arm ? (artics_per_env * p->max_dof) : kMaxArticulationDof;
        // Split-impulse position pass runs ONLY on the PairDriven path (pos_iters>0).
        const bool pos_pass = (p->pos_iters > 0u) && with_b_arm;
        // Union caches J/w in shared (cache_jw == !with_b_arm); PairDriven reads
        // them from global (bounded shared), so its carve excludes the J/w regions.
        // The position pass adds a parallel pseudo qdot tile + the dynamic-island
        // per-component tile working set (PairDriven only).
        const size_t shared_bytes = IslandSharedBytes(p->rows_per_env, p->max_dof,
                                                      qdot_floats, !with_b_arm,
                                                      pos_pass, artics_per_env);
        // One-time opt-in past the 48 KB default dynamic-shared carve-out
        // (sm_8x+ allows ~99 KB/block). Host-side attribute set — NOT a
        // stream op, so it is graph-capture-safe; cached so the steady-state
        // dispatch (and the captured plan) never re-issues it.
        {
            static size_t opted_in_bytes = 0;
            if (shared_bytes > opted_in_bytes) {
                if (cudaFuncSetAttribute(
                        SolveRowsBlockIslandKernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(shared_bytes)) != cudaSuccess) {
                    return Status::Failed;  // LOUD: slice exceeds the device limit.
                }
                opted_in_bytes = shared_bytes;
            }
        }
        // Zero every GLOBAL pseudo accumulator (rigid/particle/articulation): the
        // dynamic schedule rewrites only solved tiles, so a dropped tile reads 0 here.
        if (pos_pass) {
            if (p->total_body_count > 0u && data.body_pseudo_linear_velocity != nullptr) {
                const size_t bbytes =
                    static_cast<size_t>(p->total_body_count) * sizeof(math::Vec3);
                if (cudaMemsetAsync(data.body_pseudo_linear_velocity, 0, bbytes,
                                    stream) != cudaSuccess ||
                    cudaMemsetAsync(data.body_pseudo_angular_velocity, 0, bbytes,
                                    stream) != cudaSuccess) {
                    return Status::Failed;
                }
            }
            if (p->total_particle_count > 0u && data.particle_pseudo_vel != nullptr) {
                const size_t pbytes =
                    static_cast<size_t>(p->total_particle_count) * sizeof(math::Vec3);
                if (cudaMemsetAsync(data.particle_pseudo_vel, 0, pbytes, stream) !=
                    cudaSuccess) {
                    return Status::Failed;
                }
            }
            const size_t total_link_count =
                static_cast<size_t>(p->base_link_count) * p->env_count;
            const size_t total_artic_dof =
                static_cast<size_t>(p->articulation_count) * p->max_dof;
            if (total_link_count > 0u && data.qdot_pseudo != nullptr) {
                if (cudaMemsetAsync(data.qdot_pseudo, 0,
                                    total_link_count * sizeof(float), stream) !=
                        cudaSuccess ||
                    cudaMemsetAsync(data.link_velocity_pseudo, 0,
                                    total_link_count * sizeof(Spatial6), stream) !=
                        cudaSuccess) {
                    return Status::Failed;
                }
            }
            if (total_artic_dof > 0u && data.qdot_pseudo_flat != nullptr) {
                if (cudaMemsetAsync(data.qdot_pseudo_flat, 0,
                                    total_artic_dof * sizeof(float), stream) !=
                        cudaSuccess) {
                    return Status::Failed;
                }
            }
        }
        // Dynamic schedule: the per-step CC pass (island_quads/rows + island_count).
        // Static schedule: the cook-time arrays (model.island_row_offsets/row_order).
        const uint32_t* const islands_in = run_dynamic
            ? data.island_quads : static_cast<const uint32_t*>(model.island_row_offsets);
        const uint32_t* const row_order_in = run_dynamic
            ? data.island_rows : static_cast<const uint32_t*>(model.row_order);
        const uint32_t* const island_count_dev = run_dynamic ? data.island_count : nullptr;
        if (run_dynamic) {
            const uint32_t scalar_blocks = max_island_bound < kScalarIslandGridBlocks
                ? max_island_bound : kScalarIslandGridBlocks;
            LaunchCuda(
                SolveRowsScalarIslandsKernel, dim3(scalar_blocks),
                dim3(kScalarIslandBlockSize), 0u, stream,
                reinterpret_cast<const NkRow*>(data.urows), data.lambda,
                static_cast<const float*>(data.row_meff),
                data.body_linear_velocity, data.body_angular_velocity,
                static_cast<const float*>(data.body_inv_mass),
                static_cast<const math::Vec3*>(data.body_inv_inertia),
                static_cast<const float*>(data.particle_inv_mass), data.particle_vel,
                data.island_quads, data.island_rows,
                pos_pass ? static_cast<const float*>(data.row_penetration) : nullptr,
                pos_pass ? static_cast<float*>(data.row_pseudo_lambda) : nullptr,
                pos_pass ? data.body_pseudo_linear_velocity : nullptr,
                pos_pass ? data.body_pseudo_angular_velocity : nullptr,
                pos_pass ? data.particle_pseudo_vel : nullptr,
                data.island_count, p->rows_per_env, artics_per_env,
                static_cast<uint32_t>(p->vel_iters),
                static_cast<uint32_t>(p->pos_iters),
                p->pos_beta, p->pos_slop, p->dt,
                p->baumgarte_max_velocity);
        }
        const uint32_t island_block_size = with_b_arm
            ? kPairDrivenIslandBlockSize : kUnionIslandBlockSize;
        LaunchCuda(SolveRowsBlockIslandKernel, dim3(grid_islands),
                   dim3(island_block_size), static_cast<uint32_t>(shared_bytes),
                   stream,
                   reinterpret_cast<const NkRow*>(data.urows),
                   data.lambda,
                   static_cast<const float*>(data.chain_jacobian),
                   static_cast<const float*>(data.row_minv_jt),
                   with_b_arm ? static_cast<const float*>(data.chain_jacobian_b)
                              : nullptr,
                   with_b_arm ? static_cast<const float*>(data.row_minv_jt_b)
                              : nullptr,
                   static_cast<const float*>(data.row_meff),
                   data.qdot_flat,
                   reinterpret_cast<Spatial6*>(data.link_velocity),
                   data.qdot,
                   data.body_linear_velocity,
                   data.body_angular_velocity,
                   static_cast<const float*>(data.body_inv_mass),
                   static_cast<const math::Vec3*>(data.body_inv_inertia),
                   static_cast<const float*>(data.particle_inv_mass),
                   data.particle_vel,
                   islands_in,
                   static_cast<const uint32_t*>(model.island_color_segments),
                   row_order_in,
                   static_cast<const uint32_t*>(model.dof_to_link),
                   static_cast<const uint32_t*>(model.dof_to_component),
                   data.pd_solve_scratch,
                   pos_pass ? static_cast<const float*>(data.row_penetration) : nullptr,
                   pos_pass ? static_cast<float*>(data.row_pseudo_lambda) : nullptr,
                   pos_pass ? static_cast<float*>(data.qdot_pseudo) : nullptr,
                   pos_pass ? reinterpret_cast<Spatial6*>(data.link_velocity_pseudo) : nullptr,
                   pos_pass ? static_cast<float*>(data.qdot_pseudo_flat) : nullptr,
                   pos_pass ? data.body_pseudo_linear_velocity : nullptr,
                   pos_pass ? data.body_pseudo_angular_velocity : nullptr,
                   pos_pass ? data.particle_pseudo_vel : nullptr,
                   island_count_dev,
                   p->total_islands, p->rows_per_env, p->max_dof,
                   p->base_link_count, artics_per_env,
                   with_b_arm ? 1u : 0u,
                   static_cast<uint32_t>(p->vel_iters),
                   static_cast<uint32_t>(p->pos_iters),
                   p->pos_beta, p->pos_slop, p->dt,
                   p->baumgarte_max_velocity);
        // Flush the articulation tiles the dynamic schedule dropped (the static path
        // scatters all tiles in-kernel). cc_artic_first is BuildSolveIslands' claim table.
        if (run_dynamic && p->articulation_count > 0u && p->max_dof > 0u &&
            data.cc_artic_first != nullptr) {
            const uint32_t flush_threads = p->articulation_count * p->max_dof;
            const uint32_t flush_blocks =
                (flush_threads + kPairDrivenIslandBlockSize - 1u) /
                kPairDrivenIslandBlockSize;
            LaunchCuda(FlushOrphanArticKernel, dim3(flush_blocks),
                       dim3(kPairDrivenIslandBlockSize), 0u, stream,
                       data.cc_artic_first, p->articulation_count, artics_per_env,
                       p->max_dof, p->base_link_count, data.qdot_flat,
                       reinterpret_cast<Spatial6*>(data.link_velocity), data.qdot,
                       static_cast<const uint32_t*>(model.dof_to_link),
                       static_cast<const uint32_t*>(model.dof_to_component));
        }
        if (with_b_arm && p->base_link_count > 0u &&
            p->rows_per_env >= p->contact_rows_per_env + p->base_link_count * 2u &&
            data.joint_limit_impulse != nullptr) {
            const uint32_t total_links = p->base_link_count * p->env_count;
            const uint32_t blocks =
                (total_links + kPairDrivenIslandBlockSize - 1u) /
                kPairDrivenIslandBlockSize;
            LaunchCuda(WriteJointLimitImpulseKernel, dim3(blocks),
                       dim3(kPairDrivenIslandBlockSize), 0u, stream,
                       data.lambda, total_links, p->env_count,
                       p->base_link_count, p->rows_per_env,
                       p->contact_rows_per_env, data.joint_limit_impulse);
        }
        return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
    }

    // the legacy FUSED family was deleted. The only remaining families are
    // UnionCsr / PairDriven (handled above); anything else is an unconfigured
    // contact family with nothing to solve.
    return Status::Ok;
}

} // namespace

void RegisterNkSolveRowsOps() {
    SetCudaOp(NkOp::SolveRowsBlockIsland, &OpSolveRowsBlockIsland);
}

} // namespace nuka::phi
