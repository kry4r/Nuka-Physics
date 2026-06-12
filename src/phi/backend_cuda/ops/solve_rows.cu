// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M4 SolveRowsBlockIsland (the §3.4 device-resident
// unified row solve). TWO contact families behind the ONE op (params->family):
//
//   FUSED (kContactFamilyFusedFoot) — the legacy fused block-per-articulation
//   PGS `SolveArticulatedContactRows`, MOVED VERBATIM from the deleted
//   transitional ops/solve_articulated.cu (the line-by-line port of
//   articulation_contacts.cu; incl. the implicit joint-damping seed, the
//   warm-start apply, the 48-iteration normal/tangent sweeps and the final
//   normal sweep). The foot goldens (stand_5s / foot_contact / foot_box) pin
//   this path byte-exact — it IS those legacy rows' row math, preserved.
//
//   UNION (kContactFamilyUnionCsr) — THE §3.4 KERNEL: grid = total_islands
//   (= N_env x islands/env from the build-time SolveSchedule), block = 256;
//   per block:
//     for it in vel_iters:
//       for c in colors(island):           // device-resident segment table
//         rows of color c in parallel (thread/row) -> __syncthreads()
//   Per-row math = row_solver.cu's COMPLIANT branch, preserved numerically:
//     * jv          = per-side dispatch (CompliantSideConstraintVelocity):
//                     rigid: dot(jlin,v)+dot(jang,w); artic: serial
//                     sum_r J[r]*qdot[r] (ascending r, the legacy order);
//                     particle: dot(jlin,v); static: 0. Side a then b.
//     * update      = lambda_new = clamp(lambda + meff*(rhs*dt - jv
//                     - R*lambda), bounds) — incl. the C5c-2 regularizer
//                     feedback (-R*lambda) the legacy kernel carries.
//     * friction    = the unilateral coupled-pyramid bound
//                     [0, mu * TotalNormalLambda(group)] (IsCompliantFriction-
//                     Row semantics; the group sum over the FIXED normal slots
//                     equals the legacy compacted sum — inactive slots carry
//                     lambda == 0).
//     * apply       = per-side dispatch: rigid v += jlin*(im*dl), w += jang*
//                     invI*dl (skip im<=0); artic qdot[r] += w[r]*dl with the
//                     PRECOMPUTED w = M^-1 J^T (the ArticulationApplyImpulse
//                     inner product hoisted to AssembleRows — same products,
//                     same order); particle v += jlin*(im*dl). |dl| > 1e-12
//                     gate, side a then b. meff is the assembly-hoisted
//                     ComputeCompliantEffectiveMass (constant across iters).
//   The articulation qdot tile lives in SHARED memory for the island's env
//   (loaded from qdot_flat before the sweep, scattered back to link_velocity /
//   qdot through the cooked dof maps after — the legacy pack/scatter 1:1; a
//   contact-free env scatters its own unchanged values, a no-op). 51-DOF
//   M^-1 J^T is the per-thread register loop over the row's coalesced
//   chain_jacobian / row_minv_jt segments (fields.yaml layout note).
//   Watermark early-exit: an inactive row slot (flags bit0 clear) returns
//   immediately — max grid + early exit keeps the kernel graph-capturable
//   (plan §3.3 capacity policy).
//   pos_iters is accepted but unused: every union row is COMPLIANT, and the
//   legacy compliant path skips Baumgarte position projection entirely
//   (SolvePositionRow returns 0 for Compliant rows; the legacy union runs
//   position_iterations = 0) — preserved semantics, not an omission.
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

constexpr uint32_t kInvalidLink = ~0u;

namespace mg = ::nuka::math::gpu;

__forceinline__ __device__ float Dot3(math::Vec3 a, math::Vec3 b) {
    return mg::Dot(a, b);
}

// ===========================================================================
// FUSED-family kernel — MOVED VERBATIM from the transitional
// ops/solve_articulated.cu (M3b), which this file replaces. (See
// articulation_contacts.cu for the full algorithm + determinism contract.)
// ===========================================================================
__global__ void SolveArticulatedContactRowsKernel(ArticulationDeviceState state,
                                                  const ArticulatedContactRow* rows,
                                                  const float* normal_jacobian,
                                                  const float* tangent1_jacobian,
                                                  const float* tangent2_jacobian,
                                                  const float* inertia_M_inv,
                                                  uint32_t dof_stride,
                                                  float dt,
                                                  float friction_coefficient,
                                                  float baumgarte_max_velocity,
                                                  float* inout_lambda,
                                                  const float* joint_damping) {
    const uint32_t articulation = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (articulation >= state.articulation_count || lane != 0u) {
        return;
    }

    const uint32_t offset = state.articulation_link_offset[articulation];
    const uint32_t count = state.articulation_link_count[articulation];

    // dof_to_link[k] / dof_to_component[k] (see articulation_contacts.cu).
    uint32_t dof_to_link[kMaxArticulationDof];
    uint32_t dof_to_component[kMaxArticulationDof];
    uint32_t dof = 0u;
    for (uint32_t local = 0u; local < count && dof < kMaxArticulationDof; ++local) {
        const uint32_t link = offset + local;
        const ArticulationJointType type = state.joint_type[link];
        if (local == 0u && state.parent_link[link] == kInvalidLink &&
            type == ArticulationJointType::FloatingBase) {
            for (uint32_t b = 0u; b < 6u && dof < kMaxArticulationDof; ++b) {
                dof_to_link[dof] = link;
                dof_to_component[dof] = b;
                ++dof;
            }
            continue;
        }
        if (JointDofCountDevice(type) != 0u) {
            dof_to_link[dof] = link;
            dof_to_component[dof] = kInvalidLink;
            ++dof;
        }
    }
    if (dof == 0u || dof_stride == 0u) {
        return;
    }

    // Working joint-velocity vector. Base DOFs seed from link_velocity[root].v
    // (the omega-first base spatial velocity); scalar joint DOFs from state.qdot.
    float qdot_work[kMaxArticulationDof];
    for (uint32_t k = 0u; k < dof; ++k) {
        if (dof_to_component[k] != kInvalidLink) {
            qdot_work[k] = state.link_velocity[dof_to_link[k]].v[dof_to_component[k]];
        } else {
            qdot_work[k] = state.qdot[dof_to_link[k]];
        }
    }

    const size_t tile_stride = static_cast<size_t>(dof_stride) * dof_stride;
    const float* const Minv =
        inertia_M_inv + static_cast<size_t>(articulation) * tile_stride;

    // -- General implicit joint viscous damping (deferred from the drive). ----
    // (See articulation_contacts.cu for the backward-Euler derivation.)
    if (joint_damping != nullptr && dt > 0.0f) {
        float c_qdot[kMaxArticulationDof];
        for (uint32_t k = 0u; k < dof; ++k) {
            const float c = (dof_to_component[k] == kInvalidLink)
                                ? joint_damping[dof_to_link[k]]
                                : 0.0f;
            c_qdot[k] = c * qdot_work[k];  // C * qdot_half (qdot_half = seeded qdot_work)
        }
        for (uint32_t r = 0u; r < dof; ++r) {
            float acc = 0.0f;
            const float* const minv_row = Minv + static_cast<size_t>(r) * dof_stride;
            for (uint32_t c = 0u; c < dof; ++c) {
                acc += minv_row[c] * c_qdot[c];
            }
            qdot_work[r] -= dt * acc;
        }
    }

    // This articulation's contact slots (env-major; env == articulation in the
    // 1-articulation-per-env design).
    const uint32_t slot_base = articulation * kMaxFootContactsPerEnv;
    float lambda_n[kMaxFootContactsPerEnv];
    float lambda_t1[kMaxFootContactsPerEnv];
    float lambda_t2[kMaxFootContactsPerEnv];
    for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
        float ln = 0.0f;
        float lt1 = 0.0f;
        float lt2 = 0.0f;
        if (inout_lambda != nullptr) {
            const size_t base = static_cast<size_t>(slot_base + s) * 3u;
            ln = inout_lambda[base + 0u];
            lt1 = inout_lambda[base + 1u];
            lt2 = inout_lambda[base + 2u];
        }
        lambda_n[s] = ln;
        lambda_t1[s] = lt1;
        lambda_t2[s] = lt2;
    }

    const float inv_dt = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
    const float mu = fmaxf(friction_coefficient, 0.0f);

    // Apply any warm-started impulse so qdot_work reflects the seed lambda before
    // the first sweep (qdot_work += M^-1 J^T * actual).
    for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
        const ArticulatedContactRow row = rows[slot_base + s];
        if (row.row_type != ContactRowType::kRowNormal ||
            row.articulation != articulation) {
            continue;
        }
        const size_t row_base = static_cast<size_t>(slot_base + s) * dof_stride;
        const float* const jrows[3] = {normal_jacobian + row_base,
                                       tangent1_jacobian + row_base,
                                       tangent2_jacobian + row_base};
        const float seeds[3] = {lambda_n[s], lambda_t1[s], lambda_t2[s]};
        for (uint32_t which = 0u; which < 3u; ++which) {
            const float actual = seeds[which];
            if (actual == 0.0f) {
                continue;
            }
            const float* const jac_row = jrows[which];
            for (uint32_t r = 0u; r < dof; ++r) {
                float acc = 0.0f;
                const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                for (uint32_t c = 0u; c < dof; ++c) {
                    acc += minv_row[c] * jac_row[c];
                }
                qdot_work[r] += acc * actual;
            }
        }
    }

    for (uint32_t iter = 0u; iter < kContactSolverIterations; ++iter) {
        // -- Normal rows first, fixed slot order. ---------------------------
        for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
            const ArticulatedContactRow row = rows[slot_base + s];
            if (row.row_type != ContactRowType::kRowNormal ||
                row.articulation != articulation) {
                continue;
            }
            const float* const jn =
                normal_jacobian + static_cast<size_t>(slot_base + s) * dof_stride;
            const float bias = fminf(
                kBaumgarteBeta * inv_dt * fmaxf(row.depth - kPenetrationSlop, 0.0f),
                baumgarte_max_velocity);
            float jv = 0.0f;
            for (uint32_t k = 0u; k < dof; ++k) {
                jv += jn[k] * qdot_work[k];
            }
            const float delta = row.effective_mass * (bias - jv);
            const float old_lambda = lambda_n[s];
            // Normal impulse: lower = 0, upper = +inf.
            float new_lambda = old_lambda + delta;
            if (new_lambda < 0.0f) {
                new_lambda = 0.0f;
            }
            const float actual = new_lambda - old_lambda;
            lambda_n[s] = new_lambda;
            if (actual != 0.0f) {
                for (uint32_t r = 0u; r < dof; ++r) {
                    float acc = 0.0f;
                    const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                    for (uint32_t c = 0u; c < dof; ++c) {
                        acc += minv_row[c] * jn[c];
                    }
                    qdot_work[r] += acc * actual;
                }
            }
        }
        // -- Friction tangent rows, fixed slot order. -----------------------
        for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
            const ArticulatedContactRow row = rows[slot_base + s];
            if (row.row_type != ContactRowType::kRowNormal ||
                row.articulation != articulation) {
                continue;
            }
            const float limit = mu * fmaxf(lambda_n[s], 0.0f);
            const size_t row_base = static_cast<size_t>(slot_base + s) * dof_stride;
            // Tangent 1 then tangent 2, fixed order.
            for (uint32_t which = 0u; which < 2u; ++which) {
                const float* const jt =
                    (which == 0u ? tangent1_jacobian : tangent2_jacobian) + row_base;
                const float meff = (which == 0u) ? row.effective_mass_t1
                                                 : row.effective_mass_t2;
                float jv = 0.0f;
                for (uint32_t k = 0u; k < dof; ++k) {
                    jv += jt[k] * qdot_work[k];
                }
                const float delta = meff * (0.0f - jv);
                const float old_lambda = (which == 0u) ? lambda_t1[s] : lambda_t2[s];
                float new_lambda = old_lambda + delta;
                new_lambda = fminf(fmaxf(new_lambda, -limit), limit);
                const float actual = new_lambda - old_lambda;
                if (which == 0u) {
                    lambda_t1[s] = new_lambda;
                } else {
                    lambda_t2[s] = new_lambda;
                }
                if (actual != 0.0f) {
                    for (uint32_t r = 0u; r < dof; ++r) {
                        float acc = 0.0f;
                        const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                        for (uint32_t c = 0u; c < dof; ++c) {
                            acc += minv_row[c] * jt[c];
                        }
                        qdot_work[r] += acc * actual;
                    }
                }
            }
        }
    }

    // Final normal-row sweep (see articulation_contacts.cu for the rationale).
    for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
        const ArticulatedContactRow row = rows[slot_base + s];
        if (row.row_type != ContactRowType::kRowNormal ||
            row.articulation != articulation) {
            continue;
        }
        const float* const jn =
            normal_jacobian + static_cast<size_t>(slot_base + s) * dof_stride;
        const float bias = fminf(
            kBaumgarteBeta * inv_dt * fmaxf(row.depth - kPenetrationSlop, 0.0f),
            baumgarte_max_velocity);
        float jv = 0.0f;
        for (uint32_t k = 0u; k < dof; ++k) {
            jv += jn[k] * qdot_work[k];
        }
        const float delta = row.effective_mass * (bias - jv);
        const float old_lambda = lambda_n[s];
        float new_lambda = old_lambda + delta;
        if (new_lambda < 0.0f) {
            new_lambda = 0.0f;
        }
        const float actual = new_lambda - old_lambda;
        lambda_n[s] = new_lambda;
        if (actual != 0.0f) {
            for (uint32_t r = 0u; r < dof; ++r) {
                float acc = 0.0f;
                const float* minv_row = Minv + static_cast<size_t>(r) * dof_stride;
                for (uint32_t c = 0u; c < dof; ++c) {
                    acc += minv_row[c] * jn[c];
                }
                qdot_work[r] += acc * actual;
            }
        }
    }

    // Write the corrected velocity back for every DOF of the articulation.
    for (uint32_t k = 0u; k < dof; ++k) {
        if (dof_to_component[k] != kInvalidLink) {
            state.link_velocity[dof_to_link[k]].v[dof_to_component[k]] = qdot_work[k];
        } else {
            state.qdot[dof_to_link[k]] = qdot_work[k];
        }
    }

    // Persist this step's impulses for the next step's warm start.
    if (inout_lambda != nullptr) {
        for (uint32_t s = 0u; s < kMaxFootContactsPerEnv; ++s) {
            const size_t base = static_cast<size_t>(slot_base + s) * 3u;
            inout_lambda[base + 0u] = lambda_n[s];
            inout_lambda[base + 1u] = lambda_t1[s];
            inout_lambda[base + 2u] = lambda_t2[s];
        }
    }
}

// ===========================================================================
// UNION-family island kernel (M4 NEW — the §3.4 spec kernel).
// ===========================================================================

// Island block size. §3.4 sketched 256; MEASURED at N=1 (one island, ~170
// colors x 64 iters = ~11k __syncthreads on the critical path) a small block
// wins decisively — the per-color parallel width is 1-2 rows (each handled by
// a WARP), so wide blocks only pay barrier latency. 64 = 2 row-warps, the
// measured sweet spot; the grid (= islands = N_env x components) is what
// fills the GPU at batch (plan §3.4's own scaling argument).
constexpr uint32_t kIslandBlockSize = 64u;

// Slim per-row SOLVE record (16 f32 = 64 B), built IN-KERNEL at launch from
// the NkRow records. Rationale (MEASURED): the sweep re-reads its row record
// every iteration; broadcasting the full 128 B NkRow to every lane spilled to
// local memory and dominated the bookkeeping cost. The slim record carries
// exactly what the velocity update needs: flags / group (re-based env-LOCAL) /
// compliance terms / the side dispatch folded to {artic bits + at most ONE
// dynamic (rigid|particle) side}. A row with TWO dynamic sides (the M6
// particle x particle class — never emitted by the M4 assembly) sets the
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
};
static_assert(sizeof(SlimRow) == 16 * sizeof(float), "SlimRow must be 64 B");
constexpr uint32_t kSlimAArt = 1u << 0;
constexpr uint32_t kSlimBArt = 1u << 1;
constexpr uint32_t kSlimHasDyn = 1u << 2;
constexpr uint32_t kSlimDynIsB = 1u << 3;
constexpr uint32_t kSlimDynParticle = 1u << 4;
constexpr uint32_t kSlimFallback = 1u << 5;  // two dynamic sides: read NkRow

// The island kernel's DYNAMIC shared working-set size for a given env slice
// (see the kernel's layout carve): qdot tile + lambda + meff + slim records +
// J + w (R x D each) + order + segments. The 51-DOF / 190-row union scene
// needs ~92 KB — within sm_8x+'s opt-in dynamic shared (set once via
// cudaFuncAttributeMaxDynamicSharedMemorySize). A model whose slice exceeds
// the device limit fails LOUDLY at the op entry (Model capacity property;
// the M5 broadphase-cooked sizing revisits it).
inline size_t IslandSharedBytes(uint32_t rows_per_env, uint32_t dof_stride) {
    return sizeof(float) *
               (kMaxArticulationDof +                                   // qdot
                2ull * rows_per_env +                                   // lambda+meff
                2ull * rows_per_env * dof_stride) +                     // J + w
           sizeof(SlimRow) * rows_per_env +                             // slim
           sizeof(uint32_t) * 3ull * rows_per_env;                      // order+segs
}

// One row's velocity update — row_solver.cu SolveCompliantVelocityRow,
// preserved numerically (see the file header), executed by ONE WARP:
//   * the Jv reduction stays the LEGACY SERIAL ascending-r loop on lane 0
//     over the SHARED J slice (a tree/warp reduction would change the
//     summation order),
//   * the lambda update (bounds, clamp, regularizer feedback) is lane 0,
//   * the M^-1 J^T apply is warp-PARALLEL over the dof elements (each
//     qdot[r] += w[r]*delta is one independent multiply-add — identical
//     rounding regardless of which lane executes it; w is shared-cached).
__device__ void SolveUnionRowWarp(uint32_t ls,            // env-local slot
                                  uint32_t gslot,         // global slot
                                  uint32_t wlane,
                                  const SlimRow* slim_sh,
                                  float* lambda_sh,
                                  const float* meff_sh,
                                  const float* J_sh,
                                  const float* w_sh,
                                  float* qdot_sh,
                                  const NkRow* __restrict__ urows,
                                  math::Vec3* __restrict__ body_lin_vel,
                                  math::Vec3* __restrict__ body_ang_vel,
                                  const float* __restrict__ body_inv_mass,
                                  const math::Vec3* __restrict__ body_inv_inertia,
                                  const float* __restrict__ particle_inv_mass,
                                  math::Vec3* __restrict__ particle_vel,
                                  uint32_t dof_stride,
                                  float dt) {
    const SlimRow* const sr = slim_sh + ls;
    const uint32_t flags = sr->flags;
    if (!(flags & nk::nk_row_flags::kActive)) {
        return;  // watermark early-exit (inactive slot).
    }
    const uint32_t code = sr->code;

    float delta = 0.0f;
    if (wlane == 0u) {
        // jv contributions, then summed in the LEGACY a-then-b side order.
        float art_jv = 0.0f;
        if (code & (kSlimAArt | kSlimBArt)) {
            // Reduced-coordinate Jv = sum_r J[r]*qdot[r], ascending r (the
            // legacy serial order — NOT a warp/tree reduction; the chain is
            // the union solve's irreducible serial core).
            const float* const J = J_sh + static_cast<size_t>(ls) * dof_stride;
            for (uint32_t r = 0u; r < dof_stride; ++r) {
                art_jv += J[r] * qdot_sh[r];
            }
        }
        float dyn_jv = 0.0f;
        if (code & kSlimHasDyn) {
            const math::Vec3 jl{sr->jl[0], sr->jl[1], sr->jl[2]};
            if (code & kSlimDynParticle) {
                dyn_jv = (particle_vel != nullptr)
                             ? Dot3(jl, particle_vel[sr->dyn_index])
                             : 0.0f;
            } else {
                const math::Vec3 ja{sr->ja[0], sr->ja[1], sr->ja[2]};
                dyn_jv = Dot3(jl, body_lin_vel[sr->dyn_index]) +
                         Dot3(ja, body_ang_vel[sr->dyn_index]);
            }
        }
        float jv = 0.0f;
        if (code & kSlimAArt) jv += art_jv;
        else if ((code & kSlimHasDyn) && !(code & kSlimDynIsB)) jv += dyn_jv;
        if (code & kSlimBArt) jv += art_jv;
        else if ((code & kSlimHasDyn) && (code & kSlimDynIsB)) jv += dyn_jv;
        if (code & kSlimFallback) {
            // Two-dynamic-side row (M6 class): the slim record carries only
            // side a; read the full record and add side b's jv (rare path).
            const NkRow row = urows[gslot];
            if (row.b.kind == kNkSideParticle && particle_vel != nullptr) {
                jv += Dot3(row.b.jlin, particle_vel[row.b.index]);
            } else if (row.b.kind == kNkSideRigid) {
                jv += Dot3(row.b.jlin, body_lin_vel[row.b.index]) +
                      Dot3(row.b.jang, body_ang_vel[row.b.index]);
            }
        }

        const float effective_mass = meff_sh[ls];
        const float old_impulse = lambda_sh[ls];
        // rhs holds aref (a reference ACCELERATION); the velocity-impulse PGS
        // scales by dt. The -R*old_impulse regularizer feedback matches the
        // (A+R) denominator (the C5c-2 fix) — both legacy-preserved.
        const float rhs_v = sr->rhs * dt;
        const float lambda_inc =
            effective_mass * (rhs_v - jv - sr->R * old_impulse);
        float lower = sr->lower;
        float upper = sr->upper;
        if (flags & nk::nk_row_flags::kFriction) {
            // Unilateral coupled-pyramid edge: [0, mu*TotalNormalLambda].
            // Inactive normal slots carry lambda == 0 -> the fixed-span sum
            // equals the legacy compacted-group sum.
            float total = 0.0f;
            for (uint32_t g = 0u; g < sr->group_cnt; ++g) {
                total += fmaxf(lambda_sh[sr->group_local + g], 0.0f);
            }
            lower = 0.0f;
            upper = fmaxf(sr->mu, 0.0f) * total;
        }

        const float new_impulse =
            fminf(fmaxf(old_impulse + lambda_inc, lower), upper);
        lambda_sh[ls] = new_impulse;
        const float d = new_impulse - old_impulse;
        // The legacy |delta| > 1e-12 apply gate, folded into the broadcast.
        delta = (fabsf(d) > 1.0e-12f) ? d : 0.0f;
    }
    delta = __shfl_sync(0xffffffffu, delta, 0);
    if (delta != 0.0f) {
        // side a then b (the legacy apply order). The artic arm is the
        // dof-wide element-independent apply — warp-parallel; the rigid /
        // particle arms touch a handful of scalars — lane 0.
        for (int side = 0; side < 2; ++side) {
            const bool art = side == 0 ? (code & kSlimAArt) != 0u
                                       : (code & kSlimBArt) != 0u;
            const bool dyn = (code & kSlimHasDyn) &&
                             ((side == 1) == ((code & kSlimDynIsB) != 0u));
            if (art) {
                // qdot += w * delta with the PRECOMPUTED w = M^-1 J^T
                // (acc == w[r] bit-equal; per-element product+add identical
                // whatever lane executes it).
                const float* const w =
                    w_sh + static_cast<size_t>(ls) * dof_stride;
                for (uint32_t r = wlane; r < dof_stride; r += 32u) {
                    qdot_sh[r] += w[r] * delta;
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

__global__ void SolveRowsBlockIslandKernel(
    const NkRow* __restrict__ urows,
    float* __restrict__ lambda,
    const float* __restrict__ chain_jacobian,
    const float* __restrict__ row_minv_jt,
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
    uint32_t total_islands,
    uint32_t rows_per_env,
    uint32_t dof_stride,
    uint32_t base_link_count,
    uint32_t vel_iters,
    float dt) {
    const uint32_t island = blockIdx.x;
    if (island >= total_islands) {
        return;
    }
    const uint32_t seg_off = islands[static_cast<size_t>(island) * 4u + 0u];
    const uint32_t seg_cnt = islands[static_cast<size_t>(island) * 4u + 1u];
    const uint32_t flags = islands[static_cast<size_t>(island) * 4u + 2u];
    const uint32_t env = islands[static_cast<size_t>(island) * 4u + 3u];
    const uint32_t lane = threadIdx.x;
    const uint32_t env_row_base = env * rows_per_env;

    // DYNAMIC SHARED working set for the 64-iteration sweep (sized by the op
    // entry: qdot tile + the ENV SLICE\'s lambdas / effective masses / slim
    // records / chain-J rows / M^-1 J^T rows + the island\'s segment + order
    // tables — all re-read EVERY iteration, the measured latency core of the
    // sweep). Loading the WHOLE env slice is read-safe under multiple islands
    // per env; only THIS island\'s rows are written back (its lambdas), so
    // concurrent blocks of one env never write each other\'s slots.
    extern __shared__ unsigned char dyn_sh[];
    float* const qdot_sh = reinterpret_cast<float*>(dyn_sh);
    float* const lambda_sh = qdot_sh + kMaxArticulationDof;
    float* const meff_sh = lambda_sh + rows_per_env;
    float* const J_sh = meff_sh + rows_per_env;                  // R x D
    float* const w_sh = J_sh + static_cast<size_t>(rows_per_env) * dof_stride;
    SlimRow* const slim_sh = reinterpret_cast<SlimRow*>(
        w_sh + static_cast<size_t>(rows_per_env) * dof_stride);
    uint32_t* const order_sh = reinterpret_cast<uint32_t*>(slim_sh + rows_per_env);
    uint32_t* const seg_sh = order_sh + rows_per_env;            // 2R u32

    const bool has_artic = (flags & 1u) != 0u && dof_stride > 0u;
    if (has_artic) {
        for (uint32_t k = lane; k < dof_stride; k += blockDim.x) {
            qdot_sh[k] = qdot_flat[static_cast<size_t>(env) * dof_stride + k];
        }
    }
    for (uint32_t i = lane; i < rows_per_env; i += blockDim.x) {
        const size_t g = static_cast<size_t>(env_row_base) + i;
        lambda_sh[i] = lambda[g];
        meff_sh[i] = row_meff[g];
        // Build the slim solve record (see SlimRow).
        const NkRow row = urows[g];
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
        if (row.a.kind == kNkSideArtic) code |= kSlimAArt;
        if (row.b.kind == kNkSideArtic) code |= kSlimBArt;
        const bool a_dyn =
            row.a.kind == kNkSideRigid || row.a.kind == kNkSideParticle;
        const bool b_dyn =
            row.b.kind == kNkSideRigid || row.b.kind == kNkSideParticle;
        sr.dyn_index = 0u;
        sr.jl[0] = sr.jl[1] = sr.jl[2] = 0.0f;
        sr.ja[0] = sr.ja[1] = sr.ja[2] = 0.0f;
        if (a_dyn) {
            code |= kSlimHasDyn;
            if (row.a.kind == kNkSideParticle) code |= kSlimDynParticle;
            sr.dyn_index = row.a.index;
            sr.jl[0] = row.a.jlin.x; sr.jl[1] = row.a.jlin.y; sr.jl[2] = row.a.jlin.z;
            sr.ja[0] = row.a.jang.x; sr.ja[1] = row.a.jang.y; sr.ja[2] = row.a.jang.z;
            if (b_dyn) code |= kSlimFallback;  // two dynamic sides: full-record path
        } else if (b_dyn) {
            code |= kSlimHasDyn | kSlimDynIsB;
            if (row.b.kind == kNkSideParticle) code |= kSlimDynParticle;
            sr.dyn_index = row.b.index;
            sr.jl[0] = row.b.jlin.x; sr.jl[1] = row.b.jlin.y; sr.jl[2] = row.b.jlin.z;
            sr.ja[0] = row.b.jang.x; sr.ja[1] = row.b.jang.y; sr.ja[2] = row.b.jang.z;
        }
        sr.code = code;
        slim_sh[i] = sr;
    }
    for (size_t i = lane; i < static_cast<size_t>(rows_per_env) * dof_stride;
         i += blockDim.x) {
        const size_t g = static_cast<size_t>(env_row_base) * dof_stride + i;
        J_sh[i] = chain_jacobian[g];
        w_sh[i] = row_minv_jt[g];
    }
    // The island\'s rows are CONTIGUOUS in row_order (per-component spans).
    // COMPACT the schedule to the ACTIVE rows once (lane 0): the active set is
    // FIXED for the whole sweep (row flags never change inside the solve), so
    // dropping inactive rows / empty segments here only removes NO-OP visits —
    // the surviving execution order is the schedule\'s order, unchanged.
    __shared__ uint32_t live_seg_cnt_sh;
    if (lane == 0u) {
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
    for (uint32_t it = 0u; it < vel_iters; ++it) {
        for (uint32_t s = 0u; s < live_seg_cnt; ++s) {
            const uint32_t off = seg_sh[2u * s + 0u];
            const uint32_t cnt = seg_sh[2u * s + 1u];
            // One WARP per row of the color (rows of one color share no
            // mutable state — warp-level parallel is exactly the old
            // thread-level parallel, with the row\'s inner work distributed
            // across the warp\'s lanes).
            for (uint32_t idx = warp; idx < cnt; idx += nwarps) {
                const uint32_t gslot = order_sh[off + idx];
                SolveUnionRowWarp(gslot - env_row_base, gslot, wlane,
                                  slim_sh, lambda_sh, meff_sh, J_sh, w_sh,
                                  qdot_sh, urows,
                                  body_lin_vel, body_ang_vel, body_inv_mass,
                                  body_inv_inertia, particle_inv_mass,
                                  particle_vel, dof_stride, dt);
            }
            __syncthreads();
        }
    }

    // Write back THIS island's lambdas (the persistent warm-start/readout
    // field) — walk the island's own ACTIVE rows (the only lambdas the sweep
    // can change; inactive slots keep the assembled 0). Multi-island-per-env
    // safe: islands never share rows.
    if (live_seg_cnt > 0u) {
        const uint32_t last_off = seg_sh[2u * (live_seg_cnt - 1u) + 0u];
        const uint32_t last_cnt = seg_sh[2u * (live_seg_cnt - 1u) + 1u];
        const uint32_t island_rows = last_off + last_cnt;
        for (uint32_t i = lane; i < island_rows; i += blockDim.x) {
            const uint32_t gslot = order_sh[i];
            lambda[gslot] = lambda_sh[gslot - env_row_base];
        }
    }

    // Scatter the post-solve qdot tile back through the cooked dof maps (the
    // legacy scatter loop 1:1). A contact-free island writes back its own
    // unchanged values — a no-op, graph-stable.
    if (has_artic) {
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
    }
}

// --- op entry point ---------------------------------------------------------

Status OpSolveRowsBlockIsland(const ModelView& model, const DataView& data,
                              const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const SolveRowsBlockIslandParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }

    if (p->family == kContactFamilyUnionCsr) {
        if (p->total_islands == 0u) {
            return Status::Ok;
        }
        if (p->max_dof > kMaxArticulationDof) {
            return Status::Failed;  // shared qdot tile capacity (legacy cap).
        }
        const size_t shared_bytes = IslandSharedBytes(p->rows_per_env, p->max_dof);
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
        LaunchCuda(SolveRowsBlockIslandKernel, dim3(p->total_islands),
                   dim3(kIslandBlockSize), static_cast<uint32_t>(shared_bytes),
                   stream,
                   reinterpret_cast<const NkRow*>(data.urows),
                   data.lambda,
                   static_cast<const float*>(data.chain_jacobian),
                   static_cast<const float*>(data.row_minv_jt),
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
                   static_cast<const uint32_t*>(model.island_row_offsets),
                   static_cast<const uint32_t*>(model.island_color_segments),
                   static_cast<const uint32_t*>(model.row_order),
                   static_cast<const uint32_t*>(model.dof_to_link),
                   static_cast<const uint32_t*>(model.dof_to_component),
                   p->total_islands, p->rows_per_env, p->max_dof,
                   p->base_link_count,
                   static_cast<uint32_t>(p->vel_iters), p->dt);
        return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
    }

    // FUSED family (the legacy semantics, byte-exact).
    if (p->articulation_count == 0u || p->env_count == 0u || p->max_dof == 0u) {
        return Status::Ok;
    }
    if (p->max_dof > kMaxArticulationDof) {
        return Status::Failed;  // legacy loud-throw guard.
    }
    // total_link_count is not read by the solve kernel (it walks via the
    // articulation offset tables); 0 keeps the state honest about that.
    const ArticulationDeviceState state = MakeArticulationDeviceState(
        model, data, /*total_link_count=*/0u, p->articulation_count);
    const float* joint_damping =
        (p->apply_implicit_damping != 0u) ? data.drive_damping : nullptr;
    LaunchCuda(SolveArticulatedContactRowsKernel, dim3(p->articulation_count),
               dim3(32u), 0u, stream, state,
               reinterpret_cast<const ArticulatedContactRow*>(data.rows),
               static_cast<const float*>(data.jac_normal),
               static_cast<const float*>(data.jac_tangent1),
               static_cast<const float*>(data.jac_tangent2),
               static_cast<const float*>(data.m_inv),
               p->max_dof, p->dt, p->friction_coefficient,
               p->baumgarte_max_velocity, data.lambda, joint_damping);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

} // namespace

void RegisterNkSolveRowsOps() {
    SetCudaOp(NkOp::SolveRowsBlockIsland, &OpSolveRowsBlockIsland);
}

} // namespace nuka::phi
