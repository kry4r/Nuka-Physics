// ---------------------------------------------------------------------------
// nuka::diffsim -- IFT-at-convergence contact gradient runner kernels (p03 R3a)
// ---------------------------------------------------------------------------
//
// Implements IftRunner (see ift_runner.hpp for the full IFT adjoint + the D1 /
// near-convergence contract). Pipeline, all on phi::Buffer, zero atomics:
//
//   (1) BuildContactDelassusSystem  -> A = J M^-1 J^T   (the committed kkt_builder)
//   (2) FormRhsKernel               -> rhs = J M^-1 g   (one warp/articulation)
//   (3) CG backend Solve (BlockJacobi, exact island Cholesky) -> z = A^-1 rhs
//   (4) DistributeKernel            -> grad_qdot_free = g - J^T z (GLOBAL scatter)
//                                       grad_b_c = z              (block layout)
//
// PERF / D1 choices (mirroring kkt_builder.cu + sparse_solver_cg.cu):
//   * ONE warp per articulation, grid = block_count -> independent warps, no
//     global sync, no atomics. Each articulation owns disjoint global DOFs, so the
//     GLOBAL scatter of grad_qdot_free is race-free without atomics.
//   * The M^-1 tile (<= 18*18) and the active J rows (<= 12*18) are streamed ONCE
//     from global into SHARED (coalesced cooperative loads), then reused. The
//     dof-wide vectors w = M^-1 g (FormRhs) and Jtz = J^T z (Distribute) are
//     SHARED-resident; J^T z is a FIXED-ORDER reduction over the n active rows
//     (lane = dof component) -- atomic-free and D1.
//   * fp64 internal accumulation, fp32 store (single deterministic rounding),
//     matching the kkt_builder / CG backend so the whole IFT chain is bit-exact.
//   * __restrict__ everywhere; __device__ __forceinline__ helpers; the CG solve is
//     reused (one A^-1 apply, the IFT point); no redundant global round-trips.
//
// LIMITATIONS (honest):
//   * Channels shipped: grad_qdot_free and grad_b_c. The d/dM and d/dJ channels
//     (via d(A^-1) = -A^-1 dA A^-1 and the M^-1 J^T product rule) are DEFERRED.
//   * Fixed-base dof_to_link map only (scalar joint DOFs -> state.qdot[link]). A
//     FloatingBase root's 6 base-velocity DOFs (link_velocity[root].v) are not
//     mapped here; the validated scene is fixed-base. Mirrors the SCALAR-joint
//     branch of SolveArticulatedContactRowsKernel exactly.
//   * Sticking-equality active set (kkt_builder); sliding / cone-boundary friction
//     is nonsmooth and out of scope.
//
// VALIDATION COVERAGE (see test_ift_vs_tape_backward.cpp, honest):
//   * FD oracle runs on a PLANAR scene (mechanism in the XZ plane) so the 48-iter
//     PGS converges EXACTLY (a clean gate) -- there the second friction tangent
//     (t2, the Y direction) is identically zero (J_t2 == 0), so FD exercises the
//     normal + t1 (in-plane tangent) columns. The t2 column + the genuinely coupled
//     3x3 Delassus are exercised by the DENSE-REFERENCE oracle on a NON-PLANAR
//     scene (no PGS dependence -> matches Eigen::LDLT exactly). When a tangent
//     column is dead (or any direction is rank-deficient), A is PSD (not strictly
//     SPD). BlockJacobi's per-block factorization is a MODIFIED (pseudo) Cholesky:
//     any post-elimination pivot at or below a RELATIVE threshold (eps*max_diag,
//     eps=1e-12 -- not an absolute floor) is treated as a NULL DIRECTION -- its
//     factored diagonal is set to a safe unit value, its factor column is zeroed,
//     and the matching solution component is forced to 0 (least-norm-style). The
//     result is FINITE for every PSD block and EXACT on the range space (zero on
//     the null space), covering BOTH a structurally dead row (zero pivot) AND an
//     OBLIQUE null direction (tiny-nonzero post-elimination Schur pivot). The
//     earlier "pivot floor carries the zero pivot, the solve stays correct" wording
//     was FALSE for the oblique case: an absolute 1e-300 floor -> sqrt -> 1e-150 ->
//     divide OVERFLOWED to NaN there. The modified-Cholesky null-direction handling
//     is the fix (see sparse_solver_cg.cu kNullPivotRelEps + the oblique regression
//     test in test_cg_vs_dense.cpp). Full-rank blocks never trip the branch, so
//     their factor and every downstream bit are byte-identical to the strict-SPD
//     path (D1 preserved).
//   * Tests use a SINGLE articulation; the warp-per-articulation / disjoint-link
//     scatter makes block independence + the grad_qdot_free = g pass-through
//     structural (each warp only touches its own articulation's links / rows), but
//     a multi-articulation batch is not separately gated this round.
// ---------------------------------------------------------------------------

#include "diffsim/ift_runner.hpp"
#include "diffsim/sparse_solver_cg.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::diffsim {

namespace {

namespace art = ::nuka::runtime::articulation;
using art::ArticulatedContactRow;
using art::ContactRowType;

constexpr uint32_t kWarpSize = 32u;
constexpr uint32_t kSlotsPerEnv = art::kMaxFootContactsPerEnv;  // 4
constexpr uint32_t kMaxDof = art::kMaxContactSolverDof;         // 18

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("nuka::diffsim::IftRunner ") +
                                 operation + ": " + cudaGetErrorString(result));
    }
}

// Per-joint-type DOF count (scalar-joint convention; FloatingBase=6 is reported
// but the runner's fixed-base scope does not map its base DOFs -- see header).
// Mirrors JointDofCountDevice in articulation_contacts.cu so the local-dof
// prefix sum is byte-identical to the engine's PGS map.
__device__ __forceinline__ uint32_t JointDofCount(art::ArticulationJointType type) {
    switch (type) {
        case art::ArticulationJointType::Revolute:
        case art::ArticulationJointType::Prismatic:
            return 1u;
        case art::ArticulationJointType::FloatingBase:
            return 6u;
        case art::ArticulationJointType::Fixed:
        default:
            return 0u;
    }
}

// Build the SCALAR-joint dof column -> global link map for an articulation, into
// `dof_to_link` (length <= kMaxDof). Returns the DOF count. Fixed loop order
// (base-inclusive prefix sum over [offset, offset+count)) == LocalDofIndex /
// SolveArticulatedContactRowsKernel for fixed-base models. A FloatingBase root is
// (out of scope) treated as 6 DOFs all on the root link's qdot slot -- the
// validated scenes have no floating root, so this branch is never exercised by
// the FD/dense gates; documented as a limitation.
__device__ __forceinline__ uint32_t BuildDofToLink(
    const art::ArticulationDeviceState& state, uint32_t offset, uint32_t count,
    uint32_t* __restrict__ dof_to_link) {
    uint32_t dof = 0u;
    for (uint32_t local = 0u; local < count && dof < kMaxDof; ++local) {
        const uint32_t link = offset + local;
        const uint32_t jdof = JointDofCount(state.joint_type[link]);
        for (uint32_t d = 0u; d < jdof && dof < kMaxDof; ++d) {
            dof_to_link[dof] = link;  // scalar joint: one DOF per link
            ++dof;
        }
    }
    return dof;
}

// Fixed-order active-set scan (lane 0) over the articulation's 4 slots, EXACTLY
// the kkt_builder convention: engaged := row_type != kRowInactive AND depth > 0,
// emit normal -> tangent1 -> tangent2 per engaged slot, slots ascending. Fills
// row_dir_sh / row_slot_sh (length n) and returns n.
__device__ __forceinline__ uint32_t ScanActiveRows(
    const ArticulatedContactRow* __restrict__ rows, uint32_t slot_base,
    uint32_t* __restrict__ row_dir_sh, uint32_t* __restrict__ row_slot_sh) {
    uint32_t n = 0u;
    for (uint32_t s = 0u; s < kSlotsPerEnv && n + 3u <= kMaxBlockDim; ++s) {
        const ArticulatedContactRow row = rows[slot_base + s];
        const bool engaged =
            (row.row_type != ContactRowType::kRowInactive) && (row.depth > 0.0f);
        if (!engaged) continue;
        const uint32_t slot = slot_base + s;
        row_dir_sh[n] = 0u; row_slot_sh[n] = slot; ++n;  // normal
        row_dir_sh[n] = 1u; row_slot_sh[n] = slot; ++n;  // tangent1
        row_dir_sh[n] = 2u; row_slot_sh[n] = slot; ++n;  // tangent2
    }
    return n;
}

// ---------------------------------------------------------------------------
// (2) rhs = J M^-1 g.  One warp per articulation. Forms w = M^-1 g_local ONCE
//     (lane r owns w[r]) from the shared M^-1 tile + the gathered g slice, then
//     each active row r (lane r) writes rhs[r] = dot(J_r, w). Atomic-free.
// ---------------------------------------------------------------------------
__global__ void FormRhsKernel(const ArticulatedContactRow* __restrict__ rows,
                              const float* __restrict__ jac_normal,
                              const float* __restrict__ jac_tangent1,
                              const float* __restrict__ jac_tangent2,
                              const float* __restrict__ m_inv,
                              const float* __restrict__ g_global,
                              art::ArticulationDeviceState state,
                              float* __restrict__ rhs,
                              uint32_t block_count, uint32_t dof_stride) {
    const uint32_t blk = blockIdx.x;
    if (blk >= block_count) return;
    const uint32_t lane = threadIdx.x;

    __shared__ double minv_sh[kMaxDof * kMaxDof];   // M^-1 tile (row-major)
    __shared__ double g_sh[kMaxDof];                // gathered g (local-dof order)
    __shared__ double w_sh[kMaxDof];                // w = M^-1 g
    __shared__ double jrow_sh[kMaxBlockDim * kMaxDof];  // active J rows
    __shared__ uint32_t row_dir_sh[kMaxBlockDim];
    __shared__ uint32_t row_slot_sh[kMaxBlockDim];
    __shared__ uint32_t dof_to_link_sh[kMaxDof];
    __shared__ uint32_t n_sh;
    __shared__ uint32_t dof_sh;

    const uint32_t slot_base = blk * kSlotsPerEnv;
    if (lane == 0u) {
        const uint32_t offset = state.articulation_link_offset[blk];
        const uint32_t count = state.articulation_link_count[blk];
        dof_sh = BuildDofToLink(state, offset, count, dof_to_link_sh);
        n_sh = ScanActiveRows(rows, slot_base, row_dir_sh, row_slot_sh);
    }
    __syncwarp(0xffffffffu);
    const uint32_t n = n_sh;
    const uint32_t dof = dof_sh;

    // rhs padding for this block: zero all kMaxBlockDim slots (deterministic).
    if (lane < kMaxBlockDim) rhs[blk * kMaxBlockDim + lane] = 0.0f;
    if (n == 0u || dof == 0u) return;

    // Gather g's articulation slice into local-dof order (lane k owns g_sh[k]).
    if (lane < dof) {
        g_sh[lane] = static_cast<double>(g_global[dof_to_link_sh[lane]]);
    }

    // Stream M^-1 tile into shared (coalesced; the 32 lanes stride the tile).
    const size_t tile_stride = static_cast<size_t>(dof_stride) * dof_stride;
    const float* __restrict__ minv_tile = m_inv + static_cast<size_t>(blk) * tile_stride;
    for (uint32_t idx = lane; idx < dof_stride * dof_stride; idx += kWarpSize) {
        minv_sh[idx] = static_cast<double>(minv_tile[idx]);
    }

    // Each active row r (lane r) loads its dof-wide J row from the dir-selected buf.
    if (lane < n) {
        const uint32_t dir = row_dir_sh[lane];
        const uint32_t slot = row_slot_sh[lane];
        const float* __restrict__ jbuf =
            (dir == 0u) ? jac_normal : (dir == 1u) ? jac_tangent1 : jac_tangent2;
        const float* __restrict__ jr = jbuf + static_cast<size_t>(slot) * dof_stride;
        double* __restrict__ dst = jrow_sh + static_cast<size_t>(lane) * dof_stride;
        for (uint32_t c = 0u; c < dof_stride; ++c) dst[c] = static_cast<double>(jr[c]);
    }
    __syncwarp(0xffffffffu);

    // w = M^-1 g  (lane r owns w[r]; M^-1 symmetric). Fixed loop order -> D1.
    // Only the leading `dof` columns of g are meaningful; columns >= dof are 0.
    if (lane < dof) {
        const double* __restrict__ minv_row =
            minv_sh + static_cast<size_t>(lane) * dof_stride;
        double acc = 0.0;
        for (uint32_t c = 0u; c < dof; ++c) acc += minv_row[c] * g_sh[c];
        w_sh[lane] = acc;
    }
    __syncwarp(0xffffffffu);

    // rhs[r] = dot(J_r, w)  over dof columns. Lane r owns active row r.
    if (lane < n) {
        const double* __restrict__ jr = jrow_sh + static_cast<size_t>(lane) * dof_stride;
        double acc = 0.0;
        for (uint32_t c = 0u; c < dof; ++c) acc += jr[c] * w_sh[c];
        rhs[blk * kMaxBlockDim + lane] = static_cast<float>(acc);
    }
}

// ---------------------------------------------------------------------------
// (4) Distribute z -> grad_qdot_free = g - J^T z  (GLOBAL scatter) and
//     grad_b_c = z  (block layout).  One warp per articulation. J^T z is a
//     FIXED-ORDER reduction over the n active rows: lane = dof component k,
//     Jtz[k] = sum_r z[r] * J_r[k]; then scatter grad_qdot_free[dof_to_link[k]]
//     = g[link] - Jtz[k]. Each articulation owns disjoint links -> race-free,
//     atomic-free. DOFs / articulations with no active row keep the seeded g
//     (Jtz = 0), so this is a clean pass-through.
// ---------------------------------------------------------------------------
__global__ void DistributeKernel(const ArticulatedContactRow* __restrict__ rows,
                                 const float* __restrict__ jac_normal,
                                 const float* __restrict__ jac_tangent1,
                                 const float* __restrict__ jac_tangent2,
                                 const float* __restrict__ z,
                                 const float* __restrict__ g_global,
                                 art::ArticulationDeviceState state,
                                 float* __restrict__ grad_qdot_free,
                                 float* __restrict__ grad_b_c,
                                 uint32_t block_count, uint32_t dof_stride) {
    const uint32_t blk = blockIdx.x;
    if (blk >= block_count) return;
    const uint32_t lane = threadIdx.x;

    __shared__ double jrow_sh[kMaxBlockDim * kMaxDof];  // active J rows
    __shared__ double z_sh[kMaxBlockDim];               // z (per active row)
    __shared__ uint32_t row_dir_sh[kMaxBlockDim];
    __shared__ uint32_t row_slot_sh[kMaxBlockDim];
    __shared__ uint32_t dof_to_link_sh[kMaxDof];
    __shared__ uint32_t n_sh;
    __shared__ uint32_t dof_sh;

    const uint32_t slot_base = blk * kSlotsPerEnv;
    if (lane == 0u) {
        const uint32_t offset = state.articulation_link_offset[blk];
        const uint32_t count = state.articulation_link_count[blk];
        dof_sh = BuildDofToLink(state, offset, count, dof_to_link_sh);
        n_sh = ScanActiveRows(rows, slot_base, row_dir_sh, row_slot_sh);
    }
    __syncwarp(0xffffffffu);
    const uint32_t n = n_sh;
    const uint32_t dof = dof_sh;

    // grad_b_c = z (block layout): copy the n active components, zero the padding.
    if (lane < kMaxBlockDim) {
        const float zv = (lane < n) ? z[blk * kMaxBlockDim + lane] : 0.0f;
        grad_b_c[blk * kMaxBlockDim + lane] = zv;
    }

    if (dof == 0u) return;

    // No active row: grad_qdot_free already holds the seeded g (pass-through). The
    // runner pre-seeds the whole buffer with g, so nothing to write here.
    if (n == 0u) return;

    // Load z (active rows) + active J rows into shared.
    if (lane < n) {
        z_sh[lane] = static_cast<double>(z[blk * kMaxBlockDim + lane]);
        const uint32_t dir = row_dir_sh[lane];
        const uint32_t slot = row_slot_sh[lane];
        const float* __restrict__ jbuf =
            (dir == 0u) ? jac_normal : (dir == 1u) ? jac_tangent1 : jac_tangent2;
        const float* __restrict__ jr = jbuf + static_cast<size_t>(slot) * dof_stride;
        double* __restrict__ dst = jrow_sh + static_cast<size_t>(lane) * dof_stride;
        for (uint32_t c = 0u; c < dof_stride; ++c) dst[c] = static_cast<double>(jr[c]);
    }
    __syncwarp(0xffffffffu);

    // grad_qdot_free[dof_to_link[k]] = g[link] - (J^T z)[k]. Lane k owns DOF k.
    // (J^T z)[k] = sum_r z[r] * J_r[k], fixed row order -> D1, atomic-free.
    if (lane < dof) {
        const uint32_t k = lane;
        double jtz = 0.0;
        for (uint32_t r = 0u; r < n; ++r) {
            jtz += z_sh[r] * jrow_sh[static_cast<size_t>(r) * dof_stride + k];
        }
        const uint32_t link = dof_to_link_sh[k];
        const double gk = static_cast<double>(g_global[link]);
        grad_qdot_free[link] = static_cast<float>(gk - jtz);
    }
}

}  // namespace

IftRunner::IftRunner(const phi::DeviceContext& context, DeterminismLevel determinism)
    : context_(context),
      determinism_(determinism),
      solver_(MakeSparseSolverBackend("cg", context, determinism)) {}

void IftRunner::Run(const IftContactInputs& inputs, const float* g,
                    const IftContactGrads& grads) {
    const uint32_t bc = inputs.block_count;
    if (bc == 0u) return;
    if (inputs.rows == nullptr || inputs.jac_normal == nullptr ||
        inputs.jac_tangent1 == nullptr || inputs.jac_tangent2 == nullptr ||
        inputs.m_inv == nullptr || g == nullptr ||
        grads.grad_qdot_free == nullptr || grads.grad_b_c == nullptr) {
        throw std::invalid_argument("nuka::diffsim::IftRunner::Run: null buffer");
    }
    if (inputs.dof_stride == 0u || inputs.dof_stride > kMaxDof) {
        throw std::invalid_argument(
            "nuka::diffsim::IftRunner::Run: dof_stride out of range (1.." +
            std::to_string(kMaxDof) + ")");
    }

    phi::ScopedDeviceGuard guard(context_.device_id);
    const cudaStream_t stream = context_.stream.Native();

    // Seed grad_qdot_free with g (the pass-through baseline): DOFs with no active
    // contact keep g, active DOFs are overwritten with g - J^T z in DistributeKernel.
    CheckCuda(cudaMemcpyAsync(grads.grad_qdot_free, g,
                              static_cast<size_t>(inputs.state.total_link_count) *
                                  sizeof(float),
                              cudaMemcpyDeviceToDevice, stream),
              "seed grad_qdot_free");

    // (1) Build A = J M^-1 J^T (committed kkt_builder; same active set + gather).
    BatchedDenseSpdSystem system;
    BuildContactDelassusSystem(context_, inputs.rows, inputs.jac_normal,
                               inputs.jac_tangent1, inputs.jac_tangent2,
                               inputs.m_inv, bc, inputs.dof_stride, system,
                               delassus_);

    // (Re)allocate the block-major rhs/z scratch to the current shape.
    if (capacity_blocks_ < bc) {
        capacity_blocks_ = bc;
        rhs_ = phi::Buffer(static_cast<size_t>(bc) * kMaxBlockDim * sizeof(float),
                           phi::MemoryKind::Device);
        z_ = phi::Buffer(static_cast<size_t>(bc) * kMaxBlockDim * sizeof(float),
                         phi::MemoryKind::Device);
    }
    float* rhs = static_cast<float*>(rhs_.Data());
    float* z = static_cast<float*>(z_.Data());

    // (2) rhs = J M^-1 g.  One warp per articulation, atomic-free.
    FormRhsKernel<<<bc, kWarpSize, 0u, stream>>>(
        inputs.rows, inputs.jac_normal, inputs.jac_tangent1, inputs.jac_tangent2,
        inputs.m_inv, g, inputs.state, rhs, bc, inputs.dof_stride);
    CheckCuda(cudaGetLastError(), "FormRhsKernel launch");

    // (3) z = A^-1 rhs.  BlockJacobi = exact per-block Cholesky island solve: the
    //     IFT assumes convergence, so the exact solve is the right A^-1 apply
    //     (matches the dense Eigen::LDLT reference to machine precision). One CG
    //     solve -- THE IFT trick, replacing taping the 48 PGS iters. D1 bit-exact.
    SolveParams params;
    params.preconditioner = Preconditioner::BlockJacobi;
    params.max_iter = 64u;
    params.tol = 1.0e-6f;
    params.run_to_fixed_iters = false;
    solver_->Solve(system, rhs, z, params);

    // (4) Distribute: grad_qdot_free = g - J^T z (global scatter), grad_b_c = z.
    DistributeKernel<<<bc, kWarpSize, 0u, stream>>>(
        inputs.rows, inputs.jac_normal, inputs.jac_tangent1, inputs.jac_tangent2,
        z, g, inputs.state, grads.grad_qdot_free, grads.grad_b_c, bc,
        inputs.dof_stride);
    CheckCuda(cudaGetLastError(), "DistributeKernel launch");
}

}  // namespace nuka::diffsim
