// ---------------------------------------------------------------------------
// nuka::diffsim -- contact KKT (Delassus) builder kernel  (v0.5 p03 R2)
// ---------------------------------------------------------------------------
//
// Warp-per-articulation assembly of the batched dense SPD Delassus operator
//      A_b = J_b M_b^-1 J_b^T      (one block per articulation, block-diagonal).
// See kkt_builder.hpp for the active-set criterion, row ordering, and the D1 /
// SPD contract. The math generalizes ComputeContactEffectiveMass's diagonal
// denom = J M^-1 J^T (articulation_contacts.cu) to every (i,j) row pair.
//
// Perf-conscious choices (D1-preserving throughout):
//   * ONE warp per block, grid = block_count -> independent warps, no global
//     sync, no atomics. Each warp owns exactly its articulation's 4 contact slots
//     and its M^-1 tile.
//   * The M^-1 tile (<= 18*18) and the active J rows (<= 12 * 18) are streamed
//     ONCE from global into SHARED memory (coalesced cooperative loads across the
//     32 lanes), then every Gram inner product reads shared -- no redundant
//     global round-trips. w_i = M^-1 J_i is precomputed once per active row into
//     shared, so A[i][j] = dot(J_i, w_j) is a single dof_stride-length dot.
//   * `__restrict__` on every pointer; `__device__ __forceinline__` helpers;
//     FMA via plain a*b+c on fp64 (single deterministic rounding).
//   * Symmetry + D1 for free: lane i owns output row i and writes A[i][j] for
//     j in [i, n) into BOTH values[i*12+j] and values[j*12+i] from the SAME fp64
//     register, cast once to fp32. Each global slot is written exactly once; the
//     transpose entry is the bit-identical mirror, never an independent recompute.
//   * fp64 accumulation, fp32 store: builder-vs-host rel-err ~1e-12 (<< 1e-5) and
//     still D1 (fixed accumulation order + deterministic single rounding).
// ---------------------------------------------------------------------------

#include "diffsim/kkt_builder.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::diffsim {

namespace {

namespace art = ::nuka::runtime::articulation;
using art::ArticulatedContactRow;
using art::ContactRowType;

constexpr uint32_t kWarpSize = 32u;
constexpr uint32_t kBlockStride = kMaxBlockDim * kMaxBlockDim;  // 144 floats/block
constexpr uint32_t kSlotsPerEnv = art::kMaxFootContactsPerEnv;  // 4
constexpr uint32_t kMaxDof = art::kMaxContactSolverDof;         // 18 (tile/jac stride cap)

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("nuka::diffsim::BuildContactDelassusSystem ") +
                                 operation + ": " + cudaGetErrorString(result));
    }
}

// One warp assembles block `blk` (== articulation == env). Lane `lane` (0..31)
// owns OUTPUT matrix row `lane`; only lanes [0, n) are active rows. The warp
// scans its kSlotsPerEnv contact slots in fixed order, gathers each ENGAGED
// slot's three direction rows (normal -> t1 -> t2) into shared, forms
// w_j = M^-1 J_j, and writes the symmetric Gram block.
__global__ void BuildDelassusKernel(const ArticulatedContactRow* __restrict__ rows,
                                    const float* __restrict__ jac_normal,
                                    const float* __restrict__ jac_tangent1,
                                    const float* __restrict__ jac_tangent2,
                                    const float* __restrict__ m_inv,
                                    float* __restrict__ values,
                                    uint32_t* __restrict__ block_dim,
                                    uint32_t block_count, uint32_t dof_stride) {
    const uint32_t blk = blockIdx.x;
    if (blk >= block_count) return;
    const uint32_t lane = threadIdx.x;  // 0..31

    // Shared residency: the M^-1 tile, the active J rows, and the M^-1 J rows.
    // All fp64 so the Gram accumulation is single-rounding deterministic. Sized
    // at the compile-time caps (kMaxDof, kMaxBlockDim); the leading dof_stride /
    // n sub-extent is what we actually touch.
    __shared__ double minv_sh[kMaxDof * kMaxDof];     // M^-1 tile (row-major)
    __shared__ double jrow_sh[kMaxBlockDim * kMaxDof];  // active J rows
    __shared__ double wrow_sh[kMaxBlockDim * kMaxDof];  // M^-1 J rows
    // The compacted active-row descriptor: for each active row r, which jac
    // buffer (0=normal,1=t1,2=t2) and which global slot it came from. Built by
    // lane 0 (the fixed-order scan), broadcast via shared + __syncwarp.
    __shared__ uint32_t row_dir_sh[kMaxBlockDim];
    __shared__ uint32_t row_slot_sh[kMaxBlockDim];
    __shared__ uint32_t n_sh;

    // --- Lane 0: fixed-order active-set scan over this articulation's 4 slots.
    //     Engaged := row_type != kRowInactive AND depth > 0. Emit normal, t1, t2.
    if (lane == 0u) {
        const uint32_t slot_base = blk * kSlotsPerEnv;
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
        n_sh = n;
        block_dim[blk] = n;
    }
    __syncwarp(0xffffffffu);
    const uint32_t n = n_sh;

    // Empty articulation: block_dim already 0; values padding was pre-zeroed by
    // the caller's memset (deterministic), so nothing more to write.
    if (n == 0u) return;

    // --- Cooperatively stream the M^-1 tile into shared (coalesced; the 32 lanes
    //     stride the tile). dof_stride <= kMaxDof, so dof_stride^2 <= 324 < 32*11.
    const size_t tile_stride = static_cast<size_t>(dof_stride) * dof_stride;
    const float* __restrict__ minv_tile = m_inv + static_cast<size_t>(blk) * tile_stride;
    for (uint32_t idx = lane; idx < dof_stride * dof_stride; idx += kWarpSize) {
        minv_sh[idx] = static_cast<double>(minv_tile[idx]);
    }

    // --- Each active row r (handled by lane r) loads its dof-wide J row from the
    //     direction-selected jac buffer into shared. Lanes >= n do nothing here.
    if (lane < n) {
        const uint32_t dir = row_dir_sh[lane];
        const uint32_t slot = row_slot_sh[lane];
        const float* __restrict__ jbuf =
            (dir == 0u) ? jac_normal : (dir == 1u) ? jac_tangent1 : jac_tangent2;
        const float* __restrict__ jr = jbuf + static_cast<size_t>(slot) * dof_stride;
        double* __restrict__ dst = jrow_sh + static_cast<size_t>(lane) * dof_stride;
        for (uint32_t c = 0u; c < dof_stride; ++c) {
            dst[c] = static_cast<double>(jr[c]);
        }
    }
    __syncwarp(0xffffffffu);

    // --- w_j = M^-1 J_j  (one lane per active row). Fixed loop order over the
    //     dof_stride x dof_stride tile -> deterministic. M^-1 is symmetric, so
    //     w_j[r] = sum_c Minv[r,c] J_j[c].
    if (lane < n) {
        const double* __restrict__ jr = jrow_sh + static_cast<size_t>(lane) * dof_stride;
        double* __restrict__ wr = wrow_sh + static_cast<size_t>(lane) * dof_stride;
        for (uint32_t r = 0u; r < dof_stride; ++r) {
            double acc = 0.0;
            const double* __restrict__ minv_row = minv_sh + static_cast<size_t>(r) * dof_stride;
            for (uint32_t c = 0u; c < dof_stride; ++c) {
                acc += minv_row[c] * jr[c];  // fp64 FMA, single rounding
            }
            wr[r] = acc;
        }
    }
    __syncwarp(0xffffffffu);

    // --- Gram block: lane i owns row i, computes A[i][j] = dot(J_i, w_j) for
    //     j in [i, n) and writes BOTH (i,j) and (j,i) from the same fp64 value.
    //     Each output slot is written exactly once (i<=j region writes upper, the
    //     mirror writes the strict-lower transpose) -> no atomics, bit-symmetric.
    if (lane < n) {
        const uint32_t i = lane;
        const double* __restrict__ ji = jrow_sh + static_cast<size_t>(i) * dof_stride;
        const uint32_t base = blk * kBlockStride;
        for (uint32_t j = i; j < n; ++j) {
            const double* __restrict__ wj = wrow_sh + static_cast<size_t>(j) * dof_stride;
            double acc = 0.0;
            for (uint32_t r = 0u; r < dof_stride; ++r) {
                acc += ji[r] * wj[r];  // A[i][j] = J_i . (M^-1 J_j)
            }
            const float a_ij = static_cast<float>(acc);
            values[base + i * kMaxBlockDim + j] = a_ij;          // upper / diagonal
            if (j != i) {
                values[base + j * kMaxBlockDim + i] = a_ij;      // bit-identical mirror
            }
        }
    }
}

}  // namespace

void BuildContactDelassusSystem(const phi::DeviceContext& context,
                                const ArticulatedContactRow* rows,
                                const float* jac_normal, const float* jac_tangent1,
                                const float* jac_tangent2, const float* m_inv,
                                uint32_t block_count, uint32_t dof_stride,
                                BatchedDenseSpdSystem& out_system,
                                ContactDelassusBuffers& out_buffers) {
    out_system.block_count = block_count;
    out_system.values = nullptr;
    out_system.block_dim = nullptr;
    if (block_count == 0u) {
        out_buffers.values = phi::Buffer();
        out_buffers.block_dim = phi::Buffer();
        return;
    }
    if (rows == nullptr || jac_normal == nullptr || jac_tangent1 == nullptr ||
        jac_tangent2 == nullptr || m_inv == nullptr) {
        throw std::invalid_argument(
            "nuka::diffsim::BuildContactDelassusSystem: null input buffer");
    }
    if (dof_stride == 0u || dof_stride > kMaxDof) {
        throw std::invalid_argument(
            "nuka::diffsim::BuildContactDelassusSystem: dof_stride out of range "
            "(must be 1.." + std::to_string(kMaxDof) + ")");
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    // Allocate the owned output buffers (one fresh allocation per build).
    out_buffers.values = phi::Buffer(
        static_cast<size_t>(block_count) * kBlockStride * sizeof(float),
        phi::MemoryKind::Device);
    out_buffers.block_dim = phi::Buffer(
        static_cast<size_t>(block_count) * sizeof(uint32_t), phi::MemoryKind::Device);

    float* values = static_cast<float*>(out_buffers.values.Data());
    uint32_t* block_dim = static_cast<uint32_t*>(out_buffers.block_dim.Data());

    // Zero the ENTIRE values span first: the D1 two-run memcmp gate compares the
    // full block_count*144 buffer, so every padding lane (>= n) and every empty
    // block must be a deterministic 0 (the kernel only writes the leading n x n).
    CheckCuda(cudaMemsetAsync(values, 0,
                              static_cast<size_t>(block_count) * kBlockStride * sizeof(float),
                              stream),
              "zero values");

    // ONE warp per block; grid over blocks. Mirrors the engine's canonical
    // <<<count, 32>>> warp-per-articulation idiom (and the CG backend's launch).
    BuildDelassusKernel<<<block_count, kWarpSize, 0u, stream>>>(
        rows, jac_normal, jac_tangent1, jac_tangent2, m_inv, values, block_dim,
        block_count, dof_stride);
    CheckCuda(cudaGetLastError(), "BuildDelassusKernel launch");

    out_system.values = values;
    out_system.block_dim = block_dim;
}

}  // namespace nuka::diffsim
