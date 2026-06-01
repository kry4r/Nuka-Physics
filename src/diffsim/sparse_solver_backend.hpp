#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- sparse linear solver backend interface (v0.5 p03 R1)
// ---------------------------------------------------------------------------
//
// The IFT (Implicit Function Theorem) contact-gradient backend solves the SPD
// Delassus system  A x = b  with  A = J M^-1 J^T  in CONTACT space. That system
// is BLOCK-DIAGONAL across articulations: each robot's foot contacts couple only
// within that robot, so A is a batch of independent dense SPD blocks, one per
// articulation, each of size n <= kMaxFootContactsPerEnv(4) * 3 (normal + two
// friction rows) = 12. Because A is symmetric (A^T = A), the adjoint / transposed
// solve uses the EXACT same operator -- one solver serves forward and backward.
//
// v0.5 representation: BATCHED PER-ARTICULATION DENSE BLOCKS (NOT a general CSR
// matrix). For a block-diagonal operator a CSR SpMV is strictly slower (irregular
// access + index indirection) and is unnecessary machinery. CSR / MINRES / GMRES
// / AMG are v0.7+ extensions that plug in BEHIND this same `SparseLinearSolver`
// seam without touching call sites; they are deliberately NOT built now.
//
// D1: every backend behind this seam must be bit-exact under DeterminismLevel
// ::Strong (same input + same GPU + same driver -> byte-identical output). No
// float atomics anywhere; all reductions are fixed-order warp-shuffle butterflies.
// ---------------------------------------------------------------------------

#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
// DeterminismLevel { Strong, Weak } already exists engine-wide; reuse it rather
// than declaring a second one (keeps the D1/D2 contract single-sourced).
#include "runtime/gpu/batched_articulated_world.hpp"

#include <cstdint>
#include <memory>
#include <string_view>

namespace nuka::diffsim {

// Reuse the engine's single DeterminismLevel (Strong = D1 bit-exact, the
// default; Weak = D2 escape hatch). Defined in runtime/gpu/batched_articulated_world.hpp.
using nuka::runtime::gpu::DeterminismLevel;

// Compile-time cap on a single block's dimension: kMaxFootContactsPerEnv(4) * 3
// rows (normal + 2 friction) = 12. One warp (32 lanes) covers a block with room
// to spare; lanes >= n are inactive (carry reduction identities). An actual
// per-block n <= kMaxBlockDim is carried in `block_dim` so partially-filled and
// empty (n == 0) articulations are handled without branching the layout.
constexpr uint32_t kMaxBlockDim = 12u;

// ---------------------------------------------------------------------------
// BatchedDenseSpdSystem -- the v0.5 batched-dense SPD operator (POD handle).
//
// The blocks live in `values` with a FIXED per-block stride of
// kMaxBlockDim * kMaxBlockDim floats, ROW-MAJOR within a block:
//
//     A_block(b)[row, col] = values[b * kMaxBlockDim * kMaxBlockDim
//                                   + row * kMaxBlockDim + col]
//
// Layout rationale (coalescing + warp-per-block residency): one warp owns block
// b. With lane L handling matrix row L, the warp's first column-0 access touches
// values[base + 0], values[base + kMaxBlockDim], values[base + 2*kMaxBlockDim],
// ... i.e. a strided-by-kMaxBlockDim gather. To make the per-row load contiguous
// we instead have each lane stream its whole row (kMaxBlockDim consecutive
// floats) into registers; across the warp the 32 row-reads tile the block's
// contiguous [stride] span, so the aggregate traffic is fully contiguous and the
// block (<= 12*12*4 = 576 B, padded 12*12*4 = 576 B) is read exactly once into
// registers and never re-fetched inside the CG loop. The fixed stride (rather
// than a packed n*n) keeps every block's base at a compile-time-known offset
// (b * 144) so the KKT builder can scatter into it with a trivial index and the
// warp's base address is a pure multiply -- no prefix-sum over per-block sizes.
//
// `b` (rhs) and `x` (solution), passed separately to Solve, are
// float[block_count * kMaxBlockDim], block-major, padded lanes (>= n) ignored on
// input and written 0.0f on output (deterministic padding for the memcmp gate).
// ---------------------------------------------------------------------------
struct BatchedDenseSpdSystem {
    uint32_t block_count = 0u;  // number of articulations / SPD blocks

    // Device float buffer, length block_count * kMaxBlockDim * kMaxBlockDim.
    // Row-major dense block per articulation at stride kMaxBlockDim^2. Only the
    // leading n x n sub-block of each is meaningful; the rest is padding that the
    // solver never reads (lanes >= n are inactive).
    const float* values = nullptr;

    // Device uint32_t buffer, length block_count. block_dim[b] = n_b <= kMaxBlockDim
    // is the active dimension of block b (0 => empty articulation, skipped: its x
    // padding is zeroed and the warp exits early).
    const uint32_t* block_dim = nullptr;
};

// Which preconditioner the CG backend applies. Both are D1 bit-exact.
enum class Preconditioner : uint8_t {
    // Diagonal M^-1 = 1/diag(A). Genuine CG iteration on a coupled block; the
    // default, exercises the full CG loop + butterfly reductions.
    Jacobi = 0,
    // Per-block dense Cholesky of the WHOLE block (the exact island solve, since
    // the system is block-diagonal). M^-1 = A^-1 within the block => preconditioned
    // residual is the exact solution => CG converges in <= 1 iteration.
    BlockJacobi = 1,
};

// Solve parameters. With a fixed iteration count the path is bit-exact regardless
// of `tol`; `tol` only enables a DETERMINISTIC early exit (a fixed-order residual-
// norm compare -- same input => same residual => same break, so still D1).
struct SolveParams {
    Preconditioner preconditioner = Preconditioner::Jacobi;
    uint32_t max_iter = 64u;          // blocks are <= 12x12; 64 is ample headroom
    float tol = 1.0e-6f;              // relative-residual early-exit threshold
    bool run_to_fixed_iters = false;  // true => ignore tol, always run max_iter
                                      //         (the strictest D1 demo path)
};

// ---------------------------------------------------------------------------
// SparseLinearSolver -- the seam v0.7+ extends with CSR/MINRES/AMG backends
// WITHOUT changing call sites. v0.5 operates on the batched-dense SPD system.
// ---------------------------------------------------------------------------
class SparseLinearSolver {
public:
    virtual ~SparseLinearSolver() = default;

    // Solves A x = b for every block. `x` (device float[block_count*kMaxBlockDim])
    // is written in full incl. deterministic zero padding for inactive lanes.
    // `x` and `b` may not alias. Enqueues on the context stream; the caller
    // synchronizes (mirrors the rest of diffsim). D1 under DeterminismLevel::Strong.
    virtual void Solve(const BatchedDenseSpdSystem& system, const float* b,
                       float* x, const SolveParams& params) = 0;

    // Human-readable backend id (e.g. "cg"). For diagnostics / factory round-trip.
    virtual std::string_view Name() const = 0;
};

// Factory. v0.5's only backend is the self-written CG one ("cg" / "" / "default").
// v0.7+ registers "csr-minres" etc. here behind the same return type. Throws
// std::invalid_argument on an unknown name.
std::unique_ptr<SparseLinearSolver> MakeSparseSolverBackend(
    std::string_view name, const phi::DeviceContext& context,
    DeterminismLevel determinism = DeterminismLevel::Strong);

}  // namespace nuka::diffsim
