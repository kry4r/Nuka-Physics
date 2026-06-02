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

// ---------------------------------------------------------------------------
// CgConvergenceReport -- OPT-IN per-block convergence diagnostics (v0.7 p01).
//
// A side-channel the general (non-IFT) path uses to observe convergence on
// larger / worse-conditioned blocks: iterations actually run, the final
// relative residual ||r||/||b||, and stall / divergence flags. It is written
// ONLY when SolveParams::diagnostics is non-null; with it null the solve takes
// the EXACT v0.5 code path and emits byte-identical x (the D1 + no-IFT-regression
// guarantee). Each field is one fixed device buffer of length block_count,
// written by lane 0 of each block in fixed block order -> two-run byte-exact.
//
// The residual history (optional, separate buffer of block_count * history_cap)
// records ||r_k||/||b|| at iteration k for the κ-conditioning convergence-rate
// test; null => not recorded.
// ---------------------------------------------------------------------------
struct CgConvergenceReport {
    // Device uint32_t[block_count]: number of CG iterations the block actually
    // ran before the deterministic early exit (== max_iter if it never exits).
    uint32_t* iters_run = nullptr;
    // Device float[block_count]: final relative residual ||r||/||b|| of the block
    // (0 for an empty / zero-rhs block, by the deterministic ||b||==0 guard).
    float* final_rel_resid = nullptr;
    // Device uint8_t[block_count]: per-block status flags (CgBlockStatus bits).
    uint8_t* status = nullptr;
    // OPTIONAL device float[block_count * history_cap]: ||r_k||/||b|| at iter k,
    // row-major (block-major, stride history_cap). null => not recorded. Entries
    // past iters_run are left at their pre-initialized sentinel by the kernel
    // (the kernel only WRITES the iters it runs), so the host must zero/sentinel-
    // fill this buffer before the solve if it reads untouched tail entries.
    float* residual_history = nullptr;
    // Capacity (iterations) of one block's residual_history row. Ignored if
    // residual_history is null. Writes are capped at history_cap.
    uint32_t history_cap = 0u;
};

// Per-block convergence status bit flags (written into CgConvergenceReport.status).
// Deterministic: each flag is set by a fixed-order data-dependent compare, so the
// same input yields the same flags across runs.
enum CgBlockStatus : uint8_t {
    kCgStatusNone = 0u,
    // The deterministic relative-residual early exit fired (||r||^2 <= tol^2||b||^2).
    kCgStatusConverged = 1u << 0,
    // max_iter was exhausted without hitting tol (a budget-limited solve; on a
    // worse-conditioned block this is the honest "did not reach tol in budget").
    kCgStatusMaxIterReached = 1u << 1,
    // The residual rose above its running minimum by more than a fixed factor at
    // some iteration (divergence symptom -- should not happen for SPD + true CG,
    // surfaced so a mis-specified non-SPD system is caught rather than hidden).
    kCgStatusDiverged = 1u << 2,
    // The residual stalled: it failed to drop below kStallFactor * previous over a
    // window, i.e. CG made no useful progress (an ill-conditioned-block symptom).
    kCgStatusStalled = 1u << 3,
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
    // OPT-IN convergence diagnostics for the general path. Default (all-null) =>
    // the EXACT v0.5 path, byte-identical x. Pointers are device buffers the
    // CALLER owns + sizes to block_count (and block_count*history_cap for the
    // optional history). See CgConvergenceReport.
    CgConvergenceReport diagnostics;
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
// v0.7 p02 registers "self_minres" / "minres" (symmetric indefinite). Throws
// std::invalid_argument on an unknown name.
std::unique_ptr<SparseLinearSolver> MakeSparseSolverBackend(
    std::string_view name, const phi::DeviceContext& context,
    DeterminismLevel determinism = DeterminismLevel::Strong);

// ---------------------------------------------------------------------------
// DetectBatchedIndefinite -- v0.7 p02 auto-routing detector. READ-ONLY over the
// batched-dense `system` (reads `values` / `block_dim`; writes ONLY the single
// uint32 device flag `out_indefinite_flag`, which the caller pre-zeros). Returns
// true (sets the flag to 1) iff ANY block has a strictly-NEGATIVE post-elimination
// LDLT pivot -- i.e. the batch is SYMMETRIC INDEFINITE (Sylvester's law of inertia:
// the signs of the LDLT pivots are the inertia of A). A negative-diagonal-ENTRY
// scan would MISS saddle-point KKT systems (positive diagonal, indefinite by
// structure); we therefore compute PIVOTS, not entries.
//
// BYTE-SAFETY (the SPD-path guarantee, the load-bearing property): the pivot test
// is `djj < -relEps*max|diag|` (STRICTLY negative, with a relative margin). For any
// SPD or PSD block every pivot is >= 0 (a PSD null direction gives djj ~ 0, which
// the strict-negative test does NOT trip), so the flag stays 0 -> the caller runs
// the EXISTING CG solve untouched -> byte-identical output. NO FALSE POSITIVES on
// SPD/PSD. The detector touches NO solver state. One warp per block, lane 0 does the
// fixed-order symmetric elimination in shared, the flag is a uint32 atomicOr (uint
// atomics are D1-safe -- order-independent OR). Deterministic.
//
// HONEST LIMIT (false NEGATIVES): this is a NO-PIVOTING LDLT, so it can MISS an
// indefinite block whose leading principal minors are singular (e.g. [[0,c],[c,0]]:
// both pivots are ~0 -> not flagged, yet eigenvalues are +/-c). That is SAFE for the
// byte-identity guarantee (a miss only leaves an indefinite system on CG, which is a
// correctness/convergence issue for that exotic case, NOT an SPD-path regression).
// For robust routing on such systems, select "self_minres" EXPLICITLY via the
// backend key -- the auto-router is a convenience, the explicit key is the escape.
//
// Enqueues on the context stream; the caller synchronizes + reads the flag back.
void DetectBatchedIndefinite(const phi::DeviceContext& context,
                             const BatchedDenseSpdSystem& system,
                             uint32_t* out_indefinite_flag);

}  // namespace nuka::diffsim
