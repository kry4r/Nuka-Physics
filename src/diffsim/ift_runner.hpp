#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- IFT-at-convergence contact gradient runner  (v0.5 p03 R3a)
// ---------------------------------------------------------------------------
//
// Exit criterion #6: back-propagate through the CONVERGED contact solve in ONE
// sparse linear solve via the Implicit Function Theorem, instead of taping the
// fixed-48-iteration PGS inner loop.
//
// FORWARD (one articulation, at the converged active set):
//     A lambda = r,     A = J M^-1 J^T (SPD Delassus),  r = b_c - J qdot_free
//     qdot_post = qdot_free + M^-1 J^T lambda
// where J is the active-row chain Jacobian (sticking-equality rows: normal + two
// friction tangents per engaged slot, the kkt_builder active set), M^-1 is the
// articulation inverse-inertia tile, and b_c is the per-row target separating
// velocity (normal row: kBaumgarteBeta/dt*max(depth-slop,0); friction row: 0).
//
// IFT REVERSE -- given the downstream cotangent g = dL/dqdot_post:
//   Solve ONCE on the (symmetric) Delassus:   z = A^-1 (J M^-1 g).
//   Distribute (the two channels this runner ships):
//     grad_qdot_free = g - J^T z      (GLOBAL total_link_count layout)
//     grad_b_c       = z              (CONTACT-ROW / block layout, stride kMaxBlockDim)
// Derivation: qdot_post = qdot_free + M^-1 J^T A^-1 (b_c - J qdot_free), so
//   d qdot_post/d qdot_free = I - M^-1 J^T A^-1 J  =>  (.)^T g = g - J^T A^-1 (J M^-1 g)
//   d qdot_post/d b_c       = M^-1 J^T A^-1        =>  (.)^T g = A^-1 (J M^-1 g) = z
// (A,M symmetric so A^-T = A^-1, M^-T = M^-1). The single A^-1 CG apply IS the IFT
// trick -- it replaces taping the inner PGS. The deferred d/dM, d/dJ channels are
// NOT shipped this round (see ift_runner.cu limitation note).
//
// SCOPE / honesty:
//  * p02's tape/backward is CONTACT-FREE -- there is NO contact-tape to compare
//    against. PRIMARY validation is finite-difference through the REAL engine
//    contact step (tests/diffsim/test_ift_vs_tape_backward.cpp); the dense
//    Eigen::LDLT solve+distribute is the second oracle. "IFT vs tape" is N/A here.
//  * The IFT assumes the forward CONVERGED. The forward runs a FIXED 48 PGS iters,
//    so the test scene must be near-converged (resting normal-dominant contact);
//    sliding / at-the-cone-boundary friction is genuinely nonsmooth and out of
//    scope (matches the kkt_builder sticking-row assumption).
//
// D1 (R6): one warp per articulation, fixed-order fp64 warp-shuffle reductions,
// the CG backend's bit-exact solve, zero float atomics; grad outputs are two-run
// byte-identical (the D1TwoRunByteExact test).
// ---------------------------------------------------------------------------

#include "diffsim/kkt_builder.hpp"
#include "diffsim/sparse_solver_backend.hpp"
#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>
#include "runtime/articulation/articulation_contacts.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>

namespace nuka::diffsim {

namespace articulation = nuka::runtime::articulation;

// The engine-side contact buffers a single converged contact step exposes (the
// batched articulated step's contact members of the same name), plus the device state for
// the dof column -> global qdot index map. All raw device pointers borrowed from
// the caller; the runner does not own them.
struct IftContactInputs {
    // Active-set + geometry rows (env-major, fixed stride kMaxFootContactsPerEnv).
    const articulation::ArticulatedContactRow* rows = nullptr;
    // Chain Jacobians, per slot, dof_stride-wide (slot * dof_stride .. +dof_stride).
    const float* jac_normal = nullptr;
    const float* jac_tangent1 = nullptr;
    const float* jac_tangent2 = nullptr;
    // Per-articulation M^-1 tile, row-major, stride dof_stride^2.
    const float* m_inv = nullptr;
    // The dof column -> global qdot index map source (joint_type / parent_link /
    // articulation_link_offset / articulation_link_count). The runner reads ONLY
    // these topology fields; q/qdot are not read (the forward already ran). NOTE:
    // the dof_to_link map mirrors SolveArticulatedContactRowsKernel for SCALAR
    // joint DOFs (state.qdot[link]); a FloatingBase root's 6 base-velocity DOFs
    // live in link_velocity[root].v and are NOT mapped here -- this round supports
    // fixed-base articulations (the validated path). See ift_runner.cu.
    articulation::ArticulationDeviceState state;
    uint32_t block_count = 0u;   // articulations / SPD blocks (== env_count)
    uint32_t dof_stride = 0u;    // M^-1 / chain-Jacobian stride (== max_dof)
};

// Outputs of the IFT reverse. The caller owns the destination buffers.
struct IftContactGrads {
    // dL/dqdot_free, GLOBAL layout (device float[total_link_count]). The runner
    // first COPIES g (dL/dqdot_post) into it, then SCATTERS g - J^T z into the
    // active DOFs of each articulation -- DOFs with no active contact keep g
    // (J^T z = 0 there, a clean pass-through). This is exactly the dL/dqdot seed
    // StepBackward consumes as grad_qdot_out. May NOT alias g.
    float* grad_qdot_free = nullptr;
    // dL/db_c = z, CONTACT-ROW / block layout (device float[block_count*kMaxBlockDim],
    // block-major stride kMaxBlockDim, deterministic zero padding for lanes >= n_b).
    float* grad_b_c = nullptr;
};

// IftRunner: builds A (kkt_builder), forms rhs = J M^-1 g, solves z = A^-1 rhs
// (the self-written CG backend, BlockJacobi exact island solve), and distributes
// into grad_qdot_free / grad_b_c. Reuses the kkt_builder active-set + gather
// conventions EXACTLY so J / M^-1 indexing matches. One warp per articulation,
// zero atomics, D1. Owns its CG backend + intermediate buffers (reused across
// Run calls of the same shape).
class IftRunner {
public:
    explicit IftRunner(cudaStream_t stream, int device_id,
                       DeterminismLevel determinism = DeterminismLevel::Strong);

    // Run the IFT contact-gradient reverse. `g` is the downstream cotangent
    // dL/dqdot_post in GLOBAL layout (device float[total_link_count]); it is NOT
    // modified. The runner copies g into grads.grad_qdot_free then overwrites the
    // active DOFs with g - J^T z (pass-through DOFs keep g). Enqueues on the
    // context stream; the caller synchronizes (mirrors the rest of diffsim).
    void Run(const IftContactInputs& inputs, const float* g,
             const IftContactGrads& grads);

    // Arm the symmetric-indefinite / non-symmetric AUTO-ROUTER inside Run().
    // DEFAULT false: the validated rigid/articulation contact Delassus A = J M^-1 J^T
    // is provably bit-symmetric SPD/PSD (mirror-stored; test_kkt_build asserts the
    // byte-symmetry), so neither detector can ever fire on this path -- running them
    // (two READ-ONLY kernels + two per-solve cudaStreamSynchronize host round-trips)
    // would be pure overhead on the diff-sim BACKWARD hot path. p11 (coupling rows,
    // which CAN assemble a genuinely indefinite / non-symmetric KKT) flips this to
    // true to arm MINRES/GMRES routing. Toggling does NOT change results on an SPD
    // batch -- the router's SPD branch is byte-identical to the default CG solve.
    void SetAdaptiveRouting(bool enable) noexcept { adaptive_routing_ = enable; }

    // Frees the v2 rhs_/z_/flag Buffer* members (delassus_ frees itself; the
    // unique_ptr solver backends free themselves). Out-of-line so the incomplete
    // SparseLinearSolver / IftRunner members are complete at destruction.
    ~IftRunner();
    IftRunner(const IftRunner&) = delete;
    IftRunner& operator=(const IftRunner&) = delete;

private:
    // Auto-router body (p02 indefinite -> MINRES, p03-A non-symmetric -> GMRES, else
    // CG). Reached ONLY when adaptive_routing_ is true; see SetAdaptiveRouting().
    void RunAutoRouter(const BatchedDenseSpdSystem& system, float* rhs, float* z,
                       const SolveParams& params);

    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    DeterminismLevel determinism_;
    std::unique_ptr<SparseLinearSolver> solver_;       // CG (SPD), the default path
    // v0.7 p02: lazily-constructed MINRES backend, used ONLY when the assembled KKT
    // is detected SYMMETRIC INDEFINITE (a negative LDLT pivot). For the validated
    // (all-SPD) contact scenes the detector never fires, so this stays null and the
    // CG path runs byte-identically -- the no-regression guarantee.
    std::unique_ptr<SparseLinearSolver> minres_solver_;
    // v0.7 p03-A: lazily-constructed GMRES backend, used ONLY when the assembled KKT
    // is detected genuinely NON-SYMMETRIC. The Delassus A = J M^-1 J^T is stored
    // BIT-SYMMETRIC (mirror-stored; A[i][j] and A[j][i] are the same float), so the
    // non-symmetric detector's off-diagonal difference is EXACTLY 0.0 and never fires
    // on the contact path -> this stays null and the CG/MINRES path runs byte-
    // identically (the symmetric-path no-regression guarantee). The branch exists so
    // the auto-router covers the full solver suite if a non-symmetric system ever
    // reaches this seam.
    std::unique_ptr<SparseLinearSolver> gmres_solver_;

    // Reused intermediates (lazily (re)allocated to the current shape). phi v2
    // Buffer*; device-level DEFAULT (stream-0) alloc, all kernel work runs on
    // context_.stream over the base pointers (raw launches / copies, unchanged).
    ContactDelassusBuffers delassus_;       // A (values + block_dim)
    phi::Buffer* rhs_ = nullptr;            // J M^-1 g, block-major stride kMaxBlockDim
    phi::Buffer* z_ = nullptr;              // A^-1 rhs,  block-major stride kMaxBlockDim
    phi::Buffer* indef_flag_ = nullptr;     // uint32[1] indefinite-detector flag
    phi::Buffer* nonsym_flag_ = nullptr;    // uint32[1] non-symmetric-detector flag
    uint32_t capacity_blocks_ = 0u;
    // Gates the Run() auto-router. false (default) => skip BOTH detectors + their two
    // per-solve stream syncs and solve directly with CG (zero added cost, byte-
    // identical on the provably-SPD contact path). See SetAdaptiveRouting(); the named
    // consumer that flips it on is p11 (coupling rows can assemble non-SPD KKT).
    bool adaptive_routing_ = false;
};

}  // namespace nuka::diffsim
