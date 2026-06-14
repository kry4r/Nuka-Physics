#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- multi-step reverse-pass runner (p02-B/C)
// ---------------------------------------------------------------------------
//
// Backprops a scalar loss through an N-step CONTACT-FREE rollout to the per-step
// ACTION gradients and the LINK-MASS gradient, via gradient checkpointing,
// composing the p02-A single-step adjoint (StepBackward) once per step.
//
// THE UNIFIED "reverse a window" loop. The full-tape (recompute_on_backward==0)
// and checkpoint (==1) modes share the IDENTICAL inner reverse step; they differ
// only in HOW the state at step j is produced before StepBackward runs:
//
//   inner reverse of step j (state already restored to the START of step j):
//     1. D2D-copy live q,qdot -> q_pre,qdot_pre   (StepBackward needs pre-step)
//     2. StepOnce(action[j])                        (regen step-j ABA intermediates
//                                                     + advance live state to j+1;
//                                                     the intermediates StepBackward
//                                                     reads now hold step-j values)
//     3. zero grad_tau
//     4. StepBackward(state, inputs[j], grads)      (grad seed carries in
//                                                     grad_q/grad_qdot/grad_link_vel;
//                                                     OVERWRITTEN in place with the
//                                                     pre-step seed for step j-1)
//     5. write grad_target[j] into actions-grad slice j (StepBackward overwrote it)
//     6. grad_mass_accum += grad_mass_out           (StepBackward OVERWRITES
//                                                     grad_mass_out each call, so we
//                                                     accumulate into a separate
//                                                     buffer; fixed-order, atomic-free)
//
//   Window production:
//     full-tape:   the forward already captured a checkpoint at EVERY step, so
//                  "restore to start of j" = restore checkpoint slot j. The whole
//                  reverse is N restores + N StepBackwards, no extra recompute.
//     checkpoint:  process steps in descending WINDOWS [c .. c+K). To reverse a
//                  window, restore its base checkpoint, StepOnce forward to refill
//                  each step's pre-state into a small per-window state cache
//                  (snapshotting q/qdot/base_pose/link_velocity at the START of
//                  every step in the window), THEN reverse the window's steps in
//                  descending order, restoring each step's cached pre-state before
//                  its inner reverse. Reverse/forward time ~2x (each step's forward
//                  runs ~twice: once to record, once to refill), independent of K.
//
// Because both modes run the IDENTICAL inner reverse step on the IDENTICAL
// (bit-exact-replayed) per-step state, their grad_actions + grad_mass are
// BIT-IDENTICAL by construction -- that equality IS the replay-determinism gate.
//
// D1 / atomic-free: the cross-step + cross-window grad_mass accumulation is a
// sequence of stream-ordered fixed-order `+=` (one tiny per-link add kernel, one
// thread per link, plain load/add/store -- NO atomics). StepBackward itself is
// per-articulation fixed-order. Two runs -> bit-exact (R6).
// ---------------------------------------------------------------------------

#include "diffsim/checkpoint.hpp"
#include "diffsim/ift_runner.hpp"
#include "diffsim/recompute_orchestrator.hpp"
#include "diffsim/step_backward.hpp"
#include "diffsim/tape.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>
#include <memory>

namespace nuka::diffsim {

namespace articulation = nuka::runtime::articulation;

// gradient_mode routing seam (v0.5 p03 R3a). Selects how a step's contact rows
// (if any) are differentiated. DEFAULT is ContactFree -- the committed p02 path,
// BYTE-UNCHANGED (the IFT branch is never entered, no IftRunner is constructed).
// ContactIft routes a step's ACTIVE contact rows through the IFT-at-convergence
// path (one Delassus solve) BEFORE the contact-free connective adjoint handles
// the rest: IFT converts the downstream g = dL/dqdot_post into the dL/dqdot_free
// seed StepBackward already consumes as grad_qdot_out (qdot_free = the
// post-velocity-integration joint velocity). A full multi-step contact rollout
// through the C-ABI is a LATER integration; this round provides the seam + a
// single-contact-step entry (ApplyIftContactSeed) exercised by the FD test.
enum class GradientMode : uint8_t {
    ContactFree = 0,  // p02 default: drive -> ABA -> integrate only.
    ContactIft = 1,   // route active contact rows through the IFT path.
};

// Output gradient targets for the multi-step backward (caller-owned device
// buffers). grad_actions: [step_count * total_link_count] floats (the per-step
// action gradient, env-major per-link -- same layout as the tape actions).
// grad_mass: [total_link_count] floats (the per-link mass-parameter gradient,
// summed over all steps). Both are WRITTEN by Run (not accumulated into a
// caller's prior contents).
struct BackwardOutputs {
    float* grad_actions = nullptr;  // [step_count * total_link_count]
    float* grad_mass = nullptr;     // [total_link_count]
};

// Seeds for the loss adjoint on the FINAL state (post last step). The caller
// fills the post-step gradient dL/dq', dL/dqdot' (joints) and dL/d v_root' (the
// floating root spatial velocity). Device buffers, sized total_link_count and
// total_link_count*6 respectively. May be null -> treated as zero.
struct BackwardSeeds {
    const float* grad_q_final = nullptr;       // [total_link_count]
    const float* grad_qdot_final = nullptr;    // [total_link_count]
    const float* grad_link_velocity_final = nullptr;  // [total_link_count * 6]
};

class BackwardRunner {
public:
    BackwardRunner(const phi::DeviceContext& context,
                   RecomputeOrchestrator& orchestrator);

    // Frees the v2 scratch/seed/window Buffer* members (out-of-line).
    ~BackwardRunner();
    BackwardRunner(const BackwardRunner&) = delete;
    BackwardRunner& operator=(const BackwardRunner&) = delete;

    // Runs the full reverse pass over `tape` (step_count = tape.StepCount()),
    // restoring per-step state from `cm` (the checkpoints captured during the
    // forward), and writing grad_actions + grad_mass into `out`. `seeds` provides
    // the loss adjoint on the final state. After Run, the orchestrator's live
    // state has been moved (it was used as the replay scratch); the caller should
    // not rely on it. Synchronizes the context stream before returning.
    void Run(const Tape& tape, const CheckpointManager& cm,
             const BackwardSeeds& seeds, const BackwardOutputs& out);

    // gradient_mode routing seam (default ContactFree -> p02 path byte-unchanged).
    // Setting ContactIft enables the IFT contact-gradient branch; the IftRunner is
    // lazily constructed only when ContactIft is selected (so a ContactFree run
    // never allocates the CG backend / Delassus buffers -- the p02 tests are
    // unaffected bit-for-bit).
    void SetGradientMode(GradientMode mode) { gradient_mode_ = mode; }
    GradientMode gradient_mode() const { return gradient_mode_; }

    // Single contact-step IFT entry (the seam an integrated rollout dispatches to,
    // and the entry the FD test exercises directly). Given a converged contact
    // step's engine buffers (`inputs`) and the downstream cotangent g =
    // dL/dqdot_post in GLOBAL layout, converts g into the dL/dqdot_free seed
    // (g - J^T z, written into `grad_qdot_free`) and the contact-row dL/db_c (z,
    // written into `grad_b_c`). No-op + throws if gradient_mode is ContactFree
    // (the gate keeps the default path from ever touching IFT state). Enqueues on
    // the context stream; the caller synchronizes.
    void ApplyIftContactSeed(const IftContactInputs& inputs, const float* g,
                             float* grad_qdot_free, float* grad_b_c);

private:
    // The shared inner reverse of one step. `step` indexes the tape; the live
    // orchestrator state must already be at the START of `step`. Snapshots
    // q_pre/qdot_pre, runs StepOnce(action[step]) to regen the intermediates +
    // advance the live state, then StepBackward, writes grad_target[step], and
    // accumulates grad_mass. The grad seed (grad_q/grad_qdot/grad_link_vel) is
    // carried in the persistent seed buffers across calls.
    void ReverseStep(const Tape& tape, uint32_t step,
                     const BackwardOutputs& out);

    const phi::DeviceContext& context_;
    RecomputeOrchestrator& orch_;
    uint32_t total_link_count_ = 0u;

    // gradient_mode seam state. ContactFree by default; ift_ stays null until the
    // first ContactIft use (lazy -> the p02 path constructs nothing extra).
    GradientMode gradient_mode_ = GradientMode::ContactFree;
    std::unique_ptr<IftRunner> ift_;

    // Persistent reverse-pass scratch (sized at construction). phi v2 Buffer*;
    // device-level DEFAULT (stream-0) allocation, all kernel reads/writes run on
    // context_.stream over the base pointers (raw launches, unchanged).
    phi::Buffer* q_pre_ = nullptr;          // float[total_link_count]
    phi::Buffer* qdot_pre_ = nullptr;       // float[total_link_count]
    phi::Buffer* v_root_pre_ = nullptr;     // float[total_link_count*6] (pre-integration root vel)
    phi::Buffer* grad_q_ = nullptr;         // float[total_link_count]   (carried seed)
    phi::Buffer* grad_qdot_ = nullptr;      // float[total_link_count]   (carried seed)
    phi::Buffer* grad_link_vel_ = nullptr;  // float[total_link_count*6] (carried seed)
    phi::Buffer* grad_mass_step_ = nullptr; // float[total_link_count]   (per-step overwrite)
    phi::Buffer* grad_mass_acc_ = nullptr;  // float[total_link_count]   (running sum)
    phi::Buffer* grad_tau_ = nullptr;       // float[total_link_count]   (per-step, zeroed)

    // Per-window state cache for the checkpoint mode: snapshots q/qdot/base_pose/
    // link_velocity at the START of each step in a window so the descending-order
    // reverse can restore the exact pre-state of each step. Sized for one window
    // (checkpoint_interval steps); reused per window.
    phi::Buffer* win_q_ = nullptr;          // float[K * total_link_count]
    phi::Buffer* win_qdot_ = nullptr;       // float[K * total_link_count]
    phi::Buffer* win_base_pose_ = nullptr;  // Transform[K * articulation_count]
    phi::Buffer* win_link_vel_ = nullptr;   // LinkSpatialVel[K * total_link_count]
    uint32_t window_capacity_ = 0u;
    uint32_t articulation_count_ = 0u;
};

}  // namespace nuka::diffsim
