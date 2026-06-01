#ifndef NUKA_NUKA_DIFFSIM_H
#define NUKA_NUKA_DIFFSIM_H
// ---------------------------------------------------------------------------
// nuka_diffsim.h -- C ABI for the multi-step differentiable rollout (p02-B/C)
// ---------------------------------------------------------------------------
//
// Backprops a scalar loss through an N-step CONTACT-FREE rollout of an existing
// nuka world to (a) per-step ACTION gradients (drive_targets) and (b) the per-
// link mass-PARAMETER gradient, via gradient checkpointing. It composes the
// p02-A single-step adjoint; the §4 PARAMETER spine is the mass/gain gradient
// carried in grad_parameters_out.
//
// SCOPE: CONTACT-FREE. The differentiable forward is the exact contact-free PD
// step the p02-A adjoint reverses (explicit drive damping, drive -> ABA ->
// integrate; NO contact solve). It is INDEPENDENT of nuka_world_step's
// production contact pipeline -- a tape rollout runs its own forward on the
// world's articulation state. The floating-base ORIENTATION channel is deferred
// from p02-A: pick contact-free, near-fixed-orientation trajectories.
//
// Usage:
//   nuka_tape_handle tape;
//   nuka_tape_desc_t d = { .checkpoint_interval=10, .max_tape_entries=1024,
//                          .max_checkpoints=128, .recompute_on_backward=1 };
//   nuka_tape_create(world, &d, &tape);
//   for (step) { write actions via the buffer view; nuka_world_step_with_tape(world, tape); }
//   nuka_tape_backward(tape, grad_obs_in, grad_actions_out, grad_parameters_out);
//   nuka_tape_reset(tape);                 // reuse for the next rollout
//   nuka_tape_destroy(tape);
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nuka_tape_t* nuka_tape_handle;

// Tape configuration. checkpoint_interval (K): a checkpoint snapshot is captured
// every K forward steps (step 0 always). max_tape_entries / max_checkpoints size
// the device storage once. recompute_on_backward: 1 = checkpointed reverse
// (recompute windows; memory ~ N/K checkpoints); 0 = full-tape debug mode (a
// checkpoint EVERY step, single reverse pass, no recompute -- the CI oracle).
typedef struct nuka_tape_desc_t {
    uint32_t checkpoint_interval;
    uint32_t max_tape_entries;
    uint32_t max_checkpoints;
    uint32_t recompute_on_backward;
} nuka_tape_desc_t;

// Creates a tape bound to `world`'s articulation state + drive gains + link
// masses. The world must be articulated (single-env for the contact-free p02
// slice; the differentiable forward is independent of the world's stepper).
// Returns NUKA_RESULT_NOT_SUPPORTED for a non-articulated world.
nuka_result_t nuka_tape_create(nuka_world_handle world,
                               const nuka_tape_desc_t* desc,
                               nuka_tape_handle* out);

void nuka_tape_destroy(nuka_tape_handle tape);

// Advances the tape's differentiable rollout by ONE contact-free PD step IN
// PLACE on the world's articulation state, recording the step's action
// (drive_targets) and capturing a checkpoint on the K-cadence. The action is
// read from the world's DRIVE_TARGET buffer (write it via
// nuka_world_get_buffer_view before each call), so the tape stays in lockstep
// with how a policy drives the world. NOTE: this does NOT run nuka_world_step's
// contact pipeline -- it runs the differentiable forward.
nuka_result_t nuka_world_step_with_tape(nuka_world_handle world,
                                        nuka_tape_handle tape);

// Runs the full reverse pass over the recorded rollout.
//   grad_observations_in : host, the loss adjoint on the FINAL state. Layout is
//                          [2*total_link_count + 6*total_link_count]:
//                          [0 .. n)      = dL/dq'      (joint position)
//                          [n .. 2n)     = dL/dqdot'   (joint velocity)
//                          [2n .. 2n+6n) = dL/d v_root' (per-link spatial vel,
//                                          6 floats/link; only the floating root
//                                          is nonzero in practice). May be null
//                          -> all-zero seed.
//   grad_actions_out     : host, [step_count * total_link_count]. The per-step
//                          action gradient (env-major per-link).
//   grad_parameters_out  : host, [total_link_count]. The §4 PARAMETER spine:
//                          dL/d(link mass), summed over all steps. May be null to
//                          skip the readback.
// Bit-exact across two runs (D1). step_count == the number of recorded steps.
nuka_result_t nuka_tape_backward(nuka_tape_handle tape,
                                 const float* grad_observations_in,
                                 float* grad_actions_out,
                                 float* grad_parameters_out);

// Clears the recorded rollout (steps + checkpoints) for reuse. Does NOT reset
// the world's state; the caller resets the world (nuka_world_reset) before the
// next rollout if desired.
nuka_result_t nuka_tape_reset(nuka_tape_handle tape);

// The number of forward steps currently recorded on the tape.
uint32_t nuka_tape_step_count(nuka_tape_handle tape);
// The per-step action / parameter width (== articulation total_link_count).
uint32_t nuka_tape_link_count(nuka_tape_handle tape);

#ifdef __cplusplus
}
#endif

#endif  // NUKA_NUKA_DIFFSIM_H
