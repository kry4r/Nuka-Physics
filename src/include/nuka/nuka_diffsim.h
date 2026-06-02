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

// Sets the scalar MASS of one articulation link at runtime, reconstructing the
// link's 6x6 spatial inertia from the new mass via the SAME MakeSpatialInertia
// parameterization the engine cooked it with -- the link's COM (inertial-frame
// position) and unit-mass / rotated diagonal inertia tensor are HELD FIXED, so
// the spatial inertia is AFFINE in mass exactly as the diff-sim mass-gradient
// adjoint assumes (dI/dmass = MakeSpatialInertia(m+1) - MakeSpatialInertia(m),
// parallel-axis + COM coupling included). This is the §4 PARAMETER spine's
// forward write: the Python autograd layer pushes the optimized mass INTO the
// sim each forward via this setter so the true ∂output/∂mass matches the
// grad_parameters_out the backward returns (without it the sim would keep the
// baked mass and the gradient would be silently wrong).
//
//   link_index : GLOBAL link index in [0, total_link_count) -- the SAME ordering
//                BuildArticulationHostState concatenates the articulations and
//                grad_parameters_out / nuka_tape_link_count report. Returns
//                NUKA_RESULT_INVALID_ARG for an out-of-range index.
//   mass       : the new scalar mass (kg). Must be > 0 (a non-positive mass is
//                rejected with NUKA_RESULT_INVALID_ARG; MakeSpatialInertia's
//                mass<=0 guard would otherwise zero the inertia).
//
// The new spatial inertia is written into the world's live link_inertia device
// buffer (the SAME buffer the contact-free ABA forward -- and nuka_world_step --
// reads FRESH each step in AbaPass1's CopySpatialInertia / Mat66MulVec6(I,v));
// there is no derived/cached inertia to invalidate. The host mirror is updated
// too so a subsequent download stays consistent. Returns NUKA_RESULT_NOT_SUPPORTED
// for a non-articulated world. DETERMINISTIC: a host-driven single-link device
// write, no float atomics, byte-stable. Default behavior is UNCHANGED until
// called: a world that never calls this setter steps byte-identically.
nuka_result_t nuka_world_set_link_mass(nuka_world_handle world,
                                       uint32_t link_index, float mass);

// Sets the world's uniform gravity Z component (m/s^2). This is a plain world-
// property setter (NOT diff-sim math): it mutates step_options.gravity.z, which
// nuka_tape_create reads into RolloutParams.gravity_z at create time. It exists
// to enable the p04 A4 FLOATING mass-gradcheck, whose recipe REQUIRES g=0 so the
// deferred floating-base ORIENTATION channel (a_grav = R(base_rot)^T g) is inert
// and the finite-difference isolates the velocity channel the adjoint covers.
//
// ORDER MATTERS: call this BEFORE nuka_tape_create -- a tape created earlier has
// already captured the old gravity_z and will not see the change. Default behavior
// is UNCHANGED until called (gravity stays {0,0,-9.81}); a world that never calls
// this setter steps byte-identically (both the production stepper and the tape
// forward). Returns NUKA_RESULT_NULL_HANDLE for a null world. DETERMINISTIC: a
// host-side scalar write, no device work.
nuka_result_t nuka_world_set_gravity_z(nuka_world_handle world, float gravity_z);

// The number of forward steps currently recorded on the tape.
uint32_t nuka_tape_step_count(nuka_tape_handle tape);
// The per-step action / parameter width (== articulation total_link_count).
uint32_t nuka_tape_link_count(nuka_tape_handle tape);

// Returns a ZERO-COPY device-direct view into the tape's LIVE differentiable
// state (the orchestrator's ArticulationDeviceState, advanced in place by
// nuka_world_step_with_tape). This is the OBSERVATION side of the single-env
// tape: unlike nuka_world_get_buffer_view's single-env arm (which rebuilds from
// the world's HOST MIRROR, updated only by nuka_world_step / step_n -- NOT by
// step_with_tape, so it reads stale 0.0 after tape steps), this serves the
// device pointers the tape forward actually evolves. Mirrors
// nuka_world_get_buffer_view's contract (null-guards `out`, zeros its fields).
//   NUKA_FIELD_JOINT_POSITION -> state.q,     element_count == link_count,
//                                stride sizeof(float).
//   NUKA_FIELD_JOINT_VELOCITY -> state.qdot,  element_count == link_count,
//                                stride sizeof(float).
//   NUKA_FIELD_LINK_VELOCITY  -> state.link_velocity, element_count ==
//                                link_count, stride == 6*sizeof(float) (a
//                                LinkSpatialVel, omega-first [wx,wy,wz,vx,vy,vz];
//                                the root slot is the live base spatial velocity
//                                in the root-link body frame).
//   else                      -> NUKA_RESULT_NOT_SUPPORTED.
// Returns NUKA_RESULT_NULL_HANDLE if the tape is missing, NUKA_RESULT_INVALID_ARG
// if `out` is null. READ-only (the views back the policy observation; do not
// write through them -- write the action via the world's DRIVE_TARGET buffer).
nuka_result_t nuka_tape_state_view(nuka_tape_handle tape,
                                   nuka_state_field_t field,
                                   nuka_buffer_view_t* out);

// ---------------------------------------------------------------------------
// Sparse linear solver backend selection (v0.7 p01)
// ---------------------------------------------------------------------------
//
// The diff-sim / general sparse-solver path solves its SPD systems through the
// SparseLinearSolver seam. v0.5 shipped exactly one backend -- the self-written
// deterministic CG (Jacobi / Block-Jacobi preconditioned, D1 bit-exact). v0.7
// p01 hardens + generalizes that SAME core for larger / worse-conditioned
// systems; MINRES / GMRES are reserved for later v0.7 phases (they build on this
// CG core and are NOT YET implemented -> selecting them returns
// NUKA_RESULT_NOT_SUPPORTED).
//
// The default backend is NUKA_SOLVER_BACKEND_SELF_CG; a fresh world already uses
// it, so calling the setter is only needed to be explicit or (later) to switch.
typedef enum nuka_sparse_solver_backend_t {
    NUKA_SOLVER_BACKEND_SELF_CG = 0,      /* self-written CG; default (v0.5) */
    NUKA_SOLVER_BACKEND_SELF_MINRES = 1,  /* self-written MINRES (v0.7 p02) */
    NUKA_SOLVER_BACKEND_SELF_GMRES = 2,   /* self-written GMRES (v0.7 p03-A) */
} nuka_sparse_solver_backend_t;

// Selects the sparse linear solver backend for `world`'s diff-sim / general
// solver path. NUKA_SOLVER_BACKEND_SELF_CG is accepted (and is the default);
// the reserved MINRES / GMRES values return NUKA_RESULT_NOT_SUPPORTED until
// their phases land; any other value returns NUKA_RESULT_INVALID_ARG. Returns
// NUKA_RESULT_NULL_HANDLE for a null world. A plain host-side scalar write; no
// device work. The selection is read at solver-construction time.
nuka_result_t nuka_world_set_sparse_solver_backend(
    nuka_world_handle world, nuka_sparse_solver_backend_t backend);

// Reads back the currently selected sparse solver backend into `*out`. Returns
// NUKA_RESULT_NULL_HANDLE for a null world, NUKA_RESULT_INVALID_ARG if `out` is
// null. Round-trips the value set by nuka_world_set_sparse_solver_backend (the
// default is NUKA_SOLVER_BACKEND_SELF_CG).
nuka_result_t nuka_world_get_sparse_solver_backend(
    nuka_world_handle world, nuka_sparse_solver_backend_t* out);

#ifdef __cplusplus
}
#endif

#endif  // NUKA_NUKA_DIFFSIM_H
