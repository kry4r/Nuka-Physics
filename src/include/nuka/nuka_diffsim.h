#ifndef NUKA_NUKA_DIFFSIM_H
#define NUKA_NUKA_DIFFSIM_H

// Recorded PD adjoints use explicit damping without the production contact solve.
// Floating-base pose derivatives are incomplete for multi-step rollouts.

#include "nuka/nuka.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nuka_tape_t* nuka_tape_handle;

// Checkpoints capture the pre-step state at the given interval, including step zero.
// A zero recompute_on_backward stores a checkpoint at every step.
typedef struct nuka_tape_desc_t {
    uint32_t checkpoint_interval;
    uint32_t max_tape_entries;
    uint32_t max_checkpoints;
    uint32_t recompute_on_backward;
} nuka_tape_desc_t;

// Binds a tape to one PD-controlled articulation environment; other modes and plastic state are unsupported.
// The world must remain alive while stepping, reading state, or running backward.
nuka_result_t nuka_tape_create(nuka_world_handle world,
                               const nuka_tape_desc_t* desc,
                               nuka_tape_handle* out);

void nuka_tape_destroy(nuka_tape_handle tape);

// Records current DRIVE_TARGET values and advances one contact-free explicit-damping PD step.
// This recorded map differs from nuka_world_step's production implicit damping and contacts.
nuka_result_t nuka_world_step_with_tape(nuka_world_handle world,
                                        nuka_tape_handle tape);

// Host seeds pack [q(n), qdot(n), link spatial velocity(6n)]; null means zero seeds.
// Host outputs hold step_count*n action gradients and optional n scalar-mass gradients.
nuka_result_t nuka_tape_backward(nuka_tape_handle tape,
                                 const float* grad_observations_in,
                                 float* grad_actions_out,
                                 float* grad_parameters_out);

// Clears tape records and checkpoints; the caller resets world state separately.
nuka_result_t nuka_tape_reset(nuka_tape_handle tape);

// Sets one template link's finite positive mass in every environment without reallocating.
// COM and rotational inertia at the COM stay fixed; model and device inertia update together.
nuka_result_t nuka_world_set_link_mass(nuka_world_handle world,
                                       uint32_t link_index, float mass);

// Sets finite uniform gravity Z (m/s^2), preserving X/Y, for the next production step.
// Changed gravity invalidates the execution graph; tapes retain their creation parameters.
nuka_result_t nuka_world_set_gravity_z(nuka_world_handle world, float gravity_z);

uint32_t nuka_tape_step_count(nuka_tape_handle tape);
uint32_t nuka_tape_link_count(nuka_tape_handle tape);

// Read-only live views support JOINT_POSITION, JOINT_VELOCITY, and LINK_VELOCITY.
// A missing tape or destroyed world returns NULL_HANDLE and clears the output view.
nuka_result_t nuka_tape_state_view(nuka_tape_handle tape,
                                   nuka_state_field_t field,
                                   nuka_buffer_view_t* out);

typedef enum nuka_sparse_solver_backend_t {
    NUKA_SOLVER_BACKEND_SELF_CG = 0,
    NUKA_SOLVER_BACKEND_SELF_MINRES = 1,
    NUKA_SOLVER_BACKEND_SELF_GMRES = 2,
} nuka_sparse_solver_backend_t;

// Selects the sparse solver implementation used when constructing a solver.
// CG is the default; unknown values return INVALID_ARG.
nuka_result_t nuka_world_set_sparse_solver_backend(
    nuka_world_handle world, nuka_sparse_solver_backend_t backend);

nuka_result_t nuka_world_get_sparse_solver_backend(
    nuka_world_handle world, nuka_sparse_solver_backend_t* out);

#ifdef __cplusplus
}
#endif

#endif  // NUKA_NUKA_DIFFSIM_H
