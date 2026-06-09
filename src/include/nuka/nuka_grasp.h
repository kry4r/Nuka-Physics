#ifndef NUKA_NUKA_GRASP_H
#define NUKA_NUKA_GRASP_H
// ---------------------------------------------------------------------------
// Nuka C ABI -- the BATCHED GRASP WORLD surface (v0.8 A2). A MINIMAL extern "C"
// shim over nuka::runtime::coresident::BatchedUnifiedWorld so a Python PPO env
// (H1GraspEnv) can drive the validated batched grasp world: N parallel fixed-base
// 2-finger grippers each pinching the C7a cup by friction (NO table).
//
// WHY A C ABI (not a direct C++ class bind). The engine libnuka.so is built with
// g++-14; the nanobind binding (python/src/nuka_ext.cpp) is built with g++-10. A
// direct nb::class_<BatchedUnifiedWorld> would create a g++-10 <-> g++-14 C++ ABI
// boundary (STL container returns + mangled methods) -- a silent-corruption risk on
// a world trained on for hours. So this surface is plain C (compiler-agnostic),
// compiled INTO libnuka (g++-14), and nuka_ext.cpp binds THIS (g++-10), mirroring
// the existing nuka_world_* ABI exactly (opaque handle + nuka_result_t).
//
// The grasp world's scene is built by the SHARED, VALIDATED grasp_scene_factory
// (the SAME BatchedSceneTemplate the 21 nuka_batched_unified_world_test gates
// exercise), so a Python caller trains on a certified scene -- not a parallel,
// untested re-implementation.
//
// ERROR / HANDLE CONVENTIONS mirror nuka.h: every call returns nuka_result_t,
// out-params are zeroed on the error paths, NULL handle -> NUKA_RESULT_NULL_HANDLE,
// a bad argument -> NUKA_RESULT_INVALID_ARG, a missing cup asset ->
// NUKA_RESULT_FILE_NOT_FOUND. Buffer fills take a caller-owned float*/uint32* plus
// its expected length; a length mismatch is NUKA_RESULT_INVALID_ARG.
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"  // nuka_result_t, nuka_device_handle, stdint via nuka.h

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nuka_grasp_world_t* nuka_grasp_world_handle;

// Creation descriptor for a batched grasp world. cup_asset_path is the C7a cup
// .usda (the shared factory loads + cooks it to one convex hull); env_count is the
// number of parallel grippers; grip_force is the constant per-finger drive torque
// (N) the template seeds (per-step per-env torque is then driven via set_actions);
// friction_mu is the contact friction coefficient; gravity_z + dt are the integrator
// parameters (e.g. -9.81f, 1/240). env_count == 0 or dt <= 0 -> NUKA_RESULT_INVALID_ARG.
typedef struct nuka_grasp_world_desc_t {
    const char* cup_asset_path;  // path to the cup .usda (cooked to a convex hull).
    uint32_t    env_count;       // # parallel grasp envs (must be > 0).
    float       grip_force;      // constant per-finger grip torque the template seeds (N).
    float       friction_mu;     // per-contact isotropic friction coefficient.
    float       gravity_z;       // gravity Z (m/s^2), e.g. -9.81.
    float       fixed_dt;        // fixed integrator dt (s), e.g. 1/240. Must be > 0.
} nuka_grasp_world_desc_t;

// Create a batched grasp world from the cup asset. On success *out is a non-null
// handle; on any error *out is null and the result code says why. The device
// handle supplies the CUDA context the world runs on.
nuka_result_t nuka_grasp_world_create(nuka_device_handle device,
                                      const nuka_grasp_world_desc_t* desc,
                                      nuka_grasp_world_handle* out);
void nuka_grasp_world_destroy(nuka_grasp_world_handle world);

// ----- scalar getters (the obs/action buffer shapes a Python env needs) -------
// All write *out (zeroed on a null-handle/null-out error). action_dim ==
// per-env per-finger DOF count (the gripper's joint DOF count); num_fingertips ==
// # pinch contacts; bodies_per_env == # movable rigid bodies per env (the cup).
nuka_result_t nuka_grasp_world_env_count(nuka_grasp_world_handle world, uint32_t* out);
nuka_result_t nuka_grasp_world_action_dim(nuka_grasp_world_handle world, uint32_t* out);
nuka_result_t nuka_grasp_world_num_fingertips(nuka_grasp_world_handle world, uint32_t* out);
nuka_result_t nuka_grasp_world_bodies_per_env(nuka_grasp_world_handle world, uint32_t* out);

// ----- per-step action injection + step ---------------------------------------
// set_actions: env-major per-finger drive torque, length env_count*action_dim. The
// next step() applies it (replacing the constant template grip). A count mismatch
// is NUKA_RESULT_INVALID_ARG. step: advance EVERY env one fixed step. step_with_actions
// is the RL per-step driver (set_actions then step in one call).
nuka_result_t nuka_grasp_world_set_actions(nuka_grasp_world_handle world,
                                           const float* actions, size_t count);
nuka_result_t nuka_grasp_world_step(nuka_grasp_world_handle world);
nuka_result_t nuka_grasp_world_step_with_actions(nuka_grasp_world_handle world,
                                                 const float* actions, size_t count);

// ----- batched obs export (ONE bulk fill per buffer; host float* IS the numpy) --
// Fill the four caller-owned buffers from the LAST step()'s ExportObsState, ALL
// env-major. Expected lengths (a mismatch on any -> NUKA_RESULT_INVALID_ARG):
//   q                  : env_count * action_dim   (finger joint positions)
//   qdot               : env_count * action_dim   (finger joint velocities)
//   fingertip_world_pos: env_count * 3 * num_fingertips  (fingertip world xyz)
//   finger_normal_impulse: env_count * num_fingertips    (per-finger squeeze impulse)
// Any buffer may be NULL to skip it (its length is then ignored).
nuka_result_t nuka_grasp_world_export_obs(nuka_grasp_world_handle world,
                                          float* q, size_t q_len,
                                          float* qdot, size_t qdot_len,
                                          float* fingertip_world_pos, size_t ft_len,
                                          float* finger_normal_impulse, size_t imp_len);

// ----- cup pose / velocity + per-env contact signals (the reward inputs) -------
// Fill caller-owned buffers for ALL envs from Body(env,cup_local) / GraspReports():
//   cup_pose      : env_count * 7  (per env [px,py,pz, qw,qx,qy,qz], quat w-first)
//   cup_vel       : env_count * 6  (per env [vx,vy,vz, wx,wy,wz], linear then angular)
//   cup_z         : env_count      (cup height; == cup_pose[e*7+2], convenience)
//   cup_vz        : env_count      (cup vertical velocity; == cup_vel[e*6+2])
//   finger_contacts: env_count     (uint32, # fingertip<->cup manifold points/env)
// Any buffer may be NULL to skip it. A non-NULL buffer with the wrong length is
// NUKA_RESULT_INVALID_ARG.
nuka_result_t nuka_grasp_world_read_cup(nuka_grasp_world_handle world,
                                        float* cup_pose, size_t pose_len,
                                        float* cup_vel, size_t vel_len,
                                        float* cup_z, size_t z_len,
                                        float* cup_vz, size_t vz_len,
                                        uint32_t* finger_contacts, size_t fc_len);

// ----- per-env autoreset (deterministic randomization) ------------------------
// Reset the `count` envs in env_ids (each in [0,env_count)) to a seed-randomized
// cup pose + the nominal open gripper. Deterministic given `seed` (a re-run with
// the same seed produces identical states). count==0 (or env_ids==NULL) is a no-op
// OK; an out-of-range id -> NUKA_RESULT_INVALID_ARG (resets nothing).
nuka_result_t nuka_grasp_world_reset_envs(nuka_grasp_world_handle world,
                                          const uint32_t* env_ids, uint32_t count,
                                          uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_GRASP_H */
