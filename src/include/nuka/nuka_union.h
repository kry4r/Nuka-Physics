#ifndef NUKA_NUKA_UNION_H
#define NUKA_NUKA_UNION_H
// ---------------------------------------------------------------------------
// Nuka C ABI -- the H1 WHOLE-BODY UNION WORLD surface (v0.8 G2). A MINIMAL
// extern "C" shim over nuka::runtime::coresident::BatchedUnifiedWorld carrying
// the COMPLETE G1d union scene: N parallel floating-base 51-DOF whole-body H1s
// (h1_with_hand cook), each with the 10.9 cm cup SETTLED in its curled right
// hand, feet on a static ground, and a static table seated under the cup-proxy
// -- feet x ground + finger x cup + cup-proxy x table resolved in ONE solve.
//
// CONSTRUCTION runs the FULL deterministic authoring inside libnuka (the G2
// "settle-pre-roll seam"): MJCF cook -> bent stance -> base seat at FK -> right-
// hand curl -> placement search at the seat FK -> 200-step ORACLE SETTLE
// pre-roll (stance PD + ankle CoP law + curl hold) -> hand-frame cup carry ->
// table height 2 mm into the settled cup bottom. Same inputs -> byte-same
// initial state (D1). See src/runtime/coresident/h1_union_scene_factory.hpp.
//
// WHY A C ABI (not a C++ class bind): same reason as nuka_grasp.h -- the
// engine is g++-14, the nanobind binding g++-10; only plain C crosses.
//
// ACTION LAYOUT. Actions are per-DOF TORQUES, env-major, env_count*action_dim
// floats; action_dim == the FULL dof_stride (51 = 6 base + 45 joints). The 6
// base columns [0..5] are DEAD (the floating root has no actuator; the engine
// scatters them into the root link's drive slot, which the ABA drive ignores)
// -- by spec convention the caller's action mask zeroes them. Columns [6..50]
// are the 45 single-DOF joints in prefix-sum (DofIndexOf) order; the exported
// q/qdot at column d are the SAME link's q/qdot, so a host PD can be computed
// straight off the obs export.
//
// ERROR / HANDLE CONVENTIONS mirror nuka.h (nuka_result_t, zeroed out-params,
// NULL handle -> NUKA_RESULT_NULL_HANDLE, length mismatch -> INVALID_ARG,
// missing asset -> FILE_NOT_FOUND).
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"  // nuka_result_t, nuka_device_handle, stdint/stddef

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nuka_union_world_t* nuka_union_world_handle;

// Creation descriptor. NULL/empty paths select the factory's repo-relative
// defaults (.nuka-assets/...; the process must then run from the repo root).
// gravity_z == 0 -> -9.81f; fixed_dt <= 0 -> 1/240f (THE sentinel convention:
// both the python binding and the C++ reference pass 0 so the constructed
// worlds are byte-identical -- no host-language float-literal seam).
typedef struct nuka_union_world_desc_t {
    const char* h1_mjcf_path;   /* NULL/"" -> default h1_with_hand.xml         */
    const char* cup_usda_path;  /* NULL/"" -> default cup model.usda           */
    uint32_t    env_count;      /* # parallel union envs (must be > 0)         */
    float       gravity_z;      /* 0 -> -9.81f                                 */
    float       fixed_dt;       /* <= 0 -> 1/240f                              */
} nuka_union_world_desc_t;

// Scalar shape metadata (filled by nuka_union_world_dims).
typedef struct nuka_union_world_dims_t {
    uint32_t env_count;
    uint32_t action_dim;       /* == dof_stride (51): 6 dead base + 45 joints */
    uint32_t base_dof;         /* 6 (FloatingBase root)                        */
    uint32_t num_fingertips;   /* 30 (the H1.1 wrap)                           */
    uint32_t num_feet;         /* 4 (toe/heel per ankle)                       */
    uint32_t link_count;       /* per-env device links                         */
    uint32_t bodies_per_env;   /* 1 (the cup)                                  */
    uint32_t cup_local_index;  /* which per-env body is the cup (0)            */
    uint32_t num_grip_dofs;    /* 12 (the wrap-driven close columns)           */
    uint32_t drive_table_len;  /* entries per reference drive table            */
} nuka_union_world_dims_t;

// Scene scalars (filled by nuka_union_world_scene_info).
typedef struct nuka_union_scene_info_t {
    float    table_height;       /* static table plane z                       */
    float    cup_mass;           /* 0.2 kg                                     */
    float    total_mass;         /* Σ H1 link masses (kg)                      */
    float    poly_cx;            /* seat-time foot-polygon center x            */
    float    dt;                 /* the world's fixed dt                       */
    float    gravity_z;          /* the world's gravity                        */
    float    ankle_settle_tau[2];/* settle-final CoP ankle torques (L, R)      */
    uint32_t place_found;        /* 1 == the placement search succeeded        */
} nuka_union_scene_info_t;

nuka_result_t nuka_union_world_create(nuka_device_handle device,
                                      const nuka_union_world_desc_t* desc,
                                      nuka_union_world_handle* out);
void nuka_union_world_destroy(nuka_union_world_handle world);

nuka_result_t nuka_union_world_dims(nuka_union_world_handle world,
                                    nuka_union_world_dims_t* out);
nuka_result_t nuka_union_world_scene_info(nuka_union_world_handle world,
                                          nuka_union_scene_info_t* out);

// ----- reference choreography metadata -----------------------------------------
// The factory's PD drive tables (hold / rest / close; `which` = 0 / 1 / 2): the
// proven lift-choreography reference (curl held at the settled q; wrap backed
// off -0.25 rad; the H1.1 close PD at +0.18 rad), plus an ankle POSTURE stand-in
// for the test-side CoP balance law. Each non-NULL array must be len ==
// dims.drive_table_len. tau_d = kp*(target - q[dof]) - kd*qdot[dof], clamped to
// +/-tlim when tlim > 0; grip[i] == 1 marks the 12 wrap-driven close links.
// A choreography aid for smokes/BITEs -- G3's RL policy replaces it.
nuka_result_t nuka_union_world_drive_table(nuka_union_world_handle world,
                                           uint32_t which,
                                           uint32_t* dof, float* target,
                                           float* kp, float* kd, float* tlim,
                                           uint8_t* grip, size_t len);

// The 12 grip (wrap-driven) action columns -- the BITE kill-switch columns.
nuka_result_t nuka_union_world_grip_dofs(nuka_union_world_handle world,
                                         uint32_t* dofs, size_t len);

// Per-DOF |torque| limits (len == action_dim; 0 on the 6 dead base columns):
// the "random torques within per-joint limits" envelope (legs at the MJCF motor
// limits, torso 200, shoulder 40 / yaw 18, elbow 18, wrist 6, fingers 1 N*m).
nuka_result_t nuka_union_world_dof_limits(nuka_union_world_handle world,
                                          float* limits, size_t len);

// ----- per-step action injection + step ----------------------------------------
nuka_result_t nuka_union_world_set_actions(nuka_union_world_handle world,
                                           const float* actions, size_t count);
nuka_result_t nuka_union_world_step(nuka_union_world_handle world);
nuka_result_t nuka_union_world_step_with_actions(nuka_union_world_handle world,
                                                 const float* actions, size_t count);

// ----- per-env seams ------------------------------------------------------------
// env < 0 -> ALL envs. Disabling stops emitting that env's cup x table pair on
// the next step (the lift choreography's "remove the table"); re-enabling
// restores it. table_enabled reads one env's live toggle.
nuka_result_t nuka_union_world_set_table_enabled(nuka_union_world_handle world,
                                                 int32_t env, uint32_t enabled);
nuka_result_t nuka_union_world_table_enabled(nuka_union_world_handle world,
                                             uint32_t env, uint32_t* out);

// Overwrite env e's floating-base pose (pose7 = [px,py,pz, qw,qx,qy,qz]) -- the
// per-env IC seam (base randomization / airborne tests). OFF the step path.
nuka_result_t nuka_union_world_set_base_pose(nuka_union_world_handle world,
                                             uint32_t env, const float* pose7);

// Reset the listed envs: gripper restored to the SETTLED proto q (velocities
// zeroed), cup restored to the settled in-hand placement with a seed-
// deterministic XY jitter (+/-2.5 cm default), table toggle restored to ON.
// Same seed -> byte-same post-reset state (D1). NOTE: velocities are zeroed,
// so the post-reset state is the settled POSE at rest, not the (slightly
// moving) construction-time snapshot.
nuka_result_t nuka_union_world_reset_envs(nuka_union_world_handle world,
                                          const uint32_t* env_ids, uint32_t count,
                                          uint64_t seed);

// ----- batched obs export (ONE bulk fill per buffer; env-major) ----------------
// Expected lengths (any buffer may be NULL to skip; non-NULL + wrong length ->
// INVALID_ARG). n = env_count, ad = action_dim, nf = num_fingertips:
//   q                    : n*ad   (cols 0..5 = root filler; 6..50 joint q)
//   qdot                 : n*ad   (cols 0..5 = base spatial vel; 6..50 joint qdot)
//   base_pose            : n*7    [px,py,pz, qw,qx,qy,qz]
//   base_vel             : n*6    (root spatial velocity == qdot cols 0..5)
//   fingertip_world_pos  : n*3*nf (xyz per fingertip; LAST step's FK -- zeros
//                                  before the first step / stale across reset)
//   finger_normal_impulse: n*nf   (>= 0; LAST step's normal-row impulses)
//   last_action          : n*ad   (the exact last set_actions bytes; zeros
//                                  before the first call)
nuka_result_t nuka_union_world_export_obs(nuka_union_world_handle world,
                                          float* q, size_t q_len,
                                          float* qdot, size_t qdot_len,
                                          float* base_pose, size_t bp_len,
                                          float* base_vel, size_t bv_len,
                                          float* fingertip_world_pos, size_t ft_len,
                                          float* finger_normal_impulse, size_t fi_len,
                                          float* last_action, size_t la_len);

// ----- cup pose/vel + per-env contact state (the reward seams) ------------------
// Expected lengths (any buffer may be NULL):
//   cup_pose       : n*7   [px,py,pz, qw,qx,qy,qz]
//   cup_vel        : n*6   [vx,vy,vz, wx,wy,wz]
//   foot_rows      : n     (uint32, # foot<->ground NORMAL rows last step)
//   foot_impulse   : n     (float, Σλ over foot normal rows; cast from double)
//   table_rows     : n     (uint32, # cup<->table rows last step)
//   table_impulse  : n     (float, Σλ*j_z over table rows; cast from double)
//   finger_contacts: n     (uint32, # fingertip<->cup manifold points)
nuka_result_t nuka_union_world_read_cup_contacts(nuka_union_world_handle world,
                                                 float* cup_pose, size_t cp_len,
                                                 float* cup_vel, size_t cv_len,
                                                 uint32_t* foot_rows, size_t fr_len,
                                                 float* foot_impulse, size_t fim_len,
                                                 uint32_t* table_rows, size_t tr_len,
                                                 float* table_impulse, size_t tim_len,
                                                 uint32_t* finger_contacts, size_t fc_len);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_UNION_H */
