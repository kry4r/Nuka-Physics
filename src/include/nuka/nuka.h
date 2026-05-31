#ifndef NUKA_NUKA_H
#define NUKA_NUKA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nuka_device_t* nuka_device_handle;
typedef struct nuka_world_t* nuka_world_handle;
typedef struct nuka_buffer_t* nuka_buffer_handle;

typedef enum nuka_result_t {
    NUKA_RESULT_OK = 0,
    NUKA_RESULT_INVALID_ARG = 1,
    NUKA_RESULT_NULL_HANDLE = 2,
    NUKA_RESULT_CUDA_ERROR = 3,
    NUKA_RESULT_FILE_NOT_FOUND = 4,
    NUKA_RESULT_PARSE_ERROR = 5,
    NUKA_RESULT_OUT_OF_MEMORY = 6,
    NUKA_RESULT_NOT_SUPPORTED = 7,
    NUKA_RESULT_INTERNAL = 100
} nuka_result_t;

typedef struct nuka_version_t {
    uint16_t major;
    uint16_t minor;
    uint16_t patch;
} nuka_version_t;

nuka_version_t nuka_get_version(void);
const char* nuka_result_message(nuka_result_t result);

typedef struct nuka_device_desc_t {
    uint32_t gpu_index;
    void* cuda_stream;
    uint8_t backend_selection_layer_enabled;
} nuka_device_desc_t;

nuka_result_t nuka_device_create(const nuka_device_desc_t* desc,
                                 nuka_device_handle* out);
void nuka_device_destroy(nuka_device_handle device);

typedef struct nuka_world_desc_t {
    const char* scene_path;
    uint32_t env_count;
    float fixed_dt;
    // Determinism level (p01-W4). 0 = D1 / Strong (the default when the desc is
    // zero-initialized): the current behavior, BIT-EXACT and reproducible (no
    // float atomics, fixed kernel order). 1 = D2 / Weak: the reserved escape
    // hatch for future atomic fast-paths. NOTE: today no hot kernel benefits
    // from atomics (every hotspot is <<<articulation_count, 32>>> with per-env
    // warp reductions -- there is no cross-env ordered reduction an atomic
    // variant would accelerate), so D2 currently selects the SAME kernels as D1
    // and is behaviorally identical; it is wired as the documented mechanism and
    // is NOT held to the D1 bit-exact bar. Any value > 1 is rejected with
    // NUKA_RESULT_INVALID_ARG.
    uint8_t determinism;
} nuka_world_desc_t;

nuka_result_t nuka_world_create_from_scene(nuka_device_handle device,
                                           const nuka_world_desc_t* desc,
                                           nuka_world_handle* out);
void nuka_world_destroy(nuka_world_handle world);
nuka_result_t nuka_world_step(nuka_world_handle world);
nuka_result_t nuka_world_step_n(nuka_world_handle world, uint32_t step_count);

// p03 per-env RESET (RL autoreset). Restores selected envs to the deterministic
// creation-time initial state -- the engine's INTERNAL authoritative buffers
// (floating-base base_pose, base/link spatial velocity, joint q/qd) plus a clear
// of the carried contact warm-start. A reset written through the writable buffer
// views (e.g. ARTICULATION_LINK_POSE) is NON-authoritative for a floating base
// (the integrator overwrites it from base_pose each step), so the reset MUST go
// through here. GPU-only, D1-deterministic: bit-identical across runs, and
// reset_envs leaves every un-listed env byte-for-byte unchanged.
//
// Only the batched (env_count > 1) path supports reset; the single-env oracle
// path returns NUKA_RESULT_NOT_SUPPORTED (the RL autoreset path is always
// batched).
//
// nuka_world_reset      -- reset ALL envs to the initial snapshot.
// nuka_world_reset_envs -- reset only the `count` envs listed in `env_ids`
//                          (each in [0, env_count); an out-of-range id returns
//                          NUKA_RESULT_INVALID_ARG and resets nothing). count==0
//                          (or env_ids==NULL with count==0) is a no-op OK.
nuka_result_t nuka_world_reset(nuka_world_handle world);
nuka_result_t nuka_world_reset_envs(nuka_world_handle world,
                                    const uint32_t* env_ids,
                                    uint32_t count);

typedef enum nuka_state_field_t {
    NUKA_FIELD_RIGID_BODY_TRANSFORM = 0,
    NUKA_FIELD_ARTICULATION_LINK_POSE = 1,
    NUKA_FIELD_JOINT_POSITION = 2,
    NUKA_FIELD_JOINT_VELOCITY = 3,
    NUKA_FIELD_OBSERVATIONS = 4,
    NUKA_FIELD_CONTACT_POINTS = 5,
    // WRITABLE (batched/multi-env path only). The per-env PD position-drive
    // target buffer the batched step reads every Step. The buffer view returned
    // for this field aliases the live device buffer that
    // BatchedArticulatedStepParams::drive_targets points at, so a caller may
    // write it IN PLACE (zero-copy) and the NEXT nuka_world_step picks up the
    // new targets. Layout is IDENTICAL to NUKA_FIELD_JOINT_POSITION: a
    // float[env_count * base_link_count], env-major, index
    // (env * base_link_count + link). Only links with a non-zero drive stiffness
    // (the actuated joints) act on their target; writes to other slots are
    // silently ignored by the solver. Gains (Kp/Kd/force_limits) stay at the
    // validated rest-pose hold values and are NOT exposed here. Read of this
    // field returns the CURRENT targets (initially the rest-pose hold targets).
    //
    // Per-env slot map (go2_stand.usda articulation, base_link_count == 13):
    //   slot 0          = ROOT / trunk link. NOT an actuated joint; its drive
    //                     target is a no-op. go2_stand.usda declares no free root
    //                     joint, so the root is a FIXED base: q[0]/qd[0] are not
    //                     meaningful joint DOFs and the base WORLD POSE is the
    //                     constant cooked pose at
    //                     NUKA_FIELD_ARTICULATION_LINK_POSE[env*base_link_count].
    //                     (For a free-floating-root scene that same pose slot
    //                     instead carries the LIVE integrated 6-DOF base pose, and
    //                     base velocity lives in the engine's link_velocity[root]
    //                     -- not currently exposed through this ABI.)
    //   slots 1..12     = the 12 actuated revolute leg joints (cooked order; the
    //                     Python harness maps cooked order -> the Unitree
    //                     convention). These are the only slots that respond to a
    //                     drive-target write and the only meaningful q/qd joint
    //                     entries.
    NUKA_FIELD_DRIVE_TARGET = 6,
    // READ (batched/multi-env path only). Per-LINK spatial velocity, env-major,
    // with the SAME indexing as NUKA_FIELD_ARTICULATION_LINK_POSE:
    // element_count == env_count * base_link_count, env e's base/root is element
    // e*base_link_count (root == local link 0). Each element is 6 contiguous
    // floats (24 bytes, element_stride_bytes == 6*sizeof(float)) in OMEGA-FIRST
    // spatial order: [wx,wy,wz, vx,vy,vz] (angular 0..2, linear 3..5).
    //
    // FRAME (read carefully -- this is the load-bearing contract for the policy):
    //   * ROOT / base slot (e*base_link_count, the floating-base trunk on
    //     go2_float.usda): the 6-vector is the live base spatial velocity
    //     expressed in the ROOT-LINK BODY frame -- it is ALREADY BODY-LOCAL, NOT
    //     world. (Confirmed from the engine's floating-base integrator: the linear
    //     part v[3..5] is rotated by R(base_rot) to obtain the world translation,
    //     and the angular part v[0..2] right-multiplies the orientation as a
    //     body-frame delta-quaternion. So no world->body rotation is needed; a
    //     policy harness applies only its own fixed axis/scale remap, NOT a
    //     world->body rotate -- doing both would double-rotate.) This root entry
    //     is CURRENT after Step(): the floating-base velocity integrate and the
    //     contact solve both write link_velocity[root] within the same Step, so
    //     unlike ARTICULATION_LINK_POSE it carries NO one-step lag.
    //   * NON-ROOT link slots (1..base_link_count-1): engine-internal Featherstone
    //     per-link velocities expressed in each link's OWN local (Featherstone)
    //     frame -- NOT body, NOT world. They are populated by the ABA pass and are
    //     non-zero, but their exact timing/frame is an implementation detail and is
    //     NOT a defined observation -- do NOT rely on them. They are exposed only
    //     for index symmetry with ARTICULATION_LINK_POSE; only the ROOT slot is
    //     meaningful for a floating-base base-velocity observation. (On a FIXED-
    //     base scene the root carries no free 6-DOF velocity; use go2_float.usda
    //     for a live base.)
    NUKA_FIELD_LINK_VELOCITY = 7,
    // WRITABLE (batched/multi-env path only). The per-env PD drive GAIN buffers
    // the batched step reads every Step (BatchedArticulatedStepParams::
    // drive_stiffness / drive_damping / drive_force_limits point straight at them).
    // The view aliases the live device buffer so a caller may write it IN PLACE
    // (zero-copy) and the NEXT nuka_world_step picks up the new gains -- IDENTICAL
    // mechanism and layout to NUKA_FIELD_DRIVE_TARGET: float[env_count *
    // base_link_count], env-major, index (env*base_link_count + link), stride
    // sizeof(float). Per-env slot map matches DRIVE_TARGET: slot 0 = ROOT (not an
    // actuated joint; its gain is a no-op), slots 1..12 = the 12 actuated leg
    // joints. To drive a trained Go2 policy at its training PD gains, write
    // STIFFNESS (Kp) = 20 and DAMPING (Kd) = 0.5 on slots 1..12 of every env.
    //   STIFFNESS: per-joint proportional gain Kp (tau += Kp*(target - q)).
    //   DAMPING  : per-joint derivative gain   Kd (tau -= Kd*qdot).
    //   FORCE_LIMIT: per-joint symmetric torque clamp |tau| <= limit (Nm).
    // Reads return the CURRENT gains (initially the cooked rest-hold gains).
    NUKA_FIELD_DRIVE_STIFFNESS = 8,
    NUKA_FIELD_DRIVE_DAMPING = 9,
    NUKA_FIELD_DRIVE_FORCE_LIMIT = 10,
    // READ (batched/multi-env path only). The engine's AUTHORITATIVE per-env
    // floating-base ROOT world pose -- the internal base_pose[articulation] buffer
    // that the floating-base integrator advances each Step and that
    // nuka_world_reset_envs restores from the creation-time snapshot. element_count
    // == env_count (ONE Transform per env, NOT per link), element_stride_bytes ==
    // sizeof(math::Transform) == 7 floats [px,py,pz, qw,qx,qy,qz] (quat w-first),
    // SAME element layout as the ROOT slot of NUKA_FIELD_ARTICULATION_LINK_POSE.
    //
    // WHY THIS EXISTS (vs the root slot of ARTICULATION_LINK_POSE): the link_pose
    // FK runs at stage 4 from the PRE-integrate base, while base_pose is advanced
    // at stage 11, so after Step() returns ARTICULATION_LINK_POSE[root] is the
    // PREVIOUS step's base pose (one-step lag). base_pose has NO such lag: it is
    // current after Step() AND -- the load-bearing property for RL autoreset -- it
    // is correct IMMEDIATELY after nuka_world_reset_envs with no further step (the
    // reset writes it directly, whereas the lagged link_pose still shows the
    // pre-reset/fallen pose until the next Step's FK). A vectorized RL env reads
    // THIS field for the post-reset base orientation (projected_gravity) of the
    // just-reset envs. The non-reset envs may keep reading the (validated, golden-
    // pinned) lagged ARTICULATION_LINK_POSE -- the two intentionally differ by one
    // integration step during normal stepping.
    NUKA_FIELD_BASE_POSE = 11
} nuka_state_field_t;

typedef struct nuka_buffer_view_t {
    void* device_ptr;
    size_t element_count;
    uint32_t element_stride_bytes;
    uint8_t dtype;
} nuka_buffer_view_t;

nuka_result_t nuka_world_get_buffer_view(nuka_world_handle world,
                                         nuka_state_field_t field,
                                         nuka_buffer_view_t* out);

typedef struct nuka_invariant_violation_t {
    uint32_t invariant;
    uint32_t step;
    uint32_t env_id;
    float value;
    float threshold;
} nuka_invariant_violation_t;

nuka_result_t nuka_world_get_last_invariant_violations(nuka_world_handle world,
                                                       uint32_t* out_count,
                                                       void* out_array,
                                                       uint32_t array_capacity);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_H */
