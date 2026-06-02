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
    // Stage-1 control mode (v0.5 C-fwd). 0 = PDPosition (the default when the
    // desc is zero-initialized): the legacy PD position drive, BYTE-FOR-BYTE
    // unchanged. 1 = Torque (tau = clamp(NUKA_FIELD_TORQUE_INPUT)). 2 = Velocity
    // (tau = clamp(drive_stiffness*(NUKA_FIELD_VELOCITY_TARGET - qdot))). 3 =
    // ComputedTorque (inverse-dynamics PD: tau = M*(Kp*(DRIVE_TARGET - q) -
    // Kd*qdot) + bias, Kp/Kd reuse DRIVE_STIFFNESS/DRIVE_DAMPING). 5 = Actuator
    // (DC-motor torque-speed envelope on NUKA_FIELD_TORQUE_INPUT; tau_stall reuses
    // DRIVE_FORCE_LIMIT, no-load speed is NUKA_FIELD_ACTUATOR_NOLOAD_SPEED). Value
    // 4 = Osc (operational-space control, FORWARD position-task mode: tau =
    // J^T*Lambda*(Kp*e_x + Kd*edot_x) driving the world position of the task link
    // `osc_task_link` toward NUKA_FIELD_TASK_TARGET; Kp/Kd reuse DRIVE_STIFFNESS/
    // DRIVE_DAMPING at the task link; bounded to a 3-DOF position task on a fixed-
    // base scene). Any value > 5 is rejected with NUKA_RESULT_INVALID_ARG. Non-PD
    // modes require the BATCHED path
    // (env_count > 1); the single-env oracle path supports PDPosition only and
    // returns NUKA_RESULT_NOT_SUPPORTED for a non-PD mode (the golden oracle
    // drive site stays untouched). The implicit joint damping (#43, driven by
    // DRIVE_DAMPING) is a joint property orthogonal to the control law and runs
    // for every mode.
    uint8_t control_mode;
    // v0.5 C-fwd slice 3 (p03 R2): the OSC task link, an articulation-LOCAL link
    // index (0-based within the cooked articulation; global = env*base_link_count +
    // osc_task_link). Read ONLY when control_mode == 4 (Osc): it selects the link
    // whose world position the 3-DOF position task drives toward
    // NUKA_FIELD_TASK_TARGET. Zero-initialized (== 0, the ROOT) is the default; for
    // a fixed-base scene the root has a zero task Jacobian (the controller is a
    // no-op), so an Osc caller MUST set this to a real end-effector (e.g. a Go2
    // foot/calf local link). Ignored by every non-Osc mode (so a zero-init desc for
    // a PD/Torque/etc world is unaffected). An index >= the articulation's link
    // count makes the Osc drive a no-op (tau left unchanged).
    uint32_t osc_task_link;
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
    NUKA_FIELD_BASE_POSE = 11,
    // WRITABLE (batched/multi-env path only). The per-env per-link TORQUE input
    // buffer the batched step reads every Step when the world's control_mode is
    // Torque (1). Aliases the live device buffer (zero-copy, write IN PLACE and
    // the NEXT nuka_world_step picks it up). IDENTICAL layout to
    // NUKA_FIELD_DRIVE_TARGET: float[env_count * base_link_count], env-major,
    // index (env*base_link_count + link), stride sizeof(float). Per-env slot map
    // matches DRIVE_TARGET: slot 0 = ROOT (inert), slots 1..N = actuated joints.
    // The applied torque is clamped to +/- DRIVE_FORCE_LIMIT (when > 0). Reads
    // return the CURRENT torque input (initially all zero). Returns
    // NUKA_RESULT_NOT_SUPPORTED on the single-env path. (Allocated even for a PD
    // world; it is simply never read there.)
    NUKA_FIELD_TORQUE_INPUT = 12,
    // WRITABLE (batched/multi-env path only). The per-env per-link VELOCITY
    // TARGET buffer the batched step reads every Step when the world's
    // control_mode is Velocity (2). Same zero-copy aliasing, layout and slot map
    // as NUKA_FIELD_TORQUE_INPUT. The velocity-servo torque is
    // drive_stiffness*(velocity_target - qdot) clamped to +/- DRIVE_FORCE_LIMIT
    // (Kp_v reuses the DRIVE_STIFFNESS buffer). Reads return the CURRENT target
    // (initially all zero). Returns NUKA_RESULT_NOT_SUPPORTED on the single-env
    // path. (Allocated even for a PD world; it is simply never read there.)
    NUKA_FIELD_VELOCITY_TARGET = 13,
    // WRITABLE (batched/multi-env path only). v0.5 C-fwd slice 2: the per-env
    // per-link DC-motor NO-LOAD SPEED buffer the batched step reads every Step
    // when the world's control_mode is Actuator (5). Same zero-copy aliasing,
    // layout and slot map as NUKA_FIELD_TORQUE_INPUT. In Actuator mode the
    // commanded torque (NUKA_FIELD_TORQUE_INPUT) is clamped to the velocity-
    // dependent DC-motor envelope tau_max(qdot) = clamp(tau_stall*(1 -
    // |qdot|/qdot_noload), 0, tau_stall), where tau_stall reuses the
    // DRIVE_FORCE_LIMIT buffer and qdot_noload is THIS buffer. A per-link value
    // <= 0 disables the speed term (falls back to the plain DRIVE_FORCE_LIMIT
    // clamp == Torque mode), so an all-zero buffer is safe. Reads return the
    // CURRENT no-load speeds (initially all zero). Returns
    // NUKA_RESULT_NOT_SUPPORTED on the single-env path. (Allocated even for a
    // non-Actuator world; it is simply never read there.)
    NUKA_FIELD_ACTUATOR_NOLOAD_SPEED = 14,
    // WRITABLE (batched/multi-env path only). v0.5 C-fwd slice 3 (p03 R2): the
    // per-ENV operational-space-control TASK TARGET buffer the batched step reads
    // every Step when the world's control_mode is Osc (4). UNLIKE the per-LINK
    // control-input fields (TORQUE_INPUT / VELOCITY_TARGET / ACTUATOR_NOLOAD_SPEED),
    // this is per-ENV: element_count == env_count, element_stride_bytes == 3*sizeof
    // (float) (a float3 {x,y,z} WORLD position per env), index env*3. It is the
    // world position the task link (the desc's osc_task_link) is driven toward.
    // Same zero-copy aliasing as the other writable fields (write IN PLACE; the
    // NEXT nuka_world_step picks it up). Reads return the CURRENT targets (initially
    // all zero == the world origin). Returns NUKA_RESULT_NOT_SUPPORTED on the
    // single-env path. (Allocated for every batched world so the view always
    // succeeds; only read in Osc mode.)
    NUKA_FIELD_TASK_TARGET = 15,
    // READ (batched/multi-env path only). p14a (v0.7): the NET per-LINK contact
    // WRENCH (force + torque) in the WORLD frame, aggregated over every contact
    // incident to the link. Maintained EAGERLY on the normal forward path (the
    // batched Step computes it after the contact solve, so it is always queryable
    // after a Step -- the MuJoCo-style "net contact force on this body" readout).
    //
    // LAYOUT: float[total_link_count * 6], total_link_count == env_count *
    // base_link_count, indexed by GLOBAL link index g == env*base_link_count +
    // local_link (the SAME env-major link indexing as ARTICULATION_LINK_POSE /
    // LINK_VELOCITY). element_count == env_count*base_link_count,
    // element_stride_bytes == 6*sizeof(float). Per element g:
    //   [g*6+0 .. g*6+2] = world contact FORCE F (N): sum over the link's contacts
    //                      of (n*lambda_n + t1*lambda_t1 + t2*lambda_t2)/dt, where
    //                      lambda is the solved per-contact IMPULSE (N*s) and dt is
    //                      the step dt (force = impulse/dt). +Z is up; a foot in a
    //                      stance carries a POSITIVE F_z (the ground pushes up).
    //   [g*6+3 .. g*6+5] = world contact TORQUE tau (N*m) about the LINK FRAME
    //                      ORIGIN (NOT the link COM): sum over the link's contacts
    //                      of (contact_point - link_frame_origin_world) x F_slot.
    //                      The reference point is the link frame origin
    //                      (ARTICULATION_LINK_POSE[g].position); a consumer that
    //                      wants the torque about the COM must shift it by the
    //                      link's COM offset.
    // Links with no incident contact this step read all-zero. dt is the dt of the
    // most-recent Step.
    //
    // SCOPE (v0.7 entry): this currently aggregates the RIGID FOOT-VS-GROUND
    // contacts only -- the sole contact model wired into the per-slot contact array
    // today. When p08 (SDF) / p11 (coupling) contact rows land they MUST be plumbed
    // into the same per-slot array for this to remain the FULL net wrench; until
    // then a link touched only by a not-yet-plumbed contact model reads zero here.
    // (NUKA_FIELD_CONTACT_NORMAL carries the matching caveat.)
    // Returns NUKA_RESULT_NOT_SUPPORTED on the single-env path.
    NUKA_FIELD_LINK_CONTACT_WRENCH = 16,
    // READ (batched/multi-env path only). p14a (v0.7): per-CONTACT-SLOT world unit
    // NORMAL. Together with CONTACT_POINTS / CONTACT_FORCE / CONTACT_LINK this is
    // the per-contact enumeration surface (the mj_contactForce parity: {point,
    // normal, normal-force, friction-force, link} per contact). Aliases the
    // engine's internal per-slot contact_normal buffer (zero-copy).
    //
    // LAYOUT: identical to CONTACT_POINTS -- Vec3 (3 floats) per contact slot,
    // slot_count == env_count * kMaxFootContactsPerEnv (4 per env), env-major.
    // element_count == slot_count, element_stride_bytes == 3*sizeof(float).
    // INACTIVE slots are zero. (For the current rigid foot-vs-ground detector the
    // active normal is always world +Z = (0,0,1).) Returns
    // NUKA_RESULT_NOT_SUPPORTED on the single-env path.
    NUKA_FIELD_CONTACT_NORMAL = 17,
    // READ (batched/multi-env path only). p14a (v0.7): per-CONTACT-SLOT contact
    // FORCE in the CONTACT BASIS -- the {Fn, Ft1, Ft2} = {lambda_n, lambda_t1,
    // lambda_t2}/dt components (normal force + the two friction-force magnitudes),
    // i.e. the solved per-slot IMPULSE scaled by 1/dt. Fn is the mj-parity normal
    // force; (Ft1,Ft2) are the friction-force magnitudes along the contact's two
    // world friction tangents.
    //
    // LAYOUT: 3 floats per contact slot, slot_count == env_count *
    // kMaxFootContactsPerEnv (4 per env), env-major. element_count == slot_count,
    // element_stride_bytes == 3*sizeof(float). INACTIVE slots are zero. NOTE: the
    // tangent BASIS VECTORS (t1,t2) are NOT exposed, so the per-contact world
    // friction DIRECTION cannot be reconstructed from this field alone -- the net
    // per-link WORLD friction force is available in LINK_CONTACT_WRENCH (which sums
    // n*lambda_n + t1*lambda_t1 + t2*lambda_t2 in world). Returns
    // NUKA_RESULT_NOT_SUPPORTED on the single-env path.
    NUKA_FIELD_CONTACT_FORCE = 18,
    // READ (batched/multi-env path only). p14a (v0.7): per-CONTACT-SLOT owning
    // GLOBAL LINK index (uint32). The CSR-by-link grouping is implicit: slots are
    // env-major at fixed stride kMaxFootContactsPerEnv (4 per env), and each slot
    // carries the global link (g == env*base_link_count + local_link) the contact
    // acts on, so a consumer enumerates contacts and groups them by link via this
    // field. Aliases the engine's internal per-slot contact_link buffer (zero-copy).
    //
    // LAYOUT: 1 uint32 per contact slot, slot_count == env_count *
    // kMaxFootContactsPerEnv, env-major. element_count == slot_count,
    // element_stride_bytes == sizeof(uint32_t). dtype == 1 (UINT32) -- the ONLY
    // field with a non-float32 dtype. INACTIVE slots carry ~0u (0xFFFFFFFF, the
    // invalid-link sentinel). Returns NUKA_RESULT_NOT_SUPPORTED on the single-env
    // path.
    NUKA_FIELD_CONTACT_LINK = 19
} nuka_state_field_t;

typedef struct nuka_buffer_view_t {
    void* device_ptr;
    size_t element_count;
    uint32_t element_stride_bytes;
    // Element scalar type: 0 == float32 (every field except CONTACT_LINK), 1 ==
    // uint32 (CONTACT_LINK only -- the per-slot owning global link index).
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
