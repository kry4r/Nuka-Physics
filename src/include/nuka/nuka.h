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
    // Go2-on-stairs Phase 2a: procedural-terrain cook config (the shared
    // nuka::terrain::TerrainParams, minus ground_height which stays the model's).
    // These 5 floats are MODEL-LEVEL (one set for the whole world, NOT per env);
    // the per-env terrain TYPE + DIFFICULTY are written through
    // NUKA_FIELD_ENV_TERRAIN_TYPE / NUKA_FIELD_ENV_TERRAIN_DIFFICULTY. They are
    // applied AFTER cook + BEFORE the nk::World is constructed (the Pipeline reads
    // model.terrain at construct time). A zero-initialized desc leaves ALL FIVE at
    // 0.0 => the cook's flat terrain (SampleTerrainHeight returns ground_height for
    // every env regardless of type) => BYTE-IDENTICAL to a world created without
    // terrain (the D1 default). A caller that wants stairs sets terrain_step_height
    // / terrain_step_width / terrain_platform_width (PyramidStairs/InvertedPyramid)
    // or terrain_grid_width / terrain_grid_height_max (RandomBoxes), then writes a
    // non-zero terrain TYPE per env. See nuka::terrain::TerrainParams for the units.
    float terrain_step_height;      // vertical rise per step ring.
    float terrain_step_width;       // horizontal width of each step ring.
    float terrain_platform_width;   // full edge length of the central top platform.
    float terrain_grid_width;       // cell edge length (RandomBoxes).
    float terrain_grid_height_max;  // max per-cell random rise (RandomBoxes).
    // Multi-articulation CO-RESIDENCE (M10 dog-dog collision foundation, owner
    // bottom line): compose the authored scene with itself `instance_count` times
    // at distinct XY BEFORE cook, so ONE env holds K SEPARATE articulations that
    // co-step + physically collide (the WP1/WP5-8 multi-dog path validated by
    // tests/scenario/multi_dog_costep.cpp). Each replica i (i>=1) is placed at
    // x = i*instance_spacing (a line; the caller overrides BASE_POSE to the real
    // cluster after create) under a distinct name prefix so the cook keys K
    // disjoint kinematic trees -> articulation_count == instance_count. A value of
    // 0 OR 1 (the zero-initialized default) composes NOTHING -> the cook sees the
    // single authored scene -> BYTE-IDENTICAL to every legacy create. instance
    // co-residence is independent of env_count (use env_count == 1 for K dogs in
    // one world); the field arrays (Q / DRIVE_TARGET / ARTICULATION_LINK_POSE) are
    // then sized env_count * (K * per-dog), the K dogs concatenated per env.
    uint32_t instance_count;        // # co-resident articulation replicas (0/1 = single).
    float instance_spacing;         // X gap between replica spawns before cook (m).
    // ONE-GENERAL-SOLVER landing (L1): route this world through the GENERAL contact
    // path (CookContactFamily::PairDriven -> LBVH broadphase + cvx GJK/EPA
    // narrowphase + mixed-island solve) instead of the legacy analytic FusedFoot
    // foot-vs-ground/terrain kernel. 0 (the zero-initialized default) == FusedFoot
    // == BYTE-IDENTICAL to every legacy create. 1 == PairDriven general path: the
    // cook is taken with the PairDriven option AND a STATIC heightfield collidable
    // is baked from the SAME model.terrain (the procedural surface the FusedFoot
    // kernel samples analytically), so the feet collide against the cooked per-cell
    // TRIANGLE_PRISM grid via the general narrowphase. The heightfield geometry is
    // MODEL-LEVEL (one baked terrain type for the whole batch -- general per-env
    // terrain types are a later feature); the type baked is heightfield_terrain_type
    // (0=Flat,1=PyramidStairs,2=InvertedPyramid,3=RandomBoxes). The grid is
    // (nrow x ncol) cells of edge heightfield_cell, centred at the world origin; a 0
    // in any of the three falls back to a default (41 x 41 @ 0.25 m == a ~10 m span)
    // sized to cover a single spawn footprint. Used by the L1 acceptance rollout
    // (trained go2 policy on the general path) before the FusedFoot path is deleted.
    uint32_t contact_family;          // 0 = FusedFoot (default), 1 = PairDriven (general).
    uint32_t heightfield_terrain_type;// baked heightfield type for the general path.
    uint32_t heightfield_nrow;        // general heightfield grid rows (0 => 41).
    uint32_t heightfield_ncol;        // general heightfield grid cols (0 => 41).
    float    heightfield_cell;        // general heightfield cell edge, m (0 => 0.25).
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
    // for this field aliases the live device buffer the batched drive params'
    // drive_targets point at, so a caller may
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
    // the batched step reads every Step (the batched drive params'
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
    NUKA_FIELD_CONTACT_LINK = 19,
    // WRITE (per-env). Go2-on-stairs Phase 2a: per-env procedural-terrain TYPE
    // code (nuka::terrain::TerrainType: 0 = Flat (default seed), 1 = PyramidStairs,
    // 2 = InvertedPyramid, 3 = RandomBoxes). Aliases the engine's persistent
    // env_terrain_type Data field (zero-copy). The FusedFoot foot-contact kernel
    // samples the shared procedural heightfield h(x,y) per env instead of the
    // scalar ground plane; type 0 (the default seed) reproduces the flat plane at
    // model.ground_height (byte-identical to the legacy path). The horizontal/
    // vertical TerrainParams (step_height etc.) are set ONCE at world create via
    // nuka_world_desc_t.terrain_* (model-level cook config, NOT per env).
    //
    // LAYOUT: 1 uint32 per env, env-major. element_count == env_count,
    // element_stride_bytes == sizeof(uint32_t). dtype == 1 (UINT32). A write is
    // picked up by the NEXT Step (the curriculum-control surface). Persistent =>
    // round-trips reset.
    NUKA_FIELD_ENV_TERRAIN_TYPE = 20,
    // WRITE (per-env). Go2-on-stairs Phase 2a: per-env procedural-terrain
    // DIFFICULTY scale (f32, default seed 1.0 == unscaled). Aliases the engine's
    // persistent env_terrain_difficulty Data field (zero-copy). The foot-contact
    // kernel multiplies the terrain's VERTICAL feature magnitude (step_height +
    // grid_height_max) by this scale before sampling the heightfield, so a
    // curriculum can raise/lower the SAME step/box footprint per env (legged_gym
    // terrain-level convention). diff == 1.0 leaves the terrain unscaled; 0.0
    // collapses it to a flat plane at ground_height. Ignored by type-0 (Flat) envs
    // (SampleTerrainHeight returns ground_height regardless), so the default seed
    // is byte-identical to the legacy flat path.
    //
    // LAYOUT: 1 float per env, env-major. element_count == env_count,
    // element_stride_bytes == sizeof(float). dtype == 0 (FLOAT32). A write is
    // picked up by the NEXT Step. Persistent => round-trips reset.
    NUKA_FIELD_ENV_TERRAIN_DIFFICULTY = 21,
    // WRITE (per-link). T3 unified-actuator feed-forward joint force: a direct
    // applied torque/force ADDED to the actuator output of EVERY control mode
    // (tau += joint_f), AFTER actuator saturation (MuJoCo qfrc_applied -- a user
    // force, not an actuator output, so the actuator force-range does NOT clamp
    // it). This is the channel for gravity compensation, computed-torque
    // feed-forward, and RL torque residuals. Aliases the engine's persistent
    // joint_f Data field (zero-copy). DEFAULT 0 => no-op (the drive is then
    // byte-identical to a world without feed-forward). LAYOUT and slot map are
    // IDENTICAL to NUKA_FIELD_DRIVE_TARGET: float[env_count * base_link_count],
    // env-major, element_stride_bytes == sizeof(float), dtype == 0 (FLOAT32);
    // slot 0 is the inert root, actuated joints are [1..base_link_count). A write
    // is picked up by the NEXT Step. Persistent => round-trips reset.
    NUKA_FIELD_JOINT_FEEDFORWARD = 22
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

// ---------------------------------------------------------------------------
// Go2-on-stairs (random-XY spawn + free-roam): BATCHED arbitrary-(x,y) terrain
// height sampler -- the SINGLE host-side source for the surface height at any
// column. This is the C-ABI projection of nuka::terrain::SampleTerrainHeight +
// ScaleTerrainDifficulty (src/sensor/terrain/terrain_field.hpp), so the RL
// spawn-on-terrain reset that places the Go2 base at a RANDOM (x,y) within the
// terrain tile reads the EXACT same heightfield the foot-contact kernel uses --
// eliminating the python center-only mirror's drift for arbitrary columns.
//
// PURE HOST, no device, no world handle: a free function over plain arrays so an
// RL env can sample N per-env spawn columns in one call. Per element i it builds
// a TerrainParams from the (model-level) scalar geometry params, applies
// ScaleTerrainDifficulty(params, difficulties[i]) (the per-env curriculum
// difficulty), and writes
//   out_heights[i] = SampleTerrainHeight(types[i], xs[i], ys[i], scaled).
//
//   n              -- element count (the number of (type,x,y,diff) tuples).
//   types          -- per-element terrain TYPE code (nuka::terrain::TerrainType:
//                     0=Flat,1=PyramidStairs,2=InvertedPyramid,3=RandomBoxes).
//   xs, ys         -- per-element world column (x,y) in meters.
//   difficulties   -- per-element curriculum difficulty (scales the vertical
//                     feature magnitude; 1.0 == unscaled, 0.0 == flat).
//   ground_height  -- the base plane / floor height (NOT scaled by difficulty).
//   step_height / step_width / platform_width / grid_width / grid_height_max
//                  -- the MODEL-LEVEL TerrainParams (the create-time geometry;
//                     same five floats as nuka_world_desc_t.terrain_*).
//   out_heights    -- output array (n floats), the absolute surface z per column.
//
// Returns NUKA_RESULT_INVALID_ARG if n > 0 and any of types/xs/ys/difficulties/
// out_heights is NULL; n == 0 is a no-op OK. Never touches the device or any
// world; safe to call before/without creating a world.
nuka_result_t nuka_terrain_sample_height_batch(uint32_t n,
                                               const uint32_t* types,
                                               const float* xs,
                                               const float* ys,
                                               const float* difficulties,
                                               float ground_height,
                                               float step_height,
                                               float step_width,
                                               float platform_width,
                                               float grid_width,
                                               float grid_height_max,
                                               float* out_heights);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_H */
