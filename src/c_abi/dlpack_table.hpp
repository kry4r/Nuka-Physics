#pragma once
// ===========================================================================
// c_abi/dlpack_table.hpp — the canonical, single-source-of-truth descriptor
// table for the public RL state-buffer field enum (nuka_state_field_t).
//
// THE RL HARD CONTRACT (M9 T1). The python / RL / diffsim / DLPack stack
// depends on the per-field binary semantics BIT-FOR-BIT: a silent change to
// any field's public integer, element_stride_bytes, or wire dtype corrupts RL
// training invisibly. This table centralizes those descriptors so c_abi/
// buffer.cpp emits them from ONE place instead of scattered inline literals.
//
// It is also the FieldId <-> nuka_state_field_t bridge: each public field row
// records the nk::FieldId it corresponds to (src/nk/model/fields.yaml). M9 T1
// only AUTHORS the mapping + sources the descriptors from here; the actual
// data-source rewire (legacy world -> nk::World Data field pointer) is M9 T5.
// So buffer.cpp still reads today's legacy/batched device pointers — but the
// stride/dtype/FieldId it reports are already resolved through this table.
//
// APPEND-ONLY IN SPIRIT: never reorder/renumber/restride/redtype an existing
// public field. New fields append at the next integer. The 20 rows below are
// byte-pinned by tests/c_abi/test_multi_env_world, tests/c_abi/
// test_drive_target_io, and python/tests/test_nuka_dlpack.py (run BEFORE and
// AFTER any edit to this file or buffer.cpp — they must be value-identical).
// ===========================================================================

#include <cstddef>
#include <cstdint>

#include "nuka/nuka.h"
#include "nk/model/generated/field_ids.hpp"

namespace nuka::c_abi {

// Wire dtype codes carried in nuka_buffer_view_t::dtype (the PUBLIC contract,
// nuka.h: 0 == float32 for every field, 1 == uint32 for CONTACT_LINK only).
// These are the on-the-wire values the DLPack capsule + RL stack read; do NOT
// confuse them with nk::DlpackDtype (the codegen-side scalar code enum).
inline constexpr uint8_t kWireDtypeF32 = 0u;
inline constexpr uint8_t kWireDtypeU32 = 1u;

// Sentinel for a public field that has NO corresponding fields.yaml ordinal
// (the control-input params buffers — TORQUE_INPUT / VELOCITY_TARGET /
// ACTUATOR_NOLOAD_SPEED / TASK_TARGET — are params-carried raw device buffers,
// not Arena Data fields, so they have no nk::FieldId. See the per-row notes.)
inline constexpr nk::FieldId kNoFieldId = nk::FieldId::Count;

// One descriptor row per public nuka_state_field_t value.
//   field             : the public field enum (== its integer; rows are indexed
//                       by that integer and a build-time check enforces it).
//   element_stride_bytes : the BYTE-PINNED per-element stride emitted in the
//                       view (4 for env-major scalar f32; 28 for the 7-float
//                       quat-W-FIRST pose Transform; 24 for the 6-float
//                       omega-FIRST spatial velocity/wrench; 12 for a Vec3 /
//                       3-float contact triple).
//   dtype             : the wire dtype code (kWireDtypeF32 / kWireDtypeU32).
//   field_id          : the nk::FieldId this public field maps to (kNoFieldId
//                       for the params-carried control-input buffers). This is
//                       the bridge T5 consumes to resolve the nk Data pointer.
struct DlpackFieldRow {
    nuka_state_field_t field;
    uint32_t           element_stride_bytes;
    uint8_t            dtype;
    nk::FieldId        field_id;
};

// Strides expressed via sizeof so they track the engine math types exactly (a
// static_assert in articulation_types.cuh already pins sizeof(Transform)==28).
inline constexpr uint32_t kStrideF32   = static_cast<uint32_t>(sizeof(float));        // 4
inline constexpr uint32_t kStrideU32   = static_cast<uint32_t>(sizeof(uint32_t));     // 4
inline constexpr uint32_t kStrideVec3  = static_cast<uint32_t>(3u * sizeof(float));   // 12
inline constexpr uint32_t kStrideSpat6 = static_cast<uint32_t>(6u * sizeof(float));   // 24
inline constexpr uint32_t kStridePose  = static_cast<uint32_t>(7u * sizeof(float));   // 28

inline constexpr DlpackFieldRow kDlpackFieldTable[] = {
    // -- field 0: per-body rigid transform (Transform, quat W-FIRST). Not
    //    served on the go2 articulation path today (NOT_SUPPORTED there); the
    //    descriptor/mapping is ready for the rigid-body data source. ---------
    {NUKA_FIELD_RIGID_BODY_TRANSFORM,   kStridePose,  kWireDtypeF32, nk::FieldId::BodyPose},
    // -- field 1: live per-link WORLD pose (Transform == 7 f32 [px,py,pz,
    //    qw,qx,qy,qz], quat W-FIRST, 28 B). -----------------------------------
    {NUKA_FIELD_ARTICULATION_LINK_POSE, kStridePose,  kWireDtypeF32, nk::FieldId::LinkPose},
    // -- field 2/3: joint position/velocity scalars, env-major f32, stride 4. -
    {NUKA_FIELD_JOINT_POSITION,         kStrideF32,   kWireDtypeF32, nk::FieldId::Q},
    {NUKA_FIELD_JOINT_VELOCITY,         kStrideF32,   kWireDtypeF32, nk::FieldId::Qdot},
    // -- field 4: flat per-env observation export (obs_buffer). Not served on
    //    the current C-ABI path (NOT_SUPPORTED); mapping ready. --------------
    {NUKA_FIELD_OBSERVATIONS,           kStrideF32,   kWireDtypeF32, nk::FieldId::ObsBuffer},
    // -- field 5: per-slot world contact point (Vec3, 12 B). -----------------
    {NUKA_FIELD_CONTACT_POINTS,         kStrideVec3,  kWireDtypeF32, nk::FieldId::ContactPoint},
    // -- field 6: PD drive target (the RL action surface), env-major f32. ----
    {NUKA_FIELD_DRIVE_TARGET,           kStrideF32,   kWireDtypeF32, nk::FieldId::DriveTarget},
    // -- field 7: per-link spatial velocity (6 f32 [wx,wy,wz,vx,vy,vz],
    //    omega-FIRST, 24 B). --------------------------------------------------
    {NUKA_FIELD_LINK_VELOCITY,          kStrideSpat6, kWireDtypeF32, nk::FieldId::LinkVelocity},
    // -- field 8/9/10: PD gain buffers (Kp / Kd / force-limit), env-major f32.
    {NUKA_FIELD_DRIVE_STIFFNESS,        kStrideF32,   kWireDtypeF32, nk::FieldId::DriveStiffness},
    {NUKA_FIELD_DRIVE_DAMPING,          kStrideF32,   kWireDtypeF32, nk::FieldId::DriveDamping},
    {NUKA_FIELD_DRIVE_FORCE_LIMIT,      kStrideF32,   kWireDtypeF32, nk::FieldId::DriveForceLimit},
    // -- field 11: authoritative per-env floating-base root world pose
    //    (Transform, quat W-FIRST, 28 B; ONE per env, NOT per link). ----------
    {NUKA_FIELD_BASE_POSE,              kStridePose,  kWireDtypeF32, nk::FieldId::BasePose},
    // -- field 12: TORQUE_INPUT. The Torque control mode (drive_mode==1) reads the
    //    per-link torque command from the SAME persistent Data field as the PD
    //    target (DriveTarget): OpApplyDrives's ApplyTorqueDriveKernel consumes
    //    data.drive_target as the torque (articulation.cu ApplyTorqueDriveKernel +
    //    OpApplyDrives mode==1). So in the unified actuator (T1) TORQUE_INPUT
    //    ALIASES DriveTarget -- the public field is the SAME device buffer as
    //    DRIVE_TARGET, reinterpreted by the active preset (this is "禁止特化":
    //    ONE control buffer `u`, the preset decides its meaning). Byte-identical
    //    storage; the alias is only the wire name. The other three control-input
    //    fields (13/14/15) stay kNoFieldId until their presets are wired.
    {NUKA_FIELD_TORQUE_INPUT,           kStrideF32,   kWireDtypeF32, nk::FieldId::DriveTarget},
    {NUKA_FIELD_VELOCITY_TARGET,        kStrideF32,   kWireDtypeF32, kNoFieldId},
    {NUKA_FIELD_ACTUATOR_NOLOAD_SPEED,  kStrideF32,   kWireDtypeF32, kNoFieldId},
    // TASK_TARGET is per-ENV float3 {x,y,z} (stride 12), NOT per-link f32.
    {NUKA_FIELD_TASK_TARGET,            kStrideVec3,  kWireDtypeF32, kNoFieldId},
    // -- field 16: net per-link world contact wrench [F(3),tau(3)], 24 B. -----
    {NUKA_FIELD_LINK_CONTACT_WRENCH,    kStrideSpat6, kWireDtypeF32, nk::FieldId::LinkContactWrench},
    // -- field 17: per-slot world unit normal (Vec3, 12 B). ------------------
    {NUKA_FIELD_CONTACT_NORMAL,         kStrideVec3,  kWireDtypeF32, nk::FieldId::ContactNormal},
    // -- field 18: per-slot contact force {Fn,Ft1,Ft2} (3 f32, 12 B). --------
    {NUKA_FIELD_CONTACT_FORCE,          kStrideVec3,  kWireDtypeF32, nk::FieldId::ContactForce},
    // -- field 19: per-slot owning global link index (uint32, 4 B). THE ONLY
    //    non-float32 field — dtype == 1. -------------------------------------
    {NUKA_FIELD_CONTACT_LINK,           kStrideU32,   kWireDtypeU32, nk::FieldId::ContactLink},
    // -- field 20: Go2-on-stairs Phase 2a per-env terrain TYPE (uint32, 4 B). A
    //    writable per-env Data field (the curriculum-control surface). ----------
    {NUKA_FIELD_ENV_TERRAIN_TYPE,       kStrideU32,   kWireDtypeU32, nk::FieldId::EnvTerrainType},
    // -- field 21: Go2-on-stairs Phase 2a per-env terrain DIFFICULTY scale (f32,
    //    4 B, default 1.0). A writable per-env Data field. -----------------------
    {NUKA_FIELD_ENV_TERRAIN_DIFFICULTY, kStrideF32,   kWireDtypeF32, nk::FieldId::EnvTerrainDifficulty},
    // -- field 22: T3 unified-actuator feed-forward joint force (f32, 4 B, default
    //    0). A writable per-link Data field aliasing the persistent joint_f Data
    //    field; added to tau in every preset (default 0 => no-op). ---------------
    {NUKA_FIELD_JOINT_FEEDFORWARD,      kStrideF32,   kWireDtypeF32, nk::FieldId::JointF},
    // -- field 23/24: live particle world position / velocity (Vec3, 12 B) of a
    //    coupled world. per:particle => element_count == env_count*particles_per_env;
    //    an empty view (element_count == 0) on a no-particle world. ------------
    {NUKA_FIELD_PARTICLE_POSITION,      kStrideVec3,  kWireDtypeF32, nk::FieldId::ParticlePos},
    {NUKA_FIELD_PARTICLE_VELOCITY,      kStrideVec3,  kWireDtypeF32, nk::FieldId::ParticleVel},
    // -- field 25/26: per-body world LINEAR / ANGULAR velocity (Vec3, 12 B) of the
    //    cooked rigid SoA (per:body; empty view on a body-free world). ---------------
    {NUKA_FIELD_BODY_LINEAR_VELOCITY,   kStrideVec3,  kWireDtypeF32, nk::FieldId::BodyLinearVelocity},
    {NUKA_FIELD_BODY_ANGULAR_VELOCITY,  kStrideVec3,  kWireDtypeF32, nk::FieldId::BodyAngularVelocity},
};

inline constexpr size_t kDlpackFieldCount =
    sizeof(kDlpackFieldTable) / sizeof(kDlpackFieldTable[0]);

// The table must cover exactly the 27 public fields (0..26), one row each, in
// integer order. BODY_ANGULAR_VELOCITY == 26 is the highest public field; the count
// is one past it. (Append-only: bump this together with a new trailing row.)
// Fields 0..19 are byte-pinned by the RL contract; fields 20/21 are the Go2-on-
// stairs per-env terrain controls; field 22 is the unified-actuator feed-forward
// joint force; fields 23/24 are the coupled-world particle position/velocity; fields
// 25/26 are the per-body linear/angular velocity (all appended, never reordered).
static_assert(kDlpackFieldCount == 27u,
              "dlpack_table must hold exactly the 27 public state fields");
static_assert(static_cast<int>(NUKA_FIELD_CONTACT_LINK) == 19,
              "public field enum range changed — review the RL binary contract");
static_assert(static_cast<int>(NUKA_FIELD_JOINT_FEEDFORWARD) == 22,
              "public field enum range changed — review the RL binary contract");
static_assert(static_cast<int>(NUKA_FIELD_PARTICLE_VELOCITY) == 24,
              "public field enum range changed — review the RL binary contract");
static_assert(static_cast<int>(NUKA_FIELD_BODY_ANGULAR_VELOCITY) == 26,
              "public field enum range changed — review the RL binary contract");

// Resolve a public field's canonical descriptor. Returns nullptr if the field
// integer is out of the table's covered range. The table is laid out so row i
// describes field i; we verify that identity rather than trust it.
inline const DlpackFieldRow* FindDlpackFieldRow(nuka_state_field_t field) {
    const int idx = static_cast<int>(field);
    if (idx < 0 || idx >= static_cast<int>(kDlpackFieldCount)) {
        return nullptr;
    }
    const DlpackFieldRow& row = kDlpackFieldTable[idx];
    if (row.field != field) {
        // Table rows are not in 1:1 integer order — refuse rather than emit a
        // wrong descriptor (would silently corrupt the RL contract).
        return nullptr;
    }
    return &row;
}

} // namespace nuka::c_abi
