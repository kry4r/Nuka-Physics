// GENERATED — do not edit; regen via tools/codegen/regen.py
// Source of truth: src/nk/model/fields.yaml

#pragma once

// Per-field DLPack descriptor rows for the future c_abi buffer bridge
// (plan §3.3: c_abi/buffer.cpp queries this table instead of hand-written
// per-field branches). dtype_code is the scalar element code; ndim_hint is
// a coarse rank hint (1 = flat scalar array, 2 = packed/vector array); diff
// marks the tape-visible / autograd-exposed fields.

#include <cstdint>

#include "nk/model/generated/field_ids.hpp"

namespace nuka::nk {

enum class DlpackDtype : uint8_t { kF32, kU32, kU64 };

struct DlpackRow {
    FieldId     field;
    DlpackDtype dtype_code;
    uint8_t     ndim_hint;
    bool        diff;
    bool        readout;
};

inline constexpr DlpackRow kDlpackTable[kFieldCount] = {
    {FieldId::Q, DlpackDtype::kF32, 1, true, false},  // q
    {FieldId::Qdot, DlpackDtype::kF32, 1, true, false},  // qdot
    {FieldId::LinkPose, DlpackDtype::kF32, 2, false, false},  // link_pose
    {FieldId::BasePose, DlpackDtype::kF32, 2, true, false},  // base_pose
    {FieldId::BodyPose, DlpackDtype::kF32, 2, true, false},  // body_pose
    {FieldId::BodyInvMass, DlpackDtype::kF32, 1, false, false},  // body_inv_mass
    {FieldId::ContactCount, DlpackDtype::kU32, 1, false, false},  // contact_count
    {FieldId::Rows, DlpackDtype::kF32, 2, false, false},  // rows
    {FieldId::Lambda, DlpackDtype::kF32, 1, false, false},  // lambda
    {FieldId::MInv, DlpackDtype::kF32, 1, false, false},  // m_inv
    {FieldId::ParticlePos, DlpackDtype::kF32, 2, true, false},  // particle_pos
    {FieldId::MatBuckets, DlpackDtype::kF32, 1, false, false},  // mat_buckets
    {FieldId::MatIndex, DlpackDtype::kU32, 1, false, false},  // mat_index
    {FieldId::LinkContactWrench, DlpackDtype::kF32, 2, false, true},  // link_contact_wrench
    {FieldId::Qddot, DlpackDtype::kF32, 1, false, false},  // qddot
    {FieldId::Tau, DlpackDtype::kF32, 1, false, false},  // tau
    {FieldId::JointForce, DlpackDtype::kF32, 1, false, false},  // joint_force
    {FieldId::JointDiagonal, DlpackDtype::kF32, 1, false, false},  // joint_diagonal
    {FieldId::LinkVelocity, DlpackDtype::kF32, 2, true, false},  // link_velocity
    {FieldId::LinkAcceleration, DlpackDtype::kF32, 2, false, false},  // link_acceleration
    {FieldId::LinkVelocityBias, DlpackDtype::kF32, 2, false, false},  // link_velocity_bias
    {FieldId::LinkXup, DlpackDtype::kF32, 2, false, false},  // link_xup
    {FieldId::LinkArticulatedI, DlpackDtype::kF32, 2, false, false},  // link_articulated_I
    {FieldId::LinkBiasForce, DlpackDtype::kF32, 2, false, false},  // link_bias_force
    {FieldId::LinkUSpatial, DlpackDtype::kF32, 2, false, false},  // link_u_spatial
    {FieldId::LinkInertia, DlpackDtype::kF32, 2, false, false},  // link_inertia
    {FieldId::LinkLocalPose, DlpackDtype::kF32, 2, false, false},  // link_local_pose
    {FieldId::LinkInertialFrame, DlpackDtype::kF32, 2, false, false},  // link_inertial_frame
    {FieldId::JointMotionSubspace, DlpackDtype::kF32, 2, false, false},  // joint_motion_subspace
    {FieldId::JointAxis, DlpackDtype::kF32, 2, false, false},  // joint_axis
    {FieldId::ParentOffset, DlpackDtype::kF32, 2, false, false},  // parent_offset
    {FieldId::JointType, DlpackDtype::kU32, 1, false, false},  // joint_type
    {FieldId::ParentLink, DlpackDtype::kU32, 1, false, false},  // parent_link
    {FieldId::LinkBody, DlpackDtype::kU32, 1, false, false},  // link_body
    {FieldId::LinkToArticulation, DlpackDtype::kU32, 1, false, false},  // link_to_articulation
    {FieldId::JointDamping, DlpackDtype::kF32, 1, false, false},  // joint_damping
    {FieldId::JointArmature, DlpackDtype::kF32, 1, false, false},  // joint_armature
    {FieldId::ArticulationLinkCount, DlpackDtype::kU32, 1, false, false},  // articulation_link_count
    {FieldId::ArticulationLinkOffset, DlpackDtype::kU32, 1, false, false},  // articulation_link_offset
    {FieldId::BodyLinearVelocity, DlpackDtype::kF32, 2, true, false},  // body_linear_velocity
    {FieldId::BodyAngularVelocity, DlpackDtype::kF32, 2, true, false},  // body_angular_velocity
    {FieldId::BodyForce, DlpackDtype::kF32, 2, false, false},  // body_force
    {FieldId::BodyTorque, DlpackDtype::kF32, 2, false, false},  // body_torque
    {FieldId::BodyInvInertia, DlpackDtype::kF32, 2, false, false},  // body_inv_inertia
    {FieldId::BodyAabbLo, DlpackDtype::kF32, 2, false, false},  // body_aabb_lo
    {FieldId::BodyAabbHi, DlpackDtype::kF32, 2, false, false},  // body_aabb_hi
    {FieldId::PairCount, DlpackDtype::kU32, 1, false, false},  // pair_count
    {FieldId::CandidatePairs, DlpackDtype::kU32, 2, false, false},  // candidate_pairs
    {FieldId::ContactPoint, DlpackDtype::kF32, 2, false, false},  // contact_point
    {FieldId::ContactNormal, DlpackDtype::kF32, 2, false, false},  // contact_normal
    {FieldId::ContactDepth, DlpackDtype::kF32, 1, false, false},  // contact_depth
    {FieldId::ContactTangent, DlpackDtype::kF32, 2, false, false},  // contact_tangent
    {FieldId::ContactMaterial, DlpackDtype::kU32, 1, false, false},  // contact_material
    {FieldId::RowCount, DlpackDtype::kU32, 1, false, false},  // row_count
    {FieldId::RowSides, DlpackDtype::kU32, 2, false, false},  // row_sides
    {FieldId::ChainJacobian, DlpackDtype::kF32, 2, false, false},  // chain_jacobian
    {FieldId::RowMeff, DlpackDtype::kF32, 1, false, false},  // row_meff
    {FieldId::RowMaterial, DlpackDtype::kU32, 1, false, false},  // row_material
    {FieldId::IslandRowOffsets, DlpackDtype::kU32, 1, false, false},  // island_row_offsets
    {FieldId::IslandColorSegments, DlpackDtype::kU32, 1, false, false},  // island_color_segments
    {FieldId::RowOrder, DlpackDtype::kU32, 1, false, false},  // row_order
    {FieldId::ParticlePrevPos, DlpackDtype::kF32, 2, false, false},  // particle_prev_pos
    {FieldId::ParticleVel, DlpackDtype::kF32, 2, true, false},  // particle_vel
    {FieldId::ParticleInvMass, DlpackDtype::kF32, 1, false, false},  // particle_inv_mass
    {FieldId::DistParticleA, DlpackDtype::kU32, 1, false, false},  // dist_particle_a
    {FieldId::DistParticleB, DlpackDtype::kU32, 1, false, false},  // dist_particle_b
    {FieldId::DistRestLength, DlpackDtype::kF32, 1, false, false},  // dist_rest_length
    {FieldId::DistCompliance, DlpackDtype::kF32, 1, false, false},  // dist_compliance
    {FieldId::DistLambda, DlpackDtype::kF32, 1, false, false},  // dist_lambda
    {FieldId::BendParticles, DlpackDtype::kU32, 2, false, false},  // bend_particles
    {FieldId::BendGradients, DlpackDtype::kF32, 2, false, false},  // bend_gradients
    {FieldId::BendCompliance, DlpackDtype::kF32, 1, false, false},  // bend_compliance
    {FieldId::BendLambda, DlpackDtype::kF32, 1, false, false},  // bend_lambda
    {FieldId::VolParticles, DlpackDtype::kU32, 2, false, false},  // vol_particles
    {FieldId::VolRestTimes6, DlpackDtype::kF32, 1, false, false},  // vol_rest_times6
    {FieldId::VolCompliance, DlpackDtype::kF32, 1, false, false},  // vol_compliance
    {FieldId::VolLambda, DlpackDtype::kF32, 1, false, false},  // vol_lambda
    {FieldId::PbfPredictedPos, DlpackDtype::kF32, 2, false, false},  // pbf_predicted_pos
    {FieldId::PbfPositionDelta, DlpackDtype::kF32, 2, false, false},  // pbf_position_delta
    {FieldId::PbfDensity, DlpackDtype::kF32, 1, false, false},  // pbf_density
    {FieldId::PbfLambda, DlpackDtype::kF32, 1, false, false},  // pbf_lambda
    {FieldId::RngState, DlpackDtype::kU64, 1, false, false},  // rng_state
    {FieldId::EnvStatus, DlpackDtype::kU32, 1, false, true},  // env_status
    {FieldId::ObsBuffer, DlpackDtype::kF32, 2, false, true},  // obs_buffer
};

} // namespace nuka::nk
