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

enum class DlpackDtype : uint8_t { kF32, kU32, kU64, kU8 };

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
    {FieldId::JointMotionSubspace, DlpackDtype::kF32, 2, false, false},  // joint_motion_subspace
    {FieldId::DriveTarget, DlpackDtype::kF32, 1, false, false},  // drive_target
    {FieldId::DriveStiffness, DlpackDtype::kF32, 1, false, false},  // drive_stiffness
    {FieldId::DriveDamping, DlpackDtype::kF32, 1, false, false},  // drive_damping
    {FieldId::DriveForceLimit, DlpackDtype::kF32, 1, false, false},  // drive_force_limit
    {FieldId::SnapshotQ, DlpackDtype::kF32, 1, false, false},  // snapshot_q
    {FieldId::SnapshotQdot, DlpackDtype::kF32, 1, false, false},  // snapshot_qdot
    {FieldId::SnapshotLinkVelocity, DlpackDtype::kF32, 2, false, false},  // snapshot_link_velocity
    {FieldId::SnapshotBasePose, DlpackDtype::kF32, 2, false, false},  // snapshot_base_pose
    {FieldId::ResetEnvIds, DlpackDtype::kU32, 1, false, false},  // reset_env_ids
    {FieldId::LinkInertia, DlpackDtype::kF32, 2, false, false},  // link_inertia
    {FieldId::LinkLocalPose, DlpackDtype::kF32, 2, false, false},  // link_local_pose
    {FieldId::LinkInertialFrame, DlpackDtype::kF32, 2, false, false},  // link_inertial_frame
    {FieldId::JointAxis, DlpackDtype::kF32, 2, false, false},  // joint_axis
    {FieldId::ParentOffset, DlpackDtype::kF32, 2, false, false},  // parent_offset
    {FieldId::JointType, DlpackDtype::kU8, 1, false, false},  // joint_type
    {FieldId::ParentLink, DlpackDtype::kU32, 1, false, false},  // parent_link
    {FieldId::LinkBody, DlpackDtype::kU32, 1, false, false},  // link_body
    {FieldId::LinkToArticulation, DlpackDtype::kU32, 1, false, false},  // link_to_articulation
    {FieldId::JointDamping, DlpackDtype::kF32, 1, false, false},  // joint_damping
    {FieldId::JointArmature, DlpackDtype::kF32, 1, false, false},  // joint_armature
    {FieldId::ArticulationLinkCount, DlpackDtype::kU32, 1, false, false},  // articulation_link_count
    {FieldId::ArticulationLinkOffset, DlpackDtype::kU32, 1, false, false},  // articulation_link_offset
    {FieldId::FootShape, DlpackDtype::kF32, 1, false, false},  // foot_shape
    {FieldId::BodyLinearVelocity, DlpackDtype::kF32, 2, true, false},  // body_linear_velocity
    {FieldId::BodyAngularVelocity, DlpackDtype::kF32, 2, true, false},  // body_angular_velocity
    {FieldId::BodyForce, DlpackDtype::kF32, 2, false, false},  // body_force
    {FieldId::BodyTorque, DlpackDtype::kF32, 2, false, false},  // body_torque
    {FieldId::BodyInvInertia, DlpackDtype::kF32, 2, false, false},  // body_inv_inertia
    {FieldId::BodyAabbLo, DlpackDtype::kF32, 2, false, false},  // body_aabb_lo
    {FieldId::BodyAabbHi, DlpackDtype::kF32, 2, false, false},  // body_aabb_hi
    {FieldId::PairCount, DlpackDtype::kU32, 1, false, false},  // pair_count
    {FieldId::CandidatePairs, DlpackDtype::kU32, 2, false, false},  // candidate_pairs
    {FieldId::LbvhNodes, DlpackDtype::kF32, 2, false, false},  // lbvh_nodes
    {FieldId::LbvhMorton, DlpackDtype::kU32, 1, false, false},  // lbvh_morton
    {FieldId::LbvhIndex, DlpackDtype::kU32, 1, false, false},  // lbvh_index
    {FieldId::LbvhVisit, DlpackDtype::kU32, 1, false, false},  // lbvh_visit
    {FieldId::ContactLink, DlpackDtype::kU32, 1, false, false},  // contact_link
    {FieldId::ContactPoint, DlpackDtype::kF32, 2, false, false},  // contact_point
    {FieldId::ContactNormal, DlpackDtype::kF32, 2, false, false},  // contact_normal
    {FieldId::ContactDepth, DlpackDtype::kF32, 1, false, false},  // contact_depth
    {FieldId::ContactTangent1, DlpackDtype::kF32, 2, false, false},  // contact_tangent1
    {FieldId::ContactTangent2, DlpackDtype::kF32, 2, false, false},  // contact_tangent2
    {FieldId::ContactMaterial, DlpackDtype::kU32, 1, false, false},  // contact_material
    {FieldId::JacNormal, DlpackDtype::kF32, 1, false, false},  // jac_normal
    {FieldId::JacTangent1, DlpackDtype::kF32, 1, false, false},  // jac_tangent1
    {FieldId::JacTangent2, DlpackDtype::kF32, 1, false, false},  // jac_tangent2
    {FieldId::ContactMeffNormal, DlpackDtype::kF32, 1, false, false},  // contact_meff_normal
    {FieldId::ContactMeffTangent1, DlpackDtype::kF32, 1, false, false},  // contact_meff_tangent1
    {FieldId::ContactMeffTangent2, DlpackDtype::kF32, 1, false, false},  // contact_meff_tangent2
    {FieldId::ContactForce, DlpackDtype::kF32, 2, false, true},  // contact_force
    {FieldId::UcontactCount, DlpackDtype::kU32, 1, false, false},  // ucontact_count
    {FieldId::UcontactPoint, DlpackDtype::kF32, 2, false, false},  // ucontact_point
    {FieldId::UcontactNormal, DlpackDtype::kF32, 2, false, false},  // ucontact_normal
    {FieldId::UcontactDepth, DlpackDtype::kF32, 2, false, false},  // ucontact_depth
    {FieldId::RowCount, DlpackDtype::kU32, 1, false, false},  // row_count
    {FieldId::Urows, DlpackDtype::kF32, 2, false, false},  // urows
    {FieldId::ChainJacobian, DlpackDtype::kF32, 1, false, false},  // chain_jacobian
    {FieldId::RowMinvJt, DlpackDtype::kF32, 1, false, false},  // row_minv_jt
    {FieldId::RowMeff, DlpackDtype::kF32, 1, false, false},  // row_meff
    {FieldId::RowCjLink, DlpackDtype::kU32, 1, false, false},  // row_cj_link
    {FieldId::RowCjPoint, DlpackDtype::kF32, 2, false, false},  // row_cj_point
    {FieldId::RowCjDir, DlpackDtype::kF32, 2, false, false},  // row_cj_dir
    {FieldId::QdotFlat, DlpackDtype::kF32, 1, false, false},  // qdot_flat
    {FieldId::M, DlpackDtype::kF32, 1, false, false},  // m
    {FieldId::LinkCompositeInertia, DlpackDtype::kF32, 2, false, false},  // link_composite_inertia
    {FieldId::UnionSlots, DlpackDtype::kF32, 1, false, false},  // union_slots
    {FieldId::HullVerts, DlpackDtype::kF32, 1, false, false},  // hull_verts
    {FieldId::DofToLink, DlpackDtype::kU32, 1, false, false},  // dof_to_link
    {FieldId::DofToComponent, DlpackDtype::kU32, 1, false, false},  // dof_to_component
    {FieldId::TableEnabled, DlpackDtype::kU32, 1, false, false},  // table_enabled
    {FieldId::ShapeTable, DlpackDtype::kF32, 1, false, false},  // shape_table
    {FieldId::ExcludedPairs, DlpackDtype::kU64, 1, false, false},  // excluded_pairs
    {FieldId::SampPoints, DlpackDtype::kF32, 1, false, false},  // samp_points
    {FieldId::SampRanges, DlpackDtype::kU32, 1, false, false},  // samp_ranges
    {FieldId::SdfHeaders, DlpackDtype::kF32, 1, false, false},  // sdf_headers
    {FieldId::SdfCellCount, DlpackDtype::kU32, 1, false, false},  // sdf_cell_count
    {FieldId::SdfCellKeys, DlpackDtype::kU64, 1, false, false},  // sdf_cell_keys
    {FieldId::SdfCellValues, DlpackDtype::kF32, 1, false, false},  // sdf_cell_values
    {FieldId::SdfCellGradients, DlpackDtype::kF32, 2, false, false},  // sdf_cell_gradients
    {FieldId::IslandRowOffsets, DlpackDtype::kU32, 2, false, false},  // island_row_offsets
    {FieldId::IslandColorSegments, DlpackDtype::kU32, 2, false, false},  // island_color_segments
    {FieldId::RowOrder, DlpackDtype::kU32, 1, false, false},  // row_order
    {FieldId::ParticlePrevPos, DlpackDtype::kF32, 2, false, false},  // particle_prev_pos
    {FieldId::ParticleVel, DlpackDtype::kF32, 2, true, false},  // particle_vel
    {FieldId::ParticleInvMass, DlpackDtype::kF32, 1, false, false},  // particle_inv_mass
    {FieldId::ParticleVPre, DlpackDtype::kF32, 2, false, false},  // particle_v_pre
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
    {FieldId::GridCellKey, DlpackDtype::kU32, 1, false, false},  // grid_cell_key
    {FieldId::GridParticleIdx, DlpackDtype::kU32, 1, false, false},  // grid_particle_idx
    {FieldId::GridCellStart, DlpackDtype::kU32, 1, false, false},  // grid_cell_start
    {FieldId::GridCellEnd, DlpackDtype::kU32, 1, false, false},  // grid_cell_end
    {FieldId::GridNeighborOffset, DlpackDtype::kU32, 1, false, false},  // grid_neighbor_offset
    {FieldId::GridNeighborCount, DlpackDtype::kU32, 1, false, false},  // grid_neighbor_count
    {FieldId::GridNeighborIdx, DlpackDtype::kU32, 2, false, false},  // grid_neighbor_idx
    {FieldId::RngState, DlpackDtype::kU64, 1, false, false},  // rng_state
    {FieldId::EnvStatus, DlpackDtype::kU32, 1, false, true},  // env_status
    {FieldId::ObsBuffer, DlpackDtype::kF32, 2, false, true},  // obs_buffer
};

} // namespace nuka::nk
