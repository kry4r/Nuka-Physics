// GENERATED — do not edit; regen via tools/codegen/regen.py
// Source of truth: src/nk/model/fields.yaml

#pragma once

#include <cstdint>

namespace nuka::nk {

// One stable enumerator per field (yaml order). FieldId is the key into
// the arena segment table and the View member binding.
enum class FieldId : uint16_t {
    Q,  // q (per:link arena:persistent owner:data flags:[diff])
    Qdot,  // qdot (per:link arena:persistent owner:data flags:[diff])
    LinkPose,  // link_pose (per:link arena:persistent owner:data)
    BasePose,  // base_pose (per:articulation arena:persistent owner:data flags:[diff])
    BodyPose,  // body_pose (per:body arena:persistent owner:data flags:[diff])
    BodyInvMass,  // body_inv_mass (per:body arena:persistent owner:data flags:[param])
    ContactCount,  // contact_count (per:env arena:scratch owner:data)
    Rows,  // rows (per:contact_slot arena:scratch owner:data elem:16)
    Lambda,  // lambda (per:row_slot arena:persistent owner:data)
    MInv,  // m_inv (per:articulation_dof2 arena:scratch owner:data)
    ParticlePos,  // particle_pos (per:particle arena:persistent owner:data flags:[diff])
    MatBuckets,  // mat_buckets (per:scalar arena:persistent owner:data count:num_buckets*16 flags:[param])
    MatIndex,  // mat_index (per:body arena:persistent owner:data flags:[param])
    LinkContactWrench,  // link_contact_wrench (per:link arena:scratch owner:data flags:[readout])
    Qddot,  // qddot (per:link arena:persistent owner:data)
    Tau,  // tau (per:link arena:persistent owner:data)
    JointForce,  // joint_force (per:link arena:scratch owner:data)
    JointDiagonal,  // joint_diagonal (per:link arena:scratch owner:data)
    LinkVelocity,  // link_velocity (per:link arena:persistent owner:data flags:[diff])
    LinkAcceleration,  // link_acceleration (per:link arena:scratch owner:data)
    LinkVelocityBias,  // link_velocity_bias (per:link arena:scratch owner:data)
    LinkXup,  // link_xup (per:link arena:scratch owner:data)
    LinkArticulatedI,  // link_articulated_I (per:link arena:scratch owner:data)
    LinkBiasForce,  // link_bias_force (per:link arena:scratch owner:data)
    LinkUSpatial,  // link_u_spatial (per:link arena:scratch owner:data)
    JointMotionSubspace,  // joint_motion_subspace (per:link arena:scratch owner:data)
    DriveTarget,  // drive_target (per:link arena:persistent owner:data flags:[param])
    DriveStiffness,  // drive_stiffness (per:link arena:persistent owner:data flags:[param])
    DriveDamping,  // drive_damping (per:link arena:persistent owner:data flags:[param])
    DriveForceLimit,  // drive_force_limit (per:link arena:persistent owner:data flags:[param])
    SnapshotQ,  // snapshot_q (per:link arena:persistent owner:data)
    SnapshotQdot,  // snapshot_qdot (per:link arena:persistent owner:data)
    SnapshotLinkVelocity,  // snapshot_link_velocity (per:link arena:persistent owner:data)
    SnapshotBasePose,  // snapshot_base_pose (per:articulation arena:persistent owner:data)
    SnapshotBodyPose,  // snapshot_body_pose (per:body arena:persistent owner:data)
    SnapshotBodyLinearVelocity,  // snapshot_body_linear_velocity (per:body arena:persistent owner:data)
    SnapshotBodyAngularVelocity,  // snapshot_body_angular_velocity (per:body arena:persistent owner:data)
    ResetEnvIds,  // reset_env_ids (per:env arena:scratch owner:data)
    LinkInertia,  // link_inertia (per:link arena:persistent owner:model)
    LinkLocalPose,  // link_local_pose (per:link arena:persistent owner:model)
    LinkInertialFrame,  // link_inertial_frame (per:link arena:persistent owner:model)
    JointAxis,  // joint_axis (per:link arena:persistent owner:model)
    ParentOffset,  // parent_offset (per:link arena:persistent owner:model)
    JointType,  // joint_type (per:link arena:persistent owner:model)
    ParentLink,  // parent_link (per:link arena:persistent owner:model)
    LinkBody,  // link_body (per:link arena:persistent owner:model)
    LinkToArticulation,  // link_to_articulation (per:link arena:persistent owner:model)
    JointDamping,  // joint_damping (per:link arena:persistent owner:model flags:[param])
    JointArmature,  // joint_armature (per:link arena:persistent owner:model flags:[param])
    ArticulationLinkCount,  // articulation_link_count (per:articulation arena:persistent owner:model)
    ArticulationLinkOffset,  // articulation_link_offset (per:articulation arena:persistent owner:model)
    FootShape,  // foot_shape (per:scalar arena:persistent owner:model count:max_contacts_per_env*5)
    BodyLinearVelocity,  // body_linear_velocity (per:body arena:persistent owner:data flags:[diff])
    BodyAngularVelocity,  // body_angular_velocity (per:body arena:persistent owner:data flags:[diff])
    BodyForce,  // body_force (per:body arena:scratch owner:data)
    BodyTorque,  // body_torque (per:body arena:scratch owner:data)
    BodyInvInertia,  // body_inv_inertia (per:body arena:persistent owner:data flags:[param])
    BodyAabbLo,  // body_aabb_lo (per:body arena:scratch owner:data)
    BodyAabbHi,  // body_aabb_hi (per:body arena:scratch owner:data)
    PairCount,  // pair_count (per:env arena:scratch owner:data)
    CandidatePairs,  // candidate_pairs (per:contact_slot arena:scratch owner:data elem:2)
    LbvhNodes,  // lbvh_nodes (per:scalar arena:scratch owner:data count:lbvh_node_count)
    LbvhMorton,  // lbvh_morton (per:body arena:scratch owner:data)
    LbvhIndex,  // lbvh_index (per:body arena:scratch owner:data)
    LbvhVisit,  // lbvh_visit (per:body arena:scratch owner:data)
    LbvhSortkey,  // lbvh_sortkey (per:body arena:scratch owner:data)
    ContactLink,  // contact_link (per:contact_slot arena:scratch owner:data)
    ContactPoint,  // contact_point (per:contact_slot arena:scratch owner:data)
    ContactNormal,  // contact_normal (per:contact_slot arena:scratch owner:data)
    ContactDepth,  // contact_depth (per:contact_slot arena:scratch owner:data)
    ContactTangent1,  // contact_tangent1 (per:contact_slot arena:scratch owner:data)
    ContactTangent2,  // contact_tangent2 (per:contact_slot arena:scratch owner:data)
    ContactMaterial,  // contact_material (per:contact_slot arena:scratch owner:data)
    JacNormal,  // jac_normal (per:slot_dof arena:scratch owner:data)
    JacTangent1,  // jac_tangent1 (per:slot_dof arena:scratch owner:data)
    JacTangent2,  // jac_tangent2 (per:slot_dof arena:scratch owner:data)
    ContactMeffNormal,  // contact_meff_normal (per:contact_slot arena:scratch owner:data)
    ContactMeffTangent1,  // contact_meff_tangent1 (per:contact_slot arena:scratch owner:data)
    ContactMeffTangent2,  // contact_meff_tangent2 (per:contact_slot arena:scratch owner:data)
    ContactForce,  // contact_force (per:contact_slot arena:scratch owner:data elem:3 flags:[readout])
    UcontactCount,  // ucontact_count (per:contact_slot arena:scratch owner:data)
    UcontactPoint,  // ucontact_point (per:contact_slot arena:scratch owner:data elem:4)
    UcontactNormal,  // ucontact_normal (per:contact_slot arena:scratch owner:data elem:4)
    UcontactDepth,  // ucontact_depth (per:contact_slot arena:scratch owner:data elem:4)
    RowCount,  // row_count (per:env arena:scratch owner:data)
    Urows,  // urows (per:row_slot arena:scratch owner:data elem:32)
    ChainJacobian,  // chain_jacobian (per:row_dof arena:scratch owner:data)
    RowMinvJt,  // row_minv_jt (per:row_dof arena:scratch owner:data)
    RowMeff,  // row_meff (per:row_slot arena:scratch owner:data)
    RowDamping,  // row_damping (per:row_slot arena:scratch owner:data)
    RowCjLink,  // row_cj_link (per:row_slot arena:scratch owner:data)
    RowCjPoint,  // row_cj_point (per:row_slot arena:scratch owner:data)
    RowCjDir,  // row_cj_dir (per:row_slot arena:scratch owner:data)
    QdotFlat,  // qdot_flat (per:articulation_dof arena:scratch owner:data)
    M,  // m (per:articulation_dof2 arena:scratch owner:data)
    LinkCompositeInertia,  // link_composite_inertia (per:link arena:scratch owner:data)
    HullVerts,  // hull_verts (per:scalar arena:persistent owner:model count:max_hull_verts*3)
    DofToLink,  // dof_to_link (per:dof arena:persistent owner:model)
    DofToComponent,  // dof_to_component (per:dof arena:persistent owner:model)
    ShapeTable,  // shape_table (per:scalar arena:persistent owner:model count:max_bodies_total*13)
    ExcludedPairs,  // excluded_pairs (per:scalar arena:persistent owner:model count:max_excluded_pairs)
    SampPoints,  // samp_points (per:scalar arena:persistent owner:model count:max_samp_points*3)
    SampRanges,  // samp_ranges (per:scalar arena:persistent owner:model count:max_bodies_total*2)
    SdfHeaders,  // sdf_headers (per:scalar arena:persistent owner:model count:max_sdf_grids*8)
    SdfCellCount,  // sdf_cell_count (per:scalar arena:persistent owner:model count:max_sdf_grids)
    SdfCellKeys,  // sdf_cell_keys (per:scalar arena:persistent owner:model count:max_sdf_cells)
    SdfCellValues,  // sdf_cell_values (per:scalar arena:persistent owner:model count:max_sdf_cells)
    SdfCellGradients,  // sdf_cell_gradients (per:scalar arena:persistent owner:model count:max_sdf_cells)
    IslandRowOffsets,  // island_row_offsets (per:row_slot arena:persistent owner:model elem:4)
    IslandColorSegments,  // island_color_segments (per:row_slot arena:persistent owner:model elem:2)
    RowOrder,  // row_order (per:row_slot arena:persistent owner:model)
    ParticlePrevPos,  // particle_prev_pos (per:particle arena:persistent owner:data)
    ParticleVel,  // particle_vel (per:particle arena:persistent owner:data flags:[diff])
    ParticleInvMass,  // particle_inv_mass (per:particle arena:persistent owner:data flags:[param])
    ParticleVPre,  // particle_v_pre (per:particle arena:scratch owner:data)
    DistParticleA,  // dist_particle_a (per:dist_con arena:persistent owner:model)
    DistParticleB,  // dist_particle_b (per:dist_con arena:persistent owner:model)
    DistRestLength,  // dist_rest_length (per:dist_con arena:persistent owner:model)
    DistCompliance,  // dist_compliance (per:dist_con arena:persistent owner:model flags:[param])
    DistLambda,  // dist_lambda (per:dist_con arena:persistent owner:data)
    BendParticles,  // bend_particles (per:bend_con arena:persistent owner:model elem:4)
    BendGradients,  // bend_gradients (per:bend_con arena:persistent owner:model elem:4)
    BendCompliance,  // bend_compliance (per:bend_con arena:persistent owner:model flags:[param])
    BendLambda,  // bend_lambda (per:bend_con arena:persistent owner:data)
    VolParticles,  // vol_particles (per:vol_con arena:persistent owner:model elem:4)
    VolRestTimes6,  // vol_rest_times6 (per:vol_con arena:persistent owner:model)
    VolCompliance,  // vol_compliance (per:vol_con arena:persistent owner:model flags:[param])
    VolLambda,  // vol_lambda (per:vol_con arena:persistent owner:data)
    SmClusterOffset,  // sm_cluster_offset (per:shape_match_slot arena:persistent owner:model)
    SmClusterSize,  // sm_cluster_size (per:shape_match_slot arena:persistent owner:model)
    SmStiffness,  // sm_stiffness (per:shape_match_slot arena:persistent owner:model flags:[param])
    SmRestCentroid,  // sm_rest_centroid (per:shape_match_slot arena:persistent owner:model)
    SmParticles,  // sm_particles (per:shape_match_member arena:persistent owner:model)
    SmRestQ,  // sm_rest_q (per:shape_match_member arena:persistent owner:model)
    SmMass,  // sm_mass (per:shape_match_member arena:persistent owner:model)
    AeroTriVerts,  // aero_tri_verts (per:aero_tri arena:persistent owner:model elem:3)
    AeroTriArea,  // aero_tri_area (per:aero_tri arena:persistent owner:model)
    PbfPredictedPos,  // pbf_predicted_pos (per:particle arena:scratch owner:data)
    PbfPositionDelta,  // pbf_position_delta (per:particle arena:scratch owner:data)
    PbfDensity,  // pbf_density (per:particle arena:scratch owner:data)
    PbfLambda,  // pbf_lambda (per:particle arena:scratch owner:data)
    GridCellKey,  // grid_cell_key (per:particle arena:scratch owner:data)
    GridParticleIdx,  // grid_particle_idx (per:particle arena:scratch owner:data)
    GridCellStart,  // grid_cell_start (per:scalar arena:scratch owner:data count:max_grid_cells*env_count)
    GridCellEnd,  // grid_cell_end (per:scalar arena:scratch owner:data count:max_grid_cells*env_count)
    GridNeighborOffset,  // grid_neighbor_offset (per:particle arena:scratch owner:data)
    GridNeighborCount,  // grid_neighbor_count (per:particle arena:scratch owner:data flags:[readout])
    GridNeighborIdx,  // grid_neighbor_idx (per:scalar arena:scratch owner:data count:neighbor_pool_capacity*env_count)
    RngState,  // rng_state (per:env arena:persistent owner:data flags:[param])
    EnvStatus,  // env_status (per:env arena:scratch owner:data flags:[readout])
    ObsBuffer,  // obs_buffer (per:scalar arena:scratch owner:data count:obs_width*env_count flags:[readout])
    EnvTerrainType,  // env_terrain_type (per:env arena:persistent owner:data)
    EnvTerrainDifficulty,  // env_terrain_difficulty (per:env arena:persistent owner:data)
    JointF,  // joint_f (per:link arena:persistent owner:data flags:[param])
    LinkGeomKind,  // link_geom_kind (per:link arena:persistent owner:model)
    LinkGeomParams,  // link_geom_params (per:link arena:persistent owner:model elem:4)
    LinkGeomLocal,  // link_geom_local (per:link arena:persistent owner:model)
    BodyToLink,  // body_to_link (per:body arena:persistent owner:model)
    BodyToArticulation,  // body_to_articulation (per:body arena:persistent owner:model)
    UcontactA,  // ucontact_a (per:contact_slot arena:scratch owner:data elem:4)
    UcontactB,  // ucontact_b (per:contact_slot arena:scratch owner:data elem:4)
    UcontactGen,  // ucontact_gen (per:contact_slot arena:scratch owner:data elem:4)
    Heights,  // heights (per:scalar arena:persistent owner:model count:max_heightfield_cells)
    UcontactTangent1,  // ucontact_tangent1 (per:contact_slot arena:scratch owner:data elem:4)
    UcontactTangent2,  // ucontact_tangent2 (per:contact_slot arena:scratch owner:data elem:4)
    ChainJacobianB,  // chain_jacobian_b (per:row_dof arena:scratch owner:data)
    RowMinvJtB,  // row_minv_jt_b (per:row_dof arena:scratch owner:data)
    RowCjLinkB,  // row_cj_link_b (per:row_slot arena:scratch owner:data)
    RowCjPointB,  // row_cj_point_b (per:row_slot arena:scratch owner:data)
    RowCjDirB,  // row_cj_dir_b (per:row_slot arena:scratch owner:data)
    PdSolveScratch,  // pd_solve_scratch (per:row_slot arena:scratch owner:data elem:3)
    RowPenetration,  // row_penetration (per:row_slot arena:scratch owner:data)
    QdotPseudo,  // qdot_pseudo (per:link arena:scratch owner:data)
    LinkVelocityPseudo,  // link_velocity_pseudo (per:link arena:scratch owner:data)
    QdotPseudoFlat,  // qdot_pseudo_flat (per:articulation_dof arena:scratch owner:data)
    BodyPseudoLinearVelocity,  // body_pseudo_linear_velocity (per:body arena:scratch owner:data)
    BodyPseudoAngularVelocity,  // body_pseudo_angular_velocity (per:body arena:scratch owner:data)
    ParticlePseudoVel,  // particle_pseudo_vel (per:particle arena:scratch owner:data)
    RowPseudoLambda,  // row_pseudo_lambda (per:row_slot arena:scratch owner:data)
    UcontactAKind,  // ucontact_a_kind (per:contact_slot arena:scratch owner:data elem:4)
    UcontactBKind,  // ucontact_b_kind (per:contact_slot arena:scratch owner:data elem:4)
    SnapshotParticlePos,  // snapshot_particle_pos (per:particle arena:persistent owner:data)
    SnapshotParticlePrevPos,  // snapshot_particle_prev_pos (per:particle arena:persistent owner:data)
    SnapshotParticleVel,  // snapshot_particle_vel (per:particle arena:persistent owner:data)
    DistColorSegments,  // dist_color_segments (per:scalar arena:persistent owner:model count:xpbd_dist_colors*2)
    BendColorSegments,  // bend_color_segments (per:scalar arena:persistent owner:model count:xpbd_bend_colors*2)
    VolColorSegments,  // vol_color_segments (per:scalar arena:persistent owner:model count:xpbd_vol_colors*2)
    SmColorSegments,  // sm_color_segments (per:scalar arena:persistent owner:model count:xpbd_sm_colors*2)
    GridSortScratch,  // grid_sort_scratch (per:scalar arena:scratch owner:data count:grid_sort_scratch_bytes)
    GridMass,  // grid_mass (per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count)
    GridMomentum,  // grid_momentum (per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count)
    GridVelocity,  // grid_velocity (per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count)
    GridForce,  // grid_force (per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count)
    ParticleF,  // particle_F (per:particle arena:persistent owner:data elem:9 flags:[diff])
    ParticleC,  // particle_C (per:particle arena:persistent owner:data elem:9)
    ParticleVol0,  // particle_vol0 (per:particle arena:persistent owner:data)
    ParticlePlastic,  // particle_plastic (per:particle arena:persistent owner:data)
    ParticleMaterialId,  // particle_material_id (per:particle arena:persistent owner:data)
    SnapshotParticleF,  // snapshot_particle_F (per:particle arena:persistent owner:data elem:9)
    SnapshotParticleC,  // snapshot_particle_C (per:particle arena:persistent owner:data elem:9)
    SnapshotParticlePlastic,  // snapshot_particle_plastic (per:particle arena:persistent owner:data)
    MpmGridCellKey,  // mpm_grid_cell_key (per:particle arena:scratch owner:data)
    MpmGridPartIdx,  // mpm_grid_part_idx (per:particle arena:scratch owner:data)
    MpmSortScratch,  // mpm_sort_scratch (per:scalar arena:scratch owner:data count:mpm_grid_sort_scratch_bytes)
    MpmParticleStress,  // mpm_particle_stress (per:particle arena:scratch owner:data elem:9)
    MpmMaterialTable,  // mpm_material_table (per:scalar arena:persistent owner:data count:mpm_material_count*11)
    GridBodyDp,  // grid_body_dp (per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count)
    GridBodyOwner,  // grid_body_owner (per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count)
    MpmBodyReaction,  // mpm_body_reaction (per:body arena:scratch owner:data flags:[readout])
    MpmBodyAngReaction,  // mpm_body_ang_reaction (per:body arena:scratch owner:data flags:[readout])
    CcParent,  // cc_parent (per:row_slot arena:scratch owner:data)
    CcRoot,  // cc_root (per:row_slot arena:scratch owner:data)
    CcArticFirst,  // cc_artic_first (per:articulation arena:scratch owner:data)
    CcBodyFirst,  // cc_body_first (per:body arena:scratch owner:data)
    CcParticleFirst,  // cc_particle_first (per:particle arena:scratch owner:data)
    IslandRootSorted,  // island_root_sorted (per:row_slot arena:scratch owner:data)
    IslandRows,  // island_rows (per:row_slot arena:scratch owner:data)
    IslandQuads,  // island_quads (per:row_slot arena:scratch owner:data elem:4)
    IslandCount,  // island_count (per:scalar arena:scratch owner:data count:1)
    IslandCubTemp,  // island_cub_temp (per:scalar arena:scratch owner:data count:island_cub_temp_bytes)
    JointFrictionloss,  // joint_frictionloss (per:link arena:persistent owner:model flags:[param])
    BodyCollidableLink,  // body_collidable_link (per:body arena:persistent owner:model)
    BodyCollidableLocal,  // body_collidable_local (per:body arena:persistent owner:model)
    BodyCollidableBody,  // body_collidable_body (per:body arena:persistent owner:model)
    UcontactIdPair,  // ucontact_id_pair (per:contact_slot arena:scratch owner:data elem:4)
    UcontactIdFeature,  // ucontact_id_feature (per:contact_slot arena:scratch owner:data elem:4)
    ContactCachePair,  // contact_cache_pair (per:contact_slot arena:persistent owner:data elem:4)
    ContactCacheFeature,  // contact_cache_feature (per:contact_slot arena:persistent owner:data elem:4)
    ContactCacheLambda,  // contact_cache_lambda (per:contact_slot arena:persistent owner:data elem:12)
    ContactCacheNormal,  // contact_cache_normal (per:contact_slot arena:persistent owner:data elem:4)
    ContactCacheTangent1,  // contact_cache_tangent1 (per:contact_slot arena:persistent owner:data elem:4)
    ContactCacheTangent2,  // contact_cache_tangent2 (per:contact_slot arena:persistent owner:data elem:4)
    ContactCacheMaterial,  // contact_cache_material (per:contact_slot arena:persistent owner:data elem:4)
    ContactCacheAge,  // contact_cache_age (per:contact_slot arena:persistent owner:data elem:4)
    ContactCacheSnapshotPair,  // contact_cache_snapshot_pair (per:contact_slot arena:scratch owner:data elem:4)
    ContactCacheSnapshotFeature,  // contact_cache_snapshot_feature (per:contact_slot arena:scratch owner:data elem:4)
    ContactCacheSnapshotLambda,  // contact_cache_snapshot_lambda (per:contact_slot arena:scratch owner:data elem:12)
    ContactCacheSnapshotNormal,  // contact_cache_snapshot_normal (per:contact_slot arena:scratch owner:data elem:4)
    ContactCacheSnapshotTangent1,  // contact_cache_snapshot_tangent1 (per:contact_slot arena:scratch owner:data elem:4)
    ContactCacheSnapshotTangent2,  // contact_cache_snapshot_tangent2 (per:contact_slot arena:scratch owner:data elem:4)
    ContactCacheSnapshotMaterial,  // contact_cache_snapshot_material (per:contact_slot arena:scratch owner:data elem:4)
    ContactCacheSnapshotAge,  // contact_cache_snapshot_age (per:contact_slot arena:scratch owner:data elem:4)
    ContactCacheCurrentOwner,  // contact_cache_current_owner (per:contact_slot arena:scratch owner:data elem:4)
    ContactCacheOldKeep,  // contact_cache_old_keep (per:contact_slot arena:scratch owner:data elem:4)
    JointLimitLower,  // joint_limit_lower (per:link arena:persistent owner:model flags:[param])
    JointLimitUpper,  // joint_limit_upper (per:link arena:persistent owner:model flags:[param])
    JointLimitFlags,  // joint_limit_flags (per:link arena:persistent owner:model flags:[param])
    JointLimitImpulse,  // joint_limit_impulse (per:link arena:scratch owner:data elem:2 flags:[readout])
    ActuatorEffortRequested,  // actuator_effort_requested (per:link arena:scratch owner:data flags:[readout])
    ActuatorEffort,  // actuator_effort (per:link arena:scratch owner:data flags:[readout])
    ActuatorSaturated,  // actuator_saturated (per:link arena:scratch owner:data flags:[readout])
    PairSortScratch,  // pair_sort_scratch (per:scalar arena:scratch owner:data count:pair_sort_scratch_bytes)
    TaskTarget,  // task_target (per:articulation arena:persistent owner:data flags:[param])
    TaskRotationTarget,  // task_rotation_target (per:articulation arena:persistent owner:data flags:[param])
    TaskLocalPose,  // task_local_pose (per:articulation arena:persistent owner:data flags:[param])
    BodyInertialFrame,  // body_inertial_frame (per:body arena:persistent owner:data flags:[param])
    BodyWorldInvInertia,  // body_world_inv_inertia (per:body arena:scratch owner:data)
    StepQdotFlat,  // step_qdot_flat (per:articulation_dof arena:scratch owner:data)
    StepBodyLinearVelocity,  // step_body_linear_velocity (per:body arena:scratch owner:data)
    StepBodyAngularVelocity,  // step_body_angular_velocity (per:body arena:scratch owner:data)
    StepParticleVelocity,  // step_particle_velocity (per:particle arena:scratch owner:data)
    ContactSideAKind,  // contact_side_a_kind (per:contact_slot arena:scratch owner:data flags:[readout])
    ContactSideBKind,  // contact_side_b_kind (per:contact_slot arena:scratch owner:data flags:[readout])
    ContactSideAIndex,  // contact_side_a_index (per:contact_slot arena:scratch owner:data flags:[readout])
    ContactSideBIndex,  // contact_side_b_index (per:contact_slot arena:scratch owner:data flags:[readout])
    AeroParticleOffset,  // aero_particle_offset (per:particle arena:persistent owner:model)
    AeroParticleCount,  // aero_particle_count (per:particle arena:persistent owner:model)
    AeroIncidentTri,  // aero_incident_tri (per:aero_tri arena:persistent owner:model elem:3)
    AeroTriImpulse,  // aero_tri_impulse (per:aero_tri arena:scratch owner:data)
    BodyGyroResidual,  // body_gyro_residual (per:body arena:scratch owner:data flags:[readout])
    BodyGyroIterations,  // body_gyro_iterations (per:body arena:scratch owner:data flags:[readout])
    BodyGyroStatus,  // body_gyro_status (per:body arena:scratch owner:data flags:[readout])
    GridNeighborAttempted,  // grid_neighbor_attempted (per:particle arena:scratch owner:data flags:[readout])
    GridNeighborScanOffset,  // grid_neighbor_scan_offset (per:particle arena:scratch owner:data)
    LbvhSortScratch,  // lbvh_sort_scratch (per:scalar arena:scratch owner:data count:lbvh_sort_scratch_bytes)
    ContactCacheScratch,  // contact_cache_scratch (per:scalar arena:scratch owner:data count:contact_cache_scratch_bytes)
    ActiveRowIds,  // active_row_ids (per:row_slot arena:scratch owner:data)
    ActiveRowCount,  // active_row_count (per:env arena:scratch owner:data)
    ContactEndpointKeys,  // contact_endpoint_keys (per:row_slot arena:scratch owner:data elem:2)
    ContactEndpointCount,  // contact_endpoint_count (per:env arena:scratch owner:data)
    LinkContactBegin,  // link_contact_begin (per:link arena:scratch owner:data)
    LinkContactEnd,  // link_contact_end (per:link arena:scratch owner:data)
    ContactIndexScratch,  // contact_index_scratch (per:scalar arena:scratch owner:data count:contact_index_scratch_bytes)
    ParticleTopologyOffsets,  // particle_topology_offsets (per:scalar arena:persistent owner:model count:particle_topology_offsets)
    ParticleTopologyElements,  // particle_topology_elements (per:scalar arena:persistent owner:model count:particle_topology_incidence_count)
    ParticleContactRestPos,  // particle_contact_rest_pos (per:scalar arena:persistent owner:model count:particle_contact_rest_positions)
    MeshSurfaceInfo,  // mesh_surface_info (per:scalar arena:persistent owner:model count:mesh_surface_info_count)
    MeshTriangles,  // mesh_triangles (per:scalar arena:persistent owner:model count:max_mesh_triangles*3)
    MeshBvhNodes,  // mesh_bvh_nodes (per:scalar arena:persistent owner:model count:max_mesh_bvh_nodes)
    StepEnvStatus,  // step_env_status (per:env arena:scratch owner:data)
    StepMpmBodyImpulse,  // step_mpm_body_impulse (per:body arena:scratch owner:data)
    StepMpmBodyMoment,  // step_mpm_body_moment (per:body arena:scratch owner:data)
    StepLinkImpulse,  // step_link_impulse (per:link arena:scratch owner:data)
    StepLinkMoment,  // step_link_moment (per:link arena:scratch owner:data)
    StepJointLimitImpulse,  // step_joint_limit_impulse (per:link arena:scratch owner:data elem:2)
    ParticlePlasticF,  // particle_plastic_F (per:particle arena:persistent owner:data elem:9)
    SnapshotParticlePlasticF,  // snapshot_particle_plastic_F (per:particle arena:persistent owner:data elem:9)
    GridInvMass,  // grid_inv_mass (per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count)
    CcGridFirst,  // cc_grid_first (per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count)
    GridContactCount,  // grid_contact_count (per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count)
    GridContactOffset,  // grid_contact_offset (per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count)
    UcontactLaw,  // ucontact_law (per:contact_slot arena:scratch owner:data)
    UcontactFriction,  // ucontact_friction (per:contact_slot arena:scratch owner:data)
    GridContactAttempted,  // grid_contact_attempted (per:env arena:scratch owner:data flags:[readout])
    GridContactRetained,  // grid_contact_retained (per:env arena:scratch owner:data flags:[readout])
    GridContactPeak,  // grid_contact_peak (per:env arena:persistent owner:data flags:[readout])
    GridContactOverflow,  // grid_contact_overflow (per:env arena:scratch owner:data flags:[readout])
    MpmBoundaryImpulse,  // mpm_boundary_impulse (per:scalar arena:scratch owner:data count:mpm_boundary_count*env_count flags:[readout])
    MpmBoundaryMoment,  // mpm_boundary_moment (per:scalar arena:scratch owner:data count:mpm_boundary_count*env_count flags:[readout])
    StepMpmBoundaryImpulse,  // step_mpm_boundary_impulse (per:scalar arena:scratch owner:data count:mpm_boundary_count*env_count)
    StepMpmBoundaryMoment,  // step_mpm_boundary_moment (per:scalar arena:scratch owner:data count:mpm_boundary_count*env_count)
    SolverVelocityScratch,  // solver_velocity_scratch (per:scalar arena:scratch owner:data count:solver_velocity_scratch_bytes)
    VelocityTarget,  // velocity_target (per:link arena:persistent owner:data flags:[param])
    AccelerationTarget,  // acceleration_target (per:link arena:persistent owner:data flags:[param])
    ActuatorNoloadSpeed,  // actuator_noload_speed (per:link arena:persistent owner:data flags:[param])
    TaskNullspaceStiffness,  // task_nullspace_stiffness (per:articulation arena:persistent owner:data flags:[param])
    TaskNullspaceDamping,  // task_nullspace_damping (per:articulation arena:persistent owner:data flags:[param])
    DriveCommand,  // drive_command (per:link arena:scratch owner:data)
    DriveDissipation,  // drive_dissipation (per:link arena:scratch owner:data)
    DriveLower,  // drive_lower (per:link arena:scratch owner:data)
    DriveUpper,  // drive_upper (per:link arena:scratch owner:data)
    ControlMass,  // control_mass (per:articulation_dof2 arena:scratch owner:data)
    ControlFactor,  // control_factor (per:articulation_dof2 arena:scratch owner:data)
    ControlJacobian,  // control_jacobian (per:articulation_dof arena:scratch owner:data elem:6)
    ControlResponse,  // control_response (per:articulation_dof arena:scratch owner:data elem:6)
    ControlTaskMap,  // control_task_map (per:articulation_dof arena:scratch owner:data elem:6)
    Count
};

inline constexpr int kFieldCount = static_cast<int>(FieldId::Count);
inline constexpr const char* kFieldNames[kFieldCount] = {
    "q",
    "qdot",
    "link_pose",
    "base_pose",
    "body_pose",
    "body_inv_mass",
    "contact_count",
    "rows",
    "lambda",
    "m_inv",
    "particle_pos",
    "mat_buckets",
    "mat_index",
    "link_contact_wrench",
    "qddot",
    "tau",
    "joint_force",
    "joint_diagonal",
    "link_velocity",
    "link_acceleration",
    "link_velocity_bias",
    "link_xup",
    "link_articulated_I",
    "link_bias_force",
    "link_u_spatial",
    "joint_motion_subspace",
    "drive_target",
    "drive_stiffness",
    "drive_damping",
    "drive_force_limit",
    "snapshot_q",
    "snapshot_qdot",
    "snapshot_link_velocity",
    "snapshot_base_pose",
    "snapshot_body_pose",
    "snapshot_body_linear_velocity",
    "snapshot_body_angular_velocity",
    "reset_env_ids",
    "link_inertia",
    "link_local_pose",
    "link_inertial_frame",
    "joint_axis",
    "parent_offset",
    "joint_type",
    "parent_link",
    "link_body",
    "link_to_articulation",
    "joint_damping",
    "joint_armature",
    "articulation_link_count",
    "articulation_link_offset",
    "foot_shape",
    "body_linear_velocity",
    "body_angular_velocity",
    "body_force",
    "body_torque",
    "body_inv_inertia",
    "body_aabb_lo",
    "body_aabb_hi",
    "pair_count",
    "candidate_pairs",
    "lbvh_nodes",
    "lbvh_morton",
    "lbvh_index",
    "lbvh_visit",
    "lbvh_sortkey",
    "contact_link",
    "contact_point",
    "contact_normal",
    "contact_depth",
    "contact_tangent1",
    "contact_tangent2",
    "contact_material",
    "jac_normal",
    "jac_tangent1",
    "jac_tangent2",
    "contact_meff_normal",
    "contact_meff_tangent1",
    "contact_meff_tangent2",
    "contact_force",
    "ucontact_count",
    "ucontact_point",
    "ucontact_normal",
    "ucontact_depth",
    "row_count",
    "urows",
    "chain_jacobian",
    "row_minv_jt",
    "row_meff",
    "row_damping",
    "row_cj_link",
    "row_cj_point",
    "row_cj_dir",
    "qdot_flat",
    "m",
    "link_composite_inertia",
    "hull_verts",
    "dof_to_link",
    "dof_to_component",
    "shape_table",
    "excluded_pairs",
    "samp_points",
    "samp_ranges",
    "sdf_headers",
    "sdf_cell_count",
    "sdf_cell_keys",
    "sdf_cell_values",
    "sdf_cell_gradients",
    "island_row_offsets",
    "island_color_segments",
    "row_order",
    "particle_prev_pos",
    "particle_vel",
    "particle_inv_mass",
    "particle_v_pre",
    "dist_particle_a",
    "dist_particle_b",
    "dist_rest_length",
    "dist_compliance",
    "dist_lambda",
    "bend_particles",
    "bend_gradients",
    "bend_compliance",
    "bend_lambda",
    "vol_particles",
    "vol_rest_times6",
    "vol_compliance",
    "vol_lambda",
    "sm_cluster_offset",
    "sm_cluster_size",
    "sm_stiffness",
    "sm_rest_centroid",
    "sm_particles",
    "sm_rest_q",
    "sm_mass",
    "aero_tri_verts",
    "aero_tri_area",
    "pbf_predicted_pos",
    "pbf_position_delta",
    "pbf_density",
    "pbf_lambda",
    "grid_cell_key",
    "grid_particle_idx",
    "grid_cell_start",
    "grid_cell_end",
    "grid_neighbor_offset",
    "grid_neighbor_count",
    "grid_neighbor_idx",
    "rng_state",
    "env_status",
    "obs_buffer",
    "env_terrain_type",
    "env_terrain_difficulty",
    "joint_f",
    "link_geom_kind",
    "link_geom_params",
    "link_geom_local",
    "body_to_link",
    "body_to_articulation",
    "ucontact_a",
    "ucontact_b",
    "ucontact_gen",
    "heights",
    "ucontact_tangent1",
    "ucontact_tangent2",
    "chain_jacobian_b",
    "row_minv_jt_b",
    "row_cj_link_b",
    "row_cj_point_b",
    "row_cj_dir_b",
    "pd_solve_scratch",
    "row_penetration",
    "qdot_pseudo",
    "link_velocity_pseudo",
    "qdot_pseudo_flat",
    "body_pseudo_linear_velocity",
    "body_pseudo_angular_velocity",
    "particle_pseudo_vel",
    "row_pseudo_lambda",
    "ucontact_a_kind",
    "ucontact_b_kind",
    "snapshot_particle_pos",
    "snapshot_particle_prev_pos",
    "snapshot_particle_vel",
    "dist_color_segments",
    "bend_color_segments",
    "vol_color_segments",
    "sm_color_segments",
    "grid_sort_scratch",
    "grid_mass",
    "grid_momentum",
    "grid_velocity",
    "grid_force",
    "particle_F",
    "particle_C",
    "particle_vol0",
    "particle_plastic",
    "particle_material_id",
    "snapshot_particle_F",
    "snapshot_particle_C",
    "snapshot_particle_plastic",
    "mpm_grid_cell_key",
    "mpm_grid_part_idx",
    "mpm_sort_scratch",
    "mpm_particle_stress",
    "mpm_material_table",
    "grid_body_dp",
    "grid_body_owner",
    "mpm_body_reaction",
    "mpm_body_ang_reaction",
    "cc_parent",
    "cc_root",
    "cc_artic_first",
    "cc_body_first",
    "cc_particle_first",
    "island_root_sorted",
    "island_rows",
    "island_quads",
    "island_count",
    "island_cub_temp",
    "joint_frictionloss",
    "body_collidable_link",
    "body_collidable_local",
    "body_collidable_body",
    "ucontact_id_pair",
    "ucontact_id_feature",
    "contact_cache_pair",
    "contact_cache_feature",
    "contact_cache_lambda",
    "contact_cache_normal",
    "contact_cache_tangent1",
    "contact_cache_tangent2",
    "contact_cache_material",
    "contact_cache_age",
    "contact_cache_snapshot_pair",
    "contact_cache_snapshot_feature",
    "contact_cache_snapshot_lambda",
    "contact_cache_snapshot_normal",
    "contact_cache_snapshot_tangent1",
    "contact_cache_snapshot_tangent2",
    "contact_cache_snapshot_material",
    "contact_cache_snapshot_age",
    "contact_cache_current_owner",
    "contact_cache_old_keep",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_limit_flags",
    "joint_limit_impulse",
    "actuator_effort_requested",
    "actuator_effort",
    "actuator_saturated",
    "pair_sort_scratch",
    "task_target",
    "task_rotation_target",
    "task_local_pose",
    "body_inertial_frame",
    "body_world_inv_inertia",
    "step_qdot_flat",
    "step_body_linear_velocity",
    "step_body_angular_velocity",
    "step_particle_velocity",
    "contact_side_a_kind",
    "contact_side_b_kind",
    "contact_side_a_index",
    "contact_side_b_index",
    "aero_particle_offset",
    "aero_particle_count",
    "aero_incident_tri",
    "aero_tri_impulse",
    "body_gyro_residual",
    "body_gyro_iterations",
    "body_gyro_status",
    "grid_neighbor_attempted",
    "grid_neighbor_scan_offset",
    "lbvh_sort_scratch",
    "contact_cache_scratch",
    "active_row_ids",
    "active_row_count",
    "contact_endpoint_keys",
    "contact_endpoint_count",
    "link_contact_begin",
    "link_contact_end",
    "contact_index_scratch",
    "particle_topology_offsets",
    "particle_topology_elements",
    "particle_contact_rest_pos",
    "mesh_surface_info",
    "mesh_triangles",
    "mesh_bvh_nodes",
    "step_env_status",
    "step_mpm_body_impulse",
    "step_mpm_body_moment",
    "step_link_impulse",
    "step_link_moment",
    "step_joint_limit_impulse",
    "particle_plastic_F",
    "snapshot_particle_plastic_F",
    "grid_inv_mass",
    "cc_grid_first",
    "grid_contact_count",
    "grid_contact_offset",
    "ucontact_law",
    "ucontact_friction",
    "grid_contact_attempted",
    "grid_contact_retained",
    "grid_contact_peak",
    "grid_contact_overflow",
    "mpm_boundary_impulse",
    "mpm_boundary_moment",
    "step_mpm_boundary_impulse",
    "step_mpm_boundary_moment",
    "solver_velocity_scratch",
    "velocity_target",
    "acceleration_target",
    "actuator_noload_speed",
    "task_nullspace_stiffness",
    "task_nullspace_damping",
    "drive_command",
    "drive_dissipation",
    "drive_lower",
    "drive_upper",
    "control_mass",
    "control_factor",
    "control_jacobian",
    "control_response",
    "control_task_map",
};
inline constexpr const char* FieldName(FieldId id) {
    return static_cast<int>(id) < kFieldCount ? kFieldNames[static_cast<int>(id)] : "unknown";
}

} // namespace nuka::nk
