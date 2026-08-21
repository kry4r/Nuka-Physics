// ---------------------------------------------------------------------------
// nk::Data implementation (plan §3.3).
// ---------------------------------------------------------------------------

#include "nk/data/data.hpp"

#include "nk/model/model.hpp"   // ModelCapacities

#include <algorithm>

namespace nuka::nk {

namespace {

// Bind one data-owned field's device pointer into the DataView by FieldId. The
// switch is exhaustive over the data-owned fields; a model-owned id falls to the
// default (those are ModelView members, not here).
void BindDataPointer(phi::DataView& v, FieldId id, void* p) {
    switch (id) {
        case FieldId::Q:                   v.q = static_cast<float*>(p); break;
        case FieldId::Qdot:                v.qdot = static_cast<float*>(p); break;
        case FieldId::LinkPose:            v.link_pose = static_cast<math::Transform*>(p); break;
        case FieldId::BasePose:            v.base_pose = static_cast<math::Transform*>(p); break;
        case FieldId::BodyPose:            v.body_pose = static_cast<math::Transform*>(p); break;
        case FieldId::BodyInvMass:         v.body_inv_mass = static_cast<float*>(p); break;
        case FieldId::ContactCount:        v.contact_count = static_cast<uint32_t*>(p); break;
        case FieldId::Rows:                v.rows = static_cast<float*>(p); break;
        case FieldId::Lambda:              v.lambda = static_cast<float*>(p); break;
        case FieldId::MInv:                v.m_inv = static_cast<float*>(p); break;
        case FieldId::ParticlePos:         v.particle_pos = static_cast<math::Vec3*>(p); break;
        case FieldId::MatBuckets:          v.mat_buckets = static_cast<float*>(p); break;
        case FieldId::MatIndex:            v.mat_index = static_cast<uint32_t*>(p); break;
        case FieldId::LinkContactWrench:   v.link_contact_wrench = static_cast<Spatial6*>(p); break;
        case FieldId::Qddot:               v.qddot = static_cast<float*>(p); break;
        case FieldId::Tau:                 v.tau = static_cast<float*>(p); break;
        case FieldId::JointForce:          v.joint_force = static_cast<float*>(p); break;
        case FieldId::JointDiagonal:       v.joint_diagonal = static_cast<float*>(p); break;
        case FieldId::LinkVelocity:        v.link_velocity = static_cast<Spatial6*>(p); break;
        case FieldId::LinkAcceleration:    v.link_acceleration = static_cast<Spatial6*>(p); break;
        case FieldId::LinkVelocityBias:    v.link_velocity_bias = static_cast<Spatial6*>(p); break;
        case FieldId::LinkXup:             v.link_xup = static_cast<Mat36*>(p); break;
        case FieldId::LinkArticulatedI:    v.link_articulated_I = static_cast<Mat36*>(p); break;
        case FieldId::LinkBiasForce:       v.link_bias_force = static_cast<Spatial6*>(p); break;
        case FieldId::LinkUSpatial:        v.link_u_spatial = static_cast<Spatial6*>(p); break;
        case FieldId::JointMotionSubspace: v.joint_motion_subspace = static_cast<Spatial6*>(p); break;
        case FieldId::DriveTarget:         v.drive_target = static_cast<float*>(p); break;
        case FieldId::DriveStiffness:      v.drive_stiffness = static_cast<float*>(p); break;
        case FieldId::DriveDamping:        v.drive_damping = static_cast<float*>(p); break;
        case FieldId::DriveForceLimit:     v.drive_force_limit = static_cast<float*>(p); break;
        case FieldId::SnapshotQ:           v.snapshot_q = static_cast<float*>(p); break;
        case FieldId::SnapshotQdot:        v.snapshot_qdot = static_cast<float*>(p); break;
        case FieldId::SnapshotLinkVelocity:v.snapshot_link_velocity = static_cast<Spatial6*>(p); break;
        case FieldId::SnapshotBasePose:    v.snapshot_base_pose = static_cast<math::Transform*>(p); break;
        // M7 T1: movable rigid-body snapshot fields (restore source for the M7
        // settle consumer). Bound here so OpSnapshotState/OpRestoreState/
        // OpResetEnvs see live arena pointers (the M5-binding-bug class).
        case FieldId::SnapshotBodyPose:    v.snapshot_body_pose = static_cast<math::Transform*>(p); break;
        case FieldId::SnapshotBodyLinearVelocity:  v.snapshot_body_linear_velocity = static_cast<math::Vec3*>(p); break;
        case FieldId::SnapshotBodyAngularVelocity: v.snapshot_body_angular_velocity = static_cast<math::Vec3*>(p); break;
        case FieldId::ResetEnvIds:         v.reset_env_ids = static_cast<uint32_t*>(p); break;
        case FieldId::BodyLinearVelocity:  v.body_linear_velocity = static_cast<math::Vec3*>(p); break;
        case FieldId::BodyAngularVelocity: v.body_angular_velocity = static_cast<math::Vec3*>(p); break;
        case FieldId::BodyForce:           v.body_force = static_cast<math::Vec3*>(p); break;
        case FieldId::BodyTorque:          v.body_torque = static_cast<math::Vec3*>(p); break;
        case FieldId::BodyInvInertia:      v.body_inv_inertia = static_cast<math::Vec3*>(p); break;
        case FieldId::BodyAabbLo:          v.body_aabb_lo = static_cast<math::Vec3*>(p); break;
        case FieldId::BodyAabbHi:          v.body_aabb_hi = static_cast<math::Vec3*>(p); break;
        case FieldId::PairCount:           v.pair_count = static_cast<uint32_t*>(p); break;
        case FieldId::CandidatePairs:      v.candidate_pairs = static_cast<uint32_t*>(p); break;
        // M5 LBVH broadphase scratch (review fix: these were MISSING — the
        // LbvhBuild/LbvhQueryPairs ops read them from the DataView and would
        // have dereferenced null device pointers on the pair-driven path).
        case FieldId::LbvhNodes:           v.lbvh_nodes = static_cast<float*>(p); break;
        case FieldId::LbvhMorton:          v.lbvh_morton = static_cast<uint32_t*>(p); break;
        case FieldId::LbvhIndex:           v.lbvh_index = static_cast<uint32_t*>(p); break;
        case FieldId::LbvhVisit:           v.lbvh_visit = static_cast<uint32_t*>(p); break;
        case FieldId::LbvhSortkey:         v.lbvh_sortkey = static_cast<uint64_t*>(p); break;
        case FieldId::ContactLink:         v.contact_link = static_cast<uint32_t*>(p); break;
        case FieldId::ContactPoint:        v.contact_point = static_cast<math::Vec3*>(p); break;
        case FieldId::ContactNormal:       v.contact_normal = static_cast<math::Vec3*>(p); break;
        case FieldId::ContactDepth:        v.contact_depth = static_cast<float*>(p); break;
        case FieldId::ContactTangent1:     v.contact_tangent1 = static_cast<math::Vec3*>(p); break;
        case FieldId::ContactTangent2:     v.contact_tangent2 = static_cast<math::Vec3*>(p); break;
        case FieldId::ContactMaterial:     v.contact_material = static_cast<uint64_t*>(p); break;
        case FieldId::JacNormal:           v.jac_normal = static_cast<float*>(p); break;
        case FieldId::JacTangent1:         v.jac_tangent1 = static_cast<float*>(p); break;
        case FieldId::JacTangent2:         v.jac_tangent2 = static_cast<float*>(p); break;
        case FieldId::ContactMeffNormal:   v.contact_meff_normal = static_cast<float*>(p); break;
        case FieldId::ContactMeffTangent1: v.contact_meff_tangent1 = static_cast<float*>(p); break;
        case FieldId::ContactMeffTangent2: v.contact_meff_tangent2 = static_cast<float*>(p); break;
        case FieldId::ContactForce:        v.contact_force = static_cast<float*>(p); break;
        case FieldId::UcontactCount:       v.ucontact_count = static_cast<uint32_t*>(p); break;
        case FieldId::UcontactPoint:       v.ucontact_point = static_cast<math::Vec3*>(p); break;
        case FieldId::UcontactNormal:      v.ucontact_normal = static_cast<math::Vec3*>(p); break;
        case FieldId::UcontactDepth:       v.ucontact_depth = static_cast<float*>(p); break;
        // C1 (general contact pipeline Phase 0): per-slot collidable-id + gen
        // fields on the unified contact buffer (INERT in Phase 0; stamped by C2).
        case FieldId::UcontactA:           v.ucontact_a = static_cast<uint32_t*>(p); break;
        case FieldId::UcontactB:           v.ucontact_b = static_cast<uint32_t*>(p); break;
        case FieldId::UcontactGen:         v.ucontact_gen = static_cast<uint32_t*>(p); break;
        case FieldId::UcontactIdPair:      v.ucontact_id_pair = static_cast<uint64_t*>(p); break;
        case FieldId::UcontactIdFeature:   v.ucontact_id_feature = static_cast<uint64_t*>(p); break;
        case FieldId::ContactCachePair:    v.contact_cache_pair = static_cast<uint64_t*>(p); break;
        case FieldId::ContactCacheFeature: v.contact_cache_feature = static_cast<uint64_t*>(p); break;
        case FieldId::ContactCacheLambda: v.contact_cache_lambda = static_cast<float*>(p); break;
        case FieldId::ContactCacheNormal: v.contact_cache_normal = static_cast<math::Vec3*>(p); break;
        case FieldId::ContactCacheTangent1: v.contact_cache_tangent1 = static_cast<math::Vec3*>(p); break;
        case FieldId::ContactCacheTangent2: v.contact_cache_tangent2 = static_cast<math::Vec3*>(p); break;
        case FieldId::ContactCacheMaterial: v.contact_cache_material = static_cast<uint64_t*>(p); break;
        case FieldId::ContactCacheAge:     v.contact_cache_age = static_cast<uint32_t*>(p); break;
        case FieldId::ContactCacheSnapshotPair: v.contact_cache_snapshot_pair = static_cast<uint64_t*>(p); break;
        case FieldId::ContactCacheSnapshotFeature: v.contact_cache_snapshot_feature = static_cast<uint64_t*>(p); break;
        case FieldId::ContactCacheSnapshotLambda: v.contact_cache_snapshot_lambda = static_cast<float*>(p); break;
        case FieldId::ContactCacheSnapshotNormal: v.contact_cache_snapshot_normal = static_cast<math::Vec3*>(p); break;
        case FieldId::ContactCacheSnapshotTangent1: v.contact_cache_snapshot_tangent1 = static_cast<math::Vec3*>(p); break;
        case FieldId::ContactCacheSnapshotTangent2: v.contact_cache_snapshot_tangent2 = static_cast<math::Vec3*>(p); break;
        case FieldId::ContactCacheSnapshotMaterial: v.contact_cache_snapshot_material = static_cast<uint64_t*>(p); break;
        case FieldId::ContactCacheSnapshotAge: v.contact_cache_snapshot_age = static_cast<uint32_t*>(p); break;
        case FieldId::ContactCacheCurrentOwner: v.contact_cache_current_owner = static_cast<uint32_t*>(p); break;
        case FieldId::ContactCacheOldKeep: v.contact_cache_old_keep = static_cast<uint32_t*>(p); break;
        // Per-slot side-kind tag for each ucontact_a/ucontact_b index (a body-local
        // collidable row vs a global particle id); the assembly resolves the side
        // by this tag before the shape_table body-row lookup.
        case FieldId::UcontactAKind:       v.ucontact_a_kind = static_cast<uint32_t*>(p); break;
        case FieldId::UcontactBKind:       v.ucontact_b_kind = static_cast<uint32_t*>(p); break;
        // C4 (general contact pipeline Phase 1A): per-point tangent basis for the
        // unified PairDriven contact buffer (ComputeUnionContactTangentBasisKernel).
        case FieldId::UcontactTangent1:    v.ucontact_tangent1 = static_cast<math::Vec3*>(p); break;
        case FieldId::UcontactTangent2:    v.ucontact_tangent2 = static_cast<math::Vec3*>(p); break;
        case FieldId::RowCount:            v.row_count = static_cast<uint32_t*>(p); break;
        case FieldId::Urows:               v.urows = static_cast<float*>(p); break;
        case FieldId::ChainJacobian:       v.chain_jacobian = static_cast<float*>(p); break;
        case FieldId::RowMinvJt:           v.row_minv_jt = static_cast<float*>(p); break;
        case FieldId::RowMeff:             v.row_meff = static_cast<float*>(p); break;
        case FieldId::RowCjLink:           v.row_cj_link = static_cast<uint32_t*>(p); break;
        case FieldId::RowCjPoint:          v.row_cj_point = static_cast<math::Vec3*>(p); break;
        case FieldId::RowCjDir:            v.row_cj_dir = static_cast<math::Vec3*>(p); break;
        // S2 (general contact pipeline Phase 1B): the second per-side arm fields.
        case FieldId::ChainJacobianB:      v.chain_jacobian_b = static_cast<float*>(p); break;
        case FieldId::RowMinvJtB:          v.row_minv_jt_b = static_cast<float*>(p); break;
        case FieldId::RowCjLinkB:          v.row_cj_link_b = static_cast<uint32_t*>(p); break;
        case FieldId::RowCjPointB:         v.row_cj_point_b = static_cast<math::Vec3*>(p); break;
        case FieldId::RowCjDirB:           v.row_cj_dir_b = static_cast<math::Vec3*>(p); break;
        // PairDriven block-island GLOBAL order/segment scratch (3 u32 per row slot:
        // rows_per_env order + 2*rows_per_env segments per env, disjoint per env).
        case FieldId::PdSolveScratch:      v.pd_solve_scratch = static_cast<uint32_t*>(p); break;
        case FieldId::QdotFlat:            v.qdot_flat = static_cast<float*>(p); break;
        // Split-impulse position pass: SEPARATE pseudo-velocity accumulators (never
        // the persisted velocity). Written by the gated position sweep, read by
        // IntegratePosition as (real+pseudo)*dt. Zero on the pos_iters==0 path.
        case FieldId::RowPenetration:      v.row_penetration = static_cast<float*>(p); break;
        case FieldId::QdotPseudo:          v.qdot_pseudo = static_cast<float*>(p); break;
        case FieldId::LinkVelocityPseudo:  v.link_velocity_pseudo = static_cast<Spatial6*>(p); break;
        case FieldId::QdotPseudoFlat:      v.qdot_pseudo_flat = static_cast<float*>(p); break;
        case FieldId::BodyPseudoLinearVelocity:  v.body_pseudo_linear_velocity = static_cast<math::Vec3*>(p); break;
        case FieldId::BodyPseudoAngularVelocity: v.body_pseudo_angular_velocity = static_cast<math::Vec3*>(p); break;
        case FieldId::ParticlePseudoVel:   v.particle_pseudo_vel = static_cast<math::Vec3*>(p); break;
        case FieldId::RowPseudoLambda:     v.row_pseudo_lambda = static_cast<float*>(p); break;
        // L1-c: FieldId::TableEnabled (v.table_enabled) was DELETED with the field.
        case FieldId::M:                   v.m = static_cast<float*>(p); break;
        case FieldId::LinkCompositeInertia:v.link_composite_inertia = static_cast<Mat36*>(p); break;
        case FieldId::ParticlePrevPos:     v.particle_prev_pos = static_cast<math::Vec3*>(p); break;
        case FieldId::ParticleVel:         v.particle_vel = static_cast<math::Vec3*>(p); break;
        case FieldId::ParticleInvMass:     v.particle_inv_mass = static_cast<float*>(p); break;
        // Particle snapshot fields (restore source for a coupled-world Reset). Bound
        // here so OpSnapshotState/OpRestoreState/OpResetEnvs see live arena pointers
        // (an unbound field stays null and the snapshot/restore copy would fail).
        case FieldId::SnapshotParticlePos:     v.snapshot_particle_pos = static_cast<math::Vec3*>(p); break;
        case FieldId::SnapshotParticlePrevPos: v.snapshot_particle_prev_pos = static_cast<math::Vec3*>(p); break;
        case FieldId::SnapshotParticleVel:     v.snapshot_particle_vel = static_cast<math::Vec3*>(p); break;
        // MLS-MPM env-private background grid + per-particle continuum state +
        // snapshot triple + P2G gather scratch + material table (all data-owned;
        // an unbound field stays null and the MPM op would deref nullptr).
        case FieldId::GridMass:            v.grid_mass = static_cast<float*>(p); break;
        case FieldId::GridMomentum:        v.grid_momentum = static_cast<math::Vec3*>(p); break;
        case FieldId::GridVelocity:        v.grid_velocity = static_cast<math::Vec3*>(p); break;
        case FieldId::GridForce:           v.grid_force = static_cast<math::Vec3*>(p); break;
        case FieldId::ParticleF:           v.particle_F = static_cast<float*>(p); break;
        case FieldId::ParticleC:           v.particle_C = static_cast<float*>(p); break;
        case FieldId::ParticleVol0:        v.particle_vol0 = static_cast<float*>(p); break;
        case FieldId::ParticlePlastic:     v.particle_plastic = static_cast<float*>(p); break;
        case FieldId::ParticleMaterialId:  v.particle_material_id = static_cast<uint32_t*>(p); break;
        case FieldId::SnapshotParticleF:       v.snapshot_particle_F = static_cast<float*>(p); break;
        case FieldId::SnapshotParticleC:       v.snapshot_particle_C = static_cast<float*>(p); break;
        case FieldId::SnapshotParticlePlastic: v.snapshot_particle_plastic = static_cast<float*>(p); break;
        case FieldId::MpmGridCellKey:      v.mpm_grid_cell_key = static_cast<uint32_t*>(p); break;
        case FieldId::MpmGridPartIdx:      v.mpm_grid_part_idx = static_cast<uint32_t*>(p); break;
        case FieldId::MpmSortScratch:      v.mpm_sort_scratch = static_cast<uint8_t*>(p); break;
        case FieldId::MpmParticleStress:   v.mpm_particle_stress = static_cast<float*>(p); break;
        case FieldId::MpmMaterialTable:    v.mpm_material_table = static_cast<float*>(p); break;
        case FieldId::GridBodyDp:          v.grid_body_dp = static_cast<math::Vec3*>(p); break;
        case FieldId::GridBodyOwner:       v.grid_body_owner = static_cast<uint32_t*>(p); break;
        case FieldId::MpmBodyReaction:     v.mpm_body_reaction = static_cast<math::Vec3*>(p); break;
        case FieldId::MpmBodyAngReaction:  v.mpm_body_ang_reaction = static_cast<math::Vec3*>(p); break;
        // Dynamic solve islands (BuildSolveIslands union-find + emit working set).
        case FieldId::CcParent:            v.cc_parent = static_cast<uint32_t*>(p); break;
        case FieldId::CcRoot:              v.cc_root = static_cast<uint32_t*>(p); break;
        case FieldId::CcArticFirst:        v.cc_artic_first = static_cast<uint32_t*>(p); break;
        case FieldId::CcBodyFirst:         v.cc_body_first = static_cast<uint32_t*>(p); break;
        case FieldId::CcParticleFirst:     v.cc_particle_first = static_cast<uint32_t*>(p); break;
        case FieldId::IslandRootSorted:    v.island_root_sorted = static_cast<uint32_t*>(p); break;
        case FieldId::IslandRows:          v.island_rows = static_cast<uint32_t*>(p); break;
        case FieldId::IslandQuads:         v.island_quads = static_cast<uint32_t*>(p); break;
        case FieldId::IslandCount:         v.island_count = static_cast<uint32_t*>(p); break;
        case FieldId::IslandCubTemp:       v.island_cub_temp = static_cast<uint8_t*>(p); break;
        case FieldId::DistLambda:          v.dist_lambda = static_cast<float*>(p); break;
        case FieldId::BendLambda:          v.bend_lambda = static_cast<float*>(p); break;
        case FieldId::VolLambda:           v.vol_lambda = static_cast<float*>(p); break;
        case FieldId::PbfPredictedPos:     v.pbf_predicted_pos = static_cast<math::Vec3*>(p); break;
        case FieldId::PbfPositionDelta:    v.pbf_position_delta = static_cast<math::Vec3*>(p); break;
        case FieldId::PbfDensity:          v.pbf_density = static_cast<float*>(p); break;
        case FieldId::PbfLambda:           v.pbf_lambda = static_cast<float*>(p); break;
        // M6 coupled v_pre scratch + the M5 particle uniform-grid CSR fields (the
        // grid build op's neighbor structure — bound here so the ported thrust /
        // CSR kernels see live arena pointers instead of null).
        case FieldId::ParticleVPre:        v.particle_v_pre = static_cast<math::Vec3*>(p); break;
        case FieldId::GridCellKey:         v.grid_cell_key = static_cast<uint32_t*>(p); break;
        case FieldId::GridParticleIdx:     v.grid_particle_idx = static_cast<uint32_t*>(p); break;
        case FieldId::GridCellStart:       v.grid_cell_start = static_cast<uint32_t*>(p); break;
        case FieldId::GridCellEnd:         v.grid_cell_end = static_cast<uint32_t*>(p); break;
        case FieldId::GridNeighborOffset:  v.grid_neighbor_offset = static_cast<uint32_t*>(p); break;
        case FieldId::GridNeighborCount:   v.grid_neighbor_count = static_cast<uint32_t*>(p); break;
        case FieldId::GridNeighborIdx:     v.grid_neighbor_idx = static_cast<uint32_t*>(p); break;
        case FieldId::GridSortScratch:     v.grid_sort_scratch = static_cast<uint8_t*>(p); break;
        case FieldId::RngState:            v.rng_state = static_cast<uint64_t*>(p); break;
        case FieldId::EnvStatus:           v.env_status = static_cast<uint32_t*>(p); break;
        case FieldId::ObsBuffer:           v.obs_buffer = static_cast<float*>(p); break;
        case FieldId::EnvTerrainType:      v.env_terrain_type = static_cast<uint32_t*>(p); break;
        case FieldId::EnvTerrainDifficulty: v.env_terrain_difficulty = static_cast<float*>(p); break;
        case FieldId::JointF:              v.joint_f = static_cast<float*>(p); break;
        case FieldId::JointLimitImpulse:   v.joint_limit_impulse = static_cast<float*>(p); break;
        case FieldId::ActuatorEffortRequested: v.actuator_effort_requested = static_cast<float*>(p); break;
        case FieldId::ActuatorEffort:      v.actuator_effort = static_cast<float*>(p); break;
        case FieldId::ActuatorSaturated:   v.actuator_saturated = static_cast<float*>(p); break;
        default: break;  // model-owned field id: not a DataView member.
    }
}

}  // namespace

phi::Status Data::Allocate(phi::BufferType* bt, const ModelCapacities& caps,
                           phi::DataView* out_view) {
    const phi::Status s = arena_.Allocate(bt, caps);
    if (s != phi::Status::Ok) {
        return s;
    }
    if (out_view != nullptr) {
        FillView(out_view);
    }
    return phi::Status::Ok;
}

void Data::FillView(phi::DataView* out_view) const {
    if (out_view == nullptr) {
        return;
    }
    *out_view = phi::DataView{};
    for (const Arena::Segment& seg : arena_.Segments()) {
        BindDataPointer(*out_view, seg.field, arena_.Ptr(seg.field));
    }
}

uint64_t Data::PersistentByteSize() const {
    uint64_t span = 0;
    for (const Arena::Segment& segment : arena_.Segments()) {
        if (segment.arena == 0u) {
            span = std::max(span, segment.offset + segment.bytes);
        }
    }
    return span == 0u ? 256u : span;
}

bool Data::DownloadPersistent(std::vector<uint8_t>* out) const {
    phi::Buffer* buffer = arena_.PersistentBuffer();
    if (buffer == nullptr || out == nullptr) {
        return false;
    }
    out->assign(PersistentByteSize(), 0u);
    phi::BufferDownload(buffer, out->data(), 0, out->size());
    return true;
}

bool Data::UploadPersistent(const std::vector<uint8_t>& bytes) const {
    phi::Buffer* buffer = arena_.PersistentBuffer();
    if (buffer == nullptr || bytes.size() != PersistentByteSize()) {
        return false;
    }
    phi::BufferUpload(buffer, bytes.data(), 0, bytes.size());
    return true;
}

bool Data::Snapshot() {
    return DownloadPersistent(&snapshot_persistent_);
}

bool Data::Restore() {
    return UploadPersistent(snapshot_persistent_);
}

} // namespace nuka::nk
