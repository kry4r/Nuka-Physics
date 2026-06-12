// ---------------------------------------------------------------------------
// nk::Pipeline implementation (plan §3.2 fixed order).
// ---------------------------------------------------------------------------

#include "nk/pipeline/pipeline.hpp"

#include "nk/model/model.hpp"

namespace nuka::nk {

void Pipeline::Build(const Model& model, const SolverConfig& cfg,
                     phi::Device* device) {
    calls_.clear();
    const ModelCapacities& cap = model.capacities;

    const bool has_articulation = cap.dofs_per_env > 0 || cap.links_per_env > 0;
    const bool has_bodies       = cap.bodies_per_env > 0;
    const bool has_particles    = cap.particles_per_env > 0;
    const bool has_contacts     = cap.max_rows_per_env > 0;
    const bool has_collidables  = has_bodies || cap.links_per_env > 0;
    const bool is_union = model.contact_family == ContactFamily::UnionCsr;
    const bool is_pair_driven = model.contact_family == ContactFamily::PairDriven;
    const uint32_t family = is_union ? phi::kContactFamilyUnionCsr
                          : is_pair_driven ? phi::kContactFamilyPairDriven
                                           : phi::kContactFamilyFusedFoot;

    // M3b launch-geometry counts (the views are pure pointer aggregates, so
    // every op carries its counts in the params POD).
    const uint32_t env_count        = cap.env_count;
    const uint32_t base_link_count  = cap.links_per_env;
    const uint32_t total_link_count = cap.links_per_env * cap.env_count;
    const uint32_t articulation_cnt = has_articulation ? cap.env_count : 0u;
    const uint32_t slot_count       = cap.max_contacts_per_env * cap.env_count;
    const uint32_t max_dof          = cap.dofs_per_env;

    auto add = [&](phi::NkOp op, const void* params) {
        // Capability query (§3.1): with a device, emit only ops the backend
        // implements (unimplemented = a later milestone's system).
        if (device != nullptr && !phi::DeviceSupportsOp(device, op)) {
            return;
        }
        calls_.push_back(phi::OpCall{op, params});
    };

    // §3.2 FIXED order:
    // ApplyDrives -> AbaForward -> IntegrateVelocity (+ParticlePredict) ->
    // FkWorldPoses -> BuildAabbs -> LbvhBuild -> LbvhQueryPairs (+ParticleGridBuild)
    // -> NarrowphasePrimitives -> NarrowphaseSdf -> ContactTangentBasis ->
    // CrbaComputeM -> CrbaFactorM -> AssembleRows -> SolveRowsBlockIsland
    // (+XpbdProject / PbfDensityLambda inline) -> IntegratePosition
    // (+ParticleFinalize) -> ReadoutContactWrench.

    if (has_articulation) {
        p_apply_drives_.dt = cfg.dt;
        p_apply_drives_.total_link_count = total_link_count;
        p_apply_drives_.defer_velocity_damping = cfg.defer_velocity_damping;
        p_apply_drives_.mode = model.drive_mode;
        add(phi::NkOp::ApplyDrives, &p_apply_drives_);

        p_aba_.gravity[0] = cfg.gravity[0];
        p_aba_.gravity[1] = cfg.gravity[1];
        p_aba_.gravity[2] = cfg.gravity[2];
        p_aba_.articulation_count = articulation_cnt;
        p_aba_.total_link_count = total_link_count;
        add(phi::NkOp::AbaForward, &p_aba_);
    }

    if (has_articulation || has_bodies) {
        p_int_vel_.dt = cfg.dt;
        p_int_vel_.gravity_z = cfg.gravity[2];
        p_int_vel_.total_link_count = total_link_count;
        p_int_vel_.articulation_count = articulation_cnt;
        // M4: the rigid-body gravity kick rides IntegrateVelocity (the union
        // world kicks the cup exactly once per step, before the contact solve).
        p_int_vel_.total_body_count = cap.bodies_per_env * env_count;
        add(phi::NkOp::IntegrateVelocity, &p_int_vel_);
    }

    // M6 particle launch geometry + mode (resolved once from the Model).
    const uint32_t particle_count = cap.particles_per_env * env_count;
    const uint32_t dist_count = cap.dist_cons_per_env * env_count;
    const uint32_t bend_count = cap.bend_cons_per_env * env_count;
    const uint32_t vol_count  = cap.vol_cons_per_env * env_count;
    const Model::ModelParticles& mp = model.particles;
    const uint32_t particle_mode =
        mp.mode == Model::ParticleMode::Xpbd ? phi::kParticleModeXpbd
        : mp.mode == Model::ParticleMode::Pbf ? phi::kParticleModePbf
        : mp.mode == Model::ParticleMode::Coupled ? phi::kParticleModeCoupled
                                                  : phi::kParticleModeNone;
    const uint32_t coupled_internal =
        mp.coupled_internal == Model::CoupledInternal::Xpbd ? phi::kCoupledInternalXpbd
        : mp.coupled_internal == Model::CoupledInternal::Pbf ? phi::kCoupledInternalPbf
                                                             : phi::kCoupledInternalNone;
    // The PBF density-projection runs when the body IS a fluid: either standalone
    // PBF mode, or coupled mode with the Pbf internal sub-type.
    const bool runs_pbf = (particle_mode == phi::kParticleModePbf) ||
        (particle_mode == phi::kParticleModeCoupled &&
         coupled_internal == phi::kCoupledInternalPbf);

    if (has_particles) {
        p_part_predict_.dt = cfg.dt;
        p_part_predict_.gravity[0] = cfg.gravity[0];
        p_part_predict_.gravity[1] = cfg.gravity[1];
        p_part_predict_.gravity[2] = cfg.gravity[2];
        p_part_predict_.mode = particle_mode;
        p_part_predict_.particle_count = particle_count;
        p_part_predict_.coupled_internal = coupled_internal;
        add(phi::NkOp::ParticlePredict, &p_part_predict_);
    }

    if (has_articulation) {
        p_fk_.articulation_count = articulation_cnt;
        p_fk_.total_link_count = total_link_count;
        add(phi::NkOp::FkWorldPoses, &p_fk_);
    }

    if (has_collidables) {
        // M5 broadphase (BuildAabbs/LbvhBuild/LbvhQueryPairs). These ops do real
        // work ONLY for the PairDriven family (the union slot-template and fused-
        // foot families run their own detection and never read candidate_pairs);
        // they EARLY-EXIT for is_union / fused so the gate-pinned union
        // StepPlanned graph stays bit-identical to M4. The family + per-env body
        // geometry travel in the params (the views are pure pointer aggregates).
        const uint32_t bodies_per_env = cap.bodies_per_env;
        p_aabbs_.margin = cfg.contact_margin;
        p_aabbs_.family = family;
        p_aabbs_.env_count = env_count;
        p_aabbs_.bodies_per_env = bodies_per_env;
        add(phi::NkOp::BuildAabbs, &p_aabbs_);

        p_lbvh_build_.family = family;
        p_lbvh_build_.env_count = env_count;
        p_lbvh_build_.bodies_per_env = bodies_per_env;
        add(phi::NkOp::LbvhBuild, &p_lbvh_build_);

        p_lbvh_query_.max_pairs = cfg.max_pairs;
        p_lbvh_query_.family = family;
        p_lbvh_query_.env_count = env_count;
        p_lbvh_query_.bodies_per_env = bodies_per_env;
        p_lbvh_query_.max_contacts_per_env = cap.max_contacts_per_env;
        p_lbvh_query_.filter_cross_env = model.filter_cross_env ? 1u : 0u;
        p_lbvh_query_.excluded_count =
            static_cast<uint32_t>(model.excluded_pairs.size());
        add(phi::NkOp::LbvhQueryPairs, &p_lbvh_query_);
    }

    if (has_particles) {
        // M6: resolve the PBF uniform-grid params from the cooked Model. For a
        // PBF fluid the cell_size/query_radius == the support radius (the M5
        // grid precondition cell >= query); the grid_min/dims come from the
        // cooked domain. An XPBD-only scene leaves cell_size 0 -> the build op
        // early-exits (no PBF neighbors needed). The grid is rebuilt every step
        // over the PREDICTED positions (pbf_predicted_pos) — but ParticleGridBuild
        // reads particle_pos; M6 routes it at the PBF predicted positions via the
        // op reading particle_pos AFTER ParticlePredict... see the note below.
        p_grid_.cell_size = runs_pbf ? mp.cell_size : 0.0f;
        p_grid_.query_radius = mp.query_radius;
        p_grid_.particle_count = particle_count;
        for (int k = 0; k < 3; ++k) {
            p_grid_.grid_min[k] = (&mp.grid_min.x)[k];
            p_grid_.grid_dims[k] = mp.grid_dims[k];
        }
        // PBF builds the grid over the predicted positions (ParticlePredict runs
        // just above, seeding pbf_predicted_pos).
        p_grid_.pos_source = runs_pbf ? phi::kGridPosSourcePbfPredicted
                                      : phi::kGridPosSourceParticlePos;
        add(phi::NkOp::ParticleGridBuild, &p_grid_);
    }

    if (has_collidables) {
        p_np_prim_.contact_margin = cfg.contact_margin;
        p_np_prim_.max_contacts_per_pair = 4;
        p_np_prim_.ground_height = model.ground_height;
        p_np_prim_.foot_count = static_cast<uint32_t>(model.feet.size());
        p_np_prim_.env_count = env_count;
        p_np_prim_.base_link_count = base_link_count;
        p_np_prim_.family = family;
        p_np_prim_.union_slot_count =
            static_cast<uint32_t>(model.union_slots.size());
        p_np_prim_.bodies_per_env = cap.bodies_per_env;
        p_np_prim_.hull_vert_count =
            static_cast<uint32_t>(model.hull_verts.size() / 3u);
        p_np_prim_.particles_per_env = cap.particles_per_env;  // M6 coupling slots.
        add(phi::NkOp::NarrowphasePrimitives, &p_np_prim_);

        p_np_sdf_.contact_margin = cfg.contact_margin;
        p_np_sdf_.max_contacts_per_pair = 4;
        p_np_sdf_.family = family;          // PairDriven => sample; else no-op.
        p_np_sdf_.env_count = env_count;
        p_np_sdf_.bodies_per_env = cap.bodies_per_env;
        p_np_sdf_.max_contacts_per_env = cap.max_contacts_per_env;
        add(phi::NkOp::NarrowphaseSdf, &p_np_sdf_);

        // ContactTangentBasis: fused-family only (the union family's tangent
        // spokes are emitted inside AssembleRows, the EmitCompliantContactRows
        // per-manifold basis).
        if (!is_union) {
            p_tangent_.slot_count = slot_count;
            add(phi::NkOp::ContactTangentBasis, &p_tangent_);
        }
    }

    if (has_articulation) {
        p_crba_m_.dt = cfg.dt;
        p_crba_m_.max_dof = max_dof;
        p_crba_m_.articulation_count = articulation_cnt;
        p_crba_m_.total_link_count = total_link_count;
        p_crba_m_.fold_drive_damping = cfg.fold_drive_damping;
        add(phi::NkOp::CrbaComputeM, &p_crba_m_);
        p_crba_factor_.max_dof = max_dof;
        p_crba_factor_.articulation_count = articulation_cnt;
        add(phi::NkOp::CrbaFactorM, &p_crba_factor_);
    }

    if (has_contacts) {
        p_assemble_.dt = cfg.dt;
        p_assemble_.slot_count = slot_count;
        p_assemble_.max_dof = max_dof;
        p_assemble_.env_count = env_count;
        p_assemble_.articulation_count = articulation_cnt;
        p_assemble_.total_link_count = total_link_count;
        p_assemble_.family = family;
        p_assemble_.union_slot_count =
            static_cast<uint32_t>(model.union_slots.size());
        p_assemble_.rows_per_env = cap.max_rows_per_env;
        p_assemble_.bodies_per_env = cap.bodies_per_env;
        p_assemble_.base_link_count = base_link_count;
        for (int k = 0; k < 2; ++k) p_assemble_.solref[k] = model.union_solref[k];
        for (int k = 0; k < 5; ++k) p_assemble_.solimp[k] = model.union_solimp[k];
        p_assemble_.particles_per_env = cap.particles_per_env;  // M6 coupling.
        add(phi::NkOp::AssembleRows, &p_assemble_);
    }

    if (has_particles) {
        // XpbdProject: the distance/bend/volume sweeps (XPBD soft). Inert for a
        // PBF-only scene (all constraint counts 0). iters from the cooked Model
        // (NOT cfg.vel_iters — the XPBD GS sweep count is a soft-body property).
        p_xpbd_.dt = cfg.dt;
        p_xpbd_.iters = mp.xpbd_iters == 0u ? 1u : mp.xpbd_iters;
        p_xpbd_.dist_con_count = dist_count;
        p_xpbd_.bend_con_count = bend_count;
        p_xpbd_.vol_con_count  = vol_count;
        add(phi::NkOp::XpbdProject, &p_xpbd_);

        // PbfDensityLambda / PbfApplyDelta: the PBF density-projection. Inert for
        // an XPBD-only scene (rest_density / support_radius 0). The two ops split
        // the legacy [density,lambda,delta,apply]xN loop (density-lambda + the in-
        // loop applies for iters 0..N-2 here; the final apply in PbfApplyDelta).
        p_pbf_density_.rest_density   = mp.pbf_rest_density;
        p_pbf_density_.relaxation     = mp.pbf_cfm_epsilon;
        p_pbf_density_.support_radius = mp.pbf_support_radius;
        p_pbf_density_.particle_mass  = mp.pbf_particle_mass;
        p_pbf_density_.particle_count = particle_count;
        p_pbf_density_.iters          = mp.pbf_iters;
        p_pbf_density_.clamp_overdensity = mp.pbf_clamp_overdensity ? 1u : 0u;
        p_pbf_density_.dt             = cfg.dt;
        p_pbf_density_.boundary_enabled = mp.boundary_enabled ? 1u : 0u;
        p_pbf_density_.floor_z        = mp.floor_z;
        add(phi::NkOp::PbfDensityLambda, &p_pbf_density_);

        p_pbf_apply_.support_radius  = mp.pbf_support_radius;
        p_pbf_apply_.particle_mass   = mp.pbf_particle_mass;
        p_pbf_apply_.particle_count  = particle_count;
        p_pbf_apply_.boundary_enabled = mp.boundary_enabled ? 1u : 0u;
        p_pbf_apply_.floor_z         = mp.floor_z;
        add(phi::NkOp::PbfApplyDelta, &p_pbf_apply_);
    }

    if (has_contacts) {
        // M4: the spec-fixed SolveRowsBlockIslandParams takes over the slot
        // (the M3b transitional SolveArticulatedParams routing is deleted).
        // Semantic triplet first; appended launch geometry + the per-family
        // Model-derived solver constants (see op_schema.hpp).
        p_solve_.dt = cfg.dt;
        p_solve_.vel_iters = cfg.vel_iters;
        p_solve_.pos_iters = cfg.pos_iters;
        p_solve_.family = family;
        p_solve_.total_islands = model.schedule_island_count;
        p_solve_.max_dof = max_dof;
        p_solve_.env_count = env_count;
        p_solve_.articulation_count = articulation_cnt;
        p_solve_.rows_per_env = cap.max_rows_per_env;
        p_solve_.base_link_count = base_link_count;
        p_solve_.total_body_count = cap.bodies_per_env * env_count;
        p_solve_.friction_coefficient = model.friction_coefficient;
        p_solve_.baumgarte_max_velocity = model.baumgarte_max_velocity;
        p_solve_.apply_implicit_damping = cfg.fold_drive_damping;
        add(phi::NkOp::SolveRowsBlockIsland, &p_solve_);
    }

    if (has_articulation || has_bodies) {
        p_int_pos_.dt = cfg.dt;
        p_int_pos_.total_link_count = total_link_count;
        p_int_pos_.articulation_count = articulation_cnt;
        p_int_pos_.total_body_count = cap.bodies_per_env * env_count;
        add(phi::NkOp::IntegratePosition, &p_int_pos_);
    }

    if (has_particles) {
        p_part_finalize_.dt = cfg.dt;
        p_part_finalize_.mode = particle_mode;
        p_part_finalize_.particle_count = particle_count;
        p_part_finalize_.coupled_internal = coupled_internal;
        // PBF post-finalize polish (gated inert when the coefficient is 0).
        p_part_finalize_.support_radius = mp.pbf_support_radius;
        p_part_finalize_.particle_mass  = mp.pbf_particle_mass;
        p_part_finalize_.xsph_viscosity_c = mp.pbf_xsph_viscosity;
        p_part_finalize_.surface_tension_gamma = mp.pbf_surface_tension;
        p_part_finalize_.rest_density = mp.pbf_rest_density;
        add(phi::NkOp::ParticleFinalize, &p_part_finalize_);
    }

    // ReadoutContactWrench: fused-family only in M4 (its kernel reads the
    // fused contact_* stream; the union family's readout — wrench from the
    // urows lambdas — is the M5/M9 obs wiring, tests read the fields directly).
    if ((has_articulation || has_bodies) && !is_union) {
        p_readout_.dt = cfg.dt;
        p_readout_.env_count = env_count;
        p_readout_.base_link_count = base_link_count;
        p_readout_.max_contacts_per_env = cap.max_contacts_per_env;
        add(phi::NkOp::ReadoutContactWrench, &p_readout_);
    }
}

} // namespace nuka::nk
