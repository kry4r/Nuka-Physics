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
        add(phi::NkOp::ApplyDrives, &p_apply_drives_);

        p_aba_.gravity[0] = cfg.gravity[0];
        p_aba_.gravity[1] = cfg.gravity[1];
        p_aba_.gravity[2] = cfg.gravity[2];
        p_aba_.articulation_count = articulation_cnt;
        p_aba_.total_link_count = total_link_count;
        add(phi::NkOp::AbaForward, &p_aba_);

        p_int_vel_.dt = cfg.dt;
        p_int_vel_.gravity_z = cfg.gravity[2];
        p_int_vel_.total_link_count = total_link_count;
        p_int_vel_.articulation_count = articulation_cnt;
        add(phi::NkOp::IntegrateVelocity, &p_int_vel_);
    }

    if (has_particles) {
        p_part_predict_.dt = cfg.dt;
        p_part_predict_.gravity[0] = cfg.gravity[0];
        p_part_predict_.gravity[1] = cfg.gravity[1];
        p_part_predict_.gravity[2] = cfg.gravity[2];
        add(phi::NkOp::ParticlePredict, &p_part_predict_);
    }

    if (has_articulation) {
        p_fk_.articulation_count = articulation_cnt;
        p_fk_.total_link_count = total_link_count;
        add(phi::NkOp::FkWorldPoses, &p_fk_);
    }

    if (has_collidables) {
        p_aabbs_.margin = cfg.contact_margin;
        add(phi::NkOp::BuildAabbs, &p_aabbs_);
        add(phi::NkOp::LbvhBuild, &p_lbvh_build_);
        p_lbvh_query_.max_pairs = cfg.max_pairs;
        add(phi::NkOp::LbvhQueryPairs, &p_lbvh_query_);
    }

    if (has_particles) {
        p_grid_.cell_size = 0.0f;  // resolved from PBF params in M6.
        add(phi::NkOp::ParticleGridBuild, &p_grid_);
    }

    if (has_collidables) {
        p_np_prim_.contact_margin = cfg.contact_margin;
        p_np_prim_.max_contacts_per_pair = 4;
        p_np_prim_.ground_height = model.ground_height;
        p_np_prim_.foot_count = static_cast<uint32_t>(model.feet.size());
        p_np_prim_.env_count = env_count;
        p_np_prim_.base_link_count = base_link_count;
        add(phi::NkOp::NarrowphasePrimitives, &p_np_prim_);

        p_np_sdf_.contact_margin = cfg.contact_margin;
        p_np_sdf_.max_contacts_per_pair = 4;
        add(phi::NkOp::NarrowphaseSdf, &p_np_sdf_);

        p_tangent_.slot_count = slot_count;
        add(phi::NkOp::ContactTangentBasis, &p_tangent_);
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
        add(phi::NkOp::AssembleRows, &p_assemble_);
    }

    if (has_particles) {
        p_xpbd_.dt = cfg.dt;
        p_xpbd_.iters = cfg.vel_iters;
        add(phi::NkOp::XpbdProject, &p_xpbd_);

        p_pbf_density_.rest_density = 1.0f;
        p_pbf_density_.relaxation = 1.0e-6f;
        add(phi::NkOp::PbfDensityLambda, &p_pbf_density_);
        add(phi::NkOp::PbfApplyDelta, &p_pbf_apply_);
    }

    if (has_contacts) {
        // TRANSITIONAL (M3b, LOUD): the SolveRowsBlockIsland slot dispatches
        // the ported legacy fused articulated solver with ITS OWN params POD
        // (SolveArticulatedParams). M4's device-resident SolveRowsBlockIsland
        // restores p_solve_ (the spec-fixed POD) here.
        p_solve_.dt = cfg.dt;
        p_solve_.vel_iters = cfg.vel_iters;
        p_solve_.pos_iters = cfg.pos_iters;
        p_solve_articulated_.dt = cfg.dt;
        p_solve_articulated_.friction_coefficient = model.friction_coefficient;
        p_solve_articulated_.baumgarte_max_velocity = model.baumgarte_max_velocity;
        p_solve_articulated_.max_dof = max_dof;
        p_solve_articulated_.articulation_count = articulation_cnt;
        p_solve_articulated_.env_count = env_count;
        p_solve_articulated_.apply_implicit_damping = cfg.fold_drive_damping;
        add(phi::NkOp::SolveRowsBlockIsland, &p_solve_articulated_);
    }

    if (has_articulation) {
        p_int_pos_.dt = cfg.dt;
        p_int_pos_.total_link_count = total_link_count;
        p_int_pos_.articulation_count = articulation_cnt;
        add(phi::NkOp::IntegratePosition, &p_int_pos_);
    }

    if (has_particles) {
        p_part_finalize_.dt = cfg.dt;
        add(phi::NkOp::ParticleFinalize, &p_part_finalize_);
    }

    if (has_articulation || has_bodies) {
        p_readout_.dt = cfg.dt;
        p_readout_.env_count = env_count;
        p_readout_.base_link_count = base_link_count;
        p_readout_.max_contacts_per_env = cap.max_contacts_per_env;
        add(phi::NkOp::ReadoutContactWrench, &p_readout_);
    }
}

} // namespace nuka::nk
