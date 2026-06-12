// ---------------------------------------------------------------------------
// nk::Pipeline implementation (plan §3.2 fixed order).
// ---------------------------------------------------------------------------

#include "nk/pipeline/pipeline.hpp"

#include "nk/model/model.hpp"

namespace nuka::nk {

void Pipeline::Build(const Model& model, const SolverConfig& cfg) {
    calls_.clear();
    const ModelCapacities& cap = model.capacities;

    const bool has_articulation = cap.dofs_per_env > 0 || cap.links_per_env > 0;
    const bool has_bodies       = cap.bodies_per_env > 0;
    const bool has_particles    = cap.particles_per_env > 0;
    const bool has_contacts     = cap.max_rows_per_env > 0;
    const bool has_collidables  = has_bodies || cap.links_per_env > 0;

    auto add = [&](phi::NkOp op, const void* params) {
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
        add(phi::NkOp::ApplyDrives, &p_apply_drives_);

        p_aba_.gravity[0] = cfg.gravity[0];
        p_aba_.gravity[1] = cfg.gravity[1];
        p_aba_.gravity[2] = cfg.gravity[2];
        add(phi::NkOp::AbaForward, &p_aba_);

        p_int_vel_.dt = cfg.dt;
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
        add(phi::NkOp::NarrowphasePrimitives, &p_np_prim_);

        p_np_sdf_.contact_margin = cfg.contact_margin;
        p_np_sdf_.max_contacts_per_pair = 4;
        add(phi::NkOp::NarrowphaseSdf, &p_np_sdf_);

        add(phi::NkOp::ContactTangentBasis, &p_tangent_);
    }

    if (has_articulation) {
        add(phi::NkOp::CrbaComputeM, &p_crba_m_);
        add(phi::NkOp::CrbaFactorM, &p_crba_factor_);
    }

    if (has_contacts) {
        p_assemble_.dt = cfg.dt;
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
        p_solve_.dt = cfg.dt;
        p_solve_.vel_iters = cfg.vel_iters;
        p_solve_.pos_iters = cfg.pos_iters;
        add(phi::NkOp::SolveRowsBlockIsland, &p_solve_);
    }

    if (has_articulation) {
        p_int_pos_.dt = cfg.dt;
        add(phi::NkOp::IntegratePosition, &p_int_pos_);
    }

    if (has_particles) {
        p_part_finalize_.dt = cfg.dt;
        add(phi::NkOp::ParticleFinalize, &p_part_finalize_);
    }

    if (has_articulation || has_bodies) {
        add(phi::NkOp::ReadoutContactWrench, &p_readout_);
    }
}

} // namespace nuka::nk
