// ---------------------------------------------------------------------------
// nk::CouplingProvider implementation — the body↔particle row provider.
// ---------------------------------------------------------------------------

#include "nk/pipeline/coupling_provider.hpp"

#include <algorithm>

#include "collision/contact_capacity.hpp"
#include "collision/shape_kind.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/pipeline.hpp"

namespace nuka::nk {

void CouplingBuildCtx::Emit(phi::NkOp op, const void* params) const {
    pipeline->AddOp(op, params, device);
}

void RowCouplingProvider::PreCouple(const CouplingBuildCtx& ctx) const {
    const Model& model = *ctx.model;
    phi::NarrowphaseBodyParticleParams& p_np_body_particle = *ctx.p_np_body_particle;

    // Body/artic <-> particle narrowphase. Runs AFTER the rigid narrowphase
    // (which filled the rigid candidate slots [0, pair_count) racily) and BEFORE
    // AssembleRows. It treats each particle as a SPHERE of its radius and writes
    // its body manifolds into a RESERVED contact-slot sub-range at the TOP of the
    // per-env block, deterministic relative to the rigid slots (the cross-stream
    // ordering guard). The particle collision radius is the cooked d_min/2 (the
    // same uniform radius the particle-particle co-step uses).
    const uint32_t cands_per_particle =
        collision::kBodyParticleContactSlotsPerParticle;
    // particle_base == rigid_cap by construction: particles take the top
    // [rigid_cap, total) range above the rigid [0, rigid_cap) sub-range.
    p_np_body_particle.family = ctx.family;
    p_np_body_particle.env_count = ctx.env_count;
    p_np_body_particle.bodies_per_env = ctx.bodies_per_env;
    p_np_body_particle.particles_per_env = ctx.particles_per_env;
    p_np_body_particle.slot_stride = ctx.max_contacts_per_env;
    p_np_body_particle.particle_slot_base = ctx.rigid_cap;
    p_np_body_particle.cands_per_particle = cands_per_particle;
    // A particle is a sphere of d_min/2 on the ONE path (the cooked uniform
    // contact radius); 0 leaves the op inert (no collision radius cooked).
    p_np_body_particle.particle_radius = 0.5f * model.particles.pp_contact_d_min;
    p_np_body_particle.contact_margin = ctx.contact_margin;
    // Every row-coupled particle uses the same projected geometry.
    p_np_body_particle.fluid_pos_source = 1u;
    p_np_body_particle.n_soft_particles = 0u;
    // Grid-owned particles do not emit body-particle manifolds.
    p_np_body_particle.particle_row_base = ctx.n_mpm;
    p_np_body_particle.sdf_grid_count = model.capacities.max_sdf_grids;
    p_np_body_particle.sdf_cell_total = model.capacities.max_sdf_cells;
    p_np_body_particle.mesh_geometry = {model.capacities.max_hull_verts,
        model.capacities.max_mesh_triangles, model.capacities.max_mesh_bvh_nodes};
    // Warp-per-particle only pays off when a collider has a WIDE hull whose
    // SupportHull scan dominates; an analytic-only collider world (box/sphere/
    // plane walls) keeps thread-per-particle so 31 lanes don't idle. The
    // threshold is the cook-time max hull vcount -> a model property, not a
    // per-scene branch; both launch paths are byte-identical.
    uint32_t max_hull_vcount = 0u;
    for (const auto& sh : model.shape_table_rows)
        if (sh.kind == collision::kShapeConvexHull)
            max_hull_vcount = std::max(max_hull_vcount, sh.hull_vert_count);
    constexpr uint32_t kWarpHullVcountThreshold = 256u;
    p_np_body_particle.warp_per_particle =
        (max_hull_vcount > kWarpHullVcountThreshold) ? 1u : 0u;
    // The heightfield descriptor (the SAME single cooked field the rigid
    // heightfield narrowphase wires) so a sphere particle walks the grid.
    if (!model.heightfields.empty()) {
        const nk::HeightfieldData& hfd = model.heightfields.front();
        p_np_body_particle.has_heightfield = 1u;
        p_np_body_particle.origin_x = hfd.origin.x;
        p_np_body_particle.origin_y = hfd.origin.y;
        p_np_body_particle.origin_z = hfd.origin.z;
        p_np_body_particle.cell_size = hfd.cell_size;
        p_np_body_particle.nrow = hfd.nrow;
        p_np_body_particle.ncol = hfd.ncol;
        p_np_body_particle.min_z = hfd.min_z;
        p_np_body_particle.max_z = hfd.max_z;
        p_np_body_particle.data_offset = hfd.data_offset;
    } else {
        p_np_body_particle.has_heightfield = 0u;
    }
    ctx.Emit(phi::NkOp::NarrowphaseBodyParticle, &p_np_body_particle);
}

void RowCouplingProvider::Couple(const CouplingBuildCtx&) const {
    // The row coupling rides the shared SolveRowsBlockIsland the builder emits;
    // there is no extra op. A grid-transfer provider emits its umbrella here.
}

void MpmCouplingProvider::Couple(const CouplingBuildCtx& ctx) const {
    // Only worlds containing grid-owned particles schedule MPM transfer.
    if (ctx.has_mpm == 0u || ctx.p_mpm_step == nullptr) return;
    const Model& model = *ctx.model;
    const Model::ModelParticles& mp = model.particles;
    phi::MpmStepParams& p = *ctx.p_mpm_step;
    p.particle_count = ctx.particle_count;
    p.particles_per_env = ctx.particles_per_env;
    // Pure and mixed MPM use the same explicit grid-owned particle range.
    p.mpm_particles_per_env = ctx.n_mpm;
    p.env_count = ctx.env_count;
    p.nodes_per_env = model.capacities.mpm_grid_nodes_per_env;
    p.grid_dims[0] = mp.mpm_grid_dims[0];
    p.grid_dims[1] = mp.mpm_grid_dims[1];
    p.grid_dims[2] = mp.mpm_grid_dims[2];
    p.grid_origin[0] = mp.mpm_grid_min.x;
    p.grid_origin[1] = mp.mpm_grid_min.y;
    p.grid_origin[2] = mp.mpm_grid_min.z;
    p.dx = mp.mpm_cell_size;
    p.dt = ctx.dt;
    p.mode = ctx.particle_mode;
    p.substeps = mp.mpm_substeps == 0u ? 1u : mp.mpm_substeps;
    p.material_count = model.capacities.mpm_material_count;
    for (int k = 0; k < 3; ++k) p.gravity[k] = ctx.gravity[k];
    // The static floor plane (z-up). Coulomb mu reuses the body<->soft friction.
    p.plane_n[0] = mp.mpm_floor_normal.x;
    p.plane_n[1] = mp.mpm_floor_normal.y;
    p.plane_n[2] = mp.mpm_floor_normal.z;
    p.plane_d = mp.mpm_floor_d;
    p.plane_mu = mp.mpm_floor_friction;
    // Collidable surfaces impose grid boundaries and route reaction to their owner.
    // Unsupported geometry is reported; articulation feedback follows the substeps.
    p.dynamic_body_bc = ctx.bodies_per_env > 0u ? 1u : 0u;
    p.bite_disable_dynamic_bc = mp.mpm_bite_disable_dynamic_bc ? 1u : 0u;
    p.bodies_per_env = ctx.bodies_per_env;
    p.sdf_grid_count = model.capacities.max_sdf_grids;
    p.sdf_cell_total = model.capacities.max_sdf_cells;
    p.mesh_geometry = {model.capacities.max_hull_verts,
        model.capacities.max_mesh_triangles, model.capacities.max_mesh_bvh_nodes};
    p.body_mu = mp.mpm_body_friction;
    p.body_band = mp.mpm_body_band > 0.0f ? mp.mpm_body_band : mp.mpm_cell_size;
    // Link reactions seed the shared solve through M^-1 J^T in qdot_flat.
    p.artic_count = ctx.articulation_count;
    p.max_dof = ctx.max_dof;
    p.base_link_count = ctx.base_link_count;
    p.artics_per_env = ctx.artics_per_env;
    ctx.Emit(phi::NkOp::MpmStep, &p);
}

void RowCouplingProvider::PostCouple(const CouplingBuildCtx& ctx) const {
    const Model& model = *ctx.model;
    phi::ParticleFinalizeParams& p_part_finalize = *ctx.p_part_finalize;

    p_part_finalize.dt = ctx.dt;
    p_part_finalize.mode = ctx.particle_mode;
    p_part_finalize.particle_count = ctx.particle_count;
    p_part_finalize.coupled_internal = ctx.coupled_internal;
    // PBF post-finalize polish (gated inert when the coefficient is 0).
    p_part_finalize.support_radius = model.particles.pbf_support_radius;
    p_part_finalize.particle_mass  = model.particles.pbf_particle_mass;
    p_part_finalize.xsph_viscosity_c = model.particles.pbf_xsph_viscosity;
    p_part_finalize.surface_tension_gamma = model.particles.pbf_surface_tension;
    p_part_finalize.rest_density = model.particles.pbf_rest_density;
    p_part_finalize.n_soft_particles = ctx.n_soft;
    p_part_finalize.particles_per_env = ctx.particles_per_env;
    p_part_finalize.n_mpm_particles = ctx.n_mpm;
    // Carry the split-impulse pseudo velocity onto the final particle position
    // only when the position pass is active (the same predicate IntegratePosition
    // uses for bodies); else the pseudo pointer is null -> byte-identical.
    p_part_finalize.pos_pass = ctx.pos_pass;
    ctx.Emit(phi::NkOp::ParticleFinalize, &p_part_finalize);

}

} // namespace nuka::nk
