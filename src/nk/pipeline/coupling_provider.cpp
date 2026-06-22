// ---------------------------------------------------------------------------
// nk::CouplingProvider implementation — the body↔particle row provider.
// ---------------------------------------------------------------------------

#include "nk/pipeline/coupling_provider.hpp"

#include <algorithm>

#include "collision/cross_system_query.hpp"  // kBodyParticleContactSlotsPerParticle
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
        collision::gpu::kBodyParticleContactSlotsPerParticle;
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
    // Detect the fluid slice at its predicted position (the gravity-integrated
    // pos the density solve uses) for PBF/SoftFluid, consistent with the
    // particle grid's pos_source -- kills the one-step fluid<->body contact
    // lag. The soft slice + pure-Xpbd/Coupled keep particle_pos (n_soft=0 for
    // pure Pbf routes every particle; the SoftFluid split routes [n_soft, P)).
    p_np_body_particle.fluid_pos_source =
        (ctx.particle_mode == phi::kParticleModePbf ||
         ctx.particle_mode == phi::kParticleModeSoftFluid)
            ? 1u : 0u;
    p_np_body_particle.n_soft_particles = ctx.n_soft;
    // Warp-per-particle only pays off when a collider has a WIDE hull whose
    // SupportHull scan dominates; an analytic-only collider world (box/sphere/
    // plane walls) keeps thread-per-particle so 31 lanes don't idle. The
    // threshold is the cook-time max hull vcount -> a model property, not a
    // per-scene branch; both launch paths are byte-identical.
    uint32_t max_hull_vcount = 0u;
    for (const auto& sh : model.shape_table_rows)
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
    // Build-time gate: emit the umbrella ONLY for a cooked MPM medium, so the op
    // LIST of a non-MPM world is byte-identical to master (not merely inert).
    if (ctx.has_mpm == 0u || ctx.p_mpm_step == nullptr) return;
    const Model& model = *ctx.model;
    const Model::ModelParticles& mp = model.particles;
    phi::MpmStepParams& p = *ctx.p_mpm_step;
    p.particle_count = ctx.particle_count;
    p.particles_per_env = ctx.particles_per_env;
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
    ctx.Emit(phi::NkOp::MpmStep, &p);
}

void RowCouplingProvider::PostCouple(const CouplingBuildCtx& ctx) const {
    const Model& model = *ctx.model;
    phi::ParticleFinalizeParams& p_part_finalize = *ctx.p_part_finalize;
    phi::ParticleParticleContactParams& p_pp_contact = *ctx.p_pp_contact;

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
    // Carry the split-impulse pseudo velocity onto the final particle position
    // only when the position pass is active (the same predicate IntegratePosition
    // uses for bodies); else the pseudo pointer is null -> byte-identical.
    p_part_finalize.pos_pass = ctx.pos_pass;
    ctx.Emit(phi::NkOp::ParticleFinalize, &p_part_finalize);

    // Cross-system: the cross-system particle-particle contact co-step
    // (the op-ified cross-system particle co-step). Runs AFTER ParticleFinalize
    // (incl. the fluid-slice polish), correcting the committed union positions
    // (particle_pos) over the union grid CSR built this step. ONLY the SoftFluid
    // mode emits real work; the single-system Xpbd/Pbf paths carry the op only
    // as an inert no-op (mode-gated early-exit) so they stay byte-identical.
    p_pp_contact.contact_distance_d_min =
        ctx.particle_mode == phi::kParticleModeSoftFluid
            ? model.particles.pp_contact_d_min : 0.0f;
    p_pp_contact.compliance_alpha = model.particles.pp_contact_compliance;
    p_pp_contact.solver_iterations =
        model.particles.pp_contact_iters == 0u ? 1u : model.particles.pp_contact_iters;
    p_pp_contact.mode = ctx.particle_mode;
    p_pp_contact.particle_count = ctx.particle_count;
    p_pp_contact.n_soft_particles = ctx.n_soft;
    p_pp_contact.particles_per_env = ctx.particles_per_env;
    ctx.Emit(phi::NkOp::ParticleParticleContact, &p_pp_contact);
}

} // namespace nuka::nk
