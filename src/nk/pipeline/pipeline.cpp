// ---------------------------------------------------------------------------
// nk::Pipeline implementation (the design fixed order).
// ---------------------------------------------------------------------------

#include "nk/pipeline/pipeline.hpp"

#include <algorithm>
#include <cassert>

#include "collision/cross_system_query.hpp"  // kBodyParticleContactSlotsPerParticle
#include "constraint/contact_manifold.hpp"  // ContactManifold::kMaxPoints
#include "nk/model/model.hpp"

namespace nuka::nk {

// The per-pair manifold point capacity shared by every narrowphase op. Bound to
// the manifold struct so the param fields cannot drift from the solver layout.
inline constexpr uint8_t kMaxContactsPerPair =
    static_cast<uint8_t>(constraint::ContactManifold::kMaxPoints);

void Pipeline::AddOp(phi::NkOp op, const void* params, phi::Device* device) {
    // Capability query (the spec): with a device, emit only ops the backend
    // implements (unimplemented = a later system).
    if (device != nullptr && !phi::DeviceSupportsOp(device, op)) {
        return;
    }
    calls_.push_back(phi::OpCall{op, params});
}

void Pipeline::Build(const Model& model, const SolverConfig& cfg,
                     phi::Device* device) {
    calls_.clear();
    const ModelCapacities& cap = model.capacities;

    const bool has_articulation = cap.dofs_per_env > 0 || cap.links_per_env > 0;
    const bool has_bodies       = cap.bodies_per_env > 0;
    const bool has_particles    = cap.particles_per_env > 0;
    const bool has_contacts     = cap.max_rows_per_env > 0;
    // gate the contact pipeline (SyncLinkBodyPose / broadphase / narrowphase)
    // on actual contact capacity. For every cooked-with-contacts world this equals
    // the old structural test -- the cook sizes max_contacts_per_env ==
    // (bodies+links)*4 > 0 whenever a body/link exists, so op selection is
    // byte-identical. A contacts-OFF cook (CookToModelOptions::enable_contacts ==
    // false) zeroes max_contacts_per_env, so the broadphase + narrowphase + the
    // FK->body_pose sync are all skipped -> the world runs pure articulation+rigid
    // dynamics and StepPlanned captures cleanly (no thrust LBVH mid-graph).
    const bool has_collidables  =
        (has_bodies || cap.links_per_env > 0) && cap.max_contacts_per_env > 0;
    // deleted the FUSED runtime path; deleted the UnionCsr path. There
    // is now ONE general contact path: PairDriven. Every cooked model uses it.
    const uint32_t family = phi::kContactFamilyPairDriven;

    // launch-geometry counts (the views are pure pointer aggregates, so
    // every op carries its counts in the params POD).
    const uint32_t env_count        = cap.env_count;
    const uint32_t base_link_count  = cap.links_per_env;
    const uint32_t total_link_count = cap.links_per_env * cap.env_count;
    // Multi-articulation co-residence: articulation_count == K * env_count (K ==
    // articulations_per_env, the number of co-resident articulations per env). Every
    // forward kernel already launches dim3(articulation_count) and indexes
    // articulation_link_offset[articulation] / base_pose[articulation] /
    // m[articulation*max_dof^2], so this single scalar generalizes the launch
    // geometry. At K==1 (a single-articulation scene) this equals env_count, so the
    // launch is byte-identical. max_dof stays cap.dofs_per_env (per-articulation DOF).
    const uint32_t articulation_cnt =
        has_articulation ? cap.articulations_per_env * cap.env_count : 0u;
    const uint32_t slot_count       = cap.max_contacts_per_env * cap.env_count;
    const uint32_t max_dof          = cap.dofs_per_env;
    // rigid_cap recovers the pre-particle budget the cook grew from, so rigid keeps
    // [0, rigid_cap) and particles the top range; particle-free -> rigid_cap == total.
    const uint32_t particle_reserve =
        has_particles ? cap.particles_per_env *
                            collision::gpu::kBodyParticleContactSlotsPerParticle
                      : 0u;
    const uint32_t rigid_cap = cap.max_contacts_per_env - particle_reserve;
    // The per-ARTICULATION contact-slot stride the contact detection/assembly/solve
    // share, DATA-DRIVEN from the cooked geometry: the cook sizes max_contacts_per_env
    // from the collidable count, so the stride is the per-articulation quotient
    // max_contacts_per_env / K (bounded by kMaxContactsPerArtic). At K<=1 this is
    // kMaxFootContactsPerEnv (4) -> byte-identical layout; at K>1 it is the grown
    // per-articulation stride (room for end-effector + body + inter-articulation rows).
    const uint32_t artics_per_env =
        cap.articulations_per_env == 0u ? 1u : cap.articulations_per_env;
    // Degenerate-cook fallback (max_contacts_per_env < K): the single-articulation
    // end-effector capacity == kMaxFootContactsPerEnv (defined device-side; named
    // host-locally here so pipeline.cpp stays a pure-C++ TU).
    constexpr uint32_t kFallbackContactsPerArtic = 4u;  // == kMaxFootContactsPerEnv
    const uint32_t contact_slots_per_artic =
        (artics_per_env > 0u && cap.max_contacts_per_env >= artics_per_env)
            ? (cap.max_contacts_per_env / artics_per_env)
            : kFallbackContactsPerArtic;

    auto add = [&](phi::NkOp op, const void* params) {
        AddOp(op, params, device);
    };

    // the spec FIXED order:
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
        // The rigid-body arm: the rigid-body gravity kick rides IntegrateVelocity (the union
        // world kicks the cup exactly once per step, before the contact solve).
        p_int_vel_.total_body_count = cap.bodies_per_env * env_count;
        add(phi::NkOp::IntegrateVelocity, &p_int_vel_);
    }

    // particle launch geometry + mode (resolved once from the Model).
    const uint32_t particle_count = cap.particles_per_env * env_count;
    const uint32_t dist_count = cap.dist_cons_per_env * env_count;
    const uint32_t bend_count = cap.bend_cons_per_env * env_count;
    const uint32_t vol_count  = cap.vol_cons_per_env * env_count;
    const uint32_t sm_cluster_count = cap.shape_match_slots_per_env * env_count;
    const Model::ModelParticles& mp = model.particles;
    const uint32_t particle_mode =
        mp.mode == Model::ParticleMode::Xpbd ? phi::kParticleModeXpbd
        : mp.mode == Model::ParticleMode::Pbf ? phi::kParticleModePbf
        : mp.mode == Model::ParticleMode::Coupled ? phi::kParticleModeCoupled
        : mp.mode == Model::ParticleMode::SoftFluid ? phi::kParticleModeSoftFluid
        : mp.mode == Model::ParticleMode::Mpm ? phi::kParticleModeMpm
                                                  : phi::kParticleModeNone;
    // SoftFluid: the per-env [soft | fluid] split + stride (0 elsewhere).
    const uint32_t n_soft = mp.mode == Model::ParticleMode::SoftFluid
                                ? mp.n_soft_particles : 0u;
    const uint32_t per_env_particles = cap.particles_per_env;
    // Per-env soft-particle count selecting which per-system mu a particle side
    // reads: SoftFluid the explicit split, Xpbd all-soft, Pbf all-fluid, Coupled by type.
    const uint32_t friction_n_soft =
        mp.mode == Model::ParticleMode::SoftFluid ? mp.n_soft_particles
        : mp.mode == Model::ParticleMode::Xpbd ? per_env_particles
        : mp.mode == Model::ParticleMode::Pbf ? 0u
        : mp.mode == Model::ParticleMode::Coupled
              ? (mp.coupled_internal == Model::CoupledInternal::Pbf ? 0u
                                                                    : per_env_particles)
        : 0u;
    const uint32_t coupled_internal =
        mp.coupled_internal == Model::CoupledInternal::Xpbd ? phi::kCoupledInternalXpbd
        : mp.coupled_internal == Model::CoupledInternal::Pbf ? phi::kCoupledInternalPbf
                                                             : phi::kCoupledInternalNone;
    // The PBF density-projection runs when the body IS a fluid: standalone PBF
    // mode, coupled mode with the Pbf internal sub-type, OR the SoftFluid
    // co-residence mode (which always carries a PBF fluid slice).
    const bool runs_pbf = (particle_mode == phi::kParticleModePbf) ||
        (particle_mode == phi::kParticleModeSoftFluid) ||
        (particle_mode == phi::kParticleModeCoupled &&
         coupled_internal == phi::kCoupledInternalPbf);

    // The build-time coupling context: the row provider's PreCouple/PostCouple
    // fill the Pipeline-owned PODs from these resolved scalars at their original
    // insertion points; Couple is wired (inert for the row path) before the solve.
    CouplingBuildCtx coupling_ctx;
    coupling_ctx.model = &model;
    coupling_ctx.device = device;
    coupling_ctx.pipeline = this;
    coupling_ctx.family = family;
    coupling_ctx.env_count = env_count;
    coupling_ctx.bodies_per_env = cap.bodies_per_env;
    coupling_ctx.particles_per_env = cap.particles_per_env;
    coupling_ctx.max_contacts_per_env = cap.max_contacts_per_env;
    coupling_ctx.rigid_cap = rigid_cap;
    coupling_ctx.particle_mode = particle_mode;
    coupling_ctx.coupled_internal = coupled_internal;
    coupling_ctx.particle_count = particle_count;
    coupling_ctx.n_soft = n_soft;
    coupling_ctx.dt = cfg.dt;
    coupling_ctx.contact_margin = cfg.contact_margin;
    coupling_ctx.pos_pass =
        (has_contacts && family == phi::kContactFamilyPairDriven &&
         cfg.pos_iters > 0u) ? 1u : 0u;
    coupling_ctx.p_np_body_particle = &p_np_body_particle_;
    coupling_ctx.p_part_finalize = &p_part_finalize_;
    coupling_ctx.p_pp_contact = &p_pp_contact_;

    if (has_particles) {
        p_part_predict_.dt = cfg.dt;
        p_part_predict_.gravity[0] = cfg.gravity[0];
        p_part_predict_.gravity[1] = cfg.gravity[1];
        p_part_predict_.gravity[2] = cfg.gravity[2];
        p_part_predict_.mode = particle_mode;
        p_part_predict_.particle_count = particle_count;
        p_part_predict_.coupled_internal = coupled_internal;
        p_part_predict_.n_soft_particles = n_soft;
        p_part_predict_.particles_per_env = per_env_particles;
        add(phi::NkOp::ParticlePredict, &p_part_predict_);
    }

    if (has_articulation) {
        p_fk_.articulation_count = articulation_cnt;
        p_fk_.total_link_count = total_link_count;
        add(phi::NkOp::FkWorldPoses, &p_fk_);
    }

    if (has_collidables) {
        // General contact pipeline (B2): SyncLinkBodyPose. Runs AFTER
        // FkWorldPoses (the link_pose it reads) and BEFORE BuildAabbs (the
        // body_pose it writes, which the AABB build consumes) so articulation
        // links enter the LBVH. PairDriven-family-gated -> a no-op for the
        // UnionCsr family (the captured graph still ENQUEUES it, but the op
        // early-exits and writes nothing), so the union goldens are byte-identical.
        p_sync_body_pose_.family = family;
        p_sync_body_pose_.env_count = env_count;
        p_sync_body_pose_.links_per_env = cap.links_per_env;  // PER-ENV stride
        p_sync_body_pose_.bodies_per_env = cap.bodies_per_env;
        add(phi::NkOp::SyncLinkBodyPose, &p_sync_body_pose_);

        // broadphase (BuildAabbs/LbvhBuild/LbvhQueryPairs). These ops drive
        // the ONE general PairDriven contact path. The family + per-env body
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
        p_lbvh_query_.rigid_slot_cap = rigid_cap;  // body<->body fills only [0, rigid_cap).
        p_lbvh_query_.filter_cross_env = model.filter_cross_env ? 1u : 0u;
        p_lbvh_query_.excluded_count =
            static_cast<uint32_t>(model.excluded_pairs.size());
        add(phi::NkOp::LbvhQueryPairs, &p_lbvh_query_);
    }

    if (has_particles) {
        // resolve the PBF uniform-grid params from the cooked Model. For a
        // PBF fluid the cell_size/query_radius == the support radius (the
        // grid precondition cell >= query); the grid_min/dims come from the
        // cooked domain. An XPBD-only scene leaves cell_size 0 -> the build op
        // early-exits (no PBF neighbors needed). The grid is rebuilt every step
        // over the PREDICTED positions (pbf_predicted_pos) — but ParticleGridBuild
        // reads particle_pos; the pipeline routes it at the PBF predicted positions via the
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
        // Env-private grids: per-env cell-key offsets + the cooked cell
        // capacity (the grid_cell_start/end arena sizing the op guards).
        p_grid_.env_count = env_count;
        p_grid_.particles_per_env = cap.particles_per_env;
        p_grid_.cells_capacity = cap.max_grid_cells;
        add(phi::NkOp::ParticleGridBuild, &p_grid_);
    }

    if (has_collidables) {
        p_np_prim_.contact_margin = cfg.contact_margin;
        p_np_prim_.max_contacts_per_pair = kMaxContactsPerPair;
        p_np_prim_.ground_height = model.ground_height;
        p_np_prim_.foot_count = static_cast<uint32_t>(model.feet.size());
        p_np_prim_.env_count = env_count;
        p_np_prim_.base_link_count = base_link_count;
        p_np_prim_.family = family;
        // The PairDriven narrowphase uses union_slot_count as its CANDIDATE slot
        // STRIDE (slots per env in candidate_pairs / ucontact_*). It MUST equal
        // the broadphase's (EnvQueryPairsKernel uses max_contacts_per_env), so the
        // candidate slot index lines up across broadphase -> narrowphase -> the
        // unified buffer. (Field name retained for the shared params layout.)
        p_np_prim_.union_slot_count = cap.max_contacts_per_env;
        p_np_prim_.rigid_slot_cap = rigid_cap;  // body<->body fills only [0, rigid_cap).
        p_np_prim_.bodies_per_env = cap.bodies_per_env;
        p_np_prim_.hull_vert_count =
            static_cast<uint32_t>(model.hull_verts.size() / 3u);
        p_np_prim_.particles_per_env = cap.particles_per_env;  // coupling slots.
        // Co-resident end-effector contacts: route each into its OWNING
        // articulation's slot block when K>1 (byte-identical env-keyed at K<=1; the
        // narrowphase reads link_to_articulation only on the K>1 branch). The
        // per-articulation slot stride is the shared data-driven value (4 at K<=1).
        p_np_prim_.articulations_per_env = cap.articulations_per_env;
        p_np_prim_.max_foot_contacts = contact_slots_per_artic;
        add(phi::NkOp::NarrowphasePrimitives, &p_np_prim_);

        // (Multi-body collision — including artic-link<->artic-link and
        // body-vs-terrain — runs ENTIRELY on the GENERAL PairDriven path: LBVH
        // broadphase -> cvx GJK/EPA narrowphase + the general heightfield
        // narrowphase below -> mixed-island solve. There is no special-cased
        // op; a robot body is just a physics body on the ONE path.)

        // General contact pipeline (H3): the per-cell heightfield
        // midphase. Runs AFTER NarrowphasePrimitives (which left the (convex,
        // heightfield) candidate slots empty — kKindHeightfield hits DispatchPair's
        // default case) and BEFORE AssembleRows (which consumes the unified
        // ucontact_* buffer). For each (convex, heightfield) candidate slot it
        // emits the per-cell TRIANGLE_PRISM contacts (tread + riser) routed through
        // cvx GJK/EPA, written into THAT slot's ucontact_* with side a = convex,
        // side b = the static heightfield. Part of the ONE general PairDriven path.
        // The descriptor travels in params (the first cooked heightfield; the grid
        // rides the model `heights` field).
        p_np_hf_.family = family;
        p_np_hf_.env_count = env_count;
        p_np_hf_.slot_stride = cap.max_contacts_per_env;
        p_np_hf_.rigid_slot_cap = rigid_cap;  // body<->body fills only [0, rigid_cap).
        p_np_hf_.bodies_per_env = cap.bodies_per_env;
        p_np_hf_.hull_vert_count =
            static_cast<uint32_t>(model.hull_verts.size() / 3u);
        p_np_hf_.contact_margin = cfg.contact_margin;
        // NarrowphaseHeightfieldParams carries ONE descriptor; a model with more
        // than one cooked heightfield would silently drop all but the first.
        assert(model.heightfields.size() <= 1 &&
               "pipeline wires a single cooked heightfield; multi-heightfield "
               "needs a descriptor table in NarrowphaseHeightfieldParams");
        if (!model.heightfields.empty()) {
            const nk::HeightfieldData& hfd = model.heightfields.front();
            p_np_hf_.has_heightfield = 1u;
            p_np_hf_.origin_x = hfd.origin.x;
            p_np_hf_.origin_y = hfd.origin.y;
            p_np_hf_.origin_z = hfd.origin.z;
            p_np_hf_.cell_size = hfd.cell_size;
            p_np_hf_.nrow = hfd.nrow;
            p_np_hf_.ncol = hfd.ncol;
            p_np_hf_.min_z = hfd.min_z;
            p_np_hf_.max_z = hfd.max_z;
            p_np_hf_.data_offset = hfd.data_offset;
            p_np_hf_.hf_body_row = 0u;  // resolved per-slot from the candidate kind.
        } else {
            p_np_hf_.has_heightfield = 0u;
        }
        add(phi::NkOp::NarrowphaseHeightfield, &p_np_hf_);

        p_np_sdf_.contact_margin = cfg.contact_margin;
        p_np_sdf_.max_contacts_per_pair = kMaxContactsPerPair;
        p_np_sdf_.family = family;          // PairDriven => sample; else no-op.
        p_np_sdf_.env_count = env_count;
        p_np_sdf_.bodies_per_env = cap.bodies_per_env;
        p_np_sdf_.max_contacts_per_env = cap.max_contacts_per_env;
        p_np_sdf_.rigid_slot_cap = rigid_cap;  // body<->body fills only [0, rigid_cap).
        add(phi::NkOp::NarrowphaseSdf, &p_np_sdf_);

        // Body/artic <-> particle narrowphase (the row provider's pre-coupling
        // emission). Runs AFTER the rigid narrowphase and BEFORE AssembleRows,
        // inside has_collidables. Gated on actual particles; a particle-free world
        // emits no op -> byte-identical.
        if (has_particles) {
            row_coupling_provider_.PreCouple(coupling_ctx);
        }

        // ContactTangentBasis: the ONE general PairDriven path. The op builds the
        // tangents over the unified contact buffer (ucontact_*).
        p_tangent_.slot_count = slot_count;
        p_tangent_.family = family;
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
        // standalone backward-Euler joint viscous damping. This is GENERAL
        // articulation physics that used to ride inside the now-deleted FUSED
        // contact solve kernel (which ran every step on the go2 stand world even
        // with zero actual contacts). It applies qdot -= dt*(M+dt*C)^-1*(C*qdot)
        // using the freshly factored data.m_inv == (M+dt*C)^-1, so it MUST come
        // AFTER CrbaFactorM and BEFORE AssembleRows/SolveRowsBlockIsland +
        // IntegratePosition (the legacy single-env oracle order: factor ->
        // damping -> solve -> integrate). Gated on fold_drive_damping so worlds
        // that don't fold dt*C (e.g. the union path) emit NO new op -> their
        // captured graphs / goldens stay byte-identical.
        if (cfg.fold_drive_damping != 0u) {
            p_apply_damping_.dt = cfg.dt;
            p_apply_damping_.max_dof = max_dof;
            p_apply_damping_.articulation_count = articulation_cnt;
            p_apply_damping_.total_link_count = total_link_count;
            add(phi::NkOp::ApplyImplicitDamping, &p_apply_damping_);
        }
    }

    if (has_contacts) {
        p_assemble_.dt = cfg.dt;
        p_assemble_.slot_count = slot_count;
        p_assemble_.max_dof = max_dof;
        p_assemble_.env_count = env_count;
        p_assemble_.articulation_count = articulation_cnt;
        p_assemble_.total_link_count = total_link_count;
        p_assemble_.family = family;
        // The PairDriven assembly reads union_slot_count as the CANDIDATE-slot
        // stride (slots per env in the unified contact buffer), which must equal the
        // broadphase's max_contacts_per_env (so the candidate index lines up across
        // broadphase -> narrowphase -> assembly). (Field name retained for the
        // shared params layout.)
        p_assemble_.union_slot_count = cap.max_contacts_per_env;
        p_assemble_.rows_per_env = cap.max_rows_per_env;
        p_assemble_.bodies_per_env = cap.bodies_per_env;
        p_assemble_.base_link_count = base_link_count;
        for (int k = 0; k < 2; ++k) p_assemble_.solref[k] = model.contact_solref[k];
        for (int k = 0; k < 5; ++k) p_assemble_.solimp[k] = model.contact_solimp[k];
        p_assemble_.num_material_buckets = cap.num_material_buckets;
        p_assemble_.particles_per_env = cap.particles_per_env;  // coupling.
        // Per-particle-system friction for a body<->particle contact side: the
        // [soft | fluid] split + each slice's mu (a particle has no body material).
        p_assemble_.n_soft_particles = friction_n_soft;
        p_assemble_.particle_soft_friction = mp.soft_friction;
        p_assemble_.particle_fluid_friction = mp.fluid_friction;
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
        p_xpbd_.shape_match_cluster_count = sm_cluster_count;
        // Graph-coloring: per-family color counts + per-env strides so the colored
        // kernels iterate the single-env color ranges env-major in parallel.
        p_xpbd_.dist_colors = cap.xpbd_dist_colors;
        p_xpbd_.bend_colors = cap.xpbd_bend_colors;
        p_xpbd_.vol_colors  = cap.xpbd_vol_colors;
        p_xpbd_.sm_colors   = cap.xpbd_sm_colors;
        p_xpbd_.env_count   = env_count;
        p_xpbd_.dist_cons_per_env   = cap.dist_cons_per_env;
        p_xpbd_.bend_cons_per_env   = cap.bend_cons_per_env;
        p_xpbd_.vol_cons_per_env    = cap.vol_cons_per_env;
        p_xpbd_.sm_clusters_per_env = cap.shape_match_slots_per_env;
        p_xpbd_.sm_members_per_env  = cap.shape_match_members_per_env;
        p_xpbd_.dist_color_segments = model.dist_color_segments.data();
        p_xpbd_.bend_color_segments = model.bend_color_segments.data();
        p_xpbd_.vol_color_segments  = model.vol_color_segments.data();
        p_xpbd_.sm_color_segments   = model.sm_color_segments.data();
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
        p_pbf_density_.n_soft_particles = n_soft;
        p_pbf_density_.particles_per_env = per_env_particles;
        add(phi::NkOp::PbfDensityLambda, &p_pbf_density_);

        p_pbf_apply_.support_radius  = mp.pbf_support_radius;
        p_pbf_apply_.particle_mass   = mp.pbf_particle_mass;
        p_pbf_apply_.particle_count  = particle_count;
        p_pbf_apply_.boundary_enabled = mp.boundary_enabled ? 1u : 0u;
        p_pbf_apply_.floor_z         = mp.floor_z;
        p_pbf_apply_.n_soft_particles = n_soft;
        p_pbf_apply_.particles_per_env = per_env_particles;
        add(phi::NkOp::PbfApplyDelta, &p_pbf_apply_);
    }

    if (has_contacts) {
        // The rigid-body arm: the spec-fixed SolveRowsBlockIslandParams takes over the slot
        // (the transitional SolveArticulatedParams routing is deleted).
        // Semantic triplet first; appended launch geometry + the per-family
        // Model-derived solver constants (see op_schema.hpp).
        p_solve_.dt = cfg.dt;
        p_solve_.vel_iters = cfg.vel_iters;
        // Split-impulse position pass runs ONLY on the general PairDriven path; the
        // Union/Fused families stay velocity-only (byte-identical).
        p_solve_.pos_iters =
            (family == phi::kContactFamilyPairDriven) ? cfg.pos_iters : 0u;
        p_solve_.pos_beta = cfg.pos_beta;
        p_solve_.pos_slop = cfg.pos_slop;
        p_solve_.total_particle_count = particle_count;
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
        // (: the FUSED-family implicit-damping seed moved to the standalone
        // NkOp::ApplyImplicitDamping op (added after CrbaFactorM above); the
        // solve op no longer carries an apply_implicit_damping knob.)
        // The per-articulation slot stride MUST match the detection/assembly
        // (slot_base = articulation * stride). Data-driven; 4 at K<=1.
        p_solve_.contact_slots_per_artic = contact_slots_per_artic;
        // Pre-solve coupling seam: each registered provider's Couple funnels its
        // two-way reaction into the shared body sink before the single solve. The
        // row provider emits nothing here (it rides this solve); a grid-transfer
        // provider emits its umbrella op here.
        if (has_particles) {
            row_coupling_provider_.Couple(coupling_ctx);
        }
        add(phi::NkOp::SolveRowsBlockIsland, &p_solve_);
    }

    if (has_articulation || has_bodies) {
        p_int_pos_.dt = cfg.dt;
        p_int_pos_.total_link_count = total_link_count;
        p_int_pos_.articulation_count = articulation_cnt;
        p_int_pos_.total_body_count = cap.bodies_per_env * env_count;
        // Read the split-impulse pseudo velocity additively when the position pass
        // is active (the general PairDriven path); else velocity-only (identical).
        p_int_pos_.pos_pass =
            (has_contacts && family == phi::kContactFamilyPairDriven &&
             cfg.pos_iters > 0u) ? 1u : 0u;
        add(phi::NkOp::IntegratePosition, &p_int_pos_);
    }

    // The row provider's post-coupling emission (ParticleFinalize +
    // ParticleParticleContact). Runs AFTER IntegratePosition at the TOP level
    // (NOT inside has_collidables) and BEFORE ReadoutContactWrench, the original
    // position. Gated on actual particles -> byte-identical when absent.
    if (has_particles) {
        row_coupling_provider_.PostCouple(coupling_ctx);
    }

    // ReadoutContactWrench: the general per-env contact-wrench readout over the
    // unified PairDriven contact buffer (removed the !is_union gate — the
    // UnionCsr family that suppressed this op is gone; the path is always
    // PairDriven now).
    if (has_articulation || has_bodies) {
        p_readout_.dt = cfg.dt;
        p_readout_.env_count = env_count;
        p_readout_.base_link_count = base_link_count;
        p_readout_.max_contacts_per_env = cap.max_contacts_per_env;
        p_readout_.rows_per_env = cap.max_rows_per_env;
        add(phi::NkOp::ReadoutContactWrench, &p_readout_);
    }

    // the union-family per-env contact-observation readout
    // (ReadoutUnionContactObs) was DELETED together with the entire UnionCsr path.
    // Grasp/union moved to RL; the general per-env contact readout is
    // ReadoutContactWrench over the unified contact buffer (above).
}

} // namespace nuka::nk
