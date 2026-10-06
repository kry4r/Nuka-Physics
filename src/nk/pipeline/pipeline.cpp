// ---------------------------------------------------------------------------
// nk::Pipeline implementation (the design fixed order).
// ---------------------------------------------------------------------------

#include "nk/pipeline/pipeline.hpp"
#include "phi/articulation_contract.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "collision/shape_kind.hpp"
#include "constraint/contact_manifold.hpp"  // ContactManifold::kMaxPoints
#include "nk/model/model.hpp"
#include "nk/solve/nk_row.hpp"
#include "nk/solve/point_endpoint.hpp"
#include "sensor/state_bank.hpp"

namespace nuka::nk {

// The per-pair manifold point capacity shared by every narrowphase op. Bound to
// the manifold struct so the param fields cannot drift from the solver layout.
inline constexpr uint8_t kMaxContactsPerPair =
    static_cast<uint8_t>(constraint::ContactManifold::kMaxPoints);

void Pipeline::AddOp(phi::NkOp op, const void* params, phi::Device* device) {
    if (device != nullptr && !phi::DeviceSupportsOp(device, op)) {
        if (std::find(missing_ops_.begin(), missing_ops_.end(), op) == missing_ops_.end())
            missing_ops_.push_back(op);
        return;
    }
    calls_.push_back(phi::OpCall{op, params});
}

uint32_t Pipeline::SubstepCount(const Model& model, const SolverConfig& cfg) {
    uint32_t count = std::max(1u, cfg.substeps);
    if (model.MpmParticlesPerEnv() > 0u && model.capacities.mpm_grid_nodes_per_env > 0u)
        count = std::max(count, model.particles.mpm_substeps);
    return count;
}

phi::Status Pipeline::Build(const Model& model, const SolverConfig& cfg,
                            phi::Device* device, uint32_t readout_demand,
                            const sensor::StateSensorBank* sensors) {
    const uint32_t substeps = SubstepCount(model, cfg);
    SolverConfig interval = cfg;
    interval.dt /= static_cast<float>(substeps);
    calls_.clear();
    missing_ops_.clear();
    p_accumulate_step_.clear();
    p_energy_.clear();
    p_block_descent_substeps_.clear();
    const char* block_descent = std::getenv("NUKA_BLOCK_DESCENT");
    use_block_descent_ = block_descent != nullptr && block_descent[0] == '1' &&
        block_descent[1] == '\0';
    p_block_descent_ = {};
    if (use_block_descent_ && cfg.velocity_iterations_override)
        p_block_descent_.iterations = cfg.vel_iters;
    if (use_block_descent_ && !cfg.velocity_iterations_override) {
        const char* iterations = std::getenv("NUKA_BLOCK_DESCENT_ITERATIONS");
        if (iterations != nullptr) {
            uint32_t count = 0u;
            if (*iterations == '\0') return phi::Status::InvalidArgument;
            for (const char* digit = iterations; *digit != '\0'; ++digit) {
                if (*digit < '0' || *digit > '9') return phi::Status::InvalidArgument;
                const uint32_t value = static_cast<uint32_t>(*digit - '0');
                if (count > (std::numeric_limits<uint32_t>::max() - value) / 10u)
                    return phi::Status::InvalidArgument;
                count = count * 10u + value;
            }
            if (count == 0u) return phi::Status::InvalidArgument;
            p_block_descent_.iterations = count;
        }
    }
    if ((readout_demand & kReadoutEnergyLedger) != 0u)
        p_energy_.resize(size_t{substeps} * kEnergyStageCount);
    if (!(interval.dt > 0.0f) || !std::isfinite(interval.dt) ||
        !std::isfinite(1.0f / interval.dt) || !std::isfinite(1.0f / cfg.dt))
        return phi::Status::InvalidArgument;
    const auto status = BuildInterval(model, interval, device, readout_demand);
    if (status != phi::Status::Ok) return status;
    if (sensors && (!sensors->Before().empty() || !sensors->After().empty())) {
        const auto physics = std::move(calls_);
        calls_.clear();
        for (const auto& call : sensors->Before()) AddOp(call.op, call.params, device);
        for (const auto& call : physics) AddOp(call.op, call.params, device);
        for (const auto& call : sensors->After()) AddOp(call.op, call.params, device);
        if (!missing_ops_.empty()) { calls_.clear(); return phi::Status::Unsupported; }
    }
    const auto& cap = model.capacities;
    const bool has_mpm = model.MpmParticlesPerEnv() > 0u && cap.mpm_grid_nodes_per_env > 0u;
    const bool mpm_reaction = has_mpm;
    if (substeps == 1u && !mpm_reaction) return phi::Status::Ok;
    const uint64_t call_count = uint64_t(substeps) * (calls_.size() + 1u + uint32_t(mpm_reaction));
    if (call_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        calls_.clear();
        return phi::Status::InvalidArgument;
    }
    const auto interval_calls = calls_;
    p_int_vel_.clear_body_forces = 0u;
    calls_.clear();
    calls_.reserve(static_cast<size_t>(call_count));
    p_accumulate_step_.resize(static_cast<size_t>(substeps) * 2u);
    if (use_block_descent_) p_block_descent_substeps_.resize(substeps);
    for (uint32_t step = 0u; step < substeps; ++step) {
        auto& capture = p_accumulate_step_[static_cast<size_t>(step) * 2u];
        capture.env_count = cap.env_count;
        capture.bodies_per_env = cap.bodies_per_env;
        capture.links_per_env = cap.links_per_env;
        capture.artics_per_env = cap.articulations_per_env;
        capture.flags = phi::kAccumulateMpmImpulse;
        capture.first = step == 0u ? 1u : 0u;
        capture.last = step + 1u == substeps ? 1u : 0u;
        capture.substep_dt = interval.dt;
        capture.inv_outer_dt = 1.0f / cfg.dt;
        auto& outputs = p_accumulate_step_[static_cast<size_t>(step) * 2u + 1u];
        outputs = capture;
        outputs.flags = phi::kAccumulateOutputs;
        if (cap.bodies_per_env > 0u) outputs.flags |= phi::kFinalizeBodyForces;
        if (mpm_reaction) outputs.flags |= phi::kAccumulateMpmOutput;
        if (substeps > 1u && (readout_demand & kReadoutContactWrench) != 0u &&
            cap.max_contacts_per_env > 0u)
            outputs.flags |= phi::kAccumulateLinkWrench;
        if (substeps > 1u && cap.joint_limit_rows_per_env != 0u)
            outputs.flags |= phi::kAccumulateJointLimit;
        for (const auto& call : interval_calls) {
            if (call.op == phi::NkOp::ReadoutEnergyLedger || call.op == phi::NkOp::ReadoutContactAudit) {
                const auto& source = *static_cast<const phi::ReadoutEnergyLedgerParams*>(call.params);
                auto& energy = p_energy_[size_t{step} * kEnergyStageCount + static_cast<uint32_t>(source.stage)];
                if (step != 0u) energy = source;
                energy.slot = step;
                AddOp(call.op, &energy, device);
            } else if (call.op == phi::NkOp::BlockDescentSolve) {
                auto& solve = p_block_descent_substeps_[step];
                solve = *static_cast<const phi::BlockDescentSolveParams*>(call.params);
                solve.clear_failure_diagnostics = step == 0u ? 1u : 0u;
                solve.substep_index = step;
                AddOp(call.op, &solve, device);
            } else AddOp(call.op, call.params, device);
            if (mpm_reaction && call.op == phi::NkOp::MpmCommit)
                AddOp(phi::NkOp::AccumulateStep, &capture, device);
        }
        AddOp(phi::NkOp::AccumulateStep, &outputs, device);
    }
    if (!missing_ops_.empty()) { calls_.clear(); return phi::Status::Unsupported; }
    return phi::Status::Ok;
}

phi::Status Pipeline::BuildInterval(const Model& model, const SolverConfig& cfg,
                                    phi::Device* device, uint32_t readout_demand) {
    calls_.clear();
    missing_ops_.clear();
    const ModelCapacities& cap = model.capacities;
    float contact_margin = cfg.contact_margin;
    for (const auto& bucket : model.material_buckets)
        contact_margin = std::max(contact_margin, bucket.Profile().margin);

    const bool has_articulation = cap.dofs_per_env > 0 || cap.links_per_env > 0;
    const bool has_bodies       = cap.bodies_per_env > 0;
    const bool has_particles    = cap.particles_per_env > 0;
    const bool has_contacts     = cap.max_rows_per_env > 0;
    const uint32_t grid_particles_per_env = model.MpmParticlesPerEnv();
    if (grid_particles_per_env > cap.particles_per_env) return phi::Status::InvalidArgument;
    if (cap.vbd_vertices_per_env > 0u &&
        (cap.vbd_particle_begin < grid_particles_per_env ||
         uint64_t{cap.vbd_particle_begin} + cap.vbd_vertices_per_env > cap.particles_per_env))
        return phi::Status::InvalidArgument;
    const bool has_mpm = grid_particles_per_env > 0u && cap.mpm_grid_nodes_per_env > 0u;
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
    std::printf("[Pipeline] has_bodies=%d links_per_env=%u max_contacts_per_env=%u -> has_collidables=%d\n",
                has_bodies, cap.links_per_env, cap.max_contacts_per_env, has_collidables);
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
    // Coupled joints follow their roots; the articulation then steps in reduced coordinates.
    const uint32_t mimic_couplings  = has_articulation ? cap.mimic_couplings_per_env : 0u;
    const bool use_inverse_dynamics =
        model.drive_mode == static_cast<uint32_t>(phi::ArticulationControlMode::ComputedTorque) ||
        model.drive_mode == static_cast<uint32_t>(phi::ArticulationControlMode::Osc);
    // The same particle ownership sizes the cook reserve and the pipeline's row range.
    const uint64_t particle_reserve64 = cap.max_contacts_per_env > 0u
        ? cap.ParticleContactReserve(grid_particles_per_env) : 0u;
    if (particle_reserve64 + cap.ogc_contacts_per_env +
        cap.mpm_contact_capacity_per_env > cap.max_contacts_per_env)
        return phi::Status::InvalidArgument;
    const uint32_t particle_reserve = static_cast<uint32_t>(particle_reserve64);
    const uint32_t rigid_cap = cap.max_contacts_per_env - particle_reserve -
                               cap.ogc_contacts_per_env - cap.mpm_contact_capacity_per_env;
    const uint32_t default_pair_cap =
        cap.bodies_per_env * 4u < rigid_cap ? cap.bodies_per_env * 4u : rigid_cap;
    const uint32_t pair_emit_cap =
        cfg.max_pairs == 0u ? default_pair_cap
                            : (cfg.max_pairs < rigid_cap ? cfg.max_pairs : rigid_cap);
    const uint32_t contact_rows_per_env =
        rigid_cap * kPairDrivenRowsPerSlot +
        (particle_reserve + cap.ogc_contacts_per_env +
         cap.mpm_contact_capacity_per_env) * kPairDrivenParticleRowsPerSlot;
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

    // Contact detection and solving consume the internally projected particle state.
    // Step-start velocity remains the acceleration reference; finalization adds contact deltas once.

    const auto energy = [&](EnergyStage stage) {
        if ((readout_demand & kReadoutEnergyLedger) == 0u) return;
        auto& p = p_energy_[static_cast<uint32_t>(stage)];
        p.stage = stage;
        p.physics_diagnostics = (readout_demand & kReadoutPhysicsDiagnostics) != 0u;
        p.cloth_integrator = cfg.cloth_integrator;
        p.contact_slots_per_env = cap.max_contacts_per_env;
        p.rigid_contact_slots = rigid_cap;
        p.ogc_contact_slots = cap.ogc_contacts_per_env;
        p.pos_slop = cfg.pos_slop;
        p.dt = cfg.dt;
        std::copy(std::begin(cfg.gravity), std::end(cfg.gravity), p.gravity);
        p.env_count = env_count;
        p.substeps = cap.integration_substeps;
        p.particles_per_env = cap.particles_per_env;
        p.vbd_begin = cap.vbd_particle_begin;
        p.vbd_vertices = cap.vbd_vertices_per_env;
        p.vbd_elements = cap.vbd_elements_per_env;
        p.bodies_per_env = cap.bodies_per_env;
        p.links_per_env = cap.links_per_env;
        p.artics_per_env = has_articulation ? cap.articulations_per_env : 0u;
        p.dofs_per_artic = cap.dofs_per_env;
        p.rows_per_env = cap.max_rows_per_env;
        p.contact_rows = contact_rows_per_env;
        p.limit_rows = cap.joint_limit_rows_per_env;
        p.friction_rows = cap.joint_friction_rows_per_env;
        p.drive_rows = cap.joint_drive_rows_per_env;
        p.mimic_couplings = mimic_couplings;
        p.has_dat = cap.ogc_contacts_per_env > 0u;
        p.has_aero = cap.aero_tris_per_env > 0u &&
            (model.particles.aero_drag_normal != 0.0f || model.particles.aero_drag_tangent != 0.0f);
        p.pos_pass = !use_block_descent_ && has_contacts && cfg.pos_iters > 0u;
        if (has_mpm) p.coverage_status |= energy_status::kGridMaterial;
        if (cap.particles_per_env != cap.vbd_vertices_per_env)
            p.coverage_status |= energy_status::kOtherParticleMaterial;
        add(phi::NkOp::ReadoutEnergyLedger, &p);
        if ((stage == EnergyStage::Solved || stage == EnergyStage::End) &&
            (readout_demand & kReadoutContactAudit) != 0u)
            add(phi::NkOp::ReadoutContactAudit, &p);
    };
    energy(EnergyStage::Begin);

    if (has_contacts) {
        p_step_velocity_.env_count = env_count;
        p_step_velocity_.articulation_count = articulation_cnt;
        p_step_velocity_.max_dof = max_dof;
        p_step_velocity_.base_link_count = base_link_count;
        p_step_velocity_.total_body_count = cap.bodies_per_env * env_count;
        p_step_velocity_.total_particle_count = cap.particles_per_env * env_count;
        add(phi::NkOp::SnapshotStepVelocity, &p_step_velocity_);
    }

    if (has_articulation) {
        p_apply_drives_.dt = cfg.dt;
        p_apply_drives_.total_link_count = total_link_count;
        p_apply_drives_.defer_velocity_damping = cfg.defer_velocity_damping && cfg.fold_drive_damping;
        p_apply_drives_.links_per_env = base_link_count;
        p_apply_drives_.mode = model.drive_mode;

        p_aba_.gravity[0] = cfg.gravity[0];
        p_aba_.gravity[1] = cfg.gravity[1];
        p_aba_.gravity[2] = cfg.gravity[2];
        p_aba_.articulation_count = articulation_cnt;
        p_aba_.total_link_count = total_link_count;

        if (use_inverse_dynamics) {
            // Current kinematics and force-free acceleration share the same mass response.
            // Controllers remove passive damping from bias compensation.
            p_fk_.articulation_count = articulation_cnt;
            p_fk_.total_link_count = total_link_count;
            add(phi::NkOp::FkWorldPoses, &p_fk_);
            add(phi::NkOp::ApplyDrives, &p_apply_drives_);
            add(phi::NkOp::AbaForward, &p_aba_);

            // Inverse dynamics and contacts consume the same physical M^-1.
            p_crba_m_.dt = cfg.dt;
            p_crba_m_.max_dof = max_dof;
            p_crba_m_.articulation_count = articulation_cnt;
            p_crba_m_.total_link_count = total_link_count;
            p_crba_m_.fold_drive_damping = 0u;
            add(phi::NkOp::CrbaComputeM, &p_crba_m_);
            p_crba_factor_.max_dof = max_dof;
            p_crba_factor_.articulation_count = articulation_cnt;
            add(phi::NkOp::CrbaFactorM, &p_crba_factor_);

            p_apply_dynamics_.max_dof = max_dof;
            p_apply_dynamics_.articulation_count = articulation_cnt;
            p_apply_dynamics_.total_link_count = total_link_count;
            p_apply_dynamics_.task_link = model.osc_task_link;
            p_apply_dynamics_.mode = model.drive_mode;
            p_apply_dynamics_.links_per_env = base_link_count;
            std::copy(std::begin(cfg.gravity), std::end(cfg.gravity), p_apply_dynamics_.gravity);
            add(phi::NkOp::ApplyDynamicsDrives, &p_apply_dynamics_);
            add(phi::NkOp::AbaForward, &p_aba_);
        } else {
            add(phi::NkOp::ApplyDrives, &p_apply_drives_);
            add(phi::NkOp::AbaForward, &p_aba_);
        }
    }

    if (has_articulation || has_bodies) {
        p_int_vel_.dt = cfg.dt;
        std::copy(std::begin(cfg.gravity), std::end(cfg.gravity), p_int_vel_.gravity);
        p_int_vel_.total_link_count = total_link_count;
        p_int_vel_.articulation_count = articulation_cnt;
        // Free-body loads are applied once before the shared contact solve.
        p_int_vel_.total_body_count = cap.bodies_per_env * env_count;
        p_int_vel_.clear_body_forces = 1u;
        add(phi::NkOp::IntegrateVelocity, &p_int_vel_);
    }

    // particle launch geometry + mode (resolved once from the Model).
    const uint32_t particle_count = cap.particles_per_env * env_count;
    const uint32_t sm_cluster_count = cap.shape_match_slots_per_env * env_count;
    const Model::ModelParticles& mp = model.particles;
    const uint32_t particle_mode =
        mp.mode == Model::ParticleMode::Xpbd ? phi::kParticleModeXpbd
        : mp.mode == Model::ParticleMode::Pbf ? phi::kParticleModePbf
        : mp.mode == Model::ParticleMode::Coupled ? phi::kParticleModeCoupled
        : mp.mode == Model::ParticleMode::SoftFluid ? phi::kParticleModeSoftFluid
        : mp.mode == Model::ParticleMode::Mpm ? phi::kParticleModeMpm
        : mp.mode == Model::ParticleMode::MpmXpbd ? phi::kParticleModeMpmXpbd
                                                  : phi::kParticleModeNone;
    // SoftFluid: the per-env [soft | fluid] split + stride (0 elsewhere).
    const uint32_t n_soft = mp.mode == Model::ParticleMode::SoftFluid
                                ? mp.n_soft_particles : 0u;
    const uint32_t n_mpm = grid_particles_per_env;
    const uint32_t per_env_particles = cap.particles_per_env;
    phi::VertexBlockLayout vertex_blocks;
    vertex_blocks.begin = cap.vbd_particle_begin;
    vertex_blocks.vertices = cap.vbd_vertices_per_env;
    vertex_blocks.dynamic_vertices = cap.vbd_dynamic_vertices_per_env;
    vertex_blocks.colors = cap.vbd_colors;
    vertex_blocks.particles_per_env = per_env_particles;
    vertex_blocks.env_count = env_count;
    vertex_blocks.elements = cap.vbd_elements_per_env;
    if (use_block_descent_) {
        p_block_descent_.dt = cfg.dt;
        p_block_descent_.env_count = env_count;
        p_block_descent_.rows_per_env = cap.max_rows_per_env;
        p_block_descent_.total_particle_count = particle_count;
        p_block_descent_.total_body_count = cap.bodies_per_env * env_count;
        p_block_descent_.total_grid_count = cap.mpm_grid_nodes_per_env * env_count;
        p_block_descent_.articulation_count = articulation_cnt;
        p_block_descent_.max_dof = max_dof;
        p_block_descent_.mimic_couplings = mimic_couplings;
        p_block_descent_.material_cells_per_env = cap.mpm_stress_cells_per_env;
        p_block_descent_.grid_particles_per_env = grid_particles_per_env;
        std::copy(mp.mpm_grid_dims, mp.mpm_grid_dims + 3u, p_block_descent_.grid_dims);
        p_block_descent_.base_link_count = base_link_count;
        p_block_descent_.contact_rows_per_env = contact_rows_per_env;
        p_block_descent_.joint_limit_rows_per_env = cap.joint_limit_rows_per_env;
        p_block_descent_.workspace_words = cap.block_descent_scratch_words;
        p_block_descent_.vertex_blocks = vertex_blocks;
        const char* acceleration_radius = std::getenv("NUKA_BLOCK_DESCENT_SPECTRAL_RADIUS");
        if (acceleration_radius != nullptr) {
            char* end = nullptr;
            const float radius = std::strtof(acceleration_radius, &end);
            if (end == acceleration_radius || *end != '\0' || !std::isfinite(radius) ||
                radius < 0.0f || radius >= 1.0f) return phi::Status::InvalidArgument;
            p_block_descent_.acceleration_spectral_radius = radius;
        }
        if (cap.ogc_contacts_per_env > 0u)
            p_block_descent_.max_point_terms = kTriangleEndpointTerms;
        if (cap.vol_cons_per_env > 0u) {
            constexpr uint32_t kTetrahedronEndpointTerms = 4u;
            p_block_descent_.max_point_terms =
                std::max(p_block_descent_.max_point_terms, kTetrahedronEndpointTerms);
        }
        if (has_mpm)
            p_block_descent_.max_point_terms = std::max(
                {p_block_descent_.max_point_terms, kMpmStencilNodes, kMpmCellStencilNodes});
        const char* diagnostics = std::getenv("NUKA_CONTACT_SOLVER_DIAGNOSTICS");
        p_block_descent_.measure_vertex_audit = (readout_demand & kReadoutVbdSolveAudit) != 0u;
        p_block_descent_.measure_contact_residual = cfg.measure_contact_residual ||
            p_block_descent_.measure_vertex_audit != 0u ||
            (readout_demand & kReadoutPhysicsDiagnostics) != 0u ||
            (diagnostics != nullptr && diagnostics[0] == '1');
    }
    // Per-env soft-particle count selecting which per-system mu a particle side
    // reads: SoftFluid the explicit split, Xpbd all-soft, Pbf all-fluid, Coupled by type.
    const uint32_t friction_n_soft =
        mp.mode == Model::ParticleMode::SoftFluid ? mp.n_soft_particles
        : mp.mode == Model::ParticleMode::Xpbd ? per_env_particles
        : mp.mode == Model::ParticleMode::MpmXpbd ? per_env_particles
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

    // Distance and volume constraints are rows of the common solve; shape matching projects.
    const uint32_t xpbd_iterations = sm_cluster_count != 0u
        ? std::max<uint32_t>(mp.xpbd_iters, 1u) : 0u;
    const uint32_t pbf_iterations = runs_pbf ? std::max<uint32_t>(mp.pbf_iters, 1u) : 0u;
    const bool pp_contact = particle_mode == phi::kParticleModeSoftFluid || mp.pp_self_contact;
    // Contact runs in every pass that projects the soft constraints it competes with.
    const uint32_t pp_iterations = pp_contact && mp.pp_contact_d_min > 0.0f
        ? std::max({mp.pp_contact_iters, xpbd_iterations, 1u}) : 0u;
    const uint32_t projection_iterations = std::max({1u, xpbd_iterations, pbf_iterations, pp_iterations});
    const uint32_t coupling_iterations = cfg.coupling_passes != 0u
        ? cfg.coupling_passes : projection_iterations;
    p_xpbd_iterations_.resize(projection_iterations);
    p_solve_iterations_.resize(coupling_iterations);
    const auto iterations_before = [](uint32_t pass, uint32_t budget, uint32_t passes) {
        return static_cast<uint32_t>((static_cast<uint64_t>(pass) * budget + passes - 1u) / passes);
    };
    const auto iteration_work = [&](uint32_t pass, uint32_t budget, uint32_t passes) {
        return iterations_before(pass + 1u, budget, passes) - iterations_before(pass, budget, passes);
    };

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
    coupling_ctx.articulation_count = articulation_cnt;
    coupling_ctx.max_dof = max_dof;
    coupling_ctx.base_link_count = base_link_count;
    coupling_ctx.artics_per_env = artics_per_env;
    coupling_ctx.particles_per_env = cap.particles_per_env;
    coupling_ctx.max_contacts_per_env = cap.max_contacts_per_env;
    coupling_ctx.rigid_cap = rigid_cap;
    coupling_ctx.particle_slot_base = rigid_cap + cap.ogc_contacts_per_env;
    coupling_ctx.particle_mode = particle_mode;
    coupling_ctx.coupled_internal = coupled_internal;
    coupling_ctx.particle_count = particle_count;
    coupling_ctx.n_soft = n_soft;
    coupling_ctx.n_mpm = n_mpm;
    coupling_ctx.dt = cfg.dt;
    coupling_ctx.gravity[0] = cfg.gravity[0];
    coupling_ctx.gravity[1] = cfg.gravity[1];
    coupling_ctx.gravity[2] = cfg.gravity[2];
    coupling_ctx.contact_margin = contact_margin;
    coupling_ctx.pos_pass =
        (!use_block_descent_ && has_contacts && family == phi::kContactFamilyPairDriven &&
         cfg.pos_iters > 0u) ? 1u : 0u;
    coupling_ctx.p_np_body_particle = &p_np_body_particle_;
    coupling_ctx.p_part_finalize = &p_part_finalize_;
    coupling_ctx.p_pp_contact = &p_pp_contact_;
    coupling_ctx.p_mpm = &p_mpm_;
    coupling_ctx.has_mpm = has_mpm ? 1u : 0u;

    if (has_particles) {
        // Cloth anisotropic air drag runs BEFORE the predict so the drag-modified
        // velocity feeds the position integration. Inert (op early-exits) unless a
        // coefficient is set, so a drag-free world's op list is byte-identical.
        const uint32_t aero_tri_count = cap.aero_tris_per_env * env_count;
        if (aero_tri_count > 0u &&
            (mp.aero_drag_normal != 0.0f || mp.aero_drag_tangent != 0.0f)) {
            p_aero_drag_.dt = cfg.dt;
            p_aero_drag_.drag_normal = mp.aero_drag_normal;
            p_aero_drag_.drag_tangent = mp.aero_drag_tangent;
            p_aero_drag_.max_dv = mp.aero_drag_max_dv;
            p_aero_drag_.tri_count = aero_tri_count;
            p_aero_drag_.particle_count = cap.particles_per_env * env_count;
            add(phi::NkOp::ParticleAeroDrag, &p_aero_drag_);
        }

        p_part_predict_.dt = cfg.dt;
        p_part_predict_.gravity[0] = cfg.gravity[0];
        p_part_predict_.gravity[1] = cfg.gravity[1];
        p_part_predict_.gravity[2] = cfg.gravity[2];
        p_part_predict_.mode = particle_mode;
        p_part_predict_.particle_count = particle_count;
        p_part_predict_.coupled_internal = coupled_internal;
        p_part_predict_.n_soft_particles = n_soft;
        p_part_predict_.particles_per_env = per_env_particles;
        p_part_predict_.n_mpm_particles = n_mpm;
        p_part_predict_.vertex_blocks = vertex_blocks;
        add(phi::NkOp::ParticlePredict, &p_part_predict_);
        if (vertex_blocks.vertices > 0u) {
            p_cloth_step_.dt = cfg.dt;
            for (int k = 0; k < 3; ++k) p_cloth_step_.gravity[k] = cfg.gravity[k];
            p_cloth_step_.layout = vertex_blocks;
            p_cloth_step_.integrator = cfg.cloth_integrator;
            p_cloth_step_.truncation = 0u;
            add(phi::NkOp::ClothPredict, &p_cloth_step_);
        }
    }

    mpm_coupling_provider_.PreCouple(coupling_ctx);

    if (has_articulation && !use_inverse_dynamics) {
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
        p_aabbs_.margin = contact_margin;
        p_aabbs_.family = family;
        p_aabbs_.env_count = env_count;
        p_aabbs_.bodies_per_env = bodies_per_env;
        add(phi::NkOp::BuildAabbs, &p_aabbs_);

        p_lbvh_build_.family = family;
        p_lbvh_build_.env_count = env_count;
        p_lbvh_build_.bodies_per_env = bodies_per_env;
        p_lbvh_build_.workspace_bytes = cap.lbvh_sort_scratch_bytes;
        add(phi::NkOp::LbvhBuild, &p_lbvh_build_);

        p_lbvh_query_.max_pairs = pair_emit_cap;
        p_lbvh_query_.family = family;
        p_lbvh_query_.env_count = env_count;
        p_lbvh_query_.bodies_per_env = bodies_per_env;
        p_lbvh_query_.max_contacts_per_env = cap.max_contacts_per_env;
        p_lbvh_query_.rigid_slot_cap = rigid_cap;  // body<->body fills only [0, rigid_cap).
        p_lbvh_query_.workspace_bytes = cap.pair_sort_scratch_bytes;
        p_lbvh_query_.filter_cross_env = model.filter_cross_env ? 1u : 0u;
        p_lbvh_query_.excluded_count =
            static_cast<uint32_t>(model.excluded_pairs.size());
        add(phi::NkOp::LbvhQueryPairs, &p_lbvh_query_);
    }

    if (has_particles) {
        // Spatial constraints rebuild neighbors from the current shared working positions.
        p_grid_.cell_size = runs_pbf || pp_iterations != 0u ? mp.cell_size : 0.0f;
        p_grid_.query_radius = mp.query_radius;
        p_grid_.particle_count = particle_count;
        for (int k = 0; k < 3; ++k) {
            p_grid_.grid_min[k] = (&mp.grid_min.x)[k];
            p_grid_.grid_dims[k] = mp.grid_dims[k];
        }
        // The projected buffer is shared by every row-coupled particle material.
        p_grid_.pos_source = phi::kGridPosSourcePbfPredicted;
        // Env-private grids: per-env cell-key offsets + the cooked cell
        // capacity (the grid_cell_start/end arena sizing the op guards).
        p_grid_.env_count = env_count;
        p_grid_.particles_per_env = cap.particles_per_env;
        p_grid_.cells_capacity = cap.max_grid_cells;
        p_grid_.neighbor_capacity = static_cast<uint32_t>(cap.NeighborPoolCapacity());
    }

    if (has_particles) {
        p_xpbd_.dt = cfg.dt;
        p_xpbd_.iters = 1u;
        p_xpbd_.shape_match_cluster_count = sm_cluster_count;
        p_xpbd_.sm_colors = cap.xpbd_sm_colors;
        p_xpbd_.env_count = env_count;
        p_xpbd_.sm_clusters_per_env = cap.shape_match_slots_per_env;
        p_xpbd_.sm_members_per_env = cap.shape_match_members_per_env;
        p_xpbd_.sm_color_segments = model.sm_color_segments.data();
        for (uint32_t pass = 0u; pass < projection_iterations; ++pass) {
            p_xpbd_iterations_[pass] = p_xpbd_;
            p_xpbd_iterations_[pass].iteration_start =
                iterations_before(pass, xpbd_iterations, projection_iterations);
        }

        p_pbf_density_.rest_density = mp.pbf_rest_density;
        p_pbf_density_.relaxation = mp.pbf_cfm_epsilon;
        p_pbf_density_.support_radius = mp.pbf_support_radius;
        p_pbf_density_.particle_mass = mp.pbf_particle_mass;
        p_pbf_density_.particle_count = particle_count;
        p_pbf_density_.iters = 1u;
        p_pbf_density_.clamp_overdensity = mp.pbf_clamp_overdensity ? 1u : 0u;
        p_pbf_density_.dt = cfg.dt;
        p_pbf_density_.boundary_enabled = mp.boundary_enabled ? 1u : 0u;
        p_pbf_density_.floor_z = mp.floor_z;
        p_pbf_density_.n_soft_particles = n_soft;
        p_pbf_density_.particles_per_env = per_env_particles;

        p_pbf_apply_.support_radius = mp.pbf_support_radius;
        p_pbf_apply_.particle_mass = mp.pbf_particle_mass;
        p_pbf_apply_.particle_count = particle_count;
        p_pbf_apply_.boundary_enabled = mp.boundary_enabled ? 1u : 0u;
        p_pbf_apply_.floor_z = mp.floor_z;
        p_pbf_apply_.n_soft_particles = n_soft;
        p_pbf_apply_.particles_per_env = per_env_particles;

        p_part_projection_velocity_.dt = cfg.dt;
        p_part_projection_velocity_.particle_count = particle_count;
        p_part_projection_velocity_.particles_per_env = per_env_particles;
        p_part_projection_velocity_.active_begin_per_env =
            particle_mode == phi::kParticleModeNone ? per_env_particles : n_mpm;
        p_part_contact_delta_.dt = cfg.dt;
        p_part_contact_delta_.particle_count = particle_count;
        p_part_contact_delta_.particles_per_env = per_env_particles;
        p_part_contact_delta_.active_begin_per_env = p_part_projection_velocity_.active_begin_per_env;

        p_pp_contact_.contact_distance_d_min = pp_iterations != 0u ? mp.pp_contact_d_min : 0.0f;
        p_pp_contact_.compliance_alpha = mp.pp_contact_compliance;
        p_pp_contact_.solver_iterations = 1u;
        p_pp_contact_.mode = particle_mode;
        p_pp_contact_.particle_count = particle_count;
        p_pp_contact_.n_soft_particles = n_soft;
        p_pp_contact_.particles_per_env = per_env_particles;
        p_pp_contact_.vertex_blocks = vertex_blocks;
    }

    const auto project_particles = [&](uint32_t pass) {
        if (!has_particles) return;
        const uint32_t begin = iterations_before(pass, projection_iterations, coupling_iterations);
        const uint32_t end = iterations_before(pass + 1u, projection_iterations, coupling_iterations);
        for (uint32_t iteration = begin; iteration < end; ++iteration) {
            if (iteration_work(iteration, xpbd_iterations, projection_iterations) != 0u)
                add(phi::NkOp::XpbdProject, &p_xpbd_iterations_[iteration]);
            const bool density_work = iteration_work(iteration, pbf_iterations, projection_iterations) != 0u;
            const bool contact_work = iteration_work(iteration, pp_iterations, projection_iterations) != 0u;
            if (density_work || contact_work) add(phi::NkOp::ParticleGridBuild, &p_grid_);
            if (density_work) {
                add(phi::NkOp::PbfDensityLambda, &p_pbf_density_);
                add(phi::NkOp::PbfApplyDelta, &p_pbf_apply_);
            }
            if (contact_work) add(phi::NkOp::ParticleParticleContact, &p_pp_contact_);
            if (p_part_projection_velocity_.active_begin_per_env < per_env_particles)
                add(phi::NkOp::ParticleProjectionVelocity, &p_part_projection_velocity_);
        }
    };
    project_particles(0u);
    if (use_block_descent_)
        for (uint32_t pass = 1u; pass < coupling_iterations; ++pass) project_particles(pass);

    if ((cap.point_endpoints_per_env > 0u || cap.vbd_vertices_per_env > 0u) &&
        cap.particle_surfaces_per_env > 0u) {
        p_particle_surfaces_ = {env_count, per_env_particles, cap.particle_surfaces_per_env,
                               cap.particle_surface_triangles, cap.particle_surface_nodes_per_env,
                               cap.particle_surface_edge_nodes_per_env};
        add(phi::NkOp::RefitParticleSurfaces, &p_particle_surfaces_);
    }

    if (has_collidables) {
        p_np_prim_.contact_margin = contact_margin;
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
        p_np_hf_.contact_margin = contact_margin;
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

        p_np_sdf_.contact_margin = contact_margin;
        p_np_sdf_.max_contacts_per_pair = kMaxContactsPerPair;
        p_np_sdf_.family = family;          // PairDriven => sample; else no-op.
        p_np_sdf_.env_count = env_count;
        p_np_sdf_.bodies_per_env = cap.bodies_per_env;
        p_np_sdf_.max_contacts_per_env = cap.max_contacts_per_env;
        p_np_sdf_.rigid_slot_cap = rigid_cap;  // body<->body fills only [0, rigid_cap).
        p_np_sdf_.sdf_grid_count = cap.max_sdf_grids;
        p_np_sdf_.sample_point_count = cap.max_samp_points;
        p_np_sdf_.sdf_cell_total = cap.max_sdf_cells;
        p_np_sdf_.mesh_geometry = {cap.max_hull_verts, cap.max_mesh_triangles, cap.max_mesh_bvh_nodes};
        p_np_sdf_.max_body_samples = 0u;
        for (size_t i = 1u; i < model.samp_ranges.size(); i += 2u)
            p_np_sdf_.max_body_samples = std::max(p_np_sdf_.max_body_samples, model.samp_ranges[i]);
        const bool has_sampled_geometry = cap.max_samp_points > 0u || cap.max_sdf_grids > 0u ||
            std::any_of(model.shape_table_rows.begin(), model.shape_table_rows.end(),
                [](const Model::PairDrivenShape& shape) {
                    return shape.kind == collision::kShapeSdfMesh &&
                           (shape.contype | shape.conaffinity) != 0u;
                });
        if (has_sampled_geometry) add(phi::NkOp::NarrowphaseSdf, &p_np_sdf_);

        // Body/artic <-> particle narrowphase (the row provider's pre-coupling
        // emission). Runs AFTER the rigid narrowphase and BEFORE AssembleRows,
        // inside has_collidables. Gated on actual particles; a particle-free world
        // emits no op -> byte-identical.
        if (has_particles) {
            row_coupling_provider_.PreCouple(coupling_ctx);
        }

    }

    mpm_coupling_provider_.Couple(coupling_ctx);
    if (cap.ogc_contacts_per_env > 0u &&
        (cap.particle_surfaces_per_env > 0u || cap.mesh_vertex_source_count > 0u)) {
        p_ogc_detect_.dt = cfg.dt;
        p_ogc_detect_.margin = contact_margin;
        p_ogc_detect_.env_count = env_count;
        p_ogc_detect_.particles_per_env = cap.particles_per_env;
        p_ogc_detect_.surfaces_per_env = cap.particle_surfaces_per_env;
        p_ogc_detect_.triangles_per_env = cap.particle_surface_triangles;
        p_ogc_detect_.nodes_per_env = cap.particle_surface_nodes_per_env;
        p_ogc_detect_.edges_per_env = cap.particle_surface_edges_per_env;
        p_ogc_detect_.edge_nodes_per_env = cap.particle_surface_edge_nodes_per_env;
        p_ogc_detect_.bodies_per_env = cap.bodies_per_env;
        p_ogc_detect_.links_per_env = cap.links_per_env;
        p_ogc_detect_.articulations_per_env =
            has_articulation ? cap.articulations_per_env : 0u;
        p_ogc_detect_.mesh_vertex_sources = cap.mesh_vertex_source_count;
        p_ogc_detect_.mesh_edge_sources = cap.mesh_edge_source_count;
        p_ogc_detect_.mesh_vertices = cap.max_hull_verts;
        p_ogc_detect_.mesh_triangles = cap.max_mesh_triangles;
        p_ogc_detect_.mesh_nodes = cap.max_mesh_bvh_nodes;
        p_ogc_detect_.mesh_edges = cap.max_mesh_edges;
        p_ogc_detect_.mesh_edge_nodes = cap.max_mesh_edge_nodes;
        p_ogc_detect_.excluded_pairs = cap.max_excluded_pairs;
        p_ogc_detect_.slot_stride = cap.max_contacts_per_env;
        p_ogc_detect_.slot_base = rigid_cap;
        p_ogc_detect_.slot_capacity = cap.ogc_contacts_per_env;
        p_ogc_detect_.point_endpoints_per_env = cap.point_endpoints_per_env;
        p_ogc_detect_.point_endpoint_terms_per_env = cap.point_endpoint_terms_per_env;
        const bool mpm_grid = cap.mpm_grid_nodes_per_env > 0u;
        const uint32_t mpm_surfaces =
            cap.particle_surfaces_per_env > 0u ? cap.mpm_contact_capacity_per_env : 0u;
        p_ogc_detect_.point_endpoint_first = mpm_grid ? static_cast<uint32_t>(MpmPointEndpointCount(
            cap.particles_per_env, mpm_surfaces, cap.mpm_stress_cells_per_env)) : 0u;
        p_ogc_detect_.point_endpoint_term_first = mpm_grid ? static_cast<uint32_t>(MpmPointEndpointTermCount(
            cap.particles_per_env, mpm_surfaces, cap.mpm_stress_cells_per_env)) : 0u;
        p_ogc_detect_.workspace_bytes = cap.contact_cache_scratch_bytes;
        add(phi::NkOp::OgcDetect, &p_ogc_detect_);
    }
    if (cap.max_contacts_per_env > 0u) {
        // All contact providers finish before tangent construction and row assembly.
        p_tangent_.slot_count = slot_count;
        p_tangent_.family = family;
        add(phi::NkOp::ContactTangentBasis, &p_tangent_);
    }

    if (has_articulation && !use_inverse_dynamics) {
        p_crba_m_.dt = cfg.dt;
        p_crba_m_.max_dof = max_dof;
        p_crba_m_.articulation_count = articulation_cnt;
        p_crba_m_.total_link_count = total_link_count;
        // Implicit actuators fold their impedance into the step's mass, so contacts, the
        // position pass and the backward-Euler velocity all see the driven joints.
        p_crba_m_.fold_drive_damping = p_apply_drives_.defer_velocity_damping;
        add(phi::NkOp::CrbaComputeM, &p_crba_m_);
        p_crba_factor_.max_dof = max_dof;
        p_crba_factor_.articulation_count = articulation_cnt;
        add(phi::NkOp::CrbaFactorM, &p_crba_factor_);
        if (p_crba_m_.fold_drive_damping != 0u) {
            p_apply_damping_.dt = cfg.dt;
            p_apply_damping_.max_dof = max_dof;
            p_apply_damping_.articulation_count = articulation_cnt;
            p_apply_damping_.total_link_count = total_link_count;
            add(phi::NkOp::ApplyImplicitDamping, &p_apply_damping_);
        }
    }

    // Both mass paths have factored this step's mass and formed the step velocity.
    if (mimic_couplings > 0u) {
        p_mimic_reduce_.max_dof = max_dof;
        p_mimic_reduce_.articulation_count = articulation_cnt;
        p_mimic_reduce_.total_link_count = total_link_count;
        add(phi::NkOp::MimicReduce, &p_mimic_reduce_);
    }

    if (has_contacts) {
        p_assemble_.grid_nodes_per_env = cap.mpm_grid_nodes_per_env;
        p_assemble_.point_endpoints_per_env = cap.point_endpoints_per_env;
        p_assemble_.point_endpoint_terms_per_env = cap.point_endpoint_terms_per_env;
        // Particle constraint rows sit just before the stress tail; their endpoints end the pool.
        const uint32_t constraint_rows = cap.dist_cons_per_env + cap.vol_cons_per_env;
        const uint32_t constraint_endpoints = cap.vol_cons_per_env;
        p_assemble_.dist_cons_per_env = cap.dist_cons_per_env;
        p_assemble_.vol_cons_per_env = cap.vol_cons_per_env;
        p_assemble_.particle_constraint_row_first = cap.max_rows_per_env -
            cap.mpm_stress_cells_per_env * kMpmStressRowsPerCell - constraint_rows;
        p_assemble_.constraint_endpoint_first = cap.point_endpoints_per_env - constraint_endpoints;
        p_assemble_.constraint_term_first = cap.point_endpoint_terms_per_env - 4u * constraint_endpoints;
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
        p_assemble_.contact_rows_per_env = contact_rows_per_env;
        p_assemble_.joint_limit_rows_per_env = cap.joint_limit_rows_per_env;
        p_assemble_.joint_friction_rows_per_env = cap.joint_friction_rows_per_env;
        p_assemble_.joint_drive_rows_per_env = cap.joint_drive_rows_per_env;
        p_assemble_.mimic_couplings_per_env = mimic_couplings;
        // Layout follows the slot provider, not the particle solver mode: rigid
        // candidates are 4-point manifolds, the reserved sphere-particle tail is
        // one point. With no reserve rigid_cap == the full slot stride.
        p_assemble_.full_row_slot_count = rigid_cap;
        p_assemble_.ogc_slot_count = cap.ogc_contacts_per_env;
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
        p_assemble_.baumgarte_max_velocity = model.baumgarte_max_velocity;
        p_assemble_.contact_margin = cfg.contact_margin;
        p_assemble_.position_pass =
            (!use_block_descent_ && family == phi::kContactFamilyPairDriven && cfg.pos_iters > 0u) ? 1u : 0u;
        p_assemble_.vertex_blocks = vertex_blocks;
        if constexpr (family == phi::kContactFamilyPairDriven) {
            p_warm_start_prepare_.phase = 0u;
            p_warm_start_prepare_.env_count = env_count;
            p_warm_start_prepare_.slot_count = cap.max_contacts_per_env;
            p_warm_start_prepare_.rows_per_env = cap.max_rows_per_env;
            p_warm_start_prepare_.full_row_slot_count = rigid_cap;
            p_warm_start_prepare_.decay_steps = 2u;
            p_warm_start_prepare_.workspace_bytes = cap.contact_cache_scratch_bytes;
        }
        add(phi::NkOp::AssembleRows, &p_assemble_);
        if constexpr (family == phi::kContactFamilyPairDriven) {
            add(phi::NkOp::ContactWarmStart, &p_warm_start_prepare_);
        }
    }

    energy(EnergyStage::Free);
    if (has_contacts) {
        if (use_block_descent_) {
            if (has_particles) row_coupling_provider_.Couple(coupling_ctx);
            add(phi::NkOp::BlockDescentSolve, &p_block_descent_);
            if (has_particles && p_part_contact_delta_.active_begin_per_env < per_env_particles)
                add(phi::NkOp::ParticleContactDelta, &p_part_contact_delta_);
        } else {
            p_solve_.dt = cfg.dt;
            p_solve_.vel_iters = cfg.vel_iters;
            p_solve_.vel_tolerance = cfg.vel_tolerance;
            // The general contact solve projects overlap using split-impulse position sweeps.
            p_solve_.pos_iters =
                (family == phi::kContactFamilyPairDriven) ? cfg.pos_iters : 0u;
            p_solve_.pos_beta = cfg.pos_beta;
            p_solve_.pos_slop = cfg.pos_slop;
            p_solve_.total_particle_count = particle_count;
            p_solve_.total_grid_count = cap.mpm_grid_nodes_per_env * env_count;
            p_solve_.workspace_bytes = cap.solver_velocity_scratch_bytes;
            p_solve_.family = family;
            p_solve_.total_islands = model.schedule_island_count;
            p_solve_.max_dof = max_dof;
            p_solve_.env_count = env_count;
            p_solve_.articulation_count = articulation_cnt;
            p_solve_.rows_per_env = cap.max_rows_per_env;
            p_solve_.contact_rows_per_env = contact_rows_per_env;
            p_solve_.joint_limit_rows_per_env = cap.joint_limit_rows_per_env;
            p_solve_.base_link_count = base_link_count;
            p_solve_.total_body_count = cap.bodies_per_env * env_count;
            p_solve_.friction_coefficient = model.friction_coefficient;
            p_solve_.baumgarte_max_velocity = model.baumgarte_max_velocity;
            // The per-articulation slot stride MUST match the detection/assembly
            // (slot_base = articulation * stride). Data-driven; 4 at K<=1.
            p_solve_.contact_slots_per_artic = contact_slots_per_artic;
            p_solve_.vertex_blocks = vertex_blocks;
            p_solve_.mimic_couplings = mimic_couplings;
            // Validation A/B: force the cook-time static schedule (the byte-identity
            // reference for the dynamic islanding). Read once at Build (graph-safe).
            const char* fsi = std::getenv("NUKA_FORCE_STATIC_ISLANDS");
            p_solve_.force_static_islands = (fsi != nullptr && fsi[0] == '1') ? 1u : 0u;
            // Validation A/B: sweep the convergence bound without recooking a scene.
            const char* tolerance = std::getenv("NUKA_SOLVER_VEL_TOLERANCE");
            if (tolerance != nullptr) p_solve_.vel_tolerance = std::strtof(tolerance, nullptr);
            const char* diagnostics = std::getenv("NUKA_CONTACT_SOLVER_DIAGNOSTICS");
            p_solve_.measure_vertex_audit = (readout_demand & kReadoutVbdSolveAudit) != 0u;
            p_solve_.measure_contact_residual = cfg.measure_contact_residual ||
                p_solve_.measure_vertex_audit != 0u ||
                (readout_demand & kReadoutPhysicsDiagnostics) != 0u ||
                (diagnostics != nullptr && diagnostics[0] == '1');
            // Particle rows exchange their impulses in the common solve.
            if (has_particles) {
                row_coupling_provider_.Couple(coupling_ctx);
            }
            // The active rows define independent solve islands after assembly.
            p_islands_.family = family;
            p_islands_.env_count = env_count;
            p_islands_.rows_per_env = cap.max_rows_per_env;
            p_islands_.articulation_count = articulation_cnt;
            p_islands_.bodies_per_env = cap.bodies_per_env;
            p_islands_.particles_per_env = cap.particles_per_env;
            p_islands_.grid_nodes_per_env = cap.mpm_grid_nodes_per_env;
            add(phi::NkOp::BuildSolveIslands, &p_islands_);
            // Rows the screened sweeps leave idle are rechecked against the solved velocity by a
            // final pass, which sweeps again only if one broke and then projects positions.
            const bool verify_idle =
                family == phi::kContactFamilyPairDriven && p_solve_.force_static_islands == 0u;
            for (uint32_t pass = 0u; pass < coupling_iterations; ++pass) {
                if (pass != 0u) project_particles(pass);
                const bool last = pass + 1u == coupling_iterations;
                auto& solve = p_solve_iterations_[pass];
                solve = p_solve_;
                solve.vel_iters = static_cast<uint16_t>(iteration_work(pass, cfg.vel_iters, coupling_iterations));
                solve.pos_iters = last && !verify_idle ? p_solve_.pos_iters : 0u;
                solve.position_later = last && verify_idle && p_solve_.pos_iters != 0u ? 1u : 0u;
                solve.measure_contact_residual =
                    last && !verify_idle ? p_solve_.measure_contact_residual : 0u;
                solve.measure_vertex_audit =
                    last && !verify_idle ? p_solve_.measure_vertex_audit : 0u;
                solve.continue_impulses = pass != 0u ? 1u : 0u;
                add(phi::NkOp::SolveRowsBlockIsland, &solve);
                if (last && verify_idle) {
                    p_solve_verify_ = p_solve_;
                    p_solve_verify_.vel_iters = static_cast<uint16_t>((cfg.vel_iters + 3u) / 4u);
                    p_solve_verify_.continue_impulses = 1u;
                    p_solve_verify_.verify_idle = 1u;
                    add(phi::NkOp::SolveRowsBlockIsland, &p_solve_verify_);
                }
                if (has_particles && p_part_contact_delta_.active_begin_per_env < per_env_particles)
                    add(phi::NkOp::ParticleContactDelta, &p_part_contact_delta_);
            }
        }
        if constexpr (family == phi::kContactFamilyPairDriven) {
            p_warm_start_commit_ = p_warm_start_prepare_;
            p_warm_start_commit_.phase = 1u;
            add(phi::NkOp::ContactWarmStart, &p_warm_start_commit_);
        }
    } else {
        if (!use_block_descent_)
            for (uint32_t pass = 1u; pass < coupling_iterations; ++pass) project_particles(pass);
        if (use_block_descent_) add(phi::NkOp::BlockDescentSolve, &p_block_descent_);
    }

    if (cap.joint_drive_rows_per_env > 0u) {
        p_readout_drives_.dt = cfg.dt;
        p_readout_drives_.total_link_count = total_link_count;
        p_readout_drives_.links_per_env = base_link_count;
        p_readout_drives_.rows_per_env = cap.max_rows_per_env;
        p_readout_drives_.first_drive_row = contact_rows_per_env +
            cap.joint_limit_rows_per_env + cap.joint_friction_rows_per_env;
        add(phi::NkOp::ReadoutDrives, &p_readout_drives_);
    }

    energy(EnergyStage::Solved);
    mpm_coupling_provider_.PostCouple(coupling_ctx);

    if (cap.ogc_contacts_per_env > 0u) {
        p_dat_truncate_.dt = cfg.dt;
        p_dat_truncate_.margin = contact_margin;
        p_dat_truncate_.env_count = env_count;
        p_dat_truncate_.particles_per_env = cap.particles_per_env;
        p_dat_truncate_.vbd_particle_begin = cap.vbd_particle_begin;
        p_dat_truncate_.vbd_vertices_per_env = cap.vbd_vertices_per_env;
        p_dat_truncate_.surfaces_per_env = cap.particle_surfaces_per_env;
        p_dat_truncate_.triangles_per_env = cap.particle_surface_triangles;
        p_dat_truncate_.nodes_per_env = cap.particle_surface_nodes_per_env;
        p_dat_truncate_.edges_per_env = cap.particle_surface_edges_per_env;
        p_dat_truncate_.edge_nodes_per_env = cap.particle_surface_edge_nodes_per_env;
        p_dat_truncate_.bodies_per_env = cap.bodies_per_env;
        p_dat_truncate_.links_per_env = cap.links_per_env;
        p_dat_truncate_.articulations_per_env =
            has_articulation ? cap.articulations_per_env : 0u;
        p_dat_truncate_.max_dof = max_dof;
        p_dat_truncate_.mesh_vertices = cap.max_hull_verts;
        p_dat_truncate_.mesh_triangles = cap.max_mesh_triangles;
        p_dat_truncate_.mesh_nodes = cap.max_mesh_bvh_nodes;
        p_dat_truncate_.mesh_edges = cap.max_mesh_edges;
        p_dat_truncate_.mesh_edge_nodes = cap.max_mesh_edge_nodes;
        p_dat_truncate_.mesh_vertex_sources = cap.mesh_vertex_source_count;
        p_dat_truncate_.mesh_edge_sources = cap.mesh_edge_source_count;
        p_dat_truncate_.excluded_pairs = cap.max_excluded_pairs;
        p_dat_truncate_.slot_stride = cap.max_contacts_per_env;
        p_dat_truncate_.slot_base = rigid_cap;
        p_dat_truncate_.slot_capacity = cap.ogc_contacts_per_env;
        if (has_articulation || has_bodies)
            add(phi::NkOp::DatSnapshot, &p_dat_truncate_);
    }

    if (has_articulation || has_bodies) {
        p_int_pos_.dt = cfg.dt;
        p_int_pos_.total_link_count = total_link_count;
        p_int_pos_.articulation_count = articulation_cnt;
        p_int_pos_.total_body_count = cap.bodies_per_env * env_count;
        p_int_pos_.env_count = env_count;
        // Read the split-impulse pseudo velocity additively when the position pass
        // is active (the general PairDriven path); else velocity-only (identical).
        p_int_pos_.pos_pass =
            (!use_block_descent_ && has_contacts && family == phi::kContactFamilyPairDriven &&
             cfg.pos_iters > 0u) ? 1u : 0u;
        p_int_pos_.mimic_couplings = mimic_couplings;
        add(phi::NkOp::IntegratePosition, &p_int_pos_);
    }

    if (has_particles) row_coupling_provider_.PostCouple(coupling_ctx);

    if (cap.ogc_contacts_per_env > 0u) {
        if (has_articulation) add(phi::NkOp::FkWorldPoses, &p_fk_);
        if (has_collidables) add(phi::NkOp::SyncLinkBodyPose, &p_sync_body_pose_);
        if (cap.particle_surfaces_per_env > 0u) {
            p_dat_refit_ = p_particle_surfaces_;
            p_dat_refit_.position_source = 1u;
            add(phi::NkOp::RefitParticleSurfaces, &p_dat_refit_);
        }
        energy(EnergyStage::Projected);
        add(phi::NkOp::DatTruncate, &p_dat_truncate_);
    } else {
        energy(EnergyStage::Projected);
    }

    // Commit particle state once after all incremental coupling solves.
    if (has_particles) {
        if (cap.ogc_contacts_per_env > 0u) p_cloth_step_.truncation = 1u;
        if (vertex_blocks.vertices > 0u) add(phi::NkOp::ClothFinalize, &p_cloth_step_);
    }

    // Public poses and velocities describe the completed interval before sensor sampling.
    if (has_articulation) {
        add(phi::NkOp::FkWorldPoses, &p_fk_);
        p_fk_velocity_.articulation_count = articulation_cnt;
        p_fk_velocity_.total_link_count = total_link_count;
        add(phi::NkOp::FkLinkVelocities, &p_fk_velocity_);
    }
    if (has_collidables) add(phi::NkOp::SyncLinkBodyPose, &p_sync_body_pose_);

    energy(EnergyStage::End);

    // ReadoutContactWrench: the general per-env contact-wrench readout over the
    // unified PairDriven contact buffer. Pure readout — emitted only when a
    // consumer demanded its output fields (World flips the bit on first request
    // and rebuilds; unconsumed worlds skip the full row scan every step).
    if ((readout_demand & kReadoutContactWrench) != 0u &&
        has_contacts) {
        p_readout_.dt = cfg.dt;
        p_readout_.env_count = env_count;
        p_readout_.base_link_count = base_link_count;
        p_readout_.max_contacts_per_env = cap.max_contacts_per_env;
        p_readout_.rows_per_env = cap.max_rows_per_env;
        p_readout_.full_row_slot_count = rigid_cap;
        p_readout_.workspace_bytes = cap.contact_index_scratch_bytes;
        add(phi::NkOp::ReadoutContactWrench, &p_readout_);
    }

    if (!missing_ops_.empty()) {
        calls_.clear();
        return phi::Status::Unsupported;
    }
    return phi::Status::Ok;
}

} // namespace nuka::nk
