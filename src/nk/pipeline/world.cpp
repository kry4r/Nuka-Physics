// World owns model, state, and the common physics operator sequence.

#include "nk/pipeline/world.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <utility>

#include "nk/solve/schedule.hpp"
#include "nk/solve/nk_row.hpp"
#include "nk/solve/xpbd_coloring.hpp"

namespace nuka::nk {

namespace {

phi::Status DispatchChecked(phi::Backend* backend, const phi::ModelView& model,
                            const phi::DataView& data, const phi::OpCall& call,
                            phi::ExecutionError* error) {
    phi::Status status;
    try {
        status = phi::BackendDispatch(backend, model, data, call);
        if (status == phi::Status::Ok) return status;
        std::snprintf(error->message, sizeof(error->message),
                      "operator %u failed during eager execution", static_cast<unsigned>(call.op));
    } catch (const std::bad_alloc& exception) {
        status = phi::Status::OutOfMemory;
        std::snprintf(error->message, sizeof(error->message), "%s", exception.what());
    } catch (const std::exception& exception) {
        status = phi::Status::Failed;
        std::snprintf(error->message, sizeof(error->message), "%s", exception.what());
    } catch (...) {
        status = phi::Status::Failed;
        std::snprintf(error->message, sizeof(error->message), "unknown operator exception");
    }
    error->status = status;
    error->failed_op = call.op;
    return status;
}

}  // namespace

World::World(Model model, uint32_t env_count, phi::Device* device,
             phi::Backend* backend, const Pipeline::SolverConfig& cfg)
    : model_(std::move(model)), cfg_(cfg), device_(device), backend_(backend) {
    if (device == nullptr || backend == nullptr) {
        creation_error_ = "a live device and backend are required";
        return;
    }
    if (env_count > 0) {
        model_.capacities.env_count = env_count;
    }
    creation_status_ = model_.ValidateTopology(&creation_error_);
    if (creation_status_ != phi::Status::Ok) return;
    if (!(cfg_.dt > 0.0f) || !std::isfinite(cfg_.dt) ||
        !std::isfinite(cfg_.gravity[0]) || !std::isfinite(cfg_.gravity[1]) ||
        !std::isfinite(cfg_.gravity[2])) {
        creation_status_ = phi::Status::InvalidArgument;
        creation_error_ = "timestep and gravity must be finite with positive timestep";
        return;
    }
    creation_status_ = pipeline_->Build(model_, cfg_, device_, readout_demand_);
    for (phi::NkOp op : pipeline_->MissingOps())
        creation_error_ += "missing required op " + std::to_string(static_cast<uint32_t>(op)) + "; ";
    std::vector<phi::NkOp> state_ops{
        phi::NkOp::SnapshotState, phi::NkOp::RestoreState, phi::NkOp::ResetEnvs};
    if (model_.capacities.links_per_env > 0u) state_ops.push_back(phi::NkOp::FkWorldPoses);
    if (model_.capacities.bodies_per_env > 0u) state_ops.push_back(phi::NkOp::SyncLinkBodyPose);
    for (phi::NkOp op : state_ops) {
        if (!phi::DeviceSupportsOp(device_, op)) {
            creation_status_ = phi::Status::Unsupported;
            creation_error_ += "missing state op " + std::to_string(static_cast<uint32_t>(op)) + "; ";
        }
    }
    if (creation_status_ != phi::Status::Ok) return;
    creation_status_ = phi::Status::Failed;

    // Size the particle-grid sort/scan scratch so ParticleGridBuild captures into
    // the graph (no mid-capture cudaMalloc); 0 for a particle-free world (inert).
    try {
    const bool runs_pbf =
        model_.particles.mode == Model::ParticleMode::Pbf ||
        model_.particles.mode == Model::ParticleMode::SoftFluid ||
        (model_.particles.mode == Model::ParticleMode::Coupled &&
         model_.particles.coupled_internal == Model::CoupledInternal::Pbf);
    const uint32_t particle_grid_count = runs_pbf
        ? model_.capacities.particles_per_env * model_.capacities.env_count : 0u;
    if (particle_grid_count > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        creation_status_ = phi::Status::InvalidArgument;
        creation_error_ = "particle grid exceeds the sort item limit";
        return;
    }
    model_.capacities.grid_sort_scratch_bytes =
        phi::GridSortScratchBytes(particle_grid_count);

    // Size the MLS-MPM P2G deterministic-gather scratch the same way; 0 for a
    // non-MPM world (zero-byte segment, byte-inert).
    const uint32_t mpm_per_env = model_.MpmParticlesPerEnv();
    const uint64_t mpm_particle_sort_count =
        static_cast<uint64_t>(mpm_per_env) * model_.capacities.env_count;
    // The same workspace also compacts active P2G nodes and, when dynamic bodies
    // are enabled, stably groups projected nodes by owner for the reaction gather.
    // Both node phases need capacity for the full env-private grid.
    const uint64_t mpm_node_sort_count = mpm_per_env > 0u
        ? static_cast<uint64_t>(model_.capacities.mpm_grid_nodes_per_env) *
              model_.capacities.env_count
        : 0u;
    const uint64_t mpm_sort_count =
        mpm_particle_sort_count > mpm_node_sort_count
            ? mpm_particle_sort_count : mpm_node_sort_count;
    model_.capacities.mpm_grid_sort_scratch_bytes =
        mpm_sort_count <= static_cast<uint64_t>(std::numeric_limits<int>::max())
            ? phi::MpmSortScratchBytes(static_cast<uint32_t>(mpm_sort_count))
            : 0u;  // CUB sort/select expose an int num_items contract.

    // Size the dynamic-island cub radix-sort scratch (BuildSolveIslands) over the
    // total row capacity so the sort captures into the graph; 0 if no rows (inert).
    model_.capacities.island_cub_temp_bytes = phi::IslandSortScratchBytes(
        model_.capacities.max_rows_per_env * model_.capacities.env_count);

    // Size the broadphase pair-stream canonicalization scratch (LbvhQueryPairs)
    // over the rigid candidate-slot capacity; 0 for a non-PairDriven world (the
    // op early-exits there, so the segment stays zero-byte / byte-inert).
    const bool runs_pair_driven =
        model_.contact_family == ContactFamily::PairDriven;
    const uint64_t contact_slots =
        runs_pair_driven ? static_cast<uint64_t>(
            model_.capacities.max_contacts_per_env) * model_.capacities.env_count : 0u;
    uint64_t pair_sort_slots = 0u;
    for (const auto& call : pipeline_->Calls()) {
        if (call.op == phi::NkOp::LbvhQueryPairs) {
            const auto* query = static_cast<const phi::LbvhQueryPairsParams*>(call.params);
            pair_sort_slots = uint64_t{query->env_count} * query->rigid_slot_cap;
        }
    }
    model_.capacities.pair_sort_scratch_bytes =
        (pair_sort_slots > 0u &&
         pair_sort_slots <= static_cast<uint64_t>(std::numeric_limits<int>::max()))
            ? phi::PairSortScratchBytes(static_cast<uint32_t>(pair_sort_slots),
                                        model_.capacities.env_count)
            : 0u;  // CUB sort/scan expose an int num_items contract.

    model_.capacities.lbvh_sort_scratch_bytes = runs_pair_driven
        ? phi::LbvhSortScratchBytes(model_.capacities.env_count, model_.capacities.bodies_per_env) : 0u;
    model_.capacities.contact_cache_scratch_bytes = runs_pair_driven
        ? phi::ContactCacheScratchBytes(static_cast<uint32_t>(contact_slots * kPairDrivenPtsPerSlot),
                                        model_.capacities.env_count) : 0u;
    const auto& cap = model_.capacities;
    model_.capacities.contact_index_scratch_bytes = cap.links_per_env > 0u && cap.max_contacts_per_env > 0u
        ? phi::ContactIndexScratchBytes(cap.max_rows_per_env * cap.env_count, cap.env_count) : 0u;
    } catch (const std::exception& error) {
        creation_status_ = phi::Status::Failed;
        creation_error_ = error.what();
        return;
    }
    creation_status_ = model_.capacities.Validate(&creation_error_);
    if (creation_status_ != phi::Status::Ok) return;

    // The init-time buffer type (stream-less default-stream device type — fine
    // for the one-shot Model upload + Arena alloc, per the plan's note).
    phi::BufferType* bt = phi::DeviceBufferType(device);
    if (bt == nullptr) {
        creation_status_ = phi::Status::Failed;
        creation_error_ = "device has no buffer type";
        return;
    }

    // 0. (M4, plan §3.4) Build the worst-case device-resident solve schedule
    // ONCE on the host (the row_scheduler algorithms migrated to nk/solve) —
    // BEFORE UploadTo, which stages the triple into the Model device buffer.
    // Runtime steps never re-color (the plan-replay anchor).
    SolveSchedule::Build(&model_);

    // 0b. Graph-color the XPBD constraint families ONCE (a general solver
    // property): reorder each family into color-contiguous order so the device
    // projects a color's constraints in parallel (no shared particle => race-
    // free, no atomics) while colors run in fixed order — deterministic.
    XpbdColoring::Build(&model_);

    // 1. Upload the Model's constant tables into ONE device buffer + fill view.
    creation_status_ = model_.UploadTo(bt, &model_view_);
    if (creation_status_ != phi::Status::Ok) {
        creation_error_ = "model upload failed";
        return;
    }

    // 2. Allocate the Data arena + fill the data view.
    creation_status_ = data_.Allocate(bt, model_.capacities, &data_view_);
    if (creation_status_ != phi::Status::Ok) {
        creation_error_ = "state arena allocation failed";
        return;
    }

    // Coloring finalizes the model's constraint counts before parameter binding.
    creation_status_ = pipeline_->Build(model_, cfg_, device, readout_demand_);
    if (creation_status_ != phi::Status::Ok) return;

    ready_ = true;

    // 4. Seed the initial state (q / poses / drives) + take the device snapshot
    // that backs Reset. Failure leaves the World unbuilt (honest).
    if (!SeedInitialState()) {
        ready_ = false;
        creation_status_ = phi::Status::Failed;
        creation_error_ = "initial state or snapshot dispatch failed";
    }
}

bool World::SeedInitialState() {
    const ModelCapacities& cap = model_.capacities;
    const uint32_t L = cap.links_per_env;
    const uint32_t E = cap.env_count;

    // Snapshot/restore carry total counts; masked reset carries environment strides.
    const uint32_t total_body_count = cap.bodies_per_env * E;
    const uint32_t total_particle_count = cap.particles_per_env * E;
    snapshot_params_.total_link_count = L * E;
    snapshot_params_.env_count = E;
    snapshot_params_.articulation_count = cap.articulations_per_env * E;
    snapshot_params_.total_body_count = total_body_count;
    snapshot_params_.total_particle_count = total_particle_count;
    restore_params_.total_link_count = L * E;
    restore_params_.env_count = E;
    restore_params_.articulation_count = cap.articulations_per_env * E;
    restore_params_.dofs_per_articulation = cap.dofs_per_env;
    restore_params_.row_slot_count = cap.max_rows_per_env * E;
    restore_params_.contact_slot_count = cap.max_contacts_per_env * E;
    restore_params_.total_body_count = total_body_count;
    restore_params_.total_particle_count = total_particle_count;
    reset_params_.count = 0;
    reset_params_.env_count = E;
    reset_params_.base_link_count = L;
    reset_params_.lambda_stride = cap.max_rows_per_env;
    reset_params_.contact_slot_count = cap.max_contacts_per_env;
    // Each state category uses its own environment stride.
    reset_params_.articulation_count = cap.articulations_per_env * E;
    reset_params_.articulations_per_env = cap.articulations_per_env;
    reset_params_.dofs_per_articulation = cap.dofs_per_env;
    reset_params_.use_env_ids = 1u;
    reset_params_.body_count = cap.bodies_per_env;
    // Restored joint state immediately refreshes the selected FK poses.
    fk_params_.articulation_count = cap.articulations_per_env * E;
    fk_params_.total_link_count = L * E;
    fk_params_.articulations_per_env = cap.articulations_per_env;
    reset_params_.particle_count = cap.particles_per_env;
    reset_params_.has_particle_grid = cap.grid_sort_scratch_bytes != 0u;
    restore_params_.has_particle_grid = reset_params_.has_particle_grid;

    // -- M4: movable rigid-body template seeding (env-major replication) -----
    const uint32_t B = cap.bodies_per_env;
    if (B > 0 && !model_.body_init.empty()) {
        std::vector<math::Transform> poses(static_cast<size_t>(B) * E);
        std::vector<math::Vec3> lin(static_cast<size_t>(B) * E);
        std::vector<math::Vec3> ang(static_cast<size_t>(B) * E);
        std::vector<float> inv_mass(static_cast<size_t>(B) * E, 0.0f);
        std::vector<math::Vec3> inv_inertia(static_cast<size_t>(B) * E);
        std::vector<math::Transform> inertial_frame(static_cast<size_t>(B) * E);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t b = 0; b < B; ++b) {
                const Model::BodyInit& src =
                    b < model_.body_init.size() ? model_.body_init[b]
                                                : Model::BodyInit{};
                const size_t at = static_cast<size_t>(e) * B + b;
                poses[at] = src.pose;
                lin[at] = src.linear_velocity;
                ang[at] = src.angular_velocity;
                inv_mass[at] = src.inv_mass;
                inv_inertia[at] = src.inv_inertia;
                inertial_frame[at] = src.inertial_frame;
            }
        }
        if (!data_.UploadField(FieldId::BodyPose, poses.data(),
                               poses.size() * sizeof(math::Transform)) ||
            !data_.UploadField(FieldId::BodyLinearVelocity, lin.data(),
                               lin.size() * sizeof(math::Vec3)) ||
            !data_.UploadField(FieldId::BodyAngularVelocity, ang.data(),
                               ang.size() * sizeof(math::Vec3)) ||
            !data_.UploadField(FieldId::BodyInvMass, inv_mass.data(),
                               inv_mass.size() * sizeof(float)) ||
            !data_.UploadField(FieldId::BodyInvInertia, inv_inertia.data(),
                               inv_inertia.size() * sizeof(math::Vec3)) ||
            !data_.UploadField(FieldId::BodyInertialFrame, inertial_frame.data(),
                               inertial_frame.size() * sizeof(math::Transform))) {
            return false;
        }
    }

    // L1-c: the M4 union per-env live table toggle (FieldId::TableEnabled) seed
    // was DELETED here — both the field and table_enabled_default are gone.

    // -- The per-env terrain TYPE field, seeded 0 (flat) for EVERY env. The general
    // path's terrain is the cook-time baked heightfield collidable, so this per-env
    // type is informational; a harness may set it post-construction via
    // World::GetData().UploadField(FieldId::EnvTerrainType, ...). Persistent so it
    // round-trips Reset (the construction-time snapshot).
    {
        std::vector<uint32_t> terrain_type(E, 0u);
        if (!data_.UploadField(FieldId::EnvTerrainType, terrain_type.data(),
                               terrain_type.size() * sizeof(uint32_t))) {
            return false;
        }
    }

    // -- Go2-on-stairs Phase 2a: the per-env procedural-terrain DIFFICULTY scale.
    // Seeded 1.0 (the unscaled terrain) for EVERY env. (Historically the FUSED foot
    // kernel multiplied the terrain step_height + grid_height_max by this scale
    // before sampling per env; L1-b deletes that runtime, so the field is now
    // model-level / informational alongside the cook-time baked heightfield.)
    // The DEFAULT (1.0) keeps the unscaled terrain,
    // and for the type-0 (Flat) seed SampleTerrainHeight ignores step/grid so the
    // multiply is computed-but-unused => byte-identical to the legacy flat path
    // (D1). A training harness sets it post-construction via
    // World::GetData().UploadField(FieldId::EnvTerrainDifficulty, ...). Persistent
    // so it round-trips Reset (the construction-time snapshot).
    {
        std::vector<float> terrain_difficulty(E, 1.0f);  // 1.0 == unscaled
        if (!data_.UploadField(FieldId::EnvTerrainDifficulty,
                               terrain_difficulty.data(),
                               terrain_difficulty.size() * sizeof(float))) {
            return false;
        }
    }

    // -- M6: particle (XPBD soft / PBF fluid) initial state seeding (env-major
    // replication of the single-env template). particle_prev_pos is seeded == pos
    // (the legacy soft-upload "prev seeded = p" / fluid-upload "predicted
    // seeded = p"). The XPBD/PBF constraint lambdas + the PBF scratch stay at the
    // arena's zero init (matching the legacy zero-seed).
    if (cap.particles_per_env > 0 &&
        model_.particles.mode != Model::ParticleMode::None) {
        const Model::ModelParticles& mp = model_.particles;
        const uint32_t P = cap.particles_per_env;
        std::vector<math::Vec3> pos(static_cast<size_t>(P) * E);
        std::vector<math::Vec3> vel(static_cast<size_t>(P) * E);
        std::vector<float> inv_mass(static_cast<size_t>(P) * E, 0.0f);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t i = 0; i < P; ++i) {
                const size_t at = static_cast<size_t>(e) * P + i;
                pos[at] = i < mp.initial_pos.size() ? mp.initial_pos[i]
                                                    : math::Vec3::Zero();
                vel[at] = i < mp.initial_vel.size() ? mp.initial_vel[i]
                                                    : math::Vec3::Zero();
                inv_mass[at] = i < mp.inv_mass.size() ? mp.inv_mass[i] : 0.0f;
            }
        }
        if (!data_.UploadField(FieldId::ParticlePos, pos.data(),
                               pos.size() * sizeof(math::Vec3)) ||
            !data_.UploadField(FieldId::ParticlePrevPos, pos.data(),
                               pos.size() * sizeof(math::Vec3)) ||
            !data_.UploadField(FieldId::ParticleVel, vel.data(),
                               vel.size() * sizeof(math::Vec3)) ||
            !data_.UploadField(FieldId::ParticleInvMass, inv_mass.data(),
                               inv_mass.size() * sizeof(float))) {
            return false;
        }
        // MLS-MPM per-particle continuum seed (F=identity, vol0, material_id),
        // gated to the MPM modes so a non-MPM world writes EXACTLY today's bytes (the
        // new fields stay at their 0-byte segments). C is the arena zero default. For
        // MpmXpbd initial_F/vol0/material_id are sized to the MPM slice [0, n_mpm), so
        // the loop's identity/0 fallback fills the XPBD slice (unused by the transfer).
        if (mp.mode == Model::ParticleMode::Mpm ||
            mp.mode == Model::ParticleMode::MpmXpbd) {
            std::vector<float> F(static_cast<size_t>(P) * E * 9u, 0.0f);
            std::vector<float> vol0(static_cast<size_t>(P) * E, 0.0f);
            std::vector<uint32_t> mat(static_cast<size_t>(P) * E, 0u);
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t i = 0; i < P; ++i) {
                    const size_t at = static_cast<size_t>(e) * P + i;
                    float* f = F.data() + at * 9u;
                    if ((static_cast<size_t>(i) + 1u) * 9u <= mp.initial_F.size()) {
                        for (uint32_t k = 0; k < 9u; ++k) f[k] = mp.initial_F[i * 9u + k];
                    } else {
                        f[0] = 1.0f; f[4] = 1.0f; f[8] = 1.0f;  // identity fallback.
                    }
                    vol0[at] = i < mp.initial_vol0.size() ? mp.initial_vol0[i] : 0.0f;
                    mat[at] = i < mp.initial_material_id.size()
                                  ? mp.initial_material_id[i] : 0u;
                }
            }
            if (!data_.UploadField(FieldId::ParticleF, F.data(),
                                   F.size() * sizeof(float)) ||
                !data_.UploadField(FieldId::ParticleVol0, vol0.data(),
                                   vol0.size() * sizeof(float)) ||
                !data_.UploadField(FieldId::ParticleMaterialId, mat.data(),
                                   mat.size() * sizeof(uint32_t))) {
                return false;
            }
        }
    }

    // MLS-MPM material table (global; data-owned, flat f32 pool like mat_buckets).
    // 0 materials -> the segment is zero bytes and this block no-ops (byte-inert).
    if (cap.mpm_material_count > 0 && !model_.mpm_materials.empty()) {
        const uint32_t stride = MpmMaterial::kValueCount;
        std::vector<float> host(static_cast<size_t>(cap.mpm_material_count) * stride, 0.0f);
        for (uint32_t r = 0; r < cap.mpm_material_count &&
                             r < model_.mpm_materials.size(); ++r) {
            const MpmMaterial& mm = model_.mpm_materials[r];
            float* dst = host.data() + static_cast<size_t>(r) * stride;
            dst[0] = mm.youngs; dst[1] = mm.poisson; dst[2] = mm.density;
            dst[3] = mm.dp_friction; dst[4] = mm.dp_cohesion; dst[5] = mm.model_kind;
            dst[6] = mm.bulk_modulus; dst[7] = mm.tait_gamma; dst[8] = mm.viscosity;
        }
        if (!data_.UploadField(FieldId::MpmMaterialTable, host.data(),
                               host.size() * sizeof(float))) {
            return false;
        }
    }

    // Material bucket table (global; data-owned param field). Seeded for EVERY
    // world that has buckets — including articulation-less (bodies/particles-
    // only) worlds, which return early below. (Review fix: this block used to
    // sit below the L==0 early return, silently skipping the bucket seed for
    // any world without an articulation — the M5-binding-bug class.)
    if (cap.num_material_buckets > 0 && !model_.material_buckets.empty()) {
        const uint32_t stride = ModelMaterialBucket::kValueCount;
        std::vector<float> host(static_cast<size_t>(cap.num_material_buckets) * stride,
                                0.0f);
        for (uint32_t b = 0; b < cap.num_material_buckets &&
                             b < model_.material_buckets.size(); ++b) {
            ModelMaterialBucket canonical;
            if (!CanonicalizeContactProfileForUpload(model_.material_buckets[b], &canonical))
                return false;
            for (uint32_t k = 0u; k < stride; ++k) {
                host[static_cast<size_t>(b) * stride + k] = canonical.values[k];
            }
        }
        if (!data_.UploadField(FieldId::MatBuckets, host.data(),
                               host.size() * sizeof(float))) {
            return false;
        }
    }
    // Per-body material bucket index (data-owned param field; the cooked
    // Model::body_material_bucket template replicated env-major). (Review fix:
    // the cook fills body_material_bucket and the mat_index field is declared,
    // bound, and dlpack-visible, but NOTHING ever uploaded the cooked values —
    // the M5-binding-bug class. No op consumes it yet; seeding closes the hole
    // before the first consumer lands.)
    if (B > 0 && !model_.body_material_bucket.empty()) {
        std::vector<uint32_t> host(static_cast<size_t>(B) * E, 0u);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t b = 0; b < B; ++b) {
                host[static_cast<size_t>(e) * B + b] =
                    b < model_.body_material_bucket.size()
                        ? model_.body_material_bucket[b] : 0u;
            }
        }
        if (!data_.UploadField(FieldId::MatIndex, host.data(),
                               host.size() * sizeof(uint32_t))) {
            return false;
        }
    }

    if (L == 0) {
        if (RefreshPoses(0u) != phi::Status::Ok) return false;
        return DispatchOp(phi::NkOp::SnapshotState, &snapshot_params_) ==
               phi::Status::Ok;
    }

    const ModelArticulation& a = model_.articulation;
    // env-major replication of the per-link template arrays (the EXACT layout
    // ReplicateArticulationHostState tiles; link_velocity / qdot / qddot / tau
    // stay at the arena's zero init, matching the host-state build).
    auto replicate_f32 = [&](FieldId id, const std::vector<float>& tpl) {
        std::vector<float> host(static_cast<size_t>(L) * E, 0.0f);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t l = 0; l < L && l < tpl.size(); ++l) {
                host[static_cast<size_t>(e) * L + l] = tpl[l];
            }
        }
        return data_.UploadField(id, host.data(), host.size() * sizeof(float));
    };
    if (!replicate_f32(FieldId::Q, a.initial_q) ||
        !replicate_f32(FieldId::DriveTarget, model_.hold_drives.targets) ||
        !replicate_f32(FieldId::DriveStiffness, model_.hold_drives.stiffness) ||
        !replicate_f32(FieldId::DriveDamping, model_.hold_drives.damping) ||
        !replicate_f32(FieldId::DriveForceLimit, model_.hold_drives.force_limits)) {
        return false;
    }
    // M4 (union family): the SETTLED initial velocity state (the legacy
    // factory's pre-roll product). Empty templates keep the zero-velocity
    // arena init (the M3 path, byte-unchanged).
    if (!a.initial_qdot.empty() && !replicate_f32(FieldId::Qdot, a.initial_qdot)) {
        return false;
    }
    if (!a.initial_link_velocity.empty()) {
        // Spatial-vector element count, derived from the Spatial6 layout so the
        // per-link stride cannot drift from the field type.
        constexpr size_t kSpatialDim = sizeof(Spatial6) / sizeof(float);
        std::vector<float> host(static_cast<size_t>(L) * E * kSpatialDim, 0.0f);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t l = 0; l < L; ++l) {
                for (size_t k = 0; k < kSpatialDim; ++k) {
                    const size_t src = static_cast<size_t>(l) * kSpatialDim + k;
                    if (src < a.initial_link_velocity.size()) {
                        host[(static_cast<size_t>(e) * L + l) * kSpatialDim + k] =
                            a.initial_link_velocity[src];
                    }
                }
            }
        }
        if (!data_.UploadField(FieldId::LinkVelocity, host.data(),
                               host.size() * sizeof(float))) {
            return false;
        }
    }
    {
        std::vector<math::Transform> host(static_cast<size_t>(L) * E);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t l = 0; l < L; ++l) {
                host[static_cast<size_t>(e) * L + l] =
                    l < a.initial_link_pose.size() ? a.initial_link_pose[l]
                                                   : math::Transform::Identity();
            }
        }
        if (!data_.UploadField(FieldId::LinkPose, host.data(),
                               host.size() * sizeof(math::Transform))) {
            return false;
        }
    }
    {
        // Each environment stores K roots at e*K+k, matching FK indexing.
        const uint32_t K = cap.articulations_per_env;
        std::vector<math::Transform> host(static_cast<size_t>(K) * E);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t k = 0; k < K; ++k) {
                host[static_cast<size_t>(e) * K + k] =
                    k < a.base_poses.size() ? a.base_poses[k] : a.base_pose;
            }
        }
        if (!data_.UploadField(FieldId::BasePose, host.data(),
                               host.size() * sizeof(math::Transform))) {
            return false;
        }
    }
    // Creation and reset expose the same FK and collision proxy poses.
    if (RefreshPoses(0u) != phi::Status::Ok) return false;
    return DispatchOp(phi::NkOp::SnapshotState, &snapshot_params_) == phi::Status::Ok;
}

World::~World() {
    if (plan_ != nullptr && backend_ != nullptr) {
        phi::BackendPlanFree(backend_, plan_);
        plan_ = nullptr;
    }
}

StepResult World::Step() {
    StepResult out;
    execution_error_ = {};
    if (!ready_ || backend_ == nullptr) {
        execution_error_.status = last_status_ = phi::Status::Failed;
        std::snprintf(execution_error_.message, sizeof(execution_error_.message), "world is not ready");
        return out;
    }
    const std::vector<phi::OpCall>& calls = pipeline_->Calls();
    out.result = phi::Status::Ok;
    out.status.reserve(calls.size());
    for (const phi::OpCall& call : calls) {
        const phi::Status s = DispatchChecked(backend_, model_view_, data_view_, call, &execution_error_);
        out.status.push_back(s);
        if (s != phi::Status::Ok) {
            out.result = s;
            out.failed_op = call.op;
            break;
        }
    }
    last_status_ = out.result;
    return out;
}

phi::Status World::PrepareGraph() {
    if (!ready_ || !backend_) return last_status_ = phi::Status::Failed;
    if (plan_ != nullptr) return phi::Status::Ok;
    if (!plan_attempted_) {
        plan_attempted_ = true;
        ++capture_attempts_;
        const auto& calls = pipeline_->Calls();
        plan_ = phi::BackendPlanCreate(backend_, model_view_, data_view_, calls.data(),
                                       static_cast<int>(calls.size()), &graph_error_);
        if (!plan_ && graph_error_.status == phi::Status::Ok) {
            graph_error_.status = phi::Status::Failed;
            std::snprintf(graph_error_.message, sizeof(graph_error_.message),
                          "backend returned no graph");
        }
    }
    execution_error_ = graph_error_;
    return last_status_ = graph_error_.status;
}

phi::Status World::StepPlanned() {
    const auto status = PrepareGraph();
    if (status != phi::Status::Ok) return status;
    last_status_ = phi::BackendPlanExecute(backend_, plan_, &execution_error_);
    if (last_status_ == phi::Status::Ok) ++graph_replays_;
    return last_status_;
}

phi::Status World::SetExecutionMode(ExecutionMode mode) {
    if (mode != ExecutionMode::Eager && mode != ExecutionMode::Graph)
        return last_status_ = phi::Status::InvalidArgument;
    if (mode == ExecutionMode::Graph) {
        const auto status = PrepareGraph();
        if (status != phi::Status::Ok) return status;
    }
    execution_mode_ = mode;
    return last_status_ = phi::Status::Ok;
}

phi::Status World::StepConfigured() {
    return execution_mode_ == ExecutionMode::Graph ? StepPlanned() : Step().result;
}

phi::Status World::Synchronize() {
    if (!ready_ || !backend_) return last_status_ = phi::Status::Failed;
    return last_status_ = phi::BackendSynchronize(backend_, &execution_error_);
}

phi::Status World::Reset(const std::vector<uint32_t>& env_ids) {
    if (!ready_ || backend_ == nullptr) return phi::Status::Failed;
    const uint32_t env_count = model_.capacities.env_count;
    for (uint32_t id : env_ids) {
        if (id >= env_count) return phi::Status::Failed;
    }
    std::vector<uint32_t> selected(env_ids);
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    if (!selected.empty() &&
        !data_.UploadField(FieldId::ResetEnvIds, selected.data(),
                           selected.size() * sizeof(uint32_t))) {
        return phi::Status::Failed;
    }
    ++reset_params_.ic_episode;
    reset_params_.count = static_cast<uint32_t>(selected.size());
    phi::Status status = selected.empty()
        ? DispatchOp(phi::NkOp::RestoreState, &restore_params_)
        : DispatchOp(phi::NkOp::ResetEnvs, &reset_params_);
    if (status != phi::Status::Ok) return status;
    return RefreshPoses(reset_params_.count);
}

phi::Status World::RefreshPoses(uint32_t selected_env_count) {
    fk_params_.selected_env_count = selected_env_count;
    if (fk_params_.total_link_count > 0u) {
        const auto status = DispatchOp(phi::NkOp::FkWorldPoses, &fk_params_);
        if (status != phi::Status::Ok) return status;
    }
    const auto& capacity = model_.capacities;
    if (capacity.bodies_per_env == 0u) return phi::Status::Ok;
    phi::SyncLinkBodyPoseParams params{};
    params.family = phi::kContactFamilyPairDriven;
    params.env_count = capacity.env_count;
    params.links_per_env = capacity.links_per_env;
    params.bodies_per_env = capacity.bodies_per_env;
    params.selected_env_count = selected_env_count;
    return DispatchOp(phi::NkOp::SyncLinkBodyPose, &params);
}

phi::Status World::DispatchOp(phi::NkOp op, const void* params) {
    if (!ready_ || backend_ == nullptr) {
        return phi::Status::Failed;
    }
    const phi::OpCall call{op, params};
    execution_error_ = {};
    return last_status_ = DispatchChecked(backend_, model_view_, data_view_, call, &execution_error_);
}

phi::Status World::DemandReadout(FieldId id) {
    uint32_t bit = 0u;
    // Every field OpReadoutContactWrench produces (geometry + {Fn,Ft1,Ft2} +
    // owning link + per-link wrench) shares the one readout bit.
    if (id == FieldId::LinkContactWrench || id == FieldId::ContactForce ||
        id == FieldId::ContactPoint || id == FieldId::ContactNormal ||
        id == FieldId::ContactLink || id == FieldId::ContactSideAKind ||
        id == FieldId::ContactSideBKind || id == FieldId::ContactSideAIndex ||
        id == FieldId::ContactSideBIndex) {
        bit = Pipeline::kReadoutContactWrench;
    }
    if (bit == 0u || (readout_demand_ & bit) != 0u) {
        return last_status_ = phi::Status::Ok;
    }
    auto candidate = std::make_unique<Pipeline>();
    last_status_ = candidate->Build(model_, cfg_, device_, readout_demand_ | bit);
    if (last_status_ != phi::Status::Ok) return last_status_;
    for (const phi::OpCall& call : candidate->Calls()) {
        if (call.op == phi::NkOp::ReadoutContactWrench) {
            execution_error_ = {};
            last_status_ = DispatchChecked(backend_, model_view_, data_view_, call, &execution_error_);
            if (last_status_ != phi::Status::Ok) return last_status_;
            break;
        }
    }
    if (plan_ != nullptr && backend_ != nullptr) {
        phi::BackendPlanFree(backend_, plan_);
        plan_ = nullptr;
    }
    plan_attempted_ = false;
    graph_error_ = {};
    pipeline_ = std::move(candidate);
    readout_demand_ |= bit;
    return last_status_ = phi::Status::Ok;
}

void* World::FieldPtr(FieldId id) const {
    if (!ready_) {
        return nullptr;
    }
    // Readout outputs are produced on demand: the first request turns the op on.
    if (const_cast<World*>(this)->DemandReadout(id) != phi::Status::Ok) return nullptr;
    const FieldLayout& lay = LayoutOf(id);
    if (lay.owner == FieldOwner::Model) {
        // Resolve from the model device buffer via its segment table.
        uint64_t total = 0;
        const std::vector<Model::Segment> segs = model_.ComputeModelSegments(&total);
        phi::Buffer* buf = model_.DeviceBuffer();
        if (buf == nullptr) {
            return nullptr;
        }
        for (const Model::Segment& s : segs) {
            if (s.field == id) {
                return static_cast<uint8_t*>(phi::BufferBase(buf)) + s.offset;
            }
        }
        return nullptr;
    }
    return data_.Ptr(id);
}

} // namespace nuka::nk
