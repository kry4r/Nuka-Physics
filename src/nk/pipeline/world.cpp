// ---------------------------------------------------------------------------
// nk::World implementation (plan §3.2 / M3).
// ---------------------------------------------------------------------------

#include "nk/pipeline/world.hpp"

#include <utility>

#include "nk/solve/schedule.hpp"

namespace nuka::nk {

World::World(Model model, uint32_t env_count, phi::Device* device,
             phi::Backend* backend, const Pipeline::SolverConfig& cfg)
    : model_(std::move(model)), backend_(backend) {
    if (device == nullptr || backend == nullptr) {
        return;
    }
    if (env_count > 0) {
        model_.capacities.env_count = env_count;
    }

    // The init-time buffer type (stream-less default-stream device type — fine
    // for the one-shot Model upload + Arena alloc, per the plan's note).
    phi::BufferType* bt = phi::DeviceBufferType(device);
    if (bt == nullptr) {
        return;
    }

    // 0. (M4, plan §3.4) Build the worst-case device-resident solve schedule
    // ONCE on the host (the row_scheduler algorithms migrated to nk/solve) —
    // BEFORE UploadTo, which stages the triple into the Model device buffer.
    // Runtime steps never re-color (the plan-replay anchor).
    SolveSchedule::Build(&model_);

    // 1. Upload the Model's constant tables into ONE device buffer + fill view.
    if (model_.UploadTo(bt, &model_view_) != phi::Status::Ok) {
        return;
    }

    // 2. Allocate the Data arena + fill the data view.
    if (data_.Allocate(bt, model_.capacities, &data_view_) != phi::Status::Ok) {
        return;
    }

    // 3. Build the per-step pipeline (OpCall list, §3.2 order; the device's
    // supports_op capability query drops not-yet-implemented ops).
    pipeline_.Build(model_, cfg, device);

    ready_ = true;

    // 4. Seed the initial state (q / poses / drives) + take the device snapshot
    // that backs Reset. Failure leaves the World unbuilt (honest).
    if (!SeedInitialState()) {
        ready_ = false;
    }
}

bool World::SeedInitialState() {
    const ModelCapacities& cap = model_.capacities;
    const uint32_t L = cap.links_per_env;
    const uint32_t E = cap.env_count;

    // Fill the reset/snapshot params (stable addresses for dispatch). The body
    // arm (M7 T1): SnapshotState/RestoreState carry the env-major TOTAL body
    // count (bodies_per_env*env_count); ResetEnvs carries the PER-ENV body
    // stride. All 0 when the world has no movable bodies (articulation-only
    // worlds keep the byte-identical snapshot/restore/reset).
    const uint32_t total_body_count = cap.bodies_per_env * E;
    snapshot_params_.total_link_count = L * E;
    snapshot_params_.env_count = E;
    snapshot_params_.total_body_count = total_body_count;
    restore_params_.total_link_count = L * E;
    restore_params_.env_count = E;
    restore_params_.row_slot_count = cap.max_rows_per_env * E;
    restore_params_.total_body_count = total_body_count;
    reset_params_.count = 0;
    reset_params_.base_link_count = L;
    reset_params_.lambda_stride = cap.max_rows_per_env;
    reset_params_.articulation_count = E;
    reset_params_.body_count = cap.bodies_per_env;

    // -- M4: movable rigid-body template seeding (env-major replication) -----
    const uint32_t B = cap.bodies_per_env;
    if (B > 0 && !model_.body_init.empty()) {
        std::vector<math::Transform> poses(static_cast<size_t>(B) * E);
        std::vector<math::Vec3> lin(static_cast<size_t>(B) * E);
        std::vector<math::Vec3> ang(static_cast<size_t>(B) * E);
        std::vector<float> inv_mass(static_cast<size_t>(B) * E, 0.0f);
        std::vector<math::Vec3> inv_inertia(static_cast<size_t>(B) * E);
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
                               inv_inertia.size() * sizeof(math::Vec3))) {
            return false;
        }
    }

    // -- M4: the per-env live table toggle (union family; harmless default
    // elsewhere — the field exists for every model).
    {
        std::vector<uint32_t> enabled(E, model_.table_enabled_default ? 1u : 0u);
        if (!data_.UploadField(FieldId::TableEnabled, enabled.data(),
                               enabled.size() * sizeof(uint32_t))) {
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
    }

    // Material bucket table (global; data-owned param field). Seeded for EVERY
    // world that has buckets — including articulation-less (bodies/particles-
    // only) worlds, which return early below. (Review fix: this block used to
    // sit below the L==0 early return, silently skipping the bucket seed for
    // any world without an articulation — the M5-binding-bug class.)
    if (cap.num_material_buckets > 0 && !model_.material_buckets.empty()) {
        std::vector<float> host(static_cast<size_t>(cap.num_material_buckets) * 8u, 0.0f);
        for (uint32_t b = 0; b < cap.num_material_buckets &&
                             b < model_.material_buckets.size(); ++b) {
            for (int k = 0; k < 8; ++k) {
                host[static_cast<size_t>(b) * 8u + static_cast<size_t>(k)] =
                    model_.material_buckets[b].values[k];
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
        // No articulation, but the world may still have movable BODIES (e.g. a
        // settled cup-on-table). The body state was already seeded above
        // (BodyPose / BodyLinearVelocity / BodyAngularVelocity), and
        // snapshot_params_.total_body_count is set, so SnapshotState now
        // CAPTURES the body slice (M7 T1) — Reset restores the settled cup pose.
        // (Particle snapshot/restore is still out of scope here — the named M8
        // RL-reset debt.) When there are no bodies either, SnapshotState is a
        // genuine no-op; honest Ok.
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
        std::vector<float> host(static_cast<size_t>(L) * E * 6u, 0.0f);
        for (uint32_t e = 0; e < E; ++e) {
            for (uint32_t l = 0; l < L; ++l) {
                for (uint32_t k = 0; k < 6u; ++k) {
                    const size_t src = static_cast<size_t>(l) * 6u + k;
                    if (src < a.initial_link_velocity.size()) {
                        host[(static_cast<size_t>(e) * L + l) * 6u + k] =
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
        std::vector<math::Transform> host(E, a.base_pose);
        if (!data_.UploadField(FieldId::BasePose, host.data(),
                               host.size() * sizeof(math::Transform))) {
            return false;
        }
    }
    // Device snapshot of the seeded initial state (the Reset restore source —
    // the legacy creation-time snapshot 1:1).
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
    if (!ready_ || backend_ == nullptr) {
        return out;
    }
    const std::vector<phi::OpCall>& calls = pipeline_.Calls();
    out.status.reserve(calls.size());
    for (const phi::OpCall& call : calls) {
        // One BackendDispatch per OpCall, in the §3.2 fixed order. (Ops the
        // backend lacks were already dropped by the Build-time supports_op
        // capability filter, so a healthy step is all-Ok.)
        const phi::Status s = phi::BackendDispatch(backend_, model_view_, data_view_, call);
        out.status.push_back(s);
    }
    return out;
}

phi::Status World::StepPlanned() {
    if (!ready_ || backend_ == nullptr) {
        return phi::Status::Failed;
    }
    const std::vector<phi::OpCall>& calls = pipeline_.Calls();
    if (plan_ == nullptr) {
        plan_ = phi::BackendPlanCreate(backend_, model_view_, data_view_,
                                       calls.data(),
                                       static_cast<int>(calls.size()));
        if (plan_ == nullptr) {
            // No capturable plan: an op in the list is not graph-captureable
            // (today: the thrust sorts in ParticleGridBuild / LbvhBuild — the
            // named PBF Step-only debt). Honest Unsupported; the caller falls
            // back to Step().
            return phi::Status::Unsupported;
        }
    }
    return phi::BackendPlanExecute(backend_, plan_);
}

phi::Status World::Reset(const std::vector<uint32_t>& env_ids) {
    if (!ready_ || backend_ == nullptr) {
        return phi::Status::Failed;
    }
    if (env_ids.empty()) {
        // Bulk restore: snapshot -> live + clear qddot/tau/lambda (device-side).
        return DispatchOp(phi::NkOp::RestoreState, &restore_params_);
    }
    // Per-env masked reset: validate host-side (a bad id must never OOB-write),
    // upload the ids into the reset_env_ids field, dispatch ResetEnvs.
    const uint32_t env_count = model_.capacities.env_count;
    for (uint32_t id : env_ids) {
        if (id >= env_count) {
            return phi::Status::Failed;
        }
    }
    const uint32_t count = static_cast<uint32_t>(
        env_ids.size() > env_count ? env_count : env_ids.size());
    if (!data_.UploadField(FieldId::ResetEnvIds, env_ids.data(),
                           static_cast<uint64_t>(count) * sizeof(uint32_t))) {
        return phi::Status::Failed;
    }
    reset_params_.count = count;
    return DispatchOp(phi::NkOp::ResetEnvs, &reset_params_);
}

phi::Status World::DispatchOp(phi::NkOp op, const void* params) {
    if (!ready_ || backend_ == nullptr) {
        return phi::Status::Failed;
    }
    const phi::OpCall call{op, params};
    return phi::BackendDispatch(backend_, model_view_, data_view_, call);
}

void* World::FieldPtr(FieldId id) const {
    if (!ready_) {
        return nullptr;
    }
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
