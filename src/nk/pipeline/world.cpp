// ---------------------------------------------------------------------------
// nk::World implementation (plan §3.2 / M3).
// ---------------------------------------------------------------------------

#include "nk/pipeline/world.hpp"

#include <utility>

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

    // Fill the reset/snapshot params (stable addresses for dispatch).
    snapshot_params_.total_link_count = L * E;
    snapshot_params_.env_count = E;
    restore_params_.total_link_count = L * E;
    restore_params_.env_count = E;
    restore_params_.row_slot_count = cap.max_rows_per_env * E;
    reset_params_.count = 0;
    reset_params_.base_link_count = L;
    reset_params_.lambda_stride = cap.max_rows_per_env;
    reset_params_.articulation_count = E;

    if (L == 0) {
        return true;  // no articulation: nothing to seed.
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
    // Material bucket table (global; data-owned param field).
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
        // Dispatch honestly: M3a has no ops registered, so every dispatch
        // returns Unsupported (the expected M3a evidence). Real ops land in M3b+.
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
            // No capturable plan in M3a (no real ops) — honest Unsupported; the
            // caller can fall back to Step().
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
