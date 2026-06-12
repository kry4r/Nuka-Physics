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

    // 3. Build the per-step pipeline (OpCall list, §3.2 order).
    pipeline_.Build(model_, cfg);

    ready_ = true;
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

phi::Status World::Reset(const std::vector<uint32_t>& /*env_ids*/) {
    if (!ready_ || backend_ == nullptr) {
        return phi::Status::Failed;
    }
    // Dispatch the ResetEnvs op (Unsupported in M3a). Whatever the op returns,
    // also re-zero the data arena host-side so the World stays clean + crash-free
    // (the device ResetEnvs op replaces this in M3b).
    const phi::OpCall call{phi::NkOp::ResetEnvs, &reset_params_};
    const phi::Status s = phi::BackendDispatch(backend_, model_view_, data_view_, call);
    data_.ZeroAll();
    return s;
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
