#pragma once
// ---------------------------------------------------------------------------
// nk::World — Model + Data + Pipeline bound to a backend (plan §3.2 / M3).
//
// Ctor takes the cook product (nk::Model), the env_count, and the phi::Backend:
// it uploads the Model into ONE device buffer, allocates the Data arena, and
// builds the Pipeline (the per-step OpCall list). Step() dispatches each OpCall
// (surfacing per-op status HONESTLY — in M3a no ops are registered, so every op
// returns Unsupported, which is the EXPECTED M3a evidence). StepPlanned() builds
// a CUDA-graph plan once then replays it. Reset(env_mask) dispatches ResetEnvs.
// FieldPtr(FieldId) returns a device pointer (model- or data-owned).
//
// PURE C++ — zero CUDA tokens. All device + execution via the phi v2 vtable.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

#include "phi/backend.hpp"
#include "nk/data/data.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/pipeline.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/generated/views.hpp"

namespace nuka::nk {

// Forward placeholder for the M4 device-resident solve schedule. World owns it
// (built at construct / Reset); M3a leaves it empty.
class SolveSchedule;

// Per-op step outcome — World surfaces the status of EVERY dispatched op so a
// caller can see which ops are unimplemented (M3a: all Unsupported) vs failed.
struct StepResult {
    // Parallel to the Pipeline's OpCall list. status[i] is the dispatch result
    // of call i.
    std::vector<phi::Status> status;

    // Convenience aggregates.
    bool AllOk() const {
        for (phi::Status s : status) if (s != phi::Status::Ok) return false;
        return true;
    }
    int CountOf(phi::Status want) const {
        int n = 0;
        for (phi::Status s : status) if (s == want) ++n;
        return n;
    }
    size_t OpCount() const { return status.size(); }
};

class World {
public:
    // Construct from a cook product. The World TAKES the Model by value (move).
    // env_count overrides the Model's capacity env_count if > 0 (the Model is
    // cooked at a base env_count by CookToModel; this is the validation seam).
    // backend must be a live phi::Backend; `device` supplies the BufferType used
    // for the one-shot Model upload + Arena allocation (the plan's documented use
    // of DeviceI::get_buffer_type — a stream-less default-stream type fine for
    // init-time movement). On any device failure the World is left in an unbuilt
    // state (Ready() == false).
    World(Model model, uint32_t env_count, phi::Device* device,
          phi::Backend* backend, const Pipeline::SolverConfig& cfg = {});

    ~World();
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    bool Ready() const { return ready_; }
    uint32_t EnvCount() const { return model_.capacities.env_count; }

    // Dispatch each OpCall in order; returns the per-op status vector. In M3a all
    // ops are Unsupported (no op registered) — that is the expected evidence.
    StepResult Step();

    // Plan path: build a CUDA-graph plan over the OpCall list once, then execute.
    // Returns the plan_execute status. If plan_create fails (e.g. an op the
    // backend cannot capture in M3a), returns Status::Unsupported and falls back
    // to NOT planning (caller can use Step()).
    phi::Status StepPlanned();

    // Reset the envs selected by `env_mask` (bit e set => reset env e). M3a
    // dispatches the ResetEnvs op (Unsupported today) and re-zeros the data arena
    // host-side so the World stays in a clean, crash-free state. Empty mask =>
    // all envs.
    phi::Status Reset(const std::vector<uint32_t>& env_ids = {});

    // Device pointer of a field (model- or data-owned); null if absent/unbuilt.
    void* FieldPtr(FieldId id) const;
    template <class T> T* FieldPtr(FieldId id) const { return static_cast<T*>(FieldPtr(id)); }

    const phi::ModelView& ModelViewRef() const { return model_view_; }
    const phi::DataView&  DataViewRef()  const { return data_view_; }

    const Pipeline& GetPipeline() const { return pipeline_; }
    const Model&    GetModel()    const { return model_; }
    Data&           GetData()     { return data_; }

private:
    Model           model_;
    Data            data_;
    Pipeline        pipeline_;
    phi::Backend*   backend_ = nullptr;
    phi::ModelView  model_view_{};
    phi::DataView   data_view_{};
    phi::Plan*      plan_ = nullptr;
    bool            ready_ = false;

    // ResetEnvs op params storage (stable address for dispatch).
    phi::ResetEnvsParams reset_params_{};
};

} // namespace nuka::nk
