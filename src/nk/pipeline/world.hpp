#pragma once
// World owns model, mutable state, and one operator sequence for eager or graph execution.
// Device operations and storage use the backend interface without exposing CUDA types.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "phi/backend.hpp"
#include "nk/data/data.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/pipeline.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/generated/views.hpp"
#include "sensor/state_bank.hpp"

namespace nuka::nk {

// The fixed-capacity solve schedule depends on immutable topology and survives reset.
class SolveSchedule;

// Per-op step outcome — World surfaces the status of EVERY dispatched op so a
// caller can see which ops are unimplemented (Unsupported) vs failed.
struct StepResult {
    // Parallel to the Pipeline's OpCall list. status[i] is the dispatch result
    // of call i.
    std::vector<phi::Status> status;
    phi::Status result = phi::Status::Failed;
    phi::NkOp failed_op = phi::NkOp::Count;

    // Convenience aggregates.
    bool AllOk() const {
        if (result != phi::Status::Ok) return false;
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
    enum class ExecutionMode : uint32_t { Eager, Graph };
    // Takes ownership of the cooked model; a positive env_count overrides its capacity.
    // Device and backend must remain live; creation failures leave Ready() false.
    World(Model model, uint32_t env_count, phi::Device* device,
          phi::Backend* backend, const Pipeline::SolverConfig& cfg = {});

    ~World();
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    bool Ready() const { return ready_; }
    phi::Status CreationStatus() const { return creation_status_; }
    const std::string& CreationError() const { return creation_error_; }
    phi::Status LastStatus() const { return last_status_; }
    uint32_t EnvCount() const { return model_.capacities.env_count; }

    // Rebuild the common operator parameters; state and readout buffers retain their identity.
    phi::Status SetGravity(const math::Vec3& gravity);
    math::Vec3 Gravity() const { return {cfg_.gravity[0], cfg_.gravity[1], cfg_.gravity[2]}; }

    // Changes contact exchange frequency while retaining material iterations and interval length.
    phi::Status SetCouplingPasses(uint32_t passes);

    // Update one template link's spatial inertia in every environment without reallocating.
    phi::Status SetLinkInertia(uint32_t link_index, const Mat36& inertia);

    phi::Status AttachStateSensor(const sensor::StateSensorDesc& desc, uint32_t* id);
    phi::Status ConfigureStateSensorError(uint32_t id, uint32_t channel, const sensor::ObservationConfig& config);
    phi::Status RestoreStateSensors(const sensor::StateSensorBankSnapshot& snapshot);
    const sensor::StateSensorBank& StateSensors() const { return state_sensors_; }

    // Dispatch in order and stop at the first host or launch failure.
    StepResult Step();

    // Capture the common operator sequence once and replay it without eager fallback.
    // Failed captures are cached until the operator sequence changes.
    phi::Status StepPlanned();
    phi::Status PrepareGraph();
    phi::Status SetExecutionMode(ExecutionMode mode);
    phi::Status StepConfigured();
    phi::Status Synchronize();
    ExecutionMode GetExecutionMode() const { return execution_mode_; }
    bool GraphReady() const { return plan_ != nullptr; }
    uint64_t CaptureAttempts() const { return capture_attempts_; }
    uint64_t GraphReplays() const { return graph_replays_; }
    const phi::ExecutionError& LastExecutionError() const { return execution_error_; }

    // Restore the selected environment set; empty selects all, duplicates are ignored.
    // Invalid IDs fail without mutation. Control targets and unselected environments stay intact.
    phi::Status Reset(const std::vector<uint32_t>& env_ids = {});

    // Dispatch a single op outside the per-step pipeline (oracle harness +
    // snapshot/restore/obs plumbing). params must match the op's POD.
    phi::Status DispatchOp(phi::NkOp op, const void* params);

    // Returns the field's device pointer, enabling its readout producer when required.
    // A new producer invalidates the graph; absent or unbuilt fields return null.
    void* FieldPtr(FieldId id) const;
    template <class T> T* FieldPtr(FieldId id) const { return static_cast<T*>(FieldPtr(id)); }

    const phi::ModelView& ModelViewRef() const { return model_view_; }
    const phi::DataView&  DataViewRef()  const { return data_view_; }

    const Pipeline& GetPipeline() const { return *pipeline_; }
    const Model&    GetModel()    const { return model_; }
    Data&           GetData()     { return data_; }

    // Interop consumers share the world's backend and stream through this opaque handle.
    phi::Backend* Backend() const { return backend_; }

private:
    // Seed environment state from the model and capture the reset snapshot.
    bool SeedInitialState();
    phi::Status RefreshPoses(uint32_t selected_env_count);

    // First external request for a readout output: emit the producing op from
    // now on (rebuild pipeline, drop the plan) + backfill it from the last solve.
    phi::Status DemandReadout(FieldId id);
    phi::Status RebuildPipeline();

    Model           model_;
    sensor::StateSensorBank state_sensors_;
    Data            data_;
    std::unique_ptr<Pipeline> pipeline_ = std::make_unique<Pipeline>();
    Pipeline::SolverConfig cfg_{};
    phi::Device*    device_ = nullptr;
    uint32_t        readout_demand_ = 0;
    phi::Backend*   backend_ = nullptr;
    phi::ModelView  model_view_{};
    phi::DataView   data_view_{};
    phi::Plan*      plan_ = nullptr;
    bool            plan_attempted_ = false;
    ExecutionMode   execution_mode_ = ExecutionMode::Eager;
    uint64_t        capture_attempts_ = 0u;
    uint64_t        graph_replays_ = 0u;
    phi::ExecutionError graph_error_{};
    phi::ExecutionError execution_error_{};
    bool            ready_ = false;
    phi::Status     creation_status_ = phi::Status::Failed;
    phi::Status     last_status_ = phi::Status::Ok;
    std::string     creation_error_;

    // Reset/snapshot op params storage (stable addresses for dispatch).
    phi::ResetEnvsParams     reset_params_{};
    phi::SnapshotStateParams snapshot_params_{};
    phi::RestoreStateParams  restore_params_{};
    // FK refresh after a restore: recompute the link world poses from the restored
    // base_pose + q so a consumer renders / reads obs at rest without first stepping.
    phi::FkWorldPosesParams  fk_params_{};
};

} // namespace nuka::nk
