#pragma once
// The editor's physics thread. While it runs it makes every world call: typed commands
// apply in order at tick boundaries, and each publish hands the UI a snapshot.

#include "math/transform.hpp"
#include "nuka/nuka.h"
#include "render/render_world.hpp"
#include "runtime/app/viewer/debug_overlay.hpp"
#include "runtime/app/viewer/editor_world.hpp"
#include "runtime/app/viewer/pose_scatter.hpp"
#include "runtime/app/viewer/script_host.hpp"
#include "runtime/inference/world_controller.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nuka::runtime::app::viewer {

using WorkerClock = std::chrono::steady_clock;

enum class CommandKind : uint8_t { EditApply, Teleport, ParamSet, Perturb, Transport, ScriptCall };
enum class ParamKind : uint8_t { Field, HoldGains, ControllerCommand, DisplayEnv, ScatterRows, Overlay, Registry };
enum class TransportOp : uint8_t { Play, Pause, Step, Speed, Reset };

// A held load on one link (wrench) or free body (force and torque), rewritten before
// every step until a zero load replaces it.
struct Perturbation {
    render::PoseSource source;
    float force[3] = {0.0f, 0.0f, 0.0f};
    float torque[3] = {0.0f, 0.0f, 0.0f};
};

struct TickCommand {
    CommandKind kind = CommandKind::Transport;
    uint64_t seq = 0u;                    // assigned by Push
    uint64_t pushed_tick = 0u;            // the worker's tick when Push queued it
    PoseWrite pose;                       // EditApply, Teleport
    ParamKind param = ParamKind::Field;   // ParamSet
    nuka_state_field_t field = NUKA_FIELD_DRIVE_TARGET;
    size_t offset = 0u;                   // bytes into the field
    uint32_t index = 0u;                  // Registry parameter
    uint32_t element = 0u;                // Registry element
    std::vector<float> values;            // field values, gains (kp, kd), command, overlay flags, registry value
    uint32_t env = 0u;                    // DisplayEnv
    std::vector<ScatterRow> rows;         // ScatterRows
    Perturbation perturb;                 // Perturb
    TransportOp op = TransportOp::Pause;  // Transport
    uint32_t steps = 0u;
    float speed = 1.0f;
    std::string code;                     // ScriptCall

    static TickCommand Transport(TransportOp op);
    static TickCommand Param(ParamKind param, std::vector<float> values);
    static TickCommand Script(std::string code);
};

struct TickResult {
    uint64_t seq = 0u;  // the command answered; 0 for callback output
    std::string text;
    bool error = false;
    uint64_t pushed_tick = 0u;   // registry writes: the tick when queued
    uint64_t applied_tick = 0u;  // and the tick the write landed after
};

struct WorkerTelemetry {
    double step_ms = 0.0;  // the last batch; launch time when nothing synchronizes
    double publish_ms = 0.0;
    uint32_t batch_steps = 0u;
    uint64_t capped_batches = 0u;
    double dropped_sim_s = 0.0;
};

struct Snapshot {
    uint64_t sequence = 0u;  // gaps between reads are dropped snapshots
    uint64_t epoch = 0u;     // the binding it was taken from
    uint64_t acked = 0u;     // the last command applied before it
    uint64_t tick = 0u;
    double sim_time = 0.0;
    WorkerClock::time_point published{};
    uint32_t env = 0u;
    bool playing = false;
    bool healthy = true;
    std::vector<math::Transform> link_poses;
    std::vector<math::Transform> body_poses;
    OverlayData overlay;
    WorkerTelemetry telemetry;
};

// Three snapshots passed from one writer to one reader by exchanging slot indices.
class SnapshotRing {
public:
    Snapshot& Back() { return slots_[back_]; }
    void Publish();
    // The newest snapshot when one arrived since the last call, otherwise null.
    const Snapshot* Acquire();
    const Snapshot& Front() const { return slots_[front_]; }

private:
    static constexpr uint32_t kFresh = 4u;
    std::array<Snapshot, 3> slots_;
    uint32_t back_ = 0u;
    uint32_t front_ = 2u;
    std::atomic<uint32_t> middle_{1u};
};

// Zero-copy pose slots: the scatter writes a slot no frame pins and frames pin the latest.
class PoseSlotPool {
public:
    void Reset(uint32_t count);
    uint32_t AcquireWrite();
    void Publish(uint32_t slot, uint64_t value);
    // Pins the latest written slot; false before the first write.
    bool AcquireRead(uint32_t* slot, uint64_t* value);
    void Release(uint32_t slot);

private:
    std::mutex mu_;
    std::vector<uint32_t> pins_;
    uint32_t latest_ = ~0u;
    uint64_t latest_value_ = 0u;
    uint32_t next_ = 0u;
};

// What a world binding hands the worker; replaced at every load and rebuild.
struct WorkerBinding {
    nuka_world_handle world = nullptr;
    WorldLayout layout;
    std::vector<float> rest_targets;
    inference::WorldController* controller = nullptr;
    PoseScatter* scatter = nullptr;  // null publishes through host downloads
    uint32_t env = 0u;
    float dt = 0.005f;
    float speed = 1.0f;
    bool colliders = false;
    bool contacts = false;
    uint64_t tick = 0u;  // where the timeline continues; a migrating rebuild keeps it
    double sim_time = 0.0;
};

class PhysicsWorker {
public:
    explicit PhysicsWorker(ScriptHost* scripts) : scripts_(scripts) {}
    ~PhysicsWorker() { StopThread(); }
    PhysicsWorker(const PhysicsWorker&) = delete;
    PhysicsWorker& operator=(const PhysicsWorker&) = delete;

    // The thread paces the transport; without it, Pump runs one boundary per frame.
    void StartThread();
    void StopThread();
    bool Threaded() const { return thread_.joinable(); }

    // Runs fn on the calling thread while the worker is parked at a tick boundary, after
    // every command queued before it.
    void Exclusive(const std::function<void()>& fn);

    // Exclusive sections only. Bind publishes the first snapshot of the new binding and
    // returns its epoch.
    uint64_t Bind(WorkerBinding binding);
    void Unbind();
    double SimTime() const { return sim_time_; }
    // The last finished tick; readable from any thread.
    uint64_t Tick() const { return tick_mirror_.load(std::memory_order_relaxed); }

    uint64_t Push(TickCommand cmd);
    // Applies the queued commands, publishes and dispatches the tick callback.
    void Pump();
    std::vector<TickResult> TakeResults();
    const Snapshot* AcquireSnapshot() { return ring_.Acquire(); }
    PoseSlotPool& Slots() { return slots_; }

private:
    void Loop();
    template <class Fn>
    void Guarded(Fn&& fn);
    bool ApplyCommands(std::vector<TickCommand>& cmds);
    bool Apply(const TickCommand& cmd);
    double Advance();
    void WritePerturbations();
    void Publish();
    void DispatchTick();
    void Report(uint64_t seq, std::string text, bool error = false);
    void Report(TickResult result);
    void SetTick(uint64_t tick);

    ScriptHost* scripts_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<TickCommand> queue_;
    std::vector<TickResult> results_;
    uint64_t next_seq_ = 0u;
    bool stop_ = false;
    bool park_ = false;
    bool parked_ = false;
    // Set with a queued command or a park request so a running batch yields at its next step.
    std::atomic<bool> interrupt_{false};
    std::atomic<uint64_t> tick_mirror_{0u};
    std::thread thread_;
    SnapshotRing ring_;
    PoseSlotPool slots_;

    // Worker state; the UI touches it only while the worker is parked or not threaded.
    WorkerBinding bind_;
    bool bound_ = false;
    uint64_t epoch_ = 0u;
    uint64_t acked_ = 0u;
    uint64_t tick_ = 0u;
    double sim_time_ = 0.0;
    bool playing_ = false;
    float speed_ = 1.0f;
    uint32_t step_budget_ = 0u;
    bool healthy_ = true;
    double accum_ = 0.0;
    bool clock_reset_ = true;
    WorkerClock::time_point last_{};
    std::vector<Perturbation> perturbs_;
    std::vector<float> perturb_scratch_;
    uint64_t timeline_value_ = 0u;
    uint64_t sequence_ = 0u;
    WorkerClock::time_point last_publish_{};
    WorkerClock::time_point last_mirror_{};
    std::vector<math::Transform> mirror_links_;
    std::vector<math::Transform> mirror_bodies_;
    OverlayData overlay_;
    WorkerTelemetry telemetry_;
};

}  // namespace nuka::runtime::app::viewer
