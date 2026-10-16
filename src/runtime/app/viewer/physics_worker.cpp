#include "runtime/app/viewer/physics_worker.hpp"

#include "runtime/app/viewer/editor_scene.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

// A stalled worker resumes without replaying more than this much wall time.
constexpr double kMaxElapsedSeconds = 0.25;
// A batch past this many steps drops the rest of its backlog.
constexpr uint32_t kMaxStepsPerBatch = 64u;
// Zero-copy publishing refreshes the host pose mirror and overlay at this period.
constexpr std::chrono::milliseconds kMirrorPeriod{33};
// A long batch publishes at least this often so the view keeps moving during catch-up.
constexpr std::chrono::milliseconds kBatchPublishPeriod{33};

double Ms(WorkerClock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

bool Nonzero(const float v[3]) { return v[0] != 0.0f || v[1] != 0.0f || v[2] != 0.0f; }

bool SameSource(const render::PoseSource& a, const render::PoseSource& b) {
    return a.kind == b.kind && a.row == b.row;
}

}  // namespace

// An iteration that throws reports the error and pauses the transport.
template <class Fn>
void PhysicsWorker::Guarded(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        playing_ = false;
        Report(0u, std::string("[nuka_editor] physics paused: ") + e.what() + "\n", true);
    }
}

TickCommand TickCommand::Transport(TransportOp op) {
    TickCommand cmd;
    cmd.kind = CommandKind::Transport;
    cmd.op = op;
    return cmd;
}

TickCommand TickCommand::Param(ParamKind param, std::vector<float> values) {
    TickCommand cmd;
    cmd.kind = CommandKind::ParamSet;
    cmd.param = param;
    cmd.values = std::move(values);
    return cmd;
}

TickCommand TickCommand::Script(std::string code) {
    TickCommand cmd;
    cmd.kind = CommandKind::ScriptCall;
    cmd.code = std::move(code);
    return cmd;
}

void SnapshotRing::Publish() {
    back_ = middle_.exchange(back_ | kFresh, std::memory_order_acq_rel) & ~kFresh;
}

const Snapshot* SnapshotRing::Acquire() {
    if ((middle_.load(std::memory_order_acquire) & kFresh) == 0u) return nullptr;
    front_ = middle_.exchange(front_, std::memory_order_acq_rel) & ~kFresh;
    return &slots_[front_];
}

void PoseSlotPool::Reset(uint32_t count) {
    std::lock_guard<std::mutex> lock(mu_);
    pins_.assign(count, 0u);
    latest_ = ~0u;
    latest_value_ = 0u;
    next_ = 0u;
}

uint32_t PoseSlotPool::AcquireWrite() {
    std::lock_guard<std::mutex> lock(mu_);
    const auto count = static_cast<uint32_t>(pins_.size());
    uint32_t slot = next_ % count;
    for (uint32_t k = 0; k < count; ++k) {
        const uint32_t s = (next_ + k) % count;
        if (pins_[s] == 0u && s != latest_) {
            slot = s;
            break;
        }
    }
    next_ = slot + 1u;
    return slot;
}

void PoseSlotPool::Publish(uint32_t slot, uint64_t value) {
    std::lock_guard<std::mutex> lock(mu_);
    latest_ = slot;
    latest_value_ = value;
}

bool PoseSlotPool::AcquireRead(uint32_t* slot, uint64_t* value) {
    std::lock_guard<std::mutex> lock(mu_);
    if (latest_ == ~0u) return false;
    ++pins_[latest_];
    *slot = latest_;
    *value = latest_value_;
    return true;
}

void PoseSlotPool::Release(uint32_t slot) {
    std::lock_guard<std::mutex> lock(mu_);
    if (slot < pins_.size() && pins_[slot] > 0u) --pins_[slot];
}

void PhysicsWorker::StartThread() {
    if (thread_.joinable()) return;
    stop_ = false;
    thread_ = std::thread([this] { Loop(); });
}

void PhysicsWorker::StopThread() {
    if (!thread_.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
}

void PhysicsWorker::Exclusive(const std::function<void()>& fn) {
    if (!thread_.joinable()) {
        fn();
        return;
    }
    {
        std::unique_lock<std::mutex> lock(mu_);
        park_ = true;
        cv_.notify_all();
        cv_.wait(lock, [this] { return parked_; });
    }
    struct Unpark {
        PhysicsWorker* worker;
        ~Unpark() {
            {
                std::lock_guard<std::mutex> lock(worker->mu_);
                worker->park_ = false;
            }
            worker->cv_.notify_all();
        }
    } unpark{this};
    fn();
}

uint64_t PhysicsWorker::Bind(WorkerBinding binding) {
    bind_ = std::move(binding);
    bound_ = true;
    ++epoch_;
    tick_ = 0u;
    sim_time_ = 0.0;
    playing_ = false;
    speed_ = bind_.speed;
    step_budget_ = 0u;
    healthy_ = true;
    accum_ = 0.0;
    clock_reset_ = true;
    perturbs_.clear();
    timeline_value_ = 0u;
    last_mirror_ = WorkerClock::time_point{};
    telemetry_ = WorkerTelemetry{};
    Publish();
    return epoch_;
}

void PhysicsWorker::Unbind() {
    bind_ = WorkerBinding{};
    bound_ = false;
    ++epoch_;
    playing_ = false;
    step_budget_ = 0u;
    perturbs_.clear();
}

uint64_t PhysicsWorker::Push(TickCommand cmd) {
    uint64_t seq = 0u;
    {
        std::lock_guard<std::mutex> lock(mu_);
        seq = cmd.seq = ++next_seq_;
        queue_.push_back(std::move(cmd));
    }
    cv_.notify_all();
    return seq;
}

void PhysicsWorker::Pump() {
    std::vector<TickCommand> cmds;
    {
        std::lock_guard<std::mutex> lock(mu_);
        cmds.swap(queue_);
    }
    Guarded([&] {
        ApplyCommands(cmds);
        step_budget_ = 0u;
        if (!bound_) return;
        Publish();
        if (scripts_->Valid()) Report(0u, scripts_->DispatchTick());
    });
}

std::vector<TickResult> PhysicsWorker::TakeResults() {
    std::vector<TickResult> out;
    std::lock_guard<std::mutex> lock(mu_);
    out.swap(results_);
    return out;
}

void PhysicsWorker::Report(uint64_t seq, std::string text, bool error) {
    if (text.empty() && seq == 0u) return;
    std::lock_guard<std::mutex> lock(mu_);
    results_.push_back(TickResult{seq, std::move(text), error});
}

void PhysicsWorker::Loop() {
    if (bound_) Guarded([&] { BindWorldDevice(bind_.world); });
    std::unique_lock<std::mutex> lock(mu_);
    while (!stop_) {
        if (park_ && queue_.empty()) {
            parked_ = true;
            cv_.notify_all();
            cv_.wait(lock, [this] { return !park_ || stop_; });
            parked_ = false;
            clock_reset_ = true;
            lock.unlock();
            if (bound_) Guarded([&] { BindWorldDevice(bind_.world); });
            lock.lock();
            continue;
        }
        if (!queue_.empty()) {
            std::vector<TickCommand> cmds;
            cmds.swap(queue_);
            lock.unlock();
            Guarded([&] {
                if (ApplyCommands(cmds)) Publish();
            });
            lock.lock();
            continue;
        }
        if (!bound_ || !(playing_ || step_budget_ > 0u)) {
            cv_.wait(lock);
            continue;
        }
        lock.unlock();
        double wait_s = 0.0;
        Guarded([&] { wait_s = Advance(); });
        lock.lock();
        if (wait_s > 0.0)
            cv_.wait_for(lock, std::chrono::duration<double>(wait_s),
                         [this] { return stop_ || park_ || !queue_.empty(); });
    }
}

bool PhysicsWorker::ApplyCommands(std::vector<TickCommand>& cmds) {
    bool publish = false;
    for (const TickCommand& cmd : cmds) {
        if (bound_ || cmd.kind == CommandKind::ScriptCall) publish = Apply(cmd) || publish;
        acked_ = cmd.seq;
    }
    return publish && bound_;
}

// True when the command changed what the next snapshot shows.
bool PhysicsWorker::Apply(const TickCommand& cmd) {
    switch (cmd.kind) {
        case CommandKind::EditApply:
        case CommandKind::Teleport:
            ApplyPoseWrite(bind_.world, bind_.layout, bind_.env, cmd.pose);
            return true;
        case CommandKind::ParamSet:
            switch (cmd.param) {
                case ParamKind::Field:
                    CheckResult(nuka_world_upload_field(bind_.world, cmd.field, cmd.values.data(),
                                                        cmd.values.size() * sizeof(float), cmd.offset),
                                "nuka_world_upload_field");
                    return false;
                case ParamKind::HoldGains:
                    ApplyHoldGains(bind_.world, bind_.layout, cmd.values[0], cmd.values[1]);
                    return false;
                case ParamKind::ControllerCommand:
                    if (bind_.controller != nullptr)
                        bind_.controller->SetCommandVector(cmd.values.data(), static_cast<uint32_t>(cmd.values.size()));
                    return false;
                case ParamKind::DisplayEnv:
                    bind_.env = cmd.env % bind_.layout.env_count;
                    last_mirror_ = WorkerClock::time_point{};
                    return true;
                case ParamKind::ScatterRows:
                    if (bind_.scatter != nullptr) bind_.scatter->SetRows(bind_.world, cmd.rows);
                    return true;
                case ParamKind::Overlay:
                    bind_.colliders = cmd.values[0] != 0.0f;
                    bind_.contacts = cmd.values[1] != 0.0f;
                    last_mirror_ = WorkerClock::time_point{};
                    return true;
            }
            return false;
        case CommandKind::Perturb: {
            const auto it = std::find_if(perturbs_.begin(), perturbs_.end(), [&](const Perturbation& p) {
                return SameSource(p.source, cmd.perturb.source);
            });
            if (it != perturbs_.end()) perturbs_.erase(it);
            if (Nonzero(cmd.perturb.force) || Nonzero(cmd.perturb.torque)) perturbs_.push_back(cmd.perturb);
            return false;
        }
        case CommandKind::Transport:
            switch (cmd.op) {
                case TransportOp::Play:
                    playing_ = true;
                    clock_reset_ = true;
                    break;
                case TransportOp::Pause:
                    playing_ = false;
                    break;
                case TransportOp::Step:
                    step_budget_ += cmd.steps;
                    break;
                case TransportOp::Speed:
                    speed_ = cmd.speed;
                    break;
                case TransportOp::Reset:
                    healthy_ = ResetWorldPhysics(bind_.world, bind_.rest_targets);
                    tick_ = 0u;
                    sim_time_ = 0.0;
                    playing_ = false;
                    step_budget_ = 0u;
                    last_mirror_ = WorkerClock::time_point{};
                    Report(cmd.seq, scripts_->DispatchReset());
                    break;
            }
            return true;
        case CommandKind::ScriptCall:
            Report(cmd.seq, scripts_->RunString(cmd.code));
            last_mirror_ = WorkerClock::time_point{};
            return true;
    }
    return false;
}

// Runs the steps that are due; when none are, the seconds until the next one is.
double PhysicsWorker::Advance() {
    const WorkerClock::time_point now = WorkerClock::now();
    if (clock_reset_) {
        last_ = now;
        accum_ = 0.0;
        clock_reset_ = false;
    }
    const double dt = static_cast<double>(bind_.dt);
    const double speed = static_cast<double>(speed_);
    uint32_t due = 0u;
    if (playing_) {
        double elapsed = std::chrono::duration<double>(now - last_).count();
        last_ = now;
        if (elapsed > kMaxElapsedSeconds) {
            telemetry_.dropped_sim_s += (elapsed - kMaxElapsedSeconds) * speed;
            elapsed = kMaxElapsedSeconds;
        }
        accum_ += elapsed * speed;
        const double want = std::floor(accum_ / dt);
        if (want > static_cast<double>(kMaxStepsPerBatch)) {
            ++telemetry_.capped_batches;
            telemetry_.dropped_sim_s += accum_ - static_cast<double>(kMaxStepsPerBatch) * dt;
            due = kMaxStepsPerBatch;
            accum_ = 0.0;
        } else {
            due = static_cast<uint32_t>(want);
            accum_ -= static_cast<double>(due) * dt;
        }
    }
    due += step_budget_;
    step_budget_ = 0u;
    if (due == 0u) return (dt - accum_) / speed;

    bool healthy = true;
    for (uint32_t i = 0; i < due; ++i) {
        WritePerturbations();
        if (bind_.controller != nullptr) bind_.controller->OnStep(bind_.world, bind_.env);
        healthy = nuka_world_step(bind_.world) == NUKA_RESULT_OK && healthy;
        ++tick_;
        sim_time_ += dt;
        if (i + 1u == due || WorkerClock::now() - last_publish_ >= kBatchPublishPeriod) {
            healthy_ = healthy;
            telemetry_.step_ms = Ms(WorkerClock::now() - now);
            telemetry_.batch_steps = i + 1u;
            Publish();
        }
        DispatchTick();
    }
    return 0.0;
}

void PhysicsWorker::WritePerturbations() {
    const WorldLayout& layout = bind_.layout;
    const size_t env = bind_.env;
    for (const Perturbation& p : perturbs_) {
        if (p.source.kind == render::PoseSource::Kind::Link) {
            perturb_scratch_.assign({p.force[0], p.force[1], p.force[2], p.torque[0], p.torque[1], p.torque[2]});
            const size_t at = (env * layout.links_per_env + p.source.row) * 6u * sizeof(float);
            CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_LINK_WRENCH, perturb_scratch_.data(),
                                                6u * sizeof(float), at),
                        "upload link wrench");
        } else {
            const size_t at = (env * layout.bodies_per_env + p.source.row) * 3u * sizeof(float);
            CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_BODY_FORCE, p.force, 3u * sizeof(float), at),
                        "upload body force");
            CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_BODY_TORQUE, p.torque, 3u * sizeof(float), at),
                        "upload body torque");
        }
    }
}

void PhysicsWorker::DispatchTick() {
    if (!scripts_->Valid()) return;
    std::string text = scripts_->DispatchStep();
    text += scripts_->DispatchTick();
    Report(0u, std::move(text));
}

void PhysicsWorker::Publish() {
    const WorkerClock::time_point start = WorkerClock::now();
    Snapshot& s = ring_.Back();
    s.sequence = ++sequence_;
    s.epoch = epoch_;
    s.acked = acked_;
    s.tick = tick_;
    s.sim_time = sim_time_;
    s.env = bind_.env;
    s.playing = playing_;
    s.healthy = healthy_;
    if (bind_.scatter != nullptr) {
        const uint32_t slot = slots_.AcquireWrite();
        bind_.scatter->Scatter(bind_.world, bind_.env, bind_.layout.links_per_env, bind_.layout.bodies_per_env,
                               slot, ++timeline_value_);
        slots_.Publish(slot, timeline_value_);
    }
    if (bind_.scatter == nullptr || !playing_ || start - last_mirror_ >= kMirrorPeriod) {
        DownloadEnvPoses(bind_.world, bind_.layout, bind_.env, &mirror_links_, &mirror_bodies_);
        ReadOverlayData(bind_.world, bind_.layout, bind_.env, bind_.colliders, bind_.contacts, &overlay_);
        last_mirror_ = start;
    }
    s.link_poses = mirror_links_;
    s.body_poses = mirror_bodies_;
    s.overlay = overlay_;
    s.published = WorkerClock::now();
    last_publish_ = s.published;
    telemetry_.publish_ms = Ms(s.published - start);
    s.telemetry = telemetry_;
    ring_.Publish();
}

}  // namespace nuka::runtime::app::viewer
