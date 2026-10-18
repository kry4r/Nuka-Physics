#include "runtime/app/viewer/physics_worker.hpp"

#include "nuka/nuka_state_sensor.h"
#include "runtime/app/viewer/editor_scene.hpp"
#include "runtime/app/viewer/robot_tools.hpp"

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
// The solver log keeps this many of the newest telemetry records.
constexpr size_t kSolverLog = 4096u;
// The joint samples kept: 20 s at 200 Hz.
constexpr size_t kJointLog = 4096u;

bool Capped(const nuka_telemetry_record_t& r) { return (r.flags & NUKA_TELEMETRY_CAP) != 0u; }

double Ms(WorkerClock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

math::Vec3 Vec(const float v[3]) { return math::Vec3{v[0], v[1], v[2]}; }

// Whether two perturbations hold the same particle, link or body.
bool SameTarget(const Perturbation& a, const Perturbation& b) {
    return a.env == b.env && a.particle == b.particle &&
           (a.particle != kNoParticle || (a.source.kind == b.source.kind && a.source.row == b.source.row));
}

// A seek right before an absolute one lands nowhere the later one keeps.
bool Superseded(const std::vector<TickCommand>& cmds, size_t i) {
    const auto seek = [&](size_t k) {
        return k < cmds.size() && cmds[k].kind == CommandKind::Timeline && cmds[k].timeline == TimelineOp::Seek;
    };
    return seek(i) && seek(i + 1u) && !cmds[i + 1u].relative;
}

}  // namespace

SensorReading ReadSensor(nuka_world_handle world, uint32_t index, uint32_t plane, uint32_t env) {
    SensorReading out;
    out.plane = plane;
    nuka_result_t r = NUKA_RESULT_OK;
    if (plane == kStatePlane) {
        uint32_t envs = 0u;
        r = nuka_world_get_state_sensor_dims(world, index, &envs, &out.width);
        out.height = out.channels = 1u;
        out.values.resize(out.width);
        if (r == NUKA_RESULT_OK)
            r = nuka_world_download_state_sensor(world, index, out.values.data(), out.values.size() * sizeof(float),
                                                 size_t{env} * out.values.size() * sizeof(float));
    } else {
        r = nuka_world_render_sensors(world);
        if (r == NUKA_RESULT_OK && plane == NUKA_SENSOR_CHANNEL_RANGE) {
            r = nuka_world_get_lidar_dims(world, nullptr, nullptr, &out.width, &out.height);
            out.channels = 1u;
        } else if (r == NUKA_RESULT_OK) {
            r = nuka_world_get_sensor_dims(world, nullptr, nullptr, &out.height, &out.width, &out.channels);
            if (plane == NUKA_SENSOR_CHANNEL_DEPTH || plane == NUKA_SENSOR_CHANNEL_PRIM) out.channels = 1u;
        }
        uint64_t count = 0u;
        const auto channel = static_cast<nuka_sensor_channel_t>(plane);
        if (r == NUKA_RESULT_OK) r = nuka_world_download_sensor(world, channel, env, index, nullptr, 0u, &count);
        out.values.resize(count);
        if (r == NUKA_RESULT_OK)
            r = nuka_world_download_sensor(world, channel, env, index, out.values.data(), count, &count);
        size_t len = 0u;
        if (r == NUKA_RESULT_OK && plane == NUKA_SENSOR_CHANNEL_PRIM &&
            nuka_world_get_sensor_instances(world, nullptr, 0u, &len) == NUKA_RESULT_OK) {
            std::string paths(len, '\0');
            nuka_world_get_sensor_instances(world, paths.data(), paths.size() + 1u, &len);
            for (size_t at = 0u; at < paths.size();) {
                const size_t end = paths.find('\n', at);
                out.instances.push_back(paths.substr(at, end - at));
                at = end + 1u;
            }
        }
    }
    if (r != NUKA_RESULT_OK) {
        out.values.clear();
        out.error = nuka_result_message(r);
    }
    return out;
}

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

TickCommand TickCommand::Script(std::string code, std::string filename, bool echo) {
    TickCommand cmd;
    cmd.kind = CommandKind::ScriptCall;
    cmd.code = std::move(code);
    cmd.filename = std::move(filename);
    cmd.echo = echo;
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
    exited_.store(false, std::memory_order_relaxed);
    thread_ = std::thread([this] { Loop(); });
}

void PhysicsWorker::StopThread(const std::function<void()>& serve) {
    if (!thread_.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    while (serve && !exited_.load(std::memory_order_acquire)) serve();
    thread_.join();
}

void PhysicsWorker::ParkWhile(const std::function<void()>& wait) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        parked_ = true;
    }
    cv_.notify_all();
    wait();
    {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return !park_ || stop_; });
        parked_ = false;
    }
    clock_reset_ = true;
    if (bound_) Guarded([&] { BindWorldDevice(bind_.world); });
}

void PhysicsWorker::Exclusive(const std::function<void()>& fn) {
    if (!thread_.joinable()) {
        fn();
        return;
    }
    {
        std::unique_lock<std::mutex> lock(mu_);
        park_ = true;
        interrupt_.store(true, std::memory_order_relaxed);
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
    SetTick(bind_.tick);
    sim_time_ = bind_.sim_time;
    playing_ = false;
    speed_ = bind_.speed;
    step_budget_ = 0u;
    healthy_ = true;
    accum_ = 0.0;
    clock_reset_ = true;
    perturbs_.clear();
    ramp_ = TargetRamp{};
    track_ = TargetTrack{};
    command_ = {};
    axes_.clear();
    demo_envs_.clear();
    timeline_value_ = 0u;
    last_mirror_ = WorkerClock::time_point{};
    telemetry_ = WorkerTelemetry{};
    mirror_joints_ = JointFrame{};
    {
        std::lock_guard<std::mutex> lock(solver_mu_);
        if (solver_.mask != 0u)
            CheckResult(nuka_world_set_telemetry_mask(bind_.world, solver_.mask, solver_.alert),
                        "nuka_world_set_telemetry_mask");
        CheckResult(nuka_world_get_velocity_iterations(bind_.world, &solver_.iterations),
                    "nuka_world_get_velocity_iterations");
    }
    {
        std::lock_guard<std::mutex> lock(joints_mu_);
        joint_log_.clear();
        joint_count_ = 0u;
        ++joint_session_;
    }
    ResetHistory();
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
    ramp_ = TargetRamp{};
    track_ = TargetTrack{};
    demo_envs_.clear();
    branches_.clear();
}

uint64_t PhysicsWorker::Push(TickCommand cmd) {
    uint64_t seq = 0u;
    {
        std::lock_guard<std::mutex> lock(mu_);
        seq = cmd.seq = ++next_seq_;
        cmd.pushed_tick = tick_mirror_.load(std::memory_order_relaxed);
        queue_.push_back(std::move(cmd));
        interrupt_.store(true, std::memory_order_relaxed);
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

SolverStats PhysicsWorker::ReadSolver(uint64_t since, std::vector<nuka_telemetry_record_t>* out) {
    std::lock_guard<std::mutex> lock(solver_mu_);
    const uint64_t first = solver_.records - solver_log_.size();
    const uint64_t from = std::min(std::max(since, first), solver_.records);
    out->assign(solver_log_.begin() + static_cast<std::ptrdiff_t>(from - first), solver_log_.end());
    return solver_;
}

bool PhysicsWorker::TakeBeauty(BeautyImage* out) {
    std::lock_guard<std::mutex> lock(beauty_mu_);
    if (!beauty_ready_) return false;
    *out = std::move(beauty_);
    beauty_ready_ = false;
    return true;
}

bool PhysicsWorker::TakeGainTrace(GainTrace* out) {
    std::lock_guard<std::mutex> lock(gain_mu_);
    if (!gain_ready_) return false;
    *out = std::move(gain_);
    gain_ready_ = false;
    return true;
}

bool PhysicsWorker::TakeIk(IkReport* out) {
    std::lock_guard<std::mutex> lock(ik_mu_);
    if (!ik_ready_) return false;
    *out = std::move(ik_);
    ik_ready_ = false;
    return true;
}

bool PhysicsWorker::TakeSensorReading(SensorReading* out) {
    std::lock_guard<std::mutex> lock(sensor_mu_);
    if (!sensor_ready_) return false;
    *out = std::move(sensor_);
    sensor_ready_ = false;
    return true;
}

uint64_t PhysicsWorker::ReadJoints(uint64_t since, std::vector<JointSample>* out, uint64_t* session) {
    std::lock_guard<std::mutex> lock(joints_mu_);
    const uint64_t first = joint_count_ - joint_log_.size();
    const uint64_t from = std::min(std::max(since, first), joint_count_);
    out->assign(joint_log_.begin() + static_cast<std::ptrdiff_t>(from - first), joint_log_.end());
    *session = joint_session_;
    return joint_count_;
}

void PhysicsWorker::Report(uint64_t seq, std::string text, bool error) {
    if (text.empty() && seq == 0u) return;
    TickResult result;
    result.seq = seq;
    result.text = std::move(text);
    result.error = error;
    Report(std::move(result));
}

void PhysicsWorker::Report(TickResult result) {
    std::lock_guard<std::mutex> lock(mu_);
    results_.push_back(std::move(result));
}

void PhysicsWorker::SetTick(uint64_t tick) {
    tick_ = tick;
    tick_mirror_.store(tick, std::memory_order_relaxed);
}

void PhysicsWorker::Loop() {
    if (bound_) Guarded([&] { BindWorldDevice(bind_.world); });
    std::unique_lock<std::mutex> lock(mu_);
    while (!stop_) {
        interrupt_.store(false, std::memory_order_relaxed);
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
    exited_.store(true, std::memory_order_release);
}

bool PhysicsWorker::ApplyCommands(std::vector<TickCommand>& cmds) {
    bool publish = false;
    for (size_t i = 0; i < cmds.size(); ++i) {
        TickCommand& cmd = cmds[i];
        if (!bound_) {
            if (cmd.kind == CommandKind::ScriptCall) Apply(cmd);
        } else if (!Superseded(cmds, i)) {
            if (cmd.kind == CommandKind::EditApply || cmd.kind == CommandKind::Teleport || cmd.kind == CommandKind::IkSolve)
                cmd.env = bind_.env;
            if (cmd.kind == CommandKind::IkSolve && cmd.ik.write == 0u)
                cmd.ik.write = playing_ ? NUKA_IK_WRITE_TARGETS : NUKA_IK_WRITE_POSITIONS | NUKA_IK_WRITE_TARGETS;
            cmd.perturb.env = bind_.env;
            if (cmd.kind == CommandKind::ParamSet && cmd.param == ParamKind::TargetRamp) StampRamp(cmd);
            if (Recorded(cmd)) Record(cmd);
            publish = (cmd.kind == CommandKind::ScriptCall ? ApplyScript(cmd) : Apply(cmd)) || publish;
        }
        acked_ = cmd.seq;
    }
    return publish && bound_;
}

// True when the command changed what the next snapshot shows.
bool PhysicsWorker::Apply(const TickCommand& cmd) {
    switch (cmd.kind) {
        case CommandKind::EditApply:
        case CommandKind::Teleport:
            ApplyPoseWrite(bind_.world, bind_.layout, cmd.env, cmd.pose);
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
                    std::copy_n(cmd.values.begin(), std::min(cmd.values.size(), command_.size()), command_.begin());
                    if (bind_.controller != nullptr)
                        bind_.controller->SetCommandVector(cmd.values.data(), static_cast<uint32_t>(cmd.values.size()));
                    return false;
                case ParamKind::DisplayEnv:
                    bind_.env = cmd.env % bind_.layout.env_count;
                    bind_.shown.clear();
                    for (const float e : cmd.values) bind_.shown.push_back(static_cast<uint32_t>(e) % bind_.layout.env_count);
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
                case ParamKind::Registry: {
                    // A refused write reports without pausing; the transport keeps running.
                    TickResult r;
                    r.seq = cmd.seq;
                    r.pushed_tick = cmd.pushed_tick;
                    r.applied_tick = tick_;
                    const nuka_result_t result =
                        nuka_world_set_param(bind_.world, cmd.index, cmd.element, cmd.values[0]);
                    r.error = result != NUKA_RESULT_OK;
                    if (r.error)
                        r.text = std::string("[nuka_editor] parameter write refused: ") + nuka_result_message(result) + "\n";
                    Report(std::move(r));
                    return false;
                }
                case ParamKind::Telemetry: {
                    const auto mask = static_cast<uint32_t>(cmd.values[0]);
                    CheckResult(nuka_world_set_telemetry_mask(bind_.world, mask, cmd.values[1]),
                                "nuka_world_set_telemetry_mask");
                    std::lock_guard<std::mutex> lock(solver_mu_);
                    solver_log_.clear();
                    solver_ = SolverStats{solver_.session + 1u, mask, cmd.values[1], solver_.iterations};
                    return true;
                }
                case ParamKind::Iterations: {
                    CheckResult(nuka_world_set_velocity_iterations(bind_.world, static_cast<uint32_t>(cmd.values[0])),
                                "nuka_world_set_velocity_iterations");
                    std::lock_guard<std::mutex> lock(solver_mu_);
                    CheckResult(nuka_world_get_velocity_iterations(bind_.world, &solver_.iterations),
                                "nuka_world_get_velocity_iterations");
                    return true;
                }
                case ParamKind::Heatmap:
                    heatmap_ = cmd.values[0] != 0.0f;
                    last_mirror_ = WorkerClock::time_point{};
                    return true;
                case ParamKind::JointWatch: {
                    // Stopping keeps the recorded samples; starting begins a new record.
                    watch_ = cmd.values[0] != 0.0f;
                    if (!watch_) return false;
                    std::lock_guard<std::mutex> lock(joints_mu_);
                    joint_log_.clear();
                    joint_count_ = 0u;
                    ++joint_session_;
                    return false;
                }
                case ParamKind::TargetRamp:
                    StartRamp(cmd, tick_);
                    return true;
                case ParamKind::TargetTrack:
                    // A track of no rows stops the one driving.
                    track_ = cmd.index == 0u
                                 ? TargetTrack{}
                                 : TargetTrack{tick_, cmd.index, cmd.element, cmd.env, cmd.values, cmd.offset != 0u};
                    return true;
                case ParamKind::TeleopAxes:
                    axes_ = cmd.values;
                    return false;
            }
            return false;
        case CommandKind::Perturb:
            if (cmd.perturb.hold || cmd.perturb.particle != kNoParticle ||
                cmd.perturb.source.kind != render::PoseSource::Kind::Static) {
                Perturb(cmd.perturb);
            } else {
                std::vector<Perturbation> held;
                for (const Perturbation& p : perturbs_)
                    if (p.env == cmd.perturb.env) held.push_back(p);
                for (Perturbation& p : held) {
                    p.hold = false;
                    Perturb(p);
                }
            }
            return true;
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
                    SetTick(0u);
                    sim_time_ = 0.0;
                    playing_ = false;
                    step_budget_ = 0u;
                    perturbs_.clear();
                    ramp_ = TargetRamp{};
                    track_ = TargetTrack{};
                    last_mirror_ = WorkerClock::time_point{};
                    Report(cmd.seq, scripts_->DispatchReset());
                    ResetHistory();
                    break;
            }
            return true;
        case CommandKind::ScriptCall: {
            TickResult result;
            result.seq = cmd.seq;
            result.text = scripts_->RunString(cmd.code, cmd.filename, cmd.echo, &result.error);
            result.streamed = true;
            Report(std::move(result));
            last_mirror_ = WorkerClock::time_point{};
            return true;
        }
        case CommandKind::Timeline:
            ApplyTimeline(cmd);
            return true;
        case CommandKind::Beauty: {
            // Renders the live world at this boundary; physics waits, nothing is published.
            BeautyImage image;
            image.request = cmd.seq;
            image.width = cmd.width;
            image.height = cmd.height;
            image.spp = cmd.spp;
            image.rgb.resize(static_cast<size_t>(cmd.width) * cmd.height * 3u);
            const nuka_result_t r = nuka_world_render_beauty(bind_.world, &cmd.camera, cmd.width, cmd.height, cmd.spp, 0u,
                                                              image.rgb.data(), image.rgb.size(), nullptr);
            if (r != NUKA_RESULT_OK) {
                image.rgb.clear();
                image.error = nuka_result_message(r);
            }
            std::lock_guard<std::mutex> lock(beauty_mu_);
            beauty_ = std::move(image);
            beauty_ready_ = true;
            return false;
        }
        case CommandKind::GainTest:
            RunGainTest(cmd);
            return false;
        case CommandKind::IkSolve:
            SolveIk(cmd);
            return true;
        case CommandKind::SensorRead: {
            // Reads the live world at this boundary; physics waits, nothing is published.
            SensorReading reading = ReadSensor(bind_.world, cmd.index, cmd.element, cmd.env);
            reading.request = cmd.seq;
            reading.tick = tick_;
            reading.sim_time = sim_time_;
            std::lock_guard<std::mutex> lock(sensor_mu_);
            sensor_ = std::move(reading);
            sensor_ready_ = true;
            return false;
        }
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
    const uint32_t budget = step_budget_;
    due += budget;
    step_budget_ = 0u;
    if (due == 0u) return (dt - accum_) / speed;

    bool healthy = true;
    uint32_t done = 0u;
    while (done < due) {
        // Replayed ticks already ran their callbacks when they were recorded.
        const bool recording = !Replaying();
        healthy = StepOnce() && healthy;
        ++done;
        // A queued command or park request ends the batch so it lands at this boundary.
        const bool yield = done < due && interrupt_.load(std::memory_order_relaxed);
        if (done == due || yield || WorkerClock::now() - last_publish_ >= kBatchPublishPeriod) {
            healthy_ = healthy;
            telemetry_.step_ms = Ms(WorkerClock::now() - now);
            telemetry_.batch_steps = done;
            Publish();
        }
        if (recording) DispatchTick();
        if (yield) break;
    }
    // Steps a yield left return: explicit ones to the step budget, the rest to the clock.
    const uint32_t left = due - done;
    const uint32_t budget_left = std::min(left, budget);
    step_budget_ += budget_left;
    accum_ += static_cast<double>(left - budget_left) * dt;
    return 0.0;
}

// One step. At the head it may capture a keyframe first; behind it the log replays, and a
// recorded keyframe the replay misses replaces the replayed state.
bool PhysicsWorker::StepOnce() {
    const bool recording = !Replaying();
    if (recording && (key_due_ || tick_ >= Current().keys.back()->tick + key_interval_)) {
        CaptureKey(key_due_);
        key_due_ = false;
    }
    WritePerturbations();
    if (!ramp_.values.empty()) WriteRamp();
    // A target track drives in place of the controller.
    const bool tracking = !track_.values.empty();
    if (tracking) WriteTrack();
    if (bind_.controller != nullptr && !tracking) bind_.controller->OnStep(bind_.world, bind_.env);
    const bool healthy = nuka_world_step(bind_.world) == NUKA_RESULT_OK;
    SetTick(tick_ + 1u);
    sim_time_ += static_cast<double>(bind_.dt);
    if (watch_ && bind_.layout.links_per_env != 0u) SampleJoints();
    HistoryBranch& b = Current();
    if (recording) {
        if (!demo_envs_.empty()) SampleDemo();
        b.head = tick_;
        return healthy;
    }
    const auto at = std::lower_bound(b.keys.begin(), b.keys.end(), tick_,
                                     [](const std::shared_ptr<Keyframe>& k, uint64_t t) { return k->tick < t; });
    if (at == b.keys.end() || (*at)->tick != tick_) {
        ApplyLogged();
        return healthy;
    }
    const Keyframe& key = **at;
    ApplyLogged(key.log_pos);
    if (key.pinned) {
        Restore(key);
    } else if (!Matches(key.state)) {
        b.exact = false;
        ++history_revision_;
        Restore(key);
    } else {
        ApplyLogged();
    }
    return healthy;
}

void PhysicsWorker::Perturb(Perturbation p) {
    const auto it = std::find_if(perturbs_.begin(), perturbs_.end(), [&](const Perturbation& q) { return SameTarget(q, p); });
    const bool held = it != perturbs_.end();
    if (held) {
        p.last = it->last;
        p.primed = it->primed;
        p.inv_mass = it->inv_mass;
        perturbs_.erase(it);
    }
    // A held particle is kinematic at its target; the release gives back its inverse mass.
    if (p.particle != kNoParticle) {
        const size_t row = static_cast<size_t>(p.env) * bind_.layout.particles_per_env + p.particle;
        if (!held) p.inv_mass = DownloadRow<float>(bind_.world, NUKA_FIELD_PARTICLE_INV_MASS, row);
        const float inv_mass = p.hold ? 0.0f : p.inv_mass;
        CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_PARTICLE_INV_MASS, &inv_mass, sizeof(float),
                                            row * sizeof(float)),
                    "upload particle inverse mass");
        if (p.hold)
            CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_PARTICLE_KINEMATIC_TARGET, &p.target,
                                                sizeof(math::Vec3), row * sizeof(math::Vec3)),
                        "upload particle target");
    }
    if (p.hold) perturbs_.push_back(p);
}

// Each held load acts at its point; a spring adds k (target - point) - c v, v differenced over a step.
void PhysicsWorker::WritePerturbations() {
    const WorldLayout& layout = bind_.layout;
    for (Perturbation& p : perturbs_) {
        if (p.particle != kNoParticle) continue;
        const bool link = p.source.kind == render::PoseSource::Kind::Link;
        const size_t row =
            static_cast<size_t>(p.env) * (link ? layout.links_per_env : layout.bodies_per_env) + p.source.row;
        const math::Transform pose = DownloadRow<math::Transform>(
            bind_.world, link ? NUKA_FIELD_ARTICULATION_LINK_POSE : NUKA_FIELD_RIGID_BODY_TRANSFORM, row);
        const math::Vec3 at = pose.TransformPoint(p.point);
        math::Vec3 f = Vec(p.force);
        if (p.stiffness > 0.0f) {
            const math::Vec3 v = p.primed ? (at - p.last) * (1.0f / bind_.dt) : math::Vec3::Zero();
            f = f + (p.target - at) * p.stiffness - v * p.damping;
            p.last = at;
            p.primed = true;
        }
        const math::Vec3 t = Vec(p.torque) + (at - pose.TransformPoint(p.com)).Cross(f);
        const float wrench[6] = {f.x, f.y, f.z, t.x, t.y, t.z};
        if (link) {
            CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_LINK_WRENCH, wrench, sizeof(wrench),
                                                row * sizeof(wrench)),
                        "upload link wrench");
        } else {
            CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_BODY_FORCE, wrench, 3u * sizeof(float),
                                                row * 3u * sizeof(float)),
                        "upload body force");
            CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_BODY_TORQUE, wrench + 3, 3u * sizeof(float),
                                                row * 3u * sizeof(float)),
                        "upload body torque");
        }
    }
}

// A ramp starts from the shown environment's targets as they are when it lands.
void PhysicsWorker::StampRamp(TickCommand& cmd) {
    cmd.env = bind_.env;
    const size_t row = static_cast<size_t>(cmd.env) * bind_.layout.links_per_env;
    for (size_t i = 0; i + 2u < cmd.values.size(); i += 3u)
        cmd.values[i + 1u] =
            DownloadRow<float>(bind_.world, NUKA_FIELD_DRIVE_TARGET, row + static_cast<size_t>(cmd.values[i]));
}

void PhysicsWorker::StartRamp(const TickCommand& cmd, uint64_t start) {
    ramp_ = TargetRamp{start, cmd.index, cmd.env, cmd.values};
    if (ramp_.steps == 0u) WriteRamp();
}

// The targets for the step from the current tick, a share (tick - start + 1) / steps of the way;
// the last step's write ends the ramp.
void PhysicsWorker::WriteRamp() {
    const float alpha = ramp_.steps == 0u ? 1.0f
                                          : std::min(1.0f, static_cast<float>(tick_ + 1u - ramp_.start) /
                                                               static_cast<float>(ramp_.steps));
    const size_t row = static_cast<size_t>(ramp_.env) * bind_.layout.links_per_env;
    for (size_t i = 0; i + 2u < ramp_.values.size(); i += 3u) {
        const float target = ramp_.values[i + 1u] + alpha * (ramp_.values[i + 2u] - ramp_.values[i + 1u]);
        CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_DRIVE_TARGET, &target, sizeof(float),
                                            (row + static_cast<size_t>(ramp_.values[i])) * sizeof(float)),
                    "upload drive target");
    }
    if (alpha >= 1.0f) ramp_ = TargetRamp{};
}

// The track's row for the step from the current tick; the last row ends the track unless it loops.
void PhysicsWorker::WriteTrack() {
    const uint64_t row = (tick_ - track_.start) % track_.rows;
    const size_t links = bind_.layout.links_per_env, offset = static_cast<size_t>(track_.env) * links * sizeof(float);
    track_scratch_.resize(links);
    CheckResult(nuka_world_download_field(bind_.world, NUKA_FIELD_DRIVE_TARGET, track_scratch_.data(),
                                          links * sizeof(float), offset),
                "nuka_world_download_field");
    const float* values = track_.values.data() + track_.columns * (row + 1u);
    for (uint32_t c = 0; c < track_.columns; ++c) track_scratch_[static_cast<size_t>(track_.values[c])] = values[c];
    CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_DRIVE_TARGET, track_scratch_.data(), links * sizeof(float),
                                        offset),
                "upload drive target");
    if (!track_.loop && row + 1u >= track_.rows) track_ = TargetTrack{};
}

void PhysicsWorker::RecordDemo(std::vector<uint32_t> envs) {
    demo_envs_ = std::move(envs);
    axes_.clear();
    demo_roots_.clear();
    for (uint32_t i = 0; i < bind_.layout.links_per_env; ++i)
        if (bind_.layout.links[i].is_root != 0u) demo_roots_.push_back(i);
}

std::vector<DemoSample> PhysicsWorker::TakeDemo() {
    std::vector<DemoSample> out;
    out.swap(demo_);
    demo_count_.store(0u, std::memory_order_relaxed);
    return out;
}

void PhysicsWorker::SampleDemo() {
    for (const uint32_t env : demo_envs_) {
        DemoSample& s = demo_.emplace_back();
        s.tick = tick_;
        s.time = sim_time_;
        DownloadDemoSample(bind_.world, bind_.layout, env, demo_roots_, &s);
        s.command = command_;
        s.axes = axes_;
    }
    demo_count_.store(demo_.size(), std::memory_order_relaxed);
}

// Steps the drive alone, without loads, ramp or policy, writing one link's target each step,
// then puts the world back; the trace starts with the state before the first step.
void PhysicsWorker::RunGainTest(const TickCommand& cmd) {
    Keyframe before;
    CheckResult(nuka_world_checkpoint_capture(bind_.world, &before.state), "nuka_world_checkpoint_capture");
    GainTrace trace;
    trace.request = cmd.seq;
    trace.link = cmd.index;
    trace.sine = cmd.values[0] != 0.0f;
    trace.amplitude = cmd.values[1];
    trace.frequency = cmd.values[2];
    const size_t row = static_cast<size_t>(bind_.env) * bind_.layout.links_per_env + cmd.index;
    trace.base = DownloadRow<float>(bind_.world, NUKA_FIELD_DRIVE_TARGET, row);
    const auto sample = [&](float time, float target) {
        trace.time.push_back(time);
        trace.target.push_back(target);
        trace.q.push_back(DownloadRow<float>(bind_.world, NUKA_FIELD_JOINT_POSITION, row));
        trace.qd.push_back(DownloadRow<float>(bind_.world, NUKA_FIELD_JOINT_VELOCITY, row));
        trace.effort.push_back(DownloadRow<float>(bind_.world, NUKA_FIELD_ACTUATOR_EFFORT, row));
        trace.saturated.push_back(DownloadRow<float>(bind_.world, NUKA_FIELD_ACTUATOR_SATURATED, row));
    };
    sample(0.0f, trace.base);
    const uint32_t steps = static_cast<uint32_t>(std::lround(cmd.values[3] / bind_.dt));
    for (uint32_t i = 0; i < steps; ++i) {
        const float t = static_cast<float>(i) * bind_.dt;
        const float shape = trace.sine ? std::sin(2.0f * 3.14159265f * trace.frequency * t) : 1.0f;
        const float target = trace.base + trace.amplitude * shape;
        CheckResult(nuka_world_upload_field(bind_.world, NUKA_FIELD_DRIVE_TARGET, &target, sizeof(float),
                                            row * sizeof(float)),
                    "upload drive target");
        nuka_world_step(bind_.world);
        sample(static_cast<float>(i + 1u) * bind_.dt, target);
    }
    CheckResult(nuka_world_checkpoint_restore(bind_.world, before.state), "nuka_world_checkpoint_restore");
    std::lock_guard<std::mutex> lock(gain_mu_);
    gain_ = std::move(trace);
    gain_ready_ = true;
}

IkReport RunIk(nuka_world_handle world, uint32_t links_per_env, const TickCommand& cmd) {
    IkReport report;
    report.request = cmd.seq;
    report.q.resize(links_per_env);
    report.targets.resize(links_per_env);
    report.position_error.resize(cmd.ik_targets.size());
    report.rotation_error.resize(cmd.ik_targets.size());
    nuka_ik_request_t request = cmd.ik;
    request.env = cmd.env;
    request.targets = cmd.ik_targets.data();
    request.target_count = static_cast<uint32_t>(cmd.ik_targets.size());
    nuka_ik_result_t result{report.q.data(), report.position_error.data(), report.rotation_error.data(), 0u, 0u, 0u};
    const nuka_result_t r = nuka_world_solve_ik(world, &request, &result);
    if (r != NUKA_RESULT_OK) report.error = nuka_result_message(r);
    report.iterations = result.iterations;
    report.joints = result.joints;
    report.at_limit = result.at_limit;
    const size_t bytes = static_cast<size_t>(links_per_env) * sizeof(float);
    CheckResult(nuka_world_download_field(world, NUKA_FIELD_DRIVE_TARGET, report.targets.data(), bytes, cmd.env * bytes),
                "nuka_world_download_field");
    return report;
}

void PhysicsWorker::SolveIk(const TickCommand& cmd) {
    IkReport report = RunIk(bind_.world, bind_.layout.links_per_env, cmd);
    last_mirror_ = WorkerClock::time_point{};
    std::lock_guard<std::mutex> lock(ik_mu_);
    ik_ = std::move(report);
    ik_ready_ = true;
}

void PhysicsWorker::DispatchTick() {
    if (!scripts_->Valid()) return;
    std::string text = scripts_->DispatchStep();
    text += scripts_->DispatchTick();
    Report(0u, std::move(text));
}

// A sample at or before the last one starts a new session: the playhead went back.
void PhysicsWorker::SampleJoints() {
    JointSample sample;
    sample.tick = tick_;
    sample.time = sim_time_;
    DownloadJointSample(bind_.world, bind_.layout, bind_.env, &sample);
    std::lock_guard<std::mutex> lock(joints_mu_);
    if (!joint_log_.empty() && joint_log_.back().tick >= sample.tick) {
        joint_log_.clear();
        joint_count_ = 0u;
        ++joint_session_;
    }
    joint_log_.push_back(std::move(sample));
    ++joint_count_;
    while (joint_log_.size() > kJointLog) joint_log_.pop_front();
}

// Appends the records written since the last read; the worker is the log's only writer.
void PhysicsWorker::ReadTelemetry() {
    records_scratch_.resize(1024u);
    uint32_t count = 0u, dropped = 0u;
    CheckResult(nuka_world_read_telemetry(bind_.world, records_scratch_.data(),
                                          static_cast<uint32_t>(records_scratch_.size()), &count, &dropped),
                "nuka_world_read_telemetry");
    std::lock_guard<std::mutex> lock(solver_mu_);
    solver_.records += count;
    solver_.dropped += dropped;
    for (uint32_t i = 0; i < count; ++i) {
        solver_.caps += Capped(records_scratch_[i]) ? 1u : 0u;
        solver_log_.push_back(records_scratch_[i]);
    }
    while (solver_log_.size() > kSolverLog) solver_log_.pop_front();
    const auto recent = static_cast<std::ptrdiff_t>(std::min<size_t>(solver_log_.size(), kRecentRecords));
    solver_.recent_caps = static_cast<uint32_t>(std::count_if(solver_log_.end() - recent, solver_log_.end(), Capped));
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
    mirrored_ = bind_.scatter == nullptr || !playing_ || start - last_mirror_ >= kMirrorPeriod;
    if (mirrored_) {
        DownloadEnvPoses(bind_.world, bind_.layout, bind_.env, &mirror_links_, &mirror_bodies_);
        mirror_envs_.resize(bind_.shown.size());
        for (size_t i = 0; i < bind_.shown.size(); ++i) {
            mirror_envs_[i].tick = tick_;
            mirror_envs_[i].env = bind_.shown[i];
            DownloadEnvPoses(bind_.world, bind_.layout, bind_.shown[i], &mirror_envs_[i].links, &mirror_envs_[i].bodies);
        }
        DownloadEnvParticles(bind_.world, bind_.layout, bind_.env, &mirror_particles_);
        if (bind_.layout.links_per_env != 0u) DownloadJointFrame(bind_.world, bind_.layout, bind_.env, &mirror_joints_);
        ReadOverlayData(bind_.world, bind_.layout, bind_.env, bind_.colliders, bind_.contacts, &overlay_);
        if (solver_.mask != 0u) ReadTelemetry();
        mirror_heat_.clear();
        const size_t particles = bind_.layout.particles_per_env;
        if (heatmap_ && (solver_.mask & NUKA_TELEMETRY_VERTICES) != 0u && particles != 0u) {
            vertex_scratch_.resize(particles * bind_.layout.env_count);
            size_t count = 0u;
            CheckResult(nuka_world_read_telemetry_vertices(bind_.world, vertex_scratch_.data(), vertex_scratch_.size(),
                                                           &count),
                        "nuka_world_read_telemetry_vertices");
            const auto begin = vertex_scratch_.begin() + static_cast<std::ptrdiff_t>(bind_.env * particles);
            mirror_heat_.assign(begin, begin + static_cast<std::ptrdiff_t>(particles));
        }
        last_mirror_ = start;
    }
    s.link_poses = mirror_links_;
    s.body_poses = mirror_bodies_;
    s.envs = mirror_envs_;
    s.particles = mirror_particles_;
    s.heat = mirror_heat_;
    s.joints = mirror_joints_;
    s.perturbs.clear();
    for (const Perturbation& p : perturbs_)
        if (p.env == bind_.env) s.perturbs.push_back(p);
    s.overlay = overlay_;
    s.track_left = track_.values.empty() ? 0u : track_.rows - static_cast<uint32_t>((tick_ - track_.start) % track_.rows);
    s.published = WorkerClock::now();
    last_publish_ = s.published;
    telemetry_.publish_ms = Ms(s.published - start);
    s.telemetry = telemetry_;
    PublishTimeline(s);
    ring_.Publish();
}

}  // namespace nuka::runtime::app::viewer
