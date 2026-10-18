#include "runtime/app/viewer/physics_worker.hpp"

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
            if (cmd.kind == CommandKind::EditApply || cmd.kind == CommandKind::Teleport) cmd.env = bind_.env;
            cmd.perturb.env = bind_.env;
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
    if (bind_.controller != nullptr) bind_.controller->OnStep(bind_.world, bind_.env);
    const bool healthy = nuka_world_step(bind_.world) == NUKA_RESULT_OK;
    SetTick(tick_ + 1u);
    sim_time_ += static_cast<double>(bind_.dt);
    if (watch_ && bind_.layout.links_per_env != 0u) SampleJoints();
    HistoryBranch& b = Current();
    if (recording) {
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
    s.particles = mirror_particles_;
    s.heat = mirror_heat_;
    s.joints = mirror_joints_;
    s.perturbs.clear();
    for (const Perturbation& p : perturbs_)
        if (p.env == bind_.env) s.perturbs.push_back(p);
    s.overlay = overlay_;
    s.published = WorkerClock::now();
    last_publish_ = s.published;
    telemetry_.publish_ms = Ms(s.published - start);
    s.telemetry = telemetry_;
    PublishTimeline(s);
    ring_.Publish();
}

}  // namespace nuka::runtime::app::viewer
