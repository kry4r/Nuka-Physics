#pragma once
// The editor's physics thread. While it runs it makes every world call: typed commands
// apply in order at tick boundaries, each publish hands the UI a snapshot, and a history of
// keyframes and inputs lets the playhead seek, replay and fork.

#include "math/transform.hpp"
#include "nuka/nuka.h"
#include "render/render_world.hpp"
#include "runtime/app/viewer/debug_overlay.hpp"
#include "runtime/app/viewer/demos.hpp"
#include "runtime/app/viewer/editor_world.hpp"
#include "runtime/app/viewer/pose_scatter.hpp"
#include "runtime/app/viewer/robot_view.hpp"
#include "runtime/app/viewer/script_host.hpp"
#include "runtime/app/viewer/sensor_view.hpp"
#include "runtime/app/viewer/timeline_view.hpp"
#include "runtime/inference/world_controller.hpp"
#include "scene/format/json.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nuka::runtime::app::viewer {

using WorkerClock = std::chrono::steady_clock;

enum class CommandKind : uint8_t {
    EditApply, Teleport, ParamSet, Perturb, Transport, ScriptCall, Timeline, Beauty, GainTest, IkSolve, SensorRead,
    Observe
};
enum class ParamKind : uint8_t {
    Field, HoldGains, ControllerCommand, DisplayEnv, ScatterRows, Overlay, Registry, Telemetry, Iterations, Heatmap,
    JointWatch, TargetRamp, TargetTrack, TeleopAxes, Randomize
};
enum class TransportOp : uint8_t { Play, Pause, Step, Speed, Reset };
enum class TimelineOp : uint8_t { Seek, Fork, Branch, Rename, Ghost, Configure, Clear };

inline constexpr uint32_t kNoParticle = ~0u;

// A load on a link or free body, rewritten before every step, or a particle held kinematic at a target;
// a later command for the same target replaces or releases it, a release naming none releases them all.
struct Perturbation {
    render::PoseSource source;
    uint32_t particle = kNoParticle;  // per-environment index of a held particle
    uint32_t env = 0u;                // the worker stamps the shown environment
    bool hold = true;
    float force[3] = {0.0f, 0.0f, 0.0f};   // world [N], acting at `point`
    float torque[3] = {0.0f, 0.0f, 0.0f};  // world [N m]
    math::Vec3 point = math::Vec3::Zero();   // body frame
    math::Vec3 com = math::Vec3::Zero();     // body frame
    math::Vec3 target = math::Vec3::Zero();  // world: where the spring pulls `point`, or the particle
    float stiffness = 0.0f;  // [N/m]; zero holds the plain load
    float damping = 0.0f;    // [N s/m]
    // The worker's: the point a step ago for the damping, and the particle's own inverse mass.
    math::Vec3 last = math::Vec3::Zero();
    bool primed = false;
    float inv_mass = 0.0f;
};

struct TickCommand {
    CommandKind kind = CommandKind::Transport;
    uint64_t seq = 0u;                    // assigned by Push
    uint64_t pushed_tick = 0u;            // the worker's tick when Push queued it
    PoseWrite pose;                       // EditApply, Teleport
    ParamKind param = ParamKind::Field;   // ParamSet
    nuka_state_field_t field = NUKA_FIELD_DRIVE_TARGET;
    size_t offset = 0u;                   // bytes into the field; TargetTrack: 1 repeats its rows
    // Registry parameter; TargetRamp: its length in steps; TargetTrack: its rows; GainTest: the link.
    // SensorRead: the camera, lidar or state sensor among its kind; Randomize: the seed; Observe: the
    // environments.
    uint32_t index = 0u;
    // Registry element; TargetTrack: the links of a row; SensorRead: a nuka_sensor_channel_t or kStatePlane.
    uint32_t element = 0u;
    // Field values, gains (kp, kd), command, overlay flags, registry value, telemetry (mask, alert),
    // sweep budget, heatmap or joint watch flag, ghost flag, the keyframe interval and budget [MB],
    // a target ramp's (link, from, to) triples, the worker stamping `from`, a target track's links
    // then its rows, the teleop axes, a gain test's (sine, amplitude, frequency, duration), a
    // randomization as PhysicsWorker::Randomize reads it, or what PhysicsWorker::Observe writes.
    std::vector<float> values;
    // DisplayEnv, TargetTrack, SensorRead; the worker stamps pose writes with the shown one.
    uint32_t env = 0u;
    std::vector<ScatterRow> rows;         // ScatterRows
    Perturbation perturb;                 // Perturb
    TransportOp op = TransportOp::Pause;  // Transport
    uint32_t steps = 0u;
    float speed = 1.0f;
    std::string code;                     // ScriptCall
    std::string filename;                 // ScriptCall: what tracebacks name; empty for the console
    bool echo = false;                    // ScriptCall: print a final expression's value
    TimelineOp timeline = TimelineOp::Seek;  // Timeline
    uint64_t tick = 0u;                      // Seek target
    int64_t delta = 0;                       // Seek offset from the playhead when `relative`
    bool relative = false;
    uint32_t branch = kNoBranch;             // Branch, Rename, Ghost
    std::string label;                       // Fork, Rename
    nuka_beauty_camera_t camera{};           // Beauty: the view, its pixel size and samples per pixel
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t spp = 0u;
    // IkSolve: the solve's settings and targets; a zero `write` takes the transport's, drive
    // targets while playing and the joints at rest as well while paused.
    nuka_ik_request_t ik{};
    std::vector<nuka_ik_target_t> ik_targets;

    static TickCommand Transport(TransportOp op);
    static TickCommand Param(ParamKind param, std::vector<float> values);
    static TickCommand Script(std::string code, std::string filename = std::string(), bool echo = false);
    static TickCommand Timeline(TimelineOp op);
};

// The world after every input of its tick so far, with the loads then held.
struct Keyframe {
    uint64_t tick = 0u;
    double sim_time = 0.0;
    size_t log_pos = 0u;  // the branch inputs already in this state
    std::vector<Perturbation> perturbs;
    nuka_checkpoint_handle state = nullptr;
    uint64_t bytes = 0u;
    bool pinned = false;  // holds writes the input log cannot replay

    Keyframe() = default;
    Keyframe(const Keyframe&) = delete;
    Keyframe& operator=(const Keyframe&) = delete;
    ~Keyframe() { nuka_checkpoint_destroy(state); }
};

struct LoggedInput {
    uint64_t tick = 0u;
    TickCommand cmd;
};

// The shown environment's poses at one tick, for the ghost.
struct PoseSample {
    uint64_t tick = 0u;
    uint32_t env = 0u;
    std::vector<math::Transform> links;
    std::vector<math::Transform> bodies;
};

// One line of history. Its log and the keyframes before its fork are the parent's up to there.
struct HistoryBranch {
    uint32_t id = 0u;
    uint32_t parent = kNoBranch;
    uint64_t fork_tick = 0u;
    uint64_t head = 0u;
    std::string label;
    bool exact = true;
    uint64_t used = 0u;  // the switch count when it was last current
    std::vector<std::shared_ptr<Keyframe>> keys;  // by tick
    std::vector<LoggedInput> log;                 // by tick, from the history's start
    std::deque<PoseSample> track;                 // from the fork on
};

struct TickResult {
    uint64_t seq = 0u;  // the command answered; 0 for callback output
    std::string text;
    bool error = false;
    bool streamed = false;       // the text reached the console as it printed
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

// A path-traced image of environment 0; `error` when the render failed.
struct BeautyImage {
    uint64_t request = 0u;  // the Beauty command's seq
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t spp = 0u;
    std::vector<uint8_t> rgb;
    std::string error;
};

// What an IK command solved on the shown environment, or why the world refused it.
struct IkReport {
    uint64_t request = 0u;               // the IkSolve command's seq
    std::vector<float> q;                // per link
    std::vector<float> targets;          // the drive targets after it, per link
    std::vector<float> position_error;   // per target [m]
    std::vector<float> rotation_error;   // per target [rad]
    uint32_t iterations = 0u;
    uint32_t joints = 0u;
    uint32_t at_limit = 0u;
    std::string error;
};

// Solves an IkSolve command on environment `cmd.env` of `world`, writing what `cmd.ik.write` asks.
IkReport RunIk(nuka_world_handle world, uint32_t links_per_env, const TickCommand& cmd);

// One environment's reading of a sensor: a camera plane of `height` x `width` x `channels`, a
// lidar's ranges by azimuth (`width`) then elevation (`height`), or a state sensor's `width`
// values. PRIM values are uint32 bit patterns; `instances` names their instance field's nodes.
struct SensorReading {
    uint64_t request = 0u;  // the SensorRead command's seq
    uint64_t tick = 0u;
    double sim_time = 0.0;
    uint32_t plane = 0u;
    uint32_t width = 0u, height = 0u, channels = 0u;
    std::vector<float> values;
    std::vector<std::string> instances;
    std::string error;
};

// Renders `world`'s sensors when `plane` is an image, unless `render` is false because a read of this
// tick already did, and reads sensor `index` of its kind in `env`.
SensorReading ReadSensor(nuka_world_handle world, uint32_t index, uint32_t plane, uint32_t env, bool render = true);

// The newest records of this many count toward SolverStats::recent_caps.
inline constexpr uint32_t kRecentRecords = 256u;

// What the worker read of the solver telemetry.
struct SolverStats {
    uint64_t session = 0u;  // bumped whenever the mask is set
    uint32_t mask = 0u;  // nuka_telemetry_bits_t
    float alert = 1.0e-3f;
    uint32_t iterations = 0u;  // the sweep budget
    uint64_t records = 0u;     // read since the mask was set
    uint64_t caps = 0u;
    uint32_t recent_caps = 0u;
    uint64_t dropped = 0u;
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
    std::vector<PoseSample> envs;       // the other displayed environments' poses
    std::vector<math::Vec3> particles;  // the shown environment's
    std::vector<float> heat;            // its per-particle residuals while the heatmap reads them
    JointFrame joints;                  // its joints
    std::vector<Perturbation> perturbs;  // the loads held on it
    OverlayData overlay;
    WorkerTelemetry telemetry;
    uint32_t track_left = 0u;  // the steps the target track still drives
    uint32_t branch = 0u;  // the history branch the playhead is on
    uint64_t head = 0u;    // its newest recorded tick
    // The ghost branch's poses at this tick; empty while it matches or has none.
    std::vector<math::Transform> ghost_links;
    std::vector<math::Transform> ghost_bodies;
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
    std::vector<uint32_t> shown;  // other environments drawn beside it
    float dt = 0.005f;
    float speed = 1.0f;
    bool colliders = false;
    bool contacts = false;
    uint64_t tick = 0u;  // where the timeline continues; a migrating rebuild keeps it
    double sim_time = 0.0;
};

// The JSON manifest a recording starts with; throws when `path` holds no recording.
scene::json::Value ReadRecordingManifest(const std::string& path);

class PhysicsWorker {
public:
    explicit PhysicsWorker(ScriptHost* scripts) : scripts_(scripts) {}
    ~PhysicsWorker() { StopThread(); }
    PhysicsWorker(const PhysicsWorker&) = delete;
    PhysicsWorker& operator=(const PhysicsWorker&) = delete;

    // The thread paces the transport; without it, Pump runs one boundary per frame.
    void StartThread();
    // Joins the thread; `serve` runs meanwhile, so calls the thread waits on can finish.
    void StopThread(const std::function<void()>& serve = {});
    bool Threaded() const { return thread_.joinable(); }
    bool OnThread() const { return std::this_thread::get_id() == thread_.get_id(); }
    // On the worker thread: waits in `wait` counted as parked, so Exclusive sections may run
    // meanwhile, and resumes once none is pending.
    void ParkWhile(const std::function<void()>& wait);

    // Runs fn on the calling thread while the worker is parked at a tick boundary, after
    // every command queued before it.
    void Exclusive(const std::function<void()>& fn);

    // Exclusive sections only. Bind publishes the first snapshot of the new binding and
    // returns its epoch.
    uint64_t Bind(WorkerBinding binding);
    void Unbind();
    // Exclusive sections only. The controller run before each step from now on; null runs none.
    void SetController(inference::WorldController* controller) { bind_.controller = controller; }
    // Exclusive sections only. Each step at the head from now on records environments `envs`
    // for a demonstration; empty stops. TakeDemo moves out what was recorded.
    void RecordDemo(std::vector<uint32_t> envs);
    std::vector<DemoSample> TakeDemo();
    // The samples waiting for TakeDemo; readable from any thread.
    uint64_t DemoCount() const { return demo_count_.load(std::memory_order_relaxed); }
    // Exclusive sections only. Every `every` ticks at the head from now on, starting now, the shown
    // environment's poses wait for TakeStream; 0 stops. Returns the samples the previous stream took.
    uint64_t StreamPoses(uint32_t every);
    // Moves the waiting stream samples into `out` as a recording's track chunk; any thread.
    void TakeStream(std::string* out);
    // Exclusive sections only. Writes the current branch's line of history to `path` after
    // `manifest`, which comes back with the history's own fields; keyframes pass through `scratch`.
    scene::json::Value SaveHistory(const std::string& path, scene::json::Value manifest, const std::string& scratch);
    // Exclusive sections only. The recording at `path` becomes the whole history, one branch,
    // and the playhead returns to where it was saved.
    void LoadHistory(const std::string& path, const std::string& scratch);
    double SimTime() const { return sim_time_; }
    // The last finished tick; readable from any thread.
    uint64_t Tick() const { return tick_mirror_.load(std::memory_order_relaxed); }

    uint64_t Push(TickCommand cmd);
    // Applies the queued commands, publishes and dispatches the tick callback.
    void Pump();
    std::vector<TickResult> TakeResults();
    const Snapshot* AcquireSnapshot() { return ring_.Acquire(); }
    PoseSlotPool& Slots() { return slots_; }
    // Copies the kept records numbered `since` on, 0 being the first since the mask was set;
    // readable from any thread.
    SolverStats ReadSolver(uint64_t since, std::vector<nuka_telemetry_record_t>* out);
    // Copies the history the last snapshot described; readable from any thread.
    void ReadTimeline(TimelineInfo* info);
    // Moves the newest finished beauty render into `out` once; readable from any thread.
    bool TakeBeauty(BeautyImage* out);
    // Moves the newest finished gain test into `out` once; readable from any thread.
    bool TakeGainTrace(GainTrace* out);
    // Moves the newest IK report into `out` once; readable from any thread.
    bool TakeIk(IkReport* out);
    // Moves the newest sensor reading into `out` once; readable from any thread.
    bool TakeSensorReading(SensorReading* out);
    // Copies the joint samples numbered `since` on, 0 being the first of `session`; returns the
    // count taken in it. Readable from any thread.
    uint64_t ReadJoints(uint64_t since, std::vector<JointSample>* out, uint64_t* session);

private:
    void Loop();
    template <class Fn>
    void Guarded(Fn&& fn);
    bool ApplyCommands(std::vector<TickCommand>& cmds);
    bool Apply(const TickCommand& cmd);
    void Perturb(Perturbation p);
    double Advance();
    bool StepOnce();
    void WritePerturbations();
    void StampRamp(TickCommand& cmd);
    void StartRamp(const TickCommand& cmd, uint64_t start);
    void WriteRamp();
    void WriteTrack();
    void SampleDemo();
    void SampleStream();
    void RunGainTest(const TickCommand& cmd);
    void SolveIk(const TickCommand& cmd);
    void Randomize(const TickCommand& cmd);
    void Observe(const TickCommand& cmd);
    void SampleJoints();
    void ReadTelemetry();
    void Publish();
    void DispatchTick();
    void Report(uint64_t seq, std::string text, bool error = false);
    void Report(TickResult result);
    void SetTick(uint64_t tick);

    // History (physics_history.cpp).
    HistoryBranch& Current() { return branches_.at(branch_); }
    bool Replaying() { return tick_ < Current().head; }
    static bool Recorded(const TickCommand& cmd);
    void ResetHistory();
    void CaptureKey(bool pinned);
    void Record(const TickCommand& cmd);
    bool ApplyScript(const TickCommand& cmd);
    void Fork(std::string label);
    void KeyHead();
    void SeekTo(uint64_t target);
    void Restore(const Keyframe& key);
    void ResumeRamp();
    void ResumeTrack();
    void ResumeCommand();
    void ApplyLogged(size_t limit = ~size_t(0));
    bool Matches(nuka_checkpoint_handle state);
    void ApplyTimeline(const TickCommand& cmd);
    void Evict();
    bool EvictOne();
    void Recount();
    void SampleTrack();
    const PoseSample* GhostSample();
    uint64_t Divergence(uint32_t a, uint32_t b) const;
    void PublishTimeline(Snapshot& s);

    ScriptHost* scripts_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<TickCommand> queue_;
    std::vector<TickResult> results_;
    uint64_t next_seq_ = 0u;
    bool stop_ = false;
    bool park_ = false;
    bool parked_ = false;
    std::atomic<bool> exited_{false};
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
    // A pose blend of the shown environment's drive targets, (link, from, to) triples, over
    // `steps` steps from tick `start`; empty `values` is none.
    struct TargetRamp {
        uint64_t start = 0u;
        uint32_t steps = 0u;
        uint32_t env = 0u;
        std::vector<float> values;
    };
    TargetRamp ramp_;
    // Rows of environment `env`'s drive targets, one per step from tick `start`, written in place
    // of the controller, over and over with `loop`; empty `values` is none.
    struct TargetTrack {
        uint64_t start = 0u;
        uint32_t rows = 0u;
        uint32_t columns = 0u;  // the links of a row, the first `columns` values
        uint32_t env = 0u;
        std::vector<float> values;
        bool loop = false;
    };
    TargetTrack track_;
    std::vector<float> track_scratch_;
    // The controller command and teleop axes in force, which demonstrations record.
    std::array<float, 3> command_{};
    std::vector<float> axes_;
    // Each registry parameter a randomization scales: its elements' values before any draw.
    std::map<uint32_t, std::vector<double>> param_nominal_;
    std::vector<uint32_t> demo_envs_;
    std::vector<uint32_t> demo_roots_;
    std::vector<DemoSample> demo_;
    std::atomic<uint64_t> demo_count_{0u};
    uint32_t stream_every_ = 0u;
    uint64_t stream_taken_ = 0u;
    std::mutex stream_mu_;
    std::deque<PoseSample> stream_;
    uint64_t timeline_value_ = 0u;
    uint64_t sequence_ = 0u;
    WorkerClock::time_point last_publish_{};
    WorkerClock::time_point last_mirror_{};
    std::vector<math::Transform> mirror_links_;
    std::vector<math::Transform> mirror_bodies_;
    std::vector<PoseSample> mirror_envs_;
    std::vector<math::Vec3> mirror_particles_;
    std::vector<float> mirror_heat_;
    JointFrame mirror_joints_;
    OverlayData overlay_;
    WorkerTelemetry telemetry_;

    // The telemetry mask, alert and heatmap survive rebinds; the log holds the newest records.
    bool heatmap_ = false;
    std::vector<nuka_telemetry_record_t> records_scratch_;
    std::vector<float> vertex_scratch_;
    std::mutex solver_mu_;
    std::deque<nuka_telemetry_record_t> solver_log_;
    SolverStats solver_;

    // While watched, every step samples the shown environment's joints; a new session starts
    // with each watch, binding or step back.
    bool watch_ = false;
    std::mutex joints_mu_;
    std::deque<JointSample> joint_log_;
    uint64_t joint_count_ = 0u;
    uint64_t joint_session_ = 0u;

    // The history; behind the current branch's head its log replays from `cursor_`.
    std::map<uint32_t, HistoryBranch> branches_;
    uint32_t branch_ = 0u;
    uint32_t next_branch_ = 0u;
    size_t cursor_ = 0u;  // the current branch's inputs applied so far
    uint64_t switches_ = 0u;
    uint32_t key_interval_ = 16u;
    uint64_t key_budget_ = uint64_t{1} << 30u;
    uint64_t key_bytes_ = 0u;
    uint32_t key_count_ = 0u;
    bool key_due_ = false;  // a script changed the world since the last keyframe
    uint32_t ghost_ = kNoBranch;
    bool ghost_on_ = true;
    bool mirrored_ = false;  // this publish downloaded the host pose mirror
    uint64_t history_revision_ = 0u;
    std::mutex timeline_mu_;
    TimelineInfo timeline_;
    std::mutex beauty_mu_;
    BeautyImage beauty_;
    bool beauty_ready_ = false;
    std::mutex gain_mu_;
    GainTrace gain_;
    bool gain_ready_ = false;
    std::mutex ik_mu_;
    IkReport ik_;
    bool ik_ready_ = false;
    std::mutex sensor_mu_;
    SensorReading sensor_;
    bool sensor_ready_ = false;
};

}  // namespace nuka::runtime::app::viewer
