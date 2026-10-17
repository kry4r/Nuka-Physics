// The physics thread's history: a keyframe every few ticks, the inputs between them, and the
// shown environment's poses for the ghost. A seek restores the nearest keyframe and replays.

#include "runtime/app/viewer/physics_worker.hpp"

#include "runtime/app/viewer/editor_scene.hpp"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

// Thinning stops before a gap between keyframes would span more intervals than this.
constexpr uint64_t kMaxGapIntervals = 64u;
// Each branch keeps this many pose samples for the ghost.
constexpr size_t kTrackSamples = size_t{1} << 16u;

using KeyRef = std::shared_ptr<Keyframe>;

}  // namespace

TickCommand TickCommand::Timeline(TimelineOp op) {
    TickCommand cmd;
    cmd.kind = CommandKind::Timeline;
    cmd.timeline = op;
    return cmd;
}

// The commands that change physics; the log replays them.
bool PhysicsWorker::Recorded(const TickCommand& cmd) {
    switch (cmd.kind) {
        case CommandKind::EditApply:
        case CommandKind::Teleport:
        case CommandKind::Perturb:
            return true;
        case CommandKind::ParamSet:
            return cmd.param == ParamKind::Field || cmd.param == ParamKind::HoldGains ||
                   cmd.param == ParamKind::ControllerCommand || cmd.param == ParamKind::Registry ||
                   cmd.param == ParamKind::Iterations;
        default:
            return false;
    }
}

// A new history whose root starts at the playhead.
void PhysicsWorker::ResetHistory() {
    branches_.clear();
    HistoryBranch& root = branches_[0u];
    root.fork_tick = root.head = tick_;
    root.exact = bind_.controller == nullptr;
    branch_ = 0u;
    next_branch_ = 1u;
    cursor_ = 0u;
    ghost_ = kNoBranch;
    key_due_ = false;
    CaptureKey(false);
}

void PhysicsWorker::CaptureKey(bool pinned) {
    auto key = std::make_shared<Keyframe>();
    CheckResult(nuka_world_checkpoint_capture(bind_.world, &key->state), "nuka_world_checkpoint_capture");
    CheckResult(nuka_checkpoint_byte_size(key->state, &key->bytes), "nuka_checkpoint_byte_size");
    key->tick = tick_;
    key->sim_time = sim_time_;
    key->log_pos = cursor_;
    key->perturbs = perturbs_;
    key->pinned = pinned;
    std::vector<KeyRef>& keys = Current().keys;
    // A later keyframe of the same tick holds everything the earlier one did.
    if (!keys.empty() && keys.back()->tick == tick_) {
        key->pinned = key->pinned || keys.back()->pinned;
        keys.back() = std::move(key);
    } else {
        keys.push_back(std::move(key));
    }
    ++history_revision_;
    Evict();
}

// An input recorded behind the head starts a branch there.
void PhysicsWorker::Record(const TickCommand& cmd) {
    if (Replaying()) Fork(std::string());
    HistoryBranch& b = Current();
    b.log.push_back(LoggedInput{tick_, cmd});
    b.log.back().cmd.seq = 0u;  // a replay answers no one
    cursor_ = b.log.size();
}

// A script can write what the log cannot replay. When it changed the world the next keyframe
// is pinned, and behind the head the change starts a branch.
bool PhysicsWorker::ApplyScript(const TickCommand& cmd) {
    const uint64_t epoch = epoch_;
    Keyframe before;
    CheckResult(nuka_world_checkpoint_capture(bind_.world, &before.state), "nuka_world_checkpoint_capture");
    const bool publish = Apply(cmd);
    if (epoch_ != epoch || Matches(before.state)) return publish;
    key_due_ = true;
    if (Replaying()) Fork(std::string());
    return publish;
}

// The child shares the log and keyframes up to the playhead and gets a keyframe there.
void PhysicsWorker::Fork(std::string label) {
    const HistoryBranch& parent = Current();
    HistoryBranch child;
    child.id = next_branch_++;
    child.parent = parent.id;
    child.fork_tick = child.head = tick_;
    child.label = std::move(label);
    child.exact = parent.exact;
    child.used = ++switches_;
    child.log.assign(parent.log.begin(), parent.log.begin() + static_cast<std::ptrdiff_t>(cursor_));
    for (const KeyRef& key : parent.keys)
        if (key->tick <= tick_) child.keys.push_back(key);
    ghost_ = parent.id;
    const uint32_t id = child.id;
    branches_.emplace(id, std::move(child));
    branch_ = id;
    KeyHead();
}

// The head gets a keyframe with every input so far unless its newest one has them.
void PhysicsWorker::KeyHead() {
    if (Replaying()) return;
    const Keyframe& last = *Current().keys.back();
    if (key_due_ || last.tick != tick_ || last.log_pos != cursor_) CaptureKey(key_due_);
    key_due_ = false;
}

// Restores the newest keyframe at or before `target` and replays the log up to it.
void PhysicsWorker::SeekTo(uint64_t target) {
    const HistoryBranch& b = Current();
    target = std::clamp(target, b.keys.front()->tick, b.head);
    const auto at = std::upper_bound(b.keys.begin(), b.keys.end(), target,
                                     [](uint64_t t, const KeyRef& k) { return t < k->tick; });
    Restore(**std::prev(at));
    bool healthy = true;
    while (tick_ < target) healthy = StepOnce() && healthy;
    healthy_ = healthy;
}

void PhysicsWorker::Restore(const Keyframe& key) {
    CheckResult(nuka_world_checkpoint_restore(bind_.world, key.state), "nuka_world_checkpoint_restore");
    perturbs_ = key.perturbs;
    SetTick(key.tick);
    sim_time_ = key.sim_time;
    cursor_ = key.log_pos;
    last_mirror_ = WorkerClock::time_point{};
    ApplyLogged();
}

// The current branch's inputs up to the playhead, stopping at log position `limit`.
void PhysicsWorker::ApplyLogged(size_t limit) {
    const std::vector<LoggedInput>& log = Current().log;
    const size_t end = std::min(limit, log.size());
    while (cursor_ < end && log[cursor_].tick <= tick_) Apply(log[cursor_++].cmd);
}

// Whether the world equals a checkpoint of it.
bool PhysicsWorker::Matches(nuka_checkpoint_handle state) {
    nuka_checkpoint_handle now = nullptr;
    CheckResult(nuka_world_checkpoint_capture(bind_.world, &now), "nuka_world_checkpoint_capture");
    uint32_t equal = 0u;
    const nuka_result_t result = nuka_checkpoint_equal(now, state, &equal);
    nuka_checkpoint_destroy(now);
    CheckResult(result, "nuka_checkpoint_equal");
    return equal != 0u;
}

void PhysicsWorker::ApplyTimeline(const TickCommand& cmd) {
    if ((cmd.timeline == TimelineOp::Branch || cmd.timeline == TimelineOp::Rename ||
         (cmd.timeline == TimelineOp::Ghost && cmd.branch != kNoBranch)) &&
        branches_.count(cmd.branch) == 0u) {
        Report(cmd.seq, "[nuka_editor] no timeline branch " + std::to_string(cmd.branch) + "\n", true);
        return;
    }
    switch (cmd.timeline) {
        case TimelineOp::Seek: {
            const uint64_t back = cmd.delta < 0 ? std::min(tick_, static_cast<uint64_t>(-cmd.delta)) : 0u;
            const uint64_t target =
                !cmd.relative ? cmd.tick : cmd.delta < 0 ? tick_ - back : tick_ + static_cast<uint64_t>(cmd.delta);
            playing_ = false;
            step_budget_ = 0u;
            KeyHead();
            SeekTo(target);
            break;
        }
        case TimelineOp::Fork:
            KeyHead();
            Fork(cmd.label);
            break;
        case TimelineOp::Branch: {
            playing_ = false;
            step_budget_ = 0u;
            KeyHead();
            if (ghost_ == cmd.branch) ghost_ = branch_;
            branch_ = cmd.branch;
            Current().used = ++switches_;
            SeekTo(Current().head);
            break;
        }
        case TimelineOp::Rename:
            branches_.at(cmd.branch).label = cmd.label;
            break;
        case TimelineOp::Ghost:
            ghost_on_ = cmd.values[0] != 0.0f;
            if (cmd.branch != kNoBranch) ghost_ = cmd.branch;
            break;
        case TimelineOp::Configure:
            key_interval_ = static_cast<uint32_t>(cmd.values[0]);
            key_budget_ = static_cast<uint64_t>(static_cast<double>(cmd.values[1]) * 1048576.0);
            Evict();
            break;
        case TimelineOp::Clear:
            ResetHistory();
            break;
    }
    ++history_revision_;
}

void PhysicsWorker::Evict() {
    Recount();
    while (key_bytes_ > key_budget_ && EvictOne()) Recount();
}

// Thins first: the keyframe whose neighbours end up closest goes, the oldest on a tie, never a
// branch's first, newest, fork or pinned one. Then the least recently used leaf branch goes,
// then the current branch's oldest keyframe, which moves the start of its history.
bool PhysicsWorker::EvictOne() {
    const auto drop = [this](const Keyframe* key) {
        for (auto& [id, b] : branches_)
            b.keys.erase(std::remove_if(b.keys.begin(), b.keys.end(), [&](const KeyRef& k) { return k.get() == key; }),
                         b.keys.end());
    };
    const uint64_t limit = static_cast<uint64_t>(key_interval_) * kMaxGapIntervals;
    const Keyframe* thin = nullptr;
    uint64_t thin_gap = 0u;
    for (const auto& [id, b] : branches_)
        for (size_t i = 1u; i + 1u < b.keys.size(); ++i) {
            const Keyframe& k = *b.keys[i];
            const uint64_t gap = b.keys[i + 1u]->tick - b.keys[i - 1u]->tick;
            if (k.pinned || k.tick == b.fork_tick || gap > limit) continue;
            if (thin == nullptr || gap < thin_gap || (gap == thin_gap && k.tick < thin->tick)) {
                thin = &k;
                thin_gap = gap;
            }
        }
    if (thin != nullptr) {
        drop(thin);
        return true;
    }
    uint32_t lru = kNoBranch;
    for (const auto& [id, b] : branches_) {
        const bool parent = std::any_of(branches_.begin(), branches_.end(),
                                        [id = id](const auto& c) { return c.second.parent == id; });
        if (id == branch_ || parent) continue;
        if (lru == kNoBranch || b.used < branches_.at(lru).used) lru = id;
    }
    if (lru != kNoBranch) {
        if (ghost_ == lru) ghost_ = kNoBranch;
        branches_.erase(lru);
        return true;
    }
    const Keyframe* oldest = Current().keys.front().get();
    for (const auto& [id, b] : branches_)
        if (b.keys.front().get() == oldest && b.keys.size() < 2u) return false;
    drop(oldest);
    return true;
}

void PhysicsWorker::Recount() {
    std::unordered_set<const Keyframe*> seen;
    key_bytes_ = 0u;
    for (const auto& [id, b] : branches_)
        for (const KeyRef& k : b.keys)
            if (seen.insert(k.get()).second) key_bytes_ += k->bytes;
    key_count_ = static_cast<uint32_t>(seen.size());
}

// At the head, each downloaded pose mirror becomes a sample of the branch's track.
void PhysicsWorker::SampleTrack() {
    if (!mirrored_ || Replaying()) return;
    std::deque<PoseSample>& track = Current().track;
    while (!track.empty() && track.back().tick >= tick_) track.pop_back();
    track.push_back(PoseSample{tick_, bind_.env, mirror_links_, mirror_bodies_});
    while (track.size() > kTrackSamples) track.pop_front();
}

// The ghost branch's newest sample at or before the playhead, once the two branches diverged.
const PoseSample* PhysicsWorker::GhostSample() {
    const auto g = branches_.find(ghost_);
    if (!ghost_on_ || g == branches_.end() || ghost_ == branch_ || tick_ > g->second.head ||
        tick_ <= Divergence(branch_, ghost_))
        return nullptr;
    const std::deque<PoseSample>& track = g->second.track;
    const auto at = std::upper_bound(track.begin(), track.end(), tick_,
                                     [](uint64_t t, const PoseSample& s) { return t < s.tick; });
    if (at == track.begin()) return nullptr;
    const PoseSample& s = *std::prev(at);
    return s.env == bind_.env ? &s : nullptr;
}

// Two branches share their history up to the earlier of the fork ticks just below their common
// ancestor.
uint64_t PhysicsWorker::Divergence(uint32_t a, uint32_t b) const {
    std::vector<std::pair<uint32_t, uint64_t>> up;  // a's ancestors, each with the fork below it
    uint64_t below = ~uint64_t(0);
    for (uint32_t id = a; id != kNoBranch; id = branches_.at(id).parent) {
        up.emplace_back(id, below);
        below = branches_.at(id).fork_tick;
    }
    below = ~uint64_t(0);
    for (uint32_t id = b; id != kNoBranch; id = branches_.at(id).parent) {
        for (const auto& [x, fork] : up)
            if (x == id) return std::min(fork, below);
        below = branches_.at(id).fork_tick;
    }
    return 0u;
}

void PhysicsWorker::PublishTimeline(Snapshot& s) {
    SampleTrack();
    const HistoryBranch& b = Current();
    s.branch = branch_;
    s.head = b.head;
    s.ghost_links.clear();
    s.ghost_bodies.clear();
    if (const PoseSample* g = GhostSample()) {
        s.ghost_links = g->links;
        s.ghost_bodies = g->bodies;
    }
    std::lock_guard<std::mutex> lock(timeline_mu_);
    TimelineInfo& t = timeline_;
    t.branch = branch_;
    t.start = b.keys.front()->tick;
    t.head = b.head;
    t.tick = tick_;
    t.ghost = ghost_;
    t.ghost_on = ghost_on_;
    t.interval = key_interval_;
    t.bytes = key_bytes_;
    t.budget = key_budget_;
    t.keyframes = key_count_;
    if (t.revision == history_revision_) return;
    t.revision = history_revision_;
    t.branches.clear();
    for (const auto& [id, br] : branches_)
        t.branches.push_back(TimelineBranchInfo{id, br.parent, br.fork_tick, br.head, br.label, br.exact});
    t.keys.clear();
    t.pinned.clear();
    for (const KeyRef& k : b.keys) (k->pinned ? t.pinned : t.keys).push_back(k->tick);
}

void PhysicsWorker::ReadTimeline(TimelineInfo* info) {
    std::lock_guard<std::mutex> lock(timeline_mu_);
    *info = timeline_;
}

}  // namespace nuka::runtime::app::viewer
