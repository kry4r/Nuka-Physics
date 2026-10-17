#pragma once
// The history as the physics thread reports it and the Timeline panel shows it.

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

inline constexpr uint32_t kNoBranch = ~0u;

struct TimelineBranchInfo {
    uint32_t id = 0u;
    uint32_t parent = kNoBranch;
    uint64_t fork_tick = 0u;
    uint64_t head = 0u;   // the newest recorded tick
    std::string label;    // empty for the default name
    bool exact = true;    // no controller ran and every replayed keyframe matched
};

struct TimelineInfo {
    uint64_t revision = 0u;  // bumped whenever branches or keyframes change
    uint32_t branch = 0u;    // the current branch
    uint64_t start = 0u;     // the earliest tick it can seek to
    uint64_t head = 0u;
    uint64_t tick = 0u;      // the playhead
    std::vector<TimelineBranchInfo> branches;
    std::vector<uint64_t> keys;    // the current branch's keyframe ticks
    std::vector<uint64_t> pinned;  // the ones holding writes the input log cannot replay
    uint32_t ghost = kNoBranch;    // the branch the ghost shows
    bool ghost_on = true;
    uint32_t interval = 16u;       // ticks between keyframes
    uint64_t bytes = 0u;           // keyframe memory over all branches
    uint64_t budget = 0u;
    uint32_t keyframes = 0u;
};

}  // namespace nuka::runtime::app::viewer
