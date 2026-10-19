#pragma once
// A training run the viewport watches: the robots' velocity arrows and the run's progress.

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// Where one shown robot's arrows start, above its root, and the planar velocities they draw.
struct TrainArrow {
    float at[3] = {0.0f, 0.0f, 0.0f};
    float command[3] = {0.0f, 0.0f, 0.0f};   // commanded world vx, vy and yaw rate
    float velocity[3] = {0.0f, 0.0f, 0.0f};  // the root's world vx, vy and yaw rate
};

struct TrainView {
    bool observing = false;
    bool follow = false;  // the camera keeps the current environment's robot centred
    std::string run;
    uint32_t envs = 0u;      // shown
    uint32_t run_envs = 0u;  // trained
    uint64_t frames = 0u;    // taken since watching began
    uint64_t steps = 0u;     // the run's environment steps at the newest
    int64_t epoch = 0;
    int64_t max_epochs = 0;
    bool has_reward = false;
    double reward = 0.0;
    double fps = 0.0;  // the run's environment steps per second
    std::vector<TrainArrow> arrows;
    uint32_t current = 0u;  // the arrows' entry of the current environment
};

}  // namespace nuka::runtime::app::viewer
