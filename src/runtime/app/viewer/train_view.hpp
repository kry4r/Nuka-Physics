#pragma once
// A training run the viewport watches, with the robots' velocity arrows and the run's progress, and
// the Training panel's task configuration and runs.

#include "scene/format/json.hpp"

#include <cstdint>
#include <optional>
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

// One run of the document as the Training panel lists it.
struct TrainRunRow {
    std::string name;
    std::string state;  // starting, running, finished, stopped, failed or lost
    bool alive = false;
    bool serving = false;  // it serves an observation
    int64_t epoch = 0;
    int64_t max_epochs = 0;
    int64_t envs = 0;
    int64_t frames = 0;
    int64_t checkpoints = 0;
    std::optional<double> reward, best, fps;
    double started = 0.0;  // seconds since 1970
    double ended = 0.0;    // 0 until it ends
    std::string error;
};

// A run's epoch rows by column for the Dashboard's curves; a value a row lacks is NaN.
struct TrainSeries {
    std::string run;
    size_t next = 0u;  // the rows read so far
    std::vector<double> epoch, time, reward, length, fps, actor_loss, critic_loss, entropy, kl, lr;
};

// One save of the selected run.
struct TrainSave {
    std::string kind;  // best, final or epoch
    std::string policy;
    int64_t epoch = 0;
    int64_t frames = 0;
    std::optional<double> reward;
    double time = 0.0;
};

enum class TrainPage : uint8_t { Task, Runs, Dashboard, Checkpoints };

// The Training panel: the document's task configuration, its runs and the selected run's progress,
// refreshed by the host while the panel shows.
struct TrainPanel {
    bool shown = false;  // UI-set: the panel drew this frame
    TrainPage page = TrainPage::Task;
    scene::json::Value tasks;       // train.tasks: each task's schema
    scene::json::Value config;      // train.config: task, config, changed and errors
    uint64_t config_version = 0u;   // bumped whenever `config` is taken again
    std::string config_error;       // why there is no configuration
    scene::json::Value python;      // train.python: python, ready and error
    std::vector<TrainRunRow> runs;  // newest first
    std::string training;           // the run this editor started, while it trains
    std::string selected;           // the run the pages show; the host keeps it a listed run
    std::vector<std::string> compare;  // runs the Dashboard draws over the selected one
    std::string log;                   // the selected run's last lines
    std::vector<TrainSeries> series;   // the selected run's, then the compared ones'
    std::vector<TrainSave> saves;      // the selected run's
};

}  // namespace nuka::runtime::app::viewer
