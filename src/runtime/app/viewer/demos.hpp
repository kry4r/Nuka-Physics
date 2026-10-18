#pragma once
// Demonstrations: what the physics thread records of an environment each step, the episodes
// cut from it, and the exchange through which nuka.editor.demos writes and reads them as HDF5.

#include "nuka/nuka.h"
#include "runtime/app/viewer/editor_world.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// One environment after a step, and the inputs in force for it.
struct DemoSample {
    uint64_t tick = 0u;
    double time = 0.0;
    uint32_t env = 0u;
    std::vector<float> q, qd, target;  // per link; the target is the one the step ran with
    std::vector<float> root;           // per root link: x y z qw qx qy qz, then body-frame wx wy wz vx vy vz
    std::array<float, 3> command{};    // the controller's: vx, vy, wyaw
    std::vector<float> axes;           // the teleop device's
};

inline constexpr size_t kRootValues = 13u;

// Reads environment `env` after a step; `roots` are its root links' per-environment indices.
void DownloadDemoSample(nuka_world_handle world, const WorldLayout& layout, uint32_t env,
                        const std::vector<uint32_t>& roots, DemoSample* out);

// One environment's steps from a start to a stop, with the set's joints as columns.
struct DemoEpisode {
    uint32_t env = 0u;
    uint64_t start = 0u;  // the tick its first step started from
    uint32_t roots = 0u;
    std::vector<double> time;
    std::vector<float> q, qd, action;  // per step and joint; the action is the drive target
    std::vector<float> root_pose;      // per step and root: x y z qw qx qy qz
    std::vector<float> root_vel;       // per step and root: body-frame wx wy wz vx vy vz
    std::vector<float> command;        // per step: vx vy wyaw
    std::vector<float> axes;           // per step and axis
    size_t Steps() const { return time.size(); }
};

// Episodes and what they share: the scene, the step, the joints with their link slots, the axes.
struct DemoSet {
    std::string scene;
    float dt = 0.0f;
    std::vector<std::string> joints;
    std::vector<uint32_t> links;
    std::vector<std::string> axes;
    std::vector<DemoEpisode> episodes;
};

// Appends the episodes the samples hold, each one environment's consecutive steps.
void CutEpisodes(const std::vector<DemoSample>& samples, DemoSet* set);

// The set as a JSON manifest beside its raw arrays in `data`, as nuka.editor.demos reads it,
// and back; reading throws when the files disagree.
void WriteDemoExchange(const DemoSet& set, const std::string& manifest, const std::string& data);
DemoSet ReadDemoExchange(const std::string& manifest, const std::string& data);

}  // namespace nuka::runtime::app::viewer
