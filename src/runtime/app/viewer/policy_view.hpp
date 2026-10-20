#pragma once
// The loaded locomotion policy as the Policy panel shows it.

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

struct PolicyView {
    bool loaded = false;
    bool on = true;  // it steps; off, the drive targets hold where it left them
    bool gpu = false;
    std::string path;
    std::string contract;  // the contract it runs under, empty for the Go2 one
    std::string error;     // why the last attach failed
    bool arrows = true;    // the command and measured velocity draw over its robots
    uint32_t robots = 0u;
    uint32_t obs_dim = 0u;
    uint32_t act_dim = 0u;
    uint32_t decimation = 0u;  // steps per decision
    float rate_hz = 0.0f;      // decisions per simulated second
    float action_scale = 0.0f;
    std::vector<uint32_t> links;  // each action's link slot, robot after robot
    std::vector<float> defaults;  // each action's default joint position
    float command[3] = {0.0f, 0.0f, 0.0f};  // the [vx, vy, wyaw] last sent to it
    float norm = 0.0f;                      // its last actions' norm, the mean over the robots
    std::vector<float> norm_time, norm_value;  // the norm over simulated time while it runs
    char path_edit[512] = {0};
    bool host = false;  // the next load runs the host MLP instead of the GPU path
};

}  // namespace nuka::runtime::app::viewer
