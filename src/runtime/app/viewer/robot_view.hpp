#pragma once
// The robot workflow's joint state as the physics thread samples it and the Joints panel shows it.

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// The shown environment's joints at one step, one value per link.
struct JointSample {
    uint64_t tick = 0u;
    double time = 0.0;
    std::vector<float> q;
    std::vector<float> target;
    std::vector<float> effort;
};

// What a mirror publish reads of the shown environment's joints, one value per link.
struct JointFrame {
    std::vector<float> q;
    std::vector<float> qd;
    std::vector<float> target;
    std::vector<float> effort;
    std::vector<float> saturated;  // 1 where the actuator clamped its effort
};

// A moving joint: its link slot, its authored record and the drive on it.
struct RobotJoint {
    uint32_t link = 0u;
    std::string name;
    std::string path;  // the child link's node
    bool prismatic = false;
    bool limited = false;
    float lower = 0.0f;
    float upper = 0.0f;
    std::string drive;         // the actuator type: position, motor, velocity or force; empty without one
    float kp = 0.0f;           // a position drive's gain
    float force_limit = 0.0f;  // 0 is unlimited
};

struct RobotView {
    std::vector<RobotJoint> joints;  // in link order
    std::vector<std::string> roots;  // the articulation roots' nodes
    bool floating = false;           // some root is a free base
    JointFrame live;
    bool watch = false;  // the physics thread samples every step
    uint64_t session = 0u;
    uint64_t first = 0u;  // the number of history.front() in the worker's count
    std::deque<JointSample> history;
    int focus = -1;  // the selected link's joint, which the plot follows, by index into `joints`
    int plotted = -1;  // the joint the plot showed last frame; plotting a new one starts the watch
    std::string filter;
    float drive_kp = 100.0f;  // the gain Position drive converts with
};

}  // namespace nuka::runtime::app::viewer
