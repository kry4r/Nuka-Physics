#pragma once
// What the Teleop section shows and edits: the devices, the active one's reading, each mode's
// mapping from device inputs to retarget quantities, and what the last frame produced.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// A device as teleop lists it: its axes in [-1, 1] and its buttons, by name.
struct TeleopDeviceInfo {
    std::string id;
    std::string label;
    std::string kind;   // keyboard, gamepad, virtual, or what an extension calls it
    std::string owner;  // the extension that registered it; empty for the built-ins
    std::vector<std::string> axes;
    std::vector<std::string> buttons;
};

// One device input scaled into a retarget quantity; a button reads 0 or 1.
struct TeleopBinding {
    std::string input;
    std::string quantity;
    float gain = 1.0f;
    bool invert = false;
};

// A mode an extension registered: device and robot state in, joint targets out.
struct TeleopRetarget {
    std::string id;
    std::string label;
    std::string owner;
};

struct TeleopView {
    bool on = false;
    std::string device = "keyboard";
    std::string mode = "command";  // command, joint, ik, or a registered retarget's id
    std::vector<TeleopDeviceInfo> devices;
    std::vector<TeleopRetarget> retargets;
    std::map<std::string, std::vector<TeleopBinding>> maps;  // per mode
    float deadzone = 0.1f;
    // The active device's reading this frame, in its info's order.
    bool connected = false;
    std::vector<float> axes;
    std::vector<uint8_t> buttons;
    std::map<std::string, float> quantities;
    float command[3] = {0.0f, 0.0f, 0.0f};       // vx, vy, wyaw
    float command_base[3] = {0.0f, 0.0f, 0.0f};  // the command mode adds the device on top
    bool controller = false;                     // a loaded policy takes the command
    int joint = 0;     // the joint mode's joint, an index into the robot's joints
    std::string link;  // the ik mode's link path; empty follows the selection
    bool goal_set = false;
    float goal[3] = {0.0f, 0.0f, 0.0f};  // the ik mode's goal for the link origin
    bool hold = false;                   // PD hold: drive targets act without a policy
    float kp = 20.0f;
    float kd = 0.5f;
    std::string error;
};

}  // namespace nuka::runtime::app::viewer
