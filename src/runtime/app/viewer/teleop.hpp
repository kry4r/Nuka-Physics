#pragma once
// Teleop devices and their mapping. The built-in devices all report SDL's gamepad layout, so
// one mapping drives the keyboard, a gamepad or the virtual device; SDL gamepads live here too.

#include "runtime/app/viewer/teleop_view.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// SDL's standard gamepad axes and buttons, by its names and in its order.
inline const std::vector<std::string> kPadAxes = {"leftx", "lefty", "rightx", "righty", "lefttrigger", "righttrigger"};
inline const std::vector<std::string> kPadButtons = {"a",          "b",          "x",           "y",
                                                     "back",       "guide",      "start",       "leftstick",
                                                     "rightstick", "leftshoulder", "rightshoulder", "dpup",
                                                     "dpdown",     "dpleft",     "dpright"};
enum PadAxis : uint32_t { kLeftX, kLeftY, kRightX, kRightY };
enum PadButton : uint32_t { kLeftShoulder = 9u, kRightShoulder = 10u };

inline const std::vector<std::string> kTeleopModes = {"command", "joint", "ik"};

// What a built-in mode reads: vx, vy [m/s] and wyaw [rad/s]; the joint's rate and presses that
// pick the next or previous joint; the IK goal's world velocity [m/s].
inline const std::vector<std::string>& ModeQuantities(const std::string& mode) {
    static const std::map<std::string, std::vector<std::string>> quantities = {
        {"command", {"vx", "vy", "wyaw"}}, {"joint", {"rate", "next", "prev"}}, {"ik", {"x", "y", "z"}}};
    static const std::vector<std::string> none;
    const auto it = quantities.find(mode);
    return it != quantities.end() ? it->second : none;
}

// Stick forward and left are +x and +y, so SDL's down- and right-positive axes are inverted.
inline std::map<std::string, std::vector<TeleopBinding>> DefaultTeleopMaps() {
    return {{"command", {{"lefty", "vx", 1.0f, true}, {"leftx", "vy", 1.0f, true}, {"rightx", "wyaw", 1.0f, true}}},
            {"joint",
             {{"lefty", "rate", 1.0f, true}, {"rightshoulder", "next", 1.0f, false}, {"leftshoulder", "prev", 1.0f, false}}},
            {"ik", {{"lefty", "x", 0.2f, true}, {"leftx", "y", 0.2f, true}, {"righty", "z", 0.2f, true}}}};
}

// An input's value: an axis past the deadzone rescaled to [-1, 1], a button's 0 or 1, and 0 for
// a name the device lacks.
inline float InputValue(const TeleopDeviceInfo& device, const std::vector<float>& axes,
                        const std::vector<uint8_t>& buttons, const std::string& input, float deadzone) {
    const auto axis = std::find(device.axes.begin(), device.axes.end(), input);
    if (axis != device.axes.end()) {
        const float v = axes[static_cast<size_t>(axis - device.axes.begin())];
        return std::copysign(std::min(std::max(std::fabs(v) - deadzone, 0.0f) / (1.0f - deadzone), 1.0f), v);
    }
    const auto button = std::find(device.buttons.begin(), device.buttons.end(), input);
    return button != device.buttons.end() && buttons[static_cast<size_t>(button - device.buttons.begin())] != 0u ? 1.0f
                                                                                                                 : 0.0f;
}

// Each of `names`, and each quantity a binding reaches, as the sum of its scaled inputs.
inline std::map<std::string, float> TeleopQuantities(const TeleopDeviceInfo& device, const std::vector<float>& axes,
                                                     const std::vector<uint8_t>& buttons,
                                                     const std::vector<TeleopBinding>& bindings,
                                                     const std::vector<std::string>& names, float deadzone) {
    std::map<std::string, float> out;
    for (const std::string& name : names) out[name] = 0.0f;
    for (const TeleopBinding& b : bindings)
        out[b.quantity] += (b.invert ? -b.gain : b.gain) * InputValue(device, axes, buttons, b.input, deadzone);
    return out;
}

// SDL gamepads, virtual ones included, as devices "gamepad:<SDL id>" in the pad layout. It starts
// SDL's gamepad subsystem itself, so pads work without a window and while it is in the background.
class TeleopPads {
public:
    TeleopPads();
    ~TeleopPads();
    TeleopPads(const TeleopPads&) = delete;
    TeleopPads& operator=(const TeleopPads&) = delete;

    // Opens what was plugged in, closes what was pulled, and lists the pads into `out`.
    void Update(std::vector<TeleopDeviceInfo>* out);
    // The pad's axes and buttons; false when it is gone.
    bool Read(const std::string& id, std::vector<float>* axes, std::vector<uint8_t>* buttons) const;
    // A virtual gamepad on SDL's joystick path; its device id, or empty with *error.
    std::string Attach(const std::string& name, std::string* error);
    // Sets a virtual pad's axes and buttons (nonzero is down) by name; false with *error.
    bool Set(const std::string& id, const std::map<std::string, float>& axes,
             const std::map<std::string, float>& buttons, std::string* error);
    bool Detach(const std::string& id);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nuka::runtime::app::viewer
