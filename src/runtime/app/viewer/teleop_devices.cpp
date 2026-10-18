#include "runtime/app/viewer/teleop.hpp"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace nuka::runtime::app::viewer {

namespace {

constexpr float kAxisScale = 32767.0f;

std::string PadId(SDL_JoystickID id) { return "gamepad:" + std::to_string(id); }

// A virtual trigger spans all of Sint16 and rests at its minimum, though the pad reads it in [0, 1].
Sint16 VirtualAxis(int axis, float v) {
    if (axis >= SDL_GAMEPAD_AXIS_LEFT_TRIGGER)
        return static_cast<Sint16>(std::lround(SDL_JOYSTICK_AXIS_MIN + std::clamp(v, 0.0f, 1.0f) * 65535.0f));
    return static_cast<Sint16>(std::lround(std::clamp(v, -1.0f, 1.0f) * kAxisScale));
}

}  // namespace

struct TeleopPads::Impl {
    struct Pad {
        SDL_JoystickID id = 0;
        SDL_Gamepad* pad = nullptr;
        std::string name;
    };
    bool ok = false;
    std::vector<Pad> pads;
    std::vector<std::pair<SDL_JoystickID, SDL_Joystick*>> virtuals;  // the handles that set their inputs

    const Pad* Find(const std::string& id) const {
        for (const Pad& p : pads)
            if (PadId(p.id) == id) return &p;
        return nullptr;
    }
    bool Virtual(SDL_JoystickID id) const {
        return std::any_of(virtuals.begin(), virtuals.end(), [&](const auto& v) { return v.first == id; });
    }
};

TeleopPads::TeleopPads() : impl_(std::make_unique<Impl>()) {
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    impl_->ok = SDL_InitSubSystem(SDL_INIT_GAMEPAD);
    // Update polls the pads, so no events pile up where no window drains them.
    SDL_SetJoystickEventsEnabled(false);
    SDL_SetGamepadEventsEnabled(false);
}

TeleopPads::~TeleopPads() {
    Impl& m = *impl_;
    if (!m.ok) return;
    for (const Impl::Pad& p : m.pads) SDL_CloseGamepad(p.pad);
    for (const auto& [id, joystick] : m.virtuals) {
        SDL_CloseJoystick(joystick);
        SDL_DetachVirtualJoystick(id);
    }
    SDL_QuitSubSystem(SDL_INIT_GAMEPAD);
}

void TeleopPads::Update(std::vector<TeleopDeviceInfo>* out) {
    Impl& m = *impl_;
    if (!m.ok) return;
    SDL_UpdateGamepads();
    int count = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&count);
    const std::vector<SDL_JoystickID> now(ids, ids + count);
    SDL_free(ids);
    std::erase_if(m.pads, [&](const Impl::Pad& p) {
        if (std::find(now.begin(), now.end(), p.id) != now.end()) return false;
        SDL_CloseGamepad(p.pad);
        return true;
    });
    for (const SDL_JoystickID id : now) {
        if (std::any_of(m.pads.begin(), m.pads.end(), [&](const Impl::Pad& p) { return p.id == id; })) continue;
        if (SDL_Gamepad* pad = SDL_OpenGamepad(id)) {
            const char* name = SDL_GetGamepadName(pad);
            m.pads.push_back({id, pad, name != nullptr ? name : "Gamepad"});
        }
    }
    for (const Impl::Pad& p : m.pads)
        out->push_back({PadId(p.id), p.name, m.Virtual(p.id) ? "virtual gamepad" : "gamepad", "", kPadAxes, kPadButtons});
}

bool TeleopPads::Read(const std::string& id, std::vector<float>* axes, std::vector<uint8_t>* buttons) const {
    const Impl::Pad* p = impl_->Find(id);
    if (p == nullptr) return false;
    axes->resize(kPadAxes.size());
    buttons->resize(kPadButtons.size());
    for (size_t i = 0; i < kPadAxes.size(); ++i)
        (*axes)[i] = std::clamp(static_cast<float>(SDL_GetGamepadAxis(p->pad, static_cast<SDL_GamepadAxis>(i))) / kAxisScale,
                                -1.0f, 1.0f);
    for (size_t i = 0; i < kPadButtons.size(); ++i)
        (*buttons)[i] = static_cast<uint8_t>(SDL_GetGamepadButton(p->pad, static_cast<SDL_GamepadButton>(i)));
    return true;
}

// Every axis and button of the pad layout, in order, so SDL maps joystick input i to pad input i.
std::string TeleopPads::Attach(const std::string& name, std::string* error) {
    Impl& m = *impl_;
    if (!m.ok) {
        *error = "SDL gamepads are unavailable";
        return std::string();
    }
    SDL_VirtualJoystickDesc desc;
    SDL_INIT_INTERFACE(&desc);
    desc.type = static_cast<Uint16>(SDL_JOYSTICK_TYPE_GAMEPAD);
    desc.naxes = static_cast<Uint16>(kPadAxes.size());
    desc.nbuttons = static_cast<Uint16>(kPadButtons.size());
    desc.axis_mask = (1u << kPadAxes.size()) - 1u;
    desc.button_mask = (1u << kPadButtons.size()) - 1u;
    desc.name = name.c_str();
    const SDL_JoystickID id = SDL_AttachVirtualJoystick(&desc);
    if (id == 0) {
        *error = SDL_GetError();
        return std::string();
    }
    SDL_Joystick* joystick = SDL_OpenJoystick(id);
    for (const SDL_GamepadAxis axis : {SDL_GAMEPAD_AXIS_LEFT_TRIGGER, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER})
        SDL_SetJoystickVirtualAxis(joystick, axis, VirtualAxis(axis, 0.0f));
    m.virtuals.emplace_back(id, joystick);
    return PadId(id);
}

bool TeleopPads::Set(const std::string& id, const std::map<std::string, float>& axes,
                     const std::map<std::string, float>& buttons, std::string* error) {
    const auto it = std::find_if(impl_->virtuals.begin(), impl_->virtuals.end(),
                                 [&](const auto& v) { return PadId(v.first) == id; });
    if (it == impl_->virtuals.end()) {
        *error = "'" + id + "' is not a virtual gamepad";
        return false;
    }
    const auto index = [](const std::vector<std::string>& names, const std::string& name) {
        return static_cast<int>(std::find(names.begin(), names.end(), name) - names.begin());
    };
    for (const auto& [axis, v] : axes)
        if (index(kPadAxes, axis) == static_cast<int>(kPadAxes.size())) *error = "no axis '" + axis + "'";
    for (const auto& [button, v] : buttons)
        if (index(kPadButtons, button) == static_cast<int>(kPadButtons.size())) *error = "no button '" + button + "'";
    if (!error->empty()) return false;
    for (const auto& [axis, v] : axes)
        SDL_SetJoystickVirtualAxis(it->second, index(kPadAxes, axis), VirtualAxis(index(kPadAxes, axis), v));
    for (const auto& [button, v] : buttons) SDL_SetJoystickVirtualButton(it->second, index(kPadButtons, button), v != 0.0f);
    return true;
}

bool TeleopPads::Detach(const std::string& id) {
    Impl& m = *impl_;
    const auto it = std::find_if(m.virtuals.begin(), m.virtuals.end(), [&](const auto& v) { return PadId(v.first) == id; });
    if (it == m.virtuals.end()) return false;
    SDL_CloseJoystick(it->second);
    SDL_DetachVirtualJoystick(it->first);
    m.virtuals.erase(it);
    return true;
}

}  // namespace nuka::runtime::app::viewer
