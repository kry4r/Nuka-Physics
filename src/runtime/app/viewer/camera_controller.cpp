// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer::CameraController -- implementation (M8.5 T3).
//
// Pure host C++/math: orbit/pan/zoom -> RasterOptions camera override. NO Vulkan,
// NO ImGui, NO CUDA. See camera_controller.hpp for the control mapping.
// ---------------------------------------------------------------------------

#include "runtime/app/viewer/camera_controller.hpp"

#include <algorithm>
#include <cmath>

namespace nuka::runtime::app::viewer {

namespace {

// Standard evdev keycodes under X / Xvfb. `key` arrives as a raw keycode (the xcb
// backend forwards kp->detail, not a keysym) so we match the physical positions.
// Shift is tracked so Shift+LMB acts as pan.
//
// CAVEAT (hardcoded keycodes / evdev assumption): 50/62 are the LEFT/RIGHT SHIFT
// keycodes under the common evdev X keymap. Because xcb forwards RAW keycodes
// (not resolved keysyms), this matching is keymap-dependent and would break on a
// non-evdev layout. Resolving keysyms properly (e.g. via xcb-keysyms / an
// xkbcommon map) instead of comparing raw keycodes is a TODO.
constexpr uint32_t kKeyShiftL = 50u;
constexpr uint32_t kKeyShiftR = 62u;

}  // namespace

bool CameraController::HandleEvent(const window::WindowEvent& ev, bool allow_drag,
                                   bool allow_scroll) {
    using Type = window::WindowEvent::Type;
    switch (ev.type) {
        case Type::Key: {
            if (ev.key == kKeyShiftL || ev.key == kKeyShiftR) {
                shift_down_ = ev.pressed;
                return false;
            }
            return false;
        }
        case Type::MouseButton: {
            if (!ev.pressed) {
                // Release ends any active drag regardless of where the cursor is.
                const bool was = orbiting_ || panning_;
                orbiting_ = false;
                panning_  = false;
                return was;
            }
            if (!allow_drag) return false;  // ImGui owns this click.
            last_x_ = ev.mouse_x;
            last_y_ = ev.mouse_y;
            if (ev.button == 0u) {            // LMB
                if (shift_down_) panning_ = true;
                else orbiting_ = true;
            } else if (ev.button == 1u) {     // MMB -> pan
                panning_ = true;
            }
            return true;
        }
        case Type::MouseMove: {
            const int32_t dx = ev.mouse_x - last_x_;
            const int32_t dy = ev.mouse_y - last_y_;
            last_x_ = ev.mouse_x;
            last_y_ = ev.mouse_y;
            if (orbiting_) {
                yaw_   -= static_cast<float>(dx) * orbit_speed;
                pitch_ += static_cast<float>(dy) * orbit_speed;
                pitch_  = std::clamp(pitch_, -kMaxPitch, kMaxPitch);
                return true;
            }
            if (panning_) {
                // Pan in the camera's right/up plane, scaled by distance so the
                // grab feels 1:1 at any zoom.
                const math::Vec3 eye = ResolvedEye();
                const math::Vec3 fwd = (target_ - eye).Normalized();
                const math::Vec3 world_up{0.0f, 0.0f, 1.0f};
                math::Vec3 right = fwd.Cross(world_up);
                if (right.LengthSq() < 1e-8f) right = math::Vec3{1.0f, 0.0f, 0.0f};
                right = right.Normalized();
                const math::Vec3 up = right.Cross(fwd).Normalized();
                const float scale = pan_speed * distance_;
                target_ -= right * (static_cast<float>(dx) * scale);
                target_ += up * (static_cast<float>(dy) * scale);
                return true;
            }
            return false;
        }
        case Type::Scroll: {
            if (!allow_scroll) return false;  // ImGui owns the wheel.
            // Exponential dolly so each tick is a constant fraction.
            const float factor = std::pow(1.0f - zoom_speed, static_cast<float>(ev.scroll_delta));
            distance_ = std::clamp(distance_ * factor, kMinDistance, kMaxDistance);
            return true;
        }
        default:
            return false;
    }
}

math::Vec3 CameraController::ResolvedEye() const {
    // Spherical -> cartesian about +Z up. pitch elevates above the XY horizon.
    const float cp = std::cos(pitch_);
    const float sp = std::sin(pitch_);
    const float cy = std::cos(yaw_);
    const float sy = std::sin(yaw_);
    const math::Vec3 dir{cp * cy, cp * sy, sp};
    return target_ + dir * distance_;
}

void CameraController::WriteOptions(render::RasterOptions& out) const {
    out.use_camera_override = true;
    out.camera_eye          = ResolvedEye();
    out.camera_target       = target_;
    out.camera_up           = {0.0f, 0.0f, 1.0f};
    out.camera_fov_degrees  = fov_degrees;
}

void CameraController::FrameAabb(const math::Vec3& aabb_min, const math::Vec3& aabb_max) {
    const math::Vec3 center{0.5f * (aabb_min.x + aabb_max.x),
                            0.5f * (aabb_min.y + aabb_max.y),
                            0.5f * (aabb_min.z + aabb_max.z)};
    const math::Vec3 extent{aabb_max.x - aabb_min.x,
                            aabb_max.y - aabb_min.y,
                            aabb_max.z - aabb_min.z};
    float radius = 0.5f * extent.Length();
    if (!(radius > 0.0f) || std::isnan(radius)) radius = 1.0f;  // degenerate AABB.

    target_ = center;
    // Fit the bounding sphere into the vertical fov with a little margin.
    const float fov_rad = std::max(fov_degrees, 10.0f) * 3.14159265358979323846f / 180.0f;
    const float fit = radius / std::sin(fov_rad * 0.5f);
    distance_ = std::clamp(fit * 1.25f, kMinDistance, kMaxDistance);

    // A pleasant default 3/4 view.
    yaw_   = 0.9f;
    pitch_ = 0.45f;
}

}  // namespace nuka::runtime::app::viewer
