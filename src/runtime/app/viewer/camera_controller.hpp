#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer::CameraController -- the interactive orbit/pan/zoom
// camera for the M8.5 viewer (T3).
//
// It holds a spherical orbit state (yaw / pitch / distance) around a focus
// `target` and turns it into a render::RasterOptions camera OVERRIDE
// (use_camera_override = true, eye/target/up/fov) the present renderer consumes.
// Input arrives as window::WindowEvent (mouse move + button + scroll); the viewer
// feeds those events here AFTER giving ImGui first refusal (so dragging over a
// panel never spins the camera).
//
// Controls (recon open-Q 7/8: minimal, real input dispatch):
//   * LMB drag        -> orbit (yaw/pitch)
//   * MMB or Shift+LMB-> pan (slide the target in the view plane)
//   * scroll wheel    -> zoom (dolly distance, clamped)
//   * Reset()         -> frame the scene AABB (the viewer's "reset camera")
//
// Pure host C++/math -- NO Vulkan, NO ImGui, NO CUDA. It does not own any window
// or GPU state; it is a small value object the viewer ticks. This keeps it unit-
// reasonable and lets imgui_layer read/write its public fields for the camera
// panel readout + fov slider.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"  // RasterOptions
#include "render/window/window_surface.hpp"          // WindowEvent

#include <cstdint>

namespace nuka::runtime::app::viewer {

// Bring the window event POD into this namespace for the API signatures.
namespace window = nuka::render::window;

// ---------------------------------------------------------------------------
// CameraController -- spherical-orbit interactive camera state.
//
// Angles are radians. `yaw` rotates about the world up axis (+Z, the nuka
// convention -- gravity is -Z), `pitch` is the elevation above the horizon
// (clamped just shy of +/-90deg so the orbit never gimbal-flips). `distance` is
// the dolly radius from `target` to `eye`. Everything is clamped to sane bounds.
// ---------------------------------------------------------------------------
class CameraController {
public:
    CameraController() = default;

    // ---- input -------------------------------------------------------------
    // Apply ONE window event to the camera. `allow_drag` gates the mouse-drag
    // handlers (the viewer passes false when ImGui wants the mouse, so dragging a
    // panel never orbits). Scroll is also gated by `allow_scroll` (false when the
    // cursor is over a panel). Returns true if the event changed the camera.
    bool HandleEvent(const window::WindowEvent& ev, bool allow_drag, bool allow_scroll);

    // ---- per-frame ---------------------------------------------------------
    // Recompute eye/up from yaw/pitch/distance/target and write the override into
    // `out` (use_camera_override = true). fov is taken from this->fov_degrees so
    // the imgui fov slider drives it directly.
    void WriteOptions(render::RasterOptions& out) const;

    // ---- framing -----------------------------------------------------------
    // Frame an AABB: center the target, set distance to fit the bounding sphere
    // for the current fov, reset to a pleasant 3/4 view angle. The viewer calls
    // this once at startup (and on the "reset camera" command) with the scene
    // AABB so SOMETHING is always well-composed. Safe with a degenerate AABB.
    void FrameAabb(const math::Vec3& aabb_min, const math::Vec3& aabb_max);

    // ---- resolved readout (for the imgui camera panel) ---------------------
    math::Vec3 ResolvedEye() const;
    math::Vec3 ResolvedTarget() const { return target_; }
    float      Yaw() const { return yaw_; }
    float      Pitch() const { return pitch_; }
    float      Distance() const { return distance_; }

    // ---- public tunables the imgui panel reads/writes ----------------------
    float fov_degrees = 45.0f;   // the fov slider in the camera panel drives this
    float orbit_speed = 0.008f;  // radians per pixel of mouse drag
    float pan_speed   = 0.0018f; // world units per pixel, scaled by distance
    float zoom_speed  = 0.12f;   // fraction of distance per wheel tick

private:
    // Spherical orbit state about `target_`.
    float yaw_      = 0.9f;    // ~52deg around +Z -> a 3/4 view
    float pitch_    = 0.45f;   // ~26deg elevation
    float distance_ = 4.0f;    // dolly radius
    math::Vec3 target_ = {0.0f, 0.0f, 0.5f};

    // Drag bookkeeping.
    bool     orbiting_ = false;
    bool     panning_  = false;
    bool     shift_down_ = false;
    int32_t  last_x_ = 0;
    int32_t  last_y_ = 0;

    // Clamp limits.
    static constexpr float kMinDistance = 0.15f;
    static constexpr float kMaxDistance = 200.0f;
    static constexpr float kMaxPitch    = 1.5533f;  // ~89deg
};

}  // namespace nuka::runtime::app::viewer
