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
// Ray -- a world-space pick ray (origin + unit direction). VIEW-1 picker math.
// Pure value type; no Vulkan/CUDA. Built from the resolved camera basis so it
// matches the raster renderer's right-handed LookAt convention EXACTLY (forward
// = normalize(target-eye), right = forward x up, true-up = right x forward).
// ---------------------------------------------------------------------------
struct Ray {
    math::Vec3 origin{};
    math::Vec3 dir{0.0f, 0.0f, -1.0f};  // unit
};

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

    // Fly the orbit rig (target + eye move together) from WASD/QE input over `dt`
    // seconds. forward/right walk the horizon-projected view plane (W/S, A/D), up is
    // world +Z (Q/E); inputs in [-1,1], speed scales with dolly distance.
    void Move(float forward, float right, float up, float dt);

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

    // ---- picker math (VIEW-1; pure host, no Vulkan) ------------------------
    // Build the world-space pick ray through pixel (px, py) of a viewport of
    // (width, height) px. (px, py) is in window coords (origin top-left, +y
    // DOWN, the xcb/ImGui convention). The ray uses the SAME camera basis +
    // fov the renderer's LookAt/Perspective use, so a screen click unprojects
    // onto the geometry the user sees. width/height/fov degenerate -> a forward
    // ray from the eye. Deterministic (no time input) for the GATE-B record.
    Ray ScreenRay(float px, float py, uint32_t width, uint32_t height) const;

    // Intersect `ray` with the plane through `plane_point` whose normal is
    // `plane_normal` (need not be unit). On a hit in front of the ray origin,
    // writes the world-space hit point to *out and returns true; returns false
    // for a back-facing / parallel / behind-origin hit. The drag handler uses a
    // plane through the grabbed entity perpendicular to the view forward so the
    // entity tracks the cursor at a constant view depth.
    bool RayPlaneHit(const Ray& ray, const math::Vec3& plane_point,
                     const math::Vec3& plane_normal, math::Vec3* out) const;

    // ---- public tunables the imgui panel reads/writes ----------------------
    float fov_degrees = 45.0f;   // the fov slider in the camera panel drives this
    float orbit_speed = 0.008f;  // radians per pixel of mouse drag
    float pan_speed   = 0.0018f; // world units per pixel, scaled by distance
    float zoom_speed  = 0.12f;   // fraction of distance per wheel tick
    float move_speed  = 1.5f;    // WASD world units/sec at dolly distance 1

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
