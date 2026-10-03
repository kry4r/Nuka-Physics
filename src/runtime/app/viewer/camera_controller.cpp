// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer::CameraController -- implementation.
//
// Pure host C++/math: orbit/pan/zoom -> RasterOptions camera override. NO Vulkan,
// NO ImGui, NO CUDA. See camera_controller.hpp for the control mapping.
// ---------------------------------------------------------------------------

#include "runtime/app/viewer/camera_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nuka::runtime::app::viewer {

namespace {

// X11 keysyms (XKB_KEY_*/XK_* protocol values) for the Shift keys. The window
// backend resolves the raw keycode->keysym, so this match is KEYMAP-INDEPENDENT.
// Shift is tracked so Shift+LMB acts as pan.
constexpr uint32_t kKeyShiftL = 0xffe1u;  // XKB_KEY_Shift_L / XK_Shift_L
constexpr uint32_t kKeyShiftR = 0xffe2u;  // XKB_KEY_Shift_R / XK_Shift_R

bool RenderBounds(const render::RenderWorld& world, const scene::EntityId* selected,
                  math::Vec3& lo, math::Vec3& hi) {
    const float limit = std::numeric_limits<float>::max();
    lo = {limit, limit, limit};
    hi = {-limit, -limit, -limit};
    bool found = false;
    for (const auto& instance : world.instances) {
        if (selected && instance.entity != *selected) continue;
        if (instance.mesh_id >= world.meshes.Count()) continue;
        const auto& positions = world.meshes.Geometry(instance.mesh_id).positions;
        for (size_t i = 0; i + 2u < positions.size(); i += 3u) {
            const auto point = instance.world_xform.TransformPoint({positions[i], positions[i + 1u], positions[i + 2u]});
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
            lo.x = std::min(lo.x, point.x); lo.y = std::min(lo.y, point.y); lo.z = std::min(lo.z, point.z);
            hi.x = std::max(hi.x, point.x); hi.y = std::max(hi.y, point.y); hi.z = std::max(hi.z, point.z);
            found = true;
        }
    }
    return found;
}

}  // namespace

bool CameraController::HandleEvent(const window::WindowEvent& ev, bool allow_drag,
                                   bool allow_scroll) {
    using Type = window::WindowEvent::Type;
    switch (ev.type) {
        case Type::Key: {
            if (ev.keysym == kKeyShiftL || ev.keysym == kKeyShiftR) {
                shift_down_ = ev.pressed;
                return false;
            }
            return false;
        }
        case Type::FocusLost: {
            // Releases are lost across a focus switch: drop latched state.
            shift_down_ = false;
            orbiting_ = false;
            panning_ = false;
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
            if (!allow_drag) {
                orbiting_ = panning_ = false;
                return false;
            }
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

void CameraController::Move(float forward, float right_in, float up, float dt) {
    if (dt <= 0.0f) return;
    // Horizon-projected view forward + horizontal right (z=0) so WASD walks the
    // ground plane; Q/E rides world up. Move target_; eye follows rigidly.
    const math::Vec3 eye = ResolvedEye();
    math::Vec3 fwd = target_ - eye;
    fwd.z = 0.0f;
    if (fwd.LengthSq() < 1e-8f) fwd = math::Vec3{-std::cos(yaw_), -std::sin(yaw_), 0.0f};
    fwd = fwd.Normalized();
    const math::Vec3 world_up{0.0f, 0.0f, 1.0f};
    math::Vec3 right = fwd.Cross(world_up);
    if (right.LengthSq() < 1e-8f) right = math::Vec3{1.0f, 0.0f, 0.0f};
    right = right.Normalized();
    const float step = move_speed * std::max(distance_, 0.5f) * dt;
    target_ += fwd * (forward * step);
    target_ += right * (right_in * step);
    target_ += world_up * (up * step);
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

Ray CameraController::ScreenRay(float px, float py, uint32_t width,
                                uint32_t height) const {
    Ray ray;
    const math::Vec3 eye = ResolvedEye();
    ray.origin = eye;
    if (width == 0u || height == 0u) {
        ray.dir = (target_ - eye).Normalized();
        return ray;
    }
    // Camera basis -- EXACTLY the raster renderer's right-handed LookAt:
    //   forward = normalize(target - eye); right = forward x up; up = right x forward.
    const math::Vec3 world_up{0.0f, 0.0f, 1.0f};
    math::Vec3 fwd = (target_ - eye).Normalized();
    math::Vec3 right = fwd.Cross(world_up);
    if (right.LengthSq() < 1e-8f) right = math::Vec3{1.0f, 0.0f, 0.0f};
    right = right.Normalized();
    const math::Vec3 up = right.Cross(fwd).Normalized();

    // NDC in [-1, 1], y UP (window py is top-left/+y-down -> flip). Pixel centers
    // at +0.5 so the central pixel maps near the optical axis.
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float ndc_x = (2.0f * (px + 0.5f) / static_cast<float>(width)) - 1.0f;
    const float ndc_y = 1.0f - (2.0f * (py + 0.5f) / static_cast<float>(height));
    const float fov_rad =
        std::max(fov_degrees, 1.0f) * 3.14159265358979323846f / 180.0f;
    const float tan_half_y = std::tan(fov_rad * 0.5f);
    const float tan_half_x = tan_half_y * aspect;

    math::Vec3 dir = fwd + right * (ndc_x * tan_half_x) + up * (ndc_y * tan_half_y);
    ray.dir = dir.Normalized();
    return ray;
}

bool CameraController::RayPlaneHit(const Ray& ray, const math::Vec3& plane_point,
                                   const math::Vec3& plane_normal,
                                   math::Vec3* out) const {
    const float denom = ray.dir.Dot(plane_normal);
    if (std::fabs(denom) < 1e-8f) return false;  // parallel
    const float t = (plane_point - ray.origin).Dot(plane_normal) / denom;
    if (!(t > 0.0f)) return false;  // behind / on the origin
    if (out) *out = ray.origin + ray.dir * t;
    return true;
}

bool CameraController::FrameAabb(const math::Vec3& aabb_min, const math::Vec3& aabb_max, float aspect) {
    if (!std::isfinite(aabb_min.x) || !std::isfinite(aabb_min.y) || !std::isfinite(aabb_min.z) ||
        !std::isfinite(aabb_max.x) || !std::isfinite(aabb_max.y) || !std::isfinite(aabb_max.z) ||
        aabb_min.x > aabb_max.x || aabb_min.y > aabb_max.y || aabb_min.z > aabb_max.z ||
        !std::isfinite(fov_degrees)) return false;
    const math::Vec3 center{
        static_cast<float>((static_cast<double>(aabb_min.x) + aabb_max.x) * 0.5),
        static_cast<float>((static_cast<double>(aabb_min.y) + aabb_max.y) * 0.5),
        static_cast<float>((static_cast<double>(aabb_min.z) + aabb_max.z) * 0.5)};
    const double dx = static_cast<double>(aabb_max.x) - aabb_min.x;
    const double dy = static_cast<double>(aabb_max.y) - aabb_min.y;
    const double dz = static_cast<double>(aabb_max.z) - aabb_min.z;
    double radius = 0.5 * std::hypot(dx, dy, dz);
    if (radius == 0.0) radius = 1.0;
    const double ratio = std::isfinite(aspect) && aspect > 0.0f ? aspect : 1.0;
    const double fov = std::clamp(static_cast<double>(fov_degrees), 10.0, 170.0) * 3.14159265358979323846 / 180.0;
    const double half_fov = std::atan(std::tan(fov * 0.5) * std::min(ratio, 1.0));
    const double fit = radius / std::sin(half_fov) * 1.25;
    if (!std::isfinite(fit) || fit > kMaxDistance) return false;
    const float distance = std::max(static_cast<float>(fit), kMinDistance);
    const math::Vec3 direction{std::cos(0.45f) * std::cos(0.9f), std::cos(0.45f) * std::sin(0.9f), std::sin(0.45f)};
    const auto eye = center + direction * distance;
    if (!std::isfinite(eye.x) || !std::isfinite(eye.y) || !std::isfinite(eye.z) ||
        (eye - center).LengthSq() == 0.0f) return false;
    target_ = center;
    distance_ = distance;
    yaw_ = 0.9f;
    pitch_ = 0.45f;
    return true;
}

bool CameraController::FramePreservingView(const math::Vec3& lo, const math::Vec3& hi, float aspect) {
    CameraController framed = *this;
    if (!framed.FrameAabb(lo, hi, aspect)) return false;
    framed.yaw_ = yaw_;
    framed.pitch_ = pitch_;
    const auto eye = framed.ResolvedEye();
    if (!std::isfinite(eye.x) || !std::isfinite(eye.y) || !std::isfinite(eye.z) ||
        (eye - framed.target_).LengthSq() == 0.0f) return false;
    framed.orbiting_ = framed.panning_ = false;
    *this = framed;
    return true;
}

bool CameraController::FrameAll(const render::RenderWorld& world, float aspect) {
    math::Vec3 lo, hi;
    return RenderBounds(world, nullptr, lo, hi) && FramePreservingView(lo, hi, aspect);
}

bool CameraController::FrameSelected(const render::RenderWorld& world, scene::EntityId selected, float aspect) {
    if (selected == scene::kInvalidEntity) return false;
    math::Vec3 lo, hi;
    return RenderBounds(world, &selected, lo, hi) && FramePreservingView(lo, hi, aspect);
}

void CameraController::SetView(const math::Vec3& target, float distance, float yaw,
                              float pitch) {
    target_   = target;
    distance_ = std::clamp(distance, kMinDistance, kMaxDistance);
    yaw_      = yaw;
    pitch_    = std::clamp(pitch, -kMaxPitch, kMaxPitch);
}

}  // namespace nuka::runtime::app::viewer
