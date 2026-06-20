#pragma once
// ---------------------------------------------------------------------------
// nuka::rt::PinholeCamera -- deterministic pinhole ray generation (v0.7 p12).
//
// A bare-bones pinhole camera for the self-written CUDA ray tracer. ONE ray per
// pixel through the pixel CENTER -- NO anti-aliasing / jitter / stochastic
// sampling (those need a seeded deterministic RNG and would break D1 if added
// naively; they are deferred to p13/later). Ray generation is a pure function of
// the pixel index + camera params, so it is byte-identical run-to-run (D1).
//
// HOST-buildable, DEVICE-callable: the whole struct + GenerateRay are
// __host__ __device__ (NUKA_RT_HD) so the SAME ray-gen runs in the render kernel
// AND in the CPU brute-force oracle (no mirrored arithmetic -> the only thing
// that can diverge between GPU and oracle is the leaf slab test, which is itself
// a single shared function in ray_box.cuh).
//
// Convention: right-handed, camera looks down `forward`; `right` and `up` span
// the image plane. Image-plane half-extents derive from a vertical FOV and the
// aspect ratio. Pixel (px,py): px increases to the +right, py increases DOWN the
// image (row 0 at the TOP), which is the conventional framebuffer layout.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_RT_HD __host__ __device__
#else
#define NUKA_RT_HD
#endif

namespace nuka::rt {

// A single generated ray: origin + (normalized) direction. We keep the
// direction normalized so the returned slab `t` is a true world-space distance
// (depth), which is what the oracle and the depth buffer compare.
struct Ray {
    math::Vec3 origin;
    math::Vec3 dir; // unit length
};

// fx<=0 marks "derive focal from vfov+aspect, centered principal point" -- the
// default pinhole. Any positive fx means the camera carries explicit intrinsics.
inline constexpr float kIntrinsicsDeriveFromFov = 0.0f;

// Pinhole camera. Build it on the host (BuildPinhole), pass it by value into the
// kernel (it is trivially copyable), and call GenerateRay on device or host.
//
// The camera ALWAYS carries intrinsics. Default intrinsics (fx==fy==0 => focal
// from vfov+aspect, cx/cy centered, distortion off, clip wide-open) reproduce the
// plain-pinhole NDC ray-gen BYTE-IDENTICAL; non-default values give an off-center
// principal point, per-axis focal, and radial (Brown-Conrady) lens distortion.
struct PinholeCamera {
    math::Vec3 origin = {0.0f, 0.0f, 0.0f};
    math::Vec3 forward = {0.0f, 0.0f, 1.0f}; // unit
    math::Vec3 right = {1.0f, 0.0f, 0.0f};   // unit, image +x
    math::Vec3 up = {0.0f, 1.0f, 0.0f};      // unit, image +y (world up)
    uint32_t width = 0u;
    uint32_t height = 0u;
    // Half-extents of the image plane at unit distance (tan(fov/2)).
    float half_w = 1.0f;
    float half_h = 1.0f;

    // Per-axis focal (pixels) + principal point (pixels). fx<=0 => default: focal
    // from vfov+aspect, cx=width/2, cy=height/2 (the byte-exact NDC pinhole).
    float fx = kIntrinsicsDeriveFromFov;
    float fy = kIntrinsicsDeriveFromFov;
    float cx = 0.0f;
    float cy = 0.0f;
    // Radial distortion gated by `distortion`!=0: r2=x*x+y*y, d=1+k1*r2+k2*r2*r2.
    float k1 = 0.0f;
    float k2 = 0.0f;
    uint8_t distortion = 0u;
    // Depth AOV clip; a hit outside [near_clip, far_clip] reads as a miss.
    float near_clip = 0.01f;
    float far_clip = 1000.0f;

    // Assemble + normalize dir = forward + dx*right + dy*up in fp64 (host/device
    // identical, stored fp32). Shared tail of both the default and intrinsics paths.
    NUKA_RT_HD Ray AssembleRay(double dx, double dy) const {
        const double rx = static_cast<double>(forward.x) +
                          dx * static_cast<double>(right.x) +
                          dy * static_cast<double>(up.x);
        const double ry = static_cast<double>(forward.y) +
                          dx * static_cast<double>(right.y) +
                          dy * static_cast<double>(up.y);
        const double rz = static_cast<double>(forward.z) +
                          dx * static_cast<double>(right.z) +
                          dy * static_cast<double>(up.z);
        const double inv_len = 1.0 / sqrt(rx * rx + ry * ry + rz * rz);
        Ray ray;
        ray.origin = origin;
        ray.dir = math::Vec3{static_cast<float>(rx * inv_len),
                             static_cast<float>(ry * inv_len),
                             static_cast<float>(rz * inv_len)};
        return ray;
    }

    // Ray through pixel (px,py) offset by sub-pixel (jx,jy) in pixel units. The
    // default branch (fx<=0) is the original NDC arithmetic, bit-for-bit; the
    // intrinsics branch uses principal point + per-axis focal + radial distortion.
    NUKA_RT_HD Ray GenerateRayJitter(uint32_t px, uint32_t py, float jx, float jy) const {
        if (fx <= kIntrinsicsDeriveFromFov) {
            const double sx = (2.0 * (static_cast<double>(px) + 0.5 + jx) /
                               static_cast<double>(width)) - 1.0;
            const double sy = 1.0 - (2.0 * (static_cast<double>(py) + 0.5 + jy) /
                                     static_cast<double>(height));
            return AssembleRay(sx * static_cast<double>(half_w),
                               sy * static_cast<double>(half_h));
        }
        // Continuous pixel coord -> normalized sensor coord (y up). Image-plane
        // offset = normalized * half-extent (half_w/half_h already encode focal).
        const double u = static_cast<double>(px) + 0.5 + static_cast<double>(jx);
        const double v = static_cast<double>(py) + 0.5 + static_cast<double>(jy);
        double x = (u - static_cast<double>(cx)) / static_cast<double>(fx);
        double y = (static_cast<double>(cy) - v) / static_cast<double>(fy);
        if (distortion != 0u) {
            const double r2 = x * x + y * y;
            const double d = 1.0 + static_cast<double>(k1) * r2 +
                             static_cast<double>(k2) * r2 * r2;
            x *= d;
            y *= d;
        }
        return AssembleRay(x * static_cast<double>(half_w),
                           y * static_cast<double>(half_h));
    }

    // Deterministic ray through the CENTER of pixel (px, py). Row 0 is the TOP
    // of the image (py grows downward), matching the framebuffer layout.
    NUKA_RT_HD Ray GenerateRay(uint32_t px, uint32_t py) const {
        return GenerateRayJitter(px, py, 0.0f, 0.0f);
    }
};

// Unit-length normalize matching math::Vec3::Normalized bit-for-bit (sqrtf is the
// correctly-rounded IEEE op on host and device; len<1e-12 -> zero), HD so the same
// basis math runs on the host AND in the device camera-mount kernel.
NUKA_RT_HD inline math::Vec3 NormalizeVec(const math::Vec3& v) {
    const float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-12f) return math::Vec3{};
    return math::Vec3{v.x / len, v.y / len, v.z / len};
}

// Build the pinhole basis from an eye + a forward + an up reference: forward and
// right span the image plane, up is re-orthogonalized. ONE shared math the host
// BuildPinhole and the device camera-mount kernel both call (no duplicated basis).
NUKA_RT_HD inline PinholeCamera BuildPinholeBasis(math::Vec3 eye,
                                                  math::Vec3 forward_dir,
                                                  math::Vec3 up_ref,
                                                  float vfov_radians,
                                                  uint32_t width,
                                                  uint32_t height) {
    PinholeCamera cam;
    cam.origin = eye;
    cam.width = width;
    cam.height = height;

    const math::Vec3 fwd = NormalizeVec(forward_dir);
    const math::Vec3 rgt = NormalizeVec(fwd.Cross(up_ref));
    const math::Vec3 up = NormalizeVec(rgt.Cross(fwd));
    cam.forward = fwd;
    cam.right = rgt;
    cam.up = up;

    cam.half_h = tanf(0.5f * vfov_radians);
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    cam.half_w = cam.half_h * aspect;
    return cam;
}

// Build a pinhole camera looking from `eye` toward `target`, with `world_up`
// defining roll. `vfov_radians` is the FULL vertical field of view. The image is
// `width` x `height`; aspect = width/height. Delegates to BuildPinholeBasis with
// forward = target - eye. Deterministic (no RNG).
inline PinholeCamera BuildPinhole(math::Vec3 eye,
                                  math::Vec3 target,
                                  math::Vec3 world_up,
                                  float vfov_radians,
                                  uint32_t width,
                                  uint32_t height) {
    return BuildPinholeBasis(eye, target - eye, world_up, vfov_radians, width, height);
}

} // namespace nuka::rt

#undef NUKA_RT_HD
