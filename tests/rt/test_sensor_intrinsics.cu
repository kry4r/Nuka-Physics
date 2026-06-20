// ---------------------------------------------------------------------------
// Camera intrinsics + lens distortion gate. The PinholeCamera ALWAYS carries
// intrinsics; the DEFAULT (fx<=0 => focal from vfov+aspect, centered principal
// point, distortion off, clip wide-open) must reproduce the plain-pinhole NDC
// ray-gen BYTE-IDENTICAL. Non-default intrinsics bend off-axis rays (radial
// distortion), shift rays for an off-center principal point / per-axis focal, and
// clip the depth AOV (a hit outside [near,far] reads as a miss). Compiled
// -ffp-contract=off + --fmad=false so host==device.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/backend_cuda/rt/batched_sensor_render.hpp"
#include "phi/backend_cuda/rt/sensor_scatter.hpp"
#include "phi/interop_scatter.hpp"
#include "rt/camera.hpp"
#include "rt/material.hpp"
#include "rt/two_level_render.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

using namespace nuka;

namespace {

using math::Quat;
using math::Transform;
using math::Vec3;
using phi::ScatterFkSource;
using rt::PinholeCamera;
using rt::SensorMountRow;

#define NK_CUDA_OK(call)                                          \
    do {                                                          \
        const cudaError_t e_ = (call);                           \
        ASSERT_EQ(e_, cudaSuccess) << cudaGetErrorString(e_);    \
    } while (0)

// The ORIGINAL plain-pinhole ray-gen arithmetic (fp64-internal NDC math), kept
// here as the byte-exact reference the default-intrinsics path must reproduce.
math::Vec3 ReferencePinholeDir(const PinholeCamera& c, uint32_t px, uint32_t py,
                               float jx, float jy) {
    const double sx = (2.0 * (static_cast<double>(px) + 0.5 + jx) /
                       static_cast<double>(c.width)) - 1.0;
    const double sy = 1.0 - (2.0 * (static_cast<double>(py) + 0.5 + jy) /
                             static_cast<double>(c.height));
    const double dx = sx * static_cast<double>(c.half_w);
    const double dy = sy * static_cast<double>(c.half_h);
    const double rx = static_cast<double>(c.forward.x) +
                      dx * static_cast<double>(c.right.x) + dy * static_cast<double>(c.up.x);
    const double ry = static_cast<double>(c.forward.y) +
                      dx * static_cast<double>(c.right.y) + dy * static_cast<double>(c.up.y);
    const double rz = static_cast<double>(c.forward.z) +
                      dx * static_cast<double>(c.right.z) + dy * static_cast<double>(c.up.z);
    const double inv_len = 1.0 / std::sqrt(rx * rx + ry * ry + rz * rz);
    return math::Vec3{static_cast<float>(rx * inv_len), static_cast<float>(ry * inv_len),
                      static_cast<float>(rz * inv_len)};
}

PinholeCamera DefaultCamera(uint32_t w, uint32_t h, float vfov_rad) {
    return rt::BuildPinhole(Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, -1.0f},
                            Vec3{0.0f, 1.0f, 0.0f}, vfov_rad, w, h);
}

bool DirBitEqual(const math::Vec3& a, const math::Vec3& b) {
    return std::memcmp(&a, &b, sizeof(math::Vec3)) == 0;
}

}  // namespace

// (a) Default intrinsics => GenerateRay / GenerateRayJitter are BYTE-IDENTICAL to
// the original plain-pinhole NDC arithmetic, at center, corners, and a jittered
// sub-pixel sample (the RGB-fidelity path's call).
TEST(SensorIntrinsics, DefaultIsByteIdenticalToPinhole) {
    const uint32_t W = 64u, H = 48u;
    const PinholeCamera cam = DefaultCamera(W, H, 0.9f);
    ASSERT_LE(cam.fx, rt::kIntrinsicsDeriveFromFov) << "default camera must have fx<=0";

    const uint32_t pxs[] = {0u, W / 2u, W - 1u, 7u, W - 3u};
    const uint32_t pys[] = {0u, H / 2u, H - 1u, 5u, H - 2u};
    for (uint32_t px : pxs) {
        for (uint32_t py : pys) {
            const rt::Ray got = cam.GenerateRay(px, py);
            EXPECT_TRUE(DirBitEqual(got.dir, ReferencePinholeDir(cam, px, py, 0.0f, 0.0f)))
                << "GenerateRay default != pinhole at (" << px << "," << py << ")";
            // The fidelity path's jittered sub-pixel sample.
            const float jx = 0.37f, jy = -0.21f;
            const rt::Ray gj = cam.GenerateRayJitter(px, py, jx, jy);
            EXPECT_TRUE(DirBitEqual(gj.dir, ReferencePinholeDir(cam, px, py, jx, jy)))
                << "GenerateRayJitter default != pinhole at (" << px << "," << py << ")";
        }
    }
}

// (b) Distortion ON (k1!=0): the CENTER ray is unchanged (r2==0 => d==1) and an
// off-axis corner ray BENDS away from the pinhole corner ray.
TEST(SensorIntrinsics, DistortionBendsOffAxisOnly) {
    const uint32_t W = 64u, H = 64u;
    PinholeCamera pin = DefaultCamera(W, H, 1.0f);
    // Explicit intrinsics matching the default focal/principal point, then distort.
    PinholeCamera dis = pin;
    dis.cx = 0.5f * static_cast<float>(W);
    dis.cy = 0.5f * static_cast<float>(H);
    dis.fx = (0.5f * static_cast<float>(H)) / std::tan(0.5f * 1.0f);
    dis.fy = dis.fx;
    dis.distortion = 1u;
    dis.k1 = 0.25f;
    dis.k2 = 0.05f;

    // Center pixel: on a square image with these intrinsics r2 is ~0, so d~1; the
    // distorted center must equal the (matching-intrinsics) undistorted center.
    PinholeCamera nod = dis;
    nod.distortion = 0u;
    const uint32_t cpx = W / 2u, cpy = H / 2u;
    EXPECT_TRUE(DirBitEqual(dis.GenerateRay(cpx, cpy).dir, nod.GenerateRay(cpx, cpy).dir))
        << "distortion changed the center ray (should be identity at r2~0)";

    // A corner pixel bends: the distorted corner ray differs from the pinhole one.
    const rt::Ray corner_pin = pin.GenerateRay(0u, 0u);
    const rt::Ray corner_dis = dis.GenerateRay(0u, 0u);
    const float dx = corner_dis.dir.x - corner_pin.dir.x;
    const float dy = corner_dis.dir.y - corner_pin.dir.y;
    const float dz = corner_dis.dir.z - corner_pin.dir.z;
    EXPECT_GT(std::sqrt(dx * dx + dy * dy + dz * dz), 1e-4f)
        << "distortion did not bend the off-axis corner ray";
}

// (c) Principal-point shift + fx!=fy => the ray shifts as expected. Known-value
// check: a half-pixel principal-point shift in +x maps the pixel whose center is
// at the new cx to the optical axis (x==0 => dir == forward exactly).
TEST(SensorIntrinsics, PrincipalPointAndFocalShiftRays) {
    const uint32_t W = 80u, H = 60u;
    PinholeCamera cam = DefaultCamera(W, H, 0.8f);
    // Put the principal point at the CENTER of pixel (10, 20): cx = 10.5, cy = 20.5.
    cam.cx = 10.5f;
    cam.cy = 20.5f;
    cam.fx = 90.0f;
    cam.fy = 120.0f;  // fx != fy
    cam.distortion = 0u;

    // The pixel under the principal point has x==y==0 => dir == forward (bit-exact,
    // since AssembleRay normalizes forward which is already unit).
    const rt::Ray on_axis = cam.GenerateRay(10u, 20u);
    EXPECT_TRUE(DirBitEqual(on_axis.dir, cam.forward))
        << "pixel at the principal point should look straight down forward";

    // One pixel to the +x of cx: x = (11.5-10.5)/fx = 1/90; the image-plane offset
    // is x*half_w, so dir = normalize(forward + (1/fx)*half_w*right). Check the
    // sign + magnitude of the right-component vs forward.
    const rt::Ray off = cam.GenerateRay(11u, 20u);
    const double xexp = 1.0 / static_cast<double>(cam.fx);
    const double dxp = xexp * static_cast<double>(cam.half_w);
    const double rx = static_cast<double>(cam.forward.x) + dxp * static_cast<double>(cam.right.x);
    const double ry = static_cast<double>(cam.forward.y) + dxp * static_cast<double>(cam.right.y);
    const double rz = static_cast<double>(cam.forward.z) + dxp * static_cast<double>(cam.right.z);
    const double inv = 1.0 / std::sqrt(rx * rx + ry * ry + rz * rz);
    EXPECT_NEAR(off.dir.x, static_cast<float>(rx * inv), 1e-6f);
    EXPECT_NEAR(off.dir.y, static_cast<float>(ry * inv), 1e-6f);
    EXPECT_NEAR(off.dir.z, static_cast<float>(rz * inv), 1e-6f);

    // fx != fy: a +1 pixel step in x vs y gives DIFFERENT normalized magnitudes.
    const rt::Ray offy = cam.GenerateRay(10u, 21u);
    const float ax = std::fabs(off.dir.x - on_axis.dir.x);
    const float ay = std::fabs(offy.dir.y - on_axis.dir.y);
    EXPECT_GT(std::fabs(ax - ay), 1e-5f) << "fx==fy? per-axis focal had no effect";
}

// --- (d) far_clip below the scene depth turns far hits into misses. ---
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kRes = 32u;
constexpr float kBoxHalf = 0.4f;

rt::BlasMesh BuildBoxMesh(float half) {
    const Vec3 c[8] = {
        {-half, -half, -half}, {half, -half, -half}, {half, half, -half}, {-half, half, -half},
        {-half, -half, half},  {half, -half, half},  {half, half, half},  {-half, half, half}};
    const int faces[12][3] = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7},
                              {0, 1, 5}, {0, 5, 4}, {2, 3, 7}, {2, 7, 6},
                              {1, 2, 6}, {1, 6, 5}, {0, 4, 7}, {0, 7, 3}};
    rt::BlasMesh m;
    for (auto& f : faces) m.triangles.push_back({c[f[0]], c[f[1]], c[f[2]], 0u});
    return m;
}

phi::InstanceScatterRow BaseRow() {
    phi::InstanceScatterRow r;
    r.kind = 3u;  // Base
    r.row = 0u;
    r.cached_visual_local[0] = 0.0f;
    r.cached_visual_local[1] = 0.0f;
    r.cached_visual_local[2] = 0.0f;
    r.cached_visual_local[3] = 1.0f;
    return r;
}

rt::BatchedSensorSceneDesc MakeSceneDesc() {
    rt::BatchedSensorSceneDesc d;
    d.scene.meshes.push_back(BuildBoxMesh(kBoxHalf));
    rt::Material mat;
    mat.albedo = {0.6f, 0.6f, 0.6f};
    d.scene.materials = {mat};
    d.scene.light.directional = true;
    d.scene.light.direction = {0.3f, 0.4f, -0.85f};
    d.scene.ambient.color = {0.05f, 0.05f, 0.05f};
    d.rows.push_back(BaseRow());
    d.blas_id.push_back(0u);
    d.material_id.push_back(0u);
    return d;
}

}  // namespace

// A camera 3 m above a box at the origin sees it at ~2.6 m. far_clip below that
// depth turns every hit into a miss; the depth AOV changes (hits -> +inf miss).
TEST(SensorIntrinsics, FarClipTurnsFarHitsToMisses) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";

    // ONE env, ONE overhead camera built on device from a Base pose at the origin.
    Transform base = Transform::Identity();
    Transform* d_base = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_base, sizeof(Transform)));
    NK_CUDA_OK(cudaMemcpy(d_base, &base, sizeof(Transform), cudaMemcpyHostToDevice));
    ScatterFkSource fk;
    fk.base_pose = d_base;
    fk.link_pose = nullptr;
    fk.body_pose = nullptr;
    fk.links_per_env = 0u;
    fk.bodies_per_env = 0u;

    SensorMountRow mount;
    mount.kind = 3u;  // Base
    mount.row = 0u;
    mount.local_offset[2] = 3.0f;  // 3 m above
    mount.local_offset[3] = 1.0f;  // identity quat (local -Z looks down)
    mount.fov_y = 0.5f * kPi;
    mount.width = kRes;
    mount.height = kRes;

    auto render_depth = [&](const SensorMountRow& m) {
        SensorMountRow* d_m = nullptr;
        NK_CUDA_OK(cudaMalloc(&d_m, sizeof(SensorMountRow)));
        NK_CUDA_OK(cudaMemcpy(d_m, &m, sizeof(SensorMountRow), cudaMemcpyHostToDevice));
        PinholeCamera* d_cam = nullptr;
        NK_CUDA_OK(cudaMalloc(&d_cam, sizeof(PinholeCamera)));
        rt::ScatterEnvCameras(nullptr, fk, d_m, 1u, 1u, d_cam);
        NK_CUDA_OK(cudaDeviceSynchronize());

        rt::BatchedSensorSceneDevice scene = rt::BuildBatchedSensorScene(MakeSceneDesc());
        rt::RenderSensorsBatched(scene, fk, d_cam, 1u, 1u, kRes, kRes);
        NK_CUDA_OK(cudaDeviceSynchronize());

        const size_t pix = static_cast<size_t>(kRes) * kRes;
        std::vector<float> depth(pix);
        NK_CUDA_OK(cudaMemcpy(depth.data(), rt::SensorDepthDevice(scene),
                              depth.size() * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_m);
        cudaFree(d_cam);
        return depth;
    };

    // Wide-open clip (default): the box is visible -> some finite depths near ~2.6 m.
    const std::vector<float> wide = render_depth(mount);
    size_t hits_wide = 0u;
    float min_d = 1e30f;
    for (float d : wide) {
        if (std::isfinite(d)) {
            ++hits_wide;
            if (d < min_d) min_d = d;
        }
    }
    ASSERT_GT(hits_wide, 0u) << "overhead camera saw no box (test scaffold broken)";
    ASSERT_LT(min_d, 3.0f);
    ASSERT_GT(min_d, 2.0f) << "box depth not in the expected band";

    // far_clip BELOW the box depth: every hit is clipped to a miss (+inf depth).
    SensorMountRow clipped = mount;
    clipped.far_clip = min_d - 0.1f;
    const std::vector<float> near_only = render_depth(clipped);
    size_t hits_clipped = 0u;
    for (float d : near_only) {
        if (std::isfinite(d)) ++hits_clipped;
    }
    EXPECT_EQ(hits_clipped, 0u) << "far_clip below scene depth did not drop the hits";
    EXPECT_NE(std::memcmp(wide.data(), near_only.data(), wide.size() * sizeof(float)), 0)
        << "depth AOV unchanged after far_clip (expected hits -> misses)";

    // near_clip ABOVE the box depth: also clips everything to misses.
    SensorMountRow nearclip = mount;
    nearclip.near_clip = min_d + 0.1f;
    const std::vector<float> far_only = render_depth(nearclip);
    size_t hits_near = 0u;
    for (float d : far_only) {
        if (std::isfinite(d)) ++hits_near;
    }
    EXPECT_EQ(hits_near, 0u) << "near_clip above scene depth did not drop the hits";

    cudaFree(d_base);
}
