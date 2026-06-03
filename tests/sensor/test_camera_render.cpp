// ---------------------------------------------------------------------------
// v0.7 p14b GATE: the RGB+depth+normal+uv CAMERA SENSOR over the LIVE batched
// sim state (nuka::sensor::RenderCameraSensor). It must:
//   * SDF-INSTANCE each body's cooked SDF at its LIVE pose (DevicePoses()) and
//     render the ACTUAL sim state -- depth/normal at chosen pixels match an
//     ANALYTIC expectation for a body at a known pose; RGB == the palette color
//     for the hit body id; a missing pixel reads the depth sentinel.
//   * TRACK LIVE STATE: render, MOVE a body's pose on device, re-render, and the
//     depth/RGB for the affected pixels CHANGE (proof it is driven by DevicePoses,
//     not a static scene).
//   * camera-space NORMAL convention: the front face of a sphere viewed head-on
//     has a camera-space normal ~ (0,0,-1) (viewer-faced).
//   * be two-run byte-exact across ALL channels (D1).
//
// The poses come from a REAL runtime::gpu::BatchedDeviceWorld (DevicePoses()), so
// the test exercises the intended live-state path, not a hand-faked pose array.
// The cooked SDFs are real p07 narrow-band SDFs (CookSparseSdf), uploaded as
// device SparseSdfDevice views the SAME way nuka_rt_gpu's RenderScene does.
//
// Built -ffp-contract=off to pair with the kernel's --fmad=false so the shared
// fp64-internal SDF sphere-trace is host==device bit-for-bit.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "import/cooker/sparse_sdf_cooker.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"
#include "phi/device_context.hpp"
#include "rt/camera.hpp"
#include "rt/ray_box.cuh"  // RtMissDepth / kNoPrim
#include "runtime/gpu/batched_device_world.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "sensor/camera_render.hpp"
#include "tests/import/sdf_test_meshes.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;

// A cooked SDF + its device buffers (kept alive for the views).
struct DeviceSdf {
    runtime::sdf::SparseSdfDevice header;  // device-pointer header (one view)
    phi::Buffer keys;
    phi::Buffer values;
    phi::Buffer gradients;
};

// Cook a unit-sphere SDF (radius r at the local origin) and upload its cell arrays;
// returns a SparseSdfDevice header whose cell_* point at device memory.
DeviceSdf CookSphereSdf(float r) {
    using nuka::import::cooker::CookSparseSdf;
    using nuka::import::cooker::SparseSdfParams;
    const auto mesh = nuka::test::SphereMesh(0.0f, 0.0f, 0.0f, r, 48, 96);
    SparseSdfParams params;
    params.voxel_size = 0.05f;
    params.band_voxels = 6u;
    const auto cooked = CookSparseSdf(mesh.vertices.data(),
                                      static_cast<uint32_t>(mesh.vertices.size() / 3),
                                      mesh.indices.data(),
                                      static_cast<uint32_t>(mesh.indices.size() / 3),
                                      params);
    DeviceSdf out;
    out.header.origin = {cooked.origin[0], cooked.origin[1], cooked.origin[2]};
    out.header.voxel_size = cooked.voxel_size;
    out.header.dims[0] = cooked.dims[0];
    out.header.dims[1] = cooked.dims[1];
    out.header.dims[2] = cooked.dims[2];
    std::vector<math::Vec3> grads(cooked.CellCount());
    for (uint32_t c = 0; c < cooked.CellCount(); ++c) {
        grads[c] = {cooked.cell_gradients[3 * c], cooked.cell_gradients[3 * c + 1],
                    cooked.cell_gradients[3 * c + 2]};
    }
    out.keys = phi::UploadVector(cooked.cell_keys);
    out.values = phi::UploadVector(cooked.cell_values);
    out.gradients = phi::UploadVector(grads);
    out.header.cell_keys = static_cast<const uint64_t*>(out.keys.Data());
    out.header.cell_values = static_cast<const float*>(out.values.Data());
    out.header.cell_gradients = static_cast<const math::Vec3*>(out.gradients.Data());
    out.header.cell_count = cooked.CellCount();
    return out;
}

// Build a minimal BatchedDeviceWorld with `bodies` free rigid bodies per env over
// `envs` envs (we only use its DevicePoses() as the LIVE pose source). The bodies
// are placed at the given local positions in the template.
runtime::gpu::BatchedDeviceWorld BuildPoseWorld(uint32_t envs,
                                                const std::vector<math::Vec3>& positions) {
    scene::SceneIR ir;
    for (uint32_t b = 0; b < positions.size(); ++b) {
        scene::RigidBodyRecord body;
        body.name = "b" + std::to_string(b);
        body.mass = 1.0f;
        body.inertia = {1.0f, 1.0f, 1.0f};
        body.local_transform.position = positions[b];
        ir.AddRigidBody(std::move(body));
    }
    const auto blob = scene::CookScene(ir);
    auto world = runtime::BuildWorld(blob);
    std::vector<runtime::WorldInstance> instances(envs, world.instance);
    return runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);
}

struct RenderOut {
    std::vector<float> rgb;
    std::vector<float> depth;
    std::vector<float> normal;
    std::vector<float> uv;
};

// Run RenderCameraSensor end-to-end; download all channels. `cams` is per-env.
RenderOut Render(const phi::DeviceContext& ctx,
                 const runtime::gpu::BatchedDeviceWorld& world,
                 const runtime::sdf::SparseSdfDevice* d_views,
                 const std::vector<uint32_t>& body_to_sdf,
                 const std::vector<uint32_t>& body_material,
                 const std::vector<rt::PinholeCamera>& cams, uint32_t w, uint32_t h) {
    const uint32_t envs = world.InstanceCount();
    const uint32_t bodies = world.BodyCountPerInstance();
    const size_t pixels = static_cast<size_t>(w) * h;

    phi::Buffer d_b2s = phi::UploadVector(body_to_sdf);
    phi::Buffer d_mat = phi::UploadVector(body_material);

    phi::Buffer d_rgb(envs * pixels * 3u * sizeof(float), phi::MemoryKind::Device);
    phi::Buffer d_depth(envs * pixels * sizeof(float), phi::MemoryKind::Device);
    phi::Buffer d_normal(envs * pixels * 3u * sizeof(float), phi::MemoryKind::Device);
    phi::Buffer d_uv(envs * pixels * 2u * sizeof(float), phi::MemoryKind::Device);

    sensor::RenderCameraSensor(
        ctx, world.DevicePoses(), d_views,
        static_cast<const uint32_t*>(d_b2s.Data()),
        static_cast<const uint32_t*>(d_mat.Data()), cams.data(), envs, bodies, w, h,
        static_cast<float*>(d_rgb.Data()), static_cast<float*>(d_depth.Data()),
        static_cast<float*>(d_normal.Data()), static_cast<float*>(d_uv.Data()));
    ctx.stream.Synchronize();

    RenderOut out;
    out.rgb.resize(envs * pixels * 3u);
    out.depth.resize(envs * pixels);
    out.normal.resize(envs * pixels * 3u);
    out.uv.resize(envs * pixels * 2u);
    d_rgb.CopyToHost(out.rgb.data(), out.rgb.size() * sizeof(float));
    d_depth.CopyToHost(out.depth.data(), out.depth.size() * sizeof(float));
    d_normal.CopyToHost(out.normal.data(), out.normal.size() * sizeof(float));
    d_uv.CopyToHost(out.uv.data(), out.uv.size() * sizeof(float));
    return out;
}

}  // namespace

// =====================================================================
// (1) Analytic depth + camera-space normal + palette RGB + miss sentinel,
//     SDF-instanced at the LIVE pose. One env, one sphere body at a known pose.
// =====================================================================
TEST(CameraRender, AnalyticDepthNormalPaletteAndMiss) {
    const auto ctx = phi::MakeDefaultDeviceContext();

    // One env, one body (a unit sphere) whose local position in the template is the
    // ORIGIN; the body lives at world origin (no per-env offset).
    const float radius = 1.0f;
    auto world = BuildPoseWorld(1u, {{0.0f, 0.0f, 0.0f}});
    ASSERT_EQ(world.InstanceCount(), 1u);
    ASSERT_EQ(world.BodyCountPerInstance(), 1u);

    DeviceSdf sphere = CookSphereSdf(radius);
    std::vector<runtime::sdf::SparseSdfDevice> views{sphere.header};
    phi::Buffer d_views = phi::UploadVector(views);

    std::vector<uint32_t> body_to_sdf{0u};
    std::vector<uint32_t> body_material{2u};  // palette id 2 -> blue

    const uint32_t W = 64u, H = 64u;
    // Camera looking down +z from z=-4 toward the origin.
    std::vector<rt::PinholeCamera> cams{rt::BuildPinhole(
        {0.0f, 0.0f, -4.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.5f * kPi, W, H)};

    auto out = Render(ctx, world, static_cast<const runtime::sdf::SparseSdfDevice*>(d_views.Data()),
                      body_to_sdf, body_material, cams, W, H);

    // Center pixel: ray from z=-4 toward +z hits the sphere front face at z=-1, so
    // depth ~ 3.0 (within surface_eps + voxel).
    const uint32_t cx = W / 2, cy = H / 2;
    const size_t cp = static_cast<size_t>(cy) * W + cx;
    ASSERT_NE(out.depth[cp], rt::RtMissDepth()) << "center ray missed the sphere";
    const float analytic = 3.0f;
    const float tol = 0.2f * sphere.header.voxel_size + sphere.header.voxel_size;
    std::printf("[p14b/1] center depth=%g analytic=%g tol=%g\n", out.depth[cp], analytic, tol);
    EXPECT_LT(std::fabs(out.depth[cp] - analytic), tol);

    // Camera-space normal at center: the front face faces the camera, so the
    // camera-space normal is ~ (0,0,-1).
    const float nx = out.normal[cp * 3 + 0], ny = out.normal[cp * 3 + 1],
                nz = out.normal[cp * 3 + 2];
    std::printf("[p14b/1] center cam-normal=(%g,%g,%g)\n", nx, ny, nz);
    EXPECT_NEAR(nx, 0.0f, 0.05f);
    EXPECT_NEAR(ny, 0.0f, 0.05f);
    EXPECT_LT(nz, -0.9f);  // viewer-faced (camera-z negative)

    // RGB at center: palette[2] (blue) = (0.25,0.45,0.90).
    EXPECT_NEAR(out.rgb[cp * 3 + 0], 0.25f, 1e-4f);
    EXPECT_NEAR(out.rgb[cp * 3 + 1], 0.45f, 1e-4f);
    EXPECT_NEAR(out.rgb[cp * 3 + 2], 0.90f, 1e-4f);

    // A corner pixel misses (the sphere fills the center, not the corners with this
    // 90deg vfov from z=-4): depth sentinel + black RGB.
    const size_t corner = 0;  // (0,0)
    EXPECT_EQ(out.depth[corner], rt::RtMissDepth());
    EXPECT_EQ(out.rgb[corner * 3 + 0], 0.0f);
    EXPECT_EQ(out.rgb[corner * 3 + 1], 0.0f);
    EXPECT_EQ(out.rgb[corner * 3 + 2], 0.0f);

    // Count hits (sanity: the sphere covers a meaningful patch).
    size_t hits = 0;
    for (size_t i = 0; i < out.depth.size(); ++i)
        if (out.depth[i] != rt::RtMissDepth()) ++hits;
    std::printf("[p14b/1] hits=%zu / %u\n", hits, W * H);
    EXPECT_GT(hits, 100u);
}

// =====================================================================
// (2) LIVE TRACKING: move a body's pose on device, re-render, assert the image
//     changes for the affected pixels (the un-fakeable "tracks live state" gate).
// =====================================================================
TEST(CameraRender, ImageTracksLivePose) {
    const auto ctx = phi::MakeDefaultDeviceContext();

    auto world = BuildPoseWorld(1u, {{0.0f, 0.0f, 0.0f}});
    DeviceSdf sphere = CookSphereSdf(1.0f);
    std::vector<runtime::sdf::SparseSdfDevice> views{sphere.header};
    phi::Buffer d_views = phi::UploadVector(views);
    std::vector<uint32_t> body_to_sdf{0u};
    std::vector<uint32_t> body_material{0u};

    const uint32_t W = 64u, H = 64u;
    std::vector<rt::PinholeCamera> cams{rt::BuildPinhole(
        {0.0f, 0.0f, -4.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.5f * kPi, W, H)};

    auto before = Render(ctx, world, static_cast<const runtime::sdf::SparseSdfDevice*>(d_views.Data()),
                         body_to_sdf, body_material, cams, W, H);

    const uint32_t cx = W / 2, cy = H / 2;
    const size_t cp = static_cast<size_t>(cy) * W + cx;
    const float depth_before = before.depth[cp];
    ASSERT_NE(depth_before, rt::RtMissDepth());

    // MOVE the body toward the camera by 1.0 along -z (closer) ON DEVICE, in place.
    math::Transform* d_poses = world.DevicePoses();
    math::Transform pose0;
    ASSERT_EQ(cudaMemcpy(&pose0, d_poses, sizeof(math::Transform),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    pose0.position.z -= 1.0f;  // move from z=0 to z=-1 (closer to camera at z=-4)
    ASSERT_EQ(cudaMemcpy(d_poses, &pose0, sizeof(math::Transform),
                         cudaMemcpyHostToDevice), cudaSuccess);

    auto after = Render(ctx, world, static_cast<const runtime::sdf::SparseSdfDevice*>(d_views.Data()),
                        body_to_sdf, body_material, cams, W, H);

    const float depth_after = after.depth[cp];
    ASSERT_NE(depth_after, rt::RtMissDepth());
    std::printf("[p14b/2] center depth before=%g after=%g (moved -1 in z)\n",
                depth_before, depth_after);
    // Closer body -> smaller depth by ~1.0.
    EXPECT_NEAR(depth_after, depth_before - 1.0f, 0.1f);
    EXPECT_LT(depth_after, depth_before - 0.5f);

    // The whole depth image changed: count pixels whose depth differs.
    size_t changed = 0;
    for (size_t i = 0; i < before.depth.size(); ++i)
        if (before.depth[i] != after.depth[i]) ++changed;
    std::printf("[p14b/2] changed depth pixels=%zu\n", changed);
    EXPECT_GT(changed, 100u);
}

// =====================================================================
// (3) BATCHED: per-env poses differ -> per-env images differ. Two envs, the body
//     at a different per-env world position (env 1 shifted in x).
// =====================================================================
TEST(CameraRender, BatchedPerEnvDiffersWithPose) {
    const auto ctx = phi::MakeDefaultDeviceContext();

    auto world = BuildPoseWorld(2u, {{0.0f, 0.0f, 0.0f}});
    ASSERT_EQ(world.InstanceCount(), 2u);
    // Shift env 1's body in x on device so its image differs from env 0.
    math::Transform* d_poses = world.DevicePoses();
    const uint32_t bodies = world.BodyCountPerInstance();
    math::Transform env1;
    ASSERT_EQ(cudaMemcpy(&env1, d_poses + bodies, sizeof(math::Transform),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    env1.position.x += 0.7f;
    ASSERT_EQ(cudaMemcpy(d_poses + bodies, &env1, sizeof(math::Transform),
                         cudaMemcpyHostToDevice), cudaSuccess);

    DeviceSdf sphere = CookSphereSdf(1.0f);
    std::vector<runtime::sdf::SparseSdfDevice> views{sphere.header};
    phi::Buffer d_views = phi::UploadVector(views);
    std::vector<uint32_t> body_to_sdf{0u};
    std::vector<uint32_t> body_material{1u};

    const uint32_t W = 48u, H = 48u;
    std::vector<rt::PinholeCamera> cams(2u, rt::BuildPinhole(
        {0.0f, 0.0f, -4.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.5f * kPi, W, H));

    auto out = Render(ctx, world, static_cast<const runtime::sdf::SparseSdfDevice*>(d_views.Data()),
                      body_to_sdf, body_material, cams, W, H);

    const size_t pixels = static_cast<size_t>(W) * H;
    // The two env depth slices must differ (env 1's sphere is shifted in x).
    size_t diff = 0;
    for (size_t i = 0; i < pixels; ++i)
        if (out.depth[i] != out.depth[pixels + i]) ++diff;
    std::printf("[p14b/3] per-env depth diff pixels=%zu\n", diff);
    EXPECT_GT(diff, 50u);
}

// =====================================================================
// (4) D1 two-run byte-exact across ALL channels.
// =====================================================================
TEST(CameraRender, DeterministicTwoRunByteExact) {
    const auto ctx = phi::MakeDefaultDeviceContext();

    auto world = BuildPoseWorld(3u, {{0.0f, 0.0f, 0.0f}, {0.6f, 0.3f, 0.0f}});
    DeviceSdf sphere = CookSphereSdf(0.8f);
    std::vector<runtime::sdf::SparseSdfDevice> views{sphere.header};
    phi::Buffer d_views = phi::UploadVector(views);
    // Two bodies; body 0 has the SDF, body 1 also uses the same SDF view.
    std::vector<uint32_t> body_to_sdf{0u, 0u};
    std::vector<uint32_t> body_material{3u, 5u};

    const uint32_t W = 40u, H = 40u;
    std::vector<rt::PinholeCamera> cams(3u, rt::BuildPinhole(
        {0.0f, 0.0f, -4.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.5f * kPi, W, H));

    auto run1 = Render(ctx, world, static_cast<const runtime::sdf::SparseSdfDevice*>(d_views.Data()),
                       body_to_sdf, body_material, cams, W, H);
    auto run2 = Render(ctx, world, static_cast<const runtime::sdf::SparseSdfDevice*>(d_views.Data()),
                       body_to_sdf, body_material, cams, W, H);

    EXPECT_EQ(0, std::memcmp(run1.rgb.data(), run2.rgb.data(),
                             run1.rgb.size() * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(run1.depth.data(), run2.depth.data(),
                             run1.depth.size() * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(run1.normal.data(), run2.normal.data(),
                             run1.normal.size() * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(run1.uv.data(), run2.uv.data(),
                             run1.uv.size() * sizeof(float)));
    std::printf("[p14b/4] two-run byte-exact across rgb/depth/normal/uv OK\n");
}
