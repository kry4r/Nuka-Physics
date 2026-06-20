// ---------------------------------------------------------------------------
// GPU lidar range trace gate: lidar rides the SAME batched TLAS/LBVH/ClosestHit
// the camera path builds (sensor_scatter + batched_sensor_render). The range trace
// over [E*S*az*el] generates each (az,el) ray and writes the world distance.
//
// Gates (known-distance geometry, fp tol):
//  (1) a box at a KNOWN distance along a KNOWN (az,el) ray -> range == distance;
//      an off-target ray that misses every box -> range == max_range; the range is
//      clamped to [min_range, max_range].
//  (2) per-env determinism: every env renders the same scene at the same mount, so
//      the per-env (az,el) range tiles are byte-identical across the N per-env TLASes.
//  (3) the MOUNTED path (SetLidarMounts + RenderLidarsMounted, the shipped surface)
//      produces the same finite ranges as the device-pointer path.
//
// Convention: az swept az_min->az_max about local +Z, el about local +X; az=el=0 =>
// local +X. Range tensor is (E, S, az, el) row-major (az outer, el inner).
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/backend_cuda/rt/batched_sensor_render.hpp"
#include "phi/backend_cuda/rt/sensor_scatter.hpp"
#include "phi/interop_scatter.hpp"
#include "rt/material.hpp"
#include "rt/two_level_render.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

using namespace nuka;

namespace {

constexpr float kBoxHalf = 0.5f;

#define NK_CUDA_OK(call)                                          \
    do {                                                          \
        const cudaError_t e_ = (call);                           \
        ASSERT_EQ(e_, cudaSuccess) << cudaGetErrorString(e_);    \
    } while (0)

// One UNIT box mesh (12 triangles) in LOCAL space, shared by every instance.
rt::BlasMesh BuildBoxMesh(float half) {
    const math::Vec3 c[8] = {
        {-half, -half, -half}, {half, -half, -half}, {half, half, -half}, {-half, half, -half},
        {-half, -half, half},  {half, -half, half},  {half, half, half},  {-half, half, half}};
    const int faces[12][3] = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7},
                              {0, 1, 5}, {0, 5, 4}, {2, 3, 7}, {2, 7, 6},
                              {1, 2, 6}, {1, 6, 5}, {0, 4, 7}, {0, 7, 3}};
    rt::BlasMesh m;
    for (auto& f : faces) {
        m.triangles.push_back({c[f[0]], c[f[1]], c[f[2]], 0u});
    }
    return m;
}

phi::InstanceScatterRow BaseRow(const math::Transform& cvl) {
    phi::InstanceScatterRow r;
    r.kind = 3u;  // Base-mounted: world = base_pose[env] * cvl.
    r.row = 0u;
    r.cached_visual_local[0] = cvl.position.x;
    r.cached_visual_local[1] = cvl.position.y;
    r.cached_visual_local[2] = cvl.position.z;
    r.cached_visual_local[3] = cvl.rotation.w;
    r.cached_visual_local[4] = cvl.rotation.x;
    r.cached_visual_local[5] = cvl.rotation.y;
    r.cached_visual_local[6] = cvl.rotation.z;
    return r;
}

// Two boxes: one centered at (+5,0,0) (the +X target), one far off at (0,+8,3) so
// it is never on the swept fan -> a control "miss" geometry + the >=2 instances the
// batched LBVH build needs. Lidar sits at the env origin.
rt::BatchedSensorSceneDesc MakeSceneDesc() {
    rt::BatchedSensorSceneDesc d;
    d.scene.meshes.push_back(BuildBoxMesh(kBoxHalf));
    rt::Material mat;
    mat.albedo = {0.6f, 0.6f, 0.6f};
    d.scene.materials = {mat};
    d.scene.light.directional = true;
    d.scene.light.direction = {0.0f, 0.0f, -1.0f};

    math::Transform a;  // target box centered on +X at distance 5
    a.position = {5.0f, 0.0f, 0.0f};
    d.rows.push_back(BaseRow(a));
    d.blas_id.push_back(0u);
    d.material_id.push_back(0u);

    math::Transform b;  // off-axis box (never on the fan)
    b.position = {0.0f, 8.0f, 3.0f};
    d.rows.push_back(BaseRow(b));
    d.blas_id.push_back(0u);
    d.material_id.push_back(0u);
    return d;
}

// A lidar at world origin with a small az fan straddling +X (az=0) and one el row at
// el=0 (level). max_range deliberately short enough to test the clamp on the far box.
rt::LidarSensor MakeLidar(uint32_t az_count, uint32_t el_count, float max_range) {
    rt::LidarSensor s;
    s.origin[0] = 0.0f; s.origin[1] = 0.0f; s.origin[2] = 0.0f;
    s.rotation[0] = 1.0f; s.rotation[1] = 0.0f; s.rotation[2] = 0.0f; s.rotation[3] = 0.0f;
    s.az_count = az_count;
    s.el_count = el_count;
    s.az_min = -0.5f;  // radians; the swept band includes az=0 (+X)
    s.az_max = 0.5f;
    s.el_min = 0.0f;
    s.el_max = 0.0f;
    s.min_range = 0.1f;
    s.max_range = max_range;
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Gate 1 -- known distance, miss, clamp.
// ---------------------------------------------------------------------------
TEST(BatchedLidarRender, KnownDistanceMissAndClamp) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";

    // Odd az_count so a sample center lands exactly on az=0 (the +X ray). With
    // az in [-0.5, 0.5] and N samples, center i has az = -0.5 + (i+0.5)/N. For
    // N=11 the middle sample (i=5) has az = 0 exactly.
    const uint32_t kEnv = 4u, kAz = 11u, kEl = 1u;
    const float kMaxRange = 100.0f;

    rt::BatchedSensorSceneDevice scene = rt::BuildBatchedSensorScene(MakeSceneDesc());

    // Base pose = identity for every env (boxes at their cvl positions).
    std::vector<math::Transform> base(kEnv);
    math::Transform* d_base = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_base, kEnv * sizeof(math::Transform)));
    NK_CUDA_OK(cudaMemcpy(d_base, base.data(), kEnv * sizeof(math::Transform),
                          cudaMemcpyHostToDevice));
    phi::ScatterFkSource fk;
    fk.base_pose = d_base;
    fk.link_pose = nullptr;
    fk.body_pose = nullptr;
    fk.links_per_env = 0u;
    fk.bodies_per_env = 0u;

    // One lidar per env (S==1).
    std::vector<rt::LidarSensor> lid(kEnv, MakeLidar(kAz, kEl, kMaxRange));
    rt::LidarSensor* d_lid = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_lid, kEnv * sizeof(rt::LidarSensor)));
    NK_CUDA_OK(cudaMemcpy(d_lid, lid.data(), kEnv * sizeof(rt::LidarSensor),
                          cudaMemcpyHostToDevice));

    rt::RenderLidarsBatched(scene, fk, d_lid, kEnv, 1u, kAz, kEl);
    NK_CUDA_OK(cudaDeviceSynchronize());

    const uint64_t cells = static_cast<uint64_t>(kEnv) * kAz * kEl;
    std::vector<float> range(cells);
    NK_CUDA_OK(cudaMemcpy(range.data(), rt::SensorRangeDevice(scene),
                          range.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool finite = true;
    for (float r : range) if (!std::isfinite(r)) finite = false;
    EXPECT_TRUE(finite) << "range tensor has non-finite values";

    // The center az sample (i=5) of env 0 is the +X ray; the box front face is at
    // x = 5 - kBoxHalf = 4.5 from the origin.
    const float kExpected = 5.0f - kBoxHalf;
    const uint32_t kCenter = kAz / 2u;  // 5
    const float center_range = range[static_cast<size_t>(0) * kAz * kEl +
                                     static_cast<size_t>(kCenter) * kEl + 0u];
    EXPECT_NEAR(center_range, kExpected, 1.0e-3f)
        << "center +X ray range != box front-face distance";

    // Edge az samples (az = +/-~0.45 rad) miss the box (it subtends < 0.1 rad at 5 m)
    // -> max_range. Sample i=0 has az = -0.5 + 0.5/11 ~= -0.45.
    const float edge_range = range[0];  // env0, az0, el0
    EXPECT_FLOAT_EQ(edge_range, kMaxRange) << "edge ray should miss -> max_range";

    // Clamp: a short max_range pushes even the box hit to be clamped at max_range
    // when max_range < the true distance.
    const float kShort = 2.0f;  // < 4.5 (the true hit) -> clamp to 2.0
    std::vector<rt::LidarSensor> lid2(kEnv, MakeLidar(kAz, kEl, kShort));
    rt::LidarSensor* d_lid2 = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_lid2, kEnv * sizeof(rt::LidarSensor)));
    NK_CUDA_OK(cudaMemcpy(d_lid2, lid2.data(), kEnv * sizeof(rt::LidarSensor),
                          cudaMemcpyHostToDevice));
    rt::RenderLidarsBatched(scene, fk, d_lid2, kEnv, 1u, kAz, kEl);
    NK_CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> range2(cells);
    NK_CUDA_OK(cudaMemcpy(range2.data(), rt::SensorRangeDevice(scene),
                          range2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const float center_clamped = range2[static_cast<size_t>(kCenter) * kEl + 0u];
    EXPECT_FLOAT_EQ(center_clamped, kShort)
        << "a hit beyond max_range must clamp to max_range";

    std::printf("[diag] (1) center=%.4f (exp %.4f) edge=%.4f (max %.1f) clamped=%.4f\n",
                center_range, kExpected, edge_range, kMaxRange, center_clamped);

    cudaFree(d_base);
    cudaFree(d_lid);
    cudaFree(d_lid2);
}

// ---------------------------------------------------------------------------
// Gate 2 -- per-env tiles byte-identical (same scene + mount across N TLASes).
// ---------------------------------------------------------------------------
TEST(BatchedLidarRender, PerEnvTilesByteIdentical) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";
    const uint32_t kEnv = 6u, kAz = 9u, kEl = 4u;

    rt::BatchedSensorSceneDevice scene = rt::BuildBatchedSensorScene(MakeSceneDesc());
    std::vector<math::Transform> base(kEnv);
    math::Transform* d_base = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_base, kEnv * sizeof(math::Transform)));
    NK_CUDA_OK(cudaMemcpy(d_base, base.data(), kEnv * sizeof(math::Transform),
                          cudaMemcpyHostToDevice));
    phi::ScatterFkSource fk;
    fk.base_pose = d_base; fk.link_pose = nullptr; fk.body_pose = nullptr;
    fk.links_per_env = 0u; fk.bodies_per_env = 0u;

    rt::LidarSensor proto = MakeLidar(kAz, kEl, 50.0f);
    proto.el_min = -0.2f; proto.el_max = 0.3f;
    std::vector<rt::LidarSensor> lid(kEnv, proto);
    rt::LidarSensor* d_lid = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_lid, kEnv * sizeof(rt::LidarSensor)));
    NK_CUDA_OK(cudaMemcpy(d_lid, lid.data(), kEnv * sizeof(rt::LidarSensor),
                          cudaMemcpyHostToDevice));

    rt::RenderLidarsBatched(scene, fk, d_lid, kEnv, 1u, kAz, kEl);
    NK_CUDA_OK(cudaDeviceSynchronize());

    const size_t per = static_cast<size_t>(kAz) * kEl;
    std::vector<float> range(per * kEnv);
    NK_CUDA_OK(cudaMemcpy(range.data(), rt::SensorRangeDevice(scene),
                          range.size() * sizeof(float), cudaMemcpyDeviceToHost));

    size_t mismatches = 0u;
    for (uint32_t e = 1u; e < kEnv; ++e) {
        if (std::memcmp(range.data() + static_cast<size_t>(e) * per, range.data(),
                        per * sizeof(float)) != 0) {
            ++mismatches;
        }
    }
    // The +X box must be hit by some ray (not an all-miss fan).
    bool any_hit = false;
    for (size_t i = 0; i < per; ++i) if (range[i] < 49.0f) any_hit = true;
    EXPECT_TRUE(any_hit) << "lidar fan never hit the box (scene/convention bug)";
    EXPECT_EQ(mismatches, 0u) << "per-env range tiles diverge for identical state";
    std::printf("[diag] (2) cross-env tile mismatches = %zu / %u | any_hit=%d\n",
                mismatches, kEnv - 1u, (int)any_hit);

    cudaFree(d_base);
    cudaFree(d_lid);
}

// ---------------------------------------------------------------------------
// Gate 3 -- the MOUNTED path (the shipped surface) matches the device path.
// ---------------------------------------------------------------------------
TEST(BatchedLidarRender, MountedPathMatchesDevicePath) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";
    const uint32_t kEnv = 4u, kAz = 11u, kEl = 1u;

    // Device-pointer reference.
    rt::BatchedSensorSceneDevice ref = rt::BuildBatchedSensorScene(MakeSceneDesc());
    std::vector<math::Transform> base(kEnv);
    math::Transform* d_base = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_base, kEnv * sizeof(math::Transform)));
    NK_CUDA_OK(cudaMemcpy(d_base, base.data(), kEnv * sizeof(math::Transform),
                          cudaMemcpyHostToDevice));
    phi::ScatterFkSource fk;
    fk.base_pose = d_base; fk.link_pose = nullptr; fk.body_pose = nullptr;
    fk.links_per_env = 0u; fk.bodies_per_env = 0u;

    std::vector<rt::LidarSensor> lid(kEnv, MakeLidar(kAz, kEl, 100.0f));
    rt::LidarSensor* d_lid = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_lid, kEnv * sizeof(rt::LidarSensor)));
    NK_CUDA_OK(cudaMemcpy(d_lid, lid.data(), kEnv * sizeof(rt::LidarSensor),
                          cudaMemcpyHostToDevice));
    rt::RenderLidarsBatched(ref, fk, d_lid, kEnv, 1u, kAz, kEl);
    NK_CUDA_OK(cudaDeviceSynchronize());
    const uint64_t cells = static_cast<uint64_t>(kEnv) * kAz * kEl;
    std::vector<float> ref_range(cells);
    NK_CUDA_OK(cudaMemcpy(ref_range.data(), rt::SensorRangeDevice(ref),
                          ref_range.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Mounted path: a Base-mounted lidar SensorDesc (identity offset) with the SAME
    // pattern. SetLidarMounts + RenderLidarsMounted scatters the per-env poses.
    rt::BatchedSensorSceneDevice mnt = rt::BuildBatchedSensorScene(MakeSceneDesc());
    scene::SensorDesc sd;
    sd.type = scene::SensorType::Lidar;
    sd.mount = scene::MountFrame::Base;
    sd.mount_index = 0u;
    sd.lidar.az_count = static_cast<uint16_t>(kAz);
    sd.lidar.el_count = static_cast<uint16_t>(kEl);
    sd.lidar.az_min = -0.5f; sd.lidar.az_max = 0.5f;
    sd.lidar.el_min = 0.0f;  sd.lidar.el_max = 0.0f;
    sd.lidar.min_range = 0.1f; sd.lidar.max_range = 100.0f;
    rt::SetLidarMounts(mnt, {sd});
    rt::RenderLidarsMounted(mnt, fk, kEnv);
    NK_CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> mnt_range(cells);
    NK_CUDA_OK(cudaMemcpy(mnt_range.data(), rt::SensorRangeDevice(mnt),
                          mnt_range.size() * sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_EQ(rt::LidarAzCount(mnt), kAz);
    ASSERT_EQ(rt::LidarElCount(mnt), kEl);
    ASSERT_EQ(rt::LidarsPerEnv(mnt), 1u);
    size_t mismatches = 0u;
    for (size_t i = 0; i < cells; ++i) if (ref_range[i] != mnt_range[i]) ++mismatches;
    EXPECT_EQ(mismatches, 0u)
        << "mounted lidar ranges differ from the device-pointer ranges";
    std::printf("[diag] (3) mounted vs device mismatches = %zu / %llu\n", mismatches,
                (unsigned long long)cells);

    cudaFree(d_base);
    cudaFree(d_lid);
}
