// ---------------------------------------------------------------------------
// Per-env camera-mount gate: ScatterEnvCameras builds each env's PinholeCamera
// on device from cam_world = fk(pose) * local_offset (USD/Isaac axes: -Z forward,
// +Y up). The device camera must be BYTE-IDENTICAL (memcmp) to the host camera
// composed from the SAME pose * local_offset via math::Transform::operator* +
// the shared BuildPinholeBasis, over >=3 envs and all 3 mount kinds (Link/Body/
// Base) + a Static row. A second test drives the batched sensor trace with these
// scatter-built cameras and checks per-env tiles match an E=1 render with the
// same camera. Compiled -ffp-contract=off + --fmad=false so host==device.
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
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

Transform MakePose(float px, float py, float pz, float ax, float ay, float az,
                   float angle) {
    return Transform{Vec3{px, py, pz}, Quat::FromAxisAngle(Vec3{ax, ay, az}, angle)};
}

// The HOST reference camera: cam_world = fk * local_offset, then forward = R*(-Z),
// up = R*(+Y), then the shared BuildPinholeBasis -- the exact chain the kernel runs.
PinholeCamera HostMountCamera(const Transform* fk_pose, const Transform& local_offset,
                              float fov_y, uint32_t w, uint32_t h) {
    const Transform cam_world =
        (fk_pose == nullptr) ? local_offset : (*fk_pose) * local_offset;
    const Vec3 forward = cam_world.rotation.Rotate(Vec3{0.0f, 0.0f, -1.0f});
    const Vec3 up = cam_world.rotation.Rotate(Vec3{0.0f, 1.0f, 0.0f});
    return rt::BuildPinholeBasis(cam_world.position, forward, up, fov_y, w, h);
}

void SetOffset(SensorMountRow& r, const Transform& t) {
    r.local_offset[0] = t.position.x;
    r.local_offset[1] = t.position.y;
    r.local_offset[2] = t.position.z;
    r.local_offset[3] = t.rotation.w;
    r.local_offset[4] = t.rotation.x;
    r.local_offset[5] = t.rotation.y;
    r.local_offset[6] = t.rotation.z;
}

constexpr uint32_t kEnvCount = 4u;
constexpr uint32_t kLinksPerEnv = 5u;
constexpr uint32_t kBodiesPerEnv = 3u;

}  // namespace

// Device cameras == host BuildPinholeBasis(fk*local_offset) byte-for-byte across
// Link/Body/Base/Static mounts and >=3 envs.
TEST(SensorCamera, DeviceMatchesHostMountCompose) {
    std::vector<Transform> link_pose(static_cast<size_t>(kEnvCount) * kLinksPerEnv);
    std::vector<Transform> body_pose(static_cast<size_t>(kEnvCount) * kBodiesPerEnv);
    std::vector<Transform> base_pose(kEnvCount);
    for (uint32_t e = 0; e < kEnvCount; ++e) {
        const float s = 1.0f + 0.41f * static_cast<float>(e);
        for (uint32_t l = 0; l < kLinksPerEnv; ++l) {
            link_pose[e * kLinksPerEnv + l] =
                MakePose(0.5f * l + s, -0.3f * l, 0.2f * e + 0.1f * l, 0.2f, 0.7f,
                         0.3f, 0.4f + 0.13f * l + 0.05f * e);
        }
        for (uint32_t b = 0; b < kBodiesPerEnv; ++b) {
            body_pose[e * kBodiesPerEnv + b] =
                MakePose(-0.4f * b - s, 0.6f * b, 0.15f * e, 0.9f, 0.1f, 0.4f,
                         -0.3f + 0.21f * b + 0.07f * e);
        }
        base_pose[e] = MakePose(2.0f + e, 0.5f * e, -0.7f, 0.0f, 0.0f, 1.0f,
                                0.25f + 0.11f * e);
    }

    Transform* d_link = nullptr;
    Transform* d_body = nullptr;
    Transform* d_base = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_link, link_pose.size() * sizeof(Transform)));
    NK_CUDA_OK(cudaMalloc(&d_body, body_pose.size() * sizeof(Transform)));
    NK_CUDA_OK(cudaMalloc(&d_base, base_pose.size() * sizeof(Transform)));
    NK_CUDA_OK(cudaMemcpy(d_link, link_pose.data(),
                          link_pose.size() * sizeof(Transform), cudaMemcpyHostToDevice));
    NK_CUDA_OK(cudaMemcpy(d_body, body_pose.data(),
                          body_pose.size() * sizeof(Transform), cudaMemcpyHostToDevice));
    NK_CUDA_OK(cudaMemcpy(d_base, base_pose.data(),
                          base_pose.size() * sizeof(Transform), cudaMemcpyHostToDevice));

    // Mount table (env-invariant): all 3 frames + a Static row, distinct offsets +
    // intrinsics so the camera byte compare covers every field.
    std::vector<SensorMountRow> mounts;
    auto add = [&](uint32_t kind, uint32_t row, const Transform& off, float fov,
                   uint32_t w, uint32_t h) {
        SensorMountRow r;
        r.kind = kind;
        r.row = row;
        SetOffset(r, off);
        r.fov_y = fov;
        r.width = w;
        r.height = h;
        mounts.push_back(r);
    };
    add(1u, 0u, MakePose(0.05f, 0.0f, 0.10f, 1.0f, 0.0f, 0.0f, 0.10f), 0.80f, 64u, 48u);
    add(1u, 4u, MakePose(0.0f, 0.12f, -0.03f, 0.0f, 1.0f, 0.0f, 0.20f), 1.20f, 32u, 32u);
    add(2u, 0u, MakePose(-0.07f, 0.0f, 0.09f, 0.3f, 0.3f, 0.9f, 0.30f), 0.60f, 80u, 60u);
    add(2u, 2u, MakePose(0.0f, -0.04f, 0.0f, 1.0f, 1.0f, 0.0f, 0.40f), 1.05f, 48u, 48u);
    add(3u, 0u, MakePose(0.11f, 0.02f, 0.20f, 0.0f, 0.0f, 1.0f, 0.50f), 0.90f, 64u, 64u);
    add(0u, 0u, MakePose(1.30f, -0.5f, 0.20f, 0.5f, 0.5f, 0.7f, 0.60f), 0.70f, 100u, 50u);
    const uint32_t kSensors = static_cast<uint32_t>(mounts.size());

    SensorMountRow* d_mounts = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_mounts, kSensors * sizeof(SensorMountRow)));
    NK_CUDA_OK(cudaMemcpy(d_mounts, mounts.data(), kSensors * sizeof(SensorMountRow),
                          cudaMemcpyHostToDevice));

    const size_t total = static_cast<size_t>(kEnvCount) * kSensors;
    PinholeCamera* d_cams = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_cams, total * sizeof(PinholeCamera)));
    NK_CUDA_OK(cudaMemset(d_cams, 0, total * sizeof(PinholeCamera)));

    ScatterFkSource fk;
    fk.link_pose = d_link;
    fk.body_pose = d_body;
    fk.base_pose = d_base;
    fk.env_index = 0u;
    fk.links_per_env = kLinksPerEnv;
    fk.bodies_per_env = kBodiesPerEnv;

    rt::ScatterEnvCameras(/*stream=*/nullptr, fk, d_mounts, kEnvCount, kSensors, d_cams);
    NK_CUDA_OK(cudaDeviceSynchronize());

    std::vector<PinholeCamera> got(total);
    NK_CUDA_OK(cudaMemcpy(got.data(), d_cams, total * sizeof(PinholeCamera),
                          cudaMemcpyDeviceToHost));

    // Host reference: same fk * local_offset compose + the shared basis.
    std::vector<PinholeCamera> ref(total);
    std::memset(ref.data(), 0, total * sizeof(PinholeCamera));
    for (uint32_t e = 0; e < kEnvCount; ++e) {
        for (uint32_t s = 0; s < kSensors; ++s) {
            const SensorMountRow& m = mounts[s];
            const Transform* pose = nullptr;
            if (m.kind == 1u && m.row < kLinksPerEnv) {
                pose = &link_pose[e * kLinksPerEnv + m.row];
            } else if (m.kind == 2u && m.row < kBodiesPerEnv) {
                pose = &body_pose[e * kBodiesPerEnv + m.row];
            } else if (m.kind == 3u) {
                pose = &base_pose[e];
            }
            Transform off;
            off.position = Vec3{m.local_offset[0], m.local_offset[1], m.local_offset[2]};
            off.rotation = Quat{m.local_offset[3], m.local_offset[4], m.local_offset[5],
                                m.local_offset[6]};
            ref[static_cast<size_t>(e) * kSensors + s] =
                HostMountCamera(pose, off, m.fov_y, m.width, m.height);
        }
    }

    EXPECT_EQ(std::memcmp(got.data(), ref.data(), total * sizeof(PinholeCamera)), 0)
        << "device camera table != host fk*local_offset reference (byte-diff)";

    cudaFree(d_link);
    cudaFree(d_body);
    cudaFree(d_base);
    cudaFree(d_mounts);
    cudaFree(d_cams);
}

// --- Integration: drive the batched sensor trace with scatter-built cameras. ---
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kInstancesPerEnv = 12u;
constexpr float kEnvPitch = 6.0f;
constexpr float kLinkSpread = 0.35f;
constexpr float kLinkHalf = 0.12f;
constexpr uint32_t kRes = 48u;

rt::BlasMesh BuildBoxMesh(float half) {
    const Vec3 c[8] = {
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

uint32_t GridSide(uint32_t n) {
    uint32_t s = 1u;
    while (s * s < n) ++s;
    return s;
}

Transform LinkLocal(uint32_t k) {
    const uint32_t side = GridSide(kInstancesPerEnv);
    const uint32_t lx = k % side;
    const uint32_t ly = k / side;
    const float ox = (static_cast<float>(lx) - 0.5f * static_cast<float>(side - 1u)) * kLinkSpread;
    const float oy = (static_cast<float>(ly) - 0.5f * static_cast<float>(side - 1u)) * kLinkSpread;
    const float oz = kLinkHalf + 0.02f * static_cast<float>(k % 5u);
    const float ang = 0.1f + 0.05f * static_cast<float>(k);
    Transform t;
    t.position = {ox, oy, oz};
    t.rotation = Quat::FromAxisAngle(Vec3{0.2f, 1.0f, 0.3f}, ang);
    return t;
}

Transform EnvBasePose(uint32_t e, uint32_t n) {
    const uint32_t side = GridSide(n);
    const uint32_t ix = e % side;
    const uint32_t iy = e / side;
    const float cx = (static_cast<float>(ix) - 0.5f * static_cast<float>(side - 1u)) * kEnvPitch;
    const float cy = (static_cast<float>(iy) - 0.5f * static_cast<float>(side - 1u)) * kEnvPitch;
    Transform t;
    t.position = {cx, cy, 0.0f};
    t.rotation = Quat::FromAxisAngle(Vec3{0.0f, 0.0f, 1.0f}, 0.07f * static_cast<float>(e));
    return t;
}

rt::Material LinkMaterial() {
    rt::Material mat;
    mat.albedo = {0.65f, 0.55f, 0.4f};
    mat.metallic = 0.0f;
    mat.roughness = 0.6f;
    return mat;
}

phi::InstanceScatterRow BaseRow(const Transform& cvl) {
    phi::InstanceScatterRow r;
    r.kind = 3u;
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

rt::BatchedSensorSceneDesc MakeSceneDesc() {
    rt::BatchedSensorSceneDesc d;
    d.scene.meshes.push_back(BuildBoxMesh(kLinkHalf));
    d.scene.materials = {LinkMaterial()};
    d.scene.light.directional = true;
    d.scene.light.direction = {0.3f, 0.4f, -0.85f};
    d.scene.ambient.color = {0.05f, 0.05f, 0.05f};
    for (uint32_t k = 0; k < kInstancesPerEnv; ++k) {
        d.rows.push_back(BaseRow(LinkLocal(k)));
        d.blas_id.push_back(0u);
        d.material_id.push_back(0u);
    }
    return d;
}

// A Base-mounted camera 3 m above each env's origin, identity rotation: local -Z
// looks straight DOWN at the cluster -> guaranteed geometry per env.
SensorMountRow OverheadMount() {
    SensorMountRow r;
    r.kind = 3u;  // Base
    r.row = 0u;
    r.local_offset[0] = 0.0f;
    r.local_offset[1] = 0.0f;
    r.local_offset[2] = 3.0f;
    r.local_offset[3] = 1.0f;  // identity quaternion
    r.fov_y = 0.6f * kPi;
    r.width = kRes;
    r.height = kRes;
    return r;
}

}  // namespace

// The batched trace driven by ScatterEnvCameras renders non-empty geometry, and
// each per-env tile equals an E=1 render driven by that env's scatter-built camera.
TEST(SensorCamera, BatchedRenderWithMountedCameras) {
    ASSERT_NE(phi::ActiveBackend(), nullptr) << "no CUDA backend";

    const uint32_t kEnv = 4u;
    const size_t pix = static_cast<size_t>(kRes) * kRes;

    // Env-major Base poses + the ScatterFkSource over them.
    std::vector<Transform> base(kEnv);
    for (uint32_t e = 0; e < kEnv; ++e) base[e] = EnvBasePose(e, kEnv);
    Transform* d_base = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_base, kEnv * sizeof(Transform)));
    NK_CUDA_OK(cudaMemcpy(d_base, base.data(), kEnv * sizeof(Transform),
                          cudaMemcpyHostToDevice));
    ScatterFkSource fk;
    fk.base_pose = d_base;
    fk.link_pose = nullptr;
    fk.body_pose = nullptr;
    fk.links_per_env = 0u;
    fk.bodies_per_env = 0u;

    // ONE overhead camera per env, built ON DEVICE from the base poses.
    std::vector<SensorMountRow> mounts = {OverheadMount()};
    SensorMountRow* d_mounts = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_mounts, mounts.size() * sizeof(SensorMountRow)));
    NK_CUDA_OK(cudaMemcpy(d_mounts, mounts.data(), mounts.size() * sizeof(SensorMountRow),
                          cudaMemcpyHostToDevice));
    PinholeCamera* d_cams = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_cams, kEnv * sizeof(PinholeCamera)));
    rt::ScatterEnvCameras(/*stream=*/nullptr, fk, d_mounts, kEnv, /*sensors=*/1u, d_cams);
    NK_CUDA_OK(cudaDeviceSynchronize());

    // Batched render driven by the scatter-built cameras.
    rt::BatchedSensorSceneDevice scene = rt::BuildBatchedSensorScene(MakeSceneDesc());
    rt::RenderSensorsBatched(scene, fk, d_cams, kEnv, 1u, kRes, kRes);
    NK_CUDA_OK(cudaDeviceSynchronize());

    const uint64_t rays = static_cast<uint64_t>(kEnv) * pix;
    std::vector<float> color(rays * 3u);
    std::vector<uint32_t> prim(rays);
    NK_CUDA_OK(cudaMemcpy(color.data(), rt::SensorColorDevice(scene),
                          color.size() * sizeof(float), cudaMemcpyDeviceToHost));
    NK_CUDA_OK(cudaMemcpy(prim.data(), rt::SensorPrimDevice(scene),
                          prim.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Every env's mounted camera must see the cluster (not an all-miss image).
    for (uint32_t e = 0; e < kEnv; ++e) {
        bool hit = false;
        for (size_t p = 0; p < pix; ++p) {
            if (prim[static_cast<size_t>(e) * pix + p] != 0xFFFFFFFFu) { hit = true; break; }
        }
        EXPECT_TRUE(hit) << "env " << e << " mounted-camera render is all-miss";
    }

    // The scatter cameras (downloaded) drive an E=1 render that matches each tile.
    std::vector<PinholeCamera> cams(kEnv);
    NK_CUDA_OK(cudaMemcpy(cams.data(), d_cams, kEnv * sizeof(PinholeCamera),
                          cudaMemcpyDeviceToHost));
    for (uint32_t e = 0; e < kEnv; ++e) {
        rt::BatchedSensorSceneDevice solo = rt::BuildBatchedSensorScene(MakeSceneDesc());
        Transform sb = base[e];
        Transform* d_sb = nullptr;
        NK_CUDA_OK(cudaMalloc(&d_sb, sizeof(Transform)));
        NK_CUDA_OK(cudaMemcpy(d_sb, &sb, sizeof(Transform), cudaMemcpyHostToDevice));
        ScatterFkSource sfk;
        sfk.base_pose = d_sb;
        sfk.links_per_env = 0u;
        sfk.bodies_per_env = 0u;

        PinholeCamera one = cams[e];
        PinholeCamera* d_one = nullptr;
        NK_CUDA_OK(cudaMalloc(&d_one, sizeof(PinholeCamera)));
        NK_CUDA_OK(cudaMemcpy(d_one, &one, sizeof(PinholeCamera), cudaMemcpyHostToDevice));

        rt::RenderSensorsBatched(solo, sfk, d_one, 1u, 1u, kRes, kRes);
        NK_CUDA_OK(cudaDeviceSynchronize());

        std::vector<float> rc(pix * 3u);
        std::vector<uint32_t> rp(pix);
        NK_CUDA_OK(cudaMemcpy(rc.data(), rt::SensorColorDevice(solo),
                              rc.size() * sizeof(float), cudaMemcpyDeviceToHost));
        NK_CUDA_OK(cudaMemcpy(rp.data(), rt::SensorPrimDevice(solo),
                              rp.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        const size_t off3 = static_cast<size_t>(e) * pix * 3u;
        const size_t off1 = static_cast<size_t>(e) * pix;
        EXPECT_EQ(std::memcmp(rc.data(), color.data() + off3, rc.size() * sizeof(float)), 0)
            << "env " << e << " color tile != E=1 render with the mounted camera";
        EXPECT_EQ(std::memcmp(rp.data(), prim.data() + off1, rp.size() * sizeof(uint32_t)), 0)
            << "env " << e << " prim tile != E=1 render with the mounted camera";

        cudaFree(d_sb);
        cudaFree(d_one);
    }

    cudaFree(d_base);
    cudaFree(d_mounts);
    cudaFree(d_cams);
}
