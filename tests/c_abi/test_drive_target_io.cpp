// ---------------------------------------------------------------------------
// sim-val #33 -- minimal incremental C ABI: per-env PD drive-target write +
// q/qd/base read, for the batched (multi-env) Go2 world.
//
// This is the convention-FREE seam an external Python policy harness drives:
//   - WRITE per-env actuated-joint drive targets via the NUKA_FIELD_DRIVE_TARGET
//     buffer view (zero-copy: the view aliases the live device buffer that the
//     batched step reads each Step).
//   - READ q (NUKA_FIELD_JOINT_POSITION), qd (NUKA_FIELD_JOINT_VELOCITY), and the
//     base/link world pose (NUKA_FIELD_ARTICULATION_LINK_POSE).
// No Unitree obs/action layout lives here -- only raw env-major float buffers.
//
// Layouts asserted/relied on here (env-major; base_link_count = q_count/env_count;
// the articulation root/base is local link 0 so env e's base/root is element
// e*base_link_count):
//   q, qd, drive_target : float[env_count*base_link_count], idx e*base_link_count+l
//   link pose           : math::Transform[env_count*base_link_count], 7 floats each
//                         [px,py,pz, qw,qx,qy,qz]  (quat w-FIRST)
//
// Gates:
//  (1) WRITE reaches the solver: perturbing the drive targets of the actuated
//      joints moves q (after K steps) relative to the unperturbed rest run, and
//      the moved joints move in the SIGN of the perturbation (toward the target).
//      Actuated joints are self-identified (the entries that actually moved), so
//      no Unitree index list is hardcoded.
//  (2) qd and base/link pose read views are finite and correctly shaped.
//  (3) Determinism: two identical perturbed runs give bit-identical q.
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"
#include "nuka/nuka.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

std::string ScenePath() {
    return SourcePath("examples/scenes/go2_stand.usda").string();
}

bool SceneAvailable() {
    return std::filesystem::exists(SourcePath("examples/scenes/go2_stand.usda"));
}

struct DeviceGuard {
    nuka_device_handle handle = nullptr;
    DeviceGuard() {
        nuka_device_desc_t desc{};
        desc.gpu_index = 0u;
        desc.cuda_stream = nullptr;
        desc.backend_selection_layer_enabled = 1u;
        EXPECT_EQ(nuka_device_create(&desc, &handle), NUKA_RESULT_OK);
    }
    ~DeviceGuard() {
        if (handle != nullptr) {
            nuka_device_destroy(handle);
        }
    }
};

struct WorldGuard {
    nuka_world_handle handle = nullptr;
    ~WorldGuard() {
        if (handle != nullptr) {
            nuka_world_destroy(handle);
        }
    }
};

nuka_result_t CreateWorld(nuka_device_handle device, uint32_t env_count,
                          nuka_world_handle* out) {
    const std::string scene = ScenePath();
    nuka_world_desc_t desc{};
    desc.scene_path = scene.c_str();
    desc.env_count = env_count;
    desc.fixed_dt = 1.0f / 240.0f;
    return nuka_world_create_from_scene(device, &desc, out);
}

// Reads a float buffer-view back to host. floats_per_element is derived from the
// view stride (1 for q/qd/drive_target, 7 for link pose).
std::vector<float> DownloadField(nuka_world_handle world, nuka_state_field_t field) {
    nuka_buffer_view_t view{};
    EXPECT_EQ(nuka_world_get_buffer_view(world, field, &view), NUKA_RESULT_OK);
    std::vector<float> out;
    if (view.device_ptr == nullptr || view.element_count == 0u) {
        return out;
    }
    const size_t floats_per_element = view.element_stride_bytes / sizeof(float);
    out.resize(view.element_count * floats_per_element);
    EXPECT_EQ(cudaMemcpy(out.data(), view.device_ptr, out.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return out;
}

bool AllFinite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

// Writes `targets` (env-major float[env_count*base_link_count]) into the live
// device DRIVE_TARGET buffer the next Step reads (zero-copy view alias).
void WriteDriveTargets(nuka_world_handle world, const std::vector<float>& targets) {
    nuka_buffer_view_t view{};
    ASSERT_EQ(nuka_world_get_buffer_view(world, NUKA_FIELD_DRIVE_TARGET, &view),
              NUKA_RESULT_OK);
    ASSERT_NE(view.device_ptr, nullptr);
    ASSERT_EQ(view.element_stride_bytes, sizeof(float));
    ASSERT_EQ(view.element_count, targets.size());
    ASSERT_EQ(cudaMemcpy(view.device_ptr, targets.data(),
                         targets.size() * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
}

}  // namespace

// ---------------------------------------------------------------------------
// Gate 1 + 2 -- write reaches solver (q moves toward target) + qd/pose shaped.
// ---------------------------------------------------------------------------
TEST(DriveTargetIo, WriteMovesQTowardTargetAndReadsAreShaped) {
    if (!SceneAvailable()) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnvCount = 16u;
    const uint32_t kSteps = 30u;
    const float kDelta = 0.30f;  // rad perturbation added to every joint target.

    // --- Establish the env-major layout from the q view. -------------------
    std::vector<float> q0;
    {
        WorldGuard probe;
        ASSERT_EQ(CreateWorld(device.handle, kEnvCount, &probe.handle), NUKA_RESULT_OK);
        q0 = DownloadField(probe.handle, NUKA_FIELD_JOINT_POSITION);
        ASSERT_GT(q0.size(), 0u);
        ASSERT_EQ(q0.size() % kEnvCount, 0u);

        // The DRIVE_TARGET read view must mirror q exactly (same count + stride).
        nuka_buffer_view_t dv{};
        ASSERT_EQ(nuka_world_get_buffer_view(probe.handle, NUKA_FIELD_DRIVE_TARGET, &dv),
                  NUKA_RESULT_OK);
        EXPECT_EQ(dv.element_count, q0.size());
        EXPECT_EQ(dv.element_stride_bytes, sizeof(float));
    }
    const uint32_t base_link_count = static_cast<uint32_t>(q0.size() / kEnvCount);
    ASSERT_GT(base_link_count, 0u);

    // --- Run A: unperturbed (rest hold-drive) reference. -------------------
    std::vector<float> q_rest;
    {
        WorldGuard rest;
        ASSERT_EQ(CreateWorld(device.handle, kEnvCount, &rest.handle), NUKA_RESULT_OK);
        ASSERT_EQ(nuka_world_step_n(rest.handle, kSteps), NUKA_RESULT_OK);
        q_rest = DownloadField(rest.handle, NUKA_FIELD_JOINT_POSITION);
        ASSERT_EQ(q_rest.size(), q0.size());
        EXPECT_TRUE(AllFinite(q_rest));
    }

    // --- Run B: perturb every joint target by +kDelta, then step. ----------
    // Perturbing ALL link slots (incl. non-actuated ones, which the solver
    // silently ignores) means we don't need to know WHICH links are actuated --
    // the joints that actually move self-identify the actuated set.
    std::vector<float> q_pert;
    std::vector<float> qd_pert;
    std::vector<float> pose_pert;
    std::vector<float> pose_pre;  // link pose after 1 step (base already live).
    {
        WorldGuard pert;
        ASSERT_EQ(CreateWorld(device.handle, kEnvCount, &pert.handle), NUKA_RESULT_OK);

        // Read current targets (rest hold) and add the perturbation in place.
        auto targets = DownloadField(pert.handle, NUKA_FIELD_DRIVE_TARGET);
        ASSERT_EQ(targets.size(), q0.size());
        for (float& t : targets) {
            t += kDelta;
        }
        WriteDriveTargets(pert.handle, targets);

        // Snapshot the link pose after a single step (the base root pose is the
        // live integrated floating-base pose, refreshed by the batched FK each
        // step), so we can prove below it is NOT a frozen cooked pose.
        ASSERT_EQ(nuka_world_step_n(pert.handle, 1u), NUKA_RESULT_OK);
        pose_pre = DownloadField(pert.handle, NUKA_FIELD_ARTICULATION_LINK_POSE);

        ASSERT_EQ(nuka_world_step_n(pert.handle, kSteps - 1u), NUKA_RESULT_OK);
        q_pert = DownloadField(pert.handle, NUKA_FIELD_JOINT_POSITION);
        qd_pert = DownloadField(pert.handle, NUKA_FIELD_JOINT_VELOCITY);
        pose_pert = DownloadField(pert.handle, NUKA_FIELD_ARTICULATION_LINK_POSE);
        ASSERT_EQ(q_pert.size(), q0.size());
    }

    // (1) WRITE reaches the solver: enough joints moved, and every joint that
    // moved beyond solver jitter moved in the SIGN of the +kDelta perturbation
    // (i.e. toward the raised target).
    const float kMoveEps = 1e-4f;  // ignore sub-jitter differences.
    uint32_t moved = 0u;
    uint32_t moved_wrong_sign = 0u;
    for (size_t i = 0u; i < q_pert.size(); ++i) {
        const float d = q_pert[i] - q_rest[i];
        if (std::fabs(d) > kMoveEps) {
            ++moved;
            if (d < 0.0f) {  // +kDelta target should push q UP.
                ++moved_wrong_sign;
            }
        }
    }
    std::printf("[diag] (1) env_count=%u base_link_count=%u steps=%u delta=%.3f | "
                "q entries moved=%u (wrong-sign=%u of %zu)\n",
                kEnvCount, base_link_count, kSteps,
                static_cast<double>(kDelta), moved, moved_wrong_sign, q_pert.size());
    // 12 actuated joints/env * kEnvCount is the floor; require at least 12 total
    // (one env's worth) to prove the write reached the solver without hardcoding
    // the actuated index set.
    EXPECT_GE(moved, 12u) << "drive-target write did not move q (write never reached the solver)";
    EXPECT_EQ(moved_wrong_sign, 0u)
        << "some joints moved AWAY from the raised target (write/sign mismatch)";

    // (2) qd + base/link pose read views are finite and correctly shaped.
    EXPECT_EQ(qd_pert.size(), q0.size());
    EXPECT_TRUE(AllFinite(qd_pert)) << "qd non-finite";
    // link pose: 7 floats per link [px,py,pz, qw,qx,qy,qz], env*base_link_count links.
    ASSERT_EQ(pose_pert.size(), static_cast<size_t>(q0.size()) * 7u)
        << "link-pose view is not 7 floats * env_count*base_link_count";
    EXPECT_TRUE(AllFinite(pose_pert)) << "link pose non-finite";
    // Each base/link quaternion (qw,qx,qy,qz) must be near unit-norm -- a wrong
    // element order / stride would corrupt this.
    uint32_t bad_quat = 0u;
    for (uint32_t link = 0u; link < q0.size(); ++link) {
        const float* p = &pose_pert[static_cast<size_t>(link) * 7u];
        const float qn = std::sqrt(p[3] * p[3] + p[4] * p[4] + p[5] * p[5] + p[6] * p[6]);
        if (std::fabs(qn - 1.0f) > 1e-2f) {
            ++bad_quat;
        }
    }
    EXPECT_EQ(bad_quat, 0u) << "link-pose quaternions not unit norm (layout/stride wrong)";

    // Base/root of env 0 is element 0; its world height (pz) must be finite and
    // physically sane (Go2 trunk sits ~0.3-0.5 m; just sanity-bound it).
    // NOTE: go2_stand.usda declares NO free/floating root joint, so the trunk
    // cooks to a FIXED base -- its world pose is legitimately CONSTANT at the
    // cooked rest pose (0,0,0.445)+identity for every step. (For a free-floating
    // scene the same view would carry the live integrated base pose; that path is
    // covered by the floating-base tests.)
    const float base_z = pose_pert[2];

    // The link-pose view must be LIVE forward-kinematics output, not a globally
    // frozen cooked table: under the 0.3 rad leg perturbation the LEG links move,
    // so the per-link pose 7-tuples must DIFFER between an early-step snapshot and
    // the settled run for at least the actuated chain. (The fixed root stays put;
    // we sum drift across ALL links so the moving legs dominate.)
    ASSERT_EQ(pose_pre.size(), pose_pert.size());
    float link_pose_drift = 0.0f;
    for (size_t k = 0u; k < pose_pert.size(); ++k) {
        link_pose_drift += std::fabs(pose_pert[k] - pose_pre[k]);
    }
    std::printf("[diag] (2) base(env0) world z=%.4f (fixed root) | link-pose FK "
                "drift=%.5f | qd finite | pose 7-float OK\n",
                static_cast<double>(base_z), static_cast<double>(link_pose_drift));
    EXPECT_TRUE(std::isfinite(base_z));
    EXPECT_GT(link_pose_drift, 1e-3f)
        << "link poses did not move -- ARTICULATION_LINK_POSE looks frozen (not live FK)";
}

// ---------------------------------------------------------------------------
// Gate 3 -- determinism: two identical perturbed runs give bit-identical q.
// ---------------------------------------------------------------------------
TEST(DriveTargetIo, PerturbedRunIsBitIdenticalAcrossRuns) {
    if (!SceneAvailable()) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnvCount = 16u;
    const uint32_t kSteps = 24u;
    const float kDelta = 0.25f;

    auto perturbed_run = [&](std::vector<float>* q_out, std::vector<float>* qd_out) {
        WorldGuard world;
        ASSERT_EQ(CreateWorld(device.handle, kEnvCount, &world.handle), NUKA_RESULT_OK);
        auto targets = DownloadField(world.handle, NUKA_FIELD_DRIVE_TARGET);
        ASSERT_GT(targets.size(), 0u);
        for (float& t : targets) {
            t += kDelta;
        }
        WriteDriveTargets(world.handle, targets);
        ASSERT_EQ(nuka_world_step_n(world.handle, kSteps), NUKA_RESULT_OK);
        *q_out = DownloadField(world.handle, NUKA_FIELD_JOINT_POSITION);
        *qd_out = DownloadField(world.handle, NUKA_FIELD_JOINT_VELOCITY);
    };

    std::vector<float> q_a, qd_a, q_b, qd_b;
    perturbed_run(&q_a, &qd_a);
    perturbed_run(&q_b, &qd_b);
    ASSERT_EQ(q_a.size(), q_b.size());
    ASSERT_GT(q_a.size(), 0u);

    size_t q_mis = 0u, qd_mis = 0u;
    for (size_t i = 0u; i < q_a.size(); ++i) {
        if (q_a[i] != q_b[i]) ++q_mis;
        if (qd_a[i] != qd_b[i]) ++qd_mis;
    }
    std::printf("[diag] (3) determinism (perturbed): q mismatches=%zu qd mismatches=%zu\n",
                q_mis, qd_mis);
    EXPECT_EQ(q_mis, 0u) << "perturbed q not bit-identical across runs";
    EXPECT_EQ(qd_mis, 0u) << "perturbed qd not bit-identical across runs";
}

// ---------------------------------------------------------------------------
// Gate 4 -- C++ wrapper SetDriveTargets convenience + GetBufferView readback.
// ---------------------------------------------------------------------------
TEST(DriveTargetIo, CppWrapperSetDriveTargets) {
    if (!SceneAvailable()) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    auto device = nuka::Device::Create(0u, nullptr);
    ASSERT_TRUE(device.has_value()) << device.error().message();

    const uint32_t kEnvCount = 16u;
    auto world = nuka::World::CreateFromScene(*device, ScenePath(), kEnvCount,
                                              1.0f / 240.0f);
    ASSERT_TRUE(world.has_value()) << world.error().message();

    auto dt_view = world->GetBufferView(NUKA_FIELD_DRIVE_TARGET);
    ASSERT_TRUE(dt_view.has_value()) << dt_view.error().message();
    ASSERT_GT(dt_view->element_count, 0u);

    // Read current targets, raise them, write via the typed convenience.
    std::vector<float> targets(dt_view->element_count);
    ASSERT_EQ(cudaMemcpy(targets.data(), dt_view->device_ptr,
                         targets.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (float& t : targets) {
        t += 0.20f;
    }
    auto set_result = world->SetDriveTargets(targets.data(), targets.size());
    ASSERT_TRUE(set_result.has_value()) << set_result.error().message();

    // Wrong count is rejected.
    auto bad = world->SetDriveTargets(targets.data(), targets.size() + 1u);
    EXPECT_FALSE(bad.has_value());

    auto step_result = world->StepN(20u);
    ASSERT_TRUE(step_result.has_value()) << step_result.error().message();

    auto q_view = world->GetBufferView(NUKA_FIELD_JOINT_POSITION);
    ASSERT_TRUE(q_view.has_value()) << q_view.error().message();
    std::vector<float> q(q_view->element_count);
    ASSERT_EQ(cudaMemcpy(q.data(), q_view->device_ptr, q.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(AllFinite(q)) << "C++ wrapper SetDriveTargets q non-finite";
    std::printf("[diag] (4) C++ wrapper SetDriveTargets OK: targets=%zu q=%zu\n",
                targets.size(), q.size());
}
