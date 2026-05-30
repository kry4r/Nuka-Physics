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

#include <array>
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

// infer-enable #40: the FLOATING-base Go2 (real free 6-DOF root) used by the
// new LINK_VELOCITY + writable-gain gates. The fixed-base go2_stand has zero
// base velocity by construction, so base-velocity must be read off go2_float.
std::string Go2FloatScenePath() {
    return SourcePath("examples/scenes/go2_float.usda").string();
}

bool Go2FloatSceneAvailable() {
    return std::filesystem::exists(SourcePath("examples/scenes/go2_float.usda"));
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

// infer-enable #40: floating-base world for the base-velocity + gain gates.
nuka_result_t CreateFloatWorld(nuka_device_handle device, uint32_t env_count,
                               nuka_world_handle* out) {
    const std::string scene = Go2FloatScenePath();
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
    // Small NEAR-LINEAR perturbation (rad) added to every joint target. This must
    // stay small: a large nudge (e.g. +0.30) added to ALL 12 targets at once drives
    // the planted fixed-base stance into a NONLINEAR contact-conflict regime where
    // ground reaction pushes a handful of joints (here slots 2,4,8,10) BACKWARD,
    // away from their raised target -- a property of CONTACT, not of the write path.
    // The write-sign is what this gate tests, so we perturb in the near-linear band
    // (sim-val #42 swept it: wrong-sign = 0 at delta <= 0.10 across ground seatings,
    // and the +0.30 reversals appear only at large amplitude). 0.05 clears the 1e-4
    // jitter floor on every actuated joint while keeping the response drive-dominated.
    const float kDelta = 0.05f;

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

// ---------------------------------------------------------------------------
// infer-enable #40 -- LINK_VELOCITY read (floating base) + writable PD gains.
// ---------------------------------------------------------------------------

namespace {

// Reads the env0 root-link 6-vector spatial velocity from a LINK_VELOCITY view.
// LINK_VELOCITY element = 6 floats [wx,wy,wz,vx,vy,vz] (omega-first), env-major,
// SAME indexing as link pose -> env0 root is element 0.
std::array<float, 6> RootLinkVelocityEnv0(nuka_world_handle world) {
    auto lv = DownloadField(world, NUKA_FIELD_LINK_VELOCITY);
    std::array<float, 6> v{};
    if (lv.size() >= 6u) {
        for (uint32_t i = 0u; i < 6u; ++i) {
            v[i] = lv[i];
        }
    }
    return v;
}

float L2Norm6(const std::array<float, 6>& v) {
    float s = 0.0f;
    for (float x : v) {
        s += x * x;
    }
    return std::sqrt(s);
}

}  // namespace

// (5) LINK_VELOCITY read: a floating-base Go2 settling under gravity has a
// non-zero, finite, CHANGING base spatial velocity. Layout/shape (6 floats per
// link, env_count*base_link_count links) is asserted, mirroring DRIVE_TARGET's
// layout gate but for the new read field.
TEST(LinkVelocityIo, FloatingBaseRootVelocityIsNonZeroFiniteChanging) {
    if (!Go2FloatSceneAvailable()) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnvCount = 16u;
    WorldGuard world;
    ASSERT_EQ(CreateFloatWorld(device.handle, kEnvCount, &world.handle),
              NUKA_RESULT_OK);

    // Layout: element_count == env_count*base_link_count; stride 6 floats.
    auto q0 = DownloadField(world.handle, NUKA_FIELD_JOINT_POSITION);
    ASSERT_GT(q0.size(), 0u);
    nuka_buffer_view_t lvv{};
    ASSERT_EQ(nuka_world_get_buffer_view(world.handle, NUKA_FIELD_LINK_VELOCITY,
                                         &lvv),
              NUKA_RESULT_OK);
    ASSERT_NE(lvv.device_ptr, nullptr);
    EXPECT_EQ(lvv.element_count, q0.size());                 // links == q entries.
    EXPECT_EQ(lvv.element_stride_bytes, 6u * sizeof(float)); // omega-first 6-vec.

    // Whole velocity buffer is finite at rest (before stepping).
    auto lv_rest = DownloadField(world.handle, NUKA_FIELD_LINK_VELOCITY);
    ASSERT_EQ(lv_rest.size(), static_cast<size_t>(q0.size()) * 6u);
    EXPECT_TRUE(AllFinite(lv_rest)) << "LINK_VELOCITY non-finite at rest";

    // Step under gravity; the free trunk picks up downward (and settling) motion.
    ASSERT_EQ(nuka_world_step_n(world.handle, 20u), NUKA_RESULT_OK);
    const std::array<float, 6> v_a = RootLinkVelocityEnv0(world.handle);
    ASSERT_EQ(nuka_world_step_n(world.handle, 10u), NUKA_RESULT_OK);
    const std::array<float, 6> v_b = RootLinkVelocityEnv0(world.handle);

    auto lv_full = DownloadField(world.handle, NUKA_FIELD_LINK_VELOCITY);
    EXPECT_TRUE(AllFinite(lv_full)) << "LINK_VELOCITY non-finite after stepping";

    // Non-zero: a falling/settling base has motion (assert the linear part, vz,
    // is the dominant non-zero -- gravity pushes it; but accept the full norm).
    const float norm_a = L2Norm6(v_a);
    const float norm_b = L2Norm6(v_b);
    EXPECT_GT(norm_a, 1e-4f) << "root base velocity is ~zero (floating base not moving?)";

    // Changing: between two snapshots the base velocity differs (it is being
    // integrated/contact-corrected each step, not a frozen constant).
    float delta = 0.0f;
    for (uint32_t i = 0u; i < 6u; ++i) {
        delta += std::fabs(v_b[i] - v_a[i]);
    }
    EXPECT_GT(delta, 1e-5f) << "root base velocity did not change between snapshots";

    // Document (diagnostic) a NON-root link velocity slot for env0 -- the engine
    // populates these in ABA Pass-1 (Featherstone-local frame); we report rather
    // than over-assert, matching the nuka.h doc (root is the policy-relevant one).
    float nonroot_norm = 0.0f;
    if (lv_full.size() >= 12u) {  // link 1 of env0 = floats [6..11].
        for (uint32_t i = 6u; i < 12u; ++i) {
            nonroot_norm += lv_full[i] * lv_full[i];
        }
        nonroot_norm = std::sqrt(nonroot_norm);
    }
    std::printf("[diag] (5) LINK_VELOCITY: root|v|_a=%.5f |v|_b=%.5f changed=%.6f "
                "| root v=[%.4f %.4f %.4f, %.4f %.4f %.4f] | nonroot(link1)|v|=%.5f\n",
                static_cast<double>(norm_a), static_cast<double>(norm_b),
                static_cast<double>(delta),
                static_cast<double>(v_b[0]), static_cast<double>(v_b[1]),
                static_cast<double>(v_b[2]), static_cast<double>(v_b[3]),
                static_cast<double>(v_b[4]), static_cast<double>(v_b[5]),
                static_cast<double>(nonroot_norm));
}

// (6) Writable PD gains reach the solver: with the SAME drive target, swapping
// the cooked-soft gains for stiff training gains (Kp=20, Kd=0.5) changes the PD
// response magnitude. Proves DRIVE_STIFFNESS / DRIVE_DAMPING writes alias the
// live device buffers batched_step_params reads, exactly like DRIVE_TARGET.
TEST(DriveGainsIo, WritingStiffnessAndDampingChangesPdResponse) {
    if (!Go2FloatSceneAvailable()) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnvCount = 16u;
    const uint32_t kSteps = 25u;
    const float kDelta = 0.25f;  // raise every joint target by this (same in both).

    // Run with a chosen (Kp,Kd) on slots 1..12; return settled q. The drive
    // target perturbation is IDENTICAL across runs -- only the gains differ.
    auto run_with_gains = [&](float kp, float kd, bool write_gains,
                              std::vector<float>* q_out) {
        WorldGuard world;
        ASSERT_EQ(CreateFloatWorld(device.handle, kEnvCount, &world.handle),
                  NUKA_RESULT_OK);
        const uint32_t blc =
            static_cast<uint32_t>(
                DownloadField(world.handle, NUKA_FIELD_JOINT_POSITION).size() /
                kEnvCount);
        ASSERT_GT(blc, 0u);

        // Same target perturbation in every run.
        auto targets = DownloadField(world.handle, NUKA_FIELD_DRIVE_TARGET);
        for (float& t : targets) {
            t += kDelta;
        }
        WriteDriveTargets(world.handle, targets);

        if (write_gains) {
            // Write Kp/Kd on the actuated slots 1..12 of every env via the new
            // WRITABLE views; root slot 0 left as-is (it is a no-op anyway).
            nuka_buffer_view_t kv{};
            nuka_buffer_view_t dv{};
            ASSERT_EQ(nuka_world_get_buffer_view(world.handle,
                                                 NUKA_FIELD_DRIVE_STIFFNESS, &kv),
                      NUKA_RESULT_OK);
            ASSERT_EQ(nuka_world_get_buffer_view(world.handle,
                                                 NUKA_FIELD_DRIVE_DAMPING, &dv),
                      NUKA_RESULT_OK);
            ASSERT_NE(kv.device_ptr, nullptr);
            ASSERT_NE(dv.device_ptr, nullptr);
            ASSERT_EQ(kv.element_count, targets.size());
            ASSERT_EQ(dv.element_count, targets.size());
            ASSERT_EQ(kv.element_stride_bytes, sizeof(float));
            ASSERT_EQ(dv.element_stride_bytes, sizeof(float));
            // Stiffness, damping and force-limit must alias THREE DISTINCT device
            // buffers (and distinct from DRIVE_TARGET) -- otherwise a Kp write
            // could masquerade as a Kd write. Proves DRIVE_DAMPING specifically
            // aliases the damping buffer, not just "some gain buffer".
            nuka_buffer_view_t fv{};
            nuka_buffer_view_t tv{};
            ASSERT_EQ(nuka_world_get_buffer_view(world.handle,
                                                 NUKA_FIELD_DRIVE_FORCE_LIMIT, &fv),
                      NUKA_RESULT_OK);
            ASSERT_EQ(nuka_world_get_buffer_view(world.handle,
                                                 NUKA_FIELD_DRIVE_TARGET, &tv),
                      NUKA_RESULT_OK);
            EXPECT_NE(kv.device_ptr, dv.device_ptr);
            EXPECT_NE(kv.device_ptr, fv.device_ptr);
            EXPECT_NE(dv.device_ptr, fv.device_ptr);
            EXPECT_NE(kv.device_ptr, tv.device_ptr);
            EXPECT_NE(dv.device_ptr, tv.device_ptr);
            std::vector<float> kbuf(kv.element_count);
            std::vector<float> dbuf(dv.element_count);
            ASSERT_EQ(cudaMemcpy(kbuf.data(), kv.device_ptr,
                                 kbuf.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpy(dbuf.data(), dv.device_ptr,
                                 dbuf.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            for (uint32_t e = 0u; e < kEnvCount; ++e) {
                for (uint32_t link = 1u; link < blc; ++link) {
                    kbuf[e * blc + link] = kp;
                    dbuf[e * blc + link] = kd;
                }
            }
            ASSERT_EQ(cudaMemcpy(kv.device_ptr, kbuf.data(),
                                 kbuf.size() * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpy(dv.device_ptr, dbuf.data(),
                                 dbuf.size() * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
        }

        ASSERT_EQ(nuka_world_step_n(world.handle, kSteps), NUKA_RESULT_OK);
        *q_out = DownloadField(world.handle, NUKA_FIELD_JOINT_POSITION);
    };

    std::vector<float> q_cooked;  // cooked soft gains (no gain write).
    std::vector<float> q_stiff;   // training gains Kp=20, Kd=0.5.
    run_with_gains(0.0f, 0.0f, /*write_gains=*/false, &q_cooked);
    run_with_gains(20.0f, 0.5f, /*write_gains=*/true, &q_stiff);
    ASSERT_EQ(q_cooked.size(), q_stiff.size());
    ASSERT_GT(q_cooked.size(), 0u);
    EXPECT_TRUE(AllFinite(q_cooked));
    EXPECT_TRUE(AllFinite(q_stiff));

    // The stiffer gains pull q measurably CLOSER to the raised target after the
    // same steps -> the response MAGNITUDE differs. Sum |q_stiff - q_cooked|
    // across the actuated joints must exceed solver jitter.
    double total_diff = 0.0;
    uint32_t differing = 0u;
    for (size_t i = 0u; i < q_cooked.size(); ++i) {
        const float d = std::fabs(q_stiff[i] - q_cooked[i]);
        total_diff += d;
        if (d > 1e-4f) {
            ++differing;
        }
    }
    std::printf("[diag] (6) gains-reach-solver: sum|q_stiff-q_cooked|=%.5f "
                "joints differing=%u of %zu\n",
                total_diff, differing, q_cooked.size());
    EXPECT_GT(total_diff, 1e-3)
        << "writing Kp=20/Kd=0.5 did not change the PD response (gains never "
           "reached the solver)";
    EXPECT_GE(differing, 12u)
        << "fewer than one env's worth of actuated joints responded to the gain "
           "change";
}
