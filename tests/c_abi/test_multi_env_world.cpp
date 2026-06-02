// ---------------------------------------------------------------------------
// p01-F T7 -- multi-env C ABI surface (batched articulated-with-contacts world)
//
// Validates that the public C ABI / C++ World surface exposes the batched
// 4096-env Go2 articulated-with-contacts world (T6) when env_count > 1, WITHOUT
// changing the single-env oracle path (env_count == 1 still routes to the
// single-env Featherstone sequence; that path's golden is guarded separately by
// nuka_regression_test --gtest_filter='Go2Stand.*').
//
// Gates here (all through the C ABI, no internal headers):
//  (1) Multi-env create + step + read: env_count > 1, step K, read q/qdot via
//      nuka_world_get_buffer_view. No NaN/Inf; feet engaged (CONTACT_POINTS has
//      a non-zero contact); world steps stably; the headline 4096-env scale
//      creates + steps. q/qdot length == env_count * base_link_count.
//  (2) Determinism (D1): all envs bit-identical to env 0 (same initial state)
//      AND bit-identical across two independent world instances (q, qdot).
//  (3) C++ wrapper: World::CreateFromScene(..., env_count>1, ...) works against
//      the lifted cap (compiles + runs + reads back).
//  (4) Single-env unchanged: env_count == 1 create/step/readback still succeeds
//      and returns a base-length buffer (the byte-for-byte oracle equality is
//      proven by nuka_regression_test Go2Stand.*, reported separately).
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

// RAII C-ABI device.
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

std::vector<float> DownloadField(nuka_world_handle world, nuka_state_field_t field) {
    nuka_buffer_view_t view{};
    EXPECT_EQ(nuka_world_get_buffer_view(world, field, &view), NUKA_RESULT_OK);
    std::vector<float> out;
    if (view.device_ptr == nullptr || view.element_count == 0u) {
        return out;
    }
    // CONTACT_POINTS stride is 3 floats (Vec3); q/qdot stride is 1 float.
    const size_t floats_per_element = view.element_stride_bytes / sizeof(float);
    out.resize(view.element_count * floats_per_element);
    EXPECT_EQ(cudaMemcpy(out.data(), view.device_ptr, out.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return out;
}

size_t ElementCount(nuka_world_handle world, nuka_state_field_t field) {
    nuka_buffer_view_t view{};
    EXPECT_EQ(nuka_world_get_buffer_view(world, field, &view), NUKA_RESULT_OK);
    return view.element_count;
}

bool AllFinite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Gate 1 -- multi-env create + step + read: stability, contacts active, scale.
// ---------------------------------------------------------------------------
TEST(MultiEnvWorld, CreateStepReadStableContactsActive) {
    if (!SceneAvailable()) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnvCount = 256u;  // fast bulk-assert size.
    const uint32_t kSteps = 60u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnvCount, &world.handle), NUKA_RESULT_OK);
    ASSERT_NE(world.handle, nullptr);

    // q/qdot are env-major, length env_count * base_link_count.
    const size_t q_count = ElementCount(world.handle, NUKA_FIELD_JOINT_POSITION);
    const size_t qdot_count = ElementCount(world.handle, NUKA_FIELD_JOINT_VELOCITY);
    ASSERT_GT(q_count, 0u);
    ASSERT_EQ(q_count % kEnvCount, 0u);
    ASSERT_EQ(qdot_count, q_count);
    const uint32_t base_link_count = static_cast<uint32_t>(q_count / kEnvCount);
    ASSERT_GT(base_link_count, 0u);

    // Step one-at-a-time and track an energy-bounded proxy: max|qdot| over an
    // early penetration-correction transient vs the settled tail. Finiteness +
    // feet-loaded alone do NOT prove boundedness (a slow blow-up stays finite),
    // so we require the tail not to exceed the transient -- energy is being
    // removed by the contact solve, not injected (the same proxy T6 validates).
    const uint32_t kTransientSteps = 5u;
    float max_qdot_transient = 0.0f;
    float max_qdot_tail = 0.0f;
    for (uint32_t step = 0u; step < kSteps; ++step) {
        ASSERT_EQ(nuka_world_step_n(world.handle, 1u), NUKA_RESULT_OK);
        const auto qdot_step = DownloadField(world.handle, NUKA_FIELD_JOINT_VELOCITY);
        ASSERT_EQ(qdot_step.size(), qdot_count);
        ASSERT_TRUE(AllFinite(qdot_step)) << "qdot non-finite at step " << step;
        float step_max = 0.0f;
        for (float x : qdot_step) {
            step_max = std::fmax(step_max, std::fabs(x));
        }
        if (step < kTransientSteps) {
            max_qdot_transient = std::fmax(max_qdot_transient, step_max);
        } else {
            max_qdot_tail = std::fmax(max_qdot_tail, step_max);
        }
    }

    const auto q = DownloadField(world.handle, NUKA_FIELD_JOINT_POSITION);
    const auto qdot = DownloadField(world.handle, NUKA_FIELD_JOINT_VELOCITY);
    ASSERT_EQ(q.size(), q_count);
    ASSERT_EQ(qdot.size(), qdot_count);
    EXPECT_TRUE(AllFinite(q)) << "q non-finite after " << kSteps << " steps";
    EXPECT_TRUE(AllFinite(qdot)) << "qdot non-finite after " << kSteps << " steps";

    // Feet engaged: CONTACT_POINTS exposes per-slot Vec3 world contact points;
    // an active foot contact has a non-zero point. (Inactive slots are zeroed.)
    const auto contacts = DownloadField(world.handle, NUKA_FIELD_CONTACT_POINTS);
    ASSERT_FALSE(contacts.empty());
    EXPECT_TRUE(AllFinite(contacts));
    uint32_t active_contact_slots = 0u;
    for (size_t i = 0u; i + 2u < contacts.size(); i += 3u) {
        const float mag2 = contacts[i] * contacts[i] +
                           contacts[i + 1u] * contacts[i + 1u] +
                           contacts[i + 2u] * contacts[i + 2u];
        if (mag2 > 0.0f) {
            ++active_contact_slots;
        }
    }
    std::printf("[diag] (1) env_count=%u base_link_count=%u steps=%u | "
                "active contact slots=%u (of %zu) | max|qdot| transient(<%u)=%.5f "
                "tail=%.5f\n",
                kEnvCount, base_link_count, kSteps, active_contact_slots,
                contacts.size() / 3u, kTransientSteps,
                static_cast<double>(max_qdot_transient),
                static_cast<double>(max_qdot_tail));
    EXPECT_GT(active_contact_slots, 0u) << "feet did not engage the ground (no active contacts)";

    // Penetration bounded: the settled-tail joint velocity must not exceed the
    // early penetration-correction transient (a small margin absorbs solver
    // jitter). A diverging system / growing penetration would have tail >>
    // transient. This is the C-ABI-observable signature of bounded penetration
    // (depth itself is not exposed; CONTACT_POINTS gives the surface projection).
    EXPECT_TRUE(std::isfinite(max_qdot_transient));
    EXPECT_LT(max_qdot_tail, max_qdot_transient * 1.10f + 1.0f)
        << "joint velocity (penetration/energy proxy) grew over the run -- not bounded";
}

// Headline scale: a 4096-env create + step must work.
TEST(MultiEnvWorld, HeadlineScale4096CreateStep) {
    if (!SceneAvailable()) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnvCount = 4096u;
    const uint32_t kSteps = 30u;
    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, kEnvCount, &world.handle), NUKA_RESULT_OK);
    ASSERT_NE(world.handle, nullptr);

    const size_t q_count = ElementCount(world.handle, NUKA_FIELD_JOINT_POSITION);
    ASSERT_EQ(q_count % kEnvCount, 0u);

    ASSERT_EQ(nuka_world_step_n(world.handle, kSteps), NUKA_RESULT_OK);

    const auto q = DownloadField(world.handle, NUKA_FIELD_JOINT_POSITION);
    const auto qdot = DownloadField(world.handle, NUKA_FIELD_JOINT_VELOCITY);
    EXPECT_TRUE(AllFinite(q)) << "4096-env q non-finite";
    EXPECT_TRUE(AllFinite(qdot)) << "4096-env qdot non-finite";
    std::printf("[diag] (1) 4096-env create+step OK: q elements=%zu\n", q_count);
}

// ---------------------------------------------------------------------------
// Gate 2 -- determinism (D1): cross-env identical + two-instance identical.
// ---------------------------------------------------------------------------
TEST(MultiEnvWorld, DeterminismCrossEnvAndTwoRuns) {
    if (!SceneAvailable()) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const uint32_t kEnvCount = 64u;
    const uint32_t kSteps = 40u;

    auto run = [&](std::vector<float>* q_out, std::vector<float>* qdot_out,
                   uint32_t* base_link_count_out) {
        WorldGuard world;
        ASSERT_EQ(CreateWorld(device.handle, kEnvCount, &world.handle), NUKA_RESULT_OK);
        const size_t q_count = ElementCount(world.handle, NUKA_FIELD_JOINT_POSITION);
        *base_link_count_out = static_cast<uint32_t>(q_count / kEnvCount);
        ASSERT_EQ(nuka_world_step_n(world.handle, kSteps), NUKA_RESULT_OK);
        *q_out = DownloadField(world.handle, NUKA_FIELD_JOINT_POSITION);
        *qdot_out = DownloadField(world.handle, NUKA_FIELD_JOINT_VELOCITY);
    };

    std::vector<float> q_a, qdot_a, q_b, qdot_b;
    uint32_t base_link_count = 0u, base_link_count_b = 0u;
    run(&q_a, &qdot_a, &base_link_count);
    run(&q_b, &qdot_b, &base_link_count_b);
    ASSERT_EQ(base_link_count, base_link_count_b);
    ASSERT_GT(base_link_count, 0u);
    ASSERT_EQ(q_a.size(), q_b.size());

    // Cross-env: every env bit-identical to env 0.
    size_t cross_q_mis = 0u, cross_qdot_mis = 0u;
    for (uint32_t env = 1u; env < kEnvCount; ++env) {
        for (uint32_t l = 0u; l < base_link_count; ++l) {
            if (q_a[env * base_link_count + l] != q_a[l]) ++cross_q_mis;
            if (qdot_a[env * base_link_count + l] != qdot_a[l]) ++cross_qdot_mis;
        }
    }

    // Two-run: bit-identical across independent world instances.
    size_t run_q_mis = 0u, run_qdot_mis = 0u;
    for (size_t i = 0u; i < q_a.size(); ++i) {
        if (q_a[i] != q_b[i]) ++run_q_mis;
        if (qdot_a[i] != qdot_b[i]) ++run_qdot_mis;
    }
    std::printf("[diag] (2) determinism: cross-env q=%zu qdot=%zu | "
                "two-run q=%zu qdot=%zu\n",
                cross_q_mis, cross_qdot_mis, run_q_mis, run_qdot_mis);
    EXPECT_EQ(cross_q_mis, 0u);
    EXPECT_EQ(cross_qdot_mis, 0u);
    EXPECT_EQ(run_q_mis, 0u);
    EXPECT_EQ(run_qdot_mis, 0u);
}

// ---------------------------------------------------------------------------
// Gate 3 -- the C++ wrapper drives the lifted cap (env_count > 1).
// ---------------------------------------------------------------------------
TEST(MultiEnvWorld, CppWrapperMultiEnv) {
    if (!SceneAvailable()) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    auto device = nuka::Device::Create(0u, nullptr);
    ASSERT_TRUE(device.has_value()) << device.error().message();

    const uint32_t kEnvCount = 32u;
    auto world = nuka::World::CreateFromScene(*device, ScenePath(), kEnvCount,
                                              1.0f / 240.0f);
    ASSERT_TRUE(world.has_value()) << world.error().message();

    auto step_result = world->StepN(24u);
    ASSERT_TRUE(step_result.has_value()) << step_result.error().message();

    auto view = world->GetBufferView(NUKA_FIELD_JOINT_POSITION);
    ASSERT_TRUE(view.has_value()) << view.error().message();
    ASSERT_NE(view->device_ptr, nullptr);
    ASSERT_EQ(view->element_count % kEnvCount, 0u);
    ASSERT_GT(view->element_count, kEnvCount);

    std::vector<float> values(view->element_count);
    ASSERT_EQ(cudaMemcpy(values.data(), view->device_ptr,
                         values.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(AllFinite(values)) << "C++ wrapper multi-env q non-finite";
    std::printf("[diag] (3) C++ wrapper env_count=%u q elements=%zu (OK)\n",
                kEnvCount, values.size());
}

// ---------------------------------------------------------------------------
// Gate 4 -- env_count == 1 still routes to the single-env path (cap-removal
// guard): create/step/readback succeed and return a base-length buffer. The
// byte-for-byte oracle equality is proven by nuka_regression_test Go2Stand.*.
// ---------------------------------------------------------------------------
TEST(MultiEnvWorld, SingleEnvStillSupported) {
    if (!SceneAvailable()) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    WorldGuard world;
    ASSERT_EQ(CreateWorld(device.handle, 1u, &world.handle), NUKA_RESULT_OK);
    ASSERT_NE(world.handle, nullptr);

    const size_t single_q_count = ElementCount(world.handle, NUKA_FIELD_JOINT_POSITION);
    ASSERT_GT(single_q_count, 0u);

    ASSERT_EQ(nuka_world_step_n(world.handle, 32u), NUKA_RESULT_OK);
    const auto q = DownloadField(world.handle, NUKA_FIELD_JOINT_POSITION);
    ASSERT_EQ(q.size(), single_q_count);
    EXPECT_TRUE(AllFinite(q));

    // Single-env contacts are off (oracle path): CONTACT_POINTS is unsupported.
    nuka_buffer_view_t view{};
    EXPECT_EQ(nuka_world_get_buffer_view(world.handle, NUKA_FIELD_CONTACT_POINTS, &view),
              NUKA_RESULT_NOT_SUPPORTED);
    // p14a: the contact-force readout fields are likewise batched-only -- they live
    // inside the `record->batched != nullptr` arm, so a single-env world falls
    // through to NUKA_RESULT_NOT_SUPPORTED (buffer.cpp final return). Assert all 4.
    for (const nuka_state_field_t f :
         {NUKA_FIELD_LINK_CONTACT_WRENCH, NUKA_FIELD_CONTACT_NORMAL,
          NUKA_FIELD_CONTACT_FORCE, NUKA_FIELD_CONTACT_LINK}) {
        nuka_buffer_view_t v{};
        EXPECT_EQ(nuka_world_get_buffer_view(world.handle, f, &v),
                  NUKA_RESULT_NOT_SUPPORTED)
            << "p14a field " << static_cast<int>(f)
            << " must be NOT_SUPPORTED on the single-env oracle path";
    }

    // Multi-env (env_count=8) q length must be a multiple of the single-env length
    // (same base articulation replicated), confirming env-major layout.
    WorldGuard multi;
    ASSERT_EQ(CreateWorld(device.handle, 8u, &multi.handle), NUKA_RESULT_OK);
    const size_t multi_q_count = ElementCount(multi.handle, NUKA_FIELD_JOINT_POSITION);
    EXPECT_EQ(multi_q_count, single_q_count * 8u)
        << "multi-env q length is not env_count * single-env length";
    std::printf("[diag] (4) single-env q=%zu, 8-env q=%zu (= 8x) -- cap lifted, "
                "single-env path intact\n",
                single_q_count, multi_q_count);
}
