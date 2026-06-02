// ---------------------------------------------------------------------------
// p02-B/C -- C ABI runtime coverage for the differentiable rollout tape.
// ---------------------------------------------------------------------------
//
// Exercises the nuka_diffsim.h handle surface END-TO-END (not just symbol
// export): create a single-env articulated world, build a tape, drive N
// contact-free differentiable steps (the tape records the world's live
// DRIVE_TARGET device buffer each step), run the reverse pass, and read back
// grad_actions + grad_parameters (the mass spine). NOTE: the zero-copy
// DRIVE_TARGET buffer VIEW is served only on the batched (env_count > 1) path,
// so on this single-env world the tape records the world's built hold-pose
// targets directly (a constant action across steps). The action-recording from
// a per-step-CHANGING buffer is covered by the C++ MildActions test; here the
// goal is the C-ABI glue (create/step/backward/reset, seed layout, mass
// ordering, readback, determinism).
//
// Gates:
//   (1) The full create -> step_with_tape*N -> backward -> reset cycle returns OK
//       and produces FINITE, NONZERO grads (the rollout actually backpropagated).
//   (2) D1: the WHOLE C-ABI backward is bit-identical across two runs (same world
//       init + same actions). This pins the entire C-ABI glue -- the
//       grad_observations seed-layout unpacking, the BuildMassParams global-link
//       ordering, the device<->host readback -- to a deterministic, stable path.
//   (3) step_count / link_count accessors report the recorded rollout correctly.
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"
#include "nuka/nuka_diffsim.h"

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
        if (handle != nullptr) nuka_device_destroy(handle);
    }
};

struct WorldGuard {
    nuka_world_handle handle = nullptr;
    ~WorldGuard() {
        if (handle != nullptr) nuka_world_destroy(handle);
    }
};

struct TapeGuard {
    nuka_tape_handle handle = nullptr;
    ~TapeGuard() {
        if (handle != nullptr) nuka_tape_destroy(handle);
    }
};

// Single-env articulated world (env_count == 1 uploads the articulation_device
// the tape binds to; the differentiable forward is independent of the stepper).
nuka_result_t CreateFloatWorld(nuka_device_handle device,
                               nuka_world_handle* out) {
    const std::string scene = Go2FloatScenePath();
    nuka_world_desc_t desc{};
    desc.scene_path = scene.c_str();
    desc.env_count = 1u;
    desc.fixed_dt = 0.005f;
    return nuka_world_create_from_scene(device, &desc, out);
}

bool AllFinite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// Drives a complete tape rollout + backward via the C ABI and returns the
// concatenated [grad_actions | grad_parameters]. `world` is reset to the hold
// pose by the caller order (fresh world per call -> identical init).
std::vector<float> RunTapeRollout(nuka_device_handle device, uint32_t n_steps,
                                  uint32_t* out_step_count,
                                  uint32_t* out_link_count) {
    WorldGuard world;
    EXPECT_EQ(CreateFloatWorld(device, &world.handle), NUKA_RESULT_OK);

    nuka_tape_desc_t desc{};
    desc.checkpoint_interval = 5u;
    desc.max_tape_entries = 128u;
    desc.max_checkpoints = 64u;
    desc.recompute_on_backward = 1u;
    TapeGuard tape;
    EXPECT_EQ(nuka_tape_create(world.handle, &desc, &tape.handle), NUKA_RESULT_OK);
    EXPECT_NE(tape.handle, nullptr);

    const uint32_t n = nuka_tape_link_count(tape.handle);
    EXPECT_GT(n, 0u);

    // Drive N steps. The tape records the world's live DRIVE_TARGET device buffer
    // (the built hold-pose targets) each step + advances the differentiable
    // forward; a constant action across steps still backpropagates through the
    // N-step rollout (the state evolves under gravity / the PD hold).
    for (uint32_t s = 0u; s < n_steps; ++s) {
        EXPECT_EQ(nuka_world_step_with_tape(world.handle, tape.handle),
                  NUKA_RESULT_OK);
    }

    const uint32_t step_count = nuka_tape_step_count(tape.handle);
    EXPECT_EQ(step_count, n_steps);
    if (out_step_count) *out_step_count = step_count;
    if (out_link_count) *out_link_count = n;

    // Seed a nontrivial loss on the final state (dL/dqdot' on the joints).
    std::vector<float> seed(2u * n + 6u * n, 0.0f);
    for (uint32_t i = 0u; i < n; ++i) seed[n + i] = 0.3f;  // dL/dqdot'

    std::vector<float> grad_actions(static_cast<size_t>(step_count) * n, 0.0f);
    std::vector<float> grad_params(n, 0.0f);
    EXPECT_EQ(nuka_tape_backward(tape.handle, seed.data(), grad_actions.data(),
                                 grad_params.data()),
              NUKA_RESULT_OK);

    // reset for hygiene (proves the reset path runs).
    EXPECT_EQ(nuka_tape_reset(tape.handle), NUKA_RESULT_OK);
    EXPECT_EQ(nuka_tape_step_count(tape.handle), 0u);

    std::vector<float> out = grad_actions;
    out.insert(out.end(), grad_params.begin(), grad_params.end());
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// (1) End-to-end: create -> step_with_tape -> backward -> finite, nonzero grads.
// ---------------------------------------------------------------------------
TEST(DiffsimTapeCAbi, EndToEndProducesFiniteNonzeroGrads) {
    if (!Go2FloatSceneAvailable()) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    uint32_t step_count = 0u, link_count = 0u;
    const std::vector<float> grads =
        RunTapeRollout(device.handle, /*n_steps=*/20u, &step_count, &link_count);
    ASSERT_GT(grads.size(), 0u);
    EXPECT_TRUE(AllFinite(grads)) << "C-ABI tape backward produced non-finite grads";

    float max_abs = 0.0f;
    for (float v : grads) max_abs = std::max(max_abs, std::fabs(v));
    std::printf("[c-abi tape] steps=%u links=%u grad_size=%zu max|grad|=%.4e\n",
                step_count, link_count, grads.size(),
                static_cast<double>(max_abs));
    EXPECT_GT(max_abs, 1e-6f)
        << "C-ABI tape backward grads are all ~zero (nothing backpropagated)";
}

// ---------------------------------------------------------------------------
// (2) D1: the whole C-ABI backward is bit-identical across two runs. This pins
// the seed-layout unpacking + BuildMassParams ordering + readback to a stable,
// deterministic path.
// ---------------------------------------------------------------------------
TEST(DiffsimTapeCAbi, BackwardBitIdenticalAcrossTwoRuns) {
    if (!Go2FloatSceneAvailable()) {
        GTEST_SKIP() << "go2_float scene is not available";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);

    const std::vector<float> r1 = RunTapeRollout(device.handle, 16u, nullptr, nullptr);
    const std::vector<float> r2 = RunTapeRollout(device.handle, 16u, nullptr, nullptr);
    ASSERT_EQ(r1.size(), r2.size());
    size_t mismatches = 0u;
    for (size_t i = 0u; i < r1.size(); ++i)
        if (r1[i] != r2[i]) ++mismatches;
    std::printf("[c-abi tape] D1 two-run mismatches=%zu of %zu\n", mismatches,
                r1.size());
    EXPECT_EQ(mismatches, 0u)
        << "C-ABI tape backward not bit-identical across two runs";
}

// v0.7 p01: sparse-solver backend selection C ABI round-trip + validation.
TEST(DiffsimTapeCAbi, SparseSolverBackendSetGetRoundTrip) {
    // NULL handle is rejected without needing a world / device.
    nuka_sparse_solver_backend_t got;
    EXPECT_EQ(nuka_world_set_sparse_solver_backend(nullptr,
                                                   NUKA_SOLVER_BACKEND_SELF_CG),
              NUKA_RESULT_NULL_HANDLE);
    EXPECT_EQ(nuka_world_get_sparse_solver_backend(nullptr, &got),
              NUKA_RESULT_NULL_HANDLE);

    if (!Go2FloatSceneAvailable()) {
        GTEST_SKIP() << "go2_float scene is not available (world-backed checks)";
    }
    DeviceGuard device;
    ASSERT_NE(device.handle, nullptr);
    WorldGuard world;
    ASSERT_EQ(CreateFloatWorld(device.handle, &world.handle), NUKA_RESULT_OK);

    // Default is SELF_CG (a fresh world already uses the only shipping backend).
    EXPECT_EQ(nuka_world_get_sparse_solver_backend(world.handle, &got),
              NUKA_RESULT_OK);
    EXPECT_EQ(got, NUKA_SOLVER_BACKEND_SELF_CG);

    // Explicit set of SELF_CG round-trips.
    EXPECT_EQ(nuka_world_set_sparse_solver_backend(world.handle,
                                                   NUKA_SOLVER_BACKEND_SELF_CG),
              NUKA_RESULT_OK);
    EXPECT_EQ(nuka_world_get_sparse_solver_backend(world.handle, &got),
              NUKA_RESULT_OK);
    EXPECT_EQ(got, NUKA_SOLVER_BACKEND_SELF_CG);

    // v0.7 p02: SELF_MINRES is now a SHIPPING backend (self-written symmetric
    // indefinite MINRES + ILU(0)); it is accepted and round-trips.
    EXPECT_EQ(nuka_world_set_sparse_solver_backend(world.handle,
                                                   NUKA_SOLVER_BACKEND_SELF_MINRES),
              NUKA_RESULT_OK);
    EXPECT_EQ(nuka_world_get_sparse_solver_backend(world.handle, &got),
              NUKA_RESULT_OK);
    EXPECT_EQ(got, NUKA_SOLVER_BACKEND_SELF_MINRES);
    // Restore the default so the rest of the contract checks read SELF_CG.
    EXPECT_EQ(nuka_world_set_sparse_solver_backend(world.handle,
                                                   NUKA_SOLVER_BACKEND_SELF_CG),
              NUKA_RESULT_OK);

    // SELF_GMRES remains reserved for v0.7 phase 3 -> NOT_SUPPORTED, no mutation.
    EXPECT_EQ(nuka_world_set_sparse_solver_backend(world.handle,
                                                   NUKA_SOLVER_BACKEND_SELF_GMRES),
              NUKA_RESULT_NOT_SUPPORTED);
    EXPECT_EQ(nuka_world_get_sparse_solver_backend(world.handle, &got),
              NUKA_RESULT_OK);
    EXPECT_EQ(got, NUKA_SOLVER_BACKEND_SELF_CG) << "rejected set must not mutate";

    // An out-of-range enum is INVALID_ARG; null out-pointer is INVALID_ARG.
    EXPECT_EQ(nuka_world_set_sparse_solver_backend(
                  world.handle, static_cast<nuka_sparse_solver_backend_t>(99)),
              NUKA_RESULT_INVALID_ARG);
    EXPECT_EQ(nuka_world_get_sparse_solver_backend(world.handle, nullptr),
              NUKA_RESULT_INVALID_ARG);
}
