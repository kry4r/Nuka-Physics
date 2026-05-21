# CUDA Batched Vulkan Demo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a runnable imported-scene workflow that simulates multiple environments through CUDA `BatchedDeviceWorld` and renders the resulting batched debug view through Vulkan.

**Architecture:** Keep the existing single-scene demo intact. Add a batched demo API that imports/cooks one scene, creates multiple `WorldInstance` objects with per-environment offsets, uploads them into `runtime::gpu::BatchedDeviceWorld`, steps CUDA contacts/joints/drives, samples batched CUDA sensors, downloads final state only as the render/debug synchronization boundary, and emits one combined Vulkan debug image. CPU remains import/cooking/orchestration/reference only; production physics is CUDA and production render is Vulkan.

**Tech Stack:** C++20, CUDA, Vulkan offscreen renderer, CMake, GoogleTest.

---

### Task 1: Batched Demo API Contract

**Files:**
- Modify: `src/apps/debug_shell/scene_demo.hpp`
- Modify: `tests/apps/test_scene_demo.cpp`

**Step 1: Write the failing test**

Add `SceneDemo.ExportsBatchedCudaVulkanSceneDebugViewToPpm`.

Test setup:

```cpp
nuka::app::BatchedSceneDemoOptions options;
options.input_path = "examples/scenes/complete_robot.usda";
options.output_path = output_path.string();
options.instance_count = 4u;
options.width = 220;
options.height = 140;
options.simulation_steps = 8u;
options.dt = 1.0f / 120.0f;
options.instance_spacing = {2.0f, 0.0f, 0.0f};
```

Expected assertions:

- result uses `phi::PhysicsBackend::Cuda` and production physics is true.
- result render backend is Vulkan and production render is true.
- result instance count is 4.
- result body count per instance is 2 for `complete_robot.usda`.
- result total body count is 8.
- result batched CUDA report has simulated step count 8 and nonzero joint/drive rows.
- result batched IMU sample count is 4.
- output PPM exists and has non-background pixels.
- at least one body pose changes relative to authored initial pose.

**Step 2: Run red**

```powershell
cmake --build build --config Release --target nuka_debug_draw_test
```

Expected: build fails because `BatchedSceneDemoOptions`, `BatchedSceneDemoResult`, and `ExportBatchedImportedSceneDebugView()` do not exist.

### Task 2: Implement CUDA Batched Demo Runtime

**Files:**
- Modify: `src/apps/debug_shell/scene_demo.hpp`
- Modify: `src/apps/debug_shell/scene_demo.cpp`

**Step 1: Add result/options structs**

Add `BatchedSceneDemoOptions` with:

- `input_path`, `output_path`, `width`, `height`, `simulation_steps`, `dt`,
  `gravity`, `auto_fit_view`, `view_scale`, `view_center`
- `instance_count`, `instance_spacing`
- `physics_backend_policy` defaulting to `PreferCuda`
- `render_backend` defaulting to Vulkan

Add `BatchedSceneDemoResult` with:

- scene counts: `instance_count`, `body_count_per_instance`,
  `total_body_count`, `mesh_instance_count_per_instance`
- render counts: `debug_command_count`, `non_background_pixel_count`
- backend metadata mirroring `SceneDemoResult`
- CUDA report counters from `CudaBatchedWorldStepReport`
- batched sensor counters: IMU sample count and first sample vectors
- `std::vector<std::vector<math::Transform>> body_world_poses_by_instance`

**Step 2: Implement imported-scene batched execution**

Implement `ExportBatchedImportedSceneDebugView(const BatchedSceneDemoOptions&)`:

1. Validate input/output path and `instance_count > 0`.
2. Import scene with existing `LoadSceneForDemo()`.
3. Build one `CompiledScene` and one `BuiltWorld`.
4. Create `instance_count` copies of the world instance.
5. Offset each copy by `instance_index * instance_spacing`.
6. Resolve PHI backend and require CUDA for production physics; throw on CPU selection.
7. Upload with `runtime::gpu::UploadBatchedDeviceWorld()`.
8. Run `StepBatchedCudaWorld()` with contacts, joints, and drives enabled.
9. Query batched IMU/frame-pose sensors with one request per sensor per instance.
10. Download final batched state.
11. For each instance, apply its slice to the compiled scene, build debug commands, then append translated commands to one combined `DebugDrawList`.
12. Render combined commands through `render::RenderDebugDrawListVulkan()` and write PPM.
13. Fill result metadata and pose arrays.

**Step 3: Run green**

```powershell
cmake --build build --config Release --target nuka_debug_draw_test
ctest --test-dir build -C Release --output-on-failure -R "SceneDemo.ExportsBatchedCudaVulkanSceneDebugViewToPpm"
```

Expected: new workflow test passes and writes a non-empty Vulkan PPM.

### Task 3: CLI and Benchmark

**Files:**
- Modify: `src/apps/debug_shell/main.cpp`
- Create: `tests/perf/test_batched_vulkan_scene_demo_timing.cpp`
- Modify: `tests/CMakeLists.txt`

**Step 1: Add CLI mode**

Extend `nuka_scene_demo` so an optional final argument selects batched instance count:

```powershell
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.usda out\batched.ppm 640 360 60 0.0166667 8
```

If the final instance count is greater than one, call `ExportBatchedImportedSceneDebugView()` and print `instances`, `batched_total_bodies`, `batched_rows`, and `batched_imu_samples`.

**Step 2: Add performance test**

Add `BatchedVulkanSceneDemoTiming.ImportedUsdBatchedCudaSimulationVulkanRenderUnderOneSecond`:

- 8 instances.
- 60 fixed steps.
- Vulkan output size 320x180.
- asserts CUDA and Vulkan production backends.
- asserts non-empty debug image and `< 1000 ms`.

**Step 3: Run target benchmark**

```powershell
cmake --build build --config Release --target nuka_perf_test
ctest --test-dir build -C Release --output-on-failure -R "BatchedVulkanSceneDemoTiming"
```

Expected: benchmark passes.

### Task 4: Docs, Full Verification, Commit

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/architecture/debug-render-demo.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `docs/plans/2026-05-21-cuda-batched-vulkan-demo-plan.md`

**Step 1: Update docs**

Document the new batched imported-scene CUDA/Vulkan workflow and keep the
remaining gap explicit: richer render instancing/material/shadow support and
avoiding final render readback for interactive renderer integration.

**Step 2: Full verification**

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

Expected: full Release build and all tests pass.

**Step 3: Commit**

```powershell
git add src/apps/debug_shell/scene_demo.* src/apps/debug_shell/main.cpp tests/apps/test_scene_demo.cpp tests/perf/test_batched_vulkan_scene_demo_timing.cpp tests/CMakeLists.txt README.md docs/architecture/runtime-overview.md docs/architecture/debug-render-demo.md docs/testing/p0-regression-matrix.md docs/plans/2026-05-21-cuda-batched-vulkan-demo-plan.md
git commit -m "apps: add batched cuda vulkan scene demo"
```

---

## Result

Implemented a batched imported-scene CUDA/Vulkan workflow.

- Added `BatchedSceneDemoOptions`, `BatchedSceneDemoResult`, and
  `ExportBatchedImportedSceneDebugView()`.
- The workflow imports one XML/USD/URDF scene, builds one cooked template,
  creates offset `WorldInstance` copies, uploads them into
  `runtime::gpu::BatchedDeviceWorld`, runs CUDA batched contacts/joints/drives,
  samples batched CUDA IMU/frame-pose observations, and renders the combined
  multi-environment debug view through Vulkan.
- Extended `nuka_scene_demo` with an optional final `instance_count` argument.
  Values greater than one run the batched CUDA/Vulkan path.
- Added a global workflow regression test and a one-second benchmark.

Verification performed during implementation:

```powershell
cmake --build build --config Release --target nuka_debug_draw_test
ctest --test-dir build -C Release --output-on-failure -R "SceneDemo.ExportsBatchedCudaVulkanSceneDebugViewToPpm"
cmake --build build --config Release --target nuka_scene_demo nuka_perf_test
ctest --test-dir build -C Release --output-on-failure -R "BatchedVulkanSceneDemoTiming"
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.usda %TEMP%\nuka_cli_batched_scene_demo.ppm 220 140 8 0.008333333 4
```

Observed targeted results:

- `SceneDemo.ExportsBatchedCudaVulkanSceneDebugViewToPpm`: passed in 0.29 s.
- `BatchedVulkanSceneDemoTiming.ImportedUsdBatchedCudaSimulationVulkanRenderUnderOneSecond`:
  passed in 0.36 s test time / 0.41 s total CTest time.
- CLI smoke generated a PPM with 4 instances, 8 total bodies, CUDA production
  physics, Vulkan production rendering, 264 batched constraint rows, 4 batched
  IMU samples, and 256 non-background pixels.

Remaining global CUDA/Vulkan follow-up:

- Replace debug/PPM-focused Vulkan output with richer Vulkan render features:
  instancing, material binding, shadows, and interactive presentation.
- Keep CPU limited to import/cooking/orchestration/reference validation; batched
  production physics remains CUDA.
