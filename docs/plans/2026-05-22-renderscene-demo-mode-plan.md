# RenderScene Demo Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runnable imported-scene demo mode that renders synchronized `RenderScene` material mesh instances through Vulkan after CUDA simulation.

**Architecture:** Keep debug overlays as the default demo view, then add an explicit render output mode for materialized `RenderScene`. The simulation path remains import -> compile -> CUDA stepping -> `SceneGraph`/`RenderScene` synchronization; only the final Vulkan render call switches between debug draw commands and `RenderSceneVulkan()`.

**Tech Stack:** C++20, CUDA runtime path, Vulkan offscreen renderer, CMake, GoogleTest.

---

## File Structure

- `src/apps/debug_shell/scene_demo.hpp`: add `SceneDemoOutputMode`, output mode option, and material render counters.
- `src/apps/debug_shell/scene_demo.cpp`: route single-scene Vulkan rendering through either debug commands or `RenderSceneVulkan()`.
- `src/apps/debug_shell/main.cpp`: parse optional output mode for single-scene CLI runs and print the selected mode.
- `tests/apps/test_scene_demo.cpp`: add RED/GREEN regression coverage for imported USDA -> CUDA simulation -> RenderScene Vulkan material output.
- `tests/perf/test_vulkan_scene_demo_timing.cpp`: add timing coverage for the new material output mode.
- `README.md`, `docs/architecture/debug-render-demo.md`, `docs/architecture/runtime-overview.md`, `docs/testing/p0-regression-matrix.md`: document the mode and validation coverage.

### Task 1: Scene Demo RenderScene Mode Contract

**Files:**
- Modify: `src/apps/debug_shell/scene_demo.hpp`
- Modify: `tests/apps/test_scene_demo.cpp`

- [x] **Step 1: Write the failing test**

Add `SceneDemo.ExportsUsdSceneRenderSceneVulkanMaterialViewToPpm`:

```cpp
TEST(SceneDemo, ExportsUsdSceneRenderSceneVulkanMaterialViewToPpm) {
    const auto output_path = TempPpmPath("nuka_scene_demo_renderscene_material.ppm");
    std::filesystem::remove(output_path);

    nuka::app::SceneDemoOptions options;
    options.input_path = "examples/scenes/complete_robot.usda";
    options.output_path = output_path.string();
    options.width = 160;
    options.height = 120;
    options.simulation_steps = 8u;
    options.dt = 1.0f / 120.0f;
    options.output_mode = nuka::app::SceneDemoOutputMode::RenderSceneMaterial;

    const auto result = nuka::app::ExportImportedSceneDebugView(options);

    EXPECT_EQ(result.physics_backend, nuka::phi::PhysicsBackend::Cuda);
    EXPECT_TRUE(result.production_physics_backend);
    EXPECT_EQ(result.render_backend, nuka::app::SceneDemoRenderBackend::Vulkan);
    EXPECT_TRUE(result.production_render_backend);
    EXPECT_EQ(result.output_mode, nuka::app::SceneDemoOutputMode::RenderSceneMaterial);
    EXPECT_EQ(result.render_scene_command_count, result.mesh_instance_count);
    EXPECT_EQ(result.debug_command_count, 0u);
    EXPECT_GT(result.non_background_pixel_count, 0u);
    EXPECT_TRUE(std::filesystem::exists(output_path));

    std::filesystem::remove(output_path);
}
```

- [x] **Step 2: Run RED**

```powershell
cmake --build build --config Release --target nuka_debug_draw_test
```

Expected: build fails because `SceneDemoOutputMode`, `output_mode`, and
`render_scene_command_count` do not exist.

- [x] **Step 3: Add API fields**

Add:

```cpp
enum class SceneDemoOutputMode {
    DebugOverlay,
    RenderSceneMaterial
};
```

Add `SceneDemoOptions::output_mode = SceneDemoOutputMode::DebugOverlay`.

Add to `SceneDemoResult`:

```cpp
SceneDemoOutputMode output_mode = SceneDemoOutputMode::DebugOverlay;
uint32_t render_scene_command_count = 0;
```

### Task 2: Single-Scene Vulkan RenderScene Implementation

**Files:**
- Modify: `src/apps/debug_shell/scene_demo.cpp`
- Modify: `tests/apps/test_scene_demo.cpp`

- [x] **Step 1: Implement complete routing**

In `ExportImportedSceneDebugView()`:

- Always set `result.output_mode = options.output_mode`.
- Build debug commands only for `DebugOverlay` mode.
- For `RenderSceneMaterial`, call `render::RenderSceneVulkan(compiled.render, vulkan_options)`.
- Write the returned pixels with `WriteVulkanPpm()`.
- Set `render_scene_command_count = vulkan_result.command_count`.
- Set `debug_command_count = 0` for material mode.
- Reject `RenderSceneMaterial` with `HeadlessReference` using `std::runtime_error`.

- [x] **Step 2: Run GREEN**

```powershell
cmake --build build --config Release --target nuka_debug_draw_test
ctest --test-dir build -C Release --output-on-failure -R "SceneDemo.*RenderScene"
```

Expected: the new material output regression passes.

### Task 3: CLI, Benchmark, and Docs

**Files:**
- Modify: `src/apps/debug_shell/main.cpp`
- Modify: `tests/perf/test_vulkan_scene_demo_timing.cpp`
- Modify: `README.md`
- Modify: `docs/architecture/debug-render-demo.md`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `docs/plans/2026-05-22-renderscene-demo-mode-plan.md`

- [x] **Step 1: Add CLI mode parsing**

Accept an optional final argument after `instance_count`: `debug` or
`renderscene`. For single-scene runs, `renderscene` selects
`SceneDemoOutputMode::RenderSceneMaterial`. Batched runs remain debug-overlay
only for now.

- [x] **Step 2: Add benchmark**

Add `VulkanSceneDemoTiming.ImportedUsdSceneCudaSimulationRenderSceneMaterialUnderOneSecond`
to `tests/perf/test_vulkan_scene_demo_timing.cpp`.

- [x] **Step 3: Update docs**

Document:

- default `debug` output still renders physics overlays,
- `renderscene` output renders material mesh instances after CUDA simulation,
- remaining gap is a full graphics pipeline with real instancing/lights/shadows.

- [x] **Step 4: Full verification**

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

Expected: full build and tests pass; diff check only reports existing CRLF
normalization warnings.

- [x] **Step 5: Commit**

```powershell
git add src/apps/debug_shell/scene_demo.* src/apps/debug_shell/main.cpp tests/apps/test_scene_demo.cpp tests/perf/test_vulkan_scene_demo_timing.cpp README.md docs/architecture/debug-render-demo.md docs/architecture/runtime-overview.md docs/testing/p0-regression-matrix.md docs/plans/2026-05-22-renderscene-demo-mode-plan.md
git commit -m "apps: add renderscene vulkan demo mode"
```

---

## Implementation Notes

- Added `SceneDemoOutputMode::RenderSceneMaterial` for single-scene demo runs.
- The default `DebugOverlay` mode still renders physics debug commands through
  `RenderDebugDrawListVulkan()`.
- The new `RenderSceneMaterial` mode keeps the same import -> CUDA simulation ->
  `SceneGraph`/`RenderScene` synchronization path, then calls
  `RenderSceneVulkan()` and writes the Vulkan pixels to PPM.
- Batched demo runs currently remain debug-overlay only; the CLI rejects
  `renderscene` with `instance_count > 1` until a batched RenderScene material
  merge path is implemented.

## Verification Record

- RED target build attempt: `cmake --build build --config Release --target
  nuka_debug_draw_test` hung in MSBuild/CL before reporting diagnostics.
- RED fallback: single-file MSVC syntax check for `tests/apps/test_scene_demo.cpp`
  failed with missing `SceneDemoOutputMode`, `SceneDemoOptions::output_mode`,
  `SceneDemoResult::output_mode`, and `SceneDemoResult::render_scene_command_count`.
- GREEN syntax check: the same single-file MSVC syntax check passed with existing
  C4819 source-encoding warnings.
- Build environment correction: `C:\Softwares\code\Mangifera-build-graph-native`
  was checked and found to be a different `Mangifera` build tree, not this
  repository. Current verification used `C:\Softwares\code\nuka-physics\build`,
  whose cache points to this repository and has `NK_PHYSICS_BACKEND=CUDA`,
  `NK_REQUIRE_CUDA=ON`, and Vulkan SDK paths cached.
- MSBuild toolchain note: `vcvars64.bat` did not populate `CUDA_PATH`, so the
  CUDA targets required the cached CUDA v13.2 install to be passed as
  `/p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\"`.
  `TrackFileAccess=false` was also used to avoid a Windows file-tracker hang
  observed while rebuilding GoogleTest.
- Focused build:
  `cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_scene_demo_test -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'`
  passed with existing C4819 source-encoding warnings.
- Focused build:
  `cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_scene_demo -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'`
  passed.
- Focused build:
  `cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_perf_test -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'`
  passed with existing C4819 source-encoding warnings.
- Focused regression:
  `cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure -R "SceneDemo.*RenderScene"'`
  passed 2/2 tests in 1.02 seconds.
- Focused benchmark:
  `cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure -R "VulkanSceneDemoTiming.*RenderScene"'`
  passed 1/1 test in 0.60 seconds; the material demo timing case reported
  0.40 seconds.
- Full build:
  `cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'`
  passed in 86.1 seconds with existing C4819 source-encoding warnings.
- Full regression:
  `cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure'`
  passed 252/252 tests in 13.44 seconds.
- CLI demo:
  `cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && C:\Softwares\code\nuka-physics\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.usda %TEMP%\nuka_scene_demo_renderscene_cli.ppm 320 180 60 0.008333333 1 renderscene'`
  exported a PPM with `backend=cuda`, `production_backend=true`,
  `render_backend=vulkan`, `production_render=true`, `debug_commands=0`,
  `renderscene_commands=2`, and `lit_pixels=544`.
- Diff check: `git diff --check` exited 0 and only printed line-ending
  normalization warnings for touched files.
