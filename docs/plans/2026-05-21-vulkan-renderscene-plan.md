# Vulkan RenderScene Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Vulkan rendering entry point that consumes `render::RenderScene` mesh instances and materials instead of only renderer-independent debug draw commands.

**Architecture:** Keep the existing Vulkan offscreen compute renderer as the executable backend. Add a `RenderSceneVulkan()` adapter that converts synchronized `RenderScene` mesh instances into Vulkan draw commands with material-derived colors, then renders through the existing offscreen storage-image path. This is intentionally a real rendered `RenderScene` workflow while leaving room for a later Vulkan graphics/instancing pipeline to replace the adapter internals without changing app-level callers.

**Tech Stack:** C++20, Vulkan offscreen renderer, CMake, GoogleTest.

---

### Task 1: RenderScene Vulkan API

**Files:**
- Modify: `src/render/vulkan_renderer.hpp`
- Modify: `src/render/vulkan_renderer.cpp`
- Modify: `tests/render/test_vulkan_backend.cpp`

**Step 1: Write failing test**

Add `VulkanRenderer.RendersRenderSceneMaterialsToOffscreenImage`:

- Build a `RenderScene` with two mesh instances.
- Add two materials with distinct base colors.
- Render with `RenderSceneVulkan()`.
- Assert Vulkan production backend, output dimensions, command count equals mesh count, non-background pixels are present, and at least one pixel carries each material color.

**Step 2: Run red**

```powershell
cmake --build build --config Release --target nuka_render_test
```

Expected: build fails because `RenderSceneVulkan()` does not exist.

**Step 3: Implement adapter**

Add:

- `RenderSceneVulkan(const RenderScene&, const VulkanOffscreenOptions&)`
- material lookup by `material_id`
- RGBA conversion from `RenderMaterial::base_color` and `alpha`
- mesh-to-command conversion for sphere/capsule/box-like shapes

**Step 4: Run green**

```powershell
cmake --build build --config Release --target nuka_render_test
ctest --test-dir build -C Release --output-on-failure -R "VulkanRenderer"
```

Expected: all Vulkan renderer tests pass.

### Task 2: RenderScene Benchmark and Docs

**Files:**
- Create: `tests/perf/test_vulkan_renderscene_timing.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `README.md`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/architecture/debug-render-demo.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `docs/plans/2026-05-21-vulkan-renderscene-plan.md`

**Step 1: Add benchmark**

Add `VulkanRenderSceneTiming.RenderSceneMaterialPassUnderOneSecond` with at
least 128 mesh instances and material bindings rendered to a 320x180 image.

**Step 2: Update docs**

Document that Vulkan now has both debug draw and `RenderScene` materialized
offscreen entry points. Keep remaining gap explicit: this is still compute
offscreen raster, not a full graphics pipeline with instancing/shadows.

**Step 3: Full verification**

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

Expected: full Release build and all tests pass.

**Step 4: Commit**

```powershell
git add src/render/vulkan_renderer.* tests/render/test_vulkan_backend.cpp tests/perf/test_vulkan_renderscene_timing.cpp tests/CMakeLists.txt README.md docs/architecture/runtime-overview.md docs/architecture/debug-render-demo.md docs/testing/p0-regression-matrix.md docs/plans/2026-05-21-vulkan-renderscene-plan.md
git commit -m "render: add vulkan renderscene path"
```

---

## Implementation Notes

- `RenderSceneVulkan()` now exists as the renderer-facing API for materialized
  `RenderScene` data.
- The current implementation converts mesh instances into Vulkan offscreen
  commands and maps `RenderMaterial::base_color` / `alpha` into RGBA8, so it is
  executable today while preserving an API boundary for a later Vulkan graphics
  pipeline with instancing, lighting, shadows, and direct presentation.
- This stage intentionally keeps the CPU rasterizer out of the production render
  path; CPU rendering remains a reference artifact path.

## Verification Record

- RED: `cmake --build build --config Release --target nuka_render_test` failed
  before implementation because `RenderSceneVulkan()` did not exist.
- GREEN target build: `cmake --build build --config Release --target
  nuka_render_test` passed with existing C4819 source-encoding warnings.
- GREEN Vulkan tests: `ctest --test-dir build -C Release --output-on-failure -R
  "VulkanRenderer"` passed 4/4.
- Benchmark target build: `cmake --build build --config Release --target
  nuka_perf_test` passed with existing C4819 warnings.
- Benchmark: `ctest --test-dir build -C Release --output-on-failure -R
  "VulkanRenderSceneTiming"` passed 1/1; CTest reported 0.20-0.21 seconds for
  the 128-instance RenderScene material pass.
- Full build: `cmake --build build --config Release` passed with existing C4819
  warnings.
- Full suite: `ctest --test-dir build -C Release --output-on-failure` passed
  248/248 in 10.96 seconds.
- Whitespace check: `git diff --check` reported only CRLF normalization warnings.
