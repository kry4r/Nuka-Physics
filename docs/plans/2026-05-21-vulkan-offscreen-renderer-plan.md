# Vulkan Offscreen Renderer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a real Vulkan rendering execution path that can consume synchronized debug/render scene data and write a deterministic offscreen artifact after CUDA simulation.

**Architecture:** Keep simulation and rendering decoupled: CUDA physics updates `SceneGraph` / `RenderScene`, debug visualization emits `DebugDrawList`, and the Vulkan renderer consumes that draw list through a swapchain-free offscreen path. The first Vulkan renderer uses a compute raster pass into a storage image, read back to host for tests and PPM output; later graphics pipeline and instancing work can replace the raster internals without changing the app-level handoff.

**Tech Stack:** C++20, Vulkan 1.0+, GLSL compute compiled with `glslc`, CMake, GoogleTest.

---

### Task 1: Vulkan Offscreen Debug Renderer Core

**Status:** Completed in the current iteration. `render::RenderDebugDrawListVulkan()`
now compiles and runs a GLSL compute shader through Vulkan, renders debug draw
commands into an RGBA8 offscreen image, reads pixels back, and reports
non-background pixels plus selected device metadata.

**Files:**
- Modify: `src/render/vulkan_renderer.hpp`
- Modify: `src/render/vulkan_renderer.cpp`
- Create: `src/render/shaders/debug_draw.comp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/render/test_vulkan_backend.cpp`

**Scope:** Add `render::VulkanOffscreenOptions`, `render::VulkanOffscreenReport`,
and `render::RenderDebugDrawListVulkan()` that allocate a Vulkan instance,
device, queue, storage buffer, storage image, descriptor set, compute pipeline,
and staging readback. The shader maps `DebugDrawList` line/sphere/box/AABB/frame
commands into a deterministic RGBA8 image. The function returns non-background
pixel counts, command counts, selected device, and elapsed work metrics.

**Validation:**

```powershell
cmake --build build --config Release --target nuka_render_test
ctest --test-dir build -C Release --output-on-failure -R "VulkanRenderer"
```

**Expected:** tests fail before implementation because the offscreen API does
not exist, then pass with a non-empty Vulkan-rendered image.

### Task 2: Scene Demo Vulkan Artifact Path

**Status:** Completed in the current iteration. `nuka_scene_demo` defaults to
CUDA production physics plus Vulkan production rendering, writes its PPM artifact
from the Vulkan image readback, and exposes render backend/device metadata in
`SceneDemoResult` and CLI output.

**Files:**
- Modify: `src/apps/debug_shell/scene_demo.hpp`
- Modify: `src/apps/debug_shell/scene_demo.cpp`
- Modify: `src/apps/debug_shell/main.cpp`
- Modify: `tests/apps/test_scene_demo.cpp`
- Create: `tests/perf/test_vulkan_scene_demo_timing.cpp`
- Modify: `tests/CMakeLists.txt`

**Scope:** Extend `SceneDemoOptions` with a renderer backend choice that defaults
to Vulkan on this workstation while preserving the deterministic PPM debug
fallback for CI/reference. The demo should still run CUDA physics first, then
render the synchronized `DebugDrawList` through Vulkan and write a PPM artifact
from the Vulkan image readback.

**Validation:**

```powershell
cmake --build build --config Release --target nuka_debug_draw_test nuka_perf_test
ctest --test-dir build -C Release --output-on-failure -R "SceneDemo|VulkanSceneDemoTiming"
```

**Expected:** MJCF and USDA examples report CUDA physics plus Vulkan rendering,
produce non-empty PPM artifacts, and complete under one second.

### Task 3: Documentation, Regression Matrix, and Full Verification

**Status:** In progress.

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/debug-render-demo.md`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `docs/plans/2026-05-21-vulkan-offscreen-renderer-plan.md`

**Scope:** Document the Vulkan offscreen renderer as the current production
rendering execution path, distinguish it from the old CPU raster fallback, and
record the exact validation commands and benchmark results.

**Validation:**

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

**Expected:** full Release build and all tests pass on the CUDA/Vulkan
workstation. Commit as `render: add vulkan offscreen debug renderer`.
