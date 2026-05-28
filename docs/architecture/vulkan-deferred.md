# Deferred Vulkan Work

v0.1 Foundation Refactor is currently prioritizing CUDA physics, row
infrastructure, diagnostics, Featherstone, and the C ABI. Vulkan validation is
deferred unless a phase explicitly requires it for a physics deliverable.

## Current Policy

- Default validation focuses on physics and uses headless reference rendering
  where a rendered artifact is only a smoke check.
- Vulkan-specific tests and timing gates are opt-in through:

  ```bash
  cmake -S . -B build -DNK_BUILD_TESTS=ON -DNK_BUILD_VULKAN_VALIDATION=ON
  ```

- Do not install or modify system Vulkan/NVIDIA driver components as part of
  the v0.1 physics phases.
- When Vulkan work appears while implementing physics phases, record it in this
  document instead of blocking physics progress.

## Deferred Items

- NVIDIA Vulkan ICD is not available in the current Linux environment; Vulkan
  tests route to lavapipe and can miss the one-second render timing gates.
- `tests/render/test_vulkan_backend.cpp` is disabled by default and should be
  re-enabled under `NK_BUILD_VULKAN_VALIDATION=ON` when the Vulkan environment is
  ready.
- `tests/perf/test_vulkan_scene_demo_timing.cpp` is disabled by default.
- `tests/perf/test_batched_vulkan_scene_demo_timing.cpp` is disabled by
  default.
- `tests/perf/test_vulkan_renderscene_timing.cpp` is disabled by default.
- Scene demo tests that only need an artifact now use
  `SceneDemoRenderBackend::HeadlessReference` by default; Vulkan RenderScene
  material coverage is deferred to the opt-in validation target.
- Scene demo render timing gates are disabled by default behind
  `NK_BUILD_DEMO_TIMING_VALIDATION=ON` because they include import, CUDA context
  cold start, and artifact rendering. Core physics and CUDA kernel timing tests
  remain in default validation.
