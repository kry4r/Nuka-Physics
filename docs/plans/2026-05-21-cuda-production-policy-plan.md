# CUDA Production Policy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make high-level imported-scene APIs enforce CUDA as the production physics path while preserving explicit CPU reference validation.

**Architecture:** Keep the existing PHI backend selection layer. Add app-level policy fields that distinguish production execution from explicit reference validation, so `PreferCuda` and `ForceCuda` cannot silently use CPU when the CUDA runtime is unavailable, while `ForceCpuReference` remains available only when the caller opts into reference validation.

**Tech Stack:** C++20, CUDA/PHI backend selection, scene demo app layer, GoogleTest.

---

### Task 1: Single-Scene Production Policy Contract

**Files:**
- Modify: `src/apps/debug_shell/scene_demo.hpp`
- Modify: `src/apps/debug_shell/scene_demo.cpp`
- Modify: `tests/apps/test_scene_demo.cpp`

**Step 1: Write failing tests**

Add:

- `SceneDemo.ForceCpuReferenceRequiresExplicitReferenceValidation`
- `SceneDemo.CanRunExplicitCpuReferenceValidationPath`

The first test sets `physics_backend_policy = ForceCpuReference` with the
default options and expects `ExportImportedSceneDebugView()` to throw. The
second test also sets a new explicit reference-validation flag and expects the
result to report `CpuReference` plus `production_physics_backend == false`.

**Step 2: Run RED**

```powershell
cmake --build build --config Release --target nuka_debug_draw_test
```

Expected: build fails because the explicit reference-validation field does not
exist.

**Step 3: Implement minimal policy**

Add `allow_cpu_reference_validation = false` to `SceneDemoOptions`.

In `ExportImportedSceneDebugView()`:

- If resolved backend is `Cuda`, run CUDA production path.
- If resolved backend is `CpuReference` and
  `allow_cpu_reference_validation == false`, throw a runtime error explaining
  that CPU is reference-only and must be explicitly enabled for validation.
- If resolved backend is `CpuReference` and the flag is true, run
  `StepCompiledSceneCpuReference()`.
- In non-CUDA builds, `PreferCuda` and `ForceCuda` should throw instead of
  silently selecting CPU; `ForceCpuReference` should only run with the explicit
  validation flag.

**Step 4: Run GREEN**

```powershell
cmake --build build --config Release --target nuka_debug_draw_test
ctest --test-dir build -C Release --output-on-failure -R "SceneDemo"
```

Expected: scene demo tests pass.

### Task 2: Documentation and Regression Matrix

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/architecture/debug-render-demo.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `docs/plans/2026-05-21-cuda-production-policy-plan.md`

**Step 1: Document policy**

Document that high-level production APIs no longer silently fall back to CPU.
CPU stepping remains available only by explicitly setting the reference policy
and `allow_cpu_reference_validation`.

**Step 2: Full verification**

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

Expected: full build and all tests pass; diff check only has existing CRLF
warnings.

**Step 3: Commit**

```powershell
git add src/apps/debug_shell/scene_demo.* tests/apps/test_scene_demo.cpp README.md docs/architecture/runtime-overview.md docs/architecture/debug-render-demo.md docs/testing/p0-regression-matrix.md docs/plans/2026-05-21-cuda-production-policy-plan.md
git commit -m "phi: enforce cuda production policy"
```

---

## Implementation Notes

- Added `SceneDemoOptions::allow_cpu_reference_validation`, defaulting to
  `false`.
- `ExportImportedSceneDebugView()` now rejects CPU physics unless the caller
  explicitly selects `ForceCpuReference` and enables the validation flag.
- In non-CUDA builds, `PreferCuda` and `ForceCuda` throw instead of silently
  selecting CPU. `ForceCpuReference` remains available only as an explicit
  validation path.

## Verification Record

- RED: `cmake --build build --config Release --target nuka_debug_draw_test`
  failed because `SceneDemoOptions::allow_cpu_reference_validation` did not
  exist.
- GREEN target build: `cmake --build build --config Release --target
  nuka_debug_draw_test` passed with existing C4819 source-encoding warnings.
- GREEN demo tests: `ctest --test-dir build -C Release --output-on-failure -R
  "SceneDemo"` passed 14/14 in 4.04 seconds.
- Full build: `cmake --build build --config Release` passed with existing C4819
  warnings.
- Full suite: `ctest --test-dir build -C Release --output-on-failure` passed
  250/250 in 11.04 seconds.
- Whitespace check: `git diff --check` reported only CRLF normalization warnings.
