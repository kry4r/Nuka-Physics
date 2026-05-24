# CUDA Coupling Constraint Rows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Promote particle/rigid coupling cache slots into CUDA-resident coupling constraint rows that can become the shared rigid/cloth/deformable/fluid assembly surface.

**Architecture:** `CudaParticleWorld` keeps the existing fixed four-slot per-particle coupling cache, but each slot also owns a device-side row record with particle id, cooked shape id, body id, normal, contact point, Jacobian terms, effective mass, rhs, position error, and accumulated normal impulse. The current direct projection/impulse path continues to solve the contact, while the row data makes the GPU state inspectable and ready for a later unified PGS/XPBD assembly pass. CPU remains a validation/download boundary only.

**Tech Stack:** C++20, CUDA, PHI `Buffer`, existing `DeviceWorld`, GoogleTest, CTest, MSVC via `vcvars64.bat`.

---

## Task 1: Coupling Row State

**Files:**
- Modify: `src/runtime/gpu/cuda_particle_world.hpp`
- Modify: `src/runtime/gpu/cuda_particle_world.cu`
- Modify: `tests/runtime/test_cuda_particle_world.cpp`

**Step 1: Write failing runtime test**

Add `CudaParticleWorld.AssemblesDeviceWorldCouplingConstraintRowsOnCuda`.

The test builds the existing two-box corner scene and one particle touching both cooked boxes. After one CUDA step it downloads coupling rows and verifies:
- row count equals `particle_count * kCudaParticleCouplingSlotsPerParticle`
- slot 0 and slot 1 are active
- both rows keep `particle_index == 0`
- shape ids are `0` and `1`
- body ids match the two cooked robot-link bodies
- normal impulse and effective mass are positive
- normal length is approximately one
- position error is positive before projection
- body angular Jacobian is nonzero for the off-center floor contact

Run:

```powershell
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_cuda_runtime_test -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'
ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure -R "CudaParticleWorld\.AssemblesDeviceWorldCouplingConstraintRowsOnCuda"
```

Expected: build or test fails because coupling row download API does not exist.

**Step 2: Implement CUDA row storage**

Add:
- `CudaParticleCouplingConstraintRow`
- `CudaParticleCouplingRowsState`
- `CudaParticleWorld::DownloadCouplingRows()`
- device buffer allocation sized `particle_count * kCudaParticleCouplingSlotsPerParticle`

**Step 3: Populate rows in the coupling kernel**

When `SolveDeviceWorldShape()` produces a contact and has a valid slot:
- write active row data into the slot
- store particle id, shape id, body id
- store contact normal, contact point, particle/body Jacobians, rhs, effective mass, position error, and accumulated normal impulse
- clear untouched rows on device with invalid ids and zero row data

**Step 4: Verify runtime test**

Run the focused test until it passes.

## Task 2: Benchmark and Demo Visibility

**Files:**
- Modify: `tests/perf/test_cuda_particle_coupling_timing.cpp`
- Modify: `src/apps/cuda_particle_demo/main.cpp`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/plans/2026-05-23-cuda-physics-solver-roadmap.md`
- Modify: `docs/testing/p0-regression-matrix.md`

**Step 1: Add benchmark assertions**

Extend `DeviceWorldMultiSlotDiagnosticsUnderOneSecond` to download row state and verify active rows for thousands of particles under the existing one-second budget.

**Step 2: Demo output**

Print corner row diagnostics: active row count, effective mass, position error, and normal impulse for slot 0 and slot 1.

**Step 3: Docs**

Document that the coupling cache now includes CUDA-resident row records, while the unified solver promotion remains follow-on work.

## Task 3: Verification and Commit

Run:

```powershell
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_cuda_runtime_test -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_cuda_perf_test -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_cuda_particle_demo -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'
ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure -R "CudaParticleWorld|CudaParticleCouplingTiming"
C:\Softwares\code\nuka-physics\build\src\Release\nuka_cuda_particle_demo.exe
git diff --check
```

Then run full CTest before reporting final state:

```powershell
ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure
```

Commit:

```powershell
git add docs src tests
git commit -m "physics: assemble cuda coupling constraint rows"
```
