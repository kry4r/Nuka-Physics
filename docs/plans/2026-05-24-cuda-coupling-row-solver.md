# CUDA Coupling Row Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make CUDA particle/rigid coupling rows execute as a GPU solver path, not only as downloadable diagnostics.

**Architecture:** Keep contact detection and position projection in the particle coupling kernel for this stage, but move normal velocity impulse application into a separate CUDA row-solver kernel that consumes `CudaParticleCouplingConstraintRow` records. The row solver updates particle velocities, rigid linear/angular velocities, row normal impulses, warm-start cache values, and solver diagnostics on device, while CPU only launches kernels and downloads compact validation state. The implementation sweeps each particle's fixed coupling slots in one CUDA thread so multiple active contacts on one particle accumulate into the same particle velocity deterministically at particle scope; shared rigid-body updates still use atomics until the unified scheduler adds global coloring or island batching.

**Tech Stack:** C++20, CUDA, PHI `Buffer`, existing `DeviceWorld`, GoogleTest, CTest, MSVC via `vcvars64.bat`.

---

## Task 1: Row Solver Runtime Contract

**Files:**
- Modify: `src/runtime/gpu/cuda_particle_world.hpp`
- Modify: `tests/runtime/test_cuda_particle_world.cpp`

**Step 1: Write the failing test**

Add `CudaParticleWorld.ExecutesDeviceWorldCouplingRowsWithCudaSolverKernel`.

The test builds one dynamic cooked box and one particle moving into the box. It runs `StepCudaParticlesAgainstDeviceWorld()` with row solving enabled and verifies:
- `report.coupling_row_solver_launch_count > 0`
- `report.coupling_row_solver_impulse_count == 1`
- downloaded row slot 0 is active
- row slot 0 has positive `normal_impulse`
- particle and rigid contact normal relative speed is approximately zero after the step
- rigid linear and angular velocities changed through the row path

Run:

```powershell
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_cuda_runtime_test -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'
ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure -R "CudaParticleWorld\.ExecutesDeviceWorldCouplingRowsWithCudaSolverKernel"
```

Expected: build fails because the report fields and option do not exist.

## Task 2: CUDA Row Solver Kernel

**Files:**
- Modify: `src/runtime/gpu/cuda_particle_world.hpp`
- Modify: `src/runtime/gpu/cuda_particle_world.cu`

**Step 1: Add explicit options and report fields**

Add:
- `CudaParticleDeviceWorldCouplingOptions::solve_coupling_rows_on_cuda = true`
- `CudaParticleStepReport::coupling_row_solver_launch_count`
- `CudaParticleStepReport::coupling_row_solver_impulse_count`
- `CudaParticleStepReport::coupling_row_solver_impulse_magnitude`

**Step 2: Convert contact kernel to assemble solver-ready rows**

When row solving is enabled:
- project particle position as before
- assemble row data
- skip direct velocity and rigid impulse application inside `SolveDeviceWorldShape()`
- write `row.rhs = 0.0f` for non-restitution contact rows, matching existing `ConstraintBlock` PGS convention
- seed `row.normal_impulse` from warm-start cache only when matching shape

When row solving is disabled:
- preserve the previous direct coupling behavior for transition testing only.

**Step 3: Add row solver kernel**

Launch `SolveParticleCouplingRowsKernel` after row assembly each step. For every particle, sweep the fixed coupling slots in row order. For every active row:
- recompute `jv` from particle velocity plus rigid contact velocity
- compute `delta = effective_mass * (rhs - jv)`
- clamp accumulated normal impulse to non-negative
- apply the delta to the particle velocity accumulator and rigid linear/angular velocities using row Jacobians
- update row normal impulse and matching warm-start cache slot
- write per-row diagnostics for reduction

Add `CudaParticleWorld.AccumulatesMultipleCudaCouplingRowsIntoOneParticleVelocity` to verify one particle with two active cooked contacts receives both row impulses in its downloaded velocity.

**Step 4: Reduce row solver diagnostics**

Add a device diagnostics buffer for row solver results and fold it into `CudaParticleStepReport`.

## Task 3: Benchmark, Demo, Docs, Commit

**Files:**
- Modify: `tests/perf/test_cuda_particle_coupling_timing.cpp`
- Modify: `src/apps/cuda_particle_demo/main.cpp`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/plans/2026-05-23-cuda-physics-solver-roadmap.md`
- Modify: `docs/testing/p0-regression-matrix.md`

**Step 1: Benchmark assertions**

Extend existing particle coupling timing tests to assert row-solver launch and impulse diagnostics stay under the existing one-second budgets.

**Step 2: Demo output**

Print row solver launches, impulse count, and impulse magnitude in `nuka_cuda_particle_demo`.

**Step 3: Documentation**

Document that coupling rows now have an executing CUDA normal-solver path. Explicitly leave friction rows, compliance, and unified rigid/cloth/deformable/fluid row iteration as follow-on work.

**Step 4: Verification**

Run:

```powershell
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_cuda_runtime_test -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_cuda_perf_test -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_cuda_particle_demo -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'
ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure -R "CudaParticleWorld|CudaParticleCouplingTiming"
C:\Softwares\code\nuka-physics\build\src\Release\nuka_cuda_particle_demo.exe
git diff --check
ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure
```

Commit:

```powershell
git add docs src tests
git commit -m "physics: execute cuda coupling rows"
```
