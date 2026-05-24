# CUDA Physics Solver Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the current `nuka-physics` focus to a CUDA-first full-GPU physics core for embodied robotics, with rendering treated only as a validation boundary during this phase.

**Architecture:** The backend selection layer remains intact, but production physics state and solver work must live in CUDA buffers. CPU code is limited to import/cook, host orchestration, and reference validation. The near-term extension adds CUDA-resident particle/deformable state and rigid-collider coupling so the existing rigid robot solver has a real soft/fluid foundation to connect to in later stages.

**Tech Stack:** C++20, CUDA, PHI `Buffer`, existing `DeviceWorld`/`BatchedDeviceWorld`, GoogleTest, CTest, MSVC via `vcvars64.bat`.

---

## Phase Direction

Current renderer expansion is not the active line of work. Vulkan remains the production renderer direction, but solver architecture, GPU residency, diagnostics, and rigid/deformable/fluid coupling are the priority.

The CUDA physics core must progress as a coherent system:

1. Rigid robot dynamics stay on the existing `DeviceWorld` and `BatchedDeviceWorld` paths.
2. Particle/deformable/fluid state receives a CUDA-resident container, upload/download validation boundary, integration kernel, coupling kernels, and diagnostics.
3. Coupling starts with analytic rigid plane/sphere colliders, then moves to cooked `DeviceWorld` shape tables so robot bodies can couple to deformables and fluids without CPU stepping. Current cooked-shape coverage includes plane, sphere, box, and capsule.
4. Solver reports must expose invariants: contact count, max penetration, kinetic energy, max speed, kernel launches, and later constraint residuals.
5. Every stage requires regression tests, benchmark coverage, architecture docs, and a commit.

## Task 1: CUDA Particle Coupling Foundation

**Files:**
- Create: `src/runtime/gpu/cuda_particle_world.hpp`
- Create: `src/runtime/gpu/cuda_particle_world.cu`
- Create: `tests/runtime/test_cuda_particle_world.cpp`
- Create: `tests/perf/test_cuda_particle_coupling_timing.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/testing/p0-regression-matrix.md`

**Step 1: Write the failing tests**

Create CUDA runtime tests that require:
- Uploading particle positions, velocities, inverse masses, radii, and phases into device buffers.
- Downloading compact state for validation.
- Stepping particles on CUDA with gravity.
- Enforcing plane and sphere rigid-collider coupling on CUDA.
- Reporting particle count, contact count, max penetration, max speed, kinetic energy, and kernel launches.

Run:

```powershell
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_cuda_runtime_test -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'
```

Expected: build fails because `runtime/gpu/cuda_particle_world.hpp` does not exist yet.

**Step 2: Implement the CUDA particle world**

Add `runtime::gpu::CudaParticleWorld`, `UploadCudaParticleWorld()`, and `StepCudaParticleWorld()`.

Required behavior:
- Store SoA particle state in PHI device buffers.
- Keep dynamic particles fully on GPU during stepping.
- Treat static particles as `inv_mass <= 0`.
- Integrate velocity and position on CUDA.
- Resolve analytic plane and sphere contacts on CUDA.
- Generate device-side diagnostics and download only compact validation reports.

**Step 3: Verify the runtime tests**

Run:

```powershell
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure -R "CudaParticleWorld"'
```

Expected: all `CudaParticleWorld` tests pass.

**Step 4: Add benchmark coverage**

Create a benchmark that steps thousands of CUDA particles against plane/sphere rigid colliders for repeated substeps, plus cooked `DeviceWorld` sphere, box, and capsule coupling with rigid impulse feedback.

Run:

```powershell
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure -R "CudaParticleCouplingTiming"'
```

Expected: benchmark stays under one second and reports nonzero contact diagnostics.

**Step 5: Update docs and regression matrix**

Document the new CUDA-resident particle/deformable/fluid foundation in `runtime-overview.md` and add the tests/benchmark to `p0-regression-matrix.md`.

**Step 6: Broader verification and commit**

Run focused CUDA targets:

```powershell
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build C:\Softwares\code\nuka-physics\build --config Release --target nuka_cuda_runtime_test nuka_cuda_perf_test -- /m:1 /nodeReuse:false /p:TrackFileAccess=false /p:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\\" /v:m'
cmd.exe /d /s /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && ctest --test-dir C:\Softwares\code\nuka-physics\build -C Release --output-on-failure -R "Cuda(ParticleWorld|ParticleCouplingTiming|BatchedWorld|ConstraintSolver|Sensor)"'
```

Commit:

```powershell
git add docs src tests
git commit -m "physics: add cuda particle coupling foundation"
```

## Current Progress

- CUDA particle/DeviceWorld coupling now supports cooked plane, sphere, box, and
  capsule shapes with rigid linear/angular impulse feedback.
- `CudaParticleWorld` owns CUDA-resident per-particle coupling normal impulse
  and shape-index caches with four cooked-shape slots per particle. The cooked
  coupling kernel writes each touched slot, reports active slot count,
  warm-start count/magnitude, force magnitude, and torque magnitude, and clears
  stale cache entries when contact separates.
- `nuka_cuda_particle_demo` prints warm-start/cache diagnostics for sphere, box,
  capsule, and two-box corner coupling, so single-particle multi-contact cache
  behavior is visible outside unit tests.
- `CudaParticleCouplingTiming.DeviceWorldWarmStartDiagnosticsUnderOneSecond`
  tracks the added cache/report path under the existing one-second CUDA budget.
- `CudaParticleCouplingTiming.DeviceWorldMultiSlotDiagnosticsUnderOneSecond`
  tracks the multi-slot contact-cache and force/torque diagnostic path with
  thousands of particles under the same one-second CUDA budget.

## Follow-On Tasks

1. Promote the fixed per-particle coupling slots into a reusable
   coupling-constraint assembly path shared by rigid/cloth/deformable/fluid
   constraints, including friction rows and constraint-space Jacobians.
2. Add cloth/deformable constraints: distance, bending, volume/shape matching,
   and XPBD-style compliance.
3. Add particle-fluid constraints: density estimate, pressure/viscosity solve,
   boundary coupling, and diagnostics.
4. Add batched particle worlds for robot/RL workloads that match
   `BatchedDeviceWorld` instance-major layout.
