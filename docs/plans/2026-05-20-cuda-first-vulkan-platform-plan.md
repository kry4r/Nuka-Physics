# CUDA-First Vulkan Platform Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move Nuka Physics from a CPU-reference runtime toward the required CUDA-preferred production physics path and Vulkan production rendering path while preserving an explicit backend selection API.

**Architecture:** PHI owns backend selection, CUDA device capability checks, buffers, streams, and future kernel dispatch. Runtime keeps the existing CPU `StepWorldInstance()` as a deterministic reference implementation, then adds a CUDA world stepper with matching reports so every migrated stage can be compared against reference output. Rendering treats Vulkan as the production backend; the PPM debug renderer remains a CI artifact path.

**Tech Stack:** C++20, CUDA, Vulkan, CMake, GoogleTest.

---

### Task 1: Backend Contract and Build Policy

**Status:** Completed. Commit `3b8a71d` added the CUDA-first platform contract,
backend selector, Vulkan renderer probe, tests, and documentation.

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `src/CMakeLists.txt`
- Create: `src/phi/platform_contract.hpp`
- Modify: `src/phi/backend_cuda/device.cu`
- Create: `src/render/vulkan_renderer.hpp`
- Create: `src/render/vulkan_renderer.cpp`
- Create: `tests/phi/test_platform_contract.cpp`
- Create: `tests/render/test_vulkan_backend.cpp`
- Modify: `tests/CMakeLists.txt`

**Validation:**

```powershell
cmake --build build --config Release --target nuka_phi_test nuka_render_test
ctest --test-dir build -C Release --output-on-failure -R "PlatformContract|Phi|VulkanRenderer"
```

**Expected:** CUDA is the default production physics backend, explicit CPU selection is reference-only, and Vulkan can create an instance and enumerate a physical device.

### Task 2: CUDA Device World State

**Status:** Completed. Commit `00dd1f8` added CUDA-resident cooked runtime table
uploads, compact readback validation, and the `nuka_runtime_gpu` target.

**Files:**
- Create: `src/runtime/gpu/device_world.hpp`
- Create: `src/runtime/gpu/device_world.cu`
- Create: `tests/runtime/test_cuda_device_world.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Scope:** Upload cooked body poses, velocities, inverse masses, inertias, shape records, joint records, actuator records, and material tables into typed PHI/CUDA buffers with ownership separated from host `WorldTemplate`.

**Validation:** Upload a cooked imported scene, download a compact checksum/state snapshot, and compare body/shape/joint/actuator counts and selected table values with the CPU reference world.

### Task 3: CUDA Rigid Integration Kernel

**Status:** Completed in the current iteration. `DeviceWorld` now owns mutable
state buffers, exposes validation readback, and `StepCudaWorld()` runs fixed-step
rigid integration on CUDA. The build now forces `CMAKE_CUDA_ARCHITECTURES` from
`NK_CUDA_ARCHITECTURES`, defaulting to `native`, so this workstation compiles
kernels for the installed GPU instead of stale cached architectures.

**Files:**
- Create: `src/runtime/gpu/cuda_world_stepper.hpp`
- Create: `src/runtime/gpu/cuda_world_stepper.cu`
- Create: `tests/runtime/test_cuda_world_stepper.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Scope:** Implement gravity, force integration, velocity integration, accumulator clearing, and fixed-step loop on CUDA buffers. Preserve `WorldStepReport` fields that are meaningful before contacts.

**Validation:** Run free-fall and forced-body scenes through CUDA and CPU reference paths, compare poses/velocities within deterministic tolerances, and record timing.

Current validation command:

```powershell
ctest --test-dir build -C Release --output-on-failure -R "CudaWorldStepper|CudaDeviceWorld|CudaStepTiming"
```

Current result: 5/5 CUDA runtime and CUDA timing tests pass on the local CUDA
path.

### Task 4: CUDA Broadphase and Narrowphase

**Status:** Completed in the current iteration. CUDA now owns cooked shape
tables in `DeviceWorld`, generates shape AABBs and deterministic broadphase
pair slots on device, and emits plane, sphere-sphere, and box-style contact
manifolds into device buffers. CPU readback is used only for validation,
regression comparisons, and timing reports.

**Files:**
- Create: `src/collision/gpu/broadphase.cuh`
- Create: `src/collision/gpu/broadphase.cu`
- Create: `src/constraint/gpu/contact_generation.cuh`
- Create: `src/constraint/gpu/contact_generation.cu`
- Create: `tests/runtime/test_cuda_contacts.cpp`
- Create: `tests/perf/test_cuda_contact_timing.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `README.md`

**Scope:** Generate AABBs, broadphase candidate pairs, and plane/sphere/box contact manifolds on CUDA. Keep deterministic ordering for regression comparisons.

**Validation:** Compare contact pair counts, manifold counts, contact normals, penetration depths, and contact points against CPU reference scenes.

Current validation commands:

```powershell
ctest --test-dir build -C Release --output-on-failure -R "CudaContacts|CudaWorldStepper|CudaDeviceWorld|CudaStepTiming|CudaContactTiming"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Current result: focused CUDA runtime/performance checks pass 9/9, and the full
Release regression suite passes 211/211 on the local CUDA/Vulkan workstation.

### Task 5: CUDA Constraint Assembly and Solver

**Status:** Completed in the current iteration. CUDA now assembles contact,
joint, and drive constraint blocks from device-resident contact results and
cooked `DeviceWorld` tables, then runs deterministic PGS velocity iterations
plus contact/joint position projection on CUDA pose and velocity buffers.
Host-authored block upload is retained for validation-only differential tests.

**Files:**
- Create: `src/solver/gpu/cuda_constraint_solver.cuh`
- Create: `src/solver/gpu/cuda_constraint_solver.cu`
- Create: `tests/runtime/test_cuda_solver.cpp`
- Create: `tests/perf/test_cuda_solver_timing.cpp`
- Modify: `src/runtime/gpu/device_world.hpp`
- Modify: `src/runtime/gpu/device_world.cu`
- Modify: `src/constraint/gpu/contact_generation.cuh`
- Modify: `src/constraint/gpu/contact_generation.cu`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `README.md`

**Scope:** Assemble contact, friction, restitution, joint, and actuator rows into device-resident solver buffers and run deterministic PGS iterations on CUDA.

**Validation:** Compare resting contact, friction, restitution, joint projection, and velocity drive scenes against CPU reference metrics. Report constraint error, penetration, energy drift, solver iteration count, and GPU timing.

Current validation commands:

```powershell
ctest --test-dir build -C Release --output-on-failure -R "CudaConstraintSolver|CudaContacts|CudaWorldStepper|CudaDeviceWorld|CudaStepTiming|CudaContactTiming|CudaSolverTiming"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

### Task 6: End-to-End CUDA Scene Demo and Benchmark

**Files:**
- Modify: `src/apps/debug_shell/scene_demo.cpp`
- Create: `tests/perf/test_cuda_step_timing.cpp`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `README.md`

**Scope:** Route imported-scene simulation through backend selection. Default demo path uses CUDA on this workstation, then synchronizes state to `SceneGraph`/`RenderScene` for debug visualization and Vulkan rendering.

**Validation:** Run MJCF and USDA examples through import -> cook -> CUDA simulation -> render/debug sync. Compare against CPU reference snapshots and record GPU timing versus current CPU reference benchmarks.
