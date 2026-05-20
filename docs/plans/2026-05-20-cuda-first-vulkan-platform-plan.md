# CUDA-First Vulkan Platform Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move Nuka Physics from a CPU-reference runtime toward the required CUDA-preferred production physics path and Vulkan production rendering path while preserving an explicit backend selection API.

**Architecture:** PHI owns backend selection, CUDA device capability checks, buffers, streams, and future kernel dispatch. Runtime keeps the existing CPU `StepWorldInstance()` as a deterministic reference implementation, then adds a CUDA world stepper with matching reports so every migrated stage can be compared against reference output. Rendering treats Vulkan as the production backend; the PPM debug renderer remains a CI artifact path.

**Tech Stack:** C++20, CUDA, Vulkan, CMake, GoogleTest.

---

### Task 1: Backend Contract and Build Policy

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

**Files:**
- Create: `src/runtime/gpu/device_world.hpp`
- Create: `src/runtime/gpu/device_world.cu`
- Create: `tests/runtime/test_cuda_device_world.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Scope:** Upload cooked body poses, velocities, inverse masses, inertias, shape records, joint records, actuator records, and material tables into typed PHI/CUDA buffers with ownership separated from host `WorldTemplate`.

**Validation:** Upload a cooked imported scene, download a compact checksum/state snapshot, and compare body/shape/joint/actuator counts and selected table values with the CPU reference world.

### Task 3: CUDA Rigid Integration Kernel

**Files:**
- Create: `src/runtime/gpu/cuda_world_stepper.hpp`
- Create: `src/runtime/gpu/cuda_world_stepper.cu`
- Create: `tests/runtime/test_cuda_world_stepper.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Scope:** Implement gravity, force integration, velocity integration, accumulator clearing, and fixed-step loop on CUDA buffers. Preserve `WorldStepReport` fields that are meaningful before contacts.

**Validation:** Run free-fall and forced-body scenes through CUDA and CPU reference paths, compare poses/velocities within deterministic tolerances, and record timing.

### Task 4: CUDA Broadphase and Narrowphase

**Files:**
- Create: `src/collision/gpu/broadphase.cuh`
- Create: `src/collision/gpu/broadphase.cu`
- Create: `src/constraint/gpu/contact_generation.cuh`
- Create: `src/constraint/gpu/contact_generation.cu`
- Create: `tests/runtime/test_cuda_contacts.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Scope:** Generate AABBs, broadphase candidate pairs, and plane/sphere/box contact manifolds on CUDA. Keep deterministic ordering for regression comparisons.

**Validation:** Compare contact pair counts, manifold counts, contact normals, penetration depths, and contact points against CPU reference scenes.

### Task 5: CUDA Constraint Assembly and Solver

**Files:**
- Create: `src/solver/gpu/cuda_constraint_solver.cuh`
- Create: `src/solver/gpu/cuda_constraint_solver.cu`
- Create: `tests/runtime/test_cuda_solver.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Scope:** Assemble contact, friction, restitution, joint, and actuator rows into device-resident solver buffers and run deterministic PGS iterations on CUDA.

**Validation:** Compare resting contact, friction, restitution, joint projection, and velocity drive scenes against CPU reference metrics. Report constraint error, penetration, energy drift, solver iteration count, and GPU timing.

### Task 6: End-to-End CUDA Scene Demo and Benchmark

**Files:**
- Modify: `src/apps/debug_shell/scene_demo.cpp`
- Create: `tests/perf/test_cuda_step_timing.cpp`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `README.md`

**Scope:** Route imported-scene simulation through backend selection. Default demo path uses CUDA on this workstation, then synchronizes state to `SceneGraph`/`RenderScene` for debug visualization and Vulkan rendering.

**Validation:** Run MJCF and USDA examples through import -> cook -> CUDA simulation -> render/debug sync. Compare against CPU reference snapshots and record GPU timing versus current CPU reference benchmarks.
