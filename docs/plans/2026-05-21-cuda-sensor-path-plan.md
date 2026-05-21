# CUDA Sensor Path Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move runtime sensor queries onto the CUDA production path so imported scenes can simulate, solve, and sample state/ray sensors from GPU-resident world data.

**Architecture:** Keep existing CPU `sensor` functions as reference-only helpers. Add a `sensor::gpu` layer that consumes `runtime::gpu::DeviceWorld` after CUDA simulation, launches CUDA kernels for IMU/state and lidar depth sampling, and exposes compact readback reports for tests and demos. The first lidar path performs deterministic ray tests against cooked sphere, box-style, and plane shapes from `DeviceWorld`; later stages can replace it with BVH or RT acceleration without changing the public sensor API.

**Tech Stack:** C++20, CUDA, CMake, GoogleTest.

---

### Task 1: CUDA IMU/State Sensor Query - Completed

**Files:**
- Create: `src/sensor/gpu/cuda_sensors.cuh`
- Create: `src/sensor/gpu/cuda_sensors.cu`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Create: `tests/sensor/test_cuda_sensors.cpp`

**Scope:** Add `CudaImuSample`, `CudaImuResult`, and `QueryCudaImuSensor()`.
The kernel reads device poses, angular velocities, forces, and inverse masses
for requested body ids and writes position, angular velocity, and simplified
linear acceleration on device. Tests compare CUDA readback against the CPU
reference `QueryImuSensor()` for multiple bodies.

**Validation:**

```powershell
cmake --build build --config Release --target nuka_cuda_sensor_test
ctest --test-dir build -C Release --output-on-failure -R "CudaSensor"
```

**Result:** Implemented `sensor::gpu::QueryCudaImuSensor()` over CUDA
`DeviceWorld` pose, angular velocity, force, and inverse-mass buffers. The
test failed before the API existed, then passed with CUDA samples matching CPU
reference values.

### Task 2: CUDA Lidar/Depth Query - Completed

**Files:**
- Modify: `src/sensor/gpu/cuda_sensors.cuh`
- Modify: `src/sensor/gpu/cuda_sensors.cu`
- Modify: `tests/sensor/test_cuda_sensors.cpp`
- Create: `tests/perf/test_cuda_sensor_timing.cpp`
- Modify: `tests/CMakeLists.txt`

**Scope:** Add `CudaLidarOptions`, `CudaLidarResult`, and
`QueryCudaLidarSensor()`. The kernel casts a deterministic fan of rays from an
origin/direction basis and writes nearest depth against cooked sphere, box-style
AABB, and plane shapes in `DeviceWorld`; misses return max range. Timing tests
cover a dense ray set over an imported/cooked scene.

**Validation:**

```powershell
cmake --build build --config Release --target nuka_cuda_sensor_test nuka_cuda_perf_test
ctest --test-dir build -C Release --output-on-failure -R "CudaSensor|CudaSensorTiming"
```

**Result:** Implemented deterministic CUDA lidar ray sampling against cooked
sphere, box-style AABB, and plane geometry. Focused CUDA sensor correctness and
timing tests passed in Release.

### Task 3: Documentation, Demo Integration, and Full Verification - Completed

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `docs/plans/2026-05-21-cuda-sensor-path-plan.md`
- Modify: `src/apps/debug_shell/scene_demo.hpp`
- Modify: `src/apps/debug_shell/scene_demo.cpp`
- Modify: `src/apps/debug_shell/main.cpp`
- Modify: `tests/apps/test_scene_demo.cpp`

**Scope:** Document CPU sensors as reference-only, CUDA sensor query as the
production path, add P0 regression/timing matrix entries, and wire imported
IMU/frame-pose sensors into the default CUDA + Vulkan scene demo report so this
stage has a workflow-level validation path instead of isolated unit coverage.

**Validation:**

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

**Expected:** full Release build and all tests pass. Commit as
`sensor: add cuda sensor query path`.

**Focused validation completed before full verification:**

```powershell
cmake --build build --config Release --target nuka_cuda_sensor_test nuka_cuda_perf_test
ctest --test-dir build -C Release --output-on-failure -R "CudaSensor|CudaSensorTiming"
cmake --build build --config Release --target nuka_debug_draw_test
ctest --test-dir build -C Release --output-on-failure -R "SceneDemo\.DefaultsImportedSceneSimulationToCudaBackendWhenAvailable|CudaSensor|CudaSensorTiming"
```

**Focused results:** CUDA sensor correctness, CUDA sensor timing, and imported
scene demo CUDA sensor report checks passed.
