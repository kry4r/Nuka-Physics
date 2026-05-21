# CUDA Batched Sensor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CUDA-resident batched sensor queries so Isaac Lab-style parallel environments can sample IMU/state and lidar observations from `BatchedDeviceWorld` without CPU simulation or CPU sensor fallback.

**Architecture:** Reuse the existing `sensor::gpu` layer and add overloads that consume `runtime::gpu::BatchedDeviceWorld`. IMU queries take `(instance, body)` requests and flatten valid body ids with `instance * body_count_per_instance + body`. Lidar queries take one options record per environment, reuse shared shape tables, and search only shapes inside the requested instance. Downloads remain validation/tooling boundaries; production observations stay on CUDA buffers until explicitly read back.

**Tech Stack:** C++20, CUDA, CMake, GoogleTest.

---

### Task 1: Batched CUDA IMU Samples

**Files:**
- Modify: `src/sensor/gpu/cuda_sensors.cuh`
- Modify: `src/sensor/gpu/cuda_sensors.cu`
- Modify: `tests/sensor/test_cuda_sensors.cpp`

**Step 1: Write failing test**

Add `CudaSensor.SamplesBatchedImuFromBatchedDeviceWorldState`. Build a shared two-body template, upload two `WorldInstance` objects with different poses, angular velocities, and forces, then call:

```cpp
sensor::gpu::QueryBatchedCudaImuSensor(batch, {{0u, body}, {1u, body}});
```

Verify two samples are returned, each matching the corresponding flattened CUDA state and inverse mass.

**Step 2: Run red**

```powershell
cmake --build build --config Release --target nuka_cuda_sensor_test
```

Expected: build fails because `BatchedCudaBodyRequest` and `QueryBatchedCudaImuSensor()` do not exist.

**Step 3: Implement minimal CUDA IMU path**

Add:
- `BatchedCudaBodyRequest`
- `QueryBatchedCudaImuSensor(const runtime::gpu::BatchedDeviceWorld&, const std::vector<BatchedCudaBodyRequest>&)`
- a CUDA kernel that reads flattened pose/angular velocity/force and shared inverse mass tables.

**Step 4: Run green**

```powershell
cmake --build build --config Release --target nuka_cuda_sensor_test
ctest --test-dir build -C Release --output-on-failure -R "CudaSensor"
```

Expected: existing single-world sensor tests and new batched IMU test pass.

### Task 2: Batched CUDA Lidar Samples

**Files:**
- Modify: `src/sensor/gpu/cuda_sensors.cuh`
- Modify: `src/sensor/gpu/cuda_sensors.cu`
- Modify: `tests/sensor/test_cuda_sensors.cpp`

**Step 1: Write failing test**

Add `CudaSensor.BatchedLidarQueriesStayWithinEachInstance`. Upload two environments sharing a ground+sphere template, offset one sphere far away, then query one lidar fan per instance:

```cpp
sensor::gpu::QueryBatchedCudaLidarSensor(batch, options);
```

Verify each returned fan reads only its own instance's geometry and does not hit shapes from another instance.

**Step 2: Run red**

```powershell
cmake --build build --config Release --target nuka_cuda_sensor_test
```

Expected: build fails because `BatchedCudaLidarOptions`, `BatchedCudaLidarResult`, and `QueryBatchedCudaLidarSensor()` do not exist.

**Step 3: Implement minimal CUDA lidar path**

Add:
- `BatchedCudaLidarOptions`
- `BatchedCudaLidarResult`
- a CUDA kernel where each flattened ray belongs to exactly one lidar query and only loops over `shape_count_per_instance` shapes in that query's `instance_index`.

**Step 4: Run green**

```powershell
cmake --build build --config Release --target nuka_cuda_sensor_test
ctest --test-dir build -C Release --output-on-failure -R "CudaSensor"
```

Expected: batched lidar test passes and proves no cross-instance hits.

### Task 3: Timing, Docs, Workflow Evidence, Commit

**Files:**
- Create: `tests/perf/test_cuda_batch_sensor_timing.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `README.md`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `docs/plans/2026-05-21-cuda-batched-sensor-plan.md`

**Step 1: Add benchmark**

Add `CudaBatchSensorTiming.BatchedImuAndLidarQueriesUnderOneSecond` with at least 128 environments, 128 IMU body requests, 128 lidar fans, and 64 rays per fan.

**Step 2: Update docs**

Document that batched CUDA now covers rigid integration, contacts, joint projection, velocity drives, IMU/state samples, and lidar observations. Keep remaining gap explicit: GPU-to-Vulkan batched render synchronization / richer Vulkan renderer features.

**Step 3: Full verification**

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

Expected: full Release build and all tests pass.

**Step 4: Commit**

```powershell
git add src/sensor/gpu/cuda_sensors.* tests/sensor/test_cuda_sensors.cpp tests/perf/test_cuda_batch_sensor_timing.cpp tests/CMakeLists.txt README.md docs/architecture/runtime-overview.md docs/testing/p0-regression-matrix.md docs/plans/2026-05-21-cuda-batched-sensor-plan.md
git commit -m "sensor: add cuda batched sensor queries"
```

---

## Result

Implemented batched CUDA sensor queries for `runtime::gpu::BatchedDeviceWorld`.

- `sensor::gpu::QueryBatchedCudaImuSensor()` samples `(instance_index, body_id)`
  requests from flattened CUDA pose/angular-velocity/force buffers while reading
  shared inverse mass from the cooked template table.
- `sensor::gpu::QueryBatchedCudaLidarSensor()` accepts one lidar fan per
  environment, flattens all rays into one CUDA launch, and only scans shape rows
  inside each requested instance.
- `BatchedDeviceWorld` now exposes const device accessors for mutable velocity,
  force, and torque buffers so read-only CUDA observation paths do not require a
  mutable world reference.
- Added a 128-environment timing regression for 128 IMU samples and 128 lidar
  fans with 64 rays each.

Verification performed during implementation:

```powershell
cmake --build build --config Release --target nuka_cuda_sensor_test
ctest --test-dir build -C Release --output-on-failure -R "CudaSensor"
cmake --build build --config Release --target nuka_cuda_perf_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchSensorTiming"
```

Observed targeted results:

- `CudaSensor` filter: 5/5 passed in 0.50 s.
- `CudaBatchSensorTiming.BatchedImuAndLidarQueriesUnderOneSecond`: passed in
  0.13 s test time / 0.19 s total CTest time.

Remaining global CUDA-first follow-up:

- Keep production simulation on CUDA and CPU only as reference/import/cooking/
  orchestration.
- Connect batched GPU state/observations into the Vulkan render/debug
  synchronization path without turning CPU simulation into the hot path.
