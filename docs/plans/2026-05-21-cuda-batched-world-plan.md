# CUDA Batched World Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a CUDA production path for stepping multiple instances of the same cooked scene in one GPU-resident batched world.

**Architecture:** Keep CPU batching as metadata/reference only. Add a CUDA `BatchedDeviceWorld` that reuses one cooked `WorldTemplate`, stores per-instance mutable state in flattened device arrays, and launches CUDA kernels across `instance_count * body_count`. The first stage batches rigid integration; later stages can extend the same layout to broadphase, contacts, constraints, sensors, and render synchronization.

**Tech Stack:** C++20, CUDA, CMake, GoogleTest.

---

### Task 1: CUDA Batched State Upload - Completed

**Files:**
- Create: `src/runtime/gpu/batched_device_world.hpp`
- Create: `src/runtime/gpu/batched_device_world.cu`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Create: `tests/runtime/test_cuda_batched_world.cpp`

**Step 1: Write failing upload test**

Add `CudaBatchedWorld.UploadsMultipleInstancesOfSharedTemplate`:

```cpp
const auto blob = scene::CookScene(BuildTwoBodyScene());
auto world = runtime::BuildWorld(blob);
std::vector<runtime::WorldInstance> instances(3, world.instance);
instances[1].poses[0].position.x += 10.0f;
instances[2].linear_velocities[1] = {0.0f, 2.0f, 0.0f};

auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);
const auto snapshot = batch.DownloadState();

EXPECT_EQ(snapshot.instance_count, 3u);
EXPECT_EQ(snapshot.body_count_per_instance, 2u);
EXPECT_NEAR(snapshot.poses[2].position.x, 11.0f, 1.0e-5f);
EXPECT_NEAR(snapshot.linear_velocities[5].y, 2.0f, 1.0e-5f);
```

**Step 2: Verify red**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
```

Expected: build fails because `batched_device_world.hpp` and `UploadBatchedDeviceWorld()` do not exist.

**Step 3: Implement minimal batched upload**

Create `BatchedDeviceWorld` with:
- `InstanceCount()`
- `BodyCountPerInstance()`
- `TotalBodyCount()`
- flattened pose, linear velocity, angular velocity, force, and torque buffers
- reused body inverse mass and inverse inertia buffers copied from `WorldTemplate`
- `DownloadState()`

**Step 4: Verify green**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld"
```

Expected: upload test passes.

**Result:** Added `runtime::gpu::BatchedDeviceWorld`, flattened multi-instance
state upload, and state download validation. The upload test passed after the
implementation.

### Task 2: CUDA Batched Rigid Integration - Completed

**Files:**
- Modify: `src/runtime/gpu/batched_device_world.hpp`
- Modify: `src/runtime/gpu/batched_device_world.cu`
- Modify: `tests/runtime/test_cuda_batched_world.cpp`

**Step 1: Write failing integration test**

Add `CudaBatchedWorld.StepsInstancesIndependentlyOnDevice`:

```cpp
runtime::gpu::CudaBatchedWorldStepOptions options;
options.gravity = {0.0f, -10.0f, 0.0f};
options.dt = 0.25f;
options.step_count = 2u;
options.clear_forces_after_step = true;

const auto report = runtime::gpu::StepBatchedCudaWorld(batch, options);
const auto state = batch.DownloadState();

EXPECT_EQ(report.instance_count, 3u);
EXPECT_EQ(report.total_body_count, 6u);
EXPECT_EQ(report.kernel_launch_count, 2u);
// compare each instance against a CPU reference step with contacts disabled
```

**Step 2: Verify red**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
```

Expected: build fails because `StepBatchedCudaWorld()` does not exist.

**Step 3: Implement batched integration kernel**

Launch one CUDA thread per flattened body. Compute:

```cpp
body_index = flat_index % body_count_per_instance;
```

Use template inverse mass/inertia for that body index and mutate the flattened per-instance state slot.

**Step 4: Verify green**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld"
```

Expected: batched CUDA state matches CPU reference integration for each instance with contacts disabled.

**Result:** Added `StepBatchedCudaWorld()` with one CUDA thread per flattened
body. The batched state matched CPU reference integration for each instance with
contacts disabled.

### Task 3: Timing, Documentation, and Commit - Completed

**Files:**
- Create: `tests/perf/test_cuda_batch_timing.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `README.md`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `docs/plans/2026-05-21-cuda-batched-world-plan.md`

**Step 1: Add timing test**

Add `CudaBatchTiming.BatchedRigidIntegrationUnderOneSecond` with 256 instances,
32 bodies per instance, 120 CUDA steps, and a `< 1000 ms` threshold.

**Step 2: Verify focused timing**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test nuka_cuda_perf_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld|CudaBatchTiming"
```

Expected: correctness and timing tests pass.

**Step 3: Update docs**

Document CUDA batched world as the production path for multi-environment rigid
integration and explicitly keep CPU batch scheduler as reference/orchestration metadata.

**Step 4: Full verification**

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

Expected: full Release build and all tests pass.

**Step 5: Commit**

```powershell
git add src/runtime/gpu/batched_device_world.* src/CMakeLists.txt tests/CMakeLists.txt tests/runtime/test_cuda_batched_world.cpp tests/perf/test_cuda_batch_timing.cpp README.md docs/architecture/runtime-overview.md docs/testing/p0-regression-matrix.md docs/plans/2026-05-21-cuda-batched-world-plan.md
git commit -m "runtime: add cuda batched world integration"
```

**Focused validation completed before full verification:**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test nuka_cuda_perf_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld|CudaBatchTiming"
```

**Focused results:** CUDA batched upload, independent batched integration, and
timing checks passed. The benchmark covers 256 instances, 32 bodies per
instance, and 120 CUDA steps under the one-second threshold.
