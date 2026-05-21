# CUDA Batched Joint Drive Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `BatchedDeviceWorld` so Isaac Lab-style parallel environments can assemble and solve cooked joint and actuator constraints on CUDA without falling back to CPU stepping.

**Architecture:** Reuse one cooked `WorldTemplate` for all batch instances. Upload shared joint and actuator tables next to the existing body/shape tables, assemble one joint block per `(instance, joint)` and one drive block per `(instance, actuator)` with flattened body ids, and solve those blocks with the same batched PGS path that already handles contact rows. CPU remains validation/reference only.

**Tech Stack:** C++20, CUDA, CMake, GoogleTest.

---

### Task 1: Batched Joint and Actuator Table Upload

**Files:**
- Modify: `src/runtime/gpu/batched_device_world.hpp`
- Modify: `src/runtime/gpu/batched_device_world.cu`
- Modify: `tests/runtime/test_cuda_batched_world.cpp`

**Step 1: Write failing upload test**

Add `CudaBatchedWorld.UploadsSharedJointAndActuatorTables` using a static parent, dynamic child, revolute joint, and velocity actuator.

**Step 2: Run red**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
```

Expected: build fails because batched joint/actuator count and upload validation APIs do not exist.

**Step 3: Implement table upload**

Add shared joint buffers:
- joint types
- parent/child bodies
- axes
- parent/child frames

Add shared actuator buffers:
- actuator types
- joint ids
- gains
- force limits

Expose count/accessor helpers and validation methods.

**Step 4: Run green**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld"
```

Expected: joint/actuator upload and existing batched tests pass.

**Result:** Implemented. `CudaBatchedWorld.UploadsSharedJointAndActuatorTables`
passes and validates one shared cooked joint plus one shared cooked actuator
table uploaded into `BatchedDeviceWorld`.

### Task 2: Batched Joint and Drive Constraint Assembly

**Files:**
- Modify: `src/runtime/gpu/batched_device_world.hpp`
- Modify: `src/runtime/gpu/batched_device_world.cu`
- Modify: `tests/runtime/test_cuda_batched_world.cpp`

**Step 1: Write failing solver tests**

Add:
- `CudaBatchedWorld.ProjectsJointAnchorsIndependentlyOnDevice`
- `CudaBatchedWorld.AppliesVelocityDrivesIndependentlyOnDevice`

The tests should create two instances with different initial child poses or velocities, call `SolveBatchedCudaConstraints(batch, nullptr, config)`, and verify report counts plus per-instance state changes.

**Step 2: Run red**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
```

Expected: build fails because the generic batched constraint solver and joint/drive report fields do not exist.

**Step 3: Implement CUDA assembly**

Add kernels that assemble:
- joint blocks for `instance_count * joint_count_per_instance`
- drive blocks for `instance_count * actuator_count_per_instance`

Flatten valid local body ids with `instance_index * body_count_per_instance + local_body`. Preserve `scene::kInvalidBody` for world anchors.

**Step 4: Run green**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld"
```

Expected: batched joint projection and velocity drive tests pass with no CPU stepping in the hot path.

**Result:** Implemented. Batched CUDA now assembles joint blocks for
`instance_count * joint_count_per_instance` and drive blocks for
`instance_count * actuator_count_per_instance`, preserving world-anchor invalid
body ids and flattening valid ids per instance.

### Task 3: Step Pipeline, Timing, Docs, Commit

**Files:**
- Modify: `src/runtime/gpu/batched_device_world.hpp`
- Modify: `src/runtime/gpu/batched_device_world.cu`
- Modify: `tests/runtime/test_cuda_batched_world.cpp`
- Create: `tests/perf/test_cuda_batch_joint_drive_timing.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `README.md`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `docs/plans/2026-05-21-cuda-batched-joint-drive-plan.md`

**Step 1: Extend step pipeline**

Add `enable_joints` and `enable_drives` options to `CudaBatchedWorldStepOptions`, wire those into `StepBatchedCudaWorld()`, and extend the step report with joint and drive counts.

**Step 2: Add timing test**

Add `CudaBatchJointDriveTiming.BatchedJointDriveSolveUnderOneSecond` with at least 128 two-body environments, 60 CUDA steps, and `< 1000 ms` threshold.

**Step 3: Update docs**

Document that batched CUDA now covers rigid integration, contacts, joint projection, and velocity drives. Keep remaining gaps explicit: batched sensors and render synchronization.

**Step 4: Full verification**

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

Expected: full Release build and all tests pass.

**Step 5: Commit**

```powershell
git add src/runtime/gpu/batched_device_world.* tests/runtime/test_cuda_batched_world.cpp tests/perf/test_cuda_batch_joint_drive_timing.cpp tests/CMakeLists.txt README.md docs/architecture/runtime-overview.md docs/testing/p0-regression-matrix.md docs/plans/2026-05-21-cuda-batched-joint-drive-plan.md
git commit -m "runtime: add cuda batched joint drive solver"
```

**Result:** Implementation and focused verification complete before full-suite
verification:

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test nuka_cuda_perf_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld|CudaBatchJointDriveTiming"
```

Focused result: 11/11 tests passed. The new
`CudaBatchJointDriveTiming.BatchedJointDriveSolveUnderOneSecond` benchmark ran
128 two-body CUDA environments for 60 steps and completed under the 1000 ms
threshold in the focused CTest run.
