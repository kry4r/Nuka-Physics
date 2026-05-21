# CUDA Batched Contact Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the CUDA batched world from rigid integration to GPU-resident broadphase, contact generation, and contact solving across multiple scene instances.

**Architecture:** Reuse one cooked `WorldTemplate` for all environments and keep mutable state flattened by instance. Upload shape tables into `BatchedDeviceWorld`, generate per-instance AABBs/pairs/manifolds in CUDA kernels with no cross-instance pairs, assemble contact constraint blocks with flattened body indices, and solve them on the same batched device state. CPU remains a validation boundary through `DownloadState()` and reference tests only.

**Tech Stack:** C++20, CUDA, CMake, GoogleTest.

---

### Task 1: Batched CUDA Shape Upload

**Files:**
- Modify: `src/runtime/gpu/batched_device_world.hpp`
- Modify: `src/runtime/gpu/batched_device_world.cu`
- Modify: `tests/runtime/test_cuda_batched_world.cpp`

**Step 1: Write failing shape table test**

Add `CudaBatchedWorld.UploadsSharedShapeTablesForBatchedContacts`:

```cpp
auto world = runtime::BuildWorld(scene::CookScene(BuildPlaneBoxScene().scene));
std::vector<runtime::WorldInstance> instances(2, world.instance);

auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

EXPECT_EQ(batch.ShapeCountPerInstance(), 2u);
EXPECT_EQ(batch.TotalShapeCount(), 4u);
EXPECT_TRUE(batch.HasUploadedShapeTables());
```

**Step 2: Run red**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
```

Expected: build fails because the batched world does not expose shape counts or shape table upload.

**Step 3: Implement shape table upload**

Add shared template buffers to `BatchedDeviceWorld`:
- shape types
- shape body ids
- shape local transforms
- shape half extents
- shape radii

Expose const device pointer accessors and `ShapeCountPerInstance()`, `TotalShapeCount()`, and `HasUploadedShapeTables()`.

**Step 4: Run green**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld"
```

Expected: shape table upload and existing batched rigid tests pass.

**Result:** Added shared cooked shape-table buffers to `BatchedDeviceWorld`
with count/accessor validation. `CudaBatchedWorld.UploadsSharedShapeTablesForBatchedContacts`
passes with the existing batched rigid tests.

### Task 2: Batched CUDA Broadphase and Contact Generation

**Files:**
- Modify: `src/runtime/gpu/batched_device_world.hpp`
- Modify: `src/runtime/gpu/batched_device_world.cu`
- Modify: `tests/runtime/test_cuda_batched_world.cpp`

**Step 1: Write failing contact generation test**

Add `CudaBatchedWorld.GeneratesContactsPerInstanceWithoutCrossPairs`:

```cpp
auto world = runtime::BuildWorld(scene::CookScene(BuildPlaneBoxScene().scene));
std::vector<runtime::WorldInstance> instances(2, world.instance);
instances[1].poses[box_body].position.y = 2.0f;

auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);
auto broadphase = runtime::gpu::BuildBatchedCudaBroadphase(batch);
auto contacts = runtime::gpu::GenerateBatchedCudaContacts(batch, broadphase);
auto report = contacts.DownloadReport();
auto manifolds = contacts.DownloadManifolds();

EXPECT_EQ(report.instance_count, 2u);
EXPECT_EQ(report.pair_count, 1u);
EXPECT_EQ(report.contact_manifold_count, 1u);
EXPECT_EQ(manifolds[0].body_a, box_body);
EXPECT_EQ(manifolds[0].body_b, 0u);
EXPECT_EQ(manifolds[0].instance_index, 0u);
```

**Step 2: Run red**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
```

Expected: build fails because batched broadphase/contact APIs do not exist.

**Step 3: Implement broadphase/contact kernels**

Add result classes:
- `CudaBatchedBroadphaseResult`
- `CudaBatchedContactResult`

Run one AABB thread per flattened shape and one pair thread per `(instance, local_pair_slot)`. Store local shape pairs plus `instance_index`; contact generation converts body ids to flattened body ids by adding `instance_index * body_count_per_instance`.

**Step 4: Run green**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld"
```

Expected: batched contact report and manifold download match one colliding instance and one separated instance.

**Result:** Added batched AABB and local-pair kernels. Contact generation runs
per `(instance, local_pair_slot)`, preserves `instance_index` in downloaded
manifolds, and does not generate cross-environment pairs.

### Task 3: Batched CUDA Contact Solver

**Files:**
- Modify: `src/runtime/gpu/batched_device_world.hpp`
- Modify: `src/runtime/gpu/batched_device_world.cu`
- Modify: `tests/runtime/test_cuda_batched_world.cpp`

**Step 1: Write failing solve test**

Add `CudaBatchedWorld.SolvesContactsIndependentlyOnDevice`:

```cpp
auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);
auto broadphase = runtime::gpu::BuildBatchedCudaBroadphase(batch);
auto contacts = runtime::gpu::GenerateBatchedCudaContacts(batch, broadphase);

runtime::gpu::CudaBatchedConstraintSolverConfig config;
config.velocity_iterations = 10u;
config.position_iterations = 4u;

auto result = runtime::gpu::SolveBatchedCudaContactConstraints(batch, contacts, config);
auto report = result.DownloadReport();
auto state = batch.DownloadState();

EXPECT_EQ(report.contact_constraint_count, 1u);
EXPECT_GE(state.linear_velocities[instance0_box_flat].y, -1.0e-5f);
EXPECT_NEAR(state.poses[instance1_box_flat].position.y, 2.0f, 1.0e-5f);
```

**Step 2: Run red**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
```

Expected: build fails because the batched solver API does not exist.

**Step 3: Implement minimal batched contact solver**

Assemble batched contact blocks from contact manifolds and solve them on flattened state. The first pass supports contact blocks; joint/drive batching remains a documented next step after contacts are GPU-resident.

**Step 4: Run green**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld"
```

Expected: only the colliding instance is corrected and the separated instance remains unchanged.

**Result:** Added batched contact block assembly and PGS contact solve over
flattened body state. The colliding environment receives velocity and position
correction, while a separated environment's pose and velocity remain unchanged.

### Task 4: Batched Step Pipeline, Timing, Docs, Commit

**Files:**
- Modify: `src/runtime/gpu/batched_device_world.hpp`
- Modify: `src/runtime/gpu/batched_device_world.cu`
- Modify: `tests/runtime/test_cuda_batched_world.cpp`
- Create: `tests/perf/test_cuda_batch_contact_timing.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `README.md`
- Modify: `docs/architecture/runtime-overview.md`
- Modify: `docs/testing/p0-regression-matrix.md`
- Modify: `docs/plans/2026-05-21-cuda-batched-contact-solver-plan.md`

**Step 1: Add pipeline API test**

Extend `StepBatchedCudaWorld()` options with `enable_contacts`, solver iteration fields, slop, and Baumgarte. Add a test that calls the step function with contacts enabled and verifies the contact report fields are included in `CudaBatchedWorldStepReport`.

**Step 2: Add timing test**

Add `CudaBatchContactTiming.BatchedContactSolveUnderOneSecond` with 128 instances, a plane and box per instance, 60 CUDA steps, and `< 1000 ms` threshold.

**Step 3: Update docs**

Document that batched contacts now run through the CUDA production path; joint/drive batching remains a follow-on phase.

**Step 4: Full verification**

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

Expected: full Release build and all tests pass.

**Step 5: Commit**

```powershell
git add src/runtime/gpu/batched_device_world.* tests/runtime/test_cuda_batched_world.cpp tests/perf/test_cuda_batch_contact_timing.cpp tests/CMakeLists.txt README.md docs/architecture/runtime-overview.md docs/testing/p0-regression-matrix.md docs/plans/2026-05-21-cuda-batched-contact-solver-plan.md
git commit -m "runtime: add cuda batched contact solver"
```

**Focused validation completed before full verification:**

```powershell
cmake --build build --config Release --target nuka_cuda_runtime_test nuka_cuda_perf_test
ctest --test-dir build -C Release --output-on-failure -R "CudaBatchedWorld|CudaBatchContactTiming"
```

**Focused results:** CUDA batched shape upload, per-instance contact
generation, independent contact solve, step-pipeline contact solve, and the
128-environment / 60-step batched contact timing test passed. The measured
focused test wall time was 0.95 seconds for the selected CTest set, with the
batched contact timing test reporting 0.30 seconds under the one-second
threshold.
