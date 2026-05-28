# Nuka Physics v0.1 – Phase 7: C ABI v0.1 + V1 Oracle Setup + Go2 Stand Demo (v0.1 Exit Gate)

> **Master plan reference:** §3 Round 8 (C ABI) + §3 Round 11 (V1 oracle) + §7 v0.1 exit criteria
> **Prerequisites:** Phases 1–6 complete
> **Blocks:** v0.3 phase 1 (S1 sprint cannot begin until v0.1 closes)
> **Exit criteria gate:** **v0.1 CLOSE**
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Final phase of v0.1. Three deliverables that together close the v0.1 gate:

1. **C ABI v0.1** — `extern "C"` surface with opaque handles, no exceptions, no STL across boundary. Sufficient to create a World, step it, read state.
2. **V1 oracle for Featherstone** — golden trajectories from MuJoCo MJX checked into Git LFS; CI compares engine output against golden on every push.
3. **Go2 stand demo** — load Go2 USD via existing importer, run 5 s simulation through the new pipeline (Featherstone ABA + RowSolver + V2 monitoring), assert energy stable, output PPM frame for visual sanity.

The Go2 stand demo is the **literal v0.1 exit demonstration**. Master plan §7 v0.1 exit criteria: "Demo: Go2 stand 5 s, MJX oracle within tolerance."

## Tech Stack

- C++20
- CUDA 12+
- Git LFS (golden trajectories)
- MuJoCo MJX (reference oracle, Python harness)
- All Phase 1–6 deliverables

## Files to Create

### C ABI

- `src/include/nuka/nuka.h` — public C header (the only file external C consumers need)
- `src/include/nuka/nuka.hpp` — thin C++20 wrapper (RAII handles, expected-based errors)
- `src/c_abi/device.cpp` — `nuka_device_*` functions
- `src/c_abi/world.cpp` — `nuka_world_*` functions
- `src/c_abi/buffer.cpp` — `nuka_buffer_*` functions
- `src/c_abi/error.cpp` — `nuka_result_t` mapping
- `src/c_abi/handle_table.hpp` — opaque-handle ↔ internal-pointer mapping (per-process registry)
- `tests/c_abi/test_create_step_destroy.c` — pure C test
- `tests/c_abi/test_cpp_wrapper_raii.cpp` — C++20 wrapper test

### V1 Oracle

- `tests/oracle/golden/featherstone_go2_random_sample.bin` — Git LFS
- `tests/oracle/golden/featherstone_h1_random_sample.bin` — Git LFS
- `tests/oracle/golden/go2_stand_5s.bin` — Git LFS (full 5 s trajectory)
- `tests/oracle/featherstone_oracle_harness.cpp`
- `tools/oracle/generate_mjx_golden.py` — Python script that produces the golden files using MJX
- `tools/oracle/diff_trajectory.py` — diff utility: load engine output + golden, compute per-step / per-DOF error
- `tests/oracle/README.md`

### Go2 Stand Demo

- `examples/scenes/go2_stand.usda` — Go2 in standing pose (if not already present in `examples/scenes/`)
- `examples/scenes/go2_stand.urdf` — alternate format
- `src/apps/nuka_go2_stand_demo.cpp` — CLI demo binary
- `tests/regression/test_go2_stand_5s.cpp` — automated regression test
- `out/.gitkeep` — output directory

## Files to Modify

- `src/CMakeLists.txt` — add `c_abi/` translation units; produce `libnuka.so` / `nuka.dll` as shared library; export public headers
- `cmake/nukaConfig.cmake.in` — CMake package config so downstream `find_package(nuka)` works
- `src/apps/nuka_scene_demo.cpp` — keep existing demo; add reference to new `nuka_go2_stand_demo`
- `tests/CMakeLists.txt`

## C ABI Surface (v0.1)

`src/include/nuka/nuka.h`:

```c
#ifndef NUKA_NUKA_H
#define NUKA_NUKA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------- Opaque handles -------- */
typedef struct nuka_device_t* nuka_device_handle;
typedef struct nuka_world_t*  nuka_world_handle;
typedef struct nuka_buffer_t* nuka_buffer_handle;

/* -------- Result codes -------- */
typedef enum {
    NUKA_RESULT_OK = 0,
    NUKA_RESULT_INVALID_ARG = 1,
    NUKA_RESULT_NULL_HANDLE = 2,
    NUKA_RESULT_CUDA_ERROR = 3,
    NUKA_RESULT_FILE_NOT_FOUND = 4,
    NUKA_RESULT_PARSE_ERROR = 5,
    NUKA_RESULT_OUT_OF_MEMORY = 6,
    NUKA_RESULT_NOT_SUPPORTED = 7,
    NUKA_RESULT_INTERNAL = 100,
} nuka_result_t;

/* -------- Versioning -------- */
typedef struct { uint16_t major, minor, patch; } nuka_version_t;
nuka_version_t nuka_get_version(void);

/* -------- Device -------- */
typedef struct {
    uint32_t gpu_index;
    void*    cuda_stream;     /* cudaStream_t cast to void*; caller-owned, may be NULL for engine-default */
    uint8_t  backend_selection_layer_enabled;
} nuka_device_desc_t;

nuka_result_t nuka_device_create(const nuka_device_desc_t* desc, nuka_device_handle* out);
void          nuka_device_destroy(nuka_device_handle device);

/* -------- World -------- */
typedef struct {
    const char* scene_path;        /* USD / URDF / MJCF / .nuka */
    uint32_t    env_count;         /* number of parallel envs */
    float       fixed_dt;          /* seconds */
} nuka_world_desc_t;

nuka_result_t nuka_world_create_from_scene(nuka_device_handle device,
                                            const nuka_world_desc_t* desc,
                                            nuka_world_handle* out);
void          nuka_world_destroy(nuka_world_handle world);
nuka_result_t nuka_world_step(nuka_world_handle world);
nuka_result_t nuka_world_step_n(nuka_world_handle world, uint32_t step_count);

/* -------- State read -------- */
typedef enum {
    NUKA_FIELD_RIGID_BODY_TRANSFORM = 0,
    NUKA_FIELD_ARTICULATION_LINK_POSE = 1,
    NUKA_FIELD_JOINT_POSITION = 2,
    NUKA_FIELD_JOINT_VELOCITY = 3,
    NUKA_FIELD_OBSERVATIONS = 4,
    NUKA_FIELD_CONTACT_POINTS = 5,
} nuka_state_field_t;

/* Returns a device-pointer view (zero copy). Caller must not free. */
typedef struct {
    void*    device_ptr;
    size_t   element_count;
    uint32_t element_stride_bytes;
    uint8_t  dtype;             /* 0 f32, 1 f64, 2 i32, 3 u32 */
} nuka_buffer_view_t;

nuka_result_t nuka_world_get_buffer_view(nuka_world_handle world,
                                          nuka_state_field_t field,
                                          nuka_buffer_view_t* out);

/* -------- Diagnostics -------- */
nuka_result_t nuka_world_get_last_invariant_violations(nuka_world_handle world,
                                                       uint32_t* out_count,
                                                       /* fixed-size struct array, see header */
                                                       void* out_array,
                                                       uint32_t array_capacity);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_H */
```

C++20 wrapper `src/include/nuka/nuka.hpp` provides RAII handles, `std::expected<T, nuka_result_t>` return values, and `std::string_view` ergonomic input — but the underlying ABI is pure C.

## Tasks

### Task 7.1 — Implement C ABI translation units

- Handle table (`handle_table.hpp`): per-process map opaque pointer → internal `phi::DeviceContext` / `PhysicsWorld` / etc. Thread-safe.
- Map every C ABI function to existing engine APIs; catch any C++ exceptions internally and convert to `nuka_result_t`.
- `cudaStream_t` passed in `nuka_device_desc_t::cuda_stream` is wrapped as `phi::StreamView`. If NULL, engine creates an `OwnedStream` and uses it.

### Task 7.2 — C++ wrapper

`src/include/nuka/nuka.hpp`:

```cpp
#pragma once
#include "nuka/nuka.h"
#include <string_view>
#include <expected>
#include <cuda_runtime.h>

namespace nuka {

class Error {
public:
    explicit Error(nuka_result_t code) : code_(code) {}
    nuka_result_t code() const noexcept { return code_; }
    const char* message() const noexcept;
private:
    nuka_result_t code_;
};

class Device {
public:
    static std::expected<Device, Error> Create(uint32_t gpu_index, cudaStream_t stream);
    Device(Device&&) noexcept;
    ~Device() noexcept;
    nuka_device_handle raw() const noexcept { return h_; }
private:
    Device(nuka_device_handle h) : h_(h) {}
    nuka_device_handle h_ = nullptr;
};

class World {
public:
    static std::expected<World, Error> CreateFromScene(const Device& d, std::string_view path,
                                                       uint32_t env_count, float dt);
    World(World&&) noexcept;
    ~World() noexcept;
    std::expected<void, Error> Step();
    std::expected<void, Error> StepN(uint32_t n);
    struct BufferView { void* device_ptr; size_t element_count; uint32_t stride; uint8_t dtype; };
    std::expected<BufferView, Error> GetBufferView(nuka_state_field_t field) const;
private:
    World(nuka_world_handle h) : h_(h) {}
    nuka_world_handle h_ = nullptr;
};

} // namespace nuka
```

### Task 7.3 — Build as shared library

`src/CMakeLists.txt`:

```cmake
add_library(nuka SHARED
    # ... existing sources ...
    c_abi/device.cpp
    c_abi/world.cpp
    c_abi/buffer.cpp
    c_abi/error.cpp
)

set_target_properties(nuka PROPERTIES
    PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/src/include/nuka/nuka.h;${CMAKE_SOURCE_DIR}/src/include/nuka/nuka.hpp"
    VERSION   ${NUKA_VERSION}
    SOVERSION ${NUKA_VERSION_MAJOR}
)

install(TARGETS nuka EXPORT nukaTargets
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/nuka
)
install(EXPORT nukaTargets FILE nukaTargets.cmake NAMESPACE nuka:: DESTINATION lib/cmake/nuka)
configure_package_config_file(cmake/nukaConfig.cmake.in nukaConfig.cmake ...)
```

This produces a `libnuka.so` (Linux) / `nuka.dll` (Windows) that downstream projects consume via:

```cmake
find_package(nuka REQUIRED)
target_link_libraries(my_app PRIVATE nuka::nuka)
```

### Task 7.4 — C and C++ ABI tests

`tests/c_abi/test_create_step_destroy.c` (pure C, validates no C++ leaks across boundary):

```c
#include "nuka/nuka.h"
#include <assert.h>

int main(void) {
    nuka_device_desc_t dd = { .gpu_index = 0, .cuda_stream = NULL,
                              .backend_selection_layer_enabled = 1 };
    nuka_device_handle dev;
    assert(nuka_device_create(&dd, &dev) == NUKA_RESULT_OK);

    nuka_world_desc_t wd = { .scene_path = "examples/scenes/go2_stand.usda",
                             .env_count = 1, .fixed_dt = 1.f / 240.f };
    nuka_world_handle world;
    assert(nuka_world_create_from_scene(dev, &wd, &world) == NUKA_RESULT_OK);

    for (int i = 0; i < 100; ++i) {
        assert(nuka_world_step(world) == NUKA_RESULT_OK);
    }

    nuka_buffer_view_t view;
    assert(nuka_world_get_buffer_view(world, NUKA_FIELD_JOINT_POSITION, &view) == NUKA_RESULT_OK);
    assert(view.device_ptr != NULL);

    nuka_world_destroy(world);
    nuka_device_destroy(dev);
    return 0;
}
```

### Task 7.5 — Generate Featherstone golden trajectories

`tools/oracle/generate_mjx_golden.py`:

```python
import jax, mujoco_mjx as mjx, numpy as np

def generate_go2_random_sample_golden(out_path):
    model = mjx.put_model(mjx.load_model("examples/scenes/go2_stand.xml"))
    rng = np.random.default_rng(seed=42)
    samples = []
    for _ in range(1000):
        q = rng.uniform(-1, 1, model.nq)
        qd = rng.uniform(-1, 1, model.nv)
        tau = rng.uniform(-10, 10, model.nu)
        data = mjx.make_data(model).replace(qpos=q, qvel=qd, ctrl=tau)
        qddot = mjx.forward(model, data).qacc
        samples.append((q, qd, tau, qddot))
    # serialize to binary
    save_golden(samples, out_path)

def generate_go2_stand_5s_golden(out_path):
    # Full 1200-step trajectory of Go2 standing with PD hold; ABA + integrator
    # Save: per-step (qpos, qvel, qddot, link_poses)
    ...
```

Outputs go to `tests/oracle/golden/`. Tracked in Git LFS.

### Task 7.6 — Oracle harness in C++

`tests/oracle/featherstone_oracle_harness.cpp`:

```cpp
TEST(FeatherstoneOracle, Go2RandomSampleAgreesWithMjx) {
    auto samples = LoadGoldenBinary("tests/oracle/golden/featherstone_go2_random_sample.bin");
    auto ctx = nuka::phi::MakeDeviceContext(0, /*stream*/nullptr);
    auto articulation = LoadArticulation(ctx, "examples/scenes/go2_stand.usda");

    for (const auto& s : samples) {
        UploadJointState(articulation, s.q, s.qd);
        UploadJointTorque(articulation, s.tau);
        FeatherstoneAba::ComputeAccelerations(ctx, articulation, /*gravity_z*/ -9.81f);
        auto qddot_engine = DownloadJointAcceleration(articulation);
        for (uint32_t i = 0; i < s.qddot.size(); ++i) {
            EXPECT_NEAR(qddot_engine[i], s.qddot[i], 1e-3f);
        }
    }
}
```

Tolerance per master plan §6 V1 table: < 1e-4 joint angle over 1 s. For raw `qddot`: 1e-3 absolute is the per-step bar.

### Task 7.7 — Go2 stand demo

`src/apps/nuka_go2_stand_demo.cpp`:

```cpp
int main(int argc, char** argv) {
    const char* scene = argc > 1 ? argv[1] : "examples/scenes/go2_stand.usda";
    auto dev = nuka::Device::Create(0, nullptr).value();
    auto world = nuka::World::CreateFromScene(dev, scene, /*env_count=*/1, /*dt=*/1.f/240.f).value();

    constexpr int N = 1200;     // 5 s @ 240 Hz
    auto t0 = now();
    for (int i = 0; i < N; ++i) {
        world.Step().value();
    }
    auto elapsed = now() - t0;
    printf("Stepped %d in %.2f ms (avg %.3f us/step)\n", N, elapsed_ms, elapsed_us/N);

    // Read joint positions; assert robot still upright (no fall)
    auto view = world.GetBufferView(NUKA_FIELD_JOINT_POSITION).value();
    std::vector<float> qpos(view.element_count);
    cudaMemcpy(qpos.data(), view.device_ptr, qpos.size()*sizeof(float), cudaMemcpyDeviceToHost);
    // Check root z-coordinate (assume joint 0 is base z); print pass/fail

    return 0;
}
```

### Task 7.8 — Regression test for Go2 stand

`tests/regression/test_go2_stand_5s.cpp`:

```cpp
TEST(Go2Stand, Stands5sEnergyDriftAndOracleAgreement) {
    // 1. Run 5s
    // 2. Assert energy drift < 2% (V2 invariant)
    // 3. Compare full trajectory against tests/oracle/golden/go2_stand_5s.bin
    //    per-step max joint position error < 1e-4 (master plan tolerance)
    // 4. Verify deterministic: run twice, bit-exact match
}
```

This regression test gates v0.1 exit.

### Task 7.9 — Build / CI integration

- CMake produces `libnuka.so` and `nuka_go2_stand_demo` binary.
- CI workflow `v0.1-exit-gate.yml` runs the Go2 stand regression test on every push.
- Performance budget: 1 env × 1200 steps in < 5 seconds wall time (i.e. > 240 Hz realtime; not the S1 budget of < 1 ms/step yet — that's v0.3).

## Validation

- C ABI pure-C test creates, steps, reads, destroys without C++ ever leaking.
- C++20 wrapper test demonstrates RAII + `std::expected` ergonomic surface.
- `find_package(nuka)` works in a sample downstream project.
- Featherstone oracle harness passes Go2 + H1 random-sample tests.
- Go2 stand demo runs 5 s, robot does not fall, energy drift < 2%, deterministic bit-exact.
- Per-step joint position error vs MJX golden trajectory < 1e-4 over the 5 s window.

## Exit Criteria for Phase 7 = **v0.1 EXIT**

All exit criteria for v0.1 from master plan §7:

1. ✅ PHI Stream refactored to non-owning view; PlatformContract de-singletonized (Phase 3).
2. ✅ C ABI v0.1 working (Phase 7 Task 7.1-7.4).
3. ✅ CSR Universal Row format landed; existing Contact/Joint/Drive paths migrated through new scheduler via diff-test bridge (Phase 5).
4. ✅ Island / graph-coloring scheduler operational (Phase 5).
5. ✅ V5 + V2 validation infrastructure operational (Phases 1, 4).
6. ✅ Featherstone ABA forward dynamics CUDA-resident; advances Go2 correctly (Phase 6).
7. ✅ Codegen pipeline produces forward kernels for the four base row classes (Phase 2).
8. ✅ **Demo: Go2 stand 5 s, MJX oracle within tolerance** (Phase 7 Task 7.7-7.8).

When all above check: **declare v0.1 closed**, write a quarterly external output (blog post / talk / preprint per §5.1 rhythm), and begin v0.3 Phase 1.

## What This Phase Does Not Do

- No Vulkan ↔ CUDA interop (v0.7 / v1.0).
- No Python bindings (v0.3 / v0.5).
- No PyTorch autograd (v0.3 skeleton, v0.5 complete).
- No diff-sim (v0.5).
- No new row classes beyond v0.1 catalog (v0.7).
- No high-throughput / sub-ms perf budget (v0.3).
