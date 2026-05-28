# Nuka Physics v0.1 – Phase 3: PHI Stream Caller-Owned + PlatformContract De-Singletonize

> **Master plan reference:** §3 Round 8 (C++ ABI) + §3 Round 9 (renderer interop)
> **Prerequisites:** Phase 1 complete; can run in parallel with Phase 2
> **Blocks:** Phase 7 (C ABI v0.1) — C ABI requires caller-owned stream
> **Exit criteria gate:** v0.1
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Refactor the PHI (Physics Hardware Interface) layer to support **caller-owned `cudaStream_t`** and **per-handle `PlatformContract` configuration**. These are both prerequisites for the C ABI: external callers (UE5, PyTorch, ROS2) bring their own CUDA stream and must not be forced into a global singleton.

Current state (from `src/phi/device.hpp`, `src/phi/stream.hpp`, `src/phi/platform_contract.hpp`):

- `Stream` owns its stream (constructed → creates new `cudaStream_t`); no way to wrap an external stream.
- `SetDevice(int)` uses CUDA's thread-global current device.
- `PlatformContract` is fetched via global `GetPlatformContract()`; backend selection is also global-style.

Deliverable: a non-owning `StreamView` (wraps external `cudaStream_t`), a per-handle `DeviceContext` (carries platform contract, device id, stream), and updated call sites throughout the engine.

## Tech Stack

- C++20
- CUDA 12+
- Existing `phi` module conventions

## Files to Create

- `src/phi/stream_view.hpp` — non-owning stream wrapper
- `src/phi/device_context.hpp` — per-handle device + stream + platform contract bundle
- `src/phi/device_context.cpp` — implementation
- `tests/phi/test_stream_view_external.cpp` — wrap external stream, run kernel, verify
- `tests/phi/test_device_context_per_handle.cpp` — two contexts on same GPU, independent configs

## Files to Modify

- `src/phi/stream.hpp` — keep current owning `Stream` as `OwnedStream` (rename); document; deprecate direct use in new code
- `src/phi/stream.cpp` — rename to `owned_stream.cpp`; update build
- `src/phi/device.hpp` — keep `GetDeviceCount`, `GetDeviceInfo`; deprecate `SetDevice` in favor of per-context device binding (allow override in DeviceContext)
- `src/phi/platform_contract.hpp` — keep struct definition; remove `GetPlatformContract()` singleton; the contract now lives in `DeviceContext`
- `src/phi/platform_contract.cpp` — implementation cleanup
- All call sites: `src/runtime/world_stepper.cpp`, `src/runtime/physics_world.cpp`, `src/runtime/batch_scheduler.cpp`, `src/runtime/gpu/*`, `src/solver/gpu/cuda_constraint_solver.cu`, `src/constraint/gpu/contact_generation.cu`, `src/sensor/**`, `src/render/**` — accept `const DeviceContext&` instead of relying on globals
- `tests/**` — update tests to construct `DeviceContext` explicitly

## Tasks

### Task 3.1 — `StreamView` non-owning wrapper

`src/phi/stream_view.hpp`:

```cpp
#pragma once
#include <cuda_runtime.h>

namespace nuka::phi {

// Non-owning view over an externally-managed CUDA stream.
// Caller must outlive any StreamView referring to their stream.
class StreamView {
public:
    StreamView() = default;          // null stream
    explicit StreamView(cudaStream_t s) : handle_(s) {}

    cudaStream_t Native() const noexcept { return handle_; }
    bool IsNull() const noexcept { return handle_ == nullptr; }

    // Convenience: synchronize the underlying stream.
    void Synchronize() const;

private:
    cudaStream_t handle_ = nullptr;
};

} // namespace nuka::phi
```

`StreamView` is trivial, copyable, and intentionally does not free the stream in its destructor. The owner (caller code or `OwnedStream`) is responsible for lifecycle.

### Task 3.2 — `DeviceContext`

`src/phi/device_context.hpp`:

```cpp
#pragma once
#include "phi/platform_contract.hpp"
#include "phi/stream_view.hpp"

namespace nuka::phi {

// Per-handle physics device context. Replaces the global singleton model.
// One DeviceContext corresponds to one nuka_device_handle from the C ABI.
struct DeviceContext {
    int device_id = 0;
    StreamView stream;                  // caller-owned (or constructed internally by OwnedStream wrapper)
    PlatformContract contract = {};     // per-handle override of backend policy
};

// Construct a DeviceContext bound to a specific GPU, using an externally-supplied stream.
// Caller retains ownership of the stream.
DeviceContext MakeDeviceContext(int device_id, cudaStream_t external_stream,
                                const PlatformContract& contract = {});

} // namespace nuka::phi
```

Helper:

```cpp
DeviceContext MakeDeviceContext(int device_id, cudaStream_t external_stream,
                                const PlatformContract& contract)
{
    DeviceContext ctx;
    ctx.device_id = device_id;
    ctx.stream = StreamView{external_stream};
    ctx.contract = contract;
    // We deliberately do NOT call cudaSetDevice here; SetDevice is thread-scoped.
    // The kernel-launching call site is responsible for scoped cudaSetDevice
    // (see ScopedDeviceGuard in Task 3.3).
    return ctx;
}
```

### Task 3.3 — `ScopedDeviceGuard` RAII

For call sites that span multiple kernel launches, provide a scoped guard that sets the CUDA current device on enter, restores on exit:

`src/phi/device_context.hpp` (continued):

```cpp
class ScopedDeviceGuard {
public:
    explicit ScopedDeviceGuard(int device_id);
    ~ScopedDeviceGuard();
    ScopedDeviceGuard(const ScopedDeviceGuard&) = delete;
    ScopedDeviceGuard& operator=(const ScopedDeviceGuard&) = delete;
private:
    int prior_device_ = 0;
};
```

Usage pattern at every entry into engine code:

```cpp
void World::Step(float dt) {
    nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);
    StepImpl(ctx_.stream.Native(), dt);
}
```

### Task 3.4 — Refactor `Stream` → `OwnedStream`

Rename existing `Stream` (currently owns its stream) to `OwnedStream`. New code should prefer `StreamView`; `OwnedStream` is for cases where the engine itself wants to create a stream (e.g., the default factory path when caller does not provide one).

`src/phi/owned_stream.hpp`:

```cpp
class OwnedStream {
public:
    OwnedStream();           // creates a new cudaStream_t
    ~OwnedStream();
    OwnedStream(OwnedStream&&) noexcept;
    OwnedStream& operator=(OwnedStream&&) noexcept;
    OwnedStream(const OwnedStream&) = delete;

    StreamView View() const noexcept { return StreamView{handle_}; }
    void Synchronize();
private:
    cudaStream_t handle_ = nullptr;
};
```

### Task 3.5 — Eliminate `GetPlatformContract()` singleton

Search-and-replace every call site:

```cpp
// before
auto contract = nuka::phi::GetPlatformContract();
// after
const auto& contract = ctx.contract;   // ctx is DeviceContext passed in
```

Where the call site doesn't currently have a `DeviceContext`, plumb one through the API. This is the main churn of the phase.

Key call sites (from grep):
- `src/runtime/world_stepper.cpp`
- `src/runtime/physics_world.cpp`
- `src/runtime/batch_scheduler.cpp`
- `src/runtime/world_builder.cpp`
- `src/solver/gpu/cuda_constraint_solver.cu` (kernel launch site)
- `src/constraint/gpu/contact_generation.cu`
- `src/apps/nuka_scene_demo.cpp`
- `tests/runtime/**`, `tests/solver/**`, etc.

Strategy: introduce `DeviceContext& ctx` parameter at the topmost entry (`PhysicsWorld::Step`), then thread it through. For tests, construct a default context: `auto ctx = MakeDeviceContext(0, StreamView{}, {});` and pass it.

### Task 3.6 — Validation tests

`tests/phi/test_stream_view_external.cpp`:

```cpp
TEST(PhiStreamView, WrapsExternalStream) {
    cudaStream_t s;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    {
        nuka::phi::StreamView view{s};
        ASSERT_FALSE(view.IsNull());
        // Launch a trivial kernel on s via view.Native(); verify completion.
        ...
    }
    // StreamView's destructor must NOT free the stream.
    EXPECT_EQ(cudaStreamQuery(s), cudaSuccess);    // still valid
    ASSERT_EQ(cudaStreamDestroy(s), cudaSuccess);
}
```

`tests/phi/test_device_context_per_handle.cpp`:

```cpp
TEST(DeviceContext, TwoContextsIndependentConfig) {
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1); cudaStreamCreate(&s2);

    auto ctx1 = nuka::phi::MakeDeviceContext(0, s1, /*contract*/{});
    auto ctx2 = nuka::phi::MakeDeviceContext(0, s2, /*contract*/{});

    ctx1.contract.backend_selection_layer_enabled = false;
    ctx2.contract.backend_selection_layer_enabled = true;

    EXPECT_NE(ctx1.contract.backend_selection_layer_enabled,
              ctx2.contract.backend_selection_layer_enabled);
    EXPECT_NE(ctx1.stream.Native(), ctx2.stream.Native());

    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
}
```

## Validation

- **Existing test suite passes** after the refactor (no behavior change; only signature change).
- **Build with `nuka_run_lint`** target passes — no new STL leaks into public headers, no exceptions added.
- **External stream lifecycle**: `tests/phi/test_stream_view_external.cpp` proves view does not free.
- **Independent contexts** test proves no shared global state.

## Exit Criteria for Phase 3

1. `StreamView` exists, is non-owning, wraps external `cudaStream_t`.
2. `OwnedStream` rename complete; old `Stream` removed from public API.
3. `DeviceContext` carries device id + stream view + per-handle `PlatformContract`.
4. `GetPlatformContract()` singleton removed.
5. All physics call sites accept `DeviceContext` and pass through to kernel launches.
6. All existing tests pass (`ctest` green).
7. Two contexts on same GPU with different `PlatformContract` settings demonstrably independent.

## What This Phase Does Not Do

- Does not yet expose a C ABI (Phase 7).
- Does not implement Vulkan / D3D12 interop (deferred to v0.7 / v1.0).
- Does not change CUDA kernel internals — only how they are launched.
- Does not yet support multi-GPU `M2` from one `World` (single device per context; M2 layer comes in v0.3 perf phase).
