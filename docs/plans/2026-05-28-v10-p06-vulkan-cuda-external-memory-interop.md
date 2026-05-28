# Nuka Physics v1.0 – Phase 6: Vulkan ↔ CUDA External Memory Interop (Windows Priority)

> **Master plan reference:** §3 Round 9 (renderer interop strategy I1) + §7 v1.0 exit
> **Prerequisites:** v0.1 Phase 3 (caller-owned cudaStream_t in PHI); v0.7 Phase 13 (CUDA RT for sensor framebuffers exists)
> **Blocks:** v1.0 Phases 7-8 (interactive demos benefit from external renderer); v1.5 D3D12 phase
> **Exit criteria gate:** v1.0
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Enable **zero-copy GPU memory sharing between Nuka's CUDA physics and an external Vulkan renderer**. Master plan §3 Round 9 commits Vulkan first (S1 done in Windows + Vulkan), D3D12 second (v1.5).

After this phase, an external Vulkan-based renderer (UE5 RHI Vulkan, Godot 4 Vulkan, custom engine) can read Nuka's rigid body transforms / soft body vertices / fluid particles directly from a Vulkan buffer that is the same GPU allocation Nuka writes to. No CPU staging, no synchronization round-trip.

## Tech Stack

- CUDA 12+ external memory APIs (`cudaImportExternalMemory`, `cudaImportExternalSemaphore`)
- Vulkan 1.3+ external memory extensions (`VK_KHR_external_memory_win32`, `VK_KHR_external_semaphore_win32`)
- Windows OS APIs (NT handles)
- Existing Vulkan renderer in `src/render/`

## Files to Create

- `src/interop/vulkan_cuda_interop.hpp`
- `src/interop/vulkan_cuda_interop.cpp` — buffer / semaphore import / export
- `src/interop/exportable_buffer.hpp` — Nuka-side wrapper
- `src/interop/exportable_buffer.cu`
- `src/include/nuka/nuka_interop.h` — C ABI extension
- `src/c_abi/interop.cpp`
- `tests/interop/test_vulkan_cuda_buffer_roundtrip.cpp`
- `tests/interop/test_vulkan_cuda_semaphore.cpp`
- `examples/integration/external_vulkan_app/main.cpp` — example external Vulkan consumer
- `examples/integration/external_vulkan_app/CMakeLists.txt`
- `docs/integration/vulkan-interop-guide.md`

## Files to Modify

- `src/runtime/world_stepper.cpp` — flag exportable buffers; signal semaphores at step end
- `src/include/nuka/nuka.h` — add semaphore handle accessor

## Tasks

### Task 10.6.1 — External memory protocol

Per master plan §3 Round 9 design:
- Engine allocates CUDA memory via `cudaExternalMemoryAllocate` (Windows: with `cudaExternalMemoryHandleTypeOpaqueWin32`).
- Engine exports a Windows NT handle.
- External Vulkan app imports the handle via `vkImportMemoryWin32HandleInfoKHR`.
- Both sides now reference the same GPU allocation.

The flow:

```
1. Engine creates physics world.
2. For each render-relevant buffer (rigid transforms, joint poses, particle positions),
   engine allocates via cudaExternalMemoryAllocate.
3. Engine exports Win32 handle via cudaExternalMemoryGetMappedBuffer + Vulkan helper.
4. Engine returns handle via C ABI.
5. External app imports via vkAllocateMemory + vkImportMemoryWin32HandleKHR.
6. External app binds Vulkan VkBuffer to imported memory.
7. Per frame:
   a. Engine steps physics; writes to the shared buffer; signals a cudaExternalSemaphore.
   b. External app waits on the semaphore via vkSemaphoreWait; reads / renders.
```

### Task 10.6.2 — Exportable buffer wrapper

`src/interop/exportable_buffer.hpp`:

```cpp
class ExportableBuffer {
public:
    ExportableBuffer(const phi::DeviceContext& ctx, size_t bytes,
                     ExternalMemoryHandleType handle_type);
    ~ExportableBuffer();

    void*  CudaDevicePointer() const noexcept;
    void*  Win32Handle() const noexcept;       // for Vulkan import
    size_t Size() const noexcept;

private:
    cudaExternalMemory_t ext_mem_;
    void* device_ptr_;
    HANDLE win32_handle_;
    size_t size_;
};
```

Used by engine for all "render-visible" buffers: rigid body transforms, articulation link poses, soft body vertex positions, fluid particle positions.

### Task 10.6.3 — Exportable semaphore

`src/interop/vulkan_cuda_interop.cpp`:

```cpp
class ExportableSemaphore {
public:
    ExportableSemaphore(const phi::DeviceContext& ctx);
    ~ExportableSemaphore();

    void Signal(uint64_t value, cudaStream_t stream);     // from CUDA side
    void* Win32Handle() const;                            // for Vulkan import

private:
    cudaExternalSemaphore_t ext_sem_;
    HANDLE win32_handle_;
    uint64_t counter_ = 0;
};
```

Engine signals the semaphore at the end of each physics step; external renderer waits on it before reading.

### Task 10.6.4 — C ABI surface

```c
typedef struct nuka_exportable_buffer_t* nuka_exportable_buffer_handle;
typedef struct nuka_exportable_semaphore_t* nuka_exportable_semaphore_handle;

nuka_result_t nuka_world_get_exportable_buffer(nuka_world_handle w,
                                                nuka_state_field_t field,
                                                nuka_exportable_buffer_handle* out);

nuka_result_t nuka_exportable_buffer_get_win32_handle(nuka_exportable_buffer_handle b,
                                                       void** out_handle);

nuka_result_t nuka_world_get_step_complete_semaphore(nuka_world_handle w,
                                                      nuka_exportable_semaphore_handle* out);

nuka_result_t nuka_exportable_semaphore_get_win32_handle(nuka_exportable_semaphore_handle s,
                                                          void** out_handle);

nuka_result_t nuka_world_step_signal(nuka_world_handle w, uint64_t signal_value);
```

Last call: stepping + signaling the semaphore in one operation, allowing the external app to wait for a specific step.

### Task 10.6.5 — Example external Vulkan app

`examples/integration/external_vulkan_app/main.cpp`:

A minimal example showing:
- Initialize Vulkan instance with external memory extensions.
- Create nuka Device + World.
- Get exportable rigid transforms buffer + semaphore.
- Import into Vulkan.
- Per frame: signal nuka to step + signal; wait via Vulkan; render using imported buffer as vertex/instance data.

This becomes the canonical reference for users wiring nuka into UE5/Godot/custom Vulkan engines.

### Task 10.6.6 — Performance verification

`tests/interop/test_vulkan_cuda_buffer_roundtrip.cpp`:

```cpp
TEST(VulkanCudaInterop, ZeroCopyPerf) {
    auto world = MakeGo2Demo(...);
    auto exp_buf = GetExportableTransformBuffer(world);
    auto vk_buf = ImportIntoVulkan(exp_buf);

    auto t0 = now_us();
    for (int i = 0; i < 1000; ++i) {
        StepWorld(world);   // writes to exp_buf
        SignalSemaphore(world);
        VulkanWaitAndRender(vk_buf);
    }
    auto elapsed = now_us() - t0;
    // No CPU staging; throughput should be physics-bound, not transfer-bound
    EXPECT_LT(elapsed, 1000 * 2000);   // < 2 ms per frame avg
}
```

### Task 10.6.7 — Windows-only constraint

CUDA-Vulkan interop on Windows requires NT handle types. Linux uses fd-based handle types (also supported). macOS not supported (no CUDA on macOS).

Linux path also implemented in this phase (similar pattern with `cudaExternalMemoryHandleTypeOpaqueFd`).

### Task 10.6.8 — Documentation

`docs/integration/vulkan-interop-guide.md`:

- Architecture diagram (CUDA + Vulkan sharing memory + semaphore).
- Step-by-step integration with UE5 (theoretical; user provides UE5 plugin themselves).
- Buffer protocol reference.
- Synchronization timeline.

## Validation

- Example Vulkan app builds and runs on Windows.
- Throughput is physics-bound, not interop-bound.
- Determinism preserved.
- Semaphore wait correctly synchronizes (no torn reads).

## Exit Criteria for v1.0 Phase 6

1. ExportableBuffer + ExportableSemaphore operational on Windows.
2. Linux variant (fd-based handles) operational.
3. C ABI exports work end-to-end.
4. Example Vulkan integration app builds and runs.
5. Performance test demonstrates zero-copy benefit.
6. Documentation published.

## What This Phase Does Not Do

- No D3D12 interop (v1.5 — second priority per master plan).
- No Metal interop (out of scope; macOS unsupported).
- No multi-GPU sharing across devices (out of scope).
- No automatic UE5 plugin (user must build their own).
