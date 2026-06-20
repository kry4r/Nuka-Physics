<!-- Authoritative offline-RHI design (verified workflow, 12/12 claims held).
Supersedes + consolidates 2026-06-20-offline-rhi-render-design.md and incorporates its grounded edits.
The dual-purpose (replay video + GPU sensor) re-scope. -->

# Offline RHI + GPU Sensor Rendering — Design

A self-written **offline render RHI** that is GPU-compute-only, runs ON the PHI GPU backends (inheriting PHI's multi-backend selection), and is GPU-resident / zero-copy with the physics step. No OptiX, no RT cores, no graphics-API rasterization. The realtime Vulkan rasterizer (`src/render/`) is a **separate** abstraction, left as-is (it may adopt Slang later — out of scope).

This serves **two purposes over one general path**: (1) replay video (training-time periodic + post-train demo), and (2) in-the-loop, batched-over-envs **sensor rendering** (the strict-perf driver). The largest re-scope vs the prior draft (`subagent-plans/2026-06-20-offline-rhi-render-design.md`): render is **NOT** folded into PHI's op table — PHI stays purely the compute/device substrate, and the render *pipeline* (passes, AOV lifetimes, camera/sensor fan-out) is a new `src/rhi/` layer that **consumes** PHI. The second re-scope: the batched per-step sensor path is added as a first-class, zero-copy, tiled compute path (the prior draft was video-only).

---

## 1. Goal & owner constraints

- **PHI ≠ RHI.** PHI is the GPU compute + physics backend layer: a ggml-style multi-backend vtable (`src/phi/backend.hpp:60-94`: `BackendI` dispatch/synchronize/plan/event, `DeviceI`, `RegistryEntryI`; `BufferTypeI/BufferI` in `buffer.hpp`). CUDA today behind `NUKA_PHI2_WITH_CUDA`. PHI plays the role O3DE/Unreal call "RHI" — the hardware/device abstraction — but compute-first, not graphics-API.
- **RHI** = render hardware interface, split into **REALTIME** (Vulkan rasterizer, untouched) and **OFFLINE** (this doc). The OFFLINE RHI is **not a peer of PHI and not a second device layer** — it is a *client* of PHI: "a render pipeline expressed as a sequence of compute dispatches over PHI's Device/Buffer/Stream substrate." It plays the role O3DE calls "RPI" (passes / frame-scheduler), with PHI playing O3DE's "RHI" role.
- **Offline-first.** Do the OFFLINE RHI before any realtime Slang work. Its perf requirement is *stricter* than a normal offline renderer because the sensor path runs every render-tick, batched over many RL envs, co-resident with physics.
- **Dual purpose, one core.** ONE geometry/traversal core; replay vs sensor is a **config (`RenderProfile`), not a code fork** (CLAUDE.md ONE general path).
- **Inherits PHI selection.** The offline RHI holds the *same* `phi::Backend*`/`Device*` physics ran on. Selecting a PHI backend selects it for render; adding a PHI backend gives render that backend.
- **Pillars (state-preserving for every proposal):** D1 bit-determinism (no FP atomics, fixed reduction order); self-written / no external SDK (**Slang is an external compiler → `needs_owner_signoff`**); differentiable-sim compatibility; ONE general path (no per-scene hacks, no magic numbers, comments ≤ 2 lines).

**Naming hazard:** in Unreal/O3DE "RHI" = a *graphics-API* abstraction, which the offline path forbids. Nuka's "offline RHI" is deliberately not that — internally prefer the phrase "render pipeline on the compute substrate."

---

## 2. Current state (grounded, file:line)

**PHI is multi-backend-ready and is NOT device-0-locked; only the RT TUs lock device 0.**
- The vtable + device/stream/buffer seam exist and are backend-neutral (`backend.hpp:60-156`, `buffer.hpp`). The CUDA backend **owns its own main stream + capture stream + a per-backend `device_id`**: `DeviceInitBackendImpl` sets `cb->device_id = cd->device_id` and creates `cb->main`/`cb->capture` (`cuda_backend.cu:228-238`); `CudaBackendMainStream(b)` returns `b->main` (`cuda_backend.cu:367-369`); `BackendDispatchImpl` does `cudaSetDevice(cb->device_id)` then dispatches on `cb->main` (`cuda_backend.cu:71-76`). A grep of the PHI core finds no `cudaSetDevice(0)`/`ScopedDeviceGuard(0)`.
  - **Correction vs the prior draft:** only `dispatch` re-asserts `cudaSetDevice`; `synchronize`/`event_*`/`plan_execute` do not. This is benign (the streams are device-bound objects), but the offline-RHI entry point should itself re-assert `cudaSetDevice(cb->device_id)` before any thrust/LBVH work that needs the current device, exactly as `dispatch` does (`cuda_backend.cu:71-74`).
- **The one missing PHI capability:** ops are a **closed `NkOp` enum** (`op_schema.hpp:63-140`, **33** physics ops `ApplyDrives`…`NarrowphaseHeightfield` + `Count`, **zero render ops**), each a bespoke kernel in a `g_ops` table dispatched via `BackendDispatch(b, model, data, OpCall{op, params})` (`backend.hpp:63,109`). `BufferI` is only free/base/upload/download/memset/copy_from (`buffer.hpp:44-51`). There is **no** generic "dispatch program P over an N/2D/3D grid with buffer bindings + a push blob." `LaunchCuda` (`launch.cuh:21-29`) is a thin CUDA-only `<<<>>>` wrapper, not a backend-neutral primitive. *(The `op_schema.hpp` top-comment still says "30 ops"; it is stale — the enum is 33.)*

**The RT path bypasses PHI's launch/stream/device binding (debt, not a missing capability).**
- Three CUDA TUs hardcode device 0 + default stream 0: `two_level_render.cu` (`const int device_id = 0; ScopedDeviceGuard guard(device_id)` and `const cudaStream_t stream = nullptr; // default stream 0`, repeated at the BLAS/RenderFrame/RenderBeauty entries — verified at the offsets around `:936-938,966-968,1028-1030`), plus `scene_render.cu` and `bvh_ray_traversal.cu` with the identical pattern. Kernels are raw `RenderFrameKernel<<<>>>`/`RenderBeautyKernel<<<>>>`, synchronized on the default stream.
- **AOVs round-trip through the host.** `RenderFrame` allocates its *own* device buffers, launches, then `d_*.CopyToHost(...)` all 6 channels into a host `rt::Framebuffer` (`two_level_render.cu`, the `CopyToHost` block right after the kernel). The backend adapter then re-uploads: `CudaRtBackend::Trace` runs the host `RenderFrame` and `UploadAov`s back into the caller's device buffers — the **named RT-F2 debt**, documented at `rt_backend.hpp:20-29,108-116`.
  - **Correction vs the prior draft:** the per-pixel writes are *already device-side* (the kernel writes `out_color/out_depth/...`). The genuine debt is in the **adapter**: it downloads to a host `Framebuffer` then re-uploads (D2H→H2D). P0 removes the host round-trip and plumbs PHI/caller-resident device buffers into the kernel — it does **not** rewrite the per-pixel write logic.
- PHI is used for buffer *lifecycle* (`OwnedBuffer` → `BufferAlloc`/`BufferUpload`/`BufferDownload`/`BufferFree`), but the device is **re-derived** via `InitBestDevice()` (`registry.cpp` picks the first device of the first backend, takes no selection arg) instead of the selected `Device*`/`Backend*`. So the AOV buffers are already `phi::Buffer*` — the zero-copy fix is *easier*, not harder.

**The shared BVH + refit already exist (the key leverage).**
- The LBVH is built by the same `collision::gpu` code for both broadphase and the tracer; PHI even has broadphase as real ops (`op_schema.hpp`: `BuildAabbs`/`LbvhBuild`/`LbvhQueryPairs`).
- **Refit-not-rebuild is implemented:** `RefitLbvh(cudaStream_t stream, int device_id, LbvhNode* device_nodes, const collision::AABB* device_new_aabbs, uint32_t leaf_count, uint32_t* device_visit_scratch)` (`collision/lbvh_refit.cuh:24`, def `lbvh_refit.cu:82-105`) re-propagates fresh leaf AABBs up an **existing** tree (topology reused; leaf→body id stored in `nodes[].left`; uint32 atomic visit-counter; D1). **It currently has zero call sites** — it is an unwired primitive ready to use.
- Two-level split is already right for dynamics: BLAS is built once per unique mesh and **never refit** (`BuildBlas`, called once per mesh from `BuildTwoLevelScene`, `two_level_render.cu:943`); the TLAS is **rebuilt** per frame (`BuildFrameTlas` via `BuildLbvhForQuery`, called from RenderFrame/RenderBeauty). Refit applies to the TLAS scale (≤ `kMaxInstances`, `prim_id.cuh`) — cheap — while BLAS topology is permanent.

**The zero-copy physics→render pose seam exists (implemented; owner-verified, not CI-proven).**
- `nk::World::FieldPtr(FieldId)` returns a device pointer (`world.hpp:102`, def `world.cpp`); `LinkPose/BodyPose/BasePose` are owner:data fields so `FieldPtr` routes to the data arena. `nk::World::Backend()` returns the opaque `phi::Backend*` (`world.hpp:118`).
- The live consumer (`CudaVulkanInteropPublisher::Publish`) fills `ScatterFkSource` from `Data::Ptr(LinkPose/BodyPose/BasePose)` (== `FieldPtr` for these fields) with **no D2H** and sets `fk.world_backend = world.Backend()` (`interop_scatter.hpp:107-135`). **Correction vs the prior draft:** the cross-stream ordering is implemented with **raw** `cudaEventRecord(fk_done_, main_stream)` + `cudaStreamWaitEvent` inside `cuda_vk_scatter.cu` (casting `Backend*`→`CudaBackend` via `CudaBackendMainStream`), **not** the `phi::BackendEventRecord/Wait` vtable wrappers at `backend.hpp:118-121` (those exist but are unused by this path). And it is **owner-verified on NVIDIA hardware, never run in this CI box** (lavapipe can't share CUDA memory) — call it "implemented + fallback-tested," not "proven."
  - **Design implication:** the offline RHI must decide whether to keep this raw-CUDA reach-through or **promote ordering onto the `phi` event vtable** for multi-backend (HIP) portability. This design recommends the vtable (§5, §8). The existing scatter is also a *raster*-path artifact reading one selected env; the batched sensor render reuses the same `FieldPtr`/`Backend` seam + ordering pattern but needs its **own GPU-compute consumer**, not the SSBO scatter.

**Sensors today are CPU/host for lidar; no image sensor exists.**
- Lidar: `QueryLidarSensor` is a **host CPU loop** over `collision::StaticBVH::Raycast` (a host `std::vector` BVH, explicit-stack recursive Raycast, `std::optional` hit), one ray at a time, returning a host `SensorPacket{ std::vector<float> depth_buffer }` (`ray_sensor.cpp:11-46`, `static_bvh.hpp`, `sensor_packet.hpp`). Not batched, not on GPU.
- IMU/JointState are host analytic functions (`state_sensor.hpp`) — correct as-is. Contact-wrench **is** a GPU PHI op (`ReadoutContactWrench`) — GPU-resident, but a solver readback, not a render sensor. **Correction vs the prior draft's "sensors are CPU-only":** `sensor/contact_wrench.cu` and `sensor/noise/*.cu` (Philox) are already GPU-resident; it is the **lidar (range) + IMU/JointState** paths that are host. The accurate statement is "the range/image sensor path is host-only and unbatched."
- `SensorType` is exactly `{IMU, JointState, ContactSummary, Lidar, Depth}` (`sensor_graph.hpp:16-22`) — **no RGB/segmentation/normal/instance image sensor**. The only ray-vs-scene image render is the offline AOV tracer (`scene_render.cu`, self-described "the SENSOR renderer"), but it is single-camera, single-image, host-orchestrated, device-0/stream-0-bound. **History note:** a batched GPU camera sensor *existed and was deleted* (commit `9670a42` added `src/sensor/camera_render.cu` + `gpu/cuda_sensors.cu`; the M0 cleanup `c28a2db` removed them with the deleted `DeviceWorld`). So this is **re-home/re-create on PHI**, reusing the surviving backend-neutral RT headers + `scene_render.cu` AOV logic — not green-field invention. The scene layer already carries a data-only `scene::SensorType::Camera` + `CameraRecord` (fov/clips/transform, no buffer) the new path can consume for intrinsics.

**What is already portable (the reuse the design rests on).**
- Math (`src/math/*`, `NUKA_MATH_HD`) and the RT intersection/shading headers (`rt/intersect_primitives.cuh`, `ray_box.cuh`, `instance_transform.cuh`, `shading.cuh`, `prim_id.cuh`, `camera.hpp`) carry **no `cuda_runtime`, no `<<<>>>`**, all `NUKA_RT_HD`-gated.
  - **Caveat 1:** `bvh_traverse_impl.cuh` is **`__device__`-gated, not `NUKA_RT_HD`**, and it includes `collision/lbvh_node.cuh` which `#include <cuda_runtime.h>`. So *traversal + LBVH node defs are CUDA-locked* and need a `NUKA_RT_HD` re-macro (and host/device-clean `LbvhNode`) for a 2nd backend.
  - **Caveat 2:** `src/math/cuda_vec_ops.cuh` and `cuda_spatial_ops.cuh` pass the grep but `#error` on non-nvcc and use bare `__device__`. They are *physics-op* helpers (not on the RT path), so they don't threaten the RT port, but "all of `src/math` is backend-neutral" is false for them.
- The traversal is **`template <typename LeafFn> __device__ void TraverseRay(...)`** (`bvh_traverse_impl.cuh:45-46`, "ONE traversal, two leaves" `:8-10`). The closest-hit nest (`BlasLeaf` `:196`, `TlasLeaf` `:222`, `ClosestHit` `:254` in `two_level_render.cu`) updates the same `best_t`/`best_prim` by reference and is reused verbatim by primary, shadow, and beauty secondary rays. *(Terminology: the two "leaves" here are the two-level TLAS→BLAS functors; the header's original "two leaves" meant box-leaf vs prim-leaf — orthogonal axes, both true.)*
- **The AOV set already produced by one closest-hit is exactly the sensor channel set:** color(3f)/depth(1f)/normal(3f)/albedo(3f)/uv(2f)/prim(u32) (`framebuffer.hpp:36-41`); `depth = best_t` is true world distance because `ray.dir` is unit (`camera.hpp`). `prim` packs `(instance_high 12b, local-prim 20b)` via `PackPrimId` (`prim_id.cuh:40-49`); the tie-break `t < best || (t==best && prim < best_prim)` lives in **`ray_box.cuh:163-171`** (`RtClosestHitUpdate`) and is a total order → D1, with instance/semantic segmentation a near-free relabel.
  - **Correction:** the "DLPack contract" in `framebuffer.hpp:14-19` is a **comment describing an aspirational layout**, not implemented code. `rt::Framebuffer` is a host struct of `std::vector`; no DLPack capsule wraps it.
- **DLPack zero-copy obs ships for physics state only.** `torch.from_dlpack(world.buffer_view(Field.X))` (`python/nuka/__init__.py:14`, `autograd.py:70`, `jax_frontend.py:102`), backed by `c_abi/dlpack_table.hpp`: **23 rows, of which 20 map to live nk arena fields** (`TORQUE_INPUT` aliases `DriveTarget`) and **3 (`VELOCITY_TARGET`, `ACTUATOR_NOLOAD_SPEED`, `TASK_TARGET`) map to the `kNoFieldId` sentinel and return `NOT_SUPPORTED`** (`dlpack_table.hpp:113-116`, `buffer.cpp:82-84`). Only fields 0..19 are byte-pinned by the RL contract (`dlpack_table.hpp:144-146`). There is **no** DLPack/torch export for render/AOV outputs — AOV export is a new view *family* (§4.3).

So only three things are CUDA-locked on the RT path: the `<<<>>>` launches, the LBVH build/refit, and stream/device binding (plus the `__device__`-gated traversal header). The first and third are exactly what moving onto PHI fixes; the second is shared with physics.

---

## 3. Architecture: PHI vs RHI separation

**Dependency arrow (one direction, no cycle):**

```
scene / RenderWorld  (host, CUDA-free — the shared data product feeding raster AND offline)
        │
        ▼
src/rhi/offline      (host, CUDA-free: passes, profiles, AOV lifetimes, camera/sensor
        │             fan-out — names ONLY phi opaque handles, never a graphics API)
        ▼
src/phi              (Device / Buffer / Stream / dispatch — the compute substrate)
        │
        ▼
src/phi/backend_cuda/rt/...   (per-backend render-pass KERNEL BODIES — may name CUDA)
```

`src/rhi/offline` names only `phi::Device*` / `phi::Backend*` / `phi::Buffer*` / `phi::BufferType*` (all opaque, CUDA-free), the same zero-CUDA red-line `src/render/` already holds (`rt_backend.hpp` "names ONLY phi v2 opaque Buffer*/BufferType*").

**NEW — `src/rhi/offline/` (host, CUDA-free):**

| File | Role | Analogue |
|---|---|---|
| `render_graph.hpp/.cpp` | pass DAG + AOV/scratch buffer lifetimes + ping-pong; backend-neutral C++ orchestration | O3DE RPI FrameScheduler; LuisaCompute "Stream records commands" |
| `render_pass.hpp` | `RenderPassId` enum (RayGen, TraverseClosest, ShadePrep, ShadowConnect, ExtendGI, Compact, Accumulate, Denoise, Tonemap, WriteAov) + per-pass POD params | the render analogue of `op_schema.hpp` — a **separate** closed enum, NOT folded into `NkOp` |
| `render_profile.hpp` | `RenderProfile` (camera set, AOV mask, secondary-ray depth, SPP, denoise mode, determinism mode, output target) — selects the pass list as DATA | — |
| `offline_renderer.hpp/.cpp` | public façade: `BuildScene(RenderWorld)→Handle`; `Render(Handle, profile, RenderTargets, Stream)`; absorbs today's `render::RtBackendI` | — |
| `sensor_render.hpp/.cpp` | batched `RenderSensorBatch(scene, CameraSet, AovBatch, Stream)` (tiled `[N,H,W,C]` in one pass) + `RangeSensorBatch` (lidar/height-scan on the same TLAS) | Isaac-Lab TiledCamera + RayCaster as ONE path |
| `aov_export.hpp/.cpp` | DLPack / `__cuda_array_interface__` wrapper of the tiled device AOV → torch, reusing the `nuka_buffer_view_t` descriptor + nanobind caster | the RL obs zero-copy seam |
| `replay.hpp/.cpp` | cadence-gated frame capture + mp4 encode feed (training-time periodic + post-train demo) | Isaac-Lab `RecordVideo` analogue |

**ADD to PHI (the one missing capability):**

| File | Role |
|---|---|
| `src/phi/dispatch.hpp` | a new `BackendI` vtable slot, **co-existing** with the existing `dispatch(OpCall)` (zero churn to the 33 physics ops): `Status DispatchProgram(Backend*, ProgramId, const BindGroup&, GridDims, const void* push, size_t push_bytes)` on the backend's main stream + an opaque `BindGroup` (a list of `Buffer*`). |
| `src/phi/render_program_schema.hpp` *(optional split)* | the render-pass `ProgramId` table, beside `op_schema.hpp` but a **distinct program family**. Physically separate from `NkOp`. |

The device **capability** (DispatchProgram) belongs in PHI; the render **pass registry + pass DAG** belong in RHI. Physics ops may migrate onto `DispatchProgram` opportunistically; not required.

**PHI CUDA backend (per-backend kernel BODIES, may name CUDA — `src/phi/backend_cuda/rt/`):**
- KEEP the portable bodies (`rt/*.cuh`) and the LBVH build/refit (`collision/`).
- REWRITE `two_level_render.cu` + `rt_backend_cuda.cpp` so kernels write the caller's `phi::Buffer*` AOVs **in place** on the **selected** backend's main stream (pay RT-F2), launched via `launch.cuh` (the sole `<<<>>>` wrapper) under `DispatchProgram`, registering passes in a `g_render_programs` table parallel to `g_ops`. Kill `InitBestDevice()` re-derivation + `ScopedDeviceGuard(0)` + the default stream.
- CMake: `nuka_phi2_rt` (the RT slice of the CUDA backend, `--fmad=false`) becomes the CUDA implementation of the offline RHI. A new CUDA-free `nuka_rhi` links like `nuka_render` does today, with a `CreateCudaOfflineRenderer()` strong/weak factory mirroring `CreateCudaRtBackend` / `CreateCudaVkScatter`.

**UNCHANGED (owner mandate):** `src/render/` realtime Vulkan rasterizer stays as-is. `RenderWorld` stays the shared CUDA-free data product feeding both realtime and offline. Slang is realtime's future, not this work.

---

## 4. Offline RHI core + sensor-render path

### 4.1 One general path: shared geometry core + per-purpose config

The ONE-path pillar forbids a sensor codepath separate from a replay codepath (exactly the split Isaac Lab has: Warp RayCaster vs RTX tiled). Resolve it as **ONE geometry/traversal core + a per-purpose pass list selected by `RenderProfile`**.

**Shared core (identical bytes for both purposes):**
1. **Acceleration structure:** the shared LBVH (BLAS built once, TLAS refit/rebuilt per step from physics device buffers, §4.4).
2. **Ray generation:** a `RayGen` emitting SoA primary rays for a *camera set* — one pinhole per env-tile for sensors, one cinematic camera for replay, a direction-table per env for lidar. Same kernel, different camera/pattern buffer length.
3. **Closest-hit traversal:** the existing `ClosestHit` nest (`two_level_render.cu:254`), reused verbatim via the templated `TraverseRay` leaf — already produces depth (`best_t`), normal, albedo, uv, instance/semantic id (`UnpackPrimId`). The **only** thing varying between purposes is the leaf payload: **distance-only** (lidar/depth) vs **full G-buffer write** (rgb sensor) vs **shade** (replay) — `bvh_traverse_impl.cuh`'s leaf-functor template extended to three payloads.

**Per-purpose config (`RenderProfile`), NOT a codepath:**

| Profile field | SENSOR (in-loop) | REPLAY (cinematic) |
|---|---|---|
| camera set | N env tiles (e.g. 64×64) or lidar pattern | 1 hero camera (e.g. 1080p) |
| AOVs requested | depth / seg(id) / normal / instance (/ rgb) | RGBA color |
| secondary rays | 0 (exact geometry) or 1 shadow for rgb | shadow + AO + 1-bounce GI |
| samples/px (SPP) | 1 | 1 + ReSTIR reuse + TAA |
| denoise | none | single-frame à-trous (golden) / temporal SVGF (fast) |
| output | device buffer → obs tensor (zero-copy) | device → RGBA8 → mp4 (async D2H ok) |
| determinism | exact (no temporal, no atomics) — D1 | golden mode exact / fast mode flagged |

The pass list is **data**: a sensor profile runs `{RayGen, TraverseClosest, WriteAov}`; a replay profile runs `{RayGen, TraverseClosest, ShadePrep, (ReSTIR), ExtendGI, Compact, Accumulate, Denoise, Tonemap}`. Both lists call the **same programs over the same buffers**; replay just enables more passes. Depth/seg/normal are exact in **both**. A sensor is a replay with the shading/GI passes disabled — no second tracer, no per-scene branch, no magic numbers (CLAUDE.md). This kills the prior draft's biggest risk: a "beauty kernel" and a "sensor kernel" diverging.

### 4.2 Batched tiled layout + the occupancy problem

N envs × a small tile (64×64) is many tiny, independently-divergent ray sets. A one-block-per-tile megakernel under-occupies. Do **wavefront over ALL rays of ALL tiles at once**, persistent-thread style:

1. **One global SoA ray pool** across the whole batch: `RayGen` writes `(tile_h·tile_w·N)` primary rays into one flat `(origin, dir, pixel_id, env_id)` buffer. The grid is sized to the GPU, not to one tile — full occupancy from ray 0 (Laine/Karras, *Megakernels Considered Harmful*).
2. **TraverseClosest** is one dispatch over the whole pool against a single **env-tagged TLAS**; results to a flat hit buffer. `env_id` carried per ray → `WriteAov` scatters to `aov[env_id][pixel_id]`. A ray accepts only hits with its own `env_id` (one integer compare in the leaf — the Madrona "world-id" trick) so ONE env-tagged TLAS replaces N trees — the single biggest throughput lever (§4.4).
3. **Stream-compact** live rays between bounces (REPLAY only; sensors are one closest-hit). Compaction is order-preserving → deterministic.
4. **Tiled AOV layout** mirrors Isaac Lab's stitched buffer: one big device buffer per channel `[N_tiles, tile_h, tile_w, channels]` (env-major, tile = `f(env_index)`) so the learner reads it as one tensor with no gather; per-env slicing is a view.

**Megakernel-vs-wavefront split by purpose:** the SENSOR pass MAY stay a megakernel (shade is shallow + coherent; one closest-hit) — but it must launch over the whole batch (one grid over N·tile pixels), not per tile. The REPLAY beauty pass uses the wavefront split because deep multi-material GI is divergent. Both call the same `TraverseRay` / `rt/*.cuh` bodies; only host orchestration (one dispatch vs many) and shade depth differ.

### 4.3 Zero-copy AOV → torch (the hard contract)

For sensors the AOV tensor MUST be a device buffer the RL obs reads in place, never the host round-trip:
- AOV buffers allocated from the **active backend's** type via `BufferAlloc(BackendDeviceBufferType(backend), bytes)` (`backend.hpp:155`) so transfers + render kernels share the physics main stream (the D1 stream anchor — `backend.hpp` notes "stream choice is the D1 anchor").
- Export via DLPack capsule, **reusing** the `nuka_buffer_view_t` descriptor (`nuka.h:500-507`: device_ptr/element_count/element_stride_bytes/dtype) and the nanobind ndarray caster `make_array_from_view` (shared by `buffer_view` and `state_view`), via a new `sensor_view(SensorId, AovChannel)` C-ABI entry returning that descriptor with shape `[N,H,W,C]` + wire dtype (f32 for depth/normal/rgb-float, u32 for instance/semantic id, uint8 for rgb8), consumed as `torch.from_dlpack(world.sensor_view(...))`. **Correction:** `c_abi/dlpack_table.hpp` itself is **not** reusable — it is keyed by `nk::FieldId` arena fields and AOVs are not arena fields; the new view needs its **own** channel descriptor source (the `rt::Framebuffer` 6-channel set). The seam is thin (reuse struct + caster + a `state_view`-style entry point), not zero. The host-download path stays ONLY for the video encoder + D1 golden tests.
- **Co-residency / ordering:** render reads instance transforms from `World::FieldPtr` device buffers (no `DownloadField`), ordered behind the physics step. **This design recommends ordering via the `phi` event vtable** (`BackendEventRecord`/`BackendEventWait`, `backend.hpp:118-121`) rather than the raw-CUDA reach-through the existing raster scatter uses (`cuda_vk_scatter.cu`), so the ordering is multi-backend-portable (HIP). The pattern is otherwise identical to `ScatterFkSource.world_backend` (`interop_scatter.hpp:127-135`).

### 4.4 Sharing ONE scene/BVH between physics and render (refit per step)

1. **BLAS** (per unique mesh): built once at scene cook (`BuildBlas`, `two_level_render.cu:943`), retained, **never refit** — rigid meshes are static in their own frame. No per-step cost.
2. **TLAS** (instances): instead of allocating fresh buffers + a fresh tree every frame (`BuildFrameTlas` today), **refit the retained TLAS** with `RefitLbvh` (`lbvh_refit.cuh:24`) over current instance world-AABBs computed on-device from the physics FK field buffer (`FieldPtr`), zero-copy. Rebuild topology only on a cheap trigger (instance-count change, or a periodic SAH refresh) — refit degrades under large rearrangement (looser boxes), so a periodic rebuild bounds quality; the interval is a quality/throughput dial, **not** a per-scene hack.
3. **Env-tagged single TLAS** for batched sensors: one TLAS over all envs' instances, each leaf tagged with `env_id`; the ray's leaf functor rejects cross-env hits (one compare). Single-env replay just uses all instances.
4. **Instance transforms come from the physics state device buffer** — `DevInstance.transform` filled from `FieldPtr(LinkPose/BodyPose/BasePose)` on-device, the same data the interop scatter reads bit-exactly. The render never downloads pose.

Net: per step the render adds a TLAS refit (kernels already in `lbvh_refit.cu`, over ≤ `kMaxInstances`) + the traversal, sharing the structure physics just used.

### 4.5 Camera & range-sensor models (all are ray-gen variants, ONE path)

- **Pinhole** exists (`camera.hpp:45-141`, fp64-internal `GenerateRay`, `NUKA_RT_HD`).
- **Fisheye / equidistant** (`r = f·θ`) and **Brown-Conrady distortion** are different pixel→ray maps; add as `GenerateRay` variants in the `NUKA_RT_HD` camera header. Traverse/shade identical. Keep fp64-internal for D1.
- **Lidar / height-scan / range sensors** = a non-pinhole ray-gen *pattern* on the SAME env-tagged TLAS: spinning = (azimuth × elevation) grid, solid-state = a device direction table, height-scan = a ground grid. Per ray return `range = best_t` and optionally `intensity = cos(incidence)/r² · albedo`. Batched over envs like cameras. **This deletes the CPU `StaticBVH` lidar path** (`ray_sensor.cpp`) and subsumes it into the GPU path — ONE BVH, two leaf payloads. The project's legged_gym terrain height-scan obs becomes one such pattern on the unified path.

### 4.6 The AOV set (what one closest-hit produces, no rasterizer)

From one closest-hit (all already in `framebuffer.hpp:36-41` + `prim_id.cuh`): `depth/range = best_t`; `distance_to_image_plane = best_t·dot(dir, forward)`; `normal = ReconstructHit`; `instance_segmentation = high bits of PackPrimId`; `semantic_segmentation = per-instance class LUT[instance]` (the kernel already reads `instances[inst].material_id` — a near-free relabel); `albedo`/`uv` from the hit. The one genuinely new AOV is **motion vectors**: it needs **new state** — a per-instance prev-frame transform device buffer (none exists today) — then transform the current hit back to the prev pose, project through the prev camera, take the screen-space delta. All atomic-free, one-writer-per-pixel → D1.

### 4.7 Replay-video path (quality tracer, ONE core with sensors)

Replay is the SAME core with shade/GI/denoise passes enabled (§4.1 table). Reference Isaac Lab's recorder shape (a thin `gymnasium.RecordVideo` wrapper, not a custom renderer): a cadence-gated wrapper that, every `video_interval` steps or on a post-train demo run, points a hero camera at the selected env, renders the FAST profile, async-D2H's the RGBA8 on a copy stream (the host copy is fine here — once per clip, not in the inner obs loop), and muxes to mp4. Because replay reuses the in-loop ray pool + shared BVH, "render a training replay" and "produce a sensor image" are the same machinery at different profiles — no separate offline renderer.

Today's replay (`c_abi/recorder.cpp`, `python/nuka/recorder.py`) uses the offscreen **Vulkan rasterizer**, single env (env_index 0), PPM→ffmpeg — that is the realtime path and is orthogonal; the offline-RHI replay is the path-traced cinematic alternative, sharing the sensor core. Known reuse caveats from Isaac: `RecordVideo` leaks memory over long runs (#1996) and is version-fragile (#875) — keep the cadence wrapper thin and own the encoder feed.

---

## 5. Determinism (D1), differentiability, no-SDK

- **Sensor profile is exact by construction:** 1 closest-hit/ray, `depth = best_t`, seg/instance from the total-order `RtClosestHitUpdate` tie-break (`ray_box.cuh:163-171`: `t < best || (t==best && prim < best_prim)` — total order on the packed instance/prim id), one writer per pixel, no FP atomics → bit-identical run-to-run within a backend/device (the `framebuffer.hpp:18-19` two-run memcmp-identical property; the existing `tests/rt/` D1 two-run + analytic-anchor tests already gate this). **NO temporal reuse, ever.** This is the regression / golden / diff-sim path and the **default** for sensors.
  - **Honest caveat:** node pruning uses entry-`t` with strict `<` (`bvh_traverse_impl.cuh`), so the "winner is canonical independent of visit order" claim is overstated only at the measure-zero boundary where two prims in different subtrees have bit-identical hit `t` *and* the later one's AABB entry_t equals that `t` in fp32 (a grazing near-face hit). This does **not** affect D1 — traversal order is itself fixed across runs, so two runs are byte-identical regardless. A one-line hardening (tie-aware pruning) would close the canonicity edge if a cross-build-order guarantee is ever needed; not required for the stated determinism.
- **Replay split into two modes, ONE geometry pipeline:** GOLDEN = fixed-order single-frame à-trous, no ReSTIR/TAA, counter-based RNG keyed by (env, pixel, sample, bounce), regeneratable — the byte-exact `RenderFrameKernel`-class path. FAST = wavefront + ReSTIR + temporal SVGF + TAA, carries `temporal_reuse=on` that **hard-disables golden comparison**. Never silently loosen a tolerance or regen a golden (CLAUDE.md). Cross-vendor: bit-exact *within* a backend, equivalent-within-tolerance *across* vendors (FMA/transcendental rounding differ — bit-identical cross-vendor is not achievable; state it plainly).
- **Differentiability:** the closest-hit geometry path (depth/normal/segmentation as piecewise-smooth functions of pose; silhouette visibility is the known non-diff locus, as in every diff-renderer) is differentiable-friendly in GOLDEN mode and lives on the same arena/DLPack device buffers, so a backward pass can hang off the same device fields if a grad-through-pixels consumer ever appears. ReSTIR/denoise/temporal (FAST) are not differentiable and live only in FAST replay. For the RL-obs use, gradients flow through the policy, not the pixels — so wiring sensor AOVs into the diffsim tape is **deferred to a real consumer** (Open Decision #3).
- **No-SDK:** CUDA + a single-source CUDA/HIP macro header need no external compiler (the `rt/*.cuh` bodies are already `cuda_runtime`-free; the exceptions are `bvh_traverse_impl.cuh` + `lbvh_node.cuh`, which need a `NUKA_RT_HD` re-macro and a host/device-clean `LbvhNode` to compile under hipcc). **Vulkan/SPIR-V breadth would need Slang (an external compiler) → `needs_owner_signoff`.** CUDA+HIP cover the perf driver (NVIDIA + AMD) with zero new dependency; the hand-written-SPIR-V alternative is a multi-year compiler project and would force abandoning the header-only math core — not recommended.

---

## 6. Multi-backend (inherits PHI selection automatically)

Because the offline RHI holds a `phi::Backend*`/`Device*` (the SAME one physics ran on) and dispatches via `DispatchProgram`, selecting a PHI backend selects render for free.

- **CUDA** (unify first): the existing tracer, moved onto the selected device/stream with device-resident AOVs.
- **HIP/ROCm** (second): a single-source `__global__`/`__device__` macro header over the already-neutral `rt/*.cuh` bodies compiles under nvcc AND hipcc with near-zero rewrite. The genuine port surface beyond launch/stream/device is the `__device__`-gated **traversal** (`bvh_traverse_impl.cuh`) + **`LbvhNode`** (re-macro to `NUKA_RT_HD`, drop the `cuda_runtime` include from the node header) and the **LBVH build/refit**. Promote cross-stream ordering onto the `phi` event vtable (§4.3) so it ports too.
- **Vulkan-compute / SPIR-V** (breadth): needs **Slang** (external compiler) → `needs_owner_signoff`. Defer to this phase only.

Both physics AND render gain each new backend.

---

## 7. Phased implementation (each shippable + profiling-gated)

- **P0 — Unify RT onto PHI, device-resident AOVs (no perf change, D1 byte-exact).** Take the selected `Device*`/stream (kill `ScopedDeviceGuard(0)` + default stream in `two_level_render.cu` / `scene_render.cu` / `bvh_ray_traversal.cu`); plumb caller/PHI-resident `phi::Buffer*` AOVs into the kernel and drop the adapter's host download + `UploadAov` (pay RT-F2). Stand up `src/rhi/offline/` host skeleton + the `CreateCudaOfflineRenderer` factory. Re-assert `cudaSetDevice(cb->device_id)` at the render entry. **Gate:** D1 goldens byte-identical (`tests/rt/` two-run memcmp), no new host copy in the device path.
- **P1 — Sensor profile = the strict path.** `RenderSensorBatch`: RayGen for an N-tile camera set; wavefront intersect over one global ray pool against the env-tagged TLAS; TLAS **refit** per step via `RefitLbvh` with instance AABBs from `FieldPtr` (zero-copy); scatter depth/seg/normal/instance into one tiled device AOV; export as a DLPack obs tensor (`sensor_view` C-ABI, reusing `nuka_buffer_view_t` + `make_array_from_view`). Wire into the RL loop next to the obs export (`python/nuka/rl_games/vecenv.py`). **Gate:** rays/s, occupancy, per-pass ms, step-inflation factor vs the §9 budget at N=1024×64×64; **zero D2H per step**.
- **P2 — Unify lidar + add the dispatch primitive.** Add `DispatchProgram` + `g_render_programs` to PHI; migrate P0/P1 passes onto it (profiles become pass lists). Delete the CPU `StaticBVH` lidar path; lidar = device direction-table ray-gen on the same TLAS (`RangeSensorBatch`); add fisheye/distortion ray-gen + motion-vector (new prev-transform buffer) + semantic-LUT AOVs. **Gate:** lidar parity vs the old CPU path within tolerance; height-scan obs on the unified path matches the legged_gym obs.
- **P3 — Replay FAST profile + recorder.** Wavefront beauty passes (ShadePrep, ReSTIR DI, ExtendGI, Compact, Accumulate, SVGF/à-trous denoise, Tonemap) + the cadence wrapper + mp4 encode (training-time periodic + post-train demo). **Gate:** GOLDEN replay mode byte-exact; FAST mode flagged off the goldens; cinematic fps **measured**, not asserted.
- **P4 — Second backend (HIP) via single-source header**, then Vulkan/SPIR-V (owner-gated on Slang). **Gate:** cross-vendor within-tolerance AOV equivalence; within-backend bit-exact.

---

## 8. Honest perf (bankable vs aspirational — static extrapolation, NOT measured)

These are static extrapolations from published numbers under the no-profile constraint; they must be **measured at the P1 gate** before being asserted as achieved.

- **Physics baseline:** ~16,367 env-steps/s @N=1024 ⇒ ~61 µs/step, **measured + logged** (`out/perf/union_throughput_2026-06-15.log`) — but it is the **pre-collapse, pure-engine** union path with no action/obs/PPO in the loop. The current general-contact path (per-step LBVH broadphase + cvx narrowphase over all bodies) adds cost and is expected slower; no committed steady-state result log exists. Treat ~61 µs as a favorable *ceiling*, not the general-path step time.
- **Sensor workload:** a 64×64 tile = 4096 rays/env; N=1024 ⇒ **4.19M primary rays/step**. Cross-checks (single GPU, RTX-class): Isaac Lab tiled @64×64 ≈ 250k–300k env-FPS over 4096 envs, and its **GPU-compute Warp RayCaster** (depth/normals only) is *marginally faster* than RTX tiled — direct validation of the GPU-compute-only sensor approach (arXiv:2511.04831); Madrona CUDA batch tracer ≈ 403k steps/s @64×64 simple (37k Franka, physics-bound); ManiSkill3/SAPIEN ≈ 30k+ FPS RGBD+seg @4090 in 3.5 GB (vs Isaac's 14.1 GB — the case for a lean compute renderer).
- **Bankable conclusion:** with vision on, the **step rate becomes render-bound at a few hundred → low-thousands of env-FPS** (consistent across Isaac/Madrona/SAPIEN). Do **not** quote sub-ms/step as achieved before measurement.
- **Why device-resident is the whole point, not an optimization:** a single per-step D2H of 4.19M×4 B ≈ 17 MB at PCIe ~12 GB/s is ~1.4 ms of pure copy that **serializes** the step — on TOP of the trace, scaling with N. Today's RT-F2 host round-trip (D2H *and* H2D) is categorically disqualified for in-loop sensors. Removing it is the difference between feasible and infeasible.
- **Decimation** (Isaac `update_period`): sensors render every K physics steps; a separate `video_interval` drives replay. A render-tick decimator lets the strict sensor path co-exist with high physics throughput.
- **Replay (aspirational):** the prior draft's wavefront + CWBVH + ReSTIR + temporal SVGF @ ~30 fps@1080p target. Honest landing is ~50–80× from FP32 + wide-BVH + low-SPP + SVGF; full 30 fps needs the whole stack and must be measured — do not oversell 33 ms pre-measurement.

---

## 9. Open decisions for the owner

1. **Slang** for Vulkan/SPIR-V (and Metal) codegen breadth — external compiler, `needs_owner_signoff`. CUDA+HIP need none and cover the perf driver (NVIDIA + AMD). *Recommendation: gate Slang to P4 only.*
2. **Sensor default resolution(s) + which AOVs are in the hot loop** — 64×64 depth+seg is the Isaac-Lab-matched default and nearly free (geometry-only, exact, D1); rgb adds the shadow ray (~2× rays); higher res trades the §8 budget linearly. Owner picks the standard task resolution + the in-loop AOV mask.
3. **Wire sensor AOVs into the diffsim tape now, or keep the differentiable surface = GOLDEN sensor pass only?** Only worth it if a grad-through-pixels consumer exists; the RL-obs use does not need it. *Recommendation: defer.*
4. **TLAS rebuild interval** (refit-quality dial) — every step vs periodic full rebuild on instance-count change + every K steps. Throughput vs traversal quality. *Recommendation: conservative periodic rebuild.*
5. **Quality bar for replay** — ~30 fps@1080p (FAST stack) vs higher-res/lower-fps offline export.
6. **Confirm the module/naming layout** — a new `src/rhi/offline/` tree (host, CUDA-free) consuming PHI, with the render PASS registry + pass DAG in RHI (NOT folded into `NkOp`), and the generic `DispatchProgram` capability added to `BackendI`. This is the core re-scope vs the prior draft (which extended PHI with render ops). Confirm before P0 stands up the directory + the `CreateCudaOfflineRenderer` factory.
7. **Cross-stream ordering home** — promote FK→render ordering onto the `phi` event vtable (`backend.hpp:118-121`) for HIP portability, vs keeping the existing raw-CUDA reach-through used by the raster scatter (`cuda_vk_scatter.cu`). *Recommendation: the vtable.*

---

## References (verified file:line)

`src/phi/backend.hpp:60-156` (PHI vtable: `BackendI`/`DeviceI`/`RegistryEntryI`, event wrappers `:118-121`, `BackendDeviceBufferType` + the D1 stream-anchor note), `buffer.hpp:44-51` (BufferI surface), `op_schema.hpp:63-140` (closed `NkOp`, 33 ops, zero render ops; stale "30 ops" comment), `backend_cuda/cuda_backend.cu:228-238` (per-backend main/capture streams + `device_id`), `:367-369` (`CudaBackendMainStream`), `:71-76` (`cudaSetDevice(cb->device_id)` then dispatch on `cb->main`), `backend_cuda/launch.cuh:21-29` (sole `<<<>>>` wrapper), `backend_cuda/rt/two_level_render.cu` (BLAS-once `BuildBlas` `:943`; per-frame TLAS rebuild; `BlasLeaf`/`TlasLeaf`/`ClosestHit` `:196,:222,:254`; device-0 `ScopedDeviceGuard` + null default stream at the BLAS/RenderFrame/RenderBeauty entries ≈ `:936-938,966-968,1028-1030`; 6-channel `CopyToHost` host round-trip after the kernel), `backend_cuda/rt/rt_backend_cuda.cpp` (CUDA `Trace` = host `RenderFrame` + `UploadAov` = RT-F2; `InitBestDevice`), `backend_cuda/rt/bvh_traverse_impl.cuh:8-10,45-46` (templated `__device__ TraverseRay`; **`__device__`-gated, not `NUKA_RT_HD`**), `backend_cuda/rt/ray_box.cuh:163-171` (`RtClosestHitUpdate` total-order tie-break = D1 + free seg), `backend_cuda/rt/prim_id.cuh:40-49` (instance-high pack/unpack), `backend_cuda/rt/camera.hpp:45-141` (PinholeCamera fp64 `NUKA_RT_HD`), `rt/framebuffer.hpp:14-19,36-41` (AOV set = sensor channels; the DLPack layout is an **unimplemented comment**; two-run memcmp D1), `render/rt_backend.hpp:20-29,79-86,108-146` (`RtBackendI` device-buffer contract + RT-F2 named debt + `CreateCudaRtBackend` factory; `RtAovBuffers`), `collision/lbvh_refit.cuh:24` (refit primitive, exact signature; **zero call sites**), `collision/lbvh_refit.cu:82-105` (refit def), `collision/static_bvh.hpp` (CPU lidar tree — the gap), `sensor/ray_sensor.cpp:11-46` (CPU lidar loop), `sensor/sensor_graph.hpp:16-22` (SensorType, no image sensor), `nk/pipeline/world.hpp:102,118` (device `FieldPtr` + `Backend()`), `phi/interop_scatter.hpp:107-135` (zero-copy pose seam via `Data::Ptr`; `ScatterFkSource.world_backend` cross-stream ordering — implemented with **raw** `cudaEventRecord`/`cudaStreamWaitEvent` in `cuda_vk_scatter.cu`, not the phi event vtable; owner-verified on NVIDIA, not CI-run), `c_abi/dlpack_table.hpp:113-116,144-146` (23 rows: 20 live arena fields + 3 `kNoFieldId`; only 0..19 byte-pinned — AOVs need their own descriptor source), `python/nuka/__init__.py:14` + `autograd.py:70` + `jax_frontend.py:102` (`torch.from_dlpack(world.buffer_view(...))` seam), `python/src/nuka_ext.cpp` (`make_array_from_view` shared by `buffer_view`+`state_view`), `python/nuka/rl_games/vecenv.py` (RL obs wiring point), `c_abi/recorder.cpp` + `python/nuka/recorder.py` (current raster single-env PPM→ffmpeg replay). History: commit `9670a42` (added `src/sensor/camera_render.cu` + `gpu/cuda_sensors.cu`), `c28a2db` (M0 cleanup deleted them with `DeviceWorld`). Perf log: `out/perf/union_throughput_2026-06-15.log`. Prior draft to revise: `subagent-plans/2026-06-20-offline-rhi-render-design.md`.

External: Isaac Lab arXiv:2511.04831 (perception throughput collapse; tiled RTX @64×64; Warp RayCaster geometry-only marginally faster); Isaac Lab tiled-rendering + RecordVideo docs (TiledCamera `(num_cameras,H,W,C)` GPU tensors; annotators rgb/depth/normal/seg/instance/motion_vectors; `--video_interval`; #1996 leak, #875 fragility); Madrona / Madrona-MJX (403k steps/s @64×64 simple, 37k Franka physics-bound; on-device zero-copy); ManiSkill3/SAPIEN (30k+ FPS RGBD+seg, 3.5 GB vs 14.1 GB); O3DE Atom RHI/RPI split; LuisaCompute unified Device/Stream/Buffer/Accel runtime; Laine/Karras *Megakernels Considered Harmful*; Ylitie/Karras/Laine *CWBVH*; Bitterli *ReSTIR* + Lin *ReSTIR PT Enhanced*; Schied *SVGF*; Dammertz *à-trous*; BVH refit-vs-rebuild quality degradation; Jacco *Wavefront Path Tracing*; fisheye/Brown-Conrady ray-gen.