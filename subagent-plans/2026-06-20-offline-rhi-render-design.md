# Offline RHI — GPU-Compute Render Layer (Sensor + Replay), on PHI

A self-written **offline render RHI** that is GPU-compute-only, runs ON the PHI GPU
backends (inherits PHI's multi-backend selection), and is GPU-resident / zero-copy
with the physics step. No OptiX, no RT cores, no graphics-API rasterization. The
realtime Vulkan rasterizer is a SEPARATE abstraction and is left as-is (it may adopt
Slang later — out of scope here).

This revises the prior draft for the owner's **PHI/RHI separation** and the
**dual-purpose** re-scope: ONE offline RHI serves (1) replay video AND (2)
in-the-loop sensor rendering, on ONE general path. The single largest correction vs the
prior draft: **render is NOT folded into PHI's op table.** PHI stays purely the
compute/device substrate; the render *pipeline* (passes, AOV lifetimes, the camera/
sensor fan-out) is a NEW layer (`src/rhi/`) that CONSUMES PHI. The second correction:
the prior draft was video-only; the **batched per-step sensor path is the stricter
requirement** and is added here as a first-class, zero-copy, tiled compute path.

## 0. The two abstractions (owner, binding)

- **PHI** = GPU compute + physics backend layer. Multi-backend vtable
  (`src/phi/backend.hpp:60` `BackendI`, `:77` `DeviceI`, `:89` `RegistryEntryI`;
  `BufferTypeI/BufferI` in `buffer.hpp:31,44`). CUDA today behind `NUKA_PHI2_WITH_CUDA`.
  PHI plays the role O3DE/Unreal call "RHI" (the hardware/device abstraction) — but
  compute-first, not graphics-API.
- **RHI** = render hardware interface, split into **REALTIME** (Vulkan rasterizer,
  untouched, `src/render/` raster path) and **OFFLINE** (this doc, new `src/rhi/`).
- The OFFLINE RHI is **not a second device layer** and **not a peer of PHI** — it is a
  *client* of PHI. It is "a render pipeline expressed as a sequence of compute
  dispatches over PHI's Device / Buffer / Stream substrate." It plays the role O3DE
  calls "RPI" (passes / frame-scheduler), with PHI playing O3DE's "RHI" role. Selecting
  a PHI backend selects it for render. Adding a PHI backend gives render that backend
  for free.

Naming note (to avoid the well-known hazard): in Unreal/O3DE "RHI" = a *graphics-API*
abstraction, which the offline path explicitly forbids. Nuka's "offline RHI" is
deliberately NOT that — it is a compute render pipeline on PHI. Internally prefer the
phrase "render pipeline on the compute substrate" so nobody reaches for a graphics API.

## 1. Module separation, dependency direction, concrete layout

**Dependency arrow (one direction, no cycle):**

```
scene / RenderWorld  (host, CUDA-free, the shared data product)
        │
        ▼
src/rhi/offline      (host, CUDA-free orchestration: passes, profiles, AOV lifetimes,
        │             camera/sensor fan-out — names ONLY phi opaque handles)
        ▼
src/phi              (Device / Buffer / Stream / dispatch — the compute substrate)
        │
        ▼
src/phi/backend_cuda/...  (per-backend render-pass KERNEL BODIES — may name CUDA)
```

`src/rhi/offline` never names a graphics API and never names a CUDA type in its host
layer; it names only `phi::Device*` / `phi::Backend*` / `phi::Buffer*` / `phi::BufferType*`
(all opaque, CUDA-free per `backend.hpp:36-38`, `buffer.hpp:25-26`). This is the same
zero-CUDA red-line `src/render/` already holds (`rt_backend.hpp:49-51` "names ONLY phi
v2 opaque Buffer*/BufferType*").

**Concrete files (NEW — `src/rhi/offline/`, host, CUDA-free):**

| File | Role | Analogue |
|---|---|---|
| `render_graph.hpp/.cpp` | pass DAG + AOV/scratch buffer lifetimes + ping-pong, backend-neutral C++ orchestration | O3DE RPI FrameScheduler; LuisaCompute "Stream records commands" |
| `render_pass.hpp` | `RenderPassId` enum (RayGen, TraverseClosest, ShadePrep, ShadowConnect, ExtendGI, Compact, Accumulate, Denoise, Tonemap, WriteAov) + per-pass POD params | the render analogue of `op_schema.hpp` — a SEPARATE closed enum, NOT folded into `NkOp` |
| `render_profile.hpp` | `RenderProfile` config (camera set, AOV mask, secondary-ray depth, SPP, denoise mode, determinism mode, output target) — selects the pass list as DATA | — |
| `offline_renderer.hpp/.cpp` | the public façade: `BuildScene(RenderWorld)→Handle`, `Render(Handle, profile, RenderTargets, Stream)`; absorbs/replaces today's `render::RtBackendI` | — |
| `sensor_render.hpp/.cpp` | batched multi-camera entry `RenderSensorBatch(scene, CameraSet, AovBatch, Stream)` (tiled `[N,H,W,C]` in one pass) + `RangeSensorBatch` (lidar/height-scan on the same TLAS) | Isaac-Lab TiledCamera + RayCaster as ONE path |
| `aov_export.hpp/.cpp` | DLPack/`__cuda_array_interface__` wrapper of the tiled device AOV → torch, mirroring `c_abi/dlpack_table.hpp` | the RL obs zero-copy seam |
| `replay.hpp/.cpp` | cadence-gated frame capture + mp4 encode feed (training-time periodic + post-train demo), Isaac-Lab `RecordVideo` analogue | — |

**ADD to PHI (the one missing capability — §6):**

| File | Role |
|---|---|
| `src/phi/dispatch.hpp` | backend-neutral generic compute dispatch: `Status DispatchProgram(Backend*, ProgramId, const BindGroup&, GridDims, const void* push, size_t push_bytes)` on the backend's main stream + an opaque `BindGroup` (a list of `Buffer*`). |
| `src/phi/render_program_schema.hpp` (optional) | the render-pass `ProgramId` table, beside `op_schema.hpp` but a DISTINCT program family. Either home is fine; physically separate from `NkOp`. |

**PHI CUDA backend (per-backend kernel BODIES — `src/phi/backend_cuda/rt/`, may name CUDA):**

- KEEP `rt/*.cuh` (the portable bodies: `intersect_primitives`, `ray_box`,
  `instance_transform`, `shading`, `bvh_traverse_impl`, `prim_id`) and the LBVH build
  (`collision/`). Verified zero `cuda_runtime` / zero `<<<>>>` (grep clean).
- REWRITE `two_level_render.cu` + `rt_backend_cuda.cpp` so kernels write the caller's
  `phi::Buffer*` AOVs **in place** on the **selected** backend's main stream (pay
  RT-F2), launched via `launch.cuh` (the sole `<<<>>>` wrapper, `launch.cuh:21-29`)
  under `DispatchProgram`, and register passes in a `g_render_programs` table parallel
  to `g_ops` (`backend_cuda/ops/registry.cuh`). Kill `InitBestDevice()` re-derivation +
  `ScopedDeviceGuard(0)` + the default stream (`two_level_render.cu:936-938,966-968,
  1028-1030,999,1067`).
- CMake: `nuka_phi2_rt` (the RT slice of the CUDA backend, `--fmad=false`) becomes the
  CUDA implementation of the offline RHI. A NEW CUDA-free `nuka_rhi` links like
  `nuka_render` does today (zero-CUDA red-line), with a `CreateCudaOfflineRenderer()`
  strong/weak factory mirroring `CreateCudaRtBackend` / `CreateCudaVkScatter`
  (`rt_backend.hpp:139-146`, `interop_scatter.hpp:184-187`).

**UNCHANGED (owner mandate):** `src/render/` realtime Vulkan rasterizer
(`vulkan_renderer.*`, `raster/*`) stays as-is. `RenderWorld` (`render_world.hpp`) stays
the shared CUDA-free data product feeding BOTH realtime and offline. Slang adoption is
realtime's future, not this work.

## 2. Current state (grounded, file:line)

**PHI is multi-backend-ready, render bypasses it.**
- The vtable + device/stream/buffer seam exist and are backend-neutral
  (`backend.hpp:60-156`, `buffer.hpp:31-80`). The backend OWNS a **main stream** + a
  **capture stream** + its own `device_id` (`cuda_backend.cu:232-234` creates
  `cb->main`/`cb->capture`, `:232` sets `cb->device_id`; `:367` `CudaBackendMainStream`
  exposes `cb->main`); CUDA-graph stream-capture is wired (`:89-113`). **PHI is NOT
  device-0-locked** — only the RT TUs hardcode device 0.
- **The one missing PHI capability:** ops are a CLOSED `NkOp` enum (`op_schema.hpp:63-140`,
  exactly the ~33 physics ops `ApplyDrives`…`NarrowphaseHeightfield` + `Count`, **zero
  render ops**), each a bespoke kernel in a `g_ops` table dispatched via
  `BackendDispatch(b, model, data, OpCall{op, params})` (`backend.hpp:63,109`). There is
  **no generic "dispatch program P over N elements with buffer bindings + a push blob."**
  `LaunchCuda` (`launch.cuh:21`) is a thin CUDA-only `<<<>>>` wrapper, not a
  backend-neutral primitive.
- **The RT path bypasses PHI entirely** (`two_level_render.cu`): raw
  `RenderFrameKernel<<<>>>` (`:988`), `RenderBeautyKernel<<<>>>` (`:1057`),
  `cudaStreamSynchronize(stream=nullptr)` on the **default stream** (`:945,:999,:1067`;
  `stream=nullptr` at `:936,:966,:1028`), `ScopedDeviceGuard(device_id=0)` hardcodes
  **device 0** (`:937,:967,:1029`). AOVs return via **host round-trip** (`:1001-1006`
  D2H into `fb.*`, then `rt_backend_cuda.cpp:78-83` re-uploads via `UploadAov` — the
  named **RT-F2** debt, documented `rt_backend.hpp:20-29,108-116`). PHI is used only for
  buffer alloc, itself re-derived from `InitBestDevice()` (`rt_backend_cuda.cpp:111`)
  rather than the selected `Device*`. None of this is a missing capability — it is debt:
  the device pointer + stream the trace should use already exist.

**The shared BVH + refit already exist (the key leverage).**
- The LBVH is built by the same `collision::gpu` code for broadphase AND the tracer
  (`two_level_render.cu` uses `BuildLbvhForQuery`; `broadphase_lbvh.hpp:137`). PHI even
  has the broadphase as real ops (`op_schema.hpp:82-86` `BuildAabbs`/`LbvhBuild`/
  `LbvhQueryPairs`).
- **Refit-not-rebuild is already implemented** for the physics broadphase:
  `collision/lbvh_refit.cuh:24` `RefitLbvh(stream, device_id, nodes, new_aabbs,
  leaf_count, visit_scratch)` re-propagates fresh leaf AABBs up an EXISTING tree
  (topology reused). This is exactly the per-step primitive the sensor path needs.
- Two-level split is already correct for dynamics: **BLAS is built ONCE per unique mesh
  and never refit** (rigid; `BuildBlas` in `BuildTwoLevelScene`, `two_level_render.cu:
  928-947`), **TLAS is rebuilt per frame** over current instance world-AABBs
  (`BuildFrameTlas`, called `:973,:1032`). Refit applies to the TLAS-scale (≤4096
  instances, `prim_id.cuh:42` `kMaxInstances=4096`) — cheap — while BLAS topology is
  permanent.

**The zero-copy physics→render seam already exists.**
- `nk::World::FieldPtr(FieldId)` returns a **device pointer** to FK pose fields
  (`world.hpp:102`); `nk::World::Backend()` returns the opaque `phi::Backend*` the
  World's ops ran on (`world.hpp:118`). The interop layer already reads these device
  pointers with NO D2H (`interop_scatter.hpp:119-135` `ScatterFkSource`: "the SAME
  device pointers `Data::Ptr(FieldId)` returns … No D2H copy"), AND already carries the
  **cross-stream event-ordering** pattern (`interop_scatter.hpp:127-135` INT-F2: record
  an event on the World's main stream, the render/scatter stream waits before launching,
  via `BackendEventRecord`/`BackendEventWait`, `backend.hpp:118-121`). The render reads
  instance transforms from this same device buffer — no host round-trip needed.

**Sensors today are CPU-only and NOT GPU-resident (the gap to close).**
- Lidar: `sensor/ray_sensor.cpp:11-46` `QueryLidarSensor` is a **host CPU loop**
  (`for i<ray_count`) calling `collision::StaticBVH::Raycast` (a CPU binary BVH,
  `static_bvh.hpp:44-75`: `std::vector` nodes, explicit-stack recursive Raycast,
  `std::optional` hit), one ray at a time, returning a host `std::vector<float>`
  (`sensor_packet.hpp:23` `depth_buffer`). Not batched, not on GPU, not using the GPU
  LBVH.
- IMU / JointState: host functions over a host `BodyState` (`state_sensor.hpp`) —
  analytic, not a render (correct; not a render sensor).
- Contact wrench: IS a PHI op (`op_schema.hpp:102` `ReadoutContactWrench`) — GPU-resident
  already, but a solver readback, not a render sensor.
- **There is NO camera/depth/segmentation/normal/instance render sensor anywhere.** The
  `SensorType` enum is exactly `{IMU, JointState, ContactSummary, Lidar, Depth}`
  (`sensor_graph.hpp:16-31`) — no RGB/seg/normal IMAGE sensor. The only ray-vs-scene
  render is the offline RGB tracer, host-output and device-0/stream-0 bound. So
  in-the-loop sensor rendering does not exist yet; the tracer's geometry core is the
  thing to generalize.

**What is already portable (the reuse the design rests on).**
- Math (`src/math/*` under `NUKA_MATH_HD`) and the intersection/shading layer
  (`rt/*.cuh`: `intersect_primitives`, `ray_box`, `instance_transform`, `shading`,
  `prim_id`) have **no `cuda_runtime`, no `<<<>>>`** — backend-neutral (grep clean).
- The traversal is **`template <typename LeafFn> __device__ TraverseRay(...)`**
  (`bvh_traverse_impl.cuh:45-46`): "literally ONE traversal, two leaves" (`:8-11`). The
  closest-hit nest (`two_level_render.cu:196` `BlasLeaf`, `:222` `TlasLeaf`, `:254`
  `ClosestHit`) updates the SAME `best_t`/`best_prim` by reference and already emits
  depth (`best_t`, `:354`), normal, albedo, uv, and the packed instance/local prim id
  (`prim_id.cuh:47` `PackPrimId`, instance HIGH bits → instance/semantic seg). The
  caveat: `bvh_traverse_impl.cuh` is `__device__`-gated (NOT `NUKA_RT_HD`) — porting
  traversal to a 2nd backend needs that one header re-macro'd to `NUKA_RT_HD`.
- The AOV set (`framebuffer.hpp:36-42`: color/depth/normal/albedo/uv/prim) is EXACTLY
  the sensor channels, already documented as the DLPack export layout
  (`framebuffer.hpp:14-19`).
- The DLPack zero-copy obs contract already ships for physics state:
  `torch.from_dlpack(world.buffer_view(Field.X))` (`python/nuka/__init__.py:14`,
  `autograd.py:70`, `jax_frontend.py:102`), backed by `c_abi/dlpack_table.hpp` (23
  byte-pinned fields). AOV export extends this exact seam (§3.3) — but note: today's
  table covers only physics arena fields; render AOVs are a NEW family of views.

So only three things are CUDA-locked on the RT path: the two `__global__` launches, the
LBVH build, and stream/device binding. The first and third are exactly what moving onto
PHI fixes; the second is shared with physics.

## 3. The sensor-render path (the strict-perf driver)

Sensor rendering runs **every render-tick, for N envs**, co-resident with physics, with
the AOV staying on device as the RL obs. This is the perf bar and the thing that does
not exist today.

### 3.1 One general path: shared geometry core + per-purpose config

The ONE-path pillar forbids a sensor codepath separate from a replay codepath (exactly
the split Isaac Lab has: Warp RayCaster vs RTX tiled). Resolve it as **ONE geometry/
traversal core + a per-purpose pass list selected by a `RenderProfile`**, not two
renderers.

**Shared core (identical bytes for both purposes):**
1. The acceleration structure: the shared LBVH (BLAS built once, TLAS refit/rebuilt per
   step from physics device buffers, §4).
2. Ray generation: a `RayGen` that emits SoA primary rays for a *camera set* — one
   pinhole per env-tile for sensors, one cinematic camera for replay, a direction-table
   per env for lidar. Same kernel, different camera/pattern buffer length.
3. Closest-hit traversal: the existing `ClosestHit` nest (`two_level_render.cu:254`),
   reused verbatim via the templated `TraverseRay` leaf — it already produces depth
   (`best_t`), normal (`ReconstructHit`), albedo, uv, and instance/semantic id
   (`UnpackPrimId`). The ONLY thing that varies between a sensor and a camera is the
   leaf payload: **distance-only** (lidar/depth) vs **full G-buffer write** (rgb sensor)
   vs **shade** (replay). This is precisely the `bvh_traverse_impl.cuh:8` "ONE
   traversal, two leaves" design extended to three leaves.

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

The pass list is **data**: a sensor profile runs `{RayGen, TraverseClosest, WriteAov}`;
a replay profile runs `{RayGen, TraverseClosest, ShadePrep, (ReSTIR), ExtendGI, Compact,
Accumulate, Denoise, Tonemap}`. Both lists call the SAME programs over the SAME buffers;
replay just enables more passes. Depth/seg/normal are exact in BOTH (a sensor is a
replay with the shading/GI passes disabled). No second tracer, no per-scene branch, no
magic numbers — the profile is the only knob (CLAUDE.md ONE-path). This kills the prior
draft's biggest risk: a "beauty kernel" and a "sensor kernel" diverging.

### 3.2 Batched tiled layout + the occupancy problem

N envs × a small tile (64×64) is many tiny, independently-divergent ray sets. A
one-block-per-tile megakernel under-occupies and diverges. Do **wavefront over ALL rays
of ALL tiles at once**, persistent-thread style (the convergence Madrona/Habitat/Isaac
all use):

1. **One global SoA ray pool** across the whole batch: `RayGen` writes (tile_h·tile_w·N)
   primary rays into one flat `(origin, dir, pixel_id, env_id)` buffer. The grid is
   sized to the GPU, not to one tile — full occupancy from ray 0.
2. **TraverseClosest** is one dispatch over the whole pool against the shared TLAS;
   results to a flat hit buffer. Tile identity is just `env_id` carried in the ray record
   → `WriteAov` scatters to `aov[env_id][pixel_id]`. No per-tile launch. A ray accepts
   only hits with its own `env_id` (one integer compare in the leaf — the Madrona
   "world-id" trick — so ONE env-tagged TLAS replaces N trees, the single biggest
   throughput lever; see §4).
3. **Stream-compact** live rays between bounces (REPLAY only; sensors are 1 closest-hit,
   no compaction). Compaction is order-preserving → deterministic.
4. **The tiled AOV layout** mirrors Isaac Lab's stitched buffer: one big device buffer
   per channel `[N_tiles, tile_h, tile_w, channels]` (env-major, a deterministic tile =
   `f(env_index)`) so the learner reads it as one tensor with no gather; per-env slicing
   is a view.

Megakernel-vs-wavefront split by purpose: the SENSOR pass MAY stay a megakernel (shade
is shallow + coherent; one closest-hit) — but it must still launch over the whole batch
(one grid over N·tile pixels), not per tile. The REPLAY beauty pass uses the wavefront
split because deep multi-material GI is divergent. Both call the same `TraverseRay` /
`rt/*.cuh` bodies; only host orchestration (one dispatch vs many) and shade depth differ.

### 3.3 Zero-copy AOV → torch (the hard contract)

For sensors the AOV tensor MUST be a device buffer the RL obs reads in place, NEVER the
host round-trip in `two_level_render.cu:1001` / `rt_backend_cuda.cpp:80`:

- AOV buffers are allocated from the **active backend's** `BufferType` via
  `BufferAlloc(BackendDeviceBufferType(backend), bytes)` (`backend.hpp:155`,
  `buffer.hpp:68`) so transfers + render kernels share the physics main stream.
- Export via DLPack capsule, extending the `c_abi/dlpack_table.hpp` pattern: a new
  `sensor_view(SensorId, AovChannel)` C-ABI entry returning a `nuka_buffer_view_t`-class
  descriptor (device ptr + shape `[N,H,W,C]` + stride + wire dtype: f32 for
  depth/normal/rgb-float, u32 for instance/semantic id, uint8 for rgb8), consumed as
  `torch.from_dlpack(world.sensor_view(...))` exactly like `world.buffer_view(...)`
  today (`python/nuka/__init__.py:14`). The host-download path (`TraceToHost`) stays
  ONLY for the video encoder + D1 golden tests.
- Co-residency: the render reads instance transforms from `World::FieldPtr` device
  buffers (no `DownloadField`), ordered behind the physics step via the existing
  `BackendEventRecord`/`Wait` pattern (`ScatterFkSource::world_backend`,
  `interop_scatter.hpp:127-135`).

### 3.4 Camera & range-sensor models (all are ray-gen variants, ONE path)

- **Pinhole** exists (`camera.hpp:45-113`, fp64-internal `GenerateRay`, `NUKA_RT_HD`).
- **Fisheye / equidistant** (`r = f·θ`) and **Brown-Conrady distortion** are different
  pixel→ray maps; add them as `GenerateRay` variants in the `NUKA_RT_HD` camera header.
  Traverse/shade are identical. Keep fp64-internal for D1.
- **Lidar / height-scan / range sensors** = a non-pinhole ray-gen *pattern* on the SAME
  env-tagged TLAS: spinning = (azimuth × elevation) grid, solid-state = a device
  direction table, height-scan = a ground grid. Per ray return `range = best_t` and
  optionally `intensity = cos(incidence)/r² · albedo` (the data is at the hit already).
  Batched over envs exactly like cameras. **This deletes the CPU `StaticBVH` lidar path**
  (`ray_sensor.cpp`) and subsumes it into the GPU path — ONE BVH, two leaf payloads
  (distance-only vs shade), the CLAUDE.md ONE-path fix. The project's existing
  legged_gym terrain height-scan obs becomes one such pattern on the unified path.

### 3.5 The AOV set (what one closest-hit produces, no rasterizer)

From one closest-hit (all already in `framebuffer.hpp:36-42` + `prim_id.cuh`):
`depth/range = best_t`; `distance_to_image_plane = best_t·dot(dir, forward)`;
`normal = ReconstructHit`; `instance_segmentation = HIGH bits of PackPrimId`;
`semantic_segmentation = per-instance class LUT[instance]` (a cheap relabel — near-free);
`albedo`/`uv` from the hit. The one genuinely new AOV is **motion vectors**: store each
instance's PREV-frame transform (one extra device array), transform the current hit back
to the prev pose, project through the prev camera, take the screen-space delta. All
atomic-free, one-writer-per-pixel → D1 (§5).

### 3.6 Strict-perf budget (static extrapolation — NOT measured)

Grounded by Isaac Lab (arXiv:2511.04831, RTX Pro 6000, single GPU): state-only ≈10⁶ FPS
multi-GPU collapses to ~100k–300k FPS the moment cameras turn on ("rendering dominates
computational demand in perception tasks"); tiled RTX @64×64 ≈ 250k–300k env-FPS over
4096 envs; their **GPU-compute Warp RayCaster** (depth/normals only, "omits rendering
effects") is *marginally faster* than RTX tiled on a single GPU — direct validation of
the GPU-compute-only sensor approach. Cross-checks: Madrona CUDA batch tracer 403k
steps/s @64×64 simple (37k Franka, physics-bound); ManiSkill3/SAPIEN 30k+ FPS RGBD+seg
@4090 in 3.5 GB (vs Isaac's 14.1 GB — the case for a lean compute renderer).

- Nuka physics today: ~16,367 env-steps/s @N=1024 ⇒ **~61 µs/step** (memory:
  `nuka_union_throughput`).
- A 64×64 sensor tile = **4096 rays/env**; N=1024 ⇒ **4.19M primary rays/step**. At
  Isaac Lab's ~1 Gprimary-ray/s order on closest-hit-only geometry that is **~2–5
  ms/step**, i.e. the **vision step rate is render-bound at a few hundred env-FPS** — the
  honest reading matches Isaac Lab: with vision on, the step rate drops to the render
  rate. (Do NOT quote sub-ms as achieved before measurement.)
- **Why device-resident is the whole point, not an optimization:** a single per-step D2H
  of 4.19M×4 B ≈ 17 MB at PCIe ~12 GB/s is ~1.4 ms of pure copy that **serializes** the
  step — on TOP of the trace, and it scales with N. Today's RT-F2 host round-trip is
  therefore categorically disqualified for in-loop sensors.
- **Decimation** (Isaac `update_period`): sensors render every K physics steps, not
  every substep — multiplies effective throughput; a separate `video_interval` drives
  replay. Add a render-tick decimator so the strict sensor path co-exists with high
  physics throughput.

**Target:** match Isaac Lab's order — ~1 Gprimary-ray/s of closest-hit geometry on the
shared refit-BVH, depth/seg/normal exact, no GI/denoise for sensors, zero host copy, on
ONE GPU. This is stricter than replay because it is in the inner loop (every Kth step,
not once per clip), cannot hide the host copy, is batched over many tiny tiles
(occupancy), and shares the BVH with physics.

## 4. Sharing ONE scene/BVH between physics and render (refit per step)

The collision broadphase and the tracer already build the same `LbvhNode` type; make
them the SAME structure per step:

1. **BLAS** (per unique mesh, local prims): built once at scene cook (`BuildTwoLevelScene`,
   `two_level_render.cu:928`), retained, **never refit** — rigid meshes are static in
   their own frame. No per-step cost.
2. **TLAS** (instances): instead of allocating fresh device buffers + a fresh tree every
   frame (`BuildFrameTlas` today, `:973`), **refit the retained TLAS** with `RefitLbvh`
   (`lbvh_refit.cuh:24`) over current instance world-AABBs computed on-device from the
   physics FK field buffer (`FieldPtr`), zero-copy. Rebuild the TLAS topology only on a
   cheap trigger (instance-count change, or a periodic SAH-quality refresh) — the
   standard refit/rebuild discipline: refit degrades under large rearrangement (looser
   boxes → slower traversal), so a periodic rebuild bounds quality. For ≤4096 rigid
   instances on a heightfield this is firmly refit territory; the rebuild interval is a
   quality/throughput dial, not a per-scene hack.
3. **Env-tagged single TLAS** for batched sensors: one TLAS over ALL envs' instances,
   each leaf tagged with `env_id`; the ray's leaf functor rejects cross-env hits (one
   compare). Avoids N trees + N dispatches — the Madrona world-id trick. (Single-env
   replay just uses all instances.)
4. **Instance transforms come from the physics state device buffer** — the DevInstance
   `transform` is filled from `FieldPtr(LinkPose/BodyPose/BasePose)` on-device, the same
   data the interop scatter reads bit-exactly (`interop_scatter.hpp:112-113`). The render
   never downloads pose.

Net: per step the render adds a TLAS refit (two kernels already written in
`lbvh_refit.cu`, over ≤4096 boxes) + the traversal, sharing the structure the physics
step just used. "Refit-not-rebuild, transforms from physics state, zero-copy" is
satisfied with code that already exists.

## 5. Determinism (D1), differentiability, no-SDK

- **Sensor profile is exact by construction:** 1 closest-hit/ray, `depth = best_t`,
  seg/instance from the total-order `PackPrimId` tie-break (`prim_id.cuh:9-13`: instance
  HIGH bits ⇒ `t < best || (t==best && packed < best_packed)` is a TOTAL ORDER ⇒
  order-independent winner), one writer per pixel, no FP atomics → bit-identical
  run-to-run within a backend/device (the `framebuffer.hpp:18-19` two-run memcmp-identical
  property). NO temporal reuse, ever. This is the regression / golden / diff-sim path,
  and it is the DEFAULT for sensors (the prior draft defaulted to the video framing; the
  re-scope flips the default to GOLDEN for the strict path).
- **Replay split into two modes, ONE geometry pipeline:**
  GOLDEN = fixed-order single-frame à-trous, no ReSTIR/TAA, counter-based RNG keyed by
  (env, pixel, sample, bounce), regeneratable — the byte-exact `RenderFrameKernel`-class
  path. FAST = wavefront + ReSTIR + temporal SVGF + TAA, carries `temporal_reuse=on`
  that **HARD-DISABLES golden comparison**. Never silently loosen a tolerance or regen a
  golden (CLAUDE.md). Cross-vendor: bit-exact within a backend, equivalent-within-
  tolerance across vendors (FMA/transcendental rounding differ — bit-identical
  cross-vendor is not achievable; state it plainly).
- **Differentiability:** the closest-hit geometry path (depth/normal/segmentation as
  piecewise-smooth functions of pose; silhouette visibility is the known non-diff locus,
  same as every diff-renderer) is differentiable in GOLDEN mode and lives on the same
  arena/DLPack device buffers, so a backward pass can hang off the same device fields if
  a grad-through-pixels consumer ever appears. ReSTIR/denoise/temporal (FAST) are NOT
  differentiable and live only in FAST replay. For the RL-obs use, gradients flow through
  the policy, not the pixels — so wiring sensor AOVs into the diffsim tape is deferred to
  a real consumer (Open Decision #3).
- **No-SDK:** CUDA + a single-source CUDA/HIP macro header need no external compiler (the
  `rt/*.cuh` bodies are already `cuda_runtime`-free; the one exception is
  `bvh_traverse_impl.cuh`, `__device__`-gated, which needs a `NUKA_RT_HD` re-macro to
  compile under hipcc). **Vulkan/SPIR-V breadth would need Slang (an external compiler)
  → `needs_owner_signoff`.** CUDA+HIP cover the perf driver (NVIDIA + AMD) with zero new
  dependency; the hand-written-SPIR-V alternative is a multi-year compiler project and
  would force abandoning the header-only math core — not recommended.

## 6. The missing PHI primitive (the one thing to add)

Add to `BackendI` a **backend-neutral compute-dispatch** (a new vtable slot, co-existing
with the existing `dispatch(OpCall)` — no churn to the 33 physics ops): "launch render
program P over an N/2D/3D grid with a list of `BufferI` bindings + a small POD push blob,
on the backend's main stream." Plus a per-backend **render-program registry** mirroring
`g_ops` (`backend_cuda/ops/registry.cuh`) but for render passes (RayGen, TraverseClosest,
Shade, Compact, Denoise…). The host orchestration (profile → pass list → buffer
lifetimes → ping-pong) is backend-neutral C++ in `src/rhi/`; only the program *bodies*
are per-backend, and the math/`rt/*.cuh` bodies are reused as-is.

This is the device CAPABILITY shared by both purposes — it belongs in PHI. The render
PASS REGISTRY and the pass DAG belong in RHI, not PHI (the §1 separation). Physics ops
MAY migrate onto `DispatchProgram` opportunistically; it is not required. Until it lands,
P0 can route the existing two kernels through `LaunchCuda` on `CudaBackendMainStream`
with device-resident AOVs (kills RT-F2 + device-0/stream-0) without the generic primitive
— a byte-exact foundation step.

## 7. Replay video (the second, looser purpose)

Replay is the SAME core with the shade/GI/denoise passes enabled (§3.1 table). Reference
Isaac Lab's recorder shape (which is just a `gymnasium.RecordVideo` wrapper, not a custom
renderer): a cadence-gated wrapper that, every `video_interval` steps or on a
post-training demo run, points a hero camera at the selected env, renders the FAST
profile, async-D2H's the RGBA8 on a copy stream (the host copy is fine here — once per
clip, not in the inner obs loop), and muxes to mp4. Because replay reuses the in-loop
ray pool + shared BVH, "render a training replay" and "produce a sensor image" are the
same machinery at different profiles — no separate offline renderer (ONE-path).

Today's replay (`c_abi/recorder.cpp`, `python/nuka/recorder.py`) uses the offscreen
**Vulkan rasterizer**, single env, PPM→ffmpeg — that is the realtime path and is
orthogonal; the offline-RHI replay is the path-traced cinematic alternative, sharing the
sensor core. Quality target stays the prior draft's wavefront + CWBVH + ReSTIR +
temporal SVGF at ~30 fps@1080p; honest landing is ~50–80× from FP32 + wide-BVH + low-SPP
+ SVGF, full 30 fps only with the whole stack — state it plainly, do not oversell 33 ms
pre-measurement. Known reuse caveats from Isaac: `RecordVideo` leaks memory over long
runs and is version-fragile — treat the cadence wrapper as thin, own the encoder feed.

## 8. Multi-backend (inherits PHI selection automatically)

Because the offline RHI holds a `phi::Backend*`/`Device*` (the SAME one physics ran on)
and dispatches via `DispatchProgram`, selecting a PHI backend selects render for free.

- **CUDA** (unify first): the existing tracer, moved onto the selected device/stream with
  device-resident AOVs.
- **HIP/ROCm** (second): a single-source `__global__`/`__device__` macro header over the
  already-neutral `rt/*.cuh` bodies compiles under nvcc AND hipcc with near-zero rewrite
  (the one `__device__`-gated header, `bvh_traverse_impl.cuh`, gets re-macro'd to
  `NUKA_RT_HD`). Only `DispatchProgram` + the LBVH build/refit are per-backend.
- **Vulkan-compute / SPIR-V** (breadth): needs **Slang** (external compiler) →
  `needs_owner_signoff`. Defer to this phase only.

Both physics AND render gain each new backend.

## 9. Implementation phases (each independently shippable, profiling-gated)

- **P0 — Unify RT onto PHI, device-resident AOVs (no perf change, D1 byte-exact).**
  Route the trace through `LaunchCuda` on `CudaBackendMainStream`; take the selected
  `Device*`/stream (kill `ScopedDeviceGuard(0)` + default stream, `two_level_render.cu:
  937,967,1029,999,1067`); make AOVs device-resident, kill the RT-F2 host round-trip
  (`rt_backend_cuda.cpp:78-83`). Stand up `src/rhi/offline/` host skeleton + the
  `CreateCudaOfflineRenderer` factory. **Gate:** D1 goldens byte-identical
  (`tests/rt/test_render_world_rt.cpp` TraceToHost×2 memcmp), no new host copy in the
  device path.
- **P1 — Sensor profile = the strict path.** `RenderSensorBatch`: RayGen for an N-tile
  camera set; wavefront intersect over one global ray pool against the env-tagged TLAS;
  TLAS **refit** per step via `RefitLbvh` with instance AABBs from `FieldPtr`
  (zero-copy); scatter depth/seg/normal/instance into one tiled device AOV; export as a
  DLPack obs tensor (`sensor_view` C-ABI). Wire into the RL loop next to the obs export
  (`vecenv.py`). **Gate:** rays/s, occupancy, per-pass ms, step-inflation factor vs the
  §3.6 ~2–5 ms/step target at N=1024×64×64; zero D2H per step.
- **P2 — Unify lidar + add the dispatch primitive.** Add `DispatchProgram` +
  `g_render_programs` to PHI; migrate P0/P1 passes onto it (profiles become pass lists).
  Delete the CPU `StaticBVH` lidar path; lidar = device direction-table ray-gen on the
  same TLAS (`RangeSensorBatch`); add fisheye/distortion ray-gen + motion-vector +
  semantic-LUT AOVs. **Gate:** lidar parity vs the old CPU path within tolerance;
  height-scan obs on the unified path matches the legged_gym obs.
- **P3 — Replay FAST profile + recorder.** Wavefront beauty passes (ShadePrep, ReSTIR
  DI, ExtendGI, Compact, Accumulate, SVGF/à-trous denoise, Tonemap) + the cadence wrapper
  + mp4 encode (training-time periodic + post-train demo). **Gate:** GOLDEN replay mode
  byte-exact; FAST mode flagged off the goldens; cinematic fps measured (not asserted).
- **P4 — Second backend (HIP) via single-source header**, then Vulkan/SPIR-V
  (owner-gated on Slang). **Gate:** cross-vendor within-tolerance AOV equivalence;
  within-backend bit-exact.

## 10. Open decisions for the owner

1. **Slang** for Vulkan/SPIR-V (and Metal) codegen breadth — external compiler, flagged
   `needs_owner_signoff`. CUDA+HIP need none and cover the perf driver (NVIDIA + AMD).
2. **Sensor default resolution(s) + which AOVs are in the hot loop** — 64×64 depth+seg
   is the Isaac-Lab-matched default and nearly free; rgb adds the shadow ray (~2×);
   higher res trades the §3.6 budget directly.
3. **Wire sensor AOVs into the diffsim tape now, or keep the differentiable surface =
   GOLDEN sensor pass only?** Only worth it if a grad-through-pixels consumer exists; the
   RL-obs use does not need it.
4. **TLAS rebuild interval** (refit-quality dial) — every step vs periodic; a
   throughput/traversal-quality tradeoff, default a conservative periodic rebuild.
5. **Quality bar for replay** — 30 fps@1080p target vs higher-res/lower-fps offline.

## References

Code (this repo, verified file:line): `src/phi/backend.hpp:60-156`,
`buffer.hpp:31-80`, `op_schema.hpp:63-140` (closed NkOp, zero render ops),
`backend_cuda/cuda_backend.cu:232-234,367` (main/capture streams + device_id, not
device-0-locked), `backend_cuda/launch.cuh:21-29` (sole `<<<>>>`), `rt/two_level_render.cu`
(kernels `:988,:1057`; TLAS rebuild `:973,:1032`; device-0 `:937,:967,:1029`; stream-0
`:936,:966,:1028,:999,:1067`; host AOV `:1001-1006`; `BlasLeaf/TlasLeaf/ClosestHit`
`:196,:222,:254`), `rt/rt_backend_cuda.cpp:78-83,111` (RT-F2 + InitBestDevice),
`rt/bvh_traverse_impl.cuh:45-46` (templated `__device__ TraverseRay`, ONE traversal/two
leaves), `rt/prim_id.cuh:9-13,47` (instance-HIGH total-order tie-break = D1 + free seg),
`rt/camera.hpp:45-113` (PinholeCamera fp64 NUKA_RT_HD), `rt/framebuffer.hpp:14-19,36-42`
(AOV set = sensor channels, DLPack layout), `render/rt_backend.hpp:20-29,79-86,108-146`
(RtBackendI device-buffer contract + RT-F2 named debt + factory), `collision/lbvh_refit.cuh:24`
(refit primitive), `collision/static_bvh.hpp:44-75` (CPU lidar tree — the gap),
`sensor/ray_sensor.cpp:11-46` (CPU lidar loop), `sensor/sensor_graph.hpp:16-31` (SensorType,
no image sensor), `nk/pipeline/world.hpp:102,118` (device FieldPtr + Backend()),
`phi/interop_scatter.hpp:112-135` (zero-copy pose seam + cross-stream event ordering),
`c_abi/dlpack_table.hpp` (DLPack contract to extend), `python/nuka/__init__.py:14` +
`autograd.py` + `jax_frontend.py` (`torch.from_dlpack(world.buffer_view(...))` seam).
External: Isaac Lab arXiv:2511.04831 (perception throughput collapse 10⁶→100–300k FPS;
tiled RTX @64×64 ~250–300k env-FPS/4096 envs; Warp RayCaster marginally faster,
geometry-only); Isaac Lab tiled-rendering + RecordVideo docs (TiledCamera
`(num_cameras,H,W,C)` GPU tensors, `--video/--video_length/--video_interval/
--enable_cameras`, annotators rgb/depth/normals/seg/motion_vectors; #1996 leak, #875
fragility); Madrona/MJX (403k steps/s @64×64; on-device zero-copy); ManiSkill3/SAPIEN
(30k+ FPS RGBD+seg, 3.5 vs 14.1 GB); O3DE Atom RHI/RPI two-layer split; LuisaCompute
unified Device+Stream+Buffer/Image/Mesh/Accel (AS build in the runtime); Laine/Karras
*Megakernels Considered Harmful*; Ylitie/Karras/Laine *CWBVH*; Bitterli et al. *ReSTIR*
+ Lin et al. *ReSTIR PT Enhanced*; Schied *SVGF*; Dammertz *à-trous*; refit/rebuild
quality degradation (Springer); Jacco *Wavefront Path Tracing* (SoA, compaction,
occupancy); fisheye/equidistant + Brown-Conrady distortion ray-gen.
