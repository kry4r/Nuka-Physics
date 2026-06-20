# Offline RHI — Implementation Spec (controller-authored)

Implementation spec for subagents. Source of truth for the design + grounded
file:line is `subagent-plans/2026-06-20-offline-rhi-FINAL-design.md` (read it first).
This spec turns that design into dispatchable work items with exact scope, gates,
and constraints. Sensor-model + Python-API sections (P1) are finalized from the
sensor-model research (then folded here).

## Global constraints (every work item)

- **CLAUDE.md:** comments ≤ 2 lines; NO temporal/process wording (no Phase/M*/L1/
  Task/dates/`fix #` in code or comments); **ONE general path** — no per-scene
  hacks, no fused fast-paths, no magic numbers, no silent truncation.
- **Pillars:** D1 bit-determinism (no FP atomics, fixed reduction order); self-
  written / no external SDK (Slang is owner-gated, P4 only); differentiable-sim
  compatibility; multi-backend via PHI.
- **PHI ≠ RHI, ONE shared architecture (owner 2026-06-20):** PHI (compute/physics)
  and RHI (render) are two capability layers that BOTH bind to a SINGLE globally-
  selected architecture (the active backend = kind + device), chosen via one global
  config item — they share one device/context/memory (zero-copy). The render *pipeline*
  (passes, AOV lifetimes, profiles) lives in the new `src/rhi/` and only names PHI
  opaque handles (`phi::Device*/Backend*/Buffer*/BufferType*`) — never a graphics API,
  never CUDA. The RHI surface takes NO per-call `phi::Backend*`; it RESOLVES the global
  active backend (the same one PHI uses). **This supersedes any per-call backend
  threading described in the FINAL design.**
- **Verification is mandatory:** a work item is DONE only when it builds green AND
  its stated gate passes with output shown. Never silently reset a golden or loosen
  a tolerance. If context exceeds ~350k tokens, RETURN for relay — do not claim done.
- **Build env:** `PATH=/root/.nuka-toolchain/bin:/opt/cuda-12.8-root/usr/local/cuda-12.8/bin:/root/miniconda3/envs/nuka-v03/bin:$PATH`;
  `LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:/root/Nuka-Physics/build-cuda128/src`;
  libnuka: `make -C build-cuda128 nuka -j$(nproc)`; viewer/demos:
  `make -C build-viewer/tests <target> -j$(nproc)`; GPU0 free.

---

## P0 — Unify the offline RT onto PHI (correctness foundation)

**Goal:** the offline ray tracer runs on the *selected* PHI device + stream with
device-resident AOVs — no hardcoded device 0, no default stream, no host round-trip.
**Pure plumbing: kernel numerics UNCHANGED → D1 goldens stay byte-identical.**

**Files (exact targets, see FINAL design §2):**
- `src/phi/backend_cuda/rt/two_level_render.cu` — `const int device_id = 0;
  ScopedDeviceGuard guard(device_id)` and `const cudaStream_t stream = nullptr;`
  at the BLAS / RenderFrame / RenderBeauty entries (≈ `:936-938, :966-968,
  :1028-1030`); the 6-channel `CopyToHost` block after each kernel.
- `src/phi/backend_cuda/rt/scene_render.cu`, `bvh_ray_traversal.cu` — same
  device-0 / stream-0 pattern.
- `src/phi/backend_cuda/rt/rt_backend_cuda.cpp` — `CudaRtBackend::Trace` runs the
  host `RenderFrame` then `UploadAov`s back (the **RT-F2** D2H→H2D debt); device
  re-derived via `InitBestDevice()`.
- `src/render/rt_backend.hpp` — `RtBackendI` (may need to carry the selected
  device/stream or a `phi::Backend*`).

**Approach:**
1. **Trace the call chain** first: who constructs `CudaRtBackend`, who calls
   `Trace`/`RenderFrame`/`RenderBeauty`, and what `phi::Backend*`/`Device*` is in
   scope at that call site (the world that produced the scene — `nk::World::Backend()`
   exists, `world.hpp:118`). Plumb the selected backend down to the RT entries.
2. Source device + stream from the backend: `cb->device_id` + `CudaBackendMainStream(b)`
   (`cuda_backend.cu:367`). Replace `ScopedDeviceGuard(0)`→`ScopedDeviceGuard(device_id)`,
   `stream=nullptr`→ the backend main stream. Re-assert `cudaSetDevice(cb->device_id)`
   at the RT entry, mirroring `BackendDispatchImpl` (`cuda_backend.cu:71-74`).
3. Allocate RT buffers from the selected backend's type
   (`BackendDeviceBufferType(backend)`, `backend.hpp:155`), not `InitBestDevice()`.
4. **Pay RT-F2:** have the kernel write the caller's device `phi::Buffer*` AOVs in
   place; remove the adapter's redundant re-upload. Keep a **single** host-download
   path ONLY for the existing video/golden consumers (one D2H, not D2H+H2D). The
   per-pixel write logic is already device-side — do NOT rewrite it.

**Scope guard:** do NOT do the render→realtime reorg (P0b) here. Do NOT change any
kernel math. If device-resident AOVs require broad caller changes, land the
device/stream unification first and RELAY the AOV-residency part.

**Gate:** `build-cuda128` builds green; the `tests/rt/` D1 tests (two-run memcmp +
analytic anchors) pass **byte-identical**; `nuka_go2_terrain_demo` renders a few
frames unchanged. Show the test output.

---

## P0b — Reorg: src/render (Vulkan + ImGui) → src/rhi/realtime + offline skeleton

**Goal (owner):** make the PHI≠RHI split concrete on disk; realtime = the existing
Vulkan rasterizer + ImGui (future role: interactive single-scene world builder).

**Work:**
- Move `src/render/` (Vulkan rasterizer, ImGui, raster, window, shaders) →
  `src/rhi/realtime/`. Keep `RenderWorld` as the shared CUDA-free data product that
  feeds BOTH realtime and offline (place it in a shared `src/rhi/` location or leave
  in `scene` — choose minimal churn; state the choice).
- Create `src/rhi/offline/` skeleton (host, CUDA-free) per FINAL design §3:
  `render_graph`, `render_pass`, `render_profile`, `offline_renderer`,
  `sensor_render`, `aov_export`, `replay` — interface headers + stubs that compile.
- Update CMake targets (e.g. `nuka_render` → `nuka_rhi_realtime`; add `nuka_rhi`)
  and all include paths across the tree.

**Gate:** full tree builds green; the viewer + all demos still build and render
identically. High-churn mechanical — build after each move; commit in logical steps.

**Sequencing:** run AFTER P0 lands (don't mix a risky reorg with the correctness
change). Independent dirs, so no conflict, but serialize for clean bisection.

---

## P1 — Batched GPU sensor render path + Python control (the strict-perf core)

Tracer side in FINAL design §4; **sensor model DECIDED in
`subagent-plans/2026-06-20-sensor-model-design.md`** (read it — it has the struct,
the mount mechanism, the parse mapping tables, and the Python API). Blocked on P0
(device-resident AOVs + offline RHI core) and P0b (the `src/rhi/offline` skeleton).

Dispatch as ordered sub-items (each builds green + its gate):
1. **Unify the sensor IR.** Collapse `sensor::SensorType` + `scene::SensorType` into
   ONE `scene::SensorType`; promote `SensorRecord`/`CameraRecord` into one `SensorDesc`
   (`MountFrame`, `mount_index`, `local_offset`, `CameraIntrinsics`, `LidarPattern`,
   `aov_mask`, `update_period`). Typedef the runtime enum to the scene one. Keep
   serialization stable. Gate: build green; existing sensor/scene tests pass.
2. **Mount field + kernel.** Add `FieldId::SensorWorldPose` to `fields.yaml` + regen; a
   device kernel computes `SensorWorldPose[env,sensor] = FieldPtr(mount)[env,row] ∘
   local_offset` each step, gated by `update_period`, batched over envs, zero host
   copy. Gate: a mounted sensor's world pose matches `link_pose ∘ offset` (unit test).
3. **MJCF/USD parse → `SensorDesc`.** Extend `mjcf_importer.cpp:783-806` (site→mount,
   `<camera fovy>`, rangefinder→RangeScan, etc.) and `usd_importer.cpp:1104-1145`
   (`Camera`→intrinsics+mount; `NukaSensor` `nuka_type∈{lidar,depth}`). Mapping tables
   in the sensor design §2,§3. Gate: a kitchen/robot scene with a camera + rangefinder
   parses into the right `SensorDesc`s.
4. **`RenderSensorBatch` (the tracer path).** Tiled per-env AOVs (`[N,H,W,C]`,
   env-major tile layout) over ONE env-tagged TLAS, refit per step via `RefitLbvh`
   from `FieldPtr` (zero-copy); ONE geometry core shared with replay via `RenderProfile`
   (sensor = shade/GI off). Render kinds = Camera/Depth/Lidar/RangeScan; analytic kinds
   stay on `src/sensor`. **D1:** depth/seg/range exact closest-hit (byte-identical
   two-run). Gate: batched depth/seg for N envs, zero D2H/step, D1 two-run memcmp.
5. **Python API + zero-copy export.** `nuka_world_get_sensor_view(world,sensor_id,aov,
   &view)` returning `nuka_buffer_view_t`; `CameraCfg/LidarCfg/...` dataclasses →
   `SensorDesc`; `sensor.data(aov)` → `torch.from_dlpack` (reuse the `nuka_ext.cpp:539-621`
   caster). Gate: the Python example in the sensor design runs; `rgb/depth/range`
   tensors are GPU, correct shape, zero-copy (alias the device buffer).

## P2 — Unify lidar onto GPU + PHI `DispatchProgram`
FINAL design §4.5: add the generic `DispatchProgram` to `BackendI`; migrate passes
onto it (profiles = pass lists); delete the CPU `StaticBVH` lidar; lidar/height-scan
= device direction-table ray-gen on the shared TLAS; add fisheye/distortion ray-gen,
motion-vector (new prev-transform buffer), semantic-LUT AOVs.

## P3 — Replay FAST profile + recorder (RT-audit Tier-1 folds in)
FINAL design §4.7 + the RT speedup audit: wavefront + CWBVH + ReSTIR DI + temporal
SVGF + FP32 beauty TU + à-trous; cadence recorder + mp4 (training periodic +
post-train demo). GOLDEN byte-exact / FAST flagged off the goldens.

## P4 — Second backend (HIP), then Vulkan/Slang (owner-gated)
FINAL design §6: single-source CUDA/HIP macro header over the neutral `rt/*.cuh`;
re-macro `bvh_traverse_impl.cuh` + `LbvhNode` to `NUKA_RT_HD`; promote cross-stream
ordering onto the `phi` event vtable. Vulkan/SPIR-V via Slang needs owner sign-off.

---

## Dispatch order
P0 (now) → P0b → [sensor research → finalize P1 spec] → P1 → P2 → P3 → P4.
Each phase: controller dispatches a subagent (Opus 4.8 max, 350k-relay), reviews the
returned build/gate result, then proceeds.
