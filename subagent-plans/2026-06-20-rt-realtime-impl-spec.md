# RT near-real-time impl spec (controller-authored 2026-06-20)

Owner re-target: **near-real-time**. The dual-purpose SENSOR path (in-the-loop batched
RGB per env per step) dictates the bar; "14s video in 9 min" is rejected. This supersedes
the ROI order in `2026-06-20-rt-render-speedup-plan.md` (which stays the lever catalog).

## Measured baseline (the two enemies)

`build-viewer/tests/nuka_go2_terrain_demo --probe --gpu --placeholder --robots 10` (340 inst, sm_86):

| res | spp | render | note |
|---|---|---|---|
| 1920×1080 | 16 | 6.8 s | the README path |
| 1920×1080 | 1 | 783 ms | trace ~713ms + fixed ~70ms |
| 960×540 | 1 | 246 ms | |
| 256×256 | 1 | 80 ms | |
| 128×128 | 1 | 67 ms | |
| 84×84 | 1 | 77 ms | **FLAT** at low res |

- **Enemy 1 — fixed ~67ms/frame floor** (flat across 84²→256²): per-frame TLAS REBUILD
  (`BuildFrameTlas`: re-upload 340 inst + thrust stable_sort + tree build) + 6 `OwnedBuffer`
  malloc/free EVERY frame (`two_level_render.cu` `RenderBeauty` L1101-1126). 1% at 1080p/16spp,
  ~100% of a sensor-res frame → the SENSOR enemy.
- **Enemy 2 — FP64 trace ~340 ns/px** (~59 ns/ray, ~12M rays/1080p-frame): `--fmad=false` + double
  internal math on sm_86 (FP64 = 1/64 FP32 rate) → the VIDEO enemy.

## Hard constraints (NON-NEGOTIABLE — read CLAUDE.md)

1. **Golden FP64 path byte-exact.** `RenderFrameKernel` (two_level_render.cu:280) + the host
   oracle MUST stay byte-identical. These gate it (ALL must stay GREEN):
   `nuka_rt_two_level_test` (esp. `RtTwoLevelRender.Oracle1HostDeviceByteExact`,
   `Oracle2IdentityByteExactVsFlat`, `Oracle2RotatedNearExactVsFlat`, `AnalyticAnchorP13TriangleIdentity`,
   `AnalyticAnchorRotatedTranslatedTriangleAndPrimId`, `D1TwoRunByteExactAllAovs`,
   `InstanceTransformTracking`, `SdfLeafThroughNestIdentityAndRotated`),
   `nuka_rt_render_world_test` (`RenderWorldRt.*`), `nuka_rt_scene_render_test` (`RtSceneRender.*`),
   `nuka_rt_traversal_test` (`RtBvhTraversal.*`), and `nuka_rhi_offline_replay_test`
   (`OfflineReplay.OfflineRendererBeautyMatchesDirectTracerByteExact`).
2. **NEVER loosen a tolerance or regenerate a golden to make it pass.** If a golden moves, STOP
   and report — that means the change leaked into the FP64 path.
3. **ONE general path.** The FP32/FP64 choice is a precision template parameter + a TU-level
   compile flag, NOT scattered `#ifdef`s or a per-scene branch. No magic numbers.
4. **Comments ≤2 lines. No temporal/process words** (no "Phase", "Tier", "T1.3", dates) in code.
5. Beauty pixels MAY change vs the old FP64 beauty (it is stochastic + non-golden). There is NO
   stored beauty golden — confirm by grep before relying on it.

---

## Dispatch 1 — FP32 beauty TU + secondary-ray cut + download-mask (DETAILED)

Kills Enemy 2 (dominant video lever) + cheap co-located wins. Numerically contained: golden
untouched, beauty goes FP32.

### 1a. FP32 beauty TU split
The precision lives in the `#include`d math headers (`phi/backend_cuda/rt/{ray_box.cuh,
intersect_primitives.cuh,shading.cuh}`) which use `double` temporaries for host==device
byte-exactness — and the HOST ORACLE includes them too. The shared device fns
(`ClosestHit` L214, `ReconstructHit` L237, `TlasLeaf`, `BlasLeaf`, `IntersectBlasPrimT`) +
structs (`DevPrim/DevBlas/DevInstance/BeautyParams`) are file-local in `two_level_render.cu`;
both `RenderFrameKernel` (golden) and `RenderBeautyKernel`+`ShadeBeauty` (beauty) call them.

Mechanism (pick the cleanest; this is the load-bearing risk):
- Templatize the internal precision of the math header fns on `<typename Real = double>`. Default
  `double` ⇒ every existing caller (golden kernel + host oracle) is byte-unchanged. The beauty path
  instantiates `Real=float`. Verify the host-oracle TU and `RenderFrameKernel` NEVER pick `float`.
- Move the shared device fns + structs into a new header `two_level_render_kernels.cuh`
  (templatized on `Real` where precision flows through), included by BOTH TUs.
- Move `RenderBeautyKernel` + `ShadeBeauty` + beauty helpers (`PcgHash/BeautyRng/OrthoBasis/
  CosineHemisphere/SampleCone/SkyColor`) into a NEW TU `phi/backend_cuda/rt/two_level_render_beauty.cu`,
  instantiating the `float` path. `LaunchRenderBeauty` calls into it.
- CMake (`src/CMakeLists.txt` ~1057-1070): the new beauty TU compiles with `--fmad=true` (NOT
  folded into `two_level_render.cu`, which keeps `--fmad=false`). Add it to `nuka_phi2_rt`.
- `RenderFrameKernel` + `RenderFrame`/`RenderFrameToAovs` stay in `two_level_render.cu`, FP64.

### 1b. Secondary-ray cut (in `bvh_traverse_impl.cuh` + the beauty TU)
- **AO-ray tMax prune** (byte-exact): the AO/GI bounce ray (ShadeBeauty L533) seeds best_t=+inf but
  only consumes hits with `bt < ao_radius`. Pass `ao_radius` as the ray's max_dist so `want()` skips
  children whose entry-t ≥ max_dist. KEEP it closest-hit (gi_bounces=1 consumes the nearest hit).
  Mind `>=`/eps consistency with the existing `bt >= ao_radius` test → no 1-ULP flip.
- **Visibility any-hit**: add a `TraverseRayAnyHit` functor that returns on the FIRST hit in
  `[t_min, dist)`. Apply ONLY to pure-visibility rays: soft-shadow (L504), GI-shadow (L554).
  Boolean visibility is order-invariant.

### 1c. Download-mask (in `RenderBeauty` host path, two_level_render.cu:1120-1126)
The mp4 path consumes ONLY `color`+`prim` (`render/rt_framebuffer_to_report.cpp:49-58`). D2H-copy
only those two by default; keep ALL 6 AOVs DEVICE-RESIDENT (the denoiser in Dispatch 3 reads
normal/albedo/depth). Make it an output-mask param, not a delete.

### Verification gate (ALL must pass before the controller commits)
1. Build `nuka_phi2_rt` + the 5 RT test targets in build-cuda128, GREEN.
2. The golden gate (constraint #1 list) GREEN — proves FP64 path byte-untouched.
3. `OfflineReplay.*ByteExact` + every `*D1TwoRunByteExact*` GREEN — proves beauty determinism.
4. PERF: `--probe --gpu --placeholder --robots 10` at 1080p@{1,16}spp, 540p@1spp, 128²@1spp
   BEFORE vs AFTER; report the measured speedup factor (the deliverable number).
5. Report context size; if >~350k tokens, RETURN for relay rather than claim done.

---

## Dispatch 2 — persistent render context + build-TLAS-once + batched sensor (OUTLINE)

Kills Enemy 1 (the SENSOR enabler). Depends on the P1 SensorDesc landing.
- A `RtRenderContext` holding PERSISTENT device buffers (AOV + instance + TLAS scratch) sized for
  (res, instance-count); reused across frames (no per-frame malloc/free).
- Build the TLAS ONCE per sim-step; render ALL env cameras against it → 67ms amortizes to 67/N.
  For the video (poses change per frame) the win is persistent buffers + (optionally) a faster
  rebuild; a TLAS REFIT changes Morton order ⇒ NOT byte-exact ⇒ keep rebuild for golden paths.
- `RenderSensorBatch` over the offline-RHI `RenderProfile::Sensor` seam: many small cameras, low
  res, cheap shade profile (1spp, shadow_rays=1, ao_samples=0, gi_bounces=0). Tiled per-env AOVs;
  D1 depth/seg/range. Python `nuka_world_get_sensor_view` dlpack (zero-copy, ActiveBackend).

## Dispatch 3 — 1spp + temporal denoise for video (OUTLINE)
Self-written edge-avoiding A-trous (Dammertz) over the device-resident AOVs → drop video SPP 16→3-5;
later SVGF temporal. Pure CUDA (no OIDN/OptiX). Ship as a profile-selected beauty mode + a
deterministic no-denoise toggle. Eyeball SSIM on a real clip before lowering export SPP.

---

## Build/run env
- build: `make -C build-cuda128 nuka_phi2_rt nuka_rt_two_level_test nuka_rt_render_world_test nuka_rt_scene_render_test nuka_rt_traversal_test nuka_rhi_offline_replay_test -j$(nproc)`
- PATH=/root/.nuka-toolchain/bin:/opt/cuda-12.8-root/usr/local/cuda-12.8/bin:/root/miniconda3/envs/nuka-v03/bin:$PATH
- LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:/root/Nuka-Physics/build-cuda128/src
- CUDA_VISIBLE_DEVICES=0
- perf probe binary already built: `build-viewer/tests/nuka_go2_terrain_demo` (rebuild build-viewer to pick up RT changes for the probe, OR run the RT bench `nuka_rt_two_level_bench`).
