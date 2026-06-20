# Batched device-resident sensor core (D2) — design (controller, 2026-06-20)

The 4096-env near-real-time RGB/depth sensor path. Owner: "4096 个环境、每环境多机器人,
RGB sensor 也要接近实时." The strict-perf core — ABOVE the pretty video.

## The principle (why this is feasible)
Grounding map (read-only Explore, 2026-06-20) found **every device building block already
exists**; only the BATCHED GLUE is missing. The path must SEVER two host dependencies:
- `render::RenderWorldToTwoLevelScene` host copy `out.transform = inst.world_xform` (rt_adapter.cpp:84)
- the host `scene.instances` array that `RenderFrame*` rebuilds the TLAS from each call.
Replace both with: device poses → device transforms → device TLAS → tiled batched trace →
device AOV tensor → dlpack. No host round-trip, no per-frame re-cook, ONE TLAS build/refit per
sim-step shared across ALL env cameras.

## What EXISTS (reuse, do NOT duplicate)
- **Device poses:** `nk::World::FieldPtr(FieldId::LinkPose|BodyPose|BasePose)` → device ptr,
  env-major (`e*links_per_env + L`), 28 B/row = pos3 + quat4 (W-first). model.cpp:140-141, arena_layout.hpp:52-54.
- **Device compose kernel:** `ScatterTransformsKernel` (cuda_vk_scatter.cu:134-202) computes
  `world_xform = fk(LinkPose) ∘ cached_visual_local` on device, bit-identical to host. Input POD
  `InstanceScatterRow{kind,row,cached_visual_local[7]}` (interop_scatter.hpp:96-100) + `ScatterFkSource`
  (the device pose ptrs). It writes a Vulkan mat4 SSBO today — REUSE its arithmetic, retarget output.
- **In-place device-AOV tracer:** `rt::RenderFrameToAovs`/`RenderBeautyToAovs` + `RtDeviceAovs`
  {color,depth,normal,albedo,uv,prim device buffers} (two_level_render.hpp:128-183), `BuildTwoLevelScene`
  (BLAS once), 4096-instance cap (prim_id.cuh:42).
- **Refit primitive:** `collision::gpu::RefitLbvh` (lbvh_refit.cu, 0 call sites). **Refit ≡ rebuild
  BYTE-EXACT** because the closest-hit winner is a total order on `(t, packed-prim-id)` (prim_id.cuh:8-13),
  independent of traversal order. (Supersedes the impl-spec's earlier cautious "refit not byte-exact" note —
  but VALIDATE against the D1 goldens, never assume.)
- **SensorDesc (committed 7f2feda):** mount/mount_index/local_offset/cam/lidar/aov_mask/update_period;
  `MountFrame{Link,Body,Base}`→`FieldId{LinkPose,BodyPose,BasePose}` contract (canonical_types.hpp:56-57).
- **dlpack obs pattern:** `torch.from_dlpack(world.buffer_view(field))` zero-copy; `make_array_from_view`
  (nuka_ext.cpp:539-613); `nuka_world_get_buffer_view` + `nuka_buffer_view_t` (nuka.h:500-511).

## What to BUILD — ordered sub-dispatches (each independently verifiable)

**D2.1 — device instance-transform → device TLAS (sever the host deps).**
Factor `ScatterTransformsKernel`'s `fk∘cvl` arithmetic into a reusable device fn; add a path that writes
a DEVICE array of `rt::Instance` transforms (or world-AABBs) from `LinkPose` + a static `InstanceScatterRow`
table, and build the TLAS from THAT (not host `scene.instances`). Wire `RefitLbvh` as its first caller for
the per-step update. Gate: existing RT D1 goldens stay byte-exact (refit≡rebuild proven); a device-transform
TLAS of the demo scene matches the host-transform TLAS byte-for-byte.

**D2.2 — sensor-mount kernel.** For every (env, sensor): `cam_world_pose = LinkPose[e*links_per_env +
mount_index] ∘ local_offset` (Body/Base variants per MountFrame). Same arithmetic as D2.1; output is a
per-env `PinholeCamera`. Decide: a render-side device buffer of camera poses (no new FieldId needed) vs a
`FieldId::SensorWorldPose` arena field (only if the policy/obs must read sensor pose as a field). Prefer the
render-side buffer (no arena-schema change) unless a consumer needs the field.

**D2.3 — batched tiled multi-camera launch.** Extend the tracer to take N per-env cameras + write one
`(env, H, W, ch)` device tensor (tiled). ONE TLAS (D2.1) shared across all cameras. Cheap sensor shade
profile: 1spp, shadow_rays=1, ao_samples=0, gi_bounces=0 (RenderProfile::Sensor). D1 depth/seg/range AOVs.
Reuse the FP32 beauty TU (D1) for the per-ray cost. Gate: per-env tiles match single-camera renders;
4096-env × 64² per-step timing reported.

**D2.4 — offline RHI SensorBatchPass.** New `RenderPassKind::SensorBatch`; `RenderGraph::FromProfile`
emits it for `RenderOutput::Sensor` (today returns empty graph — render_graph.hpp:42-48); `PassContext`
gains N cameras + device AOV tensor; `RenderSensorBatch` orchestrator entry. Binds the global ActiveBackend.

**D2.5 — C-ABI + Python obs.** Attach an RtBackend + render scene handle to the C-ABI `WorldRecord`
(none today — internal.hpp). `nuka_world_get_sensor_view(world, sensor_id, view*)` → device tensor;
nanobind `World.get_sensor_view(...)` → `(env,H,W,3)` ndarray via `make_array_from_view` → `torch.from_dlpack`.
Python `CameraCfg`/`LidarCfg` author SensorDescs.

## Constraints (CLAUDE.md + D1)
ONE general path: the sensor render is the SAME tracer at a cheap RenderProfile, NOT a new solver/path; reuse
the compose kernel's math (don't duplicate it). Comments ≤2 lines, no temporal/process words. The RT D1
byte-exact goldens (nuka_rt_two_level/render_world/scene_render/traversal_test + offline_replay) stay GREEN
through every sub-dispatch; refit/device-TLAS validated against them, never loosen a tolerance.

## Verify (the deliverable numbers)
Per-step wall-time for the batched path at increasing scale (e.g. 256/1024/4096 envs × 64² × 1spp cheap-shade),
device-resident (no D2H). Target: render-step ≪ the ~16k env-steps/s physics budget when rendering every K
control steps. Plus the D1 byte-exact gate after each sub-dispatch.

## Coupling note
D2.1 touches the TLAS build in `two_level_render.cu` — wire it AFTER D1 (the FP32 split) commits, to avoid
editing the same file concurrently.
