# M11 render / interop / RT gate inventory (the viewer + zero-copy + RtBackendI stack)

> Companion to `m8-render-gates.md`. Covers the M11 additions: the RT path tracer homed behind
> `RtBackendI` (§3.10), the CUDA↔Vulkan zero-copy interop publisher (the firm shared-memory
> directive), the ImGui viewer delta (drag/MoveEntity + drive sliders + entity picker), and the
> render_scene deletion. Same two-build-dir discipline as M8 — **NEVER reconfigure build-cuda128.**

## 1. What M11 added to the render/UI stack

| Area | Files | Essence |
|---|---|---|
| **RT homing (§3.10)** | `src/render/rt_backend.hpp` (RtBackendI, CUDA-free) + `src/render/rt_adapter.{hpp,cpp}` (RenderWorldToTwoLevelScene, host) + impl in `src/phi/backend_cuda/rt/*` (homed from src/rt) | The self-written path tracer (NO OptiX) now sits behind a backend-agnostic interface; the engine/render layer is zero-CUDA, the CUDA impl lives in `nuka_phi2_rt` (dedicated lib, **`--fmad=false`**). RT output is via phi-v2 `BufferI` (6 AOV). RT is **still D1**. |
| **CUDA↔Vulkan interop** | `src/phi/interop_scatter.hpp` (CUDA-free seam) + `src/phi/backend_cuda/interop/cuda_vk_scatter.{hpp,cu}` (scatter + import + semaphore) + `src/render/raster/interop_transform_ssbo.{hpp,cpp}` + `mesh_instanced.vert` + `src/runtime/app/cuda_vulkan_interop.{hpp,cpp}` | `CudaVulkanInteropPublisher` implements the same `PosePublisher` interface as `HostDownloadPublisher`; opt-in, default-OFF. Real zero-copy = `cudaImportExternalMemory` over a `VK_KHR_external_memory_fd` SSBO. **Not run-verifiable here** (CUDA↔llvmpipe cannot share memory). |
| **Viewer delta** | `src/runtime/app/viewer/{viewer_main,imgui_layer,camera_controller}.*` + `systems.{hpp,cpp}` (MoveEntity) + `src/render/imgui/nuka_imgui.*` (RebuildForRenderPass) | Ctrl+LMB entity pick + drag → MoveEntity → live `nk::Data` `BodyPose`/`BasePose` write (never the frozen .nks); generic per-DOF drive-slider panel → `FieldId::DriveTarget`; swapchain-recreate ImGui rebuild; StepOnce same-frame. |
| **render_scene deletion** | (deleted) `src/render/render_scene.{hpp,cpp}` + the RenderScene overlay wrappers in `vulkan_renderer.{hpp,cpp}` | The dead pre-RenderWorld scene path is gone; the live command-list debug overlay survives. (The `rt::RenderScene` path tracer is unrelated and untouched.) |

## 2. The M11 gates

Same gating model as M8: most render TUs build **always** but `#ifdef NK_BUILD_VULKAN_VALIDATION`
their device-touching bodies; the present/viewer/interop targets are whole-target-gated.

| Gate | Source | What it proves | Build dir | Xvfb? |
|---|---|---|---|---|
| **RT-5 offscreen D1** | `tests/rt/test_render_world_rt.cpp` (`RenderWorldRt`) | RenderWorld → rt_adapter → RtBackendI → image: two-run dispatch 6-AOV `memcmp==0` + adapter-vs-hand-built-oracle parity + `BufferI` device-output == host-convenience (OD-12) + coverage>0. **RT still D1.** | `build-cuda128` (CUDA region, OUTSIDE NK_BUILD_VULKAN) | No |
| **RT relocation D1** | `tests/rt/test_{two_level_render,scene_render,bvh_ray_traversal}.cpp` | The homed kernels are byte-identical post-move: `D1TwoRunByteExactAllAovs`, `Oracle1HostDeviceByteExact`. | `build-cuda128` | No |
| **SDF-tier forward** | `tests/collision/test_narrowphase_dispatch.cpp` (`SdfTierWired`) | `BoxOnBoxMatchesAnalyticalNormalAndPenetration` (normal/penetration vs analytical truth) + `TwoRunByteIdenticalManifold` (the M9 carry-forward re-asserted). | `build-cuda128` | No |
| **Viewer MoveEntity** | `tests/scenario/viewer_move_entity.cpp` (`ViewerMoveEntity`) | Push MoveEntity(cup) → Frame/Publish → `DownloadField(BodyPose)` == requested (1e-5) + velocity zeroed (OD-6); StepOnce applies same-frame. Host, no Vulkan. | `build-cuda128` | No |
| **Viewer camera ray** | `tests/scenario/viewer_frame_smoke.cpp` (`ViewerCameraRay`) | ScreenRay center + known-plane-hit. Host. | `build-cuda128` | No |
| **GATE-B (extended)** | `tests/scenario/viewer_frame_smoke.cpp` | Offscreen ImGui draw-data two-run **byte-identical** with the new Drive + Entity panels at a **FIXED** ViewerUiState (R12 — no time-seeded widgets). | `build-viewer` | No |
| **Interop fallback** | `tests/scenario/interop_fallback_smoke.cpp` | On lavapipe (`vkGetMemoryFdKHR` absent) the SSBO export reports `Supported()==false` and the viewer **gracefully degrades** to `HostDownloadPublisher`. The build-link + fallback gate that stands in for the unverifiable real zero-copy. | `build-viewer` | (smoke) |
| **Offscreen unchanged (interop OFF)** | `render_physics_parity` / `render_raster_smoke` / `RtTwoLevelRender` | The interop SSBO + instanced pipeline are default-OFF → every offscreen render byte is unchanged. | both | No |

## 3. Determinism / honesty discipline (M11-specific)

- **RT `--fmad=false`:** the homed RT lib `nuka_phi2_rt` keeps `--fmad=false` (paired with the host
  oracle `-ffp-contract=off`) so device == host bit-exact. It MUST NOT be folded into `nuka_phi2`
  (nvcc default `--fmad=true` would silently diverge every RT golden). Verified in `flags.make`.
- **RT relocation = include-path-only:** the 9 device files were `git mv`'d src/rt → backend_cuda/rt
  with body-identical content (only `#include` paths changed); `PackPrimId` bit-packing moved
  bit-identical. The existing rt D1 tests are the backstop.
- **Interop byte-exact transform (R14):** the scatter kernel reproduces `HostDownloadPublisher`'s
  `world_xform = fk * cached_visual_local` with the exact fp32 op sequence on the same device FK
  buffers (the repo's `instance_transform.cuh` HD-equivalent precedent — host-only `Transform::operator*`
  is not callable from device). Its **actual** byte-exactness vs host is an on-display item (§5).
- **Viewer never mutates the frozen .nks (R13):** all live edits go through `nk::Data::UploadField`
  (BodyPose/BasePose/DriveTarget/velocities) — never `nuka_scene_set_local`/Save. `UnionCookGolden`
  stays `0.000e+00`.
- **Buffer sweep D1:** the M11 buffer_legacy → phi-v2 sweep + the DeviceContext/stream removal were
  proven byte-exact (stream-0 pure type swap; `ctest -L full` + all goldens byte-identical).

## 4. Flag-ON validation recipe (NEVER reconfigure build-cuda128)

Identical to m8-render-gates §4. Two build dirs: **`build-cuda128`** (`NK_BUILD_VULKAN_VALIDATION=OFF`,
the default; its `CMAKE_CUDA_COMPILER` cache pins `/opt/cuda-12.8-root` nvcc — the system nvcc is the
wrong CUDA 11.3, **never reconfigure**) and **`build-viewer`** (`NK_BUILD_VULKAN_VALIDATION=ON`, where
the gated TUs are real). Environment (run from repo ROOT):

```sh
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH   # git operations only
```

Run (from repo root):
- `cmake --build build-cuda128 -jN` → the RT D1 gates, SDF-tier gate, viewer host gates (ViewerMoveEntity/ViewerCameraRay), and the whole core suite (`ctest -L full`).
- `cmake --build build-viewer -jN` → the flag-ON viewer/present/interop compile+link + GATE-B + interop_fallback. The viewer present path needs Xvfb: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json xvfb-run -a -s '-screen 0 1280x720x24' build-viewer/tests/<target>`.
- The M11 exit gate (greps): `git grep -lE "buffer_legacy|\b(OwnedStream|DeviceContext|UploadVector|StreamView)\b" src/ tests/` = 0; `render::RenderScene` live = 0; `physics_smell.py` rc=0.

> **Exit-gate honesty (EGH-F2, M11-end review).** The `ctest -L full -E "GoldensMatchCudaAba"`
> exclusion drops FOUR oracle tests, all pathologically slow (re-cook/large-sample, terminated
> unfinished): `FeatherstoneOracle.{RandomSample,FloatingBaseRandomSample}GoldensMatchCudaAba`
> (the pre-existing M0 known-fail family) **and** `FeatherstoneOracle.NkWorld{RandomSample,
> FloatingBase}GoldensMatchCudaAbaByteExact` (>33 min / >12 min — too slow for any exit gate).
> This does **not** leave the async-stream-swept articulation+buffer path unverified: the FAST nk
> byte-exact gates `NkWorldGo2Stand5sMatchesGoldenAndLegacyByteExact` (#277, ~1.7 s) and
> `NkWorldBatchedContactStepPlannedByteExact` (#278, ~1.2 s) **run inside `-L full` and pass
> byte-exact**, covering the same nk articulation + phi-v2 buffer path. The exclusion hides slow
> tests, not a regression.

## 5. Manual on-display checklist (real NVIDIA + display required — unverifiable here)

These are PASS/FAIL a human runs on a real NVIDIA-GPU + physical-display machine (this box has the
NVIDIA GPU for CUDA physics **but only llvmpipe/lavapipe CPU Vulkan** — the two cannot share memory):

- [ ] **Drag-entity edits the live scene.** Ctrl+LMB pick + drag a movable body (the cup) in the
  windowed viewer; confirm it teleports to the cursor's drag-plane point, velocity zeroes, and the
  physics continues from there (a runtime Data edit, lost on reset — never written back to the .nks).
- [ ] **Drive-slider panel drives the robot.** Move a per-DOF slider; confirm the corresponding joint
  PD target tracks it live (generic per-DOF, not a scripted choreography).
- [ ] **CUDA↔Vulkan zero-copy interop (the firm directive).** On real NVIDIA-Vulkan, enable the interop
  publisher and confirm the renderer reads the physics device transform SSBO **with no D2H copy**
  (`CudaVulkanInteropPublisher` active, not the host-download fallback) and the rendered pose matches
  the live physics state with zero sync gap. Confirm the byte-exact transform (R14) matches the
  host-download render pixel-for-pixel.
- [ ] **RT path-traced beauty on a real GPU.** The offscreen RT D1 gate proves correctness on CPU/CUDA;
  re-render the RT showcase on a real NVIDIA GPU and confirm the path-traced result reads correctly
  (no banding / wrong gamma) — RT is still D1, so the image must match the offscreen oracle.
- [ ] **Swapchain-recreate (resize) does not stale the ImGui renderpass.** Resize the window; confirm
  ImGui re-binds to the recreated present pass (VIEW-5) with no validation error / no black overlay.
- [ ] **Custom ImGui theme** (teal `#2DD4BF`, rounded panels, JetBrains Mono, docked drive/entity
  panels) — the owner's custom-not-stock requirement.

## 6. M10 boundary

M11 touches NO RL / go2 PD-stance / per-link wrench-control / autoreset / control-mode / SG-spec —
those are M10 (the last milestone). The viewer drive-slider writes a generic `DriveTarget`; it is NOT
a trained policy or a scripted stance.
