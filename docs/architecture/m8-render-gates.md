# M8 / M8.5 render-gate inventory (the RenderWorld showcase stack)

> **Authoritative gate inventory** for the M8 (offscreen frame-loop + raster PBR)
> and M8.5 (realtime swapchain viewport + Dear ImGui custom UI + scalar PBR
> beauty) render milestones. Dated 2026-06-13.
>
> Companion doc: [`headless-rendering.md`](headless-rendering.md) (the v0.3
> offscreen `vulkan_renderer.cpp` path, now historical for the robot showcase — see
> its `## M8/M8.5 update` section). This doc is the gate/build/run authority; that
> doc is the headless-Vulkan background.

This is a **reference** document: it does not instruct anyone to run builds. It
records what each gate proves and the exact recipe so a reviewer can reproduce the
flag-ON validation.

---

## 1. The render stack under test

The M8/M8.5 showcase path is a new, backend-agnostic stack, distinct from the
demoted compute-only `src/render/vulkan_renderer.cpp` (the v0.3 debug-draw path):

| Layer | Files | Role |
|---|---|---|
| Host data product | `src/render/render_world.{hpp,cpp}` | `RenderWorld` (`RenderInstance` / `MeshLibrary` / `PoseSource`), built **once** by `BuildRenderWorld(Registry, SceneMap)`. Pure data — **zero CUDA tokens** (lint red-line, §5). Feeds raster, the M11 path-tracer, and the viewer with no rework. |
| Offscreen raster | `src/render/raster/vulkan_raster_renderer.{hpp,cpp}` + shaders `mesh.vert` / `mesh_pbr.frag` | Forward Cook-Torrance GGX PBR into an offscreen `VkImage` → `VulkanOffscreenReport::pixels`. This is the **D1 correctness oracle**. Carries `RendererConfig` (present-capable opt-in), a `VulkanHandles()` accessor (for ImGui init), and `RasterOptions` (`draw_ground` / `hero_framing` default-OFF). A `Render(world, opts, overlay)` seam composites an ImGui overlay. |
| Swapchain present | `src/render/raster/vulkan_present_renderer.{hpp,cpp}` | `PresentRenderer::DrawFrame` — acquire → draw → `vkQueuePresentKHR`, per-frame semaphores + in-flight fence; optional `OverlayRecordFn`. Inherently non-D1 (vsync / frames-in-flight); isolated from the offscreen oracle. |
| Windowing | `src/render/window/window_surface.{hpp,cpp}` | `XcbWindowSurface` + `MakeSurface`. Runtime-selectable backend: prefers `VK_EXT_headless_surface`, falls back to `VK_KHR_xcb_surface`. **On this box's loader (1.2.131) headless-surface is UNAVAILABLE**, so xcb-under-Xvfb is the run-verifiable path (§4.3). |
| ImGui UI | `src/render/imgui/nuka_imgui.{hpp,cpp}` + `external/imgui/` (vendored v1.92.8-docking) + `external/imgui/fonts/JetBrainsMono-{Regular,Bold}.ttf` | `NukaImGuiContext` RAII + `ApplyNukaTheme` — the custom-beautified "Nuka" theme (teal accent `#2DD4BF`, rounded panels, JetBrains Mono). **Owner requirement: custom-beautified, NOT stock ImGui.** |
| Pose publish | `src/runtime/app/pose_publisher.{hpp,cpp}` | Abstract `PosePublisher` + `HostDownloadPublisher` (`Data::DownloadField` → host → `RenderWorld`). The abstract seam is what lets the **M11 DLPack-Vulkan interop publisher** drop in with zero caller change (§6). |
| Frame loop | `src/runtime/app/simulation.{hpp,cpp}` + `systems.{hpp,cpp}` | `Simulation::Frame` = Input → Sim → [render: TransformSync → Render]. Render is **fully bypassed** when off (the G5 throughput-neutrality contract). |
| Viewer | `src/runtime/app/viewer/{viewer_main,imgui_layer,camera_controller}.{cpp,hpp}` | The custom-beautified docked realtime viewer (windowed exe). |

---

## 2. The six gates

All six are registered in `tests/CMakeLists.txt`. Three build **only** behind the
`if(NK_BUILD_VULKAN_VALIDATION)` CMake block; the other three build **always** but
internally gate their device-touching assertions (a `#ifdef NK_BUILD_VULKAN_VALIDATION`
inside the source, or no Vulkan at all). The "gated-by-flag?" column captures
exactly which.

| Gate | Source | CMake target | What it proves | Xvfb? | Gated by `NK_BUILD_VULKAN_VALIDATION`? |
|---|---|---|---|---|---|
| **GATE-A** (M8 T3a) | `tests/scenario/render_raster_smoke.cpp` | `nuka_render_raster_smoke_test` | Offscreen raster pipeline works on the lavapipe CPU ICD: synthetic 2-box `RenderWorld`, `non_background_pixel_count > 0` (G3), two `Render()` calls **byte-identical** (`memcmp == 0`, G2). Dumps `/tmp/m8_raster_smoke.ppm`. SKIPs if the renderer ctor throws (no device). | No | **Yes** — whole target inside the guard block. |
| **M8 T4** (frame-loop spine) | `tests/scenario/render_frame_loop_smoke.cpp` | `nuka_render_frame_loop_smoke_test` | The `Simulation::Frame` spine on a real cooked `.nks`. **G5 (NON-gated, runs in default build):** render-OFF `Frame()` × 16 vs bare `world.Step()` × 16 → downloaded state **byte-identical** (the render-off branch is a true bypass, not a no-op). **G1 (gated):** rendered `Frame()`s, then for sampled instances `instance.world_xform == downloaded_pose * cached_visual_local` bit-exact + `non_bg > 0`. Asset-gated SKIP. | No | **Partial** — target always builds; the **G5** case runs unconditionally, the **G1** case is `#ifdef`'d on the flag. |
| **T7 EXIT GATE** | `tests/scenario/render_physics_parity.cpp` | `nuka_render_physics_parity_test` | **THE M8 honesty gate.** Real H1 union fixture (`Load(.nks)` → `CookToModel` → `nk::World` + `BuildRenderWorld` + `Simulation`). **G1:** over **ALL** non-Static instances, `world_xform == downloaded_pose * cached_visual_local` bit-exact (must be ≥1 — never vacuous). **G2:** two `Render()`s of the same frame byte-identical (RAW pixels, never mp4). **G3:** `non_bg > 0`. Poses read from `nk::World::DownloadField` directly (D2, M9-proof). Asset-gated SKIP. | No | **Yes** — real gate body is `#ifdef`'d; in the default build it compiles to a single `RequiresVulkanValidationBuild` SKIP so the exe still links/registers. (Target itself builds always; the gate logic is flag-only.) |
| **GATE-C** (M8.5 T2) | `tests/scenario/viewport_present_smoke.cpp` | `nuka_viewport_present_smoke_test` | The swapchain **present** path: `MakeSurface` (xcb under Xvfb) → swapchain → acquire → draw → `vkQueuePresentKHR` for N=3 frames. Asserts loop completes (no `VkError`), `SwapchainImageCount() >= 2`, all 3 frames presented. SKIPs cleanly with no display / no surface ext / no device. | **Yes** | **Yes** — whole target inside the guard block. |
| **GATE-B** (M8.5 T3) | `tests/scenario/viewer_frame_smoke.cpp` | `nuka_viewer_frame_smoke_test` | The ImGui **viewer frame loop**, offscreen. Inits `NukaImGuiContext` against the renderer's offscreen render pass (`VulkanHandles()`), records the **real** `imgui_layer` panels with a **fixed** UI state, composites scene + `ImDrawData` via `Render(world, opts, overlay)`. Asserts `ImDrawData.CmdListsCount > 0`, composite `non_bg > 0`, two composites **byte-identical** (D1, fixed UI). Dumps `/tmp/m8_5_viewer_frame.ppm`. SKIPs with no device / ImGui init fail. | No | **Yes** — whole target inside the guard block. |
| **M8.5 T5** (visual-mesh cook) | `tests/scenario/visual_mesh_cook_smoke.cpp` | `nuka_visual_mesh_cook_smoke_test` | The visual-mesh cook produces real `.nka` MESH triangles: cook go2 (16 visual meshes) and, if present, h1_with_hand (49) through import → `Save(.nks/.nka)` → `Load` → `BuildRenderWorld`; asserts ≥1 MESH chunk in the `.nka`, ≥1 instance is `MeshSource::NkaMesh` with `>0` triangles, and the round-trip preserves the geometry. Scene-side only — **no Vulkan, no CUDA**. | No | **No** — builds + runs in every config (`add_test(NAME VisualMeshCook …)`, outside the guard). |

### Notes on the registrations

- `nuka_render_raster_smoke_test`, `nuka_viewport_present_smoke_test`,
  `nuka_viewer_frame_smoke_test`, and the historical `nuka_render_test`
  (`render/test_vulkan_backend.cpp`) live entirely inside the single
  `if(NK_BUILD_VULKAN_VALIDATION)` block in `tests/CMakeLists.txt`: they do **not
  exist** as build targets when the flag is OFF.
- `nuka_render_frame_loop_smoke_test`, `nuka_render_physics_parity_test`, and
  `nuka_visual_mesh_cook_smoke_test` are declared **outside** that block — they
  always build. The first two compile `NK_BUILD_VULKAN_VALIDATION` only when the
  flag is set (so their device-touching assertions are present only then); the cook
  test never needs Vulkan.

---

## 3. Determinism / honesty discipline

- **G1 — pose correctness (the honesty gate).** The renderer must draw the **real
  physics pose**, not a fabricated one. The parity gate independently re-downloads
  each instance's source FK pose (`LinkPose` / `BodyPose` / `BasePose`) for the
  selected env and asserts `world_xform == downloaded_pose * cached_visual_local`
  **bit-exact** (`memcmp` of two `math::Transform`s == 0) over **all** non-Static
  instances, with a non-vacuity check (≥1 physics-driven instance, ≥1 actually
  compared). `cached_visual_local` is computed once at `BuildRenderWorld` time
  (Decision D1), so `TransformSync` is a single multiply.
- **G2 — byte-identity (D1).** Two renders of the same frame must be
  byte-identical: `memcmp` of the two `VulkanOffscreenReport::pixels` vectors == 0,
  on **RAW framebuffer pixels, never encoded mp4 bytes** (Decision D6). The
  offscreen renderer pins a deterministic pipeline (no hardware MSAA —
  `VK_SAMPLE_COUNT_1_BIT`; `LESS_OR_EQUAL` depth; single submit + `vkQueueWaitIdle`).
  Any AA must be supersample + deterministic downsample. The realtime **present**
  path is legitimately non-D1 (vsync / frames-in-flight) and is kept off the G2/G1
  path — the offscreen `Render()` stays the sole correctness oracle.
- **G5 — throughput neutrality.** Render-off `Simulation::Frame()` must issue ZERO
  extra device work vs a bare `world.Step()` loop. Proven byte-exact by
  `render_frame_loop_smoke`'s G5 case (runs in the default build, no flag).
- **Zero-CUDA-token lint red-line.** `tools/lint/banned_patterns.yaml` scope
  `nk_engine` covers `src/nk/**`, `src/scene/**`, **`src/render/**`**, and
  **`src/runtime/app/**`** (`.hpp` + `.cpp`). It bans, as **errors**: `<<<`
  triple-chevron launches, `cuda*(` runtime calls, `#include <cuda_runtime…>`, and
  `#include "phi/backend_cuda…"`. So the entire RenderWorld/raster/present/imgui/
  viewer/publisher stack is pure host C++/Vulkan; all device pose access goes
  through `Data::DownloadField` (host) — never a CUDA scatter. This is precisely
  why M8/M8.5 pose-publish is host-download.

---

## 4. Flag-ON validation recipe (NEVER reconfigure build-cuda128)

### 4.1 The two build dirs

- **`build-cuda128`** — the default. `NK_BUILD_VULKAN_VALIDATION=OFF`. The gated
  render TUs (`nuka_imgui` / `nuka_present` / `nuka_viewer` libs, and the
  GATE-A/B/C smoke targets) are **stubbed / not built**; recorder + viewer code
  compile to `NOT_SUPPORTED`/SKIP stubs. **NEVER reconfigure this dir** — its
  `CMAKE_CUDA_COMPILER` cache pins the correct `/opt/cuda-12.8-root` nvcc; the
  system nvcc is the wrong CUDA 11.3.
- **`build-viewer`** — a separate dir configured with
  `NK_BUILD_VULKAN_VALIDATION=ON`, where the gated TUs are **real** (verified:
  `build-viewer/CMakeCache.txt` carries `NK_BUILD_VULKAN_VALIDATION:BOOL=ON`).

The flag-ON validation is done in `build-viewer` (or by recompiling the gated TU
with `-DNK_BUILD_VULKAN_VALIDATION` against the built archives) — **never** by
flipping the flag in `build-cuda128`.

### 4.2 Environment exports (run from repo ROOT)

```sh
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
# git operations only:
export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH
```

### 4.3 Running each gate

The ICD here is **lavapipe / llvmpipe (CPU Vulkan)** — no NVIDIA Vulkan on this box.
Asset-gated tests SKIP if `examples/scenes/h1_cup_table.nks` + the imported
MJCF/USD assets are absent. Renderer-construction failures SKIP cleanly.

> **Asset note (h1_with_hand .nka deferral):** the cooked `h1_with_hand` `.nka`
> (~88 MB of STL/OBJ geometry) is intentionally **NOT committed** pending an LFS
> decision. `examples/scenes/go2.nks`/`.nka` is the committed pipeline proof, and
> h1 is reproduced on demand from the source MJCF via `nuka_cook_scene` (the
> `VisualMeshCook.H1WithHandRendersRealMeshes` gate cooks+renders it live when the
> assets are present, else SKIPs).

| Target | How to run (from repo root) |
|---|---|
| `nuka_render_raster_smoke_test` | direct (offscreen, no display): `build-viewer/tests/nuka_render_raster_smoke_test` |
| `nuka_render_frame_loop_smoke_test` | direct. The **G5** case also runs from `build-cuda128` (default build, no display). |
| `nuka_render_physics_parity_test` | direct (offscreen). The real gate runs only in the flag-ON build. |
| `nuka_viewer_frame_smoke_test` | direct (offscreen ImGui composite — no window needed). |
| `nuka_viewport_present_smoke_test` | **under Xvfb** (xcb surface needs a virtual display): `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json xvfb-run -a -s '-screen 0 1280x720x24' build-viewer/tests/nuka_viewport_present_smoke_test` |
| `nuka_visual_mesh_cook_smoke_test` | direct, **any build** (scene-side; no Vulkan / no flag). |

**Only the present/window/viewer paths need `xvfb-run`** — they create a real
`VK_KHR_xcb_surface`, which needs a virtual display. The offscreen raster, the
frame-loop spine, the parity gate, the offscreen ImGui composite, and the
visual-mesh cook all run headless with **no** display.

---

## 5. Manual on-display checklist (real NVIDIA + display required)

This box has CUDA (NVIDIA GPU for physics) **plus** llvmpipe/lavapipe (CPU Vulkan
ICD only). The items below are **structurally unverifiable here** and must be
checked by a human on a real NVIDIA-GPU + physical-display machine. Each is a
PASS/FAIL a reviewer runs and eyeballs:

- [ ] **Interactive realtime viewer presents to a window.** Build + launch the
  windowed `viewer_main` (`nuka_viewer`) on a machine with a display. Headless we
  only prove that frames *present* under Xvfb (GATE-C); a human must confirm an
  **actual window opens, the robot animates, and interaction feels live** —
  smooth vsync pacing, mouse orbit/pan/zoom (`camera_controller`), and the
  play/pause/step/reset controls respond with no perceptible lag or tearing.
- [ ] **Custom ImGui theme appearance.** Confirm the **custom-beautified** Nuka
  look — teal accent `#2DD4BF`, rounded panels, JetBrains Mono fonts, the docked
  panel layout (viewport / stats / scene tree / camera / drive sliders) — and that
  it is **NOT** the default grey stock-ImGui look. This is the owner's
  "美观 / custom-not-stock" requirement and is a human judgement.
- [ ] **GPU-accurate PBR beauty colors.** The hero shots referenced in the M8.5
  close (`/tmp/m8_5_go2_pbr.png`, `/tmp/m8_5_h1_pbr.png`) were **CPU-rendered on
  lavapipe**. lavapipe shading approximates — but is not bit-identical to — a real
  GPU. Re-render the showcase on a real NVIDIA GPU and confirm the Cook-Torrance
  GGX metallic/roughness/emissive response, the 3-point light rig, ACES tonemap,
  and the grounded look read correctly (no banding / wrong gamma / blown
  highlights).
- [ ] **M11 DLPack-Vulkan zero-copy interop.** CUDA↔Vulkan external-memory sharing
  is **impossible CUDA↔llvmpipe** (different physical devices) — so the M11
  shared-memory render is not run-verifiable here at all. When the M11 interop
  publisher lands, validate on a real NVIDIA-Vulkan machine that the renderer reads
  the physics device buffers **with no D2H copy** and the rendered pose matches the
  live physics state with zero sync gap. (M8/M8.5 ship host-download; the
  `PosePublisher` seam swaps the interop publisher in with no caller change — §6.)

---

## 6. The M11 DLPack-Vulkan interop direction (host-download now → zero-rework swap)

M8 and M8.5 ship with `HostDownloadPublisher` (`Data::DownloadField` → host →
`RenderWorld`), which already renders the **real** physics poses — just via a
per-frame device-to-host copy. The abstract `PosePublisher` interface
(`pose_publisher.hpp`, `virtual Publish(world, env_index, …)`) was designed
precisely so the **M11 DLPack-Vulkan interop publisher** (the
M8-named `CudaVulkanInteropPublisher`) drops in behind the same interface with
**zero caller change** to the viewer or recorder.

At M11 the publisher reads the SAME GPU device buffers the physics engine writes
(zero-copy shared memory) via DLPack (`kDLVulkan`, or `kDLCUDA` + external-memory
import) → Vulkan imports them as external memory → the raster/RT renderer reads
instance transforms directly from the shared buffer, no D2H. This is a **firm M11
requirement** (owner directive 2026-06-13). It is build-only here — CUDA↔llvmpipe
external-memory sharing cannot work (see §5, last item).
</content>
</invoke>
