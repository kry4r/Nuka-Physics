# Headless rendering on a server with no desktop

> Scope note (2026-05-30): the v0.3 demo **video is deferred** (owner pivoted to sim
> correctness). This documents *how* we will render on a display-less server so the
> later video step is a known, low-risk path — not work to do now.

---

## M8/M8.5 update (2026-06-13): the RenderWorld showcase stack

**The v0.3 offscreen `vulkan_renderer.cpp` path described below is SUPERSEDED for
the robot showcase** by the new `RenderWorld → VulkanRasterRenderer` stack added in
M8 (offscreen frame-loop + forward PBR raster) and M8.5 (realtime swapchain
viewport + Dear ImGui custom UI + scalar PBR beauty). The v0.3 content below the
divider is kept for **historical reference** (it documents the old compute-only
debug-draw offscreen path, which M8 demoted to a debug overlay) — it is **stale for
the showcase**. For the authoritative gate inventory + build/run recipe see the
companion doc **[`m8-render-gates.md`](m8-render-gates.md)**.

### (a) What replaces the old path

| Concern | v0.3 (below, historical) | M8 / M8.5 (current) |
|---|---|---|
| Render entry | `RenderSceneVulkan(RenderScene, opts)` — single compute `vkCmdDispatch` of `debug_draw.comp`, 2D orthographic wireframe outlines | `BuildRenderWorld(Registry, SceneMap)` → `VulkanRasterRenderer::Render(world, opts)` — a real triangle **forward Cook-Torrance GGX PBR** pass into an offscreen `VkImage` |
| Scene model | `RenderScene` (SceneIR-derived) | `RenderWorld` (`RenderInstance` / `MeshLibrary` / `PoseSource`), backend-agnostic, **zero CUDA tokens** (lint red-line) |
| Pose feed | host-built `RenderScene` per frame | `Simulation::Frame` (Input → Sim → TransformSync → Render) + `HostDownloadPublisher` (`Data::DownloadField` → host → `RenderWorld`) |
| Pixel egress | `VulkanOffscreenReport::pixels` (RGBA8 readback) — **unchanged**; the offscreen report type is reused as the D1 oracle surface |

The offscreen `Render()` is the **D1 correctness oracle** (two renders
byte-identical), and the renderer draws the **real physics pose** — the M8 exit gate
`render_physics_parity` asserts, bit-exact, `world_xform == downloaded_pose *
cached_visual_local` over every physics-driven instance.

### (b) Realtime present / swapchain + custom ImGui viewer (M8.5)

M8.5 adds an **on-screen** path alongside the offscreen oracle:

- `src/render/raster/vulkan_present_renderer.{hpp,cpp}` — `PresentRenderer::DrawFrame`
  (acquire → draw → `vkQueuePresentKHR`, per-frame semaphores + in-flight fence).
- `src/render/window/window_surface.{hpp,cpp}` — `XcbWindowSurface` + `MakeSurface`,
  a **runtime-selectable** backend that prefers `VK_EXT_headless_surface` and falls
  back to `VK_KHR_xcb_surface`.
- `src/render/imgui/nuka_imgui.{hpp,cpp}` + vendored `external/imgui/` (v1.92.8-docking,
  JetBrains Mono) — `NukaImGuiContext` + `ApplyNukaTheme`, the **custom-beautified**
  Nuka theme (teal accent `#2DD4BF`), **not** stock ImGui (owner requirement).
- `src/runtime/app/viewer/{viewer_main,imgui_layer,camera_controller}` — the
  custom-beautified docked realtime viewer.

**This box's Vulkan loader is `libvulkan.so.1.2.131` (Feb 2020), which does NOT
export `vkCreateHeadlessSurfaceEXT`** — so `VK_EXT_headless_surface` is **unavailable
here** despite the lavapipe ICD `.so` carrying the string. The **run-verifiable
present path is therefore `VK_KHR_xcb_surface` under `Xvfb`** (proven by `vkcube`
exit-0 under the same setup). The realtime present loop is inherently **non-D1**
(vsync / frames-in-flight) and is kept off the G1/G2 deterministic offscreen path.

The literal monitor scan-out, vsync pacing, ImGui interaction feel, GPU-accurate PBR
beauty, and the custom-theme appearance are **NOT verifiable on this CPU-Vulkan box**
— see the **Manual on-display checklist** in
[`m8-render-gates.md`](m8-render-gates.md#5-manual-on-display-checklist-real-nvidia--display-required).

### (c) DLPack-Vulkan M11 interop direction

M8/M8.5 render the real physics poses via a per-frame **host-download**
(`HostDownloadPublisher`: `Data::DownloadField` → host → `RenderWorld`). The
abstract `PosePublisher` interface (`src/runtime/app/pose_publisher.hpp`) was
designed so the **M11 DLPack-Vulkan zero-copy interop publisher** drops in behind
the same interface with **zero caller change** (viewer / recorder untouched). At
M11 the renderer reads the SAME GPU device buffers physics writes (no D2H) via
DLPack (`kDLVulkan` / external-memory import). This is a **firm M11 requirement**
(owner directive 2026-06-13) and is build-only here: CUDA↔llvmpipe external-memory
sharing cannot work (different physical devices). See the memory note
`render-physics-shared-memory` and [`m8-render-gates.md`](m8-render-gates.md#6-the-m11-dlpack-vulkan-interop-direction-host-download-now--zero-rework-swap).

---

## Historical (v0.3, 2026-05-30) — the original offscreen compute path

> **Superseded for the robot showcase by the M8/M8.5 RenderWorld stack above.**
> Retained as background on the headless-Vulkan substrate (ICD discovery, no-display
> requirement) which the new stack still relies on. The compute-only
> `vulkan_renderer.cpp` it describes was demoted to a debug overlay in M8.

## TL;DR

Nuka's renderer is **already headless by construction**. `src/render/vulkan_renderer.cpp`
renders to an **offscreen `VkImage`** (`VK_FORMAT_R8G8B8A8_UNORM`), copies it into a
`HOST_VISIBLE` readback buffer, `vkMapMemory`+`memcpy`s it into
`VulkanOffscreenReport::pixels` (`std::vector<VulkanRgba8>`), and returns. There is **no
swapchain, no `VK_KHR_surface`, no X11/xcb/wayland/glfw, and no `DISPLAY` dependency**
(verified by grep — none of those symbols appear in `src/render/`). So no virtual
framebuffer (Xvfb), no GPU display output, and no window system are required.

Pipeline:

```
RenderSceneVulkan(scene, opts)  ->  report.pixels (RGBA8, opts.width x opts.height)
   -> write each frame as binary PPM (P6)        # trivial: header "P6 W H 255\n" + RGB bytes
   -> ffmpeg -framerate 30 -i frame_%05d.ppm -pix_fmt yuv420p out.mp4
```

`RenderSceneVulkan(const RenderScene&, const VulkanOffscreenOptions&)` and
`RenderDebugDrawListVulkan(...)` are the entry points (`src/render/vulkan_renderer.hpp`).
`VulkanOffscreenOptions` sets `width/height/view_scale/view_center/background`.

## The one runtime requirement: a Vulkan ICD (no display needed)

A Vulkan **Installable Client Driver** must be discoverable. This box has two ways, both
display-less:

1. **Lavapipe (CPU software rasterizer)** — `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`
   is already installed. Pure software, needs **no GPU and no display** at all; this is
   what lets the existing render tests (`tests/render/test_vulkan_backend.cpp`,
   `tests/perf/test_vulkan_renderscene_timing.cpp`) pass on this server. Slow but
   bulletproof for a one-off demo clip.
2. **NVIDIA GPU ICD** — present at `/tmp/nvidia_icd.json`. Select it with
   `VK_ICD_FILENAMES=/tmp/nvidia_icd.json` (env var is currently empty, so lavapipe is the
   default pick). GPU path is far faster for long/high-res renders. Vulkan offscreen needs
   the driver's compute/graphics ICD only — **not** a running X server or the (broken-here)
   `nvidia-smi`/NVML.

Render is independent of the physics GPU work: simulate on CUDA, download the state
(q/qdot + base pose via the C ABI buffer views), feed a `RenderScene`, render offscreen.

## For the eventual 16-env grid demo (v0.3 §p04, deferred)

- Step the 4096-env world; pick 16 envs; build a `RenderScene` per frame from their
  downloaded link poses; `RenderSceneVulkan` each frame at e.g. 1280x720, 30 fps.
- Dump PPM frames to a tmp dir, then `ffmpeg ... -pix_fmt yuv420p` to MP4 (confirm
  `ffmpeg` is on PATH at video time; if absent, any PPM->MP4 encoder works, or commit the
  PPM sequence + an encode script).
- GPU ICD (`VK_ICD_FILENAMES=/tmp/nvidia_icd.json`) recommended for the full clip;
  lavapipe is the fallback that needs nothing but CPU.

**Bottom line:** rendering on this display-less server is a solved, already-tested path;
the deferred video task is just "loop frames -> PPM -> ffmpeg," with the ICD env var the
only knob.
