# Headless rendering on a server with no desktop

> Scope note (2026-05-30): the v0.3 demo **video is deferred** (owner pivoted to sim
> correctness). This documents *how* we will render on a display-less server so the
> later video step is a known, low-risk path — not work to do now.

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
