# Vendored Dear ImGui — Nuka provenance record

This directory is a **copied-source vendor** of Dear ImGui (NOT a git submodule),
mirroring `external/vhacd/`. It is consumed by the `nuka_imgui` static glue lib
(`src/render/imgui/`) which is built ONLY when `-DNK_BUILD_VULKAN_VALIDATION=ON`
(M8.5 Decision D3). The default `build-cuda128` config never compiles any of this.

## Source

- **Upstream:** https://github.com/ocornut/imgui
- **Branch:** `docking` (the multi-viewport / docking branch — required for the
  docking + multi-viewport features and, more importantly, the ≥1.92 texture
  lifecycle, `ImGuiBackendFlags_RendererHasTextures` / `ImTextureData`, that the
  current `imgui_impl_vulkan` backend now depends on — M8.5 Decision D2).
- **Pinned tag:** `v1.92.8-docking`
- **Commit SHA (tag deref `^{}`):** `b61e56346a92cfcaf1f43a545ca37b0b32239654`
- **Fetched:** 2026-06-13 (network available; cloned `--depth 1 --branch v1.92.8-docking`).

Core and the `imgui_impl_vulkan` backend are copied at the **same tag/SHA** (mixing
core vs backend versions corrupts the font-atlas / texture lifecycle — pinned together).

## What was copied (and why)

### Core (`./`)
| File | Why |
|---|---|
| `imgui.cpp` | core |
| `imgui_draw.cpp` | draw-list + font atlas |
| `imgui_tables.cpp` | tables widget (used by the T3 stats/scene-tree panels) |
| `imgui_widgets.cpp` | widgets (sliders/buttons for the T3 drive/camera panels) |
| `imgui_demo.cpp` | kept — handy as a headless `ImDrawData` smoke source + a widget reference for T3 |
| `imgui.h` | public API |
| `imgui_internal.h` | needed by the backend + some style work |
| `imconfig.h` | compile-time config (left at upstream defaults; no custom defines) |
| `imstb_rectpack.h`, `imstb_textedit.h`, `imstb_truetype.h` | bundled stb deps (default font rasterizer — no FreeType dependency added) |
| `LICENSE.txt` | MIT (Dear ImGui) |

### Backend (`backends/`)
| File | Why |
|---|---|
| `imgui_impl_vulkan.h` / `.cpp` | the renderer backend (all platforms). |
| `imgui_impl_glfw.h` / `.cpp` | the PLATFORM backend on Windows (GLFW window/surface/input). Linux keeps the self-written xcb backend (no glfw backend compiled there), so it is added to the `imgui` lib only on `WIN32`. Copied at the same tag/SHA as core. |

### Fonts (`fonts/`) — the custom-beautified look (OWNER UI requirement)
| File | License | Why |
|---|---|---|
| `JetBrainsMono-Regular.ttf` | SIL OFL 1.1 (`JetBrainsMono-OFL.txt`) | UI body + numeric stats — the "engineering pro-tool" typeface (distinctive, not stock ImGui's ProggyClean) |
| `JetBrainsMono-Bold.ttf` | SIL OFL 1.1 (same) | headings / panel titles (a heavier weight in the same family → cohesive) |
| `JetBrainsMono-OFL.txt` | — | the SIL Open Font License covering both .ttf above |

Fonts fetched from https://github.com/JetBrains/JetBrainsMono (master, 2026-06-13).
JetBrains Mono is SIL OFL 1.1 — freely redistributable with the license bundled
(done). The font load + sizes live in `src/render/imgui/nuka_imgui.cpp::ApplyNukaTheme`.

## Licenses
- Dear ImGui: MIT — `LICENSE.txt`.
- JetBrains Mono: SIL OFL 1.1 — `fonts/JetBrainsMono-OFL.txt`.

## Not vendored (intentional)
- `imgui_impl_glfw.*`, `imgui_impl_sdl*.*`, `imgui_impl_*` other than Vulkan
  (platform backend is self-written xcb — D1).
- `misc/`, `examples/`, `docs/`, the `.natvis`/`.editorconfig`/CI files (not needed
  to build the static lib).
- FreeType (`misc/freetype/`) — the default stb_truetype rasterizer is sufficient
  for M8.5; no FreeType system dependency is introduced.

## Re-vendor recipe (for the next bump)
```
git clone --depth 1 --branch <newtag-docking> https://github.com/ocornut/imgui /tmp/imgui_vendor
cp /tmp/imgui_vendor/{imgui.cpp,imgui_draw.cpp,imgui_tables.cpp,imgui_widgets.cpp,\
imgui_demo.cpp,imgui.h,imgui_internal.h,imconfig.h,imstb_*.h,LICENSE.txt} external/imgui/
cp /tmp/imgui_vendor/backends/imgui_impl_vulkan.{h,cpp} external/imgui/backends/
# then update the pinned tag/SHA above.
```
