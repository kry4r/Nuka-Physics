# Vendored Dear ImGui — provenance record

Copied sources (not a submodule), built into the `imgui` static lib only when
`-DNK_BUILD_VULKAN_VALIDATION=ON`; the default core build never compiles them.

## Source

- **Upstream:** https://github.com/ocornut/imgui
- **Branch:** `docking` (docking/multi-viewport and the ≥1.92 texture lifecycle,
  `ImGuiBackendFlags_RendererHasTextures` / `ImTextureData`, used by `imgui_impl_vulkan`).
- **Pinned tag:** `v1.92.8-docking`
- **Commit SHA (tag deref `^{}`):** `b61e56346a92cfcaf1f43a545ca37b0b32239654`

Core and both backends are copied at the **same tag/SHA**; mixing core and backend
versions corrupts the font-atlas / texture lifecycle.

## What was copied

### Core (`./`)
`imgui.cpp`, `imgui_draw.cpp`, `imgui_tables.cpp`, `imgui_widgets.cpp`, `imgui_demo.cpp`
(headless `ImDrawData` smoke source and widget reference), `imgui.h`, `imgui_internal.h`,
`imconfig.h` (upstream defaults), `imstb_rectpack.h`, `imstb_textedit.h`,
`imstb_truetype.h` (stb rasterizer, no FreeType), `LICENSE.txt` (MIT).

### Backends (`backends/`)
| File | Role |
|---|---|
| `imgui_impl_vulkan.h` / `.cpp` | renderer backend |
| `imgui_impl_sdl3.h` / `.cpp` | platform backend on every desktop OS: mouse, keyboard, gamepad, clipboard, IME text-input area. SDL3 itself is fetched by CMake (`release-3.4.18`, static). |

### Fonts (`fonts/`)
| File | License | Role |
|---|---|---|
| `JetBrainsMono-Regular.ttf` | SIL OFL 1.1 | monospace stats / code |
| `JetBrainsMono-Bold.ttf` | SIL OFL 1.1 | monospace emphasis |
| `JetBrainsMono-OFL.txt` | — | license for both |

Inter (UI body/headings) and Noto Sans CJK SC (merged CJK glyphs) are downloaded at
configure time by `cmake/ViewerFonts.cmake` and packaged with the editor; font loading
lives in `src/render/imgui/nuka_imgui.cpp`.

## Not vendored
- Platform/renderer backends other than SDL3 and Vulkan.
- `misc/`, `examples/`, `docs/`, CI and editor config files.
- FreeType (`misc/freetype/`).

## Re-vendor recipe
```
git clone --depth 1 --branch <newtag-docking> https://github.com/ocornut/imgui /tmp/imgui_vendor
cp /tmp/imgui_vendor/{imgui.cpp,imgui_draw.cpp,imgui_tables.cpp,imgui_widgets.cpp,imgui_demo.cpp,imgui.h,imgui_internal.h,imconfig.h,imstb_*.h,LICENSE.txt} external/imgui/
cp /tmp/imgui_vendor/backends/imgui_impl_{vulkan,sdl3}.{h,cpp} external/imgui/backends/
# then update the pinned tag/SHA above; bump the SDL3 GIT_TAG with any backend API change.
```
