# M8.5 Recon Brief — Realtime on-screen Vulkan viewport + Dear ImGui UI + BEAUTY (美观)

**Status:** synthesis of 5 parallel readers (m8-render-stack, swapchain-windowing, imgui-integration,
beauty-path, arch-directives-verification) + live re-verification of the disputed headless claims.
**Date:** 2026-06-13. **Branch baseline:** master @ 7b09a2e (M7 closed; M8 render stack committed; M8 in progress).
**Milestone position:** owner-inserted, **between M8 and M9** — the early realization of the M11 viewer
(`src/runtime/app/viewer/{viewer_main,imgui_layer,camera_controller}`, plan L507-509) and the long-dormant
v0.8 "U1 viewport" track (scoped, never started).

> **CONTROLLER: read §1 (exit gate), §7 (decisions needed BEFORE implementation), and the
> VK_EXT_headless_surface verdict in §1.3 first.** Two of the five readers disagreed on headless verifiability;
> I re-ran the probes — the verdict below is the live-verified one.

---

## 1. Scope + Exit Gate

### 1.1 Scope

M8.5 delivers a **realtime, on-screen Vulkan raster viewport with a Dear ImGui UI shell, that looks good**.
Concretely:

1. A **present-mode** Vulkan path (surface → swapchain → acquire → draw → present + per-frame sync), built
   alongside the existing **offscreen** raster renderer (which must stay pristine for the M8 G2 determinism gate).
2. A **windowing/surface** layer (the surface source for the swapchain).
3. **Dear ImGui** vendored + `imgui_impl_vulkan` glue + a physics-showcase UI (viewport image, play/pause/step/reset/speed,
   stats, scene tree, camera, drive sliders, optional material/light inspector).
4. A **Viewer** that drives `Simulation::Frame()` at interactive rate, routes window/mouse/keyboard into the
   existing `CommandQueue` (`MoveEntity` seam), and presents each frame.
5. **BEAUTY**: complete the T3a→T3b PBR shader (Cook-Torrance GGX + lights + tonemap), a default light rig, and
   the **visual-mesh cook** so robots render as real geometry instead of placeholder boxes.

**Hard scope cap (§4):** this is a *thin showcase shell over the existing Simulation+RenderWorld+raster pipeline*,
**not** a game engine, **not** an editor, **not** an auto-barrier render graph (that is M11-tier). Host-download
pose publishing is fine for M8.5; CUDA↔Vulkan interop stays M11.

### 1.2 The "本机不可验证" constraint — what is run-verified vs build-only

This is a **headless server**: no live `DISPLAY`/`WAYLAND_DISPLAY`, no `/tmp/.X11-unix` live socket, no real
monitor (verified live: `DISPLAY=<unset>`). Therefore:

**GENUINELY UNVERIFIABLE here (build/link-only + reviewed-by-correctness):**
- An actual on-screen window presenting to a physical monitor.
- Real vsync / frame-pacing on a display; human eyeball of "美观" (beauty).
- GLFW/SDL physical-window creation (needs a live display server).

**RUN-VERIFIABLE here (this is the important, live-confirmed part):**
- The **full swapchain + present loop IS runnable headlessly via `Xvfb` + `VK_KHR_xcb_surface`** on the
  lavapipe ICD. Proven live: `VK_ICD_FILENAMES=lvp_icd.json xvfb-run -a -s '-screen 0 320x240x24' vkcube --c 3`
  → **exit 0** (vkcube uses a real `VK_KHR_xcb_surface` swapchain + `vkQueuePresentKHR`). Only the standard
  `lavapipe is not a conformant vulkan implementation` warning prints.
- The **offscreen raster path** (same shaders/pipeline/material/light code) → a lit PPM, byte-determinism (D1).
- The **ImGui `ImDrawData` overlay** rendered into an offscreen image (no platform/window backend needed for
  `imgui_impl_vulkan` — just set `io.DisplaySize`+`DeltaTime`, `NewFrame/Render/GetDrawData`,
  `ImGui_ImplVulkan_RenderDrawData(...)` between BeginRenderPass and EndRenderPass).
- Compile + link of the windowed viewer TU (against the built archives, recompiled with the gate flag).

### 1.3 VK_EXT_headless_surface VERDICT — **UNAVAILABLE on this box** (live re-verified)

The two readers disagreed; **the swapchain-windowing reader is correct, the arch reader is wrong.** Re-verified live:

- The lavapipe ICD `.so` (`libvulkan_lvp.so`) **does** contain the string `VK_EXT_headless_surface` (this is what
  the arch reader saw) — BUT that is irrelevant, because **`VK_EXT_headless_surface` is a LOADER-implemented
  instance extension**, and this box's **loader is `libvulkan.so.1.2.131` (Feb 2020)**.
- `nm -D /usr/lib/x86_64-linux-gnu/libvulkan.so.1` → **does NOT export `vkCreateHeadlessSurfaceEXT`** ("NOT exported by loader").
- The loader binary even carries the string `"VK_EXT_headless_surface extension not enabled. vkCreateHeadlessSurfaceEXT not executed!"`.
- `vulkaninfo` instance-extension enumeration on lavapipe → `VK_KHR_surface`, `VK_KHR_xcb_surface`, `VK_KHR_xlib_surface`
  present; **`VK_EXT_headless_surface` ABSENT**. `VK_KHR_swapchain` rev 70 is present (device).

**Consequence:** the verification substitute is **Xvfb + `VK_KHR_xcb_surface`**, NOT `VK_EXT_headless_surface`.
Code may *reference* headless-surface symbols (the headers `VK_HEADER_VERSION 131` define them, so it compiles),
but it will return `VK_ERROR_EXTENSION_NOT_PRESENT (-7)` at runtime here. **Make the surface backend
runtime-selectable** (try headless-surface → fall back to xcb under Xvfb) so the same binary verifies on both a
modern-loader CI box and this one.

### 1.4 PROPOSED EXIT GATE (exact pass criteria)

The gate has three tiers; **all three must pass** for M8.5 to close. Everything is gated behind
`NK_BUILD_VULKAN_VALIDATION` (OFF in build-cuda128, `CMakeLists.txt:30`), mirroring the established
render-smoke + recorder gating idioms. The gated TU is validated by **recompiling it with
`-DNK_BUILD_VULKAN_VALIDATION` and linking the built archives** (never reconfigure the main build).

**GATE-A — BUILD-GATE (build+link only; covers the genuinely-unverifiable on-screen path):**
- `nuka_viewer` lib + `viewer_main` exe (windowed: surface → swapchain → present + ImGui) **compile and link**
  cleanly behind the gate flag. Default Vulkan-less / glslc-less `build-cuda128` config still builds libnuka and
  the core (viewer is an OPT-IN target, like the Recorder D5 gating, `src/CMakeLists.txt:1051-1059`).
- Lint clean: `src/render/**` and the viewer TUs stay **zero-CUDA-token** (`tools/lint/banned_patterns.yaml:32-57`);
  ImGui/GLFW vendored as **SYSTEM includes** to suppress warnings + lint.

**GATE-B — RUN-VERIFIED offscreen viewer-frame smoke (`tests/scenario/viewer_frame_smoke.cpp`, NEW):**
- Build a `RenderWorld` (synthetic 2-box like `render_raster_smoke.cpp`, **plus** a real `.nka` MESH instance +
  a metallic/rough material + the default light rig — to exercise the BEAUTY path).
- Render twice through the **offscreen** path. Assert: `non_background_pixel_count > 0`; the two renders are
  **byte-identical** (`memcmp == 0`, D1); dump a PPM for human spot-check off-box.
- Build ImGui `ImDrawData` with a **fixed UI state** (no time-seeded animation), `ImGui_ImplVulkan_RenderDrawData`
  into the same offscreen pass. Assert: `ImDrawData.CmdListsCount > 0`; composited image `non_bg > 0`; two
  composited renders **byte-identical** (fixed UI state → deterministic).
- `GTEST_SKIP` if the renderer ctor throws (no Vulkan device) — same shape as `render_raster_smoke.cpp`.

**GATE-C — RUN-VERIFIED headless present-loop smoke (`tests/scenario/viewport_present_smoke.cpp`, NEW; run via `xvfb-run`):**
- Create a **`VK_KHR_xcb_surface`** under Xvfb → `VkSwapchainKHR` → `vkAcquireNextImageKHR` → draw the RenderWorld
  into the acquired swapchain image → `vkQueuePresentKHR`, with per-frame semaphores + in-flight fence, for N
  frames (e.g. 3). Assert the loop completes (no `VK_ERROR`), `swapchain image count >= 2`, present succeeds N
  times. Optionally read back one presented image and assert `non_bg > 0`.
- Backend is **runtime-selected**: prefer `VK_EXT_headless_surface` if the loader exports it (future CI), else
  `VK_KHR_xcb_surface` under Xvfb (this box). Skip cleanly if neither surface ext is available.
- This is the **proof the present path is real**, not just compiled (vkcube already proves the substrate works).

**EXPLICITLY NOT in the gate (cannot be verified here; deferred to a display machine / off-box review):**
the literal monitor scan-out, vsync pacing, ImGui interaction feel, and the subjective beauty judgement.
Document a **manual on-display checklist** for a machine with a display (the plan's M11 gate already accepts
"有显示器机器交互运行" — verified-elsewhere — for the windowed viewer).

**Determinism red-line (G2):** the realtime present loop is **inherently non-D1** (vsync, frames-in-flight,
acquire ordering, frame-pacing interpolation). It MUST NOT be wired into the D1/G2 gated **offscreen** `Render()`
path or the `render_physics_parity` memcmp. Keep the offscreen deterministic renderer as the correctness oracle;
isolate all non-determinism to the viewer-only present path.

---

## 2. File Manifest (dependency / build order)

Marked: **REUSE** (M8, unchanged) · **EXTEND** (M8, modify) · **NEW** (create) · **VENDOR** (third-party).
Build order is top-to-bottom: data model → present path → ImGui → viewer → beauty → gate.

### Phase 0 — Reused M8 substrate (no change)
| Path | Role | Status |
|---|---|---|
| `src/render/render_world.hpp` / `.cpp` | Backend-agnostic scene model (instances/MeshLibrary/materials/cameras/lights). Viewer-ready as-is (no Vulkan/CUDA leak). `BuildRenderWorld` builds once. | **REUSE** |
| `src/render/vulkan_offscreen_types.hpp` | POD report/pixel types (`VulkanOffscreenReport`, `VulkanRgba8`). | **REUSE** |
| `src/runtime/app/pose_publisher.{hpp,cpp}` | `PosePublisher` iface + `HostDownloadPublisher` (alloc-free hot path). Interop = M11. | **REUSE** |
| `src/runtime/app/command_queue.hpp` | MPSC queue; **`MoveEntity` already present** (`:35,:50-67`) — the viewer interaction seam. | **REUSE** |

### Phase 1 — Present-mode renderer (the chief net-new Vulkan lift)
| Path | Role / key symbols | Status |
|---|---|---|
| `src/render/raster/vulkan_raster_renderer.{hpp,cpp}` | Init (instance/device/queue/cmdpool/renderpass/pipeline/shaders) is reused **in pattern**; `CreateInstance` (`:275-288`, apiVer 1.1, zero ext), `CreateDevice` (`:324-340`, zero ext), `SelectGraphicsDevice` (`:290-322`, graphics-queue only) must gain surface/swapchain ext + present-queue support. Add a **narrow accessor seam** exposing `VkDevice/Queue/RenderPass/DescriptorPool` for ImGui `InitInfo` (currently private pimpl). Do NOT perturb the offscreen egress. | **EXTEND** |
| `src/render/raster/vulkan_present_renderer.{hpp,cpp}` (or `vulkan_swapchain_present.*`) | **NEW**: present sibling. Swapchain target (finalLayout `PRESENT_SRC_KHR`), `vkAcquireNextImageKHR` → draw → `vkQueuePresentKHR`, per-frame `imageAvailable`/`renderFinished` semaphores + in-flight fence, frames-in-flight. Persistent swapchain framebuffers + cached static vertex/index store (only push-constants change per frame). Shares the persistent device/renderpass/pipeline via strategy split. Key symbols: `PresentRenderer`, `SurfaceBackend`. | **NEW** |
| `src/render/window/` (xcb backend) | **NEW**: thin `WindowSurface` over xcb/xlib (`/usr/include/xcb/xcb.h`, `Xlib.h` **present** — buildable now, runs under Xvfb). Runtime-selectable backend: headless-surface → xcb. Key symbols: `XcbWindowSurface`, `MakeSurface`. | **NEW** |

### Phase 2 — Dear ImGui vendor + glue
| Path | Role | Status |
|---|---|---|
| `external/imgui/` | **VENDOR** Dear ImGui (pinned, **same tag** for core + backend, ≥ v1.92 → needs `ImGuiBackendFlags_RendererHasTextures` + `ImTextureData` flow): `imgui.cpp/_draw/_tables/_widgets/imconfig.h` + `backends/imgui_impl_vulkan.{h,cpp}` + a platform backend (`imgui_impl_glfw` if GLFW chosen, else a hand-written xcb backend). Mirrors `external/vhacd` (copied source). | **VENDOR** |
| `external/glfw/` | **OPTIONAL VENDOR** (windowed path only) — GLFW absent system-wide (no `/usr/include/GLFW/glfw3.h`, live-confirmed). Needed only if owner mandates GLFW over the thin xcb backend. | **VENDOR (cond.)** |
| `src/render/imgui/` | **NEW**: nuka-side ImGui glue — `nuka_imgui` STATIC lib (mirrors `nuka_render`, `src/CMakeLists.txt:132-198`, SYSTEM includes + links `Vulkan::Vulkan`), `ImGui_ImplVulkan_InitInfo` wiring (Instance/PhysDevice/Device/QueueFamily/Queue/DescriptorPool/MinImageCount/RenderPass+Subpass/MSAA=1/CheckFn), `ImGuiStyle` theme + vendored TTF font for beauty. | **NEW** |

### Phase 3 — The Viewer + UI panels (the deliverable)
| Path | Role | Status |
|---|---|---|
| `src/runtime/app/viewer/viewer_main.cpp` | **NEW**: windowed entry — window + surface + swapchain present loop driving `Simulation::Frame()` at interactive rate. Gated, opt-in exe. | **NEW** |
| `src/runtime/app/viewer/imgui_layer.cpp` | **NEW**: UI panels (viewport image, play/pause/step/reset/speed, stats step-time/fps/dof/contacts, scene tree, camera, drive sliders, optional material/light inspector) → `ImDrawData` via `imgui_impl_vulkan`. | **NEW** |
| `src/runtime/app/viewer/camera_controller.cpp` | **NEW**: interactive orbit/pan/zoom → writes `RasterOptions` camera override (`use_camera_override` + eye/target, `vulkan_raster_renderer.hpp:52-66`; `ResolveCamera` honors it, `.cpp:645-686`). | **NEW** |
| `src/runtime/app/simulation.{hpp,cpp}` | `Frame()` (`simulation.cpp:21-49`, Input→Sim→[gated]TransformSync→Render) is driven per-vsync; `EnableRendering` injects renderer; `LatestReport`/`SetEnvIndex` exposed. A **present branch** is added alongside `RenderSystem`. | **EXTEND** |
| `src/runtime/app/systems.{hpp,cpp}` | `RenderSystem::Run` (`systems.cpp:32-52`) is the single renderer call site; add a parallel present-render path / renderer pointer. `InputSystem` (`systems.hpp:48-57`) drains `CommandQueue` — fill in real window/mouse dispatch (or defer to M11 — see §7). | **EXTEND** |

### Phase 4 — BEAUTY (PBR + visual-mesh cook + lighting/tonemap)
| Path | Role | Status |
|---|---|---|
| `src/render/raster/shaders/mesh_pbr.frag` | Replace Lambert body (`:25-35`) with Cook-Torrance GGX (D_GGX + G_Smith + F_Schlick) + light loop over a light UBO + emissive add + ACES tonemap (Narkowicz) + linear→sRGB. T3b seam documented `:10-16`. | **EXTEND** |
| `src/render/raster/shaders/mesh.vert` | Pass UV + world-space tangent (normal mapping); receive full material. Currently push-block `{mvp,model,base_color}` (`:18-22`). | **EXTEND** |
| `src/render/raster/vulkan_raster_renderer.cpp` | Extend `PushBlock` (`:166-170`) + `InstanceDraw` (`:625-632`) to carry metallic/roughness/emissive/opacity; resolve from `RenderMaterial` in draw loop (`:730-736`); ADD `VkDescriptorSetLayout`+pool+per-frame light UBO (pipeline currently has **ZERO descriptor sets**, push-constant only, `:408-417`); widen push range to FRAGMENT; inject default light rig when `world.lights.empty()`. | **EXTEND** |
| `src/scene/format/nks.cpp` | **THE COOK GAP.** Add `MeshSink::AddVisualMesh` (`EncodeMesh`→`NkaTagMesh` chunk, deduped like `AddCollisionMesh :258-293`); in `SaveShape` route visual-only (contype==0) shapes to a **MESH** chunk + MESH-fourcc `AssetRef` instead of CMSH (`:323`); on load decode MESH + attach AssetRef. | **EXTEND** |
| `src/scene/scene_ir.cpp` | `ProjectShape` (`:634-641`, comment "left empty here"/"binds .nka refs") must populate `VisualMeshComponent.mesh` with the MESH AssetRef — the wire that makes `render_world.cpp:328`'s `NkaTagMesh()` branch fire. | **EXTEND** |
| `src/scene/asset/nka.cpp` | `EncodeMesh`/`DecodeMesh` (`:79/:110`, fourcc `MESH`) already exist — the cook calls them. | **REUSE** |
| `src/import/mjcf_importer.cpp` | Already loads STL/OBJ visual triangles into `shape.mesh_vertices` for TriMesh incl. contype==0 (`:536-562`) + parses `<light>` (`:683-693`). Reuse; triangles just need correct MESH routing downstream. | **REUSE** |
| `examples/scenes/h1_with_hand.nks` (or `go2.nks`) | **NEW** cooked scene WITH MESH chunks so a robot renders as real geometry. Current `h1_cup_table.nks` (40KB, 2 collision_shape + 1 SAMP, **no robot visual meshes**). | **NEW** |

### Phase 5 — Gate + docs
| Path | Role | Status |
|---|---|---|
| `tests/scenario/viewer_frame_smoke.cpp` | **NEW** GATE-B (offscreen RenderWorld + ImGui overlay → image, non-bg>0, D1, ImDrawData non-empty). Mirrors `render_raster_smoke.cpp`. | **NEW** |
| `tests/scenario/viewport_present_smoke.cpp` | **NEW** GATE-C (xcb surface under Xvfb → swapchain → acquire/present N frames). Run via `xvfb-run`. | **NEW** |
| `CMakeLists.txt` / `src/CMakeLists.txt` | Add `nuka_imgui` STATIC + `nuka_viewer`/`viewer_main` + swapchain/window/imgui sources behind the gate; `find_package(X11)` or vendored GLFW/ImGui targets; glslc custom_commands already wired (`src/CMakeLists.txt:154-187`). | **EXTEND** |
| `docs/architecture/headless-rendering.md` | Annotate/supersede: it asserts "no swapchain/surface/X11/glfw/DISPLAY" for M8 — record the M8.5 present path + the **Xvfb+xcb (NOT VK_EXT_headless_surface, unavailable on the 1.2.131 loader)** verification story. | **EXTEND (doc)** |

---

## 3. What EXISTS (reuse) vs NEW — with citations

**EXISTS and directly reusable (the big win — the data model + frame loop + interaction seam are done):**
- `RenderWorld` is already viewer-ready and backend-agnostic (no Vulkan/CUDA leak) — feeds raster, the M11
  path-tracer, AND the M8.5 viewer with no rework (`render_world.hpp`; `BuildRenderWorld` `render_world.cpp:253-445`).
- The frame loop is driveable as-is: `Simulation::Frame()` runs Input→Sim→[gated]TransformSync→Render
  (`simulation.cpp:21-49`); rendered image in `LatestReport()` (`simulation.hpp:89`).
- The **interaction seam already exists**: `CommandQueue` carries `MoveEntity` (`command_queue.hpp:35,50-67`,
  comment "anticipating the M11 viewer's drag-to-move").
- Camera override is viewer-ready (`RasterOptions` eye/target/up/fov, `vulkan_raster_renderer.hpp:52-66`;
  `ResolveCamera` `.cpp:645-686`; Vulkan-clip `LookAt/Perspective` `.cpp:139-164`).
- The render-pass + graphics pipeline + shaders are present-agnostic and transplant verbatim (R8G8B8A8 color +
  D32 depth `:351-387`, dynamic viewport/scissor `:461-499`, push-constant MVP `:166-170`, no MSAA).
- **All PBR DATA already in place:** `RenderMaterial` has base_color/metallic/roughness/emissive[3]/opacity +
  3 texture-path fields (`components.hpp:58-69`); `RenderLight` gathered from `LightComponent`
  (`render_world.cpp:425-442`). Only shader+descriptors are missing.
- **The visual-mesh CONSUMER is fully wired:** `render_world.cpp:328-336` already does
  `if (!vis.mesh.Empty() && vis.mesh.fourcc==NkaTagMesh()) → DecodeMesh → InternNkaMesh` (live-verified). It
  just never fires because no producer sets `vis.mesh`.
- `EncodeMesh`/`DecodeMesh` (MESH chunk) exist (`nka.cpp:79/110`). MJCF importer loads STL/OBJ visual triangles
  (`mjcf_importer.cpp:536-562`). Self-written `LoadStl`/`LoadObj` (`mesh_file_loader.hpp:46-59`) — no assimp/tinyobj.
- Gating + smoke idioms: `NK_BUILD_VULKAN_VALIDATION` (`CMakeLists.txt:30`, OFF); `render_raster_smoke.cpp` /
  `render_frame_loop_smoke.cpp` / `render_physics_parity.cpp` (synthetic RenderWorld → render → non_bg>0 + memcmp
  D1 + PPM + SKIP-if-no-device); Recorder D5 gating (`recorder.cpp` → `NOT_SUPPORTED` when flag off,
  `src/CMakeLists.txt:1051-1059`).

**NEW (must be created or vendored):**
- The entire **present path** (swapchain/surface/semaphores/fences/present) — tree-wide grep finds **ZERO**
  `VkSwapchainKHR`/`VkSurfaceKHR`/`vkQueuePresentKHR`/`vkAcquireNextImageKHR` in any `src/**` file.
- **Dear ImGui** — not vendored (`external/` holds only vhacd + sha256; imgui hits only `external/vhacd/VHACD.h`
  false-positive).
- A **windowing/surface backend** — no GLFW (absent system-wide), SDL2 runtime-only (no dev headers); xcb/Xlib
  dev headers ARE present (`/usr/include/xcb/xcb.h`, `X11/Xlib.h` — live-verified).
- The **MESH-chunk producer** in `nks.cpp` + `ProjectShape` wiring (the cook gap) + a **richer robot scene cook**.
- Full **T3b PBR** shader + descriptor sets + light UBO; a **default light rig**; **ACES tonemap + sRGB**.
- The **Viewer** (viewer_main/imgui_layer/camera_controller) + the three gate tests + the doc annotation.

---

## 4. Architectural Constraints (binding — do not violate)

- **PHYSICS-PRIMARY, not a game engine** (`renderer-architecture-directive.md` L17; plan §1 L35): "rendering is
  only ONE PART … it exists to SHOWCASE the physics … Do NOT turn Nuka into a game engine." M8.5 = a **thin
  imgui viewer over the existing Simulation+RenderWorld+raster pipeline**. No editor, no audio/gameplay/networking,
  no AAA subsystems. The strongest binding constraint; "美观/beautiful" is **not** license to build AnKi-class systems.
- **Borrow ARCHITECTURE patterns, not subsystems:** Mangifera (owner's own proto-design = the skeleton),
  AnKi (render-graph — the *hard* parts, NOT for M8.5), **Filament** (renderer-not-engine north star — the
  relevant borrow), MuJoCo (physics→render decouple = "thin viewer over an immutable extract").
- **The bigger render-graph/RHI arch is M11-tier / POST-v0.7** — do NOT pull a full auto-barrier render graph
  into M8.5. M8 needs only a forward PBR pass + (now) a swapchain + ImGui; **add those, not a graph.**
- **Host-download pose is fine for M8.5; interop = M11.** `HostDownloadPublisher` (single env, ~51-DOF H1,
  interactive fps) is trivially adequate; the M11 `CudaVulkanInteropPublisher` drops in behind the same
  `PosePublisher` interface with no edit (`pose_publisher.hpp:22-25,48-58`). Reuse host-download unchanged.
- **G2 determinism where it applies:** the offscreen `Render()` is pinned byte-identical (no MSAA
  `VK_SAMPLE_COUNT_1_BIT` `:354,363,476,575`; LESS_OR_EQUAL depth; single submit + `vkQueueWaitIdle` `:893`).
  Keep that path **pristine** as the correctness oracle. The realtime present loop is legitimately non-D1 (display
  time) — isolate it. Any AA must be **supersample + deterministic downsample**, never hardware MSAA, or it
  breaks the `render_physics_parity` memcmp.
- **Zero-CUDA red-line:** `src/render/**` and viewer TUs stay zero-CUDA-token (`banned_patterns.yaml:32-57`);
  ImGui/GLFW are pure host C++/Vulkan → compliant; vendor as **SYSTEM includes**.
- **Land on the M8 seams, not legacy:** M9 deletes `render_scene.*` + `scene_pipeline.*` and repoints
  c_abi→`nk::World`. M8.5 must build entirely on the M8 **RenderWorld/Simulation/nk::World** path (NOT legacy
  `RenderScene`) to avoid M9 tear-out. The demoted compute-only `vulkan_renderer.cpp` (debug overlay, no graphics
  pipeline) is **not** a present base — at most a future toggleable debug overlay.

---

## 5. BEAUTY Plan

Three sub-areas; the data plumbing for all three is already in M8 — the gaps are **producers/wiring + shader**,
not architecture. **Recommended order: (5.1) PBR shader+descriptors first (small/medium, fully offscreen-verifiable),
(5.3) lighting/tonemap (small), then (5.2) the visual-mesh cook (the one real sub-project).**

### 5.1 Full PBR shading (T3a→T3b) — small-to-medium
- **Data exists** (`RenderMaterial` all fields, `RenderLight` gathered). **Work:** extend `PushBlock`+`InstanceDraw`
  for the per-draw material scalars + widen push to FRAGMENT; add the **first descriptor set** (the pipeline has
  none today) + a per-frame **light UBO**; replace the Lambert body with Cook-Torrance GGX + light loop + emissive
  + **ACES tonemap (Narkowicz)** + linear→sRGB. glslc auto-recompiles SPIR-V on `.glsl` edit (`/root/miniconda3/bin/glslc`).
- **Fully offscreen-verifiable** (same shaders feed offscreen + present) → covered by GATE-B's lit PPM + D1.
- **Textures: DEFER.** `RenderMaterial` names 3 texture paths but **no importer populates them and no TEXB chunk
  is cooked**. Full textured PBR needs a texture-cook pass + image decoder + combined-image-sampler descriptors
  that don't exist. **Scope M8.5 to analytic PBR (metallic/roughness/emissive *scalars*)** — robot MJCF materials
  are largely flat-colored; textures → M11/later.

### 5.2 Visual-mesh cook — **honest call: a genuine SUB-PROJECT, not a one-line wire**
- **Why it's not trivial:** the consumer is wired, but **two producer gaps** must both be fixed: (1) `nks.cpp`
  routes ALL shape triangles (incl. contype==0 visual-only) to **CMSH** (collision) chunks (`:323`), with no
  `AddVisualMesh`; (2) `ProjectShape` leaves `VisualMeshComponent.mesh` **empty** (`scene_ir.cpp:634-641`). Today
  every instance falls back to a 5cm box placeholder (`render_world.cpp:347-349`). **Both** must land for robots
  to render.
- **Plus a richer cook:** the only example `.nks` is the minimal cup+table (no robot meshes). A new
  import→cook of `h1_with_hand` (49 STL) or `go2` (16 meshes) into `.nks/.nka` **with MESH chunks + per-part
  materials** is real work and inflates `.nka` size. Assets present: 70 STL under unitree_h1, 49 STL/OBJ under
  newton_assets h1_with_hand, 16 under go2.
- **Recommendation:** treat as a dedicated task (T-COOK). Start with **go2** (cheaper, 16 meshes) as the first
  proof, then h1 if budget allows (h1 is the owner's demo hero — see §7 decision).

### 5.3 Lighting / environment — small
- **No default light rig today** (`BuildRenderWorld` only emits authored `LightComponent`s; cooked robot scenes
  have none → `world.lights` empty). Add a sensible analytic **3-point key/fill/rim** default **only when
  `world.lights.empty()`** (must NOT override authored MJCF `<light>` which DO import, `mjcf_importer.cpp:683-693`).
- Add a **dark ground plane** + hemispheric ambient for a grounded look. **Recommend the default rig live
  renderer-side** (raster cpp), NOT baked into `BuildRenderWorld`, so the M11 path-tracer (shares RenderWorld) is
  unaffected and RenderWorld stays a pure data product.
- **DEFER to M11:** shadow-maps, real global illumination, and beauty-tier RT. M8.5 = cheap analytic grounded
  look only.

---

## 6. Decomposition Proposal (T1..Tn)

Execution policy is **ultracode-first (parallel)**, BUT the Vulkan/ImGui present spine is build-coupled (shared
device/renderpass/pipeline, ABI surface, vendored deps) → a genuine **"can't cleanly parallelize"** case → those
tasks are **single-flight (COUPLED-SEQUENTIAL)**. The BEAUTY shader+cook tracks are more independent.

| Task | What | Mode | Rationale / depends-on |
|---|---|---|---|
| **T1 — ImGui+windowing vendor + CMake** | Vendor Dear ImGui (+ optional GLFW) under `external/`; `nuka_imgui` STATIC; `NK_BUILD_VIEWER`/reuse `NK_BUILD_VULKAN_VALIDATION` gating; X11/xcb find_package. | **COUPLED-SEQUENTIAL** | Build-config foundation; everything links against it. Must land first + clean. Decisions D1/D2/D3 gate this. |
| **T2 — Present renderer + surface backend** | `vulkan_present_renderer.*` + `src/render/window/` xcb backend; extend instance/device/queue-select for surface+swapchain+present; runtime-selectable backend (headless-surface→xcb); persistent framebuffers + cached geometry; per-frame sync. | **COUPLED-SEQUENTIAL** | Touches the shared persistent device/renderpass/pipeline + extends the M8 renderer's private Impl. Single-flight to keep the offscreen G2 path pristine. Depends T1 (Vulkan find_package unchanged, but co-edits CMake). |
| **T3 — Viewer + ImGui glue + UI panels** | `viewer/{viewer_main,imgui_layer,camera_controller}`; `nuka_imgui` `InitInfo` wiring; UI panels; drive `Simulation::Frame()`; route window input → `CommandQueue`. | **COUPLED-SEQUENTIAL** | Depends T1 (imgui) + T2 (present). Single call-site edits to `simulation/systems`. |
| **T4 — PBR shader + descriptors + light UBO** | `mesh_pbr.frag` GGX+tonemap+sRGB; `mesh.vert` UV/tangent; `PushBlock`/`InstanceDraw` extend; first descriptor set + light UBO; default light rig (renderer-side, empty-only). | **PARALLELIZABLE** | Self-contained in the shared raster cpp + shaders; offscreen-verifiable independent of the present spine. Light coordination with T2 on the renderer cpp (flag for merge). |
| **T5 — Visual-mesh cook (SUB-PROJECT)** | `nks.cpp` `AddVisualMesh` + SaveShape MESH routing + load-side decode; `scene_ir.cpp` `ProjectShape` populates `VisualMeshComponent.mesh`; new robot scene cook (go2 first, then h1). | **PARALLELIZABLE** | Entirely in scene/format/import — **zero overlap** with the Vulkan spine; the renderer consumer is already wired (`render_world.cpp:328`). Largest scene-side task; can run fully in parallel with T1-T4. |
| **T6 — Gate (B+C) + doc** | `viewer_frame_smoke.cpp` (offscreen+ImGui D1), `viewport_present_smoke.cpp` (xcb-under-Xvfb present loop), annotate `headless-rendering.md`. | **COUPLED-SEQUENTIAL** | Depends T2/T3/T4/T5 (exercises all). Final integration. |

**Parallelization summary:** T4 and T5 run in parallel (ultracode) with the T1→T2→T3 spine. T1→T2→T3 are
single-flight (build-coupled Vulkan/ImGui). T6 closes after all. Net: ~2 parallel lanes (spine + beauty),
T5 the long pole on the beauty lane, T6 the join.

---

## 7. Risks, Open Questions, and DECISIONS needed BEFORE implementation

### Controller DECISIONS required before T1 (these gate the spine):

1. **Windowing library — GLFW vs thin self-written xcb (RECOMMEND: xcb).** GLFW is absent system-wide (must
   vendor) and still needs Xvfb to run here; a **thin xcb backend uses already-installed `libxcb`/`libX11` dev
   headers, is lighter, runs under Xvfb today, and matches the repo's self-written/no-vendor ethos** (self-written
   usdc reader, no OpenUSD). GLFW only if owner mandates portability/GLFW specifically. *(SDL2 is runtime-only,
   no dev headers — not viable.)*
2. **ImGui vendoring mechanism (RECOMMEND: copied pinned sources under `external/imgui`).** Tree precedent is
   `external/vhacd` (copied source). Pin core + `imgui_impl_vulkan` from the **same ≥v1.92 tag** (mixing corrupts
   the font-atlas/texture lifecycle). FetchContent is the alternative. Needs a license/git-lfs ruling.
3. **Gating flag — new `NK_BUILD_VIEWER` vs reuse `NK_BUILD_VULKAN_VALIDATION` (RECOMMEND: reuse).** One flag
   turns on all run-unverifiable-here render gates; viewer code stubbed otherwise (keeps default
   Vulkan-less/glslc-less `build-cuda128` building libnuka).

### Controller DECISIONS required before T4/T5 (these gate beauty scope):

4. **How much beauty in M8.5 vs M11 (RECOMMEND: analytic PBR + default rig + tonemap in M8.5; textures + shadows +
   RT in M11).** Confirm M8.5 = scalar PBR (metallic/roughness/emissive) + 3-point light + ACES + grounded look;
   **no** texture sampling (no TEXB cook / image decoder exists), **no** shadow-maps/GI.
5. **Does the visual-mesh cook land in M8.5? (RECOMMEND: YES — it's required for "美观/robots render real
   geometry"; without it every instance is a placeholder box.)** It is a genuine sub-project (T5), not a wire.
   If deferred, M8.5 beauty is limited to the cup/table + primitives.
6. **Which robot is the showcase — go2 (16 meshes, cheap first proof) or h1_with_hand (49 STL, owner's demo
   hero)?** Decides T5 size. RECOMMEND go2 first as the proof, h1 if budget allows.

### Other open questions (lower-stakes, can be decided during implementation):

7. **Present egress — render directly into the swapchain image (best, zero-copy) vs reuse offscreen `Render()`
   then blit `report.pixels` to the swapchain (simpler, keeps the gated path untouched, adds a host roundtrip)?**
   Recommend direct-to-swapchain for a smooth viewport; the offscreen path stays the D1 oracle regardless.
8. **InputSystem dispatch — fill in real window→Command dispatch in M8.5 or defer to M11?** The `MoveEntity` seam
   exists; a minimal play/pause/camera dispatch in M8.5 is cheap, full drag-to-move can defer.
9. **Frame-pacing interpolation (pos-lerp/quat-slerp between snapshots) — M8.5 or defer?** Legitimately non-D1
   (the principled reason the viewport is non-deterministic) but needs a double-buffered snapshot
   `HostDownloadPublisher` doesn't keep today. Recommend ship without interpolation, add later.
10. **Wire the demoted compute debug-overlay (wireframe/AABB/contacts) into the viewport as a toggle?** Defer —
    not required for M8.5; it's offscreen-readback-only today.

### Top technical risks (for review rigor, since the on-screen path is run-unverifiable here):

- **Over-scope into a game engine / render graph** — the strongest constraint (§4). An agent may read "美观" as
  license for AAA subsystems. Keep it a thin viewer.
- **The offscreen renderer is structurally offscreen at THREE coupled points** (renderpass finalLayout
  `TRANSFER_SRC`, owned VkImage target, copy-to-buffer+map readback) + blocking `vkQueueWaitIdle`. A naive "add a
  swapchain" edit risks breaking G2. **Strategy-split / sibling class**, do not mutate the offscreen path.
- **`SelectGraphicsDevice` picks the first graphics queue with NO present check** (`:290-322`) — a real bug seam
  if copied; must add `vkGetPhysicalDeviceSurfaceSupportKHR`.
- **lavapipe is a CPU software rasterizer on an old (1.2.131) loader** — surface caps/format/present-mode differ
  from real hardware. Keep present negotiation **defensive** (query `SurfaceFormatsKHR`/`PresentModesKHR`, don't
  hardcode). The present path validated only on lavapipe+Xvfb may hit different caps on a real GPU.
- **Two NEW third-party deps (ImGui + windowing) into a tree that has none** — supply-chain + build-config
  addition; both gated behind the headless-unverifiable path → higher review bar.
- **D1 vs ImGui:** font-atlas state / time animations / hovered-widget state break two-render memcmp. The gate
  image must pin a **fixed UI state, no time-seeded animation**; the realtime viewport itself is non-D1 and that's fine.
- **Beauty is run-unverifiable here** — validated by the offscreen lit PPM + compile/link + the Xvfb present
  smoke, **not** by eyeballing. Document a manual on-display checklist for a machine with a display.

---

## 8. Controller Decisions (RESOLVED 2026-06-13 — recon recommendations + owner directives)

- **D1 windowing = thin self-written xcb** (NOT GLFW; xcb/Xlib dev headers present, runs under Xvfb, matches the self-written ethos).
- **D2 ImGui = vendored COPIED sources under `external/imgui`** (mirror `external/vhacd`); pin core + `imgui_impl_vulkan` at the SAME recent stable tag (v1.92.x docking — needs the ≥1.92 texture lifecycle).
- **D3 gating = reuse `NK_BUILD_VULKAN_VALIDATION`** (default `build-cuda128` unaffected; viewer/imgui stubbed when OFF).
- **D4 beauty scope = analytic SCALAR PBR (metallic/roughness/emissive) + 3-point default light rig + ACES tonemap + sRGB + grounded look** in M8.5; textures/shadow-maps/RT → M11.
- **D5 visual-mesh cook = YES in M8.5** (T5 sub-project; required so robots render real geometry, not placeholder boxes).
- **D6 showcase robot = go2 FIRST (16 meshes, cheap pipeline proof) → then h1_with_hand (the demo hero) if it lands cleanly**, in T5.
- **★ OWNER UI requirement (2026-06-13d): custom-beautified, NOT stock ImGui** — a distinctive tasteful custom `ImGuiStyle` theme (colors/rounding/spacing) + a vendored TTF font + tidy custom-styled panels (a "physics-sim pro tool" aesthetic), NOT the default dark/gray ImGui look.
- **★ OWNER interop (2026-06-13e): DLPack-Vulkan shared-memory render = M11 (firm)**; M8.5 stays host-download (the PosePublisher seam swaps in with no rework). See [[render-physics-shared-memory]].
- **Open-Q resolutions:** present egress = DIRECT-to-swapchain (offscreen Render() stays the D1 oracle); InputSystem = minimal play/pause/camera dispatch in M8.5 (full drag-to-move → M11); frame-pacing interpolation = DEFER; debug-overlay-in-viewport = DEFER.
- **Execution:** spine T1→T2→T3 single-flight (build-coupled Vulkan/ImGui); T5 (cook) + T4 (PBR) = the beauty lane; T6 (gate B+C) joins. Controller builds + commits per phase; gated TUs validated by recompiling with `-DNK_BUILD_VULKAN_VALIDATION` + linking the built archives (NEVER reconfigure build-cuda128).
