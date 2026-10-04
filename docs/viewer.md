# Viewer and renderer validation

Nuka Editor combines a Vulkan viewport with dockable scene, inspector, camera,
transport, and read-only physics-debug panels. The editor executable still needs
the production physics backend. The UI library and renderer smoke tests can also
be built on a host without CUDA.

## Interaction

- Play/Pause controls the loaded scene; Step is available while paused
- Left-drag orbits, middle-drag / Shift-left-drag pans, and the wheel zooms
- Ctrl-left-click selects a movable instance; Ctrl-left-drag uses the existing
  entity-edit path. Releasing Ctrl or the mouse, losing focus, or replacing the
  scene cancels the drag
- UI and gizmo input capture stop camera gestures
- Frame Selected (`F`) uses render instances carrying the selected entity; Frame
  All (`Home`) uses all real render geometry. Both preserve the view orientation
  and set the orbit pivot to the bounds center. Debug geometry is excluded
- `F` / `Home` shortcuts require the pointer over the viewport, window focus,
  no active text/keyboard capture, no gizmo operation, and no modifier keys
- The silver-white workspace defaults to Hierarchy, Scene View and a contextual
  Inspector. File contains Open, manual-path Open, Save As and Unload; Window
  opens Camera, Drive/Teleop, Script, Statistics and Physics Debug on demand
- World/Local changes the transform gizmo coordinate frame. It does not change
  physics coordinates or data
- The camera used by picking, gizmos and rendering is one frame-consistent
  snapshot. Camera input updates become visible on the next rendered frame
- Window > Reset workspace layout restores the default. The editor stores the
  versioned layout in `%APPDATA%/nuka/viewer-layout-v2.ini` on Windows or
  `$XDG_CONFIG_HOME/nuka/viewer-layout-v2.ini` (`~/.config` fallback) on Linux.
  Tests disable persistence; old layout files are not deleted

## Default Lab and resources

Without `--scene`, the editor loads `examples/assets/nuka_lab/gripper.nks` through
the ordinary scene loader. An explicit `--scene` takes precedence; `--empty`
starts a blank workspace. The Lab uses the authored three-quarter camera, with
framing adjusted to the usable scene viewport, and the authored background color.
Loading errors keep the previous scene and expose the requested path for retry.

This file is the full-size **Lab environment**, including the equipment cabinet.
It does not contain the dynamic gripper shown in the [gallery preview](nuka-stage.md),
which adds a recorded robot pose. The viewer does not inject or fabricate that robot.

Inter 4.1 Regular/SemiBold and the full Noto Sans CJK SC font are fetched once at
**build time** from official, pinned sources. The archive and individual files
are SHA256-checked; see [font provenance](../assets/viewer/fonts/README.md).
Inter Regular is also embedded as a proportional UI fallback, with OFL licenses
retained in the repository and packaged output.
For offline font provisioning, set these CMake options to the original files:

- `-DNUKA_VIEWER_INTER_REGULAR_FONT=/absolute/path/to/Inter-Regular.ttf`
- `-DNUKA_VIEWER_INTER_SEMIBOLD_FONT=/absolute/path/to/Inter-SemiBold.ttf`
- `-DNUKA_VIEWER_CJK_FONT=/absolute/path/to/NotoSansCJKsc-Regular.otf`

Other build dependencies must already be available. A missing file, failed
download, or checksum mismatch stops configuration. No reduced-character subset
is silently substituted. Default downloads are cached in `<build>/_deps/nuka-fonts`.

Post-build packaging copies the fonts and licenses, original logo and complete
Lab assets alongside the editor. Resource lookup first checks the executable
folder, then `../share/nuka`, then the development-build path; it does not depend
on the process working directory or download anything at runtime. Keep these
resource directories with a relocated executable. Missing CJK resources produce
a readable warning, not a claim of complete Chinese support. Glyphs are rasterized
on demand, the CJK source bytes are shared between font roles, and ImGui uses
32-bit Unicode; actual coverage is the source font's coverage. This does not add
Linux IME composition support.

## Physics Debug

The Physics Debug panel only observes published state. Collider proxies and
contact markers use a separate render channel and do not become pickable scene
objects. Hide all disables both overlays.

- Green: dynamic collider proxy
- Magenta: static collider proxy
- Orange: contact-position marker, not a force or impulse visualization

Counts describe the previous overlay snapshot's emitted geometry. N/A means the
layer was disabled, unavailable, or not read because collider proxies used the
budget. This last case reports an unknown contact omission count, never a sampled
zero. Filling the budget exactly does not imply omitted geometry; known valid
instances that do not fit are counted separately. Invalid dimensions, non-finite
points/poses, and rotations outside a unit-quaternion squared-norm tolerance of
0.001 are skipped and reported separately from
unsupported shapes. Primitive cache keys preserve the exact effective dimensions;
finite nonpositive plane extents retain the finite ground-patch fallback.
Primitive colliders are supported;
convex, SDF, and heightfield proxy coverage is incomplete. Wireframe falls back
to translucent fill when the device lacks non-solid polygon support.

The Stats panel separates CPU step-plus-publish time from wall-clock frame time.
GPU time, measured draw calls, and pixel counts display N/A when not sampled;
instance count is reported separately.

## Host-only renderer tests

Install CMake, a C++20 compiler, Vulkan headers/loader, `glslc`, Vulkan validation
layers, and a Vulkan driver. Linux builds also need X11/XCB development headers.
Mesa software Vulkan can validate correctness without a discrete GPU.
Linux viewer builds additionally require xkbcommon, xkbcommon-x11, and xcb-xkb
headers/libraries (Debian: `libxkbcommon-dev libxkbcommon-x11-dev libxcb-xkb-dev`).

```sh
cmake -S . -B build-viewer-host -G Ninja \
  -DNK_REQUIRE_CUDA=OFF -DNK_PHYSICS_BACKEND=CPU_REFERENCE \
  -DNK_BUILD_TESTS=ON -DNK_BUILD_VULKAN_VALIDATION=ON
cmake --build build-viewer-host --target \
  nuka_viewer_frame_smoke_test nuka_viewport_present_smoke_test \
  nuka_render_raster_smoke_test
python tools/validation/viewer_renderer_gate.py --build-dir build-viewer-host
```

The gate requires `VK_LAYER_KHRONOS_validation`, enables synchronization
validation, and fails on process errors, skipped graphics tests, or Vulkan
validation errors. Logs and a JSON summary are written to the ignored
`.nuka-runs/viewer-validation/` directory. Set `VK_ICD_FILENAMES` and
`VK_LAYER_PATH` only when driver/layer discovery needs explicit local paths.

The present test uses a real swapchain and controlled reported surface extents
to exercise suspend/recreate behavior. It does not simulate an operating system's
native window manager. A fixed-extent native surface may keep its actual size
while the test changes the reported size. Native minimize/maximize, HiDPI,
Windows GLFW, and Linux XCB input still need platform-specific validation.

## Renderer behavior and current limits

Zero-sized framebuffer reports suspend drawing while callers keep processing
events. Restoring the surface rebuilds the swapchain. Capture requests require
both transfer-source support and an RGBA8/BGRA8 surface format. Successful frame
counts exclude failed presents.

The interactive and offscreen renderers share shading code but do not expose
identical capabilities. The present path does not yet match offline lighting,
textures, shadows, or transmission.

Linux XCB forwards layout-aware key bindings and committed UTF-8 text through
xkbcommon. Locale-dependent Compose/dead-key sequences are supported when a
Compose table is available. This is not an XIM/IBus/Fcitx preedit or full IME
integration. Glyph display depends on the loaded fonts. Detectable autorepeat
is requested; legacy release/press repeat pairs are filtered. Focus loss clears
held input even if focus returns before the next frame. The Windows viewer keeps
its existing GLFW ImGui backend and does not duplicate Linux event forwarding.

Frame Selected does not infer descendants of a hierarchy-only node without
render geometry. Invalid/non-finite or unrepresentable bounds leave the camera
unchanged. The editor's scene viewport supplies the projection aspect, Vulkan
viewport/scissor, local picking coordinates and clipped gizmo rectangle. Logical
UI coordinates are converted to aligned framebuffer pixels. A hidden/zero-size
viewport starts no new gestures. Renderer callers that do not request a scene
subrectangle retain the full-frame default.

The host-only tests do not validate CUDA physics, CUDA/Vulkan interop, or hardware
GPU performance; use the production regression pipeline for those guarantees.
