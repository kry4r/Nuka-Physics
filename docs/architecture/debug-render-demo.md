# Imported Scene Debug Render Demo

`nuka_scene_demo` is the current runnable rendering demonstration for the engine.
It runs without a window, but the default rendering path is Vulkan: the demo
uses a swapchain-free offscreen compute pass, reads back the image, and writes a
PPM artifact for CI and inspection.

## Pipeline

1. Load an input scene through `LoadMjcf`, `LoadUrdf`, or `LoadUsd`.
2. Compile it with `scene::BuildCompiledScene()`.
3. Resolve the physics backend through PHI. On this workstation the default
   policy selects the CUDA production path; CPU remains reference-only for
   explicit validation runs and is rejected unless the caller also enables
   `allow_cpu_reference_validation`.
4. Advance the compiled `PhysicsWorld` instance on CUDA by uploading the cooked
   `WorldTemplate` and mutable `WorldInstance` into `DeviceWorld`, then running
   fixed-step integration, CUDA broadphase, CUDA contact generation, and CUDA
   contact/joint/drive PGS solving.
5. Synchronize simulated body poses into `SceneGraph` and `RenderScene` with
   `scene::ApplyRuntimeStateToCompiledScene()`.
6. Generate debug overlays with `app::BuildDebugVisualization()`.
7. Convert the `DebugDrawList` to render-layer debug commands and rasterize it
   with `render::RenderDebugDrawListVulkan()` into an offscreen Vulkan storage
   image.
8. Read back the Vulkan image and write a binary PPM artifact.

The same executable also has a batched mode when the optional final
`instance_count` CLI argument is greater than one. In that mode it imports and
cooks one scene, creates offset copies of the runtime instance, uploads them
into CUDA `BatchedDeviceWorld`, runs batched contacts/joints/drives and batched
IMU/frame-pose observations on CUDA, then renders the combined multi-environment
debug view through Vulkan. CPU work in this path is limited to import/cooking,
host orchestration, and the final debug/render synchronization boundary.

The Vulkan offscreen renderer uses an X/Y physics-plane projection for debug
output and currently supports line, sphere, capsule, box, AABB, and contact
point debug primitives. `SceneDemoOptions` defaults to auto-fitting the view to
the generated debug draw bounds so simulated bodies remain visible after large
vertical motion. The CPU headless rasterizer remains available as an explicit
reference artifact path, but it is not the default production render backend.
The same synchronized `SceneGraph` / `RenderScene` state can now also be sent
directly to `render::RenderSceneVulkan()`, which renders material-colored mesh
instances through the Vulkan offscreen path. This is still a compute offscreen
adapter rather than the final instanced graphics pipeline with lighting and
shadows, but app-level callers already have a real `RenderScene` Vulkan entry
point instead of only debug overlays.

The demo currently supports `.xml`, `.urdf`, text `.usd`, and `.usda` inputs.
Binary `.usd`, `.usdc`, and `.usdz` still route through the isolated USD adapter
and return the explicit pending-OpenUSD-SDK error until the native SDK backend is
added.

## Run

```powershell
cmake --build build --config Release --target nuka_scene_demo
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.xml out\complete_robot_debug.ppm 640 360 60 0.0166667
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.usda out\complete_robot_usd_debug.ppm 640 360 60 0.0166667
.\build\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.usda out\complete_robot_batched_debug.ppm 640 360 60 0.0166667 8
```

The output image shows collision bodies, AABBs, joint axes, and centers of mass
after fixed-step CUDA simulation and Vulkan offscreen rendering on this
workstation. The CLI prints the physics backend, render backend, CUDA
constraint row counts, joint/drive/contact block counts, and maximum position
error. The default CLI and API settings run 60 steps at `dt=1/60`, and the
optional final arguments override step count and time step. The tests under
`tests/apps/test_scene_demo.cpp` verify default CUDA backend selection, Vulkan
rendering selection, simulated pose handoff, and simulation-sensitive raster
output. `tests/perf/test_cuda_scene_demo_timing.cpp` and
`tests/perf/test_vulkan_scene_demo_timing.cpp` run both
`examples/scenes/complete_robot.xml` and `examples/scenes/complete_robot.usda`
through import, cook, CUDA simulation, render/debug synchronization, Vulkan
artifact output, and one-second timing budgets.
`tests/perf/test_batched_vulkan_scene_demo_timing.cpp` adds the same global
workflow for eight CUDA batched environments rendered through Vulkan.
`tests/perf/test_vulkan_renderscene_timing.cpp` separately covers the material
`RenderScene` Vulkan entry point with 128 mesh instances. Remaining renderer
work is richer Vulkan instancing/material/shadow support and replacing
PPM/readback-oriented synchronization with direct interactive GPU presentation.
