# Imported Scene Debug Render Demo

`nuka_scene_demo` is the current runnable rendering demonstration for the engine.
It is intentionally headless so it can run in CI while still exercising the same
scene and debug visualization command source that the Vulkan production viewport
will consume.

## Pipeline

1. Load an input scene through `LoadMjcf`, `LoadUrdf`, or `LoadUsd`.
2. Compile it with `scene::BuildCompiledScene()`.
3. Resolve the physics backend through PHI. On this workstation the default
   policy selects the CUDA production path; CPU remains reference-only for
   explicit validation runs.
4. Advance the compiled `PhysicsWorld` instance on CUDA by uploading the cooked
   `WorldTemplate` and mutable `WorldInstance` into `DeviceWorld`, then running
   fixed-step integration, CUDA broadphase, CUDA contact generation, and CUDA
   contact/joint/drive PGS solving.
5. Synchronize simulated body poses into `SceneGraph` and `RenderScene` with
   `scene::ApplyRuntimeStateToCompiledScene()`.
6. Generate debug overlays with `app::BuildDebugVisualization()`.
7. Rasterize the `DebugDrawList` with `app::RasterizeDebugDrawList()`.
8. Write a binary PPM image artifact.

The headless rasterizer uses an X/Y physics-plane projection for debug output.
`SceneDemoOptions` defaults to auto-fitting the view to the generated debug draw
bounds so simulated bodies remain visible after large vertical motion. Tests also
cover a fixed-view path to prove the PPM bytes change when runtime body poses
change.

The production rendering backend is Vulkan. The headless PPM output is a
deterministic regression artifact, not the long-term interactive renderer.
The same synchronized `SceneGraph` / `RenderScene` state is the handoff point
for the Vulkan viewport.

The demo currently supports `.xml`, `.urdf`, text `.usd`, and `.usda` inputs.
Binary `.usd`, `.usdc`, and `.usdz` still route through the isolated USD adapter
and return the explicit pending-OpenUSD-SDK error until the native SDK backend is
added.

## Run

```powershell
cmake --build build2 --config Release --target nuka_scene_demo
.\build2\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.xml out\complete_robot_debug.ppm 640 360 60 0.0166667
.\build2\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.usda out\complete_robot_usd_debug.ppm 640 360 60 0.0166667
```

The output image shows collision bodies, AABBs, joint axes, and centers of mass
after fixed-step CUDA simulation on this workstation. The CLI prints the
selected backend plus CUDA constraint row counts, joint/drive/contact block
counts, and maximum position error. The default CLI and API settings run 60
steps at `dt=1/60`, and the optional final arguments override step count and
time step. The tests under `tests/apps/test_scene_demo.cpp` verify default CUDA
backend selection, simulated pose handoff, and simulation-sensitive raster
output. `tests/perf/test_cuda_scene_demo_timing.cpp` runs both
`examples/scenes/complete_robot.xml` and `examples/scenes/complete_robot.usda`
through import, cook, CUDA simulation, render/debug synchronization, and PPM
artifact output under a one-second timing budget.
