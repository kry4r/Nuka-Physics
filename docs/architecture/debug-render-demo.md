# Imported Scene Debug Render Demo

`nuka_scene_demo` is the current runnable rendering demonstration for the engine.
It is intentionally headless so it can run in CI and on machines without a window
system while still exercising the same scene and debug visualization path that a
native viewport will use.

## Pipeline

1. Load an input scene through `LoadMjcf`, `LoadUrdf`, or `LoadUsd`.
2. Compile it with `scene::BuildCompiledScene()`.
3. Advance the compiled `PhysicsWorld` instance with `runtime::StepWorldInstance()`.
4. Synchronize simulated body poses into `SceneGraph` and `RenderScene` with
   `scene::ApplyRuntimeStateToCompiledScene()`.
5. Generate debug overlays with `app::BuildDebugVisualization()`.
6. Rasterize the `DebugDrawList` with `app::RasterizeDebugDrawList()`.
7. Write a binary PPM image artifact.

The headless rasterizer uses an X/Y physics-plane projection for debug output.
`SceneDemoOptions` defaults to auto-fitting the view to the generated debug draw
bounds so simulated bodies remain visible after large vertical motion. Tests also
cover a fixed-view path to prove the PPM bytes change when runtime body poses
change.

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
after fixed-step runtime simulation. The default CLI and API settings run 60
steps at `dt=1/60`, and the optional final arguments override step count and
time step. The tests under `tests/apps/test_scene_demo.cpp` verify the same path
for both MJCF and USDA inputs, including the checked-in USD example scene, the
simulated pose handoff, and simulation-sensitive raster output.
