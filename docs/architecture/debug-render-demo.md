# Imported Scene Debug Render Demo

`nuka_scene_demo` is the current runnable rendering demonstration for the engine.
It is intentionally headless so it can run in CI and on machines without a window
system while still exercising the same scene and debug visualization path that a
native viewport will use.

## Pipeline

1. Load an input scene through `LoadMjcf`, `LoadUrdf`, or `LoadUsd`.
2. Compile it with `scene::BuildCompiledScene()`.
3. Generate debug overlays with `app::BuildDebugVisualization()`.
4. Rasterize the `DebugDrawList` with `app::RasterizeDebugDrawList()`.
5. Write a binary PPM image artifact.

The demo currently supports `.xml`, `.urdf`, `.usd`, and `.usda` inputs. Binary
`.usdc` and `.usdz` still route through the isolated USD adapter and return the
explicit pending-OpenUSD-SDK error until the native SDK backend is added.

## Run

```powershell
cmake --build build2 --config Release --target nuka_scene_demo
.\build2\src\Release\nuka_scene_demo.exe examples\scenes\complete_robot.xml out\complete_robot_debug.ppm 640 360
```

The output image shows collision bodies, AABBs, joint axes, and centers of mass
from the imported scene. The tests under `tests/apps/test_scene_demo.cpp` verify
the same path for both MJCF and USDA inputs.
