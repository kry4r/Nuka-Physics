# P0 Regression Test Matrix

This document lists the P0 (must-pass) regression tests for the Nuka Physics Engine.
All tests in this matrix must pass on every commit before merge.

## Regression Tests

## Import Regression Tests

| Test Name | File | Description | Required Records |
|-----------|------|-------------|------------------|
| MjcfImporter/ParsesSceneAuthoringRecords | `tests/import/test_mjcf_importer.cpp` | Loads a complete MJCF XML scene with robot, render, control, and sensor authoring data | body, joint, geom, material, camera, light, actuator, sensor |
| UsdImporter/ParsesRenderControlAndSensorRecords | `tests/import/test_usd_importer.cpp` | Loads a USDA/text USD scene through the isolated USD adapter and emits canonical metadata | body, joint, geom, material, camera, light, actuator, sensor |
| UsdImporter/CookedBlobContainsUsdImportedTables | `tests/import/test_usd_importer.cpp` | Verifies imported USD records flatten into cooked runtime metadata tables | body, joint, shape, material, camera, light, actuator, sensor |
| ScenePipeline/ImportedMjcfSceneBuildsPhysicsAndRenderViews | `tests/scene/test_scene_pipeline.cpp` | Verifies MJCF import converts into SceneGraph, PhysicsWorld, and RenderScene | SceneGraph, PhysicsWorld, RenderScene |
| ScenePipeline/ImportedUsdSceneBuildsPhysicsAndRenderViews | `tests/scene/test_scene_pipeline.cpp` | Verifies USD import converts into SceneGraph, PhysicsWorld, and RenderScene | SceneGraph, PhysicsWorld, RenderScene |
| ScenePipeline/RenderSceneKeepsShapeMaterialAndBodyBindings | `tests/scene/test_scene_pipeline.cpp` | Verifies RenderScene keeps shape, material, and body bindings for debug/render use | RenderScene bindings |

## Debug Visualization Regression Tests

| Test Name | File | Description | Required Overlays |
|-----------|------|-------------|-------------------|
| DebugVisualization/EmitsCollisionBodyPrimitivesAndAabbs | `tests/apps/test_debug_draw_list.cpp` | Converts RenderScene debug proxies into body primitive and AABB draw commands | box, sphere, capsule, AABB |
| DebugVisualization/EmitsJointAxesAndCentersOfMass | `tests/apps/test_debug_draw_list.cpp` | Converts PhysicsWorld joint/body tables plus SceneGraph transforms into joint-axis and center-of-mass commands | joint axis, center of mass |
| DebugVisualization/EmitsContactsAndConstraintErrors | `tests/apps/test_debug_draw_list.cpp` | Converts contact manifolds and constraint blocks into contact normal and constraint error commands | contact point, contact normal, constraint error |
| HeadlessDebugRenderer/RasterizesImportedSceneDebugCommands | `tests/apps/test_scene_demo.cpp` | Rasterizes imported MJCF debug commands into a non-empty image | PPM pixels |
| SceneDemo/ExportsMjcfSceneDebugViewToPpm | `tests/apps/test_scene_demo.cpp` | Runs the imported-scene demo path for the example MJCF scene and writes a PPM artifact | MJCF, SceneGraph, PhysicsWorld, RenderScene, DebugDrawList |
| SceneDemo/ExportsUsdSceneThroughSamePipeline | `tests/apps/test_scene_demo.cpp` | Runs the same demo path for USDA/text USD input | USD adapter, SceneGraph, PhysicsWorld, RenderScene, DebugDrawList |

## Physics Regression Tests

| Test Name | File | Description | Key Metric | Threshold |
|-----------|------|-------------|------------|-----------|
| FreeFallRegression/EnergyDriftStaysBounded | `tests/regression/test_free_fall.cpp` | 1 kg body free-falling for 1 s at dt=0.01 | Energy drift | < 1e-2 |
| FreeFallRegression/PositionMatchesKinematicFormula | `tests/regression/test_free_fall.cpp` | Position compared to y = -0.5*g*t^2 | Relative error | < 1% |
| FreeFallRegression/VelocityMatchesExpected | `tests/regression/test_free_fall.cpp` | Final velocity compared to v = g*t | Absolute error | < 1e-4 |
| RestStackRegression/PenetrationIsBounded | `tests/regression/test_rest_stack.cpp` | Two boxes stacked on ground, solver run | Max penetration | < 1e-2 |
| RestStackRegression/TopBoxStaysAboveBottomBox | `tests/regression/test_rest_stack.cpp` | Vertical ordering preserved after solve | top_y > bottom_y | true |
| RestStackRegression/BottomBoxStaysAboveGround | `tests/regression/test_rest_stack.cpp` | Bottom box does not fall through ground | bottom_y >= 0 | true |
| TwoLinkArmRegression/JointMaintainsConnectivity | `tests/regression/test_two_link_arm.cpp` | Revolute joint gap after 10 steps with gravity | Gap distance | < 0.05 |
| TwoLinkArmRegression/AnchorPointsCoincide | `tests/regression/test_two_link_arm.cpp` | Per-axis anchor point agreement | Per-axis error | < 0.05 |

## Performance Tests

| Test Name | File | Description | Metric | Threshold |
|-----------|------|-------------|--------|-----------|
| StepTiming/HundredStepsUnderOneSecond | `tests/perf/test_step_timing.cpp` | 100 steps on 10 bodies | Wall time | < 1000 ms |
| StepTiming/ThousandStepsSingleBody | `tests/perf/test_step_timing.cpp` | 1000 steps on 1 body | Wall time | < 1000 ms |
| RenderDemoTiming/ImportedSceneDebugViewUnderOneSecond | `tests/perf/test_render_demo_timing.cpp` | Import, compile, overlay, and rasterize example MJCF scene | Wall time | < 1000 ms |

## Reference Comparison Tool

The `tools/reference/compare_reference.py` script compares CSV output from a simulation
run against stored reference data. Usage:

```
python tools/reference/compare_reference.py \
    --input output.csv \
    --reference data/reference.csv \
    --tolerance 1e-6
```

Reports max absolute error per numeric column and exits with code 1 on failure.

## Adding New Regression Tests

1. Add the test source file under `tests/regression/` or `tests/perf/`.
2. Register it in `tests/CMakeLists.txt` under the appropriate target.
3. Add an entry to this matrix document with the expected metric and threshold.
4. If the test produces CSV output, store the reference in `tests/data/` and
   add a CI step that runs `compare_reference.py`.
