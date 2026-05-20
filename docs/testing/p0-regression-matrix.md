# P0 Regression Test Matrix

This document lists the P0 (must-pass) regression tests for the Nuka Physics Engine.
All tests in this matrix must pass on every commit before merge.

## Regression Tests

## Platform Contract Tests

| Test Name | File | Description | Required Backend |
|-----------|------|-------------|------------------|
| PlatformContract/RequiresCudaForProductionPhysics | `tests/phi/test_platform_contract.cpp` | Verifies the PHI platform contract keeps CUDA as the production physics backend and CPU as reference-only | CUDA, CPU reference |
| PlatformContract/DefaultSelectionPrefersCudaOnThisWorkstation | `tests/phi/test_platform_contract.cpp` | Verifies the backend selection API resolves the default policy to CUDA on this machine | CUDA |
| PlatformContract/CpuReferenceSelectionIsMarkedValidationOnly | `tests/phi/test_platform_contract.cpp` | Verifies explicit CPU selection remains validation/reference-only | CPU reference |
| VulkanRenderer/CreatesInstanceAndEnumeratesPhysicalDevices | `tests/render/test_vulkan_backend.cpp` | Verifies the Vulkan production renderer backend can create an instance and see a physical device | Vulkan |
| VulkanRenderer/DeclaresVulkanAsProductionBackend | `tests/render/test_vulkan_backend.cpp` | Verifies Vulkan is the declared production render backend | Vulkan |
| CudaDeviceWorld/UploadsCookedRuntimeTablesAndDownloadsSummary | `tests/runtime/test_cuda_device_world.cpp` | Verifies cooked runtime body, shape, joint, and actuator tables upload into CUDA buffers and can be validated by compact readback | CUDA |
| CudaWorldStepper/IntegratesGravityForceAndVelocityOnDevice | `tests/runtime/test_cuda_world_stepper.cpp` | Verifies the CUDA rigid integration kernel updates velocity, pose, and force accumulators on device buffers | CUDA |
| CudaWorldStepper/MatchesCpuReferenceForFreeFallAndForces | `tests/runtime/test_cuda_world_stepper.cpp` | Compares CUDA free-fall and forced-body integration against the CPU reference path with contacts disabled | CUDA, CPU reference |
| CudaWorldStepper/RejectsStepBeforeDeviceStateUpload | `tests/runtime/test_cuda_world_stepper.cpp` | Verifies CUDA stepping fails before kernel launch when mutable state has not been uploaded | CUDA |

## Import Regression Tests

| Test Name | File | Description | Required Records |
|-----------|------|-------------|------------------|
| MjcfImporter/ParsesSceneAuthoringRecords | `tests/import/test_mjcf_importer.cpp` | Loads a complete MJCF XML scene with robot, render, control, and sensor authoring data | body, joint, geom, material, camera, light, actuator, sensor |
| UsdImporter/ParsesRenderControlAndSensorRecords | `tests/import/test_usd_importer.cpp` | Loads a USDA/text USD scene through the isolated USD adapter and emits canonical metadata | body, joint, geom, material, camera, light, actuator, sensor |
| UsdImporter/CookedBlobContainsUsdImportedTables | `tests/import/test_usd_importer.cpp` | Verifies imported USD records flatten into cooked runtime metadata tables | body, joint, shape, material, camera, light, actuator, sensor |
| UsdImporter/DetectsUsdStageFormatsCaseInsensitively | `tests/import/test_usd_importer.cpp` | Verifies `.usd`, `.usda`, `.usdc`, and `.usdz` route through the isolated adapter boundary | extension dispatch |
| UsdImporter/LoadsTextUsdExtensionThroughAdapter | `tests/import/test_usd_importer.cpp` | Verifies text `.usd` files compile through the same adapter path as `.usda` | body, joint, geom |
| UsdImporter/BinaryUsdExtensionReportsOpenUsdSdkAdapterRequirement | `tests/import/test_usd_importer.cpp` | Verifies binary `.usd` reports the pending official OpenUSD SDK backend instead of text-parsing garbage | diagnostic |
| UsdImporter/UsdcAndUsdzRouteThroughOpenUsdSdkAdapterDiagnostics | `tests/import/test_usd_importer.cpp` | Verifies crate and package inputs hit the explicit OpenUSD adapter boundary | diagnostic |
| ScenePipeline/ImportedMjcfSceneBuildsPhysicsAndRenderViews | `tests/scene/test_scene_pipeline.cpp` | Verifies MJCF import converts into SceneGraph, PhysicsWorld, and RenderScene | SceneGraph, PhysicsWorld, RenderScene |
| ScenePipeline/ImportedUsdSceneBuildsPhysicsAndRenderViews | `tests/scene/test_scene_pipeline.cpp` | Verifies USD import converts into SceneGraph, PhysicsWorld, and RenderScene | SceneGraph, PhysicsWorld, RenderScene |
| ScenePipeline/RenderSceneKeepsShapeMaterialAndBodyBindings | `tests/scene/test_scene_pipeline.cpp` | Verifies RenderScene keeps shape, material, and body bindings for debug/render use | RenderScene bindings |
| ScenePipeline/AppliesRuntimeStateToRenderAndDebugViews | `tests/scene/test_scene_pipeline.cpp` | Verifies simulated runtime poses update SceneGraph world transforms plus render/debug/camera transforms without overwriting local authoring transforms | WorldInstance, SceneGraph, RenderScene |
| SceneCooker/CooksBodyPosesInWorldSpaceForRuntimeStepping | `tests/scene/test_scene_cooker.cpp` | Verifies cooked runtime body poses include parent transforms before physics stepping | SceneIR hierarchy, cooked body table |

## Debug Visualization Regression Tests

| Test Name | File | Description | Required Overlays |
|-----------|------|-------------|-------------------|
| DebugVisualization/EmitsCollisionBodyPrimitivesAndAabbs | `tests/apps/test_debug_draw_list.cpp` | Converts RenderScene debug proxies into body primitive and AABB draw commands | box, sphere, capsule, AABB |
| DebugVisualization/EmitsJointAxesAndCentersOfMass | `tests/apps/test_debug_draw_list.cpp` | Converts PhysicsWorld joint/body tables plus SceneGraph transforms into joint-axis and center-of-mass commands | joint axis, center of mass |
| DebugVisualization/EmitsContactsAndConstraintErrors | `tests/apps/test_debug_draw_list.cpp` | Converts contact manifolds and constraint blocks into contact normal and constraint error commands | contact point, contact normal, constraint error |
| HeadlessDebugRenderer/RasterizesImportedSceneDebugCommands | `tests/apps/test_scene_demo.cpp` | Rasterizes imported MJCF debug commands into a non-empty image | PPM pixels |
| SceneDemo/ExportsMjcfSceneDebugViewToPpm | `tests/apps/test_scene_demo.cpp` | Runs the imported-scene demo path for the example MJCF scene and writes a PPM artifact | MJCF, SceneGraph, PhysicsWorld, RenderScene, DebugDrawList |
| SceneDemo/ExportsUsdSceneThroughSamePipeline | `tests/apps/test_scene_demo.cpp` | Runs the same demo path for USDA/text USD input | USD adapter, SceneGraph, PhysicsWorld, RenderScene, DebugDrawList |
| SceneDemo/ExportsUsdExampleSceneThroughSamePipeline | `tests/apps/test_scene_demo.cpp` | Runs `examples/scenes/complete_robot.usda` through import, simulation sync, debug visualization, and raster output | USD example, DebugDrawList, PPM pixels |
| SceneDemo/SimulatesImportedSceneBeforeRendering | `tests/apps/test_scene_demo.cpp` | Steps an imported scene, synchronizes runtime poses, and renders debug overlays from the simulated state | WorldInstance, SceneGraph, RenderScene, DebugDrawList |
| SceneDemo/RenderedImageChangesWhenSimulationChangesRuntimePose | `tests/apps/test_scene_demo.cpp` | Verifies fixed-view PPM bytes change when simulation changes runtime body poses | WorldInstance, DebugDrawList, PPM pixels |

## Physics Regression Tests

| Test Name | File | Description | Key Metric | Threshold |
|-----------|------|-------------|------------|-----------|
| FreeFallRegression/EnergyDriftStaysBounded | `tests/regression/test_free_fall.cpp` | 1 kg body free-falling for 1 s at dt=0.01 | Energy drift | < 1e-2 |
| FreeFallRegression/PositionMatchesKinematicFormula | `tests/regression/test_free_fall.cpp` | Position compared to y = -0.5*g*t^2 | Relative error | < 1% |
| FreeFallRegression/VelocityMatchesExpected | `tests/regression/test_free_fall.cpp` | Final velocity compared to v = g*t | Absolute error | < 1e-4 |
| RestStackRegression/PenetrationIsBounded | `tests/regression/test_rest_stack.cpp` | Two boxes stacked on ground, solver run | Max penetration | < 1e-2 |
| RestStackRegression/TopBoxStaysAboveBottomBox | `tests/regression/test_rest_stack.cpp` | Vertical ordering preserved after solve | top_y > bottom_y | true |
| RestStackRegression/BottomBoxStaysAboveGround | `tests/regression/test_rest_stack.cpp` | Bottom box does not fall through ground | bottom_y >= 0 | true |
| ContactMaterial/FrictionImpulseIsClampedByNormalImpulse | `tests/solver/test_contact_material.cpp` | Tangential contact impulse is clamped by friction coefficient times normal impulse | friction impulse | <= mu * normal impulse |
| ContactMaterial/NoNormalImpulseProducesNoFrictionImpulse | `tests/solver/test_contact_material.cpp` | Separating contacts cannot generate tangential friction without positive normal force | friction impulse | 0 |
| ContactMaterial/RestitutionBouncesAlongContactNormal | `tests/solver/test_contact_material.cpp` | Closing normal velocity is reflected by the restitution coefficient | normal velocity | expected bounce speed |
| TwoLinkArmRegression/JointMaintainsConnectivity | `tests/regression/test_two_link_arm.cpp` | Revolute joint gap after 10 steps with gravity | Gap distance | < 0.05 |
| TwoLinkArmRegression/AnchorPointsCoincide | `tests/regression/test_two_link_arm.cpp` | Per-axis anchor point agreement | Per-axis error | < 0.05 |
| JointProjection/StaticAnchorPullsDynamicAnchorOntoPivot | `tests/solver/test_joint_projection.cpp` | Static parent anchor remains fixed while dynamic child is projected onto the pivot | Anchor gap | < 1e-3 |
| JointProjection/MassWeightedCorrectionMovesLightBodyMore | `tests/solver/test_joint_projection.cpp` | Joint projection distributes correction by inverse mass | Light body motion | > 5x heavy body motion |
| JointProjection/EccentricAnchorProjectionUsesAngularInertia | `tests/solver/test_joint_projection.cpp` | Eccentric joint anchor projection applies angular correction through inverse inertia | Gap reduction, orientation change | gap < 25% initial and rotation > 1e-3 |
| WorldStepper/AdvancesDynamicBodiesAndKeepsStaticBodiesFixed | `tests/runtime/test_world_stepper.cpp` | Fixed-step world stepping advances dynamic bodies and preserves static body state | Pose/velocity | expected symplectic Euler values |
| WorldStepper/AppliesForcesTorquesAndClearsAccumulatorsAfterEachStep | `tests/runtime/test_world_stepper.cpp` | Applies force and torque accumulators through body tables and clears them after stepping | Velocity/accumulators | expected values / zeroed accumulators |
| WorldStepper/GeneratesCookedShapeContactsAndSolvesAgainstStaticPlane | `tests/runtime/test_world_stepper.cpp` | Cooked plane and box shapes flow through broadphase, contact manifold generation, contact constraint assembly, PGS solve, and pose writeback | contact report, velocity, pose | one manifold, contact rows, no downward velocity, depenetrated pose |
| WorldStepper/UsesPlaneShapeTransformWhenGeneratingContacts | `tests/runtime/test_world_stepper.cpp` | Plane contact generation honors cooked local/world shape transforms instead of assuming y=0 | contact report, pose | one manifold, raised-plane depenetration |
| WorldStepper/SolvesCookedRevoluteJointAnchors | `tests/runtime/test_world_stepper.cpp` | Cooked revolute joint records emit runtime joint constraints and project child anchors onto parent anchors | joint report, constraint rows, child pose | one joint block, five rows, anchor gap corrected |
| WorldStepper/AppliesCookedVelocityActuatorAsDriveConstraint | `tests/runtime/test_world_stepper.cpp` | Cooked velocity actuator records emit drive constraints and change child angular velocity along the joint axis | drive report, angular velocity | one drive block, positive child angular velocity |
| SceneCooker/CooksJoints | `tests/scene/test_scene_cooker.cpp` | Cooked joint tables preserve parent/child body ids, axes, limits, and local joint frames | joint table | exact authored fields |
| WorldBuild/CopiesJointFramesAndActuatorsIntoTemplate | `tests/runtime/test_world_build.cpp` | Runtime templates retain cooked joint frames and actuator rows for stepper constraint assembly | WorldTemplate | frames, actuator count, gains, limits preserved |
| ConstraintBlock/ContactPenetrationPropagatesToPositionError | `tests/constraint/test_constraint_block.cpp` | Manifold penetration becomes position error instead of velocity rhs | position_error, rhs | penetration in position_error and zero velocity rhs |

## Performance Tests

| Test Name | File | Description | Metric | Threshold |
|-----------|------|-------------|--------|-----------|
| StepTiming/HundredStepsUnderOneSecond | `tests/perf/test_step_timing.cpp` | 100 steps on 10 bodies | Wall time | < 1000 ms |
| StepTiming/ThousandStepsSingleBody | `tests/perf/test_step_timing.cpp` | 1000 steps on 1 body | Wall time | < 1000 ms |
| StepTiming/ContactMaterialSolveUnderOneSecond | `tests/perf/test_step_timing.cpp` | 120 PGS contact-material solves over 64 bodies with friction and restitution rows | Wall time | < 1000 ms |
| StepTiming/RuntimeContactPipelineUnderOneSecond | `tests/perf/test_step_timing.cpp` | 120 fixed runtime steps over 48 cooked dynamic boxes and a static plane through broadphase, narrowphase, contact builder, solver, and integration | Wall time | < 1000 ms |
| StepTiming/RuntimeJointDrivePipelineUnderOneSecond | `tests/perf/test_step_timing.cpp` | 120 fixed runtime steps over 32 cooked revolute joints and velocity drives through joint/drive constraint assembly, PGS solve, and integration | Wall time | < 1000 ms |
| RenderDemoTiming/ImportedSceneDebugViewUnderOneSecond | `tests/perf/test_render_demo_timing.cpp` | Import, compile, simulate 60 fixed steps, overlay, and rasterize example MJCF scene | Wall time | < 1000 ms |
| CudaStepTiming/RigidIntegrationKernelUnderOneSecond | `tests/perf/test_cuda_step_timing.cpp` | 240 CUDA rigid integration steps over 4096 cooked dynamic bodies | Wall time | < 1000 ms |

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
