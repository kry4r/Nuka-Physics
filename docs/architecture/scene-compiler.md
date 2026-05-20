# Scene Compiler Pipeline

This document describes how scene descriptions are transformed from authoring
formats into runtime-ready data in the Nuka Physics Engine.

## Pipeline Stages

```
Authoring Formats          Canonical IR          Compiled Runtime Views
 (MJCF, URDF, USD)  --->  (SceneIR)       --->  SceneGraph
       |                       |            \-->  PhysicsWorld
   Importers              In-memory graph   \-->  RenderScene
                          of bodies, joints, \->  CookedBlob
                          shapes, sensors
```

### 1. Authoring Model (Importers)

Importers read external scene descriptions and populate the `SceneIR`:

| Format | Module | Entry Point |
|--------|--------|-------------|
| MJCF   | `src/import/mjcf_importer.cpp` | `LoadMjcf(path)` |
| URDF   | `src/import/urdf_importer.cpp` | `LoadUrdf(path)` |
| USD/USDA | `src/import/usd_importer.cpp` | `LoadUsd(path)` |

Each importer:
- Parses the source format: XML for MJCF/URDF, and an isolated USD adapter for USDA/text USD until the official OpenUSD C++ SDK backend is wired in.
- Maps format-specific concepts to `SceneIR` records: bodies, joints, collision geometry, materials, cameras, lights, actuators, and sensors.
- Resolves named body, joint, and material references before emitting the canonical IR.

### 2. Canonical IR (`SceneIR`)

The `SceneIR` is the engine's format-independent scene representation:

- **RigidBodyRecord** -- name, parent, local transform, mass properties, and static flag.
- **JointRecord** -- type (revolute, prismatic, fixed), parent/child body indices,
  axis, limits, drive parameters.
- **CollisionShapeRecord** -- geometry type (box, sphere, capsule, mesh), dimensions,
  local transform, and material reference.
- **MaterialRecord** -- base color, alpha, roughness, and metallic properties.
- **CameraRecord** -- attachment, local transform, FOV, and clipping range.
- **LightRecord** -- type, color, intensity, attachment, and transform.
- **ActuatorRecord** -- joint target, control type, gain, and force limit.
- **SensorRecord** -- type (IMU, lidar, force/torque), attached body, sample rate.

The IR can also be built programmatically via `SceneIR::AddRigidBody()`,
`SceneIR::AddJoint()`, etc., without going through an importer.

### 3. Unified Compiled Views

`scene::BuildCompiledScene(SceneIR)` converts imported or programmatic scenes
into the three internal views used by the simulator:

- **SceneGraph** -- hierarchical body nodes with local and world transforms.
- **PhysicsWorld** -- physics-facing tables and a `runtime::BuiltWorld` built
  from cooked body, joint, shape, actuator, and sensor data.
- **RenderScene** -- render-facing mesh instances, material records, cameras,
  lights, and debug proxies. This view shares identifiers and transforms with
  the SceneGraph instead of owning physics state.

The conversion path is the required bridge from XML/USD importers to simulation
and rendering. Tests cover programmatic scenes plus MJCF and USD imported scenes
through the same `BuildCompiledScene()` entry point.

`RenderScene` debug proxies retain shape dimensions needed by debug overlays:
box half extents, sphere radii, and capsule radii/half heights. The debug
visualization bridge uses those proxies together with `PhysicsWorld` joint/body
tables and `SceneGraph` world transforms to draw collision bodies, AABBs, joint
axes, centers of mass, contacts, and constraint errors without coupling the
render module back to simulation internals.

### 4. Cooked Blobs

`scene::CookScene(SceneIR)` serializes the IR into a compact binary blob
optimized for runtime consumption:

- Flat arrays for body poses, mass properties, and shape data.
- Joint descriptor table with pre-computed local frames.
- Sensor, material, camera, light, and actuator metadata tables.
- All data is laid out for cache-friendly sequential access.

## USD Adapter Boundary

`LoadUsd()` is the stable engine-facing entry point. The current implementation
supports ASCII `.usda` and text `.usd` files and rejects binary `.usdc`/`.usdz`
with an explicit error that points to the pending OpenUSD SDK adapter. This keeps
callers and tests aligned with the final import path while avoiding a hard SDK
dependency before build-system support is added.

The cooked blob is consumed by `runtime::BuildWorld(blob)` and
`runtime::BuildPhysicsWorld(blob)` to construct a `WorldTemplate`, an initial
`WorldInstance`, and a physics-facing world view.

## Adding a New Importer

1. Create `src/import/<format>_importer.hpp` and `.cpp`.
2. Implement a function that returns a populated `SceneIR`.
3. Register the format extension in the importer dispatch table.
4. Add tests under `tests/import/test_<format>_importer.cpp`.
