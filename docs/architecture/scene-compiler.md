# Scene Compiler Pipeline

This document describes how scene descriptions are transformed from authoring
formats into runtime-ready data in the Nuka Physics Engine.

## Pipeline Stages

```
Authoring Formats          Canonical IR           Cooked Blobs
 (MJCF, URDF, USD)  --->  (SceneIR)       --->  (Binary blob)
       |                       |                       |
   Importers              In-memory graph         Serialized for
                          of bodies, joints,      fast runtime load
                          shapes, sensors
```

### 1. Authoring Model (Importers)

Importers read external scene descriptions and populate the `SceneIR`:

| Format | Module | Entry Point |
|--------|--------|-------------|
| MJCF   | `src/import/mjcf_importer.cpp` | `ImportMJCF(path)` |
| URDF   | `src/import/urdf_importer.cpp` | `ImportURDF(path)` |
| USD    | `src/import/usd_importer.cpp`  | `ImportUSD(path)` (planned) |

Each importer:
- Parses the source format (XML for MJCF/URDF, USD API for USD).
- Maps format-specific concepts to `SceneIR` records (rigid bodies, joints, shapes).
- Resolves mesh asset paths relative to the source file.

### 2. Canonical IR (`SceneIR`)

The `SceneIR` is the engine's format-independent scene representation:

- **RigidBodyRecord** -- name, local transform, mass properties, collision shape refs.
- **JointRecord** -- type (revolute, prismatic, fixed), parent/child body indices,
  axis, limits, drive parameters.
- **ShapeRecord** -- geometry type (box, sphere, capsule, mesh), dimensions.
- **SensorRecord** -- type (IMU, lidar, force/torque), attached body, sample rate.

The IR can also be built programmatically via `SceneIR::AddRigidBody()`,
`SceneIR::AddJoint()`, etc., without going through an importer.

### 3. Cooked Blobs

`scene::CookScene(SceneIR)` serializes the IR into a compact binary blob
optimized for runtime consumption:

- Flat arrays for body poses, mass properties, and shape data.
- Joint descriptor table with pre-computed local frames.
- Sensor table.
- All data is laid out for cache-friendly sequential access.

The cooked blob is consumed by `runtime::BuildWorld(blob)` to construct a
`WorldTemplate` and an initial `WorldInstance`.

## Adding a New Importer

1. Create `src/import/<format>_importer.hpp` and `.cpp`.
2. Implement a function that returns a populated `SceneIR`.
3. Register the format extension in the importer dispatch table.
4. Add tests under `tests/import/test_<format>_importer.cpp`.
