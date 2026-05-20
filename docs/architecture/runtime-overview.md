# Runtime Overview

This document describes the runtime architecture of the Nuka Physics Engine.

## Core Abstractions

### WorldTemplate

A read-only, immutable description of a physics scene produced by the scene cooker.
Contains body tables (poses, masses, inertias), joint descriptors, collision shapes,
and sensor definitions. Multiple `WorldInstance` objects can share the same template
to enable batched simulation.

### PhysicsWorld

`runtime::PhysicsWorld` is the physics-facing compiled view created from a cooked
scene. It owns body, joint, shape, actuator, and sensor tables plus the single
instance `BuiltWorld` used by the current runtime. It is intentionally separate
from render data so simulation can step without depending on cameras, lights, or
debug draw state.

### WorldInstance

A mutable simulation state derived from a `WorldTemplate`. Holds per-body dynamic
state: positions, orientations, linear and angular velocities, forces, and torques.
Each instance evolves independently during simulation.

### World Stepper

`runtime::StepWorldInstance()` advances a `WorldInstance` against its
`WorldTemplate` with fixed-step CPU integration. It converts each body row into
the rigid `BodyState`, applies gravity and accumulated forces/torques through the
shared symplectic Euler integrator, writes poses and velocities back, and clears
force accumulators by default. Cooked body poses are stored in world space so
imported hierarchies start simulation from the same transforms used by
`SceneGraph`.

### BatchContext / BatchScheduler

Manages multiple `WorldInstance` objects that share a common `WorldTemplate`.
The `BatchScheduler` orchestrates parallel stepping of instances, enabling
domain-randomized training and large-scale evaluation scenarios.

## Scene Integration

Imported and programmatic scenes flow through `scene::BuildCompiledScene()`,
which emits:

1. `SceneGraph` for authoritative hierarchy and shared transforms.
2. `PhysicsWorld` for simulation tables and mutable runtime state.
3. `RenderScene` for mesh instances, materials, cameras, lights, and debug
   proxies.

This keeps rendering decoupled from simulation while still sharing stable body,
shape, material, camera, light, actuator, and sensor identifiers.

After simulation, `scene::ApplyRuntimeStateToCompiledScene()` copies runtime body
poses back into the compiled `SceneGraph` and updates render mesh, debug proxy,
camera, and light transforms. App layers use this bridge before drawing or
debugging a simulated frame.

### Debug Visualization Bridge

`app::BuildDebugVisualization()` consumes the compiled `RenderScene`,
`PhysicsWorld`, and `SceneGraph`, plus optional contact manifolds and constraint
blocks, and emits a GPU-independent `DebugDrawList`. The bridge covers collision
shape proxies, shape AABBs, joint axes, centers of mass, contact points/normals,
and constraint error vectors. Native shells and future renderers should consume
this command list instead of re-deriving overlay geometry from raw physics state.

`nuka_scene_demo` is the current runnable debug render path. It imports a scene,
builds the compiled runtime views, steps the runtime instance with fixed-step
simulation, synchronizes simulated poses to render/debug views, emits the debug
draw command list, and rasterizes it through the headless renderer to a PPM
image. This keeps the demo usable in CI while preserving the same command source
that a native OpenGL/ImGui viewport will consume.

## Domain Modules

### Rigid Body (`runtime/rigid/`)

- **Integrator**: Symplectic Euler integration of gravity, forces, and velocities.
  Provides `IntegrateGravity`, `IntegrateForces`, `IntegrateVelocity`, and the
  combined `StepBody` entry point.

### Articulation (`runtime/articulation/`)

- **Joint Constraints**: Builds constraint blocks for revolute, prismatic, and
  fixed joints. Each joint produces a `ConstraintBlock` consumed by the solver.
  Revolute blocks store the local parent and child anchor offsets so the solver
  can project assembled joints at the position level, not only damp relative
  velocity drift.
- **Joint Drives**: PD-style drive targets that inject torques into articulated
  bodies each frame.

## Solver Pipeline

Each simulation step follows this pipeline:

1. **Gravity integration** -- apply gravitational acceleration to all dynamic bodies.
2. **Force integration** -- apply external forces and torques.
3. **Broadphase** -- identify potentially colliding pairs.
4. **Narrow-phase / contact generation** -- compute contact manifolds.
5. **Constraint assembly** -- collect contact and joint constraint blocks.
6. **Iterative solve** (PGS) -- resolve contact, joint, and drive rows,
   producing velocity corrections.
7. **Position projection** -- apply Baumgarte contact correction and
   mass/inertia-weighted joint anchor projection. Joint projection computes
   world-space parent/child anchor separation, distributes linear correction by
   inverse mass, and applies angular correction through diagonal inverse inertia
   for eccentric anchors. Static bodies keep zero inverse mass/inertia and do
   not move.
8. **Velocity integration** -- advance positions and orientations.
9. **Sensor update** -- read joint states, compute IMU/lidar readings.

## PHI Layer

The Platform Hardware Interface (PHI) abstracts GPU compute and memory management.
When a CUDA-capable device is available the solver and broadphase can offload work
to the GPU via PHI kernels. On CPU-only builds the same API falls back to a
single-threaded reference implementation.

Key PHI concepts:

- **Device / Context** -- enumerate and select compute devices.
- **Buffer** -- typed GPU memory with upload / download helpers.
- **Kernel** -- compute shader dispatch with grid/block configuration.
- **Capabilities** -- runtime feature queries (shared memory size, warp width, etc.).
