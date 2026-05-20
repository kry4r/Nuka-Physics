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

Production simulation is selected through PHI and must prefer the CUDA backend
on this workstation. `phi::ResolvePhysicsBackend()` is the API boundary for that
selection layer: the default policy resolves to CUDA when a device is available,
while the CPU path is explicitly marked as reference-only for deterministic
validation, differential checks, and host-side debugging.

`runtime::StepWorldInstance()` is the current fixed-step CPU reference stepper.
It advances a `WorldInstance` against its `WorldTemplate` by converting each
body row into the rigid `BodyState`, applying gravity and accumulated
forces/torques, building shape proxies from cooked box/sphere/capsule/plane
data, running the dynamic broadphase over shape AABBs, generating contact
manifolds for plane, sphere, and box-style contacts, assembling contact
constraints, adding cooked joint constraints and actuator drive rows, invoking
the PGS solver, then integrating velocities and writing poses back. Cooked body
poses are stored in world space so imported hierarchies start simulation from
the same transforms used by `SceneGraph`.

`StepWorldInstance()` returns a `WorldStepReport` with step count, broadphase
pairs, contact manifolds/points, constraint blocks/rows, solver iterations, and
maximum constraint error. Tests and demos use the report to prove the imported
scene path is exercising the physical contact pipeline instead of only advancing
free bodies. The next production stepper must preserve the same report contract
while moving broadphase, narrowphase, constraint assembly, solver iterations,
and integration into CUDA-resident buffers/kernels.

Cooked joints preserve parent and child frames from `SceneIR`, so runtime joint
projection uses authoring anchors instead of assuming every joint is body-center
to body-center. Revolute joints currently map to the five-row maximal-coordinate
constraint builder; fixed joints add the sixth rotational lock row. Prismatic,
spherical, and free joints are routed through the same revolute-style anchor
projection until their specialized rows are implemented. Cooked velocity,
motor, force, and position actuator records emit drive constraint rows against
their target joint; velocity-style actuators interpret `gain` as target velocity
and `force_limit` as the clamp.

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
blocks, and emits a renderer-independent `DebugDrawList`. The bridge covers
collision shape proxies, shape AABBs, joint axes, centers of mass, contact
points/normals, and constraint error vectors. Native shells and renderers should
consume this command list instead of re-deriving overlay geometry from raw
physics state.

`nuka_scene_demo` is the current runnable debug render path. It imports a scene,
builds the compiled runtime views, steps the runtime instance with the reference
stepper, synchronizes simulated poses to render/debug views, emits the debug
draw command list, and rasterizes it through the headless renderer to a PPM
image. This keeps the demo usable in CI while preserving the same command source
that the Vulkan viewport/backend will consume.

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

### Constraint and Contact Solver (`constraint/`, `solver/`)

- **Contact manifolds**: Store contact points, warm-start impulses, friction,
  and restitution parameters before constraint assembly.
- **Contact blocks**: Build normal rows plus two tangent rows per manifold. The
  solver updates tangent limits from `friction * accumulatedNormalImpulse`, so
  tangential impulses stay inside a Coulomb friction cone and disappear when no
  normal impulse is produced.
- **Position error**: Contact penetration is stored separately from velocity
  `rhs` in `ConstraintBlock::position_error`. This lets resting contacts project
  overlap without turning penetration depth into artificial bounce velocity.
- **Restitution**: Before velocity iterations, contact normal rows convert
  closing velocity into a bounce target using the block restitution coefficient.
  This keeps the material response explicit in the same PGS path as resting
  contacts.

## Solver Pipeline

Each simulation step follows this pipeline:

1. **Gravity integration** -- apply gravitational acceleration to all dynamic bodies.
2. **Force integration** -- apply external forces and torques.
3. **Broadphase** -- identify potentially colliding pairs.
4. **Narrow-phase / contact generation** -- compute contact manifolds.
5. **Constraint assembly** -- collect contact and joint constraint blocks.
6. **Actuator assembly** -- convert cooked actuator records into drive rows.
7. **Iterative solve** (PGS) -- resolve contact, friction, restitution, joint,
   and drive rows, producing velocity corrections.
8. **Position projection** -- apply Baumgarte contact correction and
   mass/inertia-weighted joint anchor projection. Joint projection computes
   world-space parent/child anchor separation, distributes linear correction by
   inverse mass, and applies angular correction through diagonal inverse inertia
   for eccentric anchors. Static bodies keep zero inverse mass/inertia and do
   not move.
9. **Velocity integration** -- advance positions and orientations.
10. **Sensor update** -- read joint states, compute IMU/lidar readings.

## PHI Layer

The Platform Hardware Interface (PHI) abstracts physics backend selection, GPU
compute, and memory management. The default selection policy is
`PreferCuda`: on this workstation production physics resolves to CUDA. The CPU
backend remains available only as a reference path for validation, not as the
performance target.

Key PHI concepts:

- **Device / Context** -- enumerate and select compute devices.
- **Backend selection** -- resolve CUDA production or CPU reference execution
  before constructing runtime device state.
- **Buffer** -- typed GPU memory with upload / download helpers.
- **Kernel** -- compute shader dispatch with grid/block configuration.
- **Capabilities** -- runtime feature queries (shared memory size, warp width, etc.).

## Vulkan Rendering Contract

The production renderer backend is Vulkan. `render::ProbeVulkanRenderer()`
creates a Vulkan instance and enumerates physical devices so tests can prove the
required graphics path is available on the workstation. The existing headless
debug rasterizer is still useful for deterministic CI artifacts, but it is not
the production renderer target.
