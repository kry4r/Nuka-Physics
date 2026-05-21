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
free bodies. CUDA production steppers must preserve the meaningful parts of the
same reporting contract while moving broadphase, narrowphase, constraint
assembly, solver iterations, and integration into CUDA-resident
buffers/kernels.

`runtime::gpu::StepCudaWorld()` operates on `DeviceWorld` state buffers and
launches a fixed-step rigid integration kernel for gravity, external
forces/torques, linear/angular velocity updates, pose integration, and
accumulator clearing. Differential tests compare the CUDA result against the CPU
reference for free-fall and forced-body scenes with contacts disabled. The CPU
path remains validation-only while CUDA carries the production broadphase,
contact generation, constraint assembly, PGS solving, and sensor query stages.

`collision::gpu::BuildCudaBroadphase()` and
`constraint::gpu::GenerateCudaContacts()` are the current CUDA collision stage.
They consume CUDA-resident `DeviceWorld` state and cooked shape tables, generate
shape AABBs on device, fill deterministic pair slots for overlapping shapes,
and emit plane, sphere-sphere, and box-style contact manifolds into device
buffers. Host downloads are limited to validation reports, regression checks,
and timing tests; the intended production path keeps broadphase and contact
data resident for the upcoming CUDA constraint assembly and solver stage.

`solver::gpu::SolveCudaConstraints()` extends that resident path through
constraint assembly and deterministic PGS. It consumes device contact manifolds,
cooked joint tables, and cooked actuator tables, builds contact/joint/drive
constraint blocks into CUDA buffers, precomputes row effective mass, applies
friction and restitution in the velocity solve, and performs contact plus joint
position projection against CUDA pose buffers. `SolveCudaConstraintBlocks()` is
kept as a validation/tooling entry for uploading a small host-authored row set
into the same device solver; production scenes should enter through
`DeviceWorld` plus device contact results.

CUDA joint and drive assembly accepts single-ended joints where either parent or
child is `scene::kInvalidBody`. Those rows represent a static world anchor and
are solved the same way as the CPU reference path: invalid bodies contribute
zero inverse mass and inertia but the valid body still receives velocity and
position corrections. This is required for MJCF-style world-hinged bodies and
for imported actuators attached to those joints.

`sensor::gpu::QueryCudaImuSensor()` and `sensor::gpu::QueryCudaLidarSensor()`
are the CUDA production sensor entry points. They consume the same
CUDA-resident `DeviceWorld` that simulation updates, so IMU/state samples read
pose, angular velocity, force, and inverse mass from device buffers and lidar
casts deterministic fan rays against cooked sphere, box-style, and plane
geometry without invoking the CPU sensor helpers. Result downloads are compact
validation/report boundaries for tests, demos, and tooling; CPU `sensor`
queries remain reference-only helpers.

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

`BatchContext` and `BatchScheduler` describe how multiple `WorldInstance`
objects share a common `WorldTemplate`. They remain CPU-side orchestration and
reference metadata; they are not the production execution path.

`runtime::gpu::BatchedDeviceWorld` is the CUDA production container for
multi-environment stepping. It uploads one shared template's body inverse mass,
inverse inertia, and shape tables, flattens per-instance pose, velocity, force,
and torque arrays into GPU buffers, and launches one CUDA thread per flattened
body for rigid integration. `StepBatchedCudaWorld()` maps `flat_index %
body_count_per_instance` back to the shared template body row so each instance
uses the same cooked physical parameters while mutating independent state.

The batched CUDA contact path extends the same layout through broadphase,
narrowphase, and contact solving. `BuildBatchedCudaBroadphase()` generates AABBs
for `instance_count * shape_count_per_instance` shapes and pair slots only
inside each instance, so environments never collide with one another.
`GenerateBatchedCudaContacts()` emits contact manifolds with a local
`instance_index`, and `SolveBatchedCudaConstraints()` assembles contact, cooked
joint, and actuator drive blocks using flattened body ids before solving PGS
rows against the batched device state. `StepBatchedCudaWorld()` can now run
batched rigid integration, contact generation/solve, joint anchor projection,
and velocity drive solve in one CUDA pipeline. This currently covers
plane/box/sphere contact scenes plus cooked revolute/fixed-style joint rows and
velocity drive rows; batched sensors and render synchronization remain the next
CUDA-first follow-on stages rather than CPU fallbacks.

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

`nuka_scene_demo` is the current runnable imported-scene debug render path. It
imports a scene, builds the compiled runtime views, resolves the physics backend
through PHI, and on this workstation runs fixed-step CUDA integration,
broadphase/contact generation, constraint solving, and imported IMU/frame-pose
sensor sampling before downloading the final state for `SceneGraph` /
`RenderScene` synchronization. The default render path converts the resulting
`DebugDrawList` into render-layer Vulkan debug
commands, runs `render::RenderDebugDrawListVulkan()` into an offscreen storage
image, reads back the RGBA8 pixels, and writes the PPM artifact from the Vulkan
image. The CPU headless rasterizer remains an explicit reference path. Explicit
CPU physics selection remains a reference-validation path only.

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
10. **Sensor update** -- sample IMU/state and lidar through CUDA sensor kernels
    from GPU-resident runtime buffers; CPU sensors are reference-only.

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

### CUDA DeviceWorld

`runtime::gpu::DeviceWorld` is the CUDA-resident runtime container. It uploads
cooked `WorldTemplate` tables into PHI buffers and keeps body, shape, joint, and
actuator counts next to the device allocations. It also owns mutable state
buffers for poses, linear velocities, angular velocities, forces, and torques.
Shape type, body binding, local transform, half extent, and radius tables are
also uploaded so CUDA kernels can build world-space collision proxies without
returning to CPU-side cooked data. Joint type, parent/child body, axis, frame,
actuator type, actuator joint id, gain, and force-limit tables are uploaded for
CUDA constraint assembly. Tests use compact summary downloads for upload
integrity and full state/contact/constraint readbacks for CPU-reference
differential validation. Full readback is a validation and tooling boundary;
production stepping keeps state on the GPU.

CUDA sensors reuse the mutable `DeviceWorld` state instead of introducing a
separate CPU-side sensor graph in the hot path. Scene import and cooking still
own sensor descriptors and body attachments; runtime code translates those
descriptors into device queries after the CUDA step/solve sequence. Lidar is
currently a deterministic direct ray loop over cooked shape tables, with the API
kept narrow so a later BVH or hardware ray tracing acceleration path can replace
the implementation without changing demos or tests.

### CUDA BatchedDeviceWorld

`runtime::gpu::BatchedDeviceWorld` stores multi-instance mutable state as
structure-of-arrays buffers laid out by instance-major order:

```text
flat_body = instance_index * body_count_per_instance + body_index
```

Template data such as inverse mass and inverse inertia is uploaded once and
indexed by `body_index`. Shared shape, joint, and actuator tables are also
uploaded once per template and assembled into per-instance constraint blocks by
CUDA kernels. Mutable pose, velocity, force, and torque are uploaded per
flattened body. `DownloadState()` is a validation and tooling boundary; the
production path keeps batched state resident for the next CUDA stage.

The build exposes `NK_CUDA_ARCHITECTURES`, defaulting to `native`, and forces
`CMAKE_CUDA_ARCHITECTURES` from that cache variable. This keeps production
kernels compiled for the workstation GPU instead of inheriting stale CMake cache
values such as an older `sm_75` target.

## Vulkan Rendering Contract

The production renderer backend is Vulkan. `render::ProbeVulkanRenderer()`
creates a Vulkan instance and enumerates physical devices so tests can prove the
required graphics path is available on the workstation.

`render::RenderDebugDrawListVulkan()` is the current executable Vulkan render
path. It creates a swapchain-free Vulkan instance/device/compute queue, uploads
debug draw commands to a storage buffer, rasterizes line/sphere/capsule/box/AABB
commands in a GLSL compute shader into an RGBA8 storage image, copies that image
to a host-visible staging buffer, and returns pixels plus non-background counts
for artifact generation and regression checks. This offscreen path is deliberately
decoupled from windowing so CI can validate real Vulkan rendering while the
future interactive viewport reuses the same render/debug handoff.

The existing CPU headless debug rasterizer is still useful for deterministic
reference artifacts, but it is not the default production renderer target.
