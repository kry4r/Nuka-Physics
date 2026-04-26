# Runtime Overview

This document describes the runtime architecture of the Nuka Physics Engine.

## Core Abstractions

### WorldTemplate

A read-only, immutable description of a physics scene produced by the scene cooker.
Contains body tables (poses, masses, inertias), joint descriptors, collision shapes,
and sensor definitions. Multiple `WorldInstance` objects can share the same template
to enable batched simulation.

### WorldInstance

A mutable simulation state derived from a `WorldTemplate`. Holds per-body dynamic
state: positions, orientations, linear and angular velocities, forces, and torques.
Each instance evolves independently during simulation.

### BatchContext / BatchScheduler

Manages multiple `WorldInstance` objects that share a common `WorldTemplate`.
The `BatchScheduler` orchestrates parallel stepping of instances, enabling
domain-randomized training and large-scale evaluation scenarios.

## Domain Modules

### Rigid Body (`runtime/rigid/`)

- **Integrator**: Symplectic Euler integration of gravity, forces, and velocities.
  Provides `IntegrateGravity`, `IntegrateForces`, `IntegrateVelocity`, and the
  combined `StepBody` entry point.

### Articulation (`runtime/articulation/`)

- **Joint Constraints**: Builds constraint blocks for revolute, prismatic, and
  fixed joints. Each joint produces a `ConstraintBlock` consumed by the solver.
- **Joint Drives**: PD-style drive targets that inject torques into articulated
  bodies each frame.

## Solver Pipeline

Each simulation step follows this pipeline:

1. **Gravity integration** -- apply gravitational acceleration to all dynamic bodies.
2. **Force integration** -- apply external forces and torques.
3. **Broadphase** -- identify potentially colliding pairs.
4. **Narrow-phase / contact generation** -- compute contact manifolds.
5. **Constraint assembly** -- collect contact and joint constraint blocks.
6. **Iterative solve** (PGS) -- resolve constraints, producing velocity corrections.
7. **Velocity integration** -- advance positions and orientations.
8. **Sensor update** -- read joint states, compute IMU/lidar readings.

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
