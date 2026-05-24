# CUDA Coupling Tangent Friction Rows

## Goal

Move cooked `DeviceWorld` particle/rigid tangential friction from direct velocity
damping in the contact assembly kernel into the CUDA coupling row solver. The
normal row remains the primary contact row; each active coupling slot also owns
two tangent sub-rows whose impulses are clamped by `friction * normal_impulse`.

## Design

- Keep one CUDA coupling record per particle/contact slot, but extend the record
  with a tangent basis, tangent angular Jacobians, tangent effective masses,
  friction coefficient, and accumulated tangent impulses.
- The contact assembly/projection kernel computes contact point, normal,
  tangent basis, normal effective mass, tangent effective masses, position
  error, and normal warm-start. When CUDA row solving is enabled it no longer
  applies `ApplyFriction()` directly.
- `SolveParticleCouplingRowsKernel` sweeps one particle's fixed coupling slots
  in order. For each active slot it solves the normal row first, then solves two
  tangent rows using the updated particle velocity and rigid velocity. Tangent
  impulses are clamped to the Coulomb interval
  `[-friction * normal_impulse, friction * normal_impulse]`.
- CPU remains orchestration and validation only: it launches kernels and can
  download compact rows/reports for tests, demos, and benchmarks.

## Validation

- Add a runtime test that compares identical cooked box coupling scenes with
  friction disabled and enabled. The friction-enabled path must reduce particle
  tangential speed, report CUDA friction-row impulses, and expose non-zero
  tangent row impulse data in downloaded coupling rows.
- Extend the CUDA particle coupling timing checks to accumulate friction-row
  evidence under the existing one-second budgets.
- Extend `nuka_cuda_particle_demo` and the P0 regression matrix so the solver
  path is visible outside unit tests.

## Follow-On

- Warm-start tangent impulses across frames.
- Move the fixed per-particle coupling record into a general row buffer shared
  by rigid, cloth, deformable, and fluid constraints.
- Add graph coloring or island batching for deterministic cross-particle writes
  into shared rigid bodies.
