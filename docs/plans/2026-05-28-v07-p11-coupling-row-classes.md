# Nuka Physics v0.7 – Phase 11: Cross-System Coupling Row Classes (K2 + K3)

> **Master plan reference:** §3 Round 7 (K2 + K3 hybrid coupling) + §4 IR catalog extensions
> **Prerequisites:** v0.7 Phases 8 (SDF contact), 9 (XPBD), 10 (PBF)
> **Blocks:** v0.7 Phase 16 (H1 grasp demo needs full coupling)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Implement the remaining coupling row classes that wire rigid ↔ soft ↔ fluid interactions through the unified row scheduler. Master plan §3 Round 7 routing table:

| Pair | Path | Row class |
|---|---|---|
| rigid ↔ soft particle | K2 SDF | `RigidSDFContactRow` (Phase 8 already) |
| rigid ↔ fluid particle | K2 SDF | `RigidSDFContactRow` (Phase 8 already) |
| Featherstone link ↔ particle | K2 pose-aware SDF | `FeatherstoneSDFContactRow` (Phase 8 already) |
| soft ↔ fluid particle | **K3** cross-system row | `ParticleParticleContactRow` |
| soft ↔ soft (incl. self) | **K3** same-system row | `ParticleParticleContactRow` |
| cloth zero-thickness | special | `TrianglePointContactRow` |

Phase 8 already shipped rigid-particle. This phase ships `ParticleParticleContactRow` and `TrianglePointContactRow`, plus full analytical adjoints for both.

## Tech Stack

- CUDA 12+
- Phase 5 (particle grid + cross-system query)
- Phase 8 (Newton-on-summed-SDF baseline)
- Phase 9 (XPBD with compliance for soft constraint enforcement)

## Files to Create

- `tools/codegen/classes/particle_particle_contact.yaml`
- `tools/codegen/classes/triangle_point_contact.yaml`
- `tools/codegen/templates/derivatives/particle_particle_contact_adjoint_body.cu.j2`
- `tools/codegen/templates/derivatives/triangle_point_contact_adjoint_body.cu.j2`
- `src/collision/particle_particle_contact.cu`
- `src/collision/triangle_point_contact.cu`
- `src/runtime/world_stepper.cpp` — wire all 6 coupling pairs through one pass
- `tests/collision/test_particle_particle_contact_fd.cpp`
- `tests/collision/test_triangle_point_contact_fd.cpp`
- `tests/regression/test_cloth_self_collision.cpp`
- `tests/regression/test_soft_fluid_two_way.cpp`
- `tests/regression/test_rigid_cloth_two_way.cpp`

## Tasks

### Task 7.11.1 — ParticleParticleContactRow

For two particles `i, j` of radii `r_i, r_j`:

```
C(p_i, p_j) = |p_i - p_j| - (r_i + r_j)
```

Unilateral: only active when overlap (`C < 0`).

Friction: tangential restitution similar to rigid contacts; lambda_friction bounded by `mu * |lambda_normal|`.

IR (`particle_particle_contact.yaml`):

```yaml
row_class_id: 6
row_class_name: ParticleParticleContactRow

body_count_mode: fixed
body_count: 2     # particle i + particle j
jacobian_kind: maximal_6vec    # particles have only position; treat as 3-DOF
supports_friction: true
supports_compliance: true
default_gradient_mode: stop_grad_on_event   # β
default_recompute_mode: checkpoint
constraint_kind: unilateral_with_friction

forward_evaluator:
  inputs: [position_i, position_j, radius_i, radius_j, inv_mass_i, inv_mass_j, lambda]
  outputs: [lambda, delta_p_i, delta_p_j]

adjoint_evaluator:
  derivative_rules:
    - op: particle_distance_constraint
      rule: particle_particle_adjoint
```

### Task 7.11.2 — TrianglePointContactRow (cloth zero-thickness)

Cloth particles are arranged in triangles. A particle-vs-triangle (or triangle-vs-particle) test gives signed distance:

```
phi = (p - closest_point_on_triangle) · triangle_normal
```

If `|phi| < cloth_thickness/2`: contact.

IR:

```yaml
row_class_id: 7
row_class_name: TrianglePointContactRow

body_count_mode: fixed
body_count: 4    # 3 triangle vertices + 1 point
jacobian_kind: maximal_6vec    # particles
supports_friction: true
supports_compliance: false
default_gradient_mode: stop_grad_on_event
constraint_kind: unilateral_with_friction

forward_evaluator:
  inputs: [tri_v0, tri_v1, tri_v2, point, thickness, lambda]
  outputs: [lambda, normal, penetration]

adjoint_evaluator:
  reverse_dependencies: [vertices, normal_construction]
  derivative_rules:
    - op: triangle_point_distance
      rule: tri_point_jacobian   # standard barycentric + normal chain
```

Used heavily for cloth self-collision (H1's towel folds onto itself).

### Task 7.11.3 — Coupling broadphase

Particle-particle pairs come from Phase 5's particle grid (same-system or cross-system). The broadphase already lists candidate pairs; this phase just routes them through the contact resolver.

Triangle-point pairs: each cloth triangle's bounding sphere is in the particle grid; query particles near triangle's bound and check distance. Use the existing particle uniform grid with triangle-as-bound query.

### Task 7.11.4 — Row builder integration

`src/constraint/row_builder.cpp` extended:

```cpp
void RowBuilder::BuildCouplingRows() {
    // particle-particle contacts (soft-soft, soft-fluid, fluid-fluid is internal PBF)
    for (auto [i, j] : particle_grid_pairs_) {
        if (overlap_check(i, j)) EmitParticleParticleContactRow(i, j);
    }
    // triangle-point cloth self-collision
    for (auto [tri_idx, point_idx] : cloth_triangle_point_pairs_) {
        EmitTrianglePointContactRow(tri_idx, point_idx);
    }
}
```

### Task 7.11.5 — End-to-end coupling test

`tests/regression/test_rigid_cloth_two_way.cpp`:

Scenario:
- Rigid cube falls onto cloth supported at four corners.
- Cube should compress cloth (rigid pushes cloth particles via `RigidSDFContactRow`).
- Cloth should deform around cube (XPBD distance constraints + `TrianglePointContactRow` for self-collision).
- Cube should rest on cloth (two-way coupling: cloth applies reaction force to cube).

Verify: cube velocity decays; cube settles; no interpenetration.

### Task 7.11.6 — V3 FD validation

For both new row classes:

```cpp
TEST(AdjointFd, ParticleParticleContact_RelErrUnder1eM3) { ... }
TEST(AdjointFd, TrianglePointContact_RelErrUnder1eM3) { ... }
```

### Task 7.11.7 — Cross-system stability tuning

The coupling between subsystems often shows numerical instability — particles can "jitter" at coupling boundaries.

Mitigations:
- Sub-stepping: when XPBD compliance is high or PBF density iteration count is low, use 2-4 sub-steps per main step.
- Lambda warm-starting across sub-steps.
- Clamping particle position correction to a per-step maximum.

These are tuning knobs in the `SolveConfig`.

## Validation

- Rigid cube on cloth: stable contact, no jitter, no penetration.
- Cloth self-collision: cloth folds without intersecting itself.
- Soft body in fluid: soft body buoyancy approximately correct.
- 6 coupling pairs from §3 Round 7 routing table all work.
- V3 FD passes for new row classes.
- D1 determinism preserved.

## Exit Criteria for v0.7 Phase 11

1. `ParticleParticleContactRow` + `TrianglePointContactRow` registered.
2. Codegen forward + adjoint kernels for both.
3. Six coupling pair types from master plan §3 Round 7 routing table all functional.
4. Two-way coupling tests pass (rigid-cloth, cloth self, soft-fluid, etc.).
5. V3 FD validation for new row classes.
6. Stability tuning surface exposed via `SolveConfig`.

## What This Phase Does Not Do

- No deformable / fluid sensor outputs beyond raw particle / vertex arrays.
- No fluid-cloth surface tension special handling (PBF + cloth particle is sufficient).
- No tearing / fracture (future phase).
- No CCD; assume small enough dt.
