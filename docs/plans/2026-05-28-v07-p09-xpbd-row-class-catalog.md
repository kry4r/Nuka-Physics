# Nuka Physics v0.7 – Phase 9: XPBD Row Class Catalog (Distance / Bend / Volume / Shape-Match)

> **Master plan reference:** §3 Round 3 (Path B variant — XPBD for soft) + §4 IR catalog extensions
> **Prerequisites:** v0.7 Phase 5 (particle grid for neighbor lookup), v0.5 Phase 1 (adjoint codegen)
> **Blocks:** v0.7 Phase 11 (coupling rows reference XPBD soft) + Phase 16 (H1 grasp demo uses cloth)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Implement XPBD (Extended Position-Based Dynamics, Macklin et al. 2016) for soft body, cloth, and muscle. Four new row classes are introduced into the unified scheduler.

XPBD's `compliance_alpha` field (already in the v0.1 Row IR) is now actively used: `α = 0` means hard constraint (PGS-equivalent), `α > 0` adds compliance proportional to material stiffness.

This phase delivers row classes and forward kernels for cloth and soft body work in S2 (H1 wrings towel; H1 grasps soft items). PBF density is similar but lives in a separate row class because of its variable neighborhood (Phase 10).

## Tech Stack

- CUDA 12+
- Existing CSR row + codegen + island/coloring scheduler
- Particle grid from Phase 5 (for cloth self-collision queries)
- Reference: Macklin, Müller "XPBD: Position-Based Simulation of Compliant Constrained Dynamics" (2016)

## Files to Create

- `tools/codegen/classes/xpbd_distance.yaml`
- `tools/codegen/classes/xpbd_bend.yaml`
- `tools/codegen/classes/xpbd_volume.yaml`
- `tools/codegen/classes/xpbd_shape_match.yaml`
- `tools/codegen/templates/derivatives/xpbd_*_adjoint_body.cu.j2`
- `src/runtime/soft/xpbd_world.hpp` — XPBD-specific runtime state (particle positions / velocities / inverse masses)
- `src/runtime/soft/xpbd_world.cu`
- `src/runtime/soft/cloth_topology.hpp` — distance/bend constraint topology builder
- `src/runtime/soft/cloth_topology.cpp`
- `src/runtime/soft/tetmesh_topology.hpp` — distance/volume constraint topology for tet meshes
- `src/runtime/soft/tetmesh_topology.cpp`
- `src/import/cooker/xpbd_cooker.cpp` — USD/MJCF soft body → XPBD constraints
- `tests/runtime/test_xpbd_distance_oracle_vellum.cpp` — V1 oracle vs Vellum
- `tests/runtime/test_xpbd_bend_cloth_simulation.cpp`
- `tests/runtime/test_xpbd_volume_tet_simulation.cpp`
- `tests/runtime/test_xpbd_shape_match_softbody.cpp`

## Files to Modify

- `src/constraint/row_builder.cpp` — emit XPBD rows from cloth / soft topologies
- `src/runtime/world_stepper.cpp` — XPBD integration step (predict positions, project constraints, correct velocities)
- `src/import/usda_importer.cpp` / `mjcf_importer.cpp` — recognize soft body / cloth flags + MaterialX compliance attributes

## Tasks

### Task 7.9.1 — XPBD core integration loop

`src/runtime/world_stepper.cpp` (extended):

```cpp
void WorldStepper::Step(float dt) {
    // 1. Featherstone ABA computes joint accelerations (rigid path, unchanged from v0.1 p06)

    // 2. XPBD predict positions:
    //    p_pred = p + v * dt + (gravity + applied_force) * dt²
    xpbd_world_.PredictPositions(dt);

    // 3. Build all constraint rows (rigid contact, joint, drive, XPBD distance/bend/volume, etc.)
    row_builder_.Build(buffers_, /* sources */);

    // 4. Solve via row scheduler (PGS for hard, XPBD compliance for soft)
    row_solver_.Solve(ctx_, buffers_, body_state_, solve_cfg_);

    // 5. XPBD correct velocities:
    //    v = (p_corrected - p) / dt
    xpbd_world_.CorrectVelocities(dt);

    // 6. Integrate (free rigid + articulation as before)
    integrator_.Advance(dt);
}
```

### Task 7.9.2 — XPBD distance constraint row

Forward:
```
C(p1, p2) = |p1 - p2| - rest_length
α̃ = α / dt²
Δλ = (-C - α̃ λ) / (∇C · ∇C / m + α̃)
p1 += w1 ∇C Δλ
p2 -= w2 ∇C Δλ
```

Codegen IR `xpbd_distance.yaml`:

```yaml
row_class_id: 5
row_class_name: XPBDDistanceRow

body_count_mode: fixed
body_count: 2
jacobian_kind: maximal_6vec    # via particle pseudo-bodies
supports_compliance: true       # uses row.compliance_alpha
default_gradient_mode: dense_adjoint
default_recompute_mode: checkpoint

forward_evaluator:
  inputs: [particle_a_pos, particle_b_pos, rest_length, compliance_alpha, inv_mass_a, inv_mass_b, lambda]
  outputs: [particle_a_pos_corr, particle_b_pos_corr, lambda]

adjoint_evaluator:
  reverse_dependencies: [particle_positions, lambda, rest_length, compliance_alpha]
  derivative_rules:
    - op: distance_constraint
      rule: distance_xpbd_adjoint
```

### Task 7.9.3 — XPBD bend constraint (cloth)

Quadratic bending energy across edges. Reference: Bender & Müller 2017 "Position-Based Simulation Methods in Computer Graphics."

Row class IR + adjoint similar to distance, but operates on 4 particles (two adjacent triangles sharing an edge).

```yaml
row_class_name: XPBDBendRow
body_count: 4
```

### Task 7.9.4 — XPBD volume constraint (tet mesh)

Volume preservation for tetrahedral elements: `C(p1,p2,p3,p4) = det(p2-p1, p3-p1, p4-p1) - V_rest`.

```yaml
row_class_name: XPBDVolumeRow
body_count: 4
```

### Task 7.9.5 — XPBD shape-matching (rigid-as-particle-cluster)

For very stiff XPBD soft bodies (or particles forming an approximately rigid body), the shape-matching constraint preserves cluster shape.

```yaml
row_class_name: XPBDShapeMatchRow
body_count_mode: variable
# variable count: # particles in the shape-match cluster
```

### Task 7.9.6 — Cloth topology builder

`src/runtime/soft/cloth_topology.cpp`:

Given a triangle mesh:
- Distance constraints on each edge.
- Bend constraints on each edge shared by two triangles.

Topology built once at cook time; runtime just iterates the precomputed constraint list.

### Task 7.9.7 — Tet topology builder

`src/runtime/soft/tetmesh_topology.cpp`:

Given a tet mesh (tetrahedron list):
- Distance constraints on tet edges.
- Volume constraint per tetrahedron.

### Task 7.9.8 — USD MaterialX schema for XPBD parameters

USD custom attributes:
```
custom float nuka:soft:stiffness = 1.0e4    # → compliance_alpha = 1/stiffness
custom float nuka:soft:damping = 0.01       # → damping_beta
custom token nuka:soft:type = "cloth"       # "cloth" / "softbody" / "shape_match"
```

Cooker reads these, sets `compliance_alpha = 1 / stiffness`, `damping_beta = damping`.

### Task 7.9.9 — V1 oracle for XPBD via Vellum

`tests/runtime/test_xpbd_distance_oracle_vellum.cpp`:

Generate Houdini Vellum reference trajectories for known cloth/tet test scenes (golden trajectory stored in Git LFS per master plan §5.4). Engine must agree to < 1% vertex position.

Vellum has parameters: stretch_stiffness, bend_stiffness, etc. Match by setting equivalent `compliance_alpha`.

### Task 7.9.10 — V3 FD validation for each new row class

Per v0.5 Phase 1 V3 contract:

```cpp
TEST(AdjointFd, XpbdDistance_RelErrUnder1eM3) { ... }
TEST(AdjointFd, XpbdBend_RelErrUnder1eM3) { ... }
TEST(AdjointFd, XpbdVolume_RelErrUnder1eM3) { ... }
TEST(AdjointFd, XpbdShapeMatch_RelErrUnder1eM3) { ... }
```

## Validation

- Single particle pair under XPBD distance with α > 0 oscillates with correct period.
- Cloth simulation matches Vellum golden trajectory within tolerance.
- Tet soft body matches Vellum golden within tolerance.
- All 4 new row classes pass V3 FD check.
- Per-step time: 200K-particle cloth step within 200 µs on RTX 4090.
- D1 determinism preserved through XPBD path.

## Exit Criteria for v0.7 Phase 9

1. Four XPBD row classes registered + codegen produces forward + adjoint kernels.
2. XPBD predict / correct integration wired into world stepper.
3. Cloth + tet topology builders + cookers operational.
4. USD MaterialX schema extended for soft body parameters.
5. Vellum oracle agreement on test scenes.
6. V3 FD check passes for all 4 new row classes.
7. Performance budget met.

## What This Phase Does Not Do

- No PBF density (Phase 10).
- No soft-soft or soft-fluid contact (Phase 11).
- No cloth self-collision (uses Phase 5 particle grid for broadphase; resolution in Phase 11).
- No XPBD muscle / actuator (future phase if H1 muscle modeling is added).
