# Nuka Physics v0.7 – Phase 8: Newton-on-Summed-SDF Contact + Analytical Adjoint

> **Master plan reference:** §3 Round 7 (Newton-on-summed-SDF + analytical adjoint)
> **Prerequisites:** v0.7 Phase 5 (cross-system query), Phase 7 (sparse SDF)
> **Blocks:** v0.7 Phase 11 (coupling row classes) — K2 coupling uses this contact
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Implement MuJoCo 3.0-style contact resolution: **Newton's method on the sum of two SDFs** to find contact point + normal + penetration depth.

For a particle (point) vs rigid SDF, contact is trivial (sample at the point). For SDF-vs-SDF (rigid-rigid via SDFs, or articulated-link-vs-rigid), find the contact point by minimizing `φ_a(p) + φ_b(p)` via Newton iteration.

This phase also delivers the **analytical adjoint** of the SDF query and Newton iteration — required by master plan §3 Round 7 "full analytical adjoint, no finite-difference shortcuts."

## Tech Stack

- CUDA 12+
- Phase 5 (cross-system candidate lists)
- Phase 7 (sparse SDF query + gradient)
- v0.5 Phase 1 (adjoint codegen — extends with this contact form)

## Files to Create

- `src/collision/sdf_contact.hpp`
- `src/collision/sdf_contact.cu` — Newton-on-summed-SDF kernel
- `src/collision/sdf_contact_adjoint.cu` — reverse-mode adjoint
- `tools/codegen/classes/rigid_sdf_contact.yaml` — `RigidSDFContactRow` IR
- `tools/codegen/classes/featherstone_sdf_contact.yaml` — `FeatherstoneSDFContactRow` IR
- `tools/codegen/templates/derivatives/rigid_sdf_contact_adjoint_body.cu.j2`
- `tools/codegen/templates/derivatives/featherstone_sdf_contact_adjoint_body.cu.j2`
- `tests/collision/test_sdf_contact_known_geometry.cpp` — sphere/box pairs vs analytical
- `tests/collision/test_sdf_contact_adjoint_fd.cpp` — V3 FD check on the new row class
- `tests/collision/test_sdf_contact_thin_shell_robustness.cpp`

## Files to Modify

- `src/constraint/row_builder.cpp` — emit `RigidSDFContactRow` for K2 coupling candidates
- `tools/codegen/regen.py` — generate new row class evaluators
- `src/include/nuka/nuka.h` — no public API change; internal

## Tasks

### Task 7.8.1 — Newton-on-summed-SDF algorithm

For rigid-rigid SDF contact (both via Phase 7 SDFs), find `p* = argmin (φ_a(p) + φ_b(p))`:

```cuda
__device__ bool find_sdf_contact_newton(
    const SparseSdfDevice& sdf_a, const Transform& world_to_local_a,
    const SparseSdfDevice& sdf_b, const Transform& world_to_local_b,
    float3 initial_guess,
    /* output */
    float3& contact_point_world,
    float3& contact_normal_world,
    float& penetration_depth)
{
    float3 p = initial_guess;
    for (int iter = 0; iter < MAX_NEWTON; ++iter) {
        // Sample both SDFs in their local frames
        float3 grad_a, grad_b;
        float phi_a = sparse_sdf_sample(sdf_a, world_to_local_a * p, grad_a);
        float phi_b = sparse_sdf_sample(sdf_b, world_to_local_b * p, grad_b);

        float3 grad_sum = grad_a + grad_b;
        // Newton step: p -= H^{-1} grad_sum
        // Hessian H of (φ_a + φ_b) approximated via finite difference or assumed identity
        // Simplest: gradient descent with adaptive step
        float step_size = 0.01f * voxel_size;
        p -= step_size * grad_sum;

        if (length(grad_sum) < 1e-4f) break;   // converged
    }
    // Contact point is the converged p; normal is grad_a - grad_b normalized
    contact_point_world = p;
    contact_normal_world = normalize(grad_a - grad_b);
    penetration_depth = -(sample_a + sample_b);    // negative because both signed-distance go negative inside
    return penetration_depth > 0;
}
```

For particle-rigid contact (Phase 9 / 10 will use this case heavily): no Newton needed — directly sample rigid SDF at particle position; contact normal = SDF gradient; penetration = max(0, -φ - particle_radius).

### Task 7.8.2 — Analytical adjoint of SDF sample

`src/collision/sdf_contact_adjoint.cu`:

The forward: `φ(p) = trilerp(SDF cells, p_local)`. The adjoint w.r.t. `p_world`:

```
∂φ/∂p_world = ∂φ/∂p_local · ∂p_local/∂p_world
            = grad(φ at p_local) · R_world_to_local

∂φ/∂R = grad(φ) ⊗ (p_world - origin)    # through rigid transform
∂φ/∂t = grad(φ)                          # through rigid translation
```

These derivatives are written into the generated adjoint kernel by codegen. Reference: PhysX 5 DiffSim's SDF adjoint approach.

For Newton iteration, the implicit function theorem gives the gradient of the converged contact point w.r.t. inputs in **one** linear solve (rather than back-propping through every Newton step). Master plan Round 5 path 3 (IFT-at-convergence) covers this; Newton-on-summed-SDF qualifies because it converges to high precision in 5-10 iterations.

### Task 7.8.3 — Row IR for RigidSDFContactRow

`tools/codegen/classes/rigid_sdf_contact.yaml`:

```yaml
row_class_id: 4   # extends v0.1 catalog
row_class_name: RigidSDFContactRow

body_count_mode: fixed
body_count: 2     # rigid + particle OR rigid + rigid

jacobian_kind: maximal_6vec
max_rows_per_block: 6
supports_friction: true
supports_compliance: false
default_gradient_mode: stop_grad_on_event   # β contact strategy
default_recompute_mode: ift_at_convergence  # Newton converges
constraint_kind: unilateral_with_friction

forward_evaluator:
  custom_kernel: rigid_sdf_contact_forward
  inputs: [rigid_pose, sdf_handle, particle_position, particle_radius]
  outputs: [contact_point, contact_normal, penetration]

adjoint_evaluator:
  custom_kernel: rigid_sdf_contact_adjoint
  reverse_dependencies: [rigid_pose, sdf_gradient, ...]
  gradient_mode_override: ift_at_convergence
  derivative_rules:
    - op: sdf_trilinear_sample
      rule: trilerp_gradient
    - op: rigid_transform
      rule: rigid_jacobian
```

### Task 7.8.4 — Row IR for FeatherstoneSDFContactRow

Same as above but instead of `rigid_pose`, the Jacobian goes through the **Featherstone-generalized Jacobian** (Phase 6 of v0.1). The adjoint propagates through `world_to_link` transform back to joint positions along the kinematic chain.

```yaml
row_class_name: FeatherstoneSDFContactRow
body_count_mode: variable           # 1 link's worth of joints along the chain
jacobian_kind: featherstone_chain_scalar
adjoint_evaluator:
  custom_kernel: featherstone_sdf_contact_adjoint
```

### Task 7.8.5 — Build pipeline

Cross-system query (Phase 5) → for each particle-rigid candidate → call SDF contact kernel → emit row.

If penetration > 0: row gets added to `RowBuffers` with computed Jacobians.
If penetration ≤ 0: skip; no contact.

### Task 7.8.6 — V3 FD validation

`tests/collision/test_sdf_contact_adjoint_fd.cpp`:

Per v0.5 Phase 1 V3 contract: every new row class passes FD validation.

```cpp
TEST(AdjointFd, RigidSdfContactRow_RelErrUnder1eM3) {
    nuka::codegen::v3::RowClassFdValidator v(ROW_CLASS_ID_RIGID_SDF_CONTACT, ctx);
    v.GenerateTestCases(/*count=*/100, /*seed=*/42);
    auto r = v.Run();
    EXPECT_LT(r.max_rel_err, 1e-3f);
}
```

### Task 7.8.7 — Thin-shell robustness

`tests/collision/test_sdf_contact_thin_shell_robustness.cpp`:

```cpp
// Thin panel (0.5mm) + particle approaching from one side
// Verify contact normal correctly points away from the panel
// Verify particle stopped at the surface (not punched through to other side)
```

Newton's method must not pull the particle through; clamp Newton step magnitude to voxel_size.

## Validation

- Sphere-sphere SDF contact agrees with analytical sphere-sphere within 1e-4 m.
- Sphere-box SDF contact normal within 0.5° of analytical.
- Thin-shell robustness: particle does not punch through.
- Adjoint FD passes < 1e-3 rel error.
- Featherstone SDF contact agrees with corresponding maximal-link path on simple chains.
- Per-step contact resolution time < 0.5 µs / contact pair on RTX 4090.

## Exit Criteria for v0.7 Phase 8

1. Newton-on-summed-SDF kernel operational.
2. `RigidSDFContactRow` + `FeatherstoneSDFContactRow` row classes registered + codegen produces forward + adjoint kernels.
3. V3 FD check passes for both new row classes.
4. Thin-shell robustness validated.
5. Performance budget met.
6. Lint + determinism tests pass.

## What This Phase Does Not Do

- No XPBD compliance handling (Phase 9).
- No particle-particle contact (Phase 11).
- No fluid pressure on rigids (Phase 11 / 10 interaction).
- No CCD (continuous collision detection); separate future phase.
