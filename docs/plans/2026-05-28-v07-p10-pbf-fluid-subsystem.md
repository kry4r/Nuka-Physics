# Nuka Physics v0.7 – Phase 10: PBF Fluid Subsystem (Density / Viscosity / Surface Tension)

> **Master plan reference:** §3 Round 3 (PBF chosen for fluid) + §3 Round 7 (PBF density internal)
> **Prerequisites:** v0.7 Phase 5 (particle grid), Phase 9 (XPBD infrastructure shared)
> **Blocks:** v0.7 Phase 11 (coupling rows for fluid-rigid + fluid-soft)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Implement Position-Based Fluids (PBF, Macklin & Müller 2013) as the primary fluid subsystem. PBF particles enforce a density constraint via iteration; this density constraint is **internal to the fluid subsystem** (does not emit into the universal row scheduler per master plan §3 Round 7).

Coupling rows (rigid-fluid, soft-fluid) come in Phase 11.

## Tech Stack

- CUDA 12+
- Particle grid from Phase 5
- XPBD-style position projection from Phase 9

## Files to Create

- `src/runtime/fluid/pbf_world.hpp`
- `src/runtime/fluid/pbf_world.cu`
- `src/runtime/fluid/pbf_density_iteration.cu` — main density correction loop
- `src/runtime/fluid/pbf_viscosity.cu` — XSPH viscosity correction
- `src/runtime/fluid/pbf_surface_tension.cu` — vorticity confinement / surface tension
- `src/runtime/fluid/pbf_kernels.cuh` — Poly6, Spiky, viscosity SPH kernels
- `src/import/cooker/fluid_cooker.cpp` — sample particle initial state from a volume mesh
- `tests/runtime/test_pbf_density_iteration.cpp`
- `tests/runtime/test_pbf_oracle_flex_paper.cpp` — V1 oracle vs Flex paper ball-into-water case
- `tests/runtime/test_pbf_volume_conservation.cpp`
- `tests/runtime/test_pbf_surface_tension.cpp`
- `tests/perf/test_pbf_1m_particles.cpp`

## Files to Modify

- `src/runtime/world_stepper.cpp` — wire PBF substep into step loop
- `src/import/usda_importer.cpp` — recognize fluid volume markers
- `tools/codegen/classes/` — register PBF as internal subsystem (not a Row class)

## Tasks

### Task 7.10.1 — PBF integration substep

PBF runs as a sub-loop inside the main physics step. Algorithm per Macklin & Müller:

```
for each particle i:
    apply external forces: v_i += dt * (gravity + applied_force_i)
    predict position: p_i_pred = p_i + dt * v_i

for iteration in [0, N_pbf]:
    build neighbor list via particle grid (Phase 5)
    for each particle i:
        compute density rho_i = sum_j (m_j * Poly6(r_ij, h))
        compute density constraint: C_i = rho_i / rho_0 - 1
        compute lambda_i = -C_i / (sum_k |∇C_i,k|² + epsilon)
    for each particle i:
        apply position correction:
            delta_p_i = (1/rho_0) sum_j (lambda_i + lambda_j) ∇W(r_ij, h)
            p_i_pred += delta_p_i

for each particle i:
    update velocity: v_i = (p_i_pred - p_i) / dt
    apply XSPH viscosity (Schechter & Bridson)
    apply surface tension correction (optional)
    update position: p_i = p_i_pred
```

Density iteration count `N_pbf` typically 3-5; converges quickly.

### Task 7.10.2 — SPH kernels

`src/runtime/fluid/pbf_kernels.cuh`:

```cuda
__device__ float poly6(float r2, float h);             // density estimation
__device__ float3 spiky_grad(float3 r, float r_len, float h);   // pressure gradient
__device__ float viscosity_laplacian(float r, float h);    // XSPH viscosity
```

Standard SPH kernels (Müller et al. 2003). Precompute coefficients at cook time.

### Task 7.10.3 — Density iteration kernel

`src/runtime/fluid/pbf_density_iteration.cu`:

```cuda
__global__ void pbf_compute_density_kernel(...);
__global__ void pbf_compute_lambda_kernel(...);
__global__ void pbf_apply_position_correction_kernel(...);
```

Each kernel iterates particles in parallel; reads neighbors via Phase 5 grid; writes density / lambda / position correction.

D1 determinism: per-particle output depends on neighbor list (sorted by neighbor index from Phase 5), so output is bit-exact.

### Task 7.10.4 — Viscosity (XSPH)

`src/runtime/fluid/pbf_viscosity.cu`:

```
v_i_corrected = v_i + c * sum_j (m_j * (v_j - v_i) * W(r_ij, h) / rho_j)
```

Optional but recommended for visually plausible fluid.

### Task 7.10.5 — Surface tension

Per Akinci et al. 2013:
- Cohesion force based on surface curvature.
- Optional in v0.7; add if needed for visual fidelity in S2 demos.

### Task 7.10.6 — Fluid cooker

`src/import/cooker/fluid_cooker.cpp`:

Given a volume mesh (USD `nuka:fluid_volume = "true"` flag): sample particle positions filling the volume on a uniform grid; compute particle mass from density × cell volume.

### Task 7.10.7 — Memory budget

S3 envelope: 1M particles.
- Position (3 floats), velocity (3 floats), predicted position (3 floats), mass (1 float): 40 bytes / particle = 40 MB
- Density (1), lambda (1), color (1): 12 bytes / particle = 12 MB
- Neighbor list (capped at 32 per particle): 1M × 32 × 4 bytes = 128 MB
- Total ~ 180 MB for 1M particles. Fits.

### Task 7.10.8 — V1 oracle via Flex paper

`tests/runtime/test_pbf_oracle_flex_paper.cpp`:

Reproduce the ball-into-water test from NVIDIA Flex paper (Macklin et al. 2014):
- 50K particles in a tank.
- Drop a ball.
- Compare splash profile + settling time + volume conservation to paper figures.

Tolerance per master plan §6 V1 table: volume conservation ±2%, surface waveform qualitative match.

### Task 7.10.9 — V2 invariant for fluid

Add particle count check (PBF should not gain or lose particles): `tests/diffsim/v2_invariants` extended to PBF subsystem.

## Validation

- Single drop falls and forms a stable pool.
- Flex paper ball-into-water case matches qualitatively.
- Volume conservation ±2% over 5 s.
- 1M particles step within 1.5 ms on RTX 4090 (relaxes S1 sub-ms because S2 is 1× realtime).
- Determinism: bit-exact across runs.

## Exit Criteria for v0.7 Phase 10

1. PBF substep operational; density iteration converges within configured count.
2. Viscosity + (optional) surface tension implemented.
3. Flex paper oracle agreement.
4. Volume conservation invariant holds.
5. 1M-particle perf within budget.
6. Cooker samples particles from volume meshes.
7. V2 invariant extended.

## What This Phase Does Not Do

- No coupling rows (Phase 11).
- No MPM (deferred per master plan §3 Round 3).
- No surface reconstruction / rendering (visual rendering uses CUDA RT in Phase 13).
- No fluid sensor output beyond particle position arrays.
