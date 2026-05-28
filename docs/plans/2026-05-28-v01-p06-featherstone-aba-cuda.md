# Nuka Physics v0.1 – Phase 6: Featherstone ABA Forward Dynamics on CUDA

> **Master plan reference:** §3 Round 3 (Path B variant) + §3 Round 4 (dual-track articulation)
> **Prerequisites:** Phase 3 (DeviceContext), Phase 5 (Row format and scheduler)
> **Blocks:** Phase 7 (Go2 stand demo + V1 oracle)
> **Exit criteria gate:** v0.1
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Implement Featherstone's Articulated-Body Algorithm (ABA) forward dynamics on CUDA. This is the missing reduced-coordinate path for articulated chains (Go2, H1). Master plan §3 Round 3 commits to dual-track: ABA for joint dynamics, rows for constraints. Current code has only `ArticulationGraph` topology (no Featherstone state) and treats joints as maximal-coord rows, which works but is the wrong path for sim-to-real-grade robot accuracy.

Deliverable: CUDA-resident Featherstone forward dynamics for revolute, prismatic, and fixed joints (the joint types used by Unitree Go2 and H1). Output is joint accelerations `q̈` from joint torques `τ` + applied wrenches. Integration of joint state (`q`, `q̇`) is plumbed into existing time integrator.

External contacts and joint limit constraints continue to flow through the `RowSolver` (Phase 5), via the `FeatherstoneContactRow` class which uses the Featherstone-generalized Jacobian.

## Tech Stack

- C++20
- CUDA 12+
- Featherstone reference: "Rigid Body Dynamics Algorithms" (Featherstone 2008), Algorithm 7.1 (ABA)
- Validation oracle: MuJoCo MJX (or Pinocchio for offline cross-check)

## Files to Create

- `src/runtime/articulation/articulation_state.hpp` — per-link state (transform, velocity, articulated inertia, bias)
- `src/runtime/articulation/articulation_state.cpp`
- `src/runtime/articulation/featherstone_aba.hpp` — public API
- `src/runtime/articulation/featherstone_aba.cu` — three-pass ABA on CUDA
- `src/runtime/articulation/featherstone_aba.cuh`
- `src/runtime/articulation/articulation_jacobian.hpp` — Featherstone-generalized Jacobian builder (for FeatherstoneContactRow)
- `src/runtime/articulation/articulation_jacobian.cu`
- `src/runtime/articulation/articulation_cooker.hpp` — cook URDF/MJCF/USD articulation → device-side topology + inertia
- `src/runtime/articulation/articulation_cooker.cpp`
- `tests/articulation/test_aba_double_pendulum.cpp` — closed-form vs ABA
- `tests/articulation/test_aba_go2_oracle.cpp` — MJX oracle agreement
- `tests/articulation/test_aba_h1_oracle.cpp` — MJX oracle agreement on longer chain
- `tests/articulation/test_featherstone_contact_row.cpp` — articulation+contact through unified RowSolver

## Files to Modify

- `src/runtime/articulation/articulation_graph.hpp` — keep topology; add inertia/joint axis/joint type fields needed by ABA
- `src/runtime/world_stepper.cpp` — call `FeatherstoneAba::ComputeAccelerations` before `RowSolver::Solve`
- `src/runtime/world_builder.cpp` — populate articulation cooking
- `src/import/urdf_importer.cpp`, `src/import/mjcf_importer.cpp`, `src/import/usda_importer.cpp` — extract per-link inertia (mass, COM, inertia tensor) and joint frames into the articulation cooker output
- `tools/codegen/classes/featherstone_contact.yaml` — fill in the actual Jacobian layout (Phase 2 left a stub)

## ABA on CUDA — algorithm sketch

For each articulation (chain of `n_links` links), per env:

**Pass 1 (root → leaves, kinematics):**
- Compute spatial transforms from joint frame to world.
- Compute spatial velocities `v[i]` from `v[parent[i]] + S[i] * q̇[i]` where `S[i]` is joint motion subspace.

**Pass 2 (leaves → root, articulated inertia):**
- Initialize `I_a[i] = I[i]` (link spatial inertia).
- For each link from leaf to root:
  - Compute `U[i] = I_a[i] * S[i]`
  - `D[i] = S[i]^T * U[i]`
  - `u[i] = τ[i] - S[i]^T * p_a[i]` (where `p_a` is articulated bias force)
  - Propagate to parent: `I_a[parent] += transform(I_a[i] - U[i] * D[i]^-1 * U[i]^T)`

**Pass 3 (root → leaves, accelerations):**
- For each link from root to leaf:
  - Compute `a_parent` in link frame.
  - `q̈[i] = D[i]^-1 * (u[i] - U[i]^T * a_parent)`
  - `a[i] = a_parent + S[i] * q̈[i] + spatial bias`

CUDA parallelization: each articulation runs on a single warp (or block depending on link count). Within an articulation, the recursion is **sequential along the chain** — that's the O(n) cost of ABA. For Go2 (~18 links) one warp per articulation is sufficient. Across articulations (4096 envs × 1 articulation each = 4096 articulations) → 128 blocks × 32 threads.

D1 determinism: no float atomics; per-articulation accumulations are within-warp; final body-level updates use segment reduction. Reduction order within a warp is fixed by warp shuffle ordering — verify bit-exact across runs.

## Tasks

### Task 6.1 — Articulation state layout

`src/runtime/articulation/articulation_state.hpp`:

```cpp
struct LinkSpatialInertia { float I[36]; };   // 6x6 spatial inertia
struct LinkSpatialVel     { float v[6]; };
struct LinkSpatialAccel   { float a[6]; };
struct LinkArticulatedInertia { float Ia[36]; };
struct LinkBiasForce      { float p[6]; };

struct ArticulationDeviceState {
    // SoA layout: arrays of size n_articulations × max_links_per_articulation
    LinkSpatialInertia*       link_inertia;        // body-frame
    LinkSpatialVel*           link_velocity;       // world-frame
    LinkSpatialAccel*         link_acceleration;
    LinkArticulatedInertia*   link_articulated_I;
    LinkBiasForce*            link_bias_force;
    // Joint state
    float* q;                   // joint position
    float* qdot;                // joint velocity
    float* qddot;               // joint acceleration (ABA output)
    float* tau;                 // applied joint torque (input)
    // Joint axes in link-local frame
    float* joint_axis;          // 3-vec per joint
    uint8_t* joint_type;        // 0 revolute, 1 prismatic, 2 fixed
    uint32_t* parent_link;
    uint32_t* link_to_articulation;    // articulation id per link
    uint32_t* articulation_link_count; // per articulation
    uint32_t  total_link_count;
    uint32_t  articulation_count;
};
```

### Task 6.2 — ABA kernels (three passes)

`src/runtime/articulation/featherstone_aba.cu`:

```cuda
__global__ void aba_pass1_kinematics_kernel(ArticulationDeviceState s, ...);
__global__ void aba_pass2_articulated_inertia_kernel(ArticulationDeviceState s, ...);
__global__ void aba_pass3_accelerations_kernel(ArticulationDeviceState s, ...);

void FeatherstoneAba::ComputeAccelerations(const phi::DeviceContext& ctx,
                                           ArticulationDeviceState& s,
                                           float gravity_z)
{
    nuka::phi::ScopedDeviceGuard guard(ctx.device_id);
    auto stream = ctx.stream.Native();

    dim3 grid(s.articulation_count), block(32);     // warp per articulation
    aba_pass1_kinematics_kernel<<<grid, block, 0, stream>>>(s, gravity_z);
    aba_pass2_articulated_inertia_kernel<<<grid, block, 0, stream>>>(s);
    aba_pass3_accelerations_kernel<<<grid, block, 0, stream>>>(s);
}
```

Each kernel handles one articulation per warp; thread idx within warp = link idx along the chain. For longer chains (H1 has ~28 links), allow block-level handling.

### Task 6.3 — Featherstone-generalized Jacobian

For the `FeatherstoneContactRow` class (master plan §4 catalog), the Jacobian relating a point's velocity to joint velocities is a chain product of joint motion subspaces transformed along the kinematic path from root to the contact link.

`src/runtime/articulation/articulation_jacobian.cu`:

```cuda
__global__ void compute_link_point_jacobian_kernel(
    const ArticulationDeviceState s,
    const uint32_t* contact_link_indices,
    const float3*    contact_point_world,
    uint32_t         contact_count,
    float*           out_jacobian_scalars)   // CSR layout per row's body_count
{
    uint32_t contact_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (contact_idx >= contact_count) return;

    uint32_t link = contact_link_indices[contact_idx];
    // Walk from link to root, accumulating S[parent] transformed to world,
    // store one scalar per joint along the chain.
    // ...
}
```

The output feeds `RowBuffers::jacobian_data` at the offsets configured by `FeatherstoneContactRow` IR.

### Task 6.4 — Articulation cooker

`src/runtime/articulation/articulation_cooker.{hpp,cpp}`:

Consumes `SceneGraph` articulation entries (already imported by `src/import/*`) and produces device-side `ArticulationDeviceState` buffers, including:
- Per-link mass / COM / inertia (transformed to spatial inertia 6x6)
- Joint type / axis / parent / frame offsets
- Initial `q`, `qdot`

Output is consumed by `world_builder.cpp` and uploaded once per cook (or per scene change).

### Task 6.5 — Integration into world stepper

`src/runtime/world_stepper.cpp`:

```cpp
void WorldStepper::Step(float dt) {
    nuka::phi::ScopedDeviceGuard guard(ctx_.device_id);

    // 1. Featherstone ABA: compute qddot from tau + bias + gravity
    featherstone_.ComputeAccelerations(ctx_, articulation_state_, gravity_z_);

    // 2. Build constraint rows (contacts via SDF/CD; joint limits; drives)
    row_builder_.Build(buffers_, /* contact manifold, joint limits, drives */);

    // 3. RowSolver iterates PGS using codegen evaluators + coloring
    row_solver_.Solve(ctx_, buffers_, body_state_, solve_cfg_);

    // 4. Integrate: q += qdot*dt; qdot += qddot*dt (simplified; symplectic Euler)
    integrator_.AdvanceArticulationState(articulation_state_, dt);
    integrator_.AdvanceFreeBodyState(body_state_, dt);
}
```

### Task 6.6 — Validation tests

`test_aba_double_pendulum.cpp`: 2-link revolute pendulum under gravity, no actuation. Compute analytical accelerations (closed-form Lagrangian) vs ABA output; assert agreement to 1e-5.

`test_aba_go2_oracle.cpp`: Load Go2 USD; sample random poses + velocities + torques; run our ABA + run MJX; compare `qddot`. Tolerance per master plan §6 V1 table: < 1e-4 joint angle over 1 s. For per-step joint acceleration, < 1e-3 absolute.

`test_aba_h1_oracle.cpp`: same protocol for H1.

`test_featherstone_contact_row.cpp`: Articulated chain in contact with the ground; verify the contact resolution through the unified RowSolver produces stable standing.

## Validation

- Double pendulum: closed-form vs ABA agreement < 1e-5.
- Go2 vs MJX: per-step `qddot` within 1e-3 absolute on random inputs.
- H1 vs MJX: same.
- D1 determinism: same input twice → bit-exact `qddot` and link transforms.
- V2 invariants: energy drift < 2% over 5 s of free-fall pendulum.

## Exit Criteria for Phase 6

1. ABA forward dynamics CUDA-resident, three-pass, deterministic.
2. Supports revolute, prismatic, fixed joints (the set used by Go2 + H1).
3. Articulation cooker pulls per-link inertia from URDF / MJCF / USD imports.
4. Featherstone-generalized Jacobian feeds `FeatherstoneContactRow` correctly.
5. Tests:
   - Double pendulum analytical agreement.
   - MJX agreement on Go2 (random pose / vel / torque sampling).
   - MJX agreement on H1.
   - Articulated chain + ground contact stable in unified RowSolver.
6. V2 energy invariant holds over 5 s of pendulum motion.

## What This Phase Does Not Do

- No Featherstone CRBA (composite rigid body algorithm) — not needed for forward dynamics; reserved for diff-sim inverse work in v0.5.
- No ball / universal / 6-DOF joint types — Go2 / H1 use only revolute / prismatic / fixed; broader joint set lands when scenes demand it.
- No constrained articulation closure loops — Go2 / H1 are open chains.
- No adjoint of ABA (v0.5 Phase 1 ships the adjoint).
