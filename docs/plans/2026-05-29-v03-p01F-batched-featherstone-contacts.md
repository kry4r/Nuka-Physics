# p01-F — Batched Featherstone ABA + Contact Rows (4096-env Go2)

> **Foundational sub-phase** inserted into v0.3 p01 after the owner chose option (a) (see
> `2026-05-29-v03-4096env-go2-path-gap.md` §6). Builds the 4096-env Go2 articulated GPU step path that p01 Wave 2–3 and p04
> presuppose. **GPU-only; D1 strong determinism must not regress.** Authored from a read of the live code (citations below).
> Execution: subagent-driven, all opus, commits centralized by the controller with `[skip ci]`.

## 0. Key findings from the code read (load-bearing)

- **Device state is already N-articulation capable.** `ArticulationDeviceState` carries `link_to_articulation`,
  `articulation_link_count/offset`, `total_link_count`, `articulation_count` (`articulation_state.hpp:78-82`); ABA kernels grid
  per-articulation (`featherstone_aba.cu:580` `dim3 grid(state.articulation_count)`, `articulation = blockIdx.x`). Feeding 4096
  Go2 is a **data-population** problem, not a kernel-structure one.
- **ABA kernels are thread-per-articulation (lane!=0 early-returns at `featherstone_aba.cu:372,425,496`), NOT
  warp-per-articulation.** ⇒ the perf-budget "warp-per-articulation 80 µs" line and the 1000 µs `step_total` are **Wave-3
  optimization targets, NOT p01-F exit gates.** p01-F delivers *correct, stable, deterministic* 4096-env Go2; speed comes later.
- **Branching trees already work** — DFS pre-order cook (`articulation_cooker.cpp:189-196`) + reverse-local-order ABA Pass 2
  (`featherstone_aba.cu:431-432,465-468`) handle Go2's 4-leg tree.
- **Go2 = fixed/pinned base** (`go2_stand.usda`: base `kinematicEnabled=true, mass=0`; cooker forces root `Fixed`
  `articulation_cooker.cpp:157`; enum has **no Free/Floating joint** `articulation_state.hpp:18-22`). 13 bodies (base + 4×{hip,
  thigh, calf}), 12 revolute joints, feet = spheres r=0.0175 on calf links.
- **No generated kernel is live** — only references to `solver::generated` are registry-name checks in
  `tests/codegen/test_codegen_roundtrip.cpp`; the generated `featherstone_contact_forward_kernel` is a compile-only stub
  (`featherstone_contact_forward.cu:55`). The only live solver kernel is `SolveRowsSweepKernel` (`row_solver.cu:676`). ⇒ reuse
  the FeatherstoneContact **row data layout** (`row.hpp:16,49`; `tools/codegen/classes/featherstone_contact.yaml`, whose
  `forward_evaluator.input_fields` already include `articulation_id, link_index, jacobian_chain_scalar, effective_mass`) but
  implement the solve in a **new non-generated kernel** — never hand-edit `src/codegen/generated/*` (PROTECTED, `// GENERATED`).
- **Two row solvers exist:** the deterministic `solver/gpu/row_solver.cu` (host greedy graph-coloring, colored single-block
  sweep) vs the in-place `BatchedRowGroup` path in `batched_device_world.cu` which uses `atomicAdd` block allocation
  (`:1145,1174,1216`) → **order-nondeterministic, unusable for D1.** We reuse only broadphase/narrow-phase from the batched TU.
- **Single-env oracle has contacts off** (`world.cpp:255`). `test_go2_stand_5s.cpp` determinism + MJX-golden (≤1e-4) tests run
  the same ABA kernels with no contacts → no-regression must be **structural** (ABA kernels untouched; `env_count==1` path
  unchanged; coupling kernels run only on the batched/contacts path).

## 1. Per-step pipeline (one step of the 4096-env Go2 world)

New stepper `StepBatchedArticulatedWorld`; all state device-resident across steps; contact stages gated by `enable_contacts`.

| # | Stage | Kernel | budget tag |
|---|---|---|---|
| 1 | PD position-drive → joint torque | `ApplyPositionDriveKernel` (exists) | `featherstone_aba` |
| 2 | ABA pass 1 — kinematics (Xup, v, bias, world `link_pose`) | `AbaPass1KinematicsKernel` (exists) | `featherstone_aba` |
| 3 | ABA pass 2 — articulated inertia + bias (leaf→root) | `AbaPass2ArticulatedInertiaKernel` (exists) | `featherstone_aba` |
| 4 | ABA pass 3 — joint accelerations (root→leaf) | `AbaPass3AccelerationKernel` (exists) | `featherstone_aba` |
| 5 | velocity sub-step `qdot += qddot·dt` | `IntegrateArticulationKernel` (vel-only) | `integrator` |
| 6 | foot-shape world poses (from `link_pose`) | small FK-to-shape kernel (new) | `buffer_mgmt` |
| 7 | broadphase (feet vs ground, per env) | `BuildBatchedCudaBroadphase` (exists) | `contact_generation` |
| 8 | narrow-phase → manifolds | `GenerateBatchedCudaContacts` (exists) | `contact_generation` |
| 9 | full-chain contact Jacobians (foot→root) | `ComputeContactChainJacobians` (new) | `row_builder` |
| 10 | assemble FeatherstoneContact rows (normal+friction+limits), fixed order, no atomics | `AssembleArticulatedContactRows` (new) | `row_builder` |
| 11 | joint-space effective mass `1/(J M⁻¹ Jᵀ)` (CRBA M) | `ComputeContactEffectiveMass` (new) | `scheduler` |
| 12 | PGS solve, **block-per-articulation**, fixed order, warm-start | `SolveArticulatedContactRowsKernel` (new) | `row_solver` |
| 13 | apply impulses `qdot += M⁻¹ Jᵀ λ` | epilogue of 12 (or small kernel) | `row_solver` |
| 14 | position sub-step `q += qdot·dt` | `IntegrateArticulationKernel` (pos) | `integrator` |
| 15 | V2 invariant sampling (every 16 steps) | existing `invariants_gpu` | `v2_sampling` |

`enable_contacts==false` ⇒ stages 6–13 skipped ⇒ exactly the existing ABA path (oracle preserved by construction).

## 2. Contact ↔ ABA coupling (the new part)

1. **Detect in maximal coords, reuse pipeline.** Run only `BuildBatchedCudaBroadphase` + `GenerateBatchedCudaContacts`
   (`batched_device_world.cu:2667,2753`) on Go2 feet vs ground. Do **not** call `SolveBatchedCudaConstraints*` /
   `BatchedRowGroup` (atomics → breaks D1).
2. **Full-chain contact Jacobian (new).** Existing `ComputeLinkPointJacobianKernel` (`articulation_jacobian.cu:42-69`) only
   does the contact link's own joint axis — insufficient. New `ComputeContactChainJacobians` walks `parent_link` to root and
   writes a scalar per ancestor joint (revolute `dot(cross(axis_w,(p−origin_w)),n)`, prismatic `dot(axis_w,n)`) using
   `link_pose` + `joint_axis`. Output: fixed-stride `jacobian_chain_scalar` over the chain (Go2 chain ≤ 3 leg joints).
3. **Joint-space inertia M via CRBA (new).** Go2 ≈ 12 DOF (fixed base) ⇒ dense `M` (≤~18×18) per articulation is cheap.
   `ComputeContactEffectiveMass` assembles `M` (CRBA, new device code reusing spatial-inertia/transform helpers from
   `featherstone_aba.cu`), inverts/factorizes it, and computes `m_eff = 1/(J M⁻¹ Jᵀ)` per row. **Rejected alt:** ABA
   articulated-impulse O(n) method — more faithful/scalable but more device code and harder to make obviously deterministic;
   documented Wave-3 fallback if CRBA becomes a bottleneck.
4. **PGS block-per-articulation = the D1 mechanism.** Contacts on one Go2 touch only that articulation's DOFs ⇒ no
   cross-articulation sharing ⇒ no coloring across articulations, **no atomics, fixed reduction order**. `SolveArticulatedContactRowsKernel`
   grids one block per articulation; within a block: normal rows then friction (limit `friction·Σλₙ`, as `row_solver.cu:293-299`),
   warm-started; joint-limit rows appended as unilateral single-DOF rows. Fixed iteration count.
5. **Apply Δqdot.** `qdot += M⁻¹ Jᵀ λ` per articulation before the position integrate; race-free (each block owns its DOFs).

**Reused as-is:** `ArticulationDeviceState`/buffers, all ABA kernels, broadphase/narrow-phase + `CudaBatchedContactManifold`,
`constraint::Row`/`RowMaterial`/`RowJacobian6` (`row.hpp`), FeatherstoneContact class id 3.
**Extended additively:** `articulation_jacobian.cu` (+chain kernel); `featherstone_contact.yaml` only if an additive input field
is needed, regenerated via `tools/codegen/regen.py` (emitter only).
**New files:** `src/runtime/articulation/articulation_contacts.{cu,hpp}` (effective-mass, row assembly, solve, apply);
`src/runtime/gpu/batched_articulated_world.{cu,hpp}` (`StepBatchedArticulatedWorld`).

## 3. Task breakdown (ordered; subagent-driven)

Deps in [...]; effort S/M/L; each task has an integrated check (per `agent.md`: e2e/integration, not narrow units).

- **T1 — Batched N-articulation replication.** [no deps] `articulation_state.cpp/.hpp`: `ReplicateArticulationHostState(base,
  env_count)` tiles one cooked Go2 into N (offset `link_to_articulation`, `articulation_link_offset`). **Chosen:** replicate one
  cook N× (deterministic memcpy-with-offset) over cooking an N-skeleton scene (slower, order-fragile). Disjoint. **M.**
  *Check:* upload 4096 replicas, run existing ABA 100 steps, assert all 4096 `q` bit-identical to single-env trajectory.
- **T2 — Foot-shape world pose + reuse broadphase/narrow-phase for feet.** [T1] new `articulation_contacts.{cu,hpp}` (FK-to-shape
  only) + thin `BatchedShapeTables` adapter (feet + ground). Shares new file w/ T4/T5 (serialize); disjoint from T3. **M.**
  *Check:* drop 4096 Go2 so feet penetrate ground; manifold count >0 per env, points at foot/ground interface.
- **T3 — Full-chain contact Jacobian.** [T1] (∥ with T2) `articulation_jacobian.cu/.hpp`: add `ComputeContactChainJacobians`;
  leave existing single-link kernel intact. Disjoint from T2. **M.**
  *Check:* apply known vertical foot force, verify projected joint torques match an analytic 3-link leg (sign+magnitude).
- **T4 — CRBA joint-space inertia M + effective mass.** [T1,T3] `articulation_contacts.{cu,hpp}` (serialize w/ T2,T5):
  `ComputeArticulationInertiaM` + `ComputeContactEffectiveMass`; per-articulation `M`/`M⁻¹` scratch (deterministic, no atomics).
  May add additive YAML field + `regen.py` (emitter only) — likely unnecessary (`effective_mass` already in YAML). **L.**
  *Check:* compare `M` for one leg vs a CPU reference to ~1e-4.
- **T5 — Assemble + PGS-solve articulated contact/limit rows + apply Δqdot.** [T2,T3,T4] `articulation_contacts.{cu,hpp}`:
  `AssembleArticulatedContactRows`, `SolveArticulatedContactRowsKernel` (block-per-articulation, fixed order, warm-start,
  friction), `ApplyJointSpaceImpulses`. **L.**
  *Check:* single Go2 on ground reaches static rest (feet in contact, qddot→0, finite), penetration < slop.
- **T6 — Batched articulated stepper.** [T1–T5] new `batched_articulated_world.{cu,hpp}` (`StepBatchedArticulatedWorld`)
  composing stages 1–15, contacts gated. Disjoint (new files). **M.**
  *Check:* 4096-env Go2 steps 1000×, all finite, feet contact ground per env; per-kernel timings via `PerfRecorder`.
- **T7 — C ABI / C++ surface for 4096-env Go2.** [T1,T6] `c_abi/world.cpp` (+`internal.hpp`), `nuka/nuka.h` (additive desc
  fields). See §4. Disjoint from T1–T6. **M.**
  *Check:* `World::CreateFromScene(go2, 4096, dt)` + `StepN` works; `env_count==1` still routes to unchanged Featherstone path,
  oracle passes.
- **T8 — Floating (6-DOF) base. [REQUIRED for p04 locomotion — not optional]** [T1–T7] add a `Free` joint type / 6-chained-DOF
  root in the enum + cooker + ABA, isolated so it cannot perturb the fixed-base oracle numerics. Without it the base is pinned
  and the robot cannot locomote (only stand). **L.**
  *Check:* a single free-base Go2 under gravity falls/contacts ground physically; with leg drives can support its own weight;
  fixed-base oracle still bit-exact.

**Critical path:** T1 → T3 → T4 → T5 → T6 → T7 → T8. T2 ∥ T3 after T1.
**GENERATED/protected:** only T4 *might* touch `featherstone_contact.yaml` + `regen.py`; never hand-edit `src/codegen/generated/*`.

## 4. C ABI / API surface

Keep `StepWorldGpu` (single-env Featherstone, `world.cpp:151`) untouched (oracle). Add a parallel batched path:
- `nuka_world_desc_t`: `env_count` already exists; optionally add zero-default `enable_contacts` + solver-iteration fields.
- `nuka_world_create_from_scene` (`world.cpp:225`): replace the `env_count!=1 → NOT_SUPPORTED` (`:235`) with a branch —
  `==1` keeps existing construction; `>1` cooks Go2 once, `ReplicateArticulationHostState` (T1), uploads, builds foot-shape
  tables, stores a **new batched record variant** in `WorldRecord` (add batched device-world member + `is_batched` flag to
  `internal.hpp`).
- `nuka_world_step_n` (`world.cpp:294`): dispatch `is_batched ? StepBatchedArticulatedWorld : StepWorldGpu`. Buffer-view
  readback returns batched `q`/`qdot` strided by env.
- C++ `World::CreateFromScene(...env_count...)` (`nuka.hpp:86`): **no signature change** — already forwards `env_count`; works
  once the C ABI cap is lifted.

## 5. Validation (integrated/e2e per `agent.md`)

- **Headline:** new test — create 4096-env Go2 (contacts on), step 1000×; assert (a) zero NaN/Inf across all envs (invariants
  path), (b) finite bounded `q`, (c) ≥1 active foot contact at rest per env, (d) all 4096 env states identical (replicated
  determinism).
- **Oracle no-regression (structural):** `test_go2_stand_5s.cpp` MJX-golden (≤1e-4) + determinism replay pass unchanged.
- **V2 energy invariant:** passive (drives+contacts off) conserves/dissipates monotonically; contacts-on must not inject energy.
- **D1 strong determinism:** two identically-stepped 4096-env worlds produce bit-identical `q`/`qdot` — guaranteed by the
  block-per-articulation, no-atomics, fixed-order solve; assert explicitly.

## 6. Risks & unknowns

- **Floating base = required for locomotion (T8).** Core T1–T7 deliver *fixed-base stand-with-contacts* (validates the
  coupling machinery + the 4096-env path). p04 PPO *locomotion* needs the moving 6-DOF base ⇒ T8 is required, isolated to
  protect the oracle. Stated plainly.
- **Branching tree** — resolved (DFS pre-order + reverse Pass 2); validated by T3/T5.
- **VRAM @ 4096** — per-link matrices × 13 × 4096 ≈ low hundreds of MB; fits 2× RTX 4000 Ada (20 GiB) and the 5080 (16 GiB);
  tally allocations in T1/T6, flag any single buffer near limits.
- **Coupling determinism** — CRBA `M` assembly + `M⁻¹` must be fixed-order, no atomics; reused narrow-phase is slot-indexed
  (deterministic) — verify before relying on it in the D1 path.
- **Warp-per-articulation budget gap** — p01-F stays thread-per-articulation; 80 µs ABA / 1000 µs `step_total` budget lines are
  **Wave-3 targets, not p01-F gates.**
