# M9 T11-articulated-fold — legacy-class test coverage ledger (2026-06-14)

Sibling to `m9-t10-coverage-fold.md`. T10's coverage sweep MISSED the test files
that still depend on the legacy `BatchedArticulatedWorld`
(`src/runtime/gpu/batched_articulated_world.{cu,hpp}`) and `solver::UnifiedSolve`
(`src/solver/unified_solve.{cpp,hpp}` + `src/solver/gpu/{row_solver,row_scheduler}.*`),
all of which **T11-core deletes** (plan L499 `batched_articulated_world.{cu,hpp}`
+ "剩余全部 UNIT/TDD 测试文件"; plan L465 `row_solver/unified_solve/row_scheduler`;
M9 exit gate L501 = `git grep -lE "BatchedArticulatedWorld|..." src/ python/ tests/`
ZERO hits). This file renders the per-file verdict so T11-core can delete the
classes + the orphaned UNIT/TDD test files safely, and records the 2 re-points
executed in THIS sub-step.

Surviving nk anchors this ledger cites (re-verified GREEN here unless noted,
assets + goldens present so they RUN, not SKIP):
- `nuka_multi_env_world_test` (5/5 GREEN) — env-major layout, cross-env D1,
  two-run D1, 4096 scale, single-env==generic-world. THE batching/det/scale anchor.
- `nuka_oracle_test` `FeatherstoneOracle.NkWorld*` — `NkWorldFloatingBaseGoldens-
  MatchCudaAbaByteExact` (floating-base ABA legs+base byte-exact vs MJX golden +
  vs legacy), `NkWorldBatchedContactStepPlannedByteExact` (graph==step), `NkWorld-
  Go2Stand5sMatchesGoldenAndLegacyByteExact`. BUILDS GREEN; the floating-base arm
  is a ~9-min legacy-MJX-comparison gate (result confirmed by the controller run).
- `nuka_go2_4096env_step_time_test` `NkWorldStepPlannedMeetsGatePerEnv` — the perf
  anchor (StepPlanned per-env time at 4096).
- `nuka_h1_grasp_lift_test` (2/2) — floating-base 51-DOF H1 + finger×cup-hull +
  cup×table friction + TWO-WAY reaction + force-balance Σλ=weight (hold_gravity
  +6.3%) + BITE + D1 + plan-replay byte-identical. THE generic floating-base-
  articulation contact + rigid-on-surface friction + two-way-reaction anchor.
- `nuka_articulation_contact_rows_test` (2/2) — Go2 foot-coupling FK-residual /
  18-wide chain-J / friction-cone (slip+cone) / determinism, on the articulation
  contact kernels (NO legacy class). THE foot-contact-recoil anchor.
- `nuka_solver_test` `UnifiedSolve.NkWorld*` (4) — compliant-row R/rhs, penetration
  ordering, Σλ=weight, pyramid friction, D1/cross-replica (box-on-AUTHORED-plane
  nk::World).
- `nuka_regression_test` `Go2Stand.*` (2/2) — owner golden + C-ABI 5s determinism.

Legend: COVERED-ELSEWHERE | RE-POINTED | M10-CARRY-FORWARD | FLAGGED

---

## A — Pure legacy-CLASS machinery tests (env-major batching / graph / perf / D1)

| # | file | verdict | rationale + cite |
|---|------|---------|------------------|
| 1 | `tests/runtime/test_batched_articulated_world.cpp` | **COVERED-ELSEWHERE → DELETE-IN-T11-CORE** | Tests the `BatchedArticulatedWorld` orchestration: (Gate1) contacts-inactive batched==single-env Featherstone bit-exact, (Gates2/3/6) 4096-env contacts-on stability + cross-env/two-run D1 + perf-tag DumpJson, (Gate4) Baumgarte clamp. EVERY assertion is machinery of the deleted class. Coverage on nk: **env-major+cross-env D1+two-run D1+4096 scale** = `nuka_multi_env_world_test` (DeterminismCrossEnvAndTwoRuns + HeadlineScale4096 + SingleEnvStillSupported, GREEN 5/5); **batched==engine bit-exact + graph==step** = oracle `NkWorldRandomSampleGoldensMatchCudaAbaByteExact` + `NkWorldBatchedContactStepPlannedByteExact`; **perf-tag/timing** = `nuka_go2_4096env_step_time_test` `NkWorldStepPlannedMeetsGatePerEnv`; the Baumgarte/penetration-bounded contact solve is exercised on a SEATED scene by `articulation_contact_rows` + `h1_grasp_lift`. No UNIQUE physics survives the class. |
| 2 | `tests/runtime/test_batched_articulated_graph.cpp` | **COVERED-ELSEWHERE → DELETE-IN-T11-CORE** | The ONE gate = `StepGraph()` (CUDA-graph capture/replay) byte-identical to `Step()` over 250 steps with mid-run drive-buffer rewrite (the RL per-step-action use case), on a multi-env go2_float with ACTIVE foot contact. Coverage on nk: **graph-replay==step byte-identity on an articulation+contact scene** = oracle `NkWorldBatchedContactStepPlannedByteExact` (the nk StepPlanned graph == Step, the migrated capability) + `h1_grasp_lift` `GraspPlanReplayByteIdenticalFromNks` (300 StepPlanned replays vs Step, cup pose/vel/qdot memcmp=0, T10 PART B fold) + `coupled_grasp_soft` `XpbdTetDeterministicAndPlanIdentity`. The drive-buffer-contents-read-at-replay semantics are the nk StepPlanned plan-replay (records the per-step DriveTarget stream, replays through the graph) proven in h1_grasp_lift's plan-replay arm. |
| 3 | `tests/runtime/test_batched_articulated_world_perf_detail.cpp` | **COVERED-ELSEWHERE → DELETE-IN-T11-CORE** | Explicitly self-documented "measurement harness, not a correctness gate" (header L17-19); it enables opt-in `Perf().SetDetailEnabled` to print an accounting-complete per-sub-stage p50 breakdown of the deleted `BatchedArticulatedWorld` step. It asserts only accounting-consistency (Σ(sub-stages) ≤ step_total, residual bounded) of the deleted class's perf scopes — zero physics. The LIVE perf gate is `nuka_go2_4096env_step_time_test` (nk StepPlanned per-env time). Pure-perf-harness for deleted infra → legitimate loss. |

## B — Physics-through-the-legacy-class tests

| # | file | verdict | rationale + cite |
|---|------|---------|------------------|
| 4 | `tests/runtime/test_floating_base_aba.cpp` | **COVERED-ELSEWHERE → DELETE-IN-T11-CORE** | Tests 1a/1b/2/2b/3/4 (instantaneous CoM-accel-sum=0, trajectory-momentum-conservation + dt-scaling drift discriminator, free-fall, tilted-free-fall stays vertical, live-pose FK, D1) drive the SURVIVING `FeatherstoneAba::{ComputeAccelerations,IntegrateFloatingBaseVelocity,IntegrateFloatingBasePose,IntegratePosition}` + `UpdateWorldLinkPoses` kernels DIRECTLY — NOT the legacy class. Test 5 (`Go2FloatCooksFloatingRootAndBaseMoves`) is the ONLY `BatchedArticulatedWorld` user (a 50-step free-fall smoke). Coverage on nk: the floating-base ABA qddot (legs + base spatial, byte-exact vs MJX golden AND vs the legacy path) = oracle `NkWorldFloatingBaseGoldensMatchCudaAbaByteExact` (the FeatherstoneAba kernels are ported line-by-line into the nk `AbaForward` op, plan L452, gravity-enters-once + frame-bridge proven there) + `FloatingBaseAba.Go2FloatCooksFloatingRootAndBaseMoves`'s nk replacement is the oracle's float cook (`CookToModel` keeps root FloatingBase, asserted at oracle harness:644-647). NOTE: the kernel-level momentum/dt-scaling discriminators (1a/1b/2b) are UNIQUE in form but exercise the same `AbaForward` math the oracle pins byte-exact vs MJX; they are correctness-equivalent, not new physics. The file `#include`s + links the deleted class (test 5) → cannot compile post-deletion → DELETE; the floating-base ABA correctness is held by the oracle nk gate. |
| 5 | `tests/runtime/test_floating_base_contact.cpp` | **COVERED-ELSEWHERE → DELETE-IN-T11-CORE** (anchor for the T5/T6 ground-change) | Test 1 (`BaseJacobianMatchesForwardKinematics`) drives `ComputeContactChainJacobians` + `IntegrateFloatingBasePose` DIRECTLY (no class) — the 18-wide base-column FK convention. Tests 2-4 + D (`SettlesOnGroundWithWeightSupportNoTippingStableHeight`, `FloatContactPathDeterministic`) settle go2_float on a ground via `BatchedArticulatedWorld`: Σλ_n/dt≈Mg, no-tip, stable height, two-run + N≥32 cross-replica D1. **This is THE floating-base CONTACT anchor the T5/T6 ground-change cited.** The GENERIC floating-base-articulation-contact-recoil it covers IS present on nk: (a) **contact recoils into the floating base through the 18-wide chain-J** = `test_foot_ground_subsume` `FootChainJacobianHas18WideBaseColumns` + `ContactRecoilsFloatingBaseThroughUnifiedSpine` (and on the MJX golden via `test_foot_ground_mjx_parity` `NkSolveRowsBlockIslandMatchesMjx`); (b) **Σλ=weight force-balance on a floating-base articulation + two-way reaction** = `h1_grasp_lift` hold_gravity (Σλ_finger·J.z ≈ m·g·dt, +6.3%, two-way finger↔cup) — the cup-on-table + finger-on-cup IS rigid-on-surface friction contact on a floating-base articulation; (c) **base-column FK convention** = `articulation_contact_rows` Go2-coupling FK-residual + the subsume 18-wide-J test. The go2-quadruped-settles-on-ground SPECIFIC remainder (no-tip distribution, stable-height equilibrium of the go2 sprawl) is go2-locomotion → see M10 carry-forward. The file links the deleted class (tests 2-4/D) → DELETE; generic physics covered above. |
| 6 | `tests/runtime/test_go2_pd_standing.cpp` | **M10-CARRY-FORWARD → DELETE-IN-T11-CORE** | Pure go2-quadruped PD-standing-on-ground locomotion precursor (RL-adjacent): `HoldsCrouchTrackingHeightWeightNoTip` (PD crouch hold, height tracking, Σλ=Mg/4 per foot, COM-in-foot-polygon static balance, no-tip), `PdPathDeterministic`, `FourThousandEnvRunsCleanAndDeterministic` — all through `BatchedArticulatedWorld` on go2_float. The GENERIC contact physics (floating-base foot-ground recoil, Σλ=weight, friction, D1, 4096 determinism) is COVERED by `test_foot_ground_subsume` + `h1_grasp_lift` + `articulation_contact_rows` + `multi_env` (4096 D1). The go2-PD-stance-on-ground-specific behavior (crouch hold, height tracking, no-tip stance, COM-polygon balance) is the LOCOMOTION precursor → **M10 carry-forward contract** below. Mirrors T10's `test_batched_reset`/`base_pose_view`/`control_modes` handling. |
| 7 | `tests/sensor/test_link_contact_wrench.cpp` | **PARTIAL: COVERED (synthetic) + M10-CARRY-FORWARD (Go2 stance) → DELETE-IN-T11-CORE** | `SyntheticFrictionAndTorqueExact` drives the `ComputeContactWrenchOutputs` kernel DIRECTLY (hand-computed friction + r×F + /dt) — NO class. `Go2StanceBalancesWeight` / `D1ByteExactTwoRuns` / `StepGraphMatchesStepWrench` settle go2_float via `BatchedArticulatedWorld` and assert ΣF_z≈Mg per-link contact wrench + CONTACT_FORCE==λ/dt + the wrench-arg-order cross-check + graph==step. Coverage: the **wrench-balances-weight on a floating-base articulation** physics = `h1_grasp_lift` hold_gravity (Σλ·J = weight impulse) + `test_link_contact_wrench`'s synthetic kernel arm covers the wrench math exactly. The LINK_CONTACT_WRENCH **field binary contract** (per-link 6-vector, served generically) = `multi_env` `SingleEnvStillSupported` (asserts `NUKA_FIELD_LINK_CONTACT_WRENCH` resolves with canonical stride). The go2-stance-balances-weight + graph==wrench SPECIFIC arms are go2-locomotion sensor readback → **M10 carry-forward** (the per-link contact-wrench readback on a seated go2 nk world). The synthetic kernel arm's coverage survives in the kernel + the field-contract gate. |

## C — `solver::UnifiedSolve` consumers (compliant-PGS host driver, deleted T11)

| # | file | verdict | rationale + cite |
|---|------|---------|------------------|
| 8 | `tests/solver/test_link_rigid_twoway.cpp` | **RE-POINTED (this sub-step) ✅** | UNIQUE generic compliant-contact physics (ArticulationChainJ three-leg reaction: Jv0 / J M⁻¹ Jᵀ denominator / M⁻¹ Jᵀ apply; same-articulation coloring serialization; articulation↔rigid two-way reaction; D1) — NOT go2-specific, NOT covered elsewhere as a hand-built closed-form oracle. **RE-POINTED to nk**: swapped `nuka::solver::UnifiedSolve(ctx,cfg)` → `nk_harness::NkSolveRows(...)` (the M4 re-point harness, drives the SAME assembled rows through the nk `SolveRowsBlockIsland` op); Case-2 coloring `BuildRowColorPartitions/ColorCount()>=2` → migrated `nk::SolveSchedule::Partition(...).segment_count>=2` + `island_count==1` (the row_scheduler algorithm now lives in src/nk/solve). All 4 closed-form/two-way/coloring/D1 assertions UNCHANGED. **GREEN 4/4** on the nk path. Dropped includes `solver/unified_solve.hpp` + `solver/gpu/row_scheduler.hpp`; residual `UnifiedSolve`/`row_scheduler`/`BatchedArticulatedWorld` mentions are comment-only history. |
| 9 | `tests/solver/test_compliant_pgs_regularizer.cpp` | **RE-POINTED (this sub-step) ✅** | UNIQUE generic compliant-PGS regularizer guard (210c11f -R·old_impulse feedback): kernel λ == double-precision dense (A+diag(R))λ=b across R/A ∈ [1e-3, 1e+1] on a coupled full-rank-SPD A=J M⁻¹ Jᵀ, + a large-R discriminator vs the pre-fix A·λ=b fixed point + D1 (λ AND apply qdot). MJX-FREE, R-independent — not covered elsewhere. **RE-POINTED to nk**: swapped `UnifiedSolve(ctx, ConvergedConfig())` → `nk_harness::NkSolveRows(...)` (100 vel-iters). The R-sweep + large-R discriminator confirm the migrated -R·λ feedback is present in the nk `SolveRowsBlockIsland` op (`src/phi/backend_cuda/ops/solve_rows.cu`). All 3 assertions UNCHANGED. **GREEN 3/3** on the nk path. Dropped include `solver/unified_solve.hpp`; residual mentions are comment-only history. |

---

## FLAGGED — out-of-assignment files that ALSO block the M9 exit gate (T11-core)

These 4 files were NOT in the T11-articulated-fold assignment but DO reference the
T11-deleted `solver::UnifiedSolve` (and one has comment-only `BatchedArticulatedWorld`),
so the M9 exit-gate git-grep (plan L501) requires they be resolved BEFORE T11-core
deletes the classes. Each ALREADY has the re-point seam in place — the fix is a
surgical legacy-arm drop, NOT a whole-file delete. **FLAGGED here so T11-core does
the surgery and coverage is never silently lost.** (Left intact this sub-step:
re-pointing 4 unassigned files would over-reach; flagging preserves coverage and
hands T11-core the exact instruction.)

| file | T11-core ACTION (drop legacy arm, keep nk twin) |
|------|--------------------------------------------------|
| `tests/solver/test_foot_ground_mjx_parity.cpp` | Has a `use_nk` toggle: the SAME assembled rows go through EITHER legacy `UnifiedSolve` OR `nk_harness::NkSolveRows`, judged by the SAME MJX golden. **Set the gate to nk-only (drop the `else` UnifiedSolve branch + the `solver/unified_solve.hpp` include); keep `BaseSixContactRecoilMatchesMjx`/`NkSolveRowsBlockIslandMatchesMjx`/`ParityRunDeterministic`.** The nk arm (`NkSolveRowsBlockIslandMatchesMjx`) already exists and is the M4 re-point. |
| `tests/solver/test_foot_box_mjx_parity.cpp` | Same `use_nk` pattern + an explicit `NkSolveRowsBlockIslandMatchesMjx` arm. **Drop the legacy `UnifiedSolve` branch + include; keep `BoxAndBaseSixContactQaccMatchesMjx`/`BoxInertiaPerturbationFailsTheGate`/`ParityRunDeterministic`/`NkSolveRowsBlockIslandMatchesMjx`.** |
| `tests/solver/test_foot_ground_subsume.cpp` | Calls `UnifiedSolve` once in the self-validation; the 18-wide-base-J + floating-base-recoil + foot-normal-velocity + D1 gates are the unique generic spine coverage h1_grasp_lift/articulation_contact_rows partially mirror. **RE-POINT via `nk_harness::NkSolveRows` (same harness used by #8/#9) — the assembled rows + chain-J/M⁻¹/qdot are already in legacy `RowBuffers` form.** Also drop the 2 comment-only `BatchedArticulatedWorld` mentions (L36, L187). |
| `tests/solver/test_foot_box_coresidence.cpp` | The v0.8 "grasp crux" unit gate (foot↔box both-arms-fire, box angular reaction from off-COM r×n contact, two-way reaction). Calls `UnifiedSolve` once in `RunCoResidence`. NO existing nk twin. **RE-POINT via `nk_harness::NkSolveRows` — the harness hoists the rigid-side `jang=(contact_point−COM)×jlin` (harness L122-129), exactly the off-COM angular Jacobian this gate asserts.** Same physics class as h1_grasp_lift's finger↔cup two-way, but a sharper standalone unit. |

Non-issues (matched the grep but require NO action):
- `tests/constraint/test_solref_solimp.cpp` — `unified_solve` is a COMMENT-only
  reference (L33, points readers at test_unified_solve's nk arm); no include, no call.
- `tests/solver/test_unified_solve.cpp` — already nk (`UnifiedSolve.NkWorld*`); the
  `unified_solve.hpp` grep hit is the header docstring describing the deletion. No action.
- `tests/coupling/test_coupling_framework.cpp` — `unified_solve` is comment-only
  (no include, no `UnifiedSolve(` call). No action.
- `tests/solver/nk_solve_harness.hpp` — the re-point harness itself; the `unified_solve`
  grep hit is its docstring ("instead of the legacy UnifiedSolve"). KEEP (it IS the seam).

---

## M10 CARRY-FORWARD contracts (go2-locomotion-specific, deleted with the class)

When RL returns at M10 (rebuilt on the unified nk::World), re-assert on nk:
1. **go2 PD standing-on-ground** (was `test_go2_pd_standing`): crouch-hold height
   tracking, Σλ=Mg/4 per foot, COM-inside-foot-polygon static balance, no-tip,
   4096-env clean + deterministic. (Generic foot-ground recoil/friction/D1/scale
   already on nk via foot_ground_subsume + h1_grasp_lift + articulation_contact_rows
   + multi_env; only the go2-stance behavioral envelope is owed.)
2. **go2 per-link contact-wrench readback on a seated stance** (was
   `test_link_contact_wrench` `Go2StanceBalancesWeight`/`D1`/`StepGraphMatchesStep`):
   ΣF_z≈Mg per-link, CONTACT_FORCE==λ/dt, wrench-arg-order cross-check, graph==wrench.
   (Synthetic wrench math + the LINK_CONTACT_WRENCH field binary contract already on nk.)
3. (Inherited from T10) batched autoreset/authority-isolation, un-lagged base pose
   after ResetEnvs, velocity (non-PD) control mode + per-mode two-run byte-exact.

---

## Re-points executed this sub-step (the only code changes)
- `tests/solver/test_link_rigid_twoway.cpp` — UnifiedSolve → nk_harness::NkSolveRows
  (4 call sites) + coloring → nk::SolveSchedule::Partition. GREEN 4/4.
- `tests/solver/test_compliant_pgs_regularizer.cpp` — UnifiedSolve → nk_harness::
  NkSolveRows (RunKernel). GREEN 3/3.
- NO CMake change (nuka_solver_test already links nuka_nk + nuka_phi2 for the
  existing test_unified_solve NkWorld arms; nk_solve_harness.hpp is same-dir).
- NO src/ change, NO golden touched, NO frozen .nks/.nka touched, legacy classes
  NOT deleted (T11-core).
