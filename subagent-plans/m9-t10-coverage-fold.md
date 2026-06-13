# M9 T10 — coverage-fold + T11 deletion-safety checklist (2026-06-14)

T10 folds orphan test coverage into surviving gates BEFORE T11 deletes the legacy
estate, and adversarially re-verifies the M7-flagged `h1_union_parity` vel window
while the legacy oracle is still alive. This file is the per-item verdict ledger:
every coverage-at-risk item from `m9-recon.md` is marked **COVERED** (a surviving
gate already asserts it), **NEEDS-FOLD / FOLDED** (unique → folded into a named
survivor here), **RE-POINT-IN-T11** (unique but cheaply preserved by a T11/T9-style
source re-point, NOT a fold — flagged so T11 does the surgical re-point instead of a
whole-exe delete), or **ACCEPTABLE-LOSS** (redundant, legacy-only-with-golden-
authority, or RL-deferred-to-M10 whose only home is a deleted batched test).

Method: for each item, located its source file(s), checked whether the file `#include`s
a T11-deleted header (deletion order steps 12–16: `unified_solve.hpp`,
`row_solver/row_scheduler`, `narrowphase_dispatch.hpp`, `sdf_contact.*`,
`batched_articulated_world.hpp`, `batched_unified_world.hpp`, `xpbd_world/pbf_world`,
all of `tests/coresident/*`) or links a deleted lib (`nuka_scene_pipeline` IF deleted),
and whether an nk-path twin / standalone survivor asserts the same physics.

---

## PART A — h1_union_parity step-0/window vel re-verification → VERDICT: **BENIGN**

The committed `tests/scenario/h1_union_parity.cpp` widened the FP-floor-window cup
velocity bar `kFloorTolVelWin` 1e-5 → 5e-5 in M7. Re-ran the live parity test (nk vs
the still-alive legacy `BatchedUnifiedWorld`) with per-step + per-component
instrumentation (temporary; reverted — committed file unchanged). Measured:

| step | dvel (cup lin-vel, m/s) | dominant component | dqdot argmax | leg table rows |
|------|-------------------------|--------------------|--------------|----------------|
| 0    | **9.010e-08**           | dvx 7.8e-8         | link29 2.4e-7| 4              |
| 1    | 7.270e-07               | dvy -5.7e-7        | link35 1.9e-6| 3              |
| 2    | 1.373e-06               | dvy -1.28e-6       | link30 1.6e-6| 3              |
| 3    | 1.162e-06               | dvy -1.07e-6       | link40 1.8e-6| 2              |
| 4    | **2.077e-05**  ← SPIKE  | **dvz 1.694e-05**  | link35 1.8e-5| 2              |
| 5    | 6.554e-06  ← recovers   | dvx -5.2e-6        | link36 3.2e-5| 2              |
| 6–10 | 3.1e-6 … 5.5e-6         | mixed              | link35/36/40 | 2→1            |

Window result line: `floor(0..9): dcup=1.273e-07 dvel=2.077e-05 dqdot=6.056e-05`.
Full-run: `nk=1.139e-02 < chaos-floor=1.387e-02` (the 1e-6-perturbed legacy self-twin).
Plan-replay twin BYTE-IDENTICAL after 300 StepPlanned replays.

**Evidence the widening is BENIGN, not MASKING:**
1. **Step 0 is pristine.** dvel = 9.01e-08, dqdot = 2.38e-07 — the pure FMA-contraction
   ULP seed, *below* the M4-cited 2–3e-7 expectation and ~110× under the 1e-5 bar (and
   under 1e-6). A real structural composition bug is ~1e-3+ at step 0 (1.5+ orders over
   even 5e-5); the step-0 bars are UNCHANGED at 1e-6/5e-6 and pass with huge margin. The
   catcher with the most discriminating power is untouched.
2. **The excess is ONE single-step spike at step 4, then RECOVERS.** dvel goes
   9e-8 → 7e-7 → 1.4e-6 → 1.2e-6 → **2.08e-5 (step 4)** → 6.6e-6 (step 5) → back to the
   3–5e-6 band. This is the signature of a contact-event transient, NOT growth. A real
   nk-vs-legacy divergence would grow monotonically (or step and STAY), not spike-and-
   recover in one step.
3. **The spike is concentrated on the cup VERTICAL (z) velocity** (dvz = 1.694e-5 of the
   2.077e-5; the x/y are ~8–9e-6), i.e. the NORMAL/support direction — exactly where the
   cooked IC's zeroed settle-residual velocity is felt, and exactly the direction a one-
   step-different table-contact normal impulse moves the cup.
4. **It coincides with the table-contact set shedding** (leg table rows 4→3→3→2→2 over
   steps 0–4; nk AGREES on the count at every in-window step — `count_fork @16`, well
   after the window). So this is not a contact-COUNT disagreement; it is a sub-ULP
   IMPULSE difference at the moment the cup is actively settling off the table rows. The
   cooked IC bakes a POSE-only initial_state (the gripper_proto's residual settle qdot /
   link_velocity zeroed, per `union_cook`), so the two worlds — reading the SAME cooked
   template — ride a marginally different chaotic trajectory whose ULP seed crosses one
   contact toggle as a single-step z-velocity transient. This matches the M7 header note
   verbatim (measured 2.08e-5 at step 4, recovering to ~5e-6 by step 5).
5. **The chaos bound proves non-masking globally:** over the full 300-step run, the
   nk-vs-legacy divergence (1.139e-2) stays BELOW the legacy self-chaos floor (1.387e-2)
   — nk tracks legacy *better* than a 1e-6 perturbation of legacy tracks itself. A real
   divergence would push nk-vs-legacy above the self-chaos floor.

**Comparison to the M4 nk-vs-legacy scaled-bar precedent:** M4 established the per-step
ULP seed at ~2–3e-7 (FMA contraction across TUs; same expressions, different fold), with
the union Lyapunov rate (~1.3×/step) amplifying it past 1e-6 within ~10 steps. The
step-0 seed here (9e-8) is *under* that. The 5e-5 window is ~100× the step-0 seed, but
the actual in-window peak (2.08e-5) is only ~2× the old 1e-5 bar and lives for ONE step
on the contact-toggle axis. This is the same class of phenomenon M4 already accepted
(numerically-equivalent-not-bit-identical solve + chaos amplification), here with one
extra contact-event transient from the valid-but-different cooked IC.

**Verdict: BENIGN.** The 5e-5 vel window absorbs a genuine single-step, z-concentrated,
contact-toggle settle-residual transient without losing the structural-bug-catching
power (step-0 bars unchanged at 1e-6/5e-6 and passing with ~110× margin; the global
chaos bound confirms nk < legacy-self-chaos). The committed window was NOT changed
(no MASKING found). This entire witness DIES with `h1_union_parity` in T11 — which is
why the unique non-legacy coverage (plan-replay D1) is folded in PART B below.

---

## PART B — plan-replay D1 fold → **FOLDED into h1_grasp_lift**

`h1_union_parity` had two coverages: (1) nk-vs-LEGACY step-0/window parity — dies with
the legacy class (the golden/nk path is authority; no fold, this is the PART A witness);
(2) nk **plan-replay D1** — the cooked union nk::World stepped via `StepPlanned()` (the
CUDA-graph plan: ONE capture, 300 replays with AssembleRows/SolveRowsBlockIsland inside
the graph) lands BYTE-IDENTICAL to the `Step()` world.

Was (2) already covered? **No.**
- `h1_grasp_lift`'s existing D1 (`GraspHoldBiteDisturbanceFromNks`) is a two-run
  `Step()`-vs-`Step()` memcmp (run-to-run determinism) — it never calls `StepPlanned`.
- `coupled_grasp_soft.cpp` (`XpbdTetDeterministicAndPlanIdentity`) DOES assert
  `StepPlanned == Step` byte-identity, but ONLY on the TET/PARTICLE coupled scene — NOT
  the H1 UNION articulation row-solve graph (51-DOF floating base + feet + 30 fingertips
  + cup + table). The articulation island/segment graph capture is a distinct code path.

So `StepPlanned == Step` byte-identity on the UNION articulation+contact scene is unique.
**Folded:** added `TEST(H1GraspLift, GraspPlanReplayByteIdenticalFromNks)` to
`tests/scenario/h1_grasp_lift.cpp`. It cooks the .nks union world, runs the
approach→close→lift hold via `Step()` while recording the per-step `DriveTarget` stream
(closed-loop torque computed from the world's own q/qdot, exactly as `h1_union_parity`
records it), then replays the recorded stream through `StepPlanned()` on a fresh world
(table-off toggle at `kLiftAt` mirrored) and asserts cup pose/vel/angvel + per-link qdot
are byte-identical (memcmp == 0). Asset-gated SKIP like the rest of the gate. Uses the
existing cook path (`CookSceneToUnionTemplate` → `BuildNkUnionModel` → `nk::World`).

RESULT (live): `300 StepPlanned replays vs Step -> cup pose memcmp=0 vel=0 angvel=0
qdot=0`. PASS, 3.46 s.

---

## PART C — coverage-at-risk per-item verdicts

Legend: COVERED | RE-POINT-IN-T11 | FOLDED | ACCEPTABLE-LOSS

| # | item | source / home | dies in T11? | verdict |
|---|------|---------------|--------------|---------|
| 1 | dof_above18_honesty (G0 51-DOF M⁻¹/J) | `tests/runtime/test_dof_above18_honesty.cpp` | **NO** — links `nuka_runtime`/`nuka_articulation_gpu`, includes only the SURVIVING articulation math primitives (featherstone_aba / articulation_contacts / articulation_jacobian) + the kept `buffer_legacy` shim | **COVERED** — survives standalone; ALSO h1_grasp_lift cooks+steps the full 51-DOF (`ASSERT_EQ(dof_stride,51)`) |
| 2 | batched_reset (RL autoreset) | `tests/runtime/test_batched_reset.cpp` | **YES** — `#include batched_articulated_world.hpp` | **ACCEPTABLE-LOSS (M10)** — RL episode-boundary autoreset; nk::World has `Reset(env_ids)` but the batched-autoreset/authority-isolation semantics are RL-surface, deferred to M10 (recon §rulings #3, §risk register). Rebuild on nk at M10. |
| 3 | base_pose_view (episode-boundary un-lagged pose) | `tests/runtime/test_base_pose_view.cpp` | **YES** — `#include batched_articulated_world.hpp` | **ACCEPTABLE-LOSS (M10)** — the one-step-FK-lag / un-lagged-after-ResetEnvs contract is the RL episode-boundary contract; only home is the deleted batched test. FLAG: M10 must re-assert that nk step/FK reproduces the un-lagged base pose after Reset (recon risk register). |
| 4 | control_modes (torque/vel) | `tests/runtime/test_control_modes.cpp` | **YES** — `#include batched_articulated_world.hpp` | **ACCEPTABLE-LOSS (M10)** — velocity-control is non-PD control surface, batched-only, RL-adjacent → M10. The torque-drive path IS covered by the nk gates (h1_grasp_lift / featherstone oracle drive q/qdot torque); only the VELOCITY-mode + two-run-byte-exact-per-mode is lost → rebuild with the M10 non-PD control surface. |
| 5 | solref_solimp host==device byte-exact | `tests/constraint/test_solref_solimp.cpp` | **NO** — no deleted includes; standalone | **COVERED** — survives (`SolrefSolimp.FunctionMatchesHandEvaluatedTable` memcmp host vs device; `DAndRNotSwapped`). ALSO mirrored on nk in `test_unified_solve` `NkWorldRestingContactSupportsWeight` (`EXPECT_NEAR(row_R, expect.R)` / `row_rhs`). |
| 6 | test_unified_solve compliant-rows | `tests/solver/test_unified_solve.cpp` | **partial** — `#include solver/unified_solve.hpp` (legacy arms die) BUT the file ALSO carries nk twins | **COVERED** — `NkWorldRestingContactSupportsWeight` / `NkWorldStifferContactPenetratesLess` / `NkWorldPyramidFrictionGripsHighMuSlidesZeroMu` / `NkWorldD1AndCrossReplica` mirror every legacy `UnifiedSolve.*` assertion (compliant-row R/rhs, penetration, lambda-sum=weight, friction cone, D1, cross-replica). T11/T9 re-point: drop the legacy `UnifiedSolve.*` arms + the `unified_solve.hpp` include, keep the `NkWorld*` arms. |
| 7 | SDF precision oracle (4/4 cell-tol) | `tests/collision/test_sdf_tier_wired.cpp` | **YES** — `#include narrowphase_dispatch.hpp` (+ tests `sdf_contact.hpp` `find_sdf_contact_newton` THROUGH the dispatch — both deleted step 13) | **ACCEPTABLE-LOSS** — this gate tests the LEGACY SDF tier wired through `ResolveNarrowphase` + `find_sdf_contact_newton`, BOTH of which T11 deletes (the new core uses `contacts_union.cu`, not this dispatch). Testing deleted infra. NOTE: SDF *determinism* (the cook) survives separately (#16). |
| 8 | gjk/epa SphereHull shallow-pen monotonicity | `tests/collision/test_gjk_epa_convex.cpp` (`SphereHullShallowPenetrationIsMonotone`) | **YES** — `#include narrowphase_dispatch.hpp` (the test calls `ResolveNarrowphase(...)` to reach cvx::SphereHull) | **RE-POINT-IN-T11** — UNIQUE (the test itself flags: "NO always-on test catches a SphereHull monotonicity revert") AND the underlying `cvx::SphereHull` lives in the KEPT `convex_narrowphase.hpp` (controller refinement: KEEP the HD cvx header). NOT a cheap fold (pulls V-HACD cup geometry from `sdf_test_meshes.hpp`). T11 ACTION: drop the `narrowphase_dispatch.hpp` include + the `ResolveNarrowphase`-routing tests (RoutesToRealConvexHandler etc. — test deleted infra), re-express the geometry tests (incl. SphereHull-monotone, HullVsSpherePrimitive, BoxBoxAsHull, TetraTetra) to call the `cvx::` handlers DIRECTLY → exe stays alive, coverage intact. Do NOT delete the whole exe. |
| 9 | broadphase D1 | `tests/collision/test_lbvh_vs_sap_pair_set.cpp` + `test_lbvh_filtered_pairs.cpp` | **NO** — both standalone | **COVERED** — `LbvhVsSap.D1TwoRunByteExactPairList` + `LbvhFilteredPairs.D1TwoRunByteExactAndScale` (both survive). The coresident broadphase usages are setup, not the D1 witness. |
| 10 | Philox-KAT RNG D1 | `tests/sensor/test_n1_gaussian.cu` | **NO** — no deleted includes | **COVERED** — survives standalone (the coresident reference was a consumer, not the KAT witness). |
| 11 | Krylov CG/MINRES Eigen cross-checks | `tests/diffsim/solver/test_{cg_vs_dense,minres_vs_dense_indefinite,gmres_nonsymmetric,minres_indefinite,ilu0_vs_jacobi}.cpp` + `tests/diffsim/test_cg_vs_dense.cpp` | **NO** — all standalone | **COVERED** — all survive. |
| 12 | XPBD/PBF cooker determinism | `tests/import/test_xpbd_cooker.cpp` (`XpbdCookerDeterminism.RepeatedCooksAreByteIdentical`) | **re-point** — `#include runtime/soft/xpbd_world.hpp` (only the GPU stepper dies, step 15; the param structs RELOCATE into the cooker FIRST per recon) | **COVERED (via T11/T13 re-point)** — the determinism witness tests the COOKER (byte-identical constraint-set cook), not the GPU stepper. Recon plan: relocate `XpbdParticleSet/ConstraintSet/...` into `xpbd_cooker` first, then the test includes the relocated struct header instead of `xpbd_world.hpp`. Coverage survives. |
| 13 | scene_ir / compose invariants | `tests/scene/test_scene_ir.cpp` + `test_scene_compose.cpp` (bundled in exe `nuka_scene_test` which also has `test_scene_pipeline.cpp` and links `nuka_scene_pipeline`) | **conditional** — only if the whole `nuka_scene_test` exe is deleted because `nuka_scene_pipeline` is deleted | **RE-POINT-IN-T11** — FINDING (refines recon): `scene_pipeline`/`ScenePipeline` has ZERO live src consumers (`render_scene.cpp` has its OWN `BuildRenderScene` in the `render` namespace — NOT the scene_pipeline one; only `scene_pipeline.cpp` + `pose_graph.cpp` + `test_scene_pipeline.cpp` reference it) → it is an ORPHAN, deletable in M9. T11 ACTION: drop ONLY `test_scene_pipeline.cpp` from the `nuka_scene_test` exe + relink `nuka_scene_pipeline → nuka_scene` (compose/scene_ir live in `nuka_scene`, NOT pipeline). This KEEPS `test_scene_ir.cpp` (11 IR invariants) + `test_scene_compose.cpp` (12 remap invariants: body/joint/actuator/sensor remap, root placement, prefix, BaseUnchanged, DeterministicAcrossRuns) + `test_scene_cooker` + `test_contact_metadata_cook` + `test_filter_precedence` IN PLACE — NO fold needed. ALSO: `tests/scenario/scene_roundtrip.cpp` (survives, links `nuka_scene`) already USES `Compose()` and asserts roundtrip fidelity, and `test_scene_compose_h1_cup_table` (survives, links `nuka_scene_cook`) exercises 3-source compose on the real H1+cup scene — so compose is doubly anchored even on the conservative read. |
| 13b | scene_pipeline ScenePipeline views | `tests/scene/test_scene_pipeline.cpp` (6 tests) | **YES** — tests the deleted `ScenePipeline` class itself | **ACCEPTABLE-LOSS** — these test the orphan being deleted (ProgrammaticSceneBuildsAllInternalViews / ComputesWorldTransforms / Imported{Mjcf,Usd}BuildsPhysicsAndRenderViews / RenderSceneKeeps... / AppliesRuntimeState...). Coverage of a deleted class = legitimate loss. |
| 14 | phi v1 per-handle-device | `tests/phi/test_device_context_per_handle.cpp` | **NO** — standalone | **COVERED** — survives. |
| 15 | test_diffsim_tape BackwardBitIdentical | `tests/c_abi/test_diffsim_tape.cpp` | **NO** | **COVERED** — survives (recon pre-marked COVERED; the canonical D1 transition with VERBATIM tols). |
| 16 | test_aba_reverse_fd NkArenaSeam | `tests/diffsim/test_aba_reverse_fd.cpp` | **NO** | **COVERED** — survives (recon pre-marked COVERED). ALSO covers SDF-cook determinism's sibling — sparse-SDF cook determinism survives in `tests/import/test_sparse_sdf_determinism.cpp` (`RepeatedCooksAreByteIdentical` / `CookedSdfTableIsByteIdenticalAcrossCooks`). |
| 17 | featherstone *VsLegacy arms | `tests/oracle/featherstone_oracle_harness.cpp` | already dropped in T9 | **ACCEPTABLE-LOSS** — the golden (FeatherstoneAba / owner-provided MJX goldens) is authority; the legacy-comparison arm was dropped in T9 (`NkWorldBatchedContactStepPlannedByteExact` replaced it). The ABA-vs-MJX + NkWorld byte-exact + owner-golden compares STAY (reference = FeatherstoneAba/BuildWorld, kept). |
| 18 | h1_union_parity nk plan-replay D1 + step-0 byte | `tests/scenario/h1_union_parity.cpp` | **YES** (deletion order step 2) | **FOLDED (plan-replay) + ACCEPTABLE-LOSS (step-0-vs-legacy)** — plan-replay D1 folded into h1_grasp_lift (PART B). The step-0 nk-vs-LEGACY byte parity dies with the legacy class (golden/nk path is authority; PART A rendered its BENIGN verdict while the oracle was alive). |

### PART C tallies
- **COVERED:** 10 — #1, #5, #6, #9, #10, #11, #12, #14, #15, #16 (and the survivor halves of #18, #13)
- **RE-POINT-IN-T11 (unique, preserved by source re-point, NOT a fold):** 3 — #8 (SphereHull monotone), #13 (scene_ir/compose), and #6-style legacy-arm-drop on #6
- **FOLDED (this task):** 1 — #18 plan-replay D1 → h1_grasp_lift
- **ACCEPTABLE-LOSS:** 6 — #2, #3, #4 (all RL-deferred → M10), #7 (deleted SDF dispatch infra), #13b (deleted ScenePipeline class), #17 (golden is authority); + the legacy-arm of #18.

Net: exactly ONE genuine fold was needed (plan-replay D1, the recon's named T10 target).
Everything else either already survives, is preserved by a cheap T11/T9 source re-point
(flagged here so T11 does surgery not a whole-exe delete), or is legitimate loss
(deleted infra / golden-is-authority / RL-deferred-to-M10).

---

## T11 deletion-safety action list (derived from PART C)
Before/at T11, for a SAFE deletion:
1. **Drop legacy arms, keep nk twins, relink** (NOT whole-file delete):
   - `test_unified_solve.cpp`: drop `UnifiedSolve.*` legacy arms + `unified_solve.hpp`
     include, keep `NkWorld*` arms.
   - `test_gjk_epa_convex.cpp`: drop `narrowphase_dispatch.hpp` include + the
     `ResolveNarrowphase`-routing tests; re-express geometry tests (incl.
     `SphereHullShallowPenetrationIsMonotone`) to call `cvx::` handlers directly
     (KEPT `convex_narrowphase.hpp`).
   - `nuka_scene_test` exe: drop `test_scene_pipeline.cpp` source; relink
     `nuka_scene_pipeline → nuka_scene`. Keep the other 5 scene unit files in place.
   - `test_xpbd_cooker.cpp`: swap `xpbd_world.hpp` include → the relocated param-struct
     header (after T11/T13 relocates the structs into the cooker).
2. **Delete outright (ACCEPTABLE-LOSS):**
   - `test_sdf_tier_wired.cpp` (legacy SDF dispatch infra, deleted with step 13).
   - `test_scene_pipeline.cpp` (deleted ScenePipeline class).
   - `test_batched_reset.cpp` / `test_base_pose_view.cpp` / `test_control_modes.cpp`
     (RL-adjacent, batched-only → M10) — **M10 MUST re-assert** their contracts on nk:
     batched autoreset/authority-isolation, un-lagged base pose after ResetEnvs, and the
     velocity (non-PD) control mode + per-mode two-run byte-exact.
3. **No action — survives as-is:** test_dof_above18_honesty, test_solref_solimp,
   test_compliant_rows, test_lbvh_vs_sap_pair_set, test_lbvh_filtered_pairs,
   test_n1_gaussian, all diffsim Krylov tests, test_device_context_per_handle,
   test_diffsim_tape, test_aba_reverse_fd, test_sparse_sdf_determinism, scene_roundtrip,
   test_scene_compose_h1_cup_table.

## M10 carry-forward (RL-surface contracts whose only home T11 deletes)
- batched autoreset + authority-isolation (was `test_batched_reset`).
- un-lagged base pose after ResetEnvs / one-step-FK-lag boundary (was `test_base_pose_view`).
- velocity (non-PD) control mode + per-mode two-run byte-exact (was `test_control_modes`).
Rebuild these on the nk world when RL returns at M10.
