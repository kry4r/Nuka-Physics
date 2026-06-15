# M10 — H1 cup-grasp RL on the unified nk::World (execution plan)

> Synthesized from a 3-agent recon (2026-06-15) of the POST-refactor codebase.
> Binding constraint set = `docs/specs/2026-06-10-h1-whole-body-rl-grasp-spec.md`
> (goal/gates/honesty bars still bind; its impl anchors `BatchedUnifiedWorld`/
> coresident are DELETED — rebuild on nk::World per [[unified-world-no-special-grasp-binding]]).
> Owner greenlit RL 2026-06-15. Demo layout LOCKED = H1 facing out at the ISLAND
> table, kitchen backdrop (out/m10/preview/00_H1_island_frontfacing_LOCKED.png).

## Architecture shift (vs the SG-spec's anchors)
- The SG-spec targets `UnifiedCoResidentStepper::StepStandGrasp` + `BatchedUnifiedWorld`.
  Both DELETED in M9. The surviving union path is:
  `nks::Load(scene)` → `cook::CookSceneToUnionTemplate(scene,N)` →
  `coresident::BuildNkUnionModel(tmpl,N)` → `nk::World(model,N,dev,backend,cfg)`.
  (Exactly what `examples/demo/h1_grasp_video.cpp:327-561` does, but at N=1.)
- The generic single-articulation path is `cook::CookToModel` (what the C-ABI
  `create_from_scene` uses today — NOT the union).

## Gate status after recon
- **G0 (DOF honesty) = SATISFIED.** Refactor raised the cap to `kMaxArticulationDof=64`
  (≥51) with LOUD construction/launch guards (crba.cu:354, solve_rows.cu:805/857);
  old 18-DOF silent clamp removed (articulation_contacts.cu:465 comment). The
  surviving `kMaxContactSolverDof=18` is only in uncalled legacy drive launchers +
  diffsim adjoint kernels — not the forward grasp path. **No G0 work needed.**
- **G1 (batched union world): PARTIAL.** Batched step + CUDA-graph + env-major +
  torque drive + 51-DOF union cook accepting N>1 all EXIST in C++. N>1 union is
  BUILT-BUT-UNEXERCISED (demo N=1). Gaps below.
- **G2 (python substrate): MISSING the union bridge.** Generic PPO python stack
  EXISTS + works (Go2, H1-stand); H1-grasp env DELETED (M9). C-ABI can't reach the
  union world from python yet.
- **G3 (staged RL S1-S5): not started.**

## The gaps to close (engine pre-work, in priority order)
1. **★ THROUGHPUT MEASUREMENT (the feasibility gate, SG-spec G1d).** Build a C++
   harness: cook the H1+cup+table union at N∈{256,1024}, `StepPlanned` loop, measure
   env-steps/sec. Bar: one PPO stage (~100M env-steps) < 24 h ⇒ need ≳1160 env-steps/s
   sustained. Prior concern (pre-refactor): ~180/s was infeasible. Go2 frozen baseline
   is ~1.1 µs/env-step @4096 — IF the union is comparable, training is feasible.
   THIS DETERMINES THE PATH: fast ⇒ proceed to env; slow ⇒ a NAMED optimization first.
   Also validates the union steps correctly at N>1 (D1 two-run byte-identity).
2. **Union contact obs.** `ReadoutContactWrench` is `!is_union`-gated
   (pipeline.cpp:355) ⇒ per-link wrench / fingertip normal impulse NOT filled on the
   union world. The grasp reward needs finger normal impulse (force-closure hold) +
   foot contact state. Wire union contact readout (or compute the needed scalars).
3. **Per-env randomized IC reset.** `ResetEnvs` restores the SINGLE cooked snapshot
   only (readout.cu:107). RL needs per-env randomized initial conditions (cup pose,
   base pose, q jitter). Add a masked per-env IC SET (base_pose/q/cup-pose) op, or a
   set-then-snapshot path. Domain-randomization ops are declared but UNREGISTERED.
4. **Python union bridge.** Either extend the C-ABI to cook the UNION
   (CookSceneToUnionTemplate→BuildNkUnionModel) from the .nks + expose torque control
   + the obs fields, OR train in C++. (Generic stack uses NukaGymEnv on
   `create_from_scene`, which only does generic CookToModel + ingests .xml/.usd not
   .nks.) Decision deferred to after throughput (may train via a new c_abi union entry).

## RL build (after gaps 1-4)
- Reuse `python/nuka/gym/env.py` (NukaGymEnv: on-GPU obs/reward/masked autoreset) +
  `python/nuka/rl_games/vecenv.py` + `examples/training/train_*_ppo.py` + the proven
  obs/action/step recipe in `examples/sim_val/go2_policy_drive.py`. H1StandEnv
  (`python/nuka/tasks/h1_stand.py`, torque control) is the closest exemplar.
- Author `GraspEnv` (obs: q/qdot env-major, base+cup pose/vel, fingertip world pos,
  per-finger normal impulse, foot contact, last action; action: per-DOF torque
  clamped to limits; reward: dense shaping; eval: SEPARATE deterministic evaluator).
- Curriculum S1→S5 (SG-spec §G3), ONE env class config-gated, warm-start chained.
  Train to convergence (multi-session GPU time). Honesty bars (SG-spec §3) apply.

## Demo render (after a trained grasp policy)
- Dump the trained grasp rollout (à la go2_dump_walk_trajectory.py) → replay link
  poses through scene_still/a video tool in the LOCKED island layout (H1 facing out,
  cup on island, kitchen backdrop) → mp4. Per-foot contact shadows (already in the
  renderer) + the grounded studio look.

## Key paths
- Union cook bring-up: `examples/demo/h1_grasp_video.cpp:327-561`.
- Cook: `src/scene/cook/{union_cook,union_nk_model}.*`. World: `src/nk/pipeline/world.*`.
- Contact readout gate: `src/nk/pipeline/pipeline.cpp:355`. Reset: `readout.cu:107`.
- PPO stack: `python/nuka/{gym/env.py,rl_games/vecenv.py,tasks/h1_stand.py}` +
  `examples/training/train_h1_stand_ppo.py`. Recipe: `examples/sim_val/go2_policy_drive.py`.
- Perf: `tests/perf/{test_go2_4096env_step_time.cpp,nk_union_n1.cpp}`,
  `out/perf/baseline_rtx4000ada_4096_frozen.json`.
- Scenes: `examples/scenes/h1_cup_table.nks` (FROZEN union scene, has grasp block).

## Sequenced execution (post-recon 2026-06-15, owner roadmap: 完善RL → 重训go2[台阶+深度传感] → H1抓杯放下课程)

Three-agent recon produced precise anchors. Execution order by dependency/risk:

### Phase A — additive RL-completion engine ops (no python, fully D1-verifiable) ← STARTED
- **A1 Union contact obs.** New `NkOp::ReadoutUnionContactObs` (enum op_schema.hpp readout group;
  params near `ExportObsParams` op_schema.hpp:451). Kernel+entry in readout.cu (sibling of
  OpReadoutContactWrench), registered RRegisterNkReadoutOps() readout.cu:396. Reads `union_slots`
  (cls/link/row_base lane14 via LoadUnionSlot union_types.cuh:69), `lambda` (per-row impulse,
  written back solve_rows.cu:766), `ucontact_count`. Per env writes `[n_fingers normal-impulse |
  n_feet contact-scalar]` into `obs_buffer` (FieldId::ObsBuffer, fields.yaml:448, currently written
  by NO in-graph op — ExportObs not in graph). Dispatch gated `is_union` after pipeline.cpp:355
  block. Slot order FIXED feet→fingers→table (union_nk_model.cpp:136-188). VERIFY vs the host
  reduction `h1_grasp_lift.cpp:193-215` already does.
- **A2 Per-env IC randomization.** Extend ResetEnvsKernel (readout.cu:107) with gated per-axis
  Philox jitter (cup body XY, base pos, q) — `#include "sensor/noise/philox.cuh"` (host+device,
  no redline CUDA leak). New ResetEnvsParams fields (op_schema.hpp:460) ic_seed/ic_episode/
  jitter_* all default 0 → byte-identical reset (D1). world.cpp:351 bumps ic_episode/Reset.
  Cup reset_jitter (union_scene_template.hpp:186, currently DEAD) is the wiring target later.
- VERIFY A: build-cuda128 only; existing union tests (h1_grasp_lift, nk_union_n1, union_cook_golden,
  test_unified_solve) green; a fused/go2 golden test still green (proves additive ops leave the
  fused StepPlanned graph byte-identical); nonzero jitter perturbs per-env, zero jitter identical.

### Phase B — C-ABI/python bridge (enables actual training)
- **B1 Torque gate.** Lift world.cpp:144-146 PD-only rejection + buffer.cpp:79-81 TORQUE_INPUT
  NOT_SUPPORTED for torque worlds (engine law LaunchApplyTorqueDriveKernels EXISTS,
  articulation_drives.hpp:38). Un-breaks the already-written H1StandEnv (python/nuka/tasks/h1_stand.py).
- **B2 Union C-ABI create.** New `nuka_world_create_union_from_nks` (mirror world.cpp
  create_from_scene): nks::Load → CookSceneToUnionTemplate → BuildNkUnionModel(tmpl,N) → nk::World
  (the union model self-selects drive_mode=1 torque). Expose union obs/action fields via
  buffer.cpp + dlpack_table. Per-env reset already works (nuka_world_reset_envs → World::Reset).
- **B3 nanobind binding** (nuka_ext.cpp:826 pattern) + GraspEnv task (h1_stand.py is the torque
  exemplar) + vecenv.py:52 factory. Smoke a torque step from python.

### Phase C — Go2 retrain demo (stairs + depth sensor) — GREENFIELD
- **C1 Terrain.** Only a flat scalar plane exists (model.ground_height; contacts_foot.cu:80 hard z).
  Build a heightfield/stair ground rep + foot-vs-heightfield narrowphase op + cook ingest. Net-new.
- **C2 Depth/height-scan sensor.** Only a CPU stub (ray_sensor.cpp:18 returns max range) + a
  render-only RT path (rt/bvh_ray_traversal.cu RenderDepth). Build a device height-scan op (can reuse
  RT BVH/ray_box machinery), a new readable obs field, C-ABI buffer-view, python obs builder. Net-new.
- **C3 Retrain** go2 with terrain+perception obs → converge → video (reuse go2 PPO stack +
  go2_dump_walk_trajectory.py → go2_walk_video.cpp render).

### Phase D — H1 grasp curriculum (S1 reach+grasp+hold → S2 lift → S3 stand → S4 co-train → S5 place)
- GraspEnv (obs from A1 + q/qdot/base+cup pose/fingertip world pos/foot contact + last action;
  torque action; dense reward; SEPARATE deterministic evaluator). Curriculum config-gated, warm-start
  chained, trained to convergence (multi-session GPU). Honesty bars SG-spec §3. Demo render in the
  LOCKED island layout.
