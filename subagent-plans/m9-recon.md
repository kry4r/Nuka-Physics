# M9 recon + 施工图 (controller, 2026-06-14)

Source: ultracode recon `wf_884ceb0a-ec4` (6 blind facet finders → synthesis). Full
result archived in the workflow transcript. This file = the executable M9 plan with
owner rulings + controller refinements folded in.

## Verdict
M9 = the biggest, IRREVERSIBLE milestone: switch the whole C-ABI off the two legacy
steppers (single-env Featherstone `StepWorldGpu` + multi-env `BatchedArticulatedWorld`)
and the two legacy union/grasp shims onto `Scene→CookToModel→nk::World`, op-ify the
diffsim backward, delete the entire legacy estate, reorganize tests into 4 dirs / 5 exes.
NOT a pure deletion: ~5 "delete" targets are still LIVE-consumed; 2 plan-named delete
targets are LOAD-BEARING for the NEW core. ~150+ files deleted, 6 C-ABI modified, 2
created, 92-file buffer_legacy migration, full test reorg. ~15 build-coupled-serial tasks.

## ★ OWNER RULINGS (2026-06-14, AskUserQuestion)
1. **git-grep-ZERO gate** = **SCRUB** the ~12 benign 1:1-provenance comments in new-core
   files (fields.yaml/model.hpp/op_schema.hpp/articulation.cu/schedule.hpp/world.hpp/
   contact_wrench.hpp/row_articulation_refs.hpp/control_mode.hpp …) so the literal
   `git grep -lE "BatchedUnifiedWorld|BatchedArticulatedWorld|BatchedSceneTemplate|UnifiedCoResidentStepper"`
   returns zero. (Owner chose the literal-gate option over re-scoping; accept the
   provenance-history loss.)
2. **diffsim backward op granularity** = controller's call: "又快又对，尽量遵从计划".
   → Target the FAST+CORRECT form with adjoint-FD byte-identity as the HARD arbiter.
   Default = ONE monolithic `NkOp::StepBackward` (kernel body verbatim, no per-stage
   adjoint device-materialization → preserves in-register fixed-order D1, no extra
   round-trips = also fastest). Honor §3.10 by expressing AbaBackward/IntegrateBackward/
   SolveRowsBackward as INTERNAL `__device__` helpers WHERE that costs no device
   round-trip. If (and only if) a finer op split keeps `test_aba_reverse_fd` /
   `test_diffsim_tape` adjoint-FD byte-identical, follow §3.10 more literally. NEVER
   regenerate goldens; the FD口径/tolerances stay verbatim.
3. **Grasp/union RL python** = ALL RL deferred to M10 (after M9+M11, RL is LAST).
   → M9 DELETES grasp/union RL tasks + bindings outright (GraspWorld/UnionWorld bindings,
   H1GraspEnv, train_h1_grasp_ppo, grasp_catch_eval, union smoke/gate scripts). Do NOT
   expose nk union/grasp to python in M9. Rebuild RL on the nk world at M10. (Matches SG
   freeze-till-M10.) The `nuka_scene.*` + `python/nuka/scene.py` SCENE-AUTHORING surface
   is STILL authored in M9 — that is not RL.
   ★ **UNIFIED-WORLD CONSTRAINT (owner 2026-06-14, [[unified-world-no-special-grasp-binding]]):**
   NO special Grasp/Union world bindings, EVER. There is ONE unified world (`nuka_world` over
   Scene→CookToModel→nk::World). Grasp-vs-union-vs-other behavior = IMPORTED SCENE (.nks grasp
   block etc.) + a per-scene CONTROL SCRIPT selected for that scene (the `h1_grasp_choreo.py`
   choreography/drive-table layer driving the generic world via drive-target/torque fields) —
   NOT a world type. So T8 must NOT replace the deleted bindings with any new nk-grasp/nk-union
   special binding; the drive-table selection + phase schedule that legacy `union_world.cpp::
   SelectDriveTable` baked in belongs in the SCRIPT. The relocated `BatchedSceneTemplate`/
   `H1UnionDriveEntry` + `BuildNkUnionModel` are the general scene→model COOK (legit), not a
   special world; drive DATA lives in the scene, applied by the script.
4. **RT/render_scene/buffer_legacy-rt** = SLIP to M11. → M9 does NOT migrate
   `rt/scene_render.cu`/`rt/two_level_render.cu` off `render_scene`, does NOT delete
   `render_scene.{hpp,cpp}`, keeps a MINIMAL `buffer_legacy` shim for `rt/*`. M9 migrates
   only the NON-RT buffer_legacy consumers. Full render_scene + buffer_legacy deletion →
   M11 (where rt/* is reworked = RT backend interface-ification). DROPS recon T12 from M9;
   T13 becomes partial.

## ★ CONTROLLER REFINEMENTS (strictly-dominating; literal plan would break the build)
- **KEEP** `src/collision/convex_narrowphase.hpp` HD `cvx::` GJK/EPA/SphereHull header —
  the NEW core's `contacts_union.cu` (cvx::SphereHull @196), `narrowphase_prims.cu`
  (analytical_manifold), `broadphase.cu` include it. Delete ONLY the legacy host
  `ConvexNarrowphase` dispatch (`narrowphase_dispatch.hpp`) + optionally the SphereHull
  EPA shallow-pen special-case if owner later wants the dead-band cleanup.
- **RELOCATE** plain-data `BatchedSceneTemplate` + `H1UnionDriveEntry` out of the doomed
  `runtime/coresident/batched_unified_world.hpp` into a NEW `src/scene/cook` header (no
  kernels/device) → unblocks the coresident dir delete cheaply. Native
  `CookToModel→UnionCsr` does NOT exist (only the M9-deferred note at union_cook.hpp:28);
  relocation satisfies the gate without building a new native cook.
- **KEEP** `world.cpp` ingesting `.usda/.xml(MJCF)` via `LoadSceneByExtension`→SceneIR→
  CookToModel so the ~12 `World.create_from_scene` (go2_float.usda / h1_reduced_train.xml)
  + go2/h1-stand consumers don't break. (RL itself deferred to M10, but the create-path
  asset ingestion must survive for non-RL consumers + future RL.)
- **RELOCATE** soft/fluid PARAM structs (XpbdParticleSet/ConstraintSet/cloth/tetmesh
  topology + PBF params) into the cookers FIRST (alive `cook_to_model` path), delete only
  the `xpbd_world.cu`/`pbf_world.cu` GPU steppers.

## Task breakdown (updated for rulings; T12 dropped, T13 partial)
- **Phase 0 (prereqs, compile alongside legacy; logically ∥ but build-coupled-serial):**
  - T1 [HIGH] FieldId↔nuka_state_field_t bridge + buffer.cpp consult dlpack_table. RL HARD
    CONTRACT: keep all 20 public field ints (0..19), element_stride_bytes, dtype, shapes
    BYTE-IDENTICAL (dtype==1/u32 ONLY for CONTACT_LINK; quat-W-FIRST stride28; omega-first
    stride24; env-major float stride4). Pin with test_nuka_dlpack.py + test_multi_env_world
    + test_drive_target_io BEFORE and AFTER. Single most error-prone task.
  - T2 [MED] DeviceRecord acquires owned phi::Device*/phi::Backend* (InitBestDevice/
    DeviceInitBackend on create, BackendFree on destroy) so nk::World ctor + cook::Settle
    callable from C-ABI. Prereq for T4.
  - T3 [MED] Relocate BatchedSceneTemplate + H1UnionDriveEntry → src/scene/cook header;
    re-point union_cook.{hpp,cpp} (drop coresident include @38), C-ABI union path, tests.
- **Phase 1 (switch; legacy still present):**
  - T4 [MED] Author nuka_scene.h + c_abi/scene.cpp + Scene nanobind + python/nuka/scene.py
    (load/compose/find/set_local/set_physics_material/settle/save/destroy). Heed the
    Get*Mut dirty-facade + EntityId-invalidation hazard (resolve nodes AFTER mutations).
    deps T2.
  - T5 [HIGH] Rewrite world.cpp create → CookToModel→nk::World; delete StepWorldGpu inline
    (314-449) + both legacy create arms; internal.hpp WorldRecord → unique_ptr<nk::World>.
    KEEP .usda/.xml ingestion + nuka_world_create_from_scene signature. Preserve go2_stand
    golden bit-exact. deps T1,T2.
  - T6 [HIGH] diffsim.cpp 1-line arena seam → MakeArticulationDeviceStateFromViews
    (articulation_state.cpp:489, proven bit-exact by NkArenaSeamForwardReverseBitExact);
    noise.cpp DR pokes → nk FieldPtr. Keep nuka_tape_* sigs + adjoint FD口径 byte-identical.
    deps T5.
  - T7 [HIGH] Op-ify diffsim backward → phi/backend_cuda/ops/diffsim_backward.cu (monolithic
    StepBackward per ruling #2; append-only NkOp; replay = CONTACT-FREE op sequence
    ApplyDrives mode0 defer=false EXPLICIT damping — NOT World::Step; verify ops/ target
    --fmad/arch/-O == nuka_diffsim; delete raw step_backward/tape/backward_runner.cu after).
    deps T6.
- **Phase 2 (re-point off the bridge):**
  - T8 [HIGH] Delete grasp_world/union_world.cpp + nuka_grasp.h/nuka_union.h + the RL python
    (bindings, __init__ exports, H1GraspEnv, train_h1_grasp_ppo, grasp_catch_eval, union
    smoke/gate scripts) in ONE commit (eager import → `import nuka` breaks otherwise). Scrub
    the gate-symbol docstrings. deps T1,T3. ★ UNIFIED-WORLD: do NOT add any replacement
    special grasp/union binding — only the generic `nuka_world`+`nuka_scene` survive; the
    grasp choreography (SelectDriveTable + phase schedule) becomes a per-scene control SCRIPT
    reading the scene's grasp block (h1_grasp_choreo.py-style), not a C-ABI world type.
  - T9 [HIGH] Re-point the 3 KEEP union gates (h1_grasp_lift/union_cook_golden/nk_union_n1)
    onto the relocated structs; drop legacy *VsLegacy arms in featherstone_oracle_harness +
    go2_4096env perf; union_cook_golden accessors re-expressed (VALUES never regenerated).
    deps T3,T5.
  - T10 [MED] Fold orphan coverage into surviving gates BEFORE deleting their files (see
    coverage-at-risk list). deps T9.
- **Phase 3 (delete):**
  - T11 [HIGH] Delete legacy estate topologically (see deletion order). KEEP convex_narrowphase
    HD header; relocate soft/fluid param structs first. deps T8,T9,T10.
- **Phase 5 (partial per ruling #4):**
  - T13 [HIGH] Migrate NON-RT buffer_legacy consumers to phi v2 BufferI; keep a minimal
    legacy shim for rt/*; do NOT delete buffer_legacy (→ M11). deps T7,T11.
- **Phase 6:**
  - T14 [MED] Test reorg: repurpose-in-place nuka_oracle/scenario/perf/import/render_test;
    ONE gtest Environment per exe (cudaSetDevice + RegisterCudaBackendEntry() — static-lib
    backend-drop trap); labels fast/full/perf; rewrite tests/CMakeLists.txt. deps T11.
  - T15 [HIGH] Final gates: scrub provenance comments (ruling #1) → git-grep zero for the 4
    symbols in LIVE+comments; ctest -L full && -L perf green (new core); lint whole-repo;
    `import nuka` succeeds. Never reconfigure build-cuda128. (render_scene/buffer_legacy-rt
    symbols excluded — deferred to M11.) deps T13,T14.
- **DROPPED from M9 (→ M11):** T12 render_scene/RT re-point; buffer_legacy full deletion.

## Deletion order (topological, leaves→load-bearing)
1. tests/coresident/ (13 .cpp + shared headers) — pure leaves
2. tests/scenario/h1_union_parity.cpp — parity witness; fold nk plan-replay D1 + step-0 byte
   into h1_grasp_lift FIRST (adversarially re-verify the M7 vel-window 1e-5→5e-5 benign-vs-
   masking BEFORE the legacy oracle dies)
3. remaining unit/TDD tests (after orphan-fold) + python/tests/test_batched_grasp_binding.py
4. collision/gpu/narrowphase_grasp.{cu,cuh,hpp}
5. solver/diff_test_bridge.{hpp,cpp}
6. c_abi/grasp_world.cpp+nuka_grasp.h, union_world.cpp+nuka_union.h (with python binding+__init__ in one commit)
7. coresident/unified_coresident_stepper.{hpp,cpp}
8. coresident/grasp_scene_factory.{hpp,cpp}
9. coresident/h1_union_nk_model.{hpp,cpp} (after KEEP gates re-pointed)
10. coresident/batched_unified_world.{hpp,cpp} (after structs relocated T3 + c_abi re-pointed)
11. coresident/ dir removal
12. solver/gpu/row_solver.* + row_scheduler.* + unified_solve.* + cuda_constraint_row_buffer.hpp (die with coresident)
13. collision/narrowphase_dispatch.hpp + legacy broadphase.cu launcher + sdf_contact.* + sdf_contact_adjoint.* (NOT convex_narrowphase.hpp — KEEP)
14. runtime/sdf/sdf_device_world.{cu,hpp} (upload duty already in Model)
15. runtime/soft/xpbd_world.cu + runtime/fluid/pbf_world.cu/* (GPU steppers ONLY, after param structs relocated)
16. runtime/gpu/batched_articulated_world.{cu,hpp} (after world.cpp T5)
17. diffsim/{step_backward,tape,backward_runner}.cu (after T7)
18. — (scene_pipeline → M9 only if not RT-coupled; render_scene → M11 per ruling #4)
19. — (render_scene.{hpp,cpp} → M11)
20. — (buffer_legacy + phi stream types → M11; M9 keeps shim)

## NOT-YET-ORPHANED (must re-point FIRST — CRITICAL)
- batched_unified_world (BatchedUnifiedWorld/BatchedSceneTemplate/H1UnionDriveEntry): LIVE in
  c_abi/union_world.cpp:48,138 + grasp_world.cpp:42,112 (PRODUCTION libnuka) + union_cook
  (produces struct) + h1_union_parity oracle → relocate/re-point first.
- batched_articulated_world: LIVE in world.cpp:583-747 + internal.hpp:78 → world.cpp create first.
- h1_union_nk_model (BuildNkUnionModel): LIVE in c_abi + 3 KEEP gates → re-point gates first.
- grasp_scene_factory: LIVE in grasp_world.cpp:108 → delete grasp_world first.
- unified_coresident_stepper: compiled into libnuka (CMake:1168) → remove after re-point.
- union_cook.{hpp,cpp}: NOT on delete list but #includes coresident @38 + produces template → decouple.
- solver trio: LIVE via coresident + diff_test_bridge → die WITH coresident.
- grasp/union C-ABI + headers: LIVE in python (eager __init__) → remove binding+__init__+C in ONE commit.
- diffsim .cu: LIVE in diffsim.cpp:16,19,20 + adjoint-FD battery → move bodies before delete.
- soft/fluid param structs: LIVE in xpbd_cooker/fluid_cooker → relocate structs first.
- tests/c_abi/test_multi_env_world + test_drive_target_io: ONLY guard for NUKA_FIELD_* binary
  semantics → KEEP + re-point (do NOT delete as "c_abi unit").

## Coverage-at-risk (fold into surviving gates BEFORE deleting source)
dof_above18_honesty (G0 51-DOF), batched_reset (RL autoreset), base_pose_view (episode-
boundary un-lagged pose), control_modes (torque/vel), solref_solimp host==device byte-exact,
SDF precision oracle (4/4 cell-tol) + gjk/epa SphereHull shallow-pen monotonicity, broadphase
D1, Philox-KAT RNG D1, h1_union_parity nk plan-replay D1 + step-0 byte, featherstone *VsLegacy
arms (lost — golden is authority), test_unified_solve compliant-rows, Krylov CG/MINRES Eigen
cross-checks, XPBD/PBF cooker determinism, scene_ir/compose invariants (nuka_scene_test links
deleted scene_pipeline → fold into scene_roundtrip), phi v1 per-handle-device, test_diffsim_tape
BackwardBitIdentical + test_aba_reverse_fd NkArenaSeam (canonical D1 transition — VERBATIM tols).

## Risk register highlights (mitigations)
- RL buffer field-enum binary contract (highest silent-corruption): pin before/after; append-only.
- One-step FK lag contract: keep base_pose_view+batched_reset assertions; nk step/FK must reproduce.
- Adjoint D1 through op-ification: monolithic StepBackward; verify ops/ flags == nuka_diffsim.
- Replay-determinism: contact-free op sub-sequence (NOT World::Step) else grad-vs-FD silently fails.
- Static-lib backend-drop: Environment::SetUp must call RegisterCudaBackendEntry().
- Generic CookToModel single-articulation debt (M7): union scenes use the relocated-struct bridge,
  not generic cook; multi-artic non-union → loud gap.
- contact_stream_driver dangling extra-source (~10 targets, not on delete list) → resolve in T14.

## Remaining minor owner-decisions (controller defaults adopted; flag if owner objects)
- IFT/KKT/OSC/sparse-CG/SDF-IFT contact-gradient family: keep raw .cu, migrate off buffer_legacy
  with rt at M11 (RL-contact-gradient not a current target) — defer with rt.
- M6-debt particle classes / solver trio: IN the M9 delete set (gate requires their consumer gone).
- Tape/checkpoint storage: params-carried raw pointers (no fields.yaml ordinal → protects dlpack).
- Adjoint-FD suite home: nuka_oracle_test (byte/tolerance-exact).
- control_mode/determinism/osc plumbing (was batched-only): re-express on nk where cheap, else
  NAMED gap (RL deferred to M10 so non-PD control surface can be rebuilt then).
- h1_union_parity: retire (fold D1 into h1_grasp_lift).
- go2_demo_render.py + render_video.sh/README go2 refs: clean in M9 (M8.5 cooked go2.nks).
