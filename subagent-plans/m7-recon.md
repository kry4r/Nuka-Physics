# M7 Reconnaissance — "scene authoring + factory death"

Repo: `/root/Nuka-Physics` @ master HEAD `caf55fd`. READ-ONLY recon, no code changed.
Plan source: `docs/plans/2026-06-11-nk-core-platform-refactor.md` §M7 (lines 482–486).

---

## EXECUTIVE SUMMARY

M7 must (1) extend the device snapshot system from articulation-only to also cover rigid-body
+ particle state, (2) build `src/scene/cook/settle.{hpp,cpp}` + `placement.{hpp,cpp}`, (3)
author the H1 grasp scene as `examples/scenes/h1_cup_table.nks` + `.nka` (the factory's ~120
empirical constants stored once as scene data), (4) migrate the choreography table to
`python/nuka/tasks/h1_grasp_choreo.py`, (5) land `tests/scenario/h1_grasp_lift.cpp` (the CORE
GATE absorbing the dev-spike force-balance/lift/BITE assertions), then (6) delete the two
factories + 4 dev-spike tests.

**The single most important finding (drives the whole decomposition):** there is **NO
SceneIR/.nks → `ContactFamily::UnionCsr` path anywhere in `src/scene/cook`**. The union Model
(feet×ground + fingers×cup-hull + cup×table slot classes) is built ONLY by the transitional
bridge `BuildNkUnionModel(BatchedSceneTemplate)` (`src/runtime/coresident/h1_union_nk_model.cpp`),
which consumes the legacy `BatchedSceneTemplate` produced by the factory. `CookToModel`
(`src/scene/cook/cook_to_model.cpp:121`) builds the *generic foot/collidable* family and never
emits `union_slots` or reads `InitialStateComponent`. **A pure-.nks `h1_grasp_lift` gate therefore
requires NEW union-slot cook logic in `src/scene/cook` (SceneIR → UnionCsr Model) before T5 can
exist** — this is the largest hidden task in M7 and is not called out as a separate bullet in the
plan. nk::Model/pipeline/schedule already fully support UnionCsr; only the *cook entry point* is
missing.

**The factory-consumer tension verdict:** the factory (`BuildH1UnionScene`) and its product
struct (`BatchedSceneTemplate`) have FOUR live consumers: the C-ABI `union_world.cpp`
(`nuka_union_world_create`, an undocumented M7 collateral — plan only deletes it at M9), the perf
gate `nk_union_n1.cpp`, the parity oracle `h1_union_parity.cpp`, and the byte-equality guard
`test_h1_union_scene_factory.cpp`. The parity oracle builds **both** worlds (legacy
`BatchedUnifiedWorld(sc.tmpl)` AND `BuildNkUnionModel(sc.tmpl)→nk::World`) from the SAME
factory-produced `tmpl`, and `BatchedUnifiedWorld` has **NO SceneIR constructor — it constructs
ONLY from `BatchedSceneTemplate`**. The M9 exit gate (plan line 501) explicitly lists
`BatchedSceneTemplate` as a token that must reach zero grep AT M9. **Conclusion: M7 can delete
the factory *authoring* (`h1_union_scene_factory.cpp` + the ~120 constants) only if the union
`BatchedSceneTemplate`/`H1UnionScene` is reproduced from the new .nks-cooked product, AND the
parity oracle + perf gate + C-ABI union_world are re-pointed to that producer. `BatchedSceneTemplate`
the *struct* (and `BuildNkUnionModel`) must survive to M9 because `BatchedUnifiedWorld` (the
M9-alive parity oracle) only speaks `BatchedSceneTemplate`.** The plan's literal "delete ALL
`BatchedSceneTemplate` references at M7" (line 485) is in direct tension with the M9 zero-grep
gate and must be read as "delete the factory *function* that authors it; keep the struct as the
oracle's cook target until M9."

---

## ITEM 1 — THE FACTORY BEING KILLED

### 1A. `h1_union_scene_factory.{hpp,cpp}` — the H1 whole-body union scene

**Entry point:** `H1UnionScene BuildH1UnionScene(const phi::DeviceContext&, h1_mjcf, cup_usda)`
(`h1_union_scene_factory.cpp:946`). Throws `std::runtime_error` on missing asset (`:949,:954`),
failed placement (`:962`), or incomplete wrap/feet (`:968`).

**The scene it builds** (authoring pipeline, all reproduced VERBATIM from the test-side shared
headers `tests/coresident/h1_demo_shared.hpp` + `h1_union_scene_shared.hpp`):

1. **Cook** (`LoadFloatingGraspCook`, `:447`): `LoadMjcf(h1_with_hand.xml)` → drop mesh geometry
   (inertia + topology only, dodges V-HACD) → `CookScene` → `CookArticulations`. The 12 wrap-driven
   finger BODIES (`kWrapDriven[]`, `:73`) get `joint_damping=1.0` / `joint_armature=0.1`
   (`kRealDamping`/`kRealArmature`). Result = floating-base 51-DOF whole-body H1 (6 base + 45 joints).
2. **Quiescent IC** (`BuildFloatingNoContactScene`, `:501`): zero velocities, identity base
   rotation, q at cook rest. 30-sphere finger-only wrap resolved by name (handles 9000+). Cup hull
   loaded (`LoadCupHull` `:199`), scaled 1.8× (`ScaleCupHull`), COM-centered, parked FAR at +5 m X.
3. **Bent stance + feet** (`BuildFeetGroundScene`, `:596`): both legs to bent stance
   (hip_pitch/knee/ankle); 4 foot toe/heel spheres per ankle (handles 12000+); base SEATED so the
   lowest foot bottom rests 2 mm into ground (z=0); ground plane (handle 8000, μ=0.8); foot-polygon
   center x computed (`poly_cx`, CoP reference).
4. **Curl + placement** (`BuildGraspStandScene`, `:816`): right hand curled to the H1.1 curled-open
   pre-pose (`ApplyCurl`/`CurlForScale(1.8)`); 12 grip links resolved by name; cup placement searched
   at the SEAT+curled FK via `BestPlacementShallowNoPalmCaging` (`:419`, 25×25 grid).
5. **200-step ORACLE SETTLE pre-roll** (`:844–858`, `kGraspSettleSteps=200`, ~0.83 s): a
   `UnifiedCoResidentStepper` driven by `BuildGraspStanceDriveSet(close_offset=0)` + the `CopCtl`
   ankle CoP balance law, cup STILL parked 5 m away (exactly the feet-only regime). Settled
   q/qdot/link_velocity/base_pose copied back into `sc.host`. **This is the deterministic settle the
   M7 `src/scene/cook/settle.{hpp,cpp}` reproduces.**
6. **Carry placement through settle** (`:861–870`): the searched cup placement is carried by the
   HAND-FRAME delta (so the cup lands in the settled hand), cup pose set to that.
7. **Table** (`BuildGraspStandTableScene`, `:880`): static table at `cup_bottom + 2 mm`
   (`kTableRestPen`); cup-table proxy box (bbox, id 7001); table broadphase id 8500, μ=0.6.

**Contact topology:** feet×ground (4 foot spheres × +Z plane, `FootSpherePlane`), 30 fingertips ×
cup hull (`FingerSphereHull`), cup-proxy box × table plane (`BodyBoxPlane`, gated on
`table_enabled`). Emission order = feet → fingers → table (the oracle drive_pairs order;
`h1_union_nk_model.cpp:131`).

**`H1UnionScene` return struct shape** (`h1_union_scene_factory.hpp:82–119`):
`tmpl` (`BatchedSceneTemplate`); `dof_stride=51`, `base_dof=6`, `root_link`, `link_count`;
`total_mass` (Σ link masses, the m·g·dt reference); `cup_mass=0.2`; `poly_cx`; `seat_pose`;
`place_found` (bool, else throw); `table_height`; `grip_links`/`grip_dofs` (12 each); three drive
tables `drive_hold`/`drive_rest`/`drive_close` (`std::vector<H1UnionDriveEntry>`); `dof_torque_limit`
(len 51); `ankle_settle_tau[2]`.

`place_found` & settle pre-roll: `BuildGraspStandTableScene` returns early with `place.found==false`
if no caging pose (`:885`) or hand link missing (`:830`); `BuildH1UnionScene` then throws loudly
(`:962`). The 200-step settle is the per-scene IC definition; runs at the VALIDATED constants
(g=−9.81, dt=1/240) regardless of the world's later step constants (hpp comment `:26–29`).

### 1B. THE ~120 MAGIC CONSTANTS (transcribe into the .nks in T3)

Literal table (name | value | meaning | where in scene/file). Grouped by role.

**Integrator / scene scalars (`h1_union_scene_factory.hpp:50–58`, `.cpp:56–64`):**

| name | value | meaning | source |
|---|---|---|---|
| `kH1UnionGravityZ` / `kGravityZ` | −9.81 | settle + world gravity z | hpp:51 / cpp:56 |
| `kH1UnionDt` / `kDt` | 1/240 (0.0041667) | settle + world dt | hpp:52 / cpp:57 |
| `kH1UnionCupMass` / `kCupMass` | 0.2 kg | cup mass (force-balance ref) | hpp:54 |
| `kMu` | 0.8 | finger×cup friction μ | cpp:61 |
| `kSxy` | 1.8 | cup hull XY scale (→ ~10.9 cm) | cpp:62 |
| `kSz` | 1.8 | cup hull Z scale | cpp:62 |
| `kCupFarOffsetX` | 5.0 m | cup park distance during settle | cpp:64 |
| `kH1UnionCloseOffset` / `kCloseOffset` | 0.18 rad | grip-close curl-beyond | hpp:57 / cpp:69 |
| `kH1UnionRestBackOffset` | −0.25 rad | rest wrap back-off | hpp:58 |

**Cook / finger-joint physics (`.cpp:67–79`):**

| name | value | meaning | source |
|---|---|---|---|
| `kGripKp` | 4.0 | grip-close PD stiffness | cpp:67 |
| `kGripKd` | 0.4 | grip-close PD damping | cpp:68 |
| `kRealArmature` | 0.1 | driven finger armature | cpp:71 |
| `kRealDamping` | 1.0 | driven finger damping | cpp:72 |
| `kWrapDriven[12]` | (12 finger body names) | the grip/BITE-kill links | cpp:73–79 |

**Leg / stance / foot (`.cpp:84–88, 177–192`):**

| name | value | meaning | source |
|---|---|---|---|
| `kLegLinkNames[10]` | (10 leg link names) | leg link resolution | cpp:84–88 |
| `LegLimit` knee | 300 N·m | knee torque limit | cpp:92 |
| `LegLimit` ankle | 40 N·m | ankle torque limit | cpp:93 |
| `LegLimit` hip | 200 N·m | hip yaw/roll/pitch limit | cpp:94 |
| `kStanceHipPitch` | −0.40 rad | bent-stance hip pitch | cpp:177 |
| `kStanceKnee` | 0.70 rad | bent-stance knee | cpp:178 |
| `kStanceAnkle` | −0.30 rad | bent-stance ankle | cpp:179 |
| `kKpHold` | 50.0 | non-leg hold PD stiffness | cpp:180 |
| `kKdHold` | 4.0 | non-leg hold PD damping | cpp:180 |
| `kFootSphereR` | 0.025 m | foot contact sphere radius | cpp:181 |
| `kFootBottomZ` | −0.055 m | foot sphere z-offset (ankle frame) | cpp:182 |
| `kFootToeX` | 0.10 m | toe sphere x-offset | cpp:183 |
| `kFootHeelX` | −0.10 m | heel sphere x-offset | cpp:184 |
| `HoldLimitFor` torso | 200 N·m | torso hold limit | cpp:187 |
| `HoldLimitFor` elbow | 18 N·m | elbow hold limit | cpp:188 |
| `HoldLimitFor` shoulder_yaw | 18 N·m | shoulder-yaw limit | cpp:189 |
| `HoldLimitFor` shoulder | 40 N·m | shoulder limit | cpp:190 |
| `HoldLimitFor` default | 1 N·m | finger default limit | cpp:191 |

**Grasp pose / wrap-sphere geometry (`.cpp:243–278, 280–308`):**

| name | value | meaning | source |
|---|---|---|---|
| `kWrapRadius` | 0.006 m | finger wrap-sphere radius | cpp:244 |
| `kPalmRadius` | 0.010 m | palm sphere radius | cpp:245 |
| `kShallowPenMax` | 0.002 m | max shallow pre-penetration | cpp:246 |
| `kLargeCagingArcMin` | 200° | min caging arc | cpp:247 |
| finger seg names | R_index/middle/ring/pinky × _proximal/_intermediate | wrap sphere bodies | cpp:258,261 |
| `finger_x[3]` | {0.006, 0.016, 0.026} | per-finger sphere x-offsets | cpp:259 |
| thumb segs | R_thumb_intermediate, R_thumb_distal | thumb wrap bodies | cpp:267 |
| `thumb_x[3]` | {0.004, 0.012, 0.020} | thumb sphere x-offsets | cpp:266 |
| palm y-row | {−0.014, 0, 0.014} | palm sphere y-offsets | cpp:272 |
| `CurlPose.finger_prox` | 1.0 | finger proximal curl | cpp:281 |
| `CurlPose.finger_int` | 1.1 | finger intermediate curl | cpp:281 |
| `CurlPose.thumb_yaw` | 1.0 | thumb yaw | cpp:282 |
| `CurlPose.thumb_pitch` | 0.5 | thumb pitch | cpp:282 |
| `CurlPose.thumb_int/dist` | 0.6 | thumb int/distal | cpp:282 |
| `CurlForScale` relax | (sxy−1)·0.55 | scale-dependent curl relax | cpp:301 |
| curl floors | 0.45 / 0.55 / 0.35 | min curl clamps | cpp:302–306 |

**Placement search (`.cpp:419–442`):**

| name | value | meaning | source |
|---|---|---|---|
| `kN` (grid) | 25 | placement search grid resolution (25×25) | cpp:426 |
| caging-arc gate | `>kLargeCagingArcMin` (200°) | min covered arc to accept | cpp:432 |
| min genuine contacts | ≥3 | min genuine contacts to accept | cpp:432 |
| pen gate | `≤kShallowPenMax` (2 mm) | reject deep placements | cpp:434 |
| objective | min `SqueezeMag` | pick least-squeeze placement | cpp:436 |

**Stance/CoP drive law (`.cpp:677–773`):** hip_yaw kp=150/kd=8/tlim=200 (`:686`); hip_roll
kp=200/kd=12/tlim=200 (`:688`); hip_pitch kp=200/kd=14/tlim=200, target=−0.40 (`:690`); knee
kp=300/kd=18/tlim=300, target=0.70 (`:692`); wrist tlim=6 N·m (real right_hand_joint ctrlrange,
`:729`); grip close kp=4/kd=0.4/tlim=0 unclamped (`:734–736`). `CopCtl`: kp=320, kd=50 (`:750`),
ankle_kd=1.5 (`:751`), tlim=40 (`:752`), settle=15 (`:753`).

**Settle + table (`.cpp:814, 878–895`):** `kGraspSettleSteps=200` (`:814`); `kTableRestPen=0.002 m`
(`:878`); cup_table_proxy_id=7001 (`:892`); table_broadphase_id=8500 (`:893`); table_mu=0.6
(`:894`); cup broadphase_body_id=7000 (`:546`); ground broadphase_id=8000 (`:638`); foot_mu=0.8
(`:639`); fingertip handles start 9000 (`:523`); foot handles start 12000 (`:613`).

**Ankle posture stand-in (factory-only metadata, `.cpp:906–908`):** `kAnklePostureKp=400`,
`kAnklePostureKd=30`, `kAnkleTorqueLimit=40`. Used to bake the settle's final CoP feedforward into
the exported ankle drive-table rows (a python caller has no FK CoM probe).

That is ≈120 distinct numeric/name constants. **All must move into `h1_cup_table.nks` (scene geometry,
materials, stance IC, settle spec) + `h1_grasp_choreo.py` (the PD drive tables + close/rest/lift
choreography).** The cup hull verts + SAMP/SDF go to the companion `.nka`.

### 1C. `grasp_scene_factory.{hpp,cpp}` — the synthetic 2-finger grasp scene

A DIFFERENT, SMALLER factory (v0.8 A2): the synthetic fixed-base 2-finger gripper pinching the C7a
cup by friction (NO table, NO feet). Entry: `BuildGraspSceneBundle` (`:110`) → `MakeGraspTemplate`
(`:175`). Constants: `kGraspCupMass=0.2`, `kGraspCupInvMass` (hpp:39–40); fingertip catch plane
z=0.20, base z=0.30 (cpp:117–118); `kPenetration=0.0015 m` (1.5 mm shallow pre-pose, cpp:103);
fingertip_radius=0.008 (cpp:121); gripper masses {1.0, 0.05, 0.05}, inertias {1e-2, 1e-4, 1e-4}
(cpp:82–84,93–95); `cup_start_z_offset` A3 knob (default 0, cpp:111); `reset_jitter_x/y` default
`kDefaultResetCupJitterM=0.025` A5a (cpp:111). **This factory is consumed by `c_abi/grasp_world.cpp`
(the python `nuka.GraspWorld` / `H1GraspEnv`) and the 21 `test_batched_unified_world` gates — NOT by
the union path.** It is in the M7 deletion list but its consumers (grasp_world C-ABI + python
h1_grasp.py + 21 unit gates) are NOT obviously re-pointed by M7; see DECOMPOSITION IMPLICATIONS.

---

## ITEM 2 — `BatchedSceneTemplate`

Defined `src/runtime/coresident/batched_unified_world.hpp:78–187`. The per-env scene template
replicated across envs at construction; the ONLY constructor input to `BatchedUnifiedWorld`
(`hpp:259`). Fields: `bodies_per_env` (cup), `has_ground`/`box_half_extent`/`ground_height` (P2.2),
`has_grasp`/`gripper_proto` (the settled articulation)/`fingertips`/`cup`/`cup_local_index`,
`grip_torque`/`drive_force_limits`, `friction_mu`/`condim`, `has_feet`/`feet`/`ground`/`foot_mu`
(G1b), `has_table`/`table_height`/`table_mu`/`cup_table_proxy_*` (G1d), `reset_jitter_x/y` (A5a).

**Every reference (`git grep`):**
- **Producers (factories):** `h1_union_scene_factory.cpp:566` (`MakeUnionTemplate`),
  `grasp_scene_factory.cpp:175` (`MakeGraspTemplate`), and the test-side mirrors
  `tests/coresident/h1_union_scene_shared.hpp:242`, `test_batched_h1_hand_grasp.cpp:496`,
  `test_batched_unified_world.cpp` (×6), `test_batched_unified_world_perf.cpp:243`.
- **Consumers:** `BatchedUnifiedWorld` ctor (`batched_unified_world.cpp:189`, hpp:260);
  `BuildNkUnionModel` (`h1_union_nk_model.cpp:24` — the M4 bridge → nk::Model);
  `c_abi/grasp_world.cpp:108`; (union side via `H1UnionScene.tmpl`).
- **Struct def + comments:** `h1_union_scene_factory.hpp:83`, `grasp_scene_factory.hpp`,
  `nk/model/model.hpp:160` (comment only), `tests/CMakeLists.txt:2583` (comment).

**Can it die in M7?** **The struct CANNOT die in M7.** `BatchedUnifiedWorld` (alive to M9, plan
line 499 deletes `src/runtime/coresident/` whole at M9) constructs ONLY from it, and the M9 exit
gate (line 501) lists `BatchedSceneTemplate` as a token to zero-out AT M9. The *factory functions*
that author it (`MakeUnionTemplate`/`BuildH1UnionScene`, `MakeGraspTemplate`/`BuildGraspSceneBundle`)
can die in M7 IF every consumer is re-pointed to build the template from the new .nks-cooked product
(or directly from the cooked `nk::Model`). The plan's line-485 "delete ALL BatchedSceneTemplate
references at M7" is unachievable as literally written and must mean "delete the factory that
authors it." See ITEM 3 verdict.

---

## ITEM 3 — THE FACTORY-CONSUMER TENSION (who-builds-what)

### 3A. Every consumer of `BuildH1UnionScene` / `kH1UnionMjcfDefault`

1. **`src/c_abi/union_world.cpp:124`** (`nuka_union_world_create`): `BuildH1UnionScene` → stores
   `H1UnionScene` whole → constructs `BatchedUnifiedWorld(record->scene.tmpl, …)` (`:125`). LIVE
   C-ABI (python `nuka.UnionWorld`, `nuka_ext.cpp:789`). **Builds the legacy path ONLY.** Exposes the
   drive tables / grip dofs / dof limits / scene info via the C-ABI (`:194–233`, `:176–192`). NOT in
   the M7 deletion list — plan deletes `union_world.cpp` at M9 (line 499).
2. **`tests/perf/nk_union_n1.cpp:59,215`** (`NkUnionN1` + `NkGraspThroughput`): `BuildH1UnionScene`
   → `BuildNkUnionModel(sc.tmpl)` → `nk::World` (the GATED path, `:63,:220`); also constructs a
   legacy `BatchedUnifiedWorld(sc.tmpl)` for the printed before/after baseline (`:177`, NOT gated).
   **Builds BOTH; gate is the nk path.**
3. **`tests/scenario/h1_union_parity.cpp:222`** (`H1UnionParity`): the parity oracle. Builds **THREE
   worlds from the SAME `sc.tmpl`**: legacy `BatchedUnifiedWorld(sc.tmpl)` (`:233`), a self-chaos
   twin `BatchedUnifiedWorld(sc.tmpl)` +1e-6 nudge (`:237`), and `BuildNkUnionModel(sc.tmpl)→nk::World`
   (`:241`); plus a 4th replay nk twin (`:454`). Compares nk vs legacy on identical IC + closed-loop
   choreography. **Builds BOTH; the legacy world IS the oracle.**
4. **`tests/coresident/test_h1_union_scene_factory.cpp:92,184`**: the byte-equality guard — asserts
   `BuildH1UnionScene().tmpl` == the shared-header `MakeUnionTemplate(sc)` field-for-field (`:89,:93`).
   Pure factory-fidelity test; dies WITH the factory.

### 3B. Is there ANY SceneIR → `BatchedUnifiedWorld` path?

**NO.** `BatchedUnifiedWorld` has exactly one constructor (`hpp:259`), input = `BatchedSceneTemplate`.
No SceneIR/Scene/Registry/Model overload exists. Confirmed by grep (single ctor sig).

### 3C. Is there ANY SceneIR → `UnionCsr` nk::Model path?

**NO** in `src/scene/cook`. `git grep union_slots|UnionCsr|FingerSphereHull -- src/scene/` → **zero
hits**. `ContactFamily::UnionCsr` + `union_slots.push_back` are produced ONLY in
`h1_union_nk_model.cpp` (the coresident bridge from `BatchedSceneTemplate`). `CookToModel`
(`cook_to_model.cpp:121`) builds the generic family (foot pipeline `cap.max_contacts_per_env=4` when
articulation present, else collidable×4; `:366–372`) and never touches union slots. nk::Model
*supports* union (model.cpp:230 serializes UnionSlots; pipeline.cpp:21 `is_union`; schedule.cpp:269
`UnionSlotRowBases`), so the missing piece is purely the **cook authoring of union slots from a
SceneGraph/Registry**.

### 3D. The transitional bridge `h1_union_nk_model.{hpp,cpp}`

`BuildNkUnionModel(const BatchedSceneTemplate&, env_count)` (`.cpp:24`). Maps 1:1:
- articulation template ← `gripper_proto` (settled q/qdot/link_velocity/base_pose) (`:40–79`);
- flat-DOF maps `dof_to_link`/`dof_to_component` (DofIndexOf⁻¹) (`:82–104`);
- torque drive ← `grip_torque`/`drive_force_limits` (`:107–117`);
- bodies ← `bodies_per_env` cup (`:120–129`);
- union slots: feet (`FootSpherePlane`) → fingers (`FingerSphereHull`) → table (`BodyBoxPlane`, gated)
  in legacy order (`:131–183`);
- capacities (`:186–197`). Header (`.hpp:13–15`) states: "**dies with the factory at M7** (the .nks
  scene path replaces the template authoring) / the directory at M9."

### 3E. WHO-BUILDS-WHAT dependency graph

```
                         h1_with_hand.xml + cup/model.usda
                                      │
                         BuildH1UnionScene  (h1_union_scene_factory.cpp:946)
                         [cook→stance→seat→curl→placement→200-step settle→carry→table]
                                      │  produces
                                      ▼
                              H1UnionScene{ .tmpl:BatchedSceneTemplate, drive_hold/rest/close,
                                            dof_torque_limit, grip_dofs, total_mass, ... }
                       ┌──────────────┼───────────────────────────┬─────────────────────────┐
                       │ .tmpl        │ .tmpl                      │ .tmpl                    │ whole scene
                       ▼              ▼                            ▼                          ▼
        BatchedUnifiedWorld   BuildNkUnionModel(tmpl)     test_h1_union_scene_factory   c_abi/union_world.cpp
        (LEGACY, alive→M9)    →nk::Model(UnionCsr)        (byte-equality guard,         (nuka_union_world_create,
                       │              │ →nk::World         dies with factory)            LIVE python UnionWorld)
                       │              │                                                         │
        ┌──────────────┴──────┐       │                                                  BatchedUnifiedWorld(tmpl)
        ▼                     ▼       ▼
  h1_union_parity      nk_union_n1  (gated nk path: parity + perf + throughput)
  (oracle: legacy      (gate: nk;
   vs nk, alive→M9)     legacy baseline printed)
```

### 3F. VERDICT

**The factory CANNOT simply die in M7 without re-pointing the M9-alive parity oracle.** The minimal
M7-viable decomposition:

1. **NEW**: `src/scene/cook` gains a SceneIR/Registry → `ContactFamily::UnionCsr` nk::Model path
   (union-slot cook), so `h1_cup_table.nks` cooks directly to the union nk::World. This is the
   precondition for `h1_grasp_lift` to be a "pure .nks path."
2. **KEEP to M9**: the `BatchedSceneTemplate` struct + `BuildNkUnionModel` + `BatchedUnifiedWorld`
   (the parity oracle's legacy world).
3. **RE-POINT (M7)**: `h1_union_parity.cpp` + `nk_union_n1.cpp` to obtain their `BatchedSceneTemplate`
   from a NEW non-factory producer — either (a) a thin "Scene→BatchedSceneTemplate" adapter cooked
   from the .nks (so both legacy + nk worlds still share one IC), or (b) freeze the parity oracle's
   template to a committed fixture. Option (a) preserves the oracle's same-IC contract; (b) is
   simpler but loses the live cook coupling.
4. **RE-POINT or defer (M7 collateral)**: `c_abi/union_world.cpp` (`nuka_union_world_create`) also
   calls `BuildH1UnionScene`. The plan only deletes it at M9, so M7 must either re-point it to the
   .nks producer or temporarily keep `BuildH1UnionScene` alive for the C-ABI alone. **This is an
   undocumented M7 ordering hazard.**
5. **DELETE (M7, gate-green)**: `h1_union_scene_factory.cpp` (the ~120 constants, once nothing calls
   `BuildH1UnionScene`), `test_h1_union_scene_factory.cpp`, and the 4 dev-spike tests.

---

## ITEM 4 — THE SNAPSHOT SYSTEM (the D1-deferred debt M7 absorbs)

### 4A. What snapshots TODAY (articulation-only)

Snapshot fields in `src/nk/model/fields.yaml:108–113` (the ONLY `snapshot_*` fields):

| field | dtype | per | line |
|---|---|---|---|
| `snapshot_q` | f32 | link | 110 |
| `snapshot_qdot` | f32 | link | 111 |
| `snapshot_link_velocity` | spatial6 | link | 112 |
| `snapshot_base_pose` | transform | env | 113 |

`OpSnapshotState` (`readout.cu:258`) copies live→snapshot for exactly these four (D2D
`cudaMemcpyAsync` in fixed order, `:271–280`). `OpRestoreState` (`readout.cu:286`) copies
snapshot→live + memsets `qddot`/`tau`/`lambda` to 0 (`:299–318`). `OpResetEnvs` (`readout.cu:233`,
`ResetEnvsKernel:107`) restores the same four per-env + clears qddot/tau/lambda. Params:
`SnapshotStateParams{total_link_count, env_count}` (`op_schema.hpp:399`);
`RestoreStateParams{+row_slot_count}` (`:406`); `ResetEnvsParams{count, base_link_count,
lambda_stride, articulation_count}` (`:392`).

### 4B. How `World::SeedInitialState` / `World::Reset` use it (`world.cpp`)

`SeedInitialState()` (`:59`): fills snapshot/reset params (`:65–73`); seeds body template
(BodyPose/BodyLinearVelocity/BodyAngularVelocity/BodyInvMass/BodyInvInertia, env-major, `:75–108`);
seeds TableEnabled (`:110–118`); seeds particles (ParticlePos/PrevPos/Vel/InvMass, `:120–152`);
seeds MatBuckets/MatIndex (`:154–192`); then articulation Q/DriveTarget/.../Qdot/LinkVelocity/
LinkPose/BasePose (`:203–267`); finally `DispatchOp(SnapshotState)` (`:270`). **CRITICAL GAP,
explicitly commented at `:194–200`:** when `L==0` (no articulation, bodies/particles-only world),
"SnapshotState is a NO-OP today (it early-returns at total_link_count==0; **body/particle snapshot
fields do not exist yet — the M7 settle consumer extends the snapshot system**). Reset on a
bodies/particles-only world therefore restores nothing; honest Ok." And even WITH an articulation,
the SnapshotState/RestoreState only round-trip the 4 articulation fields — **the rigid-body
state (body_pose, body_linear_velocity, body_angular_velocity) and particle state (particle_pos,
particle_vel, particle_prev_pos) are seeded but NEVER snapshotted/restored.**

`Reset(env_ids)` (`:317`): empty list → `RestoreState` (bulk, `:323`); non-empty → upload ids →
`ResetEnvs` (`:340`). Both restore ONLY articulation state.

### 4C. Body-state + particle fields that T1 must add to the snapshot

Live fields that would need snapshotting for a deterministic settle→snapshot→restore of the FULL
scene (the union scene has a movable cup; soft/fluid scenes have particles):

| field | dtype | per | fields.yaml line | currently snapshotted? |
|---|---|---|---|---|
| `body_pose` | transform | body | 66 | **NO** |
| `body_linear_velocity` | vec3 | body | 149 | **NO** |
| `body_angular_velocity` | vec3 | body | 150 | **NO** |
| `particle_pos` | vec3 | particle | 76 | **NO** |
| `particle_vel` | vec3 | particle | 329 | **NO** |
| `particle_prev_pos` | vec3 | particle | 328 | **NO** |

(For the M7 union grasp scene the load-bearing additions are the three `body_*` rigid fields — the
cup. Particles are the broader soft/fluid generality.) T1 = add `snapshot_body_*` (+ optionally
`snapshot_particle_*`) fields to `fields.yaml`, extend `OpSnapshotState`/`OpRestoreState`/
`OpResetEnvs` to copy them, widen the Params, and make `SeedInitialState` snapshot post-settle
state. **(Mapping only — not designing the fix here.)**

---

## ITEM 5 — THE 4 DEV-SPIKE GATE TESTS (assertions T5 must absorb)

The 4 tests are mostly SKIP-with-finding diagnostic spikes. The **hard (EXPECT_TRUE/EXPECT_LE)**
assertions that constitute the "4 key assertions" `h1_grasp_lift` must absorb live in
`test_h1_dense_grasp.cpp` and `test_h1_grasp_feasibility_probe.cpp` (the GO config). Shared dev-spike
constants: g=−9.81, dt=1/240, `kCupMass=0.2`, `kMu=0.8`, `kKp=4.0`, `kKd=0.4`, `kCloseOffset=0.18`,
cup scale 1.8× (~10.9 cm), `kShallowPenMax=0.002`. `weight_kick = kCupMass·(−g)·dt = 0.2·9.81·(1/240)
≈ 8.175e-3` (the m·g·dt force-balance reference).

### 5A. `test_h1_dense_grasp.cpp` — `ForceClosureLiftWithDisturbance` (`:1442`, HARD GO)

The validated active force-closure proof. Settle active → 1.0 g sustained lateral push along the
escape-gap bisector + a one-shot 2.5 rad/s tilt, grip held fixed, then release+recover. Constants:
`kFcLatAccelG=1.0` (`:901`), `kFcLiftSettle=50` (`:902`), `kFcPushSteps=30` (`:903`),
`kFcReleaseSettle=70` (`:904`), `kFcTiltKickW=2.5` (`:905`), `kFcMaxDisp=0.07` (`:908`),
`kFcMaxTilt=0.35` (`:909`), `kFcMaxFinalW=1.20` (`:910`). Sub-gates (`:1475–1482`):
```cpp
translation_ok = peak_disp < 0.07;
tilt_ok        = peak_tilt < 0.35 && final_w < 1.20;
contact_ok     = contact_post >= steps_post - 12;
recover_ok     = |recovered_fimp - weight_kick| < 0.35 * weight_kick;   // ← force-balance ±35%
static_ok      = !any_static_post;
cross_ok       = recovery_cross < 5.0e-2;                                // impulse bookkeeping
held = translation_ok && tilt_ok && contact_ok && recover_ok && static_ok && cross_ok;
EXPECT_TRUE(held);                                                       // :1494
```
D1 twin `ForceClosureLiftWithDisturbanceDeterministicTwoRun` (`:1499`): `EXPECT_EQ(memcmp(cup_final))`
+ `EXPECT_EQ(memcmp(qdot))` byte-identical across 2 runs (`:1510,:1513`). **This holds the
force-balance (recover_ok = Σλ≈m·g·dt ±35%) + lift-impulse-triangle (the table→friction support
flip) + D1 assertions.**

### 5B. `test_h1_dense_grasp.cpp` — `FingerOnlyFallbackBiteGripOffVsOn` (`:1519`, HARD BITE)

```cpp
weight_kick = kCupMass*(-kGravityZ)*kDt;   // :1524
free_fall   = 0.5*(-kGravityZ)*(120*kDt)^2; // :1526  (kFall=120)
// grip=ON hold (40 steps after table off):
holds_on = |z_active - cup0.z| < 0.05 && |vz_on| < 0.5 && rep_on.cup_vertical_impulse > 0.5*weight_kick; // :1544
// grip=0 for 120 steps:
falls    = drop > 0.02 || cupF.linear_velocity.z < -0.10;  // :1552
EXPECT_TRUE(holds_on);                       // :1567
EXPECT_LE(gs.place.max_pen, kShallowPenMax + 1e-5);  // :1568  (≤2.01 mm shallow)
```
**This is the BITE assertion: grip-on holds (vertical impulse > 0.5·m·g·dt), grip-off → cup falls.**

### 5C. `test_h1_grasp_feasibility_probe.cpp` — `ConjointHonestGateVerdict` (`:771`, HARDEST CONSOLIDATION)

The cleanest single-test consolidation of all four assertions — a size sweep {8.5, 9.7, 10.9} cm
that asserts the proven 10.9 cm cell (found BY TAG, `:824–825`) threads the conjoint gate
(`:826–832`):
```cpp
EXPECT_TRUE(proven->placement_found);  // shallow ≤2 mm caging placement exists
EXPECT_TRUE(proven->hold_gravity);     // cup held after table removal (support ≈ m·g·dt)
EXPECT_TRUE(proven->hold_rotation);    // survives 1g lateral + 2.5 rad/s tilt (disp/tilt/|w| bounded)
EXPECT_TRUE(proven->grip_on_holds);    // grip-on holds
EXPECT_TRUE(proven->bite_drops);       // grip=0 → cup DROPS (the honesty discriminator)
```
Per-cell GO (`RunCell`, `:591–699`): `bite_drops = (bite_drop > 0.02) || (vz < -0.10)` (`:695`);
go = hold_gravity && hold_rotation && grip_on_holds && bite_drops (`:699`). Disturbance constants:
`kLatAccelG=1.0` (`:562`), `kPushSteps=30` (`:563`), `kTiltKickW=2.5` (`:564`), `kMaxDisp=0.07`
(`:566`), `kMaxTilt=0.35` (`:567`), `kMaxFinalW=0.50` (`:568`). **The 4 key assertions =
{hold_gravity (force-balance), hold_rotation (lift/disturbance), grip_on_holds, bite_drops}.**

### 5D. `test_h1_power_grasp_lift.cpp` — `LiftGateWrapCarriesWeightWithoutPivot` (`:809`)

Mostly the HONEST-NEGATIVE wrap spike (SKIP on rotational cage, `:931`). Hard asserts present
(`:901–917`) but on the *passive* 6 cm wrap: `EXPECT_FALSE(any_static_after)`, `EXPECT_GT(mean_fimp,
0.5*weight_kick)` (force-balance ½ floor), `EXPECT_LT(max_cross, 5e-2)`, `EXPECT_LT(max_disp, 0.05)`,
`EXPECT_GE(contact_after, kLift-8)`, `EXPECT_LT(steady_qd, 5.0)`. `balance_err = |cup_vertical_impulse
- weight_kick|/weight_kick`, max ~0.43 (`:868–870, :904`). This file's value is mainly the
`balance_err`/m·g·dt formulation reference; its GO claim is weaker than 5A/5C.

**Which hold the "4 key assertions"?** `ConjointHonestGateVerdict` (5C) is the single best fit (all
4 in one test on the proven 10.9 cm config). `ForceClosureLiftWithDisturbance` (5A) holds the
force-balance + lift-triangle + D1; `FingerOnlyFallbackBiteGripOffVsOn` (5B) holds the BITE +
shallow-pen. `h1_grasp_lift` should reproduce these on a PURE .nks union scene with the SAME
weight_kick = m·g·dt reference and the ±35% / 0.5× tolerances.

---

## ITEM 6 — CURRENT .nks AUTHORING PATH + SETTLE HOOK

### 6A. `src/scene/format/nks.{hpp,cpp}` — settle/initial_state NOT yet parsed

`grep settle|initial_state|holds nks.cpp` → **zero hits.** The format parses ONLY:
`nks_version` (`:572`), `physics_materials`/`render_materials` (`:582–583, 677–678`), `tree`
(pre-order nested, `:594, :712`), `imports` (resolved BEFORE records, `:649, :667`), and an override
`overrides` layer. **The plan §3.7 example sections `"initial_state"` (line 325) and `"settle"`
(line 326) are NOT implemented.** API: `Save(const SceneIR&, path)` (hpp:38), `Load(path)→SceneIR`
(hpp:45), `Load(base, overlay)` (hpp:51). **T3 must add `initial_state` + `settle` Save/Load to
nks.cpp.** (Note: nks currently operates on `SceneIR`, not the §3.6 SceneGraph+Registry pair — the
scene_ir facade.)

### 6B. `src/scene/cook/cook_to_model.cpp` — does NOT consume `InitialStateComponent`

`grep InitialState|qpos cook_to_model.cpp` → **zero hits** (one unrelated "Position actuators seed
stiffness" at `:222`). `InitialStateComponent{qpos, root}` is defined in
`src/scene/ecs/components.hpp:139–142` (`std::vector<float> qpos; math::Transform root = Identity()`)
but **nothing reads it during cook**. The cook seeds q from the cooked articulation rest pose
(`SeedInitialState` in world.cpp uses `model.articulation.initial_q`), not from a settle product.
**T4 must wire InitialStateComponent → Model initial_q / base_pose / body_pose so the settle output
becomes the cooked IC.** (And see ITEM 3C: cook must additionally learn UnionCsr slots.)

### 6C. Existing `.nks` template?

`find -name "*.nks"` → **none.** `examples/scenes/` holds only `.usda`/`.xml`/`.urdf` (go2 +
complete_robot). The nearest authoring reference is `tests/scenario/scene_roundtrip.cpp` (M2 gate:
LoadMjcf+LoadUsd→Compose→Save(.nks/.nka)→Load→tree-equivalence). **T3 creates the FIRST
`examples/scenes/h1_cup_table.nks` from scratch.**

---

## ITEM 7 — `test_scene_compose_h1_cup_table.cpp` (untracked, `tests/scene/`)

Currently a **pure SceneIR-compose gate** (NO .nks pipeline, NO nk::World). It composes
`kitchen.xml` + `h1_with_hand.xml` + `cup/model_large.usda` (NOTE: `model_large`, a different/larger
cup than the union factory's `model.usda`) into one `SceneIR` via `Compose` (`:283–284`), finds the
real imported kitchen-counter support surface (`FindKitchenCounterSupport`, `:216`, prefers
`counter_1_main_group_top_1`), and places the cup by a **drop-to-rest computation**
(`CupRootRelativeMinZ` `:128` → cup bottom at `support.top_z + 1 mm clearance`, `:273–276`) + the H1
hand within reach (`:279–280`). 6 TESTs: count-additivity (`:291`), uses-imported-counter (`:315`),
cooks-with-mirrored-counts (`:331`), cup-rests-on-counter (`:350`, `EXPECT_NEAR(cup_bottom.z,
top_z+clearance, 1e-4)`), cup-within-reach (`:374`, `EXPECT_LT(dist, 0.5)`), deterministic-two-run
(`:401`).

**"Upgrade to full .nks pipeline" (plan gate) means:** route this scene through
`Save(.nks/.nka)→Load→CookToModel→nk::World` (instead of staying at `SceneIR`+`CookScene`), proving
the composed kitchen+H1+cup scene authors, persists, reloads, cooks to an nk::Model, and steps —
i.e. the assertions become "the .nks round-trips and produces the same rest placement / cook counts,"
and (ideally) the cup rests on the counter after a settle. This is the second M7 gate (alongside
`h1_grasp_lift`). Its drop-to-rest logic is a reuse candidate for `FindRestPlacement` (ITEM 8).

---

## ITEM 8 — FindRestPlacement / PLACEMENT SEARCH (what exists today)

No `FindRestPlacement`, `placement.hpp`, or `settle.hpp` exists (`git grep` → only matches files
containing the *substring* "BestPlacement"). Two distinct placement algorithms exist as reuse
sources:

1. **`BestPlacementShallowNoPalmCaging`** (`h1_union_scene_factory.cpp:419`, mirrored in
   `tests/coresident/h1_demo_shared.hpp` + `h1_union_scene_shared.hpp` + the 4 dev-spike tests):
   a 25×25 XY grid search at the curl-cavity z, scoring each candidate by `MeasureSurround`
   (covered-arc ≥200°, ≥3 genuine contacts), `MaxPrePenetration` (≤2 mm), minimizing `SqueezeMag`.
   This is the GRASP placement (cup in the curled hand), NOT a rest-on-surface drop.
2. **Kitchen-counter drop-to-rest** (`test_scene_compose_h1_cup_table.cpp:128–156, 260–287`):
   `CupRootRelativeMinZ` (cup's lowest mesh vertex in root frame) → place cup bottom at the support's
   `top_z + clearance`. This IS the "drop onto a surface until rest" algorithm `FindRestPlacement`
   should generalize (find a support box top, seat the object's min-z 2 mm above/into it).

**`FindRestPlacement` (T2)** most likely generalizes #2 (geometric drop-to-rest on a support
surface), while the *settle* (`settle.cpp`, T2 sibling) generalizes the factory's 200-step
quasi-static stepping (ITEM 1A step 5) + the dynamic drop (run nk::World stepping until rest). Note
the factory does NOT do a "drop+settle until rest" for the cup — it places by FK + carries through a
robot-stance settle; the cup is positioned analytically, not dropped. So `FindRestPlacement` is a
NEW capability (closest existing kin = the kitchen-counter geometric seat).

---

## DECOMPOSITION IMPLICATIONS

The 6 M7 tasks (T1 snapshot-extend / T2 settle+placement / T3 .nks authoring / T4 python choreo /
T5 gate / T6 factory-deletion) have these coupling/ordering constraints:

### Ordering (hard dependencies)

1. **T1 (snapshot-extend) precedes T2 (settle).** `settle.cpp` runs nk::World stepping →
   `SnapshotState` → `InitialStateComponent`. Today SnapshotState only saves articulation state
   (ITEM 4) and the cup is a rigid body; a settle that does not snapshot `body_pose`/`body_*velocity`
   would lose the settled cup pose on Reset → the "two runs byte-identical" gate (and the cup IC)
   would be wrong. **T1 must add `snapshot_body_*` (the cup) before settle is meaningful.**
2. **A HIDDEN T0 (union-slot cook) precedes T3+T5.** There is no SceneIR→UnionCsr path (ITEM 3C).
   `h1_cup_table.nks` cannot cook to the union nk::World, and `h1_grasp_lift` cannot be a "pure .nks
   path," until `src/scene/cook` learns to emit `union_slots` (FootSpherePlane/FingerSphereHull/
   BodyBoxPlane) + `ContactFamily::UnionCsr` + the cup body + the 30 fingertip + 4 foot + table slot
   geometry from a Registry. **This is the single largest unscoped M7 task. Either (a) build the
   union-slot cook, or (b) cook `h1_cup_table.nks` to the same `BatchedSceneTemplate` via a thin
   adapter then reuse `BuildNkUnionModel`. Option (b) keeps `BatchedSceneTemplate` alive (fine — it
   lives to M9 anyway) and is far less risky.**
3. **T4 (settle/initial_state nks parse) precedes T3-as-gate.** `nks.cpp` cannot Save/Load the
   `settle{steps,dt,holds[]}` + `initial_state{joint_pos}` sections (ITEM 6A) — they are unimplemented.
   T3 authoring + the `test_scene_compose_h1_cup_table` upgrade both need this.
4. **T4-cook (InitialStateComponent consumption) precedes T5.** Cook ignores
   `InitialStateComponent` (ITEM 6B); the settled q/root must flow into `Model.articulation.initial_q`
   + base/body poses or the cooked IC won't match the factory's settled IC the gate asserts on.
5. **T5 (gate green) precedes T6 (deletion).** The plan's literal precondition.

### Consumers that MUST be re-pointed when the factory dies (T6)

| consumer | builds | re-point target | plan-listed for M7? |
|---|---|---|---|
| `tests/scenario/h1_union_parity.cpp` | legacy `BatchedUnifiedWorld(tmpl)` + nk `BuildNkUnionModel(tmpl)` | a non-factory `BatchedSceneTemplate` producer (cooked from .nks, OR a committed fixture) — must feed BOTH worlds the SAME IC | NO (it's the M4 oracle; M7 must re-point it) |
| `tests/perf/nk_union_n1.cpp` | nk (gate) + legacy (baseline print) | same producer as parity | NO |
| `src/c_abi/union_world.cpp` (`nuka_union_world_create`) | legacy `BatchedUnifiedWorld(tmpl)` for python `nuka.UnionWorld` | the .nks producer, OR keep `BuildH1UnionScene` alive until M9 | **NO — plan deletes union_world only at M9; this is an ordering hazard** |
| `tests/coresident/test_h1_union_scene_factory.cpp` | byte-equality vs shared header | DELETE (dies with the factory) | YES |
| `src/CMakeLists.txt:905,973` + `tests/CMakeLists.txt:673,706,2588` | compile `h1_union_scene_factory.cpp` | drop the source from these targets | implied by deletion |

### Lifetime constraints (do NOT delete in M7)

- `BatchedSceneTemplate` struct, `BuildNkUnionModel`, `BatchedUnifiedWorld` — all alive to M9 (the
  M9 exit gate, plan line 501, is the zero-grep deadline). M7 deletes the factory *function*
  `BuildH1UnionScene`, not the template type or the bridge.
- `grasp_scene_factory.{hpp,cpp}` is in M7's deletion list (line 485) but its consumers
  (`c_abi/grasp_world.cpp` → python `nuka.GraspWorld`/`h1_grasp.py`, and the 21
  `test_batched_unified_world` gates) are NOT re-pointed by M7. **Deleting it in M7 breaks the python
  grasp env + 21 unit gates unless those are migrated or the deletion is deferred.** Flag for
  owner/controller: either narrow M7's deletion to the *union* factory only, or also migrate the
  synthetic-grasp consumers. (The plan's M9 deletes `c_abi/grasp_world.cpp` + the unit-test sweep, so
  the cleaner reading is: M7 deletes the *union* factory + 4 union dev-spikes; the synthetic-grasp
  factory's deletion is effectively M9 collateral. Confirm intent.)

### Determinism gate coupling

- "settle two runs byte-identical" (T2/T5 gate) requires: (i) T1's snapshot covers every settled
  field (articulation + body, ITEM 4C); (ii) the settle stepping uses the deterministic nk::World
  StepPlanned/Step path (no host atomics); (iii) `nks.cpp` Save renders floats `%.9g`
  (binary32-lossless, hpp:23) so the persisted IC round-trips bit-exactly.
- `h1_grasp_lift` must reproduce the dev-spike numbers (ITEM 5) with the SAME `weight_kick =
  m·g·dt ≈ 8.175e-3` and tolerances (force-balance ±35% `recover_ok`, BITE drop>0.02 / vz<−0.10,
  shallow-pen ≤2 mm), on the cooked-from-.nks union nk::World — which means the .nks-cooked IC must
  match the factory's settled IC to within the parity window (else the force balances drift).

### Key file:line index (for implementers)

- Factory: `src/runtime/coresident/h1_union_scene_factory.{hpp,cpp}` (entry `:946`; settle `:844`;
  placement `:419`; drive tables `:910`; constants table ITEM 1B).
- Synthetic grasp factory: `src/runtime/coresident/grasp_scene_factory.{hpp,cpp}`.
- Bridge: `src/runtime/coresident/h1_union_nk_model.{hpp,cpp}` (`:24`).
- Template struct: `src/runtime/coresident/batched_unified_world.hpp:78–187`; world ctor `:259`.
- Consumers: `src/c_abi/union_world.cpp:124`; `tests/perf/nk_union_n1.cpp:59,215`;
  `tests/scenario/h1_union_parity.cpp:222,233,237,241,454`;
  `tests/coresident/test_h1_union_scene_factory.cpp:92,184`.
- Snapshot: `src/phi/backend_cuda/ops/readout.cu:258(Snapshot),286(Restore),233(Reset)`;
  `src/nk/model/fields.yaml:108–113(snapshot),66/149/150(body),76/328/329(particle)`;
  `src/nk/pipeline/world.cpp:59(Seed),194–200(L==0 gap note),270(snapshot),317(Reset)`;
  `src/phi/op_schema.hpp:392–410(params)`.
- nks format: `src/scene/format/nks.{hpp,cpp}` (no settle/initial_state).
- Cook: `src/scene/cook/cook_to_model.cpp:121(CookToModel)` (no union, no InitialState);
  `src/scene/ecs/components.hpp:139(InitialStateComponent)`.
- Compose gate: `tests/scene/test_scene_compose_h1_cup_table.cpp` (drop-to-rest `:128,:260`).
- Dev-spike gates: `tests/coresident/test_h1_dense_grasp.cpp:1442,1519`;
  `test_h1_grasp_feasibility_probe.cpp:771`; `test_h1_power_grasp_lift.cpp:809`;
  `test_h1_scaled_cup_grasp.cpp:847,983`.
- Python: `python/nuka/tasks/h1_grasp.py` (synthetic grasp env, NOT the union choreo);
  `h1_grasp_choreo.py` does NOT exist (M7 creates it); union C-ABI bound at
  `python/src/nuka_ext.cpp:789` (`UnionWorld`).
- CMake: `src/CMakeLists.txt:898–908(c_abi+factories),963–973`;
  `tests/CMakeLists.txt:264(compose),667–732(parity+perf),2318/2359/2404/2448(4 dev-spikes),
  2586–2619(factory byte-eq guard)`.
