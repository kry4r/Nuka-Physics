# SPEC — ONE general physics-contact solver (no families, no special-cases)

Status: **DRAFT for owner review — DO NOT IMPLEMENT until approved.**
Author: controller, 2026-06-16. Branch `collision-foundation`, HEAD `8b8cea0`.

---

## 0. 执行摘要 (中文)

Owner 最高指令:**要通用的物理求解,不要 case 特惠 hack。** 当前实现仍有三类 hack:
1. **单狗/多狗走不同路径** —— `ContactFamily` 枚举(`FusedFoot`/`UnionCsr`/`PairDriven`)就是"按场景分路径"的机制本身。
2. **`world.cpp` 里的 `instance_count` 特化** —— 多实例才进某分支。
3. **`"dog" + i` 命名** —— 把"狗"这种实体类型写进通用 compose 代码。

本 spec 的终态:**删掉 family 概念,删掉所有专用检测/装配/求解分支,只剩 ONE 通用接触管线** ——
`LBVH 宽相 → cvx GJK/EPA 窄相(+ heightfield 三角棱柱)→ 注册表解析装配 → 混合-island 耦合求解`。
机器人脚踩地、狗撞狗、手指抓杯、自由刚体、地形,全部是"一堆 collidable body 经同一条路"。
没有"单狗 vs 多狗",没有"dog/grasp/union"字样,没有按数量/类型的分支。

代价:go2 单狗 golden 和 H1 抓取 golden 的字节都会变(它们现在用 FusedFoot/UnionCsr 专用路径)。
所以必须先写**等价性证明 harness(V-PROOF)**:通用路径在容差内复现旧轨迹 → owner 签字 → 一次性重设两个 golden。
**这是需要 owner 拍板的核心决定(见 §8)。**

已完成的 Phase 0–2A(`cvx` 窄相、混合-island 求解器、通用 heightfield)**不是 hack —— 它们正是 owner 要的通用求解器本体**,
现在只是被 family flag 暂时 gated 成"validated-not-wired"。本 spec = 把一切收敛到这条路 + 删掉 gating/families/特化。

---

## 1. The principle

From [[unified-world-no-special-grasp-binding]] (owner, highest directive): **通用物理求解引擎,禁止 case 特惠.**
A robot body is just a physics body. A foot, a finger, a free box, a heightfield cell — all are collidables.
Collision is **general body↔body + general body↔heightfield**, solved by **ONE coupled island solver**, for ANY count
of ANY kind. There is no "single vs multi" path, no entity-type naming, no per-scenario branch.

What violates it TODAY (the hacks to delete):
- `enum class ContactFamily { FusedFoot, UnionCsr, PairDriven }` (model.hpp:208) — *the path-distinction mechanism itself.*
- The FUSED foot detect + FUSED single-articulation solver (go2's private path).
- `EmitUnionRowsKernel` / pre-cooked union slots (H1 grasp's private assembly).
- `dog_dog_contact.cu` (multi-dog's private O(L²) sweep + analytic body-terrain that explodes on steps).
- `c_abi/world.cpp` branching on `instance_count` and naming replicas `"dog" + i`.

---

## 2. End-state architecture (the ONE path)

ONE pipeline graph, ONE family (= none). Every world cooks to: collidable `shape_table` rows (each carrying
`body_id`, `group`) + the body→link/body→articulation registry + static collidables (ground plane, heightfield).
Every step runs:

```
FkWorldPoses → SyncLinkBodyPose (artic link FK → body_pose)
  → BuildAabbs → LBVH build → LBVH query           (broadphase, ALL bodies)
  → Narrowphase (cvx GJK/EPA analytic+hull) + NarrowphaseHeightfield (prism cells)
  → AssembleRows (registry-resolved sides → coupled NkRow)
  → SolveRowsBlockIsland (mixed-island coupled solve; SolveUnionRowWarp per row)
  → integrate
```

Contact representation (already built, Phase 0–2A): shape→body indirection (`body_id = -1` ⇒ static), contacts
stored in the unified `ucontact_*` buffer with collidable ids + world-frame point/normal + generation. The island
solver already couples two dynamic sides per row (artic-link reduced-coord reaction + free-rigid 6-DOF maximal
reaction), keyed on cross-tree contact edges. **This IS the general solver. It exists and is validated.**

Foot-ground, dog-dog, finger-cup, box-box, body-terrain are then *the same code*: a foot sphere vs the ground/
heightfield collidable, a finger hull vs the cup hull, a trunk box vs another trunk box — all (collidable, collidable)
pairs through `cvx`/heightfield narrowphase → registry → island solve.

---

## 3. DELETE list (the hacks) — with anchors

| # | Delete | Where | Replaced by |
|---|--------|-------|-------------|
| F1 | `enum class ContactFamily` + `model.contact_family` + every `family`/`is_pair_driven`/`is_union` branch | model.hpp:208/389; pipeline.cpp; assemble_rows.cu; all `*.cu` `family` params | nothing — one path, no flag |
| F2 | FUSED foot detection | `contacts_foot.cu` `DetectFootGroundContactsKernel` (:77) | foot sphere vs ground/heightfield collidable via cvx/heightfield narrowphase |
| F3 | FUSED single-artic solver | `solve_rows.cu` `SolveArticulatedContactRowsKernel` (:81-351) + `OpSolveArticulatedContactRows` (:1015) | `SolveRowsBlockIslandKernel` (:647) — already general |
| F4 | FUSED row assembly | `assemble_rows.cu` `OpAssembleRowsFused` (:1049) + its kernels | `OpAssembleRowsPairDriven` (:937) — registry-resolved |
| F5 | UNION row assembly | `assemble_rows.cu` `OpAssembleRowsUnion` (:1122) + `EmitUnionRowsKernel` (:356) + pre-cooked union slots | `OpAssembleRowsPairDriven` (general detection feeds the same island solver) |
| F6 | `dog_dog_contact.cu` WHOLESALE | file + `NkOp::DogDogContact` + `DogDogContactParams` + pipeline block + `p_dog_dog_` + registration + CMake | LBVH→cvx (dog-dog) + heightfield (body-terrain), Phase 1B+2A |
| F7 | `instance_count` 特化 in c_abi | `world.cpp` the `if (desc->instance_count > 1u)` cook/family branch | generic scene composition (§6) — composition never changes the contact path |
| F8 | `"dog" + i` naming | `world.cpp` Compose call | generic instance prefix, e.g. `"inst" + i` (or scene-derived), no entity type |

**KEEP (= the general path; NOT hacks):** everything in Phase 0–2A — `cvx` narrowphase (`convex_narrowphase.hpp`),
`narrowphase_prims.cu` PairDriven driver, `narrowphase_heightfield.cu`, `OpAssembleRowsPairDriven`,
`SolveRowsBlockIslandKernel`/`SolveUnionRowWarp`, the registry (R1/R3/R5), `SyncLinkBodyPose`, the heightfield
collidable + `CookHeightfieldGrid`, `ComputeContactChainJacobianKernel` (already fully multi-artic). The cook drops
the family flag and ALWAYS produces the general collidable set.

**Note — UnionCsr already shares the solver.** `OpAssembleRowsUnion` and `OpAssembleRowsPairDriven` both feed
`SolveRowsBlockIslandKernel`/`SolveUnionRowWarp`. So collapsing UnionCsr → general is an *assembly* change (general
detection instead of pre-cooked slots), not a solver rewrite. FusedFoot is the only one with a private solver (F3).

---

## 4. The hard part — D1 / goldens (why this needs owner sign-off)

Two byte-pinned goldens currently ride the private paths:
- **go2 single-dog** `FeatherstoneOracle.NkWorldGo2Stand5s` (FusedFoot, `max_abs=1.93119e-05`).
- **H1 grasp** `UnionCookGolden` + `h1_grasp_lift` (UnionCsr) — *currently asset-gated SKIP in this worktree, but a real pin.*

Collapsing to one path **changes both goldens' bytes** (general foot contact ≠ analytic FUSED foot detection;
general finger-cup ≠ pre-cooked union slots). There is no way to delete the families AND keep the old bytes — the
old bytes ARE the old paths. So:

**V-PROOF (equivalence harness, MUST land before any deletion):** an offline harness that runs BOTH the old private
path and the new general path on go2-stand, go2-walk, and h1-grasp, and asserts the trajectories agree to a physical
tolerance (e.g. joint angles ≤ 1e-3 rad, base pose ≤ 1e-3 m over 5 s; contact impulses ≤ few %). This proves the
general path is *physically* the same physics, not a regression — only the last-bit arithmetic differs. THEN, with
owner sign-off, regenerate both goldens from the general path and pin them. The regen is a one-time, owner-approved,
explicitly-logged re-baseline (per [[nk-refactor-progress]] D1 discipline).

**Throughput gate:** the general path adds per-step LBVH+cvx over ALL bodies vs the cheap analytic foot path. Training
needs it to stay fast. The union-solver path measured **16,367 env-steps/s @N=1024** (`nuka_union_throughput`); the
general path is union-solver-based, but the LBVH broadphase every step is new cost. **Gate: re-measure
`nuka_union_throughput` (or equivalent) on the general go2 world; require ≥ ~10k env-steps/s** (≤ ~3 h / 100M-step
stage) before declaring training-ready. If it regresses, optimize (e.g. broadphase refit vs rebuild) — do NOT add a
"fast foot path" special-case back.

---

## 5. Rollout (build-up validated, landing atomic — no lingering gated hacks)

Owner dislikes lingering gated transitional states. Resolution: the *build-up* (Phase 0–2A) is validated-not-wired
scaffolding (acceptable — it's additive and inert). The **landing is ONE coherent change** that removes ALL gating
at once, so the repo never sits in a "half-special-cased" committed state for long:

- **L0 (pre-req): V-PROOF harness** proving general ≡ go2-FusedFoot and general ≡ H1-UnionCsr to tolerance. + throughput
  re-measure. This is also where any general-path foot-contact gaps (general foot-vs-ground/heightfield must reproduce
  the analytic FUSED foot detection) surface and get FIXED. → **owner reviews V-PROOF results + signs off on re-baseline.**
- **L-INST (scene-graph instancing, §6):** add the scene-graph instancing node ("one asset → N bodies at N transforms")
  + cook flatten → N collidables; migrate demos/tests to author K instances; remove the c_abi `instance_count`/`"dog"`
  compose + the C-ABI desc fields. Can proceed in parallel with L0 (independent of the contact collapse).
- **L1 (the landing, atomic): one-path collapse.**
  - Cook: always emit the general collidable set + registry; drop `contact_family` entirely.
  - Pipeline: one graph (no `family` params, no `is_*` branches); delete the FUSED/UNION ops + `dog_dog`.
  - Delete F1–F6.
  - Regenerate go2 + H1 goldens from the general path; pin; log the re-baseline.
- **L2: verify** — go2 stand/walk reproduces (vs V-PROOF + new golden), H1 grasp holds, multi-instance dogs collide +
  walk on real terrain (no explode/no 穿模), determinism + two-run identity, throughput gate.
- **L3: unified review** — `grep -rn` proves NO `ContactFamily`/`FusedFoot`/`UnionCsr`/`dog_dog`/`instance_count`-branch/
  `"dog"`-naming survives as a live path.

This keeps the *validated* incremental build-up but makes the *removal of hacks* a single clean landing, not a
drawn-out gated migration.

---

## 6. Scene-graph instancing (F7/F8) — OWNER-RULED: a real instancing mechanism in the scene tree

Owner clarification (2026-06-16): **多实例 = 一份资产实例化成多个机器人体/物理物体 → 必须挂在场景树,要有实例化机制。**
This is a legitimate, GENERAL feature: "one asset → N physics bodies" (20 dogs from one dog asset, 100 boxes from one
box asset). It is a **scene-graph instancing mechanism**, NOT a c_abi compose special-case.

End state:
- **Scene graph gains an INSTANCING node** — a node that references an asset (sub-scene/prim) and instantiates it N
  times at N transforms (USD-point-instancer-style / scene-graph instanced reference). One asset definition, N
  lightweight instances. This lives in the SceneGraph / scene file (per [[nk-refactor-plan]] Mangifera SceneGraph).
- **The cook flattens instances into N collidable bodies / articulations** — each instance is a first-class set of
  collidable `shape_table` rows + registry entries. N=1 and N=K differ only in how many bodies exist; the cook code
  path, the pipeline graph, and the contact solver are identical for any N. No `instance_count` branch anywhere.
- **No entity-type naming.** Instances are keyed generically (asset name + index), never `"dog"`/`"grasp"`/etc.
- **c_abi has ZERO instance logic.** It loads a scene (which may contain instancing nodes) and cooks it. The
  `instance_count` / `instance_spacing` / `"dog"+i` compose currently in `c_abi/world.cpp` + the C-ABI desc fields
  (the KEPT working-tree changes) are **removed** and replaced by scene-graph instancing. Demos/tests express
  "K dogs" by authoring K instances of the dog asset in the scene, not by a C-ABI count.
- The terrain/heightfield is a static collidable authored in the scene (or cooked from the scene's terrain spec) the
  same way regardless of how many instances exist.

This also subsumes the multi-dog co-residence cleanly: K dogs = K instances of the dog asset = K articulations of
collidable bodies = the one general contact path, with zero "multi-dog" concept in code.

---

## 7. Test migration

- `multi_dog_costep.cpp`, `pairdriven_*` → all cook the one general path (no family arg). Keep the behavioral asserts
  (co-step finite/independent, dogs collide + push apart, feet load on terrain, no 穿模, body rests on steps).
- `union_cook_golden.cpp`, `h1_grasp_lift` → re-baseline against the general path (asset-gated; regen when assets present).
- go2 oracle golden → re-baseline (V-PROOF + sign-off).
- New: V-PROOF equivalence harness (`tests/scenario/` or a standalone tool) + throughput re-measure.

---

## 8. DECISIONS — owner-ruled 2026-06-16

1. **Re-baseline BOTH goldens — RULED: YES (full unification).** Regenerate *both* the go2 single-dog golden AND the
   H1 grasp (UnionCsr) golden from the general path after V-PROOF proves equivalence to tolerance + owner sign-off.
   Delete `EmitUnionRows`/UnionCsr too. True one path, no families. (Option B "keep UnionCsr" rejected — a second
   assembly path is still a family = a partial hack.)

3. **Instancing — RULED: scene-graph instancing mechanism (see §6).** "One asset → N physics bodies" is a scene-tree
   instancing node, not a c_abi compose. c_abi gets ZERO instance logic; the KEPT `instance_count`/`"dog"` compose in
   c_abi is removed and replaced by scene-graph instances.

2. **V-PROOF tolerance — DEFAULT (owner did not object):** joint ≤ 1e-3 rad, base pose ≤ 1e-3 m over 5 s, contact
   impulse ≤ few %. Re-confirm at V-PROOF review.

4. **Throughput floor — DEFAULT:** ≥ ~10k env-steps/s @N=1024 (≤ ~3 h / 100M-step stage) for "training-ready".

5. **Scope — multi-session** (V-PROOF → scene-graph instancing → one-path landing → re-baseline both goldens →
   multi-instance terrain demo). Proceeding autonomously; owner reviews at milestones.

---

## 9. What does NOT change

- The `cvx` GJK/EPA narrowphase, the heightfield prism path, the mixed-island solver — already built, already the
  general path. We are WIRING everything onto them and DELETING the alternatives, not rebuilding.
- The trained go2 height-scan policy, H1 choreography, RL substrate — they drive the world; the world's contact
  solver underneath them becomes the one general path (re-validated by V-PROOF + re-train if needed).
- `instance_count` compose *mechanism* (just de-special-cased + de-named).
