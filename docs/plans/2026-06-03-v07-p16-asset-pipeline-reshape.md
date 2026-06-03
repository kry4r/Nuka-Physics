# p16 Plan — H1 Grasp-Cup Demo (v0.7 EXIT GATE, crit 7) — Asset-Driven Reshape

Date: 2026-06-03. **Supersedes the stale `2026-05-28-v07-p16-h1-grasp-cup-demo.md`** (which assumed bare-H1 / self-authored gripper / "complex USD scene"). Keeps the structure of the completed p16 Plan-agent output; reshapes it around the owner-provided newton-assets.

## 0. Owner directives that drive this (2026-06-03)

1. "之后仓库的所有资产都从这里找" → **ALL assets from `newton-physics/newton-assets`** (canonical source; don't self-model). [[newton-assets-resource]]
2. h1_with_hand.xml → **H1 WITH dual dexterous hands** (real end-effector; resolves the "no hand to grasp with" blocker).
3. kitchen → "尝试下…能不能做出真实的物理互动" → **kitchen scene as the demo ENVIRONMENT** (real interaction backdrop, not a bare table).
4. "对于cup，可以不要cup…抓取桌子上的东西" then "USD与MJCF共存…这个也是这次必做的…kitchen场景桌上放个usd cup" → the cup briefly dropped, then **came BACK as the vehicle for a NEW v0.7 MUST-DO: USD↔MJCF coexistence** (import USD into an MJCF scene / MJCF into a USD scene). The graspable on the counter = a **USD cup** placed into the **MJCF kitchen** (the USD-in-MJCF instance); a primitive ball/cube is the acceptable simple fallback. See the dedicated workstream in §10 + [[v07-usd-mjcf-coexistence]].

**FINDING (verified, kitchen.xml, 2026-06-03):** the kitchen has **NO loose free-floating graspable** — no `<freejoint/>`/`<joint type="free"/>`; the small props (toaster/utensil_holder/plant/paper_towel_holder, scale ~0.3–0.35) are **visual meshes, fixed in place**; the scene is only fixed architecture + articulated furniture (hinge doors / slide drawers) + static appliances; counters top at **z ≈ 0.445 m**. → To "grab something on the counter," **promote one counter-top prop (default: toaster) to a free rigid body**: add a freejoint + a **primitive convex collision proxy** sized to its bbox (gate physics on the proxy — no OBJ loader needed for the gate), rest it on a counter, and render its **real OBJ mesh** over the proxy (showcase). Standard collision-proxy pattern; honest (README states proxy-collision / real-mesh-render).

## 1. Verified asset facts (GitHub API + raw fetch, 2026-06-03)

| Asset | Container | Mesh fmt | Collision |
|---|---|---|---|
| `unitree_h1/mjcf/h1_with_hand.xml` | MJCF (tinyxml2 parses) | **STL** (binary, 46 files in `../meshes`) | ⚠️ finger/hand geoms `type="mesh"`, `class="collision"`=group3/mass0, visuals `contype=0 conaffinity=0` → **fingers VISUAL-ONLY, collision DISABLED** |
| `kitchen/mjcf/kitchen.xml` | MJCF, 412 KB (Robocasa→mujoco_warp, MIT) | **OBJ** (12 appliance subdirs) | ✅ **collision geoms are PRIMITIVES (box/cylinder/sphere) and ENABLED** (omit contype/conaffinity); visuals are mesh+contype0. ~150+ body / 400+ geom / 30+ joint; floor + counters @ z=0.46; **hinge** (cabinet doors), **slide** (drawers), knob hinges |
| `manipulation_objects/cup` | **USD** (`model.usda` 500 B ASCII wrapper, scale 0.03/0.03/0.0423) → `mesh.usd` 114 KB | **usdc BINARY crate** (PXR-USDC magic) | unknown; for the gate we use a convex proxy |

**Honesty finding (project discipline, mirrors the oracle-catalog deviation framing):** the stock assets are RENDER-oriented — beautiful visual meshes with collisions OFF. "Real physics interaction" is **not import-and-go**; it requires ENABLING + COOKING collision geometry. The demo doc/README must state plainly which surfaces are real-mesh-rendered vs which carry the actual contact geometry.

## 2. The spine: GATE-CRITICAL vs SHOWCASE (advisor-confirmed)

Every asset finding sorts into one of two buckets. **Hold this line** so the v0.7 close does not stall on showcase polish.

**GATE (closes crit 7 — minimal, tight):**
- **STL loader** (binary+ASCII) — the ONE loader the gate needs.
- **h1_with_hand**, with fingertip collision ENABLED + finger meshes COOKED to convex (V-HACD) → real-mesh-DERIVED collision on a real dexterous hand (not a fake; exercises crit-4).
- **Convex cup proxy** (cylinder+base) for the grasped object's collision.
- **One support surface** with collision = a kitchen counter body (primitive collision, already enabled) OR a simple table — the gate needs ONE place-on surface, not 100 interactive appliances.
- Scripted PD grasp+place + regression test (crit-7 wording: no interpenetration; grasp success; cup arrives at target; D1 two-run bit-exact; V2 energy bound).

**SHOWCASE (after the gate is green — layered, deferrable):**
- **OBJ loader** → render the full kitchen + (converted) cup meshes.
- Real **kitchen interactivity** (open drawers/doors via the slide/hinge joints — kitchen collision is primitive+enabled, so this is the existing rigid-contact path; no mesh-cook needed for kitchen physics).
- Real **USD cup render** via offline USD→OBJ convert (see §9 — env has NO USD tooling → owner-level fork).
- RT-render-to-video (two-level path) → homepage; README rewrite + logo.

## 3. #19 mesh-loader — corrected scope

`CollisionShapeRecord.mesh_vertices/mesh_indices` (`src/scene/scene_ir.hpp:50-51`) are READ by the cooker (`cooker.cpp:237/252/265`) but NEVER written by any importer. MJCF `<asset>` parsing (`mjcf_importer.cpp:229 ParseMaterials`) handles only `<material>`, never `<mesh name= file=>`; the geom parser never reads `mesh="..."`.

- **STL (binary + ASCII) — GATE.** New `src/import/mesh_file_loader.{hpp,cpp}`. Binary: header(80)+u32 count+50B/tri. ASCII: facet/outer-loop/vertex×3. Flat `mesh_vertices`(xyz) + `mesh_indices` (file order = determinism; no dedup needed). Two consumers from ONE parse: (a) importer→SceneIR (cook path: new `ParseMeshAssets` builds `name→path` from `<asset><mesh>` honoring `<compiler meshdir>`; geom parser resolves `type="mesh" mesh="..."` → populate verts/indices → existing cooker gate → V-HACD/SDF), (b) STL→`rt::BlasMesh` adapter (render path).
- **OBJ — SHOWCASE.** Add after the gate (kitchen render + converted cup). Wavefront `v`/`f` (triangulate polygons); ignore mtl for now (materials authored in the render scene).
- **cup usdc — SHOWCASE, sidestep in-engine.** See §9. Do NOT build an in-engine USD-mesh importer for the gate.

D1: a fixed file → fixed vertex/index order; cook seam stays CPU (p07) → D1 preserved.

## 4. Grasp approach (reshaped — real dexterous hand, not authored gripper)

The completed Plan-agent recommended hand-authoring a primitive parallel-jaw gripper because stock H1 had no hand. **h1_with_hand makes that moot** — use the REAL hands:
- Drive substrate proven: `control_mode.hpp` has PDPosition/Torque/Osc(task-space)/…; H1 C-ABI drive end-to-end in `examples/sim_val/g1h1_drive_harness.py` (D1-asserted). Scripted PD finger trajectory = low-risk.
- **Enable + cook fingertip collision:** import h1_with_hand; override the fingertip (and palm) geoms to collision-enabled (contype/conaffinity), honoring MuJoCo bitmask filtering for finger self-collision; cook those finger STL meshes to convex via #19→cooker→V-HACD. Grasp = approach → close fingers (PD) → lift → carry → place → release; contact on cooked-convex fingers vs convex cup proxy (proven p08 SDF-contact + PGS λ path).
- **A2 (mesh-SDF cup) and A3 (learned policy) remain out-of-gate** (A3 = follow-up; A2 = optional fidelity upgrade post-gate).

## 5. RT-render-to-video (showcase) — unchanged from Plan-agent §5
C++ harness (model on `src/apps/nuka_go2_stand_demo.cpp`, NOT the v0.3 python stick-rasterizer). Snapshot per-frame link world poses → `rt::TwoLevelScene` (one `BlasMesh` per unique STL/OBJ via §3 adapters, BLAS built once) → per-frame TLAS rebuild → p13 shading/AOVs → PPM (P6) frames (~33 ms/frame per G1-C; offline) → ffmpeg→MP4 (reuse graceful tail in `examples/demo/render_video.sh`; commit sample PNGs as evidence; MP4 under gitignored `out/`). **prim_id cap = PASS** (kInstanceBits=12→4096 instances, kPrimBits=20→1.05M prims/BLAS; H1 largest BLAS ~47k tris ≪ 2^20; instances ≪ 4096; re-verify vs final triangle/instance counts incl. kitchen).

## 6. #29 C-ABI camera arm (separable crit-6 deferral closure) — unchanged from Plan-agent §6
Video uses the triangle path → #29 is separable from both gate and video. Stock H1 collision being primitive means its `sdf_table` is empty even after #19; #29 (if scoped) is exercised by a cooked cup/finger SDF, not H1's body. Recommend: **keep #29 as a documented deferral** unless the cup is cooked to SDF (A2). Decide at p16-E.

## 7. Exit-gate crosswalk (how all 7 close) — per Plan-agent §7
crit1/2/3 = oracle-catalog `2026-06-03-v07-oracle-catalog.md` (gate-wording deviation; crit-2 front-speed + crit-3 SDF-contact-backward re-scoped v1.0 #23/#24). crit4 = V-HACD fires on the cooked finger meshes (gate evidence). crit5 = cite `tests/collision/test_cross_system_query.cpp` (grasp is rigid-only; cross-system covered by the standing test — EXIT-AUDIT item #17). crit6 = p14a F/T+tactile + p14b camera core shipped; #29 camera C-ABI = documented deferral (or wire under A2). crit7 = THIS demo + regression test.

## 8. Decomposition (serial, each committable)
- **p16-A — #19 STL loader (binary+ASCII)** + MJCF `<mesh>` asset map + geom `mesh=` resolution + STL→BlasMesh adapter. Test: round-trip an h1_with_hand STL; cook a real finger mesh → V-HACD fires; D1. **GATE.**
- **p16-B — scene assembly + enable/cook fingertip collision** (h1_with_hand + convex cup proxy + one counter/table surface, single-env). **GATE.**
- **p16-C — scripted grasp+place + regression test** (the crit-7 gate). **GATE.**
- **p16-D — OBJ loader + RT-render-to-video** (kitchen + cup render; two-level path). **SHOWCASE.**
- **p16-E — #29 C-ABI camera arm** (or documented deferral). **SEPARATE.**
- **p16-F — README rewrite + logo embed + exit-audit crosswalk doc.** **SHOWCASE + bookkeeping.**

Gate closes on **A+B+C** (+ crit-1..6 crosswalk/citations). D/E/F are showcase/deferral-closure and may slip past the gate.

## 9. USD cup fork — RESOLVED (owner 2026-06-03: "可以不要cup")
The USD cup is **DROPPED**. The graspable is a kitchen-scene prop (§0.4) instead → **no USD anywhere in p16**, no usdc reader, no offline conversion, no owner-side step. The earlier usdc-binary / no-env-tooling fork is moot. The graspable's collision = primitive proxy (gate); its render = the prop's real OBJ mesh (showcase, via the OBJ loader). This also closes out the stale `demo-homepage-readme-directives` "complex USD scene" wording — there is no USD in the demo at all.
