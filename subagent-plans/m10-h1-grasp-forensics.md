# M10 — H1 CUP-GRASP DEMO FORENSICS + THE CORRECT-GRASP FIX PLAN

> Synthesis of 5 forensic facets (SCENE, COLLISION/PHYSICS, STANDING/BALANCE, VIDEO/RENDER, RL-PATH).
> Subject: the REJECTED `out/h1_grasp.mp4` (code `examples/demo/h1_grasp_video.cpp` + `h1_grasp_driver.hpp`,
> physics from `tests/scenario/h1_grasp_lift.cpp` on the union cook of `examples/scenes/h1_cup_table.nks`).
> READ-ONLY recon. Every claim grounded in a file:line / scene field / frame / command output. Trust code over doc.
> Companion to `subagent-plans/m10-recon.md`.

---

## 1. VERDICT

**The owner is right on every point.** The rejected video is NOT "a standing H1 grasping a cup with real
collision." It is a **51-DOF floating-base H1 frozen in a baked crouch by open-loop joint PD, geometrically
buried inside the 1 m render box that stands in for a table, holding the cup with 30 invisible 6 mm proxy
spheres** doing a soft (penetration-permitting) friction contact — while the rendered dexterous-hand STL meshes
(collision OFF, `contype=0`) are drawn passing straight through the cup. The hero camera deliberately crops the
dangling legs out of frame (the demo's own comment, `h1_grasp_video.cpp:576-580`).

**THE CORE ROOT CAUSE is a category error: the demo's *render geometry is a different object from its collision
geometry*, layered on top of a robot that was never made to stand.** The visual hand/legs/torso are `contype=0`
visual-only meshes; the only colliders are fabricated proxy primitives (fingertip spheres, foot spheres, ground
plane, cup-table plane). So what you SEE never collides, and what collides is never seen. On top of that there is
**no balance controller anywhere in `src/`** (grep `balance|zmp|stance|com_polygon` = 0 hits) and the base 6 DOF
receive zero torque — "standing" is a short-horizon artifact of a settled initial condition. The demo cannot be
patched into correctness; it must be rebuilt toward the owner's RL target.

---

## 2. THE OWNER'S SHARP QUESTION, ANSWERED HONESTLY

> *"If the hand clips through the cup, there's clearly no collision — how is it even grasping?"*

**The owner's hypothesis is exactly, mechanically correct.** The visual hand is NOT a collider at all. The grasp
is carried entirely by 30 invisible cooked proxy spheres. Two independent facts compound:

1. **The visual dexterous hand has collision turned OFF.** Every hand/finger geom in the imported
   `.nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml` is `contype="0" conaffinity="0"` — **29 such
   geoms confirmed by `grep`** (lines 274,276,281,286,291,296,305,310,317,322,329,334,341,346 …). Per the
   newton-assets finding, these dexterous fingers are visual-only.
2. **The cook then drops ALL mesh geometry anyway.** `union_cook.cpp:125-129` clears `mesh_vertices` /
   `mesh_indices` for **every** shape. So no MJCF mesh — finger, hand, or limb — ever becomes a collider.

The ONLY hand colliders are 30 fabricated `wrap_spheres` (radius **6 mm**) pinned to the finger links
(`h1_cup_table.nks:326-657`, broadphase_id 9000-9029), cooked as `kFingerSphereHull` slots
(`union_nk_model.cpp`) that contact the cup convex hull via `cvx::SphereHull`
(`contacts_union.cu:163`). The renderer, meanwhile, *selects exactly the collision-OFF meshes for display*:
`is_visual = (s.contype==0 && s.conaffinity==0)` (`h1_grasp_video.cpp:411`). So the renderer draws the open
splayed visual fingers wherever FK puts them, while the only thing resisting the cup is a sparse cluster of tiny
invisible spheres. **Render geometry ≠ collision geometry → the visual fingers freely interpenetrate the cup.**

**Is the grasp force genuine?** Yes — but soft/penetrating. The `h1_grasp_lift` gate passes:
recovered vertical impulse `Σλ·ĵz = 8.69e-3` vs `m·g·dt = 8.18e-3` (**6.3 % dev**), `finger_min=5` contacts,
BITE (grip-off) drops the cup (`drop=1.058, FALLS`). That is real friction support from the proxy spheres. But it
is achieved by a **soft (penetration-permitting) contact**: `union_solimp` has `dmax=0.95` (deliberately
compliant, never hard) with `width=0.001` (full impedance only at ~1 mm penetration), and the solver runs
`pos_iters=0` (no position-correction / NGS pass) — `model.hpp:324-325`, `assemble_rows.cu:443-454`,
`h1_grasp_lift.cpp:267`. So even the invisible spheres sit penetrating the cup, and after the table is removed
at the lift the cup **sags 3.4-3.9 cm** while still "held" (demo CUP-Z probe: `cup_com_z 1.0805@f180 →
1.0461@f220`; gate `cup_dz=0.0387`).

**Bottom line:** the cup is held by invisible, penetrating proxy spheres. The visible hand is decorative geometry
that never collides — so it clips through. The owner saw the truth. The "grasp" is honest *physics* on the wrong
*geometry*, decoupled from the render, and it is a body-crook scoop, not a finger wrap (the arm never reaches and
the fingers curl only weakly: `drive_close` grip targets `kp=4 kd=0.4`).

---

## 3. CONFIRMED ROOT CAUSES

| # | Issue | Mechanism (file:line / scene / frame) | Severity |
|---|-------|----------------------------------------|----------|
| **A** | **H1 does NOT stand on its own — uncontrolled floating base, no balance controller, feet never load the ground.** | Root cooked as **FloatingBase, 6 free DOF** (`union_nk_model.cpp:22,92`; `base_dof:6 h1_cup_table.nks:1874`) under full gravity `-9.81` (`h1_grasp_video.cpp:555`). The drive law (`h1_grasp_driver.hpp:41-55` `TableTorque`: `u=kp*(target-q)-kd*qdot`) targets only **named joints**; `ResolveDriveTable` (`union_cook.cpp:172-196`) keys entries by joint link NAME — **no entry targets the root**, so the 6 base DOF get **zero torque**. Drive tables (`drive_hold/rest/close` `nks:733/1095/1457`) are a fixed **crouch** (hip_pitch=-0.40, knee=+0.87, ankle≈-0.466). No `balance|zmp|stance|com_polygon` exists in `src/` (grep = 0). 4 foot spheres (r=0.025, ankle offset z=-0.055, `nks:658-702`) vs ground plane at `ground_height=0.0` (`nks:704`); in the crouch the soles hover **~0.05-0.34 m above z=0** → `foot×ground` rows emit (`has_feet=true union_cook.cpp:329,347-352`) but stay permanently inactive. Stability is a **short-horizon artifact** of the 200-step settled IC (`settle.steps=200 nks:295-318`): gate runs only 300 steps (`h1_grasp_lift.cpp:89`), demo ~420; neither tests a ≥5 s unsupported stand (SG-spec S3). The demo's own comment admits it: *"the dangling fixed-base legs (the H1 hangs in the air) are cropped below the bottom edge"* (`h1_grasp_video.cpp:576-580`). The gate **never** checks standing/base/feet — only cup force-balance (`h1_grasp_lift.cpp:391-556`). | **blocker** |
| **B** | **H1 is JAMMED inside the 1 m render box ("table") — pelvis/torso geometrically embedded, no torso↔table collision.** | The "table" is a **render-only box** (`MakeBoxMesh` `h1_grasp_video.cpp:117,513`; full solid draw `:494-538`), half-extents `[0.5,0.5,0.25]` at center `[0.418,-0.196,0.754]` → box spans **z[0.504,1.004]**, x[-0.082,0.918], y[-0.696,0.304]. The pelvis IC is `[0.0106,0.0012,0.9715]` (`nks:289-293`) — **inside that box's z-range AND xy footprint**. But the box is **never cooked as a collider**: the cook fabricates a flat +Z half-space plane at `table_height=1.00373` for the cup (`union_cook.cpp:356-358`), and (per RC-A) drops all leg/torso mesh collision, and emits **no torso↔table contact pair** (`union_nk_model.cpp` slots = Foot×Ground + Finger×Hull + ONE cup-proxy×TablePlane only). Nothing resists the overlap → the torso renders emerging from the top face of the cube with legs hidden inside it. First-hand: `/tmp/h1frames/frame_000…419.png`, `/tmp/recon_frames/frame005.png`+`frame220.png`, `/tmp/recon_probe/probe_frame.png` all show the torso buried in the beige cube, no legs. (The cup-on-table relationship is the one thing roughly right: cup bottom ≈1.0056 rests just above table top 1.0037 — cup does NOT sink into the table; it sags later only after the table plane is removed at lift.) | **blocker** |
| **C** | **The held cup interpenetrates / sags — soft contact permits steady-state penetration; no position correction.** | Union contacts are MuJoCo-soft: `union_solref={0.02,1.0}`, `union_solimp={dmin=0.9, dmax=0.95, width=0.001, mid=0.5, power=2}` (`model.hpp:324-325`). `assemble_rows.cu:443-454` sets `pos=-depth` → `ComputeCompliantRow` → velocity-level spring-damper with a non-zero dual regularizer `R`; `dmax=0.95` (not ~0.9999) is **deliberately compliant (never hard)**; `solref_solimp.hpp:214-220` clamps to `dmax` with `R=1/D>0`. `vel_iters=64` but **`pos_iters=0`** (`h1_grasp_lift.cpp:267`) → no NGS/Baumgarte projection → penetration removed only asymptotically; steady-state overlap persists. Net: the proxy spheres (and via the bound mesh, the whole visual hand) sit penetrating the cup; after the table plane is removed at lift the cup **sags 3.4 cm** (`cup_com_z 1.0805→1.0461`, gate `cup_dz=0.0387`). The cup render-hull and cup-vs-finger collision hull MATCH (same `tmpl.cup.hull_verts` 1796 verts, `h1_grasp_video.cpp:461` render / `union_nk_model.cpp:166` collide) — so the cup shape itself is consistent; cup-vs-**table** uses a *separate* box proxy `cup_table_proxy_half=[0.054,0.054,0.0761]` (`nks:708-712`), a coarse second cup representation. | **high** |
| **D** | **Hand-mesh-through-cup clip = render geometry ≠ collision geometry (see §2).** | Render selects `contype==0` visual STL (`h1_grasp_video.cpp:411,437,144-146`); `h1_visual.nks` = **49 contype=0** visual shapes, none a collider. Source MJCF = **29 contype=0** hand geoms (`h1_with_hand.xml`). Cook drops all mesh geometry (`union_cook.cpp:125-129`). Only hand colliders = 30 `wrap_spheres` r=0.006 (`nks:326-657`, bp 9000-9029) → `kFingerSphereHull` → `cvx::SphereHull` (`contacts_union.cu:163`). Spheres are invisible + r=6 mm; visual fingers render open/splayed beside-and-through the cup, never wrapping. Frames `/tmp/recon_000180.png`, `/tmp/recon_000220.png`, `/tmp/h1_grasp_spotcheck/spot_000150…250.png` show fingers poking out the cup's far wall. **No self-collision and no hand-mesh↔cup collision exists anywhere.** | **blocker** |
| **E** | **The behavior is an open-loop scripted body-scoop, not a reach-and-grasp; and the green gate cannot catch any visual defect.** | Fixed choreography: hold(0-10)→rest(10-100)→close(100-180)→table-off lift at `kLiftAt=180` (`h1_grasp_driver.hpp:34-36,78-84`). Arm targets are ~identical across all 3 tables (shoulder/elbow ≈0.047/0.037) — the arm **holds, never reaches**; the IC pre-poses the hand at the cup. Grip curl is weak (`drive_close kp=4 kd=0.4 tlim=0 nks:1706-1800`). The `h1_grasp_lift` gate asserts ONLY: place_found, cup vertical impulse ≈ m·g·dt (±35 %), cup didn't fall >0.07 m, no cup-table static-row cheat, fingers >0 contact, BITE drops, disturbance bounded, 2-run + plan-replay determinism (`h1_grasp_lift.cpp:391-556`). It **never** checks posture/standing/leg-table interpenetration/visual-mesh clipping → a "passing" gate ships a visually broken video. The video deliberately shares this exact physics path so it "cannot drift from the gated physics" — but the gated physics is a sphere-proxy friction-hold, not a visually-correct grasp. | **high** |

---

## 4. THE FIX — STAGED PLAN TO THE OWNER'S TARGET (stand, arms at sides, raise arm, reach, grasp with REAL collision, no clip)

> Owner target = "should be RL": H1 STANDS on its feet, arms vertical at its sides, then RAISES one arm to reach
> and grasp the cup, with REAL collision and NO clipping. Built on the ONE generic `nuka.World` (highest
> directive: one general world, no special grasp world / no per-case solver branch).

### 4.0 ENGINE/PRE-WORK GAPS (must land before any showable rebuild — these are the §3 blockers turned into work)

- **G-A. Generic-world UNION cook.** `create_from_scene` routes `Scene→CookToModel→nk::World` (`world.cpp:176`)
  and `CookToModel` *"transcribes only the FIRST articulation today"* (`world.cpp:99`) — it does NOT build the
  union (cup body + fingertip collidables + feet×ground + cup×table). Reuse the proven
  `CookSceneToUnionTemplate`/`BuildNkUnionModel` logic **behind the generic API** (the union cook IS the general
  solver fed union scene data — not a special world type).
- **G-B. Torque action mode.** `world.cpp:144` **rejects** any `control_mode != PDPosition`; the proven grasp uses
  `drive_mode=1` (direct per-DOF torque), and SG-spec §4 mandates `Action = per-DOF torque`. Expose
  `DRIVE_TARGET` as torque through `create_from_scene`.
- **G-C. name→dof accessor.** `world.cpp` discards `cooked.scene_map` (m10-recon T14) → `h1_grasp_choreo.resolve()`
  and the RL obs/action have **no supplier**. Surface the map.
- **G-D. REAL fingertip collision that MATCHES the visual hand (the clip fix).** Cook each visual fingertip as a
  small **SPHERE** (or short sphere-capsule) sized+placed on the visual finger surface so **collision == render**.
  Use ONLY the robust path: `convex_narrowphase.hpp` `SphereHull` EPA-bypass is exact + monotone at all depths
  (`narrowphase_dispatch.hpp:432-466`). **Do NOT drop raw finger STL in as colliding hulls** — capsule/box/convex
  ×hull still ride the general EPA path with a shallow-penetration **dead band** (1-1.4 mm dropout), named
  consumer = C7b-2 (`narrowphase_dispatch.hpp:441-447`). Render the cooked sphere OR keep the visual mesh but
  pose-tune the grip so the visible fingers wrap the sphere-collider's cup contact → **no clip**.
- **G-E. Harden the held contact (the sag fix).** Raise `union_solimp.dmax` toward ~0.999 and/or add a
  position-correction pass (`pos_iters>0`) so the held cup neither penetrates the fingers nor sags 3-4 cm.

### 4.1 INTERIM (NON-RL), FASTEST-TO-SHOWABLE — recommended to ship FIRST

This fixes ALL FOUR owner complaints with HONEST sphere collision and a TRUE stand, **without a single training run** — it is scene/IC + control-script authoring, which is directive-compliant (behavior = scene data + per-scene control script):

- **I-1. Standing IC + real ground contact.** Re-pose the H1 to a near-straight stance with feet **ON the z=0
  ground plane** (feet loaded), arms **VERTICAL at the sides**, base height set so soles touch ground. Verify
  `Σλ_feet ≈ Mg` via the existing `ReportNk` foot rows. (The `FootSpherePlane` contact already exists.)
- **I-2. Replace the 1 m cube with a REAL table.** Render a normal table (top ~0.7-0.8 m, thin) **at the same
  plane as the cup-support `table_height`**, in front of a STANDING H1 — not a 1 m cube the torso sits inside.
  Pelvis must be ABOVE the table top, clear of any solid.
- **I-3. Authored arm-raise → close → lift choreography.** Extend the PD drive tables with a REACH phase
  (shoulder_pitch/roll + elbow) that brings the hand TO the cup, then a CLOSE phase that wraps the cooked-to-visual
  fingertip spheres (G-D) around the cup, then lift. Same `TableTorque` mechanism, new targets.
- **I-4. Render full body, no leg-crop.** Drop the leg-cropping hero camera (`h1_grasp_video.cpp:576-591`) and show
  the whole standing robot — the crop is exactly the dishonesty the owner rejected.

**Effort: ~3-6 focused sessions** (G-A…G-E + IC/choreography authoring + lavapipe render verify). This is the
pragmatic corrected demo. Note: a hand-authored *whole-body balance* on the full 51-DOF floating base is NOT
realistically tunable — I-1 stands by a settled IC + short horizon like today, just HONESTLY (feet on ground,
no buried base, no crop). Robust balance is the RL job below.

### 4.2 THE RL PATH (SG-spec G3 — the owner's "should be RL" end-state)

ONE greenfield env class on `nuka.World` (mirror `NukaGymEnv` lifecycle; do NOT port the deleted fixed-base
2-finger synthetic `GraspWorld` env at `git 4fe7970^:python/nuka/tasks/h1_grasp.py` — wrong substrate, recon CR-6),
config-gated **stages** (not one env per stage):

- **Obs** = privileged state (q/qdot env-major, base pose/vel, cup pose/vel, fingertip world pos, per-finger normal
  impulse, foot-contact state, last action) re-derived from generic field exports
  `LINK_CONTACT_WRENCH`/`CONTACT_FORCE`/`BASE_POSE`.
- **Action** = per-DOF torque via `DRIVE_TARGET` (DLPack zero-copy), needs G-B.
- **Reward** = dense shaped + **binary success from a SEPARATE deterministic evaluator** (A5b pattern).
- **Curriculum (each warm-starts the next):** S1 reach+grasp+hold (fixed base, cup on table) → S2 +lift →
  S3 floating-base STAND (legs RL, arms PD-held at sides, feet-only support) → S4 whole-body co-train
  (stand AND grasp+lift) → S5 +place. The reduced-DOF `python/nuka/tasks/h1_stand.py` (correct reward shape:
  upright + base-height + foot-flat + foot-contact, terminate height<0.55 or tilt>60°) is the S3 template, to be
  MIGRATED onto the 51-DOF union scene (so arms+hands are present), not used as-is.

**HONEST cost.** S1 smoke-learn (catch-eval rises — the literal M10 ask) is reachable in **~1-2 sessions IF
throughput passes**. A CONVERGED, natural-looking whole-body grasp (S4) is **research-grade / multi-session** with
reward-shaping iteration. **The gate is throughput:** SG-spec bar = ~100M env-steps/stage in <24 h; the 51-DOF
batched env-major number is **UNMEASURED** (T15 must measure at N∈{32,256,1024}); prior union N=1 ≈ 24235 eps/s
(commit 260bf80) is NOT the real bar. If it misses 100M<24 h, a named throughput increment must land BEFORE
training (spec G1(d)). Re-narrate `docs/specs/2026-06-10-h1-whole-body-rl-grasp-spec.md` into nk terms (T-spec).

### 4.3 RECOMMENDED SEQUENCING

**Ship the INTERIM (4.1 on G-A…G-E) as the corrected demo NOW** (fixes all 4 complaints with real sphere collision
+ a true stand + no clip + no crop) → **THEN pursue staged RL (S1 smoke-learn first, gated on measured
throughput)** as the milestone deliverable → the full converged natural grasp explicitly flagged multi-session
research-grade. **Do NOT gate the corrected demo on a converged policy.**

---

## 5. §OWNER DECISIONS (the key forks)

| ID | Question | Options | Recommendation |
|----|----------|---------|----------------|
| **OD-A** | How to deliver the CORRECTED H1 grasp? | (a) RL full converged natural grasp; (b) RL **staged-showable** (S1 smoke-learn, demo on the interim while RL matures); (c) **hand-authored interim** (stand IC + arm-raise choreography + cooked-to-visual fingertip spheres) | **(c) THEN (b).** Ship the hand-authored interim FIRST (fixes all 4 complaints in ~3-6 sessions with honest collision, no clip, a true stand, no crop), then drive staged RL as the milestone. Do not block the showable fix on a converged policy. |
| **OD-B** | Ship the Go2 walk video now while H1 grasp is rebuilt? | (a) ship Go2 walk now (already produced/committed, separate clean deliverable); (b) hold all videos until H1 is correct | **(a).** Go2 is a CO-HEADLINE (m10-recon OD), independent of the H1 grasp defect, and gives the README a correct headline immediately while H1 is rebuilt. |
| **OD-C** | Is a CONVERGED natural whole-body grasp worth the multi-session/research-grade cost now? | (a) pursue convergence now (multi-session, throughput-gated); (b) ship interim + S1 smoke-learn as the M10 milestone, defer convergence | **(b).** M10's literal RL ask is "smoke-learn / catch-eval rises," not converged. Convergence is research-grade and throughput-gated (UNMEASURED at 51-DOF). Land interim + S1; flag convergence as post-M10. |
| **OD-D** | How to make the hand stop clipping — render the colliders, or make the visual hand collide? | (a) render the proxy wrap_spheres (cheap, "what you see is what collides"); (b) cook fingertips as spheres sized to the VISUAL fingertip surface and pose-tune the grip so the visible mesh wraps (G-D); (c) cook raw finger STL as colliding hulls | **(b).** Keeps the beautiful PBR hand visible while making collision==render via the ONLY robust path (Sphere×hull EPA-bypass). Reject (c): capsule/box/convex×hull EPA dead band (named consumer C7b-2). (a) is the cheap fallback if (b)'s pose-tuning slips. |
| **OD-E** | Where does the corrected behavior live — generic `nuka.World` or keep the C++ union-cook path? | (a) build on generic `nuka.World` per the unified-world directive (needs G-A union cook + G-B torque + G-C name→dof); (b) keep the C++ `h1_grasp_lift` union-cook path for the interim, move to generic for RL | **(b) for interim, (a) for RL.** The C++ union path is proven on this box and fastest for the interim render; the generic-world gaps (G-A/B/C) are required for the RL env regardless and should be closed for the RL track, satisfying the directive where it matters (the trained policy). |

---

## 6. THE ONE-LINE TRUTH

The H1 never stands (uncontrolled floating base, no balance, feet off the ground), is rendered embedded in a
fake 1 m cube it never collides with, and "grasps" with invisible penetrating proxy spheres while the visible
hand — which has collision turned off — clips straight through the cup. **Render geometry ≠ collision geometry,
on a robot that was never made to stand.** The owner is right; the fix is a real stand + cooked-to-visual
fingertip-sphere collision (interim, ~3-6 sessions) then staged RL (S1 smoke-learn → … → S4 converged, research-grade).
