# v0.8 Plan — Unified Collision / Contact / Coupling Subsystem

> **Status:** owner-ratified requirements (via `/grill-me`, 2026-06-03) + controller-authored architecture. NOT the protected master plan. Supersedes the p16 "grasp on a proven SDF path" assumption (FALSIFIED — see [[v07-p16-contact-pipeline-reality]]).
> **Version restructure (owner, 2026-06-03):** v0.7 CLOSES NOW (asset pipeline #19/#31/#32 + the sim/RT subsystems shipped). The contact-subsystem rebuild + the post-v0.7 roadmap engine work + most former-v1.0 engine items → **v0.8**. **v1.0 = a few demos only** (grasp, photoreal showcase, homepage). The H1 grasp demo (former v0.7 exit gate #16) is RE-HOMED as a v0.8 consumer/validation of the new contact system.

## 0. Why this exists
The grasp gate exposed that contact is **fragmented**, not general, and dynamics orchestration is a monolith. Verified state ([[v07-p16-contact-pipeline-reality]]): three disjoint contact impls — (a) CPU maximal analytical (Plane/Sphere/Box/Capsule only), (b) GPU batched **foot-vs-ground-plane only** (`FootShape`×scalar `ground_height`×hardcoded +Z, impulse to articulation DOFs only, NO reaction body), (c) GPU SDF contact components with **zero stepper callsites**. No general broadphase→narrowphase→manifold pipeline; no articulated-link↔movable-body two-way reaction. Owner's framing (correct): **collision must be GENERAL; dynamics (solver) must be SEPARATED from detection; coupling (soft/fluid/rigid) all routes through collision → unify everything.**

## 1. Ratified requirements (the `/grill-me` decision tree)
| # | Decision | Choice |
|---|---|---|
| Q1 | Execution path | **GPU-first, one step** (no CPU-oracle intermediate; D1 validated GPU-side) |
| Q2 | Narrowphase architecture | **HYBRID**: analytical primitives (multi-point/face-face) + convex non-SDF (GJK/EPA/SAT + face-clip manifolds) + **SDF = high-precision tier**. mesh collidable via BOTH non-SDF (general) and SDF (high-precision). |
| Q2 | Manifold cardinality | **multi-point / face-face** (resting + grasp stability) |
| Q3 | Body/system pairs | **ALL**: rigid↔rigid, articulated-link↔rigid, articulated↔static, articulated↔articulated, + soft/cloth/fluid↔everything |
| Q3 | Migration | **migrate EVERYTHING** to the new system; re-baseline goldens OK (incl. subsume foot-ground) |
| Q4 | Unification level | **Method X**: leverage existing **row-class** abstraction; unified DETECTION → unified `ContactManifold` → constraint ROWS → existing solver; PBD systems (XPBD/PBF) bridged via coupling rows / co-step. NOT a unified-XPBD rewrite. |
| Q5 | Differentiability | **SDF tier ONLY** (p08 adjoint rides the high-precision tier); general/analytical/convex tiers forward-only, diff = v1.0+ seam |
| Q6 | Determinism | **FULL D1** throughout (stable contact ordering by stable keys, zero nondeterministic global-append atomics, fixed iteration counts; validate via two-run bit-identity + N≥32 cross-replica identity). D2/weak = optional fast-path seam, not default. |
| Q7 | Collision filtering | **MuJoCo-parity**: contype/conaffinity bitmask + `<contact><exclude>` pairs, culled at broadphase + **system-pair enable matrix** (rigid/articulated/soft/cloth/fluid). Requires NEW fields in IR/importer/cooked-blob. |
| Q8 | Friction | **linearized pyramid + isotropic + per-material μ + MuJoCo-parity combine** (reference Newton). cone/anisotropic = v1.0 seam. |
| Q9 | Broadphase topology | **B: per-system optimal structures + cross-query dispatcher** (reuse p04 LBVH rigid/articulated + p05 grid particles; generalize `cross_system_query` into a cross-system query MATRIX; unify the OUTPUT candidate-pair stream, not the geometry) |
| Q10 | Contact formulation | **soft/compliant, MuJoCo solref/solimp parity**, restitution default 0; hard contact = compliance→0 special case. (This is the glue unifying rigid contact with XPBD compliance.) |
| Q11 | v0.8 migration depth | **(ii) FULL migration** of all existing systems into the unified pipeline |
| — | Extensibility | registration-based extension seams for the post-v0.7 roadmap (new collidable types, new narrowphase pairs, new solvers e.g. Kamino ADMM, new coupling pairs, **CCD seam**). Maintainability + roadmap-readiness is a first-class requirement. |

## 2. Target architecture (Method X)
```
 per-system accel structures            cross-system query MATRIX (Q9-B)
 ┌─────────────┬──────────────┐        rigid↔rigid (LBVH self-traverse)
 │ rigid/artic │  particles   │  ───▶  particle↔rigid (cross_system_query, p05)
 │   LBVH(p04) │  grid(p05)   │        particle↔particle (grid)
 └─────────────┴──────────────┘        cloth/soft-vert↔rigid (BVH query)
        │  filtered (Q7: contype/conaffinity/exclude/system-matrix, culled here)
        ▼
   UNIFIED CANDIDATE-PAIR STREAM  (D1: sorted by stable keys)
        ▼
   PLUGGABLE NARROWPHASE DISPATCH  (Q2, keyed by geom-type-a × geom-type-b × tier)
   ├─ primitive×primitive  → analytical multi-point / face-face
   ├─ convex×convex/prim    → GJK/EPA/SAT + face-clip → multi-point manifold
   └─ SDF tier (high-prec)  → find_sdf_contact_newton (DIFFERENTIABLE, p08)  [now WIRED]
        ▼
   UNIFIED ContactManifold  (multi-point; carries both collidable handles + their
                             reaction providers: rigid invM / artic chain-J / particle invM)
        ▼
   ROW EMISSION  (existing row-class; Q10 compliant solref/solimp + Q8 pyramid friction)
        ▼
   ONE SOLVER  (row-class PGS; TWO-WAY reaction; subsumes foot-ground)
        ▲
   PBD systems (XPBD soft/cloth p09, PBF fluid p10) consume relevant manifolds via
   coupling rows / co-step bridge (K2 un-deferred, K3 retained — p11)
```
**Separation of concerns (owner Q3):** detection / manifold / row-emission / solver are distinct stages with explicit interfaces; the solver consumes rows, never shape pairs. Co-solving joint+drive+contact rows together in ONE solve is correct (coupled physics) and retained.

## 3. Extensibility seams (roadmap-driven — Q9 directive, see roadmap doc)
- **Collidable-type registry**: rigid / articulation-link / particle today; cable-segment, cloth-triangle, MPM-particle, FEM-tet later — each registers {AABB provider, accel-structure, shape/proxy provider, reaction provider}. No type enum hardcoded across dispatch.
- **Narrowphase dispatch table**: (type×type×tier) → handler; new pairs = new table entries.
- **Solver registry**: row-class PGS today; **SolverKamino (Proximal-ADMM, closed loops, R6)** plugs in alongside Featherstone, consuming the same manifolds.
- **Coupling matrix (R8)**: rigid↔MPM↔cloth↔fluid two-way = new coupling-row classes per pair.
- **CCD seam**: discrete-only in v0.8; swept-test hook reserved for garment cloth (R3).

## 4. Full-migration inventory (Q3/Q11 — what moves onto the new system)
1. CPU maximal analytical contact → general narrowphase (primitive tier).
2. GPU foot-ground → general articulated-link↔static/rigid (re-baseline go2/h1 standing golden).
3. SDF contact (unwired) → WIRED as the high-precision differentiable tier.
4. XPBD soft/cloth (p09) → consume unified manifolds via co-step.
5. PBF fluid (p10) → consume unified manifolds via co-step.
6. K2 (particle↔rigid-SDF, was v1.0-deferred) → brought in; K3 retained (p11).

## 5. Decomposition (serial, each committable; D1-gated)
- **C1 — Filtering & contact metadata foundation**: contype/conaffinity/group + exclude-pairs + solref/solimp + per-material μ into SceneIR/CollisionShapeRecord/MaterialRecord + MJCF importer parse + cooked blob + system-pair matrix.
- **C2 — Unified broadphase dispatcher**: generalize LBVH + grid + cross_system_query → cross-system candidate-pair MATRIX, filtered (C1), D1 sorted output.
- **C3 — Unified narrowphase + multi-point ContactManifold**: analytical multi-point + convex GJK/EPA/SAT + face-clip; wire SDF tier; one D1 manifold type.
- **C4 — Manifold→rows: compliant contact + pyramid friction + reaction providers** (rigid invM / chain-J / particle invM), MuJoCo solref/solimp + combine.
- **C5 — Solver wiring + two-way reaction**: all row classes through one solve; articulated-link↔rigid (grasp crux); subsume foot-ground; re-baseline goldens.
- **C6 — Full migration**: XPBD/PBF/coupling onto unified pipeline; un-defer K2.
- **C7 — Grasp demo (former #16) as validation**: H1 hand grasps cup → places on table, on the unified system; regression (no interpen / hold / D1 / V2) + RT video (G1 tracer).
- **C8+ — roadmap engine work** (W1 IK→cable→garment→MPM→FEM→Kamino→tendon; W2 G2–G4; W3 U1–U5), value-ordered, each reusing the new contact spine.

## 6. v1.0 = demos only
Per owner: v1.0 ships **a few demos** (grasp, photoreal showcase, GitHub homepage) on the v0.8 engine. All engine/feature work (collision subsystem + roadmap W1/W2/W3 + former-v1.0 engine items: GPU SDF cook backend, AMG, SDF reverse-mode, coupling wire-in) is pulled into **v0.8**. Roadmap doc to be converged accordingly.
