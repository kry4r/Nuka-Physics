# Research — Newton & MuJoCo Warp/MJX for Nuka's unified collision/contact/coupling subsystem

> Read-only research, 2026-06-03, for the v0.8 unified collision design ([[2026-06-03-v08-unified-collision-contact]]).
> Sources: local clones `newton-physics/newton` @ main, `google-deepmind/mujoco_warp` @ main, `google-deepmind/mujoco`; MuJoCo docs. File paths below are inside those clones (`newton/` and `mujoco_warp/` roots). Citations are exact code locations — the math is quoted from source, not paraphrased.

---
## NEWTON (newton-physics/newton — on NVIDIA Warp)

### 1. Solver registry / Model–State–Control–Contacts separation  → directly informs Nuka's "solver registry + row-class"
- **`SolverBase.step(state_in, state_out, control, contacts, dt)`** (`newton/_src/solvers/solver.py:300`). The step signature is the registry contract: **`Contacts` is a first-class argument passed INTO every solver**, alongside `Model` (constant topology/params), `State` (q/qd/forces, double-buffered in/out), `Control` (actuation). `notify_model_changed(flags)` lets a solver rebuild caches when the model mutates.
- Solvers are independent subclasses, NOT a unified solve: `newton/_src/solvers/{featherstone, mujoco, xpbd, vbd, semi_implicit, implicit_mpm, kamino, style3d}/`. Public re-export `newton/solvers.py` carries a support matrix (rigid/articulation/particle/cloth/soft/differentiable per solver). One `Model`+`State` can be advanced by any registered solver; you pick ONE solver per step (no cross-solver row-mixing inside a single solve — contrast Nuka, which co-solves joint+drive+contact rows in ONE solve).
- `SolverBase` ships shared helper kernels (`integrate_bodies`, `integrate_particles`, kinematic-body effective inv-mass zeroing) so back-ends reuse integration. Kamino vendors its OWN `CollisionPipelineUnifiedKamino` (`solvers/kamino/_src/geometry/`) — i.e. a solver MAY bring a private collision path.
- → **Nuka: ADOPT** the `step(state_in, state_out, control, contacts, dt)` contract verbatim as the solver-registry interface — Contacts-as-argument is exactly the "solver consumes rows, never shape pairs" separation (v08 §2). **DIVERGE (keep)** our single co-solve of joint+drive+contact rows; Newton's "one solver per step" is weaker than our row-class unification.

### 2. Collision/contact: shared & solver-agnostic  → Nuka Q9/Q2/Q6
- **`CollisionPipeline`** (`newton/_src/sim/collide.py:464`) is solver-agnostic: built from `Model`, produces a `Contacts` object consumed by whichever solver runs. Pluggable broadphase `"nxn" | "sap" | "explicit"` (`BroadPhaseAllPairs`/`BroadPhaseSAP`/`BroadPhaseExplicit`, `geometry/broad_phase_*.py`) + pluggable `NarrowPhase`.
- Narrowphase is a **dispatch over geom-type pairs into tiers** (`geometry/narrow_phase.py`): analytical `collision_primitive.py` (sphere/capsule/plane/box/cylinder/ellipsoid pairs), convex via **GJK + MPR** (`mpr.py`, `simplex_solver.py`, `support_function.py`), mesh-vs-convex midphase, and **SDF tiers** (`sdf_contact.py`, `sdf_hydroelastic.py`, `sdf_mc.py`, `sdf_texture.py`). Multi-point via `multicontact.py` + `contact_reduction*.py` (incl. `contact_reduction_hydroelastic.py`). This is *exactly* Nuka's Q2 hybrid (analytical + convex GJK/EPA/SAT + SDF high-precision tier).
- **`Contacts`** (`newton/_src/sim/contacts.py`) is a flat SoA buffer set, split rigid vs soft (`rigid_contact_*`, `soft_contact_*`), with a `generation` counter and an `EXTENDED_ATTRIBUTES = {"force"}` opt-in **contact-force readout** (request via `request_contact_attributes`) — maps to Nuka's §4.D contact-force readout decision.
- **Reaching each solver:** shared `Contacts` is consumed directly by XPBD/VBD/Featherstone, but **`SolverMuJoCo` TRANSLATES** it: `convert_newton_contacts_to_mjwarp_kernel` / `create_convert_mjw_contacts_to_newton_kernel` (`solvers/mujoco/kernels.py`) marshal the shared format ⇄ mujoco_warp's native contact arrays. So "shared representation + per-solver adapter kernel" is the real pattern.
- → **Nuka: ADOPT** one solver-agnostic `CollisionPipeline`→`ContactManifold` with pluggable broad/narrow phase and an opt-in `force` attribute. **ADAPT** the per-solver translate-kernel idea for our PBD bridge (XPBD/PBF consume manifolds via a co-step adapter — v08 §2 already plans this).

### 3. `newton.ik` — API + algorithm  → Nuka W1 IK
- API (`newton/ik.py`): `IKSolver` over a list of `IKObjective`s (`IKObjectivePosition`, `IKObjectiveRotation`, `IKObjectiveJointLimit`), optimized by **`IKOptimizerLM` (Levenberg–Marquardt)** or **`IKOptimizerLBFGS`**, with an `IKSampler` for multi-restart (keeps lowest-cost candidate per problem). `IKJacobianType = {ANALYTIC, AUTODIFF, MIXED}` (`ik_common.py`) — Jacobian from analytic where available else Warp autodiff.
- Algorithm = **batched nonlinear least-squares**, NOT classic per-step DLS. LM IS damped-least-squares with adaptive damping: `IKOptimizerLM` (`ik_lm_optimizer.py:83`) has `lambda_initial=0.1`, `lambda_factor=2.0`, `lambda_min=1e-5`, `lambda_max=1e10`; λ scales up on rejected and down on accepted trial steps (trust-region style). Solves `(JᵀJ + λI) Δ = -Jᵀr` per iteration; massively batched over `n_batch` problems sharing one articulation.
- → **Nuka: ADAPT** — adopt LM (adaptive-λ DLS) + objective-list API + ANALYTIC/AUTODIFF/MIXED Jacobian as our IK design; the batched-over-restarts structure fits our GPU-first model. No task-space QP in Newton — keep that out of scope for v0.8.

### 4. Coupling between solvers/materials  → Nuka R8 coupling matrix
- Newton does NOT run two solvers simultaneously two-way; coupling is **intra-solver**. `SolverImplicitMPM` (`solvers/implicit_mpm/solver_implicit_mpm.py`) does two-way MPM↔rigid INSIDE itself: it ingests rigid colliders as SDF fields (`collider_position_field`, `collider_distance_field`, `collider_normal_field`) and writes an **`impulse_field`** back onto colliders (lines ~2286–2300, "Necessary fields for two-way coupling"); the MPM solve is a "coupled rheology/contact solve" (line 1567). VBD also has `rigid_vbd_kernels.py` for cloth↔rigid.
- → **Nuka: DIVERGE (we go further)** — Newton's coupling is baked per-solver; Nuka's plan routes ALL coupling through the shared manifold + coupling-row classes (v08 §3 R8), which is more uniform. **BORROW** the concrete mechanism: represent the rigid collider to the soft/fluid solver as an SDF field and read an impulse/reaction field back — that is the two-way handshake to copy for K2/K3.

### 5. Determinism + differentiability via Warp  → Nuka Q5/Q6
- **Determinism:** `CollisionPipeline(deterministic=True)` adds a **radix-sort + gather** after narrowphase so contact order is independent of GPU thread scheduling (`contact_sort.py`, `make_contact_sort_key`); default path uses `atomic_add` for the contact index (`collide.py:108`) → nondeterministic order unless sorted. `contact_matching ∈ {disabled, latest, sticky}` gives frame-to-frame stable contact identity (implies `deterministic=True`) — this is warm-start stability.
- **Differentiability:** narrowphase kernels are `enable_backward=False` (**frozen geometry**). `geometry/differentiable_contacts.py` post-processes: it re-reads frozen body-local contact points/normal/margin and **reconstructs world-space contact quantities through the differentiable `body_q` transforms**, giving first-order (tangent-plane) gradients of contact distance/points w.r.t. body pose; the **normal passes through unchanged** ("gradients flow through the contact points and distance"). Explicitly flagged experimental/"tangent approximation."
- → **Nuka: ADOPT** the deterministic radix-sort-by-stable-key (our Q6 D1 mandate — Newton validates the design). **ADOPT** `contact_matching` "sticky" as our warm-start-identity mechanism. **ALREADY-HAVE / VALIDATES:** Newton's "freeze narrowphase, differentiate only the transform reconstruction" matches Nuka's Q5 "diff on SDF tier only, narrowphase forward-only" — and warns the general-tier tangent gradient is approximate, supporting our choice to keep diff on the SDF tier.

---
## MUJOCO WARP / MJX (google-deepmind/mujoco_warp)

> **MJX vs MuJoCo Warp note:** MJX (the JAX backend, `mujoco.mjx`) and MuJoCo Warp share the **identical constraint model** (same solref/solimp impedance, same CG/Newton convex solver, same contype/conaffinity/pair/exclude semantics — all inherited from MuJoCo classic `mjModel`); they differ only in the execution backend (JAX/XLA vs NVIDIA Warp) and in narrowphase coverage/perf. All findings below (solref/solimp, filtering, friction) apply equally to MJX. mujoco_warp is the newer, MuJoCo-team-blessed successor and the one Newton wraps as `SolverMuJoCo`.

### 6. Collision driver: per-pair dispatch + broadphase + islands  → Nuka Q6/Q9
- **Per-pair-type dispatch table** `MJ_COLLISION_TABLE` (`mujoco_warp/_src/collision_driver.py:45`) — explicit `(GeomType_a, GeomType_b) → CollisionType ∈ {PRIMITIVE, CONVEX}`, mirroring MuJoCo classic's `mjCOLLISIONFUNC` (`engine_collision_driver.c`). e.g. PLANE×{SPHERE,CAPSULE,BOX,MESH}=PRIMITIVE; SPHERE×ELLIPSOID, all HFIELD×*, all *×MESH = CONVEX; `BOX×BOX=CONVEX` unless `DisableBit.NATIVECCD` flips it to PRIMITIVE.
- `_narrowphase` (`collision_driver.py:836`) partitions filtered pairs by table value and fans out to **`primitive_narrowphase`, `convex_narrowphase` (GJK, `collision_gjk.py`), `sdf_narrowphase` (`collision_sdf.py`), `flex_narrowphase`** — one kernel per tier, gathering its pair subset.
- **Broadphase:** two modes (`m.opt.broadphase`): `NXN` (all-pairs) and **`SAP` (sweep-and-prune, `_sap_project`/`_sap_range`/`_sap_broadphase` + segmented sort)**. A staged **broadphase filter** bitmask `BroadphaseFilter.{PLANE,SPHERE,AABB,OBB}` does cheap rejects before the candidate is written (`_broadphase_filter`, line 286).
- **Pair pre-filtering at MODEL-BUILD time:** the candidate pair list `nxn_geom_pair_filtered` is "filtered at model creation time to exclude pairs based on `contype`/`conaffinity`, parent-child relationships, and explicit `<exclude>` tags" (`collision_driver.py:797`). So bitmask/exclude is a compile-time set-difference, not a per-step branch.
- **Contact islands:** `island.py` builds tree-tree edges (`_tree_edges`) from contacts+equality+joints to partition the constraint problem into independent islands solved in parallel (`_create_island_solver_context`, `solver.py:60`).
- → **Nuka: ADOPT** the static `(typeA × typeB × tier) → handler` dispatch table (= our pluggable narrowphase dispatch, v08 §3) and **build the filtered pair list at cook/model-build time** (our C1 cooked-blob can bake the contype/conaffinity/exclude set-difference, matching MJ). **ADOPT** the staged cheap broadphase-filter bitmask. **CONSIDER** contact islands as a future D1-friendly parallel-solve partition (not in v0.8 scope; note for solver phase).

### 7. Constraint solver + EXACT solref/solimp  → Nuka Q7/Q10 (HIGHEST VALUE)
- **Solver:** `SolverType ∈ {CG, NEWTON}` (`types.py:443`; `PGS` explicitly unsupported in warp). Default **NEWTON = primal Newton with Cholesky of the Hessian** (`block_cholesky.py`, `solver.py`), Gauss term + exact line-search over a quadratic (`_eval_pt*`); **CG = primal conjugate-gradient**. All constraints (contact + joint limit + equality + friction-loss) are assembled into one `efc` (constraint) row set and solved together as a single **convex optimization** (the MuJoCo dual/primal soft-constraint program), NOT a hard LCP.
- **EXACT solref/solimp** — verbatim from `constraint.py:_efc_row` (lines 81–117), the source of truth Nuka's compliant contact must match:
  ```
  timeconst=solref[0]; dampratio=solref[1]
  dmin=solimp[0]; dmax=solimp[1]; width=solimp[2]; mid=solimp[3]; power=solimp[4]
  if not REFSAFE-disabled: timeconst = max(timeconst, 2*timestep)         # ref-safety clamp
  dmin,dmax = clamp(.,MJ_MINIMP,MJ_MAXIMP); width=max(MJ_MINVAL,width); mid=clamp(.,MJ_MINIMP,MJ_MAXIMP); power=max(1,power)
  # stiffness k, damping b (positive solref = (timeconst,dampratio)):
  k = 1 / (dmax^2 * timeconst^2 * dampratio^2)
  b = 2 / (dmax * timeconst)
  # NEGATIVE/direct convention (non-positive solref → (-stiffness,-damping)):
  if solref[0] <= 0:  k = -solref[0] / dmax^2
  if solref[1] <= 0:  b = -solref[1] / dmax
  # impedance sigmoid d(r) over x=|pos_imp|/width:
  x   = |pos_imp| / width
  a   = (1/mid^(power-1)) * x^power
  bb  = 1 - (1/(1-mid)^(power-1)) * (1-x)^power
  imp_y = (x < mid) ? a : bb            # C1 sigmoid, =linear when power==1
  imp = clamp(dmin + imp_y*(dmax-dmin), dmin, dmax);  if x>1: imp=dmax
  # outputs into the constraint:
  D   = 1 / max( invweight * (1-imp)/imp , MJ_MINVAL )    # diagonal stiffness/regularizer R≡1/D ∝ (1-imp)/imp
  aref = -k*imp*pos_aref - b*vel                          # reference accel (spring-damper)
  ```
  Constraint dynamics: `a_c + d·(b·v + k·r) = (1−d)·a_u`; impedance `d∈[dmin,dmax]` blends between unforced accel `a_u` (d→0, soft) and the spring-damper reference (d→1, hard). **`D` = `efc_D` is the PRIMAL constraint stiffness** (the cost weight: warp's per-row cost is `0.5·Jaref²·efc_D`, `solver.py:611`; d→1 hard ⇒ D→∞). Its inverse **`R = 1/D` is the DUAL diagonal REGULARIZER added to `A=JM⁻¹Jᵀ`** (`R ∝ invweight·(1−imp)/imp`; d→1 ⇒ R→0 ⇒ hard). Do NOT swap D and R when porting — the formula above outputs D; add `1/D` to A. (MuJoCo docs: modeling.html#solver-parameters.)
- → **Nuka: ADOPT verbatim** — port `_efc_row` math exactly (incl. the `timeconst≥2·dt` REFSAFE clamp, the non-positive→direct `(-k,-b)` convention, and `power==1`⇒linear) into our manifold→row compliance (C4). This is the literal glue that unifies rigid contact with XPBD compliance (Q10). **ADAPT** solver: our row-class PGS differs from MJ's NEWTON/CG — but soft-constraint `R=1/D` regularization makes PGS converge on the same soft program, so the *parameter semantics* port even though the solver doesn't.

### 8. contype/conaffinity + exclude + pair override — EXACT filtering  → Nuka Q7 import fidelity
- **Bitmask rule (verified):** geoms i,j collide iff `(contype_i & conaffinity_j) || (contype_j & conaffinity_i)` is nonzero (bitwise AND, two-way OR). Defaults: both `contype` and `conaffinity` = 1 (everything collides). (MuJoCo XMLreference / Collision detection.)
- **Precedence (verified MuJoCo classic, inherited by warp):** (1) explicit `<contact><pair>` always generates the contact and **overrides the bitmask** (a pair is collided even if the bitmask would reject it). (2) `<contact><exclude body1 body2>` removes a *body pair* from collision entirely. (3) Parent-child bodies auto-excluded unless re-enabled. (4) Otherwise the contype/conaffinity bitmask decides. In warp this whole set-difference is precomputed into `nxn_geom_pair_filtered` at model build (`collision_driver.py:797`).
- **Per-contact param merge when a pair has no explicit `<pair>` — VERIFIED VERBATIM from `mj_contactParam` (`mujoco/src/engine/engine_collision_driver.c` ~L2850–2930):** if `priority` differs, the **higher-priority geom supplies ALL** contact params (condim/friction/solref/solimp); if equal priority:
  ```c
  condim = mjMAX(condim1, condim2);
  for i in 0..3:  fri[i] = mju_max(friction1[i], friction2[i]);           // friction = elementwise max
  mix = (solmix1>=MINVAL && solmix2>=MINVAL) ? solmix1/(solmix1+solmix2) : ...;
  if (solref1[0] > 0 && solref2[0] > 0)                                    // both STANDARD mode
      solref[i] = mix*solref1[i] + (1-mix)*solref2[i];                     //   → solmix-weighted average
  else                                                                     // either DIRECT/negative mode
      solref[i] = mju_min(solref1[i], solref2[i]);                         //   → elementwise MIN, not mix
  // solimp also solmix-mixed; margin/gap = max of the two geoms' values (getMargin/getGap).
  ```
- Newton's own native filtering is **MuJoCo-style signed collision GROUPS**, not bitmasks: `shape_collision_group` (+`shape_collision_filter_pairs`) in `model.py:244`; `test_group_pair` (`broad_phase_common.py:133`): positive group = collide only within same group; negative = collide with everything except its own negative counterpart; group 0 = collide with all.
- → **Nuka: ADOPT** the exact bitmask + precedence (pair-override → exclude → parent → bitmask) and the per-contact merge rule into C1 importer + cooked blob; this is required for MJCF import fidelity. **NOTE** Newton chose signed-group filtering — simpler but NOT bit-for-bit MJCF-compatible; Nuka's Q7 contype/conaffinity choice is the more faithful one.

### 9. Friction: pyramid vs cone + combine  → Nuka Q8
- **Cone vs pyramid:** `m.opt.cone ∈ {PYRAMIDAL, ELLIPTIC}`. Pyramidal linearizes the friction cone; row count `nmaxpyramid = max(1, 2*(condim-1))` (`io.py:301,1088`) — e.g. condim=3 → 4 pyramid edges. Elliptic = true second-order cone `f1² ≥ Σ f_i²/μ_i²`. `condim ∈ {1,3,4,6}` (1=frictionless, 3=tangential, 4=+torsional, 6=full; **2 and 5 illegal** — tangential/rolling come in pairs). `impratio` sets normal-vs-friction constraint stiffness ratio (`impratio_invsqrt`, `io.py:245`).
- **Friction-coefficient combine across geoms = ELEMENTWISE MAX** of the two geoms' friction vectors (when equal priority; else higher-priority geom wins) — verified `mju_max` in `mj_contactParam` (see §8). Friction vector dim = `condim−1`. (warp consumes the already-merged `mjModel`/`pair_friction`.)
- → **Nuka: ADOPT** linearized **pyramidal** cone with `2*(condim−1)` edges and the **elementwise-max** friction combine (our Q8 = pyramid + isotropic + per-material μ + MuJoCo-parity combine — exact match). Keep elliptic-cone as a v1.0 seam. **ALREADY-PLANNED:** isotropic μ now; anisotropic/torsional (condim 4/6) deferred.

---
## TOP RECOMMENDATIONS FOR NUKA v0.8 COLLISION SUBSYSTEM
1. **Port `_efc_row` solref/solimp math VERBATIM** (`mujoco_warp/_src/constraint.py:81–117`): the `(k,b)` derivation, the `timeconst≥2·dt` REFSAFE clamp, the non-positive→direct `(-k,-b)` convention, the two-branch `power`-sigmoid `imp_y`, and `D=1/(invweight·(1-imp)/imp)`. This is the single most decision-critical, error-prone item (Q7/Q10).
2. **Adopt the `step(state_in, state_out, control, contacts, dt)` solver-registry contract** with **Contacts passed in** — clean detection↔solver separation, matches v08 §2; keep our single co-solve (joint+drive+contact rows together) as our deliberate strengthening over Newton's one-solver-per-step.
3. **Static `(typeA × typeB × tier) → handler` dispatch table** (MJ `MJ_COLLISION_TABLE` / `mjCOLLISIONFUNC` pattern) = our pluggable narrowphase dispatch (v08 §3 extensibility seam); SDF tier as just another table value.
4. **Bake the filtered candidate-pair list at COOK time** from contype/conaffinity + `<exclude>` + parent-child set-difference (MJ does it at model build, `collision_driver.py:797`) — Nuka C1 cooked-blob, avoids per-step filter branching.
5. **Implement the exact MJCF filtering precedence:** `<pair>` override → `<exclude>` → parent-child → `(contype_i & conaffinity_j)||(contype_j & conaffinity_i)`; and the per-contact merge (priority winner, else max-condim / max-friction / solmix-mix, min-on-direct-solref, max-margin) for import fidelity (Q7).
6. **Deterministic radix-sort-by-stable-key after narrowphase** (Newton `deterministic=True`, `contact_sort.py`) — validates Nuka's Q6 D1 mandate; avoid the default atomic-append order.
7. **Adopt `contact_matching="sticky"`-style frame-to-frame contact identity** for warm-start stability (Newton `contact_matching`) — pairs with our existing warm-start cache.
8. **Pyramidal cone with `2*(condim−1)` edges + elementwise-max friction combine** (Q8 exact MJ parity); elliptic cone + anisotropic = v1.0 seam.
9. **Two-way coupling handshake = SDF-field-in / impulse-field-out** (Newton `SolverImplicitMPM` `collider_*_field` + `impulse_field`) — concrete mechanism to copy for our K2/K3 coupling rows, but route it through the shared manifold (more uniform than Newton's per-solver baking).
10. **IK = batched Levenberg–Marquardt (adaptive-λ DLS) + objective list + ANALYTIC/AUTODIFF/MIXED Jacobian** (Newton `IKOptimizerLM`); freeze-narrowphase + differentiate-only-transform-reconstruction confirms Nuka's "diff on SDF tier only" (Q5) is the right scope — Newton itself flags the general-tier tangent gradient as approximate.

---
## SOURCES
- Newton repo (clone `newton-physics/newton` @ main): `newton/_src/solvers/solver.py` (SolverBase.step:300); `newton/solvers.py` (registry+support matrix); `newton/_src/sim/collide.py` (CollisionPipeline:464, atomic append:108, sort key:145); `newton/_src/sim/contacts.py` (Contacts, EXTENDED_ATTRIBUTES `force`); `newton/_src/geometry/{narrow_phase,collision_primitive,mpr,simplex_solver,support_function,sdf_contact,sdf_hydroelastic,multicontact,contact_reduction*,contact_sort,differentiable_contacts,broad_phase_sap,broad_phase_nxn,broad_phase_common}.py`; `newton/_src/sim/ik/{ik_solver,ik_lm_optimizer,ik_common}.py`; `newton/ik.py`; `newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py` (two-way coupling ~2286); `newton/_src/solvers/mujoco/{solver_mujoco,kernels}.py` (contact convert kernels); `newton/_src/sim/model.py` (shape_collision_group:244).
- MuJoCo Warp repo (clone `google-deepmind/mujoco_warp` @ main): `mujoco_warp/_src/collision_driver.py` (MJ_COLLISION_TABLE:45, SAP, broadphase filter:286, _narrowphase:836, filter comment:797); `constraint.py` (_efc_row:53–121 — solref/solimp); `solver.py` (CG/NEWTON contexts, line-search); `types.py` (SolverType:443, PGS unsupported); `island.py` (contact islands); `io.py` (nmaxpyramid:301, friction checks:227, impratio:245); `collision_{gjk,convex,primitive,sdf,flex}.py`.
- MuJoCo docs: https://mujoco.readthedocs.io/en/stable/computation/index.html (constraint model / soft constraints / friction cone, condim, Newton solver); https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters (solref/solimp equations, negative convention); https://mujoco.readthedocs.io/en/stable/XMLreference.html (contype/conaffinity defaults, `<pair>`, `<exclude>`, priority/solmix). MuJoCo classic `mujoco/src/engine/engine_collision_driver.c`: `mjCOLLISIONFUNC` table (referenced by warp comment) + **`mj_contactParam` ~L2850–2930 (VERIFIED verbatim: friction `mju_max`, condim `mjMAX`, solmix-weighted solref/solimp mix, direct-mode `mju_min` branch, margin/gap max)** — fetched via raw.githubusercontent.com.
- Project: `docs/plans/2026-06-03-v08-unified-collision-contact.md` (Nuka ratified design mapped against).
