# Nuka Engine — Honest Assessment + Roadmap to a General, Trainable World

**Date:** 2026-06-09
**Author:** controller (Claude)
**Trigger:** Owner pushed back on the claim "rigid-body simulation is done" and on the engine's
inability to "import a physics scene and just simulate it." Owner set the architectural
principle: **RL training MUST run in the GENERAL world, never a task-specialized world.**
This document is the honest assessment + roadmap requested before any further coding.

> Status of facts below: items marked **[verified]** were read directly from source this session
> (file:line). Items marked **[mapping]** are being confirmed by an in-progress code map and will
> be tightened. Nothing here is optimistic hand-waving — where the engine falls short of "general
> engine," it says so plainly.

---

## 1. Honest state — validated COMPONENTS, not a general ENGINE

The earlier phrasing "rigid-body sim is done" was true at the **component** level and **overclaimed**
at the **system** level. Precisely:

### ✅ Built + validated (component level)
- **Articulation dynamics** — Featherstone ABA, floating + fixed base, control modes
  (torque / PD / velocity / computed-torque / OSC / actuator). GPU. **[verified]** (used by
  standing RL + co-resident stepper).
- **Rigid bodies** — `runtime::rigid::BodyState` (inv_mass, position, orientation, linear/angular
  velocity, inv_inertia, force, torque). **[verified]** `src/runtime/rigid/body_state.hpp:11`.
- **Collision (unified spine)** — broadphase + narrowphase: sphere×sphere, sphere×box,
  sphere×plane, sphere×convexhull, GJK/EPA for convex pairs. **[verified]**
  `src/collision/narrowphase_dispatch.hpp` (dispatch table @625+). GPU broadphase
  `src/collision/gpu/broadphase.cu`. **No cylinder primitive** (irrelevant — model cup as box/hull).
- **Two-way contact solve** — `UnifiedSolve` (class-blind: ArticulationChainJ ↔ RigidInvMass ↔
  StaticNull), GPU row solver with the RigidInvMass two-way arm. **[verified]**
  `src/solver/unified_solve.hpp`, `src/solver/gpu/row_solver.cu` (ValidBody/body_count, RigidInvMass
  apply @316/358). This is the real grasp physics: it computes a movable cup gripped two-way by the
  articulated hand + resting on a table.
- **The grasp itself works** — `UnifiedCoResidentStepper::StepGrasp/StepStandGrasp` simulate the H1
  hand + movable cup hull + table with two-way friction; the dense-grasp tests pass (fixed-base
  force-closure lift holds the cup by friction alone, BITE grip-off→falls). **[verified]** (read +
  re-ran this session).
- **GPU batching for locomotion/standing** — `BatchedArticulatedWorld` runs 4096 parallel H1/Go2 with
  articulation + **foot↔ground** contact; this is the RL training substrate today. **[verified]**
  `src/runtime/gpu/batched_articulated_world.{hpp,cu}`.
- **RL stack** — rl_games PPO, nanobind binding (`_nuka_ext`/`nuka`), zero-copy DLPack buffers,
  H1StandEnv, policy→C++ bridge export. Standing policy trained + converged. **[verified via map]**.
- **Diff-sim** (v0.5) and **RT renderer** (offline path-tracer) exist.
- **Importers** — MJCF + USD (geometry; SceneIR compose layer). **[mapping — exact dynamic-body
  coverage being confirmed]**.

### ❌ NOT built (the general-engine gaps — owner is right)
1. **No "load a scene file (robot + objects + table) → simulate" general entry.** Every task is
   **hand-wired in C++**. The grasp scene is assembled by hand in test code, not loaded from a scene
   and stepped generically. **[verified — co-resident grasp is hand-assembled in tests]**.
2. **Two divergent contact/stepping paths that never merged:**
   - **General path** (`UnifiedSolve` + unified collision + `BodyState`) — does articulation ↔ rigid
     ↔ static two-way contact (the grasp). **Single-world, host-orchestrated. NOT batched.**
   - **Specialized batched path** (`BatchedArticulatedWorld`) — multi-env GPU, **foot↔ground only, no
     movable rigid body, no `BodyState`.** Used by RL.
   The grasp physics lives in the path RL can't use; the RL path can't do grasping. This is exactly
   the project's tracked **"unified spine validated-not-wired into the batched steppers"** debt.
3. **No movable rigid body in the batched/RL path.** **[verified]** (`grep BodyState` in
   `src/runtime/gpu/` is empty).
4. **No perception** (RGB/raycast sensors, image encoding) for vision-based RL. (Owner: future; my
   design responsibility.)

**Bottom line:** the physics *components* (incl. movable rigid + two-way grasp contact) are real and
validated. What does **not** exist is a **general, scene-driven, batched world** you can import a
scene into and train RL on. The owner's critique stands; "engine done" was wrong at the system level.

---

## 2. The architectural decision (owner-set)

> **Training happens in the GENERAL world, not a specialized one.**

This **rules out** the tempting shortcut of bolting a cup onto `BatchedArticulatedWorld` (the
standing-specialized path). Instead the target is **one** world that is simultaneously:

- **General** — imports a scene (articulations + free rigid bodies + static colliders) and simulates
  it via the unified collision + `UnifiedSolve` two-way contact (no per-task hand-wiring).
- **Batched** — N parallel envs on GPU (for RL throughput), determinism D1 preserved.
- **Differentiable** — keep the diff-sim pillar.
- **Trainable** — exposed to python (nanobind, zero-copy buffers) so rl_games PPO trains *in it*.
- **Renderable** — feeds the RT/raster renderer for the demo video.

The grasp demo (H1 + cup + table) becomes just one scene this general world runs — and RL grasping
is trained in that same world.

---

## 3. Roadmap (general world first, then RL grasp, then perception)

Sequenced so each phase is independently testable and de-risks the next. Effort tags are relative
(S/M/L), grounded in the reuse analysis (§4).

- **Phase 0 — this assessment + roadmap.** (done on owner sign-off.)

- **Phase 1 — General, scene-driven SINGLE-INSTANCE world. [M]**
  Generalize the co-resident stepper into a *scene-driven* general world: load a scene description
  (H1 articulation + cup rigid body [box or real hull] + table/ground static) → step via unified
  collision + `UnifiedSolve` (the proven two-way path), no hand-wiring. **Deliverable:** "import the
  cup scene → simulate the grasp" works as a general API, not test-hand-assembly. This *directly*
  answers "import a scene and simulate." Reuses ~everything; mostly wiring + a scene→bodies builder.

- **Phase 2 — BATCH the general world (the hard core). [L]**
  Make unified collision + `UnifiedSolve` run per-env across N envs on GPU. Reuse the existing GPU
  broadphase (`broadphase.cu`) + GPU row solver (`row_solver.cu`); the work is per-env batching of
  narrowphase + row assembly + the solve, with D1 determinism. This is the real engineering and the
  main risk (performance + determinism at scale). **Deliverable:** thousands of parallel general
  worlds (each = H1 + cup + table) stepping deterministically.

- **Phase 3 — General RL env on the batched general world. [M]**
  `H1GraspEnv` (mirrors H1StandEnv): obs = robot state + cup pose/vel + relative; action = arm+hand
  (+ optionally legs) torques; reward = approach + grip (finger contact impulse) + lift height + cup
  stability − regularizers; BITE-style termination (drop grip → cup falls). Register with rl_games.
  Reuses the PPO stack + bindings; add per-env cup-state buffers to python (zero-copy).

- **Phase 4 — Train RL grasping in the general world. [M, compute-bound]**
  Cup at **8–9 cm** (owner-approved, easier than 10.9 cm). Start fixed-base or with the trained
  standing policy holding balance; let RL discover the wall-grip the hand-tuning could not. Export
  the policy; **validate sim2sim** on the single-instance general world (Phase 1). Render the demo.

- **Phase 5 — Perception (future, owner flagged). [L]**
  Add RGB camera + raycast sensors to the general world; design the encoder (CNN for RGB / a
  raycast-set encoder), feed into the RL policy. State-based (Phases 3–4) first, vision after.

**Deferred (owner):** "python must also run locally; C++ vs python comparison" — revisit later.

---

## 4. Reuse vs build (grounds the effort estimate)

| Capability | State | Roadmap use |
|---|---|---|
| Articulation dynamics (ABA, GPU) | ✅ ready, batched | reuse |
| Rigid `BodyState` | ✅ ready (single-world) | reuse; replicate per-env in Phase 2 |
| Collision narrowphase (sphere×box/hull/plane, GJK/EPA) | ✅ ready (single-world) | reuse; batch in Phase 2 |
| GPU broadphase | ✅ exists (`broadphase.cu`) | reuse; confirm batched |
| `UnifiedSolve` two-way (artic↔rigid↔static) | ✅ ready, **single-world host-orchestrated** | reuse logic; **batch in Phase 2 (the core build)** |
| GPU row solver (`row_solver.cu`, RigidInvMass arm) | ✅ exists | reuse; confirm per-env |
| Scene import (MJCF/USD/SceneIR) | partial [mapping] | extend to emit dynamic bodies for Phase 1 |
| Single-instance grasp (co-resident) | ✅ works, hand-wired | generalize → Phase 1 scene-driven |
| Batched world (foot-ground) | ✅ works, **specialized** | **NOT extended** (owner: no specialized world); the general batched world supersedes it for grasp |
| RL stack (rl_games, nanobind, export) | ✅ ready | reuse; new env + buffers |
| diff-sim, RT renderer | ✅ exists | reuse for diff + demo video |

**Net:** the physics and RL machinery is largely **reusable**; the singular hard build is **Phase 2
(batching the general contact path)**. No new rigid-body physics, no specialized primitive.

---

## 5. Risks & honesty bars
- **Phase 2 perf/determinism at scale** is the main risk (variable contact counts per env, D1).
- **Grasp learnability**: the physics supports a real wall-grip (proven fixed-base); RL must find it.
  Honesty bars from the demo spec hold: real friction force-closure, BITE proof, no
  teleport/parent/gravity-zero/cup-scripting, no loosening standing thresholds.
- **Scope honesty**: this is a multi-week effort dominated by Phase 2. RL "finds the grasp" faster
  than hand-tuning, but the total time is gated by building the general batched world.

---

## 6. Sequencing — DECIDED

**Owner decision (2026-06-09): go straight to Phase 2 ("直奔 P2"). Skip the standalone Phase 1.**

Rationale this is safe to do: the **single-instance oracle that Phase 1 would have produced already
exists** in the validated `UnifiedCoResidentStepper` (StepGrasp/StepStandGrasp) — it runs the H1 hand
+ movable cup hull + table through unified collision + `UnifiedSolve` and the dense-grasp tests pass.
So the batched general world (Phase 2) has a ready reference to validate against per-env; we do not
need a separate Phase-1 deliverable to get that oracle.

What "Phase 1" still contributes is folded INTO Phase 2 as its first increment: a **scene → bodies
builder** (emit articulation + free rigid bodies + static colliders from a scene description) so the
batched world is populated generically, not hand-wired. The "import a scene → simulate" capability is
delivered by the batched general world itself rather than a single-instance precursor.

### Phase 2 increment decomposition (each independently testable, validated vs. the co-resident oracle)
1. **P2.1 — Per-env movable rigid body.** Add a per-env `BodyState` array (the cup) to the batched
   world; integrate free-body dynamics per env; **no contact yet** (free-fall / ballistic check vs.
   analytic). Deterministic (D1).
2. **P2.2 — Batched cup↔static contact.** Batched narrowphase for cup↔ground/table (sphere/hull ×
   plane) + cup rows into per-env row assembly + GPU row solver RigidInvMass arm. Validate: cup rests
   on table per env, matches co-resident single-instance.
3. **P2.3 — Batched articulation↔rigid contact (the grasp crux).** Batched finger(sphere)↔cup(hull)
   two-way rows (ArticulationChainJ ↔ RigidInvMass). Validate against the co-resident fixed-base
   force-closure lift (friction-only hold; BITE grip-off → fall) replicated across envs.
4. **P2.4 — Scene → bodies builder + full grasp scene batched.** Generic scene population (H1 + cup +
   table) across N envs; determinism + perf at scale (the main risk). This is the "import scene →
   simulate, batched" deliverable.

Phases 3–5 (H1GraspEnv, train RL grasp at 8–9 cm, perception) proceed on top of the P2 batched world
unchanged from §3.

---

## 7. P2 pre-build verification (DONE 2026-06-09) — grounds the batching architecture

Before writing any harness, three load-bearing seams were read directly (advisor-directed: the batching
story rested on the rigid arm, but the grasp is the articulation arm). Findings:

**(a) The coloring ALREADY accounts for the articulation side via a synthetic-body-index convention —
NO solver fix needed.** (This CORRECTS a first-pass conclusion: reading `RowsConflict` in isolation
suggested a race; reading the *emitter* showed the codebase already prevents it.) The GPU solve is a
single-block multi-lane graph-colored PGS (`row_solver.cu:1291` `SolveRowsSweepKernel<<<1,kBlockSize>>>`),
same-color rows on parallel lanes, articulation apply `qdot[r]+=M⁻¹Jᵀδ` non-atomic
(`reaction_provider.hpp:203`). `RowsConflict` (`row_scheduler.cu:14`) keys on the raw row body indices
`body_a/body_b`. **The emitter stuffs the articulation side's body index with the synthetic key
`body_count + art_index`** (`unified_coresident_stepper.cpp:564-566`; comment lines 526-529: *"The
articulation key (body_count+art_index) likewise serializes all foot rows on the shared base DOFs"*).
`ValidBody(body_count+art_index, body_count)` is false → the rigid arm ignores it (the articulation
arm handles it via `art_refs`), but `RowsConflict` sees the shared synthetic index → same-articulation
rows CONFLICT → different colors → serialized → **no race**. The existing test
`LinkRigidTwoWay.TwoContactsSameArticulationSerializeAndAreConsistent`
(`tests/solver/test_link_rigid_twoway.cpp:273`) asserts exactly this (`ColorCount ≥ 2`, kernel result
== sequential Gauss-Seidel). The `row_solver.cu:398` "hides at equilibrium" comment is about
`ComputeJv` dropping articulation sides (the reason the compliant third arm
`CompliantSideConstraintVelocity` exists), NOT a coloring race.
  ⇒ **The convention generalizes to cross-env batching with ZERO solver change:** env `e` → rigid
  bodies at `[e·k, (e+1)·k)`, articulation synthetic key `total_body_count + e`, `art_index = e`,
  M⁻¹/qdot tiles env-major (`art_index·dof_stride²` / `art_index·dof_stride`). Cross-env rows share no
  index (rigid or synthetic) → never conflict → parallelize across envs; within-env same-articulation
  rows share `total_body_count + e` → serialize (correct). **P2 = pure plumbing of concatenated
  per-env buffers on top of the UNMODIFIED, validated `UnifiedSolve`.**

**(a′) The REAL P2 throughput risk is the single-BLOCK kernel, not the coloring.**
`SolveRowsSweepKernel<<<1, kBlockSize>>>` runs in ONE block (≤1024 threads) striding over ALL
concatenated rows. Correct at any N, but for thousands of envs × dozens of rows it is a serial
bottleneck. ⇒ correctness milestones (P2.2–P2.4 at modest N) use it as-is; **RL-scale throughput
(P2.5) needs a multi-block batched kernel variant** (e.g. one block per env / per color-group). That
— plus batched broadphase/narrowphase — is the actual "hard core," and matches the 4–6 week estimate.

**(b) The articulation reaction context is env-major BY DESIGN.** `RowArticulationRefs.art_index`
keys the per-articulation M⁻¹ tile + qdot slice; layout is "env-major, matching
BatchedArticulatedWorld" (`row_articulation_refs.hpp:18-25`). ⇒ per-env replication = give env `e`
`art_index=e`; the co-resident assembly (`unified_coresident_stepper.cpp:864+`) is the per-env builder
to generalize. No inherently single-world assumption found.

**(c) The batched dynamics kernels are a shared component.** `FeatherstoneAba` is a static-method
class in `nuka::runtime::articulation` (`featherstone_aba.hpp`); `BatchedArticulatedWorld` merely
CALLS it (`ComputeAccelerations`/`IntegrateVelocity`/`IntegrateFloatingBase*`). ⇒ the new additive
batched general stepper calls the same methods — no edit to the owner-protected `batched_articulated_world.*`.

### Increment order (no solver change — pure plumbing; each validated vs the co-resident N=1 oracle)
- **P2.1 — New additive batched general stepper skeleton + per-env movable rigid body** (free-fall,
  no contact). New TU (`BatchedUnifiedWorld`), calls shared `FeatherstoneAba`; per-env `BodyState`
  SoA; D1 check vs analytic free-fall. Establishes the per-env-offset buffer layout.
- **P2.2 — Batched rigid↔static contact** (cup↔ground/table): per-env narrowphase → concatenated rows
  with per-env body-id offsets → single `UnifiedSolve`; validate cup-rests-on-table per env == co-resident N=1.
- **P2.3 — Batched articulation↔rigid contact (the grasp crux):** per-env finger(sphere)↔cup(hull)
  two-way rows with the synthetic articulation key `total_body_count + e` + `art_index=e` + env-major
  M⁻¹/qdot; validate vs co-resident fixed-base force-closure replicated per env (friction-only hold;
  BITE grip-off → fall). Correct by construction (convention (a)); no solver edit.
- **P2.4 — Scene→bodies builder + full grasp scene batched + throughput.** Generic scene population
  (H1 + cup + table) across N envs = the "import scene → simulate, batched" deliverable. THEN the
  throughput work: GPU per-env broadphase/narrowphase + the multi-block batched solver kernel ((a′)).
  NB: a host-side per-env narrowphase + single-block solve is a **correctness oracle only** (O(N) host
  work, far below RL throughput) — must NOT be mistaken for "done" at small N.
