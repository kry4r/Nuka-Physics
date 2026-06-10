# H1 Whole-Body RL Stand+Grasp — Controller Spec (SG)

> **Status:** controller-authored spec, 2026-06-10. Owner directive: design from code
> recon, ignore prior route docs, no step-by-step plan — this spec is the binding
> constraint set. Execution is free within it.
>
> **Goal:** an RL-TRAINED H1 that stands (floating base) on the ground, reaches a
> ~10.9 cm cup resting on a table, closes an HONEST force-closure grasp, lifts, and
> places it — trained end-to-end in the ONE general batched world.

---

## 1. Anchors (code reality this spec builds on)

- **The N=1 whole-body reference EXISTS:** `UnifiedCoResidentStepper::StepStandGrasp`
  emits the UNION every step — foot×ground + finger×cup + cup×table (flat-bottom box
  proxy, `cup_table_proxy_*`), floating base, one per-link drive vector
  (`src/runtime/coresident/unified_coresident_stepper.hpp:262-314`).
- **Env-major batching generalizes to the H1 articulation byte-exactly:** H1.2
  (`tests/coresident/test_batched_h1_hand_grasp.cpp`, Gate 1 tol-0, Gate 2 N=8).
- **Cup size LOCKED ~10.9 cm** (H1.1b narrow feasible band); an honest active grasp
  exists for the cooked H1 hand with the 30-sphere finger-only wrap (H1.1).
- **PPO learns grasp on this engine** (A5b, synthetic gripper, catch 0→1.00, eval
  observer separate from reward).
- **LANDMINE (found in this recon): 18-DOF SILENT TRUNCATION.**
  `kMaxFactorDof=18` clamps (`articulation_contacts.cu:451-452`,
  `float a[kMaxFactorDof²]`), `kMaxContactSolverDof=18` caps `dof_to_link` loops
  (`articulation_contacts.cu:700-712, 1022-1040`; `articulation_contacts.hpp:294`).
  A ~51-DOF whole-body H1 silently loses M⁻¹/J coupling beyond DOF 18 on any path
  that caps. Whole-body contact work is DISHONEST until this is closed.

## 2. Gates (deliverables; each gate is a test or a measured report)

### G0 — DOF honesty (engine precondition)
No silent DOF truncation anywhere on the coresident/batched whole-body path:
M assembly/factorization, chain-J, contact effective mass, implicit joint damping,
solve working vectors, qdot scatter, action injection.
- **Gate test:** a ≥51-DOF articulation (full-free cooked H1) where a FINGER-chain
  contact's chain-J / m_eff / solve result matches a host-side oracle computation;
  fails RED on current HEAD (proving the truncation), GREEN after the fix.
- Truncation beyond the supported max becomes a LOUD failure (construction-time
  error), never a clamp.
- All existing ≤18-DOF tests/goldens stay byte-identical.

### G1 — Batched union world
`BatchedUnifiedWorld` steps the UNION scene per env: floating-base H1 + movable cup
+ static ground + static table; pairs foot-sphere×ground-plane, fingertip-sphere×cup-hull,
cup(-proxy)×table-plane — mirroring the oracle's StepStandGrasp emission.
- **Gate (a) parity:** N=1 byte-exact (tol 0 target; ≤1e-5 only with a written
  justification) vs the StepStandGrasp oracle on the IDENTICAL scene, ≥300 steps,
  contact-rich (feet loaded, fingers closing, cup on table then lifted).
- **Gate (b) independence:** N=8 mixed-action cross-env test (H1.2 Gate-2 pattern).
- **Gate (c) determinism:** two-run byte-identity (D1).
- **Gate (d) trainability:** honest throughput lower bound at N∈{32,256,1024};
  bar: one PPO stage (≈100M env-steps) must fit <24 h wall-clock on this GPU,
  else a NAMED optimization increment lands BEFORE training starts.

### G2 — Python substrate
Nanobind exposure of the union world (template-constructed: cooked H1 + cup + table).
- `SetActions` over the full dof_stride; obs export: q/qdot (env-major), base
  pose/vel, cup pose/vel, fingertip world positions, per-finger normal impulse,
  foot contact state, last action.
- **Gates:** python N=4 rollout matches the C++ parity test trajectory; seeded reset
  deterministic; 10k random-action steps with zero NaN/inf.

### G3 — Staged RL training (ONE env class, config-gated stages)
Each stage's policy warm-starts the next. Success metrics come from a SEPARATE
deterministic evaluator (A5b pattern) — never from the reward.
- **S1 reach+grasp+hold** (fixed base; cup on table; arm+hand actions).
  Gate: hold-rate ≥0.8 over ≥256 eval envs (hold = cup off table support, finger
  vertical impulse ≥0.8·m·g·dt sustained N steps); BITE: zero grip torque → cup
  falls in ≥0.9 of held cases.
- **S2 +lift** (fixed base). Gate: lift success ≥0.8 (cup ≥15 cm above table,
  held ≥1 s, tilt <30°).
- **S3 floating-base stand** in the SAME world (legs driven, arms PD-held, no cup
  requirement). Gate: ≥0.9 envs stand ≥5 s within base height/tilt bounds,
  feet-only support.
- **S4 whole-body co-train** (stand AND reach+grasp+lift). Gate: combined success
  ≥0.5 (floor) / 0.8 (stretch): standing bounds held throughout + S2 lift criteria
  + BITE.
- **S5 +place**: carry to a target zone on the table, set down, release, hand
  clears, cup at rest in zone. Gate: ≥0.5 (floor) / 0.8 (stretch).

## 3. Honesty bars (apply to every gate)
- No cup scripting/teleporting/parenting, no gravity-off, no contact-force fakery;
  the hold is friction force-closure only.
- BITE is a standing eval, not a one-off.
- D1 (two-run byte-identity) on all engine paths; parity vs oracle for engine work.
- Stand thresholds are not loosened vs prior stand gates without written justification.
- A RED gate is a FINDING to report, not a number to tune away.
- Throughput numbers are measured lower bounds, never extrapolations.

## 4. Design decisions (controller's; binding unless owner overrides)
- ONE world: extend `BatchedUnifiedWorld` with the union mode. No new specialized
  world. Legacy `BatchedArticulatedWorld` / `world_stepper` untouched.
- Contact stays on PROVEN handlers: sphere×plane (feet, fingertips vs ground/table),
  sphere×hull (fingertip×cup, the EPA-bypass path), proxy-box/hull×plane (cup×table).
  NO hull×box, NO mesh narrowphase, NO new EPA exposure.
- Feet = authored ankle contact spheres in config (fingertip pattern). No mesh feet.
- ONE env class with stage configs (base cook override fixed/floating, action mask,
  reward stage, IC randomization) — not one env per stage.
- Action = per-DOF torque, clamped to per-joint real limits; physics dt 1/240 with
  a measured control decimation.
- Privileged-state obs only (no vision/VLA in this spec).
- Dense reward shaping allowed and expected; eval metrics stay binary success rates.

## 5. Exclusions
No VLA/vision, no renderer/demo-video work, no kitchen-scene import into the
training world (training table = plane + proxy; the composed kitchen scene stays a
demo-time concern), no C-ABI rework, no golden regeneration, no legacy-world features.

## 6. Increment discipline
TDD per increment; additive TUs preferred; production edits confined to
`src/runtime/coresident/*` + the articulation kernels G0 names; the full existing
suite stays green after every increment; one commit per increment with `[skip ci]`;
push per the established proxy procedure (deferred-batch fallback when the proxy is
down).
