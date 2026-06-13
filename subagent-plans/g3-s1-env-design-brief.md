# G3 / Stage S1 — Reach + Grasp + Hold (fixed base): Env Design Brief

> READ-ONLY design deliverable for spec `docs/specs/2026-06-10-h1-whole-body-rl-grasp-spec.md`
> (§G2, §G3 S1, §3 honesty bars, §4 design decisions). Grounds: the H1.1/H1.1b
> feasibility probe (`tests/coresident/test_h1_grasp_feasibility_probe.cpp`), the A3
> synthetic-gripper env (`python/nuka/tasks/h1_grasp*.py`), the A5b eval observer
> (`examples/training/grasp_catch_eval.py`), the stand env (`python/nuka/tasks/h1_stand.py`),
> the go2 curriculum (`python/nuka/tasks/go2_locomotion.py`), `vecenv.py`, the PPO
> cfg/scripts (`examples/training/h1_grasp_ppo_cfg.yaml` + `train_h1_grasp_ppo.py`),
> and the H1 articulation (`.nuka-assets/newton_assets/unitree_h1/mjcf/h1_with_hand.xml`).
>
> S1 is trained on the **G1 batched union world + G2 nanobind binding** (built in
> parallel). This brief is written against that obs/action surface (spec §G2), NOT the
> legacy synthetic `nuka.GraspWorld`. Where I name a binding method I follow the
> existing `GraspWorld` naming (`export_obs`/`read_cup`/`set_actions`/`reset_envs`)
> as the precedent the G2 builder will most plausibly mirror; the exact names are G2's
> call — this brief depends only on the *signals* §G2 pins.

---

## 0. Ground truth that shapes every decision

**The feasible grasp is a thin envelope** (H1.1/H1.1b, hard-asserted at 10.9 cm only):
- ONE feasible cup size: 1.8× ≈ **10.9 cm** diameter. 8.5/9.7 cm have NO shallow
  finger-only caging placement; 12.1/13.3 cm reported (not asserted) and the
  envelope *narrows* going up (fixed finger arc covers a smaller angular fraction).
- Seating is **≤2 mm shallow** (`kShallowPenMax=0.002`); coverage arc >200° but the
  GO is an **opposed 2-sided active wrap with a ~142° live escape gap**, NOT a >180°
  geometric cage. The hold is friction force-closure that **requires active squeeze**
  (`kCloseOffset=0.18` curl-beyond) — BITE: grip→0 drops ~1.1 m. There is no passive
  wedge to fall back on.
- ONE validated pre-pose: `CurlForScale(1.8)` (finger_prox≈0.45, finger_int≈0.55,
  thumb yaw 1.0 / pitch 0.5 / int≈0.42 / dist≈0.42) at the
  `BestPlacementFingerOnlyShallow` cup XY. Contact set = the **30-sphere finger-only
  wrap** (4 fingers × prox+int × 3 spheres + thumb int+dist × 3), 6 mm radius, NO palm.
- Drive law in the probe: PD Kp=4 / Kd=0.4 / offset=0.18, faithful armature 0.1 /
  damping 1.0, cup mass 0.2 kg, μ=0.8, dt 1/240.

**Implication for RL.** The A5b no-shaping success was on a synthetic 2-DOF timed-catch
task and **will NOT transfer**: that task's discriminativeness came from a *timing IC*
(cup descending; a const-max slam misses) over a 1-D actuated axis. S1 is a *static*
cup the policy must *reach* in 3-D and seat to ≤2 mm precision in a ~142°-gap wrap.
**Dense reward shaping is required**, and a **curriculum that hands the policy the thin
envelope on a plate at first, then widens** is the heart of the design (§2).

**The 18-DOF landmine (spec G0).** `kMaxFactorDof=18` / `kMaxContactSolverDof=18`
silently truncate M⁻¹/J coupling beyond DOF 18 on any capping path. **S1's action DOF
budget must stay ≤18 until G0 lands** — this is the dominant constraint on §1's scope
decision, not just a footnote.

---

## 1. DOF / action scope for S1  →  **fixed-base H1, non-arm joints Fixed at cook; actions = right arm (5) + right hand (12) = 17 DOF**

### Joint inventory (counted from `h1_with_hand.xml`, 46 `<joint>` total)
| Group | Joints | Names | MJCF actuator force limit (ctrlrange, N·m) |
|---|---|---|---|
| Legs (L+R) | 10 | hip_yaw/roll/pitch, knee, ankle ×2 | 200/200/200/300/40 per leg |
| Torso | 1 | `torso_joint` | 200 |
| Left arm | 5 | l_shoulder pitch/roll/yaw, l_elbow, `left_hand_joint` (wrist) | 40/40/18/18 ; wrist 6 |
| **Right arm** | **5** | `right_shoulder_pitch/roll/yaw_joint`, `right_elbow_joint`, `right_hand_joint` (wrist) | **40 / 40 / 18 / 18 ; wrist 6** |
| Left hand | 12 | L_thumb yaw/pitch/int/dist + {index,middle,ring,pinky}×{prox,int} | ±1 each |
| **Right hand** | **12** | `R_thumb_proximal_yaw/pitch_joint`, `R_thumb_intermediate/distal_joint`, `R_{index,middle,ring,pinky}_{proximal,intermediate}_joint` | **±1 each** |

Right-arm joint ranges (rad), for IC/limit clamping: shoulder_pitch [-2.87,2.87],
shoulder_roll [-3.11,0.34], shoulder_yaw [-4.45,1.3], elbow [-1.25,2.61],
right_hand (wrist roll) [-3.05,3.05]. Right-hand fingers: thumb_yaw [-0.1,1.3],
thumb_pitch [-0.1,0.6], thumb_int [0,0.8], thumb_dist [0,1.2], all
finger prox/int [0,1.7].

### Recommendation: **fixed base, only right arm + right hand are driven DOF (17); all of
legs/torso/left-arm/left-hand cooked `Fixed`.**

- **dof_stride both ways** (fixed base → base_dof=0; `dof_stride` = Σ driven DOF):
  - *Recommended (selective-Fixed cook):* Fixed base, everything except right arm+hand
    Fixed → **dof_stride = 17**. This is the exact `LoadH1Fixed(free_finger_bodies, …)`
    pattern the probe already uses (it Fixes every link whose body isn't in the
    driven-free set), generalized from "hand only" to "right arm + right hand".
  - *Full-free with action masking:* all 46 joints live, base Fixed → **dof_stride = 46**
    (legs 10 + torso 1 + both arms 10 + both hands 24 + 1). The 29 non-action DOF would
    be PD-held at their cook pose.

- **Why selective-Fixed, decisively:**
  1. **G0 landmine.** dof_stride=46 **exceeds the 18-DOF cap and is dishonest until G0
     lands**. dof_stride=17 is **at/under 18 by one** — S1 can train on current HEAD
     even if G0 slips, and a finger-chain contact's chain-J / m_eff stays inside the
     supported band. This alone settles it: full-free is *blocked* on G0; selective-Fixed
     is not. (S4's full-free world is what G0 *exists for*; don't pull that dependency
     into S1.)
  2. **Throughput.** Per-env M factorization is O(dof_stride²) in the env-major tiles
     (`art_index*dof_stride²`); 17²≈289 vs 46²≈2116 — ~7× less M⁻¹ work per env, plus
     a smaller chain-J gather. Matters directly to the spec G1(d) <24 h/stage bar.
  3. **Re-cook churn is a non-issue.** The probe + H1.2 already cook a fixed-base
     selective-Fixed H1; widening the free set from "hand" to "arm+hand" is a one-line
     change to the free-body list, not new machinery. S4 re-cooks to floating base
     *anyway* (different root joint), so a fixed→floating re-cook is unavoidable at the
     stage boundary regardless of S1's DOF choice — keeping S1 small costs nothing later.
  4. **Honesty.** Driving the left arm / legs in a fixed-base reach task is pure
     reward-surface noise the policy must learn to null; cooking them Fixed removes a
     whole class of exploits (e.g. flailing the left arm for an action-rate artifact).

- **Decimation:** physics dt 1/240 is locked (§4). Control decimation is "to be
  measured" (§G2). **Recommend starting at decimation = 2 (control dt = 1/120 ≈ 8.3 ms)**
  and treating it as the first measured knob: the probe runs the PD close at the full
  240 Hz, and the ≤2 mm seating envelope is sensitive to per-step penetration, so do NOT
  start coarser than 2. Episode length is specified in §4 below in *control* steps so it
  is decimation-invariant.

- **Wrist note.** `right_hand_joint` (the ±6 N·m wrist roll) is grouped into the arm
  here because it orients the palm into the wrap and the probe's `CurlForScale` does not
  touch it. It is a *driven* action DOF (the policy gets wrist roll for free), but its IC
  is seeded from the validated pre-pose's wrist angle (0 in the probe's `ApplyCurl`).

---

## 2. IC randomization + curriculum  (THE HEART OF THE BRIEF)

The thin envelope (≤2 mm seat, ONE size, ONE pre-pose, ~142° gap) means a from-scratch
"open hand, random arm, find the cup" task has **effectively zero reward gradient at
cold start** — the analog of the go2 PPO-collapse the locomotion env documents
(`go2_locomotion.py`: a metastable IC gives no climbable floor → value head collapses).
The design mirrors go2's two proven levers: **(a) seed the IC at the known-good
configuration** (go2's exact-default-pose teleport), and **(b) a success-gated
curriculum that widens the distribution only as the policy earns it** (go2's
`fixed_command` → broaden).

### Recommended IC + curriculum: **a 3-phase, success-rate-gated schedule on ONE env
class via a `stage_config`.** All three phases share S1's reward/eval; only the IC
sampler + which DOF the policy must move changes.

**Pre-pose seeding (all phases).** Seed the right hand at `CurlForScale(1.8)` and the
right arm at an analytically-solved *approach pose* that places the wrap cavity centroid
at the validated `BestPlacementFingerOnlyShallow` cup XY (compute once at cook from the
probe's `CavityCenterZ` / `BestPlacement` helpers; bake into the stage config as
`q_init_arm`, `q_init_hand`, `cup_xy0`). This is the privileged-state analog of go2's
"reset to the exact default" — it is honest (it's an IC, not scripting; the policy still
has to *act* to hold), and it is the only way to put the policy inside the envelope's
basin of attraction.

**Phase C0 — close-only (hand DOF active, arm Fixed-at-pose).**
- IC: hand pre-posed *just short of* the wrap (curl scaled to ≈0.7× the close offset so
  fingers are open ~few mm off the cup), arm pinned at `q_init_arm`, cup at `cup_xy0`,
  **zero jitter**.
- Action mask: only the 12 right-hand DOF are live; arm torques forced to a PD hold of
  `q_init_arm`. (Implemented as an action-mask vector in the stage config, NOT a re-cook
  — the 5 arm DOF still exist in dof_stride=17, they're just PD-servoed to the pre-pose
  and excluded from the policy's action slice.)
- Purpose: prove the policy can learn the **squeeze-into-force-closure** sub-skill in the
  envelope with no reach burden — the direct analog of the A5b "does PPO learn the
  grasp" de-risk, but on the *real H1 hand*. Graduation gate: hold-rate ≥0.8 on the C0
  IC (eval, §5).

**Phase C1 — approach + close (arm + hand active, tight jitter).**
- IC: hand pre-posed OPEN (curl ≈ 0, fingers extended), arm at a *retracted* pose
  ~8–12 cm back along the approach axis from `q_init_arm`, cup at `cup_xy0` with **tight
  jitter**.
- Cup jitter geometry (reach-aware): the right hand approaches roughly along +X/−Y of
  the torso. Jitter the cup **on the table plane (X,Y), zero Z** (the cup *rests* on the
  table — Z is determined by the table, jittering it would be dishonest teleporting).
  Start at **±1 cm box on X and Y** (smaller than the A3 ±1.5 cm because here the policy
  must also *reach*, and the ≤2 mm seat tolerance punishes XY error far harder than the
  1-D synthetic task). Add a small arm-start jitter (±0.05 rad per arm joint) so the
  policy doesn't memorize one trajectory.
- Action mask: all 17 DOF live.
- Purpose: learn reach + pre-shape + seat + squeeze end-to-end.

**Phase C2 — widen (curriculum-tightened jitter on a success schedule).**
- Same as C1 but the cup-XY jitter half-box grows on a **success-rate schedule**
  (go2's curriculum pattern, but on jitter radius instead of command range): every K
  eval cycles, if eval hold-rate ≥0.8, multiply the jitter half-box by 1.25 (cap at the
  reachable workspace — measure it; the A3 finding that *X-extreme cups land past the
  fixed reach* applies here too, so the cap is a hard reach limit, not a free parameter).
  If hold-rate drops <0.6, shrink ×0.8. The **gate metric for S1 is hold-rate ≥0.8 at the
  C1 jitter level** (the spec's ≥256-env bar); C2 widening is "bank the margin", reported
  but not gate-blocking.

**Why staged close-only → approach+close (not the reverse, not all-at-once).** The
≤2 mm / 142°-gap envelope is the hard part; reach is comparatively forgiving. Learning
the squeeze first (C0) gives the warm-start a *force-closure prior* before the reach
problem is added, exactly as go2 pins the command to hand the policy a climbable floor
before broadening. Starting from a fully-open hand at a random arm pose with no warm
start is the configuration most likely to reproduce the go2 collapse.

**Open-vs-pre-shaped at C1.** Start C1 from an OPEN hand (not pre-curled): if the hand
is pre-curled the policy can learn a degenerate "don't move the fingers, just translate
the arm until the cup falls into the static curl" that is brittle to jitter and will fail
S2's lift. Open-hand-at-approach forces it to learn the pre-shape→seat→squeeze sequence.
C0's pre-curl is acceptable *only* because C0's arm is pinned (no translate-into-curl
exploit available).

**Determinism.** Reuse the A3 `_next_seed()` monotonic counter for autoreset IC draws so
the whole run is reproducible given the base seed (D1, spec §3).

---

## 3. Reward terms (dense, named; each with its exploit analysis)

Structure mirrors `h1_grasp_rewards.py`: a sum of named per-env terms, **the positive
manipulation terms GATED by an honest force-closure predicate** (the A3 lesson that
defeated transit-farming), velocity/effort negatives as small nudges, **NO
`only_positive_rewards` clamp** (the negatives must be able to push a knock-the-cup-off
policy below a do-nothing's return). The force-closure gate is built from the engine's
honest per-finger normal-impulse signal (`finger_normal_impulse`, exported per §G2;
populated from the stepper's `cup_vertical_impulse` / `finger_contacts`).

Define `m·g·dt` = cup weight kick (the probe's `weight_kick`, 0.2·9.81·(1/240)≈8.2e-3).
`fc` = force-closure gate = (≥3 finger spheres in contact this step) AND (cup vertical
finger impulse ≥ 0.5·m·g·dt) AND (cup within the wrap band). `fc` is the *same gate* the
S1 evaluator's hold-detector uses, minus the "table-support-lost" condition (reward must
fire *before* lift, eval fires *after*).

| Term | Sign | Form | Exploit it blocks / rationale |
|---|---|---|---|
| `reach` | + | `exp(-‖palm_or_fingertip_centroid − cup_center‖² / σ_r²)`, σ_r≈8 cm, **ungated** | The ONLY ungated positive. Needed as the cold-start gradient toward the cup (the force-closure terms are flat-zero until the hand arrives). Bounded and saturating so it cannot dominate a held grasp. Exploit: a policy that hovers the palm near the cup without grasping — bounded by σ_r and dwarfed by the gated terms below, and the cup-disturbance penalty punishes the bump-and-hover. |
| `preshape` | + | `exp(-‖q_hand − q_hand_target‖² / σ_p²) · reach_near`, gated by `reach`≈near | Rewards curling toward the validated `CurlForScale(1.8)` pose **only when the hand is near the cup** (gated by reach), so the policy pre-shapes on approach. Exploit: curling in free space for the term — killed by the `reach_near` gate. |
| `force_closure` | + | `fc` (binary→float), **the genuine grasp signal** | The A3 hard-AND on per-finger impulse, generalized to ≥3 spheres. Mere one-finger contact / batting earns nothing. This is the term the warm-start must light up in C0. |
| `seat_quality` | + | `fc · exp(-(cup_tilt)² / σ_t²) · exp(-‖cup_xy − cup_xy0‖²/σ_s²)` | Rewards a *seated upright* grasp (cup not knocked askew). Pushes the policy toward the ≤2 mm shallow-seat envelope and away from a glancing edge-grab. Gated by `fc` so it only scores a real grasp. |
| `lift_ready` | + | `fc · clamp((cup_finger_impulse − m·g·dt)/(m·g·dt), 0, 1)` | Rewards delivering ≥weight of *vertical* finger impulse while seated — the force-closure margin that S2 will need. Gated by `fc`. **Caps at 1** so it can't be farmed by crushing. Lays the runway for S2 without over-fitting S1 to a marginal hold. |
| `cup_lin_vel` | − | `fc · ‖cup_lin_vel‖²`, weight ~−0.005 (A3-sized) | Penalizes a HELD cup that wobbles (anti-bat). **Gated by `fc`** — exactly the A3 fix: an ungated velocity penalty on an unsaveable/free-falling cup *inverts the gate* on the jittered distribution (a waiting policy accrues more than a slam). Gating makes it faithful. |
| `cup_ang_vel` | − | `fc · ‖cup_ang_vel‖²`, weight ~−0.0006 (A3-sized, ~33× smaller than lin) | Same gating rationale; sized tiny because a real 2-sided wrap lets the cup tumble against point-fingers (A3's large raw ang_vel finding) — must not make a real grasp net-negative. |
| `cup_disturbance` | − | `max(0, cup_xy_displacement − ε) ` + `max(0, table_z − cup_z)·penalty` (cup knocked off / below table), **ungated**, sharp | **THE classic exploit term.** Penalizes shoving the cup across/off the table or below the rim. This is the S1 analog of A3's transit-farming: here the exploit is *batting the cup off the table* for a transient `reach`/contact reward then autoreset → fresh cup → repeat. A sharp ungated penalty on cup XY displacement past a small ε and on cup falling off the table support makes that net-negative. |
| `table_collision` | − | `(any non-fingertip-sphere finger link below table_z) · penalty`, or per the engine's fingertip×table-plane contact rows if exposed | Penalizes slamming fingers/palm through the table to "scoop". Without it the policy learns to drive the hand into the table and rake the cup. If the union world exposes fingertip×ground/table contact impulse (§G2 foot-contact pattern reused for the hand), use that; else use a kinematic below-plane test on the wrap-sphere world positions (exported per §G2 fingertip world positions). |
| `action_sq` | − | `‖action‖²`, weight ~−0.01 | Effort cost; prefers a gentle well-timed close over a constant-max crush. Does not by itself defeat any exploit (the physics + gates do). |
| `action_rate` | − | `‖aₜ − aₜ₋₁‖²`, weight ~−0.01 (stand-env pattern) | Smoothness; prevents a chattering policy that exploits per-step contact noise. Carry `_prev_action` and zero it on autoreset (the go2/stand autoreset-bookkeeping fix). |

**Weight discipline (A3 lesson, restated for S1):** size the gated negatives so a real
held grasp stays *strongly positive every step* while a no-grasp policy sits at ~0 and a
knock-off policy goes negative. The A3 report's failure mode — a velocity penalty large
enough to make a clean grasp net-negative — must be re-checked here with a per-term
breakdown logged in early runs (reuse the `return_terms` dict the A3 reward exposes).

**Anti-exploit summary.** Transit-farming has no direct analog (cup is static), but its
*shape* recurs as **batting-the-cup-off-the-table-for-a-transient-positive-then-autoreset**
— defeated by (i) the force-closure gate on all manipulation positives, (ii) the sharp
ungated `cup_disturbance` penalty, (iii) **no autoreset reward farming**: terminate (not
truncate) on cup-off-table so the episode *ends with the negative*, no fresh-cup bonus.

---

## 4. Termination / truncation

- **Terminate (failure, terminated=True):**
  - cup off table support: `cup_z < table_z − 0.05` (fell) OR
    `‖cup_xy − cup_xy0‖ > r_off` (knocked off the footprint, r_off ≈ table half-extent).
  - cup toppled: cup tilt vs upright > **45°** (S1; S2 tightens to 30° per the spec's
    lift criterion — S1 is looser so the policy can recover a slightly-tipped seat).
  - non-finite obs (NaN/inf guard, stand-env pattern).
- **Truncate (timeout, truncated=True, `info['time_outs']`):** episode length
  **= 256 control steps** (≈2.1 s at decimation 2). Rationale: the probe settles the
  active grip in ~70 steps at 240 Hz (≈35 control steps at decimation 2); a 256-step
  horizon gives ample room to reach (~80), pre-shape+seat (~60), and *sustain* the hold
  long enough that the evaluator's "sustained N steps" window (§5) fits inside truncation.
  This is `value_bootstrap`-compatible (the cfg already sets it) and decimation-invariant
  (specified in control steps).
- **No success early-termination.** Let a successful grasp **run to truncation** (do NOT
  terminate on success). Two reasons: (i) S2 needs the policy to *keep holding* after the
  grasp, and an early-success bonus teaches "grab then relax"; (ii) the evaluator's hold
  definition requires a *sustained* impulse window, which only exists if the episode
  continues. Award the per-step gated positives every step instead of a terminal bonus —
  a sustained hold then naturally out-returns a momentary one.
- **Autoreset:** legged_gym/A3 in-place autoreset (reset done envs, return post-reset obs,
  zero `last_action`/`_prev_action`/`episode_step`/reward-history for done envs).

---

## 5. The S1 evaluator (separate from reward; A5b pattern)

A **separate deterministic eval env** (`CatchRateAlgoObserver` clone →
`HoldRateAlgoObserver`), wired at `after_init`, sharing the training env's `nuka.Device`
but owning an independent union world; run every K epochs; **never perturbs the PPO
distribution** (the A5b invariant). Deterministic policy action = `get_action_values
({'obs': obs})['mus'].clamp(-1,1)` (mean, eval-mode normalizer, restore train mode after).

**Eval config:** ≥**256 eval envs** (spec bar), **fixed eval seed** (123, as A5b) so the
hold delta is the policy's not the cup's, on the **C1-level jitter distribution** (the
gate distribution; C2-widened jitter reported separately).

**Hold operationalized (spec §G3 S1):** a cup is *held* in env e at step t iff, sustained
for **N = 30 consecutive control steps**:
1. cup is **off table support** — either `table_vertical_impulse ≈ 0` (table pair carries
   no load; the union world exposes this per the stepper report) OR the cup is lifted
   `cup_z > table_z + 5 mm`; AND
2. **finger vertical impulse ≥ 0.8·m·g·dt** (the spec's threshold; from
   `cup_vertical_impulse`, per-env, exported via §G2 per-finger normal impulse summed); AND
3. cup tilt < 30° and `‖cup_xy − cup_xy0‖` within the footprint (still a real seated
   grasp, not a fling).

N=30: long enough to exclude a one-step transient (the probe's disturbance settle runs
70 steps at 240 Hz ≈ 35 control steps; 30 is just under that, a genuine *sustained*
hold), short enough to fit inside the 256-step horizon with reach+seat preceding it.
**hold_rate = fraction of eval envs that achieve a 30-step sustained hold before
truncation.** Gate: **hold_rate ≥ 0.8**.

**BITE protocol (eval only, standing — spec §3):** for each env that reached a hold,
**zero the right-hand grip torques** (set the 12 finger-action DOF to 0; keep the arm PD
holding so we isolate the *grip*, not the arm) and continue stepping ~120 control steps.
Assert the cup **falls** (`cup_z` drops > 2 cm OR `cup_vz < −0.10`), exactly the probe's
`bite_drops` discriminator. **BITE pass-rate ≥ 0.9 of held cases** (spec bar). A hold that
survives grip→0 is a passive wedge = FAKE and must fail. Run BITE as a standing eval every
eval cycle (not a one-off), logged alongside hold_rate.

**Logging cadence:** every K epochs print + CSV-append `epoch,hold_rate,bite_rate,
mean_return,finite` (A5b's CSV pattern, `NUKA_*_LOG` env var). Epoch-0 baseline = the
random-init floor.

---

## 6. PPO config deltas from `h1_grasp_ppo_cfg.yaml`

Clone the A3 cfg + `train_h1_grasp_ppo.py` scaffolding (`a2c_continuous`,
`continuous_a2c_logstd`, adaptive-KL, `value_bootstrap`, normalized input/value, the
`--smoke`/`--save-frequency`/eval-observer plumbing). Deltas:

| Key | A3 value | S1 value | Why |
|---|---|---|---|
| `env_name` | `nuka_h1_grasp` | `nuka_h1_s1` (or `nuka_h1_unified` with a `stage` env_config) | New `_ENV_FACTORIES` entry in `vecenv.py` (§7); ONE env class, stage-gated. |
| `env_config` | `{}` | `{stage: "S1", phase: "C1", ...}` | §4 binds: ONE env class, **stage configs not subclasses** — pass the stage/phase + IC params here. The new env **must accept `**kwargs` or the override hygiene** (`train_h1_grasp_ppo.py` already pops `episode_length_s`) — declare the stage keys explicitly and keep popping `episode_length_s`. |
| obs dim | 27 | **set by G2** — q/qdot(17×2=34) + base pose/vel(13) + cup pose/vel(13) + fingertip world pos(30 spheres? or palm+5 tips=18) + per-finger normal impulse(5–6) + foot contact(0 fixed-base, keep field=0) + last action(17). **≈ 110–130**; pin exactly to the G2 export. | Network input auto-sized from `single_observation_space`. |
| action dim | 2 | **17** (right arm 5 + right hand 12) | §1. |
| `mlp.units` | [256,128,64] | **[512,256,128]** | 17-D action / ~120-D obs is materially bigger than 2-D/27-D; widen the net one tier. Keep elu, separate=False. |
| `horizon_length` | 22 | **32** | 256-step episode → 32 tiles cleanly (8 rollouts/episode), and a longer horizon helps credit-assign the reach→seat→hold sequence. |
| `num_actors` | 4096 | **measured-bounded; start 1024, target ≥4096** | **Gated on G1(d) throughput.** The union-world H1 (dof_stride 17, ~30 fingertip + foot + cup×table contacts) is far heavier per env than the 2-DOF synthetic. Set `num_actors` from the *measured* env-steps/sec so one ~100M-env-step stage fits <24 h (spec G1(d)); if it doesn't, a named optimization lands BEFORE training (spec). Do NOT extrapolate — measure at N∈{256,1024,4096} first. |
| `minibatch_size` | 22528 | `num_actors*horizon/4` (e.g. 1024·32/4 = 8192) | Same tiling rule as A3. |
| `learning_rate` / `lr_schedule` | 1e-3 / adaptive | **keep 1e-3 adaptive KL (kl_threshold 0.01)** | The adaptive-KL + 1e-3 was the anti-collapse setting proven on go2/A3; don't change without a measured reason. |
| `entropy_coef` | 0.01 | **0.01, consider 0.005 for C2** | Higher entropy aids C1 exploration (find the seat); once C2 warm-starts from a competent C0/C1 policy, lower entropy stabilizes the hold. Justified by the warm-start curriculum, not a guess. |
| `max_epochs` | 1500 | **C0 ~300, C1 ~1500, C2 ~1000** (per-phase, warm-started) | Each phase warm-starts the next via `--checkpoint` (the script already has the flag); spec §G3 "each stage's policy warm-starts the next" — apply it *within* S1's phases too. |
| `bounds_loss_coef` | 0.0001 | keep | Keeps actions inside the box; needed with 17 torque DOF. |

**Action mapping:** `[-1,1] × per-joint MJCF force limit` (the stand-env pattern with a
`torque_limits` vector — here a 17-vector: arm 40/40/18/18/6, fingers ±1 each). §4 binds
"torque clamped to per-joint real limits"; use the ctrlrange table in §1 verbatim.

---

## 7. ONE env class, stage-config surface (binds §4)

Do NOT write S1/S2/S3/S4 subclasses. Write **one `H1UnifiedEnv`** over the G2 union
binding with a `StageConfig` dataclass passed via `env_config`:

```
@dataclass
class StageConfig:
    stage: str           # "S1" | "S2" | "S3" | "S4" | "S5"
    base_mode: str       # "fixed" | "floating"   -> cook override (root joint)
    action_mask: list    # which of the dof_stride DOF the policy drives this stage/phase
    reward_stage: str    # selects the term set + weights (S1 grasp-hold, S2 +lift, ...)
    ic_sampler: str      # "C0_close_only" | "C1_approach" | "C2_widen" | ...
    jitter_xy: float     # cup XY jitter half-box (curriculum-mutable at runtime)
    episode_steps: int   # control-step horizon
    ...
```

- `base_mode` selects the cook (fixed-base selective-Fixed for S1/S2; floating for
  S3/S4) — the cook override §4 names. S1 uses `"fixed"` + the right-arm+hand free set.
- `action_mask` realizes C0's "hand-only, arm PD-held" vs C1/C2's "all 17" **without a
  re-cook** (masked DOF are PD-servoed to their IC pose inside `step()`).
- `reward_stage` selects the §3 term set; S2 swaps in `+lift` weights (§8) — same env,
  different config.
- Register `nuka_h1_unified` in `_ENV_FACTORIES` (`vecenv.py`) alongside the existing
  go2/h1_stand/h1_grasp entries; the `NukaVecEnv` IVecEnv contract (single-env spaces,
  4-tuple step, float dones, `info['time_outs']`) is unchanged — S1's env just needs the
  same duck-typed surface (`single_*_space`, `num_envs`, `reset`/`step` 5-tuple,
  `sample_actions`, `close`). The eval observer reaches the training env via
  `algo.vec_env.env` exactly as A5b does.

---

## 8. Stage S2 (+lift) delta sketch — so S1 doesn't paint S2 into a corner

S2 is the **same `H1UnifiedEnv`, `stage="S2"`, `base_mode="fixed"`, warm-started from the
best S1 checkpoint**. Reward changes: add a **`lift_height`** positive,
`fc · clamp((cup_z − table_z)/h_target, 0, 1)` with `h_target = 15 cm` (the spec's
lift bar), and **relax the `cup_disturbance` XY term** to permit intentional lateral motion
while *keeping* the off-table-edge and topple penalties. The `lift_ready` term S1 already
trains (≥weight vertical impulse) is exactly the prerequisite — that's why it's in S1's
reward now. Termination tightens cup-tilt to **30°** (the spec's S2 criterion) and adds a
**lift-success early state** (cup ≥15 cm, held ≥1 s ≈ 120 control steps, tilt <30°).
Evaluator: same hold-detector with the off-table condition replaced by the lift criterion
(cup ≥15 cm above table sustained ≥1 s), BITE unchanged. **Decisions S1 must NOT foreclose,
and doesn't:** (i) episode horizon 256 already has room for reach+seat+hold; S2 will extend
to ~400 control steps for the lift+1 s-hold — keep `episode_steps` a stage config, never a
module constant (it is, §7); (ii) S1's `lift_ready` impulse-margin term means the S1 policy
already grasps with the *vertical* force-closure surplus a lift needs, so the S1→S2
warm-start lands the policy in S2's basin rather than a marginal hold that drops the moment
the table support is removed.

---

## Open items the implementer must pin against G1/G2 (not designable from here)

1. **Exact obs dim + field layout** — from the G2 export (q/qdot env-major width =
   dof_stride·2 = 34; fingertip world positions = #wrap spheres or palm+tips; per-finger
   normal impulse grouping). §6's ~110–130 is an estimate; pin to the binding.
2. **Whether the union world exposes fingertip×table contact impulse** (drives whether
   `table_collision` uses contact rows or a kinematic below-plane test, §3).
3. **Measured control decimation + measured throughput** (§1, §6) — G1(d) numbers gate
   `num_actors` and the <24 h/stage bar; measure, do not extrapolate (§3 honesty bar).
4. **Reachable-workspace cap** for the C2 jitter curriculum (§2) — measure the fixed-base
   right-arm reach envelope around `cup_xy0`; the A3 "X-extreme past reach" finding says
   this is a hard cap, not a free knob.
