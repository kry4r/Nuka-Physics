# Driving the real trained Go2 policy in Nuka — sim-val report (#41 + #42)

**Harness:** `examples/sim_val/go2_policy_drive.py`
**Run:** `export CUDA_VISIBLE_DEVICES=0; /root/miniconda3/envs/nuka-v03/bin/python examples/sim_val/go2_policy_drive.py`
**Date:** 2026-05-30 · branch `v03`

## TL;DR — verdict (qualified; read the qualification)

The real trained Go2 policy (`motion.pt`, 48→12 MLP) driven in-the-loop on Nuka
produces clean, command-tracking **WALKING — but only when the physics is sub-stepped
to dt=0.001** (decimation 20; the policy itself still runs at 50 Hz, matching
training). At that config it tracks the commanded 0.5 m/s forward, upright and stable,
cross-validated by two independent buffers. **At the NATIVE training dt=0.005 the
policy FLAILS** — that is a real, separately-characterized limitation, not swept under
the rug (§5).

Two distinct findings came out of this task; do **not** conflate them:

| # | finding | status |
|---|---|---|
| **#42** | the c_abi/nanobind **PD stance** collapsed at dt≥0.002 (deep 0.03 m foot seating) | **FIXED** (engine: seat 0.03→0.002 m); stance now holds at every dt incl. native 0.005 |
| **#41** | the **learned policy** (soft gains Kp=20/Kd=0.5) flails at native dt=0.005 | **OPEN** mechanism; mitigated by sub-stepping physics to dt=0.001 |

| metric | value | note |
|---|---|---|
| diagnostic ladder D0–D5 | **ALL PASS** | conventions proven before any rollout |
| Nuka↔URDF joint permutation | **identity `[0..11]`** | quadrant + cooked-q signature (hard gate) |
| **walk** fwd vel (LINK_VELOCITY, body x) | **+0.51 m/s** (last-50) | cmd 0.5, **@ dt=0.001** |
| **walk** fwd vel (LINK_POSE, world px) | **+0.49 m/s** (dx +2.95 m / 6 s) | independent cross-check, agrees |
| **walk** tilt / height | mean 3.8° (max 6.2°) / z 0.423 m | upright, **@ dt=0.001** |
| native-dt PD stance (Kp=60/Kd=4) | z 0.432 m, tilt 0.22° | **HOLDS @ dt=0.005** (the #42 fix) |
| native-dt soft-gain policy | max tilt 56° | **FLAILS @ dt=0.005** (the open limit) |
| 4096-env policy smoke | finite, no NaN | production env count runs |

**STATUS: conventions + engine fix DONE; policy walks only sub-stepped; native-dt
policy stability OPEN.**

## 1. Conventions (the contract), proven in rung D0

From `docs/plans/2026-05-30-v03-unitree-policy-config.md`, re-confirmed bit-exact
against the golden (`/root/third_party/go2_pr62/golden_io.json`) in **D0**:

- **obs(48):** `[0:3]` base_lin_vel ×2.0, `[3:6]` base_ang_vel ×0.25, `[6:9]`
  projected_gravity (gravity unit vector in BODY frame, no scale), `[9:12]`
  cmd×`[2,2,0.25]`, `[12:24]` (q−default)×1.0, `[24:36]` qd×0.05, `[36:48]` previous
  raw action. Clip ±100.
- **action(12):** target_q = default + 0.25·action. **gains Kp=20, Kd=0.5** (uniform),
  target_vel=0; force limits hip/thigh 23.7, calf 35.55 N·m via `DRIVE_FORCE_LIMIT`.
- **default_angles** (URDF order FL,FR,RL,RR × hip,thigh,calf):
  `[0.1,0.8,-1.5, -0.1,0.8,-1.5, 0.1,1.0,-1.5, -0.1,1.0,-1.5]`.
- **velocity frame:** base lin/ang vel read directly from `LINK_VELOCITY` root slot
  (omega-first, already BODY frame, no lag) — no world→body rotation.
- **projected_gravity:** `quat_rotate_inverse(base_quat_wxyz, [0,0,-1])`.
- **control rate:** policy 50 Hz. Physics dt = **0.001 s**, decimation **20** (=50 Hz)
  for the rollout. See §5 for why 0.001 (not training's 0.005).

## 2. Nuka↔URDF joint permutation — identity `[0..11]`

Nuka's actuated slots 1–12 are already in URDF order FL,FR,RL,RR × (hip,thigh,calf).
Established in **D1**, strongest last:

1. **Leg quadrant** from `ARTICULATION_LINK_POSE` (forward +x, left +y).
2. **Cooked-q signature [HARD GATE].** The scene cooks each leg to (hip≈0, thigh≈+0.8,
   calf≈−1.5); within a leg the slot with q≈+0.8 is the thigh, ≈−1.5 the calf, ≈0 the
   hip — exact and unambiguous. Measured: slot2/5/8/11 q=+0.80 (thigh), slot3/6/9/12
   q=−1.50 (calf), slot1/4/7/10 q≈0 (hip).
3. **Directional own-q nudge (NON-load-bearing).** Nudging the mapped FL_calf target
   makes that joint respond, but under ground contact the free base smears motion
   across legs, so it does **not** reliably dominate — reported neutrally, varies
   run-to-run, NOT relied on. (Permutation rests on 1+2 and the walk below.)
4. **End-to-end:** an MLP is not permutation-equivariant, so coherent command-tracking
   walking is only possible if both the obs read- and action write-indexing are correct.

## 3. Convention diagnostic ladder (D0–D5) — all PASS

- **D0** offline obs-assembly bit-exact vs golden (input + policy-output diff 0.0).
- **D1** identity permutation (above).
- **D2** projected_gravity: upright→`[0,0,-1]`, known +90° roll→`[0,-1,0]` (catches
  sign/transpose).
- **D3** base lin/ang vel from `LINK_VELOCITY` root, finite & sane (sag picks up
  downward lin-vel + pitch ang-vel), no rotation applied.
- **D4** warm-start (1 s @ Kp=20): upright (tilt 2.3°) and settled (max|qd| 1.2),
  base_z 0.380. Soft-Kp sag max|q−default| ≈ 0.45 rad (held against gravity, not the
  training teleport-reset — a robustness offset the policy recovers from).
- **D5** live obs → policy → 12 finite actions; obs[12:24] live-matches (q−default).

## 4. THE #42 ENGINE FIX — c_abi foot seating 0.03 → 0.002 m (PD stance)

The first attempt to run at the native dt=0.005 collapsed **the PD baseline too** — the
PD baseline that is supposed to reproduce the proven C++ standing test. Root-caused
**cleanly** (this is the part that is fully isolated):

- The C++ standing test (`tests/runtime/test_go2_pd_standing.cpp`) HOLDS the Go2 crouch
  rock-solid (tilt 0.0044 rad / 1500 steps) at Kp=60/Kd=4, dt=1/240, on the **same**
  stepper (`gpu::BatchedArticulatedWorld::Step`) the c_abi path uses.
- The **only** difference between the validated-stable path and the collapsing c_abi
  path is the **ground seating depth** passed to that stepper: the C++ test seats at
  **0.002 m** (`SeatGround`), the c_abi path baked **0.03 m** (`DeriveGroundHeight`,
  `kRestFootPenetration`). A clean A/B (only this constant changed, same gains/pose/dt):

  | seating | dt=1/240 | dt=0.005 | dt=0.002 | dt=0.001 |
  |---|---|---|---|---|
  | **0.03 m** (old) | COLLAPSE 1.4 s | COLLAPSE 3.6 s | COLLAPSE 0.75 s | HOLD |
  | **0.002 m** (fix) | HOLD | **HOLD** | HOLD | HOLD |

- **Fix:** `kRestFootPenetration` 0.03 → 0.002 in `src/c_abi/world.cpp` (now matches the
  validated C++ seat). The production (c_abi/nanobind) path now holds the PD stance at
  **every dt incl. native 0.005** (PD baseline: z 0.432 m, tilt 0.22°).
- **Honesty on the mechanism:** the *causal variable* (penetration depth) is isolated;
  the *mechanism* (deeper seat → larger startup contact-correction transient) is only
  inferred and not fully pinned — the 0.03 m collapse is non-monotonic in dt (0.002 s
  dies fastest). We anchor to the empirical A/B + the validated C++ seat, not a
  mechanism. T6 only ever proved the contact solve *resolves* a deep penetration over a
  few steps; it never validated long-horizon floating-base PD stance stability, so
  "T6-validated" did not license 0.03 m here.
- **Regression:** all c_abi tests (multi_env, drive_target_io, create_step_destroy,
  cpp_wrapper_raii), runtime ground-truth (go2_pd_standing 3/3, batched_articulated
  3/3, floating_base_contact 3/3), and nanobind pytest (22) PASS at 0.002 m. The
  `test_drive_target_io` perturbation was reduced 0.30→0.05 rad — NOT loosening: a
  +0.30 nudge on all 12 fixed-base targets drives a nonlinear contact-conflict regime
  that reverses ~4 joints/env (proven: wrong-sign=0 at δ≤0.10 across seatings, only
  appears at large δ), confounding the write-sign property the gate tests; 0.05 rad
  isolates it. The write path itself is untouched by a ground-seating constant.

## 5. THE POLICY FLAIL @ NATIVE dt=0.005 — separate, OPEN limit

With #42 fixed, the **PD stance** holds at native dt — but the **learned policy** does
not. At dt=0.005 the soft-gain policy flails (tilt → 110°). This is **not** #42 (the PD
stance holds) and **not** a convention bug (D0–D5 pass). What we isolated, and what
remains open:

- **Isolated to the soft policy gains.** A 2×2 static-hold gain sweep at dt=0.005
  (same scene, same default-pose target, only Kp/Kd vary):

  | | Kd=0.5 | Kd=4 |
  |---|---|---|
  | **Kp=20** | **FLAIL** (tilt 93°) | HOLD (tilt 12°) |
  | **Kp=60** | HOLD (tilt 0.7°) | HOLD (tilt 1.2°) |

  Only the (Kp=20, Kd=0.5) corner — the policy's exact contract — flails. Raising
  **either** Kp or Kd, or sub-stepping to dt=0.001, restores the hold.
- **Mechanism OPEN (honestly).** It is *not* "fast policy targets" — a **static**
  soft-gain hold also flails (the sweep above uses no policy). It is *not* a clean
  damping ratio ζ∝Kd/√Kp — the lowest-ζ corner (Kp=60,Kd=0.5) **holds**. It needs both
  soft Kp and soft Kd interacting with explicit integration + contact at the coarse dt;
  the precise threshold is unproven and left as future work (candidate: implicit
  joint-drive integration).
- **Mitigation (principled, not a rug).** The policy's Kp/Kd are FIXED by its deploy
  contract, so we **sub-step the physics to dt=0.001** (decimation 20; policy unchanged
  at 50 Hz) — a finer, *more accurate* integration of the soft drive. At that config
  the policy walks (§6). `native_dt_characterization()` in the harness runs dt=0.005
  explicitly every time, so the artifact self-documents this gap rather than hiding it.

## 6. The walk (dt=0.001, sub-stepped) — WALKS-LIKE, cross-confirmed

300-policy-step rollout (6.0 s sim, cmd=[0.5,0,0], 64 envs):

- Stands up out of the warm-start (z 0.380 → 0.431), holds height (mean 0.423 m),
  upright (tilt mean 3.8°, max 6.2°).
- Forward vel **+0.51 m/s** (last-50, `LINK_VELOCITY` body-x) tracks cmd 0.5.
- **Independent cross-check** (`ARTICULATION_LINK_POSE` base px, different buffer): net
  **dx = +2.95 m / 6 s → vx_world = +0.49 m/s**; lateral dy +0.33 m. Two buffers agree
  → real locomotion, not a velocity-buffer artifact.
- Gait-like leg motion (max|qd| ~9 rad/s), no NaN. 4096-env smoke finite.

## 7. PD-baseline regression (@ dt=0.001)

| | policy (cmd 0.5) | PD baseline (hold) | delta |
|---|---|---|---|
| base_z mean | 0.423 m | 0.431 m | ~same height |
| tilt mean | 3.8° | 0.40° | policy tilts a bit (actively walking) |
| forward vel | +0.51 m/s | −0.004 m/s | policy ADDS commanded forward motion |

## 8. What is proven / what is open

**Proven:** obs/action conventions (bit-exact), joint permutation, the C-ABI drive +
state-readback loop, the #42 PD-stance seating fix (stance holds at every dt, 43 tests
green), and command-tracking locomotion under a stable (sub-stepped) integration
config, two-buffer confirmed.

**Open / not claimed:** native-dt (0.005) stability of the *soft-gain policy* (the
mechanism behind the 2×2 isolation); zero-shot sim2sim fidelity vs Isaac Gym (different
solver/contact/integrator); energy-drift §7 exit gate on the policy-driven loop;
non-flat ground / friction-cone exercise. "Go2 WALKS" is true **only** with the
dt=0.001 qualification — not at the native training dt.
