# v0.3 — General implicit joint damping (#43) + go2_stand_5s golden regeneration

**Date:** 2026-05-31 · **Branch:** v03 · **Owner-authorized protected-file change.**

## What changed and why

The learned Go2 policy uses contractually-soft gains (Kp=20, Kd=0.5). At the native
training dt=0.005 the robot flailed (tilt→110°); it only walked when the physics was
sub-stepped to dt=0.001 (decimation 20, 5× physics cost). Root cause (#43): the
explicit joint-drive damping term `-Kd*qdot`, integrated by semi-implicit Euler, is
only **conditionally** stable — instability ∝ `dt*Kd/m_eff`, and contact coupling
shrinks the effective inertia `m_eff`, so the term self-excites at the native dt. A
fixed-base sweep pinned it: the buzz amplitude GROWS with Kd (Kd=0→max|qd|≈5,
Kd=0.5→376 at force_limit=200) and is Kp-INDEPENDENT (376 at Kp=20/40/60) — a clean
explicit-damping instability.

## The fix — general implicit joint viscous damping (MuJoCo-style), not a PD hack

The drive emits only the Kp stiffness torque (`defer_velocity_damping`); the per-joint
linear damping coefficient `c_j` (which the PD layer happens to populate from Kd) is
folded into the constrained-velocity solve:

- CRBA `M` kernel adds `dt*c_j` to the joint diagonals ⇒ the factored inverse is
  `(M + dt*C)^-1`.
- The solve seeds `qdot ← qdot_half − dt*(M + dt*C)^-1 * (C * qdot_half)` (backward-
  Euler joint damping), using the **full coupled** `M` so leg-joint damping reacts onto
  the floating base through the dense inverse. A diagonal-`D` velocity decay cannot
  reproduce that coupling (a failed earlier attempt).

Both the batched production path AND the `env_count==1` single-env C-ABI oracle path use
this ONE implicit scheme (unified per owner choice). New general kernel
`ApplyImplicitJointDamping`. D1 determinism preserved (0 mismatches at 1 and 4096 envs).
(The unified single-env path is validated **fixed-base only** — the go2_stand oracle; a
floating-base `env_count==1` c_abi scene is not exercised by any current test. Production
locomotion runs batched, so this is not on a shipped path.)

**Result:** the policy now walks at the native dt=0.005 (dx +2.93 m / 6 s, vx 0.488 ≈
cmd 0.5, tilt < 5.4° — same quality as the old dt=0.001 sub-step) ⇒ **decimation 4 vs 20
= 5× fewer physics steps** for soft-gain policies. No more sub-stepping.

## Protected golden: `tests/oracle/golden/go2_stand_5s.bin` regenerated (implicit)

The go2_stand_5s golden is a MuJoCo/MJX ground-truth trajectory. Its generator
(`tools/oracle/generate_mjx_golden.py`, `generate_stand_trajectory`) previously
integrated the PD drive with **explicit** damping `-Kd*qvel` (semi-implicit Euler) — the
scheme the OLD Nuka matched to <1e-4. Unifying Nuka to implicit moved its trajectory
~6.8e-3 rad off that explicit golden, so the golden was **regenerated with the matching
implicit scheme**: `step_once` now emits stiffness-only torque, takes one `mjx.forward`
for `qacc` + the dense mass matrix `M`, half-steps the velocity, then applies the same
`(M+dt*C)^-1` backward-Euler joint damping. MuJoCo still supplies the forward dynamics
and `M`, so it remains a genuine oracle (it now also cross-validates Nuka's CRBA `M`).

| | SHA-256 | size |
| --- | --- | ---: |
| old (explicit) | `8630566e67958545c102d3e8f0dcbf4112d1ca527edcbb682e99efd26ec6ddb3` | 62450 |
| new (implicit) | `2db6695fad67d7c8760dc7e495c5f2e0fc77c418bd79f91c3928c9e59f7ddc76` | 62450 |

**Validation:** Nuka-implicit vs the regenerated MuJoCo-implicit golden →
`max_abs = 1.93e-05` (< the **unchanged** 1e-4 tolerance, ~5× margin). Regenerate with:

```bash
.nuka-oracle-venv/bin/python tools/oracle/generate_mjx_golden.py \
  --model examples/scenes/go2_stand.usda --mode stand-trajectory \
  --steps 1200 --dt 0.004166666666666667 --out /tmp/nuka_owner_candidates/go2_stand_5s.bin
```

## Out of scope / unrelated

The two pre-existing `*MatchCudaAba*` / `Phase6CudaAbaMatchesGo2AndH1MuJoCoOracle`
ABA-vs-MJX oracle failures are orthogonal (bit-identical magnitudes on the clean tree,
they never touch the damping/integration path) and are NOT addressed here.
