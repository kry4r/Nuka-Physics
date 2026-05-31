# Nuka Physics v0.3 (S1) — Sprint Retrospective

**Date:** 2026-05-31 · **Branch:** `v03` · **Anchor (S1):** a 4096-env Unitree Go2 PPO
locomotion pipeline on the in-house CUDA physics engine, trained to a walking gait.

This is the retrospective + source material for the §5.1 quarterly external write-up. The
companion operational doc (exit-criteria scorecard, regen commands, owner-deferred items) is
[`2026-05-31-v03-status-and-handoff.md`](2026-05-31-v03-status-and-handoff.md).

---

## 1. What shipped (the headline)

**A 4096-env Go2 PPO converged to a command-conditioned walking gait**, end-to-end on the
Nuka engine: CUDA Featherstone dynamics → C ABI → nanobind/DLPack zero-copy → `NukaGymEnv` →
rl_games, **D1-deterministic** at scale, no NaN. The trained policy tracks a velocity command
monotonically and stands still on command (independently re-verified):

| cmd vx (m/s) | −0.5 | 0.0 | +0.5 | +1.0 |
|---|---|---|---|---|
| body vx | −0.462 | **+0.002** | +0.467 | +1.032 |
| tilt | 2.2° | 2.8° | 3.0° | 4.3° |

0 terminations across the sweep (220 deterministic-μ steps × 64 envs per command);
`last_mean_rewards` 18.4 at 250 epochs. The **cmd-0 →
+0.002** stand-still is the discriminator that rules out a degenerate "always walks forward"
policy. Qualified: tracking is approximate (small rail under/overshoot) and `bounds_loss` is
untidy — honest, neither breaks the gait.

## 2. The engineering arc

The functional spine was built bottom-up across four phases:

- **p01 — physics substrate.** Batched N-articulation Featherstone (ABA), full-chain contact
  Jacobian (base-inclusive), CRBA joint-space inertia, PGS-solved articulated contact/limit
  rows, a batched stepper, and a **floating 6-DOF base** (required for locomotion). In-engine
  timers + a frozen RTX-4000-Ada 4096-env baseline.
- **p02 — PyTorch binding.** nanobind `_nuka_ext`, DLPack zero-copy (torch interop with
  **zero libtorch linkage** — ABI-neutral), and a forward-only `torch.autograd.Function`
  skeleton + caller-owned CUDA stream.
- **p03 — RL integration.** An engine per-env **reset** primitive (D1, masked autoreset),
  `NukaGymEnv` (gymnasium 5-tuple, on-GPU obs **bit-exact** to the numpy oracle), and an
  rl_games 1.6.5 `IVecEnv` adapter — training launches at 4096×24.
- **p04 — PPO + exit.** Authoritative `BASE_POSE` view, a verbatim legged_gym reward port,
  command curriculum, the demo video, and the convergence run.

## 3. The hard problems (the real story — and the lessons)

Every convergence blocker this sprint was a **found root cause**, not a tuning wall. The
pattern that repeatedly paid off: *build the discriminator that tells substrate-bug from
config-gap before touching hyperparameters.*

1. **Batched-contact dt-instability (#42).** The C-ABI batched-contact path went unstable at
   the native dt. Root-caused and fixed in the contact assembly, not papered over with a
   smaller dt.
2. **Native-dt soft-gain joint-drive instability (#43).** Soft PD gains (Kp=20/Kd=0.5) blew
   up at native dt=0.005 under explicit damping. Fixed with **general implicit joint damping**
   so soft-gain policies walk at native dt with no sub-stepping (5× fewer physics steps); the
   golden was regenerated implicit. *Lesson: an open-loop soft-gain hold is posturally
   unstable by design — "zero-action standing must be stable" is the wrong proxy for substrate
   health; the right one is "does the trained policy walk in the actual env."*
3. **The action-space rescale bug — the convergence blocker.** The first 1500-epoch run
   collapsed (reward≡0, ep_len~22, value-head collapse). The instinct-trap was to read this as
   a reward-shaping or exploration wall. Instrumenting the **real** training step instead of
   an IID sim revealed `mean|a|=64` — the policy's [-1,1] output was reaching the env at ~64.
   Root cause: rl_games (`clip_actions=True`) **rescales** the policy output onto the declared
   action-space bounds, and the gym space had been declared at ±100 (the internal safety
   clamp) — a **100× amplification**. Fix: declare the action space at ±1 (one line) so the
   rescale is the identity. *Lesson: when a framework sits between your policy and your env,
   instrument the value that actually crosses the boundary; a 100× scale error hides perfectly
   behind a plausible "RL is just hard" narrative.*
4. **Reset authority + FK lag.** `ARTICULATION_LINK_POSE[root]` writes are non-authoritative
   (the integrator overwrites base pose from an internal buffer) and the link-pose view lags
   one step after a reset. This forced the p03 engine reset primitive (write the authoritative
   `base_pose`/`link_velocity`/q/qd) and the p04 un-lagged `BASE_POSE` view feeding post-reset
   obs for done envs.

Throughout, **D1 strong determinism was never traded away** (two-run bit-exact + 4096-env
determinism ctest stayed green across every change).

## 4. What's owner-deferred (documented, not failed)

- **§7 perf gate (<1 ms/env-step @ 4090).** No 4090 on this box (2× RTX-4000-Ada, ~3× slower);
  the relative baseline is frozen. End-to-end the RL loop runs ~92k env-steps/s here.
- **External publication (§5.1).** This doc + the status/handoff are the source material.
- **Gait polish.** Sharper command tracking + `bounds_loss` tidy are more-epochs / one-coef
  follow-ups; deliberately not tuned at the finish.
- **Unify exit #3 + #6.** The demo video currently replays the proven external PR#62 policy
  (honestly labeled); regenerating it from the in-house converged checkpoint
  (`out/go2_policy/go2_walk_randomcmd_ep250.pth`) would make it a single from-scratch artifact.

## 5. Process notes (for the next sprint)

- **Subagent-driven with a single committing controller** worked: parallel build-out, adversarial
  phase-end reviews, controller-gated commits. The one failure mode was **running two
  engine/python-package-touching subagents concurrently** (a rebuild race broke `import nuka`
  mid-run) — engine-touching agents must be sequential.
- **The commit gate that caught the most was the independent re-verification** — re-deriving the
  convergence eval (hand-rebuilt deterministic-μ forward + own env instance) rather than
  trusting the subagent's reported numbers. It reproduced them exactly; had it not, the
  "converged" claim would have failed honestly.
- **Regression tests should encode the bug's mechanism, not just its symptom.** The p04-T2
  guards assert the action-space bound *and* that rl_games' own `rescale_actions` is the
  identity for it (with a ±100 positive control) — they fail the instant the ±100 declaration
  regresses.

## 6. Metrics summary

- 50/50 python tests; D1 ctest 11/11 (determinism / reset / base-pose / 4096-env). Full Runner
  `--smoke` launches clean with healthy losses (c_loss finite, entropy stable, no collapse).
- Converged policy: 250 epochs, `last_mean_rewards` 18.4, command-conditioned, 0 terminations
  in eval. Checkpoints durable in gitignored `out/go2_policy/`.
- D1 strong determinism preserved end-to-end; ABI-clean PyTorch interop (no libtorch linkage).
