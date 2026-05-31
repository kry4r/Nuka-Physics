# Nuka Physics v0.3 (S1) — Status & Owner Handoff

**Date:** 2026-05-31 · **Branch:** `v03` · **Anchor:** S1 = 4096-env Unitree Go2 PPO locomotion.

This is the living status/handoff for v0.3. It records what landed, the exit-criteria
scorecard, what is owner-deferred, and exact regen/handoff commands. (The from-scratch
**convergence** iteration — exit #3 — is in progress; that section is marked accordingly.)

---

## 1. Phase status

| Phase | Scope | Status |
|---|---|---|
| **p01 — Perf baseline + 4096-env articulated path** | batched Featherstone + contacts, floating base, in-engine timers, RTX-4000-Ada baseline | **DONE** (perf gate itself owner-deferred, see §3) |
| **p02 — PyTorch binding** | nanobind `_nuka_ext`, DLPack zero-copy, forward-only `autograd.Function` skeleton, caller CUDA stream, wheel | **DONE + reviewed** |
| **p03 — RL integration** | engine per-env reset, `NukaGymEnv`, `isaaclab_compat`, on-GPU Go2 obs, rl_games adapter, train launches | **DONE + reviewed** |
| **p04 — PPO + exit demo** | base_pose obs fix, legged_gym reward port, command curriculum, scene, demo video, convergence run | **IN PROGRESS** (punch-list + video done; convergence iterating) |

## 2. Exit-criteria scorecard (master-plan §7 + §5.1)

| # | Criterion | Status |
|---|---|---|
| 1 | Step time < 1 ms/env-step @ 4096 on RTX 4090 | **OWNER-DEFERRED** — no 4090 here; box is 2× RTX-4000-Ada (~3× slower). RTX-4000-Ada baseline frozen (`out/perf/baseline_rtx4000ada_4096_frozen.json`, commit b5eae4c). 4096-env step runs ~92k env-steps/s end-to-end in the RL loop. |
| 2 | Energy drift < 2 % over 1000 steps | **COVERED** — passive-dynamics invariant, asserted at `energy_drift_rel ≤ 0.02` in the V01 foundation pipeline + smoke configs (`double_pendulum`/`single_falling_box`/`two_body_collision`); V2 invariant system in `src/core/diagnostics/invariants*`. (Driven locomotion is not energy-conserving by design.) |
| 3 | 4096-env Go2 PPO converges to stable walking gait | **IN PROGRESS** — pipeline runs at scale + D1-stable; first from-scratch run collapsed (reward≡0 under `only_positive_rewards` + a sub-stable stand + `entropy_coef=0`); owner-authorized iteration underway (align to Unitree's official rsl_rl recipe; fix the zero-action standing substrate first). NOT claimed converged unless it genuinely walks. |
| 4 | PyTorch `autograd.Function` skeleton wired (forward-only) | **DONE** — `python/nuka/autograd.py`; caller-facing signature locked, backward = zero stub. |
| 5 | V1 Featherstone oracle passing (carry-forward) | **CARRY-FORWARD** — 2 pre-existing ABA-vs-MJX oracle gaps (`FeatherstoneOracle.RandomSampleGoldensMatchCudaAba`, `V01…Phase6`) remain, orthogonal to v0.3 work (documented since sim-val #43). |
| 6 | Demo: 4096-env Go2 locomotion video | **DONE (proven policy)** — `examples/demo/` renders a 16-env walking grid (MP4) from the externally-trained, Nuka-validated Go2 policy (motion.pt, PR#62). Satisfies "a Go2 locomotion video"; it is NOT from-scratch convergence (that is #3). |
| 7 | Quarterly external output (§5.1) | **OWNER** — publication is an owner action; this doc + the retrospective are the source material. |

## 3. Owner-deferred (not achievable on this box; hand off with docs)

- **§7 perf gate (< 1 ms @ 4090)** — owner validates on RTX 4090 (or the supplementary < 1.3 ms @ 4096 on RTX 5080 with Nsight). This box records the relative RTX-4000-Ada baseline only.
- **External publication (#7).**
- **Multi-iteration gait tuning toward exit #3**, IF the in-progress iteration hits a wall — the owner decides whether to invest further reward-shaping / curriculum / contact-force exposure.

## 4. Key engineering wins (the real S1 deliverable, independent of #3)

- 4096-env Go2 RL pipeline: engine → C ABI → nanobind/DLPack zero-copy → `NukaGymEnv` → rl_games, running at scale, **D1-deterministic**, no NaN.
- Engine per-env **reset** primitive (`nuka_world_reset`/`reset_envs`, D1, masked autoreset) + authoritative **`BASE_POSE`** view (un-lagged, correct post-reset).
- legged_gym Go2 reward **ported** (learnable in isolation: per-step ~1e-3, tracking-dominated), command curriculum, post-reset obs.
- On-GPU 48-dim obs **bit-exact** to the proven numpy oracle; structural joint permutation.
- #43 implicit joint damping → soft-gain policies walk at native dt (5× fewer physics steps).

## 5. Regen / handoff commands

```bash
# Python wheel + env
cd python && /root/miniconda3/envs/nuka-v03/bin/pip install -e . --no-build-isolation
/root/miniconda3/envs/nuka-v03/bin/python -m pytest python/tests/ -q     # 46 pass

# Demo video (proven policy, exit #6)
CUDA_VISIBLE_DEVICES=0 examples/demo/render_video.sh                      # -> out/go2_demo/*.mp4

# Train (the SAME script does smoke + full)
CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py --smoke
CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py --num_actors 4096 --max_epochs 1500

# Engine + determinism
cmake --build build-cuda128 -j && (cd build-cuda128 && ctest -R 'Reset|Determinism|Go2PdStanding|BasePoseView')
```

## 6. Remaining p04 work (after the convergence iteration resolves)

- Finalize exit #3 status (converged / learning-demonstrated / wall + cause) honestly.
- Regression test for the full RL loop (smoke-train perf regression).
- p04 phase-end consolidated review + v0.3 close checklist + retrospective (§5.1 source).

---

*Honesty note: the demo video (#6) replays a proven external policy and is labeled as such; it
is not evidence of in-house from-scratch convergence (#3). No partial training run is reported
as a "stable walking gait" unless it genuinely walks.*
