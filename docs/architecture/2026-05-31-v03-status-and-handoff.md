# Nuka Physics v0.3 (S1) — Status & Owner Handoff

**Date:** 2026-05-31 · **Branch:** `v03` · **Anchor:** S1 = 4096-env Unitree Go2 PPO locomotion.

This is the living status/handoff for v0.3. It records what landed, the exit-criteria
scorecard, what is owner-deferred, and exact regen/handoff commands. (The from-scratch
**convergence** iteration — exit #3 — is **DONE**: the 4096-env Go2 PPO converged to a
command-conditioned walking gait, independently re-verified; see #3 below.)

---

## 1. Phase status

| Phase | Scope | Status |
|---|---|---|
| **p01 — Perf baseline + 4096-env articulated path** | batched Featherstone + contacts, floating base, in-engine timers, RTX-4000-Ada baseline; **+ (perf track re-opened) opt-in CUDA-graph step replay, D1/D2 determinism toggle + C ABI, parameterized perf-gate test** | **DONE** (absolute §7 gate owner-deferred to the 5080, see §3; the re-opened mechanisms are delivered + byte-exact-verified) |
| **p02 — PyTorch binding** | nanobind `_nuka_ext`, DLPack zero-copy, forward-only `autograd.Function` skeleton, caller CUDA stream, wheel | **DONE + reviewed** |
| **p03 — RL integration** | engine per-env reset, `NukaGymEnv`, `isaaclab_compat`, on-GPU Go2 obs, rl_games adapter, train launches | **DONE + reviewed** |
| **p04 — PPO + exit demo** | base_pose obs fix, legged_gym reward port, command curriculum, scene, demo video, convergence run | **DONE** (punch-list + video done; 4096-env PPO **converged** to a command-conditioned walk — exit #3) |

## 2. Exit-criteria scorecard (master-plan §7 + §5.1)

| # | Criterion | Status |
|---|---|---|
| 1 | Step time < 1 ms/env-step @ 4096 on RTX 4090 | **OWNER-DEFERRED** (absolute assertion) — no 4090 here; box is 2× RTX-4000-Ada (~3× slower). RTX-4000-Ada baseline frozen (`out/perf/baseline_rtx4000ada_4096_frozen.json`). Median per-env-step **≈ 0.92 µs** (CUDA-event, reproduced) — far under 1000 µs even on this slow box. **Parameterized perf-gate test now present** (`tests/perf/test_go2_4096env_step_time.cpp`, ctest `Go2_4096env_StepTime`): records-and-skips off the validation GPU (records 2.07 µs conservative wall-clock here), asserts `< NUKA_PERF_GATE_US` only on `NUKA_PERF_VALIDATION_GPU` — owner runs it on the 5080. |
| 2 | Energy drift < 2 % over 1000 steps | **COVERED** — passive-dynamics invariant, asserted at `energy_drift_rel ≤ 0.02` in the V01 foundation pipeline + smoke configs (`double_pendulum`/`single_falling_box`/`two_body_collision`); V2 invariant system in `src/core/diagnostics/invariants*`. (Driven locomotion is not energy-conserving by design.) |
| 3 | 4096-env Go2 PPO converges to stable walking gait | **DONE (qualified)** — the 4096-env PPO **converged to a command-conditioned walking gait** (250 epochs, `last_mean_rewards`=18.4). **Root cause of the earlier collapse was a found BUG, not hyperparameters:** the gym action space was declared at ±100 (`ACTION_CLIP`), but rl_games (`clip_actions=True`) *rescales* the policy's [-1,1] output onto the action-space bounds (`rescale_actions`: `a·(high−low)/2`), so ±100 = a **100× amplification** (measured `mean\|a\|=64`) → joints slam → robot tips in ~21 steps → all-negative reward → `only_positive_rewards` zeros it → value collapse → PPO dies. Fix: declare the space at ±1 (`go2_obs.ACTION_SPACE_LIMIT=1.0`) so the rescale is the identity. Config also aligned to the official rsl_rl recipe (`entropy_coef` 0→0.01, net [512,256,128] elu, lr 1e-3, base-contact-only termination). **Independently re-verified** (controller, hand-rebuilt deterministic-μ forward + correct RunningMeanStd norm, own env instance) — command sweep `cmd vx → vx_body`, 0 terminations (220 deterministic-μ steps × 64 envs per command): **−0.5 → −0.462 · 0.0 → +0.002 · +0.5 → +0.467 · +1.0 → +1.032** (monotonic, sign-correct; the **cmd-0 → 0 stand-still rules out a degenerate forward-walker**); tilt 2–4°. *Qualified:* tracking is **approximate** (small under/overshoot at the rails) and `bounds_loss` is untidy (~35, μ leans on the clamp) — both honest, neither breaks the gait. |
| 4 | PyTorch `autograd.Function` skeleton wired (forward-only) | **DONE** — `python/nuka/autograd.py`; caller-facing signature locked, backward = zero stub. |
| 5 | V1 Featherstone oracle passing (carry-forward) | **CARRY-FORWARD** — 2 pre-existing ABA-vs-MJX oracle gaps (`FeatherstoneOracle.RandomSampleGoldensMatchCudaAba`, `V01…Phase6`) remain, orthogonal to v0.3 work (documented since sim-val #43). |
| 6 | Demo: 4096-env Go2 locomotion video | **DONE (in-house policy — unifies #3+#6)** — `examples/demo/go2_demo_capture_inhouse.py` renders a 16-env grid (MP4, `out/go2_demo/`, GIF `docs/media/go2_walk_4096env.gif`) driven by the **in-house from-scratch-PPO-trained converged policy** (`go2_walk_randomcmd_ep250.pth`, exit #3). The grid is a forward-speed ramp `cmd vx ∈ [0,1]` showcasing **command-conditioning**: corr(cmd_vx, vx_world)=**+0.99**, env0 (cmd 0) stands still, faster commands walk proportionally faster (cmd 0.5→~0.42, 1.0→~0.87), all 16 upright (max tilt 5.8°). The video now demonstrates the in-house S1 achievement, NOT an external replay. (The external-PR#62 capture `go2_demo_capture.py` is kept for reference.) |
| 7 | Quarterly external output (§5.1) | **OWNER** — publication is an owner action; this doc + the retrospective are the source material. |

## 3. Owner-deferred (not achievable on this box; hand off with docs)

- **§7 perf gate (< 1 ms @ 4090)** — owner validates on RTX 4090 (or the supplementary < 1.3 ms @ 4096 on RTX 5080 with Nsight). This box records the relative RTX-4000-Ada baseline only.
- **External publication (#7).**
- **Multi-iteration gait tuning toward exit #3**, IF the in-progress iteration hits a wall — the owner decides whether to invest further reward-shaping / curriculum / contact-force exposure.

## 4. Key engineering wins (the real S1 deliverable)

- **4096-env Go2 PPO converges to a command-conditioned walking gait** (exit #3) — the
  anchor deliverable. The blocker was a found rl_games action-space rescale bug (±100 →
  100× amplification), not a substrate or hyperparameter wall; fix is a one-line
  action-space declaration (`ACTION_SPACE_LIMIT=1.0`) + recipe alignment. Converged
  policy independently re-verified (see #3). Checkpoints: `out/go2_policy/` (gitignored,
  durable on this box) — `go2_walk_randomcmd_ep250.pth` (shipped random-curriculum
  config) + `go2_walk_fixedcmd_best.pth` (fixed-[0.5,0,0] single-speed control).
- 4096-env Go2 RL pipeline: engine → C ABI → nanobind/DLPack zero-copy → `NukaGymEnv` → rl_games, running at scale, **D1-deterministic**, no NaN.
- **Perf track (owner re-open):** opt-in **CUDA-graph step replay** (`StepGraph`, bit-exact to `Step()` by a byte-compare ctest), **D1/D2 determinism toggle + C ABI** (`nuka_world_desc_t.determinism`; D1 byte-exact default, D2 opt-in weak), and the **parameterized perf-gate regression test** (Task 3.1.6, records-and-skips). All D1-safe with zero change to `Step()`'s FP sequence; full suite green except the 2 pre-existing orthogonal oracle fails. Median per-env-step **≈ 0.92 µs** confirmed.
- Engine per-env **reset** primitive (`nuka_world_reset`/`reset_envs`, D1, masked autoreset) + authoritative **`BASE_POSE`** view (un-lagged, correct post-reset).
- legged_gym Go2 reward **ported** (learnable in isolation: per-step ~1e-3, tracking-dominated), command curriculum, post-reset obs.
- On-GPU 48-dim obs **bit-exact** to the proven numpy oracle; structural joint permutation.
- #43 implicit joint damping → soft-gain policies walk at native dt (5× fewer physics steps).

## 5. Regen / handoff commands

```bash
# Python wheel + env
cd python && /root/miniconda3/envs/nuka-v03/bin/pip install -e . --no-build-isolation
/root/miniconda3/envs/nuka-v03/bin/python -m pytest python/tests/ -q     # 50 pass
# (test_ppo_loop_regression.py = the p04-T2 guards locking the exit-#3 action-space fix
#  + a D1 byte-compare on the masked-autoreset teleport branch)

# Demo video (proven policy, exit #6)
CUDA_VISIBLE_DEVICES=0 examples/demo/render_video.sh                      # -> out/go2_demo/*.mp4

# Train (the SAME script does smoke + full). Shipped cfg is recipe-faithful + LEARNS;
# converges to a command-conditioned walk by ~250 epochs (last_mean_rewards ~18).
CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py --smoke
CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py --num_actors 4096 --max_epochs 1500

# Engine + determinism (11/11 pass; no rebuild needed for the python-only p04 fix)
cmake --build build-cuda128 -j && (cd build-cuda128 && ctest -R 'Reset|Determinism|Go2PdStanding|BasePoseView')
```

### Convergence re-verification (exit #3) — the commit-gate eval

The deterministic command-sweep eval the controller used to confirm exit #3 (loads the
shipped checkpoint, hand-rebuilds the μ forward + RunningMeanStd norm, pins the command
per phase, reads body-frame `vx` from the env): `/tmp/nuka_eval_ckpt.py` (throwaway;
the regen recipe is in §6 below). Decisive numbers in the #3 row above — the **cmd-0 →
+0.002** stand-still is what rules out a degenerate always-forward policy.

## 6. Remaining work + follow-ups

- **Regression test for the full RL loop** (smoke-train perf/learning regression) — p04-T2.
- **p04 phase-end consolidated review + v0.3 close checklist + retrospective** (§5.1 source) — p04-T3.
- **(DONE — owner re-open directive) Regenerated the demo video (#6) from the in-house
  converged policy** (`go2_walk_randomcmd_ep250.pth`) instead of the external PR#62
  `motion.pt` — unifies #3+#6 into a single from-scratch artifact
  (`examples/demo/go2_demo_capture_inhouse.py`; command-conditioned forward-speed-ramp grid,
  corr 0.99). The README now showcases this GIF.
- **(Follow-up) Tracking sharpening + `bounds_loss` tidy** — more epochs would tighten the
  approximate tracking; a `bounds_loss_coef` 1e-4→~1e-3 bump would pull μ off the ±1 rails.
  Deliberately not tuned at the finish line.

---

*Honesty notes: (1) exit #3 is **converged but qualified** — command-conditioned walking with
**approximate** tracking and an untidy (benign) `bounds_loss`; it genuinely walks and the
cmd-0 stand-still rules out a degenerate forward-only policy, but it is not a polished gait.
(2) The demo video (#6) is now driven by the **in-house converged policy**
(`go2_walk_randomcmd_ep250.pth`), unifying #3+#6 — it shows command-conditioned walking
(corr 0.99) rendered as a debug-skeleton grid (honest: a schematic line render, not a
photoreal mesh). (3) The CUDA-graph path is bit-exact but its launch-overhead win is
within-noise at 4096-env production batch (compute-bound); its value is the
determinism-preserving mechanism, not a speedup — see the perf report §6.*
