#!/usr/bin/env python
"""A5b -- per-epoch CATCH-RATE eval observer for the H1 cup-grasp PPO run.

THE QUESTION A5b answers: does PPO LEARN to grasp on this engine, i.e. does the
policy's CATCH RATE climb OFF ZERO?  Return is NOT a faithful proxy (the cup's
policy-independent TRANSIT through the catch plane can farm reward even with the
force-closure gate), so the metric is the DETERMINISTIC held-to-truncation catch
fraction of the CURRENT policy on the jittered (training) reset distribution.

MECHANISM (advisor-checked):
  * A SEPARATE eval H1GraspEnv (NOT the training env) so the eval's reset/step
    NEVER perturbs the PPO experience distribution. rl_games carries ``self.obs``
    across epochs and resets the env only once at train() start; resetting the
    TRAINING env mid-run would desync it. The eval env shares the training env's
    nuka.Device (device= -> _owns_device=False, so close() won't free the shared
    device) but owns an INDEPENDENT GraspWorld (proven to coexist on one device).
  * The model is stateless across forward passes, so we share it. Deterministic
    action = ``algo.get_action_values({'obs': obs})['mus'].clamp(-1,1)`` -- this is
    the policy MEAN (mu), NOT a sample, and rl_games' own method applies the input
    normalizer in EVAL mode (model.eval()), so the eval obs does NOT pollute the
    running obs-normalizer stats. Training re-sets train() next epoch.
  * Catch measurement reuses test_h1_grasp_env.rollout_jittered_episode's logic
    VERBATIM: single-episode, alive-masked, ``caught = trunc & alive`` (held to
    truncation). FIXED eval seed (123) every epoch so the catch delta is the
    policy's, not cup variance.

OUTPUT: every K epochs, prints ``[catch-eval] epoch=.. catch=.. mean_return=..``
to stdout AND appends a CSV row ``epoch,catch,mean_return`` to the log file
(NUKA_GRASP_CATCH_LOG env var, default runs/grasp_catch_eval.csv) the controller
can poll while the background train runs.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from train_go2_ppo import StdoutAlgoObserver  # noqa: E402

# repo-root so ``import nuka`` + the repo-relative cup asset resolve.
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


DEFAULT_EVAL_ENVS = 256
DEFAULT_EVAL_EVERY = 5
DEFAULT_EVAL_SEED = 123
DEFAULT_LOG = os.path.join(_REPO_ROOT, "runs", "grasp_catch_eval.csv")


@torch.no_grad()
def deterministic_catch_rate(algo, eval_env, *, seed: int):
    """Run the CURRENT policy (mu only, clamped) through ONE jittered episode of
    ``eval_env`` and return (caught_rate, mean_return, all_finite).

    Mirrors test_h1_grasp_env.rollout_jittered_episode: single episode from
    reset(seed=...), alive-masked reward accrual, caught = held-to-truncation.
    """
    DEV = torch.device("cuda")
    obs, _ = eval_env.reset(seed=seed)
    ret = torch.zeros(eval_env.num_envs, device=DEV)
    alive = torch.ones(eval_env.num_envs, dtype=torch.bool, device=DEV)
    caught = torch.zeros(eval_env.num_envs, dtype=torch.bool, device=DEV)
    all_finite = True
    horizon = eval_env.max_episode_length
    for _ in range(horizon):
        # Deterministic policy MEAN (mu), clamped to the action box. Uses the
        # algo's own preproc (input-normalizer in eval mode) so eval obs does NOT
        # update the running obs stats.
        res = algo.get_action_values({"obs": obs})
        a = res["mus"].clamp(-1.0, 1.0)
        obs, reward, term, trunc, _info = eval_env.step(a)
        ret += reward * alive.float()
        caught = caught | (trunc & alive)
        alive = alive & (~term)
        if not (torch.isfinite(obs).all() and torch.isfinite(reward).all()):
            all_finite = False
    return float(caught.float().mean()), float(ret.mean()), all_finite


class CatchRateAlgoObserver(StdoutAlgoObserver):
    """StdoutAlgoObserver (per-epoch loss/reward dump) + a periodic DETERMINISTIC
    catch-rate eval on a SEPARATE env. Wires the eval at after_init (the model +
    vec_env exist by then) and runs it in after_print_stats every K epochs."""

    def __init__(self, eval_envs: int = DEFAULT_EVAL_ENVS,
                 eval_every: int = DEFAULT_EVAL_EVERY,
                 eval_seed: int = DEFAULT_EVAL_SEED,
                 log_path: str | None = None):
        super().__init__()
        self._eval_envs = int(eval_envs)
        self._eval_every = int(eval_every)
        self._eval_seed = int(eval_seed)
        self._log_path = log_path or os.environ.get(
            "NUKA_GRASP_CATCH_LOG", DEFAULT_LOG)
        self._eval_env = None
        self._algo = None

    def after_init(self, algo):
        super().after_init(algo)
        self._algo = algo
        # Build the SEPARATE eval env, sharing the training env's nuka.Device.
        from nuka.tasks.h1_grasp import H1GraspEnv
        train_env = algo.vec_env.env  # NukaVecEnv.env == H1GraspEnv
        shared_device = train_env._device
        self._eval_env = H1GraspEnv(self._eval_envs, device=shared_device)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        with open(self._log_path, "w") as fh:
            fh.write("epoch,catch,mean_return,finite\n")
        print(f"[catch-eval] observer attached: eval_envs={self._eval_envs} "
              f"every={self._eval_every} seed={self._eval_seed} "
              f"log={self._log_path}", flush=True)
        # Epoch-0 baseline (the RANDOM-init policy catch rate -- the floor we
        # measure the climb against).
        self._run_eval(0)

    def _run_eval(self, epoch_num):
        if self._eval_env is None or self._algo is None:
            return
        was_training = self._algo.model.training
        try:
            catch, mret, finite = deterministic_catch_rate(
                self._algo, self._eval_env, seed=self._eval_seed)
        finally:
            # Restore the model's train/eval mode (get_action_values left it eval).
            self._algo.model.train(was_training)
        print(f"[catch-eval] epoch={epoch_num} catch={catch:.4f} "
              f"mean_return={mret:.3f} finite={finite}", flush=True)
        with open(self._log_path, "a") as fh:
            fh.write(f"{epoch_num},{catch:.4f},{mret:.4f},{int(finite)}\n")

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        if epoch_num % self._eval_every == 0:
            self._run_eval(epoch_num)
