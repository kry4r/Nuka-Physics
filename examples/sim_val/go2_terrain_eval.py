#!/usr/bin/env python
"""Deterministic terrain-sweep EVAL for a trained Go2 terrain policy.

Loads an rl_games checkpoint into a deterministic (mu) player and rolls the
Go2LocomotionEnv out under the FREE-ROAM command curriculum, but with the per-env
terrain TYPE + DIFFICULTY PINNED (curriculum promotion DISABLED) so we can measure
the policy's competence at a fixed (type, difficulty) grid:

  * Stairs (1) / Pit (2) / Boxes (3)  x  difficulty {0.0, 0.5, 0.8, 1.0}.

Per cell over ~`--steps` control steps it reports, per completed episode:
  * mean ep_len           (survival; max_episode_length = episode_length_s/dt)
  * fall rate             (# terminated / # done -- a "fell" not a time-limit trunc)
  * mean forward speed    (body-frame +x linear velocity, m/s)
  * mean NET displacement (||base_xy_done - start_xy||)
  * mean PATH length      (sum of per-step ||delta base_xy|| over the episode)

This DIRECTLY answers (a) can it walk? (survives flat), (b) did the curriculum
reach higher steps? (survives 0.15 m or only flat), and (c) the gate-blocker proof
that NET displacement << PATH length under free-roam (so the old net-disp gate
rarely fired).

Single GPU only. Run as::

  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=... PYTHONPATH=/root/Nuka-Physics/python \
    python examples/sim_val/go2_terrain_eval.py \
      --checkpoint out/go2_terrain_policy/nn/go2_terrain_policy.pth
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = os.path.join(_HERE, "..", "training", "go2_ppo_cfg.yaml")

TYPE_NAMES = {1: "Stairs", 2: "Pit", 3: "Boxes"}


def build_player(checkpoint: str, cfg_path: str):
    """Build a deterministic rl_games PpoPlayerContinuous from the training yaml
    + restore the checkpoint (loads the policy net AND the input RunningMeanStd).

    We do NOT let the player construct the env (it would spin up a second world);
    we only use its network + obs normalizer + get_action(mu). So we stub the
    env-info on the player after construction with our env's spaces.
    """
    import nuka.rl_games  # noqa: F401  (registers nuka_go2)
    import nuka.tasks.go2_obs as G
    from gymnasium import spaces
    from rl_games.torch_runner import Runner

    with open(cfg_path, "r") as fh:
        raw = yaml.safe_load(fh)
    params = raw["params"] if "params" in raw else raw
    # Inject env_info directly so the player does NOT spin up a second Go2 world
    # (BasePlayer skips env creation when config['env_info'] is set). The spaces
    # match Go2LocomotionEnv's single-env spaces (obs 48, action 12 in [-1,1]).
    obs_space = spaces.Box(low=-G.OBS_CLIP, high=G.OBS_CLIP,
                           shape=(G.GO2_OBS_DIM,), dtype=np.float32)
    act_space = spaces.Box(low=-G.ACTION_SPACE_LIMIT, high=G.ACTION_SPACE_LIMIT,
                           shape=(G.GO2_ACTION_DIM,), dtype=np.float32)
    params["config"]["env_info"] = {
        "observation_space": obs_space,
        "action_space": act_space,
        "agents": 1,
        "value_size": 1,
    }
    runner = Runner()
    runner.load_config(params)
    player = runner.create_player()
    player.restore(checkpoint)
    player.has_batch_dimension = True
    player.is_deterministic = True
    # The eval feeds raw cuda tensors to get_action; flag them as tensor obs so the
    # player's _preproc_obs keeps them on-GPU.
    player.is_tensor_obses = True
    return player


def make_eval_env(num_envs: int, episode_length_s: float, seed: int):
    """Build a Go2LocomotionEnv with terrain ON, then DISABLE curriculum
    promotion so per-env (type, difficulty) stays whatever we pin it to."""
    from nuka.tasks.go2_locomotion import make_env

    # terrain config = the yaml defaults (so geometry matches training cook).
    with open(_CFG, "r") as fh:
        raw = yaml.safe_load(fh)
    tcfg = dict(raw["params"]["config"]["env_config"]["terrain"])
    tcfg["enable"] = True

    env = make_env(
        num_envs,
        terrain=tcfg,
        episode_length_s=episode_length_s,
        termination_height=raw["params"]["config"]["env_config"]["termination_height"],
        termination_tilt_deg=raw["params"]["config"]["env_config"]["termination_tilt_deg"],
        seed=seed,
    )
    # FREEZE the curriculum: make update_on_done a no-op so promote/demote never
    # changes the pinned per-env type/difficulty during the eval rollout.
    env._curriculum.update_on_done = lambda *a, **k: None  # type: ignore[assignment]
    return env


def pin_terrain(env, terrain_type: int, difficulty: float) -> None:
    """Pin EVERY env to a fixed terrain type + difficulty (the curriculum tensors
    the reset reads), then full-reset so the engine fields + spawn z are rewritten
    from the pinned assignment."""
    cur = env._curriculum
    cur.terrain_type[:] = int(terrain_type)
    cur.difficulty[:] = float(difficulty)


def rollout_cell(env, player, terrain_type, difficulty, steps, device):
    """Roll ~`steps` control steps at a pinned (type, difficulty). Accumulate
    per-episode survival/fall/speed/net-disp/path-len over the COMPLETED episodes
    (using the env autoreset done mask). Returns a metrics dict."""
    pin_terrain(env, terrain_type, difficulty)
    obs, _ = env.reset()

    n = env.num_envs
    # Per-env running accumulators (reset at each episode start = each done).
    ep_len = torch.zeros(n, dtype=torch.long, device=device)
    path_len = torch.zeros(n, device=device)
    speed_sum = torch.zeros(n, device=device)
    start_xy = env._curriculum.start_xy.clone()  # spawn xy the reset just set.
    prev_xy = env._obs.base_pos()[:, :2].detach().clone()

    # Completed-episode aggregates.
    done_lens, done_falls, done_speeds, done_net, done_path = [], [], [], [], []

    for _ in range(steps):
        with torch.no_grad():
            action = player.get_action(obs, is_deterministic=True)
        # base_pos BEFORE the step's autoreset clobbers done envs -> the terminal
        # xy. compute it from a pre-step read after the step via env internals:
        obs, reward, terminated, truncated, _ = env.step(action)

        cur_xy = env._obs.base_pos()[:, :2].detach()
        # Per-step path increment (||delta xy||). For NON-done envs this is the live
        # motion; for done envs the post-autoreset pose teleported, so we must use
        # the TERMINAL xy the env captured (env._last_base_xy) before autoreset.
        term_xy = env._last_base_xy  # (N,2) terminal xy captured pre-autoreset.
        done = terminated | truncated

        # Path increment uses terminal xy for done envs (true last step), live for others.
        step_xy = torch.where(done.unsqueeze(1), term_xy, cur_xy)
        path_len += (step_xy - prev_xy).norm(dim=1)
        ep_len += 1
        # Forward speed = body-frame +x lin vel magnitude this step.
        speed_sum += env._obs.base_lin_vel()[:, 0].abs()

        if bool(done.any()):
            d = done
            net = (term_xy - start_xy).norm(dim=1)
            done_lens.append(ep_len[d].float().cpu())
            done_falls.append(terminated[d].float().cpu())
            done_speeds.append((speed_sum[d] / ep_len[d].clamp(min=1).float()).cpu())
            done_net.append(net[d].cpu())
            done_path.append(path_len[d].cpu())
            # Reset accumulators for done envs; new start_xy = the just-set spawn.
            ep_len[d] = 0
            path_len[d] = 0.0
            speed_sum[d] = 0.0
            start_xy[d] = env._curriculum.start_xy[d]
            # After autoreset the live base_pos for done envs is the fresh spawn.
            cur_xy = env._obs.base_pos()[:, :2].detach()
        prev_xy = cur_xy.clone()

    def cat_mean(lst):
        if not lst:
            return float("nan"), 0
        t = torch.cat(lst)
        return float(t.mean()), int(t.numel())

    mean_len, n_ep = cat_mean(done_lens)
    fall_rate, _ = cat_mean(done_falls)
    mean_speed, _ = cat_mean(done_speeds)
    mean_net, _ = cat_mean(done_net)
    mean_path, _ = cat_mean(done_path)
    return dict(
        type=terrain_type, diff=difficulty, n_ep=n_ep,
        ep_len=mean_len, fall_rate=fall_rate, speed=mean_speed,
        net=mean_net, path=mean_path,
        max_ep_len=env.max_episode_length,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cfg", default=_CFG)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--episode_length_s", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"[eval] checkpoint={args.checkpoint} num_envs={args.num_envs} "
          f"steps={args.steps} ep_len_s={args.episode_length_s}", flush=True)

    player = build_player(args.checkpoint, args.cfg)
    env = make_eval_env(args.num_envs, args.episode_length_s, args.seed)
    print(f"[eval] max_episode_length = {env.max_episode_length} control steps "
          f"(survive_frac*max = {0.8*env.max_episode_length:.0f})", flush=True)

    grid_types = [1, 2, 3]
    grid_diffs = [0.0, 0.5, 0.8, 1.0]
    rows = []
    for tt in grid_types:
        for dd in grid_diffs:
            r = rollout_cell(env, player, tt, dd, args.steps, device)
            rows.append(r)
            print(f"  [{TYPE_NAMES[tt]:>6} d{dd:.2f}] "
                  f"n_ep={r['n_ep']:4d}  ep_len={r['ep_len']:7.1f}  "
                  f"fall={r['fall_rate']:.3f}  fwd_speed={r['speed']:.3f}  "
                  f"net={r['net']:.3f}m  path={r['path']:.3f}m", flush=True)

    print("\n" + "=" * 92)
    print(f"{'terrain':>8} {'diff':>5} {'n_ep':>5} {'ep_len':>8} {'survive%':>8} "
          f"{'fall%':>6} {'fwd m/s':>8} {'net m':>7} {'path m':>7} {'net/path':>8}")
    print("=" * 92)
    for r in rows:
        survive = 100.0 * r["ep_len"] / r["max_ep_len"]
        npr = (r["net"] / r["path"]) if r["path"] > 1e-6 else float("nan")
        print(f"{TYPE_NAMES[r['type']]:>8} {r['diff']:>5.2f} {r['n_ep']:>5d} "
              f"{r['ep_len']:>8.1f} {survive:>7.1f}% {100*r['fall_rate']:>5.1f}% "
              f"{r['speed']:>8.3f} {r['net']:>7.3f} {r['path']:>7.3f} {npr:>8.3f}")
    print("=" * 92)

    # NET-disp-vs-PATH-len proof aggregated over ALL cells (the gate-blocker).
    nets = np.array([r["net"] for r in rows if r["n_ep"] > 0])
    paths = np.array([r["path"] for r in rows if r["n_ep"] > 0])
    print(f"\n[net-vs-path proof] mean NET disp over all cells = {nets.mean():.3f} m  "
          f"mean PATH len = {paths.mean():.3f} m  "
          f"=> net/path = {nets.mean()/paths.mean():.3f}")
    old_gate = (nets > 3.0).mean() * 100.0
    new_gate = (paths > 5.0).mean() * 100.0
    print(f"[gate reachability]  OLD gate (net>3.0m): {old_gate:.0f}% of cells reach it  | "
          f"NEW gate (path>5.0m): {new_gate:.0f}% of cells reach it")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
