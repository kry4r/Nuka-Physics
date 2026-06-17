#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# L1 ACCEPTANCE -- drive the TRAINED go2 height-scan policy closed-loop on the
# GENERAL contact path (contact_family=1 -> PairDriven + baked heightfield
# collidable) and A/B it against the legacy FusedFoot path, BEFORE FusedFoot is
# deleted + the goldens re-baselined. Owner-endorsed gate: "用训练好的 go2 策略在
# 通用路径地形上跑一次单狗 rollout, 验证 (a) 策略仍能走 (b) 下沉是否可见".
#
# Faithful inference: reuse the rl_games PpoPlayer (obs normalizer + net + the
# deterministic get_action) exactly as examples/sim_val/go2_terrain_eval.py does,
# the PROVEN driver for this checkpoint. We recreate the env per (terrain-type)
# because the general path bakes ONE model-level heightfield at creation; FUSED
# pins the type at runtime. Both pin the SAME difficulty (1.0 == the baked full
# step geometry) so the comparison is apples-to-apples. The policy was trained on
# FusedFoot, so a modest general-vs-fused gap is EXPECTED (sim-to-sim, closed at
# the post-L1 retrain) -- the bar is: WALKS + survives + does NOT tunnel (穿模).
#
# Run (conda nuka-v03, GPU 1):
#   CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64 \
#   python examples/demo/l1_general_path_rollout.py \
#     --checkpoint /root/Nuka-Physics/out/go2_terrain_hs/nn/last_go2_terrain_hs_ep_1500_rew_10.463228.pth
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Neutralize the skbuild editable finder -> load THIS worktree's nuka + fresh ext.
_WT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.meta_path = [
    f for f in sys.meta_path
    if type(f).__name__ != "ScikitBuildRedirectingFinder"
]
for _m in [m for m in sys.modules if m == "nuka" or m.startswith("nuka.")]:
    del sys.modules[_m]
sys.path.insert(0, _WT)

import torch  # noqa: E402
import yaml  # noqa: E402
import nuka  # noqa: E402
assert nuka.__file__.startswith(_WT), f"wrong nuka pkg: {nuka.__file__}"

_HERE = os.path.dirname(os.path.abspath(__file__))
# The training yaml (network arch + env_config). Worktree copy if present, else main.
_CFG_CANDIDATES = [
    os.path.join(_WT, "..", "examples", "training", "go2_ppo_cfg.yaml"),
    "/root/Nuka-Physics/examples/training/go2_ppo_cfg.yaml",
]
_CFG = next(p for p in _CFG_CANDIDATES if os.path.exists(p))

TYPE_NAMES = {0: "Flat", 1: "Stairs", 2: "Pit", 3: "Boxes"}


def build_player(checkpoint, cfg_path):
    """Deterministic rl_games PpoPlayer from the training yaml + checkpoint (loads
    the net AND the obs RunningMeanStd). Mirrors go2_terrain_eval.build_player."""
    import nuka.rl_games  # noqa: F401  (registers nuka_go2)
    import nuka.tasks.go2_obs as G
    from gymnasium import spaces
    from rl_games.torch_runner import Runner

    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    params = raw["params"] if "params" in raw else raw
    # The height-scan policy has obs 48+187=235; size the env_info obs_space to the
    # checkpoint's REAL obs dim (from its running_mean_std) so the player builds a
    # matching-input network (the canonical eval stubbed 48 = blind policies only).
    _sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _rm = _sd["model"]
    _key = "running_mean_std.running_mean"
    _key = _key if _key in _rm else "_orig_mod." + _key
    ckpt_obs_dim = int(_rm[_key].numel())
    obs_space = spaces.Box(low=-G.OBS_CLIP, high=G.OBS_CLIP,
                           shape=(ckpt_obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-G.ACTION_SPACE_LIMIT, high=G.ACTION_SPACE_LIMIT,
                           shape=(G.GO2_ACTION_DIM,), dtype=np.float32)
    params["config"]["env_info"] = {
        "observation_space": obs_space, "action_space": act_space,
        "agents": 1, "value_size": 1,
    }
    runner = Runner()
    runner.load_config(params)
    player = runner.create_player()
    player.restore(checkpoint)
    player.has_batch_dimension = True
    player.is_deterministic = True
    player.is_tensor_obses = True
    return player, raw


def build_env(num_envs, ttype, difficulty, family, raw, episode_length_s, seed):
    from nuka.tasks.go2_locomotion import make_env

    tcfg = dict(raw["params"]["config"]["env_config"]["terrain"])
    tcfg["enable"] = True
    # The height-scan obs (48 -> 235) is yaml-default OFF (training enabled it via
    # --height-scan); force it ON so the obs matches the 235-input checkpoint.
    hs = dict(tcfg.get("height_scan", {}))
    hs["enable"] = True
    tcfg["height_scan"] = hs
    ec = raw["params"]["config"]["env_config"]
    env = make_env(
        num_envs,
        terrain=tcfg,
        episode_length_s=episode_length_s,
        termination_height=ec["termination_height"],
        termination_tilt_deg=ec["termination_tilt_deg"],
        contact_family=int(family),
        heightfield_terrain_type=int(ttype),   # general path bakes THIS type.
        seed=seed,
    )
    env._curriculum.update_on_done = lambda *a, **k: None  # freeze curriculum.
    env._curriculum.terrain_type[:] = int(ttype)
    env._curriculum.difficulty[:] = float(difficulty)
    return env


def rollout(env, player, steps, settle, device):
    """Closed-loop rollout. Accumulate per-completed-episode survival/fall/speed/
    path AND per-step (alive) base clearance + worst foot-link sink above the
    local terrain surface (the 穿模 indicator)."""
    obs, _ = env.reset()
    n = env.num_envs
    foot_slot = env._foot_link_slot

    ep_len = torch.zeros(n, dtype=torch.long, device=device)
    path_len = torch.zeros(n, device=device)
    speed_sum = torch.zeros(n, device=device)
    prev_xy = env._obs.base_pos()[:, :2].detach().clone()
    done_lens, done_falls, done_speeds, done_path = [], [], [], []

    win = torch.zeros(n, device=device)
    sum_clear = torch.zeros(n, device=device)
    sum_sink = torch.zeros(n, device=device)
    min_clear = torch.full((n,), 1e9, device=device)
    min_sink = torch.full((n,), 1e9, device=device)

    for k in range(steps):
        with torch.no_grad():
            action = player.get_action(obs, is_deterministic=True)
        # rl_games (clip_actions=True) clips the policy action to the action space
        # [-1,1] BEFORE env.step; we call env.step directly, so replicate the clip.
        action = action.clamp(-1.0, 1.0)
        obs, _r, terminated, truncated, _ = env.step(action)
        done = terminated | truncated

        bp = env._obs.base_pos()
        cur_xy = bp[:, :2].detach()
        path_len += (cur_xy - prev_xy).norm(dim=1) * (~done).to(torch.float32)
        ep_len += 1
        speed_sum += env._obs.base_lin_vel()[:, 0].abs()

        # clearance / sink above the local analytic surface (same sampler both paths).
        surf_b = env._terrain_surface(cur_xy[:, 0:1], cur_xy[:, 1:2])[:, 0]
        clear = bp[:, 2] - surf_b
        fpose = env._pose_view[:, foot_slot, :]
        surf_f = env._terrain_surface(fpose[:, :, 0], fpose[:, :, 1])
        foot_clear = (fpose[:, :, 2] - surf_f).min(dim=1).values
        alive = ~done
        if k >= settle:
            mw = alive.to(torch.float32)
            win += mw
            sum_clear += clear * mw
            sum_sink += foot_clear * mw
            min_clear = torch.where(alive, torch.minimum(min_clear, clear), min_clear)
            min_sink = torch.where(alive, torch.minimum(min_sink, foot_clear), min_sink)

        if bool(done.any()):
            d = done
            done_lens.append(ep_len[d].float().cpu())
            done_falls.append(terminated[d].float().cpu())
            done_speeds.append((speed_sum[d] / ep_len[d].clamp(min=1).float()).cpu())
            done_path.append(path_len[d].cpu())
            ep_len[d] = 0; path_len[d] = 0.0; speed_sum[d] = 0.0
            cur_xy = env._obs.base_pos()[:, :2].detach()
        prev_xy = cur_xy.clone()

    def cm(lst):
        if not lst:
            return float("nan"), 0
        t = torch.cat(lst)
        return float(t.mean()), int(t.numel())

    mean_len, n_ep = cm(done_lens)
    fall, _ = cm(done_falls)
    speed, _ = cm(done_speeds)
    path, _ = cm(done_path)
    c = win > 0
    return dict(
        n_ep=n_ep, ep_len=mean_len, fall=fall, speed=speed, path=path,
        max_ep=env.max_episode_length,
        clear_med=float((sum_clear[c] / win[c]).median()) if c.any() else float("nan"),
        clear_min=float(min_clear[min_clear < 1e8].min()) if (min_clear < 1e8).any() else float("nan"),
        sink_med=float((sum_sink[c] / win[c]).median()) if c.any() else float("nan"),
        sink_min=float(min_sink[min_sink < 1e8].min()) if (min_sink < 1e8).any() else float("nan"),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cfg", default=_CFG)
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--settle", type=int, default=30)
    p.add_argument("--episode_length_s", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    print(f"[L1-accept] ckpt={os.path.basename(args.checkpoint)} cfg={args.cfg}\n"
          f"[L1-accept] N={args.num_envs} steps={args.steps}", flush=True)
    player, raw = build_player(args.checkpoint, args.cfg)

    # Cells: Flat (basic walk) + Stairs@1.0 (the baked full-step geometry).
    cells = [(0, 0.0), (1, 1.0)]
    SINK = -0.05  # vproof no-穿模 bar.
    overall_ok = True
    for (tt, dd) in cells:
        print(f"\n--- terrain={TYPE_NAMES[tt]} difficulty={dd} ---", flush=True)
        res = {}
        for fam in (0, 1):
            env = build_env(args.num_envs, tt, dd, fam, raw,
                            args.episode_length_s, args.seed)
            r = rollout(env, player, args.steps, args.settle, device)
            env.close()
            res[fam] = r
            tag = "FUSED  " if fam == 0 else "GENERAL"
            surv = 100.0 * r["ep_len"] / r["max_ep"]
            print(f"  {tag}: survive={surv:5.1f}% fall={100*r['fall']:4.1f}% "
                  f"fwd={r['speed']:.3f}m/s path={r['path']:.2f}m | "
                  f"base-clear med={r['clear_med']:+.3f} min={r['clear_min']:+.3f} | "
                  f"foot-sink med={r['sink_med']:+.3f} min={r['sink_min']:+.3f}",
                  flush=True)
        g, f = res[1], res[0]
        walks = g["path"] > 0.3 and g["speed"] > 0.05
        survives = g["ep_len"] >= 0.5 * f["ep_len"] or (g["ep_len"] / g["max_ep"]) >= 0.5
        no_tunnel = (g["clear_min"] > SINK) and (g["sink_min"] > SINK)
        ok = walks and survives and no_tunnel
        overall_ok = overall_ok and ok
        print(f"  => GENERAL walks={walks} survives={survives} no-穿模={no_tunnel} "
              f"=> {'ACCEPT' if ok else 'REVIEW'}", flush=True)

    print(f"\n[L1-accept] OVERALL: {'*** ACCEPT (general path is a sound walk surface) ***' if overall_ok else '*** REVIEW ***'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
