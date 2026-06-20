#!/usr/bin/env python
"""Fine-grained per-step termination diagnostic for the Go2 handstand policy.

Drives the trained S2 policy deterministically for a few dozen control steps and,
each step, prints the termination breakdown (flip / collapse / nonfinite /
runaway) + the raw orientation/height so we can see EXACTLY why the policy dies
immediately. NO autoreset confusion -- we read the env's own compute_terminated
predicate components by reimplementing them on the live engine state.
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import nuka  # noqa
from nuka.tasks.go2_handstand import make_env, COLLAPSE_FLOOR_Z
from nuka.tasks import go2_obs as G
from go2_handstand_eval import load_actor


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stage", default="S2")
    p.add_argument("--theta-cmd", type=float, default=90.0)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero-action", action="store_true",
                   help="Ignore policy; hold the IC PD target (action=0) to test "
                        "whether the IC itself is stable/terminates.")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = make_env(args.num_envs, stage=args.stage, theta_cmd_obs=True, seed=args.seed)
    dev = env._torch_device; n = env.num_envs
    forward, ep, rew = load_actor(args.ckpt, env._obs_dim, G.GO2_ACTION_DIM, dev)
    print(f"[diag] ckpt epoch={ep} rew={rew} stage={args.stage} "
          f"zero_action={args.zero_action} n={n}", flush=True)

    obs, _ = env.reset(seed=args.seed)
    env._theta_cmd_deg[:] = float(args.theta_cmd)
    obs = env._append_theta(obs[:, :G.GO2_OBS_DIM])

    front_slot = env._front_foot_slot; rear_slot = env._rear_foot_slot
    alive = torch.ones(n, dtype=torch.bool, device=dev)

    # Read the IC state BEFORE any step.
    b = env._obs
    pg0 = b.projected_gravity(); bz0 = b.base_pos()[:, 2]
    print(f"[diag] IC (pre-step): pgx med={float(pg0[:,0].median()):+.3f} "
          f"pgz med={float(pg0[:,2].median()):+.3f} base_z med={float(bz0.median()):.3f} "
          f"base_z min={float(bz0.min()):.3f}", flush=True)

    print(f"[diag] {'step':>4} {'newdead':>7} {'alive':>5} | "
          f"{'pgx':>6} {'pgz':>6} {'bz':>6} | "
          f"{'flip':>4} {'coll':>4} {'nonf':>4} {'runa':>4} | "
          f"{'qdmax':>6} {'rearFz':>7} {'frntFz':>7}", flush=True)

    for k in range(args.steps):
        if args.zero_action:
            act = torch.zeros(n, G.GO2_ACTION_DIM, device=dev)
        else:
            act = forward(obs)
        obs_next, reward, terminated, truncated, info = env.step(act)
        env._theta_cmd_deg[:] = float(args.theta_cmd)

        b = env._obs
        pg = b.projected_gravity()
        base_z = b.base_pos()[:, 2]
        q = b.q_urdf(); qd = b.qd_urdf()
        rear_fz = env._wrench_view[:, rear_slot, 2].abs().sum(dim=1)
        front_fz = env._wrench_view[:, front_slot, 2].abs().sum(dim=1)

        # Reproduce the env termination predicate components (NOSE-DOWN convention).
        flipped = (pg[:, 2] > 0.7) & (pg[:, 0] > 0.0)
        collapsed = base_z < COLLAPSE_FLOOR_Z
        nonfinite = (~torch.isfinite(b.base_pos()).all(dim=1)
                     | ~torch.isfinite(q).all(dim=1) | ~torch.isfinite(qd).all(dim=1))
        runaway = qd.abs().amax(dim=1) > 50.0

        new = terminated & alive
        # count predicate among newly-dead.
        def cnt(mask):
            return int((mask & new).sum())
        alive = alive & (~terminated)

        print(f"[diag] {k:4d} {int(new.sum()):7d} {int(alive.sum()):5d} | "
              f"{float(pg[:,0].median()):+6.3f} {float(pg[:,2].median()):+6.3f} "
              f"{float(base_z.median()):6.3f} | "
              f"{cnt(flipped):4d} {cnt(collapsed):4d} {cnt(nonfinite):4d} {cnt(runaway):4d} | "
              f"{float(qd.abs().amax(dim=1).median()):6.1f} "
              f"{float(rear_fz.median()):7.1f} {float(front_fz.median()):7.1f}",
              flush=True)
        obs = obs_next
        if int(alive.sum()) == 0 and k > 2:
            print("[diag] all envs dead -- stopping.", flush=True)
            break

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
