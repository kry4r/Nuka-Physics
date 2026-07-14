#!/usr/bin/env python3
"""Flat-ground achieved-speed / fall-rate check of the v8 walk policy at the training
rate vs the corridor harness rate -- isolates rate mismatch from corridor obstacles."""
import sys
from pathlib import Path

import numpy as np
import torch

import nuka  # noqa: F401
from nuka.tasks.bdx_locomotion import BdxWalkEnv
from nuka.tasks import bdx_obs as B

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "training"))
from bdx_walk_eval import load_actor  # noqa: E402

SCENE = str(REPO / "examples" / "scenes" / "bdx_stand.nks")
CKPT = ("/data/xtzhang25/_work/activate/out/bdx_walk_v8/nn/"
        "last_bdx_walk_v8_ep_1600_rew_541.5958.pth")
REFERENCE_PKL = str(REPO / "examples" / "assets" / "bdx" /
                    "polynomial_coefficients.pkl")


def run(dt, decim, label, steps=400, n=64, cmdx=0.15):
    env = BdxWalkEnv(SCENE, n, dt=dt, decimation=decim, fixed_command=True,
                     command=(cmdx, 0.0, 0.0), seed=7,
                     reference_pkl=REFERENCE_PKL)
    dev = env._dev
    fwd, ep, rew = load_actor(CKPT, B.BDX_OBS_DIM, B.BDX_ACTION_DIM, dev)
    obs, _ = env.reset(seed=7)
    env.command[:] = torch.tensor([cmdx, 0.0, 0.0], device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    ever_fell = torch.zeros(n, dtype=torch.bool, device=dev)
    sv = torch.zeros(n, device=dev)
    sz = torch.zeros(n, device=dev)
    cnt = torch.zeros(n, device=dev)
    for k in range(steps):
        mu = fwd(obs)
        obs, r, term, trunc, _ = env.step(mu)
        v = env._obs.base_lin_vel()[:, 0]
        z = env._obs.base_pos()[:, 2]
        m = alive
        cnt[m] += 1.0
        sv[m] += v[m]
        sz[m] += z[m]
        ever_fell |= term
        alive = alive & (~term)
    valid = cnt > 0
    avx = float((sv[valid] / cnt[valid]).mean())
    az = float((sz[valid] / cnt[valid]).mean())
    fr = float(ever_fell.float().mean())
    surv = float(alive.float().mean())
    ctrl_dt = dt * decim
    print(f"[rate {label}] dt={dt:.5f} decim={decim} ctrl_dt={ctrl_dt:.4f} cmdx={cmdx} "
          f"achieved_vx={avx:+.4f} stand_z={az:.3f} fall_rate={fr*100:.1f}% "
          f"surv={surv*100:.1f}% steps={steps}({steps*ctrl_dt:.1f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    run(0.005, 4, "flat_train200Hz")
    run(1.0 / 240.0, 5, "flat_corridor240Hz")
    print("RATECHECK DONE", flush=True)
