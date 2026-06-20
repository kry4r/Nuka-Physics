#!/usr/bin/env python
"""DETERMINISTIC eval of a TRAINED Go2 HANDSTAND policy (倒立) -- honest verdict.

Loads an rl_games checkpoint's ACTOR (the policy net + the obs running_mean_std
normalizer), drives the Go2HandstandEnv deterministically (action = mu, NO sigma
sampling, NO kinematic assist), and measures whether the policy ACHIEVES & HOLDS
an inverted handstand.

Unlike go2_handstand_probe.py (a scripted PD-authority probe that WRITES a base-
pitch assist), this drives the REAL trained policy closed-loop: reset seats the
stage IC (S2 = near-handstand pose, theta_cmd in [75,90]), we FORCE theta_cmd=90
on every env (the full-handstand command), then run the policy for several hundred
control steps. We track, PER ENV, the steps survived before the env's first
termination (collapse / flip / runaway), and average the orientation / height /
contact metrics over a HOLD WINDOW (after a short settle) -- counting an env ONLY
while it is still on its FIRST episode (so an autoreset does not inject a fresh
near-IC pose into the "is it holding" statistics).

Metrics (the handstand orientation target at 90 deg is proj_grav -> [+1,0,0]):
  * proj_grav_x   : 1.0 = fully inverted (forward axis straight up). 0.85 ~= 58 deg.
  * base_z        : trunk height (collapse floor 0.10).
  * rear-foot contact frac, front-foot lift / contact (the defining handstand
    feature = front feet OFF, rear feet planted).
  * survival      : fraction of envs that never terminated over the window;
    mean steps-to-first-termination.

VERDICT: a REAL sustained handstand needs proj_grav_x >= ~0.8 held over the
window with the body up on the rear legs and high survival. We print the numbers
and a plain PASS/FAIL -- no faking.

Run (GPU1, keep GPU use short)::

  CUDA_VISIBLE_DEVICES=1 \
    LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:/root/Nuka-Physics/build-cuda128/src \
    PYTHONPATH=/root/Nuka-Physics/build-cuda128/src:/root/Nuka-Physics/python \
    /root/miniconda3/envs/nuka-v03/bin/python examples/training/go2_handstand_eval.py \
      --ckpt out/go2_handstand_s2/nn/last_go2_handstand_s2_ep_1500_rew_521.9076.pth \
      --stage S2 --theta-cmd 90 --num-envs 256 --steps 400
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import nuka  # noqa: F401  (ensures the extension is importable)
from nuka.tasks.go2_handstand import make_env
from nuka.tasks import go2_obs as G


def load_actor(ckpt_path, obs_dim, act_dim, device):
    """Load the rl_games actor (obs normalizer + MLP -> mu) from a checkpoint.

    Returns a closure ``forward(obs)->mu`` that (1) normalizes obs with the trained
    running_mean_std, (2) runs the actor MLP (elu) to the mean action mu. Sigma is
    ignored (deterministic eval). The checkpoint keys are torch.compile-wrapped
    (``_orig_mod.`` prefix); we strip it.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = sd["model"]
    g = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
         for k, v in m.items()}

    # obs normalizer (rl_games normalize_input: True).
    rmean = g["running_mean_std.running_mean"].to(device).float()
    rvar = g["running_mean_std.running_var"].to(device).float()
    assert rmean.numel() == obs_dim, (
        f"checkpoint obs-normalizer dim {rmean.numel()} != env obs dim {obs_dim} "
        f"-- stage/theta_cmd_obs mismatch")
    eps = 1e-5

    # actor MLP [512,256,128] elu -> mu(act_dim).
    w0 = g["a2c_network.actor_mlp.0.weight"].to(device).float()
    b0 = g["a2c_network.actor_mlp.0.bias"].to(device).float()
    w2 = g["a2c_network.actor_mlp.2.weight"].to(device).float()
    b2 = g["a2c_network.actor_mlp.2.bias"].to(device).float()
    w4 = g["a2c_network.actor_mlp.4.weight"].to(device).float()
    b4 = g["a2c_network.actor_mlp.4.bias"].to(device).float()
    wmu = g["a2c_network.mu.weight"].to(device).float()
    bmu = g["a2c_network.mu.bias"].to(device).float()
    assert wmu.shape[0] == act_dim, f"mu dim {wmu.shape[0]} != act {act_dim}"

    elu = torch.nn.functional.elu

    @torch.no_grad()
    def forward(obs):
        x = (obs - rmean) / torch.sqrt(rvar + eps)
        x = x.clamp(-5.0, 5.0)  # rl_games RunningMeanStd default norm clamp.
        x = elu(torch.nn.functional.linear(x, w0, b0))
        x = elu(torch.nn.functional.linear(x, w2, b2))
        x = elu(torch.nn.functional.linear(x, w4, b4))
        mu = torch.nn.functional.linear(x, wmu, bmu)
        return mu

    return forward, sd.get("epoch"), sd.get("last_mean_rewards")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Deterministic Go2 handstand eval.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stage", default="S2", choices=("S0", "S1", "S2"))
    p.add_argument("--theta-cmd", type=float, default=90.0,
                   help="Forced per-env pitch command (deg). 90 = full handstand.")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=400,
                   help="Control steps to roll out.")
    p.add_argument("--settle", type=int, default=40,
                   help="Control steps to skip before the HOLD measurement window.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(args.num_envs, stage=args.stage, theta_cmd_obs=True,
                   reward_mode="handstand", seed=args.seed)
    dev = env._torch_device
    n = env.num_envs
    obs_dim = env._obs_dim
    act_dim = G.GO2_ACTION_DIM

    forward, ckpt_epoch, ckpt_rew = load_actor(args.ckpt, obs_dim, act_dim, dev)
    print(f"[eval] ckpt={args.ckpt}\n[eval] ckpt epoch={ckpt_epoch} "
          f"last_mean_rewards={ckpt_rew}  stage={args.stage} "
          f"theta_cmd_forced={args.theta_cmd}deg num_envs={n} obs_dim={obs_dim}",
          flush=True)

    # Reset (seats the stage IC + samples theta in band), then FORCE the pitch
    # command to args.theta_cmd on every env so the obs theta slot + the metrics
    # use the full-handstand target. (The IC was seated at the sampled per-env
    # theta; forcing the cmd here just pins the obs/metric target -- the policy
    # then drives toward 90 from a near-90 IC, which is exactly the S2 task.)
    obs, _ = env.reset(seed=args.seed)
    env._theta_cmd_deg[:] = float(args.theta_cmd)
    # Re-append theta to the obs we'll feed the net (reset appended the sampled one).
    obs = obs[:, :G.GO2_OBS_DIM]
    obs = env._append_theta(obs)

    # NOSE-DOWN handstand convention (corrected 2026-06-16): proj_grav_x -> -1 at
    # the 90 deg handstand (rear feet planted, front paws up). The inversion metric
    # is INV = -proj_grav_x toward +1; the target inversion at theta_cmd is
    # sin(theta_cmd). All gates below use INV (= -pgx).
    target_inv = float(np.sin(np.radians(args.theta_cmd)))   # 1.0 at 90.

    # Per-env first-termination tracking (so the HOLD stats count only envs on
    # their FIRST episode -- an autoreset re-seats a near-IC pose we must exclude).
    alive = torch.ones(n, dtype=torch.bool, device=dev)        # still on episode 1
    steps_to_term = torch.full((n,), float(args.steps), device=dev)

    # HOLD-window accumulators (per env, summed over the window while alive).
    win_count = torch.zeros(n, device=dev)
    sum_pgx = torch.zeros(n, device=dev)
    sum_pgy = torch.zeros(n, device=dev)
    sum_pgz = torch.zeros(n, device=dev)
    sum_bz = torch.zeros(n, device=dev)
    sum_rear = torch.zeros(n, device=dev)      # rear-foot contact frac (0..1)
    sum_front_c = torch.zeros(n, device=dev)   # front-foot contact count (0..2)
    sum_front_z = torch.zeros(n, device=dev)   # front-foot mean world z (lift)

    front_slot = env._front_foot_slot
    rear_slot = env._rear_foot_slot

    traj_pgx = []   # median proj_grav_x over alive envs, per step (for the log)

    for k in range(args.steps):
        mu = forward(obs)
        # Deterministic: apply mu directly (action space +-1 => no rescale).
        obs_next, reward, terminated, truncated, info = env.step(mu)
        # Re-pin theta_cmd (step's autoreset may have re-sampled done envs; and we
        # always want the forced target in the obs/metrics) BEFORE we read metrics.
        env._theta_cmd_deg[:] = float(args.theta_cmd)

        # Read engine state for the metrics (post-step).
        b = env._obs
        pg = b.projected_gravity()                  # (N,3)
        base_z = b.base_pos()[:, 2]                  # (N,)
        rear_fz = env._wrench_view[:, rear_slot, 2]
        front_fz = env._wrench_view[:, front_slot, 2]
        rear_loaded = (rear_fz.abs() > 1.0).to(torch.float32).sum(dim=1) / 2.0  # 0..1
        front_loaded = (front_fz.abs() > 1.0).to(torch.float32).sum(dim=1)      # 0..2
        front_z = env._pose_view[:, front_slot, 2].mean(dim=1)

        # First-termination bookkeeping (terminated only -- truncation = window end,
        # not a fall). Record steps-to-term the first time an env terminates.
        newly_term = terminated & alive
        if bool(newly_term.any()):
            steps_to_term[newly_term] = float(k + 1)
        alive = alive & (~terminated)

        # HOLD-window accumulation for envs still alive (on episode 1) past settle.
        if k >= args.settle:
            mwin = alive
            win_count += mwin.to(torch.float32)
            sum_pgx += (-pg[:, 0]) * mwin                    # INV = -pgx (toward +1)
            sum_pgy += pg[:, 1].abs() * mwin
            sum_pgz += pg[:, 2] * mwin
            sum_bz += base_z * mwin
            sum_rear += rear_loaded * mwin
            sum_front_c += front_loaded * mwin
            sum_front_z += front_z * mwin

        if k % 20 == 0 or k == args.steps - 1:
            a = alive
            if bool(a.any()):
                med_inv = float((-pg[a, 0]).median())
                med_bz = float(base_z[a].median())
            else:
                med_inv = float("nan"); med_bz = float("nan")
            traj_pgx.append(med_inv)
            print(f"[eval] step {k:4d}  alive={int(a.sum())}/{n}  "
                  f"median(alive) inv(-pgx)={med_inv:+.3f} base_z={med_bz:.3f}",
                  flush=True)

        obs = obs_next

    # ---- aggregate the HOLD-window metrics across envs that contributed ----
    contributed = win_count > 0
    nc = int(contributed.sum())
    # Per-env mean over its own contributed window, then median/mean across envs.
    def per_env_mean(s):
        return (s[contributed] / win_count[contributed])

    inv_e = per_env_mean(sum_pgx)          # INV = -pgx (1.0 = fully inverted)
    pgy_e = per_env_mean(sum_pgy)
    pgz_e = per_env_mean(sum_pgz)
    bz_e = per_env_mean(sum_bz)
    rear_e = per_env_mean(sum_rear)
    frontc_e = per_env_mean(sum_front_c)
    frontz_e = per_env_mean(sum_front_z)

    survived = (steps_to_term >= args.steps)
    surv_frac = float(survived.to(torch.float32).mean())
    mean_steps = float(steps_to_term.mean())

    # "held inverted" = per-env window-mean inversion (-pgx) >= 0.8 AND survived.
    hold_thresh = 0.8
    held_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    held_mask[contributed] = inv_e >= hold_thresh
    held_mask = held_mask & survived
    hold_frac = float(held_mask.to(torch.float32).mean())

    def fmt(t):
        return (f"median={float(t.median()):+.3f} mean={float(t.mean()):+.3f} "
                f"p10={float(t.quantile(0.1)):+.3f} p90={float(t.quantile(0.9)):+.3f}")

    pitch_from_inv = np.degrees(np.arcsin(np.clip(float(inv_e.median()), -1, 1)))

    print("\n[eval] ================ HOLD-WINDOW VERDICT ================", flush=True)
    print(f"[eval] stage={args.stage} theta_cmd={args.theta_cmd}deg NOSE-DOWN "
          f"(target inversion -pgx={target_inv:.3f})  window=[{args.settle},{args.steps}) "
          f"control steps; {nc}/{n} envs contributed", flush=True)
    print(f"[eval] inversion -proj_grav_x (1.0=fully inverted): {fmt(inv_e)}", flush=True)
    print(f"[eval]   -> sustained rear-up pitch from median inv ~= {pitch_from_inv:.1f} deg", flush=True)
    print(f"[eval] |proj_grav_y| (off-sagittal): {fmt(pgy_e)}", flush=True)
    print(f"[eval] proj_grav_z (0 at handstand): {fmt(pgz_e)}", flush=True)
    print(f"[eval] base_z (m): {fmt(bz_e)}", flush=True)
    print(f"[eval] rear-foot contact frac [0,1]: {fmt(rear_e)}", flush=True)
    print(f"[eval] front-foot contact count [0,2] (0=clean handstand): {fmt(frontc_e)}",
          flush=True)
    print(f"[eval] front-foot world z (m, lift): {fmt(frontz_e)}", flush=True)
    print(f"[eval] survival (no term over {args.steps} steps): {surv_frac:.3f}  "
          f"mean steps-to-first-term={mean_steps:.1f}/{args.steps}", flush=True)
    print(f"[eval] HELD-INVERTED fraction (window-mean -pgx>={hold_thresh} AND survived): "
          f"{hold_frac:.3f}", flush=True)

    real = (float(inv_e.median()) >= hold_thresh and surv_frac >= 0.6
            and float(frontc_e.median()) < 0.5 and float(rear_e.median()) >= 0.5)
    print(f"\n[eval] RESULT: {'*** REAL SUSTAINED HANDSTAND ***' if real else '*** NOT A CLEAN HANDSTAND ***'}",
          flush=True)
    if not real:
        print(f"[eval]   (need median -pgx>={hold_thresh} [got {float(inv_e.median()):+.3f}], "
              f"survival>=0.6 [got {surv_frac:.2f}], front-contact<0.5 "
              f"[got {float(frontc_e.median()):.2f}], rear-contact>=0.5 "
              f"[got {float(rear_e.median()):.2f}])", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
