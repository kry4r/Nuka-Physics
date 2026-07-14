#!/usr/bin/env python3
"""Continuous one-shot film driven by REAL closed-loop policy physics.

The trained BDX walk policy drives the duck's ten leg joints while the base is a
free body under the general solver, so the feet imprint the gravel, the head lifts
the door curtain and the trunk swings the slab -- every contact is real solver
output. The base pose is never scripted: it is whatever the physics produces."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

import nuka
from nuka.tasks import bdx_obs as B
from nuka.tasks.bdx_locomotion import _foot_capsules
from nuka.tasks.bdx_reference_motion import BdxReferenceMotion

import bdx_oneshot_author as A  # noqa: F401  (scene constants)
import bdx_oneshot_choreo as C
from bdx_oneshot_film import camera_at, OUTROOT

sys.path.insert(0, "/root/Nuka-Physics/examples/training")
from bdx_walk_eval import load_actor  # noqa: E402

PKL = "/root/Nuka-Physics/examples/assets/bdx/polynomial_coefficients.pkl"
DT = C.DT


def sim_advance(w, n):
    try:
        w.step_n(n)
    except (AttributeError, TypeError):
        for _ in range(n):
            w.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cmd-vx", type=float, default=0.15)
    ap.add_argument("--cmd-vy", type=float, default=0.0)
    ap.add_argument("--cmd-wyaw", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=6.0)   # walk duration after settle
    ap.add_argument("--settle", type=int, default=300)      # sim steps to settle media (policy zero-cmd)
    ap.add_argument("--cmd-ramp", type=int, default=30)     # control steps to ramp cmd 0 -> target
    ap.add_argument("--decimation", type=int, default=5)    # control_dt = decim/240 ~= 0.02
    ap.add_argument("--x-stop", type=float, default=3.8)    # zero the command once base x reaches here
    ap.add_argument("--spp", type=int, default=48)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--tag", default="realsim")
    ap.add_argument("--probe", action="store_true")         # no render; log trajectory + coupling
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    with nuka.Device.create(0) as dev:
        w = C.build_world(dev)
        decim = args.decimation
        control_dt = DT * decim
        dev_t = torch.device("cuda")

        ankle = [int(w.dof_index("^left_ankle$")[0]), int(w.dof_index("^right_ankle$")[0])]
        try:
            caps = _foot_capsules(C.OUT)
        except Exception:
            caps = []
        if len(caps) < 2:  # box feet: the kinematic contact proxy is unused in eval.
            caps = [(np.zeros(3, np.float32), 0.01), (np.zeros(3, np.float32), 0.01)]

        obs = B.BdxObsBuilder(w, ankle, caps)
        obs.set_target_rate_cap(control_dt)
        obs.apply_pd_gains()
        ref = BdxReferenceMotion(PKL, dev_t)
        forward, ep, rew = load_actor(args.ckpt, B.BDX_OBS_DIM, B.BDX_ACTION_DIM, dev_t)
        print(f"[realsim] ckpt ep={ep} rew={rew} decim={decim} control_dt={control_dt:.4f} "
              f"blc={obs.blc} nb_steps={ref.nb_steps}", flush=True)

        command = torch.tensor([[args.cmd_vx, args.cmd_vy, args.cmd_wyaw]],
                               dtype=torch.float32, device=dev_t)
        zero_cmd = torch.zeros(1, 3, device=dev_t)
        last_action = torch.zeros(1, B.BDX_ACTION_DIM, device=dev_t)
        phase_i = torch.zeros(1, dtype=torch.long, device=dev_t)

        def p_base():
            return obs.base_pos()[0].detach().cpu().numpy()

        print(f"[realsim] cooked base={p_base().round(3)} tilt={float(obs.tilt_deg()[0]):.1f}",
              flush=True)

        # settle the media while the duck balances in place (policy, zero command).
        for _ in range(max(1, args.settle // decim)):
            o = obs.compute_obs(zero_cmd, last_action, ref.phase_features(phase_i))
            last_action = obs.write_action(forward(o))
            sim_advance(w, decim)
            phase_i = (phase_i + 1) % ref.nb_steps
        print(f"[realsim] settled base={p_base().round(3)} tilt={float(obs.tilt_deg()[0]):.1f}",
              flush=True)

        outdir = os.path.join(OUTROOT, f"frames_{args.tag}")
        os.makedirs(outdir, exist_ok=True)
        every = max(1, int(round(1.0 / (args.fps * control_dt))))
        n_ctrl = int(round(args.seconds / control_dt))

        traj, frame, stopped = [], 0, False
        for k in range(n_ctrl):
            r = min(1.0, (k + 1) / max(1, args.cmd_ramp))
            cmd_k = zero_cmd if stopped else command * r
            o = obs.compute_obs(cmd_k, last_action, ref.phase_features(phase_i))
            last_action = obs.write_action(forward(o))
            sim_advance(w, decim)
            phase_i = (phase_i + 1) % ref.nb_steps

            bp = obs.base_pos()[0].detach().cpu().numpy()
            tilt = float(obs.tilt_deg()[0])
            vx = float(obs.base_lin_vel()[0, 0])
            traj.append((bp[0], bp[1], bp[2], tilt, vx))
            if not bool(np.isfinite(bp).all()):
                print(f"[realsim] NaN base at ctrl {k}", flush=True)
                break
            if bp[0] >= args.x_stop:
                stopped = True

            if args.probe:
                if k % 15 == 0:
                    npart = C.particles(w).shape[0]
                    print(f"[realsim] ctrl {k:4d} x={bp[0]:+.3f} y={bp[1]:+.3f} z={bp[2]:.3f} "
                          f"tilt={tilt:4.1f} vx={vx:+.3f} npart={npart}", flush=True)
                continue

            if k % every == 0:
                eye, look, fov = camera_at(float(bp[0]))
                img = w.render_beauty(eye=tuple(float(v) for v in eye),
                                      look=tuple(float(v) for v in look),
                                      fov_deg=float(fov), width=args.width,
                                      height=args.height, spp=args.spp)
                Image.fromarray(np.ascontiguousarray(img)).save(
                    os.path.join(outdir, f"f_{frame:05d}.png"))
                print(f"[realsim] frame {frame} x={bp[0]:.3f} tilt={tilt:.1f}", flush=True)
                frame += 1

        tr = np.array(traj)
        if len(tr):
            fell = bool((tr[:, 2].min() < 0.10) or (tr[:, 3].max() > 60.0))
            print(f"[realsim] SUMMARY ctrl={len(tr)} x {tr[0,0]:+.3f}->{tr[-1,0]:+.3f} "
                  f"dx={tr[-1,0]-tr[0,0]:+.3f} zmin={tr[:,2].min():.3f} "
                  f"tiltmax={tr[:,3].max():.1f} vx_mean={tr[:,4].mean():+.3f} FELL={fell}",
                  flush=True)
        w.destroy()


if __name__ == "__main__":
    main()
