#!/usr/bin/env python3
"""Closed-loop traversal + per-medium interaction probe (logs/npy only, no render).

The trained BDX walk policy drives the ten leg joints; the base is a free body under
the general solver (never scripted). Logs base trajectory and, per medium (gravel,
debris, door cloth, hung slab/cable), the solver-produced coupling response so the
controller can read PRESENT(magnitude)/ABSENT(evidence) with hard numbers."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

import nuka
from nuka.tasks import bdx_obs as B
from nuka.tasks.bdx_locomotion import _foot_capsules
from nuka.tasks.bdx_reference_motion import BdxReferenceMotion

import bdx_oneshot_author as A
import bdx_oneshot_choreo as C

sys.path.insert(0, "/root/Nuka-Physics/examples/training")
from bdx_walk_eval import load_actor  # noqa: E402

PKL = "/root/Nuka-Physics/examples/assets/bdx/polynomial_coefficients.pkl"
DT = C.DT
OUTDIR = "/data/xtzhang25/_work/activate/out"
FOOT_LINKS = (5, 14)          # left/right ankle-foot (lowest-z links)
HEAD_LINKS = (8, 9)           # head_yaw / head_roll (forward head)
NEAR_R = 0.06                 # foot-neighbourhood radius for local media response

# Disjoint corridor zones (final world metres) -> robust particle-to-medium labels.
ZONES = {
    "gravel": dict(x=(0.90, 2.32), y=(-0.26, 0.26), z=(-0.05, 0.20)),
    "debris": dict(x=(2.40, 3.12), y=(-0.22, 0.22), z=(-0.05, 0.20)),
    "cloth":  dict(x=(-0.12, 0.56), y=(-0.32, 0.32), z=(0.20, 0.95)),
    "slab":   dict(x=(3.58, 3.92), y=(-0.20, 0.20), z=(0.28, 0.82)),
}
# Where the duck must be for that medium to be in reach (base-x gate for head media).
HEAD_GATE = {"cloth": (0.00, 0.60), "slab": (3.30, 3.98)}


def zmask(p, zd):
    return ((p[:, 0] >= zd["x"][0]) & (p[:, 0] < zd["x"][1]) &
            (p[:, 1] >= zd["y"][0]) & (p[:, 1] < zd["y"][1]) &
            (p[:, 2] >= zd["z"][0]) & (p[:, 2] < zd["z"][1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cmd-vx", type=float, default=0.15)
    ap.add_argument("--seconds", type=float, default=22.0)
    ap.add_argument("--settle", type=int, default=300)
    ap.add_argument("--cmd-ramp", type=int, default=30)
    ap.add_argument("--decimation", type=int, default=5)
    ap.add_argument("--x-stop", type=float, default=4.20)
    ap.add_argument("--tag", default="interact")
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
        if len(caps) < 2:
            caps = [(np.zeros(3, np.float32), 0.01), (np.zeros(3, np.float32), 0.01)]

        obs = B.BdxObsBuilder(w, ankle, caps)
        obs.set_target_rate_cap(control_dt)
        obs.apply_pd_gains()
        ref = BdxReferenceMotion(PKL, dev_t)
        forward, ep, rew = load_actor(args.ckpt, B.BDX_OBS_DIM, B.BDX_ACTION_DIM, dev_t)

        lcw = torch.from_dlpack(w.buffer_view(nuka.LINK_CONTACT_WRENCH))   # (1,15,6)
        lpose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))  # (1,15,7)
        print(f"[probe] ckpt ep={ep} rew={rew} decim={decim} control_dt={control_dt:.4f} "
              f"blc={obs.blc} nb_steps={ref.nb_steps} cmd_vx={args.cmd_vx}", flush=True)

        command = torch.tensor([[args.cmd_vx, 0.0, 0.0]], dtype=torch.float32, device=dev_t)
        zero_cmd = torch.zeros(1, 3, device=dev_t)
        last_action = torch.zeros(1, B.BDX_ACTION_DIM, device=dev_t)
        phase_i = torch.zeros(1, dtype=torch.long, device=dev_t)

        def parts():
            p = np.asarray(w.download_field(nuka.Field.PARTICLE_POSITION),
                           dtype=np.float32).reshape(-1, 3)
            v = np.asarray(w.download_field(nuka.Field.PARTICLE_VELOCITY),
                           dtype=np.float32).reshape(-1, 3)
            return p, v

        bp = obs.base_pos()[0].detach().cpu().numpy()
        print(f"[probe] cooked base={bp.round(3)} tilt={float(obs.tilt_deg()[0]):.1f}", flush=True)

        # Settle media with the duck balancing in place (policy, zero command).
        settle_speed = {k: 0.0 for k in ZONES}
        t_s0 = time.time()
        for _ in range(max(1, args.settle // decim)):
            o = obs.compute_obs(zero_cmd, last_action, ref.phase_features(phase_i))
            last_action = obs.write_action(forward(o))
            w.step_n(decim)
            phase_i = (phase_i + 1) % ref.nb_steps
        print(f"[probe] settle done in {time.time()-t_s0:.1f}s", flush=True)

        # Fixed per-medium particle masks from the settled rest positions (Lagrangian).
        p_rest, v_rest = parts()
        masks = {k: zmask(p_rest, ZONES[k]) for k in ZONES}
        counts = {k: int(masks[k].sum()) for k in ZONES}
        for k in ZONES:
            if counts[k]:
                settle_speed[k] = float(np.linalg.norm(v_rest[masks[k]], axis=1).max())
        print(f"[probe] settled base={obs.base_pos()[0].detach().cpu().numpy().round(3)} "
              f"tilt={float(obs.tilt_deg()[0]):.1f} media_counts={counts} "
              f"settle_speed_floor={ {k: round(settle_speed[k],4) for k in ZONES} }",
              flush=True)

        n_ctrl = int(round(args.seconds / control_dt))
        traj, stopped, down_run = [], False, 0
        t_loop0 = time.time()
        # Per-medium running peaks + whether the duck reached it.
        peak = {k: dict(disp=0.0, dz=0.0, speed=0.0, wr=0.0, near_dz=0.0,
                        near_speed=0.0, x=float("nan"), reached=False) for k in ZONES}
        dump = {}

        for k in range(n_ctrl):
            r = min(1.0, (k + 1) / max(1, args.cmd_ramp))
            cmd_k = zero_cmd if stopped else command * r
            o = obs.compute_obs(cmd_k, last_action, ref.phase_features(phase_i))
            last_action = obs.write_action(forward(o))
            w.step_n(decim)
            phase_i = (phase_i + 1) % ref.nb_steps

            bp = obs.base_pos()[0].detach().cpu().numpy()
            tilt = float(obs.tilt_deg()[0])
            v = obs.base_lin_vel()[0].detach().cpu().numpy()
            lp = lpose[0].detach().cpu().numpy()          # (15,7)
            wr = lcw[0].detach().cpu().numpy()            # (15,6)
            footxy = lp[list(FOOT_LINKS), 0:2]            # (2,2)
            foot_z = lp[list(FOOT_LINKS), 2]
            headxy = lp[list(HEAD_LINKS), 0:2]
            fwr = float(np.linalg.norm(wr[list(FOOT_LINKS), 3:6], axis=1).max())
            hwr = float(np.linalg.norm(wr[list(HEAD_LINKS), 3:6], axis=1).max())
            twr = float(np.linalg.norm(wr[0, 3:6]))
            traj.append((bp[0], bp[1], bp[2], tilt, v[0], v[1],
                         float(foot_z[0]), float(foot_z[1]), fwr, hwr, twr))

            if not np.isfinite(bp).all():
                print(f"[probe] NaN base at ctrl {k}", flush=True)
                break
            if bp[0] >= args.x_stop:
                stopped = True
            # Cut the run once a fall is terminal (down for ~0.8 s and not recovering).
            down_run = down_run + 1 if (bp[2] < 0.10 or tilt > 60.0) else 0
            if down_run >= 40:
                print(f"[probe] terminal fall confirmed at ctrl {k} x={bp[0]:.3f}", flush=True)
                break

            p_now, v_now = parts()
            disp = np.linalg.norm(p_now - p_rest, axis=1)
            dz = p_now[:, 2] - p_rest[:, 2]
            spd = np.linalg.norm(v_now, axis=1)
            for name, zd in ZONES.items():
                m = masks[name]
                if counts[name] == 0:
                    continue
                pk = peak[name]
                pk["disp"] = max(pk["disp"], float(disp[m].max()))
                pk["dz"] = min(pk["dz"], float(dz[m].min()))       # compaction (negative)
                pk["speed"] = max(pk["speed"], float(spd[m].max()))
                # Foot-local response (gravel/debris) or head reach gate (cloth/slab).
                if name in ("gravel", "debris"):
                    over = ((footxy[:, 0] >= zd["x"][0]) & (footxy[:, 0] < zd["x"][1]) &
                            (np.abs(footxy[:, 1]) < 0.26))
                    if bool(over.any()):
                        pk["reached"] = True
                        pk["wr"] = max(pk["wr"], fwr)
                        pm = p_rest[m]
                        d2 = np.min(
                            ((pm[:, None, 0] - footxy[over][None, :, 0]) ** 2 +
                             (pm[:, None, 1] - footxy[over][None, :, 1]) ** 2), axis=1)
                        near = d2 < NEAR_R ** 2
                        if bool(near.any()):
                            mn = np.where(m)[0][near]
                            pk["near_dz"] = min(pk["near_dz"], float(dz[mn].min()))
                            pk["near_speed"] = max(pk["near_speed"], float(spd[mn].max()))
                            if float(spd[mn].max()) > 0.02 and name not in dump:
                                dump[name] = (k, p_now.copy())
                else:
                    lo, hi = HEAD_GATE[name]
                    if lo <= bp[0] < hi:
                        pk["reached"] = True
                        pk["wr"] = max(pk["wr"], hwr if name == "cloth" else max(hwr, twr))
                        if float(disp[m].max()) > 0.01 and name not in dump:
                            dump[name] = (k, p_now.copy())
                if pk["speed"] >= peak[name]["speed"]:
                    pk["x"] = float(bp[0])

            if k % 15 == 0:
                rate = (time.time() - t_loop0) / max(1, k + 1)
                print(f"[probe] ctrl {k:4d} x={bp[0]:+.3f} y={bp[1]:+.3f} z={bp[2]:.3f} "
                      f"tilt={tilt:4.1f} vx={v[0]:+.3f} fz=({foot_z[0]:.3f},{foot_z[1]:.3f}) "
                      f"fwr={fwr:.2f} hwr={hwr:.2f} [{rate:.2f}s/step]", flush=True)

        tr = np.array(traj)
        os.makedirs(OUTDIR, exist_ok=True)
        np.save(os.path.join(OUTDIR, f"bdx_{args.tag}_traj.npy"), tr)
        for name, (kk, pn) in dump.items():
            np.save(os.path.join(OUTDIR, f"bdx_{args.tag}_{name}_k{kk}.npy"), pn)
        np.save(os.path.join(OUTDIR, f"bdx_{args.tag}_rest.npy"), p_rest)

        if len(tr):
            fell = bool((tr[:, 2].min() < 0.10) or (tr[:, 3].max() > 60.0))
            fell_i = -1
            for i in range(len(tr)):
                if tr[i, 2] < 0.10 or tr[i, 3] > 60.0:
                    fell_i = i
                    break
            fell_x = float(tr[fell_i, 0]) if fell_i >= 0 else float("nan")
            print(f"[probe] SUMMARY ctrl={len(tr)} x {tr[0,0]:+.3f}->{tr[-1,0]:+.3f} "
                  f"xmax={tr[:,0].max():+.3f} dx={tr[-1,0]-tr[0,0]:+.3f} "
                  f"zmin={tr[:,2].min():.3f} tiltmax={tr[:,3].max():.1f} "
                  f"vx_mean={tr[:,4].mean():+.3f} FELL={fell} fell_i={fell_i} "
                  f"fell_x={fell_x:.3f}", flush=True)
            # Sustained speed over the post-ramp window before any fall.
            lo = min(args.cmd_ramp, len(tr) - 1)
            hi = fell_i if fell_i >= 0 else len(tr)
            if hi > lo:
                print(f"[probe] sustained vx_mean[{lo}:{hi}]={tr[lo:hi,4].mean():+.3f} "
                      f"max_foot_wrench={tr[:,8].max():.2f} max_head_wrench={tr[:,9].max():.2f}",
                      flush=True)
        for name in ZONES:
            pk = peak[name]
            print(f"[probe] MEDIUM {name:7s} count={counts[name]:5d} reached={pk['reached']} "
                  f"peak_disp={pk['disp']:.4f} peak_compaction_dz={pk['dz']:.4f} "
                  f"peak_speed={pk['speed']:.4f} (floor={settle_speed[name]:.4f}) "
                  f"near_dz={pk['near_dz']:.4f} near_speed={pk['near_speed']:.4f} "
                  f"contact_wrench={pk['wr']:.3f} x@peak={pk['x']:.3f}", flush=True)
        w.destroy()
    print("[probe] DONE", flush=True)


if __name__ == "__main__":
    main()
