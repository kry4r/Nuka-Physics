#!/usr/bin/env python3
"""[p04-T2 demo] Capture a Go2 locomotion rollout for the v0.3 demo video.

WHAT THIS IS (read the honesty framing before trusting the video)
-----------------------------------------------------------------
This produces the v0.3 **exit-criterion #6 deliverable -- "a 4096-env Go2
locomotion video"** -- by REPLAYING the ALREADY-PROVEN trained Go2 policy
(motion.pt, from unitree_rl_gym PR#62) in Nuka. It is a *replay/visualize* of a
policy that was trained ELSEWHERE and then validated in Nuka (sim-val #41/#43:
dx +2.93 m / 6 s, vx 0.488 ~= cmd 0.5, tilt < 5.4 deg at native dt=0.005).

It DOES NOT satisfy exit-criterion #3 ("from-scratch PPO convergence to a stable
gait"). That is a SEPARATE gate. This video shows a PROVEN, externally-trained
policy walking in Nuka -- NOT a self-trained policy and NOT evidence of in-house
PPO convergence. Do not conflate the two.

PIPELINE
--------
  1. Import the PROVEN policy-driving logic from
     examples/sim_val/go2_policy_drive.py (load_policy, build_obs, warm_start,
     read_state_urdf, the D1 structural joint permutation, all scales/defaults).
     Capturing via IMPORT (not reimplementation) guarantees the rendered motion
     is bit-for-bit the same validated obs->action->drive pipeline.
  2. Run the policy in-the-loop on a BATCHED 16-env world. Unlike the validation
     harness (which drives env0 and broadcasts), this reads ALL 16 envs, runs the
     policy BATCHED (16,48)->(16,12), and writes PER-ENV drive targets, so the
     grid is a genuine 16-instance batched rollout. Each env gets its own forward
     command -- a spread around the ONE command validated in Nuka (cmd=[0.5,0,0]):
     vx in [0.35,0.6], vy=0. This shows visual variety; the off-0.5 speeds are NOT
     separately pre-validated and we make NO claim about the policy's TRAINING
     command distribution -- each shown env is instead verified to walk (forward +
     upright) by the per-env walk check in the renderer, and any env that does not
     is flagged, not silently shown. (The policy obs is body-frame + (q-default),
     i.e. translation-invariant, so same-cmd envs would otherwise be identical
     clones -- hence the per-env command spread.)
  3. GATE on the walk: assert the mean world +x advance of the 16 envs over the
     rollout is forward (dx > 0) and the envs stay upright. If the capture does
     NOT walk, abort -- we will not render a non-walk.
  4. Dump per-frame ARTICULATION_LINK_POSE (16 envs x 13 links x 7) +
     per-env command + base px/py to a compact .npz. The renderer
     (go2_demo_render.py) turns that into PPM frames; render_video.sh -> MP4.

Run:
  export CUDA_VISIBLE_DEVICES=0
  /root/miniconda3/envs/nuka-v03/bin/python examples/demo/go2_demo_capture.py \
      --envs 16 --seconds 6 --out out/go2_demo/go2_rollout.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Import the PROVEN policy-driving logic. Capturing via import (not a copy) makes
# the rendered motion provably the same validated pipeline.
_SIM_VAL = os.path.join(os.path.dirname(__file__), "..", "sim_val")
sys.path.insert(0, os.path.abspath(_SIM_VAL))

import nuka  # noqa: E402
import go2_policy_drive as G  # noqa: E402


def read_all_envs_urdf(w, urdf_from_nuka_slot, env_count):
    """Vectorized read of ALL envs' state in URDF order. Returns
    (q_urdf(N,12), qd_urdf(N,12), base_lin_vel(N,3), base_ang_vel(N,3),
     proj_grav(N,3), base_xyz(N,3), base_quat_wxyz(N,4), tilt_deg(N))."""
    q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
    qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
    pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
    vel = torch.from_dlpack(w.buffer_view(nuka.LINK_VELOCITY))

    q_slots = q[:env_count, 1:G.GO2_BLC].detach().cpu().numpy()      # (N,12) nuka order
    qd_slots = qd[:env_count, 1:G.GO2_BLC].detach().cpu().numpy()
    q_urdf = q_slots[:, urdf_from_nuka_slot]                          # -> URDF order
    qd_urdf = qd_slots[:, urdf_from_nuka_slot]

    base = pose[:env_count, 0, :].detach().cpu().numpy()             # (N,7)
    base_xyz = base[:, 0:3]
    base_quat = base[:, 3:7]                                          # wxyz
    pg = np.stack([G.projected_gravity_body(base_quat[i]) for i in range(env_count)])
    tilt = np.degrees(np.arccos(np.clip(-pg[:, 2], -1.0, 1.0)))

    root_vel = vel[:env_count, 0, :].detach().cpu().numpy()          # (N,6) omega-first
    base_ang_vel = root_vel[:, 0:3]
    base_lin_vel = root_vel[:, 3:6]
    return q_urdf, qd_urdf, base_lin_vel, base_ang_vel, pg, base_xyz, base_quat, tilt


def capture(dev, policy, urdf_from_nuka_slot, env_count, policy_steps, cmds):
    """Drive the policy BATCHED across env_count envs (per-env command `cmds`,
    shape (N,3)), capturing per-frame link poses. Returns a dict for np.savez."""
    print("\n" + "-" * 78)
    print(f"CAPTURE: batched {env_count}-env policy rollout, {policy_steps} steps "
          f"@ 50 Hz ({policy_steps * G.DT * G.DECIMATION:.1f} s sim)")
    print(f"  per-env cmd vx range [{cmds[:,0].min():.2f},{cmds[:,0].max():.2f}] "
          f"(validated regime around 0.5)")
    print("-" * 78, flush=True)

    frames = []          # list of (N,13,7) link-pose snapshots
    base_track = []       # list of (N,3) base xyz per frame
    cmds = np.asarray(cmds, np.float32)

    with G.make_world(dev, env_count) as w:
        # Warm-start ALL envs to the lightly-sagged upright default (same as the
        # validation harness; set_gains/warm_start operate on every env).
        G.warm_start(w, urdf_from_nuka_slot)
        pose_buf = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))

        (_, _, _, _, _, base_xyz0, _, tilt0) = read_all_envs_urdf(
            w, urdf_from_nuka_slot, env_count)
        base_xy0 = base_xyz0[:, 0:2].copy()
        last_action = np.zeros((env_count, 12), np.float32)

        nan_seen = False
        for k in range(policy_steps):
            (q_u, qd_u, blv, bav, pg, base_xyz, base_quat, tilt) = read_all_envs_urdf(
                w, urdf_from_nuka_slot, env_count)
            obs = G.build_obs(blv, bav, pg, cmds, q_u, qd_u, last_action)
            if not np.isfinite(obs).all():
                nan_seen = True
                print(f"  [abort] non-finite obs at step {k}", flush=True)
                break
            action = G.policy_forward(policy, obs)                    # (N,12) batched
            action = np.clip(action, -G.ACTION_CLIP, G.ACTION_CLIP)
            last_action = action.copy()
            target_urdf = G.DEFAULT_ANGLES[None, :] + G.ACTION_SCALE * action  # (N,12)
            G.write_targets_urdf(w, urdf_from_nuka_slot, target_urdf)
            w.step_n(G.DECIMATION)

            # snapshot AFTER the step so the frame reflects the new pose.
            nuka.sync()
            frames.append(pose_buf[:env_count].detach().cpu().numpy().copy())
            base_track.append(
                pose_buf[:env_count, 0, 0:3].detach().cpu().numpy().copy())

            if (k % 25) == 0 or k == policy_steps - 1:
                dx = float((base_track[-1][:, 0] - base_xy0[:, 0]).mean())
                print(f"  t={k*G.DT*G.DECIMATION:5.2f}s step{k:4d}  "
                      f"mean base_z={base_xyz[:,2].mean():.3f}  "
                      f"mean tilt={tilt.mean():5.1f}deg  "
                      f"mean net dx={dx:+.3f}m", flush=True)

        (_, _, _, _, _, base_xyzf, _, tiltf) = read_all_envs_urdf(
            w, urdf_from_nuka_slot, env_count)

    frames = np.stack(frames) if frames else np.zeros((1, env_count, 13, 7), np.float32)
    base_track = np.stack(base_track) if base_track else np.zeros((1, env_count, 3))
    sim_time = max(len(frames), 1) * G.DT * G.DECIMATION
    dx_world = base_xyzf[:, 0] - base_xy0[:, 0]      # (N,) per-env net world-x advance
    dy_world = base_xyzf[:, 1] - base_xy0[:, 1]
    vx_world = dx_world / sim_time

    return dict(
        frames=frames.astype(np.float32),       # (T,N,13,7) world link pose [xyz,wxyz]
        base_track=base_track.astype(np.float32),
        cmds=cmds, env_count=env_count, sim_time=sim_time,
        dt=G.DT, decimation=G.DECIMATION, policy_steps=len(frames),
        dx_world=dx_world.astype(np.float32), dy_world=dy_world.astype(np.float32),
        vx_world=vx_world.astype(np.float32),
        tilt0=tilt0.astype(np.float32), tilt_final=tiltf.astype(np.float32),
        nan=nan_seen,
    )


def make_cmds(env_count):
    """Per-env forward command: vx sweeps [0.35,0.60] across the grid; vy=0,
    wyaw=0. This is a spread AROUND the one command validated in Nuka
    (cmd=[0.5,0,0]); the off-0.5 speeds are NOT separately pre-validated and make
    NO claim about the policy's training command range -- each env is instead
    verified to walk by the renderer's per-env check (and flagged if it doesn't)."""
    vx = np.linspace(0.35, 0.60, env_count, dtype=np.float32)
    cmds = np.zeros((env_count, 3), np.float32)
    cmds[:, 0] = vx
    return cmds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default="out/go2_demo/go2_rollout.npz")
    ap.add_argument("--uniform-cmd", action="store_true",
                    help="drive every env at the exact validated cmd=[0.5,0,0] "
                         "(clones) instead of the per-env vx spread")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 78)
    print("[p04-T2 demo] Capturing the PROVEN Go2 policy walk for the v0.3 video")
    print("HONESTY: replays an EXTERNALLY-trained, Nuka-VALIDATED policy (exit #6).")
    print("         NOT from-scratch PPO convergence (exit #3, a separate gate).")
    print("=" * 78, flush=True)

    policy = G.load_policy()
    dev = nuka.Device.create(0)
    try:
        # D1: discover the structural joint permutation (the proven path).
        ladder = G.Ladder()
        perm, perm_valid = G.d1_permutation(ladder, dev)
        if not perm_valid:
            print("[ABORT] joint permutation invalid -> refusing to capture a "
                  "scrambled-joint rollout.", flush=True)
            return 1

        policy_steps = int(round(args.seconds / (G.DT * G.DECIMATION)))
        if args.uniform_cmd:
            cmds = np.tile([[0.5, 0.0, 0.0]], (args.envs, 1)).astype(np.float32)
        else:
            cmds = make_cmds(args.envs)

        cap = capture(dev, policy, perm, args.envs, policy_steps, cmds)
    finally:
        dev.close()

    # ---- WALK GATE: refuse to ship a non-walk video. ----
    mean_dx = float(np.mean(cap["dx_world"]))
    mean_vx = float(np.mean(cap["vx_world"]))
    max_tilt_final = float(np.max(cap["tilt_final"]))
    walked = (not cap["nan"]) and mean_dx > 0.3 and max_tilt_final < 35.0
    print("\n" + "=" * 78)
    print("CAPTURE VERDICT (walk gate)")
    print("=" * 78)
    print(f"  envs={cap['env_count']}  sim_time={cap['sim_time']:.2f}s  "
          f"frames={cap['policy_steps']}")
    print(f"  per-env net world dx: min {cap['dx_world'].min():+.3f}  "
          f"mean {mean_dx:+.3f}  max {cap['dx_world'].max():+.3f} m  "
          f"(world-position advance from ARTICULATION_LINK_POSE px)")
    print(f"  per-env vx_world: mean {mean_vx:+.4f} m/s (cmd ~0.5)")
    print(f"  final tilt: max {max_tilt_final:.2f} deg  "
          f"(start mean {float(np.mean(cap['tilt0'])):.2f})")
    print(f"  NaN seen: {cap['nan']}")
    print(f"  WALK GATE: {'PASS -> capturing rollout for render' if walked else 'FAIL -> NOT a clean walk'}")
    print("=" * 78, flush=True)

    if not walked:
        print("[ABORT] capture did not produce a clean forward walk; not writing "
              "an npz for rendering. Investigate before rendering.", flush=True)
        return 1

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **cap)
    sz = os.path.getsize(out_path)
    print(f"\nWROTE {out_path}  ({sz/1e6:.2f} MB)")
    print("Next: render with examples/demo/go2_demo_render.py, encode with "
          "examples/demo/render_video.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
