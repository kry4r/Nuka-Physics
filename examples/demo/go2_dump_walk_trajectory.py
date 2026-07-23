#!/usr/bin/env python3
"""[demo] Dump a DETERMINISTIC Go2 WALKING trajectory for the C++ video replay.

WHAT THIS IS
------------
This is the python PRE-PASS for the beautiful Go2 demo video (owner: WALKING via
policy replay, NOT a static stance). It rolls the PROVEN, golden-validated Go2
walking policy (motion.pt, unitree_rl_gym PR#62) through the unified `nuka.World`
HEADLESS (no render), and DUMPS a per-physics-step trajectory to a flat binary
file. A separate C++ video tool then REPLAYS that trajectory to reproduce + render
the exact walk -- torch never enters C++.

It REPLAYS an externally-trained, Nuka-validated policy (sim-val #41/#43: dx +2.93
m / 6 s, vx 0.488 ~= cmd 0.5, tilt < 5.4 deg at native dt=0.005). It is NOT
from-scratch PPO convergence. Do not conflate the two.

WHY go2_float.usda (not go2.nks) FOR THE PHYSICS
------------------------------------------------
`nuka.World.create_from_scene` ingests USDA (the create-path LoadSceneByExtension
handles .xml/.urdf/.usd/.usda, NOT .nks -- a .nks must go through nuka.Scene which
does not return a steppable World). The validated walk is on go2_float.usda, whose
12-joint articulation is the SAME Go2 (base z=0.445, joints FL/FR/RL/RR x
hip/thigh/calf, cooked crouch hip 0 / thigh +0.8 / calf -1.5) as the beauty-mesh
go2.nks. So we SIMULATE on go2_float.usda and ALSO dump the full per-step link
poses, so the C++ renderer can drive the go2.nks visual meshes directly from the
recorded kinematics (pose-replay) OR re-simulate from the recorded DriveTargets
(physics-replay). Both are provided; see the file layout below.

GROUND: the harness requests contact_family=1, which cooks an explicit flat
heightfield at z=0 on the general contact path. go2_float.usda authors no ground;
the base starts at z=0.445 and settles to ~0.30 as the feet seat on z=0. The C++
replay must render a ground plane at WORLD z = 0.

THE DUMP (what the C++ tool reads)
----------------------------------
We record EVERY physics sub-step (DECIMATION=4 sub-steps per 50 Hz control step),
because the world advances DECIMATION sim-steps per control step and the C++
physics-replay must upload the (held-constant) DriveTarget each sim-step. Per
recorded sim-step we store, for env 0:
  * drive_target[13]  -- the FULL DRIVE_TARGET buffer (slot 0 = root, UNUSED/0;
                         slots 1..12 = the 12 actuated joints in NUKA-SLOT order).
                         This is the EXACT array uploaded into Field.DriveTarget;
                         a C++ replay that uploads it bit-for-bit reproduces the
                         walk (the world is D1/deterministic).
  * link_pose[13][7]  -- ARTICULATION_LINK_POSE for all 13 links AFTER the step:
                         [px,py,pz, qw,qx,qy,qz] world frame (quat w-first). Link 0
                         = base; links 1..12 = the joints in NUKA-SLOT order
                         (FL,FR,RL,RR x hip,thigh,calf -- the cook order).

BINARY LAYOUT (little-endian, the host byte order; flat, no padding)
--------------------------------------------------------------------
  HEADER (all int32 unless noted):
    [0]  magic           = 0x474F3257  ('GO2W')
    [1]  version         = 1
    [2]  num_steps       = N  (number of recorded PHYSICS sub-steps)
    [3]  dof_count       = 13 (DRIVE_TARGET / link slot count incl. root slot 0)
    [4]  link_count      = 13 (ARTICULATION_LINK_POSE link count)
    [5]  pose_floats     = 7  (per link: px,py,pz,qw,qx,qy,qz)
    [6]  decimation      = 4  (sim sub-steps per control step)
    [7]  control_hz_x1000 = 50000 (control rate * 1000, = 50.0 Hz; int)
    [8]  float dt        = 0.005 (float32 reinterpreted; physics step dt seconds)
    [9]  float ctrl_dt   = 0.020 (float32; dt*decimation, the control period)
  Then, contiguous, N records, each:
    drive_target : dof_count   float32   (= 13 floats)
    link_pose    : link_count*pose_floats float32  (= 13*7 = 91 floats)
  => each record = (13 + 91) = 104 float32 = 416 bytes.
  Total file = 10*4 (header) + N*416 bytes.

  Also a SIDECAR JSON (out/go2_walk_trajectory.meta.json) with the human-readable
  setup contract: scene, ground, control mode, gains, initial stance, joint order,
  cmd, walk-validation numbers. The .bin is self-describing for replay; the JSON
  is the contract for the C++ tool author.

Run:
  export CUDA_VISIBLE_DEVICES=0
  export LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
  /root/miniconda3/envs/nuka-v03/bin/python examples/demo/go2_dump_walk_trajectory.py \
      --seconds 8 --out out/go2_walk_trajectory.bin
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

# Import the PROVEN policy-driving logic (load_policy, build_obs, warm_start,
# read_state_urdf, the D1 structural joint permutation, all scales/defaults/gains).
# Capturing via IMPORT (not a copy) makes the dumped motion provably the same
# validated obs->action->drive pipeline as sim-val #41/#43.
_SIM_VAL = os.path.join(os.path.dirname(__file__), "..", "sim_val")
sys.path.insert(0, os.path.abspath(_SIM_VAL))

import nuka  # noqa: E402
import go2_policy_drive as G  # noqa: E402

MAGIC = 0x474F3257  # 'GO2W'
VERSION = 1


def dump_walk(dev, policy, urdf_from_nuka_slot, policy_steps, cmd_vec):
    """Drive env 0 with the policy at cmd_vec, recording EVERY physics sub-step's
    DRIVE_TARGET (the held-constant per-control-step target, uploaded once and
    held across DECIMATION sub-steps) and the per-sub-step ARTICULATION_LINK_POSE.

    Returns a dict with the flat record arrays + walk-validation stats. We step the
    world ONE sim-step at a time (instead of step_n(DECIMATION)) so we can snapshot
    every sub-step; the DriveTarget written before the 4 sub-steps is identical to
    what go2_policy_drive.py's step_n(DECIMATION) sees -- the policy writes the
    target ONCE per control step then advances 4 sim-steps, exactly mirrored here.
    """
    print("\n" + "-" * 78)
    print(f"DUMP: env0 policy walk, {policy_steps} control steps "
          f"@ 50 Hz ({policy_steps * G.DT * G.DECIMATION:.1f} s sim), cmd={cmd_vec}; "
          f"recording every sim sub-step (decim={G.DECIMATION})")
    print("-" * 78, flush=True)

    cmd = np.array([cmd_vec], np.float32)
    drive_records = []   # list of (13,) float32  -- DRIVE_TARGET per sim-step
    pose_records = []    # list of (13,7) float32 -- ARTICULATION_LINK_POSE per sim-step

    with G.make_world(dev, 1) as w:
        # Warm-start to the lightly-sagged upright default (the validated start;
        # set_gains/warm_start operate on every env -> here env0). This SEEDS the
        # PD gains (Kp=20/Kd=0.5) + force limits the policy loop then keeps.
        G.warm_start(w, urdf_from_nuka_slot)

        pose_buf = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
        drive_buf = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))

        # Capture the SETTLED warm-start initial stance (BEFORE the policy loop) --
        # this is what the C++ replay must reproduce as the world's state at the
        # start of the dumped trajectory.
        nuka.sync()
        init_link_pose = pose_buf[0].detach().cpu().numpy().copy()      # (13,7)
        init_drive_target = drive_buf[0].detach().cpu().numpy().copy()  # (13,)
        (qi, qdi, _, _, _, z_init, tilt_init) = G.read_state_urdf(
            w, urdf_from_nuka_slot)
        base_xy0 = pose_buf[0, 0, 0:2].detach().cpu().numpy().copy()

        last_action = np.zeros((1, 12), np.float32)
        nan_seen = False
        for k in range(policy_steps):
            (q_u, qd_u, blv, bav, pg, base_z, tilt) = G.read_state_urdf(
                w, urdf_from_nuka_slot)
            obs = G.build_obs(blv[None, :], bav[None, :], pg[None, :], cmd,
                              q_u[None, :], qd_u[None, :], last_action)
            if not np.isfinite(obs).all():
                nan_seen = True
                print(f"  [abort] non-finite obs at control step {k}", flush=True)
                break
            action = G.policy_forward(policy, obs)
            action = np.clip(action, -G.ACTION_CLIP, G.ACTION_CLIP)
            last_action = action.copy()
            target_urdf = G.DEFAULT_ANGLES + G.ACTION_SCALE * action[0]
            # Write the per-control-step DriveTarget ONCE (URDF -> nuka-slot mapped).
            G.write_targets_urdf(w, urdf_from_nuka_slot, target_urdf)
            nuka.sync()
            # The DriveTarget that will be held across the DECIMATION sub-steps:
            dt_now = drive_buf[0].detach().cpu().numpy().copy()  # (13,) nuka-slot order
            # Advance DECIMATION sim sub-steps, snapshotting each (the C++ physics
            # replay uploads `dt_now` and steps once, per recorded sim-step).
            for _ in range(G.DECIMATION):
                w.step()
                nuka.sync()
                drive_records.append(dt_now.copy())
                pose_records.append(pose_buf[0].detach().cpu().numpy().copy())

            if (k % 50) == 0 or k == policy_steps - 1:
                print(f"  t={k*G.DT*G.DECIMATION:5.2f}s ctrl{k:4d}  "
                      f"base_z={base_z:.4f} tilt={tilt:5.1f}deg  "
                      f"fwd_vel(body_x)={blv[0]:+.3f}", flush=True)

        nuka.sync()
        base_xyf = pose_buf[0, 0, 0:2].detach().cpu().numpy().copy()
        (qf, qdf, _, _, _, z_final, tilt_final) = G.read_state_urdf(
            w, urdf_from_nuka_slot)

    drive_arr = (np.stack(drive_records) if drive_records
                 else np.zeros((1, G.GO2_BLC), np.float32)).astype(np.float32)
    pose_arr = (np.stack(pose_records) if pose_records
                else np.zeros((1, G.GO2_BLC, 7), np.float32)).astype(np.float32)
    num_steps = drive_arr.shape[0]
    sim_time = num_steps * G.DT
    dx_world = float(base_xyf[0] - base_xy0[0])
    dy_world = float(base_xyf[1] - base_xy0[1])
    z_min = float(pose_arr[:, 0, 2].min())
    z_max = float(pose_arr[:, 0, 2].max())

    return dict(
        drive=drive_arr, pose=pose_arr, num_steps=num_steps,
        sim_time=sim_time, nan=nan_seen,
        dx_world=dx_world, dy_world=dy_world,
        vx_world=dx_world / sim_time if sim_time > 0 else 0.0,
        z_init=z_init, z_final=z_final, z_min=z_min, z_max=z_max,
        tilt_init=tilt_init, tilt_final=tilt_final,
        init_link_pose=init_link_pose, init_drive_target=init_drive_target,
        q_init_urdf=qi, q_final_urdf=qf,
    )


def write_binary(out_path, cap):
    """Write the flat binary trajectory (see module docstring for the layout)."""
    drive = cap["drive"]          # (N, 13) f32
    pose = cap["pose"]            # (N, 13, 7) f32
    N = cap["num_steps"]
    dof_count = drive.shape[1]    # 13
    link_count = pose.shape[1]    # 13
    pose_floats = pose.shape[2]   # 7

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        # Header: 8 int32 then 2 float32 (dt, ctrl_dt), all little-endian.
        f.write(struct.pack(
            "<8i2f",
            MAGIC, VERSION, N, dof_count, link_count, pose_floats,
            G.DECIMATION, int(round(1.0 / (G.DT * G.DECIMATION) * 1000)),
            float(G.DT), float(G.DT * G.DECIMATION),
        ))
        # Records: interleave [drive(13), pose(13*7)] per step, contiguous f32.
        # Build a single (N, 13 + 91) array and write it raw (little-endian f32).
        flat = np.concatenate(
            [drive.reshape(N, dof_count),
             pose.reshape(N, link_count * pose_floats)],
            axis=1,
        ).astype("<f4")
        f.write(flat.tobytes(order="C"))
    return os.path.getsize(out_path)


def write_meta(meta_path, cap, out_bin):
    """Write the human-readable setup-contract sidecar JSON for the C++ tool."""
    meta = {
        "format": "go2_walk_trajectory.bin v1 (flat little-endian)",
        "binary_file": os.path.basename(out_bin),
        "magic_hex": "0x474F3257",
        "magic_note": (
            "On disk the first 4 bytes are little-endian: 0x57 0x32 0x4F 0x47 "
            "(= ASCII 'W2OG'). Read as a little-endian int32 this equals "
            "0x474F3257; reading those 4 chars in source order spells 'GO2W'. "
            "C++ tool: read a uint32 little-endian and compare == 0x474F3257u."
        ),
        "version": VERSION,
        "header_layout": (
            "8 int32 [magic, version, num_steps, dof_count(13), link_count(13), "
            "pose_floats(7), decimation(4), control_hz_x1000(50000)] then 2 float32 "
            "[dt(0.005), ctrl_dt(0.020)]"
        ),
        "record_layout": (
            "per recorded PHYSICS sub-step: drive_target[dof_count=13] f32 "
            "(slot0=root unused, slots1..12 = 12 joints in nuka-slot order) then "
            "link_pose[link_count=13][pose_floats=7] f32 ([px,py,pz,qw,qx,qy,qz] "
            "world, quat w-first; link0=base, links1..12 = joints in nuka-slot order)"
        ),
        "record_bytes": int((cap["drive"].shape[1] + cap["pose"].shape[1] *
                             cap["pose"].shape[2]) * 4),
        "num_steps": int(cap["num_steps"]),
        "dof_count": int(cap["drive"].shape[1]),
        "link_count": int(cap["pose"].shape[1]),
        "pose_floats": int(cap["pose"].shape[2]),
        "dt": float(G.DT),
        "decimation": int(G.DECIMATION),
        "control_period_s": float(G.DT * G.DECIMATION),
        "control_hz": float(1.0 / (G.DT * G.DECIMATION)),
        "sim_time_s": float(cap["sim_time"]),
        # ---- the C++-replay SETUP CONTRACT ----
        "setup_contract": {
            "physics_scene": "examples/scenes/go2_float.usda",
            "render_scene": "examples/scenes/go2.nks",
            "note_scene": (
                "Physics simulated on go2_float.usda (USDA, what nuka.World loads); "
                "the go2.nks visual meshes share the SAME 12-joint articulation + "
                "cook order, so the dumped link_pose drives go2.nks meshes 1:1 "
                "(pose-replay). For physics-replay, re-cook go2_float.usda and upload "
                "drive_target each sim-step (D1 reproduces this walk)."
            ),
            "ground_plane": (
                "IMPLICIT half-space at world z = 0 (model.ground_height=0, "
                "src/c_abi/world.cpp:168). No explicit ground mesh in the scene; the "
                "renderer should draw a ground plane at z=0."
            ),
            "control_mode": "PD_POSITION (control_mode=0)",
            "policy_gains": {"Kp": G.KP, "Kd": G.KD,
                             "force_limit_urdf_order": G.FORCE_LIMIT_URDF.tolist()},
            "drive_target_upload_order": (
                "DRIVE_TARGET buffer is (env, 13); slot 0 = root (leave 0), slots "
                "1..12 = the 12 actuated joints in NUKA-SLOT (cook) order. The dumped "
                "drive_target[13] is ALREADY in this layout -- upload it verbatim."
            ),
            "joint_order_nuka_slot": (
                "links/slots 1..12 = FL,FR,RL,RR x (hip,thigh,calf) = the cook order "
                "(verified by the D1 structural permutation + cooked-q signature)"
            ),
            "urdf_from_nuka_slot": cap["_perm"],
            "default_angles_urdf": G.DEFAULT_ANGLES.tolist(),
            "initial_base_z_cooked": 0.445,
            "initial_stance_after_warmstart": {
                "warm_seconds": 0.2, "warm_gains": {"Kp": G.KP, "Kd": G.KD},
                "base_z": float(cap["z_init"]), "tilt_deg": float(cap["tilt_init"]),
                "q_urdf": [round(float(x), 5) for x in cap["q_init_urdf"]],
                "init_link_pose_link0": [round(float(x), 5)
                                         for x in cap["init_link_pose"][0].tolist()],
                "note": (
                    "The trajectory's step-0 link_pose IS the state right AFTER the "
                    "first sim sub-step from this warm-started stance. A pose-replay "
                    "just plays link_pose frames. A physics-replay must first warm- "
                    "start: hold DEFAULT_ANGLES at Kp=20/Kd=0.5 for 0.2 s (40 steps "
                    "@ dt=0.005), THEN upload the dumped drive_targets."
                ),
            },
            "cmd": [0.5, 0.0, 0.0],
            "policy": "/root/third_party/go2_pr62/motion.pt (golden-validated)",
        },
        # ---- walk validation ----
        "walk_validation": {
            "walks": bool((not cap["nan"]) and cap["dx_world"] > 0.3
                          and cap["tilt_final"] < 35.0 and cap["z_min"] > 0.18),
            "base_dx_world_m": round(cap["dx_world"], 4),
            "base_dy_world_m": round(cap["dy_world"], 4),
            "vx_world_mps": round(cap["vx_world"], 4),
            "cmd_vx_mps": 0.5,
            "base_z_init_m": round(cap["z_init"], 4),
            "base_z_final_m": round(cap["z_final"], 4),
            "base_z_min_m": round(cap["z_min"], 4),
            "base_z_max_m": round(cap["z_max"], 4),
            "tilt_init_deg": round(cap["tilt_init"], 3),
            "tilt_final_deg": round(cap["tilt_final"], 3),
            "nan_seen": bool(cap["nan"]),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="sim seconds to dump (default 8 -> ~8 s clip)")
    ap.add_argument("--cmd-vx", type=float, default=0.5,
                    help="forward command vx (validated 0.5)")
    ap.add_argument("--out", default="out/go2_walk_trajectory.bin")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 78)
    print("[demo] Dumping the PROVEN Go2 policy WALK trajectory for C++ replay")
    print("HONESTY: replays an EXTERNALLY-trained, Nuka-VALIDATED policy.")
    print("=" * 78, flush=True)

    policy = G.load_policy()
    dev = nuka.Device.create(0)
    try:
        # D1: discover the structural joint permutation (the proven path). Refuse to
        # dump a scrambled-joint rollout.
        ladder = G.Ladder()
        perm, perm_valid = G.d1_permutation(ladder, dev)
        if not perm_valid:
            print("[ABORT] joint permutation invalid -> refusing to dump a "
                  "scrambled-joint trajectory.", flush=True)
            return 1

        policy_steps = int(round(args.seconds / (G.DT * G.DECIMATION)))
        cmd_vec = (float(args.cmd_vx), 0.0, 0.0)
        cap = dump_walk(dev, policy, perm, policy_steps, cmd_vec)
        cap["_perm"] = perm.tolist()
    finally:
        dev.close()

    # ---- WALK GATE: refuse to ship a non-walk trajectory. ----
    walked = (not cap["nan"]) and cap["dx_world"] > 0.3 \
        and cap["tilt_final"] < 35.0 and cap["z_min"] > 0.18
    print("\n" + "=" * 78)
    print("DUMP VERDICT (walk gate)")
    print("=" * 78)
    print(f"  num_steps(sim sub-steps)={cap['num_steps']}  "
          f"sim_time={cap['sim_time']:.2f}s")
    print(f"  base net world dx={cap['dx_world']:+.3f} m (dy={cap['dy_world']:+.3f}) "
          f"-> vx_world={cap['vx_world']:+.4f} m/s (cmd {args.cmd_vx})")
    print(f"  base_z: init {cap['z_init']:.4f} -> final {cap['z_final']:.4f} "
          f"(min {cap['z_min']:.4f}, max {cap['z_max']:.4f}) m")
    print(f"  tilt: init {cap['tilt_init']:.2f} -> final {cap['tilt_final']:.2f} deg")
    print(f"  NaN seen: {cap['nan']}")
    print(f"  WALK GATE: {'PASS -> writing trajectory' if walked else 'FAIL -> NOT a clean walk'}")
    print("=" * 78, flush=True)

    if not walked:
        print("[ABORT] dump did not produce a clean forward walk; not writing the "
              "trajectory. Investigate before rendering.", flush=True)
        return 1

    out_bin = os.path.abspath(args.out)
    sz = write_binary(out_bin, cap)
    meta_path = os.path.splitext(out_bin)[0] + ".meta.json"
    write_meta(meta_path, cap, out_bin)

    print(f"\nWROTE {out_bin}  ({sz/1e6:.3f} MB, {cap['num_steps']} steps)")
    print(f"WROTE {meta_path}  (setup contract for the C++ replay tool)")
    print("Layout: header [8 int32 + 2 float32] then num_steps records of "
          "[drive_target(13) + link_pose(13*7=91)] float32 little-endian.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
