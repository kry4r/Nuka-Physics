#!/usr/bin/env python3
"""Replay LIBERO actions through CUDA physics; timestamps exclude initialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--mode", choices=("actions", "drives"), default="actions")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--quality", choices=("preview", "high", "ultra"), default="high")
    parser.add_argument("--contact-window", type=float, nargs=2, metavar=("START", "END"))
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--margin", type=float)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    recording = np.load(args.run / "rollout.npz")
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    # Recover the authored step from the recorded integer rate instead of
    # promoting the np.float32 metadata to a different Python integration step.
    dt = 1.0 / summary["control"]["physics_rate_hz"]
    ticks = min(round(args.seconds / dt), len(recording["target_positions"]))
    events = dict(zip(recording["action_ticks"].tolist(), recording["actions"]))
    scene = args.scene or REPO / summary["scene"]
    rows = {name: [] for name in (
        "target_positions", "eef_poses", "gripper_positions", "joints",
        "drive_targets", "rigid_poses", "finger_forces", "finger_contact_counts",
        "peak_contact_force", "target_error", "eef_error", "joint_error",
        "joint_velocities", "finger_poses", "finger_wrenches",
    )}
    contacts = {name: [] for name in (
        "contact_tick", "contact_slot", "contact_link", "contact_force",
        "contact_point", "contact_normal",
        "contact_side_a_kind", "contact_side_b_kind", "contact_side_a_index", "contact_side_b_index",
    )}
    overrides = {} if args.margin is None else {"solver_contact_margin": args.margin}
    with nuka.Device.create(0) as device:
        c = LiberoBlackBowlController(
            scene, device, dt=dt, control_backend=summary["control"]["control_backend"],
            render_quality=args.quality, **overrides,
        )
        try:
            # Allocate readouts before the first step, avoiding a graph rebuild
            # in the middle of the contact event being measured.
            force = torch.from_dlpack(c.world.buffer_view(nuka.CONTACT_FORCE)).reshape(-1, 3)
            link = torch.from_dlpack(c.world.buffer_view(nuka.CONTACT_LINK)).reshape(-1)
            point = torch.from_dlpack(c.world.buffer_view(nuka.CONTACT_POINTS)).reshape(-1, 3)
            normal = torch.from_dlpack(c.world.buffer_view(nuka.CONTACT_NORMAL)).reshape(-1, 3)
            wrench = torch.from_dlpack(c.world.buffer_view(nuka.LINK_CONTACT_WRENCH)).reshape(-1, 6)
            side_views = [torch.from_dlpack(c.world.buffer_view(field)).reshape(-1)
                          for field in (nuka.Field.CONTACT_SIDE_A_KIND, nuka.Field.CONTACT_SIDE_B_KIND,
                                        nuka.Field.CONTACT_SIDE_A_INDEX, nuka.Field.CONTACT_SIDE_B_INDEX)]
            target_body_index, plate_body_index = c.target_body_index, c.plate_body_index
            for _ in range(int(recording["settle_steps"])):
                c.step([0.0] * 6 + [-1.0])
            c.camera_images()
            first_mismatch = None
            for tick in range(ticks):
                if tick in events:
                    c.camera_images()
                    if args.mode == "actions":
                        c.set_policy_action(events[tick])
                if args.mode == "drives":
                    c.task_target[0] = torch.as_tensor(recording["task_targets"][tick], device=c.q.device)
                    c.task_rotation_target[0] = torch.as_tensor(recording["task_rotations"][tick], device=c.q.device)
                    c.drive_target[0, 1:] = torch.as_tensor(recording["drive_targets"][tick], device=c.q.device)
                metrics = c.step(advance_gripper=args.mode == "actions")
                if not metrics["finite"]:
                    raise RuntimeError(f"nonfinite physics at tick {tick}")
                bowl = c.target_position().detach().cpu().numpy().copy()
                eef = c.eef_pose().detach().cpu().numpy().copy()
                joints = c.q[0, 1:].detach().cpu().numpy().copy()
                f = force.detach().cpu().numpy()
                links = link.detach().cpu().numpy()
                finger_forces = []
                finger_counts = []
                for finger in (8, 9):
                    mask = links == finger
                    finger_forces.append(f[mask].sum(axis=0))
                    finger_counts.append(int(np.count_nonzero(f[mask, 0] > 1e-4)))
                target_error = float(np.max(np.abs(bowl - recording["target_positions"][tick])))
                eef_error = float(np.max(np.abs(eef[:3] - recording["eef_positions"][tick])))
                joint_error = float(np.max(np.abs(joints - recording["joints"][tick])))
                if first_mismatch is None and max(target_error, eef_error, joint_error) > 1e-6:
                    first_mismatch = tick * dt
                values = (
                    bowl, eef, joints[7:], joints,
                    c.drive_target[0, 1:].detach().cpu().numpy().copy(),
                    c.rigid_pose.detach().cpu().numpy().copy(),
                    finger_forces, finger_counts, float(np.max(np.abs(f[:, 0]))),
                    target_error, eef_error, joint_error,
                    c.qd[0, 1:].detach().cpu().numpy().copy(),
                    c.link_pose[0, 8:10].detach().cpu().numpy().copy(),
                    wrench[8:10].detach().cpu().numpy().copy(),
                )
                for name, value in zip(rows, values):
                    rows[name].append(value)
                if args.contact_window and args.contact_window[0] <= tick * dt <= args.contact_window[1]:
                    active = np.flatnonzero(f[:, 0] > 1e-4)
                    values = (
                        np.full(len(active), tick, dtype=np.int32), active,
                        links[active].copy(), f[active].copy(),
                        point.detach().cpu().numpy()[active].copy(),
                        normal.detach().cpu().numpy()[active].copy(),
                        *(view.detach().cpu().numpy()[active].copy() for view in side_views),
                    )
                    for name, value in zip(contacts, values):
                        contacts[name].append(value)
                if args.render and tick % max(1, round(0.05 / dt)) == 0:
                    c.camera_images()
                    pixels = c.third_person_image(width=820, height=615, spp=4)
                    Image.fromarray(pixels).save(args.out / f"frame_{tick:05d}.png")
                if tick % max(1, round(0.5 / dt)) == 0:
                    print(json.dumps({
                        "time": tick * dt, "bowl": bowl.tolist(), "eef": eef[:3].tolist(),
                        "gripper_mm": (joints[7:] * 1000).tolist(),
                        "finger_normal_N": [float(v[0]) for v in finger_forces],
                        "target_error_mm": target_error * 1000,
                    }), flush=True)
        finally:
            c.close()
    arrays = {name: np.asarray(values) for name, values in rows.items()}
    contact_arrays = {name: np.concatenate(values) for name, values in contacts.items() if values}
    np.savez_compressed(args.out / "trajectory.npz", dt=dt, **arrays, **contact_arrays)
    results = {
        "mode": args.mode, "seconds": ticks * dt, "render": args.render,
        "target_body_index": target_body_index, "plate_body_index": plate_body_index,
        "first_mismatch_s": first_mismatch,
        "max_target_z": float(arrays["target_positions"][:, 2].max()),
        "min_target_z": float(arrays["target_positions"][:, 2].min()),
        "final_target_position": arrays["target_positions"][-1].tolist(),
        "peak_contact_force_N": float(arrays["peak_contact_force"].max()),
        "max_target_error_m": float(arrays["target_error"].max()),
        "max_joint_error_rad": float(arrays["joint_error"].max()),
    }
    maps = Path("/proc/self/maps")
    if maps.is_file():
        results["libraries"] = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                                        if "libnuka.so" in line})
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
