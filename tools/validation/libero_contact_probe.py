#!/usr/bin/env python3
"""Run real CUDA LIBERO bowl drops without policy inference.

The bowl starts at rest above the table, plate, or a second bowl. Save the
complete trajectory and optional live frames, including the impact interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.xml"
NUKA_NAMESPACE = "https://nuka.physics/mjcf"


def drop_scene(case: str, output: Path, height: float) -> tuple[Path, np.ndarray, np.ndarray]:
    # Nuka MJCF extensions predate namespace declarations in generated assets.
    text = SCENE.read_text(encoding="utf-8")
    if "xmlns:nuka=" not in text:
        text = text.replace("<mujoco ", f'<mujoco xmlns:nuka="{NUKA_NAMESPACE}" ', 1)
    ET.register_namespace("nuka", NUKA_NAMESPACE)
    root = ET.fromstring(text)
    compiler = root.find("compiler")
    for key in ("meshdir", "texturedir"):
        compiler.set(key, str((SCENE.parent / compiler.get(key, ".")).resolve()))
    world = root.find("worldbody")
    target = world.find("body[@name='akita_black_bowl_1_main']")
    other = world.find("body[@name='akita_black_bowl_2_main']")
    plate = world.find("body[@name='plate_1_main']")
    # Start supports at table height, so their initial fall cannot mask a drop.
    for body, z in ((other, 0.8984), (plate, 0.8991)):
        position = np.fromstring(body.get("pos"), sep=" ")
        position[2] = z
        body.set("pos", " ".join(f"{value:.9g}" for value in position))
    target_position = np.fromstring(target.get("pos"), sep=" ")
    target_position[2] = 0.8984 + height
    if case == "plate":
        target_position[:2] = np.fromstring(plate.get("pos"), sep=" ")[:2]
        target_position[2] += 0.006
    elif case == "bowl":
        target_position[:2] = np.fromstring(other.get("pos"), sep=" ")[:2]
        target_position[2] += 0.055
    target.set("pos", " ".join(f"{value:.9g}" for value in target_position))
    scene = output / f"{case}.xml"
    ET.ElementTree(root).write(scene, encoding="utf-8", xml_declaration=True)
    return scene, target_position, np.fromstring(plate.get("pos"), sep=" ")


def run_case(device, case: str, margin: float, args) -> dict:
    output = args.out / f"{case}_margin{margin:g}_dt{args.dt:g}"
    output.mkdir(parents=True, exist_ok=True)
    scene, target, plate = drop_scene(case, output, args.height)
    controller = LiberoBlackBowlController(
        scene, device, dt=args.dt, control_backend="osc", render_quality="preview",
        solver_contact_margin=margin, target_reference=target, place_reference=plate,
    )
    poses = []
    try:
        steps = round(args.seconds / args.dt)
        for tick in range(steps + 1):
            position = controller.target_position().detach().cpu().numpy().copy()
            poses.append(position)
            if args.render and (tick % max(1, round(0.05 / args.dt)) == 0 or tick == steps):
                image = controller.world.render_beauty(
                    eye=(float(target[0]) + 0.17, float(target[1]) - 0.19, 1.07),
                    look=(float(target[0]), float(target[1]), 0.933),
                    up=(0.0, 0.0, 1.0), fov_deg=45.0,
                    width=640, height=480, spp=8,
                )
                Image.fromarray(image).save(output / f"frame_{tick:05d}.png")
            if tick != steps:
                state = controller.step(advance_gripper=False)
                if not state["finite"]:
                    raise RuntimeError(f"nonfinite state at tick {tick}")
        positions = np.asarray(poses)
        tail = positions[-max(2, round(0.5 / args.dt)):]
        summary = {
            "case": case, "dt": args.dt, "margin": margin,
            "initial_position": positions[0].tolist(),
            "final_position": positions[-1].tolist(),
            "minimum_z": float(positions[:, 2].min()),
            "minimum_z_time": float(positions[:, 2].argmin() * args.dt),
            "dip_below_final_z_m": float(positions[-1, 2] - positions[:, 2].min()),
            "tail_motion_m": float(np.linalg.norm(np.ptp(tail, axis=0))),
            "plate_position": controller.plate_position().detach().cpu().tolist(),
            "finite": bool(np.isfinite(positions).all()),
        }
        np.savez_compressed(output / "trajectory.npz", target_positions=positions, dt=args.dt)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        controller.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "out/libero/contact_probe")
    parser.add_argument("--cases", nargs="+", choices=("table", "plate", "bowl"), default=["table", "plate", "bowl"])
    parser.add_argument("--margins", type=float, nargs="+", default=[0.003, 0.0])
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--height", type=float, default=0.04)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.dt <= 0 or args.seconds <= 0 or args.height <= 0 or min(args.margins) < 0:
        parser.error("dt, seconds, height must be positive; margins must be non-negative")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with nuka.Device.create(0) as device:
        summaries = [run_case(device, case, margin, args) for margin in args.margins for case in args.cases]
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
