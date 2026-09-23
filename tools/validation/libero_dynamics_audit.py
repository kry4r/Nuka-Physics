#!/usr/bin/env python3
"""Measure joint dry-friction routing and free-body angular-frame consistency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
import nuka


def quaternion_product(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def friction_probe(device, output, mode, friction):
    fixture = (REPO / "tests/data/osc_passive_damping.xml").read_text()
    fixture = fixture.replace('damping="100"', f'damping="0" frictionloss="{friction}"')
    scene = output / f"friction_{mode}_{friction}.xml"
    scene.write_text(fixture)
    world = nuka.World.create_from_scene(
        device, str(scene), env_count=2, dt=0.002,
        control_mode=nuka.CONTROL_MODE_OSC if mode == "osc" else nuka.CONTROL_MODE_TORQUE,
        osc_task_link=1, contact_family=2,
    )
    try:
        qd = torch.from_dlpack(world.buffer_view(nuka.JOINT_VELOCITY)).view(2, 3)
        for field in (nuka.DRIVE_STIFFNESS, nuka.DRIVE_DAMPING, nuka.DRIVE_TARGET):
            torch.from_dlpack(world.buffer_view(field)).zero_()
        qd.zero_()
        qd[:, 2] = 0.05
        nuka.sync()
        world.step()
        nuka.sync()
        after = qd[:, 2].cpu().numpy().copy()
        return dict(mode=mode, frictionloss=friction, before=0.05,
                    after=after.tolist(), expected=max(0, 0.05-friction*0.002/1.1))
    finally:
        qd = None
        world.destroy()


def rotation_probe(device, output):
    scene = output / "free_rotation.xml"
    scene.write_text('''<mujoco><worldbody>
      <body name="free" pos="0 0 2" quat="0.707106781187 0 0.707106781187 0">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <geom type="box" size="0.1 0.2 0.3"/>
      </body>
    </worldbody></mujoco>''')
    world = nuka.World.create_from_scene(device, str(scene), env_count=1, dt=0.002)
    try:
        pose = torch.from_dlpack(world.buffer_view(nuka.RIGID_BODY_TRANSFORM)).view(-1, 7)
        omega = torch.from_dlpack(world.buffer_view(nuka.Field.BODY_ANGULAR_VELOCITY)).view(-1, 3)
        before = pose[0, 3:].cpu().numpy().copy()
        omega.zero_()
        omega[0, 0] = 1.0
        nuka.sync()
        for _ in range(10):
            world.step()
        nuka.sync()
        after = pose[0, 3:].cpu().numpy().copy()
        delta = np.array([np.cos(0.01), np.sin(0.01), 0., 0.])
        expected = quaternion_product(delta, before)
        right = quaternion_product(before, delta)
        return dict(before=before.tolist(), after=after.tolist(), expected_world=expected.tolist(),
                    world_error=float(np.linalg.norm(after-expected)),
                    body_frame_error=float(np.linalg.norm(after-right)))
    finally:
        pose = omega = None
        world.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with nuka.Device.create(0) as device:
        result = {"friction": [friction_probe(device, args.out, mode, friction)
                                for mode in ("osc", "torque") for friction in (0., 1.)],
                  "rotation": rotation_probe(device, args.out)}
    result["libraries"] = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                  if "libnuka.so" in line})
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
