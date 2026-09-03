#!/usr/bin/env python3
"""Read narrowphase diagnostic values encoded in contact_point buffer."""
import numpy as np
import torch
from pathlib import Path

import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

REPO = Path(__file__).parent
SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"

def main():
    print("=== Reading Narrowphase Diagnostics from upoint ===\n")

    with nuka.Device.create(0) as device:
        controller = LiberoBlackBowlController(
            SCENE,
            device,
            dt=0.005,
            control_backend="joint_pd",
            render_quality="preview",
        )
        world = controller.world

        # Advance to grasp pose
        for _ in range(100):
            controller.step(advance_gripper=False)

        gripper_open = 0.04
        gripper_closed = 0.0
        for step in range(100):
            t = step / 100.0
            grip = gripper_open * (1 - t) + gripper_closed * t
            controller.q[0, 7] = grip
            controller.q[0, 8] = grip
            controller.step(advance_gripper=False)

        # Read contact_point buffer
        contact_point_view = world.buffer_view(nuka.CONTACT_POINTS)
        contact_point_raw = torch.from_dlpack(contact_point_view)
        point_slot_count = contact_point_raw.numel() // (world.env_count * 3)
        contact_point = contact_point_raw.view(world.env_count, point_slot_count, 3).cpu().numpy()

        # Decode diagnostics
        print("=== Diagnostic Values (from narrowphase kernel) ===")
        live = int(contact_point[0, 0, 0])
        rigid_slot_cap = int(contact_point[0, 0, 1])
        slot_stride = int(contact_point[0, 0, 2])
        print(f"live (pair_count[0]): {live}")
        print(f"rigid_slot_cap: {rigid_slot_cap}")
        print(f"slot_stride: {slot_stride}")

        env_count = int(contact_point[0, 1, 0])
        bodies_per_env = int(contact_point[0, 1, 1])
        print(f"env_count: {env_count}")
        print(f"bodies_per_env: {bodies_per_env}")

        print("\n=== Body Pairs (slots 0-9) ===")
        for slot in range(min(10, point_slot_count - 2)):
            idx = slot + 2
            if idx < point_slot_count:
                a = int(contact_point[0, idx, 0])
                b = int(contact_point[0, idx, 1])
                n = int(contact_point[0, idx, 2])
                print(f"Slot {slot}: pair=({a:2d}, {b:2d})  manifold_count={n}")

        print("\n=== Analysis ===")
        print(f"Narrowphase condition: slot < live && slot < rigid_slot_cap")
        print(f"For slot to run: slot < {live} && slot < {rigid_slot_cap}")
        if live == 0:
            print("\n[!] live=0 means broadphase generated ZERO pairs!")
            print("    The LBVH is not detecting any collisions.")
        elif rigid_slot_cap == 0:
            print("\n[!] rigid_slot_cap=0 means contact capacity is ZERO!")
            print("    Contacts are disabled or misconfigured.")
        elif rigid_slot_cap < live:
            print(f"\n[!] rigid_slot_cap ({rigid_slot_cap}) < live ({live})")
            print(f"    Only slots 0..{rigid_slot_cap-1} run, rest are filtered!")

if __name__ == "__main__":
    main()
