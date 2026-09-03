#!/usr/bin/env python3
"""Diagnose broadphase pair generation for pi0.5 grasping issue.

Reads the candidate_pairs buffer to see what body pairs the broadphase
actually generates for the narrowphase.
"""
import numpy as np
import torch
from pathlib import Path

import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

REPO = Path(__file__).parent
SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"

def main():
    print("=== Broadphase Pair Generation Diagnostic ===\n")

    with nuka.Device.create(0) as device:
        controller = LiberoBlackBowlController(
            SCENE,
            device,
            dt=0.005,
            control_backend="joint_pd",
            render_quality="preview",
        )
        world = controller.world

        print(f"World: {world.env_count} envs, dt={world.dt}")
        print(f"Contact family: 2 (PairDriven)")

        # Advance to grasp pose
        gripper_open = 0.04
        gripper_closed = 0.0
        for _ in range(100):
            controller.step(advance_gripper=False)

        # Close gripper
        for step in range(100):
            t = step / 100.0
            grip = gripper_open * (1 - t) + gripper_closed * t
            controller.q[0, 7] = grip
            controller.q[0, 8] = grip
            controller.step(advance_gripper=False)

        # Try to read candidate_pairs buffer
        print("\n=== Checking Broadphase Output ===")
        try:
            pairs_view = world.buffer_view("candidate_pairs")
            pairs_raw = torch.from_dlpack(pairs_view)
            print(f"candidate_pairs buffer shape: {pairs_raw.shape}")
            print(f"candidate_pairs dtype: {pairs_raw.dtype}")

            # Pairs are stored as (env, slot, 2) where last dim is (a, b)
            if pairs_raw.numel() > 0:
                pairs = pairs_raw.view(-1, 2).cpu().numpy()
                print(f"Total pair slots: {len(pairs)}")

                # Count non-zero pairs
                nonzero_mask = (pairs[:, 0] != 0) | (pairs[:, 1] != 0)
                nonzero_pairs = pairs[nonzero_mask]
                print(f"Non-zero pairs: {len(nonzero_pairs)}")

                if len(nonzero_pairs) > 0:
                    print("\nFirst 20 pairs:")
                    for i, (a, b) in enumerate(nonzero_pairs[:20]):
                        print(f"  Slot {i:3d}: ({a:2d}, {b:2d})")

                # Check finger body indices (need to identify which bodies are fingers)
                print("\n=== Body identification ===")
                print("Panda finger bodies are typically the last few rigid bodies in the chain.")
                print("For Panda: base=0, links 1-6, link7 (hand), leftfinger, rightfinger")
                print("So fingers might be bodies 8 and 9 (if 10 total rigid bodies)")

        except Exception as e:
            print(f"Could not read candidate_pairs: {e}")

        # Also try pair_count
        try:
            count_view = world.buffer_view("pair_count")
            count_raw = torch.from_dlpack(count_view)
            count = count_raw.cpu().numpy()
            print(f"\n=== pair_count buffer ===")
            print(f"pair_count[0] = {count[0]}")
        except Exception as e:
            print(f"Could not read pair_count: {e}")

if __name__ == "__main__":
    main()
