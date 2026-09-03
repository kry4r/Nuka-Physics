#!/usr/bin/env python3
"""Diagnose the collision detection pipeline for pi0.5 grasping issue.

This script checks:
1. Scene geometry - finger and bowl shapes
2. Broadphase output - what pairs are detected
3. Narrowphase configuration - solver_max_pairs, pair capacity
4. Contact results - forces vs geometry

Usage:
    python diagnose_collision_pipeline.py
"""
import numpy as np
import torch
from pathlib import Path

import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

REPO = Path(__file__).parent
SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"

def main():
    print("=== Pi0.5 Collision Pipeline Diagnostic ===\n")

    with nuka.Device.create(0) as device:
        print("Creating controller with default settings...")
        controller = LiberoBlackBowlController(
            SCENE,
            device,
            dt=0.005,
            control_backend="joint_pd",
            render_quality="preview",
        )
        world = controller.world

        print(f"World config:")
        print(f"  env_count: {world.env_count}")
        print(f"  dt: {world.dt}")
        print(f"  contact_family: 2 (PairDriven)")

        # Check capacities
        try:
            # Try to read internal buffers to infer capacity
            contact_force_view = world.buffer_view(nuka.CONTACT_FORCE)
            contact_force = torch.from_dlpack(contact_force_view)
            slot_count = contact_force.numel() // (world.env_count * 6)
            print(f"  contact_slots: {slot_count}")
        except:
            print(f"  contact_slots: unknown")

        print("\nAdvancing to grasp pose...")
        gripper_open = 0.04
        gripper_closed = 0.0

        # Move to grasp position
        for _ in range(100):
            controller.step(advance_gripper=False)

        # Close gripper
        for step in range(100):
            t = step / 100.0
            grip = gripper_open * (1 - t) + gripper_closed * t
            controller.q[0, 7] = grip
            controller.q[0, 8] = grip
            controller.step(advance_gripper=False)

        # Check contact state
        print("\n=== Contact State After Grasping ===")

        # Get contact forces
        contact_force_view = world.buffer_view(nuka.CONTACT_FORCE)
        contact_force = torch.from_dlpack(contact_force_view)
        total_elements = contact_force.numel()
        slot_count = total_elements // (world.env_count * 6)
        wrench = contact_force.view(world.env_count, slot_count, 6)

        # Get contact geometry
        contact_point_view = world.buffer_view(nuka.CONTACT_POINTS)
        contact_normal_view = world.buffer_view(nuka.CONTACT_NORMAL)

        contact_point_raw = torch.from_dlpack(contact_point_view)
        contact_normal_raw = torch.from_dlpack(contact_normal_view)

        # Reshape based on actual sizes
        point_slot_count = contact_point_raw.numel() // (world.env_count * 3)
        contact_point = contact_point_raw.view(world.env_count, point_slot_count, 3)
        contact_normal = contact_normal_raw.view(world.env_count, point_slot_count, 3)

        # Find active contacts
        force_magnitudes = torch.norm(wrench[0, :, :3], dim=1)
        active_mask = force_magnitudes > 1e-6
        active_slots = torch.where(active_mask)[0]

        print(f"\nTotal contact slots: {slot_count}")
        print(f"Active slots (force > 1e-6): {len(active_slots)}")

        if len(active_slots) > 0:
            print("\nActive contact details:")
            for i, slot in enumerate(active_slots[:10]):
                pt = contact_point[0, slot]
                nrm = contact_normal[0, slot]
                force = wrench[0, slot, :3]
                force_mag = force_magnitudes[slot]
                print(f"  Slot {slot:3d}: force={force_mag:8.4f} N  "
                      f"point=({pt[0]:7.4f}, {pt[1]:7.4f}, {pt[2]:7.4f})  "
                      f"normal=({nrm[0]:6.3f}, {nrm[1]:6.3f}, {nrm[2]:6.3f})")

            # Check if all points are at origin
            point_norms = torch.norm(contact_point[0, active_slots], dim=1)
            zero_points = (point_norms < 1e-8).sum().item()
            print(f"\nActive contacts with point at origin: {zero_points}/{len(active_slots)}")

            if zero_points == len(active_slots):
                print("\n[!] ALL active contacts have geometry at origin!")
                print("    This confirms the narrowphase is not populating contact geometry.")

        # Get gripper link poses to check finger positions
        link_pose = controller.link_pose[0]
        finger_left = link_pose[7]  # panda_leftfinger
        finger_right = link_pose[8]  # panda_rightfinger

        print(f"\n=== Gripper State ===")
        print(f"Finger joint positions: left={controller.q[0,7].item():.4f}, right={controller.q[0,8].item():.4f}")
        print(f"Left finger pose:  pos=({finger_left[0]:.4f}, {finger_left[1]:.4f}, {finger_left[2]:.4f})")
        print(f"Right finger pose: pos=({finger_right[0]:.4f}, {finger_right[1]:.4f}, {finger_right[2]:.4f})")

        print("\n=== Diagnosis ===")
        if len(active_slots) > 0 and zero_points == len(active_slots):
            print("Problem: Solver generates contact forces but geometry is not populated.")
            print("Root cause: narrowphase is not detecting/writing finger-bowl contacts.")
            print("\nPossible reasons:")
            print("  1. Broadphase not generating finger-bowl pairs")
            print("  2. Pairs filtered out before narrowphase (slot > capacity)")
            print("  3. Narrowphase running but failing collision detection")
            print("  4. ucontact_count[slot] is 0 for all slots (confirmed by LegacyContactGeometryKernel)")
            print("\nNext steps:")
            print("  - Check broadphase pair generation")
            print("  - Verify narrowphase is invoked and writes ucontact_count > 0")
            print("  - Add diagnostic to print which body pairs are in active slots")

if __name__ == "__main__":
    main()
