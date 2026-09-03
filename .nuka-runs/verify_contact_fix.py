#!/usr/bin/env python3
"""Verify that the contact fix populates legacy fields from ucontact_*."""
import sys
sys.path.insert(0, 'build-win-editor/python')
import nuka
import numpy as np

device = nuka.Device(0)
desc = nuka.WorldDesc()
desc.env_count = 1
desc.dt = 0.005
desc.control_mode = nuka.ControlMode.OSC
desc.osc_task_link = 7
desc.contact_family = 1
desc.heightfield_terrain_type = 0

scene = ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"
world = nuka.create_from_scene(device, scene, desc)

# Step to steady state
for _ in range(100):
    world.step()

# Close gripper fully
world.osc_position_command[:, :3] = [0.39, -0.002, 0.96]
world.osc_position_command[:, 6] = 0.0  # close to 0mm

for _ in range(200):
    world.step()

# Check contact telemetry
cf = world.contact_force.copy()
cn = world.contact_normal.copy()
cp = world.contact_point.copy()

active_slots = np.where(np.abs(cf[:, 0]) > 1e-6)[0]
print(f"Active contact slots: {len(active_slots)}")

# Check for finger-bowl contacts near rim height (z ~ 0.93-0.96)
rim_contacts = []
for slot in active_slots:
    pt = cp[slot]
    if 0.90 < pt[2] < 0.97:
        rim_contacts.append(slot)
        print(f"slot {slot:3d}: point=[{pt[0]:7.4f}, {pt[1]:7.4f}, {pt[2]:7.4f}]  "
              f"normal=[{cn[slot,0]:6.3f}, {cn[slot,1]:6.3f}, {cn[slot,2]:6.3f}]  "
              f"force={cf[slot,0]:7.2f}N")

if len(rim_contacts) > 0:
    print(f"\n✓ SUCCESS: {len(rim_contacts)} finger-bowl contacts detected near rim")
    sys.exit(0)
else:
    print("\n✗ FAIL: No finger-bowl contacts detected")
    sys.exit(1)
