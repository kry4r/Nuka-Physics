#!/usr/bin/env python3
"""Verify that the contact fix populates legacy fields from ucontact_*."""
import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
RIM_Z = 0.9320

device = nuka.Device.create(0)
controller = LiberoBlackBowlController(SCENE, device, control_backend="osc")
controller.reset()
world = controller.world

# Step to steady state
for _ in range(100):
    controller.step(advance_gripper=False)

# Close gripper fully
target = controller.target_position().detach().cpu().numpy()
goal = np.array([target[0], target[1], RIM_Z - 0.02])

for step in range(200):
    eef = controller.eef_pose()[:3].detach().cpu().numpy()
    cmd = np.clip((goal - eef) / 0.05, -1.0, 1.0)
    grip = max(0.0, 1.0 - step / 100.0)  # close over 100 steps
    action = torch.from_numpy(np.array([cmd[0], cmd[1], cmd[2], 0, 0, 0, grip], dtype=np.float32))
    controller.set_policy_action(action)
    for _ in range(10):
        controller.step()

# Check contact telemetry
cp_buf = world.buffer_view(nuka.CONTACT_POINTS)
cf_buf = world.buffer_view(nuka.CONTACT_FORCE)
cn_buf = world.buffer_view(nuka.CONTACT_NORMAL)

cp = torch.from_dlpack(cp_buf).reshape(2, -1, 3).cpu().numpy()
cf = torch.from_dlpack(cf_buf).reshape(2, -1, 3).cpu().numpy()
cn = torch.from_dlpack(cn_buf).reshape(2, -1, 3).cpu().numpy()

active_slots = np.where(np.abs(cf[0, :, 0]) > 1e-6)[0]
print(f"Active contact slots: {len(active_slots)}")

# Check for finger-bowl contacts near rim height (z ~ 0.90-0.97)
rim_contacts = []
print(f"\nChecking {len(active_slots)} active slots for rim contacts (0.90 < z < 0.97):")
for i, slot in enumerate(active_slots[:10]):  # Show first 10
    pt = cp[0, slot]
    print(f"  slot {slot:3d}: z={pt[2]:7.4f}  fn={cf[0,slot,0]:7.2f}N")

for slot in active_slots:
    pt = cp[0, slot]
    if 0.90 < pt[2] < 0.97:
        rim_contacts.append(slot)
        print(f"RIM slot {slot:3d}: point=[{pt[0]:7.4f}, {pt[1]:7.4f}, {pt[2]:7.4f}]  "
              f"normal=[{cn[0,slot,0]:6.3f}, {cn[0,slot,1]:6.3f}, {cn[0,slot,2]:6.3f}]  "
              f"force={cf[0,slot,0]:7.2f}N")

if len(rim_contacts) > 0:
    print(f"\nSUCCESS: {len(rim_contacts)} finger-bowl contacts detected near rim")
    sys.exit(0)
else:
    print(f"\nFAIL: No finger-bowl contacts detected at rim height")
    print(f"All {len(active_slots)} contacts are at other heights (ground, etc)")
    sys.exit(1)
