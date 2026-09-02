#!/usr/bin/env python3
"""Quick test: Try LIBERO with inverted Y/Z action axes."""

from pathlib import Path
import sys
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController, LIBERO_HOME
from nuka.vla.pi05_libero import LiberoPi05Policy

SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"
CHECKPOINT = REPO / ".nuka_cache/pi05-libero"
TOKENIZER = REPO / ".nuka_cache/paligemma-tokenizer"
TASK = "pick up the black bowl from table center and place it on the plate"

device = nuka.Device.create(0)
controller = LiberoBlackBowlController(
    SCENE, device, dt=0.005, control_backend="joint_pd"
)
controller.reset(LIBERO_HOME)
controller.attach_policy_cameras()
controller.world.step(100)  # settle

print("Loading pi0.5 policy...")
policy = LiberoPi05Policy(CHECKPOINT, TOKENIZER)
print(f"Policy loaded: {policy.parameter_count:,} parameters")

# Run 5 queries with axis inversion experiments
for experiment in ["baseline", "invert_y", "invert_z", "invert_yz", "negate_all"]:
    controller.reset(LIBERO_HOME)
    controller.world.step(100)

    print(f"\n=== Experiment: {experiment} ===")

    for query_id in range(5):
        cameras = controller.camera_images()
        state = controller.state8()

        result = policy.predict_chunk(
            state=state,
            agentview=cameras[0],
            eye_in_hand=cameras[1],
            task=TASK,
        )

        action = result.actions[0].clone()  # First action of chunk

        # Apply experiment transform
        if experiment == "invert_y":
            action[1] = -action[1]
        elif experiment == "invert_z":
            action[2] = -action[2]
        elif experiment == "invert_yz":
            action[1] = -action[1]
            action[2] = -action[2]
        elif experiment == "negate_all":
            action[:6] = -action[:6]

        controller.set_policy_action(action.cpu().numpy())
        controller.world.step(10)  # execute_steps=1 * 10 substeps

        eef = controller.eef_pose()[:3].cpu().numpy()
        target = controller.target_position().cpu().numpy()
        dist = np.linalg.norm(eef - target)

        print(f"  Query {query_id}: EEF={eef}, dist={dist:.3f}m, action[:3]={action[:3].cpu().numpy()}")

print("\nDone. Check which experiment moved toward the bowl.")
