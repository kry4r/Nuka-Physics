"""Verify Nuka's settled gripper aperture against measured real LIBERO values."""

import sys

sys.path.insert(0, "/mnt/c/Softwares/code/Nuka-Physics/python")

import torch

import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

SCENE = (
    "/mnt/c/Softwares/code/Nuka-Physics/"
    ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"
)

# Measured from real LIBERO after reset plus ten 20 Hz no-op open actions.
LIBERO_EEF = [-0.209911, -0.009284, 1.182197]
LIBERO_GRIPPER = [0.038734, -0.038725]

with nuka.Device.create(0) as device:
    controller = LiberoBlackBowlController(
        SCENE, device, control_backend="osc", render_quality="preview"
    )
    dummy_open = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], device=controller.q.device
    )
    print("tick  gripper_qpos[0]  gripper_qpos[1]  drive_target[8]")
    for tick in range(101):
        if tick and tick % 10 == 0 or tick in (1, 2, 4, 5):
            state = controller.state8()[0].tolist()
            print(
                "%4d  %+.6f        %+.6f        %+.6f"
                % (tick, state[6], state[7], float(controller.drive_target[0, 8]))
            )
        controller.step(dummy_open)

    state = controller.state8()[0].tolist()
    print()
    print("=== after 100 ticks (0.5 s) ===")
    print("nuka  eef     :", [round(v, 6) for v in state[:3]])
    print("libero eef    :", LIBERO_EEF)
    print(
        "eef error (m) : %.6f"
        % max(abs(state[i] - LIBERO_EEF[i]) for i in range(3))
    )
    print("nuka  gripper :", [round(state[6], 6), round(state[7], 6)])
    print("libero gripper:", LIBERO_GRIPPER)
    print(
        "gripper error : %.6f"
        % max(abs(state[6] - LIBERO_GRIPPER[0]), abs(state[7] - LIBERO_GRIPPER[1]))
    )
    controller.close()
