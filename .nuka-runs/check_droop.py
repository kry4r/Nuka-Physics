"""Measure steady-state arm droop off the LIBERO home pose per control backend."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import torch

import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"

# Official LIBERO reset reference (robosuite grasp site).
LIBERO_EEF = [-0.208700, -0.005800, 1.189800]

backend = sys.argv[1] if len(sys.argv) > 1 else "joint_pd"

with nuka.Device.create(0) as device:
    controller = LiberoBlackBowlController(
        str(SCENE), device, control_backend=backend, render_quality="preview"
    )
    hold_open = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], device=controller.q.device
    )
    home = controller.drive_target[0, 1:8].clone()
    for _ in range(200):
        controller.step(hold_open)

    q = controller.q[0, 1:8]
    err = (q - home).tolist()
    eef = controller.state8()[0].tolist()[:3]
    print("backend       :", backend)
    print("joint err(rad):", [round(v, 5) for v in err])
    print("max joint err :", "%.5f" % max(abs(v) for v in err))
    print("nuka  eef     :", [round(v, 5) for v in eef])
    print("libero eef    :", LIBERO_EEF)
    print(
        "eef err (m)   : %.5f"
        % max(abs(eef[i] - LIBERO_EEF[i]) for i in range(3))
    )
    controller.close()
