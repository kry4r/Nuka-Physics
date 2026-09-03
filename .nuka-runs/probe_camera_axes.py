"""Determine the raw framebuffer orientation by displacing the bowl in world space."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np
import torch
from PIL import Image

import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"
OUT = REPO / ".nuka-runs/camera_axes"
OUT.mkdir(parents=True, exist_ok=True)


def raw_agentview(controller):
    controller.attach_policy_cameras()
    controller.world.render_sensors()
    view = torch.from_dlpack(
        controller.world.get_sensor_view(nuka.SensorChannel.COLOR)
    )
    return view[0, 0].detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy().copy()


def arrival_centroid(base_rgb, moved_rgb):
    """Centroid of where the displaced body arrived, isolated by frame differencing."""
    a = np.asarray(base_rgb, dtype=np.float32).mean(axis=2)
    b = np.asarray(moved_rgb, dtype=np.float32).mean(axis=2)
    diff = b - a
    # Bowl is darker than the table, so arrival darkens pixels it now covers.
    mask = diff < -25.0
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    h, w = mask.shape
    return xs.mean() / w, ys.mean() / h


with nuka.Device.create(0) as device:
    controller = LiberoBlackBowlController(
        str(SCENE), device, control_backend="osc", render_quality="preview"
    )
    body = controller.target_body_index
    pos = controller.rigid_pose

    base = pos[body, :3].clone()
    print("bowl world start:", np.round(base.cpu().numpy(), 4).tolist())

    raw = {}
    for label, delta in (
        ("base", (0.0, 0.0, 0.0)),
        ("x_plus", (0.25, 0.0, 0.0)),
        ("y_plus", (0.0, 0.25, 0.0)),
    ):
        pos[body, 0] = base[0] + delta[0]
        pos[body, 1] = base[1] + delta[1]
        pos[body, 2] = base[2] + delta[2]
        nuka.sync()
        raw[label] = raw_agentview(controller)
        Image.fromarray(raw[label]).save(OUT / f"raw_{label}.png")

    pos[body, :3] = base
    nuka.sync()
    controller.close()

print("raw framebuffer, no flip applied (x,y in 0..1 image coords)")
for label in ("x_plus", "y_plus"):
    spot = arrival_centroid(raw["base"], raw[label])
    if spot is None:
        print("%-7s no arrival blob found" % label)
        continue
    print("world %s +0.25m -> image x %.3f y %.3f" % (label[0], spot[0], spot[1]))
