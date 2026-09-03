"""Compare Nuka's policy-camera orientation against the real LIBERO reference."""

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
REF = REPO / ".nuka-runs/libero_reference/libero_agentview_policy_input.png"
OUT = REPO / ".nuka-runs/camera_orient"


def gray_centroid(rgb):
    a = np.asarray(rgb, dtype=np.float32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    mask = (sat < 0.18) & (mx > 40) & (mx < 150)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return xs.mean() / w, ys.mean() / h


OUT.mkdir(parents=True, exist_ok=True)
with nuka.Device.create(0) as device:
    controller = LiberoBlackBowlController(
        str(SCENE), device, control_backend="osc", render_quality="preview"
    )
    images = controller.camera_images().detach().clamp(0, 1)
    pixels = images.mul(255).to(torch.uint8).cpu().numpy()
    agent = Image.fromarray(pixels[0])
    agent.save(OUT / "nuka_agentview_policy_input.png")
    controller.close()

ref_x, ref_y = gray_centroid(Image.open(REF).convert("RGB"))
nk_x, nk_y = gray_centroid(agent)
print("gray-structure centroid (x, y) in 0..1")
print("  reference : %.2f %.2f" % (ref_x, ref_y))
print("  nuka      : %.2f %.2f" % (nk_x, nk_y))
print("  dx %+.2f  dy %+.2f" % (nk_x - ref_x, nk_y - ref_y))
same_x = (ref_x < 0.5) == (nk_x < 0.5)
same_y = (ref_y < 0.5) == (nk_y < 0.5)
print("  same half x: %s   same half y: %s" % (same_x, same_y))
print("ORIENTATION MATCH" if same_x and same_y else "ORIENTATION MISMATCH")
