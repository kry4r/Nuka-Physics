"""Sweep the four camera orientations and report how close the policy gets."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

import nuka  # noqa: E402
from nuka.tasks import libero_black_bowl as lbb  # noqa: E402

SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"

ORIENTATIONS = {
    "none": (),
    "flipud": (0,),
    "fliplr": (1,),
    "rot180": (0, 1),
}


def patch(dims):
    """Re-derive the requested orientation from the shipped flipud handoff."""
    original = lbb.LiberoBlackBowlController.camera_images

    def camera_images(self):
        images = original(self)
        # Shipped code returns flipud(raw); undo it, then apply the sweep choice.
        raw = torch.flip(images, dims=(1,))
        if not dims:
            return raw
        return torch.flip(raw, dims=tuple(d + 1 for d in dims))

    lbb.LiberoBlackBowlController.camera_images = camera_images


def run(label, dims, seconds):
    from nuka.tasks.libero_black_bowl import LIBERO_TASK
    from nuka.vla.pi05_libero import LiberoPi05Policy

    CHECKPOINT = REPO / ".nuka_cache/pi05-libero"
    TOKENIZER = REPO / ".nuka_cache/paligemma-tokenizer"

    patch(dims)
    device_ctx = nuka.Device.create(0)
    device = device_ctx.__enter__()
    ctrl = lbb.LiberoBlackBowlController(str(SCENE), device, control_backend="osc")
    ctrl.reset()
    ctrl.attach_policy_cameras()
    for _ in range(100):
        ctrl.step()

    policy = LiberoPi05Policy(CHECKPOINT, TOKENIZER)
    ticks = int(seconds / ctrl.dt)
    execute_steps = 10
    action_period = int(round(1.0 / (20.0 * ctrl.dt)))

    bowl0 = ctrl.target_position().detach().cpu().numpy().copy()
    min_d, min_closed, max_z = 1e9, 1e9, float(bowl0[2])
    chunk, idx, next_tick, queries = None, 0, 0, 0

    for tick in range(ticks):
        if tick >= next_tick:
            if chunk is None or idx >= execute_steps:
                images = ctrl.camera_images()
                chunk = policy.predict_chunk(
                    ctrl.state8(), images[0], images[1], LIBERO_TASK,
                    seed=queries, profile=False,
                ).actions
                idx = 0
                queries += 1
            ctrl.set_policy_action(chunk[idx])
            idx += 1
            next_tick += action_period
        ctrl.step()
        eef = ctrl.eef_pose()[:3].detach().cpu().numpy()
        bowl = ctrl.target_position().detach().cpu().numpy()
        d = float(np.linalg.norm(eef - bowl))
        min_d = min(min_d, d)
        max_z = max(max_z, float(bowl[2]))
        if float(ctrl.q[0, 8] + ctrl.q[0, 9]) < 0.02:
            min_closed = min(min_closed, d)

    bowl1 = ctrl.target_position().detach().cpu().numpy().copy()
    ctrl.close()
    device_ctx.__exit__(None, None, None)
    return {
        "orientation": label,
        "queries": queries,
        "min_eef_bowl_m": round(min_d, 4),
        "min_when_closed_m": round(min_closed, 4) if min_closed < 1e8 else None,
        "bowl_moved_m": round(float(np.linalg.norm(bowl1 - bowl0)), 4),
        "bowl_rise_m": round(max_z - float(bowl0[2]), 4),
        "lifted": bool(max_z - float(bowl0[2]) > 0.04),
    }


if __name__ == "__main__":
    which = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    print("RESULT " + json.dumps(run(which, ORIENTATIONS[which], secs)))
