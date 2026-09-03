"""Measure finger reaction wrench during a scripted rim pinch.

LINK_CONTACT_WRENCH is assembled from the solved contact rows, so it reports
force on the finger links even when the FUSED contact_* slots stay empty.
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import nuka  # noqa: E402

SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"
RIM_Z, RIM_R = 0.9320, 0.0471


def load_batched():
    """Import the shipped controller with its single-env views widened to 2."""
    src = (REPO / "python/nuka/tasks/libero_black_bowl.py").read_text()
    src = src.replace("env_count=1,", "env_count=2,")
    src = src.replace(".view(1, 10, 7)", ".view(2, 10, 7)[:1][0].unsqueeze(0)")
    src = re.sub(r"\.view\(1, (10|7|3|4)\)",
                 lambda m: ".view(2, %s)[:1]" % m.group(1), src)
    ns = {"__name__": "lbb_batched", "__file__": str(REPO / "python/nuka/tasks/x.py")}
    exec(compile(src, "lbb_batched", "exec"), ns)
    return ns


def main():
    ns = load_batched()
    with nuka.Device.create(0) as device:
        c = ns["LiberoBlackBowlController"](
            str(SCENE), device, control_backend="osc", render_quality="preview"
        )
        c.reset()
        for _ in range(150):
            c.step()

        w = c.world
        wrench = torch.from_dlpack(
            w.buffer_view(nuka.LINK_CONTACT_WRENCH)
        ).reshape(2, -1, 6)
        print("link wrench shape", tuple(wrench.shape))

        bp = c.rigid_pose[c.target_body_index, :3].cpu().numpy().copy()
        print("bowl", np.round(bp, 4).tolist(), "rim_z", RIM_Z)

        def hold(goal, grip, ticks):
            for _ in range(ticks):
                c.task_target[0, 0] = goal[0]
                c.task_target[0, 1] = goal[1]
                c.task_target[0, 2] = goal[2]
                c.drive_target[0, 8] = grip
                c.drive_target[0, 9] = grip
                c.step(advance_gripper=False)

        def report(tag):
            f = wrench[0, :, :3].detach().cpu().numpy()
            mag = np.linalg.norm(f, axis=-1)
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            bz = float(c.rigid_pose[c.target_body_index, 2])
            top = np.argsort(mag)[::-1][:4]
            print("%-8s grip %5.2fmm  eef %s  bowlz %.4f" % (
                tag, float(c.drive_target[0, 8]) * 1000.0,
                np.round(eef, 4).tolist(), bz))
            print("         finger|f| l8 %.3f l9 %.3f   top links %s" % (
                mag[8], mag[9],
                [(int(i), round(float(mag[i]), 3)) for i in top if mag[i] > 1e-4]))

        # Straddle the rim wall, then pinch it.
        near = np.array([bp[0], bp[1] + RIM_R, RIM_Z + 0.030])
        hold(near, 0.040, 250)
        report("above")

        at = np.array([bp[0], bp[1] + RIM_R, RIM_Z - 0.018])
        hold(at, 0.040, 250)
        report("at_wall")

        for step in range(24):
            hold(at, 0.040 - 0.040 * (step + 1) / 24.0, 12)
        report("closed")

        lift = np.array([bp[0], bp[1] + RIM_R, RIM_Z + 0.120])
        hold(lift, 0.0, 400)
        report("lift")
        print("rise %+.4f" % (float(c.rigid_pose[c.target_body_index, 2]) - bp[2]))
        c.close()


if __name__ == "__main__":
    main()
