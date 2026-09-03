"""Ask whether a finger-blocked gripper reports drive resistance.

If the pads couple to the articulation, closing onto the rim must leave a
tracking error between drive_target and q. If they are decoupled, the fingers
reach the commanded aperture exactly.
"""
import sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'.nuka-runs')); sys.path.insert(0, str(REPO/'python'))
from wrenchprobe import load_batched, SCENE, RIM_Z, RIM_R
import nuka
out = []
ns = load_batched()
with nuka.Device.create(0) as device:
    c = ns['LiberoBlackBowlController'](str(SCENE), device, control_backend='osc', render_quality='preview')
    c.reset()
    for _ in range(150): c.step()
    bp = c.rigid_pose[c.target_body_index,:3].cpu().numpy().copy()
    at = [float(bp[0]), float(bp[1]+RIM_R), float(RIM_Z-0.018)]
    def hold(g, n):
        for _ in range(n):
            c.task_target[0,0], c.task_target[0,1], c.task_target[0,2] = at
            c.drive_target[0,8]=g; c.drive_target[0,9]=g
            c.step(advance_gripper=False)
    hold(0.040, 300)
    out.append('straddling rim: q8 %.5f q9 %.5f (cmd 0.04000)' % (float(c.q[0,8]), float(c.q[0,9])))
    # Close hard onto the rim wall. A coupled pad cannot reach 0.
    hold(0.0, 400)
    out.append('closed on rim : q8 %.5f q9 %.5f (cmd 0.00000)' % (float(c.q[0,8]), float(c.q[0,9])))
    out.append('  -> residual aperture %.5f m' % (float(c.q[0,8])+float(c.q[0,9])))
    # Now close in FREE SPACE, far from the bowl, as the control comparison.
    free = [float(bp[0]), float(bp[1]+0.35), float(RIM_Z+0.10)]
    for _ in range(400):
        c.task_target[0,0], c.task_target[0,1], c.task_target[0,2] = free
        c.drive_target[0,8]=0.040; c.drive_target[0,9]=0.040
        c.step(advance_gripper=False)
    for _ in range(400):
        c.task_target[0,0], c.task_target[0,1], c.task_target[0,2] = free
        c.drive_target[0,8]=0.0; c.drive_target[0,9]=0.0
        c.step(advance_gripper=False)
    out.append('closed in air : q8 %.5f q9 %.5f' % (float(c.q[0,8]), float(c.q[0,9])))
    out.append('  -> residual aperture %.5f m' % (float(c.q[0,8])+float(c.q[0,9])))
    c.close()
Path(REPO/'.nuka-runs/coupling.log').write_text('\n'.join(out))
