"""Straddle the rim wall with the gripper CENTERED on the ring, then pinch.

The pads span 0.080 m when open and the ring diameter is 2*0.0471 = 0.0942 m,
so the fingers cannot straddle the full ring. Center on one wall segment: put
the gripper axis across the rim wall at the +Y extreme of the ring.
"""
import sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'.nuka-runs')); sys.path.insert(0, str(REPO/'python'))
from wrenchprobe import load_batched, SCENE, RIM_Z, RIM_R
import nuka
out=[]
ns = load_batched()
with nuka.Device.create(0) as device:
    c = ns['LiberoBlackBowlController'](str(SCENE), device, control_backend='osc', render_quality='preview')
    c.reset()
    for _ in range(150): c.step()
    bp = c.rigid_pose[c.target_body_index,:3].cpu().numpy().copy()
    def eef(): return c.eef_pose()[:3].detach().cpu().numpy().copy()
    def bowl(): return c.rigid_pose[c.target_body_index,:3].detach().cpu().numpy().copy()
    def fingers():
        lp = c.link_pose[0,:,:3].detach().cpu().numpy(); return lp[8], lp[9]
    def drive(goal, g, n):
        for _ in range(n):
            c.task_target[0,0],c.task_target[0,1],c.task_target[0,2]=goal
            c.drive_target[0,8]=g; c.drive_target[0,9]=g
            c.step(advance_gripper=False)
    # Finger midpoint sits ~0.053 m +Y of the eef site, so bias the waypoint -Y
    # to land the pad pair across the ring wall at y = bp[1] + RIM_R.
    f8, f9 = fingers(); e = eef()
    bias = float((f8[1]+f9[1])*0.5 - e[1])
    out.append('finger-midpoint Y bias vs eef site: %+.4f m' % bias)
    wall_y = float(bp[1] + RIM_R)
    above = [float(bp[0]), wall_y - bias, float(RIM_Z + 0.055)]
    drive(above, 0.040, 400)
    f8,f9 = fingers()
    out.append('above wall: eef %s' % np.round(eef(),4).tolist())
    out.append('  pads y %.4f / %.4f  (wall y %.4f)  bowl %s' % (f8[1], f9[1], wall_y, np.round(bowl(),4).tolist()))
    down = [above[0], above[1], float(RIM_Z - 0.015)]
    drive(down, 0.040, 400)
    f8,f9 = fingers()
    out.append('straddle : eef %s' % np.round(eef(),4).tolist())
    out.append('  pads z %.4f / %.4f  y %.4f / %.4f  bowl %s' % (f8[2],f9[2],f8[1],f9[1], np.round(bowl(),4).tolist()))
    drive(down, 0.0, 500)
    out.append('pinched  : q8 %.5f q9 %.5f residual %.5f' % (float(c.q[0,8]), float(c.q[0,9]), float(c.q[0,8])+float(c.q[0,9])))
    out.append('  bowl %s' % np.round(bowl(),4).tolist())
    lift = [above[0], above[1], float(RIM_Z + 0.13)]
    for _ in range(700):
        c.task_target[0,0],c.task_target[0,1],c.task_target[0,2]=lift
        c.drive_target[0,8]=0.0; c.drive_target[0,9]=0.0
        c.step(advance_gripper=False)
    b = bowl()
    out.append('lifted   : eef %s bowl %s  rise %+.4f m' % (
        np.round(eef(),4).tolist(), np.round(b,4).tolist(), b[2]-bp[2]))
    out.append('  => %s' % ('GRASPED' if b[2]-bp[2] > 0.04 else 'not lifted'))
    c.close()
Path(REPO/'.nuka-runs/centered.log').write_text('\n'.join(out))
