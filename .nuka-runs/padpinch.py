"""Pinch the rim wall using PAD-referenced waypoints.

Pads sit within +/-6mm of the eef site (rows 20..27), so the eef site itself is
the pinch point. Straddle the wall, close, lift.
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
    for _ in range(200): c.step()
    bp = c.rigid_pose[c.target_body_index,:3].cpu().numpy().copy()
    def eef(): return c.eef_pose()[:3].detach().cpu().numpy().copy()
    def bowl(): return c.rigid_pose[c.target_body_index,:3].detach().cpu().numpy().copy()
    def pads():
        rp = c.rigid_pose.detach().cpu().numpy(); return rp[20:28,:3]
    def drive(goal, g, n):
        for _ in range(n):
            c.task_target[0,0],c.task_target[0,1],c.task_target[0,2]=goal
            c.drive_target[0,8]=g; c.drive_target[0,9]=g
            c.step(advance_gripper=False)
    wall = np.array([bp[0], bp[1] + RIM_R, RIM_Z])
    # Pads bracket the eef site in Y, so target the wall with the eef site.
    drive([float(wall[0]), float(wall[1]), float(RIM_Z + 0.08)], 0.040, 500)
    p = pads(); out.append('above : eef %s pad z %.4f..%.4f' % (
        np.round(eef(),4).tolist(), p[:,2].min(), p[:,2].max()))
    # Descend so pads bracket the rim wall vertically (wall spans ~z 0.925..0.932).
    drive([float(wall[0]), float(wall[1]), float(RIM_Z - 0.004)], 0.040, 600)
    p = pads(); out.append('straddle: eef %s pad z %.4f..%.4f  pad y %.4f..%.4f' % (
        np.round(eef(),4).tolist(), p[:,2].min(), p[:,2].max(), p[:,1].min(), p[:,1].max()))
    out.append('  bowl %s (rim wall y %.4f)' % (np.round(bowl(),4).tolist(), wall[1]))
    drive([float(wall[0]), float(wall[1]), float(RIM_Z - 0.004)], 0.0, 600)
    out.append('pinch : q8 %.5f q9 %.5f residual %.5f  bowl %s' % (
        float(c.q[0,8]), float(c.q[0,9]), float(c.q[0,8])+float(c.q[0,9]),
        np.round(bowl(),4).tolist()))
    for _ in range(900):
        c.task_target[0,0]=float(wall[0]); c.task_target[0,1]=float(wall[1])
        c.task_target[0,2]=float(RIM_Z+0.15)
        c.drive_target[0,8]=0.0; c.drive_target[0,9]=0.0
        c.step(advance_gripper=False)
    b = bowl()
    out.append('lift  : eef %s bowl %s rise %+.4f' % (
        np.round(eef(),4).tolist(), np.round(b,4).tolist(), b[2]-bp[2]))
    out.append('  => %s' % ('GRASPED' if b[2]-bp[2] > 0.04 else 'not lifted'))
    c.close()
Path(REPO/'.nuka-runs/padpinch.log').write_text('\n'.join(out))
