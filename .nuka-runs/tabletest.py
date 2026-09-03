"""Press the closed gripper straight down into the table.

The table is a large static/rigid box on the SAME general contact path. If the
finger links resolve to the articulation side, driving the eef below the table
surface must leave a standoff. If they do not, the eef sinks through.
"""
import sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'.nuka-runs')); sys.path.insert(0, str(REPO/'python'))
from wrenchprobe import load_batched, SCENE
import nuka
out=[]
ns = load_batched()
with nuka.Device.create(0) as device:
    c = ns['LiberoBlackBowlController'](str(SCENE), device, control_backend='osc', render_quality='preview')
    c.reset()
    for _ in range(150): c.step()
    e0 = c.eef_pose()[:3].detach().cpu().numpy().copy()
    out.append('settled eef %s' % np.round(e0,4).tolist())
    # Table top sits at ~0.9120 (rigid[0]/rigid[11] z). Drive well below it.
    goal = [float(e0[0]), float(e0[1]), 0.80]
    for _ in range(600):
        c.task_target[0,0],c.task_target[0,1],c.task_target[0,2]=goal
        c.drive_target[0,8]=0.0; c.drive_target[0,9]=0.0
        c.step(advance_gripper=False)
    e1 = c.eef_pose()[:3].detach().cpu().numpy()
    lp = c.link_pose[0,:,:3].detach().cpu().numpy()
    out.append('commanded z 0.8000 -> eef %s' % np.round(e1,4).tolist())
    out.append('  link8 z %.4f  link9 z %.4f' % (lp[8,2], lp[9,2]))
    out.append('  => %s' % ('BLOCKED by table (artic contact works)' if e1[2] > 0.86
                            else 'SANK THROUGH (finger links not on artic side)'))
    c.close()
Path(REPO/'.nuka-runs/tabletest.log').write_text('\n'.join(out))
