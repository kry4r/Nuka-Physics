"""Press the CLOSED gripper down onto the bowl rim from directly above.

The table test proved finger<->rigid contact resists. Repeating it against the
bowl separates 'no contact at all' from 'contact but wrong pinch geometry'.
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
    # Closed gripper, driven straight down THROUGH the rim ring.
    goal = [float(bp[0]), float(bp[1]+RIM_R), 0.86]
    for _ in range(700):
        c.task_target[0,0],c.task_target[0,1],c.task_target[0,2]=goal
        c.drive_target[0,8]=0.0; c.drive_target[0,9]=0.0
        c.step(advance_gripper=False)
    e = c.eef_pose()[:3].detach().cpu().numpy()
    bz = float(c.rigid_pose[c.target_body_index,2])
    out.append('closed gripper driven to z 0.860 over rim (rim_z %.4f)' % RIM_Z)
    out.append('  eef %s   bowlz %.4f (start %.4f)' % (np.round(e,4).tolist(), bz, bp[2]))
    out.append('  => %s' % ('rim RESISTED' if e[2] > RIM_Z - 0.005 else 'passed rim height'))
    # Now the same push offset OFF the bowl entirely (pure table reference).
    c.reset()
    for _ in range(150): c.step()
    goal2 = [float(bp[0]), float(bp[1]+0.30), 0.86]
    for _ in range(700):
        c.task_target[0,0],c.task_target[0,1],c.task_target[0,2]=goal2
        c.drive_target[0,8]=0.0; c.drive_target[0,9]=0.0
        c.step(advance_gripper=False)
    e2 = c.eef_pose()[:3].detach().cpu().numpy()
    out.append('same push over bare table: eef %s' % np.round(e2,4).tolist())
    out.append('  rim adds %+.4f m of standoff' % (e[2] - e2[2]))
    c.close()
Path(REPO/'.nuka-runs/pinchgeom.log').write_text('\n'.join(out))
