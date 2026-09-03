"""Report where the pad collidables actually are relative to the rim ring."""
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
    rp = c.rigid_pose.detach().cpu().numpy()
    out.append('rigid rows %d ; bowl body %d at %s' % (rp.shape[0], c.target_body_index, np.round(bp,4).tolist()))
    # Which rigid rows sit near the rim ring (the bowl's 40 boxes + proxies)?
    near = [(i, np.round(rp[i,:3],4).tolist()) for i in range(rp.shape[0])
            if abs(rp[i,2]-RIM_Z) < 0.02 and np.linalg.norm(rp[i,:2]-bp[:2]) < 0.09]
    out.append('rigid rows within 20mm of rim_z and 90mm of bowl axis: %d' % len(near))
    for i,p in near[:12]: out.append('   row %-4d %s' % (i,p))
    # Drive the OPEN gripper down so pads straddle the rim, then log pad rows.
    goal = [float(bp[0]), float(bp[1]+RIM_R), float(RIM_Z-0.018)]
    for _ in range(400):
        c.task_target[0,0],c.task_target[0,1],c.task_target[0,2]=goal
        c.drive_target[0,8]=0.040; c.drive_target[0,9]=0.040
        c.step(advance_gripper=False)
    lp = c.link_pose[0,:,:3].detach().cpu().numpy()
    rp2 = c.rigid_pose.detach().cpu().numpy()
    out.append('after straddle: eef %s' % np.round(c.eef_pose()[:3].detach().cpu().numpy(),4).tolist())
    out.append('  link8 %s  link9 %s' % (np.round(lp[8],4).tolist(), np.round(lp[9],4).tolist()))
    # Rigid rows that moved with the fingers == the pad collidables.
    moved = [(i, np.round(rp2[i,:3],4).tolist()) for i in range(rp2.shape[0])
             if np.linalg.norm(rp2[i,:3]-rp[i,:3]) > 0.05]
    out.append('rigid rows that moved >50mm (pads+arm proxies): %d' % len(moved))
    for i,p in moved[:16]: out.append('   row %-4d %s' % (i,p))
    c.close()
Path(REPO/'.nuka-runs/where.log').write_text('\n'.join(out))
