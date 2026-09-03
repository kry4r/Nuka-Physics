"""Locate the pad collidable rows in the eef frame so waypoints can target them."""
import sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'.nuka-runs')); sys.path.insert(0, str(REPO/'python'))
from wrenchprobe import load_batched, SCENE, RIM_Z, RIM_R
import nuka
def quat_conj(q): return np.array([q[0],-q[1],-q[2],-q[3]])
def qrot(q,v):
    w,x,y,z = q; u=np.array([x,y,z]); return v + 2*np.cross(u, np.cross(u,v)+w*v)
out=[]
ns = load_batched()
with nuka.Device.create(0) as device:
    c = ns['LiberoBlackBowlController'](str(SCENE), device, control_backend='osc', render_quality='preview')
    c.reset()
    for _ in range(200): c.step()
    ep = c.eef_pose().detach().cpu().numpy(); e, q = ep[:3], ep[3:]
    lp = c.link_pose[0,:,:3].detach().cpu().numpy()
    rp = c.rigid_pose.detach().cpu().numpy()
    out.append('eef site %s  quat %s' % (np.round(e,4).tolist(), np.round(q,4).tolist()))
    out.append('link8 %s  link9 %s' % (np.round(lp[8],4).tolist(), np.round(lp[9],4).tolist()))
    # Pad proxy rows: earlier probe showed rows 20..27 tracking the fingers.
    qc = quat_conj(q)
    for i in range(18, 32):
        d = rp[i,:3] - e
        loc = qrot(qc, d)
        out.append('row %-3d world %s  eef-local %s' % (i, np.round(rp[i,:3],4).tolist(), np.round(loc,4).tolist()))
    lowest = min(range(20,28), key=lambda i: rp[i,2])
    out.append('lowest pad row %d at z %.4f -> %+.4f m BELOW eef site' % (
        lowest, rp[lowest,2], rp[lowest,2]-e[2]))
    c.close()
Path(REPO/'.nuka-runs/padgeom.log').write_text('\n'.join(out))
