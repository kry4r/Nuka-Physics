import sys, numpy as np, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'.nuka-runs')); sys.path.insert(0, str(REPO/'python'))
from wrenchprobe import load_batched, SCENE
import nuka
out=[]
out.append('fields with BODY/ARTIC/LINK: %s' % [n for n in dir(nuka) if any(k in n for k in ('BODY_TO','ARTIC','LINK'))])
ns = load_batched()
with nuka.Device.create(0) as device:
    c = ns['LiberoBlackBowlController'](str(SCENE), device, control_backend='osc', render_quality='preview')
    c.reset()
    for _ in range(50): c.step()
    out.append('rigid_pose rows %d  link_pose %s  q %s' % (
        c.rigid_pose.shape[0], tuple(c.link_pose.shape), tuple(c.q.shape)))
    out.append('target_body_index %d' % c.target_body_index)
    # Where are the finger links vs rigid bodies?
    lp = c.link_pose[0,:, :3].cpu().numpy()
    out.append('link8 %s link9 %s' % (np.round(lp[8],4).tolist(), np.round(lp[9],4).tolist()))
    rp = c.rigid_pose[:, :3].cpu().numpy()
    for i in range(min(rp.shape[0], 24)):
        out.append('  rigid[%2d] %s' % (i, np.round(rp[i],4).tolist()))
    c.close()
Path(REPO/'.nuka-runs/maps.log').write_text('\n'.join(out))
