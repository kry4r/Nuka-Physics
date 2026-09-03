import sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'.nuka-runs')); sys.path.insert(0, str(REPO/'python'))
from wrenchprobe import load_batched, SCENE, RIM_Z, RIM_R
import nuka
out=[]
out.append('has ENV_STATUS: %s' % ('ENV_STATUS' in dir(nuka)))
ns = load_batched()
with nuka.Device.create(0) as device:
    c = ns['LiberoBlackBowlController'](str(SCENE), device, control_backend='osc', render_quality='preview')
    c.reset()
    es = torch.from_dlpack(c.world.buffer_view(nuka.ENV_STATUS)).reshape(-1)
    for _ in range(150): c.step()
    out.append('settle env_status %s' % es.cpu().numpy().tolist())
    bp = c.rigid_pose[c.target_body_index,:3].cpu().numpy().copy()
    at = [float(bp[0]), float(bp[1]+RIM_R), float(RIM_Z-0.018)]
    def hold(g,n):
        for _ in range(n):
            c.task_target[0,0],c.task_target[0,1],c.task_target[0,2]=at
            c.drive_target[0,8]=g; c.drive_target[0,9]=g
            c.step(advance_gripper=False)
    hold(0.040,300); out.append('at_wall env_status %s' % es.cpu().numpy().tolist())
    hold(0.0,400);   out.append('closed  env_status %s' % es.cpu().numpy().tolist())
    out.append('PairOverflow bit set: %s' % [bool(int(v) & 1) for v in es.cpu().numpy().tolist()])
    c.close()
Path(REPO/'.nuka-runs/status.log').write_text('\n'.join(out))
