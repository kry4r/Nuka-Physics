import sys, numpy as np, torch
from pathlib import Path
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
    hold(0.040, 250)
    for k in range(24): hold(0.040 - 0.040*(k+1)/24.0, 12)
    lw = torch.from_dlpack(c.world.buffer_view(nuka.LINK_CONTACT_WRENCH)).reshape(2,-1,6)
    cf = torch.from_dlpack(c.world.buffer_view(nuka.CONTACT_FORCE)).reshape(2,-1,3)
    m = np.linalg.norm(lw[0,:,:3].cpu().numpy(), axis=-1)
    out.append('fresh wrench %s' % np.round(m,3).tolist())
    out.append('Fn sum %.4f max %.4f' % (float(cf[0,:,0].abs().sum()), float(cf[0,:,0].abs().max())))
    hold(0.0, 5)
    m2 = np.linalg.norm(lw[0,:,:3].cpu().numpy(), axis=-1)
    out.append('after step %s' % np.round(m2,3).tolist())
    out.append('Fn2 sum %.4f max %.4f' % (float(cf[0,:,0].abs().sum()), float(cf[0,:,0].abs().max())))
    out.append('bowlz %.4f (start %.4f)' % (float(c.rigid_pose[c.target_body_index,2]), bp[2]))
    c.close()
Path(REPO/'.nuka-runs/wr2.log').write_text('\n'.join(out))
