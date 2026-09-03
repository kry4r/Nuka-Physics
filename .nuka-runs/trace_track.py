import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
with nuka.Device.create(0) as device:
    c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100):
        c.step()
    start = c.eef_pose()[:3].detach().cpu().numpy().copy()
    print('settled eef:', np.round(start,4).tolist())
    # Drive a constant full-scale +X command and see how far the EEF actually goes.
    cmd = torch.zeros(7, device=c.q.device); cmd[0] = 1.0; cmd[6] = -1.0
    for i in range(1, 41):
        c.set_policy_action(cmd)
        c.step()
        if i % 10 == 0:
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            tgt = c.task_target[0].detach().cpu().numpy()
            print('step %2d  eef %s  target %s  lag %.4f' % (
                i, np.round(eef,4).tolist(), np.round(tgt,4).tolist(),
                float(np.linalg.norm(eef-tgt))))
    eef = c.eef_pose()[:3].detach().cpu().numpy()
    print('total eef travel in +X: %+.4f m over 40 steps' % float(eef[0]-start[0]))
    c.close()
