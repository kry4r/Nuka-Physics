import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'

def run(substeps, label):
    with nuka.Device.create(0) as device:
        c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
        c.reset()
        for _ in range(100):
            c.step()
        print('dt=%.5f  control_hz implied=%.1f' % (c.dt, 1.0/c.dt))
        start = c.eef_pose()[:3].detach().cpu().numpy().copy()
        cmd = torch.zeros(7, device=c.q.device); cmd[0] = 1.0; cmd[6] = -1.0
        # 40 policy actions, each held for `substeps` physics ticks.
        for _ in range(40):
            c.set_policy_action(cmd)
            for _ in range(substeps):
                c.step()
        eef = c.eef_pose()[:3].detach().cpu().numpy()
        tgt = c.task_target[0].detach().cpu().numpy()
        print('%-26s travel %+.4f m   lag %.4f' % (
            label, float(eef[0]-start[0]), float(np.linalg.norm(eef-tgt))))
        c.close()

run(1,  '1 tick/action (current)')
run(25, '25 ticks/action (robosuite)')
