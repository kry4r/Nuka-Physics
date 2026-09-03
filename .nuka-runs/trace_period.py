import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'

def run(substeps, actions, label):
    with nuka.Device.create(0) as device:
        c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
        c.reset()
        for _ in range(100):
            c.step()
        start = c.eef_pose()[:3].detach().cpu().numpy().copy()
        cmd = torch.zeros(7, device=c.q.device); cmd[0] = 1.0; cmd[6] = -1.0
        for _ in range(actions):
            c.set_policy_action(cmd)
            for _ in range(substeps):
                c.step()
        eef = c.eef_pose()[:3].detach().cpu().numpy()
        print('%-30s travel %+.4f m  (%d ticks total)' % (
            label, float(eef[0]-start[0]), substeps*actions))
        c.close()

# Demo: 10 ticks/action. Compare equal wall-clock budgets (400 ticks = 2 s).
run(10, 40, 'demo 10 ticks/action')
run(25, 16, 'robosuite 25 ticks/action')
run(1, 400, '1 tick/action')
