import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'

def run(kp_scale, limit_scale, label):
    with nuka.Device.create(0) as device:
        c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
        c.reset()
        c.drive_kp[0, 7] = lbb.LIBERO_OSC_KP * kp_scale
        c.drive_kd[0, 7] = lbb.LIBERO_OSC_KD * np.sqrt(kp_scale)
        c._osc_task_kp = lbb.LIBERO_OSC_KP * kp_scale
        c._osc_task_kd = lbb.LIBERO_OSC_KD * np.sqrt(kp_scale)
        c.drive_limit[:, 1:8] = torch.as_tensor(
            lbb.PANDA_FORCE_LIMIT[:7] * limit_scale, device=c.q.device)
        for _ in range(100):
            c.step()
        start = c.eef_pose()[:3].detach().cpu().numpy().copy()
        cmd = torch.zeros(7, device=c.q.device); cmd[0] = 1.0; cmd[6] = -1.0
        for _ in range(40):
            c.set_policy_action(cmd); c.step()
        eef = c.eef_pose()[:3].detach().cpu().numpy()
        tgt = c.task_target[0].detach().cpu().numpy()
        print('%-22s travel %+.4f m   lag %.4f' % (
            label, float(eef[0]-start[0]), float(np.linalg.norm(eef-tgt))))
        c.close()

run(1.0,  1.0, 'baseline')
run(1.0, 10.0, 'limit x10')
run(8.0,  1.0, 'kp x8')
run(8.0, 10.0, 'kp x8 + limit x10')
