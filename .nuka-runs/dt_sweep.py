import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
for dt, sub in ((0.005, 10), (0.002, 25), (0.001, 50), (0.0005, 100)):
    with nuka.Device.create(0) as device:
        c = lbb.LiberoBlackBowlController(SCENE, device,
                                          control_backend="osc", dt=dt)
        c.reset()
        for _ in range(int(0.5/dt)):
            c.step(advance_gripper=False)
        b0 = c.target_position().detach().cpu().numpy().copy()
        rim = b0 + np.array([0.047, 0.0, 0.034])
        def drive(goal, grip, n):
            for _ in range(n):
                eef = c.eef_pose()[:3].detach().cpu().numpy()
                cmd = np.clip((goal-eef)/0.05, -1.0, 1.0)
                c.set_policy_action(torch.from_numpy(
                    np.array([cmd[0],cmd[1],cmd[2],0,0,0,grip], dtype=np.float32)))
                for _ in range(sub):
                    c.step()
        drive(rim+np.array([0,0,0.08]), -1.0, 60)
        drive(rim,                      -1.0, 60)
        drive(rim,                      +1.0, 50)
        bg = c.target_position().detach().cpu().numpy().copy()
        fing = c.q[0,8].item()
        drive(rim+np.array([0,0,0.15]), +1.0, 80)
        b1 = c.target_position().detach().cpu().numpy()
        print('dt=%.4f sub=%3d | grip_after_close %5.2fmm | bowl_at_close %s | rise %+.4f | drift %.3f'
              % (dt, sub, fing*1000, np.round(bg,3).tolist(), b1[2]-b0[2],
                 np.linalg.norm(b1[:2]-b0[:2])), flush=True)
        c.close()
