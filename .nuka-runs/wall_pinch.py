import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
RIM_Z, RIM_R = 0.9320, 0.0471
for depth in (0.010, 0.018, 0.026):
    with nuka.Device.create(0) as device:
        c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
        c.reset()
        for _ in range(100): c.step(advance_gripper=False)
        bp = c.target_position().detach().cpu().numpy().copy()
        # Centre the finger span on the rim wall: offset +Y by the rim radius so
        # one pad drops inside the bowl and the other stays outside.
        goal = np.array([bp[0], bp[1] + RIM_R, RIM_Z - depth])
        def drive(g, grip, n):
            for _ in range(n):
                eef = c.eef_pose()[:3].detach().cpu().numpy()
                cmd = np.clip((g-eef)/0.05, -1.0, 1.0)
                c.set_policy_action(torch.from_numpy(
                    np.array([cmd[0],cmd[1],cmd[2],0,0,0,grip], dtype=np.float32)))
                for _ in range(10): c.step()
        drive(goal + np.array([0,0,0.10]), -1.0, 70)
        drive(goal, -1.0, 70)
        drive(goal, +1.0, 60)
        g_mm = c.q[0,8].item()*1000
        bz_close = c.target_position()[2].item()
        drive(goal + np.array([0,0,0.18]), +1.0, 90)
        b1 = c.target_position().detach().cpu().numpy()
        print('depth %.3f | grip %5.2fmm | bowl_z@close %.4f | rise %+.4f | xy_drift %.3f'
              % (depth, g_mm, bz_close, b1[2]-bp[2],
                 np.linalg.norm(b1[:2]-bp[:2])), flush=True)
        c.close()
