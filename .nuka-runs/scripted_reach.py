import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
with nuka.Device.create(0) as device:
    c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100):
        c.step(advance_gripper=False)
    bowl = c.target_position().detach().cpu().numpy().copy()
    print('bowl', np.round(bowl,4).tolist())
    # Scripted P-controller straight at the bowl, gripper held open.
    for phase, (goal, grip, n) in enumerate([
        (bowl + np.array([0,0,0.10]), -1.0, 80),   # above
        (bowl + np.array([0,0,0.005]), -1.0, 80),  # descend
        (bowl + np.array([0,0,0.005]), +1.0, 60),  # close
        (bowl + np.array([0,0,0.18]), +1.0, 80),   # lift
    ]):
        for i in range(n):
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            err = goal - eef
            cmd = np.clip(err / 0.05, -1.0, 1.0)
            act = np.array([cmd[0],cmd[1],cmd[2],0,0,0,grip], dtype=np.float32)
            c.set_policy_action(torch.from_numpy(act))
            for _ in range(10):
                c.step()
        eef = c.eef_pose()[:3].detach().cpu().numpy()
        b = c.target_position().detach().cpu().numpy()
        print('phase%d eef %s  d=%.4f  bowl_z %.4f  grip_mm %.1f' % (
            phase, np.round(eef,4).tolist(), float(np.linalg.norm(eef-b)),
            b[2], c.q[0,8].item()*1000), flush=True)
    b = c.target_position().detach().cpu().numpy()
    print('RESULT bowl rise %+.4f m' % (b[2]-bowl[2]))
    c.close()
