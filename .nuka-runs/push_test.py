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
    b0 = c.target_position().detach().cpu().numpy().copy()
    print('bowl start', np.round(b0, 4).tolist())
    # Sweep the closed gripper laterally straight through the bowl centre.
    start = b0 + np.array([0.18, 0.0, 0.030])
    end   = b0 + np.array([-0.18, 0.0, 0.030])
    def drive(goal, grip, n):
        for _ in range(n):
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            cmd = np.clip((goal-eef)/0.05, -1.0, 1.0)
            c.set_policy_action(torch.from_numpy(
                np.array([cmd[0],cmd[1],cmd[2],0,0,0,grip], dtype=np.float32)))
            for _ in range(10):
                c.step()
    drive(start, +1.0, 80)
    print('at start  eef', np.round(c.eef_pose()[:3].detach().cpu().numpy(),4).tolist(),
          'bowl', np.round(c.target_position().detach().cpu().numpy(),4).tolist())
    drive(end, +1.0, 120)
    b1 = c.target_position().detach().cpu().numpy()
    print('after sweep eef', np.round(c.eef_pose()[:3].detach().cpu().numpy(),4).tolist())
    print('bowl end  ', np.round(b1,4).tolist())
    print('BOWL MOVED %.4f m' % np.linalg.norm(b1-b0))
    c.close()
