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
    print('bowl origin', np.round(bowl, 4).tolist())
    # Rim wall sits ~0.047 m out radially and ~0.034 m above the origin.
    rim = bowl + np.array([0.047, 0.0, 0.034])
    print('rim target  ', np.round(rim, 4).tolist())

    def drive(goal, grip, n, tag):
        for _ in range(n):
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            cmd = np.clip((goal - eef) / 0.05, -1.0, 1.0)
            act = np.array([cmd[0], cmd[1], cmd[2], 0, 0, 0, grip], dtype=np.float32)
            c.set_policy_action(torch.from_numpy(act))
            for _ in range(10):
                c.step()
        eef = c.eef_pose()[:3].detach().cpu().numpy()
        b = c.target_position().detach().cpu().numpy()
        print('%-8s eef %s bowl_z %.4f grip_mm %5.1f' % (
            tag, np.round(eef, 4).tolist(), b[2], c.q[0, 8].item() * 1000), flush=True)

    drive(rim + np.array([0, 0, 0.08]), -1.0, 70, 'above')
    drive(rim,                          -1.0, 70, 'at_rim')
    drive(rim,                          +1.0, 70, 'close')
    drive(rim + np.array([0, 0, 0.15]), +1.0, 90, 'lift')
    b = c.target_position().detach().cpu().numpy()
    print('RESULT bowl rise %+.4f m  (start %.4f -> %.4f)' % (
        b[2] - bowl[2], bowl[2], b[2]))
    c.close()
