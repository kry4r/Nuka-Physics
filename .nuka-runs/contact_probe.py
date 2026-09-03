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
    w = c.world
    print('world attrs with contact:', [a for a in dir(w) if 'contact' in a.lower()])
    for a in ('get_contact_count','contact_count','get_contacts'):
        if hasattr(w, a):
            print('has', a)
    bowl = c.target_position().detach().cpu().numpy().copy()
    rim = bowl + np.array([0.047, 0.0, 0.034])
    def drive(goal, grip, n):
        for _ in range(n):
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            cmd = np.clip((goal-eef)/0.05, -1.0, 1.0)
            c.set_policy_action(torch.from_numpy(
                np.array([cmd[0],cmd[1],cmd[2],0,0,0,grip], dtype=np.float32)))
            for _ in range(10):
                c.step()
    drive(rim+np.array([0,0,0.08]), -1.0, 70)
    drive(rim, -1.0, 70)
    for a in ('get_contact_count','contact_count'):
        if hasattr(w, a):
            v = getattr(w, a)
            print('at_rim', a, v() if callable(v) else v)
    drive(rim, +1.0, 70)
    for a in ('get_contact_count','contact_count'):
        if hasattr(w, a):
            v = getattr(w, a)
            print('closed', a, v() if callable(v) else v)
    print('bowl z %.4f  fingers %.4f %.4f' % (
        c.target_position()[2].item(), c.q[0,8].item(), c.q[0,9].item()))
    c.close()
