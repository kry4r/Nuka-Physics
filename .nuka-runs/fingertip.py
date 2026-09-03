import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
with nuka.Device.create(0) as device:
    c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100): c.step(advance_gripper=False)
    lp = c.link_pose if hasattr(c, 'link_pose') else None
    print('controller link attrs:', [a for a in dir(c) if 'link' in a.lower()])
    bp = c.target_position().detach().cpu().numpy()
    rim_z, rim_r = 0.9320, 0.0471
    goal = bp + np.array([rim_r, 0.0, rim_z - bp[2]])
    def drive(goal, grip, n):
        for _ in range(n):
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            cmd = np.clip((goal-eef)/0.05, -1.0, 1.0)
            c.set_policy_action(torch.from_numpy(
                np.array([cmd[0],cmd[1],cmd[2],0,0,0,grip], dtype=np.float32)))
            for _ in range(10): c.step()
    drive(goal + np.array([0,0,0.10]), -1.0, 70)
    drive(goal, -1.0, 70)
    eef = c.eef_pose()[:3].detach().cpu().numpy()
    print('site   ', np.round(eef,4).tolist())
    if lp is not None:
        for li in (7, 8, 9):
            try:
                print('link%-2d %s' % (li, np.round(
                    lp[0, li, :3].detach().cpu().numpy(), 4).tolist()))
            except Exception as e:
                print('link', li, 'err', e)
    print('rim target Z %.4f  bowl top %.4f' % (goal[2], rim_z))
    print('bowl now', np.round(c.target_position().detach().cpu().numpy(),4).tolist())
    c.close()
