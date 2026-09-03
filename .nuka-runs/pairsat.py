import sys, importlib.util
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
spec = importlib.util.spec_from_file_location('lbb3', '.nuka-runs/_lbb3.py')
lbb3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(lbb3)
SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
RIM_Z, RIM_R = 0.9320, 0.0471
with nuka.Device.create(0) as device:
    c = lbb3.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100): c.step(advance_gripper=False)
    w = c.world
    cp = torch.from_dlpack(w.buffer_view(nuka.CONTACT_POINTS)).reshape(2,-1,3)
    cf = torch.from_dlpack(w.buffer_view(nuka.CONTACT_FORCE)).reshape(2,-1,3)
    cn = torch.from_dlpack(w.buffer_view(nuka.CONTACT_NORMAL)).reshape(2,-1,3)
    print('slots/env', cf.shape[1], flush=True)
    bp = c.target_position().detach().cpu().numpy().copy()
    goal = np.array([bp[0], bp[1] + RIM_R, RIM_Z - 0.018])
    def drive(g, grip, n, tag):
        for _ in range(n):
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            cmd = np.clip((g-eef)/0.05, -1.0, 1.0)
            c.set_policy_action(torch.from_numpy(
                np.array([cmd[0],cmd[1],cmd[2],0,0,0,grip], dtype=np.float32)))
            for _ in range(10): c.step()
        p = cp[0].detach().cpu().numpy(); f = cf[0].detach().cpu().numpy()
        act = np.abs(f).sum(-1) > 1e-6
        # contacts whose point is near the rim height, i.e. finger/bowl candidates
        near = act & (p[:,2] > 0.90)
        print('%-8s grip %5.2fmm act %3d near_rim %2d bowlz %.4f' % (
            tag, c.q[0,8].item()*1000, int(act.sum()), int(near.sum()),
            c.target_position()[2].item()), flush=True)
        for i in np.nonzero(near)[0][:6]:
            print('    slot %3d p %s Fn %.3f n %s' % (i, np.round(p[i],4).tolist(),
                  f[i,0], np.round(cn[0,i].cpu().numpy(),2).tolist()), flush=True)
    drive(goal+np.array([0,0,0.10]), -1.0, 70, 'above')
    drive(goal, -1.0, 70, 'at_wall')
    drive(goal, +1.0, 60, 'closed')
    print('eef', np.round(c.eef_pose()[:3].cpu().numpy(),4).tolist())
    print('l8 ', np.round(c.link_pose[0,8,:3].cpu().numpy(),4).tolist())
    print('l9 ', np.round(c.link_pose[0,9,:3].cpu().numpy(),4).tolist())
    drive(goal+np.array([0,0,0.18]), +1.0, 90, 'lift')
    print('rise %+.4f' % (c.target_position()[2].item()-bp[2]))
