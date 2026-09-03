import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
RIM_Z, RIM_R = 0.9320, 0.0471
src = open('python/nuka/tasks/libero_black_bowl.py').read()
patched = src.replace('env_count=1,', 'env_count=2,')
open('.nuka-runs/_lbb2.py','w').write(patched)
sys.path.insert(0, '.nuka-runs')
import importlib.util
spec = importlib.util.spec_from_file_location('lbb2', '.nuka-runs/_lbb2.py')
lbb2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(lbb2)

with nuka.Device.create(0) as device:
    c = lbb2.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100): c.step(advance_gripper=False)
    w = c.world
    cf = w.buffer_view(nuka.CONTACT_FORCE)
    cl = w.buffer_view(nuka.CONTACT_LINK)
    print('CONTACT_FORCE shape', tuple(cf.shape), 'CONTACT_LINK', tuple(cl.shape))
    bp = c.target_position().detach().cpu().numpy().copy()
    goal = np.array([bp[0], bp[1] + RIM_R, RIM_Z - 0.018])
    def drive(g, grip, n, tag=None):
        for _ in range(n):
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            cmd = np.clip((g-eef)/0.05, -1.0, 1.0)
            c.set_policy_action(torch.from_numpy(
                np.array([cmd[0],cmd[1],cmd[2],0,0,0,grip], dtype=np.float32)))
            for _ in range(10): c.step()
        if tag:
            f = cf.torch()[0].detach().cpu().numpy()
            l = cl.torch()[0].detach().cpu().numpy()
            print('%-8s grip %5.2fmm links %s |f| %s' % (
                tag, c.q[0,8].item()*1000, l.tolist(),
                np.round(np.linalg.norm(f,axis=-1),3).tolist()), flush=True)
    drive(goal+np.array([0,0,0.10]), -1.0, 70, 'above')
    drive(goal, -1.0, 70, 'at_wall')
    drive(goal, +1.0, 60, 'closed')
    drive(goal+np.array([0,0,0.18]), +1.0, 90, 'lift')
    print('rise %+.4f' % (c.target_position()[2].item()-bp[2]))
