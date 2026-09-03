import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
RIM_Z, RIM_R = 0.9320, 0.0471

def test_pair_cap(max_pairs, label):
    print(f'\n=== Testing with max_pairs={max_pairs} ({label}) ===', flush=True)
    with nuka.Device.create(0) as device:
        c = LiberoBlackBowlController(SCENE, device, control_backend="osc",
                                       solver_max_pairs=max_pairs)
        c.reset()
        for _ in range(100): c.step(advance_gripper=False)
        w = c.world
        cp = torch.from_dlpack(w.buffer_view(nuka.CONTACT_POINTS)).reshape(2,-1,3)
        cf = torch.from_dlpack(w.buffer_view(nuka.CONTACT_FORCE)).reshape(2,-1,3)
        cn = torch.from_dlpack(w.buffer_view(nuka.CONTACT_NORMAL)).reshape(2,-1,3)

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
            near = act & (p[:,2] > 0.90)
            print('%-8s grip %5.2fmm act %3d near_rim %2d bowlz %.4f' % (
                tag, c.q[0,8].item()*1000, int(act.sum()), int(near.sum()),
                c.target_position()[2].item()), flush=True)
            if near.sum() > 0:
                for i in np.nonzero(near)[0][:6]:
                    print('    slot %3d p %s Fn %.3f' % (
                        i, np.round(p[i],4).tolist(), f[i,0]), flush=True)

        drive(goal+np.array([0,0,0.10]), -1.0, 70, 'above')
        drive(goal, -1.0, 70, 'at_wall')
        drive(goal, +1.0, 60, 'closed')
        drive(goal+np.array([0,0,0.18]), +1.0, 90, 'lift')
        print('rise %+.4f' % (c.target_position()[2].item()-bp[2]))

test_pair_cap(512, 'default LIBERO_SOLVER_MAX_PAIRS')
test_pair_cap(2048, '4x increase')
test_pair_cap(4096, '8x increase')
