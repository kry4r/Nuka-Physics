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
    print('%-6s %-9s %-9s %-9s %-9s' % ('phase','tgt8','q8','tgt9','q9'))
    for grip, tag in ((-1.0,'open'), (+1.0,'close'), (-1.0,'reopen')):
        for _ in range(60):
            c.set_policy_action(torch.tensor(
                [0,0,0,0,0,0,grip], dtype=torch.float32))
            for _ in range(10):
                c.step()
        print('%-6s %-9.5f %-9.5f %-9.5f %-9.5f' % (
            tag, c.drive_target[0,8].item(), c.q[0,8].item(),
            c.drive_target[0,9].item(), c.q[0,9].item()), flush=True)
    print('drive_limit 8,9 =', c.drive_limit[0,8].item(), c.drive_limit[0,9].item())
    print('drive_kp    8,9 =', c.drive_kp[0,8].item(), c.drive_kp[0,9].item())
    c.close()
