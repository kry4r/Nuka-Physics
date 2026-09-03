import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
with nuka.Device.create(0) as device:
    c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100):
        c.step()
    bowl = c.target_position().detach().cpu().numpy()
    eef  = c.eef_pose()[:3].detach().cpu().numpy()
    print('bowl origin :', np.round(bowl,4).tolist())
    print('eef  (settle):', np.round(eef,4).tolist())
    print('finger qpos :', [round(float(c.q[0,8]),5), round(float(c.q[0,9]),5)])
    print('LIBERO_TARGET_BOWL:', np.round(lbb.LIBERO_TARGET_BOWL,4).tolist())
    # Panda finger half-span when open, and bowl radius from the akita bowl asset.
    print()
    print('gap eef->origin  : %.4f' % float(np.linalg.norm(eef-bowl)))
    print('vertical  dz     : %+.4f' % float(eef[2]-bowl[2]))
    print('horizontal dxy   : %.4f' % float(np.linalg.norm(eef[:2]-bowl[:2])))
    c.close()
