import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
from pathlib import Path
import nuka.tasks.libero_black_bowl as lbb
from nuka.vla.pi05_libero import LiberoPi05Policy

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
pol = LiberoPi05Policy(Path(".nuka_cache/pi05-libero"),
                       Path(".nuka_cache/paligemma-tokenizer"))
with nuka.Device.create(0) as device:
    c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100):
        c.step(advance_gripper=False)
    best = (9.9, None, None, None)
    chunk=None; idx=0
    for tick in range(400):
        if chunk is None or idx >= 10:
            im = c.camera_images()
            chunk = pol.predict_chunk(c.state8(), im[0], im[1], lbb.LIBERO_TASK,
                                      seed=tick, profile=False).actions
            idx = 0
        c.set_policy_action(chunk[idx]); idx += 1
        for _ in range(10):
            c.step()
        eef = c.eef_pose()[:3].detach().cpu().numpy().copy()
        b = c.target_position().detach().cpu().numpy().copy()
        d = float(np.linalg.norm(eef-b))
        if d < best[0]:
            best = (d, eef, b, (tick, c.q[0,8].item()*1000))
    d, eef, b, meta = best
    print('closest %.4f m at tick %d  grip_mm %.1f' % (d, meta[0], meta[1]))
    print('eef  ', np.round(eef,4).tolist())
    print('bowl ', np.round(b,4).tolist())
    print('delta (eef-bowl)', np.round(eef-b,4).tolist())
    c.close()
