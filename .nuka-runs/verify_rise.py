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
    b0 = c.target_position().detach().cpu().numpy().copy()
    print('after settle bowl z=%.4f' % b0[2])
    chunk=None; idx=0
    for tick in range(600):
        if chunk is None or idx >= 10:
            im = c.camera_images()
            chunk = pol.predict_chunk(c.state8(), im[0], im[1], lbb.LIBERO_TASK,
                                      seed=tick, profile=False).actions
            idx = 0
        c.set_policy_action(chunk[idx]); idx += 1
        for _ in range(10):
            c.step()
        if tick % 75 == 0 or tick == 599:
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            b = c.target_position().detach().cpu().numpy()
            print('t%3d bowl z %.4f  dz %+.4f  d_eef %.4f  grip_mm %.1f' % (
                tick, b[2], b[2]-b0[2], float(np.linalg.norm(eef-b)),
                c.q[0,8].item()*1000), flush=True)
    c.close()
