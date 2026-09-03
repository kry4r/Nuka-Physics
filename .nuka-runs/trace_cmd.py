import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb
from nuka.vla.pi05_libero import LiberoPi05Policy
from pathlib import Path
CHECKPOINT = Path(".nuka_cache/pi05-libero")
TOKENIZER = Path(".nuka_cache/paligemma-tokenizer")

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
with nuka.Device.create(0) as device:
    c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100):
        c.step(advance_gripper=False)
    pol = LiberoPi05Policy(CHECKPOINT, TOKENIZER)
    bowl = c.target_position().detach().cpu().numpy().copy()
    print('bowl', np.round(bowl,4).tolist())
    print('%4s %-24s %-24s %7s %7s %6s' % (
        'tick','eef','cum_cmd_delta','d_bowl','grip','close'))
    cum = np.zeros(3, dtype=np.float64)
    chunk = None; idx = 0
    for tick in range(600):
        if chunk is None or idx >= 10:
            imgs = c.camera_images()
            r = pol.predict_chunk(c.state8(), imgs[0], imgs[1],
                                  lbb.LIBERO_TASK, seed=tick, profile=False)
            chunk = r.actions; idx = 0
        a = chunk[idx]; idx += 1
        an = a.detach().cpu().numpy()
        cum += an[:3] * 0.05
        c.set_policy_action(a)
        for _ in range(10):
            c.step()
        if tick % 60 == 0:
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            b = c.target_position().detach().cpu().numpy()
            print('%4d %-24s %-24s %7.4f %7.3f %6d' % (
                tick, np.round(eef,3).tolist(), np.round(cum,3).tolist(),
                float(np.linalg.norm(eef-b)), float(an[6]),
                int(c.q[0,8].item()*1000)))
    c.close()
