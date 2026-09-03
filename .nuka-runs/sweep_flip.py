import sys
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
from pathlib import Path
import nuka.tasks.libero_black_bowl as lbb
from nuka.vla.pi05_libero import LiberoPi05Policy

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
CHECKPOINT = Path(".nuka_cache/pi05-libero")
TOKENIZER = Path(".nuka_cache/paligemma-tokenizer")
VARIANTS = {
    'none':  (None, None),
    'agent_lr': ((1,), None),
    'agent_ud': ((0,), None),
    'agent_180': ((0,1), None),
}
orig = lbb.LiberoBlackBowlController.camera_images

def make(av, wr):
    def f(self):
        imgs = orig(self)
        a = torch.flip(imgs[0], dims=(1,)) if False else imgs[0]
        raw = self._camera_tensor[0]
        a = raw[0] if av is None else torch.flip(raw[0], dims=av)
        b = raw[1] if wr is None else torch.flip(raw[1], dims=wr)
        return torch.stack((a, b))
    return f

pol = LiberoPi05Policy(CHECKPOINT, TOKENIZER)
for name, (av, wr) in VARIANTS.items():
    lbb.LiberoBlackBowlController.camera_images = make(av, wr)
    with nuka.Device.create(0) as device:
        c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
        c.reset()
        for _ in range(100):
            c.step(advance_gripper=False)
        bowl0 = c.target_position().detach().cpu().numpy().copy()
        best = 9.9; zmax = bowl0[2]; chunk=None; idx=0
        for tick in range(600):
            if chunk is None or idx >= 10:
                im = c.camera_images()
                chunk = pol.predict_chunk(c.state8(), im[0], im[1],
                          lbb.LIBERO_TASK, seed=tick, profile=False).actions
                idx = 0
            c.set_policy_action(chunk[idx]); idx += 1
            for _ in range(10):
                c.step()
            eef = c.eef_pose()[:3].detach().cpu().numpy()
            b = c.target_position().detach().cpu().numpy()
            best = min(best, float(np.linalg.norm(eef-b))); zmax = max(zmax, float(b[2]))
        print('%-11s closest %.4f m   bowl_rise %+.4f m' % (
            name, best, zmax-bowl0[2]), flush=True)
        c.close()
