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
    imgs = c.camera_images()
    print('shape', tuple(imgs.shape), 'dtype', imgs.dtype)
    a = imgs.detach().cpu().numpy()
    np.save('.nuka-runs/policy_imgs.npy', a)
    # Row-brightness profile: the table/scene should occupy lower rows in an
    # upright agentview frame, so compare top vs bottom band means.
    for i, name in enumerate(('agentview', 'wrist')):
        f = a[i].astype(np.float32)
        if f.ndim == 3 and f.shape[0] in (3, 4):
            f = f.transpose(1, 2, 0)
        g = f.mean(axis=2) if f.ndim == 3 else f
        h = g.shape[0]
        print('%s  h=%d w=%d  top-third %.1f  mid %.1f  bottom-third %.1f' % (
            name, g.shape[0], g.shape[1],
            g[:h//3].mean(), g[h//3:2*h//3].mean(), g[2*h//3:].mean()))
    c.close()
