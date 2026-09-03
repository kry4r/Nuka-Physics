import sys, re
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

xml = open('.nuka-assets/generated/libero/libero_spatial_black_bowl.xml').read()
body = xml.split('<body name="akita_black_bowl_1_main"')[1].split('</body>')[0]
P, S = [], []
for g in re.findall(r'<geom\b[^>]*>', body):
    if 'official_collision' not in g: continue
    mp = re.search(r'\bpos="([^"]+)"', g); ms = re.search(r'\bsize="([^"]+)"', g)
    if mp and ms:
        P.append([float(x) for x in mp.group(1).split()])
        S.append([float(x) for x in ms.group(1).split()])
P, S = np.array(P), np.array(S)

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
with nuka.Device.create(0) as device:
    c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100): c.step(advance_gripper=False)
    bq = c.target_orientation().detach().cpu().numpy() if hasattr(c,'target_orientation') else None
    bp = c.target_position().detach().cpu().numpy()
    print('bowl pos ', np.round(bp,4).tolist())
    print('bowl quat', None if bq is None else np.round(bq,4).tolist())
    if bq is None:
        print('attrs:', [a for a in dir(c) if 'target' in a or 'orient' in a])
    else:
        w,x,y,z = bq
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
        W = (R @ P.T).T + bp
        print('world Z of collision geoms: %.4f .. %.4f' % (W[:,2].min(), W[:,2].max()))
        top = W[W[:,2] > W[:,2].max()-0.004]
        print('top-ring count %d  world XY radius %.4f..%.4f  Z~%.4f' % (
            len(top), np.linalg.norm(top[:,:2]-bp[:2],axis=1).min(),
            np.linalg.norm(top[:,:2]-bp[:2],axis=1).max(), top[:,2].mean()))
        print('=> rim pinch: descend to Z %.4f, radial offset %.4f' % (
            top[:,2].mean()-0.004, np.linalg.norm(top[:,:2]-bp[:2],axis=1).mean()))
    c.close()
