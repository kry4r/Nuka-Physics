import sys, re
sys.path.insert(0, 'python')
import numpy as np, torch, nuka
import nuka.tasks.libero_black_bowl as lbb

xml = open('.nuka-assets/generated/libero/libero_spatial_black_bowl.xml').read()
body = xml.split('<body name="akita_black_bowl_1_main"')[1].split('</body>')[0]
P = []
for g in re.findall(r'<geom\b[^>]*>', body):
    if 'official_collision' not in g: continue
    mp = re.search(r'\bpos="([^"]+)"', g)
    if mp: P.append([float(x) for x in mp.group(1).split()])
P = np.array(P)

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
with nuka.Device.create(0) as device:
    c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100): c.step(advance_gripper=False)
    pose = c.rigid_pose[c.target_body_index].detach().cpu().numpy()
    bp, bq = pose[:3], pose[3:7]
    print('bowl pos ', np.round(bp,4).tolist())
    print('bowl quat', np.round(bq,4).tolist())
    for order, (w,x,y,z) in (('wxyz', bq), ('xyzw', np.roll(bq,-1))):
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
        W = (R @ P.T).T + bp
        rad = np.linalg.norm(W[:,:2]-bp[:2], axis=1)
        top = W[:,2] > W[:,2].max()-0.004
        print('%s: worldZ %.4f..%.4f  radius %.4f..%.4f  topring n=%d Z=%.4f rad=%.4f'
              % (order, W[:,2].min(), W[:,2].max(), rad.min(), rad.max(),
                 top.sum(), W[top,2].mean(), rad[top].mean()))
    c.close()
