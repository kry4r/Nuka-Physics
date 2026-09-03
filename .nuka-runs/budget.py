"""Close the gripper on the rim at several pair budgets and family settings.

A budget-starved broadphase must show a residual aperture once the cap is high
enough to emit the finger<->bowl candidate pairs.
"""
import re, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'python'))
import nuka
SCENE = REPO / '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
RIM_Z, RIM_R = 0.9320, 0.0471

def build(family):
    src = (REPO/'python/nuka/tasks/libero_black_bowl.py').read_text()
    src = src.replace('env_count=1,', 'env_count=2,')
    src = src.replace('contact_family=1,', 'contact_family=%d,' % family)
    src = src.replace('.view(1, 10, 7)', '.view(2, 10, 7)[:1][0].unsqueeze(0)')
    src = re.sub(r'\.view\(1, (10|7|3|4)\)', lambda m: '.view(2, %s)[:1]' % m.group(1), src)
    ns = {'__name__': 'lbb_b', '__file__': str(REPO/'python/nuka/tasks/x.py')}
    exec(compile(src, 'lbb_b', 'exec'), ns)
    return ns

out = []
for family in (1, 0):
    try:
        ns = build(family)
        with nuka.Device.create(0) as device:
            c = ns['LiberoBlackBowlController'](str(SCENE), device,
                    control_backend='osc', render_quality='preview')
            c.reset()
            for _ in range(150): c.step()
            bp = c.rigid_pose[c.target_body_index,:3].cpu().numpy().copy()
            at = [float(bp[0]), float(bp[1]+RIM_R), float(RIM_Z-0.018)]
            def hold(g,n):
                for _ in range(n):
                    c.task_target[0,0],c.task_target[0,1],c.task_target[0,2]=at
                    c.drive_target[0,8]=g; c.drive_target[0,9]=g
                    c.step(advance_gripper=False)
            hold(0.040, 300)
            hold(0.0, 400)
            res = float(c.q[0,8]) + float(c.q[0,9])
            bz = float(c.rigid_pose[c.target_body_index,2])
            out.append('family=%d  residual %.5f m  bowlz %.4f (start %.4f)' % (
                family, res, bz, bp[2]))
            c.close()
    except Exception as exc:
        out.append('family=%d FAILED %s: %s' % (family, type(exc).__name__, exc))
Path(REPO/'.nuka-runs/budget.log').write_text('\n'.join(out))
