import sys
sys.path.insert(0, 'python')
import nuka, numpy as np
SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
with nuka.Device.create(0) as device:
    w = nuka.World.create_from_scene(device, SCENE, env_count=1, dt=0.005,
        determinism=nuka.DETERMINISM_STRONG,
        control_mode=nuka.CONTROL_MODE_OSC, osc_task_link=7,
        contact_family=1, heightfield_terrain_type=0)
    for a in sorted(dir(w)):
        if any(k in a.lower() for k in ('shape','geom','collid','body_count','num_')):
            print('ATTR', a)
    w.close()
