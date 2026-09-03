import nuka
import torch
from pathlib import Path
p = __import__('sys').argv[1]
c = nuka.Device.create(0)
w = nuka.World.create_from_scene(c, p, env_count=1, dt=0.005, determinism=nuka.DETERMINISM_STRONG, control_mode=nuka.CONTROL_MODE_PD_POSITION, contact_family=1)
print('created', p, flush=True)
w.step()
print('stepped', flush=True)
