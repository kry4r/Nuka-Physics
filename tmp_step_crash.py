import torch
import nuka
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController
c = LiberoBlackBowlController('/mnt/c/Softwares/code/Nuka-Physics/.nuka-assets/generated/libero/libero_spatial_black_bowl.nks', nuka.Device.create(0), control_backend='joint_pd')
print('created', flush=True)
c.step(torch.tensor([0,0,0,0,0,0,-1.], device=c.q.device))
print('after', flush=True)
