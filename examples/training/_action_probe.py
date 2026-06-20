import argparse, numpy as np, torch, nuka
from nuka.tasks.go2_handstand import make_env
from nuka.tasks import go2_obs as G
from go2_handstand_eval import load_actor
p=argparse.ArgumentParser(); p.add_argument("--ckpt",required=True); p.add_argument("--stage",default="S2")
p.add_argument("--theta-cmd",type=float,default=90.0); p.add_argument("--n",type=int,default=64); a=p.parse_args()
torch.manual_seed(0)
env=make_env(a.n,stage=a.stage,theta_cmd_obs=True,seed=0); dev=env._torch_device
fwd,ep,rew=load_actor(a.ckpt,env._obs_dim,G.GO2_ACTION_DIM,dev)
env.reset(seed=0); env._theta_cmd_deg[:]=a.theta_cmd; nuka.sync(); env._world.step_n(env.decimation); nuka.sync()
b=env._obs; last=torch.zeros(a.n,12,device=dev)
print(f"stage={a.stage} theta={a.theta_cmd} ckpt rew={rew}")
for k in range(8):
    obs=env._append_theta(b.compute_obs(env.command,last)); act=fwd(obs)
    print(f"step {k}: action |mu| med={float(act.abs().median()):.3f} max={float(act.abs().max()):.3f}  "
          f"qd|max|(per env) med={float(b.qd_urdf().abs().amax(1).median()):.1f}  "
          f"pgx med={float(b.projected_gravity()[:,0].median()):+.3f} bz={float(b.base_pos()[:,2].median()):.3f}")
    last=b.write_action(act); nuka.sync(); env._world.step_n(env.decimation); nuka.sync()
env.close()
