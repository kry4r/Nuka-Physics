"""Debug v2 (256 envs): report WHICH parameters go non-finite first and the
loss values around the event."""
import sys

import torch

sys.path.insert(0, "python")

import nuka.rl_games  # noqa: F401
import yaml

from rl_games.algos_torch import a2c_continuous as A

orig_calc = A.A2CAgent.calc_gradients


def dbg_calc(self, input_dict):
    orig_calc(self, input_dict)
    bad = [n for n, p in self.model.named_parameters()
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    if bad:
        adv = input_dict["advantages"]
        print(f"NONFINITE GRAD PARAMS: {bad}", flush=True)
        print(f"  adv finite={float(torch.isfinite(adv).float().mean()):.4f} "
              f"max|adv|={float(adv.abs().max()):.3e} "
              f"ret max|.|={float(input_dict['returns'].abs().max()):.3e} "
              f"old_val max|.|={float(input_dict['old_values'].abs().max()):.3e}",
              flush=True)
        # Forward stats under current (still-clean?) weights.
        with torch.no_grad():
            bd = {"is_train": True, "prev_actions": input_dict["actions"],
                  "obs": self._preproc_obs(input_dict["obs"])}
            r = self.model(bd)
            print(f"  forward: mu_max|.|="
                  f"{float(r['mus'].abs().max()):.3e} "
                  f"sigma_range=[{float(r['sigmas'].min()):.3e},"
                  f"{float(r['sigmas'].max()):.3e}] "
                  f"val_finite={bool(torch.isfinite(r['values']).all())}",
                  flush=True)
        raise SystemExit(0)
    return None


A.A2CAgent.calc_gradients = dbg_calc

with open("configs/rl_games/go2_backflip.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

from rl_games.torch_runner import Runner  # noqa: E402

runner = Runner()
runner.load(cfg)
runner.run({"train": True, "play": False})
print("no nonfinite grads in window")
