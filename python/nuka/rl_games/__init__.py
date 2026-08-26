"""Nuka <-> rl_games (1.6.5) integration.

Importing this package registers the 'NUKA' vecenv type and the Nuka env names
with rl_games, so an rl_games yaml ``config.env_name: nuka_go2`` (and the skill
tasks) resolves to the on-GPU env. See :mod:`nuka.rl_games.vecenv` for the exact
IVecEnv contract matched.

Also carries two documented compat shims for upstream 1.6.5 defects:
    1. ModelA2CContinuous.Network.forward crashes in its inference branch and
       trains against a key the model never ships ('value' vs 'values').
    2. Optimizer steps with non-finite gradients are refused instead of
       poisoning the policy weights permanently.
"""

from __future__ import annotations

import torch

from rl_games.algos_torch import model_builder

from .bdx_perception_network import BdxDepthFusionBuilder
from .vecenv import NukaVecEnv, register


def _patch_continuous_a2c_forward() -> None:
    """Upstream bug: the INFERENCE branch reads the local ``entropy`` that only
    the training branch assigns -- every rollout action raises UnboundLocalError;
    and calc_gradients reads ``res_dict['values']`` while the model ships only
    ``'value'``. Rebind a corrected forward computing entropy in both branches
    and providing both value keys (identical math otherwise)."""
    from rl_games.algos_torch import models as _models

    def forward(self, input_dict):
        is_train = input_dict.get("is_train", True)
        prev_actions = input_dict.get("prev_actions", None)
        input_dict["obs"] = self.norm_obs(input_dict["obs"])
        mu, sigma, value, states = self.a2c_network(input_dict)
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        entropy = distr.entropy().sum(dim=-1)
        if is_train:
            prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
            # RAW value: prepare_dataset normalizes before the loss.
            return {
                "prev_neglogp": torch.squeeze(prev_neglogp),
                "value": value,
                "values": value,
                "entropy": entropy,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }
        selected_action = distr.sample().squeeze()
        neglogp = -distr.log_prob(selected_action).sum(dim=-1)
        val = self.denorm_value(value)
        return {
            "neglogpacs": torch.squeeze(neglogp),
            "values": val,
            "value": val,
            "actions": selected_action,
            "entropy": entropy,
            "rnn_states": states,
            "mus": mu,
            "sigmas": sigma,
        }

    _models.ModelA2CContinuous.Network.forward = forward


def _patch_nan_grad_guard() -> None:
    """Second-layer safety net: refuse optimizer steps whose gradient batch
    contains non-finite values. A single poisoned minibatch would otherwise kill
    the policy weights permanently (sigma -> NaN; every later rollout then dies
    on ``normal expects std >= 0``). The env layer already guarantees finite
    obs/rewards into the buffer; this guards the update itself."""
    from rl_games.algos_torch import a2c_continuous as A

    if getattr(A.A2CAgent, "_nuka_guarded", False):
        return
    orig_init = A.A2CAgent.__init__

    def guarded_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        opt = self.optimizer
        orig_step = opt.step
        skip_diag_left = [5]

        def safe_step(closure=None):
            bad = []
            for name, p in self.model.named_parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    bad.append(name)
            if bad:
                if skip_diag_left[0] > 0:
                    skip_diag_left[0] -= 1
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1e12)
                    adv = self.experience_buffer.tensor_dict.get(
                        "advantages") if hasattr(self, "experience_buffer") else None
                    print(f"[nuka-guard] skipped batch: bad_params={bad[:4]} "
                          f"gnorm_pre={float(gnorm):.3e}", flush=True)
                opt.zero_grad(set_to_none=True)
                return None
            return orig_step(closure)

        opt.step = safe_step

        opt.step = safe_step

    guarded_init._nuka_guarded = True
    A.A2CAgent.__init__ = guarded_init
    A.A2CAgent._nuka_guarded = True


_patch_continuous_a2c_forward()
_patch_nan_grad_guard()

# rl_games resolves custom networks through this process-global registry.  Keep
# registration beside the env registration so one ``import nuka.rl_games`` wires
# the complete structured-observation pipeline.
model_builder.register_network("bdx_depth_fusion", BdxDepthFusionBuilder)

__all__ = ["NukaVecEnv", "BdxDepthFusionBuilder", "register"]
