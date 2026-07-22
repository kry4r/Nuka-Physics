"""rl_games 1.6.5 ``IVecEnv`` adapter for the Nuka Go2 locomotion env.

rl_games does NOT consume a raw gymnasium ``VectorEnv``: it drives an
:class:`rl_games.common.ivecenv.IVecEnv` whose construction is dispatched by a
``vecenv_type`` registered in ``rl_games.common.vecenv`` and whose name is
resolved through ``rl_games.common.env_configurations``. We model this exactly
like the bundled ``BraxEnv`` (the other GPU-tensor, already-vectorized backend):

THE CONTRACT WE MATCH (verified against the installed 1.6.5 source):

  * ``get_env_info()`` -> dict with ``observation_space`` / ``action_space`` set
    to the **single-env** Box spaces (Box(48,) / Box(12,)). rl_games prepends
    ``num_actors`` itself, so handing it the batched space would mis-size the
    network input. Also ``agents=1``, ``value_size=1``.
  * ``get_number_of_agents()`` -> 1.
  * ``reset()`` -> obs ONLY (a CUDA float32 tensor ``(N,48)``). rl_games'
    ``cast_obs`` sees a ``torch.Tensor`` and flips ``is_tensor_obses=True``, then
    wraps it to ``{'obs': tensor}`` -- so we return the bare tensor (NukaGymEnv's
    ``reset`` 2-tuple is unpacked; ``info`` dropped).
  * ``step(actions)`` -> 4-tuple ``(obs, rewards, dones, infos)``:
        - ``actions``: rl_games hands us a CUDA tensor (is_tensor_obses) -> pass
          straight to ``NukaGymEnv.step`` (kept on GPU, no host copy).
        - ``rewards``: ``(N,)`` cuda float32 (rl_games does ``.unsqueeze(1)``).
        - ``dones``:   ``(N,)`` cuda float32 = ``terminated | truncated`` (a
          SINGLE merged done; matched to brax's float dones to satisfy the
          experience-buffer dtype).
        - ``infos['time_outs']``: the ``truncated`` mask ``(N,)`` bool -- read by
          a2c_common ONLY when ``value_bootstrap: True`` for terminal-value
          bootstrapping (legged_gym convention); harmless otherwise.

Registration (executes at import of this module):
    vecenv.register('NUKA', <factory>)
    env_configurations.register('nuka_go2', {'vecenv_type': 'NUKA', 'env_creator': <ctor>})
so a yaml ``env_name: nuka_go2`` resolves: ``create_vec_env`` looks up the
``vecenv_type`` ('NUKA') and calls our factory ``(config_name, num_actors,
**env_config)`` DIRECTLY (the ``env_creator`` is vestigial for non-RAY types --
only RayVecEnv calls it -- so all env construction happens in ``__init__`` here,
exactly like BraxEnv).
"""

from __future__ import annotations

import torch
from rl_games.common.ivecenv import IVecEnv
from rl_games.common import vecenv as _vecenv
from rl_games.common import env_configurations as _env_configurations

from ..tasks.go2_locomotion import make_env as make_go2_env
from ..tasks.h1_stand import make_env as make_h1_stand_env
from ..tasks.go2_handstand import make_env as make_handstand_env
from ..tasks.bdx_locomotion import make_env as make_bdx_env
from ..tasks.bdx_perception import make_env as make_bdx_perception_env
from ..tasks.bdx_corridor import make_env as make_bdx_corridor_teacher_env
from ..tasks.go2_corridor import make_env as make_go2_corridor_teacher_env


_ENV_FACTORIES = {
    "nuka_go2": make_go2_env,
    "nuka_h1_stand": make_h1_stand_env,
    "nuka_go2_handstand": make_handstand_env,
    "nuka_bdx": make_bdx_env,
    "nuka_bdx_perception": make_bdx_perception_env,
    "nuka_bdx_corridor_teacher": make_bdx_corridor_teacher_env,
    "nuka_go2_corridor_teacher": make_go2_corridor_teacher_env,
}


class NukaVecEnv(IVecEnv):
    """rl_games IVecEnv wrapping the on-GPU :class:`Go2LocomotionEnv`.

    Constructed by ``rl_games.common.vecenv.create_vec_env`` as
    ``NukaVecEnv(config_name, num_actors, **env_config)``. ``env_config`` is the
    yaml ``config.env_config`` block (e.g. ``device``, ``command``, ``dt``,
    ``decimation``, ``episode_length_s``, ``seed``). ``num_actors`` is the env
    count.
    """

    def __init__(self, config_name: str, num_actors: int, **kwargs) -> None:
        self.config_name = config_name
        self.num_actors = int(num_actors)
        # rl_games passes a cuda action tensor; keep everything on GPU.
        self._device = torch.device("cuda")
        # ``device`` here would be a nuka.Device -- not provided via yaml, so the
        # env creates+owns its own (ordinal 0). All other kwargs forwarded.
        kwargs.pop("num_actors", None)  # defensive: never double-pass.
        try:
            make_env = _ENV_FACTORIES[config_name]
        except KeyError as exc:
            raise ValueError(
                f"unknown Nuka rl_games env config {config_name!r}; "
                f"available={sorted(_ENV_FACTORIES)}"
            ) from exc
        self.env = make_env(self.num_actors, **kwargs)

        # Single-env spaces (rl_games prepends num_actors).
        self.observation_space = self.env.single_observation_space
        self.action_space = self.env.single_action_space

    # -- IVecEnv API --------------------------------------------------------
    def get_number_of_agents(self) -> int:
        return 1

    def get_env_info(self) -> dict:
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": 1,
            "value_size": 1,
        }

    def reset(self):
        obs, _info = self.env.reset()
        return obs

    def step(self, actions):
        # actions: cuda tensor (N,12) handed over by rl_games.
        obs, reward, terminated, truncated, _info = self.env.step(actions)
        # Single merged done; float to match the bundled GPU-tensor backend
        # (brax returns float dones -> avoids any experience-buffer dtype edge).
        dones = (terminated | truncated).to(dtype=torch.float32)
        reward = reward.to(dtype=torch.float32)
        # legged_gym-style time-limit bootstrap signal (read iff value_bootstrap).
        infos = {"time_outs": truncated}
        return obs, reward, dones, infos

    def seed(self, seed):
        # NukaGymEnv reseeds its RNG on reset(seed=...); rl_games seeds globals
        # itself, so this is a no-op hook for compatibility.
        return seed

    def close(self):
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None


def _nuka_vecenv_factory(config_name, num_actors, **kwargs):
    return NukaVecEnv(config_name, num_actors, **kwargs)


def register() -> None:
    """Register Nuka vecenv type + env names with rl_games.

    Idempotent: safe to call multiple times (e.g. once per process import).
    """
    _vecenv.register("NUKA", _nuka_vecenv_factory)
    for env_name, make_env in _ENV_FACTORIES.items():
        _env_configurations.register(
            env_name,
            {
                "vecenv_type": "NUKA",
                # Vestigial for non-RAY vecenv types (only RayVecEnv calls
                # env_creator); provided so env_configurations is internally
                # consistent + a direct ``env_creator(**cfg)`` smoke works.
                "env_creator": lambda _make_env=make_env, **kw: _make_env(
                    kw.pop("num_actors", 256), **kw
                ),
            },
        )


# Register at import so ``import nuka.rl_games`` is enough to make
# ``env_name: nuka_go2`` resolvable.
register()
