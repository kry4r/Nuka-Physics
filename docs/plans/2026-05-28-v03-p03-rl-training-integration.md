# Nuka Physics v0.3 – Phase 3: RL Training Integration + Isaac Lab Adapter Stub

> **Master plan reference:** §3 Round 8 (Isaac Lab drop-in scope) + §3 Round 13 (S1 4096-env PPO)
> **Prerequisites:** v0.3 Phase 2 (Python bindings + autograd skeleton)
> **Blocks:** v0.3 Phase 4 (Go2 PPO training)
> **Exit criteria gate:** v0.3
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Plumb the engine into a real RL training loop. The deliverable enables the v0.3 exit demo: **4096 Go2 envs running PPO end-to-end**.

Two integration paths shipped together:

1. **Pure gym interface** (`nuka.gym.NukaGymEnv`) — works with rl_games, Stable-Baselines3, SKRL, any gym consumer. Minimal Python surface.
2. **Isaac Lab compat layer** (`nuka.isaaclab_compat`) — drop-in for the RL training subset of Isaac Lab's `ManagerBasedRLEnv`. Master plan §3 Round 8 scopes this strictly to the RL training path (no UI / editor / Omniverse).

Reward and observation computation in v0.3 lives in **Python** (PyTorch tensor ops on the zero-copy DLPack views). Master plan §3 Round 8 / 23: v1 reward in Python (Isaac Lab parity), v2 codegen DSL fused into CUDA. v0.3 is v1.

## Tech Stack

- Python 3.10+
- PyTorch 2.4+
- rl_games (or alternate: SKRL / SB3)
- gymnasium 0.29+
- nanobind extension from Phase 2

## Files to Create

### gym layer

- `python/nuka/gym/__init__.py`
- `python/nuka/gym/env.py` — `NukaGymEnv` (vectorized gym env)
- `python/nuka/gym/vec_env_protocol.py` — IsaacGym-compatible vectorized env protocol
- `tests/python/test_gym_env.py`

### Isaac Lab compat layer

- `python/nuka/isaaclab_compat/__init__.py`
- `python/nuka/isaaclab_compat/envs.py` — `ManagerBasedRLEnv`, `ManagerBasedRLEnvCfg`
- `python/nuka/isaaclab_compat/scene.py` — `InteractiveScene`
- `python/nuka/isaaclab_compat/assets.py` — `Articulation`, `ArticulationCfg`
- `python/nuka/isaaclab_compat/managers.py` — `ObservationManager`, `RewardManager`, `TerminationManager`, `ActionManager`, `CommandManager`
- `python/nuka/isaaclab_compat/sim.py` — `SimulationContext` (thin wrapper around `nuka.World`)
- `tests/python/test_isaaclab_compat_minimal.py`

### Example task

- `python/nuka/tasks/__init__.py`
- `python/nuka/tasks/go2_locomotion.py` — Go2 locomotion task (manager-based RL env config)
- `examples/training/train_go2_ppo.py` — end-to-end training script using rl_games

### Action / observation plumbing in C ABI

- `src/c_abi/action_buffer.cpp` — expose action buffer write surface
- `src/c_abi/observation_buffer.cpp` — observation buffer read surface
- `src/include/nuka/nuka.h` — add `nuka_world_get_action_buffer_view` and `nuka_world_get_observation_buffer_view`

## Tasks

### Task 3.3.1 — Action / observation buffer C ABI

The C ABI returns a writable view onto the action buffer and a readable view onto the observation buffer (both GPU-resident; DLPack-exported in Python).

```c
typedef struct {
    void*    device_ptr;
    size_t   element_count;        /* env_count × action_dim */
    uint32_t element_stride_bytes;
    uint32_t action_dim;
    uint32_t env_count;
    uint8_t  dtype;
} nuka_action_buffer_view_t;

nuka_result_t nuka_world_get_action_buffer_view(nuka_world_handle w, nuka_action_buffer_view_t* out);
nuka_result_t nuka_world_get_observation_buffer_view(nuka_world_handle w, nuka_observation_buffer_view_t* out);
```

Engine internally consumes the action buffer at the start of each step (translates actions into joint drive setpoints or torques) and writes observations at the end.

For Go2, observation layout (per master plan §3 Round 8 / 24):
- 12 joint positions
- 12 joint velocities
- Base linear velocity (3)
- Base angular velocity (3)
- Projected gravity (3)
- Commanded velocity (3, from `CommandManager`)
- Last action (12)
- = 48-dim observation

Action: 12 joint position targets (PD-controlled).

### Task 3.3.2 — `NukaGymEnv` (gym layer)

`python/nuka/gym/env.py`:

```python
import gymnasium as gym
import torch
from .. import _core, World, Device

class NukaGymEnv(gym.vector.VectorEnv):
    """Vectorized gym env backed by Nuka physics."""

    def __init__(self, scene: str, num_envs: int, device: Device | None = None):
        device = device or Device(0)
        self._world = World(device, scene, num_envs=num_envs)
        self._num_envs = num_envs

        action_view = _core.world_get_action_buffer_view(self._world._handle)
        obs_view = _core.world_get_observation_buffer_view(self._world._handle)
        self._action_dim = action_view.action_dim
        self._obs_dim = obs_view.obs_dim

        # gym spaces (vectorized)
        self.single_action_space = gym.spaces.Box(low=-1, high=1, shape=(self._action_dim,))
        self.single_observation_space = gym.spaces.Box(low=-100, high=100, shape=(self._obs_dim,))

    @property
    def actions(self) -> torch.Tensor:
        capsule = _core.world_get_action_buffer_dlpack(self._world._handle)
        return torch.utils.dlpack.from_dlpack(capsule).view(self._num_envs, self._action_dim)

    @property
    def observations(self) -> torch.Tensor:
        capsule = _core.world_get_observation_buffer_dlpack(self._world._handle)
        return torch.utils.dlpack.from_dlpack(capsule).view(self._num_envs, self._obs_dim)

    def reset(self, *, seed=None, options=None):
        _core.world_reset(self._world._handle, seed if seed is not None else 0)
        return self.observations, {}

    def step(self, actions: torch.Tensor):
        # Write actions into the engine's action buffer (zero-copy if already in place)
        self.actions.copy_(actions)
        self._world.step()
        obs = self.observations
        rewards = self._compute_rewards()
        terminated = self._compute_terminated()
        truncated = torch.zeros_like(terminated)
        return obs, rewards, terminated, truncated, {}

    def _compute_rewards(self) -> torch.Tensor:
        # Override in subclass or via RewardManager (isaaclab_compat)
        return torch.zeros(self._num_envs, device="cuda")

    def _compute_terminated(self) -> torch.Tensor:
        return torch.zeros(self._num_envs, dtype=torch.bool, device="cuda")
```

### Task 3.3.3 — Isaac Lab compat layer (RL training subset only)

The clone implements the minimal API surface to run a vectorized RL environment with manager-based reward, observation, termination, and action. **No UI, no editor, no Omniverse Kit dependency.**

`python/nuka/isaaclab_compat/managers.py`:

```python
import torch
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class RewardTerm:
    name: str
    func: Callable[["ManagerBasedRLEnv"], torch.Tensor]
    weight: float = 1.0

class RewardManager:
    def __init__(self, env: "ManagerBasedRLEnv", terms: list[RewardTerm]):
        self._env = env
        self._terms = terms

    def compute(self) -> torch.Tensor:
        total = torch.zeros(self._env.num_envs, device="cuda")
        for term in self._terms:
            r = term.func(self._env)
            total += term.weight * r
        return total

@dataclass
class ObservationTerm:
    name: str
    func: Callable[["ManagerBasedRLEnv"], torch.Tensor]

class ObservationManager:
    def __init__(self, env: "ManagerBasedRLEnv", terms: list[ObservationTerm]):
        self._env = env
        self._terms = terms

    def compute(self) -> torch.Tensor:
        return torch.cat([t.func(self._env) for t in self._terms], dim=-1)

# Similar minimal: TerminationManager, ActionManager, CommandManager
```

`python/nuka/isaaclab_compat/envs.py`:

```python
from dataclasses import dataclass
import torch
import nuka

@dataclass
class ManagerBasedRLEnvCfg:
    scene_path: str
    num_envs: int = 1
    sim_dt: float = 1/240
    decimation: int = 4               # control freq = sim freq / decimation
    episode_length_s: float = 20.0
    reward_terms: list = None
    obs_terms: list = None
    termination_terms: list = None
    action_term: object = None

class ManagerBasedRLEnv:
    """Isaac Lab API drop-in (RL training path)."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg):
        self.cfg = cfg
        self._device = nuka.Device(0)
        self._world = nuka.World(self._device, cfg.scene_path, num_envs=cfg.num_envs, dt=cfg.sim_dt)
        self.num_envs = cfg.num_envs

        self.reward_manager = RewardManager(self, cfg.reward_terms or [])
        self.observation_manager = ObservationManager(self, cfg.obs_terms or [])
        self.termination_manager = TerminationManager(self, cfg.termination_terms or [])
        self.action_manager = ActionManager(self, cfg.action_term)
        self.command_manager = CommandManager(self)

    def step(self, actions: torch.Tensor):
        self.action_manager.process_action(actions)
        for _ in range(self.cfg.decimation):
            self._world.step()
        obs = self.observation_manager.compute()
        rewards = self.reward_manager.compute()
        terminated = self.termination_manager.compute()
        truncated = self._compute_truncated()
        return obs, rewards, terminated, truncated, {}

    def reset(self, env_ids: torch.Tensor | None = None):
        # ... reset specific envs by id ...
        return self.observation_manager.compute(), {}
```

This is a deliberately minimal clone. Users familiar with Isaac Lab can write reward / observation functions that match by signature and migrate trivially.

### Task 3.3.4 — Go2 locomotion task

`python/nuka/tasks/go2_locomotion.py`:

```python
from nuka.isaaclab_compat import ManagerBasedRLEnvCfg, ObservationTerm, RewardTerm

def obs_joint_pos(env): return env._world.joint_positions
def obs_joint_vel(env): return env._world.joint_velocities
def obs_base_lin_vel(env): return env._world.base_linear_velocity
def obs_base_ang_vel(env): return env._world.base_angular_velocity
def obs_projected_gravity(env): return env._world.projected_gravity
def obs_commanded_velocity(env): return env.command_manager.command_velocity
def obs_last_action(env): return env.action_manager.last_action

def reward_track_lin_vel_xy(env):
    cmd = env.command_manager.command_velocity[:, :2]
    actual = env._world.base_linear_velocity[:, :2]
    err = (cmd - actual).pow(2).sum(dim=-1)
    return torch.exp(-err / 0.25)

def reward_alive(env):
    return torch.ones(env.num_envs, device="cuda")

def reward_joint_acc(env):
    return -env._world.joint_accelerations.pow(2).sum(dim=-1)

def terminated_fell(env):
    return env._world.base_height < 0.2

GO2_CFG = ManagerBasedRLEnvCfg(
    scene_path="examples/scenes/go2_locomotion.usda",
    num_envs=4096,
    sim_dt=1/240,
    decimation=4,
    obs_terms=[
        ObservationTerm("joint_pos", obs_joint_pos),
        ObservationTerm("joint_vel", obs_joint_vel),
        ObservationTerm("base_lin_vel", obs_base_lin_vel),
        ObservationTerm("base_ang_vel", obs_base_ang_vel),
        ObservationTerm("projected_gravity", obs_projected_gravity),
        ObservationTerm("commanded_velocity", obs_commanded_velocity),
        ObservationTerm("last_action", obs_last_action),
    ],
    reward_terms=[
        RewardTerm("track_lin_vel", reward_track_lin_vel_xy, weight=1.0),
        RewardTerm("alive", reward_alive, weight=0.1),
        RewardTerm("joint_acc_penalty", reward_joint_acc, weight=-2.5e-7),
    ],
    termination_terms=[
        # termination terms attach here
    ],
)
```

### Task 3.3.5 — Training script (rl_games)

`examples/training/train_go2_ppo.py`:

```python
import rl_games.torch_runner as runner
import yaml
from nuka.isaaclab_compat import ManagerBasedRLEnv
from nuka.tasks.go2_locomotion import GO2_CFG

def make_env(**kwargs):
    return ManagerBasedRLEnv(GO2_CFG)

with open("examples/training/go2_ppo_cfg.yaml") as f:
    cfg = yaml.safe_load(f)

runner.Runner(make_env=make_env).run(cfg)
```

YAML config (rl_games-compatible) specifies: PPO hyperparameters, network architecture (MLP 256x128x64), learning rate, batch size = 4096 × 24 (24 steps per env per rollout), epochs = 5 per rollout.

### Task 3.3.6 — Tests

`tests/python/test_gym_env.py`:

```python
def test_gym_env_step_returns_correct_shape():
    env = NukaGymEnv("examples/scenes/go2_locomotion.usda", num_envs=64)
    obs, _ = env.reset()
    assert obs.shape == (64, env._obs_dim)

    actions = torch.zeros(64, env._action_dim, device="cuda")
    obs, rewards, terminated, truncated, _ = env.step(actions)
    assert obs.shape == (64, env._obs_dim)
    assert rewards.shape == (64,)
```

`tests/python/test_isaaclab_compat_minimal.py`:

```python
def test_minimal_isaaclab_env_runs():
    env = ManagerBasedRLEnv(GO2_CFG)
    obs, _ = env.reset()
    for _ in range(10):
        actions = torch.zeros(env.num_envs, env._world.action_dim, device="cuda")
        obs, rewards, terminated, _, _ = env.step(actions)
    assert obs.shape[0] == env.num_envs
    assert rewards.shape == (env.num_envs,)
```

## Validation

- `NukaGymEnv` passes a smoke RL loop with random policy (no NaN, no crash).
- `ManagerBasedRLEnv` reward/observation managers compose correctly; output tensors are on `cuda`.
- rl_games can consume the env and start a training run (training quality validated in Phase 4).
- Action / observation buffer C ABI is zero-copy (verified via data_ptr comparison).
- Step time at 4096 envs with full RL plumbing still < 1.2 ms (Phase 1 budget + 20% RL overhead headroom).

## Exit Criteria for v0.3 Phase 3

1. C ABI exposes action + observation buffer views.
2. `nuka.gym.NukaGymEnv` operational with rl_games / SB3 / SKRL.
3. `nuka.isaaclab_compat.ManagerBasedRLEnv` implements the RL training subset of Isaac Lab's API.
4. `nuka.tasks.go2_locomotion` defines a complete Go2 PPO task.
5. `examples/training/train_go2_ppo.py` launches and begins training (convergence is Phase 4).
6. Smoke tests pass.

## What This Phase Does Not Do

- No full PPO convergence run (Phase 4).
- No actual reward DSL codegen (v0.7).
- No Isaac Lab UI / editor / Omniverse anything (out of scope per master plan).
- No sim-to-real noise (v0.5).
- No reset randomization (basic only; domain randomization comes with N2 in v0.5).
