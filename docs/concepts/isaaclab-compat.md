# Isaac Lab compatibility

Nuka provides the RL-training subset of an Isaac Lab manager-based workflow: a
GPU-resident vectorized simulation, zero-copy tensors, composable MDP terms,
masked reset, and a Gymnasium-compatible step loop.

It is a migration layer, not a replacement for Omniverse Kit or the full Isaac
Lab API.

## Concept mapping

| Isaac Lab concept | Nuka equivalent |
|---|---|
| Simulation context | `nuka.isaaclab_compat.SimulationContext` |
| Manager-based environment | `ManagerBasedRLEnv` + `ManagerBasedRLEnvCfg` |
| Parallel scene instances | One `nuka.World` with `num_envs=N` |
| Observation terms | `ObservationTerm` / `ObservationManager` |
| Reward terms | `RewardTerm` / `RewardManager` |
| Terminations and timeouts | `TerminationTerm` / `TerminationManager` |
| Action processing | `ActionTerm` / `ActionManager` |
| Command source | `CommandManager` |
| Per-environment reset | `world.reset_envs(env_ids)` |
| Tensor state access | `world.buffer_view(field)` via DLPack |

## Simulation context

`SimulationContext` owns a `nuka.Device` and `nuka.World` unless an existing
device is supplied:

```python
from nuka.isaaclab_compat import SimulationContext

sim = SimulationContext(
    scene_path="examples/scenes/go2_locomotion.usda",
    num_envs=4096,
    dt=0.005,
)
try:
    sim.step(4)
    sim.sync()
    sim.reset_envs([0, 7, 42])
finally:
    sim.close()
```

The compatibility package depends on PyTorch, but plain `import nuka` does not.

## Zero-copy tensors

State and drive buffers stay in CUDA memory:

```python
import torch
import nuka

q = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))
target = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))
target[:, 1:].copy_(policy_actions)
world.step()
```

The tensors alias live engine memory. Slot `0` is the root; policy-controlled
joints occupy slots `[1:]`. To keep PyTorch and physics on the same stream,
create the device with `stream_ptr=nuka.torch_stream_ptr()`.

## Manager-based environments

`ManagerBasedRLEnvCfg` collects simulation timing and lists of term objects:

```python
from nuka.isaaclab_compat import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

cfg = ManagerBasedRLEnvCfg(
    scene_path="examples/scenes/go2_locomotion.usda",
    num_envs=4096,
    sim_dt=0.005,
    decimation=4,
    episode_length_s=20.0,
    obs_terms=observation_terms,
    reward_terms=reward_terms,
    termination_terms=termination_terms,
    action_term=action_term,
)

with ManagerBasedRLEnv(cfg) as env:
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(actions)
```

Term callbacks receive the environment and operate on CUDA tensors. The step
order is action processing, `decimation` physics steps, observation/reward/
termination evaluation, then masked autoreset.

## Supported migration surface

- Single-GPU batched environments and per-environment reset
- Gymnasium five-value `step()` results
- Manager-style observation, reward, termination, action, and command terms
- Zero-copy PyTorch state and action buffers
- rl_games integration through `examples/training/train_go2_ppo.py`
- MJCF, URDF, text USD, and cooked NKS scene sources

## Important differences

- Nuka does not include Omniverse Kit, an RTX viewport, extensions, or the
  complete Isaac Lab configuration hierarchy.
- Physics results are Nuka's deterministic CUDA dynamics, not PhysX numerics.
- Text USD is supported; binary `.usdc` and `.usdz` are not.
- The manager compatibility layer is focused on RL. Scene authoring and beauty
  rendering use `nuka.author` and `World.render_beauty` directly.
- `ManagerBasedRLEnv` is intentionally small. Project-specific observations,
  rewards, resets, curricula, and randomization remain task code.

## Migration checklist

1. Export the robot or scene as MJCF, URDF, text USD, or NKS.
2. Replace the simulation context with `SimulationContext` or configure a
   `ManagerBasedRLEnv`.
3. Port observation, reward, termination, and action functions into term
   callbacks that operate on CUDA tensors.
4. Replace PhysX tensor access with DLPack views from `world.buffer_view(...)`.
5. Write actions into `DRIVE_TARGET[:, 1:]` or provide an `ActionTerm`.
6. Use `world.reset_envs(env_ids)` for masked reset behavior.
7. Validate observation ordering, joint ordering, control rate, and reset
   semantics before reusing an existing policy.
