# Migrating from Isaac Lab

Nuka targets the **RL-training subset** of an Isaac Lab / Isaac Gym workflow: a
batched, GPU-resident vectorized environment that exposes observations and
accepts actions through zero-copy tensors. This page maps the concepts and is
honest about where parity is partial.

This is a **subset mapping, not a drop-in replacement.** Nuka does not reimplement
the Isaac Lab Python API surface; it provides the equivalent capabilities for the
parts that matter for on-GPU RL and adds differentiability and strong determinism
on top.

## Concept mapping

| Isaac Lab / Isaac Gym | Nuka equivalent |
|------------------------|-----------------|
| Simulation context / device | `nuka.Device.create(0)` |
| Vectorized env (N parallel) | `nuka.World.create_from_scene(dev, scene, env_count=N)` — one cooked template, all envs on one GPU |
| Scene / USD stage | `.usda` (text USD), MJCF (`.xml`), or URDF imported by the cooker (no OpenUSD SDK; no binary USD) |
| Per-env reset / autoreset | `world.reset()` (all envs) / `world.reset_envs(env_ids)` (masked) |
| Step | `world.step()` / `world.step_n(n)` |
| Observation / state tensors | `world.buffer_view(field)` → zero-copy DLPack → `torch.from_dlpack(...)` |
| Action application (PD targets) | write into `DRIVE_TARGET` slots `[1:]` (zero-copy in place) |
| rl_games / gym vec-env adapter | shipped — see `examples/training/train_go2_ppo.py` |

## Zero-copy tensor interop

Engine buffers are exposed as DLPack-capable CUDA arrays that alias the device
memory with **no copy**. Both PyTorch and JAX consume them:

```python
import torch
q   = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))   # zero-copy view
tgt = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))
tgt[:, 1:].copy_(policy_actions)                                  # write actions in place
world.step()
```

Key layout note that trips up newcomers: the `DRIVE_TARGET` / `JOINT_POSITION`
buffers are `base_link_count` wide. **Slot 0 is the (inert) root link**; the
actuated joints a policy controls occupy slots `[1:]`. So `action_dim ==
base_link_count - 1` (12 for Go2), and you write a policy's
`(env_count, action_dim)` tensor into `tgt[:, 1:]`.

To share ordering between torch ops and physics without explicit syncs, pin the
engine to torch's current CUDA stream:

```python
dev = nuka.Device.create(0, stream_ptr=nuka.torch_stream_ptr())
```

## What is compatible

- **Large batched env counts on a single GPU** (4096+ for Go2), one cooked
  template shared across all envs.
- **Zero-copy observation/action tensors** for PyTorch *and* JAX.
- **A working rl_games PPO pipeline** (the Go2 locomotion task) and a gymnasium
  vec-env + engine-side per-env reset primitive.
- **MJCF and URDF import** (the most common robot description formats), plus text
  USD.

## What differs (be aware)

- **Strong determinism by default.** Isaac Lab / PhysX runs are not bit-exact
  across runs; Nuka's D1 default is. (A D2 weak-determinism toggle is reserved at
  the C ABI.) If you depend on the *exact* PhysX numerics, expect different —
  though deterministic — results.
- **Differentiability.** Nuka exposes a full analytical adjoint through rigid +
  articulated dynamics (PhysX / Isaac Lab are not differentiable). This is a
  *capability Nuka adds*, not a parity item — see [diff-sim](diff-sim.md). The
  differentiable rollout path is single-env and contact-free in v0.5; the
  floating-base orientation channel and d/dM, d/dJ contact channels are v0.7.
- **No OpenUSD SDK.** Scene import is text USD (`.usda`), MJCF, and URDF only.
  Binary `.usdc` / `.usdz` and the full OpenUSD stage API are not supported.
- **Rendering.** Nuka ships an *optional* C++ offscreen Vulkan renderer; it is
  not in the Python runtime. There is no Omniverse/RTX viewport. Sensors (IMU,
  joint, lidar) run on GPU; RGB/depth/tactile/F-T sensors are v0.7.
- **Physics breadth.** v0.5 is rigid + articulated only. Soft body, fluid, and
  cross-system coupling are v0.7 (see the
  [master plan](../plans/2026-05-28-nuka-physics-master-plan.md) §7).
- **No Python-only deployment.** Nuka is a C++ engine with a stable C ABI; it is
  designed to embed in a C++ host as well as drive from Python.

## Migrating an RL task — checklist

1. Convert the robot to MJCF / URDF / `.usda` (no binary USD).
2. Replace the sim-context + vec-env construction with
   `nuka.Device.create` + `nuka.World.create_from_scene(..., env_count=N)`.
3. Replace observation reads with `torch.from_dlpack(world.buffer_view(field))`.
4. Replace action application with an in-place `copy_` into `DRIVE_TARGET[:, 1:]`.
5. Wire reset to `world.reset_envs(env_ids)` for the masked-autoreset path.
6. Use the rl_games adapter in `examples/training/` as the reference integration.

## Compared to Brax / MJX

Brax and MJX are differentiable, GPU-based RL simulators — closer to Nuka's
diff-sim than Isaac Lab is. The distinction: Nuka uses a **full analytical
reverse-mode adjoint** through Featherstone dynamics, whereas Brax's contact
gradient is a **stop-gradient** approximation. Nuka also guarantees **strong
(D1) bit-exact determinism**, which the JAX-XLA-based simulators do not promise
across runs/hardware.
