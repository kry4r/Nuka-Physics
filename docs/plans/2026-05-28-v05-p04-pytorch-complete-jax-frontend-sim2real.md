# Nuka Physics v0.5 – Phase 4: PyTorch autograd Complete + JAX Frontend + Sim-to-Real N1+N2

> **Master plan reference:** §3 Round 8 (Python + JAX) + §3 Round 9 (autograd v1.0 complete) + §3 Round 10 (sim2real noise) + §7 v0.5 exit
> **Prerequisites:** v0.5 Phases 1–3 (adjoint codegen, tape, IFT)
> **Blocks:** v0.5 Phase 5 (exit demo needs diff-sim end-to-end + JAX + noise)
> **Exit criteria gate:** v0.5
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Complete the differentiable simulation user surface and add domain-randomization noise. Master plan §7 v0.5 exit criteria touched by this phase:

1. Diff-sim end-to-end through rigid + Featherstone.
2. `torch.autograd.Function` adjoint FD check passing for all base row classes.
3. JAX `custom_vjp` operational.
4. Sim-to-real N1 + N2 implemented.

cuDSS integration (already in Phase 3) is the IFT backend; the user-facing path is the PyTorch / JAX autograd hookup.

## Tech Stack

- PyTorch 2.4+
- JAX 0.4+ (latest stable)
- DLPack (already in v0.3)
- nanobind (already in v0.3)

## Files to Create

### PyTorch complete

- `python/nuka/autograd.py` (rewrite from v0.3 skeleton)
- `python/src/binding_autograd.cpp` — nanobind wrappers for tape + backward
- `python/src/binding_tape.cpp`
- `tests/python/test_autograd_grad_through_step.py`
- `tests/python/test_autograd_grad_check_vs_fd.py`

### JAX frontend

- `python/nuka/jax_frontend.py` — `step` as `jax.custom_vjp`
- `python/nuka/jax_state_pytree.py` — world state as JAX pytree
- `python/src/binding_jax_helpers.cpp`
- `tests/python/test_jax_custom_vjp.py`
- `tests/python/test_jax_grad_agreement_pytorch.py` — gradients via JAX match gradients via PyTorch on same scene

### Sim-to-real noise

- `src/sensor/noise/n1_gaussian.cu` — Gaussian noise on sensor readings
- `src/sensor/noise/n1_poisson.cu` — Poisson noise (depth / lidar)
- `src/sensor/noise/n2_domain_randomization.cpp` — per-episode parameter sampling
- `src/sensor/noise/noise_config.hpp` — configuration struct
- `src/include/nuka/nuka_noise.h` — C ABI extension
- `src/c_abi/noise.cpp`
- `python/nuka/noise.py`
- `tools/codegen/classes/*.yaml` — `noise_passthrough` flag for sensor-output row classes
- `tests/sensor/test_n1_gaussian.cu`
- `tests/sensor/test_n1_poisson.cu`
- `tests/sensor/test_n2_domain_randomization.cpp`

## Tasks

### Task 5.4.1 — Complete PyTorch autograd.Function

`python/nuka/autograd.py`:

```python
import torch
from . import _core

class _NukaPhysicsStepWithTape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, world_handle: int, tape_handle: int, actions: torch.Tensor) -> torch.Tensor:
        ctx.world_handle = world_handle
        ctx.tape_handle = tape_handle
        # Write actions into action buffer (assumed already in place if zero-copy)
        _core.world_write_actions(world_handle, actions.data_ptr())
        # Step with tape
        _core.world_step_with_tape(world_handle, tape_handle)
        # Read observations (zero-copy DLPack)
        obs_capsule = _core.world_buffer_dlpack(world_handle, _core.NUKA_FIELD_OBSERVATIONS)
        obs = torch.utils.dlpack.from_dlpack(obs_capsule)
        return obs

    @staticmethod
    def backward(ctx, grad_obs: torch.Tensor):
        # Allocate output grad buffer
        action_grad_capsule = _core.world_alloc_grad_buffer(ctx.world_handle, "actions")
        action_grad = torch.utils.dlpack.from_dlpack(action_grad_capsule)

        # Run tape backward
        _core.tape_backward(
            ctx.tape_handle,
            grad_obs.contiguous().data_ptr(),
            action_grad.data_ptr(),
            None  # grad_parameters; not used in this path
        )
        return None, None, action_grad

def differentiable_step(world, tape, actions: torch.Tensor) -> torch.Tensor:
    return _NukaPhysicsStepWithTape.apply(world._handle, tape._handle, actions)
```

### Task 5.4.2 — Gradient correctness tests

`tests/python/test_autograd_grad_check_vs_fd.py`:

```python
def test_grad_matches_fd():
    dev = nuka.Device(0)
    world = nuka.World(dev, "examples/scenes/go2_stand.usda", num_envs=1)
    tape = nuka.Tape(world, checkpoint_interval=10)

    actions = torch.zeros(1, world.action_dim, device="cuda", requires_grad=True)

    # Analytical gradient via autograd
    obs = differentiable_step(world, tape, actions)
    loss = obs.sum()
    loss.backward()
    grad_analytical = actions.grad.clone()

    # Numerical gradient via finite difference
    grad_numerical = torch.zeros_like(actions)
    eps = 1e-4
    for i in range(actions.shape[1]):
        world_pos = nuka.World(dev, "examples/scenes/go2_stand.usda", num_envs=1)
        actions_pos = actions.detach().clone(); actions_pos[0, i] += eps
        obs_pos = differentiable_step(world_pos, tape, actions_pos).sum().item()

        world_neg = nuka.World(dev, "examples/scenes/go2_stand.usda", num_envs=1)
        actions_neg = actions.detach().clone(); actions_neg[0, i] -= eps
        obs_neg = differentiable_step(world_neg, tape, actions_neg).sum().item()

        grad_numerical[0, i] = (obs_pos - obs_neg) / (2 * eps)

    rel_err = ((grad_analytical - grad_numerical) / (grad_numerical.abs() + 1e-6)).abs().max()
    assert rel_err < 1e-2, f"Gradient mismatch: {rel_err}"
```

### Task 5.4.3 — JAX custom_vjp

`python/nuka/jax_frontend.py`:

```python
import jax
import jax.numpy as jnp
from jax.dlpack import to_dlpack, from_dlpack as jax_from_dlpack
from . import _core

@jax.custom_vjp
def step(world_handle: int, tape_handle: int, world_state_pytree, actions):
    """JAX-style differentiable physics step.

    world_state_pytree: read-only pytree of current state (returned post-step)
    actions: jax.Array on GPU
    Returns: new world_state_pytree
    """
    # Forward: convert actions via DLPack, step, return new state
    _core.world_write_actions_dlpack(world_handle, to_dlpack(actions))
    _core.world_step_with_tape(world_handle, tape_handle)
    # Read new state via DLPack
    obs_capsule = _core.world_buffer_dlpack(world_handle, _core.NUKA_FIELD_OBSERVATIONS)
    obs = jax_from_dlpack(obs_capsule)
    # ... assemble new pytree ...
    return new_state

def step_fwd(world_handle, tape_handle, state, actions):
    out = step(world_handle, tape_handle, state, actions)
    return out, (world_handle, tape_handle, actions)

def step_bwd(res, grad_out):
    world_handle, tape_handle, actions = res
    # Allocate grad buffer
    grad_actions_capsule = _core.world_alloc_grad_buffer(world_handle, "actions")
    # Convert grad_out to a DLPack tensor; nuka reads it
    _core.tape_backward_jax(tape_handle, to_dlpack(grad_out), grad_actions_capsule, None)
    grad_actions = jax_from_dlpack(grad_actions_capsule)
    return (None, None, None, grad_actions)

step.defvjp(step_fwd, step_bwd)
```

### Task 5.4.4 — JAX state as pytree

`python/nuka/jax_state_pytree.py`:

```python
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp

@register_pytree_node_class
class NukaWorldState:
    def __init__(self, joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel):
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel
        self.body_pos = body_pos
        self.body_quat = body_quat
        self.body_lin_vel = body_lin_vel
        self.body_ang_vel = body_ang_vel

    def tree_flatten(self):
        return ((self.joint_pos, self.joint_vel, self.body_pos, self.body_quat,
                 self.body_lin_vel, self.body_ang_vel), None)
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
```

This allows JAX users to JIT, vmap, and grad through world state cleanly.

### Task 5.4.5 — PyTorch vs JAX gradient agreement test

`tests/python/test_jax_grad_agreement_pytorch.py`:

```python
def test_jax_pytorch_gradients_agree():
    # Same scene, same input
    # PyTorch path: differentiable_step + loss.backward()
    # JAX path: jax.grad(loss_fn)(actions)
    # Both gradients should agree to 1e-4 relative
```

### Task 5.4.6 — Sim-to-real N1 noise

`src/sensor/noise/n1_gaussian.cu`:

```cuda
__global__ void apply_gaussian_noise_kernel(
    float* __restrict__ sensor_data,
    uint32_t element_count,
    float mean,
    float stddev,
    uint64_t rng_state)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= element_count) return;
    // Deterministic per-element RNG (Philox / cuRAND counter-based)
    float n = generate_normal(rng_state, idx, mean, stddev);
    sensor_data[idx] += n;
}
```

Counter-based RNG (Philox) ensures D1 determinism: same seed + same idx = same noise. The RNG state lives in the checkpoint so reverse pass replays the same noise.

### Task 5.4.7 — Sim-to-real N2 domain randomization

`src/sensor/noise/n2_domain_randomization.cpp`:

```cpp
struct DomainRandomizationConfig {
    struct Range { float lo, hi; };
    Range mass_multiplier        = { 0.8f, 1.2f };
    Range friction_multiplier    = { 0.5f, 1.5f };
    Range restitution_offset     = { -0.1f, 0.1f };
    Range joint_armature_offset  = { 0.0f, 0.05f };
    Range gravity_z_offset       = { -0.5f, 0.5f };   // m/s²
    // Camera intrinsics, lighting (v0.7 RGB onward)
};

void ApplyPerEpisodeRandomization(WorldState& w, const DomainRandomizationConfig& cfg,
                                  uint64_t seed_per_env)
{
    // Sample per-env multipliers; update per-env mass / friction / etc.
    // Called on env reset.
}
```

The randomization is recorded in the checkpoint at reset; reverse pass uses recorded values (does not re-sample). This keeps backward deterministic.

### Task 5.4.8 — C ABI noise extension

`src/include/nuka/nuka_noise.h`:

```c
typedef enum {
    NUKA_NOISE_NONE = 0,
    NUKA_NOISE_GAUSSIAN = 1,
    NUKA_NOISE_POISSON = 2,
} nuka_noise_kind_t;

typedef struct {
    nuka_noise_kind_t kind;
    float param1;        /* mean for Gaussian / lambda for Poisson */
    float param2;        /* stddev for Gaussian */
} nuka_sensor_noise_desc_t;

nuka_result_t nuka_world_set_sensor_noise(nuka_world_handle w,
                                          nuka_state_field_t sensor_field,
                                          const nuka_sensor_noise_desc_t* desc);

typedef struct {
    /* mirrors DomainRandomizationConfig fields */
    float mass_mul_lo, mass_mul_hi;
    float friction_mul_lo, friction_mul_hi;
    /* ... etc ... */
} nuka_domain_randomization_desc_t;

nuka_result_t nuka_world_set_domain_randomization(nuka_world_handle w,
                                                  const nuka_domain_randomization_desc_t* desc);
```

### Task 5.4.9 — Python noise wrapper

```python
class GaussianNoise:
    def __init__(self, mean: float = 0.0, stddev: float = 0.01):
        self.mean, self.stddev = mean, stddev

class DomainRandomization:
    def __init__(self, mass_range=(0.8, 1.2), friction_range=(0.5, 1.5), ...):
        ...

# Usage
world.set_sensor_noise(nuka.SensorField.IMU, GaussianNoise(stddev=0.02))
world.set_sensor_noise(nuka.SensorField.JOINT_VELOCITY, GaussianNoise(stddev=0.05))
world.set_domain_randomization(DomainRandomization())
```

## Validation

- PyTorch FD gradient check passes for full step (not just per-row class).
- JAX vs PyTorch gradient agreement < 1e-4 relative.
- Gaussian noise determinism: same seed → bit-exact noise across runs (counter-based RNG).
- Domain randomization: per-episode samples saved + replayable in reverse.
- Sim-to-real noise does not break V1 oracle (oracle scenes have noise disabled).
- Sim-to-real noise does not break V2 invariants (noise is bounded; energy / momentum still tracked accurately).

## Exit Criteria for v0.5 Phase 4

1. PyTorch `autograd.Function` complete: full backward path with adjoint kernels + tape.
2. End-to-end gradient check vs FD passes for v0.1 row classes.
3. JAX `custom_vjp` frontend operational; same-scene gradient agreement with PyTorch.
4. World state JAX pytree registered; jit / vmap / grad work.
5. Sim-to-real noise N1 (Gaussian + Poisson on sensors) and N2 (per-episode domain randomization) operational.
6. RNG state captured in checkpoint; backward replays noise deterministically.
7. Tests pass; documentation written.

## What This Phase Does Not Do

- No N3 (lens distortion, rolling shutter, beam divergence) — v1.0 / v2.0.
- No reward DSL codegen (v0.7).
- No new row classes (v0.7).
- No diff rendering (v2.0).
