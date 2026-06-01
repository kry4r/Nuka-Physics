# Differentiable simulation

Nuka exposes a **full analytical reverse-mode adjoint** through its rigid +
articulated (Featherstone) dynamics. You can backpropagate a scalar loss through
a simulation rollout and get exact, engine-consistent gradients with respect to
the actions you applied and the physical parameters of the model.

This is fundamentally different from a stop-gradient or sample-based estimator:
the gradient is the analytical reverse of the same deterministic kernels the
forward pass ran.

## What is differentiable in v0.5

**Shipped:**

- **Reverse-mode adjoint through rigid + articulated dynamics** — Featherstone
  ABA differentiated in reverse mode (a 3-pass reverse over the articulation
  tree).
- **A recorded tape + gradient checkpointing** — the backward replays the
  forward from checkpoints to reconstruct intermediates on demand, keeping memory
  bounded. See [diff-sim tape + checkpointing](../architecture/diffsim-tape.md).
- **IFT-at-convergence contact gradient** — at the converged contact solve, the
  gradient is taken through the implicit function theorem rather than unrolling
  the iterative solver. The linear system is solved with a self-written
  deterministic CG + Jacobi / Block-Jacobi solver (fixed-order reductions, D1
  bit-exact). This path uses **constant memory** (no per-iteration tape) and is
  D1-deterministic.
- **Parameter gradients** — e.g. gradient w.r.t. a link mass (the system-ID
  channel).
- **Control-mode gradients** — for the v0.5 control modes.

**Observation (loss) channels you can seed:** post-rollout joint position (`q`),
joint velocity (`qdot`), and the floating-base root spatial **velocity**
(`v_root`).

**Deferred to v0.7 (stated honestly):**

- The floating-base **orientation** channel. The step adjoint is exact on the
  joint channel and the floating-base *velocity* channel (its gyroscopic bias is
  linearized at the pre-integration root velocity); the base **orientation** /
  quaternion-pose dependence is held fixed and not differentiated.
- The **d/dM** and **d/dJ** contact-derivative channels (gradients of the contact
  solve with respect to the mass matrix and Jacobian entries).

## The two paths

### 1. Tape-based rollout adjoint (recommended)

The whole multi-step rollout is wrapped in one `torch.autograd.Function` whose
`backward` is a **single** `tape.backward` call — the engine's own deterministic
reverse pass over the recorded tape. This is the path the system-ID demo uses.

Key properties (all enforced by the design):

- **Single-env, contact-free PD rollout.** `world.env_count` must be `1`. The
  rollout mutates engine state in place.
- **Fresh world + tape per evaluation.** `world.reset()` is *not supported* on
  this path, and `tape.reset()` alone is not bit-identical to a fresh rollout, so
  each forward evaluation builds a new world + tape (cheap — a few ms).
- **Keep world + tape alive until after `backward()`.** The adjoint reads the
  live tape; destroying it first raises `Tape.backward: empty tape`.

### 2. IFT contact gradient (constant memory)

For a converged contact solve, the contact gradient is taken via the implicit
function theorem at the fixed point, solving the KKT/Schur linear system with the
self-written deterministic CG solver. The honest performance picture: measured
IFT-vs-tape on the rigid + Featherstone scope is **0.65–0.69×** (i.e. *slower*
than the tape) — it is launch-bound at small active-row counts. The value of the
IFT path is **not** raw speed; it is **constant memory** (no tape that grows with
step count), **D1 determinism**, and **correctness at the fixed point**. The
broader perf target is framed as a qualified close, not a speedup claim.

## The PyTorch frontend

`nuka.autograd.differentiable_rollout` drives a `(K, action_dim)` action sequence
into the engine, recording each step onto a tape, and returns the observation
channel selected by `obs`. The parameter you want a gradient for is passed as a
**torch tensor** via `params=` — the engine's `grad_parameters` populates
`params.grad`. Calling `.item()` on the parameter would convert it to a Python
float and **sever the autograd graph**, leaving `params.grad` as `None`.

```python
import torch
import nuka
from nuka.autograd import differentiable_rollout

with nuka.Device.create(0) as dev:
    world = nuka.World.create_from_scene(dev, "examples/scenes/go2_system_id.usda", 1)
    world.set_gravity_z(-9.81)
    tape = nuka.Tape.create(
        world, checkpoint_interval=3, max_tape_entries=4096,
        max_checkpoints=512, recompute_on_backward=1,
    )

    mass = torch.nn.Parameter(torch.tensor([0.9], device="cuda"))   # tensor, NOT .item()
    action_dim = world.base_link_count - 1                          # 12 for Go2
    actions = torch.zeros(30, action_dim, device="cuda")            # (K, action_dim)

    obs = differentiable_rollout(
        world, tape, actions,
        params=mass, param_link_indices=[2], obs="qdot",            # link index 2 = a thigh
    )
    loss = obs.pow(2).mean()
    loss.backward()

    print("dLoss/dmass =", mass.grad.item())   # gradient is now populated
    tape.destroy()                              # only AFTER reading the gradient
    world.destroy()
```

A `differentiable_step` (K=1) wrapper exists for single-step use.

## The JAX frontend

`nuka.jax_frontend.differentiable_rollout` is a `jax.custom_vjp` over the same
nanobind binding. It mirrors the torch path line-for-line on every
engine-touching call, **reusing the same seed builder and the same
`tape.backward` reverse pass**. Because both frontends drive the *same*
deterministic engine adjoint, `jax.grad` and torch's `loss.backward()` produce
gradients that **agree to engine round-off** — the regression gate is a tight
rel-err `< 1e-4`, and the typical observed agreement is ~1e-6.

JAX usage notes:

- It runs under **eager** `jax.grad` / `jax.vjp` (concrete arrays). It is **not**
  `jax.jit`-able: the engine step is a foreign side effect that would need
  `io_callback` under tracing — deliberately out of scope. For pure-jax
  `jit`/`vmap`/`grad` over state arrays, use `nuka.jax_state_pytree.NukaWorldState`
  (which carries no engine step).
- Same single-env / fresh-world-and-tape rule as the torch path.

## Determinism and gradients

The strong (D1) determinism guarantee holds end to end. The mass-parameter
gradient is read back from the engine's `grad_parameters` via a host→device copy
that is itself bit-deterministic, so two runs of the *same* optimization produce
a **bit-identical** parameter trajectory (verified in the system-ID demo's D1
check). Note the distinction:

- **Bit-identical** applies to **two runs of the same path** (the D1 contract).
- **Cross-framework** (PyTorch vs JAX) gradients **agree to engine round-off**
  (~1e-6), not bit-for-bit — they take different host arithmetic routes into the
  same engine adjoint.

## Where to go next

- The end-to-end demo: [system identification](../examples/system_identification.md).
- The tape internals: [diff-sim tape + checkpointing](../architecture/diffsim-tape.md).
- Migrating diff-sim from Brax/MJX/Isaac Lab: [Isaac Lab compatibility](isaaclab-compat.md).
