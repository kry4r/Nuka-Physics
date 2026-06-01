# Nuka Physics

**A GPU-resident, strongly deterministic, differentiable CUDA physics engine for
robotics and large-scale reinforcement learning.**

Nuka runs thousands of articulated environments on a single GPU, reproduces
runs bit-for-bit, and exposes a full **analytical reverse-mode adjoint** through
its rigid + articulated dynamics — so you can backpropagate a loss through the
simulator itself, not just sample-estimate it.

## Showcase — 4096-env Go2 locomotion, trained in-engine (v0.3)

![Go2 walking — batched environments driven by an in-engine PPO policy](docs/media/go2_walk_4096env.gif)

*16 of 4096 batched Go2 environments (debug-skeleton render of the headless
batched path), driven by a PPO policy trained end-to-end on Nuka.* The same
cooked world template runs all 4096 environments on one GPU; the policy was
trained from scratch with [rl_games](examples/training/) PPO and converges to a
command-conditioned walking gait (forward-velocity tracking across
`-0.5 … +1.0 m/s`). **This is a forward-simulation + RL training result** (v0.3),
not the differentiable path — see
[docs/examples/go2_locomotion.md](docs/examples/go2_locomotion.md).

## Highlights — what makes Nuka different

1. **Strong (D1) determinism, bit-exact across two runs.** No floating-point
   atomics, fixed reduction order in every physics path. Re-running a simulation
   from the same inputs reproduces the result *byte-for-byte*. This is enforced
   by a physics-smell lint that must pass on every PR.

2. **Full analytical reverse-mode adjoint** through rigid + articulated dynamics
   (Featherstone ABA, reverse-mode), with a recorded tape + gradient
   checkpointing and an implicit-function-theorem (IFT) contact gradient at
   convergence. This is a *real* analytical adjoint — not a stop-gradient
   approximation through contact (Brax) — so gradients are exact and
   engine-consistent. (Scope honestly stated: floating-base **orientation**-channel
   and the d/dM, d/dJ contact-derivative channels are deferred to v0.7.)

3. **Zero-copy PyTorch + JAX interop via DLPack.** Engine buffers alias torch /
   jax tensors with no copy. Both frontends drive the *same* deterministic engine
   adjoint, so `jax.grad` and `loss.backward()` produce gradients that **agree to
   engine round-off** (a tight rel-err < 1e-4 gate; ~1e-6 in practice).

4. **Sim-to-real noise, deterministic and replayable.** N1 per-field sensor noise
   (Gaussian / Poisson) + N2 per-episode domain randomization, both driven by a
   counter-based Philox4x32-10 RNG — so noise is a pure function of
   `(seed, index, sequence)`, two-run bit-exact, and re-derived identically on the
   diff-sim backward. Off by default, so oracle scenes stay byte-identical.

## Quick start

Nuka is built and run on **Linux** with **CUDA 12.8** and **g++-10**. A
CUDA-capable GPU is required for the production physics path. (A `pip install
nuka-physics` from PyPI is *planned*, not yet published — build from source for
now.)

### 1. Build the C++ engine (GPU)

```bash
export CC=gcc-10 CXX=g++-10
cmake -S . -B build-cuda128 \
  -DNK_BUILD_TESTS=ON \
  -DNK_REQUIRE_CUDA=ON \
  -DNK_PHYSICS_BACKEND=CUDA
cmake --build build-cuda128 -j
```

Always use the `build-cuda128` directory (never the legacy `build/`). See
[CONTRIBUTING.md](CONTRIBUTING.md) and [`python/README.md`](python/README.md) for
the toolchain environment variables.

### 2. Install the Python bindings

```bash
pip install -e python
```

### 3. Backpropagate through the simulator — a link-mass gradient

The differentiable rollout lives in `nuka.autograd` (it is *not* re-exported at
the top level, so `import nuka` never hard-requires torch). Pass the parameter
you want a gradient for as a **torch tensor** via `params=` — calling `.item()`
on it would sever the autograd graph.

```python
import torch
import nuka
from nuka.autograd import differentiable_rollout

scene = "examples/scenes/go2_system_id.usda"   # fixed-base Go2, single env

with nuka.Device.create(0) as dev:
    # SINGLE-ENV, contact-free differentiable path: one fresh world + tape.
    world = nuka.World.create_from_scene(dev, scene, 1)
    world.set_gravity_z(-9.81)
    tape = nuka.Tape.create(
        world, checkpoint_interval=3, max_tape_entries=4096,
        max_checkpoints=512, recompute_on_backward=1,
    )

    # The mass of one thigh link (GLOBAL link index 2) as a differentiable leaf.
    mass = torch.nn.Parameter(torch.tensor([0.9], device="cuda"))

    # A fixed K=30-step rest-hold PD target sequence: (K, action_dim).
    action_dim = world.base_link_count - 1            # 12 for Go2
    actions = torch.zeros(30, action_dim, device="cuda")

    # Drive the rollout; observe post-rollout joint velocities (qdot).
    obs = differentiable_rollout(
        world, tape, actions,
        params=mass, param_link_indices=[2], obs="qdot",
    )
    loss = obs.pow(2).mean()
    loss.backward()

    print("dLoss/dmass =", mass.grad.item())        # read the gradient...
    tape.destroy()                                   # ...BEFORE destroying the tape
    world.destroy()
```

`world.reset()` is **not supported** on this single-env path; build a fresh
world + tape per evaluation (this is cheap — a few ms). The full worked example
is the system-ID demo: [docs/examples/system_identification.md](docs/examples/system_identification.md).

### Run the RL locomotion training (forward sim)

```bash
CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py --num_actors 4096
# fast launch proof:
CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py --smoke
```

## Status

**v0.5 — initial public release.** What ships:

- 4096+ parallel articulated environments on a single GPU (v0.3 forward sim).
- Differentiable simulation through **rigid + articulated (Featherstone)**
  dynamics: analytical adjoint, tape + gradient checkpointing, IFT-at-convergence
  contact gradient, parameter (link-mass) and control-mode gradients.
- Zero-copy PyTorch + JAX (DLPack) interop.
- Sim-to-real noise N1 (sensor) + N2 (domain randomization).
- Self-written deterministic sparse solver (CG + Jacobi / Block-Jacobi,
  fixed-order reductions, D1 bit-exact) for the IFT path — no closed-source SDK.

**Not yet shipped (roadmap):** soft body, fluid, and rigid↔soft↔fluid
cross-system coupling are **v0.7**; floating-base **orientation**-channel and
d/dM, d/dJ contact gradients are **v0.7**. See the
[master plan](docs/plans/2026-05-28-nuka-physics-master-plan.md) §7.

## Documentation

- [Getting started](docs/getting-started.md) — prerequisites, build, hello-world,
  RL quickstart.
- Concepts: [architecture](docs/concepts/architecture.md) ·
  [differentiable simulation](docs/concepts/diff-sim.md) ·
  [migrating from Isaac Lab](docs/concepts/isaaclab-compat.md)
- Examples: [Go2 locomotion (v0.3)](docs/examples/go2_locomotion.md) ·
  [system identification (v0.5)](docs/examples/system_identification.md)
- Architecture deep-dives: [Runtime overview](docs/architecture/runtime-overview.md) ·
  [diff-sim tape + checkpointing](docs/architecture/diffsim-tape.md) ·
  [sim-to-real noise](docs/architecture/sim2real-noise.md) ·
  [scene compiler](docs/architecture/scene-compiler.md)
- Launch blog: [Nuka Physics v0.5](docs/blog/2026-06-01-v05-launch.md)
- [Master plan](docs/plans/2026-05-28-nuka-physics-master-plan.md) — architecture +
  long-term roadmap (the source of truth for the version plan).

## Scene import

Scenes import from **MJCF** (`.xml`), **URDF**, and **text USD** (`.usda`). The
`.usda` parser is hand-written using only the C++ standard library — there is no
OpenUSD SDK dependency and no binary `.usdc` / `.usdz` support. XML is parsed via
tinyxml2. See [NOTICE](NOTICE).

## Contributing

Contributions are welcome under a signed [Contributor License Agreement](CLA.md)
(the CLA-assistant bot checks this on every PR). Read
[CONTRIBUTING.md](CONTRIBUTING.md) for the build, lint, and test gates, and the
[Code of Conduct](CODE_OF_CONDUCT.md). For security issues, follow
[SECURITY.md](SECURITY.md).

## License

**Apache License 2.0** with a Contributor License Agreement. See
[LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party attributions.
