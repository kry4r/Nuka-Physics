# Getting started

This guide takes you from a clean Linux box to a built engine, installed Python
bindings, a hello-world forward step, a first differentiable rollout, and the RL
training quickstart.

## Prerequisites

Nuka's production physics path is **CUDA-only** and is developed and supported on
**Linux**:

- **Linux** (x86-64).
- **CUDA 12.8** toolkit and a CUDA-capable GPU. The GPU is required to build and
  run the production physics path.
- **g++-10** (the C++11 ABI is pinned to `1`, which is the ABI the Python
  bindings expect). Use this compiler for both the engine and the bindings.
- **CMake 3.20+**.
- **Vulkan SDK** *(optional)* — only needed if you build the optional offscreen
  renderer component (`nuka_render`). Vulkan is **not** part of the shipped
  Python runtime; the `_nuka_ext` extension links only the engine library and the
  CUDA runtime. You can skip the Vulkan SDK entirely for physics + diff-sim + RL.

> `pip install nuka-physics` from PyPI is **planned but not yet published**.
> Build from source as below.

## 1. Build the C++ engine (GPU)

Mirror the build flow in [CONTRIBUTING.md](../CONTRIBUTING.md). Always build into
`build-cuda128` — never the legacy `build/` directory.

```bash
export CC=gcc-10 CXX=g++-10
cmake -S . -B build-cuda128 \
  -DNK_BUILD_TESTS=ON \
  -DNK_REQUIRE_CUDA=ON \
  -DNK_PHYSICS_BACKEND=CUDA
cmake --build build-cuda128 -j
```

## 2. Install the Python bindings

The Python extension (`_nuka_ext`) is built against the engine in
`build-cuda128`:

```bash
pip install -e python
```

See [`python/README.md`](../python/README.md) for the environment variables
(`NUKA_BUILD_DIR`, `CUDA_LIB_DIR`, `CMAKE_CXX_COMPILER`) that point the build at
the engine library and the matching ABI compiler.

PyTorch and JAX are **optional** runtime interop targets — `import nuka` does not
require either. Install `torch` to use `nuka.autograd`, and `jax` to use
`nuka.jax_frontend`.

## 3. Hello world — a forward step

`nuka.World.create_from_scene(device, scene_path, env_count)` builds a batched
world from a scene; buffers are exposed zero-copy via DLPack. The example below
writes PD drive targets in place and steps.

```python
import torch
import nuka

scene = "examples/scenes/go2_locomotion.usda"

with nuka.Device.create(0) as dev:
    world = nuka.World.create_from_scene(dev, scene, env_count=4096)

    # Zero-copy views: torch tensors aliasing the engine's device buffers.
    q   = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))
    tgt = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))

    # Hold the current joint positions as the PD target (actuated slots [1:]).
    tgt[:, 1:].copy_(q[:, 1:])

    world.step()        # one fixed step (default dt = 1/240 s)
    nuka.sync()         # kernels are async; sync before reading results back

    world.destroy()
```

Notes:
- The `DRIVE_TARGET` / `JOINT_POSITION` buffers are `base_link_count` wide; slot
  0 is the (inert) root link and the actuated joints occupy slots `[1:]`. A
  policy emits an `(env_count, action_dim)` tensor where
  `action_dim == base_link_count - 1` (12 for Go2).
- Engine kernels run asynchronously. Call `nuka.sync()` (or pin the engine to a
  torch stream via `nuka.Device.create(0, stream_ptr=nuka.torch_stream_ptr())`)
  before reading buffers back on the host side.

## 4. Your first gradient

The differentiable rollout is single-env and contact-free. It lives in
`nuka.autograd` (torch-optional design — not re-exported at top level). Pass the
parameter you differentiate as a **torch tensor** via `params=`.

```python
import torch
import nuka
from nuka.autograd import differentiable_rollout

scene = "examples/scenes/go2_system_id.usda"

with nuka.Device.create(0) as dev:
    world = nuka.World.create_from_scene(dev, scene, 1)   # SINGLE env
    world.set_gravity_z(-9.81)
    tape = nuka.Tape.create(
        world, checkpoint_interval=3, max_tape_entries=4096,
        max_checkpoints=512, recompute_on_backward=1,
    )

    mass = torch.nn.Parameter(torch.tensor([0.9], device="cuda"))
    action_dim = world.base_link_count - 1
    actions = torch.zeros(30, action_dim, device="cuda")

    obs = differentiable_rollout(
        world, tape, actions,
        params=mass, param_link_indices=[2], obs="qdot",
    )
    obs.pow(2).mean().backward()
    print("dLoss/dmass =", mass.grad.item())

    tape.destroy()      # destroy AFTER reading the gradient
    world.destroy()
```

The mass gradient comes from the engine's deterministic `grad_parameters` (a
single `tape.backward`), host→device copied — it is **not** an autograd trace of
`set_link_mass`. See [docs/concepts/diff-sim.md](concepts/diff-sim.md) for the
full model, and [docs/examples/system_identification.md](examples/system_identification.md)
for the end-to-end demo.

## 5. RL training quickstart

The Go2 PPO locomotion task trains on-GPU via rl_games. The launcher is
`examples/training/train_go2_ppo.py` (single-GPU only — always set
`CUDA_VISIBLE_DEVICES=0`):

```bash
# Full run (4096 actors):
CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py --num_actors 4096

# Tiny launch proof (3 epochs, 256 actors):
CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py --smoke
```

See [docs/examples/go2_locomotion.md](examples/go2_locomotion.md) for the full
walkthrough.

## Running the tests

```bash
# C++ tests:
ctest --test-dir build-cuda128 --output-on-failure

# Python tests:
pytest python/tests/

# Physics-smell lint (must pass on every PR):
python tools/lint/physics_smell.py
```

### Two known-fail oracle tests (expected — not regressions)

Two oracle tests are **documented, pre-existing, carry-forward failures**,
reflecting an ABA-vs-MuJoCo/MJX oracle gap that predates current work. A clean
checkout will show exactly these two failing, and that is expected:

- `V01FoundationE2E.Phase6CudaAbaMatchesGo2AndH1MuJoCoOracle`
- `FeatherstoneOracle.RandomSampleGoldensMatchCudaAba`

Your changes must not introduce any *new* failures and must not "fix" these by
weakening the oracle. See [CONTRIBUTING.md](../CONTRIBUTING.md).
