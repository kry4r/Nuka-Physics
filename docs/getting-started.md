# Getting started

This guide covers the supported source build, a first GPU step, and the Python
authoring path used by the demos.

## Requirements

- Linux x86-64
- A CUDA-capable GPU and CUDA 12.8
- GCC / g++ 10
- CMake 3.20+
- Python 3.10+

Vulkan is optional. It is needed for Vulkan validation targets, but not for the
core CUDA simulator or Python bindings.

## Build and install

From the repository root:

```bash
export CC=gcc-10 CXX=g++-10
cmake -S . -B build-cuda128 \
  -DNK_BUILD_TESTS=ON \
  -DNK_REQUIRE_CUDA=ON \
  -DNK_PHYSICS_BACKEND=CUDA
cmake --build build-cuda128 -j
pip install -e python
```

The Python extension links against `build-cuda128/src/libnuka.so`. Advanced
build-directory and CUDA-library overrides are listed in
[`python/README.md`](../python/README.md).

Check the installation:

```bash
python -c "import nuka; print(nuka.__engine_version__)"
```

## Step a batched world

`nuka.World` owns GPU-resident state. Buffer views use DLPack, so PyTorch reads
and writes the engine buffers without a host copy.

```python
import torch
import nuka

with nuka.Device.create(0) as device:
    world = nuka.World.create_from_scene(
        device,
        "examples/scenes/go2_locomotion.usda",
        env_count=4096,
    )

    q = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))
    target = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))
    target[:, 1:].copy_(q[:, 1:])

    world.step()
    nuka.sync()
    world.destroy()
```

Slot `0` is the root link. Actuated joints occupy `target[:, 1:]`. Engine work
is asynchronous; call `nuka.sync()` before a host read, or create the device
with `stream_ptr=nuka.torch_stream_ptr()` to share PyTorch's current stream.

## Author a demo scene

The high-level authoring API separates four concerns:

| Part | API | Purpose |
|---|---|---|
| Geometry | `nuka.author.morphs` | Robots, rigid primitives, cloth grids, soft bodies, fluids, granular beds, and cables |
| Physics | `nuka.author.materials` | Mass, friction, XPBD, PBF, and MLS-MPM parameters |
| Appearance | `materials.Render`, `surfaces` | PBR appearance and deformable render surfaces |
| Assembly | `nuka.author.Scene` | Cook the entities into one GPU world |

This complete scene drops a rigid box onto a ground plane and saves a rendered
frame:

```python
from pathlib import Path

import numpy as np
from PIL import Image

import nuka
from nuka.author import Scene, SimOptions, materials, morphs

scene = Scene(
    SimOptions(dt=1.0 / 240.0, env_count=1, gravity=(0.0, 0.0, -9.81))
)
scene.add_entity(
    morphs.Ground(),
    materials.Rigid(static=True, friction=0.8),
)
scene.add_entity(
    morphs.Box(half_extents=(0.18, 0.18, 0.18), pos=(0.0, 0.0, 1.0)),
    materials.Rigid(mass=1.0, friction=0.6),
    render=materials.Render(
        "signal_red", base_color=(0.75, 0.08, 0.06), roughness=0.65
    ),
)

with nuka.Device.create(0) as device:
    world = scene.build(device)
    world.step_n(240)
    image = world.render_beauty(
        eye=(2.2, -2.2, 1.5),
        look=(0.0, 0.0, 0.3),
        width=1280,
        height=720,
        spp=64,
    )

    output = Path("out/first-scene.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image)).save(output)
    world.destroy()
```

Use `morphs.NKS("path/to/scene.nks")` to start from a cooked robot or world,
then add rigid or deformable entities through the same `Scene`. Unsupported
material/geometry combinations fail during `scene.build()` instead of silently
selecting another solver.

Representative authoring examples:

```bash
python examples/demo/render_realism_demo.py --spp 64
python examples/demo/go2_cloth_drape.py
python examples/demo/soft_ball_py.py --steps 180
```

## Train a vectorized task

The Go2 launcher uses the GPU-resident Gymnasium / rl_games adapter:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/training/train_go2_ppo.py --num_actors 4096
```

Use `--smoke` for a short launch check.

## Validate a change

```bash
ctest --test-dir build-cuda128 --output-on-failure
pytest python/tests/
python tools/lint/physics_smell.py
```

The lint enforces the deterministic physics rules, public-header constraints,
and generated-file policy. Contribution requirements live in
[`CONTRIBUTING.md`](../CONTRIBUTING.md).
