# `nuka` -- Python binding for the Nuka physics engine

A [nanobind](https://nanobind.readthedocs.io/) extension that replaces the
ctypes stopgap (`examples/sim_val/nuka_cabi.py`) with a typed, **zero-copy**
Python binding. Observations / state are handed to torch and PD drive targets
are written back **without any host round-trip** via the DLPack protocol, at
4096 envs.

## The ABI rule (read this first)

The Nuka engine static/shared libs are built with **g++-10, CXX11 ABI = 1**
(the GCC default). This extension is compiled the same way and links
`libnuka.so`.

**libtorch is NOT linked into this extension.** The pip `torch` wheel ships
libtorch built with the *old* pre-CXX11 ABI (`_GLIBCXX_USE_CXX11_ABI=0`).
Linking it next to ABI=1 engine symbols would clash. Instead:

```
import nuka     # CXX11 ABI = 1  (links libnuka.so)
import torch    # ABI = 0        (its own libtorch)
```

coexist in one process and exchange tensors **only as DLPack capsules** -- an
ABI-neutral C struct protocol. That is the whole reason DLPack is used here.

## Build / install

Prereqs (already in the `nuka-v03` conda env): torch 2.6, nanobind 2.12,
scikit-build-core, ninja, numpy; CUDA 12.8 toolkit; g++-10; the engine built at
`build-cuda128/src/libnuka.so`.

```bash
export CUDA_VISIBLE_DEVICES=0
export CC=gcc-10 CXX=g++-10
cd python
# editable install (no build isolation so it reuses the env's nanobind/torch)
/root/miniconda3/envs/nuka-v03/bin/pip install -e . --no-build-isolation
# or a wheel:
/root/miniconda3/envs/nuka-v03/bin/pip wheel . --no-build-isolation -w dist
```

scikit-build-core drives `CMakeLists.txt`. Key knobs (override with
`-C cmake.define.NAME=VALUE` or env):

| var | default | meaning |
|-----|---------|---------|
| `NUKA_ROOT` | `..` | repo root holding `src/include/nuka/nuka.h` |
| `NUKA_BUILD_DIR` | `$NUKA_ROOT/build-cuda128` | dir holding `src/libnuka.so` |
| `CUDA_LIB_DIR` | `/opt/cuda-12.8-root/.../cuda-12.8/lib64` | `libcudart.so.12` dir |
| `CMAKE_CXX_COMPILER` | `g++-10` | must match the engine ABI |

The CMake pins `_GLIBCXX_USE_CXX11_ABI=1` and sets an `INSTALL_RPATH` to the
engine + CUDA lib dirs, so the extension finds `libnuka.so` / `libcudart.so.12`
with no `LD_LIBRARY_PATH`.

## Zero-copy DLPack usage

```python
import nuka, torch

with nuka.Device.create(0) as dev:
    world = nuka.World.create_from_scene(
        dev, "examples/scenes/go2_float.usda", env_count=4096)

    # READ: zero-copy CUDA tensor aliasing the engine's live q buffer.
    q = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))   # (4096, 13)
    assert q.is_cuda and q.shape == (world.env_count, world.base_link_count)

    # WRITE: DRIVE_TARGET view is writable -> torch writes in place.
    tgt = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))    # (4096, 13)
    tgt[:, 1:13] = my_policy_action                                  # in place
    world.step()    # the next step reads the new PD targets

    # Ergonomic alternative (host or CUDA 1D array -> drive buffer copy):
    world.set_drive_targets(flat_targets)   # len == env_count*base_link_count

    world.destroy()
```

`world.buffer_view(field)` returns an `nb::ndarray` exposing `__dlpack__` /
`__dlpack_device__`; `torch.from_dlpack` consumes it with **no copy**, so the
returned tensor's `data_ptr()` equals the engine's device pointer. The
`field` argument is generic -- a future field (e.g. base velocity) works with no
binding change.

### Fields (`nuka.Field`, mirrors `nuka_state_field_t`)

| field | shape (go2) | writable | notes |
|-------|-------------|----------|-------|
| `JOINT_POSITION` | `(env_count, 13)` | no | q; slot 0 root, 1..12 actuated joints |
| `JOINT_VELOCITY` | `(env_count, 13)` | no | qd |
| `ARTICULATION_LINK_POSE` | `(env_count, 13, 7)` | no | world pose `[px,py,pz, qw,qx,qy,qz]` (quat **w-first**) |
| `LINK_VELOCITY` | `(env_count, 13, 6)` | no | spatial velocity `[wx,wy,wz, vx,vy,vz]` (**omega-first**); see frame note below |
| `DRIVE_TARGET` | `(env_count, 13)` | **yes** | per-env PD targets; next `step()` applies |
| `DRIVE_STIFFNESS` | `(env_count, 13)` | **yes** | per-joint Kp; next `step()` applies (Go2 policy: 20 on slots 1..12) |
| `DRIVE_DAMPING` | `(env_count, 13)` | **yes** | per-joint Kd; next `step()` applies (Go2 policy: 0.5 on slots 1..12) |
| `DRIVE_FORCE_LIMIT` | `(env_count, 13)` | **yes** | per-joint symmetric torque clamp `|tau| <= limit` (Nm) |
| `OBSERVATIONS` | -- | -- | not served on the batched path (engine returns NOT_SUPPORTED) |

### `LINK_VELOCITY` frame (read carefully -- the policy contract)

The **ROOT slot** (`vel[:, 0, :]`, the floating-base trunk on `go2_float.usda`)
is the live base spatial velocity expressed in the **root-link BODY frame** --
it is **already body-local, NOT world**. So a Go2 locomotion observation applies
only its own fixed axis/scale remap to it, **not** a world->body rotation (doing
both would double-rotate). It carries **no one-step lag** (unlike
`ARTICULATION_LINK_POSE`): the floating-base velocity integrate and the contact
solve both write it within the same `step()`. Order is **omega-first**:
`[wx,wy,wz, vx,vy,vz]`.

The **non-root slots** (`vel[:, 1:, :]`) are engine-internal Featherstone
per-link velocities in each link's own local frame; they are non-zero but their
exact timing/frame is an implementation detail and **not** a defined observation
-- do not rely on them. They are exposed only for index symmetry with the link
pose. Use `go2_float.usda` (not the fixed-base `go2_stand`, whose root has no
free 6-DOF velocity) for a live base.

### Driving a trained policy at its training gains

```python
kp = torch.from_dlpack(world.buffer_view(nuka.DRIVE_STIFFNESS))  # (env, 13)
kd = torch.from_dlpack(world.buffer_view(nuka.DRIVE_DAMPING))
kp[:, 1:13] = 20.0   # Go2 training Kp on the 12 actuated joints (in place)
kd[:, 1:13] = 0.5    # Go2 training Kd
vel = torch.from_dlpack(world.buffer_view(nuka.LINK_VELOCITY))    # (env, 13, 6)
base_vel_body = vel[:, 0, :]   # [wx,wy,wz,vx,vy,vz], root-link body frame
world.step()   # the next step uses the new gains
```

## Layout metadata

`world.env_count`, `world.base_link_count` (13 for go2), `world.dt`. Indexing is
env-major: element `(env, link)` for the flat fields, `(env, link, 7)` for link
pose.

## Tests

```bash
export CUDA_VISIBLE_DEVICES=0
/root/miniconda3/envs/nuka-v03/bin/python -m pytest python/tests -v
```

Proves: import nuka+torch together (no ABI crash); DLPack device-ptr match
(zero-copy) for q/qd/pose/`DRIVE_TARGET`/`LINK_VELOCITY`/gain fields; drive write
-> q moves sign-correctly after step; bit-exact determinism; live floating base
and live root `LINK_VELOCITY` on `go2_float`; writing `DRIVE_STIFFNESS`/
`DRIVE_DAMPING` (Kp=20/Kd=0.5) reaches the solver.

## Known C ABI gaps (flagged, not worked around)

* No explicit `base_link_count` accessor -- derived as
  `element_count(JOINT_POSITION) / env_count`.
* `OBSERVATIONS` is not served on the batched (multi-env) path.
