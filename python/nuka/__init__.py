"""nuka -- Python package for the Nuka physics engine.

Thin re-export layer over the nanobind extension ``_nuka_ext`` plus a few
ergonomic helpers. See ``python/README.md`` for the ABI rules and the DLPack
zero-copy usage.

Typical use::

    import nuka
    import torch

    with nuka.Device.create(0) as dev:
        world = nuka.World.create_from_scene(dev, scene_path, env_count=4096)
        q = torch.from_dlpack(world.buffer_view(nuka.Field.JOINT_POSITION))
        tgt = torch.from_dlpack(world.buffer_view(nuka.Field.DRIVE_TARGET))
        tgt.copy_(...)          # write PD targets in place (zero-copy)
        world.step()            # next step applies them
        world.destroy()

Optional torch surface (``nuka.autograd``, ``nuka.torch_stream_ptr``): imported
**lazily** so ``import nuka`` never hard-requires torch. torch is only loaded
the moment you touch one of those names. See ``python/README.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._nuka_ext import (  # noqa: F401
    Device,
    Field,
    World,
    sync,
    __engine_version__,
)

# Re-export the field enum members at top level for convenience.
RIGID_BODY_TRANSFORM = Field.RIGID_BODY_TRANSFORM
ARTICULATION_LINK_POSE = Field.ARTICULATION_LINK_POSE
JOINT_POSITION = Field.JOINT_POSITION
JOINT_VELOCITY = Field.JOINT_VELOCITY
OBSERVATIONS = Field.OBSERVATIONS
CONTACT_POINTS = Field.CONTACT_POINTS
DRIVE_TARGET = Field.DRIVE_TARGET
# infer-enable #40: base velocity (read) + writable PD gains.
LINK_VELOCITY = Field.LINK_VELOCITY
DRIVE_STIFFNESS = Field.DRIVE_STIFFNESS
DRIVE_DAMPING = Field.DRIVE_DAMPING
DRIVE_FORCE_LIMIT = Field.DRIVE_FORCE_LIMIT
# p03 episode-boundary fix: authoritative per-env root pose (un-lagged; correct
# immediately after reset_envs). Shape (env_count, 7) == [px,py,pz, qw,qx,qy,qz].
BASE_POSE = Field.BASE_POSE

# ARTICULATION_LINK_POSE element = 7 floats [px,py,pz, qw,qx,qy,qz] (quat w-first).
LINK_POSE_FLOATS = 7
# LINK_VELOCITY element = 6 floats [wx,wy,wz, vx,vy,vz] (omega-first spatial vel).
# Root slot is the live base spatial velocity in the ROOT-LINK BODY frame (already
# body-local, NOT world); non-root slots are engine-internal Featherstone-local.
LINK_VELOCITY_FLOATS = 6

def torch_stream_ptr() -> int:
    """The current torch CUDA stream as a ``uintptr_t`` (for ``Device.create``).

    ``nuka.Device.create(ordinal, stream_ptr=nuka.torch_stream_ptr())`` makes the
    engine run every kernel on torch's *current* CUDA stream, so torch ops and
    physics share ordering with no explicit ``nuka.sync()`` /
    ``cudaStreamSynchronize`` between them.

    Imports torch lazily: calling this requires torch, but ``import nuka`` does
    not. (To pin a specific stream, pass that stream's ``.cuda_stream`` to
    ``Device.create`` directly, e.g. under ``with torch.cuda.stream(s):`` use
    ``nuka.torch_stream_ptr()``, which then returns ``s``'s pointer.)
    """
    import torch  # lazy: torch is an optional dep of `nuka`

    return torch.cuda.current_stream().cuda_stream


# --- PEP 562 lazy submodule import (keeps torch optional) --------------------
# `nuka.autograd` pulls in torch at its own import time. We must NOT eagerly
# `from . import autograd` here, or `import nuka` would hard-require torch.
# Instead resolve `nuka.autograd` on first attribute access; `import nuka`
# (without torch) keeps working, and the torch ImportError surfaces only when
# autograd is actually used.
if TYPE_CHECKING:  # for type checkers / IDEs only; no runtime torch import.
    from . import autograd  # noqa: F401


def __getattr__(name: str):
    if name == "autograd":
        import importlib

        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod  # cache so subsequent access is a normal attr.
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Device",
    "World",
    "Field",
    "sync",
    "__engine_version__",
    "RIGID_BODY_TRANSFORM",
    "ARTICULATION_LINK_POSE",
    "JOINT_POSITION",
    "JOINT_VELOCITY",
    "OBSERVATIONS",
    "CONTACT_POINTS",
    "DRIVE_TARGET",
    "LINK_VELOCITY",
    "DRIVE_STIFFNESS",
    "DRIVE_DAMPING",
    "DRIVE_FORCE_LIMIT",
    "BASE_POSE",
    "LINK_POSE_FLOATS",
    "LINK_VELOCITY_FLOATS",
    "torch_stream_ptr",
    # NOTE: "autograd" is deliberately NOT in __all__. It is resolved lazily via
    # __getattr__ (PEP 562) so `import nuka` / `nuka.autograd.step(...)` work
    # without eagerly importing torch. Listing it here would make
    # `from nuka import *` trigger the torch import (fails in a torch-less env).
    # `torch_stream_ptr` is safe to export: its torch import is inside the body,
    # not triggered by binding the name.
]
