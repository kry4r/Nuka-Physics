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
"""

from __future__ import annotations

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

# ARTICULATION_LINK_POSE element = 7 floats [px,py,pz, qw,qx,qy,qz] (quat w-first).
LINK_POSE_FLOATS = 7
# LINK_VELOCITY element = 6 floats [wx,wy,wz, vx,vy,vz] (omega-first spatial vel).
# Root slot is the live base spatial velocity in the ROOT-LINK BODY frame (already
# body-local, NOT world); non-root slots are engine-internal Featherstone-local.
LINK_VELOCITY_FLOATS = 6

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
    "LINK_POSE_FLOATS",
    "LINK_VELOCITY_FLOATS",
]
