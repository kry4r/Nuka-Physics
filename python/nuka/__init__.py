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
    Tape,
    # Built-scene authoring: program rigid primitives + TAGGED media records, then
    # build() cooks them through the SAME CookSceneToModel a file scene uses (the
    # general media authoring substrate -- tet-soft / fluid / a second cloth).
    SceneBuilder,
    sync,
    __engine_version__,
    DETERMINISM_STRONG,
    DETERMINISM_WEAK,
    # v0.5 C-fwd stage-1 control modes (the create_from_scene control_mode= ints).
    CONTROL_MODE_PD_POSITION,
    CONTROL_MODE_TORQUE,
    CONTROL_MODE_VELOCITY,
    CONTROL_MODE_COMPUTED_TORQUE,
    CONTROL_MODE_OSC,
    CONTROL_MODE_ACTUATOR,
    # SceneBuilder.add_rigid_primitive(kind=...) codes.
    PRIMITIVE_PLANE,
    PRIMITIVE_BOX,
    PRIMITIVE_SPHERE,
    PRIMITIVE_CAPSULE,
    # SceneBuilder.add_media(kind=..., method=...) codes.
    MEDIA_CLOTH,
    MEDIA_SOFT_TET,
    MEDIA_FLUID,
    MEDIA_GRANULAR,
    MEDIA_CABLE,
    MEDIA_METHOD_XPBD,
    MEDIA_METHOD_PBF,
    MEDIA_METHOD_MLSMPM,
    # SceneBuilder.add_media(cloth_pin=...) pin-set codes.
    CLOTH_PIN_FROM_FREE,
    CLOTH_PIN_NONE,
    CLOTH_PIN_PERIMETER,
    CLOTH_PIN_EDGE_X0,
    CLOTH_PIN_EDGE_X1,
    CLOTH_PIN_EDGE_Y0,
    CLOTH_PIN_EDGE_Y1,
    # SceneBuilder.add_media(cable_pin=...) endpoint-pin codes.
    CABLE_PIN_START,
    CABLE_PIN_END,
    CABLE_PIN_BOTH,
    CABLE_PIN_NONE,
    # Device-resident batched camera sensor: the AOV plane + the mount frame.
    SensorChannel,
    SensorMount,
)

import dataclasses as _dataclasses
from typing import Sequence, Tuple


@_dataclasses.dataclass
class CameraCfg:
    """One camera authoring a World.attach_camera_sensor call.

    mount/mount_index name the FK pose the camera follows; pos + quat (w,x,y,z) is
    its offset in that frame (camera-local axes: -Z forward, +Y up). width/height
    size every env image; vfov_deg is the vertical field of view. Calling .attach
    for several CameraCfgs at the SAME width/height gives S cameras per env (each on
    its own mount) -- the sensor view is then (E,S,H,W,ch).
    """

    width: int
    height: int
    vfov_deg: float = 60.0
    mount: int = SensorMount.BASE
    mount_index: int = 0
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def attach(self, world: "World") -> None:
        """Attach this camera to `world` (appends a per-env camera)."""
        local_offset: Sequence[float] = (
            self.pos[0], self.pos[1], self.pos[2],
            self.quat[0], self.quat[1], self.quat[2], self.quat[3],
        )
        # mount may be a SensorMount enum or a plain int (.value pulls the ordinal).
        mount = self.mount.value if hasattr(self.mount, "value") else int(self.mount)
        world.attach_camera_sensor(
            int(mount), int(self.mount_index), list(local_offset),
            float(self.vfov_deg), int(self.width), int(self.height),
        )

# v0.5 p04 Task 5.4.9 sim-to-real noise config (PURE python -- no torch/jax dep,
# so eager import is safe; see noise.py). GaussianNoise / PoissonNoise wrap the
# per-field sensor-noise C-ABI; DomainRandomization wraps the per-episode DR.
from .noise import (  # noqa: F401
    GaussianNoise,
    PoissonNoise,
    DomainRandomization,
    NOISE_NONE,
    NOISE_GAUSSIAN,
    NOISE_POISSON,
)

# M9 T4: the GENERIC scene-authoring surface (PURE python wrapper over the
# _nuka_ext.Scene C-ABI; no torch dep, so eager import is safe -- mirrors noise).
# ONE entry loads any format (mjcf/urdf/usd/nks); uniform compose/find/set_local/
# set_physics_material/settle/save. NOT a special grasp/union type.
from .scene import Scene  # noqa: F401

# Authoring facade (pure python over the prebuilt bindings): geometry-only morphs,
# constitutive materials, surfaces + SimOptions assemble a coupled world; control
# drives it. The facade assembler is nuka.author.Scene (the top-level nuka.Scene
# stays the generic scene-graph editor above); these helpers do not conflict.
from . import author  # noqa: F401
from .author import SimOptions, materials, morphs, surfaces  # noqa: F401

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
# v0.5 C-fwd: writable per-link control inputs for the non-PD control modes.
TORQUE_INPUT = Field.TORQUE_INPUT
VELOCITY_TARGET = Field.VELOCITY_TARGET
ACTUATOR_NOLOAD_SPEED = Field.ACTUATOR_NOLOAD_SPEED
# v0.5 C-fwd slice 3: writable per-ENV Osc task target (float3 {x,y,z} per env).
TASK_TARGET = Field.TASK_TARGET
# p03 episode-boundary fix: authoritative per-env root pose (un-lagged; correct
# immediately after reset_envs). Shape (env_count, 7) == [px,py,pz, qw,qx,qy,qz].
BASE_POSE = Field.BASE_POSE
# p14a (v0.7) contact-force readout (batched/multi-env only). LINK_CONTACT_WRENCH:
# net per-link world wrench, shape (env, base_link, 6) == [Fx,Fy,Fz, Tx,Ty,Tz],
# tau about the link frame origin. CONTACT_NORMAL / CONTACT_FORCE: per-slot, shape
# (env, 4, 3). CONTACT_LINK: per-slot owning GLOBAL link index, shape (env, 4),
# dtype uint32 (the only non-float32 field).
LINK_CONTACT_WRENCH = Field.LINK_CONTACT_WRENCH
CONTACT_NORMAL = Field.CONTACT_NORMAL
CONTACT_FORCE = Field.CONTACT_FORCE
CONTACT_LINK = Field.CONTACT_LINK
# Go2-on-stairs Phase 2a: writable per-env procedural-terrain controls.
# ENV_TERRAIN_TYPE (uint32, default 0=Flat): 0=Flat, 1=PyramidStairs,
# 2=InvertedPyramid, 3=RandomBoxes; buffer_view shape (env_count, 1).
# ENV_TERRAIN_DIFFICULTY (float32, default 1.0): per-env curriculum scale of the
# terrain's vertical feature height; buffer_view shape (env_count,). The terrain
# geometry (step_height etc.) is set at create via
# World.create_from_scene(terrain_step_height=..., ...).
ENV_TERRAIN_TYPE = Field.ENV_TERRAIN_TYPE
ENV_TERRAIN_DIFFICULTY = Field.ENV_TERRAIN_DIFFICULTY
# T3 unified-actuator feed-forward joint force (per-link, default 0): tau += joint_f
# in every control mode (gravcomp / computed-torque / RL residual). Writable
# zero-copy; same layout/slot map as DRIVE_TARGET.
JOINT_FEEDFORWARD = Field.JOINT_FEEDFORWARD

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
if TYPE_CHECKING:  # for type checkers / IDEs only; no runtime torch/jax import.
    from . import autograd  # noqa: F401
    from . import jax_frontend  # noqa: F401
    from . import jax_state_pytree  # noqa: F401


# Submodules that pull in an OPTIONAL heavy dep at their own import time and so
# must stay lazy (resolved on first attribute access) -- NEVER eagerly imported,
# or `import nuka` would hard-require that dep:
#   * "autograd"          -> torch
#   * "jax_frontend"      -> jax + torch (the custom_vjp engine adjoint)
#   * "jax_state_pytree"  -> jax (the NukaWorldState pytree)
_LAZY_SUBMODULES = ("autograd", "jax_frontend", "jax_state_pytree")


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod  # cache so subsequent access is a normal attr.
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Device",
    "World",
    "Tape",
    "Scene",
    "SceneBuilder",
    "PRIMITIVE_PLANE",
    "PRIMITIVE_BOX",
    "PRIMITIVE_SPHERE",
    "PRIMITIVE_CAPSULE",
    "MEDIA_CLOTH",
    "MEDIA_SOFT_TET",
    "MEDIA_FLUID",
    "MEDIA_GRANULAR",
    "MEDIA_CABLE",
    "CLOTH_PIN_FROM_FREE",
    "CLOTH_PIN_NONE",
    "CLOTH_PIN_PERIMETER",
    "CLOTH_PIN_EDGE_X0",
    "CLOTH_PIN_EDGE_X1",
    "CLOTH_PIN_EDGE_Y0",
    "CLOTH_PIN_EDGE_Y1",
    "CABLE_PIN_START",
    "CABLE_PIN_END",
    "CABLE_PIN_BOTH",
    "CABLE_PIN_NONE",
    "MEDIA_METHOD_XPBD",
    "MEDIA_METHOD_PBF",
    "MEDIA_METHOD_MLSMPM",
    "author",
    "morphs",
    "materials",
    "surfaces",
    "SimOptions",
    "Field",
    "sync",
    "__engine_version__",
    "DETERMINISM_STRONG",
    "DETERMINISM_WEAK",
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
    "TORQUE_INPUT",
    "VELOCITY_TARGET",
    "ACTUATOR_NOLOAD_SPEED",
    "TASK_TARGET",
    "BASE_POSE",
    "LINK_CONTACT_WRENCH",
    "CONTACT_NORMAL",
    "CONTACT_FORCE",
    "CONTACT_LINK",
    "JOINT_FEEDFORWARD",
    "CONTROL_MODE_PD_POSITION",
    "CONTROL_MODE_TORQUE",
    "CONTROL_MODE_VELOCITY",
    "CONTROL_MODE_COMPUTED_TORQUE",
    "CONTROL_MODE_OSC",
    "CONTROL_MODE_ACTUATOR",
    "LINK_POSE_FLOATS",
    "LINK_VELOCITY_FLOATS",
    "torch_stream_ptr",
    # v0.5 p04 Task 5.4.9 sim-to-real noise config (pure-python, eager-safe).
    "GaussianNoise",
    "PoissonNoise",
    "DomainRandomization",
    "NOISE_NONE",
    "NOISE_GAUSSIAN",
    "NOISE_POISSON",
    # NOTE: "autograd" is deliberately NOT in __all__. It is resolved lazily via
    # __getattr__ (PEP 562) so `import nuka` / `nuka.autograd.step(...)` work
    # without eagerly importing torch. Listing it here would make
    # `from nuka import *` trigger the torch import (fails in a torch-less env).
    # `torch_stream_ptr` is safe to export: its torch import is inside the body,
    # not triggered by binding the name.
]
