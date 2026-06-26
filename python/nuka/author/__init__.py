"""nuka.author -- the Python authoring facade over the prebuilt nuka bindings.

A THIN structured layer: geometry-only ``morphs``, constitutive ``materials``,
and ``surfaces`` are assembled with ``SimOptions`` into a ``Scene``, whose
``build(device)`` cooks a coupled ``World``. A scene of just a robot + cloth takes
the ``create_coupled_from_scene`` fast path; any scene with a soft/fluid medium or
a rigid primitive builds through ``nuka.SceneBuilder`` (the general media cook).
Per-step choreography lives in ``control`` (it drives the world via the public
field I/O); camera math is in ``render``. No cook, stepping, or trace math is
re-implemented in Python -- the facade only assembles kwargs and drives controls.

Cloth drape over a robot (the fast path)::

    import nuka
    from nuka.author import Scene, SimOptions, morphs, materials, surfaces

    s = Scene(SimOptions(dt=1/240, env_count=1, contact_family=1))
    s.add_entity(morphs.NKS("examples/scenes/go2.nks"))
    s.add_entity(morphs.Grid(55, 51, 0.024, origin=(0, 0, 0.90)),
                 materials.Cloth.XPBD(mass=0.012, friction=1.8,
                                      bend_alpha=0.09, iters=80),
                 surfaces.Cloth(free=True, aero=(30.0, 0.12, 0.16)),
                 contact_radius=0.022)
    with nuka.Device.create(0) as dev:
        world = s.build(dev)

Soft/fluid media on a ground (the general builder path)::

    s = Scene(SimOptions(dt=1/240, env_count=1))
    s.add_entity(morphs.Ground(), materials.Rigid(friction=0.8))
    s.add_entity(morphs.TetSphere((0, 0, 0.3), 0.10, 8),
                 materials.Soft.XPBD(volume_alpha=1.5e-6, iters=40))
    with nuka.Device.create(0) as dev:
        world = s.build(dev)
"""

from __future__ import annotations

from . import control, materials, morphs, render, surfaces  # noqa: F401
from .scene import Scene, SimOptions  # noqa: F401

__all__ = [
    "Scene",
    "SimOptions",
    "morphs",
    "materials",
    "surfaces",
    "control",
    "render",
]
