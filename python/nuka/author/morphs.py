"""Geometry-only morphs for the authoring facade.

A morph carries GEOMETRY, never physics behavior: ``NKS`` names a cooked scene
file (the robot/world), ``Grid`` is a flat cloth lattice. The constitutive
material + the surface/membrane are attached at ``Scene.add_entity``; the
per-step choreography lives in ``nuka.author.control``. These map directly onto
``World.create_coupled_from_scene`` kwargs -- no cook math runs here.
"""

from __future__ import annotations

import dataclasses as _dc
from typing import Tuple

import numpy as _np


@_dc.dataclass(frozen=True)
class NKS:
    """A cooked scene file (``.nks`` / ``.xml`` / ``.urdf`` / ``.usd``).

    Maps to ``create_coupled_from_scene(scene_path=path)`` -- the robot/world the
    media couple to. ``.nks`` resolves its imports internally.
    """

    path: str


@_dc.dataclass(frozen=True)
class Grid:
    """A flat ``nx`` x ``ny`` cloth lattice at ``spacing`` m, centred on ``origin``.

    Geometry only -- the XPBD params come from a ``materials.Cloth`` and the
    membrane/aero from a ``surfaces.Cloth``. Maps to the
    ``cloth_nx/ny/spacing/origin_*`` cook kwargs.
    """

    nx: int
    ny: int
    spacing: float
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def rest_positions(self) -> "_np.ndarray":
        """The ``(nx*ny, 3)`` flat rest lattice in the engine's row-major
        ``j*nx+i`` order (the cloth cook layout), centred on ``origin``.

        The control layer uploads these (with a park/release offset) into
        ``PARTICLE_POSITION``; index 0 is the ``(i=0, j=0)`` corner.
        """
        nx, ny, s = int(self.nx), int(self.ny), float(self.spacing)
        ox, oy, oz = (float(c) for c in self.origin)
        xs = (_np.arange(nx, dtype=_np.float32) - 0.5 * (nx - 1)) * s + ox
        ys = (_np.arange(ny, dtype=_np.float32) - 0.5 * (ny - 1)) * s + oy
        g = _np.zeros((ny, nx, 3), dtype=_np.float32)
        g[..., 0] = xs[None, :]
        g[..., 1] = ys[:, None]
        g[..., 2] = oz
        return g.reshape(-1, 3)


__all__ = ["NKS", "Grid"]
