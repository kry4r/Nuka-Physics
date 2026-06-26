"""Geometry-only morphs for the authoring facade.

A morph carries GEOMETRY, never physics behavior. ``NKS`` names a cooked scene
file (the robot/world); the media lattices are ``Grid`` (cloth), ``TetSphere``
(solid soft body) and ``FluidBox``; the rigid primitives are ``Box`` / ``Sphere``
/ ``Capsule`` / ``Plane`` / ``Ground``. The constitutive material + the surface
are attached at ``Scene.add_entity`` and the per-step choreography lives in
``nuka.author.control``. A media morph exposes the geometry block of
``SceneBuilder.add_media`` (``media_geometry_kwargs``); a rigid morph exposes the
geometry block of ``SceneBuilder.add_rigid_primitive``
(``primitive_geometry_kwargs``). No cook or sampling math runs here.
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

    MEDIA_KIND = "cloth"

    nx: int
    ny: int
    spacing: float
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def media_geometry_kwargs(self) -> dict:
        """The cloth geometry block for ``SceneBuilder.add_media`` (used when the
        scene also carries a primitive/tet/fluid, so it routes through the
        builder); the one-cloth fast path reads the fields directly instead."""
        return dict(cloth_nx=int(self.nx), cloth_ny=int(self.ny),
                    cloth_spacing=float(self.spacing),
                    cloth_origin=[float(c) for c in self.origin])

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


@_dc.dataclass(frozen=True)
class TetSphere:
    """A solid sphere soft body: a tet rest-lattice of ``cells`` per axis spanning
    ``radius`` at ``center``, each grid cell ``cell_len`` m (``0`` => a snug
    ``2*radius/cells``). Pair with a ``materials.Soft`` (XPBD or MLS-MPM)."""

    MEDIA_KIND = "soft_tet"

    center: Tuple[float, float, float]
    radius: float
    cells: int
    cell_len: float = 0.0

    def media_geometry_kwargs(self) -> dict:
        cl = (float(self.cell_len) if self.cell_len > 0.0
              else 2.0 * float(self.radius) / int(self.cells))
        return dict(tet_center=[float(c) for c in self.center],
                    tet_radius=float(self.radius), tet_cells=int(self.cells),
                    tet_cell_len=cl)


@_dc.dataclass(frozen=True)
class FluidBox:
    """A fluid volume: the AABB ``[min, max]`` sampled on a uniform ``spacing``
    lattice. Pair with a ``materials.Fluid`` (PBF or MLS-MPM)."""

    MEDIA_KIND = "fluid"

    min: Tuple[float, float, float]
    max: Tuple[float, float, float]
    spacing: float

    def media_geometry_kwargs(self) -> dict:
        return dict(fluid_min=[float(c) for c in self.min],
                    fluid_max=[float(c) for c in self.max],
                    fluid_spacing=float(self.spacing))


@_dc.dataclass(frozen=True)
class Box:
    """A rigid box of ``half_extents`` (x,y,z), at ``pos`` with orientation ``quat``
    (w,x,y,z). Movable/static + friction/mass come from a ``materials.Rigid``."""

    PRIMITIVE = "box"
    FORCE_STATIC = False

    half_extents: Tuple[float, float, float]
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def primitive_geometry_kwargs(self) -> dict:
        return dict(dims=[float(x) for x in self.half_extents],
                    pos=[float(c) for c in self.pos],
                    quat=[float(q) for q in self.quat])


@_dc.dataclass(frozen=True)
class Sphere:
    """A rigid sphere of ``radius`` at ``pos`` (rotation-invariant)."""

    PRIMITIVE = "sphere"
    FORCE_STATIC = False

    radius: float
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def primitive_geometry_kwargs(self) -> dict:
        return dict(dims=[float(self.radius)], pos=[float(c) for c in self.pos],
                    quat=[])


@_dc.dataclass(frozen=True)
class Capsule:
    """A rigid capsule of ``radius`` + ``half_height`` (local Z axis), at ``pos``
    with orientation ``quat`` (w,x,y,z)."""

    PRIMITIVE = "capsule"
    FORCE_STATIC = False

    radius: float
    half_height: float
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def primitive_geometry_kwargs(self) -> dict:
        return dict(dims=[float(self.radius), float(self.half_height)],
                    pos=[float(c) for c in self.pos],
                    quat=[float(q) for q in self.quat])


@_dc.dataclass(frozen=True)
class Plane:
    """An infinite static ground plane (+Z normal) at ``pos``."""

    PRIMITIVE = "plane"
    FORCE_STATIC = True

    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def primitive_geometry_kwargs(self) -> dict:
        return dict(dims=[], pos=[float(c) for c in self.pos], quat=[])


@_dc.dataclass(frozen=True)
class Ground:
    """The world floor: a static +Z plane at the origin (``Plane`` at ``(0,0,0)``)."""

    PRIMITIVE = "plane"
    FORCE_STATIC = True

    def primitive_geometry_kwargs(self) -> dict:
        return dict(dims=[], pos=[0.0, 0.0, 0.0], quat=[])


__all__ = [
    "NKS", "Grid", "TetSphere", "FluidBox",
    "Box", "Sphere", "Capsule", "Plane", "Ground",
]
