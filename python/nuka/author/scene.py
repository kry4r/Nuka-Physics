"""The authoring-facade ``Scene``: assemble morph + material + surface entities,
then ``build(device)`` a coupled ``nk::World`` via ``create_coupled_from_scene``.

A THIN kwarg assembler. It does NOT re-implement the cloth cook, constraint
generation, stepping, or surface/trace math -- it maps the geometry-only morph +
the constitutive material + the surface onto the existing coupled-create binding.
An unsupported combination (e.g. a cloth ``Grid`` with a non-cloth material, or
two scene morphs) raises a LOUD exception, never a silent clamp.

This is the simulation assembler, distinct from the generic ``nuka.Scene`` scene
graph editor (load / compose / settle / save); reach it as ``nuka.author.Scene``.
"""

from __future__ import annotations

import dataclasses as _dc
from typing import List, Optional

import nuka as _nuka

from . import materials as _materials
from . import morphs as _morphs
from . import surfaces as _surfaces


@_dc.dataclass
class SimOptions:
    """World-level options shared by every entity: the fixed ``dt``, ``env_count``,
    the ``control_mode`` (0 PD-position, 1 torque), and the static collidable.

    ``contact_family == 1`` bakes a flat heightfield the robot stands on and the
    cloth hem pools onto (``heightfield_terrain_type`` selects its feature set,
    0 == flat). ``contact_radius`` (m) is the particle sphere radius shared by the
    coupled media (0 == half the medium's lattice spacing).

    The ``solver_*`` + ``baumgarte_max_velocity`` knobs override the world
    ``nk::Pipeline::SolverConfig`` (the contact solve a drape needs to conform);
    each 0 keeps the engine default, so a defaults-only world is byte-identical.
    """

    dt: float = 1.0 / 240.0
    env_count: int = 1
    control_mode: int = 0
    contact_family: int = 0
    heightfield_terrain_type: int = 0
    contact_radius: float = 0.0
    solver_vel_iters: int = 0
    solver_pos_iters: int = 0
    solver_contact_margin: float = 0.0
    solver_max_pairs: int = 0
    baumgarte_max_velocity: float = 0.0


@_dc.dataclass
class _Entity:
    morph: object
    material: Optional[object] = None
    surface: Optional[object] = None
    contact_radius: Optional[float] = None


class Scene:
    """Assemble a coupled world from authored entities, then ``build`` it."""

    def __init__(self, options: Optional[SimOptions] = None) -> None:
        self.options = options if options is not None else SimOptions()
        self._entities: List[_Entity] = []

    def add_entity(self, morph, material=None, surface=None, *,
                   contact_radius: Optional[float] = None) -> "Scene":
        """Add one authored entity (a ``morph`` plus its optional ``material`` /
        ``surface``). The robot is an ``NKS`` morph; a cloth sheet is a ``Grid``
        morph with a ``materials.Cloth`` + a ``surfaces.Cloth``. Returns ``self``.
        """
        self._entities.append(_Entity(morph, material, surface, contact_radius))
        return self

    def build(self, device) -> "_nuka.World":
        """Cook + build the coupled ``World`` (``create_coupled_from_scene``).

        Resolves the single scene (``NKS``) morph and the single cloth (``Grid``)
        morph into the create kwargs. Raises if the entity set is unsupported.
        """
        o = self.options
        kw = dict(
            device=device, env_count=int(o.env_count), dt=float(o.dt),
            control_mode=int(o.control_mode),
            contact_family=int(o.contact_family),
            heightfield_terrain_type=int(o.heightfield_terrain_type),
            solver_vel_iters=int(o.solver_vel_iters),
            solver_pos_iters=int(o.solver_pos_iters),
            solver_contact_margin=float(o.solver_contact_margin),
            solver_max_pairs=int(o.solver_max_pairs),
            baumgarte_max_velocity=float(o.baumgarte_max_velocity),
        )

        scenes = [e for e in self._entities if isinstance(e.morph, _morphs.NKS)]
        grids = [e for e in self._entities if isinstance(e.morph, _morphs.Grid)]
        unknown = [e for e in self._entities
                   if not isinstance(e.morph, (_morphs.NKS, _morphs.Grid))]
        if unknown:
            raise TypeError(
                "Scene.build: unsupported morph "
                f"{type(unknown[0].morph).__name__} (use morphs.NKS / morphs.Grid)")
        if len(scenes) != 1:
            raise ValueError(
                f"Scene.build needs exactly one NKS scene morph (got {len(scenes)})")
        kw["scene_path"] = str(scenes[0].morph.path)

        contact_radius = float(o.contact_radius)
        if len(grids) > 1:
            raise ValueError(
                f"Scene.build supports one cloth Grid morph (got {len(grids)})")
        if grids:
            e = grids[0]
            mat = e.material if e.material is not None else _materials.Cloth.XPBD()
            surf = e.surface if e.surface is not None else _surfaces.Cloth()
            if not isinstance(mat, _materials.ClothMaterial):
                raise TypeError(
                    "Scene.build: a Grid morph needs a Cloth material "
                    f"(materials.Cloth.XPBD); got {type(mat).__name__}")
            if not isinstance(surf, _surfaces.ClothSurface):
                raise TypeError(
                    "Scene.build: a Grid morph needs a Cloth surface "
                    f"(surfaces.Cloth); got {type(surf).__name__}")
            g = e.morph
            kw.update(
                cloth_nx=int(g.nx), cloth_ny=int(g.ny),
                cloth_spacing=float(g.spacing),
                cloth_origin_x=float(g.origin[0]),
                cloth_origin_y=float(g.origin[1]),
                cloth_origin_z=float(g.origin[2]),
            )
            kw.update(mat.cook_kwargs())
            kw.update(surf.cook_kwargs())
            if e.contact_radius is not None:
                contact_radius = float(e.contact_radius)
        kw["contact_radius"] = contact_radius
        return _nuka.World.create_coupled_from_scene(**kw)


__all__ = ["Scene", "SimOptions"]
