"""The authoring-facade ``Scene``: assemble morph + material + surface entities,
then ``build(device)`` a coupled ``nk::World``.

A THIN kwarg assembler. It does NOT re-implement any cook, constraint generation,
stepping, or surface/trace math -- it maps the geometry-only morph + the
constitutive material + the surface onto an existing create binding. ``build``
routes by the authored entity set:

* a scene of exactly one ``NKS`` + (optionally) one cloth ``Grid`` and nothing
  else takes the ``create_coupled_from_scene`` fast path;
* any scene that adds a ``TetSphere`` / ``FluidBox`` / ``GranularBed`` medium or
  a rigid primitive
  (``Box`` / ``Sphere`` / ``Capsule`` / ``Plane`` / ``Ground``) builds through
  ``nuka.SceneBuilder`` (base = the ``NKS`` path if present, else empty), one
  ``add_rigid_primitive`` / ``add_media`` per entity, then ``build``.

An unsupported combination (a morph paired with an incompatible material, two
scene morphs, an illegal medium x method) raises a LOUD exception, never a silent
clamp. This is the simulation assembler, distinct from the generic ``nuka.Scene``
scene-graph editor; reach it as ``nuka.author.Scene``.
"""

from __future__ import annotations

import dataclasses as _dc
from typing import List, Optional, Tuple

import nuka as _nuka

from . import materials as _materials
from . import morphs as _morphs
from . import surfaces as _surfaces

# String token -> binding constant (the assembler is the only place the builder
# codes are resolved; the param bags stay binding-free).
_MEDIA_KIND = {"cloth": _nuka.MEDIA_CLOTH, "soft_tet": _nuka.MEDIA_SOFT_TET,
               "fluid": _nuka.MEDIA_FLUID, "granular": _nuka.MEDIA_GRANULAR}
_MEDIA_METHOD = {"xpbd": _nuka.MEDIA_METHOD_XPBD, "pbf": _nuka.MEDIA_METHOD_PBF,
                 "mlsmpm": _nuka.MEDIA_METHOD_MLSMPM}
_PRIMITIVE = {"plane": _nuka.PRIMITIVE_PLANE, "box": _nuka.PRIMITIVE_BOX,
              "sphere": _nuka.PRIMITIVE_SPHERE, "capsule": _nuka.PRIMITIVE_CAPSULE}
_DEFAULT_MEDIA_MATERIAL = {"cloth": _materials.Cloth.XPBD,
                           "soft_tet": _materials.Soft.XPBD,
                           "fluid": _materials.Fluid.PBF,
                           "granular": _materials.Granular.MPM}


@_dc.dataclass
class SimOptions:
    """World-level options shared by every entity: the fixed ``dt``, ``env_count``,
    the ``control_mode`` (0 PD-position, 1 torque), and the static collidable.

    ``contact_family == 1`` bakes a flat heightfield the robot stands on and the
    cloth hem pools onto (``heightfield_terrain_type`` selects its feature set,
    0 == flat). ``gravity`` (m/s^2) is the world acceleration; all-zero is the
    engine default (standard Earth).

    The ``solver_*`` + ``baumgarte_max_velocity`` ``nk::Pipeline::SolverConfig``
    overrides apply on BOTH paths (the cloth fast path and the general
    builder path), so an authored soft body can request a tighter contact solve
    (more position-correction sweeps, a speculative contact margin). Each 0 keeps
    the engine default, so a defaults-only world is byte-identical. The
    ``contact_radius`` (m) particle sphere radius applies on the cloth fast path
    only; the general builder path derives the particle radius from the lattice
    spacing, so setting ``contact_radius`` there raises.
    """

    dt: float = 1.0 / 240.0
    env_count: int = 1
    control_mode: int = 0
    contact_family: int = 0
    heightfield_terrain_type: int = 0
    contact_radius: float = 0.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
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
    render: Optional[object] = None  # materials.RenderAppearance


@_dc.dataclass(frozen=True)
class Environment:
    """The scene's beauty backdrop: an equirect ``hdri`` path lighting the sky +
    ambient (``""`` keeps the procedural studio sky), its ``yaw_deg`` /
    ``intensity``, and ``use_scene_materials`` (render with the AUTHORED
    materials instead of the studio hero palette)."""

    hdri: str = ""
    yaw_deg: float = 0.0
    intensity: float = 1.0
    use_scene_materials: bool = True


_MEDIA_MORPHS = (_morphs.Grid, _morphs.TetSphere, _morphs.FluidBox,
                 _morphs.GranularBed)
_RIGID_MORPHS = (_morphs.Box, _morphs.Sphere, _morphs.Capsule, _morphs.Plane,
                 _morphs.Ground)


class Scene:
    """Assemble a coupled world from authored entities, then ``build`` it."""

    def __init__(self, options: Optional[SimOptions] = None) -> None:
        self.options = options if options is not None else SimOptions()
        self._entities: List[_Entity] = []
        self._environment: Optional[Environment] = None

    def add_entity(self, morph, material=None, surface=None, *,
                   contact_radius: Optional[float] = None,
                   render=None) -> "Scene":
        """Add one authored entity (a ``morph`` plus its optional ``material`` /
        ``surface``). The robot is an ``NKS`` morph; a cloth/tet/fluid medium is a
        ``Grid`` / ``TetSphere`` / ``FluidBox`` with its ``materials`` +
        ``surfaces``; a rigid primitive is a ``Box`` / ``Sphere`` / ``Capsule`` /
        ``Plane`` / ``Ground`` with a ``materials.Rigid``. ``render`` binds a
        ``materials.Render`` appearance (flat PBR + image maps) to the entity's
        beauty look. Returns ``self``.
        """
        if render is not None and not isinstance(render, _materials.RenderAppearance):
            raise TypeError(
                "Scene.add_entity: render must be a materials.Render appearance; "
                f"got {type(render).__name__}")
        self._entities.append(_Entity(morph, material, surface, contact_radius,
                                      render))
        return self

    def set_environment(self, hdri: str = "", yaw_deg: float = 0.0,
                        intensity: float = 1.0,
                        use_scene_materials: bool = True) -> "Scene":
        """Set the beauty environment: an equirect ``.hdr`` path backing the sky +
        ambient light (``""`` keeps the procedural studio sky), its rotation /
        intensity, and the authored-materials render policy. Returns ``self``."""
        self._environment = Environment(hdri, float(yaw_deg), float(intensity),
                                        bool(use_scene_materials))
        return self

    def build(self, device) -> "_nuka.World":
        """Cook + build the coupled ``World``. Routes the cloth fast path
        (``create_coupled_from_scene``) or the general ``SceneBuilder`` path by the
        authored entity set. Raises if the entity set is unsupported.
        """
        scenes = [e for e in self._entities if isinstance(e.morph, _morphs.NKS)]
        grids = [e for e in self._entities if isinstance(e.morph, _morphs.Grid)]
        tets = [e for e in self._entities if isinstance(e.morph, _morphs.TetSphere)]
        fluids = [e for e in self._entities if isinstance(e.morph, _morphs.FluidBox)]
        grans = [e for e in self._entities
                 if isinstance(e.morph, _morphs.GranularBed)]
        rigids = [e for e in self._entities
                  if isinstance(e.morph, _RIGID_MORPHS)]
        unknown = [e for e in self._entities
                   if not isinstance(e.morph, (_morphs.NKS,) + _MEDIA_MORPHS
                                     + _RIGID_MORPHS)]
        if unknown:
            raise TypeError(
                "Scene.build: unsupported morph "
                f"{type(unknown[0].morph).__name__} (use morphs.NKS / Grid / "
                "TetSphere / FluidBox / GranularBed / Box / Sphere / Capsule / "
                "Plane / Ground)")
        # Any authored appearance / environment routes to the general builder path
        # (the SceneBuilder carries the material + environment records).
        appearance = self._environment is not None or any(
            e.render is not None for e in self._entities)
        if tets or fluids or grans or rigids or appearance:
            return self._build_general(device, scenes, grids, tets, fluids + grans,
                                       rigids)
        return self._build_coupled(device, scenes, grids)

    # -- the cloth fast path (one NKS + optional one cloth Grid) --------------
    def _build_coupled(self, device, scenes, grids) -> "_nuka.World":
        o = self.options
        if any(float(g) != 0.0 for g in o.gravity):
            raise ValueError(
                "Scene.build: the cloth fast path runs at the engine default "
                "(Earth) gravity; author a Ground/Plane + media to set gravity.")
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

    # -- the general builder path (tet/fluid media or a rigid primitive) ------
    def _build_general(self, device, scenes, grids, tets, fluids,
                       rigids) -> "_nuka.World":
        o = self.options
        media = grids + tets + fluids
        self._reject_contact_radius(media)
        if len(scenes) > 1:
            raise ValueError(
                f"Scene.build needs at most one NKS base scene (got {len(scenes)})")
        base_path = str(scenes[0].morph.path) if scenes else ""

        builder = _nuka.SceneBuilder.create(base_path)
        try:
            # Authored render materials first (rigids bind by name, media by id).
            material_ids = {}
            for e in self._entities:
                r = e.render
                if r is not None and r.name not in material_ids:
                    material_ids[r.name] = builder.add_material(
                        **r.add_material_kwargs())
            if self._environment is not None:
                env = self._environment
                import os
                builder.set_environment(
                    hdri=os.path.abspath(env.hdri) if env.hdri else "",
                    yaw_deg=env.yaw_deg, intensity=env.intensity,
                    use_scene_materials=env.use_scene_materials)
            for e in rigids:
                self._add_rigid(builder, e)
            for e in media:
                self._add_media(builder, e, material_ids)
            gx, gy, gz = (float(c) for c in o.gravity)
            world = builder.build(
                device, env_count=int(o.env_count), dt=float(o.dt),
                control_mode=int(o.control_mode),
                contact_family=int(o.contact_family),
                heightfield_terrain_type=int(o.heightfield_terrain_type),
                gravity_x=gx, gravity_y=gy, gravity_z=gz,
                solver_vel_iters=int(o.solver_vel_iters),
                solver_pos_iters=int(o.solver_pos_iters),
                solver_contact_margin=float(o.solver_contact_margin),
                solver_max_pairs=int(o.solver_max_pairs),
                baumgarte_max_velocity=float(o.baumgarte_max_velocity))
        finally:
            builder.destroy()
        return world

    def _reject_contact_radius(self, media) -> None:
        o = self.options
        if float(o.contact_radius) != 0.0 or any(
                e.contact_radius is not None for e in media):
            raise ValueError(
                "Scene.build: contact_radius applies only on the cloth fast path; "
                "the general media path derives the particle radius from the "
                "lattice spacing.")

    def _add_rigid(self, builder, e) -> None:
        morph = e.morph
        mat = e.material if e.material is not None else _materials.Rigid()
        if not isinstance(mat, _materials.RigidMaterial):
            raise TypeError(
                f"Scene.build: a {type(morph).__name__} primitive needs a "
                f"materials.Rigid material; got {type(mat).__name__}")
        if e.surface is not None:
            raise TypeError(
                f"Scene.build: a {type(morph).__name__} primitive takes no surface")
        kw = morph.primitive_geometry_kwargs()
        kw.update(mat.rigid_kwargs())
        if morph.FORCE_STATIC:
            kw["static"] = True
        if e.render is not None:
            kw["material"] = e.render.name
        builder.add_rigid_primitive(kind=_PRIMITIVE[morph.PRIMITIVE], **kw)

    def _add_media(self, builder, e, material_ids=None) -> None:
        morph = e.morph
        kind = morph.MEDIA_KIND
        mat = (e.material if e.material is not None
               else _DEFAULT_MEDIA_MATERIAL[kind]())
        mat_kind = getattr(mat, "MEDIA_KIND", None)
        if mat_kind != kind:
            raise TypeError(
                f"Scene.build: a {type(morph).__name__} ({kind}) medium needs a "
                f"matching material; got {type(mat).__name__} ({mat_kind}).")
        kw = morph.media_geometry_kwargs()
        kw.update(mat.media_material_kwargs())
        surf = e.surface
        if surf is not None:
            if kind == "granular":
                raise TypeError(
                    "Scene.build: a granular medium renders as particles and "
                    "takes no surface")
            ok = (isinstance(surf, _surfaces.ClothSurface) if kind == "cloth"
                  else isinstance(surf, _surfaces.SoftSurface))
            if not ok:
                raise TypeError(
                    f"Scene.build: a {kind} medium got an incompatible surface "
                    f"{type(surf).__name__}")
            kw.update(surf.media_kwargs())
        if e.render is not None and material_ids:
            kw["render_material_id"] = material_ids[e.render.name]
        builder.add_media(kind=_MEDIA_KIND[kind],
                          method=_MEDIA_METHOD[mat.MEDIA_METHOD], **kw)


__all__ = ["Scene", "SimOptions", "Environment"]
