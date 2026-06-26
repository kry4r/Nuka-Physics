"""Constitutive materials for the authoring facade.

A material is a parameter bag that also SELECTS the solver method for its medium:
``Cloth.XPBD`` (cloth, XPBD), ``Soft.XPBD`` / ``Soft.MPM`` (tet soft body),
``Fluid.PBF`` / ``Fluid.MPM`` (fluid), and ``Rigid`` (a rigid primitive). Each
maps onto the ``SceneBuilder.add_media`` / ``add_rigid_primitive`` kwargs through a
small ``*_kwargs`` method (mirroring ``ClothMaterial.cook_kwargs``) and declares
the medium ``MEDIA_KIND`` it solves + the ``MEDIA_METHOD`` it uses, so the scene
assembler can reject a morph paired with an incompatible material. No cook math
runs here -- a material only carries parameters.
"""

from __future__ import annotations

import dataclasses as _dc
from typing import Optional, Tuple


@_dc.dataclass(frozen=True)
class ClothMaterial:
    """XPBD cloth parameters -> the cloth_* cook kwargs (fast path) or the xpbd_*
    media block (builder path; the aero/free surface adds the rest)."""

    MEDIA_KIND = "cloth"
    MEDIA_METHOD = "xpbd"

    mass: float
    friction: float
    bend_alpha: float
    iters: int

    def cook_kwargs(self) -> dict:
        return dict(
            cloth_particle_mass=float(self.mass),
            cloth_friction=float(self.friction),
            cloth_bend_alpha=float(self.bend_alpha),
            cloth_iters=int(self.iters),
        )

    def media_material_kwargs(self) -> dict:
        return dict(
            xpbd_particle_mass=float(self.mass),
            xpbd_friction=float(self.friction),
            xpbd_bend_alpha=float(self.bend_alpha),
            xpbd_iters=int(self.iters),
        )


class Cloth:
    """Cloth constitutive models. ``Cloth.XPBD(...)`` builds an XPBD material."""

    @staticmethod
    def XPBD(mass: float = 0.01, friction: float = 0.6,
             bend_alpha: float = 1.0e-4, iters: int = 24) -> ClothMaterial:
        """An XPBD cloth membrane: per-particle ``mass`` kg, Coulomb ``friction``
        (the cone bound is mu*normal-lambda, mu>1 honoured), ``bend_alpha`` bend
        compliance (higher = softer folds), and ``iters`` solver iterations."""
        if iters < 1:
            raise ValueError("Cloth.XPBD: iters must be >= 1")
        return ClothMaterial(mass, friction, bend_alpha, iters)


@_dc.dataclass(frozen=True)
class SoftXpbdMaterial:
    """XPBD tet-soft parameters -> the xpbd_* media block (distance + volume)."""

    MEDIA_KIND = "soft_tet"
    MEDIA_METHOD = "xpbd"

    mass: float
    friction: float
    distance_alpha: float
    volume_alpha: float
    iters: int

    def media_material_kwargs(self) -> dict:
        return dict(
            xpbd_particle_mass=float(self.mass),
            xpbd_friction=float(self.friction),
            xpbd_distance_alpha=float(self.distance_alpha),
            xpbd_volume_alpha=float(self.volume_alpha),
            xpbd_iters=int(self.iters),
        )


@_dc.dataclass(frozen=True)
class MpmMaterial:
    """MLS-MPM parameters -> the mpm_* media block. ``model_kind`` 0 corotated,
    2 neo-hookean, 3 weakly-compressible fluid; ``MEDIA_KIND`` (soft_tet / fluid)
    is set by the factory so the scene assembler can match it to the morph."""

    MEDIA_METHOD = "mlsmpm"

    media_kind: str
    youngs: float
    poisson: float
    density: float
    model_kind: float
    bulk_modulus: float
    tait_gamma: float
    viscosity: float
    dx: float
    substeps: int
    floor_normal: Tuple[float, float, float]
    floor_d: float
    floor_friction: float

    @property
    def MEDIA_KIND(self) -> str:
        return self.media_kind

    def media_material_kwargs(self) -> dict:
        return dict(
            mpm_youngs=float(self.youngs), mpm_poisson=float(self.poisson),
            mpm_density=float(self.density), mpm_model_kind=float(self.model_kind),
            mpm_bulk_modulus=float(self.bulk_modulus),
            mpm_tait_gamma=float(self.tait_gamma),
            mpm_viscosity=float(self.viscosity), mpm_dx=float(self.dx),
            mpm_substeps=int(self.substeps),
            mpm_floor_normal=[float(c) for c in self.floor_normal],
            mpm_floor_d=float(self.floor_d),
            mpm_floor_friction=float(self.floor_friction),
        )


class Soft:
    """Tet-soft constitutive models: ``Soft.XPBD`` (distance+volume) or
    ``Soft.MPM`` (a continuum solid)."""

    @staticmethod
    def XPBD(mass: float = 0.01, friction: float = 0.5,
             distance_alpha: float = 0.0, volume_alpha: float = 1.5e-6,
             iters: int = 40) -> SoftXpbdMaterial:
        """An XPBD tet soft body: per-vertex ``mass`` kg, contact ``friction``,
        ``distance_alpha`` / ``volume_alpha`` edge + tet-volume compliance (higher
        = softer), and ``iters`` solver iterations."""
        if iters < 1:
            raise ValueError("Soft.XPBD: iters must be >= 1")
        return SoftXpbdMaterial(mass, friction, distance_alpha, volume_alpha, iters)

    @staticmethod
    def MPM(youngs: float = 1.0e5, poisson: float = 0.3, density: float = 1000.0,
            model_kind: float = 0.0, bulk_modulus: float = 0.0,
            tait_gamma: float = 0.0, viscosity: float = 0.0, dx: float = 0.02,
            substeps: int = 20,
            floor_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
            floor_d: float = 0.0, floor_friction: float = 0.4) -> MpmMaterial:
        """An MLS-MPM continuum solid: ``model_kind`` 0 corotated / 2 neo-hookean,
        ``youngs`` / ``poisson`` elasticity, ``density`` kg/m^3, grid cell ``dx`` m,
        ``substeps`` per step, and a separating floor at ``floor_normal . x =
        floor_d``. ``dx`` must be > 0 for the background grid."""
        if dx <= 0.0:
            raise ValueError("Soft.MPM: dx must be > 0 (the background grid cell)")
        if substeps < 1:
            raise ValueError("Soft.MPM: substeps must be >= 1")
        return MpmMaterial("soft_tet", youngs, poisson, density, model_kind,
                           bulk_modulus, tait_gamma, viscosity, dx, substeps,
                           floor_normal, floor_d, floor_friction)


@_dc.dataclass(frozen=True)
class FluidPbfMaterial:
    """Position-based-fluid parameters -> the pbf_* media block. ``walls`` is
    ``None`` (open domain) or a ``(min_xyz, max_xyz)`` confinement box."""

    MEDIA_KIND = "fluid"
    MEDIA_METHOD = "pbf"

    rest_density: float
    support_scale: float
    iters: int
    friction: float
    walls: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    clamp_overdensity: bool
    floor_z: float
    boundary_layers: int

    def media_material_kwargs(self) -> dict:
        kw = dict(
            pbf_rest_density=float(self.rest_density),
            pbf_support_scale=float(self.support_scale),
            pbf_iters=int(self.iters), pbf_friction=float(self.friction),
            pbf_clamp_overdensity=bool(self.clamp_overdensity),
            pbf_floor_z=float(self.floor_z),
        )
        if self.walls is not None:
            wmin, wmax = self.walls
            kw.update(pbf_walls_enabled=True,
                      pbf_walls_min=[float(c) for c in wmin],
                      pbf_walls_max=[float(c) for c in wmax],
                      pbf_boundary_layers=int(self.boundary_layers))
        return kw


class Fluid:
    """Fluid constitutive models: ``Fluid.PBF`` (incompressible PBD) or
    ``Fluid.MPM`` (a weakly-compressible MLS-MPM fluid)."""

    @staticmethod
    def PBF(rest_density: float = 1000.0, support_scale: float = 1.5,
            iters: int = 4, friction: float = 0.1, walls=None,
            clamp_overdensity: bool = True, floor_z: float = 0.0,
            boundary_layers: int = 2) -> FluidPbfMaterial:
        """A position-based fluid: ``rest_density`` kg/m^3, SPH support radius =
        ``support_scale`` * lattice spacing, ``iters`` density iterations, contact
        ``friction``. ``walls=(min_xyz, max_xyz)`` builds a pinned confinement
        container (with ``boundary_layers`` thickness); ``None`` is an open pool."""
        if iters < 1:
            raise ValueError("Fluid.PBF: iters must be >= 1")
        return FluidPbfMaterial(rest_density, support_scale, iters, friction,
                                walls, clamp_overdensity, floor_z, boundary_layers)

    @staticmethod
    def MPM(density: float = 1000.0, bulk_modulus: float = 2.0e5,
            tait_gamma: float = 7.0, viscosity: float = 0.4, dx: float = 0.011,
            substeps: int = 40,
            floor_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
            floor_d: float = 0.0, floor_friction: float = 0.0) -> MpmMaterial:
        """A weakly-compressible MLS-MPM fluid (``model_kind`` 3, Tait EOS):
        ``density`` rho0, ``bulk_modulus`` K (sound speed sqrt(K/rho0)),
        ``tait_gamma`` EOS exponent, ``viscosity`` damping, grid cell ``dx`` m,
        ``substeps`` per step, and a free-slip floor BC."""
        if dx <= 0.0:
            raise ValueError("Fluid.MPM: dx must be > 0 (the background grid cell)")
        if substeps < 1:
            raise ValueError("Fluid.MPM: substeps must be >= 1")
        return MpmMaterial("fluid", 0.0, 0.0, density, 3.0, bulk_modulus,
                           tait_gamma, viscosity, dx, substeps,
                           floor_normal, floor_d, floor_friction)


@_dc.dataclass(frozen=True)
class RigidMaterial:
    """A rigid primitive's physics bag -> the add_rigid_primitive kwargs.
    ``static`` pins the body (a ``Plane``/``Ground`` is always static); ``mass``
    kg applies when movable; ``friction`` < 0 inherits the material default."""

    static: bool = False
    mass: float = 1.0
    friction: float = -1.0

    def rigid_kwargs(self) -> dict:
        return dict(static=bool(self.static), mass=float(self.mass),
                    friction=float(self.friction))


def Rigid(friction: float = -1.0, mass: float = 1.0,
          static: bool = False) -> RigidMaterial:
    """A rigid primitive material: Coulomb ``friction`` (< 0 inherits the default),
    ``mass`` kg when movable, and ``static`` to pin it."""
    return RigidMaterial(static=static, mass=mass, friction=friction)


__all__ = [
    "Cloth", "ClothMaterial", "Soft", "SoftXpbdMaterial", "MpmMaterial",
    "Fluid", "FluidPbfMaterial", "Rigid", "RigidMaterial",
]
