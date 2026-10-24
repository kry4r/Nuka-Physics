"""Constitutive materials for the authoring facade.

Materials declare constitutive parameters and compatible discretizations for
``SceneBuilder.add_media`` / ``add_rigid_primitive``. ``MEDIA_KIND`` and
``MEDIA_METHOD`` let the assembler validate geometry compatibility; all media
join the shared dynamics and contact solve. Cooking runs in the engine.
"""

from __future__ import annotations

import dataclasses as _dc
import math as _math
from typing import Tuple


@_dc.dataclass(frozen=True)
class ClothMaterial:
    """Vertex-block cloth parameters -> the cloth_* cook kwargs (fast path) or the
    vbd_* media block (builder path; the aero/free surface adds the rest)."""

    MEDIA_KIND = "cloth"
    MEDIA_METHOD = "vbd"

    mass: float
    friction: float
    stretch_stiffness: float
    poisson: float
    bend_stiffness: float
    damping: float
    half_thickness: float = 0.0
    surface_density: float = 0.0

    def cook_kwargs(self) -> dict:
        return dict(
            cloth_particle_mass=float(self.mass),
            cloth_friction=float(self.friction),
            cloth_stretch_stiffness=float(self.stretch_stiffness),
            cloth_bend_stiffness=float(self.bend_stiffness),
            cloth_poisson=float(self.poisson),
            cloth_damping=float(self.damping),
        )

    def media_material_kwargs(self) -> dict:
        return dict(
            vbd_particle_mass=float(self.mass),
            vbd_friction=float(self.friction),
            vbd_stretch_stiffness=float(self.stretch_stiffness),
            vbd_poisson=float(self.poisson),
            vbd_bend_stiffness=float(self.bend_stiffness),
            vbd_damping=float(self.damping),
            vbd_half_thickness=float(self.half_thickness),
            vbd_surface_density=float(self.surface_density),
        )


@_dc.dataclass(frozen=True)
class FieldMaterial:
    """Synthetic shell material: the small-strain adapter (law 1, valid through 5% surface strain) or the
    Lagrangian log-strain orthotropic Hill law (law 2, valid through ``strain_range``). The compatibility
    encoding contains 25 named material scalars and contact friction; law 2 appends its code and 6 scalars, the
    mixed-mode fracture law (2) appends the law block, its code and 2 scalars, and the effective section (2) appends
    the law and fracture blocks, its code and 2 scalars."""

    MEDIA_KIND = "cloth"
    MEDIA_METHOD = "field"

    density: float
    thickness: float
    youngs: float
    poisson: float
    friction: float = 0.6
    yield_stress: float = 0.0
    hardening: float = 0.0
    yield_moment: float = 0.0
    bend_hardening: float = 0.0
    maxwell_modulus: Tuple[float, float] = (0.0, 0.0)
    maxwell_time: Tuple[float, float] = (0.0, 0.0)
    bend_maxwell_modulus: Tuple[float, float] = (0.0, 0.0)
    bend_maxwell_time: Tuple[float, float] = (0.0, 0.0)
    tear_strength: float = 0.0
    tear_energy: float = 0.0
    weibull_modulus: float = 0.0
    grain: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    grain_ratio: float = 0.0
    shear_strength: float = 0.0
    weibull_reference_area: float = 0.0
    schema_version: int = 1
    law: int = 1
    youngs_cd: float = 0.0
    shear_modulus: float = 0.0
    yield_stress_cd: float = 0.0
    yield_shear: float = 0.0
    yield_biaxial: float = 0.0
    strain_range: float = 0.0
    fracture_law: int = 1
    shear_tear_energy: float = 0.0
    mixity_exponent: float = 0.0
    section_law: int = 1
    bend_stiffness: float = 0.0
    curvature_range: float = 0.0

    def media_material_kwargs(self) -> dict:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(f"unsupported field material schema version: {self.schema_version}")
        values = (self.density, self.thickness, self.youngs, self.poisson, self.yield_stress,
                  self.hardening, self.yield_moment, self.bend_hardening, *self.maxwell_modulus,
                  *self.maxwell_time, *self.bend_maxwell_modulus, *self.bend_maxwell_time,
                  self.tear_strength, self.tear_energy, self.weibull_modulus, *self.grain, self.grain_ratio,
                  self.shear_strength, self.weibull_reference_area)
        if self.law == 2 or self.fracture_law == 2 or self.section_law == 2:
            values += (float(self.law), self.youngs_cd, self.shear_modulus, self.yield_stress_cd, self.yield_shear,
                       self.yield_biaxial, self.strain_range)
        if self.law not in (1, 2):
            raise ValueError(f"unsupported field material law: {self.law}")
        if self.fracture_law == 2:
            values += (2.0, self.shear_tear_energy, self.mixity_exponent)
        elif self.fracture_law != 1:
            raise ValueError(f"unsupported field fracture law: {self.fracture_law}")
        elif self.section_law == 2:
            values += (1.0, 0.0, 0.0)
        if self.section_law == 2:
            values += (2.0, self.bend_stiffness, self.curvature_range)
        elif self.section_law != 1:
            raise ValueError(f"unsupported field section law: {self.section_law}")
        return dict(vbd_friction=float(self.friction), field_material=[float(v) for v in values],
                    field_contract_version=self.schema_version)


class Cloth:
    """Cloth constitutive models. ``Cloth.VBD(...)`` builds a vertex-block shell,
    ``Cloth.Field(...)`` a material-field shell."""

    @staticmethod
    def VBD(mass: float = 0.01, friction: float = 0.6,
            stretch_stiffness: float = 5.0e3, poisson: float = 0.3,
            bend_stiffness: float = 5.0e-5, damping: float = 0.0,
            thickness: float = 0.0, areal_density: float = 0.0) -> ClothMaterial:
        """StVK membrane with dihedral bending and optional areal density (kg/m²).
        ``thickness`` sets the contact shell; ``mass`` is per vertex when density is zero."""
        if not stretch_stiffness > 0.0:
            raise ValueError("Cloth.VBD: stretch_stiffness must be > 0")
        if not 0.0 <= poisson < 0.5:
            raise ValueError("Cloth.VBD: poisson must be in [0, 0.5)")
        if not _math.isfinite(thickness) or thickness < 0.0:
            raise ValueError("Cloth.VBD: thickness must be finite and nonnegative")
        if not _math.isfinite(areal_density) or areal_density < 0.0:
            raise ValueError("Cloth.VBD: areal_density must be finite and nonnegative")
        if areal_density > 0.0:
            mass = 0.0
        return ClothMaterial(mass, friction, stretch_stiffness, poisson,
                             bend_stiffness, damping, thickness * 0.5, areal_density)

    @staticmethod
    def Field(density: float = 1.0e3, thickness: float = 5.0e-4, youngs: float = 1.0e6,
              poisson: float = 0.3, friction: float = 0.6, yield_stress: float = 0.0,
              hardening: float = 0.0, yield_moment: float = 0.0, bend_hardening: float = 0.0,
              maxwell_modulus: Tuple[float, float] = (0.0, 0.0),
              maxwell_time: Tuple[float, float] = (0.0, 0.0),
              bend_maxwell_modulus: Tuple[float, float] = (0.0, 0.0),
              bend_maxwell_time: Tuple[float, float] = (0.0, 0.0),
              tear_strength: float = 0.0, tear_energy: float = 0.0,
              weibull_modulus: float = 0.0, grain: Tuple[float, float, float] = (0.0, 0.0, 0.0),
              grain_ratio: float = 0.0, shear_strength: float = 0.0,
              weibull_reference_area: float = 0.0, law: str = "small_strain",
              youngs_cd: float = 0.0, shear_modulus: float = 0.0, yield_stress_cd: float = 0.0,
              yield_shear: float = 0.0, yield_biaxial: float = 0.0,
              strain_range: float = 0.0, fracture_law: str = "isotropic", shear_tear_energy: float = 0.0,
              mixity_exponent: float = 0.0, bend_stiffness: float = 0.0,
              curvature_range: float = 0.0) -> FieldMaterial:
        """Plane-stress membrane (kg/m³, m, Pa) with plate bending D = Y t³ / (12 (1 - ν²)),
        optional membrane and hinge plasticity and two Maxwell branches each; seams part at
        ``tear_strength`` (Pa) after spending ``tear_energy`` (J/m²). A positive ``weibull_modulus``
        lets any edge crack at its own strength, ``tear_strength`` scattered with that Weibull modulus.
        A positive ``grain_ratio`` (>= 1) makes a crack across ``grain``, the paper's machine direction,
        that many times stronger than one along it.

        ``law="log_strain_orthotropic"`` measures membrane strain as (1/2) ln C in the grain frame: ``youngs``
        and ``poisson`` act along ``grain`` (nonzero), ``youngs_cd`` across it and ``shear_modulus`` in shear,
        with Hill yield at ``yield_stress`` along, ``yield_stress_cd`` across, ``yield_shear`` in shear and
        ``yield_biaxial`` equibiaxially (zeros take the isotropic values); plates bend with D = Q t³ / 12.

        ``fracture_law="mixed_mode"`` parts seams in opening at ``tear_strength`` after ``tear_energy`` and in
        sliding at ``shear_strength`` after ``shear_tear_energy`` (zeros take the opening values), with toughness
        G_I + (G_S - G_I) b^``mixity_exponent`` (zero: 1) at sliding energy share b; compression does not damage.

        A positive ``curvature_range`` (1/m) makes the plate an effective section that declares the largest principal
        curvature its moment law holds for (zero: surface strain 5%); its ``bend_stiffness`` (N m along the grain,
        zero: the plate value) replaces the plate stiffness."""
        if not (density > 0.0 and thickness > 0.0 and youngs > 0.0):
            raise ValueError("Cloth.Field: density, thickness and youngs must be > 0")
        if law not in ("small_strain", "log_strain_orthotropic"):
            raise ValueError("Cloth.Field: law must be 'small_strain' or 'log_strain_orthotropic'")
        if fracture_law not in ("isotropic", "mixed_mode"):
            raise ValueError("Cloth.Field: fracture_law must be 'isotropic' or 'mixed_mode'")
        mixed = fracture_law == "mixed_mode"
        if not all(_math.isfinite(v) and v >= 0.0 for v in (shear_tear_energy, mixity_exponent)) or \
                (not mixed and (shear_tear_energy or mixity_exponent)):
            raise ValueError("Cloth.Field: shear_tear_energy and mixity_exponent must be finite, nonnegative and only "
                             "set with the mixed-mode fracture law")
        if not all(_math.isfinite(v) and v >= 0.0 for v in (bend_stiffness, curvature_range)) or \
                (bend_stiffness > 0.0 and not curvature_range > 0.0):
            raise ValueError("Cloth.Field: bend_stiffness and curvature_range must be finite and nonnegative, and an "
                             "effective section must declare its curvature_range")
        orthotropic = law == "log_strain_orthotropic"
        if not orthotropic and not 0.0 <= poisson < 0.5:
            raise ValueError("Cloth.Field: poisson must be in [0, 0.5)")
        extra = (youngs_cd, shear_modulus, yield_stress_cd, yield_shear, yield_biaxial, strain_range)
        if not all(_math.isfinite(v) and v >= 0.0 for v in extra) or \
                (not orthotropic and any(extra)):
            raise ValueError("Cloth.Field: orthotropic parameters must be finite, nonnegative and only set "
                             "with the log-strain law")
        if orthotropic and not (youngs_cd > 0.0 and shear_modulus > 0.0 and strain_range > 0.0 and
                                poisson >= 0.0 and poisson * poisson * youngs_cd < youngs and any(grain)):
            raise ValueError("Cloth.Field: the log-strain law needs positive youngs_cd, shear_modulus and "
                             "strain_range, 0 <= poisson with poisson² youngs_cd < youngs, and a nonzero grain")
        rest = (yield_stress, hardening, yield_moment, bend_hardening, *maxwell_modulus,
                *maxwell_time, *bend_maxwell_modulus, *bend_maxwell_time, tear_strength, tear_energy,
                weibull_modulus, grain_ratio, shear_strength, weibull_reference_area)
        if len(rest) != 18 or not all(_math.isfinite(v) and v >= 0.0 for v in rest):
            raise ValueError("Cloth.Field: plastic, Maxwell, tear, Weibull and grain-ratio parameters must be "
                             "finite and nonnegative")
        if len(grain) != 3 or not all(_math.isfinite(g) for g in grain) or \
                (grain_ratio > 0.0 and not (grain_ratio >= 1.0 and any(grain))):
            raise ValueError("Cloth.Field: grain must be 3 finite floats, nonzero under a grain_ratio of at least 1")
        return FieldMaterial(float(density), float(thickness), float(youngs), float(poisson),
                             float(friction), float(yield_stress), float(hardening),
                             float(yield_moment), float(bend_hardening), tuple(maxwell_modulus),
                             tuple(maxwell_time), tuple(bend_maxwell_modulus), tuple(bend_maxwell_time),
                             float(tear_strength), float(tear_energy), float(weibull_modulus),
                             tuple(float(g) for g in grain), float(grain_ratio),
                             float(shear_strength), float(weibull_reference_area), 1,
                             2 if orthotropic else 1, *(float(v) for v in extra), 2 if mixed else 1,
                             float(shear_tear_energy), float(mixity_exponent), 2 if curvature_range > 0.0 else 1,
                             float(bend_stiffness), float(curvature_range))


@_dc.dataclass(frozen=True)
class CableMaterial:
    """Vertex-block rod parameters -> the vbd_* media block. The optional
    slab lattice reuses the same particle pool."""

    MEDIA_KIND = "cable"
    MEDIA_METHOD = "vbd"

    mass: float
    friction: float
    stretch_stiffness: float
    bend_stiffness: float
    damping: float

    def media_material_kwargs(self) -> dict:
        return dict(
            vbd_particle_mass=float(self.mass),
            vbd_friction=float(self.friction),
            vbd_stretch_stiffness=float(self.stretch_stiffness),
            vbd_bend_stiffness=float(self.bend_stiffness),
            vbd_damping=float(self.damping),
        )


class Cable:
    """Cable (rope) constitutive models. ``Cable.VBD(...)`` builds a vertex-block rod."""

    @staticmethod
    def VBD(mass: float = 0.05, friction: float = 0.4,
            stretch_stiffness: float = 2.0e3, bend_stiffness: float = 0.0,
            damping: float = 0.0) -> CableMaterial:
        """A rope: per-bead ``mass`` kg, contact ``friction``, axial
        ``stretch_stiffness`` EA (N), ``bend_stiffness`` EI (N m^2) and Rayleigh
        ``damping`` (s)."""
        if not stretch_stiffness > 0.0:
            raise ValueError("Cable.VBD: stretch_stiffness must be > 0")
        return CableMaterial(mass, friction, stretch_stiffness, bend_stiffness, damping)


@_dc.dataclass(frozen=True)
class MpmMaterial:
    """MLS-MPM parameters -> the mpm_* media block. ``model_kind`` 0 corotated,
    2 neo-hookean, 3 weakly-compressible fluid, 4 granular Drucker-Prager;
    ``MEDIA_KIND`` (soft_tet / fluid / granular) is set by the factory so the
    scene assembler can match it to the morph."""

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
    dp_friction: float = 0.0
    dp_cohesion: float = 0.0
    # Extra +z grid headroom (m) so kicked/lofted debris stays inside the grid.
    loft_headroom: float = 0.0
    yield_stress: float = 0.0
    hardening_modulus: float = 0.0
    contact_capacity: int = 0

    def __post_init__(self):
        if not isinstance(self.contact_capacity, int) or not 0 <= self.contact_capacity <= 0xFFFFFFFF:
            raise ValueError("MPM contact_capacity must be a nonnegative uint32")
        if self.model_kind != 5.0:
            if self.yield_stress != 0.0 or self.hardening_modulus != 0.0:
                raise ValueError("Plasticity parameters require Hencky J2 (model_kind 5)")
            return
        values = (self.youngs, self.poisson, self.density,
                  self.yield_stress, self.hardening_modulus)
        if (not all(_math.isfinite(v) for v in values) or
                self.youngs <= 0.0 or not -1.0 < self.poisson < 0.5 or
                self.density <= 0.0 or self.yield_stress <= 0.0 or
                self.hardening_modulus < 0.0):
            raise ValueError("Hencky J2 requires finite E,rho,yield > 0, -1 < nu < 0.5, H >= 0")

    @property
    def MEDIA_KIND(self) -> str:
        return self.media_kind

    def media_material_kwargs(self) -> dict:
        return dict(
            mpm_youngs=float(self.youngs), mpm_poisson=float(self.poisson),
            mpm_density=float(self.density), mpm_model_kind=float(self.model_kind),
            mpm_dp_friction=float(self.dp_friction),
            mpm_dp_cohesion=float(self.dp_cohesion),
            mpm_bulk_modulus=float(self.bulk_modulus),
            mpm_tait_gamma=float(self.tait_gamma),
            mpm_viscosity=float(self.viscosity), mpm_dx=float(self.dx),
            mpm_substeps=int(self.substeps),
            mpm_floor_normal=[float(c) for c in self.floor_normal],
            mpm_floor_d=float(self.floor_d),
            mpm_floor_friction=float(self.floor_friction),
            mpm_contact_capacity=self.contact_capacity,
            mpm_loft_headroom=float(self.loft_headroom),
            mpm_yield_stress=float(self.yield_stress),
            mpm_hardening_modulus=float(self.hardening_modulus),
        )

    def fill_kwargs(self) -> dict:
        """The CONSTITUTIVE-only kwargs for ``SceneBuilder.add_mpm_fill`` (a
        heterogeneous sub-fill). The grid scalars (dx / substeps / floor / loft) come
        from the base medium, so they are omitted here."""
        return dict(
            youngs=float(self.youngs), poisson=float(self.poisson),
            density=float(self.density), dp_friction=float(self.dp_friction),
            dp_cohesion=float(self.dp_cohesion), model_kind=float(self.model_kind),
            bulk_modulus=float(self.bulk_modulus), tait_gamma=float(self.tait_gamma),
            viscosity=float(self.viscosity), yield_stress=float(self.yield_stress),
            hardening_modulus=float(self.hardening_modulus))


class Soft:
    """Tet-soft constitutive models: ``Soft.MPM`` (a continuum solid)."""

    @staticmethod
    def MPM(youngs: float = 1.0e5, poisson: float = 0.3, density: float = 1000.0,
            model_kind: float = 0.0, bulk_modulus: float = 0.0,
            tait_gamma: float = 0.0, viscosity: float = 0.0, dx: float = 0.02,
            substeps: int = 20,
            floor_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
            floor_d: float = 0.0, floor_friction: float = 0.4,
            loft_headroom: float = 0.0, yield_stress: float = 0.0,
            hardening_modulus: float = 0.0) -> MpmMaterial:
        """An MLS-MPM continuum solid: ``model_kind`` 0 corotated / 2 neo-hookean,
        ``youngs`` / ``poisson`` elasticity, ``density`` kg/m^3, grid cell ``dx`` m,
        ``substeps`` per step, and a separating floor at ``floor_normal . x =
        floor_d``. ``dx`` must be > 0 for the background grid. ``loft_headroom`` m
        raises the +z grid ceiling for lofted motion."""
        if dx <= 0.0:
            raise ValueError("Soft.MPM: dx must be > 0 (the background grid cell)")
        if substeps < 1:
            raise ValueError("Soft.MPM: substeps must be >= 1")
        return MpmMaterial("soft_tet", youngs, poisson, density, model_kind,
                           bulk_modulus, tait_gamma, viscosity, dx, substeps,
                           floor_normal, floor_d, floor_friction,
                           loft_headroom=loft_headroom, yield_stress=yield_stress,
                           hardening_modulus=hardening_modulus)

    @staticmethod
    def ElastoPlastic(*, yield_stress: float, hardening_modulus: float = 0.0,
                     youngs: float = 1.0e5, poisson: float = 0.3,
                     density: float = 1000.0, dx: float = 0.02, substeps: int = 20,
                     floor_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                     floor_d: float = 0.0, floor_friction: float = 0.4,
                     loft_headroom: float = 0.0) -> MpmMaterial:
        """Hencky J2 solid; yield stress and linear isotropic hardening are in Pa.
        Plastic flow is isochoric; yield uses equivalent Kirchhoff stress."""
        return Soft.MPM(youngs=youngs, poisson=poisson, density=density, model_kind=5.0,
                        dx=dx, substeps=substeps, floor_normal=floor_normal,
                        floor_d=floor_d, floor_friction=floor_friction,
                        loft_headroom=loft_headroom, yield_stress=yield_stress,
                        hardening_modulus=hardening_modulus)


class Fluid:
    """Fluid constitutive models: ``Fluid.MPM`` (a weakly-compressible MLS-MPM fluid)."""

    @staticmethod
    def MPM(density: float = 1000.0, bulk_modulus: float = 2.0e5,
            tait_gamma: float = 7.0, viscosity: float = 0.4, dx: float = 0.011,
            substeps: int = 40,
            floor_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
            floor_d: float = 0.0, floor_friction: float = 0.0,
            loft_headroom: float = 0.0) -> MpmMaterial:
        """A weakly-compressible MLS-MPM fluid (``model_kind`` 3, Tait EOS):
        ``density`` rho0, ``bulk_modulus`` K (sound speed sqrt(K/rho0)),
        ``tait_gamma`` EOS exponent, ``viscosity`` damping, grid cell ``dx`` m,
        ``substeps`` per step, and a free-slip floor BC. ``loft_headroom`` m raises
        the +z grid ceiling for lofted motion."""
        if dx <= 0.0:
            raise ValueError("Fluid.MPM: dx must be > 0 (the background grid cell)")
        if substeps < 1:
            raise ValueError("Fluid.MPM: substeps must be >= 1")
        return MpmMaterial("fluid", 0.0, 0.0, density, 3.0, bulk_modulus,
                           tait_gamma, viscosity, dx, substeps,
                           floor_normal, floor_d, floor_friction,
                           loft_headroom=loft_headroom)


class Granular:
    """Granular constitutive models: ``Granular.MPM`` (Drucker-Prager sand)."""

    @staticmethod
    def MPM(youngs: float = 3.0e5, poisson: float = 0.3, density: float = 1600.0,
            friction_angle: float = 35.0, cohesion: float = 0.0, dx: float = 0.01,
            substeps: int = 30,
            floor_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
            floor_d: float = 0.0, floor_friction: float = 0.5,
            loft_headroom: float = 0.0) -> MpmMaterial:
        """An MLS-MPM Drucker-Prager granular medium (``model_kind`` 4):
        ``friction_angle`` deg sets the repose slope, ``cohesion`` Pa the tensile
        bind (0 = dry sand), ``youngs`` / ``poisson`` the elastic predictor,
        ``density`` kg/m^3, grid cell ``dx`` m, ``substeps`` per step.
        ``loft_headroom`` m raises the +z grid ceiling for foot-lofted debris."""
        if dx <= 0.0:
            raise ValueError("Granular.MPM: dx must be > 0 (the background grid cell)")
        if substeps < 1:
            raise ValueError("Granular.MPM: substeps must be >= 1")
        if not 0.0 <= friction_angle < 90.0:
            raise ValueError("Granular.MPM: friction_angle must be in [0, 90) deg")
        return MpmMaterial("granular", youngs, poisson, density, 4.0,
                           0.0, 0.0, 0.0, dx, substeps,
                           floor_normal, floor_d, floor_friction,
                           dp_friction=friction_angle, dp_cohesion=cohesion,
                           loft_headroom=loft_headroom)


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


@_dc.dataclass(frozen=True)
class RenderAppearance:
    """A render (appearance) material: flat PBR scalars plus optional image maps.

    ``albedo_map`` (sRGB) multiplies ``base_color``; ``roughness_map`` (linear)
    multiplies ``roughness``; ``normal_map`` is a tangent-space GL normal map.
    ``uv_scale`` tiles the maps (tiles per metre under ``triplanar``); triplanar
    projects the maps in body-local space so no authored UVs are needed."""

    name: str
    base_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 1.0
    sheen: float = 0.0
    albedo_map: str = ""
    roughness_map: str = ""
    normal_map: str = ""
    uv_scale: float = 1.0
    triplanar: bool = True

    def add_material_kwargs(self) -> dict:
        import os
        def ap(p: str) -> str:
            return os.path.abspath(p) if p else ""
        return dict(
            name=self.name, base_color=[float(c) for c in self.base_color],
            metallic=float(self.metallic), roughness=float(self.roughness),
            sheen=float(self.sheen), albedo_map=ap(self.albedo_map),
            roughness_map=ap(self.roughness_map), normal_map=ap(self.normal_map),
            uv_scale=float(self.uv_scale), triplanar=bool(self.triplanar),
        )


def Render(name: str, base_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
           metallic: float = 0.0, roughness: float = 1.0, sheen: float = 0.0,
           albedo_map: str = "", roughness_map: str = "", normal_map: str = "",
           uv_scale: float = 1.0, triplanar: bool = True) -> RenderAppearance:
    """An appearance material for any entity (rigid primitive or medium). Image
    maps are file paths (PNG/JPG); empty strings keep the flat PBR scalars."""
    if not name:
        raise ValueError("materials.Render: name must be non-empty")
    return RenderAppearance(name, base_color, metallic, roughness, sheen,
                            albedo_map, roughness_map, normal_map, uv_scale,
                            triplanar)


__all__ = [
    "Cloth", "ClothMaterial", "Cable", "CableMaterial", "Soft",
    "MpmMaterial", "Fluid", "Granular",
    "Rigid", "RigidMaterial", "Render", "RenderAppearance",
]
