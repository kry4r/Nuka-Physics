"""Constitutive materials for the authoring facade.

``Cloth.XPBD(...)`` is an XPBD membrane material mapping to the
``cloth_particle_mass / cloth_friction / cloth_bend_alpha / cloth_iters`` cook
kwargs. ``Fluid`` (PBF) and ``Soft`` (MLS-MPM) are reserved for the other media
on the SAME coupled cook -- a material only carries parameters, never cook math.
"""

from __future__ import annotations

import dataclasses as _dc


@_dc.dataclass(frozen=True)
class ClothMaterial:
    """XPBD cloth parameters -> the cloth_* cook kwargs."""

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


__all__ = ["Cloth", "ClothMaterial"]
