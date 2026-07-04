"""Surface / membrane behaviors for the authoring facade.

``Cloth(free=, aero=)`` is the drape membrane: ``free`` skips the perimeter pin
(a free drape) and ``aero`` is the anisotropic drag ``(normal, tangent, max_dv)``.
``Soft(offset=, smooth_iters=, smooth_lambda=)`` is the render skin baked over a
tet/fluid medium (a normal offset + Laplacian smoothing). A surface maps onto the
``cloth_free``/``aero_*`` cook kwargs (fast path) or the matching
``SceneBuilder.add_media`` kwargs (builder path); all-zero aero keeps the cook
byte-identical.
"""

from __future__ import annotations

import dataclasses as _dc
from typing import Tuple


@_dc.dataclass(frozen=True)
class ClothSurface:
    """A cloth membrane: ``free`` perimeter + anisotropic ``aero`` drag."""

    free: bool = False
    aero: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _aero3(self) -> Tuple[float, float, float]:
        if len(self.aero) != 3:
            raise ValueError("surfaces.Cloth: aero must be (normal, tangent, max_dv)")
        return tuple(float(x) for x in self.aero)

    def cook_kwargs(self) -> dict:
        n, t, dv = self._aero3()
        return dict(cloth_free=bool(self.free), aero_normal=n,
                    aero_tangent=t, aero_max_dv=dv)

    def media_kwargs(self) -> dict:
        """The cloth free/aero block for ``SceneBuilder.add_media`` (builder path)."""
        n, t, dv = self._aero3()
        return dict(cloth_free=bool(self.free), xpbd_aero_normal=n,
                    xpbd_aero_tangent=t, xpbd_aero_max_dv=dv)


@_dc.dataclass(frozen=True)
class SoftSurface:
    """A render skin baked over a tet/fluid medium: a ``offset`` m normal inflation
    and ``smooth_iters`` Laplacian passes at ``smooth_lambda`` (consumed downstream
    by the surface bake, not the cook)."""

    offset: float = 0.0
    smooth_iters: int = 0
    smooth_lambda: float = 0.5

    def media_kwargs(self) -> dict:
        return dict(skin_normal_offset=float(self.offset),
                    skin_smooth_iters=int(self.smooth_iters),
                    skin_smooth_lambda=float(self.smooth_lambda))


@_dc.dataclass(frozen=True)
class GrainSkin:
    """The instanced-sphere grain look for a particle medium (granular / MPM /
    cable beads): ``round`` swaps the octahedron grains for analytic spheres
    (perfectly round at any zoom, cheaper BLAS), and ``radius_jitter`` /
    ``tint_jitter`` add deterministic per-grain radius + albedo scatter. All
    0/False (default) keeps today's uniform octahedra (render byte-identical)."""

    round: bool = False
    radius_jitter: float = 0.0
    tint_jitter: float = 0.0

    def media_kwargs(self) -> dict:
        return dict(skin_grain_round=1 if self.round else 0,
                    skin_grain_radius_jitter=float(self.radius_jitter),
                    skin_grain_tint_jitter=float(self.tint_jitter))


# The public surface names (``surfaces.Cloth`` / ``surfaces.Soft`` / ``surfaces.Grains``).
Cloth = ClothSurface
Soft = SoftSurface
Grains = GrainSkin

__all__ = ["Cloth", "ClothSurface", "Soft", "SoftSurface", "Grains", "GrainSkin"]
