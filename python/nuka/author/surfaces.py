"""Surface / membrane behaviors for the authoring facade.

``Cloth(free=, aero=)`` is the drape membrane: ``free`` skips the perimeter pin
(a free drape instead of a taut membrane) and ``aero`` is the anisotropic
aerodynamic drag ``(normal, tangent, max_dv)``. Maps to ``cloth_free`` +
``aero_normal / aero_tangent / aero_max_dv``; all-zero aero keeps the cook
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

    def cook_kwargs(self) -> dict:
        if len(self.aero) != 3:
            raise ValueError("surfaces.Cloth: aero must be (normal, tangent, max_dv)")
        n, t, dv = (float(x) for x in self.aero)
        return dict(cloth_free=bool(self.free), aero_normal=n,
                    aero_tangent=t, aero_max_dv=dv)


# The public surface name (``surfaces.Cloth(free=..., aero=...)``).
Cloth = ClothSurface

__all__ = ["Cloth", "ClothSurface"]
