"""Render helpers for the authoring facade.

Pure host camera math for ``World.render_beauty``: a hero orbit that sweeps from
a wide side profile (``e == 0``) to a tighter front-three-quarter (``e == 1``)
over a target point, pulling in + down as the clip settles.
"""

from __future__ import annotations

import math
from typing import Tuple

Vec3 = Tuple[float, float, float]


def smoothstep(t: float) -> float:
    """The cubic ``3t^2 - 2t^3`` eased clamp of ``t`` to ``[0, 1]``."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def hero_orbit(e: float, center: Vec3 = (0.05, 0.0, 0.42),
               radius: float = 1.95, pull_in: float = 0.55,
               drop: float = 0.24) -> Tuple[Vec3, Vec3]:
    """A hero orbit ``(eye, look)`` at normalised time ``e`` in ``[0, 1]``.

    Sweeps azimuth from a side profile to a front-three-quarter, eases elevation,
    and pulls the eye in by ``pull_in`` m while the look point drops by ``drop`` m.
    """
    e = max(0.0, min(1.0, e))
    az = -0.50 + 1.05 * e
    elev = math.radians(18.0 - 4.0 * e + 4.0 * math.sin(e * math.pi))
    r = radius - pull_in * e
    look = (center[0], center[1], center[2] - drop * e)
    eye = (look[0] + r * math.cos(elev) * math.sin(az),
           look[1] - r * math.cos(elev) * math.cos(az),
           look[2] + r * math.sin(elev))
    return eye, look


__all__ = ["smoothstep", "hero_orbit"]
