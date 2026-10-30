"""Bounded triangle representations of analytic collision geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TriangleApproximation:
    """An inscribed closed mesh and its Hausdorff distance bound in metres."""

    vertices: np.ndarray
    triangles: np.ndarray
    error_bound: float


def cone_mesh(radius: float, half_height: float, *, max_error: float) -> TriangleApproximation:
    """Approximate a solid +Z cone with base at -half_height and tip at +half_height.
    The base polygon's sagitta bounds solid Hausdorff and support function error."""
    radius, half_height, max_error = map(float, (radius, half_height, max_error))
    if not all(math.isfinite(v) and v > 0 for v in (radius, half_height, max_error)):
        raise ValueError("Cone dimensions and max_error must be finite and positive")
    angle = 2 * math.asin(math.sqrt(min(max_error / radius, 1.0) / 2))
    sides = max(3, math.ceil(math.pi / angle))
    if sides > np.iinfo(np.int32).max - 2:
        raise ValueError("Requested error requires too many mesh vertices")
    theta = np.arange(sides) * (2 * math.pi / sides)
    vertices = np.zeros((sides + 2, 3), dtype=np.float64)
    vertices[:sides, 0] = radius * np.cos(theta)
    vertices[:sides, 1] = radius * np.sin(theta)
    vertices[:sides, 2] = -half_height
    vertices[sides, 2] = half_height
    vertices[sides + 1, 2] = -half_height
    index = np.arange(sides, dtype=np.int32)
    following = (index + 1) % sides
    triangles = np.concatenate((np.stack((index, following, np.full(sides, sides)), axis=1),
                                np.stack((np.full(sides, sides + 1), following, index), axis=1)))
    bound = 2 * radius * math.sin(math.pi / (2 * sides)) ** 2
    return TriangleApproximation(vertices, triangles.astype(np.int32), bound)
