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


def ellipsoid_mesh(semiaxes, *, max_error: float) -> TriangleApproximation:
    """Approximate a centered ellipsoid with an inscribed closed triangle mesh.
    The unit mesh's inscribed sphere gives a solid Hausdorff and support bound."""
    axes = np.asarray(semiaxes, dtype=np.float64)
    max_error = float(max_error)
    if axes.shape != (3,) or not np.isfinite(axes).all() or np.any(axes <= 0) or \
            not math.isfinite(max_error) or max_error <= 0:
        raise ValueError("Ellipsoid semiaxes and max_error must be finite and positive")
    radius = float(axes.max())
    rounding = 8 * np.finfo(np.float64).eps * radius
    if max_error <= rounding:
        raise ValueError("Requested error is below the mesh construction precision")
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                        dtype=np.float64)
    triangles = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                          [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], dtype=np.int32)
    while True:
        faces = vertices[triangles]
        edge_squared = max(float(np.square(faces[:, a] - faces[:, b]).sum(axis=1).max())
                           for a, b in ((0, 1), (1, 2), (2, 0)))
        # Every triangle point has squared radius at least 1 - max_edge_squared / 3.
        # Its inscribed sphere bounds the radial gap to the unit sphere.
        fraction = min(edge_squared / 3.0, 1.0)
        bound = radius * fraction / (1.0 + math.sqrt(1.0 - fraction)) + rounding
        if bound <= max_error:
            return TriangleApproximation(vertices * axes, triangles, bound)
        if 4 * len(triangles) > np.iinfo(np.int32).max:
            raise ValueError("Requested error requires too many mesh triangles")
        edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        edges, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
        if len(vertices) + len(edges) > np.iinfo(np.int32).max:
            raise ValueError("Requested error requires too many mesh vertices")
        midpoint = vertices[edges[:, 0]] + vertices[edges[:, 1]]
        midpoint /= np.linalg.norm(midpoint, axis=1, keepdims=True)
        ab, bc, ca = (inverse.reshape(3, -1) + len(vertices)).astype(np.int32)
        a, b, c = triangles.T
        triangles = np.concatenate((np.stack((a, ab, ca), axis=1), np.stack((ab, b, bc), axis=1),
                                    np.stack((ca, bc, c), axis=1), np.stack((ab, bc, ca), axis=1)))
        vertices = np.concatenate((vertices, midpoint))
