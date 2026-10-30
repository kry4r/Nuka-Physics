"""Portable baked media assets in SI units and explicit particle order."""

from pathlib import Path
import hashlib
import struct

import numpy as np


def _floats(values, shape, name, *, optional=False):
    if values is None and optional:
        return np.empty(0, dtype="<f4")
    array = np.asarray(values, dtype="<f4")
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} requires finite values of shape {shape}")
    return np.ascontiguousarray(array)


def _indices(values, shape, name, upper):
    array = np.asarray(values)
    if array.shape != shape or array.dtype.kind not in "iu" or (array < 0).any() or (array >= upper).any():
        raise ValueError(f"{name} requires integer indices of shape {shape} below {upper}")
    return np.ascontiguousarray(array, dtype="<u4")


def _write_chunks(path, chunks):
    path = Path(path).resolve()
    offset = 12 + 28 * len(chunks)
    entries = []
    for tag, data in chunks:
        digest = int.from_bytes(hashlib.sha256(data).digest()[:8], "little")
        entries.append(struct.pack("<IQQQ", int.from_bytes(tag.encode("ascii"), "big"), offset, len(data), digest))
        offset += len(data)
    path.write_bytes(b"NKA1" + struct.pack("<II", 1, len(chunks)) + b"".join(entries)
                     + b"".join(data for _, data in chunks))
    return str(path)


def write_tet_mesh(path, positions, tetrahedra, *, rest_positions=None, rest_inverse=None,
                   velocities=None, inverse_masses=None, material_ids=None):
    """Write TETS/0; zero inverse masses pin vertices. Row-major rest inverse overrides rest positions.
    Without either, positions define rest shape; material IDs index the scene's tetrahedral table."""
    shape, tet_shape = np.shape(positions), np.shape(tetrahedra)
    if len(shape) != 2 or shape[1] != 3 or not 0 < shape[0] <= 0xFFFFFFFF:
        raise ValueError("positions require a nonempty (N, 3) array")
    if len(tet_shape) != 2 or tet_shape[1] != 4 or not 0 < tet_shape[0] <= 0x3FFFFFFF:
        raise ValueError("tetrahedra require a nonempty (T, 4) array")
    n, t = shape[0], tet_shape[0]
    q = _floats(positions, (n, 3), "positions")
    rest = _floats(rest_positions, (n, 3), "rest_positions", optional=True)
    velocity = _floats(velocities, (n, 3), "velocities", optional=True)
    mass = _floats(inverse_masses, (n,), "inverse_masses", optional=True)
    if (mass < 0).any():
        raise ValueError("inverse_masses must be nonnegative")
    indices = _indices(tetrahedra, (t, 4), "tetrahedra", n)
    if (np.diff(np.sort(indices, axis=1), axis=1) == 0).any():
        raise ValueError("tetrahedra must name four distinct vertices")
    inverse = _floats(rest_inverse, (t, 3, 3), "rest_inverse", optional=True)
    ids = (np.empty(0, dtype="<u4") if material_ids is None else
           _indices(material_ids, (t,), "material_ids", 0x100000000))
    streams = (rest, velocity, mass, inverse, ids)
    flags = sum((1 << i) for i, stream in enumerate(streams) if stream.size)
    payload = struct.pack("<IIII", 1, n, t, flags) + b"".join(
        a.tobytes() for a in (q, rest, velocity, mass, indices, inverse, ids))
    return _write_chunks(path, [("TETS", payload)]) + "#TETS/0"


def write_particle_set(path, positions, inverse_masses, rest_volumes, *, velocities=None, material_ids=None,
                       elastic_deformation=None, affine_velocity=None, plastic_deformation=None,
                       equivalent_plastic_strain=None):
    """Write PSET/0 in m, kg, s with row-major matrices; missing F/Fp are identity, velocity/C zero.
    Plastic deformation and equivalent plastic strain are histories of the Hencky J2 material."""
    shape = np.shape(positions)
    if len(shape) != 2 or shape[1] != 3 or not 0 < shape[0] <= 0xFFFFFFFF:
        raise ValueError("positions require a nonempty (N, 3) array")
    n = shape[0]
    q = _floats(positions, (n, 3), "positions")
    mass = _floats(inverse_masses, (n,), "inverse_masses")
    volume = _floats(rest_volumes, (n,), "rest_volumes")
    if (mass < 0).any() or (volume <= 0).any():
        raise ValueError("inverse_masses must be nonnegative and rest_volumes positive")
    velocity = _floats(velocities, (n, 3), "velocities", optional=True)
    ids = (np.empty(0, dtype="<u4") if material_ids is None else
           _indices(material_ids, (n,), "material_ids", 0x100000000))
    f = _floats(elastic_deformation, (n, 3, 3), "elastic_deformation", optional=True)
    c = _floats(affine_velocity, (n, 3, 3), "affine_velocity", optional=True)
    fp = _floats(plastic_deformation, (n, 3, 3), "plastic_deformation", optional=True)
    strain = _floats(equivalent_plastic_strain, (n,), "equivalent_plastic_strain", optional=True)
    if any(a.size and (np.linalg.det(a.astype(np.float64)) <= 0).any() for a in (f, fp)):
        raise ValueError("deformation gradients require positive determinants")
    if (strain < 0).any():
        raise ValueError("equivalent_plastic_strain must be nonnegative")
    streams = (velocity, ids, f, c, fp, strain)
    flags = sum((1 << i) for i, stream in enumerate(streams) if stream.size)
    payload = struct.pack("<IIII", 1, n, flags, 0) + b"".join(a.tobytes() for a in (q, mass, volume) + streams)
    return _write_chunks(path, [("PSET", payload)]) + "#PSET/0"


__all__ = ["write_tet_mesh", "write_particle_set"]
