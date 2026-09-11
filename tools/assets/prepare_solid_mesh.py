"""Close oriented OBJ boundary loops and save a solid mesh with unit-mass inertia."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def quaternion(matrix):
    values, vectors = np.linalg.eigh(np.array([
        [matrix[0, 0]-matrix[1, 1]-matrix[2, 2], matrix[0, 1]+matrix[1, 0],
         matrix[0, 2]+matrix[2, 0], matrix[2, 1]-matrix[1, 2]],
        [matrix[0, 1]+matrix[1, 0], matrix[1, 1]-matrix[0, 0]-matrix[2, 2],
         matrix[1, 2]+matrix[2, 1], matrix[0, 2]-matrix[2, 0]],
        [matrix[0, 2]+matrix[2, 0], matrix[1, 2]+matrix[2, 1],
         matrix[2, 2]-matrix[0, 0]-matrix[1, 1], matrix[1, 0]-matrix[0, 1]],
        [matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0],
         matrix[1, 0]-matrix[0, 1], np.trace(matrix)],
    ]) / 3)
    xyzw = vectors[:, np.argmax(values)]
    if xyzw[3] < 0:
        xyzw *= -1
    return xyzw[[3, 0, 1, 2]]


def triangulate_boundary(loop, vertices, normal):
    axes = [axis for axis in range(3) if axis != int(np.argmax(abs(normal)))]
    points = vertices[loop][:, axes]
    cross = lambda a, b: a[0]*b[1] - a[1]*b[0]
    orientation = np.sign(sum(cross(a, b) for a, b in zip(points, np.roll(points, -1, axis=0))))
    remaining, result = list(range(len(loop))), []
    while len(remaining) > 3:
        for position, current in enumerate(remaining):
            before, after = remaining[position-1], remaining[(position+1) % len(remaining)]
            a, b, c = points[[before, current, after]]
            if orientation*cross(b-a, c-b) <= 0:
                continue
            contains = False
            for other in remaining:
                if other in (before, current, after):
                    continue
                p = points[other]
                if min(orientation*cross(b-a, p-a), orientation*cross(c-b, p-b),
                       orientation*cross(a-c, p-c)) >= 0:
                    contains = True
                    break
            if not contains:
                result.append([loop[after], loop[current], loop[before]])
                del remaining[position]
                break
        else:
            raise ValueError("The boundary cannot be triangulated without crossings")
    result.append([loop[index] for index in reversed(remaining)])
    return result


def prepare(source, destination, extent, y_up):
    vertices, triangles = [], []
    for line in source.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append(list(map(float, fields[1:4])))
        elif fields[0] == "f":
            ids = [int(field.split("/")[0]) for field in fields[1:]]
            ids = [index-1 if index > 0 else len(vertices)+index for index in ids]
            if len(ids) != 3:
                raise ValueError("The input must have triangulated faces")
            triangles.append(ids)
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not triangles:
        raise ValueError("The OBJ has no usable triangle surface")
    vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices *= extent / np.ptp(vertices, axis=0).max()
    if y_up:
        vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
    edges = defaultdict(list)
    for triangle in triangles:
        for a, b in zip(triangle, triangle[1:] + triangle[:1]):
            edges[tuple(sorted((a, b)))].append((a, b))
    if any(len(value) > 2 or (len(value) == 2 and value[0] != value[1][::-1])
           for value in edges.values()):
        raise ValueError("The source has nonmanifold edges or inconsistent winding")
    boundary = [value[0] for value in edges.values() if len(value) == 1]
    outgoing = {}
    for a, b in boundary:
        if a in outgoing:
            raise ValueError("A boundary vertex belongs to more than one loop")
        outgoing[a] = b
    loops = []
    while outgoing:
        start = min(outgoing)
        loop, current = [], start
        while current in outgoing:
            loop.append(current)
            current = outgoing.pop(current)
        if current != start or len(loop) < 3:
            raise ValueError("An open boundary is not a simple loop")
        loops.append(loop)
    original_vertices = len(vertices)
    original_triangles = len(triangles)
    caps = []
    for loop in loops:
        points = vertices[loop]
        center = points.mean(axis=0)
        normal = np.sum(np.cross(points-center, np.roll(points, -1, axis=0)-center), axis=0)
        if np.linalg.norm(normal) == 0:
            raise ValueError("A boundary loop has no cap orientation")
        normal /= np.linalg.norm(normal)
        triangles.extend(triangulate_boundary(loop, vertices, normal))
        caps.append({"vertices": len(loop), "center_m": center.tolist(),
                     "maximum_plane_distance_m": float(np.max(abs((points-center) @ normal)))})
    triangles = np.asarray(triangles, dtype=np.int64)
    faces = vertices[triangles]
    volume = np.einsum("ij,ij->i", faces[:, 0], np.cross(faces[:, 1], faces[:, 2])) / 6
    total = volume.sum()
    if total <= 0:
        raise ValueError("The closed surface does not have positive oriented volume")
    sums = faces.sum(axis=1)
    center = (volume[:, None]*sums/4).sum(axis=0) / total
    products = np.einsum("ni,nj->nij", sums, sums) + np.einsum("nki,nkj->nij", faces, faces)
    covariance = (volume[:, None, None]*products/20).sum(axis=0)/total - np.outer(center, center)
    inertia = np.trace(covariance)*np.eye(3) - covariance
    moments, axes = np.linalg.eigh(inertia)
    if np.any(moments <= 0):
        raise ValueError("The mass tensor is not positive definite")
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="\n") as output:
        for vertex in vertices:
            output.write("v " + " ".join(f"{value:.10g}" for value in vertex) + "\n")
        for triangle in triangles:
            output.write("f " + " ".join(str(index+1) for index in triangle) + "\n")
    report = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "extent_m": extent, "source_y_up": y_up, "source_vertices": original_vertices,
        "source_triangles": original_triangles, "closed_triangles": len(triangles),
        "boundary_caps": caps, "volume_m3": float(total), "center_of_mass_m": center.tolist(),
        "inertia_tensor_per_kg": inertia.tolist(), "principal_inertia_per_kg": moments.tolist(),
        "principal_rotation_wxyz": quaternion(axes).tolist(),
        "inertia_reconstruction_relative_error": float(np.linalg.norm(axes@np.diag(moments)@axes.T-inertia)
                                                        / np.linalg.norm(inertia)),
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--extent", type=float, required=True)
    parser.add_argument("--y-up", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.extent) or args.extent <= 0:
        parser.error("extent must be finite and positive")
    if args.source.resolve() == args.destination.resolve():
        parser.error("source and destination must differ")
    prepare(args.source, args.destination, args.extent, args.y_up)


if __name__ == "__main__":
    main()
