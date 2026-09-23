"""Repose a sewn surface onto joint landmarks without changing its topology."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull, QhullError


def body_landmarks(path: Path, names, joint_positions=None, base_height=None):
    """Evaluate source MJCF kinematics with MuJoCo; meshes are unnecessary for landmarks."""
    import mujoco

    root = ET.parse(path).getroot()
    for tag in ("asset", "actuator", "sensor", "contact", "equality", "keyframe", "tendon"):
        for element in list(root.findall(tag)):
            root.remove(element)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom":
                parent.remove(child)
    if base_height is not None:
        body = root.find("./worldbody/body")
        position = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        position[2] = base_height
        body.set("pos", " ".join(map(str, position)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    for name, value in (joint_positions or {}).items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected a named hinge joint: {name}")
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    positions = {}
    for name in names:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            raise ValueError(f"Unknown body landmark: {name}")
        positions[name] = data.xpos[body].copy()
    return positions


def body_envelopes(path: Path, joint_positions, base_height):
    """Read posed visual meshes as conservative convex envelopes for offline fitting."""
    import mujoco

    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((path.parent / compiler.get("meshdir", ".")).resolve()))
    body = root.find("./worldbody/body")
    position = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
    position[2] = base_height
    body.set("pos", " ".join(map(str, position)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    for name, value in joint_positions.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"Unknown joint: {name}")
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    envelopes = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 2:
            continue
        mesh = model.geom_dataid[geom]
        first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[first:first+count].astype(np.float64)
        vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        try:
            hull = ConvexHull(vertices)
        except QhullError:
            continue
        envelopes.append({"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh),
                          "planes": hull.equations, "min": vertices.min(0), "max": vertices.max(0)})
    return envelopes


def fit_clearance(vertices, triangles, envelopes, clearance, iterations=24):
    """Relax an initial garment outside body envelopes without changing its rest pattern."""
    if not np.isfinite(clearance) or clearance < 0 or iterations < 1:
        raise ValueError("Fitting clearance must be nonnegative and iterations positive")
    initial = np.asarray(vertices, dtype=np.float64)
    fitted = initial.copy()
    graph, _ = surface_graph(initial, triangles)
    adjacency = graph.copy()
    adjacency.data.fill(1.0)
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1.0)
    violations = []
    for iteration in range(iterations):
        if iteration:
            displacement = fitted - initial
            fitted = initial + 0.5 * (displacement + adjacency @ displacement / degree[:, None])
        maximum = 0.0
        for envelope in envelopes:
            candidates = np.flatnonzero(np.all((fitted >= envelope["min"]-clearance) &
                                               (fitted <= envelope["max"]+clearance), axis=1))
            if not len(candidates):
                continue
            planes = envelope["planes"]
            signed = fitted[candidates] @ planes[:, :3].T + planes[:, 3]
            face = signed.argmax(axis=1)
            distance = signed[np.arange(len(candidates)), face]
            correction = np.maximum(clearance-distance, 0)
            maximum = max(maximum, float(correction.max()))
            fitted[candidates] += planes[face, :3] * correction[:, None]
        violations.append(maximum)
    displacement = np.linalg.norm(fitted-initial, axis=1)
    return fitted, {"method": "convex_body_clearance_relaxation", "clearance": clearance,
                    "iterations": iterations, "envelopes": [envelope["name"] for envelope in envelopes],
                    "projection_max": violations, "displacement_max": float(displacement.max()),
                    "displacement_rms": float(np.sqrt(np.mean(displacement**2))),
                    "displaced_vertices": int(np.count_nonzero(displacement > 1e-6))}


def surface_graph(vertices, triangles):
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                                    triangles[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Garment fitting requires a manifold triangle surface")
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    adjacency = coo_matrix((lengths, (edges[:, 0], edges[:, 1])),
                           shape=(len(vertices), len(vertices))).tocsr()
    adjacency = adjacency + adjacency.T
    boundary = edges[counts == 1]
    vertices_on_boundary, inverse = np.unique(boundary, return_inverse=True)
    local_edges = inverse.reshape(-1, 2)
    boundary_graph = coo_matrix((np.ones(len(boundary)), (local_edges[:, 0], local_edges[:, 1])),
                               shape=(len(vertices_on_boundary), len(vertices_on_boundary)))
    count, labels = connected_components(boundary_graph, directed=False)
    loops = [vertices_on_boundary[labels == index] for index in range(count)]
    return adjacency, loops


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def segment_transform(vertices, source, target, radial_scale):
    source_axis = source[1] - source[0]
    target_axis = target[1] - target[0]
    source_length = np.linalg.norm(source_axis)
    target_length = np.linalg.norm(target_axis)
    if min(source_length, target_length) <= 0:
        raise ValueError("Bone segments must have positive length")
    a, b = source_axis / source_length, target_axis / target_length
    cross = np.cross(a, b)
    cosine = np.dot(a, b)
    if cosine < -1.0 + 1e-10:
        tangent = np.eye(3)[np.argmin(np.abs(a))]
        normal = np.cross(a, tangent)
        normal /= np.linalg.norm(normal)
        rotation = 2 * np.outer(normal, normal) - np.eye(3)
    else:
        x, y, z = cross
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        rotation = np.eye(3) + skew + skew @ skew / (1.0 + cosine)
    relative = vertices - source[0]
    axial = (relative @ a)[:, None] * a
    resized = radial_scale * (relative - axial) + (target_length / source_length) * axial
    return resized @ rotation.T + target[0]


def retarget_surface(vertices, triangles, source_chains, target_chains, *, scale,
                     source_anchor, target_anchor, blend_fraction=0.2):
    """Blend limb transforms using surface distance from each open cuff."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if not np.isfinite(vertices).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Garment vertices and scale must be finite with positive scale")
    if not 0 < blend_fraction < 0.5:
        raise ValueError("Limb blend_fraction must lie in (0, 0.5)")
    if len(source_chains) != len(target_chains):
        raise ValueError("Source and target must have the same limb count")
    graph, loops = surface_graph(vertices, triangles)
    if len(loops) < len(source_chains):
        raise ValueError("Each limb needs a separate open cuff")
    torso = (vertices - source_anchor) * scale + target_anchor
    weighted = np.zeros_like(vertices)
    total_weight = np.zeros(len(vertices))
    used_loops = set()
    report = []
    for source, target in zip(source_chains, target_chains):
        source, target = np.asarray(source), np.asarray(target)
        if source.shape != (3, 3) or target.shape != (3, 3):
            raise ValueError("Limb landmarks must be root, middle and cuff positions")
        distances = [np.linalg.norm(vertices[loop].mean(0) - source[-1])
                     if i not in used_loops else np.inf for i, loop in enumerate(loops)]
        selected = int(np.argmin(distances))
        used_loops.add(selected)
        cuff = loops[selected]
        along_surface = dijkstra(graph, indices=cuff, min_only=True)
        lengths = np.linalg.norm(np.diff(source, axis=0), axis=1)
        reach = lengths.sum()
        blend = blend_fraction * reach
        limb_weight = 1.0 - smoothstep((along_surface - reach + blend) / (2 * blend))
        lower_blend = blend_fraction * min(lengths)
        lower_weight = 1.0 - smoothstep((along_surface - lengths[1] + lower_blend) / (2 * lower_blend))
        upper = segment_transform(vertices, source[:2], target[:2], scale)
        lower = segment_transform(vertices, source[1:], target[1:], scale)
        limb = upper * (1 - lower_weight[:, None]) + lower * lower_weight[:, None]
        weighted += limb * limb_weight[:, None]
        total_weight += limb_weight
        report.append({"cuff_vertices": len(cuff), "source_cuff_distance": distances[selected],
                       "source_chain": source.tolist(), "target_chain": target.tolist(),
                       "affected_vertices": int(np.count_nonzero(limb_weight > 0))})
    denominator = np.maximum(total_weight, 1.0)
    fitted = weighted / denominator[:, None] + torso * np.maximum(1 - total_weight[:, None], 0)
    return fitted, {"method": "surface_geodesic_limb_blend", "blend_fraction": blend_fraction,
                    "source_anchor": np.asarray(source_anchor).tolist(),
                    "target_anchor": np.asarray(target_anchor).tolist(), "limbs": report}
