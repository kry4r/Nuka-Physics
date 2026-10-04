"""External mesh trajectory audit; no collision algorithm enters the physics engine."""

import json
from pathlib import Path

import numpy as np

from .capture import iter_record_blocks, sha256, write_json


CCD_CONFIGURATION = {"algorithm": "ipctk.TightInclusionCCD", "tolerance": 1e-6,
                     "min_distance": 0, "conservative_rescaling": 1}


def audit_mesh_trace(source, geometry):
    import ipctk
    source, geometry = Path(source), Path(geometry)
    target = source / "ipctk_audit.json"
    if target.exists():
        raise FileExistsError(target)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "closed" or manifest["substeps"] != 1:
        raise ValueError("audit requires a closed trace with every physics interval's positions")
    geometry_hash = sha256(geometry)
    if manifest.get("initial_geometry_sha256") != geometry_hash:
        raise ValueError("geometry hash differs from the recorded initial physics geometry")
    metadata = manifest.get("metadata")
    initial_equals_rest = isinstance(metadata, dict) and metadata.get("initial_positions_equal_rest") is True
    with np.load(geometry, allow_pickle=False) as data:
        if "positions" in data:
            position_key = "positions"
        elif "position" in data:
            position_key = "position"
        elif initial_equals_rest and "rest" in data:
            position_key = "rest"
        else:
            raise ValueError("initial geometry requires positions or position; rest requires "
                             "metadata.initial_positions_equal_rest=true")
        initial = np.asarray(data[position_key])
        if not np.issubdtype(initial.dtype, np.number) or np.iscomplexobj(initial):
            raise ValueError("initial positions require real numeric coordinates")
        initial = initial.astype(np.float64)
        if "faces" not in data or "edges" not in data:
            raise ValueError("initial geometry requires faces and edges")
        faces, edges = np.asarray(data["faces"]), np.asarray(data["edges"])
    initial_position_source = f"{geometry.name}:{position_key}"
    if position_key == "rest":
        initial_position_source += " (metadata.initial_positions_equal_rest=true)"
    if initial.ndim == 2 and manifest["env_count"] == 1:
        initial = initial[np.newaxis, ...]
    if initial.ndim != 3 or initial.shape[-1] != 3 or not np.isfinite(initial).all():
        raise ValueError("initial geometry requires finite three-dimensional vertices")
    if initial.shape[0] != manifest["env_count"]:
        raise ValueError("geometry environment extent differs from the physics trace")
    vertices = initial.shape[1]
    if vertices == 0 or vertices > np.iinfo(np.int32).max:
        raise ValueError("initial geometry vertex extent is invalid")
    for name, topology, width in (("faces", faces, 3), ("edges", edges, 2)):
        if topology.ndim != 2 or topology.shape[1] != width or not np.issubdtype(topology.dtype, np.integer):
            raise ValueError(f"{name} require integer vertex indices with width {width}")
        if np.any(topology < 0) or np.any(topology >= vertices):
            raise ValueError(f"{name} contain invalid vertex indices")
        if np.any(np.diff(np.sort(topology, axis=1), axis=1) == 0):
            raise ValueError(f"{name} repeat a vertex within an element")
    if len(faces) == 0 and len(edges) == 0:
        raise ValueError("initial geometry topology must not be empty")
    faces, edges = faces.astype(np.int32), edges.astype(np.int32)
    meshes = [ipctk.CollisionMesh(np.asfortranarray(points), edges, faces) for points in initial]
    initial_intersections = [bool(ipctk.has_intersections(mesh, np.asfortranarray(points)))
                             for mesh, points in zip(meshes, initial)]
    initial_degenerate_faces, initial_degenerate_edges = [], []
    for points in initial:
        triangle_edges = points[faces[:, 1:]] - points[faces[:, :1]]
        areas = np.cross(triangle_edges[:, 0], triangle_edges[:, 1])
        initial_degenerate_faces.append(int(np.count_nonzero((areas == 0).all(axis=1))))
        initial_degenerate_edges.append(int(np.count_nonzero((points[edges[:, 0]] == points[edges[:, 1]]).all(axis=1))))
    invalid_initial = [crossing or faces_count > 0 or edges_count > 0 for crossing, faces_count, edges_count
                       in zip(initial_intersections, initial_degenerate_faces, initial_degenerate_edges)]
    ccd = ipctk.TightInclusionCCD(tolerance=1e-6, conservative_rescaling=1.0)
    previous = initial.copy()
    events, alarms, segments, ccd_segments = [], 0, 0, 0
    for block, block_manifest in iter_record_blocks(source, states=True):
        if block_manifest != manifest:
            raise ValueError("physics manifest changed during the trajectory audit")
        if "state_PARTICLE_POSITION" not in block:
            raise ValueError("physics trace requires particle positions at every interval")
        states = np.asarray(block["state_PARTICLE_POSITION"])
        steps = block["policy_step"]
        expected_shapes = ((len(steps),) + initial.shape,
                           (len(steps), manifest["env_count"] * vertices, 3),
                           (len(steps), manifest["env_count"] * vertices * 3))
        if states.shape not in expected_shapes:
            raise ValueError("particle state shape differs from the initial geometry and environment extent")
        if not np.issubdtype(states.dtype, np.number) or np.iscomplexobj(states):
            raise ValueError("physics trajectory requires finite real particle coordinates")
        states = states.reshape((len(steps),) + initial.shape)
        for step, current in zip(steps, states):
            for env, mesh in enumerate(meshes):
                start = np.asfortranarray(previous[env], dtype=np.float64)
                end = np.asfortranarray(current[env], dtype=np.float64)
                if not np.isfinite(end).all():
                    raise ValueError(f"nonfinite trajectory at policy step {step}")
                segments += 1
                if invalid_initial[env]:
                    continue
                fraction = ipctk.compute_collision_free_stepsize(mesh, start, end,
                    min_distance=0.0, narrow_phase_ccd=ccd)
                if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                    raise ValueError(f"invalid CCD fraction at policy step {step}")
                if fraction < 1.0:
                    events.append({"policy_step": int(step), "environment": env, "fraction": float(fraction)})
                safety_fraction = ipctk.compute_collision_free_stepsize(mesh, start, end)
                if not np.isfinite(safety_fraction) or not 0 <= safety_fraction <= 1:
                    raise ValueError(f"invalid safety-alarm fraction at policy step {step}")
                if safety_fraction < 1.0:
                    alarms += 1
                ccd_segments += 1
            previous[:] = current
    manifest_hash = sha256(manifest_path)
    report = {"manifest_sha256": manifest_hash, "source_manifest_sha256": manifest_hash,
        "geometry_sha256": geometry_hash,
        "geometry": str(geometry), "scope": "All vertices and faces supplied in the geometry artifact",
        "segments": segments, "zero_distance_crossings": len(events), "events": events,
        "ccd_segments": ccd_segments,
        "initial_intersections_by_environment": initial_intersections,
        "initial_degenerate_faces_by_environment": initial_degenerate_faces,
        "initial_degenerate_edges_by_environment": initial_degenerate_edges,
        "skipped_invalid_initial_environments": [env for env, invalid in enumerate(invalid_initial) if invalid],
        "default_safety_alarms": alarms,
        "configuration": CCD_CONFIGURATION,
        "default_alarm_configuration": {"conservative_rescaling": 0.8},
        "initial_position_source": initial_position_source,
        "covered_contact_domains": ["particle_mesh_self"],
        "trajectory": "Piecewise linear particle vertex trajectories",
        "thickness_scope": "Zero-distance mesh crossing; finite offsets and rigid rotation require separate checks"}
    write_json(target, report)
    return report
