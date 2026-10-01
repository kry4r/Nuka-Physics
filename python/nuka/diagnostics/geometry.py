"""External mesh trajectory audit; no collision algorithm enters the physics engine."""

import json
from pathlib import Path

import numpy as np

from .capture import sha256, write_json


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
    with np.load(geometry, allow_pickle=False) as data:
        rest = np.asarray(data["rest"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int32)
        edges = np.asarray(data["edges"], dtype=np.int32)
    if rest.ndim == 2:
        rest = np.broadcast_to(rest, (manifest["env_count"],) + rest.shape).copy()
    if rest.shape[0] != manifest["env_count"]:
        raise ValueError("geometry environment extent differs from the physics trace")
    if rest.ndim != 3 or rest.shape[-1] != 3 or not np.isfinite(rest).all():
        raise ValueError("initial geometry requires finite three-dimensional vertices")
    meshes = [ipctk.CollisionMesh(np.asfortranarray(vertices), edges, faces) for vertices in rest]
    ccd = ipctk.TightInclusionCCD(tolerance=1e-6, conservative_rescaling=1.0)
    previous = rest.copy()
    events, alarms, segments = [], 0, 0
    next_step = 1
    for chunk in manifest["chunks"]:
        path = source / chunk["file"]
        if sha256(path) != chunk["sha256"]:
            raise ValueError(f"evidence hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            states = data["state_PARTICLE_POSITION"].reshape(len(data["policy_step"]), *rest.shape)
            steps = data["policy_step"].copy()
        if chunk["first_policy_step"] != next_step or len(steps) != chunk["steps"] or not np.array_equal(
                steps, np.arange(next_step, next_step + len(steps))):
            raise ValueError("physics evidence has a discontinuity")
        next_step += len(steps)
        for step, current in zip(steps, states):
            for env, mesh in enumerate(meshes):
                start = np.asfortranarray(previous[env], dtype=np.float64)
                end = np.asfortranarray(current[env], dtype=np.float64)
                if not np.isfinite(end).all():
                    raise ValueError(f"nonfinite trajectory at policy step {step}")
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
                segments += 1
            previous[:] = current
    if next_step != manifest["steps"] + 1:
        raise ValueError("audited trajectory extent differs from the physics manifest")
    report = {"manifest_sha256": sha256(manifest_path), "geometry_sha256": geometry_hash,
        "geometry": str(geometry), "scope": "All vertices and faces supplied in the geometry artifact",
        "segments": segments, "zero_distance_crossings": len(events), "events": events,
        "default_safety_alarms": alarms,
        "configuration": CCD_CONFIGURATION,
        "default_alarm_configuration": {"conservative_rescaling": 0.8},
        "trajectory": "Piecewise linear between each recorded production interval"}
    write_json(target, report)
    return report
