"""Independent float64 elastic energies and gradients from cooked VBD topology."""

from pathlib import Path

import numpy as np

from ..energy import EnergyColumn as E
from .capture import load_records, sha256


def unpack_elements(words):
    words = np.asarray(words)
    if words.dtype != np.uint32 or words.ndim != 2 or words.shape[1] != 16:
        raise ValueError("VBD elements require sixteen uint32 words per cooked record")
    values = np.ascontiguousarray(words).view(np.float32)
    if not np.isfinite(values[:, 5:14]).all():
        raise ValueError("cooked VBD parameters must be finite")
    return words[:, 0], words[:, 1:5], values[:, 5:13].astype(np.float64), values[:, 13]


def elastic_quantities(words, positions, *, rates=None, dt=None, precision=np.float64):
    """Evaluate all elastic families; invalid geometry remains an explicit error."""
    kinds, vertices, parameters, damping = unpack_elements(words)
    if np.dtype(precision) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("elastic analysis requires float32 or float64 arithmetic")
    x = np.asarray(positions, dtype=precision)
    exact = parameters
    parameters = parameters.astype(precision)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("elastic evaluation requires finite vertex positions")
    if not np.isin(kinds, [0, 1, 2, 3]).all():
        raise ValueError("unknown cooked VBD material family")
    if rates is not None:
        rates = np.asarray(rates, dtype=precision)
        if rates.shape != x.shape or not np.isfinite(rates).all() or dt is None or not dt > 0:
            raise ValueError("relative elastic rates require matching finite vectors and a positive step")
    gradient = np.zeros_like(x)
    energy_by_kind = {}
    for kind in range(4):
        selected = kinds == kind
        if not selected.any():
            continue
        count = 4 if kind == 1 else 2 if kind == 2 else 3
        ids = vertices[selected, :count]
        if (ids >= len(x)).any():
            raise ValueError("cooked VBD vertex index is outside the captured range")
        p, local = parameters[selected], x[ids]
        local = local - local[:, :1]
        motion = np.zeros_like(local)
        if rates is not None:
            velocity = rates[ids]
            motion = (velocity - velocity[:, :1]) * precision(dt)
            local = local + motion
        if kind == 0:
            inverse = p[:, :4].reshape(-1, 2, 2)
            if np.dtype(precision) == np.dtype(np.float64):
                f = np.transpose(local[:, 1:], (0, 2, 1)) @ inverse
                strain = .5 * (np.transpose(f, (0, 2, 1)) @ f - np.eye(2, dtype=precision))
                ratio = np.linalg.norm(np.cross(f[:, :, 0], f[:, :, 1]), axis=1)
                delta = ratio - 1
            else:
                # Production arithmetic: exact start edges give the start strain in float64.
                edges = np.asarray(positions, dtype=np.float64)[ids]
                edges = edges - edges[:, :1]
                start = np.transpose(edges[:, 1:], (0, 2, 1)) @ exact[selected, :4].reshape(-1, 2, 2)
                base = (.5 * (np.transpose(start, (0, 2, 1)) @ start - np.eye(2))).astype(precision)
                start = start.astype(precision)
                moved = np.transpose(motion[:, 1:], (0, 2, 1)) @ inverse
                cross = np.transpose(start, (0, 2, 1)) @ moved
                strain = base + precision(.5) * (cross + np.transpose(cross, (0, 2, 1)) +
                                                 np.transpose(moved, (0, 2, 1)) @ moved)
                f = start + moved
                square = 2 * (strain[:, 0, 0] + strain[:, 1, 1]) + 4 * (
                    strain[:, 0, 0] * strain[:, 1, 1] - strain[:, 0, 1] * strain[:, 1, 0])
                delta = np.where(square > -1, square / (np.sqrt(np.maximum(1 + square, 0)) + 1), -1)
                delta = delta.astype(precision)
                ratio = 1 + delta
            trace = np.trace(strain, axis1=1, axis2=2)
            area, mu, lam = p[:, 4], p[:, 5], p[:, 6]
            normal = np.cross(f[:, :, 0], f[:, :, 1])
            if (ratio <= 0).any():
                raise ValueError("elastic membrane has zero area")
            compression = delta < 0
            barrier = np.where(compression, delta - np.log1p(delta), 0)
            if np.dtype(precision) == np.dtype(np.float32):
                series = delta * delta * (.5 - delta * (1 / 3 - .25 * delta))
                barrier = np.where(compression & (delta > -1e-2), series, barrier)
            energy = area * (mu * (strain * strain).sum(axis=(1, 2)) + .5 * lam * trace**2 + mu * barrier)
            stress = f @ (2 * mu[:, None, None] * strain + lam[:, None, None] * trace[:, None, None] * np.eye(2, dtype=precision))
            exact_path = np.dtype(precision) == np.dtype(np.float64)
            n = normal / (ratio if exact_path else np.linalg.norm(normal, axis=1))[:, None]
            loss = (1 - 1 / ratio) if exact_path else delta / ratio
            stress[:, :, 0] += (mu * compression * loss)[:, None] * np.cross(f[:, :, 1], n)
            stress[:, :, 1] += (mu * compression * loss)[:, None] * np.cross(n, f[:, :, 0])
            corner = area[:, None, None] * (stress @ np.transpose(inverse, (0, 2, 1)))
            gradients = np.stack([-corner.sum(axis=2), corner[:, :, 0], corner[:, :, 1]], axis=1)
        elif kind == 1:
            edge = local[:, 1]
            normal_a = np.cross(edge, local[:, 2])
            normal_b = np.cross(local[:, 3], edge)
            edge2 = (edge * edge).sum(axis=1)
            a2, b2 = (normal_a * normal_a).sum(axis=1), (normal_b * normal_b).sum(axis=1)
            if ((edge2 <= 0) | (a2 <= 0) | (b2 <= 0)).any():
                raise ValueError("elastic hinge has degenerate geometry")
            length = np.sqrt(edge2)
            a, b = normal_a / np.sqrt(a2)[:, None], normal_b / np.sqrt(b2)[:, None]
            angle = np.arctan2((np.cross(a, b) * edge).sum(axis=1) / length, (a * b).sum(axis=1))
            g2, g3 = -normal_a * (length / a2)[:, None], -normal_b * (length / b2)[:, None]
            s1, s2 = (local[:, 2] * edge).sum(axis=1) / edge2, (local[:, 3] * edge).sum(axis=1) / edge2
            gradients = np.stack([-(1 - s1)[:, None] * g2 - (1 - s2)[:, None] * g3,
                                   -s1[:, None] * g2 - s2[:, None] * g3, g2, g3], axis=1)
            delta = angle - p[:, 0]
            energy = p[:, 1] * delta**2
            gradients *= (2 * p[:, 1] * delta)[:, None, None]
            if np.dtype(precision) == np.dtype(np.float32):
                # Production energy: the angle from double start differences plus float32 motion.
                exact_local = np.asarray(positions, dtype=np.float64)[ids]
                exact_local = exact_local - exact_local[:, :1] + motion.astype(np.float64)
                edge64 = exact_local[:, 1]
                a64, b64 = np.cross(edge64, exact_local[:, 2]), np.cross(exact_local[:, 3], edge64)
                angle64 = np.arctan2((np.cross(a64, b64) * edge64).sum(axis=1) / np.linalg.norm(edge64, axis=1),
                                     (a64 * b64).sum(axis=1))
                energy = (exact[selected, 1].astype(np.float64) * (angle64 - exact[selected, 0]) ** 2).astype(precision)
        elif kind == 2:
            edge = local[:, 1]
            length = np.linalg.norm(edge, axis=1)
            if (length <= 0).any():
                raise ValueError("elastic spring has zero length")
            delta = length - p[:, 0]
            energy = .5 * p[:, 1] * delta**2
            g = edge * (p[:, 1] * delta / length)[:, None]
            gradients = np.stack([-g, g], axis=1)
        else:
            first, second = local[:, 1], local[:, 2] - local[:, 1]
            a, b = np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1)
            if ((a <= 0) | (b <= 0)).any():
                raise ValueError("elastic rod has zero segment length")
            t1, t2 = first / a[:, None], second / b[:, None]
            cosine = (t1 * t2).sum(axis=1)
            dc1, dc2 = (t2 - t1 * cosine[:, None]) / a[:, None], (t1 - t2 * cosine[:, None]) / b[:, None]
            energy = p[:, 0] * (1 - cosine)
            gradients = p[:, 0, None, None] * np.stack([dc1, dc2 - dc1, -dc2], axis=1)
        for corner in range(count):
            np.add.at(gradient, ids[:, corner], gradients[:, corner])
        energy_by_kind[str(kind)] = float(energy.sum())
    return {"energy_j": float(sum(energy_by_kind.values())), "energy_by_kind_j": energy_by_kind,
            "gradient_n": gradient, "elastic_gradient_sum_n": gradient.sum(axis=0),
            "damping_present": bool((damping > 0).any())}


def verify_elastic_trace(source, geometry):
    values, manifest = load_records(source, states=True)
    if manifest["status"] != "closed" or manifest["substeps"] != 1:
        raise ValueError("elastic verification requires a closed trace sampled at every physics interval")
    if sha256(geometry) != manifest.get("initial_geometry_sha256"):
        raise ValueError("elastic verification geometry differs from the recorded initial geometry")
    if not (values["energy"][..., E.VALID] == 1).all():
        raise ValueError("elastic verification requires valid production interval readouts")
    with np.load(geometry, allow_pickle=False) as initial:
        rest, elements = initial["rest"], initial["vbd_elements"].copy()
    if rest.ndim == 2:
        rest = np.broadcast_to(rest, (manifest["env_count"],) + rest.shape)
    positions = values["state_PARTICLE_POSITION"].reshape((manifest["steps"],) + rest.shape)
    begin, count = manifest["metadata"]["vbd_particle_begin"], manifest["metadata"]["vbd_vertices"]
    energy_error, gradient_sum = [], []
    for step in range(manifest["steps"]):
        for env in range(manifest["env_count"]):
            result = elastic_quantities(elements, positions[step, env, begin:begin + count])
            energy_error.append(result["energy_j"] - float(values["energy"][step, env, 0, E.END_ELASTIC]))
            gradient_sum.append(np.linalg.norm(result["elastic_gradient_sum_n"]))
    return {"status": "measured", "source_manifest_sha256": sha256(Path(source) / "manifest.json"),
        "geometry_sha256": sha256(geometry), "intervals_per_environment": manifest["steps"],
        "max_elastic_energy_difference_j": float(np.abs(energy_error).max()),
        "max_internal_gradient_sum_n": float(np.max(gradient_sum)),
        "scope": "Independent float64 cooked VBD elastic potential at recorded END positions",
        "claims_full_physics_acceptance": False}
