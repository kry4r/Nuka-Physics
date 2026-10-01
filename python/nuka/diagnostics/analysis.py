"""Model-based predictions and matched physical-time comparisons."""

import numpy as np
from pathlib import Path

from ..energy import EnergyColumn as E
from .capture import load_records, sha256
from .schema import Stage, StageColumn as C


def particle_state_quantities(position, velocity, inverse_mass, gravity):
    """Independently evaluate dynamic particle state quantities in float64."""
    position, velocity = np.asarray(position, dtype=np.float64), np.asarray(velocity, dtype=np.float64)
    inverse_mass, gravity = np.asarray(inverse_mass, dtype=np.float64), np.asarray(gravity, dtype=np.float64)
    if position.shape != velocity.shape or position.ndim < 2 or position.shape[-1] != 3:
        raise ValueError("particle positions and velocities require matching (..., particles, 3) extents")
    if inverse_mass.shape != position.shape[:-1] or gravity.shape != (3,):
        raise ValueError("inverse mass and gravity extents differ from the particle states")
    if not all(np.isfinite(value).all() for value in (position, velocity, inverse_mass, gravity)) or (inverse_mass < 0).any():
        raise ValueError("particle state quantities require finite states and nonnegative inverse mass")
    mass = np.divide(1.0, inverse_mass, out=np.zeros_like(inverse_mass), where=inverse_mass > 0)
    momentum = mass[..., None] * velocity
    return {"mass_kg": mass.sum(axis=-1),
            "kinetic_j": .5 * (mass * np.sum(velocity * velocity, axis=-1)).sum(axis=-1),
            "gravity_j": -(mass * np.sum(position * gravity, axis=-1)).sum(axis=-1),
            "linear_momentum_kg_mps": momentum.sum(axis=-2),
            "angular_momentum_kg_m2ps": np.cross(position, momentum).sum(axis=-2)}


def vbd_discrete_momentum(velocity, previous_velocity, inverse_mass, effective_dt, dt):
    """Evaluate each vertex's actual BE/BDF2 momentum measure independently."""
    velocity = np.asarray(velocity, dtype=np.float64)
    previous_velocity = np.asarray(previous_velocity, dtype=np.float64)
    inverse_mass = np.asarray(inverse_mass, dtype=np.float64)
    effective_dt = np.asarray(effective_dt, dtype=np.float32)
    if velocity.shape != previous_velocity.shape or velocity.ndim < 2 or velocity.shape[-1] != 3:
        raise ValueError("VBD velocities require matching (..., vertices, 3) extents")
    if inverse_mass.shape != velocity.shape[:-1] or effective_dt.shape != inverse_mass.shape:
        raise ValueError("VBD mass and step extents differ from the velocity states")
    if not dt > 0 or not all(np.isfinite(value).all() for value in
            (velocity, previous_velocity, inverse_mass, effective_dt)) or (inverse_mass < 0).any():
        raise ValueError("VBD momentum requires finite states, nonnegative inverse mass and positive dt")
    h = np.float32(dt)
    second_order_h = np.float32(2 / 3) * h
    if not ((effective_dt == h) | (effective_dt == second_order_h)).all():
        raise ValueError("VBD effective steps must match the actual BE or BDF2 step")
    second_order = effective_dt < h
    discrete = np.where(second_order[..., None], 1.5 * velocity - .5 * previous_velocity, velocity)
    mass = np.divide(1.0, inverse_mass, out=np.zeros_like(inverse_mass), where=inverse_mass > 0)
    return (mass[..., None] * discrete).sum(axis=-2)


def verify_particle_trace(source, geometry):
    """Cross-check captured particle-only world readouts against independent state formulas."""
    values, manifest = load_records(source, states=True)
    metadata = manifest["metadata"]
    if manifest["status"] != "closed" or manifest["substeps"] != 1:
        raise ValueError("state verification requires a closed trace sampled at every physics interval")
    if metadata.get("state_system") != "dynamic_particles":
        raise ValueError("world-wide particle readout comparison requires a declared particle-only system")
    if sha256(geometry) != manifest.get("initial_geometry_sha256"):
        raise ValueError("state verification geometry differs from the recorded initial geometry")
    with np.load(geometry, allow_pickle=False) as initial:
        rest = initial["rest"]
        initial_velocity = initial["velocity"].copy()
    if rest.ndim == 2:
        rest = np.broadcast_to(rest, (manifest["env_count"],) + rest.shape)
        initial_velocity = np.broadcast_to(initial_velocity, rest.shape)
    shape = (manifest["steps"],) + rest.shape
    positions = values["state_PARTICLE_POSITION"].reshape(shape)
    velocities = values["state_PARTICLE_VELOCITY"].reshape(shape)
    inverse_mass = values["state_PARTICLE_INV_MASS"].reshape(shape[:-1])
    effective_dt = values["state_VBD_EFFECTIVE_DT"].reshape(shape[:-1])
    if not (values["energy"][..., E.VALID] == 1).all():
        raise ValueError("state verification requires valid production interval readouts")
    if not np.array_equal(inverse_mass, np.broadcast_to(inverse_mass[:1], inverse_mass.shape)):
        raise ValueError("discrete momentum verification requires constant particle masses")
    host = particle_state_quantities(positions, velocities, inverse_mass, metadata["gravity"])
    previous_velocity = np.concatenate([initial_velocity[None], velocities[:-1]])
    discrete = vbd_discrete_momentum(velocities, previous_velocity, inverse_mass, effective_dt, metadata["dt"])
    stages = values["stages"][:, :, 0, Stage.END].astype(np.float64)
    energy = values["energy"][:, :, 0].astype(np.float64)
    comparisons = {
        "kinetic_j": (host["kinetic_j"], energy[..., E.END_KINETIC]),
        "gravity_j": (host["gravity_j"], energy[..., E.END_GRAVITY]),
        "mass_kg": (host["mass_kg"], stages[..., C.MASS]),
        "linear_momentum_kg_mps": (host["linear_momentum_kg_mps"],
                                   stages[..., C.LINEAR_MOMENTUM_X:C.LINEAR_MOMENTUM_Z + 1]),
        "angular_momentum_kg_m2ps": (host["angular_momentum_kg_m2ps"],
                                    stages[..., C.ANGULAR_MOMENTUM_X:C.ANGULAR_MOMENTUM_Z + 1]),
        "discrete_momentum_kg_mps": (discrete,
                                     stages[..., C.VBD_DISCRETE_MOMENTUM_X:C.VBD_DISCRETE_MOMENTUM_Z + 1]),
    }
    errors = {name: {"max_absolute_difference": float(np.abs(a - b).max()),
                     "max_absolute_host_value": float(np.abs(a).max())} for name, (a, b) in comparisons.items()}
    dynamic = inverse_mass > 0
    bdf = ((effective_dt < np.float32(metadata["dt"])) & dynamic).sum(axis=-1)
    dynamic_count = dynamic.sum(axis=-1)
    mixed = (bdf > 0) & (bdf < dynamic_count)
    return {"status": "measured", "source_manifest_sha256": sha256(Path(source) / "manifest.json"),
        "geometry_sha256": sha256(geometry), "intervals_per_environment": manifest["steps"],
        "environments": manifest["env_count"], "scope": "Dynamic particles; independent float64 K/G/P/L and actual mixed BE/BDF2 momentum",
        "errors": errors, "mixed_integrator_intervals": int(mixed.sum()),
        "bdf_vertex_counts_match": bool(np.array_equal(bdf, stages[..., C.VBD_BDF_PARTICLES])),
        "claims_full_physics_acceptance": False,
        "interpretation": "Readout agreement is separate from conservation, solver convergence and trajectory acceptance."}


def predict_linear_modes(mass, stiffness, displacement, velocity, dt, steps, *, integrator="bdf2"):
    """Predict isolated linear modes with BE startup and the actual BDF2 recurrence."""
    mass, stiffness = np.asarray(mass, dtype=np.float64), np.asarray(stiffness, dtype=np.float64)
    if mass.ndim == 1:
        mass = np.diag(mass)
    if mass.shape != stiffness.shape or mass.ndim != 2 or mass.shape[0] != mass.shape[1]:
        raise ValueError("mass and stiffness must be matching square matrices")
    if not np.isfinite(mass).all() or not np.isfinite(stiffness).all() or not dt > 0 or steps < 1:
        raise ValueError("finite matrices, positive dt and steps are required")
    if not np.allclose(mass, mass.T, rtol=1e-12, atol=1e-14) or not np.allclose(stiffness, stiffness.T, rtol=1e-12, atol=1e-14):
        raise ValueError("modal mass and stiffness must be symmetric")
    factor = np.linalg.cholesky(mass)
    first = np.linalg.solve(factor, stiffness)
    matrix = np.linalg.solve(factor, first.T).T
    omega2, basis = np.linalg.eigh(matrix)
    roundoff = np.finfo(np.float64).eps * 64 * max(np.linalg.norm(matrix, ord=2), 1)
    if omega2.min() < -roundoff:
        raise ValueError("linearization has an unstable mode")
    omega2 = np.maximum(omega2, 0)
    q = basis.T @ factor.T @ np.asarray(displacement, dtype=np.float64)
    v = basis.T @ factor.T @ np.asarray(velocity, dtype=np.float64)
    previous_q, previous_v = q.copy(), v.copy()
    energies = [float(.5 * np.sum(v * v + omega2 * q * q))]
    for step in range(steps):
        if integrator == "be" or step == 0:
            h, qhat, vhat = dt, q, v
        elif integrator == "bdf2":
            h = 2 * dt / 3
            qhat, vhat = (4 * q - previous_q) / 3, (4 * v - previous_v) / 3
        else:
            raise ValueError("integrator must be be or bdf2")
        next_q = (qhat + h * vhat) / (1 + h * h * omega2)
        next_v = vhat - h * omega2 * next_q
        previous_q, previous_v, q, v = q, v, next_q, next_v
        energies.append(float(.5 * np.sum(v * v + omega2 * q * q)))
    return {"time_s": np.arange(steps + 1) * dt, "predicted_energy_j": np.asarray(energies),
            "omega_squared": omega2, "scope": "Isolated linear conservative system; no contacts, controls or restarts"}


def compare_runs(sources):
    """Align completed state samples at common physical times without interpolating contacts."""
    if len(sources) < 2:
        raise ValueError("comparison requires at least two independent runs")
    runs = [load_records(source, states=True) for source in sources]
    baseline, base_manifest = runs[0]
    results = []
    for source, (values, manifest) in zip(sources[1:], runs[1:]):
        a, b = base_manifest["metadata"], manifest["metadata"]
        if a.get("physical_input_sha256") is None or a.get("physical_input_sha256") != b.get("physical_input_sha256"):
            raise ValueError("comparison requires identical recorded physical inputs and control law")
        if base_manifest["substeps"] != 1 or manifest["substeps"] != 1:
            raise ValueError("comparison requires state samples at every physics interval")
        if base_manifest["env_count"] != manifest["env_count"]:
            raise ValueError("comparison requires identical environment extents")
        dt_a, dt_b = float(a["dt"]), float(b["dt"])
        time_a = np.arange(1, len(baseline["energy"]) + 1) * dt_a
        candidates = np.rint(time_a / dt_b).astype(np.int64) - 1
        mask = (candidates >= 0) & (candidates < len(values["energy"]))
        mask &= np.isclose((candidates + 1) * dt_b, time_a, rtol=1e-10, atol=1e-12)
        selected_a, selected_b = np.flatnonzero(mask), candidates[mask]
        if not len(selected_a):
            raise ValueError("runs have no common sampled physical times")
        position_key = "state_PARTICLE_POSITION"
        velocity_key = "state_PARTICLE_VELOCITY"
        if any(key not in run for run in (baseline, values) for key in (position_key, velocity_key)):
            raise ValueError("comparison requires captured physical states")
        if any(baseline[key].shape[1:] != values[key].shape[1:] for key in (position_key, velocity_key)):
            raise ValueError("comparison requires matching physical state extents")
        delta_x = baseline[position_key][selected_a].astype(np.float64) - values[position_key][selected_b]
        delta_v = baseline[velocity_key][selected_a].astype(np.float64) - values[velocity_key][selected_b]
        energy_a = baseline["energy"][selected_a, ..., E.END_KINETIC:E.END_ELASTIC + 1].sum(axis=-1)
        energy_b = values["energy"][selected_b, ..., E.END_KINETIC:E.END_ELASTIC + 1].sum(axis=-1)
        results.append({"source": str(source), "common_samples": len(selected_a),
            "common_end_s": float(time_a[selected_a[-1]]), "baseline_dt_s": dt_a, "candidate_dt_s": dt_b,
            "position_rms_m": float(np.sqrt(np.mean(delta_x * delta_x))),
            "position_max_m": float(np.max(np.abs(delta_x))),
            "velocity_rms_mps": float(np.sqrt(np.mean(delta_v * delta_v))),
            "energy_difference_max_j": float(np.max(np.abs(energy_a - energy_b))),
            "baseline_max_momentum_velocity_defect_mps": float(np.max(baseline["stages"][selected_a, ..., Stage.SOLVED, C.VBD_MOMENTUM_VELOCITY_ERROR])),
            "candidate_max_momentum_velocity_defect_mps": float(np.max(values["stages"][selected_b, ..., Stage.SOLVED, C.VBD_MOMENTUM_VELOCITY_ERROR])),
            "claims_speedup": False, "scope": "Matched state differences; independent physical acceptance remains required"})
    return {"baseline": str(sources[0]), "comparisons": results}
