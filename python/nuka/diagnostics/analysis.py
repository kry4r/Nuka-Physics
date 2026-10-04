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
    sources = tuple(Path(source) for source in sources)
    if len(sources) < 2:
        raise ValueError("comparison requires at least two independent runs")
    if len({source.resolve() for source in sources}) != len(sources):
        raise ValueError("comparison requires distinct source captures")
    runs = [load_records(source, states=True) for source in sources]
    baseline, base_manifest = runs[0]
    physical_input = base_manifest["metadata"].get("physical_input_sha256")
    if not isinstance(physical_input, str) or not physical_input:
        raise ValueError("comparison requires a recorded physical input hash")
    position_key, velocity_key = "state_PARTICLE_POSITION", "state_PARTICLE_VELOCITY"
    timesteps, horizons, energy_totals, evidence = [], [], [], []
    baseline_initial, initial_matches = None, []

    def initial_state(source, manifest, values):
        path = source / "initial.npz"
        if not path.is_file() or not manifest.get("initial_geometry_sha256"):
            raise ValueError(f"comparison requires hash-bound initial particle states: {source}")
        initial_hash = sha256(path)
        if initial_hash != manifest["initial_geometry_sha256"]:
            raise ValueError(f"comparison initial state hash mismatch: {source}")
        with np.load(path, allow_pickle=False) as initial:
            position_name = next((name for name in ("positions", "position") if name in initial), None)
            if position_name is None and manifest["metadata"].get("initial_positions_equal_rest") is True and "rest" in initial:
                position_name = "rest"
            mass_name = next((name for name in ("inverse_mass", "inv_mass") if name in initial), None)
            if position_name is None or "velocity" not in initial or mass_name is None:
                raise ValueError(f"comparison requires actual initial positions, velocity and inverse mass: {source}")
            position, velocity, inverse_mass = initial[position_name], initial["velocity"], initial[mass_name]
        count = manifest["env_count"]
        if not position.size or position.size % (count * 3):
            raise ValueError(f"comparison initial particle extent differs from the environment count: {source}")
        particles = position.size // (count * 3)
        shape = (count, particles, 3)
        for value in (position, velocity):
            if (value.size != position.size or value.ndim not in (1, 2, 3) or
                    value.ndim == 2 and value.shape[-1] != 3 or
                    value.ndim == 3 and value.shape != shape):
                raise ValueError(f"comparison initial position/velocity extents must match: {source}")
        if inverse_mass.shape not in ((count * particles,), (count, particles)):
            raise ValueError(f"comparison initial inverse mass extent differs from particle states: {source}")
        if any(value.dtype.kind not in "fiu" or not np.isfinite(value).all() for value in (position, velocity, inverse_mass)):
            raise ValueError(f"comparison requires finite numeric initial particle states: {source}")
        if (inverse_mass < 0).any():
            raise ValueError(f"comparison requires nonnegative initial inverse mass: {source}")
        if any(values[key].size != manifest["steps"] * position.size for key in (position_key, velocity_key)):
            raise ValueError(f"comparison initial and recorded particle extents differ: {source}")
        state = {"positions": position.reshape(shape), "velocity": velocity.reshape(shape),
                 "inverse_mass": inverse_mass.reshape(count, particles)}
        match = {"source": str(source), "initial_geometry_sha256": initial_hash,
                 "positions_field": position_name, "inverse_mass_field": mass_name,
                 "env_count": count, "particles_per_env": particles}
        return state, match

    for source, (values, manifest) in zip(sources, runs):
        metadata = manifest["metadata"]
        if manifest.get("status") != "closed" or manifest.get("stop_reason") != "completed" or manifest.get("error") is not None:
            raise ValueError("comparison requires closed, completed captures without capture errors")
        expected = metadata.get("expected_policy_steps")
        if type(expected) is not int or expected < 1 or expected != manifest["steps"]:
            raise ValueError("comparison requires the complete declared physical horizon")
        if physical_input != metadata.get("physical_input_sha256"):
            raise ValueError("comparison requires identical recorded physical inputs and control law")
        if manifest["substeps"] != 1:
            raise ValueError("comparison requires state samples at every physics interval")
        if base_manifest["env_count"] != manifest["env_count"]:
            raise ValueError("comparison requires identical environment extents")
        dt = metadata.get("dt")
        if type(dt) not in (int, float) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("comparison requires positive finite recorded timesteps")
        dt = float(dt)
        horizon = manifest["steps"] * dt
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("comparison requires a positive finite physical horizon")
        for name in ("duration", "duration_s"):
            if name in metadata and (type(metadata[name]) not in (int, float) or
                    not np.isfinite(metadata[name]) or metadata[name] <= 0 or
                    not np.isclose(horizon, metadata[name], rtol=1e-10, atol=0)):
                raise ValueError("comparison trace does not cover its declared physical duration")
        if timesteps and dt >= timesteps[-1]:
            raise ValueError("comparison timesteps must be distinct and ordered from coarse to fine")
        if horizons and not np.isclose(horizon, horizons[0], rtol=1e-10, atol=0):
            raise ValueError("comparison requires identical complete physical horizons")
        if not (values["energy"][..., E.VALID] == 1).all() or np.any(values["energy_status"]) or np.any(values["env_status"]):
            raise ValueError("comparison requires valid production readouts and environment states")
        if not np.allclose(values["energy"][..., E.DT], dt, rtol=4 * np.finfo(np.float32).eps, atol=0):
            raise ValueError("comparison timestep differs from recorded production intervals")
        if any(key not in values or not values[key].size for key in (position_key, velocity_key)):
            raise ValueError("comparison requires captured physical states")
        if any(baseline[key].shape[1:] != values[key].shape[1:] for key in (position_key, velocity_key)):
            raise ValueError("comparison requires matching physical state extents")
        initial, match = initial_state(source, manifest, values)
        if baseline_initial is None:
            baseline_initial = initial
        for name in ("positions", "velocity", "inverse_mass"):
            if not np.array_equal(initial[name], baseline_initial[name]):
                raise ValueError(f"comparison initial {name} differs from the baseline: {source}")
        initial_matches.append(match)
        physical_arrays = [value for name, value in values.items() if name.startswith("state_")]
        physical_arrays += [values["energy"], values["stages"], values["force_residual_work"]]
        if any(not np.isfinite(value).all() for value in physical_arrays):
            raise ValueError("comparison requires finite physical states and readouts")
        if np.any(values["stages"][..., C.NONFINITE_PARTICLES:C.NONFINITE_LINKS + 1]):
            raise ValueError("comparison rejects recorded nonfinite entity states")
        totals = values["energy"][..., E.END_KINETIC:E.END_ELASTIC + 1].astype(np.float64).sum(axis=-1, dtype=np.float64)
        if not np.isfinite(totals).all():
            raise ValueError("comparison total energy must be representable in float64")
        timesteps.append(dt)
        horizons.append(horizon)
        energy_totals.append(totals)
        evidence.append({"source": str(source), "source_manifest_sha256": sha256(source / "manifest.json"),
            "physical_input_sha256": physical_input, "initial_geometry_sha256": manifest.get("initial_geometry_sha256"),
            "chunks": [{"file": entry["file"], "sha256": entry["sha256"]} for entry in manifest["chunks"]]})

    def common_samples(indices):
        times = np.arange(1, base_manifest["steps"] + 1, dtype=np.float64) * timesteps[0]
        selected = [np.arange(base_manifest["steps"], dtype=np.int64)]
        mask = np.ones(len(times), dtype=bool)
        for index in indices[1:]:
            candidates = np.rint(times / timesteps[index]).astype(np.int64) - 1
            mask &= (candidates >= 0) & (candidates < runs[index][1]["steps"])
            mask &= np.isclose((candidates + 1) * timesteps[index], times, rtol=1e-10, atol=0)
            selected.append(candidates)
        if not mask.any():
            raise ValueError("runs have no common sampled physical times")
        return times[mask], [selection[mask] for selection in selected]

    def rms(delta):
        if not np.isfinite(delta).all():
            raise ValueError("comparison differences must be representable in float64")
        scale = float(np.max(np.abs(delta)))
        return 0.0 if scale == 0 else float(scale * np.sqrt(np.mean((delta / scale) ** 2, dtype=np.float64)))

    def differences(first, second, selected_first, selected_second):
        a, b = runs[first][0], runs[second][0]
        delta_x = a[position_key][selected_first].astype(np.float64) - b[position_key][selected_second].astype(np.float64)
        delta_v = a[velocity_key][selected_first].astype(np.float64) - b[velocity_key][selected_second].astype(np.float64)
        delta_energy = energy_totals[first][selected_first] - energy_totals[second][selected_second]
        return {"position_rms_m": rms(delta_x), "velocity_rms_mps": rms(delta_v),
            "energy_rms_j": rms(delta_energy),
            "position_max_m": float(np.max(np.abs(delta_x))),
            "energy_difference_max_j": float(np.max(np.abs(delta_energy)))}

    results = []
    for index, source in enumerate(sources[1:], start=1):
        times, (selected_a, selected_b) = common_samples((0, index))
        values = runs[index][0]
        results.append({"source": str(source), "source_manifest_sha256": evidence[index]["source_manifest_sha256"],
            "initial_particle_state_matched": True,
            "common_samples": len(times), "common_start_s": float(times[0]), "common_end_s": float(times[-1]),
            "baseline_dt_s": timesteps[0], "candidate_dt_s": timesteps[index],
            **differences(0, index, selected_a, selected_b),
            "baseline_max_momentum_velocity_defect_mps": float(np.max(baseline["stages"][selected_a, ..., Stage.SOLVED, C.VBD_MOMENTUM_VELOCITY_ERROR].astype(np.float64))),
            "candidate_max_momentum_velocity_defect_mps": float(np.max(values["stages"][selected_b, ..., Stage.SOLVED, C.VBD_MOMENTUM_VELOCITY_ERROR].astype(np.float64))),
            "claims_physical_acceptance": False,
            "claims_speedup": False, "scope": "Matched state differences; independent physical acceptance remains required"})
    convergence = {"status": "unmeasured", "reason": "Observed order requires three inputs at h, h/2, h/4"}
    if len(runs) == 3 and np.allclose(np.asarray(timesteps[:-1]) / np.asarray(timesteps[1:]), 2.0, rtol=1e-10, atol=0):
        times, selected = common_samples((0, 1, 2))
        pairs = {"coarse_medium": differences(0, 1, selected[0], selected[1]),
                 "medium_fine": differences(1, 2, selected[1], selected[2]),
                 "coarse_fine": differences(0, 2, selected[0], selected[2])}
        orders = {}
        for name in ("position_rms_m", "velocity_rms_mps", "energy_rms_j"):
            numerator, denominator = pairs["coarse_medium"][name], pairs["medium_fine"][name]
            if numerator == 0 or denominator == 0:
                orders[name] = {"status": "roundoff_limited", "observed_order": None,
                    "reason": "A sampled pairwise difference is zero; a finite observed order cannot be inferred"}
            else:
                orders[name] = {"status": "measured", "observed_order": float(np.log2(numerator) - np.log2(denominator))}
        convergence = {"status": "measured_differences", "dt_s": timesteps, "common_samples": len(times),
            "common_physical_times_s": times.tolist(), "pairwise": pairs, "observed_order": orders,
            "claims_physical_acceptance": False,
            "scope": "All pairs use the intersection of the three sampled physical timelines; no state interpolation"}
    return {"baseline": str(sources[0]), "source_manifest_sha256": evidence[0]["source_manifest_sha256"],
            "source_hashes": evidence, "physical_input_sha256": physical_input, "physical_horizon_s": horizons[0],
            "initial_state_match": {"status": "matched", "method": "np.array_equal without tolerance",
                "fields": ["positions", "velocity", "inverse_mass"], "sources": initial_matches,
                "scope": "Particle initial states only; effective timestep and topology are not inferred from END states"},
            "comparisons": results, "timestep_convergence": convergence, "claims_physical_acceptance": False}
