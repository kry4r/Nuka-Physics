"""Level-1 physics fixtures recorded through the common production diagnostic session."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import nuka
from nuka.author import Scene, SimOptions, materials, morphs, surfaces
from nuka.diagnostics import DiagnosticSession, DiagnosticThresholds
from nuka.diagnostics.capture import sha256, write_json
from nuka.diagnostics.analysis import particle_state_quantities, vbd_discrete_momentum, verify_particle_trace
from nuka.diagnostics.constitutive import elastic_quantities, unpack_elements, verify_elastic_trace
from nuka.diagnostics.report import make_report
from nuka.diagnostics.schema import Stage, StageColumn

from bench_cloth_twist import topology


PARTICLE_FIELDS = (nuka.Field.PARTICLE_POSITION, nuka.Field.PARTICLE_VELOCITY,
                   nuka.Field.PARTICLE_INV_MASS, nuka.Field.VBD_EFFECTIVE_DT)
CONTACT_FIELDS = (nuka.Field.DAT_FAILURE_WITNESS, nuka.Field.CONTACT_SIDE_A_KIND, nuka.Field.CONTACT_SIDE_A_INDEX,
                  nuka.Field.CONTACT_SIDE_B_KIND, nuka.Field.CONTACT_SIDE_B_INDEX)
LINK_FIELDS = (nuka.Field.ARTICULATION_LINK_POSE, nuka.Field.JOINT_POSITION, nuka.Field.JOINT_VELOCITY,
               nuka.Field.JOINT_LIMIT_IMPULSE, nuka.Field.LINK_CONTACT_WRENCH)
# Budget replays keep per-entity states; the fixed-capacity contact slots stay in the record only.
REPLAY_FIELDS = (nuka.Field.PARTICLE_POSITION, nuka.Field.PARTICLE_VELOCITY)
# One POINT_ENDPOINT_TERMS record: the endpoint velocity sums its columns times the referenced particle's velocity.
ENDPOINT_TERM = np.dtype([("kind", "<u4"), ("index", "<u4"), ("columns", "<f4", (3, 3))])
HAND_URDF = (Path(__file__).resolve().parents[1] /
             ".nuka-assets/src/unitree_ros/robots/g1_description/inspire_hand/FTP_right_hand.urdf")
HAND_JOINTS = ("right_thumb_1_joint", "right_thumb_2_joint", "right_index_1_joint", "right_middle_1_joint",
               "right_ring_1_joint", "right_little_1_joint")
# The cooker bounds a simplified collision mesh's sampled one-sided Hausdorff error by 1 mm.
SIMPLIFIED_MESH_ERROR = 1e-3


def production_tolerance():
    tolerance = os.environ.get("NUKA_SOLVER_VEL_TOLERANCE")
    if tolerance is None:
        raise ValueError("set the production NUKA_SOLVER_VEL_TOLERANCE explicitly for reproducibility")
    return float(tolerance)


def policy_steps(duration, dt):
    steps = round(duration / dt)
    if steps < 1 or not math.isclose(steps * dt, duration, rel_tol=1e-9):
        raise ValueError("duration must be a positive integer multiple of dt")
    return steps


def replay_budget_list(spec):
    """Sweep budgets as first:last, inclusive, or as a comma list."""
    if ":" in spec:
        first, last = (int(value) for value in spec.split(":"))
        return list(range(first, last + 1))
    return [int(value) for value in spec.split(",") if value]


def fold_layers(nx, arc_edges):
    """Vertices per flat layer of a lattice folded over ``arc_edges`` hinge chords."""
    if arc_edges < 1 or (nx - arc_edges + 1) % 2 or nx - arc_edges + 1 < 4:
        raise ValueError("a half fold needs nx - arc_edges + 1 even and at least two vertices per layer")
    return (nx - arc_edges + 1) // 2


def ramp_fold(rest, nx, spacing, arc_edges, ramp_edges, gap):
    """Fold a flat lattice over a semicircular hinge, then run the upper layer down a straight ramp to ``gap``.
    Every chord keeps its rest length (a cylindrical isometry), so the cooked rest shape stays flat."""
    side = fold_layers(nx, arc_edges)
    radius = spacing / (2.0 * math.sin(math.pi / (2 * arc_edges)))
    drop = 2.0 * radius - gap
    if not 0.0 < drop < ramp_edges * spacing or ramp_edges >= side - 1:
        raise ValueError("the ramp must descend 2R - gap in fewer chords than the upper layer has")
    slope = math.asin(drop / (ramp_edges * spacing))
    offsets = [(radius * math.cos(math.pi * (k / arc_edges - 0.5)),
                radius * (1.0 + math.sin(math.pi * (k / arc_edges - 0.5)))) for k in range(1, arc_edges + 1)]
    for k in range(1, side):
        run = min(k, ramp_edges)
        offsets.append((-(run * math.cos(slope) + k - run) * spacing, 2.0 * radius - run * spacing * math.sin(slope)))
    lattice = rest.reshape(-1, nx, 3).copy()
    hinge_x, base_z = lattice[:, side - 1, 0].copy(), lattice[:, 0, 2].copy()
    for column, (dx, dz) in enumerate(offsets, start=side):
        lattice[:, column, 0] = hinge_x + dx
        lattice[:, column, 2] = base_z + dz
    return lattice.reshape(-1, 3)


def closest_on_triangles(p, a, b, c):
    """Row-wise closest points on triangles (closest-point regions of Ericson, RTCD 5.1.5)."""
    ab, ac = b - a, c - a
    d1, d2 = np.einsum("ij,ij->i", ab, p - a), np.einsum("ij,ij->i", ac, p - a)
    d3, d4 = np.einsum("ij,ij->i", ab, p - b), np.einsum("ij,ij->i", ac, p - b)
    d5, d6 = np.einsum("ij,ij->i", ab, p - c), np.einsum("ij,ij->i", ac, p - c)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab, t_ac, t_bc = d1 / (d1 - d3), d2 / (d2 - d6), (d4 - d3) / ((d4 - d3) + (d5 - d6))
        v, w = vb / (va + vb + vc), vc / (va + vb + vc)
    regions = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (vc <= 0) & (d1 >= 0) & (d3 <= 0),
               (d6 >= 0) & (d5 <= d6), (vb <= 0) & (d2 >= 0) & (d6 <= 0), (va <= 0) & (d4 >= d3) & (d5 >= d6)]
    closest = [a, b, a + ab * t_ab[:, None], c, a + ac * t_ac[:, None], b + (c - b) * t_bc[:, None]]
    return np.select([r[:, None] for r in regions], closest, a + ab * v[:, None] + ac * w[:, None])


def point_triangle_distance(p, a, b, c):
    """Row-wise distance from points to triangles."""
    return np.linalg.norm(p - closest_on_triangles(p, a, b, c), axis=1)


def layer_distances(positions, nx, ny, side, first, faces):
    """Distance from upper-layer vertices in columns >= ``first`` to the lower-layer surface; vertices whose
    nearest lower vertex lies on the lower boundary are left out, so no sheet edge is measured."""
    column, row = np.arange(nx * ny) % nx, np.arange(nx * ny) // nx
    x = positions.astype(np.float64)
    lower = faces[(column[faces] < side).all(axis=1)]
    lower_vertices = np.flatnonzero(column < side)
    boundary = (row == 0) | (row == ny - 1) | (column == 0) | (column == side - 1)
    points = np.flatnonzero(column >= first)
    nearest = lower_vertices[np.argmin(((x[points, None] - x[None, lower_vertices]) ** 2).sum(axis=-1), axis=1)]
    points = points[~boundary[nearest]]
    a, b, c = x[lower[:, 0]], x[lower[:, 1]], x[lower[:, 2]]
    return np.array([point_triangle_distance(np.broadcast_to(x[p], a.shape), a, b, c).min() for p in points])


def layer_offset(positions, nx, ny, side, first):
    """Lateral offset (m) and yaw (rad) of the flat upper layer relative to the lower layer."""
    lattice = positions.reshape(ny, nx, 3).astype(np.float64)
    lower, upper = lattice[:, 1:side - 1], lattice[:, first:][:, ::-1]
    yaw = [math.atan2(*(layer[:, -1].mean(axis=0) - layer[:, 0].mean(axis=0))[1::-1]) for layer in (lower, upper)]
    return (float(upper[..., 1].mean() - lower[..., 1].mean()),
            float((yaw[1] - yaw[0] + math.pi) % (2 * math.pi) - math.pi))


def base_metadata(args, physical, steps, tolerance, vertices):
    return {**physical, "dt": args.dt, "integrator": args.integrator, "sweeps": args.sweeps,
        "expected_policy_steps": steps, "ogc_contact_capacity": args.contact_capacity,
        "solver_velocity_tolerance_mps": tolerance, "position_iterations": 4,
        "momentum_scope": "vbd_subsystem", "aerodynamic_forces": False, "gravity": [0, 0, args.gravity_z],
        "vbd_particle_begin": 0, "vbd_vertices": vertices,
        "physical_input_sha256": hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest(),
        "render_acceptance": "unmeasured", "build_record": str(args.build_record),
        "binary_hashes": (args.build_record / "binaries.sha256").read_text(), "fixture_sha256": sha256(__file__)}


def cloth_linear_reference(rest, inv_mass, elements, spacing, output):
    """Linearize the independent cooked elastic gradient without removing rigid modes."""
    result = {"status": "invalid", "precision": "float64", "difference_step_m": spacing * 1e-5,
              "error_estimate": "central difference at h and h/2, plus float64 roundoff",
              "negative_eigenvalues_clamped": False, "rigid_modes_removed": False}
    try:
        rest = np.asarray(rest, dtype=np.float64)
        inv_mass = np.asarray(inv_mass, dtype=np.float64).reshape(-1)
        if not np.isfinite(inv_mass).all() or not (inv_mass > 0).all():
            raise ValueError("the full free-cloth modal reference requires finite positive inverse masses")
        _, _, _, damping = unpack_elements(elements)
        if not len(damping) or not np.isfinite(damping).all() or not (damping == damping[0]).all():
            result.update(status="unmeasured", reason="cooked element damping is not one finite uniform beta")
            return result, None
        beta = float(damping[0])
        if beta < 0:
            raise ValueError("negative cooked damping cannot define a passive modal reference")
        base = elastic_quantities(elements, rest, precision=np.float64)
        gradient = base["gradient_n"].reshape(-1)
        matrices, actual_steps, gradient_scale = [], [], np.linalg.norm(gradient)
        for step in (spacing * 1e-5, spacing * .5e-5):
            stiffness = np.empty((rest.size, rest.size), dtype=np.float64)
            actual = []
            for column in range(rest.size):
                plus, minus = rest.copy(), rest.copy()
                plus.flat[column] += step
                minus.flat[column] -= step
                extent = plus.flat[column] - minus.flat[column]
                if not extent > 0:
                    raise ValueError("the central difference step is not representable")
                positive = elastic_quantities(elements, plus, precision=np.float64)["gradient_n"].reshape(-1)
                negative = elastic_quantities(elements, minus, precision=np.float64)["gradient_n"].reshape(-1)
                gradient_scale = max(gradient_scale, np.linalg.norm(positive), np.linalg.norm(negative))
                stiffness[:, column] = (positive - negative) / extent
                actual.append(extent / 2)
            matrices.append(stiffness)
            actual_steps.append(actual)
        stiffness, refined = matrices
        if not all(np.isfinite(value).all() for value in (stiffness, refined, gradient)):
            raise ValueError("nonfinite elastic linearization")
        scale = float(np.linalg.norm(stiffness, ord="fro"))
        roundoff = 64 * np.finfo(np.float64).eps * gradient_scale * math.sqrt(rest.size) / min(actual_steps[1])
        uncertainty = float(4 * np.linalg.norm(stiffness - refined, ord="fro") / 3 + roundoff)
        asymmetry = float(np.linalg.norm(stiffness - stiffness.T, ord="fro"))
        matrix_path = output / "linear_reference_matrices.npz"
        np.savez_compressed(matrix_path, stiffness_raw_npm=stiffness, stiffness_half_step_npm=refined,
            rest_gradient_n=gradient, actual_difference_steps_m=np.asarray(actual_steps))
        result.update(actual_difference_steps_m=actual_steps, stiffness_frobenius_npm=scale,
            stiffness_difference_error_bound_npm=uncertainty, stiffness_symmetry_error_npm=asymmetry,
            stiffness_symmetry_relative_error=asymmetry / scale if scale > 0 else 0,
            elastic_base_energy_j=base["energy_j"], rest_gradient_norm_n=float(np.linalg.norm(gradient)),
            rest_gradient_sum_n=base["elastic_gradient_sum_n"].tolist(), damping_beta_s=beta,
            matrices={"file": matrix_path.name, "sha256": sha256(matrix_path)})
        if asymmetry > 2 * uncertainty:
            raise ValueError("stiffness asymmetry exceeds the measured finite difference uncertainty")
        stiffness = .5 * (stiffness + stiffness.T)
        inverse_sqrt_mass = np.repeat(np.sqrt(inv_mass), 3)
        weighted = inverse_sqrt_mass[:, None] * stiffness * inverse_sqrt_mass[None, :]
        eigenvalues, modes = np.linalg.eigh(weighted)
        spectral_error = uncertainty * float(inv_mass.max())
        result.update(eigenvalues_per_s2=eigenvalues.tolist(), mode_count=rest.size,
            negative_eigenvalues=int((eigenvalues < 0).sum()), eigenvalue_error_bound_per_s2=spectral_error,
            lowest_eigenvalue_per_s2=float(eigenvalues.min()),
            eigenvector_orthogonality_error=float(np.linalg.norm(modes.T @ modes - np.eye(rest.size))))
        if not np.isfinite(eigenvalues).all() or not np.isfinite(modes).all():
            raise ValueError("nonfinite eigendecomposition")
        if (eigenvalues < -spectral_error).any():
            raise ValueError("negative eigenvalues exceed the measured finite difference uncertainty")
        result["status"] = "measured"
        return result, {"stiffness": stiffness, "gradient": gradient, "eigenvalues": eigenvalues,
            "modes": modes, "sqrt_mass": 1 / inverse_sqrt_mass,
            "affine_force": modes.T @ (inverse_sqrt_mass * gradient), "beta": beta}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        result["reason"] = str(error)
        return result, None


def cloth_fixture_metrics(args, session, reason, rest, initial_position, initial_velocity, inv_mass,
                          samples, reference, linear):
    """Keep raw trajectories and every modal contribution alongside explicit measurement limits."""
    positions = np.stack([initial_position, *[s["state_PARTICLE_POSITION"].reshape(-1, 3) for s in samples]]).astype(np.float64)
    velocities = np.stack([initial_velocity, *[s["state_PARTICLE_VELOCITY"].reshape(-1, 3) for s in samples]]).astype(np.float64)
    rest = np.asarray(rest, dtype=np.float64)
    h = float(np.float32(args.dt))
    times = np.arange(len(positions), dtype=np.float64) * h
    mass = 1 / np.asarray(inv_mass, dtype=np.float64).reshape(-1)
    finite = np.isfinite(positions).all(axis=(1, 2)) & np.isfinite(velocities).all(axis=(1, 2))
    flags = [np.asarray(s["env_status"]).tolist() for s in samples]
    first_failure = next((i + 1 for i, s in enumerate(samples) if np.any(s["env_status"])), None)
    first_nonfinite = next((int(i) for i in np.flatnonzero(~finite)), None)
    field_finite = {field.name: [bool(np.isfinite(s["state_" + field.name]).all()) for s in samples]
                    for field in PARTICLE_FIELDS}
    first_field_nonfinite = next((i + 1 for i in range(len(samples))
        if any(not values[i] for values in field_finite.values())), None)
    constant_mass = all(np.array_equal(s["state_PARTICLE_INV_MASS"].reshape(-1), inv_mass.reshape(-1)) for s in samples)
    metrics = {"policy_steps": session.steps, "stop_reason": reason,
        "production_status": {"status": "completed" if reason == "completed" else "failed_or_incomplete",
            "first_env_status_policy_step": first_failure, "env_status": flags,
            "energy_status": [np.asarray(s["energy_status"]).tolist() for s in samples],
            "first_nonfinite_state_policy_step": first_nonfinite, "finite_state": finite.tolist(),
            "particle_field_finite": field_finite, "first_nonfinite_particle_field_policy_step": first_field_nonfinite,
            "cooked_masses_unchanged": constant_mass},
        "window": {"first_policy_step": 0, "last_policy_step": session.steps,
            "nominal_duration_s": session.steps * args.dt, "physical_float32_step_s": h,
            "recorded_duration_s": float(times[-1]), "expected_policy_steps": session.manifest["metadata"]["expected_policy_steps"]},
        "bindings": {"manifest_sha256": sha256(args.output / "manifest.json"),
            "initial_geometry_sha256": session.manifest["initial_geometry_sha256"],
            "physical_input_sha256": session.manifest["metadata"]["physical_input_sha256"],
            "fixture_sha256": session.manifest["metadata"]["fixture_sha256"],
            "binary_hashes": session.manifest["metadata"]["binary_hashes"]},
        "modal_reference": reference, "claims_full_physics_acceptance": False,
        "angular_momentum_convergence": {"status": "unmeasured", "required_timesteps": [args.dt, args.dt / 2, args.dt / 4],
            "reason": "full nonlinear spin and angular momentum need matched h, h/2, h/4 production runs"}}
    curves = {"time_s": times, "positions": positions, "physical_velocity": velocities,
              "finite_state": finite}
    momentum, angular, angular_com, centers, discrete, discrete_production = [], [], [], [], [], []
    for index, (position, velocity) in enumerate(zip(positions, velocities)):
        if not finite[index]:
            for values in (momentum, angular, angular_com, centers, discrete):
                values.append(np.full(3, np.nan))
            continue
        values = particle_state_quantities(position, velocity, inv_mass.reshape(-1), (0, 0, 0))
        center = (mass[:, None] * position).sum(axis=0) / mass.sum()
        momentum.append(values["linear_momentum_kg_mps"])
        angular.append(values["angular_momentum_kg_m2ps"])
        angular_com.append(np.cross(position - center, mass[:, None] * velocity).sum(axis=0))
        centers.append(center)
        if index == 0:
            discrete.append(momentum[-1])
        else:
            try:
                discrete.append(vbd_discrete_momentum(velocity, velocities[index - 1], inv_mass.reshape(-1),
                    samples[index - 1]["state_VBD_EFFECTIVE_DT"].reshape(-1), args.dt))
            except ValueError:
                discrete.append(np.full(3, np.nan))
    for sample in samples:
        end = sample["stages"][0, 0, Stage.END]
        discrete_production.append(end[StageColumn.VBD_DISCRETE_MOMENTUM_X:StageColumn.VBD_DISCRETE_MOMENTUM_Z + 1])
    momentum, centers, discrete = np.asarray(momentum), np.asarray(centers), np.asarray(discrete)
    curves.update(linear_momentum_kg_mps=momentum, angular_momentum_origin_kg_m2ps=np.asarray(angular),
        angular_momentum_com_kg_m2ps=np.asarray(angular_com), center_of_mass_m=centers,
        bdf_discrete_momentum_kg_mps=discrete,
        production_bdf_discrete_momentum_kg_mps=np.asarray(discrete_production))
    drift = np.linalg.norm(momentum[1:] - momentum[0], axis=1)
    baseline = float(np.linalg.norm(momentum[0]))
    relative_rate = drift / (baseline * times[1:]) if baseline > 0 else np.full(len(drift), np.nan)
    drift_measured = bool(len(drift) and constant_mass and np.isfinite(relative_rate).all())
    complete = reason == "completed" and session.steps == session.manifest["metadata"]["expected_policy_steps"]
    threshold = session.thresholds.momentum_relative_per_second
    curves["linear_momentum_relative_drift_per_s"] = relative_rate
    curves["center_of_mass_translation_error_m"] = centers - centers[0] - times[:, None] * momentum[0] / mass.sum()
    curves["displacement_mass_rms_m"] = np.sqrt((mass[None, :, None] * (positions - initial_position) ** 2).sum(axis=(1, 2)) / mass.sum())
    metrics["linear_momentum"] = {"status": "unmeasured" if not drift_measured else
        "passed" if complete and (relative_rate < threshold).all() else "failed",
        "initial_kg_mps": momentum[0].tolist(), "threshold_relative_per_s": threshold,
        "max_relative_drift_per_s": float(relative_rate.max()) if drift_measured else None,
        "normalization": "norm(P(t)-P(0))/(norm(P(0))*t), every recorded prefix",
        "reason": None if baseline > 0 else "initial linear momentum is zero; relative drift is unmeasured"}
    metrics["displacement"] = {"final_mass_rms_m": float(curves["displacement_mass_rms_m"][-1]),
        "max_center_translation_error_m": float(np.linalg.norm(curves["center_of_mass_translation_error_m"], axis=1).max())}
    metrics["discrete_momentum"] = {"status": "measured" if samples and constant_mass and np.isfinite(discrete).all() else "unmeasured",
        "max_change_kg_mps": float(np.linalg.norm(discrete - discrete[0], axis=1).max()),
        "scope": "independent actual per-vertex BE/BDF2 momentum; original physical P drift is retained"}
    metrics["modal_total_energy"] = {"status": "unmeasured", "threshold_relative_error": .05}
    metrics["per_mode_energy"] = {"status": "unmeasured", "threshold_relative_error": .05,
        "reason": "a valid full modal reference is required"}
    metrics["resolvable_elastic_mode_energy"] = {"status": "unmeasured",
        "reason": "a valid full modal reference is required", "scope": "reference observability only"}
    if linear is not None:
        modes, sqrt_mass = linear["modes"], linear["sqrt_mass"]
        eigenvalues, force, beta = linear["eigenvalues"], linear["affine_force"], linear["beta"]
        actual_q = ((positions - rest).reshape(len(positions), -1) * sqrt_mass) @ modes
        actual_w = (velocities.reshape(len(velocities), -1) * sqrt_mass) @ modes
        predicted_q, predicted_w = [actual_q[0].copy()], [actual_w[0].copy()]
        for index in range(1, len(positions)):
            second_order = args.integrator == "bdf2" and index > 1
            ht = 2 * h / 3 if second_order else h
            offset = (predicted_q[-1] - predicted_q[-2]) / 3 if second_order else np.zeros_like(predicted_q[-1])
            velocity_target = 4 * predicted_w[-1] / 3 - predicted_w[-2] / 3 if second_order else predicted_w[-1]
            target = predicted_q[-1] + offset + ht * velocity_target
            damping = beta * eigenvalues * ht * ht / h
            q = (target + damping * predicted_q[-1] - ht * ht * force) / (1 + eigenvalues * ht * ht + damping)
            w = (q - predicted_q[-1] - offset) / ht
            predicted_q.append(q)
            predicted_w.append(w)
        predicted_q, predicted_w = np.asarray(predicted_q), np.asarray(predicted_w)
        actual_energy = .5 * actual_w ** 2 + .5 * eigenvalues * actual_q ** 2 + force * actual_q
        predicted_energy = .5 * predicted_w ** 2 + .5 * eigenvalues * predicted_q ** 2 + force * predicted_q
        actual_total, predicted_total = actual_energy.sum(axis=1), predicted_energy.sum(axis=1)
        relative_error = np.abs(actual_total - predicted_total) / np.abs(predicted_total)
        energy_finite = constant_mass and np.isfinite(relative_error).all() and np.isfinite(actual_energy).all() and np.isfinite(predicted_energy).all()
        curves.update(actual_mode_displacement=actual_q, actual_mode_physical_velocity=actual_w,
            predicted_mode_displacement=predicted_q, predicted_mode_velocity=predicted_w,
            actual_mode_energy_j=actual_energy, predicted_mode_energy_j=predicted_energy,
            actual_linear_total_energy_j=actual_total, predicted_linear_total_energy_j=predicted_total,
            modal_total_energy_relative_error=relative_error, eigenvalues_per_s2=eigenvalues,
            modes=modes, mass_sqrt=sqrt_mass, stiffness_npm=linear["stiffness"], rest_gradient_n=linear["gradient"],
            affine_modal_force=force)
        metrics["modal_total_energy"].update({"status": "unmeasured" if not energy_finite else
            "passed" if complete and (relative_error < .05).all() else "failed",
            "max_relative_error": float(relative_error.max()) if energy_finite else None,
            "mode_count": len(eigenvalues), "energy_definition": "all-mode kinetic + g0 dot displacement + 0.5 displacement.T K displacement",
            "elastic_base_energy_j": reference["elastic_base_energy_j"],
            "normalization": "absolute error divided by absolute predicted linear total energy at every sample",
            "scope": "linearized cooked material, uniform beta; nonlinear spin still requires timestep convergence"})
        denominator = np.abs(predicted_energy)
        denominator_valid = np.isfinite(denominator) & (denominator > 0)
        mode_error = np.divide(np.abs(actual_energy - predicted_energy), denominator,
            out=np.full_like(denominator, np.nan), where=denominator_valid)
        spectral_error = reference["eigenvalue_error_bound_per_s2"]
        resolvable = eigenvalues > spectral_error
        translation_only = args.stretch_strain == 0 and not np.any(np.asarray(args.spin))
        per_mode = []
        for mode in range(len(eigenvalues)):
            measured = bool(samples and constant_mass and denominator_valid[:, mode].all() and
                np.isfinite(mode_error[:, mode]).all())
            decay_observable = bool(measured and resolvable[mode] and not translation_only)
            status = ("unmeasured" if not measured or translation_only else
                "passed" if complete and (mode_error[:, mode] < .05).all() else "failed")
            per_mode.append({"mode": mode, "eigenvalue_per_s2": float(eigenvalues[mode]), "status": status,
                "max_relative_error": float(mode_error[:, mode].max()) if measured else None,
                "invalid_reference_energy_samples": int((~denominator_valid[:, mode]).sum()),
                "resolvable_elastic_mode": bool(resolvable[mode]),
                "decay_status": status if decay_observable else "unmeasured",
                "reason": "zero or nonfinite reference energy, nonfinite error, no intervals or changed mass"
                    if not measured else "translation-only input does not establish modal decay excitation"
                    if translation_only else None})
        metrics["per_mode_energy"] = {"status": "failed" if any(mode["status"] == "failed" for mode in per_mode)
            else "unmeasured" if any(mode["status"] == "unmeasured" for mode in per_mode) else "passed",
            "threshold_relative_error": .05, "normalization": "absolute per-mode energy error divided by absolute corresponding predicted mode energy",
            "zero_or_nonfinite_denominator": "unmeasured; no floor", "translation_only_input": bool(translation_only),
            "modes": per_mode, "scope": "every reference mode retained; decay requires excited resolvable elastic modes"}
        elastic_actual = actual_energy[:, resolvable].sum(axis=1)
        elastic_predicted = predicted_energy[:, resolvable].sum(axis=1)
        elastic_denominator = np.abs(elastic_predicted)
        elastic_valid = np.isfinite(elastic_denominator) & (elastic_denominator > 0)
        elastic_error = np.divide(np.abs(elastic_actual - elastic_predicted), elastic_denominator,
            out=np.full_like(elastic_denominator, np.nan), where=elastic_valid)
        elastic_measured = bool(samples and constant_mass and resolvable.any() and elastic_valid.all() and
            np.isfinite(elastic_error).all())
        metrics["resolvable_elastic_mode_energy"] = {"status": "measured" if elastic_measured else "unmeasured",
            "mode_count": int(resolvable.sum()), "outside_range_mode_count": int((~resolvable).sum()),
            "eigenvalue_lower_bound_per_s2": spectral_error,
            "max_relative_error": float(elastic_error.max()) if elastic_measured else None,
            "normalization": "absolute aggregate energy error divided by absolute predicted aggregate energy",
            "scope": "eigenvalue > measured spectral error; reference observability only, no modes removed from prediction or individual acceptance",
            "replaces_per_mode_acceptance": False, "claims_full_physics_acceptance": False}
        curves.update(mode_energy_relative_error=mode_error, mode_reference_energy_denominator_valid=denominator_valid,
            resolvable_elastic_modes=resolvable, resolvable_elastic_actual_energy_j=elastic_actual,
            resolvable_elastic_predicted_energy_j=elastic_predicted,
            resolvable_elastic_energy_relative_error=elastic_error)
    path = args.output / "fixture_curves.npz"
    np.savez_compressed(path, **curves)
    metrics["curves"] = {"file": path.name, "sha256": sha256(path)}
    return metrics


def run_s1_fixture(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    steps = policy_steps(args.duration, args.dt)
    tolerance = production_tolerance()
    if args.gravity_z != 0:
        raise ValueError("the free-cloth invariant fixture requires zero gravity")
    if args.nx < 2 or args.ny < 2 or not np.isfinite(args.spacing) or not args.spacing > 0:
        raise ValueError("the free cloth needs a finite positive spacing and at least two vertices per axis")
    if not np.isfinite(args.damping) or args.damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    if not np.isfinite([*args.translation, *args.spin, args.stretch_strain, args.height]).all() or args.stretch_strain <= -1:
        raise ValueError("initial motion and height must be finite and stretch strain must exceed -1")
    physical = {key: getattr(args, key) for key in ("nx", "ny", "spacing", "height", "duration", "density", "stretch",
        "poisson", "bend", "thickness", "friction", "gravity_z", "damping", "translation", "spin", "stretch_strain")}
    metadata = {"fixture": args.fixture.upper(), **base_metadata(args, physical, steps, tolerance, args.nx * args.ny),
        "state_system": "dynamic_particles", "required_physical_systems": ["vbd"], "required_ccd_contact_domains": [],
        "boundary": "Free flat cooked cloth without external contact objects; prescribed translation, spin and in-plane dilation",
        "reference": "Full mass-orthonormal linearization of float64 cooked elastic gradients; BE startup, then selected integrator"}
    grid = morphs.Grid(args.nx, args.ny, args.spacing, origin=(0, 0, args.height))
    scene = Scene(SimOptions(dt=args.dt, gravity=(0, 0, 0),
        cloth_integrator=0 if args.integrator == "bdf2" else 1, solver_vel_iters=args.sweeps, solver_pos_iters=4,
        ogc_contact_capacity=args.contact_capacity, baumgarte_max_velocity=0))
    scene.add_entity(grid, materials.Cloth.VBD(areal_density=args.density, friction=args.friction,
        stretch_stiffness=args.stretch, poisson=args.poisson, bend_stiffness=args.bend, thickness=args.thickness,
        damping=args.damping), surfaces.Cloth(free=True))
    device = nuka.Device.create(0)
    world = scene.build(device)
    session, samples, reason = None, [], "completed"
    try:
        world.set_gravity_z(0)
        rest = np.asarray(world.download_field(nuka.Field.PARTICLE_POSITION)).reshape(-1, 3).copy()
        cooked_velocity = np.asarray(world.download_field(nuka.Field.PARTICLE_VELOCITY)).reshape(-1, 3).copy()
        inv_mass = np.asarray(world.download_field(nuka.Field.PARTICLE_INV_MASS)).reshape(-1).copy()
        elements = np.asarray(world.download_field(nuka.Field.VBD_ELEMENTS), dtype=np.uint32).reshape(-1, 16).copy()
        if not np.allclose(rest, grid.rest_positions(), atol=1e-6):
            raise ValueError("the cooked cloth lattice differs from the authored grid")
        if not np.isfinite(inv_mass).all() or not (inv_mass > 0).all():
            raise ValueError("the free cloth requires finite positive cooked inverse masses")
        mass = 1 / inv_mass.astype(np.float64)
        center = (rest.astype(np.float64) * mass[:, None]).sum(axis=0) / mass.sum()
        offset = rest.astype(np.float64) - center
        positions = (offset * (1 + args.stretch_strain) + center).astype(rest.dtype)
        velocity = (np.asarray(args.translation) + np.cross(np.asarray(args.spin), offset)).astype(cooked_velocity.dtype)
        world.upload_field(nuka.Field.PARTICLE_POSITION, np.ascontiguousarray(positions))
        world.upload_field(nuka.Field.PARTICLE_VELOCITY, np.ascontiguousarray(velocity))
        audit = getattr(nuka.Field, "VBD_SOLVE_AUDIT", None)
        fields = PARTICLE_FIELDS + CONTACT_FIELDS + (() if audit is None else (audit,))
        session = DiagnosticSession(world, args.output, metadata, chunk_steps=args.chunk_steps,
            state_fields=fields, thresholds=DiagnosticThresholds(velocity_tolerance_mps=tolerance))
        if session.substeps != 1:
            raise ValueError("the free-cloth modal fixture requires one recorded substep per policy step")
        faces, edges = topology(args.nx, args.ny)
        np.savez_compressed(args.output / "initial.npz", rest=positions, positions=positions, velocity=velocity,
            flat_rest=rest, inv_mass=inv_mass, vbd_elements=elements, faces=faces, edges=edges)
        session.manifest["initial_geometry_sha256"] = sha256(args.output / "initial.npz")
        write_json(args.output / "inputs.json", {"parameters": {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}, "metadata": metadata})
        reference, linear = cloth_linear_reference(rest, inv_mass, elements, args.spacing, args.output)
        write_json(args.output / "linear_reference.json", reference)
        try:
            for step in range(steps):
                if step + 1 == args.replay_step:
                    replayed = session.replay_budgets(replay_budget_list(args.replay_budgets), state_fields=fields)
                    print(json.dumps({"budget_replay": str(replayed)}), flush=True)
                sample = session.step()
                samples.append(sample)
                if np.any(sample["env_status"]):
                    reason = "physics_failure"
                    break
                if any(not np.isfinite(sample["state_" + field.name]).all() for field in PARTICLE_FIELDS):
                    reason = "nonfinite_state"
                    break
                if step + 1 == args.replay_step and args.replay_stop:
                    reason = "replay_complete"
                    break
        except BaseException as error:
            reason = "exception"
            session.close(reason, error=repr(error))
            raise
        finally:
            session.close(reason)
            metrics = cloth_fixture_metrics(args, session, reason, rest, positions, velocity, inv_mass,
                samples, reference, linear)
            write_json(args.output / "fixture_metrics.json", metrics)
            if session.steps:
                for filename, verify in (("particle_state_check.json", verify_particle_trace),
                                          ("elastic_state_check.json", verify_elastic_trace)):
                    try:
                        check = verify(args.output, args.output / "initial.npz")
                    except (ValueError, FloatingPointError) as error:
                        check = {"status": "unmeasured", "reason": str(error), "claims_full_physics_acceptance": False}
                    write_json(args.output / filename, check)
                try:
                    report = make_report(args.output, args.output / "report")
                    metrics["production_report"] = {"status": "measured", "file": "report/report.json",
                        "sha256": sha256(args.output / "report/report.json"),
                        "passes_full_physics_acceptance": report["passes_full_physics_acceptance"]}
                except (ValueError, ImportError, FloatingPointError) as error:
                    metrics["production_report"] = {"status": "unmeasured", "reason": str(error)}
            else:
                metrics["production_report"] = {"status": "unmeasured", "reason": "no completed production intervals"}
            write_json(args.output / "fixture_metrics.json", metrics)
            print(json.dumps({"output": str(args.output), "policy_steps": session.steps, "stop_reason": reason,
                "claims_full_physics_acceptance": False}), flush=True)
    except BaseException as error:
        if session is not None:
            session.close("exception", error=repr(error))
        raise
    finally:
        world.destroy()
        device.close()


def run_s5(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    steps = policy_steps(args.duration, args.dt)
    tolerance = production_tolerance()
    physical = {key: getattr(args, key) for key in ("nx", "ny", "spacing", "arc_edges", "ramp_edges", "gap", "height",
        "duration", "density", "stretch", "poisson", "bend", "thickness", "friction", "table_friction", "gravity_z")}
    side = fold_layers(args.nx, args.arc_edges)
    flat = side - 1 + args.arc_edges + args.ramp_edges
    first = flat + (args.nx - flat) // 2
    grid = morphs.Grid(args.nx, args.ny, args.spacing, origin=(0, 0, args.height))
    planned = ramp_fold(grid.rest_positions().astype(np.float64), args.nx, args.spacing, args.arc_edges,
                        args.ramp_edges, args.gap)
    table_center = 0.5 * (planned[:, :2].min(axis=0) + planned[:, :2].max(axis=0))
    if np.any(0.5 * np.ptp(planned[:, :2], axis=0) + 0.02 > 0.2):
        raise ValueError("the folded cloth must lie 2 cm inside the 0.4 m table top")
    metadata = {"fixture": "S5", **base_metadata(args, physical, steps, tolerance, args.nx * args.ny),
        "state_system": "dynamic_particles",
        "required_physical_systems": ["vbd"],
        "required_ccd_contact_domains": ["particle_mesh_self", "particle_static"],
        "boundary": "Static table top at z = 0; the cloth starts at rest folded over a semicircular hinge and a straight "
                    "ramp onto a flat upper layer, its lower layer at the given height, over a flat cooked rest shape",
        "gap_scope": "Exact vertex-to-triangle distance from the outer half of the flat upper layer to the lower layer; "
                     "vertices nearest the lower-layer boundary are left out",
        "layer_columns": {"lower": [0, side], "flat_upper": [flat, args.nx], "measured": [first, args.nx]},
        "reference": {"layer_gap_m": args.thickness, "source": "two touching half-thickness contact shells"}}
    scene = Scene(SimOptions(dt=args.dt, gravity=(0, 0, args.gravity_z),
        cloth_integrator=0 if args.integrator == "bdf2" else 1, solver_vel_iters=args.sweeps, solver_pos_iters=4,
        ogc_contact_capacity=args.contact_capacity, baumgarte_max_velocity=0))
    scene.add_entity(grid, materials.Cloth.VBD(areal_density=args.density, friction=args.friction,
        stretch_stiffness=args.stretch, poisson=args.poisson, bend_stiffness=args.bend, thickness=args.thickness),
        surfaces.Cloth(free=True))
    scene.add_entity(morphs.Box((0.2, 0.2, 0.01), pos=(float(table_center[0]), float(table_center[1]), -0.01)),
        materials.Rigid(static=True, friction=args.table_friction))
    device = nuka.Device.create(0)
    world = scene.build(device)
    session = None
    try:
        world.set_gravity_z(args.gravity_z)
        rest = np.asarray(world.download_field(nuka.Field.PARTICLE_POSITION)).reshape(-1, 3).copy()
        if not np.allclose(rest, grid.rest_positions(), atol=1e-6):
            raise ValueError("the cooked cloth lattice differs from the authored grid")
        folded = ramp_fold(rest, args.nx, args.spacing, args.arc_edges, args.ramp_edges, args.gap)
        world.upload_field(nuka.Field.PARTICLE_POSITION, np.ascontiguousarray(folded))
        velocity = np.asarray(world.download_field(nuka.Field.PARTICLE_VELOCITY)).reshape(-1, 3).copy()
        inv_mass = np.asarray(world.download_field(nuka.Field.PARTICLE_INV_MASS)).copy()
        elements = np.asarray(world.download_field(nuka.Field.VBD_ELEMENTS), dtype=np.uint32).reshape(-1, 16).copy()
        session = DiagnosticSession(world, args.output, metadata, chunk_steps=args.chunk_steps,
            state_fields=PARTICLE_FIELDS + CONTACT_FIELDS,
            thresholds=DiagnosticThresholds(velocity_tolerance_mps=tolerance))
        faces, edges = topology(args.nx, args.ny)
        np.savez_compressed(args.output / "initial.npz", rest=folded, positions=folded, velocity=velocity, vbd_elements=elements,
                            faces=faces, edges=edges, inv_mass=inv_mass, flat_rest=rest)
        session.manifest["initial_geometry_sha256"] = sha256(args.output / "initial.npz")
        write_json(args.output / "inputs.json", {"parameters": vars(args) | {"output": str(args.output),
            "build_record": str(args.build_record)}, "metadata": metadata})
        reason, positions, velocities = "completed", folded, velocity
        for step in range(steps):
            if step + 1 == args.replay_step:
                replayed = session.replay_budgets(replay_budget_list(args.replay_budgets),
                                                  state_fields=REPLAY_FIELDS)
                print(json.dumps({"budget_replay": str(replayed)}), flush=True)
            sample = session.step()
            positions = sample["state_PARTICLE_POSITION"].reshape(-1, 3)
            velocities = sample["state_PARTICLE_VELOCITY"].reshape(-1, 3)
            if np.any(sample["env_status"]):
                reason = "physics_failure"
                break
            if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                reason = "nonfinite_state"
                break
            if step + 1 == args.replay_step and args.replay_stop:
                reason = "replay_complete"
                break
        session.close(reason)
        gaps = layer_distances(positions, args.nx, args.ny, side, first, faces)
        offset, yaw = layer_offset(positions, args.nx, args.ny, side, flat)
        mass = np.divide(1.0, inv_mass, out=np.zeros(inv_mass.shape), where=inv_mass > 0)
        median = float(np.median(gaps)) if gaps.size else None
        metrics = {"policy_steps": session.steps, "stop_reason": reason, "reference_layer_gap_m": args.thickness,
            "layer_gap_median_m": median, "layer_gap_min_m": float(gaps.min()) if gaps.size else None,
            "layer_gap_max_m": float(gaps.max()) if gaps.size else None, "layer_gap_samples": int(gaps.size),
            "layer_gap_relative_error": abs(median / args.thickness - 1.0) if gaps.size else None,
            "layer_gap_within_10_percent": float(np.mean(np.abs(gaps / args.thickness - 1.0) <= 0.1)) if gaps.size
                else None,
            "upper_layer_offset_y_m": offset, "upper_layer_yaw_rad": yaw,
            "table_clearance_min_m": float(positions[:, 2].min()),
            "final_speed_max_mps": float(np.linalg.norm(velocities, axis=1).max()),
            "final_kinetic_energy_j": float(0.5 * (mass * (velocities.astype(np.float64) ** 2).sum(axis=1)).sum()),
            "claims_full_physics_acceptance": False}
        write_json(args.output / "fixture_metrics.json", metrics)
        print(json.dumps({"output": str(args.output), **metrics}), flush=True)
    except BaseException as error:
        if session is not None:
            session.close("exception", error=repr(error))
        raise
    finally:
        world.destroy()
        device.close()


def urdf_vector(element, key, default="0 0 0"):
    text = element.get(key) if element is not None else None
    return np.array((text or default).split(), dtype=np.float64)


def hand_mesh(urdf, filename):
    """The vendor hand URDFs name their meshes relative to g1_description."""
    path = (Path(urdf).resolve().parent.parent / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def binary_stl(path):
    """Triangles (n, 3, 3) of a binary STL file."""
    data = Path(path).read_bytes()
    count = int(np.frombuffer(data, np.uint32, 1, 80)[0]) if len(data) >= 84 else -1
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path} is not a binary STL")
    record = np.dtype([("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(data, record, count, 84)["corners"].astype(np.float64)


def segment_distance(p0, p1, q0, q1):
    """Row-wise distance between segments p0 p1 and nondegenerate q0 q1 (Ericson, RTCD 5.1.9)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    dot = lambda u, v: np.einsum("...i,...i->...", u, v)
    a, b, c, e, f = dot(d1, d1), dot(d1, d2), dot(d1, r), dot(d2, d2), dot(d2, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = a * e - b * b
        s = np.where(denominator > 0, np.clip((b * f - c * e) / denominator, 0, 1), 0)
        t = (b * s + f) / e
        s = np.where(t < 0, np.where(a > 0, np.clip(-c / a, 0, 1), 0),
                     np.where(t > 1, np.where(a > 0, np.clip((b - c) / a, 0, 1), 0), s))
    return np.linalg.norm(r + d1 * s[..., None] - d2 * np.clip(t, 0, 1)[..., None], axis=-1)


class Hand:
    """Host kinematics of a URDF hand: revolute and fixed joints, binary STL collisions and link inertials."""

    def __init__(self, urdf):
        robot = ET.parse(urdf).getroot()
        self.joints, self.meshes, self.inertials, self.upper, self.effort = {}, {}, {}, {}, {}
        for joint in robot.findall("joint"):
            kind, mimic, limit = joint.get("type"), joint.find("mimic"), joint.find("limit")
            if kind not in ("revolute", "fixed"):
                raise ValueError(f"unsupported joint type {kind}")
            axis = urdf_vector(joint.find("axis"), "xyz", "1 0 0")
            self.joints[joint.find("child").get("link")] = {
                "name": joint.get("name"), "parent": joint.find("parent").get("link"), "revolute": kind == "revolute",
                "xyz": urdf_vector(joint.find("origin"), "xyz"),
                "axis": axis / np.linalg.norm(axis) if kind == "revolute" else axis,
                "rotation": Rotation.from_euler("xyz", urdf_vector(joint.find("origin"), "rpy")).as_matrix(),
                "mimic": None if mimic is None else (mimic.get("joint"), float(mimic.get("multiplier", "1")),
                                                    float(mimic.get("offset", "0")))}
            if kind == "revolute" and mimic is None:
                self.upper[joint.get("name")] = float(limit.get("upper"))
                self.effort[joint.get("name")] = float(limit.get("effort"))
        roots = {joint["parent"] for joint in self.joints.values()} - set(self.joints)
        if len(roots) != 1:
            raise ValueError("the hand must form one tree")
        self.root = roots.pop()
        for link in robot.findall("link"):
            name, inertial = link.get("name"), link.find("inertial")
            if inertial is not None:
                self.inertials[name] = (float(inertial.find("mass").get("value")),
                                        urdf_vector(inertial.find("origin"), "xyz"))
            triangles = []
            for collision in link.findall("collision"):
                origin, mesh = collision.find("origin"), collision.find("geometry/mesh")
                if (mesh is None or np.any(urdf_vector(origin, "xyz")) or np.any(urdf_vector(origin, "rpy")) or
                        np.any(urdf_vector(mesh, "scale", "1 1 1") != 1.0)):
                    raise ValueError(f"{name}: expected unscaled mesh collisions at the link origin")
                triangles.append(binary_stl(hand_mesh(urdf, mesh.get("filename"))))
            if triangles:
                self.meshes[name] = np.concatenate(triangles)

    def angles(self, positions):
        """Every revolute joint's value; a mimic joint follows its source unless it is given."""
        by_name = {joint["name"]: joint for joint in self.joints.values()}

        def angle(joint):
            if joint["name"] in positions or joint["mimic"] is None:
                return float(positions.get(joint["name"], 0.0))
            source, multiplier, offset = joint["mimic"]
            return multiplier * angle(by_name[source]) + offset

        return {joint["name"]: angle(joint) for joint in self.joints.values() if joint["revolute"]}

    def frames(self, positions):
        """Root-frame rotation and origin of every link."""
        angles = self.angles(positions)
        frames, pending = {self.root: (np.eye(3), np.zeros(3))}, dict(self.joints)
        while pending:
            ready = [child for child, joint in pending.items() if joint["parent"] in frames]
            if not ready:
                raise ValueError("the hand joints do not form a tree")
            for child in ready:
                joint = pending.pop(child)
                rotation, origin = frames[joint["parent"]]
                turn = (Rotation.from_rotvec(joint["axis"] * angles[joint["name"]]).as_matrix() if joint["revolute"]
                        else np.eye(3))
                frames[child] = (rotation @ joint["rotation"] @ turn, origin + rotation @ joint["xyz"])
        return frames

    def subtree(self, link):
        return [link] + [name for child, joint in self.joints.items() if joint["parent"] == link
                         for name in self.subtree(child)]

    def triangles(self, frames, links):
        """Root-frame triangles of the meshed ``links`` and the link owning each triangle."""
        links = [link for link in links if link in self.meshes]
        placed = [self.meshes[link] @ frames[link][0].T + frames[link][1] for link in links]
        return np.concatenate(placed), np.repeat(np.array(links), [len(block) for block in placed])

    def holding_torque(self, positions, joint, gravity, step=1e-6):
        """Derivative of the gravitational potential along one active joint: the torque that holds it."""
        def potential(value):
            frames = self.frames({**positions, joint: value})
            return -sum(mass * gravity @ (frames[link][0] @ center + frames[link][1])
                        for link, (mass, center) in self.inertials.items())
        return (potential(positions[joint] + step) - potential(positions[joint] - step)) / (2.0 * step)


def project_on_triangles(point, triangles):
    """Closest point of a triangle set to ``point`` and its triangle index."""
    corners = triangles.transpose(1, 0, 2)
    closest = closest_on_triangles(np.broadcast_to(point, corners[0].shape), *corners)
    index = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
    return closest[index], index


def closest_pair(first, second, iterations=20):
    """Closest points of two triangle sets: the closest vertex pair refined by alternating exact projections, which
    never increase the distance."""
    vertices = np.unique(first.reshape(-1, 3), axis=0)
    distance, _ = cKDTree(np.unique(second.reshape(-1, 3), axis=0)).query(vertices)
    p = vertices[int(np.argmin(distance))]
    for _ in range(iterations):
        q, j = project_on_triangles(p, second)
        p, i = project_on_triangles(q, first)
    return p, i, q, j


def strip_distance(points, frame, width, top, length):
    """Distance from points to the strip rectangle |a| <= width / 2, top <= b <= top + length of ``frame``."""
    center, normal, across, down = frame
    offset = points - center
    a, b = offset @ across, offset @ down
    outside = np.stack([np.maximum(np.abs(a) - 0.5 * width, 0.0),
                        np.maximum(np.maximum(top - b, b - top - length), 0.0), offset @ normal], axis=-1)
    return np.linalg.norm(outside, axis=-1)


def rectangle_distance(triangles, frame, width, top, length):
    """Exact distance from a triangle set to the strip rectangle of ``frame``, zero where they intersect.
    Disjoint convex polygons are closest at a vertex or between two edges; far bounding spheres are skipped."""
    center, normal, across, down = frame
    corners = np.array([center + a * 0.5 * width * across + b * down
                        for a, b in ((-1, top), (1, top), (1, top + length), (-1, top + length))])
    middle = triangles.mean(axis=1)
    radius = np.linalg.norm(triangles - middle[:, None], axis=2).max(axis=1)
    vertices = strip_distance(triangles, frame, width, top, length).min(axis=1)
    kept = triangles[strip_distance(middle, frame, width, top, length) - radius <= vertices.min()]
    a, b, c = kept[:, 0], kept[:, 1], kept[:, 2]
    best = vertices.min()
    for corner in corners:
        best = min(best, float(point_triangle_distance(np.broadcast_to(corner, a.shape), a, b, c).min()))
    starts, ends = kept, np.roll(kept, -1, axis=1)
    for k in range(4):
        best = min(best, float(segment_distance(starts, ends, corners[k], corners[(k + 1) % 4]).min()))
    # A triangle edge crossing the rectangle, or a rectangle edge crossing a triangle, touches it.
    heights = [(points - center) @ normal for points in (starts, ends)]
    normals = np.cross(b - a, c - a)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = starts + (ends - starts) * (heights[0] / (heights[0] - heights[1]))[..., None]
        if ((heights[0] * heights[1] < 0) & (strip_distance(crossing, frame, width, top, length) <= 1e-12)).any():
            return 0.0
        for k in range(4):
            u, v = corners[k], corners[(k + 1) % 4]
            hu, hv = np.einsum("ij,ij->i", u - a, normals), np.einsum("ij,ij->i", v - a, normals)
            point = u + (v - u) * (hu / (hu - hv))[:, None]
            if ((hu * hv < 0) & (point_triangle_distance(point, a, b, c) <= 1e-12)).any():
                return 0.0
    return best


def strip_positions(rest, height, top):
    """Hang the flat lattice in the world x = 0 plane: rest x runs along world y, rest y runs down from ``top``."""
    down = rest[:, 1] - rest[:, 1].min() + top
    return np.stack([np.zeros(len(rest)), rest[:, 0], height - down], axis=1)


def pinch_geometry(hand, args):
    """Place the pinch in the hand root frame: pads one cloth thickness apart, the strip hanging from their midpoint,
    the index one gap outside its shell, and the pinch force set by the thumb torque limit through the thumb lever."""
    shell = 0.5 * args.thickness
    thumb_pad, index_pad = hand.subtree("right_thumb_3"), hand.subtree("right_index_2")
    pads = thumb_pad + index_pad
    simplified = [link for link in hand.meshes if link not in pads]

    def pose(thumb, index):
        return {"right_thumb_1_joint": hand.upper["right_thumb_1_joint"],
                "right_thumb_2_joint": float(thumb) * hand.upper["right_thumb_2_joint"],
                "right_index_1_joint": float(index) * hand.upper["right_index_1_joint"]}

    def pad_pair(closure):
        frames = hand.frames(pose(closure, closure))
        (thumb, thumb_links), (index, index_links) = hand.triangles(frames, thumb_pad), hand.triangles(frames, index_pad)
        p, i, q, j = closest_pair(thumb, index)
        return float(np.linalg.norm(q - p)), p, str(thumb_links[i]), q, str(index_links[j]), frames

    if not pad_pair(args.open)[0] > args.thickness:
        raise ValueError("the pads must be more than one cloth thickness apart at the open closure")
    closed = next((c for c in np.arange(args.open, 1.0, 0.01) if pad_pair(c)[0] < args.thickness), None)
    if closed is None:
        raise ValueError("thumb and index never close to one cloth thickness")
    low, high = closed - 0.01, closed
    for _ in range(30):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if pad_pair(middle)[0] > args.thickness else (low, middle)
    closure = 0.5 * (low + high)
    gap, p, thumb_link, q, index_link, frames = pad_pair(closure)
    normal, center = (q - p) / np.linalg.norm(q - p), 0.5 * (p + q)
    width, top, length = (args.nx - 1) * args.spacing, -args.overhang, (args.ny - 1) * args.spacing
    others, pad_triangles = hand.triangles(frames, simplified)[0], hand.triangles(frames, pads)[0]
    first = np.cross(normal, np.eye(3)[int(np.argmin(np.abs(normal)))])
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)

    def clearance(down):
        frame = (center, normal, np.cross(normal, down), down)
        return min(rectangle_distance(others, frame, width, top, length) - SIMPLIFIED_MESH_ERROR,
                   rectangle_distance(pad_triangles, frame, width, args.pinch_depth, top + length - args.pinch_depth))

    angles = np.radians(np.arange(0.0, 360.0, 5.0))
    scores = [clearance(math.cos(a) * first + math.sin(a) * second) for a in angles]
    best = int(np.argmax(scores))
    down = math.cos(angles[best]) * first + math.sin(angles[best]) * second
    across = np.cross(normal, down)
    frame = (center, normal, across, down)

    def strip_clearance(positions, links, strip=frame):
        return float(rectangle_distance(hand.triangles(hand.frames(positions), links)[0], strip, width, top, length))

    target = shell + args.index_gap
    low, high = args.open, closure
    if not strip_clearance(pose(args.open, low), index_pad) > target > strip_clearance(pose(args.open, high), index_pad):
        raise ValueError("the index cannot rest one gap outside the strip shell")
    for _ in range(30):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if strip_clearance(pose(args.open, middle), index_pad) > target else (low, middle)
    initial = pose(args.open, low)
    local = frames[thumb_link][0].T @ (p - frames[thumb_link][1])
    angle = closure * hand.upper["right_thumb_2_joint"]

    def contact(value):
        moved = hand.frames({**pose(closure, closure), "right_thumb_2_joint": value})
        return moved[thumb_link][0] @ local + moved[thumb_link][1]

    lever = float(normal @ (contact(angle + 1e-6) - contact(angle - 1e-6)) / 2e-6)
    if not lever > 0.0:
        raise ValueError("closing the thumb must move its pad toward the index")
    torque = args.pinch_force * lever
    final = angle + args.index_gap / lever + 5.0 * torque / args.kp
    if final > hand.upper["right_thumb_2_joint"]:
        raise ValueError("the thumb target exceeds its joint limit")
    # Holding, the thumb has pushed the strip one gap toward the resting index.
    held = {**pose(closure, low), "right_thumb_2_joint": angle + args.index_gap / lever}
    pushed = (center + args.index_gap * normal, normal, across, down)
    clearances = {"thumb_pad_m": strip_clearance(initial, thumb_pad), "index_pad_m": strip_clearance(initial, index_pad),
                  "simplified_initial_m": strip_clearance(initial, simplified),
                  "simplified_holding_m": strip_clearance(held, simplified, pushed)}
    required = shell + SIMPLIFIED_MESH_ERROR + 1e-3
    if not (clearances["thumb_pad_m"] > target and min(clearances["simplified_initial_m"],
                                                        clearances["simplified_holding_m"]) > required):
        raise ValueError(f"only the pads may reach the strip shell: {clearances}, simplified links need {required}")
    world_from_root = np.stack([normal, across, -down])
    gravity = world_from_root.T @ np.array([0.0, 0.0, args.gravity_z])
    position = np.array([0.0, 0.0, args.height]) - world_from_root @ center
    x, y, z, w = (float(value) for value in Rotation.from_matrix(world_from_root).as_quat())
    return {"closure": float(closure), "pad_gap_m": gap, "thumb_pad_link": thumb_link, "index_pad_link": index_link,
            "source_mesh_links": pads, "center_root_m": center.tolist(), "normal_root": normal.tolist(),
            "down_root": down.tolist(), "hanging_clearance_score_m": float(scores[best]), "index_closure": float(low),
            "clearance_m": clearances, "simplified_required_m": required, "thumb_lever_m": lever,
            "thumb_torque_limit_nm": float(torque), "thumb_open_rad": initial["right_thumb_2_joint"],
            "thumb_target_rad": float(final), "holding_positions": held,
            "holding_torque_nm": {joint: float(hand.holding_torque(held, joint, gravity))
                                  for joint in ("right_thumb_2_joint", "right_index_1_joint")},
            "initial_positions": initial, "world_from_root": world_from_root.tolist(),
            "root_position_m": position.tolist(), "root_quat_wxyz": [w, x, y, z]}


def hand_scene(args, directory, hand, geometry):
    """Author the hand URDF (absolute meshes, a contact triangle limit off the pads) and save it as a scene with a
    static root at the pinch placement and position drives on the six motor joints."""
    tree = ET.parse(args.hand_urdf)
    robot = tree.getroot()
    for mesh in robot.iter("mesh"):
        mesh.set("filename", str(hand_mesh(args.hand_urdf, mesh.get("filename"))))
    for link in robot.findall("link"):
        for collision in link.findall("collision"):
            if link.get("name") not in geometry["source_mesh_links"] and collision.find("geometry/mesh") is not None:
                ET.SubElement(collision, "nuka:mesh_triangle_limit", value=str(args.triangle_limit))
    ET.indent(tree, space="  ")
    urdf = directory / "hand.urdf"
    tree.write(urdf, encoding="utf-8", xml_declaration=True)
    path = directory / "hand.nks"
    nuka.Scene.load(str(urdf)).save(str(path))
    document = json.loads(path.read_text())
    root = next(node for node in document["tree"] if node["name"] == hand.root)
    root["rigid_body"]["is_static"] = True
    root["transform"] = {**root.get("transform", {}), "pos": geometry["root_position_m"],
                         "quat": geometry["root_quat_wxyz"]}
    angles = hand.angles(geometry["initial_positions"])
    for node in document["tree"]:
        joint = node.get("joint")
        if joint is None or joint.get("name") not in angles:
            continue
        joint["initial_position"] = angles[joint["name"]]
        if joint["name"] in HAND_JOINTS:
            limit = (geometry["thumb_torque_limit_nm"] if joint["name"] == "right_thumb_2_joint"
                     else hand.effort[joint["name"]])
            node["actuator"] = {"name": joint["name"] + "_drive", "type": "position",
                                "joint_id": node["actuator"]["joint_id"], "gain": args.kp, "force_limit": limit}
        else:
            node.pop("actuator", None)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return urdf, path


def kinematics_error(hand, links, sample, geometry):
    """Largest link position (m) and rotation (rad) gap between the engine poses and host kinematics evaluated at
    the engine's own joint positions."""
    q = sample["state_JOINT_POSITION"].reshape(-1)
    if q.size != len(links):
        raise ValueError("expected one joint coordinate per link")
    frames = hand.frames({hand.joints[link]["name"]: float(q[k]) for k, link in enumerate(links)
                          if link in hand.joints and hand.joints[link]["revolute"]})
    world_from_root, origin = np.array(geometry["world_from_root"]), np.array(geometry["root_position_m"])
    poses = sample["state_ARTICULATION_LINK_POSE"].reshape(len(links), 7).astype(np.float64)
    position = rotation = 0.0
    for k, link in enumerate(links):
        engine = Rotation.from_quat(poses[k, [4, 5, 6, 3]])
        host = Rotation.from_matrix(world_from_root @ frames[link][0])
        position = max(position, float(np.linalg.norm(poses[k, :3] - world_from_root @ frames[link][1] - origin)))
        rotation = max(rotation, float((engine.inv() * host).magnitude()))
    return {"position_m": position, "rotation_rad": rotation}


def pad_contacts(world, sample, vertices, links):
    """(vertex, link) pairs of the active particle-link contact rows; link indices run over all environments.
    A face or edge endpoint on a link pairs every particle it interpolates, read from the latest interval."""
    kinds = [sample[f"state_CONTACT_SIDE_{side}_KIND"].reshape(-1) for side in "AB"]
    indices = [sample[f"state_CONTACT_SIDE_{side}_INDEX"].reshape(-1).astype(np.int64) for side in "AB"]
    particle, link, endpoint = (int(kind.value) for kind in (nuka.ContactSideKind.PARTICLE,
                                                             nuka.ContactSideKind.LINK,
                                                             nuka.ContactSideKind.POINT_ENDPOINT))
    ranges = np.asarray(world.download_field(nuka.Field.POINT_ENDPOINT_RANGES)).reshape(-1, 2)
    terms = np.ascontiguousarray(world.download_field(nuka.Field.POINT_ENDPOINT_TERMS))
    terms = terms.reshape(-1, ENDPOINT_TERM.itemsize).view(ENDPOINT_TERM).reshape(-1)
    pairs = set()
    for a, b in ((0, 1), (1, 0)):
        mask = (kinds[a] == particle) & (kinds[b] == link) & (indices[a] < vertices)
        pairs.update(zip(indices[a][mask].tolist(), (indices[b][mask] % links).tolist()))
        mask = (kinds[a] == endpoint) & (kinds[b] == link)
        for point, owner in zip(indices[a][mask].tolist(), (indices[b][mask] % links).tolist()):
            first, count = ranges[point]
            record = terms[first:first + count]
            used = (record["kind"] == particle) & (record["index"] < vertices) & record["columns"].any(axis=(1, 2))
            pairs.update((int(vertex), owner) for vertex in record["index"][used])
    return pairs


def pad_slip(pairs, positions, poses, seconds):
    """Drift speed of each vertex in the frame of the link it touches over the window."""
    speeds = {}
    for vertex, link in pairs:
        local = [Rotation.from_quat(pose[link, [4, 5, 6, 3]]).inv().apply(position[vertex] - pose[link, :3])
                 for position, pose in ((positions[0], poses[0]), (positions[-1], poses[-1]))]
        speeds[vertex, link] = float(np.linalg.norm(local[1] - local[0]) / seconds)
    return speeds


def run_s7(args):
    scene_dir = args.output.with_name(args.output.name + "_scene")
    for path in (args.output, scene_dir):
        if path.exists():
            raise FileExistsError(path)
    tolerance = production_tolerance()
    ramp, release, settle, hold = (policy_steps(value, args.dt) for value in (args.ramp, args.close, args.settle,
                                                                               args.hold))
    if ramp > release:
        raise ValueError("the thumb must finish closing before the strip is released")
    steps, begin = release + settle + hold, release + settle
    physical = {key: getattr(args, key) for key in ("nx", "ny", "spacing", "overhang", "pinch_depth", "height", "open",
        "index_gap", "pinch_force", "kp", "kd", "triangle_limit", "ramp", "close", "settle", "hold", "density",
        "stretch", "poisson", "bend", "thickness", "friction", "gravity_z")}
    physical["hand_urdf_sha256"] = sha256(args.hand_urdf)
    hand = Hand(args.hand_urdf)
    if set(hand.upper) != set(HAND_JOINTS):
        raise ValueError("the hand's active joints differ from its six motor joints")
    geometry = pinch_geometry(hand, args)
    scene_dir.mkdir(parents=True)
    urdf, scene_path = hand_scene(args, scene_dir, hand, geometry)
    metadata = {"fixture": "S7", **base_metadata(args, physical, steps, tolerance, args.nx * args.ny),
        "state_system": "particles_and_articulation", "required_physical_systems": ["vbd", "articulation"],
        "required_ccd_contact_domains": ["particle_mesh_self", "particle_articulation", "articulation_self"],
        "boundary": "Static hand root placed so the pinch normal is world x and the strip hangs along world -z; the "
                    "thumb closes on a force-limited position drive with gravity feedforward while the index holds its "
                    "position; the strip's top row is kinematic until release, after which only the pads hold it",
        "hold_window_policy_steps": [begin + 1, steps],
        "reference": {"support_ratio": 1.0, "support_tolerance": 0.05, "slip_speed_mps": 1e-3,
                      "source": "the pads carry the whole off-ground strip weight without slipping"},
        "pinch": geometry, "hand_scene": {"urdf": str(urdf), "urdf_sha256": sha256(urdf), "scene": str(scene_path),
                                          "scene_sha256": sha256(scene_path)}}
    grid = morphs.Grid(args.nx, args.ny, args.spacing)
    scene = Scene(SimOptions(dt=args.dt, gravity=(0, 0, args.gravity_z),
        cloth_integrator=0 if args.integrator == "bdf2" else 1, solver_vel_iters=args.sweeps, solver_pos_iters=4,
        ogc_contact_capacity=args.contact_capacity, baumgarte_max_velocity=0))
    scene.add_entity(morphs.NKS(str(scene_path)))
    scene.add_entity(grid, materials.Cloth.VBD(areal_density=args.density, friction=args.friction,
        stretch_stiffness=args.stretch, poisson=args.poisson, bend_stiffness=args.bend, thickness=args.thickness),
        surfaces.Cloth(free=True))
    device = nuka.Device.create(0)
    world = scene.build(device)
    session = None
    try:
        world.set_gravity_z(args.gravity_z)
        names = list(world.dof_names())
        child = {joint["name"]: link for link, joint in hand.joints.items()}
        links = [child.get(name, name) for name in names]
        if sorted(links) != sorted([hand.root, *hand.joints]):
            raise ValueError(f"the engine links differ from the authored hand: {names}")
        slot = {name: names.index(name) for name in HAND_JOINTS}
        drive = {field: np.asarray(world.download_field(field), dtype=np.float32).reshape(-1).copy()
                 for field in (nuka.Field.DRIVE_STIFFNESS, nuka.Field.DRIVE_DAMPING, nuka.Field.DRIVE_FORCE_LIMIT,
                               nuka.Field.DRIVE_TARGET, nuka.Field.JOINT_FEEDFORWARD)}
        initial = hand.angles(geometry["initial_positions"])
        for name in HAND_JOINTS:
            drive[nuka.Field.DRIVE_STIFFNESS][slot[name]] = args.kp
            drive[nuka.Field.DRIVE_DAMPING][slot[name]] = args.kd
            drive[nuka.Field.DRIVE_FORCE_LIMIT][slot[name]] = (geometry["thumb_torque_limit_nm"]
                if name == "right_thumb_2_joint" else hand.effort[name])
            drive[nuka.Field.DRIVE_TARGET][slot[name]] = initial[name]
        for name, torque in geometry["holding_torque_nm"].items():
            drive[nuka.Field.JOINT_FEEDFORWARD][slot[name]] = torque
        for field in (nuka.Field.DRIVE_STIFFNESS, nuka.Field.DRIVE_DAMPING, nuka.Field.DRIVE_FORCE_LIMIT,
                      nuka.Field.JOINT_FEEDFORWARD):
            world.upload_field(field, np.ascontiguousarray(drive[field]))
        targets = drive[nuka.Field.DRIVE_TARGET]
        world.set_drive_targets(np.ascontiguousarray(targets))
        rest = np.asarray(world.download_field(nuka.Field.PARTICLE_POSITION)).reshape(-1, 3).copy()
        if not np.allclose(rest, grid.rest_positions(), atol=1e-6):
            raise ValueError("the cooked strip lattice differs from the authored grid")
        placed = strip_positions(rest.astype(np.float64), args.height, -args.overhang).astype(np.float32)
        world.upload_field(nuka.Field.PARTICLE_POSITION, np.ascontiguousarray(placed))
        world.upload_field(nuka.Field.PARTICLE_KINEMATIC_TARGET, np.ascontiguousarray(placed))
        free = np.asarray(world.download_field(nuka.Field.PARTICLE_INV_MASS)).copy()
        pinned = rest[:, 1] == rest[:, 1].min()
        held = free.copy()
        held[pinned] = 0.0
        world.upload_field(nuka.Field.PARTICLE_INV_MASS, held)
        velocity = np.asarray(world.download_field(nuka.Field.PARTICLE_VELOCITY)).reshape(-1, 3).copy()
        elements = np.asarray(world.download_field(nuka.Field.VBD_ELEMENTS), dtype=np.uint32).reshape(-1, 16).copy()
        metadata.update(owner_names={"LINK": names}, kinematic_tree=[
            {key: link[key] for key in ("parent_index", "articulation_index", "joint_type")}
            for link in world.kinematic_tree()])
        session = DiagnosticSession(world, args.output, metadata, chunk_steps=args.chunk_steps,
            state_fields=PARTICLE_FIELDS + CONTACT_FIELDS + LINK_FIELDS,
            thresholds=DiagnosticThresholds(velocity_tolerance_mps=tolerance))
        if session.substeps != 1:
            raise ValueError("the pinch fixture records one substep per policy step")
        faces, edges = topology(args.nx, args.ny)
        np.savez_compressed(args.output / "initial.npz", rest=placed, positions=placed, velocity=velocity, vbd_elements=elements,
                            faces=faces, edges=edges, inv_mass=free, flat_rest=rest, pinned=pinned)
        session.manifest["initial_geometry_sha256"] = sha256(args.output / "initial.npz")
        write_json(args.output / "inputs.json", {"parameters": {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}, "metadata": metadata})
        thumb, motors = slot["right_thumb_2_joint"], [slot[name] for name in HAND_JOINTS]
        opening, closing = geometry["thumb_open_rad"], geometry["thumb_target_rad"]
        stages, positions, poses, wrenches, contacts = [], [], [], [], []
        reason, kinematics, sample, released = "completed", None, None, None
        for step in range(steps):
            if step == release:
                # Releasing is only meaningful once both pads touch the strip.
                touching = {links[link] for _, link in pad_contacts(world, sample, len(rest), len(links))}
                released = {"thumb_angle_rad": float(sample["state_JOINT_POSITION"].reshape(-1)[thumb]),
                            "touching_links": sorted(touching)}
                if not all(touching & set(hand.subtree(pad)) for pad in ("right_thumb_3", "right_index_2")):
                    reason = "pinch_not_established"
                    break
                world.upload_field(nuka.Field.PARTICLE_INV_MASS, free)
            targets[thumb] = opening + (closing - opening) * min((step + 1) / ramp, 1.0)
            world.set_drive_targets(np.ascontiguousarray(targets))
            if step + 1 == args.replay_step:
                replayed = session.replay_budgets(replay_budget_list(args.replay_budgets),
                    state_fields=REPLAY_FIELDS + LINK_FIELDS, controls=targets[motors].copy())
                print(json.dumps({"budget_replay": str(replayed)}), flush=True)
            sample = session.step(controls=targets[motors].copy())
            stages.append(sample["stages"][0, 0])
            if np.any(sample["env_status"]):
                reason = "physics_failure"
                break
            if not (np.isfinite(sample["state_PARTICLE_POSITION"]).all() and
                    np.isfinite(sample["state_PARTICLE_VELOCITY"]).all()):
                reason = "nonfinite_state"
                break
            if kinematics is None:
                kinematics = kinematics_error(hand, links, sample, geometry)
            if step >= begin:
                positions.append(sample["state_PARTICLE_POSITION"].reshape(-1, 3).astype(np.float64))
                poses.append(sample["state_ARTICULATION_LINK_POSE"].reshape(len(links), 7).astype(np.float64))
                wrenches.append(sample["state_LINK_CONTACT_WRENCH"].reshape(len(links), 6).astype(np.float64))
                contacts.append(pad_contacts(world, sample, len(rest), len(links)))
            if step + 1 == args.replay_step and args.replay_stop:
                reason = "replay_complete"
                break
        session.close(reason)
        mass = float(np.sum(1.0 / free[free > 0]))
        weight = mass * -args.gravity_z
        metrics = {"policy_steps": session.steps, "stop_reason": reason, "strip_mass_kg": mass,
                   "strip_weight_n": weight, "kinematics_error": kinematics, "at_release": released}
        if reason == "completed":
            window = np.array(stages[begin:], dtype=np.float64)
            solved = window[:, Stage.SOLVED]
            rows, gravity, boundary = (float(solved[:, column].sum()) for column in (StageColumn.VBD_ROW_IMPULSE_Z,
                StageColumn.VBD_GRAVITY_IMPULSE_Z, StageColumn.VBD_ELASTIC_BOUNDARY_IMPULSE_Z))
            momentum = StageColumn.VBD_DISCRETE_MOMENTUM_Z
            support = rows / -gravity
            pads = {links.index(link) for link in geometry["source_mesh_links"]}
            touching = sorted({links[link] for pairs in contacts for _, link in pairs})
            pinched = {pair for pair in set.intersection(*contacts) if pair[1] in pads}
            slip = pad_slip(pinched, positions, poses, (len(positions) - 1) * args.dt)
            wrench = np.mean(wrenches, axis=0)
            thumb_force = wrench[[links.index(link) for link in hand.subtree("right_thumb_1")], :3].sum(axis=0)
            index_force = wrench[[links.index(link) for link in hand.subtree("right_index_1")], :3].sum(axis=0)
            metrics.update(hold_steps=len(window), row_impulse_z_ns=rows, gravity_impulse_z_ns=gravity,
                boundary_impulse_z_ns=boundary,
                momentum_change_z_ns=float(window[-1, Stage.END, momentum] - window[0, Stage.BEGIN, momentum]),
                gravity_impulse_over_weight=-gravity / (weight * len(window) * args.dt), support_ratio=support,
                support_error=abs(support - 1.0), support_within_5_percent=bool(abs(support - 1.0) < 0.05),
                touching_links=touching, only_pads_touch=set(touching) <= set(geometry["source_mesh_links"]),
                pinched_contacts=len(pinched), pinched_vertices=len({vertex for vertex, _ in pinched}),
                slip_speed_max_mps=max(slip.values()) if slip else None,
                slip_speed_median_mps=float(np.median(list(slip.values()))) if slip else None,
                slip_below_1_mm_per_s=bool(slip) and max(slip.values()) < 1e-3,
                thumb_contact_force_n=thumb_force.tolist(), index_contact_force_n=index_force.tolist(),
                pad_force_z_over_weight=float((thumb_force[2] + index_force[2]) / weight),
                strip_lowest_z_m=float(positions[-1][:, 2].min()),
                final_speed_max_mps=float(np.linalg.norm(sample["state_PARTICLE_VELOCITY"].reshape(-1, 3),
                                                         axis=1).max()))
        metrics["claims_full_physics_acceptance"] = False
        write_json(args.output / "fixture_metrics.json", metrics)
        print(json.dumps({"output": str(args.output), **metrics}), flush=True)
    except BaseException as error:
        if session is not None:
            session.close("exception", error=repr(error))
        raise
    finally:
        world.destroy()
        device.close()


def cloth_arguments(parser):
    parser.add_argument("--density", type=float, default=.4)
    parser.add_argument("--stretch", type=float, default=5000)
    parser.add_argument("--poisson", type=float, default=.3)
    parser.add_argument("--bend", type=float, default=5e-5)
    parser.add_argument("--thickness", type=float, default=.003)
    parser.add_argument("--friction", type=float, default=.5)


def common_arguments(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=1 / 600)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--integrator", choices=("bdf2", "be"), default="bdf2")
    parser.add_argument("--sweeps", type=int, default=512)
    parser.add_argument("--contact-capacity", type=int, default=65536)
    parser.add_argument("--chunk-steps", type=int, default=128)
    parser.add_argument("--replay-step", type=int, default=0, help="policy step replayed at each budget first")
    parser.add_argument("--replay-budgets", default="", help="velocity sweep budgets, first:last or a comma list")
    parser.add_argument("--replay-stop", action="store_true", help="end the record after the replayed step")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fixtures = parser.add_subparsers(dest="fixture", required=True)
    for name, integrator in (("s1", "bdf2"), ("s2", "be")):
        free = fixtures.add_parser(name, help="free cloth translation, spin and small in-plane stretch")
        common_arguments(free)
        cloth_arguments(free)
        free.set_defaults(integrator=integrator, gravity_z=0)
        free.add_argument("--nx", type=int, default=10)
        free.add_argument("--ny", type=int, default=10)
        free.add_argument("--spacing", type=float, default=.01)
        free.add_argument("--height", type=float, default=1)
        free.add_argument("--duration", type=float, default=.1)
        free.add_argument("--damping", type=float, default=.002)
        free.add_argument("--translation", type=float, nargs=3, default=(.2, .1, -.05), metavar=("VX", "VY", "VZ"))
        free.add_argument("--spin", type=float, nargs=3, default=(0, 0, .1), metavar=("WX", "WY", "WZ"))
        free.add_argument("--stretch-strain", type=float, default=1e-4)
    s5 = fixtures.add_parser("s5", help="folded cloth dropped onto a static table (self-contact)")
    common_arguments(s5)
    s5.add_argument("--nx", type=int, default=133)
    s5.add_argument("--ny", type=int, default=21)
    s5.add_argument("--spacing", type=float, default=.005)
    s5.add_argument("--arc-edges", type=int, default=12)
    s5.add_argument("--ramp-edges", type=int, default=20)
    s5.add_argument("--gap", type=float, default=.0033)
    s5.add_argument("--height", type=float, default=.0065)
    s5.add_argument("--duration", type=float, default=2.0)
    cloth_arguments(s5)
    s5.add_argument("--table-friction", type=float, default=.4)
    s7 = fixtures.add_parser("s7", help="cloth strip held by a thumb-index pinch of the Inspire hand")
    common_arguments(s7)
    cloth_arguments(s7)
    s7.add_argument("--hand-urdf", type=Path, default=HAND_URDF)
    s7.add_argument("--nx", type=int, default=5)
    s7.add_argument("--ny", type=int, default=21)
    s7.add_argument("--spacing", type=float, default=.005)
    s7.add_argument("--overhang", type=float, default=.005, help="strip length above the pad midpoint (m)")
    s7.add_argument("--pinch-depth", type=float, default=.01, help="strip length below the midpoint the pads may reach")
    s7.add_argument("--height", type=float, default=.2, help="world height of the pad midpoint (m)")
    s7.add_argument("--open", type=float, default=.5, help="initial thumb closure fraction")
    s7.add_argument("--index-gap", type=float, default=5e-4, help="index pad clearance to the strip shell (m)")
    s7.add_argument("--pinch-force", type=float, default=.2, help="thumb pad normal force at its torque limit (N)")
    s7.add_argument("--kp", type=float, default=20.0)
    s7.add_argument("--kd", type=float, default=.2)
    s7.add_argument("--triangle-limit", type=int, default=400)
    s7.add_argument("--ramp", type=float, default=.2, help="thumb closing time (s)")
    s7.add_argument("--close", type=float, default=.3, help="release time of the pinned top row (s)")
    s7.add_argument("--settle", type=float, default=.4)
    s7.add_argument("--hold", type=float, default=.8)
    args = parser.parse_args()
    {"s1": run_s1_fixture, "s2": run_s1_fixture, "s5": run_s5, "s7": run_s7}[args.fixture](args)


if __name__ == "__main__":
    main()
