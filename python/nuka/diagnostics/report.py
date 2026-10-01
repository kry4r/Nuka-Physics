"""Layered physics decisions from preserved production records."""

import json
from pathlib import Path

import numpy as np

from .._nuka_ext import ContactSideKind
from ..energy import EnergyColumn as E, EnergyStatus
from .analysis import vbd_discrete_momentum
from .capture import load_records, sha256, write_json
from .geometry import CCD_CONFIGURATION
from .schema import AuditCount as A, AuditMetric as M, DiagnosticThresholds, Stage as S, StageColumn as C


def decode_maxima(packed):
    packed = np.asarray(packed, dtype=np.uint64)
    values = (packed >> np.uint64(32)).astype(np.uint32).view(np.float32)
    ids = np.bitwise_not(packed.astype(np.uint32))
    return values, ids, packed != 0


def decision(complete, passed, **details):
    return {"status": "unmeasured" if not complete else "passed" if passed else "failed", **details}


def window_ratios(residual, throughput, dt, seconds):
    elapsed = np.cumsum(dt)
    positive = np.r_[0.0, np.cumsum(np.maximum(residual, 0))]
    flow = np.r_[0.0, np.cumsum(throughput)]
    start = np.searchsorted(elapsed, elapsed - seconds, side="right")
    loss, work = positive[1:] - positive[start], flow[1:] - flow[start]
    ratio = np.divide(loss, work, out=np.full_like(work, np.inf), where=work > 0)
    ratio[(loss == 0) & (work == 0)] = 0
    return ratio


def energy_analysis(records, flags, limits):
    finite = np.isfinite(records).all(axis=1)
    covered = (flags == 0) & (records[:, E.VALID] == 1) & finite
    reasons = [s.name for s in EnergyStatus if np.any(flags & int(s))]
    if not covered.all():
        return decision(False, False, covered_records=int(covered.sum()), records=len(records),
                        coverage_reasons=reasons)
    r, flow = records[:, E.RESIDUAL], records[:, E.THROUGHPUT]
    ratios = window_ratios(r, flow, records[:, E.DT], limits.window_seconds)
    total_flow = float(flow.sum())
    losses = records[:, E.FRICTION_LOSS:E.RAYLEIGH_LOSS + 1]
    negative = int((losses < -limits.negative_loss_tolerance_j).sum())
    trunc = records[:, E.DAT_KINETIC_LOSS]
    gates = {
        "net_residual": abs(r.sum()) <= limits.energy_net_ratio * total_flow,
        "positive_window": bool((ratios <= limits.energy_positive_window_ratio).all()),
        "dat_energy": bool((trunc >= -limits.negative_loss_tolerance_j).all() and
                            0 <= trunc.sum() <= limits.dat_energy_ratio * total_flow),
        "nonnegative_dissipation": negative == 0,
        "energy_spike": bool((np.abs(r) <= limits.energy_spike_j).all()),
        "nonnegative_throughput": bool((flow >= 0).all()),
    }
    return decision(True, all(gates.values()), gates=gates, records=len(records),
        net_residual_j=float(r.sum()), absolute_residual_j=float(np.abs(r).sum()),
        positive_residual_j=float(np.maximum(r, 0).sum()), throughput_j=total_flow,
        max_positive_window_ratio=float(ratios.max()), negative_loss_terms=negative,
        negative_loss_terms_by_class={E(E.FRICTION_LOSS + i).name: int((losses[:, i] <
            -limits.negative_loss_tolerance_j).sum()) for i in range(losses.shape[1])},
        max_absolute_residual_j=float(np.abs(r).max()), dat_kinetic_loss_j=float(trunc.sum()),
        terms_j={c.name: float(records[:, c].sum()) for c in E if E.DRIVE_WORK <= c <= E.DAT_POTENTIAL})


def particle_history(values, env, manifest, initial_velocity):
    """END velocities of one environment with the two velocities before each, masses and effective steps."""
    if manifest["substeps"] != 1 or initial_velocity is None or "state_PARTICLE_VELOCITY" not in values:
        return None
    steps, count = len(values["sweeps"]), manifest["env_count"]
    velocity = values["state_PARTICLE_VELOCITY"].reshape(steps, count, -1, 3)[:, env].astype(np.float64)
    initial = np.broadcast_to(np.asarray(initial_velocity, dtype=np.float64), (count,) + velocity.shape[1:])[env]
    history = np.concatenate([np.broadcast_to(initial, (2,) + initial.shape), velocity])
    return {"velocity": velocity, "previous": history[1:-1], "before": history[:-2],
            "inverse_mass": values["state_PARTICLE_INV_MASS"].reshape(steps, count, -1)[:, env],
            "effective_dt": values["state_VBD_EFFECTIVE_DT"].reshape(steps, count, -1)[:, env]}


def momentum_analysis(stages, records, flags, metadata, limits, tolerance, particles=None):
    discrete = slice(C.VBD_DISCRETE_MOMENTUM_X, C.VBD_DISCRETE_MOMENTUM_Z + 1)
    dt = records[:, E.DT]
    begin, solved, projected, end = (stages[:, s] for s in (S.BEGIN, S.SOLVED, S.PROJECTED, S.END))
    trunc = projected[:, discrete] - end[:, discrete]
    gravity = solved[:, C.VBD_GRAVITY_IMPULSE_X:C.VBD_GRAVITY_IMPULSE_Z + 1]
    boundary = solved[:, C.VBD_ELASTIC_BOUNDARY_IMPULSE_X:C.VBD_ELASTIC_BOUNDARY_IMPULSE_Z + 1]
    rows = solved[:, C.VBD_ROW_IMPULSE_X:C.VBD_ROW_IMPULSE_Z + 1]
    impulse = gravity + boundary + rows - trunc
    residual = end[:, discrete] - begin[:, discrete] - impulse
    equation = solved[:, C.VBD_MOMENTUM_DEFECT_X:C.VBD_MOMENTUM_DEFECT_Z + 1]
    norm = np.linalg.norm(residual, axis=1)
    scale = np.maximum(np.linalg.norm(end[:, discrete], axis=1), np.linalg.norm(gravity, axis=1))
    relative = limits.momentum_relative_per_second * scale * dt
    covered = bool(metadata.get("momentum_scope") == "vbd_subsystem" and
        not metadata.get("aerodynamic_forces", True) and (flags == 0).all() and
        (records[:, E.VALID] == 1).all() and
        (records[:, E.RAYLEIGH_LOSS] == 0).all() and np.isfinite(residual).all() and
        (solved[:, C.VBD_DYNAMIC_PARTICLES] > 0).all())
    original = decision(covered, bool((norm <= relative).all()),
        momentum_relative_per_second=limits.momentum_relative_per_second,
        max_allowed_kg_mps=float(np.max(relative)), intervals_over_bound=int((norm > relative).sum()))
    # A vertex within the solver tolerance leaves an impulse defect of at most tol m h / ht per component, so the
    # residual norm stays below sqrt(3) tol sum_i m_i h / ht_i; the host-recomputed balance must close within it.
    checked, bound = {}, None
    if particles is not None:
        p = particles
        try:
            host_end = vbd_discrete_momentum(p["velocity"], p["previous"], p["inverse_mass"],
                                             p["effective_dt"], metadata["dt"])
            host_begin = vbd_discrete_momentum(p["previous"], p["before"], p["inverse_mass"],
                                               p["effective_dt"], metadata["dt"])
        except ValueError as error:
            checked["independent_recomputation_error"] = str(error)
        else:
            inverse_mass = p["inverse_mass"].astype(np.float64)
            dynamic = inverse_mass > 0
            mass = np.divide(1.0, inverse_mass, out=np.zeros_like(inverse_mass), where=dynamic)
            ratio = dt[:, None] / np.where(dynamic, p["effective_dt"].astype(np.float64), 1.0)
            bound = np.sqrt(3.0) * tolerance * (mass * ratio).sum(axis=1)
            closure = np.linalg.norm(host_end - host_begin - impulse + equation, axis=1)
            checked.update(tolerance_bound_kg_mps=[float(bound.min()), float(bound.max())],
                intervals_over_tolerance_bound=int((norm > bound).sum()),
                max_residual_over_tolerance_bound=float((norm / bound).max()),
                max_independent_closure_kg_mps=float(closure.max()),
                max_closure_over_tolerance_bound=float((closure / bound).max()),
                intervals_over_closure_bound=int((closure > bound).sum()),
                host_readout_difference_kg_mps={"begin": float(np.abs(host_begin - begin[:, discrete]).max()),
                                                "end": float(np.abs(host_end - end[:, discrete]).max())})
    complete = covered and bound is not None and metadata.get("solver_velocity_tolerance_mps") is not None
    passed = bound is not None and bool((norm <= bound).all() and (closure <= bound).all())
    report = decision(complete, passed,
        scope="VBD subsystem; step-local BE/BDF2 discrete momentum, prescribed boundaries included",
        criterion="residual norm <= sqrt(3) tol sum_i m_i h / ht_i; host-recomputed balance plus solver defect within it",
        solver_velocity_tolerance_mps=tolerance, max_residual_kg_mps=float(np.max(norm)), **checked,
        max_balance_plus_equation_defect_kg_mps=float(np.max(np.linalg.norm(residual + equation, axis=1))),
        original_relative_gate=original,
        dat_removed_discrete_momentum_kg_mps=np.sum(trunc, axis=0).tolist(),
        integrator_switching="Each interval uses its actual inertia; mixed measures do not telescope.")
    return report, residual


def contact_analysis(counts, metrics, limits, readout_complete=True):
    totals = counts.sum(axis=0, dtype=np.uint64)
    value, ids, valid = decode_maxima(metrics)
    worst = {}
    for column in M:
        i = int(np.argmax(value[:, column]))
        worst[column.name] = {"value": float(value[i, column]), "interval": i + 1,
                              "global_row": int(ids[i, column]) if valid[i, column] else None}
    checked = totals[A.CONTACTS]
    complete = (readout_complete and checked > 0 and
                totals[A.UNMEASURED_GAPS] == 0 and totals[A.INVALID_ROWS] == 0)
    passed = all(totals[c] == 0 for c in (A.INVALID_ROWS, A.LINEAR_CLOSURE_VIOLATIONS,
        A.CONE_VIOLATIONS, A.NEGATIVE_NORMAL_IMPULSES, A.GAP_VIOLATIONS))
    return decision(bool(complete), bool(passed), contacts=int(checked),
        observed_contact_intervals=int(np.count_nonzero(counts[:, A.CONTACTS])),
        readout_valid=readout_complete,
        totals={c.name: int(totals[c]) for c in A}, worst=worst,
        thresholds={"linear_impulse_relative": limits.linear_impulse_relative,
                    "cone_relative_slack": limits.friction_cone_relative_slack,
                    "position_slop_m": limits.position_slop_m},
        time_layers={"impulse_and_work": "SOLVED", "closest_feature_gap": "END"},
        gap_scope="Final OGC particle VF/EE features; unavailable feature kinds block acceptance.",
        frozen_gap="Auxiliary frozen-Jacobian estimate; excluded from the closest-feature gap gate.")


SIDE_FIELDS = ("CONTACT_SIDE_A_KIND", "CONTACT_SIDE_A_INDEX", "CONTACT_SIDE_B_KIND", "CONTACT_SIDE_B_INDEX")


def contact_owners(values, env, manifest):
    keys = ["state_" + name for name in SIDE_FIELDS]
    if any(key not in values for key in keys):
        return None
    labels = {int(member.value): name for name, member in ContactSideKind.__members__.items()}
    static = int(ContactSideKind.STATIC.value)
    names = manifest["metadata"].get("owner_names") or {}
    sides = [values[key].reshape(len(values[key]), manifest["env_count"], -1)[:, env].astype(np.int64)
             for key in keys]
    def owner(kind, index):
        label = labels.get(kind, str(kind))
        table = names.get(label)
        if kind == static:
            return label
        return f"{label}:{table[index % len(table)]}" if table else f"{label}:{index}"
    pairs = {}
    for step, (a_kind, a_index, b_kind, b_index) in enumerate(zip(*sides)):
        active = ~((a_kind == static) & (b_kind == static))
        rows, counts = np.unique(np.stack([a_kind[active], a_index[active], b_kind[active], b_index[active]],
                                          axis=1), axis=0, return_counts=True)
        present = {}
        for (ka, ia, kb, ib), count in zip(rows.tolist(), counts.tolist()):
            pair = tuple(sorted((owner(ka, ia), owner(kb, ib))))
            present[pair] = present.get(pair, 0) + count
        for pair, count in present.items():
            entry = pairs.setdefault(pair, {"a": pair[0], "b": pair[1], "first_policy_step": step + 1,
                                            "steps": 0, "max_slots": 0, "slot_steps": 0})
            entry.update(last_policy_step=step + 1, steps=entry["steps"] + 1,
                         max_slots=max(entry["max_slots"], count), slot_steps=entry["slot_steps"] + count)
    return {"pairs": sorted(pairs.values(), key=lambda e: (-e["slot_steps"], e["a"], e["b"])),
            "scope": "Active contact slots at each recorded policy step, from the CONTACT_SIDE readouts."}


DAT_FAILURE_REASONS = ("radius", "motion", "degenerate", "overlap")
DAT_WITNESS_KINDS = ("RIGID", "LINK", "SURFACE", "STATIC")


def link_relation(tree, a, b):
    """Relate two env-local links through clusters joined by fixed joints."""
    def parent(index):
        value = tree[index]["parent_index"]
        return value if 0 <= value < len(tree) else None
    def cluster(index):
        while tree[index]["joint_type"] == "fixed" and parent(index) is not None:
            index = parent(index)
        return index
    ca, cb = cluster(a), cluster(b)
    if ca == cb:
        return "same_rigid_cluster"
    pa, pb = parent(ca), parent(cb)
    if (pa is not None and cluster(pa) == cb) or (pb is not None and cluster(pb) == ca):
        return "cluster_parent_child"
    return "separate_clusters"


def dat_failures(values, env, manifest):
    key = "state_DAT_FAILURE_WITNESS"
    if key not in values:
        return None
    names = manifest["metadata"].get("owner_names") or {}
    tree = manifest["metadata"].get("kinematic_tree")
    words = values[key].reshape(len(values[key]), manifest["env_count"], -1)[:, env]
    reasons = len(DAT_FAILURE_REASONS)
    def owner(code):
        if code == 0x3FFFFFFF:
            return "none"
        label, index = DAT_WITNESS_KINDS[code >> 28], code & 0x0FFFFFFF
        table = names.get(label)
        return f"{label}:{table[index]}" if table and index < len(table) else f"{label}:{index}"
    steps, pairs = [], {}
    for step, row in enumerate(words.tolist()):
        totals = dict(zip(DAT_FAILURE_REASONS, row[:reasons]))
        if any(totals.values()) or row[reasons]:
            steps.append({"policy_step": step + 1, **totals, "unrecorded": row[reasons]})
        slots = row[reasons + 1:]
        for packed, count, depth in zip(slots[0::3], slots[1::3], slots[2::3]):
            if not packed:
                continue
            reason = DAT_FAILURE_REASONS[(packed >> 60) - 1]
            codes = (packed >> 30) & 0x3FFFFFFF, packed & 0x3FFFFFFF
            a, b = owner(codes[0]), owner(codes[1])
            depth = float(np.array([depth], dtype=np.uint32).view(np.float32)[0])
            entry = pairs.setdefault((reason, a, b), {"reason": reason, "a": a, "b": b,
                "first_policy_step": step + 1, "steps": 0, "max_count": 0, "total": 0, "max_overlap_m": 0.0})
            if tree and all(code >> 28 == 1 and (code & 0x0FFFFFFF) < len(tree) for code in codes):
                entry["relation"] = link_relation(tree, codes[0] & 0x0FFFFFFF, codes[1] & 0x0FFFFFFF)
            entry.update(last_policy_step=step + 1, steps=entry["steps"] + 1,
                         max_count=max(entry["max_count"], count), total=entry["total"] + count,
                         max_overlap_m=max(entry["max_overlap_m"], depth))
    return {"steps": steps, "pairs": sorted(pairs.values(), key=lambda e: (-e["total"], e["reason"], e["a"], e["b"])),
            "scope": "Last DAT pass of each recorded policy step, from DAT_FAILURE_WITNESS; pairs are unordered. "
                     "max_overlap_m is the deepest separating-axis overlap of a failing primitive pair."}


def stream_analysis(values, env, manifest, limits, initial_velocity=None):
    metadata = manifest["metadata"]
    records = values["energy"][:, env].reshape(-1, len(E)).astype(np.float64)
    flags = values["energy_status"][:, env].reshape(-1)
    stages = values["stages"][:, env].reshape(-1, len(S), len(C)).astype(np.float64)
    counts = values["audit_counts"][:, env].reshape(-1, len(A))
    metrics = values["audit_metrics"][:, env].reshape(-1, len(M))
    readout_complete = bool((records[:, E.VALID] == 1).all())
    tolerance = float(metadata.get("solver_velocity_tolerance_mps", limits.velocity_tolerance_mps))
    energy = energy_analysis(records, flags, limits)
    momentum, momentum_residual = momentum_analysis(stages, records, flags, metadata, limits, tolerance,
                                                    particle_history(values, env, manifest, initial_velocity))
    bad_counts = stages[:, :, C.NONFINITE_PARTICLES:C.NONFINITE_LINKS + 1]
    finite = np.isfinite(stages).all(axis=2) & (bad_counts == 0).all(axis=2)
    bad = np.argwhere(~finite)
    first = None if not len(bad) else {"interval": int(bad[0, 0] + 1), "stage": S(int(bad[0, 1])).name,
        "nonfinite_entities": bad_counts[tuple(bad[0])].tolist()}
    contacts = contact_analysis(counts, metrics, limits, readout_complete)
    equation_error = stages[:, S.SOLVED, C.VBD_MOMENTUM_VELOCITY_ERROR]
    newton, vertex, measured = decode_maxima(values["newton_metrics"][:, env])
    solver_complete = (readout_complete and manifest["substeps"] == 1 and
                       metadata.get("solver_velocity_tolerance_mps") is not None)
    solver_pass = bool((equation_error <= tolerance).all() and (newton[:, 0] <= tolerance).all())
    contact_residual, contact_row, contact_measured = decode_maxima(values["contact_metrics"][:, env])
    solver_pass &= bool((contact_residual[:, :2] <= tolerance).all())
    truncations = values["dat_truncations"][:, env]
    dat_ratio = float(np.count_nonzero(truncations) / len(truncations))
    state = decision(readout_complete, bool(finite.all() and (values["env_status"][:, env] == 0).all()),
        invalid_readout_intervals=int(np.count_nonzero(records[:, E.VALID] != 1)),
        first_nonfinite_stage=first, environment_status_values=sorted(map(int, np.unique(values["env_status"][:, env]))))
    capacity = metadata.get("ogc_contact_capacity")
    complete_capacity = readout_complete and capacity is not None
    capacity_report = decision(complete_capacity, bool(capacity is not None and
        (values["ogc_contacts"][:, env] <= capacity).all() and (values["env_status"][:, env] == 0).all()),
        ogc_peak=int(values["ogc_contacts"][:, env].max()), ogc_capacity=capacity,
        scope="OGC reserve and explicit environment failure flags; other capacities need their own counters.")
    truncation = decision(readout_complete and manifest["substeps"] == 1, bool(dat_ratio < limits.dat_step_ratio and
        (values["dat_failures"][:, env] == 0).all() and (values["dat_query_limits"][:, env] == 0).all()),
        clipped_policy_steps=int(np.count_nonzero(truncations)), clipped_step_ratio=dat_ratio,
        contract="Substep counts are required; last-substep fields cannot certify multi-substep rates.")
    elapsed = np.cumsum(records[:, E.DT])
    force_work = values["force_residual_work"][:, env].reshape(-1)
    timeline = anomaly_timeline(records, counts, equation_error, newton[:, 0], force_work,
                                contact_residual, truncations, manifest, tolerance, limits)
    finite_r = np.flatnonzero(np.isfinite(records[:, E.RESIDUAL]) & (records[:, E.VALID] == 1))
    selected = finite_r[np.argsort(np.abs(records[finite_r, E.RESIDUAL]))[-10:][::-1]]
    events = []
    for index in selected:
        policy = index // manifest["substeps"]
        hints = []
        if equation_error[index] > tolerance:
            hints.append("force_equation_not_converged")
        if records[index, E.DAT_KINETIC_LOSS] != 0:
            hints.append("dat_changed_kinetic_energy")
        if records[index, E.NORMAL_LOSS] < -limits.negative_loss_tolerance_j:
            hints.append("normal_rows_added_energy")
        if records[index, E.FRICTION_LOSS] < -limits.negative_loss_tolerance_j:
            hints.append("tangent_rows_added_energy")
        events.append({"environment": env, "interval": int(index + 1), "time_s": float(elapsed[index]),
            "policy_step": int(policy + 1), "substep": int(index % manifest["substeps"]),
            "residual_j": float(records[index, E.RESIDUAL]), "force_residual_work_j": float(force_work[index]),
            "momentum_residual_kg_mps": momentum_residual[index].tolist(),
            "force_equation_velocity_error_mps": float(equation_error[index]),
            "newton_velocity_correction_mps": float(newton[policy, 0]),
            "newton_vertex": int(vertex[policy, 0]) if measured[policy, 0] else None,
            "observations": hints, "energy_terms": {c.name: float(records[index, c]) for c in E}})
    layers = {"state": state, "L2_energy": energy, "L2_momentum": momentum,
        "L3_contact": contacts, "capacity": capacity_report, "DAT": truncation,
        "solver": decision(solver_complete, solver_pass,
            max_force_equation_velocity_error_mps=float(equation_error.max()),
            max_newton_correction_mps=float(newton[:, 0].max()),
            max_contact_normal_residual_mps=float(contact_residual[:, 0].max()),
            max_contact_tangent_residual_mps=float(contact_residual[:, 1].max()),
            tolerance_mps=tolerance, mean_sweeps=float(values["sweeps"][:, env].mean()))}
    return {"environment": env, "intervals": len(records), "elapsed_s": float(elapsed[-1]),
            "layers": layers, "events": events, "timeline": timeline,
            "contact_owners": contact_owners(values, env, manifest),
            "dat_failures": dat_failures(values, env, manifest)}, records, stages


def anomaly_timeline(records, counts, force_error, newton_error, force_work,
                     contact_residual, truncations, manifest, tolerance, limits):
    elapsed = np.cumsum(records[:, E.DT])
    valid = records[:, E.VALID] == 1
    def occurrence(mask):
        indices = np.flatnonzero(mask & valid)
        first = None if not len(indices) else {"interval": int(indices[0] + 1),
            "policy_step": int(indices[0] // manifest["substeps"] + 1),
            "substep": int(indices[0] % manifest["substeps"]), "time_s": float(elapsed[indices[0]])}
        return {"first": first, "intervals": len(indices)}
    masks = {
        "contacts_observed": counts[:, A.CONTACTS] > 0,
        "force_equation_over_tolerance": force_error > tolerance,
        "positive_energy_residual": records[:, E.RESIDUAL] > 0,
        "normal_rows_added_energy": records[:, E.NORMAL_LOSS] < -limits.negative_loss_tolerance_j,
        "tangent_rows_added_energy": records[:, E.FRICTION_LOSS] < -limits.negative_loss_tolerance_j,
        "final_gap_over_slop": counts[:, A.GAP_VIOLATIONS] > 0,
    }
    if manifest["substeps"] == 1:
        masks.update(newton_correction_over_tolerance=newton_error > tolerance,
            normal_contact_residual_over_tolerance=contact_residual[:, 0] > tolerance,
            tangent_contact_residual_over_tolerance=contact_residual[:, 1] > tolerance,
            DAT_truncation=truncations > 0)
    timeline = {name: occurrence(mask) for name, mask in masks.items()}
    positive = masks["positive_energy_residual"] & valid
    timeline["invalid_readout_intervals"] = int(np.count_nonzero(~valid))
    timeline["force_residual_work_comparison"] = {
        "positive_residual_intervals": int(positive.sum()),
        "positive_intervals_with_force_work_at_least_residual": int(np.count_nonzero(
            positive & np.isfinite(force_work) & (force_work >= records[:, E.RESIDUAL]))),
        "max_positive_residual_less_force_work_j": float(np.maximum(
            records[positive, E.RESIDUAL] - force_work[positive], 0).max()) if positive.any() else 0.0,
        "interpretation": "An algebraic comparison, not proof of cause; original R is unchanged."}
    return timeline


def write_parquet(root, values):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p, e, s, _ = values["energy"].shape
    shape = (p, e, s)
    indices = np.indices(shape)
    columns = {"policy_step": indices[0].reshape(-1) + 1,
               "environment": indices[1].reshape(-1), "substep": indices[2].reshape(-1)}
    columns.update({c.name: values["energy"][..., c].reshape(-1) for c in E})
    columns["energy_coverage_status"] = values["energy_status"].reshape(-1)
    columns["force_residual_work_j"] = values["force_residual_work"].reshape(-1)
    pq.write_table(pa.table(columns), root / "ledger.parquet", compression="zstd")


def plot_diagnostics(root, env, records, stages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time = np.cumsum(records[:, E.DT])
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for col in (E.END_KINETIC, E.END_GRAVITY, E.END_ELASTIC):
        axes[0, 0].plot(time, records[:, col], label=col.name)
    axes[0, 0].set(ylabel="J", title="Physical state energy")
    for col in (E.RESIDUAL, E.NORMAL_LOSS, E.FRICTION_LOSS, E.DAT_KINETIC_LOSS):
        axes[0, 1].plot(time, records[:, col] * 1000, label=col.name)
    axes[0, 1].set(ylabel="mJ", title="Original balance and signed losses")
    for axis, name in enumerate("xyz"):
        axes[1, 0].plot(time, stages[:, S.END, C.VBD_DISCRETE_MOMENTUM_X + axis], label=name)
    axes[1, 0].set(ylabel="kg m/s", title="Step-local discrete momentum")
    axes[1, 1].semilogy(time, np.maximum(stages[:, S.SOLVED, C.VBD_MOMENTUM_VELOCITY_ERROR], 1e-16))
    axes[1, 1].set(ylabel="m/s", title="Physical force-equation velocity defect")
    for col in (E.KINEMATIC_ELASTIC_WORK, E.KINEMATIC_CONTACT_WORK, E.DRIVE_WORK, E.AERO_WORK):
        axes[2, 0].plot(time, np.cumsum(records[:, col]), label=col.name)
    axes[2, 0].set(ylabel="J", title="Actual input and boundary work")
    axes[2, 1].plot(time, stages[:, S.SOLVED, C.ENERGY_INJECTING_PASSIVE_ROWS], label="rows adding energy")
    axes[2, 1].plot(time, stages[:, S.SOLVED, C.ACTIVE_ROWS], label="active rows")
    axes[2, 1].set(ylabel="count", title="Constraint activity")
    for ax in axes.flat:
        ax.set_xlabel("Physical time (s)")
        ax.grid(alpha=.2)
        if ax.lines and ax.lines[0].get_label()[0] != "_":
            ax.legend(fontsize=7)
    fig.savefig(root / f"diagnostics_env{env}.png", dpi=160)
    plt.close(fig)


def make_report(source, output, *, plot=False):
    values, manifest = load_records(source, states=True)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    limits = DiagnosticThresholds(**manifest["thresholds"])
    geometry, initial_velocity = Path(source) / "initial.npz", None
    if geometry.exists() and sha256(geometry) == manifest.get("initial_geometry_sha256"):
        with np.load(geometry, allow_pickle=False) as initial:
            initial_velocity = initial["velocity"].copy()
    streams = []
    for env in range(manifest["env_count"]):
        stream, energy, stages = stream_analysis(values, env, manifest, limits, initial_velocity)
        streams.append(stream)
        if plot:
            plot_diagnostics(root, env, energy, stages)
    expected = manifest["metadata"].get("expected_policy_steps")
    horizon = decision(expected is not None, expected == manifest["steps"] and
        manifest.get("stop_reason") == "completed", actual_policy_steps=manifest["steps"],
        expected_policy_steps=expected, stop_reason=manifest.get("stop_reason"))
    layers = {"horizon": horizon, "L1_analytic_and_convergence": decision(False, False),
              "external_CCD": decision(False, False), "reference_engines": decision(False, False)}
    audit_path = Path(source) / "ipctk_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        complete = (audit.get("manifest_sha256") == sha256(Path(source) / "manifest.json") and
            audit.get("geometry_sha256") == manifest.get("initial_geometry_sha256") and
            manifest.get("initial_geometry_sha256") is not None and
            audit.get("configuration") == CCD_CONFIGURATION and
            audit.get("segments") == manifest["steps"] * manifest["env_count"] * manifest["substeps"] and
            isinstance(audit.get("events"), list) and
            audit.get("zero_distance_crossings") == len(audit["events"]))
        layers["external_CCD"] = decision(complete, audit.get("zero_distance_crossings") == 0,
                                           evidence=str(audit_path), result=audit)
    complete = all(layer["status"] == "passed" for layer in layers.values())
    complete &= all(layer["status"] == "passed" for stream in streams for layer in stream["layers"].values())
    report = {"schema_version": 1, "source": str(source), "source_manifest_sha256": sha256(Path(source) / "manifest.json"),
        "metadata": manifest["metadata"], "thresholds": manifest["thresholds"],
        "passes_full_physics_acceptance": bool(complete), "shared_layers": layers, "environments": streams,
        "interpretation": "Observations identify defects; residual work is never subtracted from the original R.",
        "coverage": "This report certifies measured channels only; other physical systems require their own covered balances."}
    for filename, name in (("particle_state_check.json", "independent_state_readout"),
                           ("elastic_state_check.json", "independent_elastic_readout")):
        state_check = Path(source) / filename
        if state_check.exists():
            check = json.loads(state_check.read_text())
            matched = (check.get("source_manifest_sha256") == report["source_manifest_sha256"] and
                       check.get("geometry_sha256") == manifest.get("initial_geometry_sha256"))
            report[name] = {"evidence_matches_trace": matched, "evidence": str(state_check), "result": check}
    write_json(root / "report.json", report)
    write_json(root / "events.json", [event for stream in streams for event in stream["events"]])
    write_json(root / "contact_audit.json", [stream["layers"]["L3_contact"] for stream in streams])
    write_json(root / "diagnostic_timeline.json", [stream["timeline"] for stream in streams])
    owners = [stream["contact_owners"] for stream in streams]
    if any(owners):
        write_json(root / "contact_owners.json", owners)
    failures = [stream["dat_failures"] for stream in streams]
    if any(failures):
        write_json(root / "dat_failures.json", failures)
    write_json(root / "perf.json", {"status": "unmeasured_gpu_latency",
        "host_step_wall_median_ms": float(np.nanmedian(values["step_wall_seconds"]) * 1000),
        "scope": manifest["timing_scope"], "five_process_equal_quality_comparison": "unmeasured"})
    write_parquet(root, values)
    lines = ["# Physics diagnostics", "", "Full physics acceptance: " + ("passed" if complete else "incomplete or failed"),
             "", "| Environment | Check | Status |", "|---|---|---|"]
    for name, layer in layers.items():
        lines.append(f"| all | {name} | {layer['status']} |")
    for stream in streams:
        for name, layer in stream["layers"].items():
            lines.append(f"| {stream['environment']} | {name} | {layer['status']} |")
    lines += ["", "## Evidence", "", "- Original thresholds: report.json",
              "- Energy rows: ledger.parquet", "- Worst intervals and raw terms: events.json",
              "- Row witnesses: contact_audit.json", "- Timing scope: perf.json",
              "- First occurrences and interval counts: diagnostic_timeline.json"]
    if any(owners):
        lines.append("- Contact owner pairs per recorded step: contact_owners.json")
    if any(failures):
        lines.append("- DAT failures by reason and owner pair: dat_failures.json")
    lines += ["", "Missing coverage, analytic checks or geometry evidence prevents a full acceptance result."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return report
