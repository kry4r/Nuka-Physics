"""Quantitative acceptance of evidence bound to a captured physics trace."""

import json
import numbers
from pathlib import Path

import numpy as np

from ..energy import EnergyColumn as E
from .capture import sha256
from .schema import AuditCount as A, Stage as S, StageColumn as C


def _decision(complete, passed, **details):
    return {"status": "unmeasured" if not complete else "passed" if passed else "failed", **details}


def _scalar(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and np.isfinite(value)


def evidence_path(source, filename):
    if not isinstance(filename, str) or Path(filename).is_absolute():
        raise ValueError("evidence must name a relative file in the captured run")
    root = Path(source).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("evidence file is absent or outside the captured run")
    return path


def quantitative_evidence(source, manifest, layer, filename):
    definitions = manifest["metadata"].get("quantitative_acceptance", {}).get(layer)
    path = Path(source) / filename
    if not isinstance(definitions, list) or not definitions or not path.is_file():
        return _decision(False, False, reason="Declared checks and quantitative evidence are required",
                         evidence=str(path), required_checks=definitions)
    payload = json.loads(path.read_text())
    matched = (payload.get("source_manifest_sha256") == sha256(Path(source) / "manifest.json") and
               payload.get("physical_input_sha256") == manifest["metadata"].get("physical_input_sha256") and
               manifest["metadata"].get("physical_input_sha256") is not None)
    checks, names = [], set()
    for definition in definitions:
        if not isinstance(definition, dict):
            return _decision(False, False, reason="Acceptance checks must be quantitative definitions")
        name, unit = definition.get("name"), definition.get("unit")
        lower, upper = definition.get("minimum"), definition.get("maximum")
        valid = (isinstance(name, str) and bool(name) and name not in names and
                 isinstance(unit, str) and bool(unit) and (lower is not None or upper is not None) and
                 (lower is None or _scalar(lower)) and (upper is None or _scalar(upper)))
        valid = valid and (lower is None or upper is None or lower <= upper)
        if isinstance(name, str):
            names.add(name)
        else:
            name = None
        result = payload.get("checks", {}).get(name, {})
        artifacts = result.get("artifacts", [])
        hashes = bool(artifacts)
        try:
            for artifact in artifacts:
                hashes &= sha256(evidence_path(source, artifact.get("file"))) == artifact.get("sha256")
        except (ValueError, OSError):
            hashes = False
        value = result.get("value")
        complete = bool(valid and matched and hashes and result.get("unit") == unit and _scalar(value))
        passed = complete and (lower is None or value >= lower) and (upper is None or value <= upper)
        checks.append(_decision(complete, passed, name=name, unit=unit, value=value,
            minimum=lower, maximum=upper, artifact_hashes_match=bool(hashes), artifacts=artifacts))
    complete = bool(matched and checks and all(c["status"] != "unmeasured" for c in checks))
    return _decision(complete, all(c["status"] == "passed" for c in checks), evidence=str(path),
                     evidence_matches_trace=matched, checks=checks)


def independent_readout(source, manifest, filename, *, elastic=False):
    path = Path(source) / filename
    if not path.is_file():
        return _decision(False, False, evidence=str(path), reason="Independent readout evidence is required")
    payload = json.loads(path.read_text())
    matched = (payload.get("source_manifest_sha256") == sha256(Path(source) / "manifest.json") and
               payload.get("geometry_sha256") == manifest.get("initial_geometry_sha256") and
               manifest.get("initial_geometry_sha256") is not None and payload.get("status") == "measured" and
               payload.get("intervals_per_environment") == manifest["steps"] and
               payload.get("environments") == manifest["env_count"])
    if elastic:
        values = {"elastic_energy_j": payload.get("max_elastic_energy_difference_j"),
                  "internal_gradient_sum_n": payload.get("max_internal_gradient_sum_n")}
    else:
        names = ("kinetic_j", "gravity_j", "mass_kg", "linear_momentum_kg_mps",
                 "angular_momentum_kg_m2ps", "discrete_momentum_kg_mps")
        values = {name: payload.get("errors", {}).get(name, {}).get("max_absolute_difference") for name in names}
    bounds = manifest["metadata"].get("independent_readout_limits", {})
    complete = bool(matched and all(_scalar(bounds.get(name)) and bounds[name] >= 0 for name in values))
    passed = all(_scalar(value) and 0 <= value <= bounds.get(name, -1) for name, value in values.items())
    if not elastic:
        passed &= payload.get("bdf_vertex_counts_match") is True
    return _decision(complete, bool(passed), evidence=str(path), evidence_matches_trace=bool(matched),
                     max_absolute_differences=values, absolute_limits=bounds, result=payload,
                     scope="Readout accuracy and internal-force cancellation; separate from physical conservation")


def solver_budget_evidence(source, manifest):
    entries = manifest.get("budget_replays", [])
    tolerance = manifest["metadata"].get("solver_velocity_tolerance_mps")
    if not entries or not _scalar(tolerance) or tolerance <= 0 or manifest["substeps"] != 1:
        return _decision(False, False, reason="Hashed single-substep replays and the production tolerance are required")
    reports = []
    for entry in entries:
        try:
            path = evidence_path(source, entry.get("file"))
            if sha256(path) != entry.get("sha256"):
                raise ValueError("Replay hash differs from the manifest")
            with np.load(path, allow_pickle=False) as archive:
                budgets, actual = archive["budgets"], archive["actual_budgets"]
                if (budgets.ndim != 1 or len(budgets) < 2 or budgets.dtype.kind != "u" or
                        actual.dtype.kind != "u" or not np.array_equal(budgets, actual) or
                        not np.array_equal(budgets, entry.get("requested_budgets")) or
                        not np.array_equal(actual, entry.get("actual_budgets")) or
                        not np.all(np.diff(budgets.astype(np.int64)) > 0) or
                        (budgets < 1).any() or (budgets > 65535).any()):
                    raise ValueError("Replay requires increasing, distinct, verified active budgets")
                size, envs = len(budgets), manifest["env_count"]
                expected = {"energy": (size, envs, 1, len(E)),
                    "energy_status": (size, envs, 1), "sweeps": (size, envs),
                    "stages": (size, envs, 1, len(S), len(C)),
                    "newton_metrics": (size, envs, 2), "contact_metrics": (size, envs, 8),
                    "audit_counts": (size, envs, 1, len(A)), "env_status": (size, envs)}
                values = {key: archive[key].copy() for key in expected}
                if any(values[key].shape != shape for key, shape in expected.items()):
                    raise ValueError("Replay measurement extents differ from the captured world")
            packed_vertex, packed_contact = values["newton_metrics"], values["contact_metrics"]
            if any(value.dtype.kind != "u" or value.dtype.itemsize != 8
                   for value in (packed_vertex, packed_contact)):
                raise ValueError("Replay residuals must preserve packed unsigned measurements")
            newton = (packed_vertex >> np.uint64(32)).astype(np.uint32).view(np.float32)
            contact = (packed_contact >> np.uint64(32)).astype(np.uint32).view(np.float32)
            stage, energy = values["stages"][:, :, 0], values["energy"][:, :, 0]
            begin = stage[:, :, S.BEGIN, :C.ANGULAR_MOMENTUM_Z + 1]
            same_begin = np.array_equal(begin, np.broadcast_to(begin[:1], begin.shape))
            has_contact = values["audit_counts"][:, :, 0, A.CONTACTS] > 0
            measured = bool((packed_vertex[..., 0] != 0).all() and
                ((packed_contact[..., :2] != 0) | ~has_contact[..., None]).all())
            valid = bool(same_begin and measured and (energy[..., E.VALID] == 1).all() and
                (values["energy_status"] == 0).all() and (values["env_status"] == 0).all() and
                (values["sweeps"] > 0).all() and (values["sweeps"] <= budgets[:, None]).all() and
                np.isfinite(energy).all() and np.isfinite(stage).all() and
                (energy[..., E.DT] > 0).all() and
                np.all(stage[..., C.DT] == energy[..., E.DT, None]) and
                np.isfinite(newton).all() and np.isfinite(contact).all())
            force = stage[:, :, S.SOLVED, C.VBD_MOMENTUM_VELOCITY_ERROR]
            gates = (force <= tolerance) & (newton[..., 0] <= tolerance) & (contact[..., :2] <= tolerance).all(axis=-1)
            rows = [{"active_budget": int(budget), "environments": [
                {"environment": env, "force_velocity_defect_mps": float(force[index, env]),
                 "newton_correction_mps": float(newton[index, env, 0]),
                 "normal_residual_mps": float(contact[index, env, 0]),
                 "tangent_residual_mps": float(contact[index, env, 1]),
                 "within_production_tolerance": bool(gates[index, env])} for env in range(envs)]}
                for index, budget in enumerate(budgets)]
            reports.append(_decision(valid, bool(gates[-1].all()), evidence=str(path),
                checkpoint_policy_step=entry.get("checkpoint_policy_step"),
                production_budget=entry.get("production_budget"), same_begin_state_quantities=same_begin,
                tolerance_mps=tolerance, budgets=rows))
        except (ValueError, OSError, KeyError, TypeError) as error:
            reports.append(_decision(False, False, evidence=entry.get("file"), reason=str(error)))
    complete = all(report["status"] != "unmeasured" for report in reports)
    return _decision(complete, all(report["status"] == "passed" for report in reports), replays=reports,
        scope="Largest replay budget must meet the production residual tolerance; production-budget quality is checked separately")
