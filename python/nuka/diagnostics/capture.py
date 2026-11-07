"""Read-only capture and append-only chunks from any production World."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .._nuka_ext import Field
from ..energy import EnergyColumn, EnergyLedgerSampler
from .schema import (AuditCount, AuditMetric, DiagnosticThresholds, SolverPhase, SolverResidual, SolverWork, Stage,
                     StageColumn, VbdSolveAuditColumn, stage_units, vbd_solve_audit_units)

INTERVAL_STATE_FIELDS = (
    "INTERVAL_STATE_DT", "INTERVAL_PARTICLE_POSITION", "INTERVAL_PARTICLE_VELOCITY",
    "INTERVAL_PARTICLE_F", "INTERVAL_PARTICLE_C", "INTERVAL_PARTICLE_PLASTIC_F",
    "INTERVAL_PARTICLE_PLASTIC", "INTERVAL_BODY_POSE", "INTERVAL_BODY_LINEAR_VELOCITY",
    "INTERVAL_BODY_ANGULAR_VELOCITY",
    "INTERVAL_Q", "INTERVAL_QDOT", "INTERVAL_JOINT_ROTATION",
    "INTERVAL_JOINT_ANGULAR_VELOCITY", "INTERVAL_LINK_VELOCITY", "INTERVAL_BASE_POSE",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    Path(path).write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")


class DiagnosticSession:
    """Request all readouts before the first step; preserve completed chunks on failure."""

    @staticmethod
    def performance(device, world, output, metadata):
        from .performance import PerformanceCapture
        return PerformanceCapture(device, world, output, metadata)

    def __init__(self, world, output, metadata, *, env_count=1, chunk_steps=128,
                 state_fields=(), thresholds=None, trace_sweeps=False, interval_states=False):
        self.world = world
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.env_count = int(env_count)
        self.chunk_steps = int(chunk_steps)
        if self.chunk_steps < 1:
            raise ValueError("chunk_steps must be positive")
        self.closed = False
        self.samples = []
        self.steps = 0
        self.interval_states = bool(interval_states)
        counts = world.resource_info() if interval_states else {}
        interval_fields = [name for name in INTERVAL_STATE_FIELDS if interval_states and (
            name == "INTERVAL_STATE_DT" or
            (name.startswith("INTERVAL_PARTICLE_") and counts["particles_per_env"] > 0) or
            (name.startswith("INTERVAL_BODY_") and counts["bodies_per_env"] > 0) or
            (name in ("INTERVAL_Q", "INTERVAL_QDOT", "INTERVAL_JOINT_ROTATION",
                      "INTERVAL_JOINT_ANGULAR_VELOCITY", "INTERVAL_LINK_VELOCITY", "INTERVAL_BASE_POSE")
             and world.base_link_count > 0))]
        requested_states = tuple(getattr(Field, name) for name in interval_fields)
        self.state_fields = tuple(dict.fromkeys(tuple(state_fields) + requested_states))
        self.thresholds = thresholds or DiagnosticThresholds()
        # Sweep telemetry keeps each env's residuals per sweep and the solve's phase times.
        self.trace_sweeps = world.velocity_iterations() if trace_sweeps else 0
        if self.trace_sweeps:
            world.set_telemetry_mask(world.telemetry_mask() | 3)
        world.download_field(Field.PHYSICS_STAGE_METRICS)
        world.download_field(Field.CONTACT_AUDIT_COUNTS)
        world.download_field(Field.CONTACT_AUDIT_METRICS)
        for field in self.state_fields:
            world.download_field(field)
        self.energy = EnergyLedgerSampler(world, self.env_count)
        self.substeps = self.energy.substeps
        self.manifest = {
            "schema_version": 2, "status": "recording", "metadata": metadata,
            "env_count": self.env_count, "substeps": self.substeps,
            "axes": ["policy_step", "environment", "substep"],
            "thresholds": self.thresholds.to_dict(), "chunks": [], "steps": 0,
            "stages": [c.name for c in Stage], "stage_columns": [c.name for c in StageColumn],
            "stage_units": [stage_units(c) for c in StageColumn],
            "energy_columns": [c.name for c in EnergyColumn],
            "audit_counts": [c.name for c in AuditCount], "audit_metrics": [c.name for c in AuditMetric],
            "solver_residual_columns": [c.name for c in SolverResidual],
            "solver_residual_units": "m/s", "trace_sweeps": self.trace_sweeps,
            "sweep_budget": world.velocity_iterations(),
            "solver_phases": [c.name for c in SolverPhase],
            "solver_work_columns": [c.name for c in SolverWork],
            "dense_motion_units": "m/s",
            "state_fields": [f.name for f in self.state_fields],
            "contact_motion_policy": "discrete_constraints",
            "interval_state_fields": interval_fields,
            "interval_state_axes": ["environment", "substep", "owner", "component"],
            "timing_scope": "Host step call wall time; excludes capture and file I/O; not GPU event latency.",
        }
        audit_field = getattr(Field, "VBD_SOLVE_AUDIT", None)
        if audit_field is not None and audit_field in self.state_fields:
            self.manifest["vbd_solve_audit_columns"] = [c.name for c in VbdSolveAuditColumn]
            self.manifest["vbd_solve_audit_units"] = [vbd_solve_audit_units(c) for c in VbdSolveAuditColumn]
            self.manifest["vbd_solve_audit_scope"] = "last solve call"
        write_json(self.output / "manifest.json", self.manifest)

    def read(self, *, step_wall_seconds=np.nan, controls=None, state_fields=None):
        energy, energy_status = self.energy.sample()
        extent = (self.env_count, self.substeps)
        def read(field, shape):
            return np.asarray(self.world.download_field(field)).reshape(shape).copy()
        sample = {
            "energy": energy, "energy_status": energy_status,
            "stages": read(Field.PHYSICS_STAGE_METRICS, extent + (len(Stage), len(StageColumn))),
            "audit_counts": read(Field.CONTACT_AUDIT_COUNTS, extent + (len(AuditCount),)),
            "audit_metrics": read(Field.CONTACT_AUDIT_METRICS, extent + (len(AuditMetric),)),
            "force_residual_work": read(Field.VBD_FORCE_RESIDUAL_WORK, extent),
            "env_status": read(Field.ENV_STATUS, (self.env_count,)),
            "ogc_contacts": read(Field.OGC_CONTACT_COUNT, (self.env_count,)),
            "dat_truncations": read(Field.DAT_TRUNCATION_COUNT, (self.env_count,)),
            "dat_failures": read(Field.DAT_FAILURE_COUNT, (self.env_count,)),
            "dat_query_limits": read(Field.DAT_QUERY_LIMIT_COUNT, (self.env_count,)),
            "sweeps": read(Field.VBD_VELOCITY_SWEEP_COUNT, (self.env_count,)),
            "solver_residual": read(Field.SOLVER_RESIDUAL, (self.env_count, len(SolverResidual))),
            "newton_metrics": read(Field.VBD_SOLVE_METRICS, (self.env_count, 2)),
            "contact_metrics": read(Field.CONTACT_SOLVE_METRICS, (self.env_count, 8)),
            "step_wall_seconds": np.asarray(step_wall_seconds),
        }
        if self.trace_sweeps:
            sample["sweep_trace"] = read(Field.SOLVER_SWEEP_TRACE,
                                         (self.env_count, self.trace_sweeps, len(SolverResidual)))
            sample["work_trace"] = read(Field.SOLVER_WORK_TRACE,
                                        (self.env_count, self.trace_sweeps, len(SolverWork)))
            sample["dense_motion_trace"] = read(Field.SOLVER_DENSE_MOTION_TRACE, (self.env_count, self.trace_sweeps))
            sample["phase_ns"] = read(Field.SOLVER_PHASE_TIME, (self.env_count, len(SolverPhase)))
        for field in self.state_fields if state_fields is None else state_fields:
            value = np.asarray(self.world.download_field(field)).copy()
            if self.interval_states and field.name in INTERVAL_STATE_FIELDS:
                if field.name == "INTERVAL_STATE_DT":
                    value = value.reshape(extent)
                else:
                    width = (7 if field.name in ("INTERVAL_BODY_POSE", "INTERVAL_BASE_POSE") else
                             6 if field.name == "INTERVAL_LINK_VELOCITY" else
                             4 if field.name == "INTERVAL_JOINT_ROTATION" else
                             9 if field.name.endswith(("_F", "_C")) else
                             1 if field.name in ("INTERVAL_PARTICLE_PLASTIC", "INTERVAL_Q", "INTERVAL_QDOT") else 3)
                    count = value.size // (self.env_count * self.substeps * width)
                    value = value.reshape(extent + (count, width))
            sample["state_" + field.name] = value
        if controls is not None:
            sample["controls"] = np.asarray(controls).copy()
        return sample

    def record(self, *, step_wall_seconds=np.nan, controls=None):
        if self.closed:
            raise RuntimeError("diagnostic session is closed")
        sample = self.read(step_wall_seconds=step_wall_seconds, controls=controls)
        self.steps += 1
        sample["policy_step"] = np.asarray(self.steps, dtype=np.uint64)
        self.samples.append(sample)
        if len(self.samples) >= self.chunk_steps:
            self.flush()
        return sample

    def step(self, *, controls=None):
        started = time.perf_counter()
        self.world.step()
        elapsed = time.perf_counter() - started
        return self.record(step_wall_seconds=elapsed, controls=controls)

    def replay_budgets(self, budgets, *, state_fields=(), controls=None):
        """Replay the next step from one checkpoint per active solve iteration budget.
        The world returns to the checkpoint at the production budget, so the record is unchanged."""
        if self.closed:
            raise RuntimeError("diagnostic session is closed")
        budgets = tuple(int(budget) for budget in budgets)
        state_fields = tuple(state_fields)
        if not budgets or any(budget < 1 or budget > 65535 for budget in budgets):
            raise ValueError("replay budgets must contain iteration counts in 1..65535")
        path = self.output / f"budget_replay_step_{self.steps + 1:06d}.npz"
        if path.exists():
            raise FileExistsError(path)
        production = self.world.velocity_iterations()
        if not 1 <= production <= 65535:
            raise ValueError("the production iteration budget cannot be restored through the runtime API")
        samples, actual_budgets = [], []
        with self.world.capture_checkpoint() as checkpoint:
            try:
                for budget in budgets:
                    self.world.set_velocity_iterations(budget)
                    actual = self.world.velocity_iterations()
                    if actual != budget:
                        raise RuntimeError(f"requested iteration budget {budget}, active budget is {actual}")
                    self.world.restore_checkpoint(checkpoint)
                    self.world.synchronize()
                    started = time.perf_counter()
                    self.world.step()
                    self.world.synchronize()
                    elapsed = time.perf_counter() - started
                    samples.append(self.read(step_wall_seconds=elapsed, controls=controls, state_fields=state_fields))
                    actual_budgets.append(actual)
            finally:
                try:
                    self.world.set_velocity_iterations(production)
                finally:
                    self.world.restore_checkpoint(checkpoint)
        payload = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
        with path.open("xb") as target:
            np.savez_compressed(target, budgets=np.asarray(budgets, dtype=np.uint32),
                actual_budgets=np.asarray(actual_budgets, dtype=np.uint32),
                timing_scope=np.asarray("Host step wall time through World.synchronize(); excludes checkpoint restore, "
                    "budget changes, diagnostic readout and file I/O; includes GPU completion and synchronization overhead; "
                    "may include graph capture after a budget rebuild; not GPU event latency."), **payload)
        replay_record = {
            "file": path.name, "sha256": sha256(path), "checkpoint_policy_step": self.steps,
            "requested_budgets": list(budgets), "actual_budgets": actual_budgets,
            "production_budget": production}
        audit_field = getattr(Field, "VBD_SOLVE_AUDIT", None)
        if audit_field is not None and audit_field in state_fields:
            replay_record["vbd_solve_audit_columns"] = [c.name for c in VbdSolveAuditColumn]
            replay_record["vbd_solve_audit_units"] = [vbd_solve_audit_units(c) for c in VbdSolveAuditColumn]
            replay_record["vbd_solve_audit_scope"] = "last solve call"
        self.manifest.setdefault("budget_replays", []).append(replay_record)
        write_json(self.output / "manifest.json", self.manifest)
        return path

    def flush(self):
        if not self.samples:
            return
        keys = self.samples[0].keys()
        if any(sample.keys() != keys for sample in self.samples):
            raise ValueError("capture columns changed inside a chunk")
        payload = {key: np.stack([row[key] for row in self.samples]) for key in keys}
        index = len(self.manifest["chunks"])
        path = self.output / f"chunk_{index:06d}.npz"
        with path.open("xb") as target:
            np.savez_compressed(target, **payload)
        self.manifest["chunks"].append({"file": path.name, "sha256": sha256(path),
            "first_policy_step": int(payload["policy_step"][0]), "steps": len(self.samples)})
        self.manifest["steps"] = self.steps
        self.samples.clear()
        write_json(self.output / "manifest.json", self.manifest)

    def close(self, reason="completed", *, error=None):
        if self.closed:
            return
        self.flush()
        self.manifest.update(status="closed", stop_reason=reason, error=error)
        write_json(self.output / "manifest.json", self.manifest)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.close("exception" if error else "completed", error=None if error is None else repr(error))


def iter_record_blocks(root, *, states=False):
    """Yield validated chunks; exhaust the iterator to verify the declared complete step extent."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("diagnostic manifest must be an object")
    if not isinstance(manifest.get("metadata"), dict):
        raise ValueError("diagnostic metadata must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] not in (1, 2):
        raise ValueError("unsupported diagnostic schema")
    stage_columns = list(StageColumn) if manifest["schema_version"] == 2 else list(StageColumn)[:39]
    columns = {"stages": Stage, "stage_columns": stage_columns,
               "audit_counts": AuditCount, "audit_metrics": AuditMetric}
    for name, enum in columns.items():
        if manifest.get(name) != [column.name for column in enum]:
            raise ValueError(f"diagnostic column order mismatch: {name}")
    energy_columns = manifest.get("energy_columns", [c.name for c in EnergyColumn][:31])
    if (not isinstance(energy_columns, list) or len(energy_columns) not in (31, 43, 44, len(EnergyColumn)) or
            energy_columns != [c.name for c in EnergyColumn][:len(energy_columns)]):
        raise ValueError("diagnostic energy column order mismatch")
    for name in ("env_count", "substeps", "steps"):
        if type(manifest.get(name)) is not int or manifest[name] < 1:
            raise ValueError(f"diagnostic {name} must be a positive integer")
    state_fields = manifest.get("state_fields", [])
    if (not isinstance(state_fields, list) or
            any(not isinstance(name, str) or not name for name in state_fields) or
            len(state_fields) != len(set(state_fields))):
        raise ValueError("diagnostic state_fields must contain distinct field names")
    if not isinstance(manifest.get("chunks"), list) or not manifest["chunks"]:
        raise ValueError("no completed physics records")
    extent = (manifest["env_count"], manifest["substeps"])
    environments = (manifest["env_count"],)
    shapes = {"energy": extent + (len(energy_columns),), "energy_status": extent,
              "stages": extent + (len(Stage), len(stage_columns)),
              "audit_counts": extent + (len(AuditCount),),
              "audit_metrics": extent + (len(AuditMetric),), "force_residual_work": extent,
              "env_status": environments, "ogc_contacts": environments,
              "dat_truncations": environments, "dat_failures": environments,
              "dat_query_limits": environments, "sweeps": environments,
              "newton_metrics": environments + (2,), "contact_metrics": environments + (8,),
              "step_wall_seconds": (), "policy_step": ()}
    floating = {"energy", "stages", "force_residual_work", "step_wall_seconds"}
    if "solver_residual_columns" in manifest:
        if manifest["solver_residual_columns"] != [c.name for c in SolverResidual]:
            raise ValueError("diagnostic solver residual column order mismatch")
        shapes["solver_residual"] = environments + (len(SolverResidual),)
        floating.add("solver_residual")
        if manifest["trace_sweeps"]:
            shapes["sweep_trace"] = environments + (manifest["trace_sweeps"], len(SolverResidual))
            shapes["phase_ns"] = environments + (len(SolverPhase),)
            floating.add("sweep_trace")
    if "solver_work_columns" in manifest and manifest["trace_sweeps"]:
        shapes["work_trace"] = environments + (manifest["trace_sweeps"], len(SolverWork))
    if "dense_motion_units" in manifest and manifest["trace_sweeps"]:
        shapes["dense_motion_trace"] = environments + (manifest["trace_sweeps"],)
        floating.add("dense_motion_trace")
    unsigned = shapes.keys() - floating
    required_states = {"state_" + name for name in state_fields}
    state_shapes, columns, paths = {}, None, set()
    next_step = 1
    for entry in manifest["chunks"]:
        if not isinstance(entry, dict):
            raise ValueError("diagnostic chunk entries must be objects")
        for name in ("steps", "first_policy_step"):
            if type(entry.get(name)) is not int or entry[name] < 1:
                raise ValueError(f"diagnostic chunk {name} must be a positive integer")
        filename = entry.get("file")
        if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
            raise ValueError("diagnostic chunk requires a relative filename")
        path = (root / filename).resolve()
        if path == root or not path.is_relative_to(root) or path in paths:
            raise ValueError("diagnostic chunk filenames must be distinct and stay inside the source")
        paths.add(path)
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"evidence hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            if len(keys) != len(data.files) or not shapes.keys() <= keys or not required_states <= keys:
                raise ValueError("diagnostic chunk has missing or duplicate columns")
            if columns is not None and keys != columns:
                raise ValueError("diagnostic capture columns changed between chunks")
            columns = keys
            block = {}
            for name in data.files:
                value = data[name]
                if value.ndim < 1 or value.shape[0] != entry["steps"]:
                    raise ValueError(f"diagnostic step extent mismatch: {name}")
                if name in shapes and value.shape != (entry["steps"],) + shapes[name]:
                    raise ValueError(f"diagnostic extent mismatch: {name}")
                if name in unsigned and value.dtype.kind != "u":
                    raise ValueError(f"diagnostic unsigned integer dtype required: {name}")
                if name in ("audit_metrics", "newton_metrics", "contact_metrics") and value.dtype.itemsize != 8:
                    raise ValueError(f"diagnostic packed maxima require 64-bit words: {name}")
                if name in floating and value.dtype.kind != "f":
                    raise ValueError(f"diagnostic floating dtype required: {name}")
                if name.startswith("state_"):
                    if value.dtype.kind not in "fiu":
                        raise ValueError(f"diagnostic state fields require numeric data: {name}")
                    shape = value.shape[1:]
                    if name in state_shapes and state_shapes[name] != shape:
                        raise ValueError(f"diagnostic state extent changed between chunks: {name}")
                    state_shapes[name] = shape
                if states or not name.startswith("state_") and name != "controls":
                    block[name] = value
        if entry["first_policy_step"] != next_step or not np.array_equal(
                block["policy_step"], np.arange(next_step, next_step + entry["steps"], dtype=np.uint64)):
            raise ValueError("physics evidence has a discontinuity")
        next_step += entry["steps"]
        if next_step - 1 > manifest["steps"]:
            raise ValueError("diagnostic chunks exceed the declared total steps")
        if len(stage_columns) < len(StageColumn):
            block["stages"] = np.pad(block["stages"],
                [(0, 0)] * (block["stages"].ndim - 1) + [(0, len(StageColumn) - len(stage_columns))])
        if len(energy_columns) < len(EnergyColumn):
            block["energy"] = np.pad(block["energy"],
                [(0, 0)] * (block["energy"].ndim - 1) + [(0, len(EnergyColumn) - len(energy_columns))],
                constant_values=np.nan)
        yield block, manifest
    if next_step - 1 != manifest["steps"]:
        raise ValueError("diagnostic chunks do not cover the declared total steps")


def load_records(root, *, states=False):
    blocks, manifest = [], None
    for block, manifest in iter_record_blocks(root, states=states):
        blocks.append(block)
    if not blocks:
        raise ValueError("no completed physics records")
    values = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    return values, manifest
