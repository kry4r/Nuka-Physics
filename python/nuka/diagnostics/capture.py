"""Read-only capture and append-only chunks from any production World."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .._nuka_ext import Field
from ..energy import EnergyColumn, EnergyLedgerSampler
from .schema import (AuditCount, AuditMetric, DiagnosticThresholds, Stage, StageColumn,
                     VbdSolveAuditColumn, stage_units, vbd_solve_audit_units)


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

    def __init__(self, world, output, metadata, *, env_count=1, chunk_steps=128,
                 state_fields=(), thresholds=None):
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
        self.state_fields = tuple(state_fields)
        self.thresholds = thresholds or DiagnosticThresholds()
        world.download_field(Field.PHYSICS_STAGE_METRICS)
        world.download_field(Field.CONTACT_AUDIT_COUNTS)
        world.download_field(Field.CONTACT_AUDIT_METRICS)
        for field in self.state_fields:
            world.download_field(field)
        self.energy = EnergyLedgerSampler(world, self.env_count)
        self.substeps = self.energy.substeps
        self.manifest = {
            "schema_version": 1, "status": "recording", "metadata": metadata,
            "env_count": self.env_count, "substeps": self.substeps,
            "axes": ["policy_step", "environment", "substep"],
            "thresholds": self.thresholds.to_dict(), "chunks": [], "steps": 0,
            "stages": [c.name for c in Stage], "stage_columns": [c.name for c in StageColumn],
            "stage_units": [stage_units(c) for c in StageColumn],
            "energy_columns": [c.name for c in EnergyColumn],
            "audit_counts": [c.name for c in AuditCount], "audit_metrics": [c.name for c in AuditMetric],
            "state_fields": [f.name for f in state_fields],
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
            "newton_metrics": read(Field.VBD_SOLVE_METRICS, (self.env_count, 2)),
            "contact_metrics": read(Field.CONTACT_SOLVE_METRICS, (self.env_count, 8)),
            "step_wall_seconds": np.asarray(step_wall_seconds),
        }
        for field in self.state_fields if state_fields is None else state_fields:
            sample["state_" + field.name] = np.asarray(self.world.download_field(field)).copy()
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
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported diagnostic schema")
    columns = {"stages": Stage, "stage_columns": StageColumn,
               "audit_counts": AuditCount, "audit_metrics": AuditMetric}
    for name, enum in columns.items():
        if manifest.get(name) != [column.name for column in enum]:
            raise ValueError(f"diagnostic column order mismatch: {name}")
    if "energy_columns" in manifest and manifest["energy_columns"] != [c.name for c in EnergyColumn]:
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
    shapes = {"energy": extent + (len(EnergyColumn),), "energy_status": extent,
              "stages": extent + (len(Stage), len(StageColumn)),
              "audit_counts": extent + (len(AuditCount),),
              "audit_metrics": extent + (len(AuditMetric),), "force_residual_work": extent,
              "env_status": environments, "ogc_contacts": environments,
              "dat_truncations": environments, "dat_failures": environments,
              "dat_query_limits": environments, "sweeps": environments,
              "newton_metrics": environments + (2,), "contact_metrics": environments + (8,),
              "step_wall_seconds": (), "policy_step": ()}
    floating = {"energy", "stages", "force_residual_work", "step_wall_seconds"}
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
