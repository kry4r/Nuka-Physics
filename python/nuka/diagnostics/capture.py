"""Read-only capture and append-only chunks from any production World."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .._nuka_ext import Field
from ..energy import EnergyColumn, EnergyLedgerSampler
from .schema import AuditCount, AuditMetric, DiagnosticThresholds, Stage, StageColumn, stage_units


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
        """Replay the next step from one checkpoint per velocity sweep budget.
        The world returns to the checkpoint at the production budget, so the record is unchanged."""
        path = self.output / f"budget_replay_step_{self.steps + 1:06d}.npz"
        if path.exists():
            raise FileExistsError(path)
        production = self.world.velocity_iterations()
        samples = []
        with self.world.capture_checkpoint() as checkpoint:
            try:
                for budget in budgets:
                    self.world.set_velocity_iterations(int(budget))
                    self.world.restore_checkpoint(checkpoint)
                    started = time.perf_counter()
                    self.world.step()
                    elapsed = time.perf_counter() - started
                    samples.append(self.read(step_wall_seconds=elapsed, controls=controls, state_fields=state_fields))
            finally:
                self.world.set_velocity_iterations(production)
                self.world.restore_checkpoint(checkpoint)
        payload = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
        with path.open("xb") as target:
            np.savez_compressed(target, budgets=np.asarray(budgets, dtype=np.uint32), **payload)
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


def load_records(root, *, states=False):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported diagnostic schema")
    columns = {"stages": Stage, "stage_columns": StageColumn,
               "audit_counts": AuditCount, "audit_metrics": AuditMetric}
    for name, enum in columns.items():
        if manifest.get(name) != [column.name for column in enum]:
            raise ValueError(f"diagnostic column order mismatch: {name}")
    if "energy_columns" in manifest and manifest["energy_columns"] != [c.name for c in EnergyColumn]:
        raise ValueError("diagnostic energy column order mismatch")
    if min(manifest["env_count"], manifest["substeps"]) < 1:
        raise ValueError("diagnostic environment and substep extents must be positive")
    extent = (manifest["env_count"], manifest["substeps"])
    shapes = {"energy": extent + (len(EnergyColumn),), "energy_status": extent,
              "stages": extent + (len(Stage), len(StageColumn)),
              "audit_counts": extent + (len(AuditCount),),
              "audit_metrics": extent + (len(AuditMetric),), "force_residual_work": extent}
    blocks = []
    next_step = 1
    for entry in manifest["chunks"]:
        path = root / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"evidence hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            block = {key: data[key].copy() for key in data.files
                     if states or not key.startswith("state_") and key != "controls"}
        if len(block["policy_step"]) != entry["steps"]:
            raise ValueError("chunk extent does not match its manifest")
        if entry["first_policy_step"] != next_step or not np.array_equal(
                block["policy_step"], np.arange(next_step, next_step + entry["steps"])):
            raise ValueError("physics evidence has a discontinuity")
        for name, shape in shapes.items():
            if block[name].shape != (entry["steps"],) + shape:
                raise ValueError(f"diagnostic extent mismatch: {name}")
        if blocks and block.keys() != blocks[0].keys():
            raise ValueError("diagnostic capture columns changed between chunks")
        next_step += entry["steps"]
        blocks.append(block)
    if not blocks:
        raise ValueError("no completed physics records")
    values = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    if not np.array_equal(values["policy_step"], np.arange(1, manifest["steps"] + 1)):
        raise ValueError("physics evidence has a discontinuity")
    return values, manifest
