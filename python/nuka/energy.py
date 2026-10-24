"""Demand-driven energy records from the production physics intervals."""

from enum import IntEnum, IntFlag

import numpy as np

from ._nuka_ext import Field


class EnergyColumn(IntEnum):
    BEGIN_KINETIC = 0
    BEGIN_GRAVITY = 1
    BEGIN_ELASTIC = 2
    FREE_KINETIC = 3
    SOLVED_KINETIC = 4
    BEFORE_DAT_KINETIC = 5
    BEFORE_DAT_GRAVITY = 6
    BEFORE_DAT_ELASTIC = 7
    END_KINETIC = 8
    END_GRAVITY = 9
    END_ELASTIC = 10
    DRIVE_WORK = 11
    AERO_WORK = 12
    EXTERNAL_WORK = 13
    KINEMATIC_ELASTIC_WORK = 14
    KINEMATIC_CONTACT_WORK = 15
    FRICTION_LOSS = 16
    NORMAL_LOSS = 17
    LIMIT_LOSS = 18
    PASSIVE_LOSS = 19
    RAYLEIGH_LOSS = 20
    POSITION_POTENTIAL = 21
    DAT_KINETIC_LOSS = 22
    DAT_POTENTIAL = 23
    RESIDUAL = 24
    THROUGHPUT = 25
    DT = 26
    ROW_RATE_WORK = 27
    ROW_PHYSICAL_IMPULSE_WORK = 28
    UNCLASSIFIED_ROW_WORK = 29
    VALID = 30
    PLASTIC_LOSS = 31
    VISCOUS_LOSS = 32
    COHESIVE_LOSS = 33
    NUMERICAL_KINETIC = 34
    NUMERICAL_POTENTIAL = 35


class EnergyStatus(IntFlag):
    NONFINITE = 1 << 0
    UNCLASSIFIED_ROWS = 1 << 1
    OTHER_PARTICLE_MATERIAL = 1 << 2
    GRID_MATERIAL = 1 << 3
    EXTERNAL_BODY_LOAD = 1 << 4
    FLOATING_POSITION_WORK = 1 << 5
    RIGID_ANGULAR_POSITION_WORK = 1 << 6
    ARMATURE_ENERGY = 1 << 7
    PHYSICS_FAILURE = 1 << 8


class EnergyLedgerSampler:
    """Request before stepping; copy all substep records once per policy step."""

    def __init__(self, world, env_count=1):
        self.world = world
        self.env_count = int(env_count)
        if self.env_count <= 0:
            raise ValueError("env_count must be positive")
        values = np.asarray(world.download_field(Field.ENERGY_LEDGER))
        if values.size % (self.env_count * len(EnergyColumn)):
            raise ValueError("energy record extent does not match env_count")
        self.substeps = values.size // (self.env_count * len(EnergyColumn))
        world.download_field(Field.ENERGY_LEDGER_STATUS)

    def sample(self):
        values = np.asarray(self.world.download_field(Field.ENERGY_LEDGER), dtype=np.float32)
        status = np.asarray(self.world.download_field(Field.ENERGY_LEDGER_STATUS), dtype=np.uint32)
        return (values.reshape(self.env_count, self.substeps, len(EnergyColumn)).copy(),
                status.reshape(self.env_count, self.substeps).copy())


def window_ratios(residual, throughput, dt, seconds):
    """Positive residual over throughput in each window spanning `seconds`; a shorter stream is one window."""
    elapsed = np.cumsum(dt)
    positive = np.r_[0.0, np.cumsum(np.maximum(residual, 0))]
    flow = np.r_[0.0, np.cumsum(throughput)]
    start = np.searchsorted(elapsed, elapsed - seconds, side="right")
    loss, work = positive[1:] - positive[start], flow[1:] - flow[start]
    ratio = np.divide(loss, work, out=np.full_like(work, np.inf), where=work > 0)
    ratio[(loss == 0) & (work == 0)] = 0
    return ratio[elapsed >= min(seconds, elapsed[-1]) - 0.5 * dt]


def _summarize_stream(values, status):
    """Report original energy gates; uncovered records cannot pass acceptance."""
    records = np.asarray(values, dtype=np.float64).reshape(-1, len(EnergyColumn))
    flags = np.asarray(status, dtype=np.uint32).reshape(-1)
    if len(records) != len(flags) or not len(records):
        raise ValueError("energy records and coverage flags must have matching nonzero extents")
    c = EnergyColumn
    covered = (flags == 0) & (records[:, c.VALID] == 1)
    covered &= np.isfinite(records).all(axis=1)
    report = {
        "records": len(records), "covered_records": int(covered.sum()),
        "coverage_flags": sorted(map(int, np.unique(flags))),
        "column_units": {column.name: ("s" if column == c.DT else "bool" if column == c.VALID else "J")
                         for column in c},
        "terms_j": {column.name: float(records[:, column].sum())
                    for column in c if column not in (c.DT, c.VALID, c.RESIDUAL, c.THROUGHPUT)},
        "passes_energy_gates": False,
    }
    if not covered.all():
        return report
    throughput = float(records[:, c.THROUGHPUT].sum())
    residual = records[:, c.RESIDUAL]
    ratios = window_ratios(residual, records[:, c.THROUGHPUT], records[:, c.DT], 1.0)
    absolute_residual = float(np.abs(residual).sum())
    net_residual = float(residual.sum())
    truncation = float(records[:, c.DAT_KINETIC_LOSS].sum())
    negative_losses = int((records[:, c.FRICTION_LOSS:c.RAYLEIGH_LOSS + 1] < -1e-9).sum())
    report.update({
        "throughput_j": throughput, "net_residual_j": net_residual,
        "absolute_residual_j": absolute_residual, "positive_residual_j": float(np.maximum(residual, 0).sum()),
        "max_positive_window_ratio": float(ratios.max()), "negative_loss_terms": negative_losses,
        "negative_dat_records": int((records[:, c.DAT_KINETIC_LOSS] < -1e-9).sum()),
        "residual_gate": abs(net_residual) <= 0.01 * throughput,
        "positive_window_gate": bool((ratios <= 0.001).all()),
        "dat_energy_gate": -1e-9 <= truncation <= 0.001 * throughput,
    })
    report["passes_energy_gates"] = bool(throughput >= 0 and report["residual_gate"] and
        report["positive_window_gate"] and report["dat_energy_gate"] and negative_losses == 0 and
        report["negative_dat_records"] == 0)
    return report


def summarize(values, status):
    """Keep each environment's time series separate for the one-second gate."""
    records = np.asarray(values)
    flags = np.asarray(status)
    if records.ndim == 4:
        reports = [_summarize_stream(records[:, env], flags[:, env])
                   for env in range(records.shape[1])]
        return {"environments": reports,
                "passes_energy_gates": all(report["passes_energy_gates"] for report in reports)}
    if records.ndim != 2:
        raise ValueError("expected intervals by columns, or policy steps by env by substeps by columns")
    return _summarize_stream(records, flags)
