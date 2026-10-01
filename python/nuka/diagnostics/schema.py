"""Backend independent units and acceptance contracts for physics evidence."""

from dataclasses import asdict, dataclass
from enum import IntEnum


class Stage(IntEnum):
    BEGIN = 0
    FREE = 1
    SOLVED = 2
    PROJECTED = 3
    END = 4


class StageColumn(IntEnum):
    MASS = 0
    LINEAR_MOMENTUM_X = 1
    LINEAR_MOMENTUM_Y = 2
    LINEAR_MOMENTUM_Z = 3
    ANGULAR_MOMENTUM_X = 4
    ANGULAR_MOMENTUM_Y = 5
    ANGULAR_MOMENTUM_Z = 6
    VBD_MOMENTUM_DEFECT_X = 7
    VBD_MOMENTUM_DEFECT_Y = 8
    VBD_MOMENTUM_DEFECT_Z = 9
    VBD_ANGULAR_DEFECT_X = 10
    VBD_ANGULAR_DEFECT_Y = 11
    VBD_ANGULAR_DEFECT_Z = 12
    VBD_MOMENTUM_VELOCITY_ERROR = 13
    VBD_FORCE = 14
    NONFINITE_PARTICLES = 15
    NONFINITE_BODIES = 16
    NONFINITE_LINKS = 17
    VBD_MIN_EFFECTIVE_DT = 18
    VBD_MAX_EFFECTIVE_DT = 19
    VBD_DYNAMIC_PARTICLES = 20
    DT = 21
    ACTIVE_ROWS = 22
    ENERGY_INJECTING_PASSIVE_ROWS = 23
    MAX_PASSIVE_ROW_ENERGY_GAIN = 24
    MAX_VERTEX_RESIDUAL_WORK = 25
    VBD_DISCRETE_MOMENTUM_X = 26
    VBD_DISCRETE_MOMENTUM_Y = 27
    VBD_DISCRETE_MOMENTUM_Z = 28
    VBD_GRAVITY_IMPULSE_X = 29
    VBD_GRAVITY_IMPULSE_Y = 30
    VBD_GRAVITY_IMPULSE_Z = 31
    VBD_ELASTIC_BOUNDARY_IMPULSE_X = 32
    VBD_ELASTIC_BOUNDARY_IMPULSE_Y = 33
    VBD_ELASTIC_BOUNDARY_IMPULSE_Z = 34
    VBD_ROW_IMPULSE_X = 35
    VBD_ROW_IMPULSE_Y = 36
    VBD_ROW_IMPULSE_Z = 37
    VBD_BDF_PARTICLES = 38


class AuditCount(IntEnum):
    CONTACTS = 0
    ROWS = 1
    INVALID_ROWS = 2
    LINEAR_CLOSURE_VIOLATIONS = 3
    CONE_VIOLATIONS = 4
    NEGATIVE_NORMAL_IMPULSES = 5
    UNMEASURED_GAPS = 6
    GAP_VIOLATIONS = 7


class AuditMetric(IntEnum):
    LINEAR_CLOSURE_RELATIVE = 0
    CONE_IMPULSE_EXCESS = 1
    NEGATIVE_NORMAL_IMPULSE = 2
    FROZEN_GAP_PENETRATION = 3
    POSITIVE_NORMAL_WORK = 4
    POSITIVE_TANGENT_WORK = 5
    GAP_PENETRATION = 6


@dataclass(frozen=True)
class DiagnosticThresholds:
    energy_net_ratio: float = 0.01
    energy_positive_window_ratio: float = 0.001
    window_seconds: float = 1.0
    dat_step_ratio: float = 0.01
    dat_energy_ratio: float = 0.001
    energy_spike_j: float = 0.01
    momentum_relative_per_second: float = 1e-5
    linear_impulse_relative: float = 1e-6
    friction_cone_relative_slack: float = 1e-3
    negative_loss_tolerance_j: float = 1e-9
    velocity_tolerance_mps: float = 3e-4
    position_slop_m: float = 0.001

    def to_dict(self):
        return asdict(self)


def stage_units(column):
    c = StageColumn(column)
    if c == StageColumn.MASS:
        return "kg"
    if 1 <= c <= 3 or 7 <= c <= 9 or 26 <= c <= 37:
        return "kg m/s"
    if 4 <= c <= 6 or 10 <= c <= 12:
        return "kg m^2/s"
    if c == StageColumn.VBD_MOMENTUM_VELOCITY_ERROR:
        return "m/s"
    if c == StageColumn.VBD_FORCE:
        return "N"
    if c in (StageColumn.VBD_MIN_EFFECTIVE_DT, StageColumn.VBD_MAX_EFFECTIVE_DT, StageColumn.DT):
        return "s"
    if c in (StageColumn.MAX_PASSIVE_ROW_ENERGY_GAIN, StageColumn.MAX_VERTEX_RESIDUAL_WORK):
        return "J"
    return "count"
