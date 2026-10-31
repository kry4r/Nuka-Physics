"""Measured contact violations remain failures when gap coverage is incomplete."""

import numpy as np

from nuka.diagnostics.report import contact_analysis, interval_solver_analysis
from nuka.diagnostics.schema import AuditCount, AuditMetric, DiagnosticThresholds, Stage, StageColumn


def test_contact_failure_survives_missing_gap_coverage():
    counts = np.zeros((1, len(AuditCount)), dtype=np.uint32)
    metrics = np.zeros((1, len(AuditMetric)), dtype=np.uint64)
    counts[0, AuditCount.CONTACTS] = 1
    counts[0, AuditCount.UNMEASURED_GAPS] = 1
    limits = DiagnosticThresholds()
    incomplete = contact_analysis(counts, metrics, limits, limits.velocity_tolerance_mps)
    assert incomplete["status"] == "unmeasured"
    assert not incomplete["coverage_complete"]
    counts[0, AuditCount.CONE_VIOLATIONS] = 1
    failed = contact_analysis(counts, metrics, limits, limits.velocity_tolerance_mps)
    assert failed["status"] == "failed"
    assert not failed["coverage_complete"]
    assert failed["known_violations"] == {"CONE_VIOLATIONS": 1}
    untrusted = contact_analysis(counts, metrics, limits, limits.velocity_tolerance_mps, readout_complete=False)
    assert untrusted["status"] == "unmeasured"
    assert not untrusted["known_violations"]


def test_earlier_physical_interval_residual_is_not_replaced_by_latest_solve():
    stages = np.zeros((2, 3, len(Stage), len(StageColumn)), dtype=np.float32)
    stages[..., Stage.SOLVED, StageColumn.SOLVER_RESIDUAL_MEASURED] = 1
    stages[0, 0, Stage.SOLVED, StageColumn.SOLVER_PRIMAL_RESIDUAL] = 0.1
    report = interval_solver_analysis(stages, 1e-4)
    assert report["status"] == "failed"
    assert report["coverage_complete"]
    assert report["measured_interval_records"] == 6
    assert report["budget_violations"] == 1
    stages[0, 0, Stage.SOLVED, StageColumn.SOLVER_PRIMAL_RESIDUAL] = 0
    assert interval_solver_analysis(stages, 1e-4)["status"] == "passed"


def test_missing_interval_residuals_remain_unmeasured():
    stages = np.zeros((2, 3, len(Stage), len(StageColumn)), dtype=np.float32)
    report = interval_solver_analysis(stages, 1e-4)
    assert report["status"] == "unmeasured"
    assert not report["coverage_complete"]
    assert report["measured_interval_records"] == 0
