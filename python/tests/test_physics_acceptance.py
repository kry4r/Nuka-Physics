"""Measured contact violations remain failures when gap coverage is incomplete."""

import numpy as np

from nuka.diagnostics.report import contact_analysis
from nuka.diagnostics.schema import AuditCount, AuditMetric, DiagnosticThresholds


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
