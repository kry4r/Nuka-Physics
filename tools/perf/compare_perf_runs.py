#!/usr/bin/env python3
"""
compare_perf_runs.py - Diff two perf baseline JSON files and flag regressions.

Reads two baseline files produced by run_perf_baseline.py, matches their
`sweep` entries by `n_envs`, and for each matched env count compares the
chosen per-tag metric (default p50_us). A tag is flagged as a regression when
B exceeds A by more than --regress-pct percent.

Tags or env counts present in only one of the two runs are reported but never
counted as regressions.

NOTE (M9 T14): this script only diffs baseline JSON files; it produces no
regression baselines itself. Those baselines come from run_perf_baseline.py,
whose perf-harness CLI was deleted in M9 T11 (see that file's docstring). Until
the M11 perf-CLI rework reintroduces a --perf-json-capable harness, no fresh
baselines can be captured; this comparator stays valid for any baseline JSON
that already honours the documented per_tag schema.

Usage:
    python3 tools/perf/compare_perf_runs.py baseline_a.json baseline_b.json
    python3 tools/perf/compare_perf_runs.py a.json b.json --metric p99_us \\
        --regress-pct 5.0 --fail-on-regress
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METRIC_CHOICES = ["count", "mean_us", "p50_us", "p99_us", "min_us", "max_us"]
MISSING = "—"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Diff two perf baseline JSON files and flag per-kernel regressions."
    )
    parser.add_argument("baseline_a", type=Path, help="Reference baseline JSON (A).")
    parser.add_argument("baseline_b", type=Path, help="Candidate baseline JSON (B).")
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default="p50_us",
        help="Per-tag metric to compare (default: p50_us).",
    )
    parser.add_argument(
        "--regress-pct",
        type=float,
        default=5.0,
        help="Percent increase (B over A) that counts as a regression (default: 5.0).",
    )
    parser.add_argument(
        "--fail-on-regress",
        action="store_true",
        help="Exit with code 1 if any regression is detected.",
    )
    return parser.parse_args(argv)


def load_baseline(path):
    """Load a baseline JSON file, returning the parsed dict."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"baseline not found: {path}")
    except json.JSONDecodeError as error:
        raise SystemExit(f"baseline is not valid JSON ({path}): {error}")
    if "sweep" not in data or not isinstance(data["sweep"], list):
        raise SystemExit(f"baseline {path} is missing a 'sweep' list")
    return data


def index_sweep_by_envs(data):
    """Return {n_envs: per_tag dict} for a loaded baseline."""
    index = {}
    for entry in data["sweep"]:
        n = entry.get("n_envs")
        if n is None:
            continue
        index[n] = entry.get("per_tag", {})
    return index


def metric_value(per_tag, tag, metric):
    """Return the metric value for a tag, or None if tag/metric is absent."""
    tag_stats = per_tag.get(tag)
    if not isinstance(tag_stats, dict):
        return None
    value = tag_stats.get(metric)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def fmt(value):
    """Format a metric value (or the missing marker) for table display."""
    if value is None:
        return MISSING
    return f"{value:.3f}"


def compare(data_a, data_b, metric, regress_pct):
    """Compare two baselines; print tables and return True if any regression."""
    index_a = index_sweep_by_envs(data_a)
    index_b = index_sweep_by_envs(data_b)

    all_envs = sorted(set(index_a) | set(index_b))
    any_regression = False

    print(f"Comparing metric '{metric}' (regression threshold: +{regress_pct:.2f}%)")
    print(f"  A: {data_a.get('gpu_label', '?')}  ({data_a.get('timestamp_utc', '?')})")
    print(f"  B: {data_b.get('gpu_label', '?')}  ({data_b.get('timestamp_utc', '?')})")

    for n in all_envs:
        print("")
        if n not in index_a:
            print(f"[N={n}] present only in B — skipped (no A to compare against)")
            continue
        if n not in index_b:
            print(f"[N={n}] present only in A — skipped (no B to compare against)")
            continue

        per_tag_a = index_a[n]
        per_tag_b = index_b[n]
        tags = sorted(set(per_tag_a) | set(per_tag_b))

        print(f"[N={n}]")
        print(f"  {'tag':<20} {'A':>12} {'B':>12} {'delta':>12} {'pct':>9}  flag")
        print(f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 9}  ----")
        for tag in tags:
            a_val = metric_value(per_tag_a, tag, metric)
            b_val = metric_value(per_tag_b, tag, metric)

            if a_val is None or b_val is None:
                # Tag (or metric) missing on one side: report, never flag.
                note = "B-only" if a_val is None else "A-only"
                print(
                    f"  {tag:<20} {fmt(a_val):>12} {fmt(b_val):>12} "
                    f"{MISSING:>12} {MISSING:>9}  {note}"
                )
                continue

            delta = b_val - a_val
            if a_val == 0.0:
                # Avoid divide-by-zero: any positive B from a zero A is "inf%".
                pct_str = "inf" if delta > 0 else "0.00"
                is_regression = delta > 0
            else:
                pct = delta / a_val * 100.0
                pct_str = f"{pct:+.2f}"
                is_regression = pct > regress_pct

            flag = "REGRESS" if is_regression else ""
            if is_regression:
                any_regression = True
            print(
                f"  {tag:<20} {a_val:>12.3f} {b_val:>12.3f} "
                f"{delta:>+12.3f} {pct_str:>9}  {flag}"
            )

    print("")
    if any_regression:
        print("Result: REGRESSION(S) DETECTED")
    else:
        print("Result: no regressions")
    return any_regression


def main(argv=None):
    args = parse_args(argv)
    data_a = load_baseline(args.baseline_a)
    data_b = load_baseline(args.baseline_b)
    any_regression = compare(data_a, data_b, args.metric, args.regress_pct)
    if any_regression and args.fail_on_regress:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
