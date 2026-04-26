#!/usr/bin/env python3
"""
compare_reference.py - Compare simulation output against reference data.

Reads two CSV files (test output and reference) and reports the maximum
absolute error per numeric column.

Usage:
    python compare_reference.py --input result.csv --reference ref.csv
    python compare_reference.py --input result.csv --reference ref.csv --tolerance 1e-4
"""

import argparse
import csv
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare numeric columns between test output and reference data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the test output CSV file.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the reference CSV file.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Acceptable absolute error threshold (default: 1e-6).",
    )
    return parser.parse_args()


def read_csv(path):
    """Read a CSV file and return (headers, rows) where rows is a list of dicts."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows


def is_numeric(value):
    """Check if a string can be parsed as a float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def compare(input_path, reference_path, tolerance):
    """Compare two CSV files column-by-column and return per-column max errors."""
    in_headers, in_rows = read_csv(input_path)
    ref_headers, ref_rows = read_csv(reference_path)

    # Validate matching structure
    common_columns = [h for h in in_headers if h in ref_headers]
    if not common_columns:
        print("ERROR: No common columns found between input and reference.")
        return False

    if len(in_rows) != len(ref_rows):
        print(
            f"WARNING: Row count mismatch: input has {len(in_rows)}, "
            f"reference has {len(ref_rows)}. Comparing min({len(in_rows)}, {len(ref_rows)}) rows."
        )

    num_rows = min(len(in_rows), len(ref_rows))
    max_errors = {}
    passed = True

    for col in common_columns:
        col_max_error = 0.0
        col_is_numeric = False

        for i in range(num_rows):
            in_val = in_rows[i].get(col, "")
            ref_val = ref_rows[i].get(col, "")

            if is_numeric(in_val) and is_numeric(ref_val):
                col_is_numeric = True
                error = abs(float(in_val) - float(ref_val))
                col_max_error = max(col_max_error, error)

        if col_is_numeric:
            max_errors[col] = col_max_error
            status = "PASS" if col_max_error <= tolerance else "FAIL"
            if status == "FAIL":
                passed = False
            print(f"  {col:30s}  max_error={col_max_error:.8e}  [{status}]")

    return passed


def main():
    args = parse_args()

    print(f"Comparing: {args.input}")
    print(f"Reference: {args.reference}")
    print(f"Tolerance: {args.tolerance}")
    print("-" * 60)

    passed = compare(args.input, args.reference, args.tolerance)

    print("-" * 60)
    if passed:
        print("Result: ALL COLUMNS PASSED")
        sys.exit(0)
    else:
        print("Result: SOME COLUMNS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
