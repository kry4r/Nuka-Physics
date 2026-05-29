#!/usr/bin/env python3
"""Preflight owner-provided v0.1 golden candidates outside protected storage.

This helper is intentionally read-only with respect to `tests/oracle/golden/**`.
It validates the candidate files, checks the local C++20 `<expected>` toolchain
contract, and can run the existing CUDA oracle tests against a candidate
directory via `NUKA_GOLDEN_DIR`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_DIR = Path("/tmp/nuka_owner_candidates")
DEFAULT_BUILD_DIR = ROOT / "build-cuda128"
GOLDEN_SPECS = (
    (
        "featherstone_go2_random_sample.bin",
        ("--kind", "random-qacc"),
        ("--samples", "1000"),
        ("--qpos-count", "13"),
        ("--qvel-count", "13"),
        ("--qacc-count", "13"),
        ("--model-name", "go2_mjx.xml"),
    ),
    (
        "featherstone_h1_random_sample.bin",
        ("--kind", "random-qacc"),
        ("--samples", "1000"),
        ("--qpos-count", "20"),
        ("--qvel-count", "20"),
        ("--qacc-count", "20"),
        ("--model-name", "h1.xml"),
    ),
    (
        "go2_stand_5s.bin",
        ("--kind", "joint-trajectory"),
        ("--samples", "1200"),
        ("--qpos-count", "13"),
        ("--qvel-count", "0"),
        ("--qacc-count", "0"),
        ("--model-name", "go2_stand.usda"),
    ),
)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"/root/miniconda3/bin:/root/.local/bin:{env.get('PATH', '')}"
    return env


def _flatten(parts: tuple[str, ...] | tuple[tuple[str, str], ...]) -> list[str]:
    flattened: list[str] = []
    for item in parts:
        flattened.extend(item)
    return flattened


def run(label: str, cmd: list[str], *, env: dict[str, str] | None = None) -> bool:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=env or _env())
    if result.returncode == 0:
        print(f"PASS: {label}", flush=True)
        return True
    print(f"FAIL: {label} exited {result.returncode}", flush=True)
    return False


def check_expected_header(cxx: str | None) -> bool:
    compiler = cxx or os.environ.get("CXX") or "/root/.nuka-toolchain-gcc14/bin/x86_64-conda-linux-gnu-g++"
    source = '#include <expected>\nint main(){std::expected<int,int> x=1;return *x==1?0:1;}\n'
    result = subprocess.run(
        [compiler, "-std=c++20", "-x", "c++", "-", "-c", "-o", os.devnull],
        input=source,
        text=True,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        print(f"PASS: {compiler} supports <expected> in -std=c++20")
        return True
    print(f"BLOCKED: {compiler} does not support <expected> in -std=c++20")
    if result.stderr:
        print(result.stderr.strip().splitlines()[-1])
    return False


def check_candidate_files(candidate_dir: Path) -> bool:
    ok = True
    for filename, *spec in GOLDEN_SPECS:
        path = candidate_dir / filename
        if not path.exists():
            print(f"FAIL: missing candidate {path}")
            ok = False
            continue
        cmd = [
            sys.executable,
            "tools/oracle/validate_golden_candidate.py",
            "--path",
            str(path),
            *_flatten(tuple(spec)),
        ]
        ok = run(f"validate {filename}", cmd) and ok
    return ok


def run_oracle_preflight(candidate_dir: Path, build_dir: Path) -> bool:
    env = _env()
    env["NUKA_GOLDEN_DIR"] = str(candidate_dir)
    return run(
        "candidate CUDA oracle preflight",
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-R",
            "FeatherstoneOracle|Go2Stand.OwnerGoldenTrajectoryMatchesWithinTolerance",
            "--output-on-failure",
            "-j1",
        ],
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR, type=Path)
    parser.add_argument("--build-dir", default=DEFAULT_BUILD_DIR, type=Path)
    parser.add_argument("--cxx", help="C++ compiler to use for the <expected> probe")
    parser.add_argument(
        "--skip-oracle-ctest",
        action="store_true",
        help="only validate candidate file shape/hash-level invariants",
    )
    args = parser.parse_args()

    candidate_dir = args.candidate_dir
    build_dir = args.build_dir if args.build_dir.is_absolute() else ROOT / args.build_dir

    checks = [
        ("candidate files", check_candidate_files(candidate_dir)),
        ("C++20 <expected>", check_expected_header(args.cxx)),
    ]
    if not args.skip_oracle_ctest:
        checks.append(("candidate CUDA oracle preflight", run_oracle_preflight(candidate_dir, build_dir)))

    failed = [name for name, ok in checks if not ok]
    print("\n== v0.1 owner preflight summary ==")
    if not failed:
        print(
            "PASS: owner candidate preflight is green; protected copy and "
            "strict gate are still required"
        )
        return 0
    for name in failed:
        print(f"BLOCKED: {name}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
