#!/usr/bin/env python3
"""
run_perf_baseline.py - Drive a perf sweep and write a baseline JSON.

Runs an engine perf-harness over a sweep of environment counts
(N in {64, 256, 1024, 4096} by default) and assembles a single baseline
JSON file. The harness is invoked per N with the documented contract:

    <harness> --scene <usda> --envs <N> --steps <M> --determinism <d1|d2> --perf-json -

and is expected to print to stdout a JSON object exactly of the form
produced by the engine's PerfRecorder.DumpJson():

    { "schema_version": 1,
      "per_tag": { "<tag>": { "count": ..., "mean_us": ..., "p50_us": ...,
                              "p99_us": ..., "min_us": ..., "max_us": ... } } }

The per_tag schema is the contract between the in-engine recorder and this
tooling; it is kept byte-compatible.

NOTE (M9 T14) -- THE OLD perf_harness CLI WAS DELETED.
    The previous `--harness build/perf_harness ... --perf-json -` binary
    (tools/perf/perf_harness.cpp) was removed in M9 T11 with the legacy
    batched-articulated world it constructed. There is currently NO shipping
    CLI that emits the --perf-json contract above; the live perf gates are now
    the gtest perf binaries (the nuka_perf_test category exe -- nk_union_n1
    <=5ms, the grasp throughput >=11k env-steps/s gate, go2 4096-env step-time),
    which assert thresholds in-process and do NOT emit per_tag JSON. A
    --perf-json-capable CLI is deferred to the M11 perf-CLI rework. Until then
    these scripts are dormant baseline-capture tooling: --dry-run still prints
    the planned sweep, but a real capture requires the caller to pass the path
    of a (future) harness that honours the --perf-json contract above. There is
    intentionally NO default --harness path -- the deleted build/perf_harness is
    not referenced anywhere.

Usage:
    # Plan the sweep + print exact harness commands without running anything:
    python3 tools/perf/run_perf_baseline.py --dry-run

    # Capture a real baseline (requires a harness honouring the --perf-json
    # contract above; the M11 perf-CLI, once it exists):
    python3 tools/perf/run_perf_baseline.py \\
        --harness <path-to-perf-json-harness> \\
        --gpu-label "RTX 4000 Ada (dev)" \\
        --out out/perf/baseline_2026-05-29.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fixed device identity for the dev/optimization box. The absolute master-plan
# §7 gate (< 1 ms/env-step @ 4096 on an RTX 4090) is not reachable here; this
# baseline records RTX-4000-Ada-relative numbers (see docs/architecture/perf-budget.md).
DEFAULT_GPU = "NVIDIA RTX 4000 Ada Generation"
DEFAULT_GPU_LABEL = "RTX 4000 Ada (dev/relative reference, not 4090/5080)"

DEFAULT_SCENE = "examples/scenes/go2_stand.usda"
DEFAULT_ENVS = "64,256,1024,4096"
DEFAULT_STEPS = 5000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Drive a perf sweep over multiple env counts and write a baseline JSON."
        )
    )
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE,
        help=f"Scene USDA passed to the harness (default: {DEFAULT_SCENE}).",
    )
    parser.add_argument(
        "--envs",
        default=DEFAULT_ENVS,
        help=(
            "Comma-separated env counts to sweep, in order "
            f"(default: {DEFAULT_ENVS})."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Steps per env for each run (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--determinism",
        choices=["d1", "d2"],
        default="d1",
        help="Determinism tier passed to the harness (default: d1).",
    )
    parser.add_argument(
        "--harness",
        default=None,
        help=(
            "Path to the engine perf-harness CLI. Optional for --dry-run "
            "(a placeholder is shown); required for a real capture."
        ),
    )
    parser.add_argument(
        "--gpu-label",
        default=DEFAULT_GPU_LABEL,
        help="Free-text GPU label recorded in the baseline (default: dev reference).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output baseline JSON path "
            "(default: out/perf/baseline_<date>.json)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned sweep and the exact harness commands that WOULD "
            "run, then exit 0 without invoking anything."
        ),
    )
    return parser.parse_args(argv)


def parse_envs(envs_arg):
    """Parse a comma-separated env-count string into an ordered list of ints."""
    values = []
    for token in envs_arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            n = int(token)
        except ValueError:
            raise SystemExit(f"invalid --envs entry: {token!r} is not an integer")
        if n <= 0:
            raise SystemExit(f"invalid --envs entry: {n} must be positive")
        values.append(n)
    if not values:
        raise SystemExit("--envs produced no valid entries")
    return values


def build_command(harness, scene, n_envs, steps, determinism):
    """Build the documented harness argv for a single sweep point.

    Uses the placeholder token "<harness>" when no harness path is provided,
    so the command line is still inspectable in --dry-run mode.
    """
    return [
        harness if harness is not None else "<harness>",
        "--scene",
        scene,
        "--envs",
        str(n_envs),
        "--steps",
        str(steps),
        "--determinism",
        determinism,
        "--perf-json",
        "-",
    ]


def run_harness(command):
    """Invoke the harness and parse its stdout JSON.

    Returns the parsed dict (expected: schema_version + per_tag). Raises
    SystemExit on non-zero exit or unparseable output.
    """
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise SystemExit(f"harness not found: {command[0]!r}")
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"harness exited {error.returncode} for command: {' '.join(command)}\n"
            f"stderr:\n{error.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"harness stdout was not valid JSON for command: {' '.join(command)}\n"
            f"error: {error}"
        )
    if "per_tag" not in payload:
        raise SystemExit(
            f"harness JSON is missing 'per_tag' for command: {' '.join(command)}"
        )
    return payload


def main(argv=None):
    args = parse_args(argv)

    env_counts = parse_envs(args.envs)
    now = datetime.now(timezone.utc)
    timestamp_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str = now.strftime("%Y-%m-%d")
    out_path = Path(args.out) if args.out else Path(f"out/perf/baseline_{date_str}.json")

    commands = [
        build_command(args.harness, args.scene, n, args.steps, args.determinism)
        for n in env_counts
    ]

    if args.dry_run:
        print("Planned perf sweep (dry-run; nothing executed):")
        print(f"  scene        : {args.scene}")
        print(f"  envs         : {', '.join(str(n) for n in env_counts)}")
        print(f"  steps        : {args.steps}")
        print(f"  determinism  : {args.determinism}")
        print(f"  gpu          : {DEFAULT_GPU}")
        print(f"  gpu_label    : {args.gpu_label}")
        print(f"  out          : {out_path}")
        print(f"  harness      : {args.harness if args.harness else '<harness> (unset)'}")
        print("")
        print("Harness commands that WOULD run (one per env count):")
        for n, command in zip(env_counts, commands):
            print(f"  [N={n}] {' '.join(command)}")
        return 0

    if args.harness is None:
        raise SystemExit(
            "--harness is required for a real capture (only optional with --dry-run)"
        )

    sweep = []
    device_gpu = None
    for n, command in zip(env_counts, commands):
        print(f"[N={n}] running: {' '.join(command)}", file=sys.stderr)
        payload = run_harness(command)
        # Prefer the device-reported GPU name (read by the harness via
        # cudaGetDeviceProperties) over the fixed DEFAULT_GPU constant, so the
        # baseline records the hardware that was actually measured. Take it from
        # the first payload that carries it.
        if device_gpu is None and isinstance(payload.get("gpu"), str):
            device_gpu = payload["gpu"]
        entry = {
            "n_envs": n,
            "steps": args.steps,
            "per_tag": payload["per_tag"],
        }
        # Self-document the warm-up provenance when the harness reports it (the
        # harness default is max(100, steps/20), so this is not simply 0).
        if isinstance(payload.get("warmup"), int):
            entry["warmup"] = payload["warmup"]
        sweep.append(entry)

    baseline = {
        "schema_version": 1,
        # Device-reported name when the harness supplied it; otherwise the
        # fixed dev-box default.
        "gpu": device_gpu if device_gpu is not None else DEFAULT_GPU,
        "gpu_label": args.gpu_label,
        "timestamp_utc": timestamp_utc,
        "scene": args.scene,
        "determinism": args.determinism,
        "sweep": sweep,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
        f.write("\n")
    print(f"wrote baseline: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
