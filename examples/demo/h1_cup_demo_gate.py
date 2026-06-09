#!/usr/bin/env python
"""Run the H1 cup-demo presentability gates.

This is a headless demo gate, not a renderer: it verifies the two pieces the demo
needs to be credible today:
  1. the exported H1 standing policy crosses the Python->C++ bridge and stands in
     the full co-resident world with 4/4 foot contacts and torque limits;
  2. the H1 hand/cup dense-grasp force-closure gate holds under disturbance, bites
     when grip is off, and is deterministic.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
BUILD_DIR = REPO / "build-cuda128"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"/root/.nuka-toolchain-gcc14/bin:{env.get('PATH', '')}"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    ld = env.get("LD_LIBRARY_PATH", "")
    cuda_lib = "/opt/cuda-12.8-root/lib64"
    env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{ld}" if ld else cuda_lib
    return env


def _run(cmd: list[str], *, env: dict[str, str], allow_skip: bool = False) -> str:
    print("[h1_demo] $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, end="", flush=True)
    if not allow_skip and ("[  SKIPPED ]" in proc.stdout or "YOU HAVE 0 DISABLED TESTS" in proc.stdout):
        raise RuntimeError("demo gate encountered a skipped gtest; assets/artifacts are not presentable")
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run H1 standing + cup grasp demo gates.")
    parser.add_argument("--skip-build", action="store_true", help="Assume test binaries are already built.")
    args = parser.parse_args(argv)

    env = _env()
    if not args.skip_build:
        _run([
            "cmake", "--build", str(BUILD_DIR), "--target",
            "nuka_h1_bridge_spike_test", "nuka_h1_dense_grasp_test",
            f"-j{os.cpu_count() or 1}",
        ], env=env, allow_skip=True)

    _run([
        str(BUILD_DIR / "tests/nuka_h1_bridge_spike_test"),
        "--gtest_filter=H1BridgeSpike.Gate1BridgeParity:H1BridgeSpike.Gate2ClosedLoopStandSoft",
    ], env=env)
    _run([
        str(BUILD_DIR / "tests/nuka_h1_dense_grasp_test"),
        "--gtest_filter=H1DenseGraspLargeCup.ForceClosureLiftWithDisturbance:"
        "H1DenseGraspLargeCup.ForceClosureLiftWithDisturbanceDeterministicTwoRun:"
        "H1DenseGraspLargeCup.FingerOnlyFallbackBiteGripOffVsOn",
    ], env=env)
    print("[h1_demo] PASS: H1 stands through the bridge and holds the cup under disturbance.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
