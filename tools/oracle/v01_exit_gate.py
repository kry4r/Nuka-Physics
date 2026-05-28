#!/usr/bin/env python3
"""Run the local v0.1/p07 exit-gate checks and report remaining blockers.

This script is intentionally strict: missing owner-provided golden files,
missing MJX dependencies, or a compiler without <expected> make the gate fail.
It does not create or modify protected golden files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CUDA_ROOT = Path("/opt/cuda-12.8-root/usr/local/cuda-12.8")
REQUIRED_GOLDENS = (
    ROOT / "tests/oracle/golden/featherstone_go2_random_sample.bin",
    ROOT / "tests/oracle/golden/featherstone_h1_random_sample.bin",
    ROOT / "tests/oracle/golden/go2_stand_5s.bin",
)
REQUIRED_ASSETS = (
    ROOT / ".nuka-assets/mujoco_menagerie/unitree_go2/go2_mjx.xml",
    ROOT / ".nuka-assets/mujoco_menagerie/unitree_h1/h1.xml",
)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"/root/miniconda3/bin:/root/.local/bin:{env.get('PATH', '')}"
    env.setdefault("CC", "/usr/bin/gcc-10")
    env.setdefault("CXX", "/usr/bin/g++-10")
    return env


def run(label: str, cmd: list[str], *, cwd: Path = ROOT) -> bool:
    print(f"\n== {label} ==")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env=_env())
    if result.returncode == 0:
        print(f"PASS: {label}")
        return True
    print(f"FAIL: {label} exited {result.returncode}")
    return False


def check_file(path: Path) -> bool:
    if path.exists() and path.is_file():
        print(f"PASS: found {path.relative_to(ROOT)}")
        return True
    print(f"FAIL: missing {path.relative_to(ROOT)}")
    return False


def check_expected_header() -> bool:
    env = _env()
    compiler = env.get("CXX") or shutil.which("g++-10", path=env.get("PATH")) or shutil.which(
        "g++", path=env.get("PATH")
    )
    if compiler is None:
        print("FAIL: no C++ compiler found for <expected> probe")
        return False
    source = '#include <expected>\nint main(){std::expected<int,int> x=1;return *x==1?0:1;}\n'
    result = subprocess.run(
        [compiler, "-std=c++20", "-x", "c++", "-", "-c", "-o", os.devnull],
        input=source,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        print(f"PASS: {compiler} supports <expected>")
        return True
    print(f"FAIL: {compiler} does not support <expected>")
    if result.stderr:
        print(result.stderr.strip().splitlines()[-1])
    return False


def check_python_deps(python: str) -> bool:
    code = (
        "import jax\n"
        "import mujoco\n"
        "from mujoco import mjx\n"
        "print('MJX deps available')\n"
    )
    result = subprocess.run(
        [python, "-c", code],
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        print("PASS: jax, mujoco, and mujoco.mjx are importable")
        return True
    print("FAIL: jax, mujoco, or mujoco.mjx is not importable")
    if result.stderr:
        print(result.stderr.strip().splitlines()[-1])
    return False


def check_git_lfs() -> bool:
    result = subprocess.run(
        ["git", "lfs", "version"],
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        print(f"PASS: {result.stdout.strip()}")
        return True
    print("FAIL: git-lfs is not available")
    if result.stderr:
        print(result.stderr.strip().splitlines()[-1])
    return False


def run_engine_trajectory_diff(build_dir: Path) -> bool:
    demo = build_dir / "src/nuka_go2_stand_demo"
    golden = ROOT / "tests/oracle/golden/go2_stand_5s.bin"
    if not golden.exists():
        print("FAIL: cannot diff Go2 trajectory without tests/oracle/golden/go2_stand_5s.bin")
        return False
    if not demo.exists():
        print(f"FAIL: missing demo binary {demo}")
        return False

    with tempfile.TemporaryDirectory(prefix="nuka_v01_exit_") as tmp:
        out_dir = Path(tmp)
        actual = out_dir / "go2_engine.bin"
        ppm = out_dir / "go2.ppm"
        env = _env()
        env["LD_LIBRARY_PATH"] = (
            f"{build_dir / 'src'}:{CUDA_ROOT / 'lib64'}:"
            f"{env.get('LD_LIBRARY_PATH', '')}"
        )
        print("\n== generate Go2 engine trajectory ==")
        result = subprocess.run(
            [
                str(demo),
                str(ROOT / "examples/scenes/go2_stand.usda"),
                str(ppm),
                str(actual),
            ],
            cwd=ROOT,
            env=env,
        )
        if result.returncode != 0:
            print(f"FAIL: Go2 demo trajectory export exited {result.returncode}")
            return False
        return run(
            "diff Go2 engine trajectory against MJX golden",
            [
                sys.executable,
                "tools/oracle/diff_trajectory.py",
                "--actual",
                str(actual),
                "--golden",
                str(golden),
                "--tolerance",
                "1e-4",
            ],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build-cuda128", type=Path)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter with jax, mujoco, and mujoco.mjx installed",
    )
    args = parser.parse_args()
    build_dir = args.build_dir if args.build_dir.is_absolute() else ROOT / args.build_dir

    checks: list[tuple[str, bool]] = []
    checks.append(("C++20 <expected>", check_expected_header()))
    checks.append(("MJX Python deps", check_python_deps(args.python)))
    checks.append(("Git LFS", check_git_lfs()))
    for asset in REQUIRED_ASSETS:
        checks.append((str(asset.relative_to(ROOT)), check_file(asset)))
    for golden in REQUIRED_GOLDENS:
        checks.append((str(golden.relative_to(ROOT)), check_file(golden)))

    checks.append(("build", run("build", ["cmake", "--build", str(build_dir), "--", "-j2"])))
    checks.append((
        "ctest p07",
        run(
            "ctest p07",
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "-R",
                "CAbi|CppWrapper|FeatherstoneOracle|Go2Stand",
                "--output-on-failure",
                "-j1",
            ],
        ),
    ))
    checks.append(("physics smell", run("physics smell", [sys.executable, "tools/lint/physics_smell.py", "--time"])))
    checks.append(("Go2 MJX trajectory diff", run_engine_trajectory_diff(build_dir)))

    failed = [name for name, ok in checks if not ok]
    print("\n== v0.1 exit gate summary ==")
    if not failed:
        print("PASS: v0.1 p07 local exit gate is green")
        return 0
    for name in failed:
        print(f"BLOCKED: {name}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
