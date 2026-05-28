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


def check_golden_lfs_attributes(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    result = subprocess.run(
        ["git", "check-attr", "filter", "diff", "merge", "text", "--", str(relative)],
        cwd=ROOT,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"FAIL: git check-attr failed for {relative}")
        if result.stderr:
            print(result.stderr.strip().splitlines()[-1])
        return False

    attrs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split(": ", 2)
        if len(parts) == 3:
            attrs[parts[1]] = parts[2]

    expected = {
        "filter": "lfs",
        "diff": "lfs",
        "merge": "lfs",
        "text": "unset",
    }
    mismatches = [
        f"{name}={attrs.get(name, 'missing')}"
        for name, value in expected.items()
        if attrs.get(name) != value
    ]
    if mismatches:
        print(
            f"FAIL: {relative} is not covered by Git LFS attributes "
            f"({', '.join(mismatches)})"
        )
        return False

    print(f"PASS: {relative} has Git LFS attributes")
    return True


def check_golden_lfs_pointer(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if not path.exists():
        print(f"SKIP: cannot verify Git LFS pointer for missing {relative}")
        return True

    result = subprocess.run(
        ["git", "show", f":{relative}"],
        cwd=ROOT,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(f"FAIL: {relative} is present but is not tracked in the Git index")
        if result.stderr:
            print(result.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1])
        return False

    blob = result.stdout
    required_markers = (
        b"version https://git-lfs.github.com/spec/v1\n",
        b"oid sha256:",
        b"\nsize ",
    )
    if all(marker in blob for marker in required_markers):
        print(f"PASS: {relative} is tracked as a Git LFS pointer")
        return True

    print(f"FAIL: {relative} is tracked as a regular Git blob, not a Git LFS pointer")
    return False


def check_find_package(build_dir: Path) -> bool:
    with tempfile.TemporaryDirectory(prefix="nuka_find_package_") as tmp:
        root = Path(tmp)
        install_dir = root / "install"
        sample_dir = root / "downstream"
        sample_build = sample_dir / "build"
        sample_dir.mkdir()
        (sample_dir / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(nuka_downstream_check LANGUAGES C)\n"
            "find_package(nuka REQUIRED)\n"
            "add_executable(check main.c)\n"
            "target_link_libraries(check PRIVATE nuka::nuka)\n"
        )
        (sample_dir / "main.c").write_text(
            "#include <nuka/nuka.h>\n"
            "int main(void) {\n"
            "    nuka_version_t v = nuka_get_version();\n"
            "    return (v.major == 0 && v.minor == 1) ? 0 : 1;\n"
            "}\n"
        )

        if not run(
            "install nuka package",
            [
                "cmake",
                "--install",
                str(build_dir),
                "--prefix",
                str(install_dir),
            ],
        ):
            return False
        if not run(
            "configure downstream find_package(nuka)",
            [
                "cmake",
                "-S",
                str(sample_dir),
                "-B",
                str(sample_build),
                f"-DCMAKE_PREFIX_PATH={install_dir}",
                f"-DCUDAToolkit_ROOT={CUDA_ROOT}",
            ],
        ):
            return False
        if not run(
            "build downstream find_package(nuka)",
            ["cmake", "--build", str(sample_build), "--", "-j2"],
        ):
            return False

        env = _env()
        env["LD_LIBRARY_PATH"] = (
            f"{install_dir / 'lib'}:{CUDA_ROOT / 'lib64'}:"
            f"{env.get('LD_LIBRARY_PATH', '')}"
        )
        result = subprocess.run([str(sample_build / "check")], env=env, cwd=sample_dir)
        if result.returncode == 0:
            print("PASS: downstream find_package(nuka) executable runs")
            return True
        print(f"FAIL: downstream find_package(nuka) executable exited {result.returncode}")
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
        checks.append((f"{golden.relative_to(ROOT)} LFS attributes", check_golden_lfs_attributes(golden)))
        checks.append((f"{golden.relative_to(ROOT)} LFS pointer", check_golden_lfs_pointer(golden)))
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
    checks.append(("find_package(nuka)", check_find_package(build_dir)))
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
