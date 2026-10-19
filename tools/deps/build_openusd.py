"""Fetches and builds the OpenUSD that nuka's USD importer links against.

CMake runs this at configure time when NK_OPENUSD_ROOT holds no OpenUSD yet; it also runs by hand.
It clones OpenUSD at the pinned tag into <work>/src and runs its build_usd.py: shared libraries
without Python, imaging, usdview or the command-line tools, plus the Alembic (.abc) and Draco
plugins, installed to <prefix>. On Windows, run it from an x64 MSVC developer prompt.

    python tools/deps/build_openusd.py --prefix .nuka-deps/openusd-v26.08 [--work .nuka-deps] [--jobs N]
"""

import argparse
import os
import pathlib
import shutil
import stat
import subprocess
import sys

TAG = "v26.08"
REPO = "https://github.com/PixarAnimationStudios/OpenUSD.git"
# nuka's own library prefix, C++ namespace and plugin-path variable: another pxr in the same process
# (usd-core, Isaac Sim) shares neither nuka's libraries nor its plugins.
USD_ARGS = ("-DPXR_LIB_PREFIX=nukausd_ -DPXR_SET_INTERNAL_NAMESPACE=pxrInternal_nuka "
            "-DPXR_OVERRIDE_PLUGINPATH_NAME=NUKA_USD_PLUGINPATH -DPXR_BUILD_USD_VALIDATION=OFF")
# The plugins' own dependencies link statically: Windows resolves a plugin DLL's imports from the
# application's directory and PATH, never from the plugin's directory.
STATIC_ARGS = "-DBUILD_SHARED_LIBS=OFF -DALEMBIC_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON"


def _remove_read_only(func, path, _):
    """Git keeps its pack files read-only, which Windows refuses to delete."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", type=pathlib.Path, required=True)
    parser.add_argument("--work", type=pathlib.Path, default=repo / ".nuka-deps")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    work = args.work.resolve()
    src = work / "src" / f"OpenUSD-{TAG}"
    if not src.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", TAG, REPO, str(src)], check=True)
    build = work / "build" / f"openusd-{TAG}"
    command = [sys.executable, str(src / "build_scripts" / "build_usd.py"), str(args.prefix.resolve()),
               "--src", str(work / "src"), "--build", str(build), "--build-variant", "release",
               "--no-python", "--no-imaging", "--no-usdview", "--no-materialx", "--alembic", "--draco",
               "--no-tools", "--no-tests", "--no-examples", "--no-tutorials", "--no-docs", "-j", str(args.jobs),
               "--build-args", f"USD,{USD_ARGS}", f"OpenEXR,{STATIC_ARGS}", f"Alembic,{STATIC_ARGS}",
               f"Draco,{STATIC_ARGS}"]
    if subprocess.run(command).returncode:
        return 1
    # The build tree takes several GB and nothing reads it once the install is done.
    shutil.rmtree(build, onerror=_remove_read_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
