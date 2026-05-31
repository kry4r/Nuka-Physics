#!/usr/bin/env bash
# build_python_wheel.sh -- build the `nuka` wheel (nanobind ext + python pkg) and
# smoke-test it (import nuka; create a Device + World; one step) in the
# nuka-v03 env. p02-T3.
#
# ABI / build facts this follows (see python/README.md, python/CMakeLists.txt):
#   * compiled g++-10, _GLIBCXX_USE_CXX11_ABI=1 (match the engine); torch interop
#     is DLPack-only (libtorch is NOT linked into _nuka_ext).
#   * the extension's INSTALL_RPATH is absolute to <repo>/build-cuda128/src
#     (libnuka.so) + the CUDA lib dir, so the installed wheel imports with no
#     LD_LIBRARY_PATH. We therefore smoke-test via `pip install --target` into a
#     throwaway dir (NOT --force-reinstall into the env, which would clobber the
#     developer's editable install).
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   scripts/build_python_wheel.sh
#
# Env overrides:
#   PYTHON   python interpreter (default: the nuka-v03 env python)
#   PIP      pip            (default: the nuka-v03 env pip)
#   WHEEL_OUT  wheel output dir (default: <repo>/python/dist)
#   SCENE    smoke scene    (default: <repo>/examples/scenes/go2_float.usda)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/nuka-v03/bin/python}"
PIP="${PIP:-/root/miniconda3/envs/nuka-v03/bin/pip}"
WHEEL_OUT="${WHEEL_OUT:-${REPO_ROOT}/python/dist}"
SCENE="${SCENE:-${REPO_ROOT}/examples/scenes/go2_float.usda}"

export CC="${CC:-gcc-10}"
export CXX="${CXX:-g++-10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "== [1/3] building wheel into ${WHEEL_OUT} =="
rm -rf "${WHEEL_OUT}"
mkdir -p "${WHEEL_OUT}"
# --no-build-isolation reuses the env's nanobind/scikit-build-core/ninja and the
# pinned g++-10 ABI. --no-deps keeps it a pure `nuka` wheel.
"${PIP}" wheel "${REPO_ROOT}/python" --no-build-isolation --no-deps -w "${WHEEL_OUT}"

WHEEL="$(ls -1 "${WHEEL_OUT}"/nuka-*.whl | head -n1)"
echo "built wheel: ${WHEEL}"

echo "== [2/3] ABI check: readelf -d NEEDED (must be NO libtorch / libc10) =="
SMOKE_DIR="$(mktemp -d)"
trap 'rm -rf "${SMOKE_DIR}"' EXIT
"${PIP}" install --no-deps --target "${SMOKE_DIR}" "${WHEEL}" >/dev/null
EXT_SO="$(ls -1 "${SMOKE_DIR}"/nuka/_nuka_ext*.so | head -n1)"
readelf -d "${EXT_SO}" | grep NEEDED
if readelf -d "${EXT_SO}" | grep -Eiq 'libtorch|libc10'; then
  echo "FATAL: wheel links libtorch/libc10 -- ABI rule violated" >&2
  exit 1
fi

echo "== [3/3] smoke: import nuka (from installed wheel) + Device + World + step =="
# An EDITABLE `nuka` install (developer convenience) registers a meta-path
# finder that shadows any `--target` dir on sys.path, so the smoke would import
# the source tree instead of the wheel. To prove the WHEEL imports, temporarily
# uninstall an editable/regular `nuka`, smoke, then restore the editable install.
HAD_EDITABLE=0
if "${PIP}" show nuka >/dev/null 2>&1; then
  HAD_EDITABLE=1
  "${PIP}" uninstall -y nuka >/dev/null 2>&1 || true
fi
restore_editable() {
  if [ "${HAD_EDITABLE}" = "1" ]; then
    ( cd "${REPO_ROOT}/python" && "${PIP}" install -e . --no-build-isolation >/dev/null 2>&1 ) \
      && echo "(restored developer editable install)"
  fi
}
trap 'rm -rf "${SMOKE_DIR}"; restore_editable' EXIT

PYTHONPATH="${SMOKE_DIR}" "${PYTHON}" - "${SCENE}" "${SMOKE_DIR}" <<'PY'
import sys
import nuka
scene, smoke_dir = sys.argv[1], sys.argv[2]
assert smoke_dir in nuka.__file__, f"imported {nuka.__file__}, not the wheel"
print("nuka from:", nuka.__file__)
print("engine_version:", nuka.__engine_version__)
with nuka.Device.create(0) as dev:
    with nuka.World.create_from_scene(dev, scene, env_count=4) as w:
        print("env_count:", w.env_count, "action_dim:", w.action_dim)
        w.step()
        print("SMOKE OK: device create + world create + one step (from wheel)")
PY

echo "== wheel build + smoke PASSED =="
