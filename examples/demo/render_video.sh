#!/usr/bin/env bash
# [p04-T2 demo] End-to-end: capture the PROVEN Go2 policy walk -> PPM frames -> MP4.
#
# HONESTY: this renders the v0.3 exit-#6 deliverable ("a 4096-env Go2 locomotion
# video") by REPLAYING the externally-trained, Nuka-VALIDATED Go2 policy (motion.pt,
# unitree_rl_gym PR#62; sim-val #41/#43). It does NOT demonstrate exit-#3
# (from-scratch PPO convergence) -- that is a separate gate. We render a
# representative 16-env grid; the 4096-env throughput claim is carried by the
# existing frozen perf baseline (out/perf/baseline_rtx4000ada_4096_frozen.json)
# and the harness's 4096-env finite smoke, NOT by rendering 4096 robots.
#
# Frame encode tail (PPM P6 -> MP4) is the documented path from
# docs/architecture/headless-rendering.md. The FRAMES themselves come from the
# self-contained Python rasterizer go2_demo_render.py (side-view stick skeleton),
# NOT the offscreen-Vulkan path -- see that file's header for why.
#
# Usage:
#   examples/demo/render_video.sh [out_dir] [seconds] [envs] [fps]
# Defaults: out/go2_demo, 6 s, 16 envs, 30 fps (policy is 50 Hz; 30 fps ~ slight
# slow-mo for readability -- use 50 for real-time).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/root/miniconda3/envs/nuka-v03/bin/python"

OUT_DIR="${1:-${ROOT}/out/go2_demo}"
SECONDS_SIM="${2:-6}"
ENVS="${3:-16}"
FPS="${4:-30}"

NPZ="${OUT_DIR}/go2_rollout.npz"
FRAMES="${OUT_DIR}/frames"
MP4="${OUT_DIR}/go2_locomotion_16env.mp4"

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES=0

echo "=== [1/3] capture proven policy rollout (${ENVS} envs, ${SECONDS_SIM}s) ==="
# Defensive guard: `import nuka` can be broken in a dirty working tree if
# python/nuka/__init__.py references a Field (e.g. BASE_POSE) the BUILT _nuka_ext
# does not yet export (a stale-binding / uncommitted-edit hazard). If a plain
# `import nuka` fails, temporarily restore the COMMITTED __init__.py for the
# capture step only, then restore the working-tree version byte-for-byte. Render
# + encode (below) do not import nuka. Requires git on PATH (git-lfs toolchain).
NUKA_INIT="${ROOT}/python/nuka/__init__.py"
RESTORE_INIT=0
if ! "${PY}" -c "import nuka" >/dev/null 2>&1; then
    if command -v git >/dev/null 2>&1 && [ -f "${NUKA_INIT}" ]; then
        echo "  [guard] 'import nuka' is broken in the working tree; temporarily"
        echo "          restoring the committed python/nuka/__init__.py for capture."
        cp "${NUKA_INIT}" "${NUKA_INIT}.demo_wip_bak"
        git -C "${ROOT}" show HEAD:python/nuka/__init__.py > "${NUKA_INIT}"
        RESTORE_INIT=1
    fi
fi
restore_init() {
    if [ "${RESTORE_INIT}" = "1" ] && [ -f "${NUKA_INIT}.demo_wip_bak" ]; then
        cp "${NUKA_INIT}.demo_wip_bak" "${NUKA_INIT}"
        rm -f "${NUKA_INIT}.demo_wip_bak"
        echo "  [guard] restored the working-tree python/nuka/__init__.py."
    fi
}
trap restore_init EXIT
"${PY}" "${ROOT}/examples/demo/go2_demo_capture.py" \
    --envs "${ENVS}" --seconds "${SECONDS_SIM}" --out "${NPZ}"
restore_init
trap - EXIT

echo "=== [2/3] rasterize -> PPM frames (4x4 side-view grid) ==="
rm -f "${FRAMES}"/frame_*.ppm 2>/dev/null || true
"${PY}" "${ROOT}/examples/demo/go2_demo_render.py" \
    --npz "${NPZ}" --frames-dir "${FRAMES}" --width 1280 --height 720

echo "=== [3/3] encode PPM -> MP4 via ffmpeg (documented headless tail) ==="
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg NOT on PATH. Frames are in ${FRAMES}. Encode elsewhere with:"
    echo "  ffmpeg -framerate ${FPS} -i ${FRAMES}/frame_%05d.ppm -c:v libx264 \\"
    echo "         -pix_fmt yuv420p ${MP4}"
    exit 2
fi
ffmpeg -y -framerate "${FPS}" -i "${FRAMES}/frame_%05d.ppm" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 "${MP4}"

echo ""
echo "DONE. MP4: ${MP4}"
ls -la "${MP4}"
echo "Sample PNGs (for committing small evidence):"
for f in 00000 00075 00149; do
    if [ -f "${FRAMES}/frame_${f}.ppm" ]; then
        ffmpeg -y -i "${FRAMES}/frame_${f}.ppm" "${OUT_DIR}/sample_${f}.png" \
            >/dev/null 2>&1 && echo "  ${OUT_DIR}/sample_${f}.png"
    fi
done
