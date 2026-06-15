#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# h1_grasp_video.sh -- M10 Phase A: render the HEADLINE H1 cup-grasp demo to mp4.
#
# Drives the unified offscreen C++ video tool (nuka_h1_grasp_video) to render the
# proven 51-DOF H1 cup-grasp choreography (approach -> close -> lift -> sustained
# friction-only hold) as a PBR-beautiful PPM sequence on lavapipe-CPU, then muxes
# it to an mp4 with ffmpeg/libx264.
#
# PHYSICS: the FROZEN examples/scenes/h1_cup_table.nks union cook (D1-untouched);
#   the per-step TableTorque drive is byte-for-byte the h1_grasp_lift gate path.
# RENDER MESHES: the committed, self-contained examples/scenes/h1_visual.{nks,nka}
#   (no .nuka-assets/ dependency -- fresh-clone reproducible). The cup is the
#   union cook's hull point cloud, convex-hull-triangulated in the tool.
#
# Built behind NK_BUILD_VULKAN_VALIDATION (build-viewer). Needs NO display
# (offscreen lavapipe ICD). ffmpeg at /usr/bin/ffmpeg.
#
# Usage:  examples/demo/h1_grasp_video.sh [out_mp4] [frames] [width] [height] [fps]
# Defaults: out/h1_grasp.mp4, 420 frames (8x slow-mo), 1280x720, 30 fps.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

OUT_MP4="${1:-${ROOT}/out/h1_grasp.mp4}"
FRAMES="${2:-420}"
WIDTH="${3:-1280}"
HEIGHT="${4:-720}"
FPS="${5:-30}"

EXE="${ROOT}/build-viewer/tests/nuka_h1_grasp_video"
if [[ ! -x "${EXE}" ]]; then
    echo "build first:  cmake --build build-viewer -j8 --target nuka_h1_grasp_video" >&2
    exit 2
fi

FRAME_DIR="$(mktemp -d /tmp/h1_grasp_frames.XXXXXX)"
trap 'rm -rf "${FRAME_DIR}"' EXIT

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}"

echo "=== [1/2] render ${FRAMES} frames (${WIDTH}x${HEIGHT}) offscreen on lavapipe ==="
"${EXE}" --frames "${FRAMES}" --width "${WIDTH}" --height "${HEIGHT}" \
         --out-dir "${FRAME_DIR}"

echo "=== [2/2] encode -> ${OUT_MP4} (${FPS} fps, libx264 crf 18) ==="
mkdir -p "$(dirname "${OUT_MP4}")"
/usr/bin/ffmpeg -y -framerate "${FPS}" -i "${FRAME_DIR}/frame_%06d.ppm" \
    -pix_fmt yuv420p -crf 18 -c:v libx264 "${OUT_MP4}"

echo "DONE -> ${OUT_MP4}"
/usr/bin/ffprobe -v error -show_entries format=duration,size:stream=width,height,codec_name \
    -of default=noprint_wrappers=1 "${OUT_MP4}" || true
