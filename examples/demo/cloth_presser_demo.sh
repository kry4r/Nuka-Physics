#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cloth_presser_demo.sh -- one-command driver for the body<->particle coupling clip.
#
# STEPS nk::World on CUDA (a real 576-particle XPBD cloth draped over two ridges,
# pressed + released by a rigid sphere on the ONE general row solver) and renders
# the live deforming cloth offscreen on lavapipe, then encodes the PPM sequence to
# an mp4 with ffmpeg + writes 3 hero stills (settled / pressed / recovered) PNGs.
#
# Build the exe first (build-viewer / lavapipe):
#   cmake --build build-viewer --target nuka_cloth_presser_demo -j8
#
# Usage (from repo root):
#   bash examples/demo/cloth_presser_demo.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXE="build-viewer/tests/nuka_cloth_presser_demo"
W="${NK_W:-1920}"
H="${NK_H:-1080}"
STRIDE="${NK_STRIDE:-2}"
FPS="${NK_FPS:-50}"
FRAMES_DIR="${NK_OUTDIR:-/tmp/cloth_presser_frames}"
PNG_DIR="${NK_PNGDIR:-out/cloth_demo}"
OUT_MP4="out/cloth_presser.mp4"

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

if [[ ! -x "$EXE" ]]; then
  echo "[cloth_presser_demo.sh] missing exe $EXE -- build it first:" >&2
  echo "  cmake --build build-viewer --target nuka_cloth_presser_demo -j8" >&2
  exit 2
fi

rm -rf "$FRAMES_DIR"
mkdir -p "$FRAMES_DIR" "$PNG_DIR" out

echo "[cloth_presser_demo.sh] simulating + rendering ${W}x${H} stride=${STRIDE} -> $FRAMES_DIR"
"$EXE" --width "$W" --height "$H" --stride "$STRIDE" --out-dir "$FRAMES_DIR" --png-dir "$PNG_DIR"

NFRAMES=$(ls "$FRAMES_DIR"/frame_*.ppm 2>/dev/null | wc -l)
echo "[cloth_presser_demo.sh] encoding $NFRAMES frames @ ${FPS} fps -> $OUT_MP4"
/usr/bin/ffmpeg -y -framerate "$FPS" -i "$FRAMES_DIR/frame_%06d.ppm" \
  -pix_fmt yuv420p -crf 18 -c:v libx264 "$OUT_MP4"

echo "[cloth_presser_demo.sh] DONE -> $OUT_MP4 + hero stills in $PNG_DIR"
/usr/bin/ffprobe -v error -show_entries format=duration,size:stream=width,height \
  -of default=noprint_wrappers=1 "$OUT_MP4" || true
