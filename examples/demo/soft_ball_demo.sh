#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# soft_ball_demo.sh -- one-command driver for the soft<->rigid coupling clip.
#
# STEPS nk::World on CUDA (a real tetrahedral soft ball -- a sphere rest-lattice
# tessellated into distance + volume tets -- dropped onto a static ground plane on
# the ONE general body<->particle row solver) and renders the live deforming
# surface on the CUDA RT backend (beauty) or lavapipe raster, then encodes the PPM
# sequence to an mp4 + writes 3 hero stills (airborne / max-squash / recovered).
#
# Build the exe first (build-viewer / lavapipe + CUDA RT):
#   cmake --build build-viewer --target nuka_soft_ball_demo -j8
#
# Usage (from repo root):
#   bash examples/demo/soft_ball_demo.sh            # beauty hero (GPU RT)
#   NK_FLAT=1 bash examples/demo/soft_ball_demo.sh  # raster dailies (lavapipe)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXE="build-viewer/tests/nuka_soft_ball_demo"
W="${NK_W:-1920}"
H="${NK_H:-1080}"
STRIDE="${NK_STRIDE:-2}"
FPS="${NK_FPS:-50}"
SAMPLES="${NK_SAMPLES:-16}"
SHADOW_MAP="${NK_SHADOW_MAP:-2048}"
FRAMES_DIR="${NK_OUTDIR:-/tmp/soft_ball_frames}"
PNG_DIR="${NK_PNGDIR:-out/soft_ball_demo}"
OUT_MP4="out/soft_ball.mp4"

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

if [[ ! -x "$EXE" ]]; then
  echo "[soft_ball_demo.sh] missing exe $EXE -- build it first:" >&2
  echo "  cmake --build build-viewer --target nuka_soft_ball_demo -j8" >&2
  exit 2
fi

# Beauty hero (GPU RT) by default; NK_FLAT=1 uses the lavapipe raster dailies.
RENDER_ARGS=(--gpu --beauty --samples "$SAMPLES" --shadow-map "$SHADOW_MAP")
if [[ "${NK_FLAT:-0}" == "1" ]]; then RENDER_ARGS=(); fi

rm -rf "$FRAMES_DIR"
mkdir -p "$FRAMES_DIR" "$PNG_DIR" out

echo "[soft_ball_demo.sh] simulating + rendering ${W}x${H} stride=${STRIDE} -> $FRAMES_DIR"
"$EXE" --width "$W" --height "$H" --stride "$STRIDE" \
  --out-dir "$FRAMES_DIR" --png-dir "$PNG_DIR" "${RENDER_ARGS[@]}"

NFRAMES=$(ls "$FRAMES_DIR"/frame_*.ppm 2>/dev/null | wc -l)
echo "[soft_ball_demo.sh] encoding $NFRAMES frames @ ${FPS} fps -> $OUT_MP4"
/usr/bin/ffmpeg -y -framerate "$FPS" -i "$FRAMES_DIR/frame_%06d.ppm" \
  -pix_fmt yuv420p -crf 18 -c:v libx264 "$OUT_MP4"

echo "[soft_ball_demo.sh] DONE -> $OUT_MP4 + hero stills in $PNG_DIR"
/usr/bin/ffprobe -v error -show_entries format=duration,size:stream=width,height \
  -of default=noprint_wrappers=1 "$OUT_MP4" || true
