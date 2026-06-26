#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# go2_cloth_drape_demo.sh -- one-command driver for the cloth<->rigid coupling clip.
#
# STEPS nk::World on CUDA (a free, unpinned square cloth released flat above a
# PD-standing Go2 cooked from go2.nks for its FULL per-link collision skeleton,
# draped over the body on the ONE general body<->particle row solver) and renders
# the live deforming cloth + the FK-rebound Go2 visuals offscreen, then encodes the
# PPM sequence to an mp4 with ffmpeg + writes 3 hero stills (falling / draping /
# settled). Default is the lavapipe raster preview; pass NK_GPU=1 for the CUDA
# ray-traced beauty hero (--gpu --beauty).
#
# Build the exe first (build-cuda128: an OPTIMISED build; build-viewer is -O0 and
# is far too slow to simulate this large fine cloth):
#   make -C build-cuda128 nuka_go2_cloth_drape_demo -j"$(nproc)"
#
# Usage (from repo root):
#   bash examples/demo/go2_cloth_drape_demo.sh           # flat dailies
#   NK_GPU=1 NK_SAMPLES=16 bash examples/demo/go2_cloth_drape_demo.sh   # beauty hero
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXE="build-cuda128/tests/nuka_go2_cloth_drape_demo"
W="${NK_W:-1920}"
H="${NK_H:-1080}"
STRIDE="${NK_STRIDE:-2}"
FPS="${NK_FPS:-50}"
SAMPLES="${NK_SAMPLES:-20}"
DRAPE="${NK_DRAPE:-800}"
FRAMES_DIR="${NK_OUTDIR:-/tmp/go2_cloth_frames}"
PNG_DIR="${NK_PNGDIR:-out/go2_cloth_demo}"
OUT_MP4="out/go2_cloth_drape.mp4"

GPU_ARGS=()
if [[ "${NK_GPU:-0}" != "0" ]]; then
  GPU_ARGS=(--gpu --beauty --samples "$SAMPLES")
fi

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="build-cuda128/src:/opt/cuda-12.8-root/lib64:/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

if [[ ! -x "$EXE" ]]; then
  echo "[go2_cloth_drape_demo.sh] missing exe $EXE -- build it first:" >&2
  echo "  make -C build-cuda128 nuka_go2_cloth_drape_demo -j\"\$(nproc)\"" >&2
  exit 2
fi

rm -rf "$FRAMES_DIR"
mkdir -p "$FRAMES_DIR" "$PNG_DIR" out

echo "[go2_cloth_drape_demo.sh] simulating + rendering ${W}x${H} stride=${STRIDE} ${GPU_ARGS[*]:-raster} -> $FRAMES_DIR"
"$EXE" --width "$W" --height "$H" --stride "$STRIDE" --drape "$DRAPE" --out-dir "$FRAMES_DIR" --png-dir "$PNG_DIR" "${GPU_ARGS[@]}"

NFRAMES=$(ls "$FRAMES_DIR"/frame_*.ppm 2>/dev/null | wc -l)
echo "[go2_cloth_drape_demo.sh] encoding $NFRAMES frames @ ${FPS} fps -> $OUT_MP4"
/usr/bin/ffmpeg -y -framerate "$FPS" -i "$FRAMES_DIR/frame_%06d.ppm" \
  -pix_fmt yuv420p -crf 18 -c:v libx264 "$OUT_MP4"

echo "[go2_cloth_drape_demo.sh] DONE -> $OUT_MP4 + hero stills in $PNG_DIR"
/usr/bin/ffprobe -v error -show_entries format=duration,size:stream=width,height \
  -of default=noprint_wrappers=1 "$OUT_MP4" || true
