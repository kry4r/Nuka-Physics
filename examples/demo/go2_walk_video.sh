#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# go2_walk_video.sh -- one-command driver for the BEAUTIFUL Go2 WALK demo video.
#
# POSE-REPLAYS the proven, golden-validated Go2 walk (out/go2_walk_trajectory.bin,
# the policy rolls +3.91 m forward, deterministic/D1) through the offscreen
# lavapipe raster renderer with the M8.5 go2 visual meshes, a premium dark-body /
# gunmetal-joint material override, a base-tracking hero follow-camera, and a
# studio ground at z=0 -- then encodes the PPM sequence to an mp4 with ffmpeg.
#
# NO physics / NO policy / NO torch in C++: the walk was simulated + dumped by
# examples/demo/go2_dump_walk_trajectory.py. This tool only REPLAYS the recorded
# per-frame link poses (see the .meta.json / the .cpp header for the layout).
#
# CADENCE: the trajectory is 1600 physics sub-steps (8.0 s sim @ dt=0.005). We
# render every 2nd sub-step (stride=2 -> 800 frames) and encode at 50 fps -> a
# smooth ~16 s cinematic clip (a ~0.5x slow-motion walk that reads the gait +
# body bob clearly). Override via env: NK_STRIDE, NK_FPS, NK_W, NK_H, NK_OUTDIR.
#
# Build the exe first (build-viewer / lavapipe):
#   cmake --build build-viewer --target nuka_go2_walk_video -j8
#
# Usage (from repo root):
#   bash examples/demo/go2_walk_video.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXE="build-viewer/tests/nuka_go2_walk_video"
BIN="out/go2_walk_trajectory.bin"
STRIDE="${NK_STRIDE:-2}"
FPS="${NK_FPS:-50}"
W="${NK_W:-1920}"
H="${NK_H:-1080}"
FRAMES_DIR="${NK_OUTDIR:-/tmp/go2_walk_frames}"
OUT_MP4="out/go2_walk.mp4"

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

if [[ ! -x "$EXE" ]]; then
  echo "[go2_walk_video.sh] missing exe $EXE -- build it first:" >&2
  echo "  cmake --build build-viewer --target nuka_go2_walk_video -j8" >&2
  exit 2
fi
if [[ ! -f "$BIN" ]]; then
  echo "[go2_walk_video.sh] missing trajectory $BIN" >&2
  exit 2
fi

rm -rf "$FRAMES_DIR"
mkdir -p "$FRAMES_DIR" out

echo "[go2_walk_video.sh] rendering ${W}x${H} stride=${STRIDE} -> $FRAMES_DIR"
"$EXE" --stride "$STRIDE" --width "$W" --height "$H" --bin "$BIN" --out-dir "$FRAMES_DIR"

NFRAMES=$(ls "$FRAMES_DIR"/frame_*.ppm 2>/dev/null | wc -l)
echo "[go2_walk_video.sh] encoding $NFRAMES frames @ ${FPS} fps -> $OUT_MP4"
/usr/bin/ffmpeg -y -framerate "$FPS" -i "$FRAMES_DIR/frame_%06d.ppm" \
  -pix_fmt yuv420p -crf 18 -c:v libx264 "$OUT_MP4"

echo "[go2_walk_video.sh] DONE -> $OUT_MP4"
/usr/bin/ffprobe -v error -show_entries format=duration,size:stream=width,height \
  -of default=noprint_wrappers=1 "$OUT_MP4" || true
