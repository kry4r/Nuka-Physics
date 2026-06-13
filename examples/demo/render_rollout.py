#!/usr/bin/env python
"""[M8 T6 / G4] One command: render rollout mp4(s) with visual meshes + materials.

This is the M8 G4 deliverable -- ONE python command that drives the host frame
loop over an ``nk::World`` (Decision D2: the nk-backed union world), through the
offscreen Vulkan forward PBR raster renderer, and muxes an mp4 of the rollout.

WHAT RENDERS TODAY
------------------
  * H1 (``examples/scenes/h1_cup_table.nks``) -- the whole-body H1 + in-hand cup
    + table union scene. Cooked to an nk::World, rendered with visual meshes +
    PBR materials, captured to PPM P6, muxed to mp4. END-TO-END.

  * go2 -- a NAMED, SCOPED asset deferral (recon Decision D2/D3). There is no
    cooked go2 ``.nks``/``.nka`` yet, so go2 is deferred to M9 / the asset pass.
    This driver LOGS that clearly and does NOT fake a go2 render. The G4
    mechanism is proven by the H1 mp4; the go2 half is asset-gated, not
    code-gated (the same Recorder renders it the moment a go2 ``.nks`` exists).

VULKAN GATING (Decision D5)
---------------------------
The recorder lives behind ``NK_BUILD_VULKAN_VALIDATION``. On a Vulkan-less
``libnuka`` ``Recorder.create`` raises ``RuntimeError`` (NUKA_RESULT_NOT_SUPPORTED);
this driver reports that as a skip rather than crashing.

USAGE
-----
    python examples/demo/render_rollout.py [--out-dir DIR] [--frames N]
                                           [--fps F] [--scene PATH]

Run from the repo root (the scene's assets are repo-relative).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

# The scenes this driver knows how to render. Each entry is (name, scene_path,
# asset_ready). go2 is intentionally asset-pending (no cooked .nks yet).
H1_SCENE = "examples/scenes/h1_cup_table.nks"
GO2_SCENE = "examples/scenes/go2_walk.nks"  # does not exist yet (M9 / asset pass)

ROLLOUTS = [
    ("h1", H1_SCENE, True),
    ("go2", GO2_SCENE, False),
]


def _assets_present(scene_path: str) -> bool:
    return (REPO / scene_path).exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render rollout mp4(s) (M8 G4).")
    parser.add_argument(
        "--out-dir",
        default=str(REPO / "out" / "render_rollout"),
        help="output directory for frames + mp4s",
    )
    parser.add_argument("--frames", type=int, default=120, help="frames per rollout")
    parser.add_argument("--fps", type=int, default=30, help="mp4 frame rate (D6: 30)")
    parser.add_argument(
        "--scene",
        default=None,
        help="render ONLY this .nks scene (overrides the built-in rollout list)",
    )
    parser.add_argument("--width", type=int, default=0, help="0 -> 1920 (D6)")
    parser.add_argument("--height", type=int, default=0, help="0 -> 1080 (D6)")
    args = parser.parse_args()

    out_root = pathlib.Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        import nuka  # noqa: E402
        from nuka.recorder import Recorder, Camera  # noqa: E402
    except Exception as exc:  # pragma: no cover -- env/import issue
        print(f"[render_rollout] FAILED to import nuka: {exc}", file=sys.stderr)
        return 2

    # Custom single-scene mode.
    if args.scene is not None:
        rollouts = [(pathlib.Path(args.scene).stem, args.scene, _assets_present(args.scene))]
    else:
        rollouts = ROLLOUTS

    produced: list[str] = []
    skipped: list[str] = []

    dev = None
    try:
        dev = nuka.Device.create(0)
    except Exception as exc:
        print(f"[render_rollout] no CUDA device: {exc}", file=sys.stderr)
        return 3

    # A presentable 3/4 side shot of the H1 + cup (recorded on the Camera; the C
    # path auto-frames the scene AABB today, so this is informational).
    camera = Camera(eye=(2.6, -2.6, 1.7), target=(0.0, 0.0, 0.95), fov_degrees=42.0)

    for name, scene_path, ready in rollouts:
        if not ready or not _assets_present(scene_path):
            # go2 (and any other not-yet-cooked scene): NAMED asset deferral.
            print(
                f"[render_rollout] {name}: ASSET-PENDING -- no cooked '{scene_path}' "
                f"exists yet. Deferred to M9 / the asset pass (the same Recorder "
                f"renders it once a {name} .nks is cooked). NOT faking a render.",
                flush=True,
            )
            skipped.append(name)
            continue

        frames_dir = out_root / name / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = out_root / f"{name}.mp4"
        print(
            f"[render_rollout] {name}: cooking '{scene_path}' -> nk::World, "
            f"rendering {args.frames} frames @ {args.width or 1920}x"
            f"{args.height or 1080} with visual meshes + PBR materials ...",
            flush=True,
        )
        try:
            with Recorder(
                dev,
                scene_path,
                camera=camera,
                width=args.width,
                height=args.height,
            ) as rec:
                n = rec.capture(str(frames_dir), n_frames=args.frames)
                print(f"[render_rollout] {name}: captured {n} PPM frames -> {frames_dir}",
                      flush=True)
                try:
                    rec.to_video(str(mp4_path), out_dir=str(frames_dir), fps=args.fps)
                    print(f"[render_rollout] {name}: mp4 -> {mp4_path}", flush=True)
                    produced.append(str(mp4_path))
                except RuntimeError as exc:
                    # R6: ffmpeg missing/failed -> frames are on disk, mux skipped.
                    print(
                        f"[render_rollout] {name}: ffmpeg mux unavailable ({exc}). "
                        f"PPM frames are in {frames_dir}; encode elsewhere.",
                        file=sys.stderr,
                    )
                    skipped.append(name + " (ffmpeg)")
        except RuntimeError as exc:
            # D5: Vulkan-less libnuka, missing assets, or no graphics device.
            print(
                f"[render_rollout] {name}: recorder unavailable ({exc}). "
                f"Likely a Vulkan-less libnuka (build with NK_BUILD_VULKAN_VALIDATION) "
                f"or no Vulkan graphics device.",
                file=sys.stderr,
            )
            skipped.append(name + " (no-vulkan)")

    print("\n[render_rollout] SUMMARY", flush=True)
    for p in produced:
        print(f"  produced: {p}", flush=True)
    for s in skipped:
        print(f"  skipped:  {s}", flush=True)

    # G4 mechanism is proven if at least one mp4 (H1) was produced.
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
