#!/usr/bin/env python3
"""Run the pretrained G1 dance policy in Nuka and capture live-world video."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
import nuka
from nuka.tasks.g1_dance import G1DanceController, G1OnnxActor
from nuka.tasks.g1_motion_contract import load_g1_motion

SCENE_XML = REPO / ".nuka-assets/generated/g1/g1_dance_stage.xml"
SCENE_NKS = REPO / ".nuka-assets/generated/g1/g1_dance_stage.nks"
MOTION = REPO / ".nuka-assets/src/g1-moves/dance/J_Dance17_Shuffle/training/J_Dance17_Shuffle.npz"
POLICY = REPO / ".nuka-assets/src/g1-moves/dance/J_Dance17_Shuffle/policy/J_Dance17_Shuffle_policy.onnx"


def ensure_nuka_scene_bundle() -> Path:
    """Persist imported STL visuals as the MESH chunks used by beauty rendering."""
    nka = SCENE_NKS.with_suffix(".nka")
    stale = (
        not SCENE_NKS.exists()
        or not nka.exists()
        or SCENE_NKS.stat().st_mtime < SCENE_XML.stat().st_mtime
    )
    if stale:
        scene = nuka.Scene.load(str(SCENE_XML))
        try:
            scene.save(str(SCENE_NKS))
        finally:
            scene.destroy()
    return SCENE_NKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=12.0,
                        help="captured dance duration after preroll")
    parser.add_argument("--preroll", type=float, default=0.0,
                        help="simulate before capture (Shuffle starts immediately)")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--control-backend", choices=("external_torque", "engine_pd"),
                        default="engine_pd")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--out", default="out/g1_shuffle")
    args = parser.parse_args()
    if args.seconds <= 0.0 or args.preroll < 0.0:
        raise ValueError("seconds must be positive and preroll must be non-negative")

    output = REPO / args.out
    output.mkdir(parents=True, exist_ok=True)
    frames = output / "frames"
    if args.video:
        if frames.exists():
            shutil.rmtree(frames)
        frames.mkdir(parents=True)

    scene_path = ensure_nuka_scene_bundle()
    motion = load_g1_motion(MOTION)
    total_seconds = min(args.preroll + args.seconds, motion.duration)
    captured_seconds = total_seconds - args.preroll
    if captured_seconds <= 0.0:
        raise ValueError("preroll begins after the motion clip ends")

    metrics: list[dict[str, float]] = []
    captured_q: list[np.ndarray] = []
    captured_root_xy: list[np.ndarray] = []
    frame_count = 0
    actor = None if args.reference_only else G1OnnxActor.load(POLICY)
    with nuka.Device.create(0) as device:
        controller = G1DanceController(
            str(scene_path), motion, device, actor,
            control_backend=args.control_backend, dt=args.dt,
            kp_scale=args.kp_scale, kd_scale=args.kd_scale)
        try:
            steps = int(total_seconds / args.dt)
            frame_stride = max(1, int(round(1.0 / (args.fps * args.dt))))
            capture_start_step = int(round(args.preroll / args.dt))
            for step in range(steps):
                time_s = step * args.dt
                sample = controller.step(time_s, reference_only=args.reference_only)
                sample["step"] = step
                sample["time_s"] = time_s
                metrics.append(sample)
                if sample["finite"] != 1.0:
                    raise RuntimeError(f"non-finite G1 state at step {step}")
                if step < capture_start_step:
                    continue

                capture_step = step - capture_start_step
                if capture_step % controller.policy_stride == 0:
                    captured_q.append(
                        controller.q[0, 1:].detach().cpu().numpy().copy())
                    captured_root_xy.append(
                        controller.base[0, :2].detach().cpu().numpy().copy())
                if args.video and capture_step % frame_stride == 0:
                    base = controller.base[0, :3].detach().cpu().numpy()
                    progress = min(1.0, capture_step * args.dt / captured_seconds)
                    orbit = math.radians(-30.0 + 65.0 * progress)
                    radius = 2.72
                    eye = (
                        float(base[0] + radius * math.cos(orbit)),
                        float(base[1] + radius * math.sin(orbit)),
                        float(1.34 + 0.08 * math.sin(progress * math.pi)),
                    )
                    look = (float(base[0]), float(base[1]), 0.82)
                    image = controller.world.render_beauty(
                        eye=eye, look=look, fov_deg=35.0,
                        width=args.width, height=args.height, spp=args.spp)
                    Image.fromarray(np.ascontiguousarray(image)).save(
                        frames / f"frame_{frame_count:05d}.png")
                    frame_count += 1
        finally:
            controller.close()

    q_samples = np.asarray(captured_q, dtype=np.float32)
    root_samples = np.asarray(captured_root_xy, dtype=np.float32)
    joint_span = np.ptp(q_samples, axis=0)
    root_path = float(np.linalg.norm(np.diff(root_samples, axis=0), axis=1).sum())
    joint_span_mean = float(joint_span.mean())
    joint_span_max = float(joint_span.max())
    dance_visible = joint_span_mean > 0.08 and joint_span_max > 0.35
    min_root_height = float(min(row["root_z_m"] for row in metrics))
    final_root_height = float(metrics[-1]["root_z_m"])
    upright = min_root_height > 0.55 and final_root_height > 0.55
    summary = {
        "mode": "reference_only" if args.reference_only else "pretrained_onnx_actor_160x29",
        "policy": None if args.reference_only else str(POLICY.relative_to(REPO)),
        "scene": str(scene_path.relative_to(REPO)),
        "visual_asset_mode": "nks_nka_real_g1_meshes",
        "control_backend": args.control_backend,
        "engine_version": nuka.__engine_version__,
        "physics_dt": args.dt,
        "kp_scale": args.kp_scale,
        "kd_scale": args.kd_scale,
        "policy_hz": 50,
        "reference_fps": motion.fps,
        "preroll_seconds": args.preroll,
        "captured_seconds": captured_seconds,
        "steps": len(metrics),
        "video_frames": frame_count,
        "finite": all(row["finite"] == 1.0 for row in metrics),
        "upright": upright,
        "dance_visible": dance_visible,
        "joint_motion_span_mean_rad": joint_span_mean,
        "joint_motion_span_max_rad": joint_span_max,
        "root_xy_path_m": root_path,
        "observation_abs_max": float(max(row["observation_abs_max"] for row in metrics)),
        "action_abs_max": float(max(row["action_abs_max"] for row in metrics)),
        "target_abs_max_rad": float(max(row["target_abs_max_rad"] for row in metrics)),
        "joint_rmse_mean_rad": float(np.mean([row["joint_rmse_rad"] for row in metrics])),
        "joint_rmse_p95_rad": float(np.percentile([row["joint_rmse_rad"] for row in metrics], 95)),
        "root_height_min_m": min_root_height,
        "root_height_final_m": final_root_height,
    }
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics), encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.video:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for --video")
        subprocess.run([
            ffmpeg, "-y", "-framerate", str(args.fps),
            "-i", str(frames / "frame_%05d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(output / "g1_dance.mp4"),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(json.dumps(summary, sort_keys=True))
    if not summary["finite"] or not upright or (not args.reference_only and not dance_visible):
        raise RuntimeError(
            "G1 rollout failed gate: "
            f"finite={summary['finite']} upright={upright} dance_visible={dance_visible} "
            f"min_root_height={min_root_height:.3f}m")


if __name__ == "__main__":
    main()
