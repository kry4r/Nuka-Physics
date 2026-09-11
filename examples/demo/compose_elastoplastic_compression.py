"""Validate and compose an unlabelled compression video from simulated states."""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def read_capture(directory):
    config = json.loads((directory / "config.json").read_text())
    with (directory / "metrics.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    columns = {key: np.array([float(row[key]) for row in rows])
               for key in rows[0] if key != "stage"}
    if "stage" in rows[0]:
        columns["stage"] = [row["stage"] for row in rows]
    return config, columns


def analyze(directory):
    config, data = read_capture(directory)
    time = data["time_s"]
    reference = (time >= 0.2) & (time <= 0.3)
    mild = time < 2.1
    recovered = (time >= 1.85) & (time < 2.1)
    final = time >= 4.55
    height = float(np.mean(data["height_m"][reference]))
    elastic_height = float(np.mean(data["height_m"][recovered]))
    plastic_height = float(np.mean(data["height_m"][final]))
    widths = [float(abs(np.mean(data[key][recovered]) / np.mean(data[key][reference]) - 1))
              for key in ("width_x_m", "width_y_m")]
    energy = sum(data[key] for key in ("elastic_j", "hardening_j", "kinetic_j", "gravity_j"))
    completion = json.loads((directory / "completion.json").read_text())
    result = {
        "config": config,
        "reference_height_m": height,
        "elastic_loaded_height_m": float(np.min(data["height_m"][mild])),
        "elastic_recovered_height_m": elastic_height,
        "elastic_height_error_fraction": abs(elastic_height / height - 1),
        "elastic_width_error_fraction": max(widths),
        "mild_alpha_max": float(np.max(data["alpha_max"][mild])),
        "plastic_recovered_height_m": plastic_height,
        "residual_compression_fraction": 1 - plastic_height / height,
        "final_height_range_fraction": float(np.ptp(data["height_m"][final]) / height),
        "final_alpha_mean": float(data["alpha_mean"][-1]),
        "final_alpha_max": float(data["alpha_max"][-1]),
        "alpha_increment_min": float(np.min(data["alpha_increment_min"])),
        "det_fp_error_max": float(np.max(data["det_fp_error"])),
        "det_fe_min": float(np.min(data["det_fe_min"])),
        "peak_force_n": float(np.max(data["top_force_n"])),
        "mild_peak_force_n": float(np.max(data["top_force_n"][mild])),
        "boundary_work_j": float(data["boundary_work_j"][-1]),
        "plastic_dissipation_j": float(data["plastic_dissipation_j"][-1]),
        "unresolved_energy_loss_j": float(energy[0] + data["boundary_work_j"][-1] -
                                           energy[-1] - data["plastic_dissipation_j"][-1]),
        "momentum_balance_error_max_ns": float(np.max(abs(data["external_impulse_z_ns"] -
                                                            data["momentum_z_ns"]))),
        "reset_passed": completion["reset_passed"],
        "material_reset_between_loads": completion["material_reset_between_loads"],
    }
    result["checks"] = {
        "complete_capture": len(time) == completion["frames"],
        "finite_state": all(np.all(np.isfinite(values)) for key, values in data.items() if key != "stage"),
        "elastic_history": result["mild_alpha_max"] <= 2e-6,
        "elastic_height_recovery": result["elastic_height_error_fraction"] <= 0.02,
        "elastic_width_recovery": result["elastic_width_error_fraction"] <= 0.03,
        "plastic_history": result["final_alpha_mean"] >= 0.03,
        "permanent_deformation": result["residual_compression_fraction"] >= 0.10,
        "settled_residual_height": result["final_height_range_fraction"] <= 0.01,
        "monotonic_history": result["alpha_increment_min"] >= -2e-6,
        "isochoric_plasticity": result["det_fp_error_max"] <= 2e-3,
        "positive_elastic_volume": result["det_fe_min"] > 0,
        "environment_status": bool(np.all(data["env_status"] == 0)),
        "reset": completion["reset_passed"] and not completion["material_reset_between_loads"],
    }
    result["passed"] = all(result["checks"].values())
    return result


def compare(base, candidate):
    for key in ("specimen_m", "youngs", "poisson", "density", "yield_stress", "hardening_modulus",
                "gravity_z", "friction", "base_z", "gaps_m", "sample_hz", "body_contact_band"):
        if base["config"][key] != candidate["config"][key]:
            raise ValueError(f"Convergence comparison changed {key}")
    same_grid = base["config"]["dx"] == candidate["config"]["dx"]
    if same_grid and not np.isclose(candidate["config"]["dt"] * 2, base["config"]["dt"], rtol=1e-6, atol=0):
        raise ValueError("The timestep comparison must halve dt")
    height = base["reference_height_m"]
    result = {
        "kind": "timestep" if same_grid else "grid",
        "candidate_physics_passed": candidate["passed"],
        "recovered_height_difference_fraction": abs(base["plastic_recovered_height_m"] -
                                                      candidate["plastic_recovered_height_m"]) / height,
        "elastic_height_difference_fraction": abs(base["elastic_recovered_height_m"] -
                                                   candidate["elastic_recovered_height_m"]) / height,
        "plastic_strain_relative_difference": abs(candidate["final_alpha_mean"] /
                                                    base["final_alpha_mean"] - 1),
        "dissipation_relative_difference": abs(candidate["plastic_dissipation_j"] /
                                                 base["plastic_dissipation_j"] - 1),
        "peak_force_relative_difference": abs(candidate["peak_force_n"] / base["peak_force_n"] - 1),
    }
    if same_grid:
        result["timestep_checks_passed"] = (result["recovered_height_difference_fraction"] <= 0.02 and
            result["elastic_height_difference_fraction"] <= 0.02 and
            result["plastic_strain_relative_difference"] <= 0.05 and
            result["dissipation_relative_difference"] <= 0.05)
    return result


class Composer:
    def __init__(self, args, analysis):
        self.args, self.analysis = args, analysis
        self.config, self.data = read_capture(args.capture)
        first = Image.open(args.frames / "frame_000000.ppm")
        self.size = first.size

    def frame(self, index):
        canvas = Image.open(self.args.frames / f"frame_{index:06d}.ppm").convert("RGB")
        width, height = canvas.size
        left, right = width * 0.30, width * 0.70
        bottom, top = height * 0.975, height * 0.925
        times = self.data["time_s"]
        points = [(left + times[i] / times[-1] * (right-left),
                   bottom - max(0, self.data["top_force_n"][i]) /
                   self.analysis["peak_force_n"] * (bottom-top))
                  for i in range(index+1)]
        overlay = Image.new("RGBA", canvas.size)
        draw = ImageDraw.Draw(overlay)
        draw.line((left, bottom, right, bottom), fill=(205, 215, 224, 36), width=1)
        if len(points) > 1:
            draw.line(points, fill=(238, 181, 122, 78), width=8)
            overlay = overlay.filter(ImageFilter.GaussianBlur(4))
            draw = ImageDraw.Draw(overlay)
            draw.line(points, fill=(250, 214, 172, 218), width=2)
        x, y = points[-1]
        draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 240, 216, 245))
        return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def encode_video(composer, args, stem, keyframes):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("ffmpeg and ffprobe are required to encode and verify the video")
    width, height = composer.size
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{width}x{height}", "-framerate", str(args.fps), "-i", "pipe:0", "-an",
               "-vf", "hqdn3d=1.2:1.2:2.0:2.0", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out_dir / (stem + ".mp4"))]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(len(composer.data["time_s"])):
            frame = composer.frame(index)
            process.stdin.write(frame.tobytes())
            if index in keyframes:
                frame.save(args.out_dir / f"keyframe_{index:06d}.png")
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoding failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    probe = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(args.out_dir / (stem + ".mp4"))], text=True))
    stream = probe["streams"][0]
    frame_count = len(composer.data["time_s"])
    if (int(stream["nb_read_frames"]) != frame_count or stream["width"] != width or
            stream["height"] != height or stream["r_frame_rate"] != f"{args.fps}/1" or
            abs(float(probe["format"]["duration"]) - frame_count / args.fps) > 0.001):
        raise RuntimeError("The encoded video does not match the captured frames")
    metadata = {
        "capture": str(args.capture), "frames": str(args.frames), "frame_count": frame_count,
        "frame_rate": args.fps, "simulation_sample_hz": composer.config["sample_hz"],
        "slow_motion_factor": composer.config["sample_hz"] / args.fps,
        "overlay": "boundary force history, without labels", "state_interpolation": False,
        "encoding_command": command,
        "verification": probe,
    }
    (args.out_dir / "playback.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--compare", type=Path, action="append", default=[])
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("fps must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(args.capture)
    result["comparisons"] = {str(path): compare(result, analyze(path)) for path in args.compare}
    result["comparison_requirements_passed"] = all(value["candidate_physics_passed"] and
        (value["kind"] != "timestep" or value["timestep_checks_passed"]) for value in result["comparisons"].values())
    result["passed"] = result["passed"] and result["comparison_requirements_passed"]
    (args.out_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if args.analyze_only:
        return 0 if result["passed"] else 1
    if not result["passed"]:
        raise RuntimeError("The material capture does not pass the frozen demonstration checks")
    args.frames = args.frames or args.capture / "frames"
    composer = Composer(args, result)
    if args.preview:
        for path in sorted(args.frames.glob("frame_*.ppm")):
            index = int(path.stem.split("_")[-1])
            composer.frame(index).save(args.out_dir / f"frame_{index:06d}.png")
        return 0
    encode_video(composer, args, "elastoplastic_compression", (36, 120, 240, 360, 444, 576))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
