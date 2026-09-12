"""Check a dynamic gripper capture and compose its two camera views."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def file_sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_capture(directory):
    config = json.loads((directory / "config.json").read_text())
    with (directory / "metrics.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    data = {key: np.array([float(row[key]) for row in rows])
            for key in rows[0] if key != "stage"}
    return config, data


def rigid_aligned_shape_rms(directory):
    path = directory / "states.bin"
    with path.open("rb") as source:
        header = np.fromfile(source, dtype="<u4", count=7)
    if len(header) != 7 or tuple(header[:2]) != (0x4E554B41, 2):
        raise ValueError("Unsupported state capture")
    particles, bodies, links, frames = map(int, header[2:6])
    stride = particles * 3 + (bodies + links) * 7 + links
    if particles == 0 or path.stat().st_size != 32 + frames * stride * 4:
        raise ValueError("Incomplete state capture")
    states = np.memmap(path, dtype="<f4", mode="r", offset=32, shape=(frames, stride))
    reference = states[0, :particles * 3].reshape(particles, 3).astype(np.float64)
    reference -= reference.mean(axis=0)
    result = np.empty(frames)
    for frame in range(frames):
        current = states[frame, :particles * 3].reshape(particles, 3).astype(np.float64)
        current -= current.mean(axis=0)
        left, _, right = np.linalg.svd(reference.T @ current)
        left[:, -1] *= 1.0 if np.linalg.det(left @ right) >= 0.0 else -1.0
        difference = reference @ (left @ right) - current
        result[frame] = np.sqrt(np.einsum("ij,ij->", difference, difference) / particles)
    return result


def analyze(directory):
    config, data = read_capture(directory)
    completion = json.loads((directory / "completion.json").read_text())
    time = data["time_s"] / config.get("schedule_time_scale", 1.0)
    rest = (time >= 0.3) & (time <= 0.5)
    mild = time <= 2.6
    recovery = (time >= 2.2) & (time <= 2.6)
    final = time >= 5.2
    execution = config.get("execution_mode")
    process_steps = completion["frames"] - 1 - max(0, completion.get("resume_frame", 0) - 1)
    graph_replays = completion.get("process_graph_replays")
    if not all(np.any(mask) for mask in (rest, recovery, final)):
        raise ValueError("The capture must contain both loading and recovery cycles")
    aligned_shape = rigid_aligned_shape_rms(directory)
    if len(aligned_shape) != len(time):
        raise ValueError("State capture and metrics have different frame counts")
    width = float(np.mean(data["central_width_y_m"][rest]))
    force = 0.5 * (data["left_force_n"] + data["right_force_n"])
    result = {
        "config": config,
        "provenance": {"composer_sha256": file_sha256(Path(__file__)),
                       **{name: file_sha256(directory / name)
                          for name in ("states.bin", "metrics.csv", "completion.json")}},
        "frames": len(time),
        "execution_mode": execution,
        "process_graph_replays": graph_replays,
        "reference_central_width_m": width,
        "mild_min_central_width_m": float(np.min(data["central_width_y_m"][mild])),
        "mild_width_reduction_m": float(width - np.min(data["central_width_y_m"][mild])),
        "mild_recovery_width_error_fraction": float(abs(np.mean(data["central_width_y_m"][recovery]) / width - 1)),
        "mild_recovery_shape_rms_m": float(np.mean(data["shape_rms_m"][recovery])),
        "mild_rigid_aligned_recovery_rms_m": float(np.mean(aligned_shape[recovery])),
        "mild_alpha_max": float(np.max(data["alpha_max"][mild])),
        "peak_mean_finger_force_n": float(np.max(force)),
        "mild_peak_mean_finger_force_n": float(np.max(force[mild])),
        "mild_bilateral_force_n": float(np.max(np.minimum(data["left_force_n"][mild],
                                                        data["right_force_n"][mild]))),
        "final_finger_force_max_n": float(np.max(np.abs(force[final]))),
        "final_alpha_mean": float(data["alpha_mean"][-1]),
        "final_shape_rms_m": float(np.mean(data["shape_rms_m"][final])),
        "final_shape_rms_range_m": float(np.ptp(data["shape_rms_m"][final])),
        "final_rigid_aligned_shape_rms_m": float(np.mean(aligned_shape[final])),
        "final_rigid_aligned_shape_range_m": float(np.ptp(aligned_shape[final])),
        "shape_metric_scope": "CSV shape RMS removes translation; rigid-aligned RMS also removes the best-fit proper rotation using the same material-particle identities.",
        "final_central_width_m": float(np.mean(data["central_width_y_m"][final])),
        "alpha_increment_min": float(np.min(data["alpha_increment_min"])),
        "det_fp_error_max": float(np.max(data["det_fp_error"])),
        "det_fe_min": float(np.min(data["det_fe_min"])),
        "det_fe_max": float(np.max(data["det_fe_max"])),
        "normal_residual_max_m_s": float(np.max(data["normal_residual_m_s"])),
        "friction_cone_violation_max_ns": float(np.max(data["friction_cone_violation_ns"])),
        "penetration_max_m": float(np.max(data["penetration_m"])),
        "linear_balance_error_max_ns": float(np.max(data["linear_balance_error_ns"])),
        "angular_balance_error_max_nms": float(np.max(data["angular_balance_error_nms"])),
        "grid_contact_peak": int(np.max(data["grid_contact_peak"])),
        "penetration_budget_m": 0.002,
        "plastic_dissipation_j": float(data["plastic_dissipation_j"][-1]),
        "actuator_work_estimate_j": float(data["actuator_work_estimate_j"][-1]),
        "energy_scope": "Material elastic, hardening, plastic dissipation, APIC kinetic and gravity energies; actuator work is a sampled estimate, not a closed energy budget.",
        "angular_balance_scope": config["angular_balance"],
    }
    size = config["specimen_m"]
    result["checks"] = {
        "complete": len(time) == completion["frames"] and time[-1] >= 5.6,
        "finite": all(np.all(np.isfinite(values)) for values in data.values()),
        "elastic_history": result["mild_alpha_max"] <= 2e-6,
        "elastic_loading": result["mild_width_reduction_m"] >= 5e-5 and result["mild_bilateral_force_n"] >= 0.01,
        "elastic_recovery": result["mild_recovery_shape_rms_m"] <= 0.02 * max(size),
        "plastic_history": result["final_alpha_mean"] >= 0.02,
        "residual_shape": result["final_shape_rms_m"] >= 0.01 * min(size),
        "settled_shape": result["final_shape_rms_range_m"] <= 0.01 * max(size),
        "rigid_aligned_residual_shape": result["final_rigid_aligned_shape_rms_m"] >= 0.01 * min(size),
        "rigid_aligned_settled_shape": result["final_rigid_aligned_shape_range_m"] <= 0.01 * max(size),
        "released": result["final_finger_force_max_n"] <= 1e-5,
        "monotonic_history": result["alpha_increment_min"] >= -2e-6,
        "isochoric_plasticity": result["det_fp_error_max"] <= 2e-3,
        "positive_elastic_volume": result["det_fe_min"] > 0,
        "normal_constraint": result["normal_residual_max_m_s"] <= 1e-3,
        "friction_cone": result["friction_cone_violation_max_ns"] <= 1e-10,
        "spatial_contact": result["penetration_max_m"] <= result["penetration_budget_m"],
        "linear_momentum": result["linear_balance_error_max_ns"] <= 1e-5,
        "angular_momentum_sampled_gravity": result["angular_balance_error_max_nms"] <= 1e-5,
        "environment_status": bool(np.all(data["env_status"] == 0)),
        "contact_capacity": bool(np.all(data["grid_contact_overflow"] == 0)),
        "dynamic_control": config["pinned_particles"] == 0 and not config["prescribed_joint_poses"],
        "configured_execution": execution is None or
            (execution == "graph" and graph_replays == process_steps) or
            (execution == "eager" and graph_replays == 0),
        "reset": completion["reset_passed"] and not completion["material_reset_between_loads"],
    }
    result["checks"] = {key: bool(value) for key, value in result["checks"].items()}
    result["passed"] = all(result["checks"].values())
    return result


def compare(base, candidate):
    for key in ("specimen_m", "youngs", "poisson", "density", "yield_stress", "hardening_modulus",
                "gravity_z", "finger_friction", "body_band_m", "finger_targets_m", "control_hz",
                "schedule_time_scale", "execution_mode", "capture_frames"):
        if base["config"].get(key) != candidate["config"].get(key):
            raise ValueError(f"Convergence comparison changed {key}")
    changed = [key for key in ("dx", "dt", "velocity_iterations")
               if base["config"][key] != candidate["config"][key]]
    if len(changed) != 1:
        raise ValueError("Each comparison must change exactly one grid, timestep or iteration parameter")
    result = {
        "parameter": changed[0],
        "candidate_physics_passed": candidate["passed"],
        "shape_difference_m": abs(base["final_shape_rms_m"] - candidate["final_shape_rms_m"]),
        "rigid_aligned_shape_difference_m": abs(base["final_rigid_aligned_shape_rms_m"] -
                                                  candidate["final_rigid_aligned_shape_rms_m"]),
        "central_width_difference_m": abs(base["final_central_width_m"] - candidate["final_central_width_m"]),
        "plastic_history_relative_difference": abs(candidate["final_alpha_mean"] / base["final_alpha_mean"] - 1),
        "peak_force_relative_difference": abs(candidate["peak_mean_finger_force_n"] / base["peak_mean_finger_force_n"] - 1),
        "penetration_difference_m": candidate["penetration_max_m"] - base["penetration_max_m"],
    }
    spatial = changed[0] == "dx"
    result["checks"] = {
        "candidate": candidate["passed"],
        "shape": result["shape_difference_m"] <= (1e-3 if spatial else 2e-4),
        "rigid_aligned_shape": result["rigid_aligned_shape_difference_m"] <= (1e-3 if spatial else 2e-4),
        "width": result["central_width_difference_m"] <= (1e-3 if spatial else 2e-4),
        "plastic_history": result["plastic_history_relative_difference"] <= (0.25 if spatial else 0.05),
        "force": result["peak_force_relative_difference"] <= (0.25 if spatial else 0.05),
    }
    result["passed"] = all(result["checks"].values())
    return result


def plot_metrics(directory, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, data = read_capture(directory)
    time = data["time_s"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for key, label in (("left_force_n", "Left"), ("right_force_n", "Right")):
        axes[0, 0].plot(time, data[key], label=label)
    axes[0, 0].set_ylabel("Finger reaction (N)")
    axes[0, 1].plot(time, 1e3 * data["central_width_y_m"], label="Central section")
    axes[0, 1].plot(time, 1e3 * data["width_y_m"], label="Whole specimen")
    axes[0, 1].set_ylabel("Particle-center width (mm)")
    axes[1, 0].plot(time, data["alpha_mean"], label="Mean equivalent plastic strain")
    axes[1, 0].set_ylabel("Plastic strain")
    for key, label in (("elastic_j", "Elastic"), ("hardening_j", "Hardening"),
                       ("plastic_dissipation_j", "Plastic dissipation"), ("kinetic_j", "APIC kinetic")):
        axes[1, 1].plot(time, 1e3 * data[key], label=label)
    axes[1, 1].set_ylabel("Material energy (mJ)")
    for axis in axes.flat:
        axis.set_xlabel("Simulation time (s)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def encode(directory, frames, output, view, fps):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("ffmpeg and ffprobe must be on PATH")
    config, data = read_capture(directory)
    count = len(data["time_s"])
    size = Image.open(frames / f"{view}_000000.ppm").size
    name = "robot_elastoplastic" + ("_close" if view == "close" else "")
    video = output / f"{name}.mp4"
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(video)]
    force = 0.5 * (data["left_force_n"] + data["right_force_n"])
    peak = max(float(force.max()), 1e-12)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(count):
            frame = Image.open(frames / f"{view}_{index:06d}.ppm").convert("RGB")
            if frame.size != size:
                raise ValueError("Frame dimensions changed")
            draw = ImageDraw.Draw(frame, "RGBA")
            points = [(size[0] * (0.3 + 0.4 * i / (count - 1)),
                       size[1] * (0.975 - 0.05 * max(0, force[i]) / peak)) for i in range(index + 1)]
            if len(points) > 1:
                draw.line(points, fill=(250, 214, 172, 210), width=2)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoding failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    probe = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration", "-of", "json", str(video)], text=True))
    stream = probe["streams"][0]
    if (int(stream["nb_read_frames"]) != count or stream["width"] != size[0] or
            stream["height"] != size[1] or stream["r_frame_rate"] != f"{fps}/1" or
            abs(float(probe["format"]["duration"]) - count / fps) > 0.001):
        raise RuntimeError("Decoded video differs from the captured sequence")
    return dict(video=str(video), frames=count, fps=fps, simulation_hz=config["control_hz"],
                slow_motion_factor=config["control_hz"] / fps, state_interpolation=False,
                overlay="mean bilateral finger reaction history, without labels", verification=probe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--compare", type=Path, action="append", default=[])
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--visual-preview", action="store_true",
                        help="Encode a preview while retaining failed physical checks")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    if args.fps <= 0 or (not args.analyze_only and args.frames is None):
        parser.error("Positive fps and rendered frames are required for video composition")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(args.capture)
    result["comparisons"] = {str(path): compare(result, analyze(path)) for path in args.compare}
    result["passed"] &= all(value["passed"] for value in result["comparisons"].values())
    result["convergence_parameters"] = sorted({value["parameter"] for value in result["comparisons"].values()})
    result["convergence_complete"] = {"dx", "dt", "velocity_iterations"} <= set(result["convergence_parameters"])
    result["physical_acceptance"] = result["passed"] and result["convergence_complete"]
    result["publication_status"] = "visual_preview" if args.visual_preview else "physical_validation"
    (args.out_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_metrics(args.capture, args.out_dir / "physical_curves.png")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"] and not args.visual_preview:
        raise SystemExit("Physical acceptance failed")
    if not args.analyze_only:
        required = ("complete", "finite", "configured_execution", "environment_status", "contact_capacity", "reset")
        if not all(result["checks"][key] for key in required):
            raise SystemExit("Capture integrity checks failed")
        videos = [encode(args.capture, args.frames, args.out_dir, view, args.fps) for view in ("wide", "close")]
        for video in videos:
            video["publication_status"] = result["publication_status"]
            video["physical_acceptance"] = result["physical_acceptance"]
        (args.out_dir / "playback.json").write_text(json.dumps(videos, indent=2) + "\n")


if __name__ == "__main__":
    main()
