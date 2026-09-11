"""Validate the rigid-mesh impact and encode its unlabelled material demonstration."""

import argparse
import json
from pathlib import Path

import numpy as np

from compose_elastoplastic_compression import Composer, encode_video, read_capture


def analyze(directory):
    config, data = read_capture(directory)
    completion = json.loads((directory / "completion.json").read_text())
    time = data["time_s"]
    contact = np.flatnonzero(abs(data["contact_force_n"]) > 1e-3)
    if not len(contact):
        raise ValueError("The impactor never contacts the material")
    before = np.arange(len(time)) < contact[0]
    final = time >= time[-1]-0.25
    z0 = data["bunny_com_z_m"][0]
    gravity = config["gravity_z"]
    momentum_error = np.column_stack([data[f"momentum_{axis}_ns"] -
                                     data[f"external_impulse_{axis}_ns"] for axis in "xyz"])
    energy = sum(data[key] for key in ("elastic_j", "hardening_j", "kinetic_j", "gravity_j"))
    impact_energy = config["bunny_mass_kg"] * abs(gravity) * config["drop_height_m"]
    combined_energy = energy + data["plastic_dissipation_j"]
    center = float(np.mean(data["center_top_m"][final]))
    result = {
        "config": config, "first_contact_s": float(time[contact[0]]),
        "precontact_alpha_max": float(np.max(data["alpha_max"][before])),
        "freefall_position_error_m": float(np.max(abs(data["bunny_com_z_m"][before] -
                                                        (z0+0.5*gravity*time[before]**2)))),
        "freefall_velocity_error_m_s": float(np.max(abs(data["bunny_vz_m_s"][before]-gravity*time[before]))),
        "final_center_top_m": center,
        "center_depression_m": float(data["center_top_m"][0]-center),
        "final_bunny_bottom_m": float(np.mean(data["bunny_bottom_m"][final])),
        "bunny_indentation_m": float(data["pad_top_m"][0]-np.mean(data["bunny_bottom_m"][final])),
        "rim_rise_m": float(np.max(data["rim_top_m"])-data["rim_top_m"][0]),
        "penetration_max_m": float(np.max(data["penetration_m"])),
        "final_alpha_mean": float(data["alpha_mean"][-1]),
        "final_alpha_max": float(data["alpha_max"][-1]),
        "alpha_increment_min": float(np.min(data["alpha_increment_min"])),
        "det_fp_error_max": float(np.max(data["det_fp_error"])),
        "det_fe_min": float(np.min(data["det_fe_min"])),
        "peak_force_n": float(np.max(data["contact_force_n"])),
        "settled_force_n": float(np.mean(data["contact_force_n"][final])),
        "final_contact_impulse_z_ns": float(data["contact_impulse_z_ns"][-1]),
        "plastic_dissipation_j": float(data["plastic_dissipation_j"][-1]),
        "energy_increase_max_j": float(np.max(combined_energy-combined_energy[0])),
        "initial_impact_energy_j": impact_energy,
        "unresolved_energy_loss_j": float(combined_energy[0]-combined_energy[-1]),
        "momentum_balance_error_max_ns": float(np.max(np.linalg.norm(momentum_error, axis=1))),
        "final_bunny_speed_m_s": float(data["bunny_speed_m_s"][-1]),
        "final_bunny_angular_speed_rad_s": float(data["bunny_angular_speed_rad_s"][-1]),
        "reset_passed": completion["reset_passed"], "unloaded_indentation": False,
    }
    result["checks"] = {
        "complete_capture": len(time) == config["frames"] == completion["frames"],
        "finite_state": all(np.all(np.isfinite(values)) for values in data.values()),
        "environment_status": bool(np.all(data["env_status"] == 0)),
        "elastic_before_contact": result["precontact_alpha_max"] <= 2e-6,
        "freefall_position": result["freefall_position_error_m"] <= 1e-3,
        "freefall_velocity": result["freefall_velocity_error_m_s"] <= 0.01,
        "plastic_history": result["final_alpha_mean"] >= 0.002 and result["final_alpha_max"] >= 0.05,
        "center_depression": result["center_depression_m"] >= 0.008,
        "bunny_indentation": result["bunny_indentation_m"] >= 0.005,
        "contact_resolution": result["penetration_max_m"] <= config["dx"],
        "monotonic_history": result["alpha_increment_min"] >= -2e-6,
        "isochoric_plasticity": result["det_fp_error_max"] <= 2e-3,
        "positive_elastic_volume": result["det_fe_min"] > 0,
        "linear_momentum": result["momentum_balance_error_max_ns"] <= 0.02,
        "nonincreasing_energy": result["energy_increase_max_j"] <= 0.05*impact_energy,
        "reset": completion["reset_passed"],
        "free_rigid_motion": not completion["body_pose_written_after_release"],
        "continuous_material_history": not completion["material_reset_during_capture"],
    }
    if "external_angular_impulse_x" in data:
        angular_error = np.column_stack([data[f"angular_momentum_{axis}"] -
                                        data[f"external_angular_impulse_{axis}"] for axis in "xyz"])
        result["angular_momentum_balance_error_max_nms"] = float(np.max(np.linalg.norm(angular_error, axis=1)))
        scale = config["bunny_mass_kg"] * np.sqrt(2*abs(gravity)*config["drop_height_m"]) * config["source_asset"]["extent_m"]
        result["checks"]["angular_momentum"] = bool(result["angular_momentum_balance_error_max_nms"] <= 0.005*scale)
    result["passed"] = all(result["checks"].values())
    return result


def compare(base, candidate):
    for key in ("bunny_mass_kg", "drop_height_m", "specimen_m", "youngs", "poisson", "density",
                "yield_stress", "hardening_modulus", "gravity_z", "friction", "frames", "sample_hz", "base_z"):
        if base["config"][key] != candidate["config"][key]:
            raise ValueError(f"Convergence comparison changed {key}")
    if base["config"]["source_asset"]["output_sha256"] != candidate["config"]["source_asset"]["output_sha256"]:
        raise ValueError("Convergence comparison changed the rigid geometry")
    same_grid = base["config"]["dx"] == candidate["config"]["dx"]
    if same_grid and base["config"]["body_contact_band"] != candidate["config"]["body_contact_band"]:
        raise ValueError("Timestep comparison changed the contact envelope")
    if same_grid and not np.isclose(candidate["config"]["dt"] * 2, base["config"]["dt"], rtol=1e-6, atol=0):
        raise ValueError("The timestep comparison must halve dt")
    result = {
        "kind": "timestep" if same_grid else "grid",
        "candidate_physics_passed": candidate["passed"],
        "center_depth_difference_m": abs(base["final_center_top_m"]-candidate["final_center_top_m"]),
        "plastic_strain_relative_difference": abs(candidate["final_alpha_mean"]/base["final_alpha_mean"]-1),
        "dissipation_relative_difference": abs(candidate["plastic_dissipation_j"]/base["plastic_dissipation_j"]-1),
        "impulse_relative_difference": abs(candidate["final_contact_impulse_z_ns"]/base["final_contact_impulse_z_ns"]-1),
    }
    if same_grid:
        result["timestep_checks_passed"] = (result["center_depth_difference_m"] <= 0.0025 and
            result["plastic_strain_relative_difference"] <= 0.05 and result["dissipation_relative_difference"] <= 0.05)
    return result


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
        raise RuntimeError("The material impact does not pass its frozen checks")
    args.frames = args.frames or args.capture / "frames"
    composer = Composer(args, result)
    composer.data["top_force_n"] = composer.data["contact_force_n"]
    if args.preview:
        for path in sorted(args.frames.glob("frame_*.ppm")):
            index = int(path.stem.split("_")[-1])
            composer.frame(index).save(args.out_dir / f"frame_{index:06d}.png")
        return 0
    end = len(composer.data["time_s"])-1
    encode_video(composer, args, "elastoplastic_bunny", (0, 24, 36, 48, 120, end))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
