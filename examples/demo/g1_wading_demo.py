"""Run a G1 velocity actor on an authored NKS scene and capture physical state and images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image
import torch

import nuka
from nuka.provenance import record_run_identity
from nuka.tasks.g1_locomotion import G1VelocityActor, G1VelocityController, locomotion_outcome, projected_gravity
from nuka.tasks.g1_perception import G1RgbdHistory, ProprioceptionConfig, RgbdConfig
from nuka.tasks.g1_policy import G1FusionPolicy
from nuka.tasks.g1_terrain import TerrainConfig


ROOT = Path(__file__).resolve().parents[2]


def elapsed_time():
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW) if hasattr(time, "CLOCK_MONOTONIC_RAW") else time.perf_counter()


def input_identity(scene, output, extras=()):
    return record_run_identity(ROOT, output, [scene, *extras],
        ["src", "python/src", "python/nuka", "tools/assets", "examples/demo/g1_wading_demo.py"])


def publish_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def publish_state(path, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **fields)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def capture_camera(camera, output, frame):
    packet = camera.latest_capture
    depth = packet["metric_depth"][0, 0].cpu().numpy()
    valid = packet["depth_valid"][0, 0].cpu().numpy()
    normalized = np.where(valid, depth / camera.config.far_clip, 0)
    depth_color = np.stack((1-normalized, 1-np.abs(2*normalized-1), normalized), axis=-1)
    depth_color[~valid] = 0
    if "rgb" in packet:
        rgb = packet["rgb"][0].float().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(output / f"camera_rgb_{frame:05d}.png")
    Image.fromarray(np.uint8(np.clip(depth_color, 0, 1) * 255)).save(output / f"camera_depth_{frame:05d}.png")
    Image.fromarray(valid.astype(np.uint8)*255).save(output / f"camera_valid_{frame:05d}.png")
    observation = camera.observation()
    terrain = {}
    if camera.projector is not None:
        terrain = {"terrain_history": observation["terrain"].cpu().numpy(),
            "captured_terrain": packet["terrain"].cpu().numpy(),
            "camera_pose_in_base": packet["camera_pose_in_base"].cpu().numpy(),
            "estimated_up": packet["estimated_up"].cpu().numpy()}
    np.savez_compressed(output / f"camera_{frame:05d}.npz", depth=depth, valid=valid, **terrain,
        sample_time=packet["sample_time"], available_time=packet["available_time"],
        history_age=observation["age"].cpu().numpy(), frame_valid=observation["frame_valid"].cpu().numpy())


def verify_actor(actor, path, observation):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = [observation[:1].detach().cpu().numpy()]
    generator = np.random.default_rng(20260914)
    inputs.extend(generator.normal(0, 0.3, inputs[0].shape).astype(np.float32) for _ in range(8))
    absolute = []
    with torch.no_grad():
        for value in inputs:
            expected = session.run(None, {session.get_inputs()[0].name: value})[0]
            actual = actor(torch.as_tensor(value, device=observation.device)).cpu().numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            absolute.append(float(np.max(np.abs(actual - expected))))
    return {"max_absolute_error": max(absolute), "inputs": len(inputs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / ".nuka-assets/generated/g1_coupled_course/wading.nks")
    parser.add_argument("--policy", type=Path, default=ROOT / ".nuka-assets/policies/g1_velocity_v0/policy.onnx")
    parser.add_argument("--deployment", type=Path, default=ROOT / ".nuka-assets/policies/g1_velocity_v0/deploy.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--gains", choices=("deployment", "training"), default="deployment")
    parser.add_argument("--execution", choices=("graph", "eager"), default="graph")
    parser.add_argument("--coupling-passes", type=int, default=0,
                        help="Contact exchanges per interval; 0 follows material iteration count")
    # 12 sweeps track the 32-sweep trajectory to 0.7% at 1.94x the speed.
    parser.add_argument("--solver-vel-iters", type=int, default=12,
                        help="velocity sweep budget; 0 keeps the scene's own value")
    parser.add_argument("--command", nargs=3, type=float, default=(0.4, 0.0, 0.0))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--camera", default="follow")
    parser.add_argument("--verify-onnx", action="store_true")
    parser.add_argument("--hold", action="store_true")
    parser.add_argument("--sensors", action=argparse.BooleanOptionalAction, default=None,
                        help="Acquire onboard depth/RGB-D, required by terrain or image policies")
    parser.add_argument("--proprioception", choices=("ideal", "sensors"))
    parser.add_argument("--proprioception-config", type=Path)
    parser.add_argument("--sensor-config", type=Path)
    parser.add_argument("--terrain-config", type=Path)
    parser.add_argument("--manifest", type=Path, help="Course geometry for the same completion checks used in training")
    parser.add_argument("--finish-x", type=float)
    args = parser.parse_args()
    if (args.seconds < 0 or not np.isfinite(args.seconds) or args.frame_stride < 1
            or min(args.width, args.height, args.spp) < 1):
        parser.error("Duration must be finite and nonnegative; capture dimensions and stride must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    trained = args.policy.suffix == ".pt"
    checkpoint = torch.load(args.policy, map_location="cpu", weights_only=True) if trained else {}
    training = checkpoint.get("training", {})
    uses_images = trained and checkpoint["specification"]["fusion"].get("use_camera", True)
    uses_terrain = trained and checkpoint["specification"]["fusion"].get("use_terrain", False)
    if (uses_images or uses_terrain) and args.sensors is False:
        parser.error("This checkpoint requires an onboard depth camera")
    observation_source = args.proprioception or ("sensors" if trained else "ideal")
    if trained and observation_source != "sensors":
        parser.error("The fused policy requires measured proprioception")
    if trained and args.verify_onnx:
        parser.error("ONNX alignment applies to the imported velocity actor")
    task_config = training.get("task", {})
    manifest_path = args.manifest or (Path(task_config["manifest"]) if task_config.get("manifest") else None)
    course = json.loads(manifest_path.read_text())["course"] if manifest_path is not None else {}
    finish_x = args.finish_x if args.finish_x is not None else task_config.get("finish_x")
    if finish_x is None:
        finish_x = course.get("finish_x")
    seed = task_config.get("seed", 20260915)
    proprioception_values = task_config.get("proprioception", {})
    if args.proprioception_config is not None:
        proprioception_values = {**proprioception_values, **json.loads(args.proprioception_config.read_text())}
    proprioception = ProprioceptionConfig(**{"seed": seed, **proprioception_values})
    solver_options = ({"solver_vel_iters": args.solver_vel_iters}
                      if args.solver_vel_iters > 0 else None)
    controller = G1VelocityController(args.scene, args.deployment, num_envs=args.envs,
        dt=args.dt, gains_source=args.gains, execution=args.execution, commands=args.command,
        observation_source=observation_source, proprioception_config=proprioception,
        solver_options=solver_options)
    try:
        controller.world.set_coupling_passes(args.coupling_passes)
        camera = None
        if args.sensors or args.sensor_config or args.terrain_config or uses_images or uses_terrain:
            values = json.loads(args.sensor_config.read_text()) if args.sensor_config else (task_config.get("camera") or {})
            config = RgbdConfig(**{"seed": seed + 1, **values})
            terrain_values = (json.loads(args.terrain_config.read_text()) if args.terrain_config
                              else task_config.get("terrain"))
            camera = G1RgbdHistory(controller, config, TerrainConfig(**terrain_values)
                                   if terrain_values is not None else None)
            (args.out / "camera.json").write_text(json.dumps(camera.metadata(), indent=2), encoding="utf-8")
        actor = (G1FusionPolicy.load(args.policy, device=controller.device)[0] if trained
                 else G1VelocityActor.from_onnx(args.policy, device=controller.device)).eval()
        if (uses_images or uses_terrain) and camera.config.history_length != actor.config.history_length:
            raise ValueError("Camera history differs from the trained policy")
        if uses_images and not camera.config.image_observation:
            raise ValueError("An image policy requires RGB-D observations")
        if uses_terrain and camera.projector is None:
            raise ValueError("A terrain policy requires measured terrain projection")
        input_identity(args.scene, args.out,
            [args.policy, args.deployment, args.sensor_config, args.proprioception_config, args.terrain_config, manifest_path])
        observation = controller.observation()
        report = {"scene": str(args.scene), "policy": str(args.policy), "gains_source": args.gains,
            "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
            "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
            "dt": args.dt, "policy_dt": controller.contract.step_dt, "envs": args.envs,
            "command": args.command, "particles": controller.world.particle_count,
            "coupling_passes": args.coupling_passes,
            "solver_vel_iters": args.solver_vel_iters,
            "execution": args.execution, "observation_source": observation_source,
            "camera_enabled": camera is not None, "policy_uses_images": uses_images,
            "policy_uses_terrain": uses_terrain,
            "recording_camera": args.camera,
            "host_clock": "CLOCK_MONOTONIC_RAW" if hasattr(time, "CLOCK_MONOTONIC_RAW") else "perf_counter"}
        if controller.proprioception is not None:
            report["proprioception"] = controller.proprioception.metadata()
        if args.verify_onnx:
            report["onnx_alignment"] = verify_actor(actor, args.policy, observation)
        timings = {"control_and_physics": 0.0, "beauty_render": 0.0,
                   "sensor_acquisition": 0.0, "evidence_write": 0.0}
        start = elapsed_time()
        rows, positions, joints = [], [], []
        diagnostics = {}
        start_position = controller.base[:, :3].clone()
        finish_streak = torch.zeros(args.envs, device=controller.device, dtype=torch.int64)
        finish_hold_seconds = task_config.get("finish_hold_seconds", 0.6)
        finish_hold_steps = math.ceil(finish_hold_seconds / controller.contract.step_dt)
        captured = 0
        state_directory = args.out / "steps"
        state_directory.mkdir()
        steps = round(args.seconds / controller.contract.step_dt)

        def progress(phase, step, **extra):
            publish_json(args.out / "progress.json", {
                "phase": phase, "completed_step": step,
                "simulation_time": step * controller.contract.step_dt,
                "requested_steps": steps, "wall_seconds": elapsed_time() - start,
                "completion_seconds": timings.copy(), **extra})

        for step in range(steps + 1):
            controller.world.synchronize()
            base = controller.base.detach().cpu().numpy().copy()
            q = controller.q.detach().cpu().numpy().copy()
            qd = controller.qd.detach().cpu().numpy()
            velocity = controller.velocity[:, 0].detach().cpu().numpy()
            up = -projected_gravity(controller.base[:, 3:])[:, 2].cpu().numpy()
            flags = controller.status.cpu().numpy().astype(np.uint32)
            finite = bool(np.isfinite(base).all() and np.isfinite(q).all() and np.isfinite(qd).all())
            fallen, off_course, reached = [value.cpu().numpy() for value in
                locomotion_outcome(controller, start_position, finish_x=finish_x, course_width=course.get("width"),
                    finish_speed_tolerance=task_config.get("finish_speed_tolerance", 0.25),
                    finish_streak=finish_streak, finish_hold_steps=finish_hold_steps)]
            row = {"step": step, "time": step * controller.contract.step_dt,
                "base": base.tolist(), "up": up.tolist(), "base_velocity": velocity.tolist(),
                "qvel_abs_max": float(np.abs(qd).max()), "flags": flags.tolist(), "finite": finite,
                "fallen": fallen.tolist(),
                "off_course": off_course.tolist(), "reached": reached.tolist(),
                "finish_hold_seconds": (finish_streak * controller.contract.step_dt).cpu().tolist(),
                "motor_effort_abs_max": float(controller.effort.abs().max().cpu()),
                "saturated_motor_count": int(controller.saturated.to(torch.int32).sum().cpu())}
            measured = controller.measurements()
            if os.environ.get("NUKA_CONTACT_SOLVER_DIAGNOSTICS", "").startswith("1"):
                packed = np.asarray(controller.world.download_field(nuka.Field.CONTACT_SOLVE_METRICS),
                                    dtype=np.uint64).reshape(-1, 8)
                error_bits = (packed >> np.uint64(32)).astype(np.uint32)
                row["contact_solver"] = {
                    "scope": "latest contact solve before integration; contact blocks only",
                    "metrics": ["normal_velocity", "tangent_velocity", "normal_velocity_violation",
                                "normal_impulse_violation", "complementarity_work", "friction_impulse_violation",
                                "friction_work_violation", "friction_dissipation_work"],
                    "units": ["m/s", "m/s", "m/s", "N*s", "J", "N*s", "J", "J"],
                    "maximum": error_bits.view(np.float32).tolist(),
                    "global_row": (~packed.astype(np.uint32)).tolist(),
                    "counts": np.asarray(controller.world.download_field(nuka.Field.CONTACT_SOLVE_COUNTS),
                                         dtype=np.uint32).reshape(-1, 2).tolist(),
                }
            truth_gravity = projected_gravity(controller.base[:, 3:])
            dot = torch.nn.functional.cosine_similarity(measured["projected_gravity"], truth_gravity, dim=-1)
            row["attitude_error_degrees"] = torch.rad2deg(dot.clamp(-1, 1).acos()).cpu().tolist()
            values = {"time": row["time"], "qvel": qd.copy(),
                "link_velocity": controller.velocity,
                "foot_pose": controller.link_pose[:, controller.foot_slots],
                "contact_wrench": controller.contact_wrench[:, controller.foot_slots],
                "action": controller.last_action, "drive_target": controller.drive_target,
                "effort": controller.effort, "effort_requested": controller.effort_requested,
                "saturated": controller.saturated, "gravity_measured": measured["projected_gravity"],
                "gravity_truth": truth_gravity, "angular_velocity_measured": measured["base_ang_vel"]}
            if controller.proprioception is not None:
                values.update(specific_force=controller.proprioception.specific_force,
                              foot_wrench_measured=controller.proprioception.foot_wrench,
                              motor_effort_measured=controller.proprioception.motor_effort)
            snapshot = {"step": step, "base": base, "q": q}
            for name, value in values.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy().copy()
                diagnostics.setdefault(name, []).append(value)
                snapshot[name] = value
            evidence_start = elapsed_time()
            publish_state(state_directory / f"step_{step:07d}.npz", snapshot)
            with (args.out / "samples.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**row, "completion_seconds": timings.copy()}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            timings["evidence_write"] += elapsed_time() - evidence_start
            progress("state_saved", step, state=f"steps/step_{step:07d}.npz",
                     state_contract="Diagnostic readout, not a restart checkpoint")
            rows.append(row)
            positions.append(base)
            joints.append(q)
            if step % 25 == 0:
                print(row, flush=True)
            failed = not finite or np.any(flags) or np.any(fallen) or np.any(off_course)
            if failed:
                report["failure"] = row
            if step % args.frame_stride == 0 or failed or step == steps:
                if args.capture:
                    progress("capturing_state", step)
                    fields = {"time": row["time"], "base": base, "q": q, "qvel": qd,
                        "link_pose": controller.link_pose.detach().cpu().numpy(),
                        "link_velocity": controller.velocity.detach().cpu().numpy()}
                    if controller.world.particle_count:
                        fields["particles"] = np.asarray(controller.world.download_field(nuka.Field.PARTICLE_POSITION))
                    publish_state(args.out / f"state_{captured:05d}.npz", fields)
                if args.render:
                    progress("rendering", step)
                    render_start = elapsed_time()
                    rgb = controller.world.render_beauty(camera=args.camera,
                        width=args.width, height=args.height, spp=args.spp)
                    Image.fromarray(np.asarray(rgb)).save(args.out / f"frame_{captured:05d}.png")
                    timings["beauty_render"] += elapsed_time() - render_start
                if camera is not None and (args.capture or args.render):
                    progress("saving_camera", step)
                    capture_camera(camera, args.out, captured)
                with (args.out / "captures.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"frame": captured, "step": step, "time": row["time"],
                        "state": args.capture, "beauty": args.render,
                        "camera": camera is not None and (args.capture or args.render)}) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                captured += 1
            if step == steps or failed or np.all(reached):
                break
            with torch.no_grad():
                inputs = {"proprio": observation, "tactile": controller.proprioception.extra_observation(),
                          **(camera.observation() if uses_images or uses_terrain else {})} if trained else observation
                action = torch.zeros_like(controller.last_action) if args.hold else actor(inputs)
            progress("physics_in_flight", step, pending_step=step + 1)
            physics_start = elapsed_time()
            observation = controller.step(action)
            controller.world.synchronize()
            timings["control_and_physics"] += elapsed_time() - physics_start
            if camera is not None:
                progress("acquiring_sensors", step, pending_step=step + 1)
                sensor_start = elapsed_time()
                camera.advance()
                controller.world.synchronize()
                timings["sensor_acquisition"] += elapsed_time() - sensor_start
        report["wall_seconds"] = elapsed_time() - start
        report["completion_seconds"] = timings
        report["steps"] = len(rows) - 1
        report["displacement"] = (positions[-1][:, :3] - positions[0][:, :3]).tolist()
        report["minimum_upright"] = np.min([row["up"] for row in rows], axis=0).tolist()
        report["maximum_joint_speed"] = max(row["qvel_abs_max"] for row in rows)
        report["finish_x"] = finish_x
        report["required_finish_hold_seconds"] = finish_hold_seconds
        if finish_x is not None:
            report["completed"] = reached.tolist()
        report["execution_info"] = controller.world.execution_info
        report["joint_names"] = controller.contract.joint_names
        report["joint_slots"] = controller.joint_slots.cpu().tolist()
        report["foot_slots"] = controller.foot_slots.cpu().tolist()
        np.savez_compressed(args.out / "trajectory.npz", base=np.asarray(positions), q=np.asarray(joints),
                            **{name: np.asarray(values) for name, values in diagnostics.items()})
        (args.out / "metrics.json").write_text(json.dumps({**report, "samples": rows}, indent=2), encoding="utf-8")
        progress("rollout_finished", len(rows) - 1,
                 traversal_completed=report.get("completed"), failure=report.get("failure"))
        print(report, flush=True)
        if args.render and shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                str(1 / (controller.contract.step_dt * args.frame_stride)), "-i",
                str(args.out / "frame_%05d.png"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                str(args.out / "preview.mp4")], check=True)
        if "failure" in report:
            raise RuntimeError("The rollout fell, left the course, or reported invalid state; see metrics.json")
        if finish_x is not None and not all(report["completed"]):
            raise RuntimeError("The rollout did not reach the requested finish line; see metrics.json")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
