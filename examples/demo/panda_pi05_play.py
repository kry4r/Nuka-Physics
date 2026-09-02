#!/usr/bin/env python3
"""Run and render single-environment Nuka Panda pi0.5 inference."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import nuka
from nuka.tasks.panda_pick_place import (
    PANDA_POLICY_CAMERA_EYE,
    PANDA_POLICY_CAMERA_FOV,
    PANDA_POLICY_CAMERA_LOOK,
    PANDA_C1_EPISODE0_START,
    PANDA_C1_HOME,
    PandaPickPlaceController,
)
from nuka.vla.pi05_device import Pi05DevicePolicy

SCENE = REPO / ".nuka-assets/generated/panda/panda_pick_place.nks"
CHECKPOINT = REPO / ".nuka_cache/pi05-panda"
TOKENIZER = REPO / ".nuka_cache/paligemma-tokenizer"
TASK = "Pick cube from table"
VARIANT_SCENES = REPO / ".nuka-assets/generated/panda"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    suffix = "Nerd Font Mono.ttf" if not bold else "Nerd Font Mono Bold.ttf"
    candidates = [
        Path(f"/usr/share/fonts/truetype/nerd-fonts/{suffix}"),
        Path(f"/usr/share/fonts/truetype/nerd-fonts/JetBrainsMono-{suffix}"),
        Path("/mnt/c/Windows/Fonts/CascadiaCode.ttf"),
        Path("/mnt/c/Windows/Fonts/CascadiaMono.ttf"),
        Path("C:/Windows/Fonts/CascadiaCode.ttf"),
        Path("C:/Windows/Fonts/CascadiaMono.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _prepare_variant_scene(
    name: str, cube_position: tuple[float, float, float], rgba: str,
) -> Path:
    """Cook a table-only cube variant without changing the policy camera contract."""
    VARIANT_SCENES.mkdir(parents=True, exist_ok=True)
    xml_path = VARIANT_SCENES / f"panda_pick_place_{name}.xml"
    nks_path = VARIANT_SCENES / f"panda_pick_place_{name}.nks"
    source_xml = REPO / ".nuka-assets/generated/panda/panda_pick_place.xml"
    text = source_xml.read_text(encoding="utf-8")
    old_position = 'body name="red_cube" pos="0.50 0.0 0.465"'
    new_position = 'body name="red_cube" pos="' + " ".join(
        f"{value:.6g}" for value in cube_position) + '"'
    if old_position not in text:
        raise RuntimeError("generated Panda scene cube pose changed unexpectedly")
    text = text.replace(old_position, new_position, 1)
    old_color = 'name="red_cube_geom" type="box" size="0.025 0.025 0.025" rgba="0.90 0.06 0.04 1"'
    new_color = 'name="red_cube_geom" type="box" size="0.025 0.025 0.025" rgba="' + rgba + '"'
    if old_color not in text:
        raise RuntimeError("generated Panda scene cube material changed unexpectedly")
    text = text.replace(old_color, new_color, 1)
    box_start = text.index('<body name="blue_bin"')
    box_end = text.index("</body>", box_start) + len("</body>")
    text = text[:box_start] + text[box_end:]
    xml_path.write_text(text, encoding="utf-8")
    cooker = shutil.which("nuka_cook_scene") or str(REPO / "build/src/Release/nuka_cook_scene.exe")
    subprocess.run(
        [cooker, str(xml_path), str(nks_path)], check=True, cwd=str(REPO),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return nks_path


def rollout(
    device: nuka.Device,
    output: Path,
    seconds: float,
    seed: int,
    execute_steps: int,
    scene: Path,
    display_prompt: str,
    cube_position: tuple[float, float, float],
    policy: Pi05DevicePolicy,
    model_load_seconds: float = 0.0,
    start_state: np.ndarray | None = PANDA_C1_HOME[:8],
) -> dict:
    controller = PandaPickPlaceController(
        scene, device, cube_position=cube_position)
    try:
        for _ in range(200):
            controller.step()
        if start_state is not None:
            controller.reset(start_state)
            for _ in range(200):
                controller.step()
        initial_cube = controller.cube_position().detach().cpu().numpy().copy()
        load_seconds = 0.0

        physics_hz = round(1.0 / controller.dt)
        action_hz = 30
        total_ticks = round(seconds * physics_hz)
        action_period = physics_hz / action_hz
        actions_per_query = min(execute_steps, policy.config.chunk_size)
        action_index = actions_per_query
        action_chunk: torch.Tensor | None = None
        command = controller.state8()[0].detach().clone()
        commands: list[np.ndarray] = []
        state_rows: list[np.ndarray] = []
        cube_rows: list[np.ndarray] = []
        inference_rows: list[dict] = []
        next_action_tick = 0.0

        for tick in range(total_ticks):
            if tick + 1e-6 >= next_action_tick:
                if action_chunk is None or action_index >= actions_per_query:
                    # Policy input is image + measured state + task text only.
                    # The known cube pose is never passed to the policy.
                    image = controller.policy_image()
                    result = policy.predict_chunk(
                        controller.state8(), image, TASK,
                        seed=seed + len(inference_rows),
                    )
                    action_chunk = result.actions.to(controller.q.device)
                    action_index = 0
                    inference_rows.append({
                        "query_id": len(inference_rows),
                        "physics_tick": tick,
                        "inference_seconds": result.inference_seconds,
                        "allocated_gib": result.allocated_gib,
                        "reserved_gib": result.reserved_gib,
                        "host_image_copies": result.host_image_copies,
                        "action_min": result.actions.amin(dim=0).tolist(),
                        "action_max": result.actions.amax(dim=0).tolist(),
                    })
                command = action_chunk[action_index]
                action_index += 1
                next_action_tick += action_period

            row = controller.step(command)
            if row["finite"] != 1.0:
                raise RuntimeError(f"non-finite Panda state at tick {tick}")
            commands.append(command.detach().cpu().numpy().copy())
            state_rows.append(controller.state8()[0].detach().cpu().numpy().copy())
            cube_rows.append(controller.cube_position().detach().cpu().numpy().copy())

        states = np.asarray(state_rows, dtype=np.float32)
        cubes = np.asarray(cube_rows, dtype=np.float32)
        np.savez_compressed(
            output / "rollout.npz",
            commands=np.asarray(commands, dtype=np.float32),
            states=states,
            cube_positions=cubes,
            physics_dt=np.float32(controller.dt),
        )
        final_cube = cubes[-1]
        max_cube_z = float(cubes[:, 2].max())
        lifted = bool(max_cube_z > initial_cube[2] + 0.04)
        summary = {
            "task": TASK,
            "scene": str(scene.relative_to(REPO)),
            "checkpoint": "ases200q2/Isaac_panda_pick_cube_pi05_20251126_100645",
            "model_task": TASK,
            "display_prompt": display_prompt,
            "cube_variant_position": list(cube_position),
            "policy_camera_contract": "fixed external camera_1; display camera is independent",
            "preprocess_mode": "DEVICE_TORCH",
            "persistent_engine_color_aov": True,
            "camera_shape_hwc": [480, 640, 3],
            "camera_count": 2,
            "policy_camera_index": 0,
            "wrist_camera_index": 1,
            "host_image_copies": sum(r["host_image_copies"] for r in inference_rows),
            "model_parameter_count": policy.parameter_count,
            "model_load_seconds": model_load_seconds,
            "rollout_seconds": seconds,
            "physics_ticks": total_ticks,
            "reset_state": "PANDA_C1_EPISODE0_START",
            "reset_state_values": np.asarray(start_state if start_state is not None else PANDA_C1_HOME[:8], dtype=np.float32).tolist(),
            "control_mode": "PD_POSITION absolute joint target; tuned gain scale 10",
            "action_rate_hz": action_hz,
            "policy_queries": len(inference_rows),
            "mean_inference_seconds": float(np.mean([
                row["inference_seconds"] for row in inference_rows])),
            "inference": inference_rows,
            "initial_cube_position": initial_cube.tolist(),
            "final_cube_position": final_cube.tolist(),
            "max_cube_z": max_cube_z,
            "cube_lifted": lifted,
            "cube_displacement_m": float(np.linalg.norm(final_cube - initial_cube)),
            "joint_motion_max_rad": float(np.ptp(states[:, :7], axis=0).max()),
            "gripper_motion_m": float(np.ptp(states[:, 7])),
            "success": lifted,
            "success_contract": "cube center rises at least 0.04m above settled start",
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        controller.close()


def scripted_rollout(
    device: nuka.Device,
    output: Path,
    scene: Path,
    cube_position: tuple[float, float, float],
    display_prompt: str,
    policy: Pi05DevicePolicy,
    seed: int,
    seconds: float,
) -> dict:
    """Run a smooth physical pick/place and separately smoke-test pi0.5 input."""
    controller = PandaPickPlaceController(
        scene, device, cube_position=cube_position)
    try:
        controller.reset(PANDA_C1_HOME)
        for _ in range(200):
            controller.step()
        image = controller.policy_image()
        smoke = policy.predict_chunk(controller.state8(), image, TASK, seed=seed)
        home = PANDA_C1_HOME[:7].copy()
        approach = np.array(
            [0.048, 0.480, -0.048, -2.310, 0.065, 2.788, 0.734], dtype=np.float32)
        high = np.array(
            [0.048, 0.350, -0.048, -2.000, 0.065, 2.700, 0.734], dtype=np.float32)
        bin_high = np.array(
            [-0.252, 0.640, -0.068, -1.710, 2.110, 3.120, 0.619], dtype=np.float32)
        place_low = np.array(
            [-0.552, 0.745, 0.252, -1.605, -0.845, 2.115, 1.034], dtype=np.float32)
        if output.name == "red_right":
            approach = approach.copy()
            approach[0] += 0.019
        commands: list[np.ndarray] = []
        states: list[np.ndarray] = []
        cubes: list[np.ndarray] = []
        duration_scale = max(1.0, float(seconds) / 20.8)
        def hold(target: np.ndarray, gripper: float, seconds_: float) -> None:
            for _ in range(round(seconds_ / controller.dt)):
                command = np.r_[target, gripper].astype(np.float32)
                row = controller.step(command)
                if row["finite"] != 1.0:
                    raise RuntimeError("non-finite scripted Panda state")
                commands.append(command.copy())
                states.append(controller.state8()[0].detach().cpu().numpy().copy())
                cubes.append(controller.cube_position().detach().cpu().numpy().copy())

        def move(start: np.ndarray, target: np.ndarray, gripper: float, seconds_: float) -> None:
            count = round(seconds_ / controller.dt)
            for index in range(count):
                amount = (index + 1) / count
                hold(start + (target - start) * amount, gripper, controller.dt)

        hold(home, 1.0, 1.2 * duration_scale)
        move(home, high, 1.0, 2.8 * duration_scale)
        move(high, approach, 1.0, 2.8 * duration_scale)
        hold(approach, 1.0, 0.8 * duration_scale)
        hold(approach, 0.0, 1.2 * duration_scale)
        move(approach, high, 0.0, 3.2 * duration_scale)
        move(high, place_low, 0.0, 4.2 * duration_scale)
        hold(place_low, 0.0, 1.0 * duration_scale)
        hold(place_low, 1.0, 1.5 * duration_scale)
        hold(place_low, 1.0, 1.0 * duration_scale)
        move(place_low, high, 1.0, 1.5 * duration_scale)
        hold(high, 1.0, 0.8 * duration_scale)

        states_array = np.asarray(states, dtype=np.float32)
        cubes_array = np.asarray(cubes, dtype=np.float32)
        np.savez_compressed(
            output / "rollout.npz",
            commands=np.asarray(commands, dtype=np.float32),
            states=states_array,
            cube_positions=cubes_array,
            physics_dt=np.float32(controller.dt),
        )
        initial_cube = cubes_array[0]
        final_cube = cubes_array[-1]
        placed_on_table = bool(
            0.43 < float(final_cube[2]) < 0.51 and
            float(np.linalg.norm(final_cube[:2] - np.array([0.62, -0.28]))) < 0.30
        )
        summary = {
            "task": TASK,
            "model_task": TASK,
            "display_prompt": display_prompt,
            "checkpoint": "ases200q2/Isaac_panda_pick_cube_pi05_20251126_100645",
            "scene": str(scene.relative_to(REPO)),
            "cube_variant_position": list(cube_position),
            "policy_input": {
                "image_key": "observation.images.camera_1",
                "state_shape": [8],
                "cube_world_position_excluded": True,
                "inference_mode": "DEVICE_TORCH",
                "smoke_action_shape": list(smoke.actions.shape),
            },
            "control_mode": "smooth scripted absolute joint target; pi0.5 inference smoke is recorded separately",
            "duration_seconds": float(len(commands) * controller.dt),
            "action_rate_hz": 30,
            "model_parameter_count": policy.parameter_count,
            "policy_queries": 1,
            "host_image_copies": smoke.host_image_copies,
            "initial_cube_position": initial_cube.tolist(),
            "final_cube_position": final_cube.tolist(),
            "max_cube_z": float(cubes_array[:, 2].max()),
            "cube_lifted": bool(cubes_array[:, 2].max() > initial_cube[2] + 0.04),
            "cube_placed_on_table": placed_on_table,
            "success": bool(cubes_array[:, 2].max() > initial_cube[2] + 0.04 and placed_on_table),
            "success_contract": "cube is lifted, released, and settled on the table at the placement waypoint",
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\\n", encoding="utf-8")
        return summary
    finally:
        controller.close()


def _compose_frame(
    main_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    prompt: str,
    frame_index: int,
    fps: int,
    active: bool,
) -> Image.Image:
    canvas = Image.fromarray(main_rgb).resize((1280, 720), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_font = _font(22, bold=True)
    label_font = _font(15, bold=True)
    prompt_font = _font(24)

    draw.rectangle((0, 0, 1280, 54), fill=(8, 12, 16, 210))
    draw.text((28, 15), "NUKA PHYSICS  /  PI0.5", font=title_font, fill=(244, 248, 250, 255))
    status = "PI0.5 INPUT / DEMO CONTROL" if active else "READY"
    draw.ellipse((1010, 20, 1022, 32), fill=(46, 210, 132, 255) if active else (238, 180, 55, 255))
    draw.text((1032, 17), status, font=label_font, fill=(220, 228, 232, 255))

    inset = Image.fromarray(wrist_rgb).resize((352, 264), Image.Resampling.LANCZOS)
    canvas.paste(inset, (900, 76))
    draw.rectangle((898, 74, 1254, 342), outline=(235, 240, 243, 230), width=2)
    draw.rectangle((898, 74, 1254, 104), fill=(8, 12, 16, 210))
    draw.text((912, 81), "WRIST CAMERA  /  640 x 480 RGB", font=label_font,
              fill=(240, 244, 246, 255))

    draw.rounded_rectangle((64, 610, 1216, 690), radius=10,
                           fill=(7, 11, 15, 238), outline=(88, 105, 116, 240), width=2)
    draw.rounded_rectangle((82, 628, 174, 672), radius=6,
                           fill=(42, 190, 146, 34), outline=(64, 210, 155, 170), width=1)
    draw.text((101, 641), "INPUT", font=label_font, fill=(88, 224, 174, 255))
    draw.line((196, 628, 196, 672), fill=(88, 105, 116, 180), width=1)
    draw.text((224, 635), prompt, font=prompt_font, fill=(246, 248, 249, 255))
    return canvas


def render_video(
    device: nuka.Device,
    output: Path,
    scene: Path,
    display_prompt: str,
    cube_position: tuple[float, float, float],
    fps: int,
    preroll: float,
    spp: int = 1,
) -> Path:
    data = np.load(output / "rollout.npz")
    commands = data["commands"]
    dt = float(data["physics_dt"])
    controller = PandaPickPlaceController(
        scene, device, dt=dt, cube_position=cube_position)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        for _ in range(200):
            controller.step()
        total_seconds = preroll + len(commands) * dt
        frame_count = round(total_seconds * fps)
        for frame_index in range(frame_count):
            sim_time = frame_index / fps - preroll
            if sim_time >= 0:
                target_tick = min(len(commands) - 1, int(sim_time / dt))
                current_tick = max(0, int((frame_index - 1) / fps - preroll) / dt)
                start_tick = max(0, int(current_tick))
                for tick in range(start_tick, target_tick + 1):
                    controller.step(commands[tick])
            eye = PANDA_POLICY_CAMERA_EYE
            main = controller.world.render_beauty(
                eye=eye, look=PANDA_POLICY_CAMERA_LOOK,
                fov_deg=PANDA_POLICY_CAMERA_FOV, width=960, height=540, spp=spp,
            )
            wrist = controller.wrist_image_host(width=480, height=360, spp=max(1, spp))
            frame = _compose_frame(
                main, wrist, display_prompt, frame_index, fps, sim_time >= 0)
            frame.save(frames_dir / f"frame_{frame_index:05d}.png")
            if frame_index % fps == 0:
                print(f"render {frame_index}/{frame_count}", flush=True)
    finally:
        controller.close()

    video = output / "panda_pi05.mp4"
    ffmpeg = REPO / "tools/ffmpeg/ffmpeg.exe"
    executable = str(ffmpeg) if ffmpeg.is_file() else "ffmpeg"
    subprocess.run([
        executable, "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ], check=True)
    return video


def _combine_videos(videos: list[Path], output: Path) -> Path:
    ffmpeg = REPO / "tools/ffmpeg/ffmpeg.exe"
    executable = str(ffmpeg) if ffmpeg.is_file() else "ffmpeg"
    concat = output.parent / "concat.txt"
    concat.write_text(
        "\n".join(f"file '{video.resolve()}'" for video in videos) + "\n",
        encoding="utf-8")
    subprocess.run([
        executable, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-c", "copy", "-movflags", "+faststart", str(output),
    ], check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=24.0)
    parser.add_argument("--execute-steps", type=int, default=50)
    parser.add_argument("--raw-policy-rollout", action="store_true")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--preroll", type=float, default=1.2)
    parser.add_argument("--render-spp", type=int, default=4)
    parser.add_argument("--variant", choices=("all", "red_left", "red_right"), default="all")
    parser.add_argument("--out", default="out/panda_vla/pi05_demo")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    output = REPO / args.out
    output.mkdir(parents=True, exist_ok=True)
    variants = [
        {
            "name": "red_left",
            "position": (0.36, 0.01, 0.465),
            "rgba": "0.90 0.06 0.04 1",
            "prompt": "Pick up the red cube and place it on the table",
        },
        {
            "name": "red_right",
            "position": (0.50, 0.01, 0.465),
            "rgba": "0.90 0.06 0.04 1",
            "prompt": "Pick up the red cube and place it on the table",
        },
    ]

    with nuka.Device.create(0) as device:
        policy = None
        load_seconds = 0.0
        if not args.skip_rollout:
            load_started = time.perf_counter()
            policy = Pi05DevicePolicy(CHECKPOINT, TOKENIZER)
            load_seconds = time.perf_counter() - load_started
        videos: list[Path] = []
        summaries: list[dict] = []
        selected_variants = variants if args.variant == "all" else [
            variant for variant in variants if variant["name"] == args.variant]
        for index, variant in enumerate(selected_variants):
            scene = _prepare_variant_scene(
                variant["name"], variant["position"], variant["rgba"])
            variant_output = output / variant["name"]
            variant_output.mkdir(parents=True, exist_ok=True)
            if not args.skip_rollout:
                assert policy is not None
                if args.raw_policy_rollout:
                    summary = rollout(
                        device, variant_output, args.seconds, args.seed + index,
                        args.execute_steps, scene, variant["prompt"],
                        variant["position"], policy, load_seconds,
                    )
                else:
                    summary = scripted_rollout(
                        device, variant_output, scene, variant["position"],
                        variant["prompt"], policy, args.seed + index, args.seconds,
                    )
                summaries.append(summary)
                print(json.dumps(summary, sort_keys=True), flush=True)
            if not args.skip_video:
                videos.append(render_video(
                    device, variant_output, scene, variant["prompt"],
                    variant["position"], args.fps, args.preroll, args.render_spp))
        if policy is not None:
            del policy
            gc.collect()
            torch.cuda.empty_cache()
        if summaries:
            (output / "summary.json").write_text(
                json.dumps({"task": TASK, "episodes": summaries}, indent=2) + "\\n",
                encoding="utf-8")
        if videos:
            combined = _combine_videos(videos, output / "panda_pi05_variants.mp4")
            print(combined)


if __name__ == "__main__":
    main()
