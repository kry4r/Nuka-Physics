#!/usr/bin/env python3
"""Run the Nuka-native LIBERO Spatial demo with the official pi0.5 checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import nuka
from nuka.tasks.libero_black_bowl import (
    LIBERO_CAMERA_SIZE,
    LIBERO_OBJECT_TASK,
    LIBERO_TASK,
    LiberoBlackBowlController,
    LiberoObjectController,
)
from nuka.vla.pi05_libero import LiberoPi05Policy

SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"
TASK_TEXT = LIBERO_TASK
RUN_SUITE = "libero_spatial"
CHECKPOINT = REPO / ".nuka_cache/pi05-libero-peft"
BASE_CHECKPOINT = REPO / ".nuka_cache/pi05-base"
TOKENIZER = REPO / ".nuka_cache/paligemma-tokenizer"
MODEL_REPO = "ases200q2/lerobot_libero_pickplace_10tasks_pi05_peft_tuned_20260325_0900"
MODEL_REVISION = "2e550fcaef64ebc9b5d2104f2ad563c0887b40bd"
MODEL_SHA256 = "724e94b63768cc6f1af2f48b396f0dc502906d4f1e9c31df971524746b41ad29"

# Plate proxy half height from the scene generator; the bowl rests this far above
# the plate body origin when it is genuinely ON the plate.
PLATE_HALF_Z = 0.004


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(
            "/usr/share/fonts/truetype/nerd-fonts/JetBrainsMonoNerdFont-"
            + ("Bold.ttf" if bold else "Regular.ttf")
        ),
        Path("/mnt/c/Windows/Fonts/CascadiaCode.ttf"),
        Path("/mnt/c/Windows/Fonts/CascadiaMono.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _save_rgb(path: Path, image: torch.Tensor) -> None:
    array = image.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
    Image.fromarray(array).save(path)


def _query_row(result, query_id: int, tick: int) -> dict:
    return {
        "query_id": query_id,
        "physics_tick": tick,
        "inference_seconds": result.inference_seconds,
        "allocated_gib": result.allocated_gib,
        "reserved_gib": result.reserved_gib,
        "host_image_copies": result.host_image_copies,
        "preprocess_seconds": result.preprocess_seconds,
        "model_seconds": result.model_seconds,
        "postprocess_seconds": result.postprocess_seconds,
        "synchronize_seconds": result.synchronize_seconds,
        "action_min": result.actions.amin(dim=0).detach().cpu().tolist(),
        "action_max": result.actions.amax(dim=0).detach().cpu().tolist(),
        "first_action": result.actions[0].detach().cpu().tolist(),
    }


def rollout(
    controller: LiberoBlackBowlController,
    policy: LiberoPi05Policy,
    output: Path,
    *,
    seconds: float,
    execute_steps: int,
    seed: int,
    model_load_seconds: float,
    settle_steps: int,
    max_queries: int | None,
) -> dict:
    physics_hz = round(1.0 / controller.dt)
    action_hz = 20
    total_ticks = round(seconds * physics_hz)
    action_period = physics_hz / action_hz
    execute_steps = min(execute_steps, policy.config.chunk_size)
    progress_path = output / "rollout_progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")
    progress_stream = progress_path.open("a", encoding="utf-8")
    started_wall = time.perf_counter()

    def progress(event: str, **values: object) -> None:
        row = {
            "event": event,
            "wall_seconds": time.perf_counter() - started_wall,
            **values,
        }
        progress_stream.write(json.dumps(row) + "\n")
        progress_stream.flush()
        if event in {
            "rollout_start",
            "initial_cameras_begin",
            "initial_cameras_done",
            "query_begin",
            "query_cameras_done",
            "query_state_done",
            "query_done",
            "physics_checkpoint",
            "physics_loop_complete",
            "final_cameras_begin",
            "final_cameras_done",
        }:
            print(json.dumps(row), flush=True)

    progress("rollout_start", total_ticks=total_ticks, execute_steps=execute_steps)
    initial_target = controller.target_position().detach().cpu().numpy().copy()
    plate = controller.plate_position().detach().cpu().numpy().copy()
    progress("initial_cameras_begin")
    initial_images = controller.camera_images()
    progress("initial_cameras_done", shape=list(initial_images.shape))
    _save_rgb(output / "initial_agentview.png", initial_images[0])
    _save_rgb(output / "initial_eye_in_hand.png", initial_images[1])
    del initial_images

    action_chunk: torch.Tensor | None = None
    action_index = execute_steps
    next_action_tick = 0.0
    current_action = torch.zeros(7, device=controller.q.device)
    current_action[6] = -1.0
    query_rows: list[dict] = []
    action_events: list[np.ndarray] = []
    action_event_ticks: list[int] = []
    drive_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    joint_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    eef_rows: list[np.ndarray] = []
    gripper_rows: list[np.ndarray] = []
    task_target_rows: list[np.ndarray] = []
    task_rotation_rows: list[np.ndarray] = []

    def checkpoint(tick: int) -> None:
        np.savez_compressed(
            output / "rollout_partial.npz",
            actions=np.asarray(action_events, dtype=np.float32),
            action_ticks=np.asarray(action_event_ticks, dtype=np.int32),
            drive_targets=np.asarray(drive_rows, dtype=np.float32),
            states=np.asarray(state_rows, dtype=np.float32),
            joints=np.asarray(joint_rows, dtype=np.float32),
            target_positions=np.asarray(target_rows, dtype=np.float32),
            eef_positions=np.asarray(eef_rows, dtype=np.float32),
            gripper_positions=np.asarray(gripper_rows, dtype=np.float32),
            task_targets=np.asarray(task_target_rows, dtype=np.float32)
            if task_target_rows
            else np.empty((0, 3), dtype=np.float32),
            task_rotations=np.asarray(task_rotation_rows, dtype=np.float32)
            if task_rotation_rows
            else np.empty((0, 4), dtype=np.float32),
            checkpoint_tick=np.int32(tick),
        )

    checkpoint(0)

    for tick in range(total_ticks):
        if tick + 1e-6 >= next_action_tick:
            if action_chunk is None or action_index >= execute_steps:
                if max_queries is not None and len(query_rows) >= max_queries:
                    break
                progress("query_begin", query_id=len(query_rows), physics_tick=tick)
                images = controller.camera_images()
                progress(
                    "query_cameras_done",
                    query_id=len(query_rows),
                    shape=list(images.shape),
                )
                state_input = controller.state8()
                progress("query_state_done", query_id=len(query_rows))
                result = policy.predict_chunk(
                    state_input,
                    images[0],
                    images[1],
                    TASK_TEXT,
                    seed=seed + len(query_rows),
                    profile=True,
                )
                action_chunk = result.actions
                action_index = 0
                query_rows.append(_query_row(result, len(query_rows), tick))
                progress(
                    "query_done",
                    query_id=len(query_rows) - 1,
                    physics_tick=tick,
                    inference_seconds=result.inference_seconds,
                    preprocess_seconds=result.preprocess_seconds,
                    model_seconds=result.model_seconds,
                    postprocess_seconds=result.postprocess_seconds,
                    synchronize_seconds=result.synchronize_seconds,
                )
                del images, state_input, result
            current_action = action_chunk[action_index]
            action_index += 1
            controller.set_policy_action(current_action)
            action_events.append(current_action.detach().cpu().numpy().copy())
            action_event_ticks.append(tick)
            next_action_tick += action_period

        progress("physics_begin", physics_tick=tick)
        metrics = controller.step()
        if metrics["finite"] != 1.0:
            raise RuntimeError(f"non-finite LIBERO state at physics tick {tick}")
        drive_rows.append(
            controller.drive_target[0, 1:].detach().cpu().numpy().copy()
        )
        state_rows.append(controller.state8()[0].detach().cpu().numpy().copy())
        joint_rows.append(controller.q[0, 1:].detach().cpu().numpy().copy())
        target_position = controller.target_position().detach().cpu().numpy().copy()
        target_rows.append(target_position)
        eef_rows.append(controller.eef_pose()[:3].detach().cpu().numpy().copy())
        gripper_rows.append(controller.q[0, 8:10].detach().cpu().numpy().copy())
        if tick % 25 == 0:
            progress("physics_done", physics_tick=tick)
        if controller.control_backend == "osc":
            task_target_rows.append(
                controller.task_target[0].detach().cpu().numpy().copy()
            )
            task_rotation_rows.append(
                controller.task_rotation_target[0].detach().cpu().numpy().copy()
            )
        if tick % 100 == 0:
            progress("physics_checkpoint", physics_tick=tick)
            checkpoint(tick)

    progress("physics_loop_complete", physics_ticks=len(target_rows))
    checkpoint(len(target_rows))
    if not target_rows:
        raise RuntimeError("LIBERO rollout produced no physics steps")
    progress("final_cameras_begin")
    final_images = controller.camera_images()
    progress("final_cameras_done", shape=list(final_images.shape))
    _save_rgb(output / "final_agentview.png", final_images[0])
    _save_rgb(output / "final_eye_in_hand.png", final_images[1])
    del final_images, action_chunk, current_action

    drives = np.asarray(drive_rows, dtype=np.float32)
    states = np.asarray(state_rows, dtype=np.float32)
    joints = np.asarray(joint_rows, dtype=np.float32)
    targets = np.asarray(target_rows, dtype=np.float32)
    eefs = np.asarray(eef_rows, dtype=np.float32)
    grippers = np.asarray(gripper_rows, dtype=np.float32)
    task_targets = (
        np.asarray(task_target_rows, dtype=np.float32)
        if task_target_rows
        else np.empty((0, 3), dtype=np.float32)
    )
    task_rotations = (
        np.asarray(task_rotation_rows, dtype=np.float32)
        if task_rotation_rows
        else np.empty((0, 4), dtype=np.float32)
    )
    plate = controller.plate_position().detach().cpu().numpy().copy()
    action_array = np.asarray(action_events, dtype=np.float32)
    np.savez_compressed(
        output / "rollout.npz",
        actions=action_array,
        action_ticks=np.asarray(action_event_ticks, dtype=np.int32),
        drive_targets=drives,
        states=states,
        joints=joints,
        target_positions=targets,
        eef_positions=eefs,
        gripper_positions=grippers,
        task_targets=task_targets,
        task_rotations=task_rotations,
        camera_layout="third_person_main_agentview_wrist_aux",
        plate_position=plate,
        initial_target_position=initial_target,
        physics_dt=np.float32(controller.dt),
        settle_steps=np.int32(settle_steps),
    )

    final_target = targets[-1]
    eef_target_distance = np.linalg.norm(eefs - targets, axis=1)
    close_mask = grippers.max(axis=1) < 0.012
    contact_distance = (
        float(eef_target_distance[close_mask].min())
        if np.any(close_mask)
        else float("inf")
    )
    max_target_z = float(targets[:, 2].max())
    lifted_mask = targets[:, 2] > initial_target[2] + 0.04
    lifted = bool(np.any(lifted_mask))
    grasp_mask = close_mask & lifted_mask & (eef_target_distance < 0.09)
    grasped = bool(np.any(grasp_mask))
    tail_count = min(len(targets), max(2, round(0.5 / controller.dt)))
    tail_motion = float(np.linalg.norm(np.ptp(targets[-tail_count:], axis=0)))
    settled = tail_motion < 0.012

    # The same geometric contract applies to both suites; the placement body is
    # a basket for object tasks and a plate for the spatial bowl task.
    place_label = "basket" if RUN_SUITE == "libero_object" else "plate"
    place_xy_error = float(np.linalg.norm(final_target[:2] - plate[:2]))
    target_above_place = bool(final_target[2] >= plate[2])
    on_place_xy = bool(place_xy_error < 0.032)
    resting_z = float(plate[2] + PLATE_HALF_Z)
    target_contacts_place = bool(abs(float(final_target[2]) - resting_z) < 0.012)
    placed = bool(target_above_place and on_place_xy and target_contacts_place and settled)
    success = bool(grasped and placed)
    summary = {
        "task": TASK_TEXT,
        "libero_suite": RUN_SUITE,
        "libero_task_index": 0 if RUN_SUITE == "libero_object" else 2,
        "scene": str(SCENE.relative_to(REPO)),
        "checkpoint": MODEL_REPO,
        "checkpoint_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "model_format": "PEFT LoRA adapter",
        "base_model": "lerobot/pi05_base",
        "base_checkpoint": str(BASE_CHECKPOINT.relative_to(REPO)),
        "base_parameter_count": getattr(policy, "base_parameter_count", policy.parameter_count),
        "adapter_parameter_count": policy.parameter_count - getattr(
            policy, "base_parameter_count", policy.parameter_count
        ),
        "trainable_parameter_count": policy.trainable_parameter_count,
        "lora": {
            "method": "LORA",
            "rank": 16,
            "alpha": 8,
            "target_modules": "gemma_expert q/k/v attention + MLP; action projections/time MLP saved modules",
        },
        "model_parameter_count": policy.parameter_count,
        "model_normalization": "QUANTILES",
        "model_empty_cameras": policy.config.empty_cameras,
        "model_load_seconds": model_load_seconds,
        "preprocess": {
            "image_keys": [
                "observation.images.image",
                "observation.images.image2",
            ],
            "image_shape_hwc": [LIBERO_CAMERA_SIZE, LIBERO_CAMERA_SIZE, 3],
            "state_shape": [8],
            "nuka_camera_transform": "row flip on both policy cameras before the checkpoint processor",
            "checkpoint_processor_transform": "LeRobot LIBERO processor applies H/W 180-degree flip to both streams",
            "host_image_copies": sum(row["host_image_copies"] for row in query_rows),
        },
        "control": {
            "policy_action_shape": [7],
            "chunk_size": 50,
            "execute_steps": execute_steps,
            "action_rate_hz": action_hz,
            "physics_rate_hz": physics_hz,
            "position_scale_m": 0.05,
            "rotation_scale_rad": 0.5,
            "gripper": "-1 open, +1 closed",
            "control_backend": controller.control_backend,
            "render_quality": controller.render_quality,
            "adapter": (
                "native 6D operational-space torque control"
                if controller.control_backend == "osc"
                else "geometric Jacobian damped least squares to Panda joint PD"
            ),
        },
        "requested_rollout_seconds": seconds,
        "executed_physics_ticks": len(targets),
        "executed_policy_actions": len(action_events),
        "policy_queries": len(query_rows),
        "mean_inference_seconds": float(
            np.mean([row["inference_seconds"] for row in query_rows])
        ),
        "inference": query_rows,
        "initial_target_position": initial_target.tolist(),
        "final_target_position": final_target.tolist(),
        "place_body": place_label,
        "place_position": plate.tolist(),
        "container_position": plate.tolist(),
        "max_target_z": max_target_z,
        "target_lifted": lifted,
        "grasped_with_closed_fingers": grasped,
        "grasp_candidate_ticks": int(np.count_nonzero(grasp_mask)),
        "max_eef_target_distance_m": float(eef_target_distance.max()),
        "min_eef_target_distance_m": float(eef_target_distance.min()),
        "min_eef_target_distance_when_closed_m": contact_distance,
        "gripper_final_opening_m": float(grippers[-1].max()),
        "gripper_closed_during_rollout": bool(np.any(close_mask)),
        "final_place_xy_error_m": place_xy_error,
        "final_container_xy_error_m": place_xy_error,
        "tail_target_motion_m": tail_motion,
        "target_settled": settled,
        "target_placed_in_container": placed,
        "target_placed_on_plate": placed,
        "target_above_place": target_above_place,
        "on_place_xy_within_0p03": on_place_xy,
        "on_place_resting_contact": target_contacts_place,
        "expected_resting_z": resting_z,
        "final_plate_xy_error_m": place_xy_error,
        "target_placed_on_plate": placed,
        "on_plate_bowl_above": target_above_place,
        "on_plate_xy_within_0p03": on_place_xy,
        "on_plate_resting_contact": target_contacts_place,
        "on_plate_expected_resting_z": resting_z,
        "joint_motion_max_rad": float(np.ptp(joints[:, :7], axis=0).max()),
        "gripper_motion_m": float(np.ptp(joints[:, 7:], axis=0).max()),
        "success": success,
        "success_contract": (
            f"LIBERO {RUN_SUITE} goal: the closed gripper lifts the target 0.04m, "
            f"then the target ends above the {place_label}, within 0.03m XY, "
            f"rests in contact with it, and settles"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _compose_frame(
    third_person: np.ndarray,
    agent: np.ndarray,
    wrist: np.ndarray,
    *,
    seconds: float,
    success: bool,
) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), (13, 16, 17))
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_font = _font(24, bold=True)
    body_font = _font(17)
    small_font = _font(15, bold=True)
    main = Image.fromarray(third_person).resize((820, 615), Image.Resampling.LANCZOS)
    agent_image = Image.fromarray(agent).resize((420, 292), Image.Resampling.LANCZOS)
    wrist_image = Image.fromarray(wrist).resize((420, 292), Image.Resampling.LANCZOS)
    canvas.paste(main, (0, 0))
    canvas.paste(agent_image, (840, 0))
    canvas.paste(wrist_image, (840, 306))
    draw.rectangle((0, 615, 1280, 720), fill=(8, 11, 12, 248))
    draw.rectangle((840, 0, 1279, 598), outline=(196, 202, 198, 220), width=2)
    draw.line((840, 300, 1279, 300), fill=(196, 202, 198, 180), width=2)
    draw.text((858, 14), "AGENTVIEW / POLICY INPUT", font=small_font, fill=(238, 241, 237, 255))
    draw.text((858, 320), "WRIST / POLICY INPUT", font=small_font, fill=(238, 241, 237, 255))
    draw.text((34, 636), f"NUKA / PI0.5 / LIBERO {RUN_SUITE.upper()}", font=title_font, fill=(242, 244, 241, 255))
    draw.text((34, 676), TASK_TEXT, font=body_font, fill=(222, 227, 222, 255))
    draw.text((1010, 640), f"t = {seconds:05.2f}s", font=small_font, fill=(164, 174, 168, 255))
    status = "SUCCESS" if success else "POLICY ROLLOUT"
    color = (53, 205, 139, 255) if success else (238, 185, 66, 255)
    draw.ellipse((1010, 679, 1024, 693), fill=color)
    draw.text((1034, 674), status, font=small_font, fill=(228, 232, 227, 255))
    return canvas


def render_video(
    device: nuka.Device,
    output: Path,
    *,
    fps: int,
    render_quality: str = "ultra",
) -> Path:
    data = np.load(output / "rollout.npz")
    drives = data["drive_targets"]
    dt = float(data["physics_dt"])
    settle_steps = int(data["settle_steps"])
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    control_backend = summary.get("control", {}).get("control_backend", "joint_pd")
    if control_backend not in ("joint_pd", "osc"):
        raise ValueError(f"unsupported recorded control backend: {control_backend!r}")
    task_targets = data["task_targets"] if "task_targets" in data else np.empty((0, 3))
    task_rotations = data["task_rotations"] if "task_rotations" in data else np.empty((0, 4))
    if control_backend == "osc" and (
        len(task_targets) != len(drives) or len(task_rotations) != len(drives)
    ):
        raise RuntimeError("OSC rollout is missing per-tick task targets for replay")
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    controller_type = (
        LiberoObjectController
        if RUN_SUITE == "libero_object"
        else LiberoBlackBowlController
    )
    controller = controller_type(
        SCENE,
        device,
        dt=dt,
        control_backend=control_backend,
        render_quality=render_quality,
    )
    try:
        for _ in range(settle_steps):
            controller.step(advance_gripper=False)
        frame_count = max(2, int(np.ceil(len(drives) * dt * fps)))
        current_tick = 0
        for frame_index in range(frame_count):
            target_tick = min(len(drives) - 1, int((frame_index / fps) / dt))
            while current_tick <= target_tick:
                target = torch.as_tensor(
                    drives[current_tick], device=controller.q.device
                )
                if control_backend == "osc":
                    controller.task_target[0] = torch.as_tensor(
                        task_targets[current_tick], device=controller.q.device
                    )
                    controller.task_rotation_target[0] = torch.as_tensor(
                        task_rotations[current_tick], device=controller.q.device
                    )
                    controller.drive_target[0, 8:10] = target[7:9]
                else:
                    controller.drive_target[0, 1:] = target
                controller.step(advance_gripper=False)
                current_tick += 1
            policy_images = controller.camera_images().detach().clamp(0, 1)
            policy_pixels = policy_images.mul(255).to(torch.uint8).cpu().numpy()
            render_spp = {"preview": 4, "high": 16, "ultra": 32}[render_quality]
            third_pixels = controller.third_person_image(
                width=820, height=615, spp=render_spp
            )
            frame = _compose_frame(
                third_pixels,
                policy_pixels[0],
                policy_pixels[1],
                seconds=target_tick * dt,
                success=bool(summary["success"] and frame_index == frame_count - 1),
            )
            frame.save(frames_dir / f"frame_{frame_index:05d}.png")
            if frame_index % max(1, fps * 2) == 0:
                print(f"render {frame_index}/{frame_count}", flush=True)
    finally:
        controller.close()

    video = output / "libero_pi05.mp4"
    bundled = REPO / "tools/ffmpeg/ffmpeg.exe"
    executable = str(bundled) if bundled.is_file() else shutil.which("ffmpeg")
    if executable is None:
        raise FileNotFoundError("ffmpeg")
    subprocess.run(
        [
            executable,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    return video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-backend", choices=("joint_pd", "osc"), default="joint_pd")
    parser.add_argument("--suite", choices=("spatial", "object"), default="spatial")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=1,
        help="policy actions to execute before refreshing the observation (1 = 20 Hz inference)",
    )
    # LiberoEnv.reset() settles with ten 20 Hz no-op gripper-open actions
    # (0.5 s). Nuka advances physics at 200 Hz, so that is 100 ticks.
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--render-quality", choices=("preview", "high", "ultra"), default="high"
    )
    parser.add_argument("--out", default="out/libero/pi05_black_bowl")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    global SCENE, TASK_TEXT, RUN_SUITE
    if args.suite == "object":
        SCENE = REPO / ".nuka-assets/generated/libero/libero_object_orange_juice.nks"
        TASK_TEXT = LIBERO_OBJECT_TASK
        RUN_SUITE = "libero_object"
        if args.out == "out/libero/pi05_black_bowl":
            args.out = "out/libero/pi05_object_orange_juice"
    controller_type = LiberoObjectController if args.suite == "object" else LiberoBlackBowlController

    if not SCENE.is_file():
        raise FileNotFoundError(
            f"missing cooked scene {SCENE}; run tools/assets/convert_libero_pi05.py --cook"
        )
    output = REPO / args.out
    output.mkdir(parents=True, exist_ok=True)
    startup_path = output / "startup_progress.jsonl"
    startup_path.write_text("", encoding="utf-8")
    startup_started = time.perf_counter()

    def startup(event: str, **values: object) -> None:
        row = {"event": event, "wall_seconds": time.perf_counter() - startup_started, **values}
        with startup_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        print(json.dumps(row), flush=True)

    startup("process_start", pid=__import__("os").getpid())

    with nuka.Device.create(0) as device:
        startup("device_ready")
        if args.render_only:
            video = render_video(
                device, output, fps=args.fps, render_quality=args.render_quality
            )
            print(video)
            return

        controller = controller_type(
            SCENE,
            device,
            control_backend=args.control_backend,
            render_quality=args.render_quality,
        )
        startup("world_ready", control_backend=args.control_backend)
        policy = None
        try:
            startup("settle_begin", steps=args.settle_steps, control_hz=20)
            for _ in range(args.settle_steps):
                # Match LiberoEnv.reset(): ten 20 Hz dummy open actions.
                controller.step(
                    torch.tensor(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                        device=controller.q.device,
                    )
                )
            startup("settle_done")
            # Nuka creates CUDA resources first so Torch joins an initialized GPU
            # process instead of forcing the engine through a late context change.
            startup("policy_load_begin")
            started = time.perf_counter()
            policy = LiberoPi05Policy(CHECKPOINT, TOKENIZER)
            load_seconds = time.perf_counter() - started
            startup("policy_load_done", seconds=load_seconds)
            summary = rollout(
                controller,
                policy,
                output,
                seconds=args.seconds,
                execute_steps=args.execute_steps,
                seed=args.seed,
                model_load_seconds=load_seconds,
                settle_steps=args.settle_steps,
                max_queries=args.max_queries,
            )
            print(json.dumps(summary, indent=2), flush=True)
        finally:
            if policy is not None:
                del policy
            gc.collect()
            torch.cuda.empty_cache()
            controller.close()

        if not args.skip_video:
            video = render_video(
                device, output, fps=args.fps, render_quality=args.render_quality
            )
            print(video)


if __name__ == "__main__":
    main()
