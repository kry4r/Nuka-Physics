#!/usr/bin/env python3
"""Headless MuJoCo contract gate for the pretrained G1 Moves ONNX policy.

This compares the literal standalone runner with the policy's serialized mjlab
training contract. It is a diagnostic/demo gate, not a unit test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
from nuka.tasks.g1_motion_contract import G1_JOINT_NAMES, load_g1_motion

MODEL = REPO / ".nuka-assets/src/g1-moves/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
MOTION = REPO / ".nuka-assets/src/g1-moves/dance/J_Dance17_Shuffle/training/J_Dance17_Shuffle.npz"
POLICY = REPO / ".nuka-assets/src/g1-moves/dance/J_Dance17_Shuffle/policy/J_Dance17_Shuffle_policy.onnx"

DEFAULT_Q = np.zeros(29, dtype=np.float32)
for _index, _name in enumerate(G1_JOINT_NAMES):
    if _name.endswith("hip_pitch_joint"):
        DEFAULT_Q[_index] = -0.312
    elif _name.endswith("knee_joint"):
        DEFAULT_Q[_index] = 0.669
    elif _name.endswith("ankle_pitch_joint"):
        DEFAULT_Q[_index] = -0.363
    elif _name.endswith("elbow_joint"):
        DEFAULT_Q[_index] = 0.6
    elif _name in ("left_shoulder_pitch_joint", "left_shoulder_roll_joint"):
        DEFAULT_Q[_index] = 0.2
    elif _name == "right_shoulder_pitch_joint":
        DEFAULT_Q[_index] = 0.2
    elif _name == "right_shoulder_roll_joint":
        DEFAULT_Q[_index] = -0.2

ACTION_SCALE = np.array([
    0.5475464629911068 if ("hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint") else
    0.35066146637882434 if ("hip_roll" in n or "knee" in n) else
    0.07450087032950714 if ("wrist_pitch" in n or "wrist_yaw" in n) else
    0.43857731392336724
    for n in G1_JOINT_NAMES
], dtype=np.float32)


def _control_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kp = np.empty(29, dtype=np.float64)
    kd = np.empty(29, dtype=np.float64)
    limit = np.empty(29, dtype=np.float64)
    armature = np.empty(29, dtype=np.float64)
    for index, name in enumerate(G1_JOINT_NAMES):
        if any(part in name for part in ("elbow", "shoulder", "wrist_roll")):
            values = (14.25062309787429, 0.907222843292423, 25.0, 0.003609725)
        elif any(part in name for part in ("hip_pitch", "hip_yaw")) or name == "waist_yaw_joint":
            values = (40.17923863450712, 2.557889775413375, 88.0, 0.01017752004132231)
        elif "hip_roll" in name or "knee" in name:
            values = (99.09842777666111, 6.308801853496639, 139.0, 0.025101925)
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            values = (16.77832748089279, 1.06814150219, 5.0, 0.00425)
        else:
            values = (28.50124619574858, 1.814445686584846, 50.0, 0.00721945)
        kp[index], kd[index], limit[index], armature[index] = values
    return kp, kd, limit, armature


def _load_training_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(MODEL))
    kp, kd, effort, armature = _control_parameters()
    for index, name in enumerate(G1_JOINT_NAMES):
        joint = spec.joint(name)
        joint.armature = armature[index]
        joint.frictionloss = 0.0
        actuator = spec.add_actuator(name=name, target=name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = kp[index]
        actuator.biasprm[1] = -kp[index]
        actuator.biasprm[2] = -kd[index]
        actuator.ctrllimited = False
        actuator.forcelimited = True
        actuator.forcerange[:] = (-effort[index], effort[index])
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0.0, 0.0, 0.05),
        contype=1, conaffinity=1, condim=3, friction=(0.6, 0.005, 0.0001),
    )
    return spec.compile()


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _sixd(rotation: np.ndarray) -> np.ndarray:
    return rotation[:, :2].reshape(-1).astype(np.float32)


def _anchor_observation(
    robot_pos: np.ndarray,
    robot_quat: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    robot_rotation = _quat_to_matrix(robot_quat)
    anchor_rotation = _quat_to_matrix(anchor_quat)
    position = robot_rotation.T @ (anchor_pos - robot_pos)
    orientation = _sixd(robot_rotation.T @ anchor_rotation)
    return position.astype(np.float32), orientation


def _sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    start = int(model.sensor_adr[sensor_id])
    size = int(model.sensor_dim[sensor_id])
    return data.sensordata[start:start + size].astype(np.float32).copy()


def run(seconds: float, contract: str) -> dict[str, object]:
    model = _load_training_model()
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.opt.tolerance = 1e-8
    data = mujoco.MjData(model)
    motion = load_g1_motion(MOTION)
    session = ort.InferenceSession(str(POLICY), providers=["CPUExecutionProvider"])

    joint_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt))
    if (model.nq, model.nv, joint_names) != (36, 35, G1_JOINT_NAMES):
        raise RuntimeError(f"G1 topology mismatch: nq={model.nq} nv={model.nv} joints={joint_names}")

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    anchor_index = 0 if contract == "official_script" else torso_id - 1
    initial = motion.sample(0.0)
    data.qpos[:3] = initial["body_position_world"][0]
    data.qpos[3:7] = initial["body_quaternion_wxyz"][0]
    data.qpos[7:36] = initial["joint_position"]
    data.qvel[:3] = initial["body_linear_velocity_world"][0]
    data.qvel[3:6] = _quat_to_matrix(data.qpos[3:7]).T @ initial["body_angular_velocity_world"][0]
    data.qvel[6:35] = initial["joint_velocity"]
    mujoco.mj_forward(model, data)

    last_action = np.zeros(29, dtype=np.float32)
    target = initial["joint_position"].astype(np.float64)
    control_steps = min(int(seconds / 0.02), int(motion.duration / 0.02))
    heights: list[float] = []
    tilts: list[float] = []
    errors: list[float] = []
    action_abs_max = 0.0
    target_abs_max = 0.0

    for control_step in range(control_steps):
        frame = min(int(control_step * 0.02 * motion.fps), motion.frame_count - 1)
        ref_jp = motion.joint_position[frame]
        ref_jv = motion.joint_velocity[frame]
        if contract == "official_script":
            robot_anchor_pos = data.qpos[:3]
            robot_anchor_quat = data.qpos[3:7]
        else:
            robot_anchor_pos = data.xpos[torso_id]
            robot_anchor_quat = data.xquat[torso_id]
        anchor_pos, anchor_ori = _anchor_observation(
            robot_anchor_pos, robot_anchor_quat,
            motion.body_position_world[frame, anchor_index],
            motion.body_quaternion_wxyz[frame, anchor_index],
        )
        base_ang_vel = _sensor(model, data, "imu_ang_vel")
        base_lin_vel = _sensor(model, data, "imu_lin_vel")
        if contract == "official_script":
            velocity_terms = (base_ang_vel, base_lin_vel)
            joint_offset = data.qpos[7:36]
        else:
            velocity_terms = (base_lin_vel, base_ang_vel)
            joint_offset = data.qpos[7:36] - DEFAULT_Q
        obs = np.concatenate((
            ref_jp, ref_jv, anchor_pos, anchor_ori,
            *velocity_terms, joint_offset, data.qvel[6:35], last_action,
        )).astype(np.float32)
        action = session.run(["actions"], {"obs": obs[None]})[0][0]
        if not np.isfinite(action).all():
            raise RuntimeError(f"non-finite ONNX action at control step {control_step}")
        last_action = action.copy()
        target = action.astype(np.float64) if contract == "official_script" else (DEFAULT_Q + ACTION_SCALE * action).astype(np.float64)
        action_abs_max = max(action_abs_max, float(np.max(np.abs(action))))
        target_abs_max = max(target_abs_max, float(np.max(np.abs(target))))

        for _ in range(4):
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(f"non-finite MuJoCo state at control step {control_step}")
            heights.append(float(data.qpos[2]))
            rotation = _quat_to_matrix(data.qpos[3:7])
            tilts.append(float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
            errors.append(float(np.sqrt(np.mean((data.qpos[7:36] - target) ** 2))))

    return {
        "backend": "mujoco",
        "contract": contract,
        "model": str(MODEL.relative_to(REPO)),
        "policy": str(POLICY.relative_to(REPO)),
        "motion": str(MOTION.relative_to(REPO)),
        "mujoco_version": mujoco.__version__,
        "onnxruntime_version": ort.__version__,
        "physics_hz": 200,
        "policy_hz": 50,
        "reference_fps": motion.fps,
        "control_steps": control_steps,
        "seconds": control_steps * 0.02,
        "action_abs_max": action_abs_max,
        "target_abs_max_rad": target_abs_max,
        "joint_target_rmse_mean_rad": float(np.mean(errors)),
        "root_height_min_m": float(np.min(heights)),
        "root_height_final_m": heights[-1],
        "root_tilt_max_rad": float(np.max(tilts)),
        "root_tilt_final_rad": tilts[-1],
        "upright_final": bool(heights[-1] > 0.55 and tilts[-1] < 0.9),
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--contract", choices=("official_script", "training_contract"), default="training_contract")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = run(args.seconds, args.contract)
    out = REPO / (args.out or f"out/g1_shuffle/mujoco_{args.contract}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
