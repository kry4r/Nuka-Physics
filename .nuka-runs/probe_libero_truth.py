"""Print real LIBERO reset geometry so Nuka's scene can be checked against it."""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, "/mnt/c/Softwares/code/Nuka-Physics/.nuka_cache/LIBERO")

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils
import robosuite.utils.transform_utils as T

suite_name = "libero_spatial"
task_id = 2

suite = benchmark.get_benchmark_dict()[suite_name]()
task = suite.get_task(task_id)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print("task_name:", task.name)
print("language :", task.language)
print("bddl     :", os.path.basename(bddl))

info = BDDLUtils.get_problem_info(bddl)
print("problem_info:", info)

import robosuite

# LIBERO's OffScreenRenderEnv wrapper loads OSC_POSE; the raw domain would
# otherwise default to joint control with an 8D action.
controller_configs = robosuite.load_controller_config(default_controller="OSC_POSE")
print("controller_configs:", controller_configs)

cls = TASK_MAPPING[info["problem_name"]]
env = cls(
    bddl_file_name=bddl,
    robots=["Panda"],
    controller_configs=controller_configs,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    ignore_done=True,
    control_freq=20,
)

obs = env.reset()
init_states = suite.get_task_init_states(task_id)
print("init_states shape:", np.asarray(init_states).shape)
env.sim.set_state_from_flattened(init_states[0])
env.sim.forward()
# LiberoEnv.reset() settles with ten 20 Hz no-op open actions.
for _ in range(10):
    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=float)
axis_angle = T.quat2axisangle(eef_quat)
grip = np.asarray(obs["robot0_gripper_qpos"], dtype=float)

print()
print("=== ROBOT (world frame, after reset+settle) ===")
print("robot0_eef_pos     :", np.round(eef_pos, 6).tolist())
print("robot0_eef_quat    :", np.round(eef_quat, 6).tolist())
print("eef axis_angle     :", np.round(axis_angle, 6).tolist())
print("robot0_gripper_qpos:", np.round(grip, 6).tolist())
print("STATE_8            :", np.round(np.concatenate([eef_pos, axis_angle, grip]), 6).tolist())
print("robot base xpos    :", np.round(env.robots[0].robot_model.base_xpos_offset["table"](1.0), 6).tolist())

sim = env.sim
print()
print("=== ARENA ===")
for attr in ("table_offset", "workspace_offset", "table_full_size"):
    if hasattr(env, attr):
        print(f"{attr:18}:", np.asarray(getattr(env, attr)).tolist())

print()
print("=== OBJECT BODY POSITIONS (world) ===")
for i in range(sim.model.nbody):
    name = sim.model.body_id2name(i)
    if not name:
        continue
    low = name.lower()
    if any(k in low for k in ("bowl", "plate", "cookies", "ramekin", "cabinet", "stove", "table")):
        print(f"  {name:42}", np.round(sim.data.body_xpos[i], 6).tolist())

print()
print("=== GRIP SITE ===")
for sname in ("gripper0_grip_site", "gripper0_ft_frame"):
    try:
        sid = sim.model.site_name2id(sname)
        print(f"  {sname:24}", np.round(sim.data.site_xpos[sid], 6).tolist())
    except Exception as exc:  # noqa: BLE001
        print(f"  {sname}: {exc}")

print()
print("=== JOINTS ===")
print("robot0_joint_pos:", np.round(np.asarray(obs["robot0_joint_pos"], dtype=float), 6).tolist())

env.close()
