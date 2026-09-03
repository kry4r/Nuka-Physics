"""Measure robosuite's real Panda gripper aperture response for LIBERO."""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/c/Softwares/code/Nuka-Physics/.nuka_cache/LIBERO")

import numpy as np
import robosuite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(2)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
info = BDDLUtils.get_problem_info(bddl)

env = TASK_MAPPING[info["problem_name"]](
    bddl_file_name=bddl,
    robots=["Panda"],
    controller_configs=robosuite.load_controller_config(default_controller="OSC_POSE"),
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    ignore_done=True,
    control_freq=20,
)

env.reset()
init_states = suite.get_task_init_states(2)
env.sim.set_state_from_flattened(init_states[0])
env.sim.forward()

robot = env.robots[0]
gripper = robot.gripper
sim = env.sim

print("gripper class :", type(gripper).__name__)
print("gripper.dof   :", gripper.dof)
print("gripper.speed :", gripper.speed)
print("gripper.init_qpos:", np.asarray(gripper.init_qpos).tolist())
print("actuators     :", list(gripper.actuators))
idxs = [sim.model.actuator_name2id(a) for a in gripper.actuators]
print("ctrlrange     :", sim.model.actuator_ctrlrange[idxs].tolist())
joint_idxs = [sim.model.joint_name2id(j) for j in gripper.joints]
print("joint range   :", sim.model.jnt_range[joint_idxs].tolist())
print("qpos after set_init_state:", np.round(sim.data.qpos[sim.model.jnt_qposadr[joint_idxs]], 6).tolist())

print("\nstep  action  current_action        ctrl               gripper_qpos")


def show(step: int, action: float) -> None:
    qpos = sim.data.qpos[sim.model.jnt_qposadr[joint_idxs]]
    current = np.atleast_1d(np.asarray(gripper.current_action, dtype=float))
    current_text = " ".join("%+.4f" % v for v in current)
    print(
        "%4d  %+5.1f  [%s]  [%+.5f %+.5f]  [%+.6f %+.6f]"
        % (
            step,
            action,
            current_text,
            sim.data.ctrl[idxs[0]],
            sim.data.ctrl[idxs[1]],
            qpos[0],
            qpos[1],
        )
    )


step = 0
show(step, float("nan"))
for phase_action, count in ((-1.0, 15), (1.0, 30), (-1.0, 15)):
    for _ in range(count):
        env.step([0, 0, 0, 0, 0, 0, phase_action])
        step += 1
        if step <= 12 or step % 5 == 0:
            show(step, phase_action)
    print("  --- phase change ---")

env.close()
