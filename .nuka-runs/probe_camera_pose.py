#!/usr/bin/env python
"""Print real LIBERO agentview camera world pose."""
import sys
sys.path.insert(0, "/mnt/c/Softwares/code/Nuka-Physics/.nuka_cache/LIBERO")
import numpy as np
from scipy.spatial.transform import Rotation as R
from libero.libero.envs import PickPlaceDistractorCan, TASK_MAPPING
import robosuite

info_dict = {
    "problem_name": "LIBERO_SPATIAL_TASK_2_0",
    "bddl_file": "/root/.libero/bddl_files/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
}
controller_configs = robosuite.load_controller_config(default_controller="OSC_POSE")
cls = TASK_MAPPING[info_dict["problem_name"]]
env = cls(
    bddl_file_name=info_dict["bddl_file"],
    robots=["Panda"],
    controller_configs=controller_configs,
    has_renderer=False,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_object_obs=False,
    use_camera_obs=True,
    reward_shaping=True,
    control_freq=20,
    camera_names=["agentview"],
    camera_heights=360,
    camera_widths=360,
)
env.reset()

cam_id = env.sim.model.camera_name2id("agentview")
cam_pos = env.sim.model.cam_pos[cam_id].copy()
cam_quat_wxyz = env.sim.model.cam_quat[cam_id].copy()

print("Real LIBERO agentview cam_pos:", cam_pos)
print("Real LIBERO agentview cam_quat (wxyz):", cam_quat_wxyz)
rot = R.from_quat([cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]])
print("Real LIBERO agentview euler (deg, xyz):", rot.as_euler("xyz", degrees=True))
print("Real LIBERO agentview matrix:")
print(rot.as_matrix())

# Also check if it's a world camera or a mounted camera
print("\nChecking camera body id:", env.sim.model.cam_bodyid[cam_id])
print("body id 0 = world, >0 = attached to body")
