"""Print real LIBERO camera extrinsics and FOV for comparison with Nuka."""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/c/Softwares/code/Nuka-Physics/.nuka_cache/LIBERO")

import numpy as np
import robosuite
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils

_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

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
for _ in range(10):
    env.step([0, 0, 0, 0, 0, 0, -1])

sim = env.sim
model = sim.model


def mat_to_quat(mat):
    """Return (w, x, y, z) from a 3x3 rotation matrix."""
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 / s,
                (mat[2, 1] - mat[1, 2]) * s,
                (mat[0, 2] - mat[2, 0]) * s,
                (mat[1, 0] - mat[0, 1]) * s,
            ]
        )
    idx = int(np.argmax([mat[0, 0], mat[1, 1], mat[2, 2]]))
    if idx == 0:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        return np.array(
            [
                (mat[2, 1] - mat[1, 2]) / s,
                0.25 * s,
                (mat[0, 1] + mat[1, 0]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
            ]
        )
    if idx == 1:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        return np.array(
            [
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[0, 1] + mat[1, 0]) / s,
                0.25 * s,
                (mat[1, 2] + mat[2, 1]) / s,
            ]
        )
    s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
    return np.array(
        [
            (mat[1, 0] - mat[0, 1]) / s,
            (mat[0, 2] + mat[2, 0]) / s,
            (mat[1, 2] + mat[2, 1]) / s,
            0.25 * s,
        ]
    )


print("=== CAMERAS (world frame) ===")
for name in ("agentview", "robot0_eye_in_hand"):
    cid = model.camera_name2id(name)
    pos = sim.data.cam_xpos[cid]
    mat = sim.data.cam_xmat[cid].reshape(3, 3)
    quat = mat_to_quat(mat)
    print(f"\n{name}")
    print("  fovy      :", float(model.cam_fovy[cid]))
    print("  world pos :", np.round(pos, 6).tolist())
    print("  world quat (w,x,y,z):", np.round(quat, 6).tolist())
    print("  forward (-Z col):", np.round(-mat[:, 2], 6).tolist())
    print("  up      (+Y col):", np.round(mat[:, 1], 6).tolist())

print("\n=== LINK / BASE POSES (world) ===")
for body in ("robot0_link7", "robot0_base", "robot0_right_hand"):
    try:
        bid = model.body_name2id(body)
        bmat = sim.data.body_xmat[bid].reshape(3, 3)
        print(f"{body:22} pos={np.round(sim.data.body_xpos[bid], 6).tolist()}")
        print(f"{'':22} quat={np.round(mat_to_quat(bmat), 6).tolist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"{body}: {exc}")

# Wrist camera relative to link7, which is how Nuka attaches it.
lid = model.body_name2id("robot0_link7")
lpos = sim.data.body_xpos[lid]
lmat = sim.data.body_xmat[lid].reshape(3, 3)
cid = model.camera_name2id("robot0_eye_in_hand")
cpos = sim.data.cam_xpos[cid]
cmat = sim.data.cam_xmat[cid].reshape(3, 3)
local_pos = lmat.T @ (cpos - lpos)
local_mat = lmat.T @ cmat
print("\n=== eye_in_hand RELATIVE TO link7 ===")
print("  local pos :", np.round(local_pos, 8).tolist())
print("  local quat (w,x,y,z):", np.round(mat_to_quat(local_mat), 8).tolist())

env.close()
