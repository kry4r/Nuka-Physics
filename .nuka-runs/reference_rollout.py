"""Closed-loop reference: run the pi0.5 checkpoint inside real LIBERO."""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/c/Softwares/code/Nuka-Physics/.nuka_cache/LIBERO")
sys.path.insert(0, "/mnt/c/Softwares/code/Nuka-Physics/python")

import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
import torch
from PIL import Image

_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils

from nuka.vla.pi05_libero import LiberoPi05Policy

REPO = "/mnt/c/Softwares/code/Nuka-Physics"
CHECKPOINT = f"{REPO}/.nuka_cache/pi05-libero"
TOKENIZER = f"{REPO}/.nuka_cache/paligemma-tokenizer"
OUT = f"{REPO}/.nuka-runs/libero_reference_rollout"
SIZE = 360
EXECUTE_STEPS = 10
MAX_ACTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
# openpi's LIBERO conversion stores agentview/wrist rotated 180 degrees.
FLIP = "--flip" in sys.argv

os.makedirs(OUT, exist_ok=True)

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(2)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
info = BDDLUtils.get_problem_info(bddl)
print("task:", task.language)

env = TASK_MAPPING[info["problem_name"]](
    bddl_file_name=bddl,
    robots=["Panda"],
    controller_configs=robosuite.load_controller_config(default_controller="OSC_POSE"),
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    use_object_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=SIZE,
    camera_widths=SIZE,
    ignore_done=True,
    control_freq=20,
)

env.reset()
init_states = suite.get_task_init_states(2)
env.sim.set_state_from_flattened(init_states[0])
env.sim.forward()
for _ in range(10):
    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

bowl_id = env.sim.model.body_name2id("akita_black_bowl_1_main")
plate_id = env.sim.model.body_name2id("plate_1_main")
bowl0 = env.sim.data.body_xpos[bowl_id].copy()
print("bowl at reset:", np.round(bowl0, 5).tolist())

policy = LiberoPi05Policy(CHECKPOINT, TOKENIZER)
print("policy loaded")


def state8(observation):
    return np.concatenate(
        [
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            T.quat2axisangle(np.asarray(observation["robot0_eef_quat"], dtype=np.float64)).astype(np.float32),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )


def maybe_flip(img: np.ndarray, mode: str) -> np.ndarray:
    """Apply transformation per FLIP_MODE."""
    if mode == "none":
        return img
    elif mode == "rot180":
        return img[::-1, ::-1]
    elif mode == "agent_rot180":
        return img[::-1, ::-1]
    elif mode == "wrist_fliplr":
        return img[:, ::-1]
    elif mode == "both_custom":
        return img  # handled in predict_chunk call
    raise ValueError(f"unknown mode {mode}")


FLIP_MODE = os.environ.get("FLIP_MODE", "none")
print(f"FLIP_MODE: {FLIP_MODE}")


Image.fromarray(np.asarray(obs["agentview_image"], dtype=np.uint8)).save(
    f"{OUT}/initial_agentview.png"
)

rows = []
chunk = None
index = EXECUTE_STEPS
print("\nact  grip_cmd  dz_cmd   eef_z   dist_bowl  bowl_z  fingers")
for step in range(MAX_ACTIONS):
    if index >= EXECUTE_STEPS:
        result = policy.predict_chunk(
            state8(obs),
            maybe_flip(np.asarray(obs["agentview_image"], dtype=np.uint8)),
            maybe_flip(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)),
            task.language,
            seed=20260828 + step,
        )
        chunk = result.actions.detach().cpu().numpy()
        index = 0

    action = chunk[index].astype(np.float64)
    index += 1
    obs, _, _, _ = env.step(action.tolist())

    eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
    bowl = env.sim.data.body_xpos[bowl_id].copy()
    dist = float(np.linalg.norm(eef - bowl))
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
    rows.append(
        dict(step=step, grip_cmd=float(action[6]), eef=eef.tolist(), bowl=bowl.tolist(), dist=dist)
    )
    if step % 10 == 0 or step < 5:
        print(
            "%3d  %+.3f    %+.3f   %.3f   %.4f     %.4f  %.5f"
            % (step, action[6], action[2], eef[2], dist, bowl[2], grip[0])
        )

bowl = env.sim.data.body_xpos[bowl_id].copy()
plate = env.sim.data.body_xpos[plate_id].copy()
lift = float(bowl[2] - bowl0[2])
print("\n=== REFERENCE RESULT ===")
print("bowl start z %.4f  end z %.4f  lift %.4f m" % (bowl0[2], bowl[2], lift))
print("bowl end xy  :", np.round(bowl[:2], 4).tolist())
print("plate xy     :", np.round(plate[:2], 4).tolist())
print("xy error     : %.4f m" % float(np.linalg.norm(bowl[:2] - plate[:2])))
print("closest approach: %.4f m" % min(r["dist"] for r in rows))
grips = np.array([r["grip_cmd"] for r in rows])
print("gripper dim: mean %+.3f  frac_positive %.2f" % (grips.mean(), float((grips > 0).mean())))
print("first action gripper: %+.4f" % grips[0])
print("LIFTED" if lift > 0.04 else "NOT LIFTED")

Image.fromarray(np.asarray(obs["agentview_image"], dtype=np.uint8)).save(
    f"{OUT}/final_agentview.png"
)
env.close()
