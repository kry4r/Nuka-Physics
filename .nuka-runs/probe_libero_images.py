"""Render real LIBERO policy cameras and find which flip aligns Nuka's frames."""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/c/Softwares/code/Nuka-Physics/.nuka_cache/LIBERO")

import numpy as np
from PIL import Image
import robosuite
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils

REPO = "/mnt/c/Softwares/code/Nuka-Physics"
RUN = f"{REPO}/.nuka-runs/libero_pi05_osc_signfix"
OUT = f"{REPO}/.nuka-runs/libero_reference"
SIZE = 360

os.makedirs(OUT, exist_ok=True)

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(2)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
info = BDDLUtils.get_problem_info(bddl)

controller_configs = robosuite.load_controller_config(default_controller="OSC_POSE")
env = TASK_MAPPING[info["problem_name"]](
    bddl_file_name=bddl,
    robots=["Panda"],
    controller_configs=controller_configs,
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

print("IMAGE_CONVENTION:", robosuite.macros.IMAGE_CONVENTION)
print("eef_pos:", np.round(np.asarray(obs["robot0_eef_pos"], float), 6).tolist())
print("gripper_qpos:", np.round(np.asarray(obs["robot0_gripper_qpos"], float), 6).tolist())

pairs = {
    "agentview": ("agentview_image", "initial_agentview.png"),
    "eye_in_hand": ("robot0_eye_in_hand_image", "initial_eye_in_hand.png"),
}

TRANSFORMS = {
    "identity": lambda a: a,
    "fliplr": lambda a: a[:, ::-1],
    "flipud": lambda a: a[::-1, :],
    "rot180": lambda a: a[::-1, ::-1],
}


def gray_small(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8)).convert("L").resize((48, 48))
    vector = np.asarray(image, dtype=np.float64).ravel()
    return (vector - vector.mean()) / (vector.std() + 1e-8)


for label, (obs_key, nuka_name) in pairs.items():
    raw = np.asarray(obs[obs_key], dtype=np.uint8)
    Image.fromarray(raw).save(f"{OUT}/libero_{label}_raw.png")
    Image.fromarray(raw[::-1, ::-1]).save(f"{OUT}/libero_{label}_policy_input.png")

    nuka_path = f"{RUN}/{nuka_name}"
    if not os.path.isfile(nuka_path):
        print(f"[{label}] missing Nuka frame: {nuka_path}")
        continue
    nuka = np.asarray(Image.open(nuka_path).convert("RGB"), dtype=np.uint8)

    print(f"\n=== {label}: real LIBERO raw {raw.shape} vs Nuka saved {nuka.shape} ===")
    reference = gray_small(raw)
    scores = {}
    for name, fn in TRANSFORMS.items():
        candidate = gray_small(np.ascontiguousarray(fn(nuka)))
        scores[name] = float(np.dot(reference, candidate) / reference.size)
    for name, value in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  nuka[{name:8}] vs libero_raw  correlation = {value:+.4f}")
    print(f"  BEST = {max(scores, key=scores.get)}")

    best = max(scores, key=scores.get)
    combo = Image.new("RGB", (SIZE * 3, SIZE), "black")
    combo.paste(Image.fromarray(raw), (0, 0))
    combo.paste(Image.fromarray(nuka).resize((SIZE, SIZE)), (SIZE, 0))
    combo.paste(
        Image.fromarray(
            np.ascontiguousarray(TRANSFORMS[best](nuka)).astype(np.uint8)
        ).resize((SIZE, SIZE)),
        (SIZE * 2, 0),
    )
    combo.save(f"{OUT}/compare_{label}.png")

print("\nwrote", OUT)
env.close()
