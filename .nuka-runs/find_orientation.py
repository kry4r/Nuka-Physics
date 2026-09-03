"""Find which flip of Nuka's framebuffer matches real LIBERO's policy input."""

import numpy as np
from PIL import Image

REF = ".nuka-runs/libero_reference"
# This run's saved frames are rot180 of Nuka's raw framebuffer.
RUN = "out/libero/pi05_rot180_fix"

TRANSFORMS = {
    "identity": lambda a: a,
    "fliplr": lambda a: a[:, ::-1],
    "flipud": lambda a: a[::-1, :],
    "rot180": lambda a: a[::-1, ::-1],
}


def grid(array: np.ndarray, n: int = 16) -> np.ndarray:
    gray = np.asarray(Image.fromarray(array).convert("L").resize((n, n)), dtype=float)
    flat = gray.ravel()
    return (flat - flat.mean()) / (flat.std() + 1e-9)


for label, ref_name, run_name in (
    ("agentview", "libero_agentview_raw.png", "initial_agentview.png"),
    ("eye_in_hand", "libero_eye_in_hand_raw.png", "initial_eye_in_hand.png"),
):
    libero_raw = np.asarray(Image.open(f"{REF}/{ref_name}").convert("RGB"))
    # The reference rollout proved the policy wants LIBERO raw rotated 180.
    want = np.ascontiguousarray(libero_raw[::-1, ::-1])

    saved = np.asarray(Image.open(f"{RUN}/{run_name}").convert("RGB"))
    nuka_raw = np.ascontiguousarray(saved[::-1, ::-1])

    target = grid(want)
    print(f"=== {label}: Nuka raw framebuffer vs required policy input ===")
    scores = {}
    for name, fn in TRANSFORMS.items():
        cand = grid(np.ascontiguousarray(fn(nuka_raw)))
        scores[name] = float(np.dot(target, cand) / target.size)
    for name, value in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  flip(nuka_raw, {name:8}) correlation = {value:+.4f}")
    print(f"  BEST = {max(scores, key=scores.get)}\n")
