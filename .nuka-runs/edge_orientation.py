"""Match Nuka's framebuffer to LIBERO using lighting-invariant edge structure."""

import numpy as np
from PIL import Image

REF = ".nuka-runs/libero_reference"
RUN = "out/libero/pi05_rot180_fix"

TRANSFORMS = {
    "identity": lambda a: a,
    "fliplr": lambda a: a[:, ::-1],
    "flipud": lambda a: a[::-1, :],
    "rot180": lambda a: a[::-1, ::-1],
}


def edge_signature(array: np.ndarray, n: int = 24) -> np.ndarray:
    """Sobel gradient magnitude, pooled to n x n and standardized."""
    gray = np.asarray(Image.fromarray(array).convert("L"), dtype=float)
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    pooled = np.asarray(
        Image.fromarray(magnitude.astype(np.float32)).resize((n, n), Image.BILINEAR),
        dtype=float,
    ).ravel()
    return (pooled - pooled.mean()) / (pooled.std() + 1e-9)


for label, ref_name, run_name in (
    ("agentview", "libero_agentview_raw.png", "initial_agentview.png"),
    ("eye_in_hand", "libero_eye_in_hand_raw.png", "initial_eye_in_hand.png"),
):
    libero_raw = np.asarray(Image.open(f"{REF}/{ref_name}").convert("RGB"))
    saved = np.asarray(Image.open(f"{RUN}/{run_name}").convert("RGB"))
    # This run wrote rot180 of the raw framebuffer for both cameras.
    nuka_raw = np.ascontiguousarray(saved[::-1, ::-1])

    print(f"=== {label}: which flip maps Nuka raw onto LIBERO raw ===")
    target = edge_signature(libero_raw)
    scores = {}
    for name, fn in TRANSFORMS.items():
        scores[name] = float(
            np.dot(target, edge_signature(np.ascontiguousarray(fn(nuka_raw))))
            / target.size
        )
    for name, value in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  nuka_raw[{name:8}] vs libero_raw  edge corr = {value:+.4f}")
    best = max(scores, key=scores.get)
    runner = sorted(scores.values(), reverse=True)[1]
    print(f"  BEST = {best}  (margin over 2nd: {scores[best] - runner:+.4f})\n")
