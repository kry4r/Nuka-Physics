"""Check left-right alignment of Nuka's policy input against real LIBERO."""

import numpy as np
from PIL import Image

RUN = ".nuka-runs/libero_pi05_osc_signfix"
REF = ".nuka-runs/libero_reference"


def policy_input(array: np.ndarray) -> np.ndarray:
    """Apply the LeRobot LIBERO processor's documented H/W 180-degree flip."""
    return np.ascontiguousarray(array[::-1, ::-1])


def col_profile(array: np.ndarray, bins: int = 12) -> np.ndarray:
    gray = np.asarray(Image.fromarray(array).convert("L"), dtype=float)
    return gray.mean(axis=0).reshape(bins, -1).mean(axis=1)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())


def dark_centroid(array: np.ndarray) -> tuple[float, float]:
    gray = np.asarray(Image.fromarray(array).convert("L"), dtype=float)
    threshold = np.percentile(gray, 3.0)
    ys, xs = np.nonzero(gray <= threshold)
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean() / gray.shape[1]), float(ys.mean() / gray.shape[0])


for label, nuka_name in (
    ("agentview", "initial_agentview.png"),
    ("eye_in_hand", "initial_eye_in_hand.png"),
):
    lib_raw = np.asarray(Image.open(f"{REF}/libero_{label}_raw.png").convert("RGB"))
    nuk_raw = np.asarray(Image.open(f"{RUN}/{nuka_name}").convert("RGB"))
    lib = policy_input(lib_raw)
    nuk = policy_input(nuk_raw)

    pl = col_profile(lib)
    pn = col_profile(nuk)
    pm = pn[::-1]
    print(f"=== {label} (policy-input orientation) ===")
    print("  libero cols   :", " ".join("%5.1f" % v for v in pl))
    print("  nuka   cols   :", " ".join("%5.1f" % v for v in pn))
    print("  mirrored nuka :", " ".join("%5.1f" % v for v in pm))
    print("  corr(libero, nuka)          = %+.4f" % corr(pl, pn))
    print("  corr(libero, mirrored nuka) = %+.4f" % corr(pl, pm))
    lx, ly = dark_centroid(lib)
    nx, ny = dark_centroid(nuk)
    print("  dark centroid libero (x,y) = (%.3f, %.3f)" % (lx, ly))
    print("  dark centroid nuka   (x,y) = (%.3f, %.3f)" % (nx, ny))
    print("  mirrored nuka x            = %.3f" % (1.0 - nx))
    print()
