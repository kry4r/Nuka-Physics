"""Compare Nuka render statistics against the checkpoint's training image stats."""

import numpy as np
from PIL import Image
from safetensors.torch import load_file

STATS = ".nuka_cache/pi05-libero/policy_preprocessor_step_2_normalizer_processor.safetensors"
stats = load_file(STATS)

targets = {
    "observation.images.image": ".nuka-runs/libero_reference/libero_agentview_raw.png",
    "observation.images.image2": ".nuka-runs/libero_reference/libero_eye_in_hand_raw.png",
}
nuka_runs = {
    "observation.images.image": [
        (".nuka-runs/libero_pi05_osc_signfix/initial_agentview.png", "preview(old)"),
        ("out/libero/pi05_gripper_fix/initial_agentview.png", "preview(fixed)"),
    ],
    "observation.images.image2": [
        (".nuka-runs/libero_pi05_osc_signfix/initial_eye_in_hand.png", "preview(old)"),
        ("out/libero/pi05_gripper_fix/initial_eye_in_hand.png", "preview(fixed)"),
    ],
}


def rgb_stats(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    except FileNotFoundError:
        return None
    return array.reshape(-1, 3).mean(axis=0), array.reshape(-1, 3).std(axis=0)


for key, ref_path in targets.items():
    train_mean = stats[f"{key}.mean"].flatten().numpy()
    train_std = stats[f"{key}.std"].flatten().numpy()
    print(f"=== {key} ===")
    print("  training  mean %s  std %s" % (np.round(train_mean, 4), np.round(train_std, 4)))

    ref = rgb_stats(ref_path)
    if ref is not None:
        print("  LIBERO    mean %s  std %s" % (np.round(ref[0], 4), np.round(ref[1], 4)))

    for path, label in nuka_runs[key]:
        got = rgb_stats(path)
        if got is None:
            print(f"  {label:16} MISSING {path}")
            continue
        drift = np.abs(got[0] - train_mean).max()
        print(
            "  %-16s mean %s  std %s   |mean-train| max %.4f"
            % (label, np.round(got[0], 4), np.round(got[1], 4), drift)
        )
    print()
