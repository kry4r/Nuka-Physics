#!/usr/bin/env python3
"""Validate the Panda Nuka asset, stable PD hold, and real visual meshes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
import nuka
from nuka.tasks.panda_pick_place import PANDA_C1_HOME, PandaPickPlaceController

SCENE_XML = REPO / ".nuka-assets/generated/panda/panda_pick_place.xml"
SCENE_NKS = REPO / ".nuka-assets/generated/panda/panda_pick_place.nks"


def ensure_scene_bundle() -> Path:
    nka = SCENE_NKS.with_suffix(".nka")
    stale = (
        not SCENE_NKS.exists() or not nka.exists()
        or SCENE_NKS.stat().st_mtime < SCENE_XML.stat().st_mtime
    )
    if stale:
        scene = nuka.Scene.load(str(SCENE_XML))
        try:
            scene.save(str(SCENE_NKS))
        finally:
            scene.destroy()
    return SCENE_NKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--out", default="out/panda_vla/asset_gate")
    args = parser.parse_args()
    output = REPO / args.out
    output.mkdir(parents=True, exist_ok=True)

    scene_path = ensure_scene_bundle()
    rows: list[dict[str, float]] = []
    with nuka.Device.create(0) as device:
        controller = PandaPickPlaceController(scene_path, device, dt=args.dt)
        try:
            initial = controller.q[0, 1:].detach().cpu().numpy().copy()
            for _ in range(int(args.seconds / args.dt)):
                row = controller.step()
                rows.append(row)
                if row["finite"] != 1.0:
                    raise RuntimeError("non-finite Panda state")
            final = controller.q[0, 1:].detach().cpu().numpy().copy()
            image = controller.world.render_beauty(
                eye=(-0.75, -1.35, 1.25), look=(0.38, 0.0, 0.62),
                fov_deg=42.0, width=960, height=540, spp=8)
            Image.fromarray(np.ascontiguousarray(image)).save(output / "panda_scene.png")
        finally:
            controller.close()

    drift = np.abs(final - np.asarray(PANDA_C1_HOME))
    summary = {
        "scene": str(scene_path.relative_to(REPO)),
        "visual_asset_mode": "nks_nka_real_panda_meshes",
        "steps": len(rows),
        "finite": all(row["finite"] == 1.0 for row in rows),
        "arm_hold_error_max_rad": float(drift[:7].max()),
        "finger_hold_error_max_m": float(drift[7:].max()),
        "joint_rmse_max_rad": float(max(row["joint_rmse_rad"] for row in rows)),
        "max_joint_speed": float(max(row["max_joint_speed"] for row in rows)),
        "initial_state": initial.tolist(),
        "final_state": final.tolist(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if (not summary["finite"] or summary["arm_hold_error_max_rad"] > 0.08
            or summary["finger_hold_error_max_m"] > 0.01):
        raise RuntimeError(f"Panda stable-hold gate failed: {summary}")


if __name__ == "__main__":
    main()
