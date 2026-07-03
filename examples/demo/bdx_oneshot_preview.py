#!/usr/bin/env python3
"""Beauty-preview the bdx_oneshot scene: cook it (optionally with the D01 cloth or
the D02 granular medium), settle a few hundred steps, then offline-render a set of
composition frames along the +x walkway (1280x720, high spp) with the authored
textured-PBR materials + HDRI environment.

    python examples/demo/bdx_oneshot_preview.py [--media none|cloth|granular]
        [--spp 160] [--settle 250] [--out DIR]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image
import nuka
from bdx_oneshot_author import add_cloth, add_granular

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nks")

# (name, eye, look) camera stations along the walkway; eye/look in world metres.
VIEWS = [
    ("00_establish",  (2.4, -3.1, 1.7),  (2.8, 0.0, 0.20)),
    ("01_zoneA_stairs_door", (0.55, -1.75, 0.50), (0.30, 0.0, 0.24)),
    ("02_zoneB_trough", (2.2, -1.55, 0.48), (2.25, 0.0, 0.04)),
    ("03_zoneC_rocks", (3.78, -0.72, 0.26), (3.78, 0.0, 0.02)),
    ("04_zoneD_rope",  (5.0, -2.05, 0.52), (5.0, 0.05, 0.42)),
    ("05_travel_low",  (1.0, -2.0, 0.40), (3.2, 0.0, 0.12)),
    ("06_overhead_debug", (3.0, 0.0, 5.5), (3.0, 0.0001, 0.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", choices=["none", "cloth", "granular"], default="none")
    ap.add_argument("--spp", type=int, default=160)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--settle", type=int, default=250)
    ap.add_argument("--scene", default=OUT)
    ap.add_argument("--out", default="/data/xtzhang25/_work/activate/out/oneshot_scene")
    ap.add_argument("--only", default="", help="comma-separated view name substrings")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with nuka.Device.create(0) as dev:
        b = nuka.SceneBuilder.create(args.scene)
        if args.media == "cloth":
            add_cloth(b)
        elif args.media == "granular":
            add_granular(b)
        w = b.build(dev, env_count=1, dt=1.0 / 240.0, control_mode=0)
        b.destroy()
        for _ in range(args.settle):
            w.step()
        views = VIEWS
        if args.only:
            keys = [k for k in args.only.split(",") if k]
            views = [v for v in VIEWS if any(k in v[0] for k in keys)]
        for name, eye, look in views:
            img = w.render_beauty(eye=eye, look=look, width=args.width,
                                  height=args.height, spp=args.spp)
            path = os.path.join(args.out, f"{args.media}_{name}.png")
            Image.fromarray(np.ascontiguousarray(img)).save(path)
            print(f"[preview] {path}")
        w.destroy()


if __name__ == "__main__":
    main()
