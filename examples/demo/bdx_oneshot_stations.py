#!/usr/bin/env python3
"""Fast composition probe: settle once, brisk-glide the duck to each beauty
station and render it at low spp, so framing / lighting / prop layout can be
judged quickly before the slow-glide coupling render pass."""
from __future__ import annotations
import argparse
import os
import numpy as np
from PIL import Image
import nuka
import bdx_oneshot_author as A
import bdx_oneshot_choreo as C

OUT = "/data/xtzhang25/_work/activate/out/oneshot_scene_v2"
LIFT = A.SCENE_LIFT

# (name, duck target x, sink, eye, look, fov). Teleport (not glide) to each x.
STATIONS = [
    ("S0_establish", None, 0.0, (-0.52, -0.85, 0.46 + LIFT), (1.1, 0.0, 0.16 + LIFT), 48.0),
    ("S1_stairs_door_cloth", 0.18, 0.0, (0.62, -0.58, 0.22 + LIFT), (0.26, 0.0, 0.30 + LIFT), 42.0),
    ("S2_gravel_lowtrack", 1.60, -0.010, (1.58, -0.55, 0.11 + LIFT), (1.60, 0.12, 0.04 + LIFT), 42.0),
    ("S3_smallparts_macro", 2.72, 0.0, (2.86, -0.36, 0.10 + LIFT), (2.70, 0.0, 0.04 + LIFT), 45.0),
    ("S4_slab_strike", 3.35, -0.012, (3.00, -0.82, 0.32 + LIFT), (3.66, 0.0, 0.30 + LIFT), 47.0),
    ("S5_reverse_depth", None, 0.0, (4.62, -0.52, 0.36 + LIFT), (3.2, 0.05, 0.20 + LIFT), 46.0),
    ("S6_overhead", None, 0.0, (2.0, 0.0, 4.5), (2.0, 0.0001, 0.0 + LIFT), 60.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    keys = [k for k in args.only.split(",") if k]
    with nuka.Device.create(0) as dev:
        w = C.build_world(dev)
        d = C.Driver(w)
        d.hold(300)
        for name, tx, sink, eye, look, fov in STATIONS:
            if keys and not any(k in name for k in keys):
                continue
            if tx is not None:
                d.set_base(tx, 0.0, C.z_floor(tx) + A.BASE_OVER_FLOOR + sink)
                d.hold(90)  # let the PD stance + contact seat the feet
                if "gravel" in name or "smallparts" in name:
                    d.dip(0.02, 100)  # a stomp: print the gravel / kick the parts
                    d.hold(40)
            img = w.render_beauty(eye=eye, look=look, fov_deg=fov,
                                  width=1280, height=720, spp=args.spp)
            p = os.path.join(OUT, f"{args.tag}_{name}.png")
            Image.fromarray(np.ascontiguousarray(img)).save(p)
            print(f"[stations] {p}", flush=True)
        w.destroy()


if __name__ == "__main__":
    main()
