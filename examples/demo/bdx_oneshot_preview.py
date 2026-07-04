#!/usr/bin/env python3
"""Beauty stations for the bdx_oneshot corridor. ONE continuous world: settle,
then walk the duck beat to beat with the choreo driver and render each station
at its action moment (the acceptance frames double as the one-shot spline's key
poses). Low grazing cameras throughout; every eye stays inside the corridor
sight-lines and above z=0.08.

    python examples/demo/bdx_oneshot_preview.py [--spp 512] [--only S1,S4]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image
import nuka
import bdx_oneshot_author as A
import bdx_oneshot_choreo as C

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spp", type=int, default=512)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--scene", default=OUT)
    ap.add_argument("--out", default="/data/xtzhang25/_work/activate/out/oneshot_scene_v2")
    ap.add_argument("--only", default="", help="comma-separated station keys")
    ap.add_argument("--fov", type=float, default=40.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    keys = [k for k in args.only.split(",") if k]

    def want(name):
        return not keys or any(k in name for k in keys)

    def shoot(w, name, eye, look, fov=None):
        if not want(name):
            return
        img = w.render_beauty(eye=eye, look=look, fov_deg=fov or args.fov,
                              width=args.width, height=args.height, spp=args.spp)
        path = os.path.join(args.out, f"{name}.png")
        Image.fromarray(np.ascontiguousarray(img)).save(path)
        print(f"[preview] {path}", flush=True)

    with nuka.Device.create(0) as dev:
        w = C.build_world(dev, args.scene)
        d = C.Driver(w)
        d.hold(500)  # drape forms, gravel compacts, slab stills.

        # S0 establish: duck on the platform, seen diagonally through the door.
        shoot(w, "S0_establish", (-0.55, -0.70, 0.32), (0.30, 0.0, 0.25))
        shoot(w, "S6_overhead_debug", (2.0, 0.0, 5.0), (2.0, 0.0001, 0.0), fov=55.0)

        # S1: on the step, head pushing the curtain up (D01).
        d.glide(0.20, 0.15)
        d.hold(60)
        shoot(w, "S1_stairs_door_cloth", (0.60, -0.50, 0.20), (0.28, 0.0, 0.32))

        # S2: mid-gravel with a stomped print behind (D02).
        d.glide(0.60, 0.15)
        d.glide(1.35, 0.25, sink=-0.010)
        d.dip(0.014, 160)
        d.glide(1.62, 0.25, sink=-0.010)
        d.hold(80)
        shoot(w, "S2_gravel_lowtrack", (1.62, -0.52, 0.10), (1.55, 0.10, 0.05))

        # S3: macro on the feet amid the micro objects (D03).
        d.glide(2.30, 0.3)
        d.glide(2.72, 0.5)
        d.hold(60)
        shoot(w, "S3_smallparts_macro", (2.82, -0.34, 0.10), (2.70, 0.0, 0.05),
              fov=45.0)

        # S4: head against the hanging slab (D04); slab nudged mid-contact.
        d.glide(3.60, 0.5)
        d.glide(3.82, 0.3)
        d.hold(30)
        shoot(w, "S4_slab_strike", (3.42, -0.85, 0.34), (3.80, 0.0, 0.40))

        # S5: reverse wide back down the corridor.
        shoot(w, "S5_reverse_depth", (4.42, 0.42, 0.28), (3.55, 0.0, 0.30))
        w.destroy()


if __name__ == "__main__":
    main()
