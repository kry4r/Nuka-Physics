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

    L = A.SCENE_LIFT
    with nuka.Device.create(0) as dev:
        w = C.build_world(dev, args.scene)
        d = C.Driver(w)
        d.hold(320)  # drape forms, gravel compacts, slab stills.

        # S0 establish: duck on the platform looking down the whole corridor.
        shoot(w, "S0_establish", (-0.52, -0.85, 0.46 + L), (1.1, 0.0, 0.16 + L),
              fov=48.0)
        shoot(w, "S6_overhead_debug", (2.0, 0.0, 4.5), (2.0, 0.0001, 0.0 + L), fov=60.0)

        # S1: head emerging through the door having lifted the curtain (D01).
        d.glide(0.32, 0.35)
        d.hold(50)
        shoot(w, "S1_stairs_door_cloth", (0.72, -0.54, 0.30 + L), (0.30, 0.0, 0.34 + L),
              fov=40.0)

        # S2: mid-gravel with a stomped print behind (D02).
        d.glide(0.90, 0.6)
        d.glide(1.35, 0.5, sink=-0.010)
        d.dip(0.014, 150)
        d.glide(1.62, 0.5, sink=-0.010)
        d.hold(70)
        shoot(w, "S2_gravel_lowtrack", (1.50, -0.60, 0.16 + L), (1.66, 0.08, 0.12 + L),
              fov=44.0)

        # S3: macro on a foot amid the micro objects, a stomp kicking them (D03).
        d.glide(2.30, 0.7)
        d.glide(2.66, 0.3)
        d.dip(0.02, 90)
        d.hold(40)
        shoot(w, "S3_smallparts_macro", (2.98, -0.40, 0.17 + L), (2.60, 0.0, 0.11 + L),
              fov=46.0)

        # S4: head strikes the hanging slab then eases back so it swings clear (D04).
        d.glide(3.62, 0.7)
        d.glide(3.80, 0.3)
        d.glide(3.71, 0.5)
        d.hold(32)
        shoot(w, "S4_slab_strike", (3.30, -0.80, 0.30 + L), (3.78, 0.0, 0.30 + L),
              fov=45.0)

        # S5: reverse wide back down the corridor.
        shoot(w, "S5_reverse_depth", (4.62, -0.52, 0.36 + L), (3.2, 0.05, 0.20 + L),
              fov=46.0)
        w.destroy()


if __name__ == "__main__":
    main()
