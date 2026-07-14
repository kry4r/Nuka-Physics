#!/usr/bin/env python3
"""Render-timing probe: build, settle a little, render S0 at a couple of spp so
the beauty-render cost per frame is known before the full station pass."""
from __future__ import annotations
import os
import time
import numpy as np
from PIL import Image
import nuka
import bdx_oneshot_choreo as C

OUT = "/data/xtzhang25/_work/activate/out/oneshot_scene_v2"


def main():
    os.makedirs(OUT, exist_ok=True)
    with nuka.Device.create(0) as dev:
        w = C.build_world(dev)
        d = C.Driver(w)
        d.hold(300)
        for spp in (64, 256):
            t0 = time.time()
            img = w.render_beauty(eye=(-0.55, -0.70, 0.32), look=(0.30, 0.0, 0.25),
                                  fov_deg=40.0, width=1280, height=720, spp=spp)
            dt = time.time() - t0
            p = os.path.join(OUT, f"probe_S0_spp{spp}.png")
            Image.fromarray(np.ascontiguousarray(img)).save(p)
            print(f"[rprobe] spp={spp} {dt:.1f}s -> {p}", flush=True)
        w.destroy()


if __name__ == "__main__":
    main()
