#!/usr/bin/env python3
"""Continuous one-shot film of the bdx corridor: one settle, then an unbroken
walk the length of the walkway while a keyframed camera dollies through each
beat (door curtain, gravel imprint, kicked metal debris, hanging-slab strike).

A numbered beauty frame is written every sim tick-group. `--shard i/n` splits
the frames across processes: each shard runs the identical deterministic sim
but only renders the frames it owns, so two GPUs halve the wall-clock."""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

import nuka
import bdx_oneshot_author as A
import bdx_oneshot_choreo as C

OUTROOT = "/data/xtzhang25/_work/activate/out/oneshot_film"
LIFT = A.SCENE_LIFT
DT = C.DT


def _k(x, eye, look, fov):
    return (float(x), np.array(eye, np.float32), np.array(look, np.float32), float(fov))


# Camera spine: (duck_x, eye, look, fov_deg), dollying on the open -y lane so the
# duck stays framed while each beat reveals in turn.
KEYS = [
    _k(-0.20, (-0.50, -0.95, 0.52 + LIFT), (0.55, 0.00, 0.26 + LIFT), 52.0),   # establish wide
    _k(0.15, (0.00, -1.00, 0.48 + LIFT), (0.65, 0.00, 0.30 + LIFT), 50.0),     # door: outside on -y, down corridor
    _k(0.75, (0.62, -0.90, 0.44 + LIFT), (1.15, 0.00, 0.28 + LIFT), 48.0),     # emerged into the corridor
    _k(1.45, (1.30, -0.82, 0.40 + LIFT), (1.85, 0.05, 0.26 + LIFT), 47.0),     # gravel track, whole duck
    _k(2.15, (2.02, -0.70, 0.34 + LIFT), (2.55, 0.00, 0.20 + LIFT), 47.0),     # approach the debris
    _k(2.68, (2.55, -0.55, 0.28 + LIFT), (2.95, 0.00, 0.15 + LIFT), 47.0),     # debris macro, metal-bead bed
    _k(3.18, (3.05, -0.78, 0.36 + LIFT), (3.55, 0.00, 0.28 + LIFT), 46.0),     # rise toward the slab
    _k(3.74, (3.45, -0.90, 0.48 + LIFT), (3.90, 0.00, 0.42 + LIFT), 46.0),     # slab strike, look up
    _k(4.05, (4.50, -0.62, 0.50 + LIFT), (3.70, 0.05, 0.30 + LIFT), 50.0),     # pull back, look back
]


def _smooth(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def camera_at(x):
    if x <= KEYS[0][0]:
        return KEYS[0][1], KEYS[0][2], KEYS[0][3]
    if x >= KEYS[-1][0]:
        return KEYS[-1][1], KEYS[-1][2], KEYS[-1][3]
    for i in range(len(KEYS) - 1):
        x0, e0, l0, f0 = KEYS[i]
        x1, e1, l1, f1 = KEYS[i + 1]
        if x0 <= x <= x1:
            t = _smooth((x - x0) / max(1e-6, x1 - x0))
            return (1 - t) * e0 + t * e1, (1 - t) * l0 + t * l1, (1 - t) * f0 + t * f1
    return KEYS[-1][1], KEYS[-1][2], KEYS[-1][3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--fps", type=float, default=24.0)   # rendered frames per sim second
    ap.add_argument("--speed", type=float, default=0.28)  # duck advance speed (m/s)
    ap.add_argument("--x-end", type=float, default=4.05)
    ap.add_argument("--settle", type=int, default=300)
    ap.add_argument("--tag", default="film")
    ap.add_argument("--shard", default="0/1")            # i/n : own frames where idx%n==i
    ap.add_argument("--sweep", type=int, default=0)      # >0: teleport-sweep N x's (camera check)
    ap.add_argument("--x-start", type=float, default=-0.20)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    si, sn = (int(v) for v in args.shard.split("/"))
    outdir = os.path.join(OUTROOT, f"frames_{args.tag}")
    os.makedirs(outdir, exist_ok=True)

    every = max(1, int(round(1.0 / (args.fps * DT))))    # sim steps between rendered frames
    step_i = [0]

    with nuka.Device.create(0) as dev:
        w = C.build_world(dev)
        d = C.Driver(w)
        d.hold(args.settle)  # drape forms, gravel compacts, slab comes to rest

        def on_step(_x):
            i = step_i[0]
            step_i[0] += 1
            if i % every:
                return
            frame = i // every
            if frame % sn != si:
                return
            eye, look, fov = camera_at(d.pos[0])
            img = w.render_beauty(eye=tuple(float(v) for v in eye),
                                  look=tuple(float(v) for v in look),
                                  fov_deg=float(fov), width=args.width,
                                  height=args.height, spp=args.spp)
            p = os.path.join(outdir, f"f_{frame:05d}.png")
            Image.fromarray(np.ascontiguousarray(img)).save(p)
            print(f"[film] frame {frame} x={d.pos[0]:.3f} -> {p}", flush=True)

        if args.sweep > 0:
            # Teleport-sweep: static scene, camera path framing check at N x's.
            for i in range(args.sweep):
                x = args.x_start + (args.x_end - args.x_start) * i / (args.sweep - 1)
                d.set_base(x, 0.0, C.z_floor(x) + A.BASE_OVER_FLOOR)
                d.hold(24)  # seat the stance + march the gait pose in place
                eye, look, fov = camera_at(x)
                img = w.render_beauty(eye=tuple(float(v) for v in eye),
                                      look=tuple(float(v) for v in look),
                                      fov_deg=float(fov), width=args.width,
                                      height=args.height, spp=args.spp)
                p = os.path.join(outdir, f"sweep_{i:03d}.png")
                Image.fromarray(np.ascontiguousarray(img)).save(p)
                print(f"[film] sweep {i} x={x:.3f} -> {p}", flush=True)
        else:
            if abs(args.x_start - d.pos[0]) > 0.02:
                d.set_base(args.x_start, 0.0, C.z_floor(args.x_start) + A.BASE_OVER_FLOOR)
                d.hold(60)  # seat the stance at the teleported glide start
            d.glide(args.x_end, args.speed, on_step=on_step)
        w.destroy()
    print(f"[film] shard {si}/{sn} done", flush=True)


if __name__ == "__main__":
    main()
