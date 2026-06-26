#!/usr/bin/env python3
"""Drop a soft tet ball onto a ground -- authored ENTIRELY from Python.

Builds a coupled world through the nuka authoring facade: a static floor slab (the
PHYSICS ground) + a ``TetSphere`` soft body (an XPBD tet rest-lattice with distance
+ volume constraints) authored above it, cooked through the general ``SceneBuilder``
media path (``Scene.build``). The DROP is a CONTROL input (``SoftDrop`` holds the
ball airborne, then releases it) and the floor is held by a CONTROL pose-pin
(``PinBody``) -- never baked onto a morph. The ball falls under gravity, the volume
constraint squashes + recovers it on the floor through the ONE general
body<->particle row solver, and each frame beauty-renders the LIVE world with
``World.render_beauty``: the deforming tet boundary surface (rebuilt from the live
particle field each frame) over a studio floor, hero-orbited like the cloth demo.

Pure Python over the prebuilt nuka module -- NO recompile. Writes
out/soft_py/frame_####.png (and an mp4 when ffmpeg is on PATH).

Run from the repo root:
    python examples/demo/soft_ball_py.py
"""

import argparse
import math
import os
import shutil
import subprocess

import numpy as np
from PIL import Image

import nuka
from nuka.author import Scene, SimOptions, materials, morphs, surfaces
from nuka.author.control import PinBody, SoftDrop
from nuka.author.render import smoothstep

# A ~0.10 m soft sphere dropped from this centre height onto a floor slab whose top
# is z = 0. A coarse tet cage (its boundary faces render, smoothed by the render
# skin); a light, gentle drop so the squash + recovery reads on the floor.
RADIUS, CELLS, DROP_Z = 0.10, 10, 0.20
FLOOR_HALF_Z = 0.5            # a thick slab so the ball cannot squeeze out the bottom.
HOLD, WIDTH, HEIGHT, SPP = 8, 1280, 720, 16


def main():
    ap = argparse.ArgumentParser(description="soft tet ball drop (python facade)")
    ap.add_argument("--out", default="out/soft_py")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = nuka.Device.create(0)

    # Assemble the coupled world: a static box-slab floor (top at z = 0, the physics
    # ground) + a tet-soft sphere authored above it (XPBD distance + volume), with a
    # render skin (a thin outward inflation + Laplacian relax) so the coarse tet
    # boundary reads smooth. The general SceneBuilder path runs at default (Earth)
    # gravity. (A plane primitive has a degenerate broadphase AABB for particle
    # contact, so the floor is a finite slab; studio_beauty draws its own floor.)
    scene = Scene(SimOptions(dt=1.0 / 240.0, env_count=1))
    scene.add_entity(morphs.Box((1.5, 1.5, FLOOR_HALF_Z), pos=(0.0, 0.0, -FLOOR_HALF_Z)),
                     materials.Rigid(friction=0.8, static=True))
    scene.add_entity(
        morphs.TetSphere((0.0, 0.0, DROP_Z), RADIUS, CELLS),
        materials.Soft.XPBD(mass=0.0015, friction=0.6,
                            distance_alpha=3.0e-6, volume_alpha=1.0e-6, iters=40),
        surfaces.Soft(offset=0.004, smooth_iters=6, smooth_lambda=0.5))
    world = scene.build(dev)
    print(f"soft world: links={world.base_link_count} particles={world.particle_count}")

    # CONTROL layer: pin the static floor slab at its authored pose each step (a
    # built-scene static body carries no seeded device pose), and the drop -- hold the
    # ball airborne, release it, then a gentle settle drag.
    floor = PinBody(world, pos=(0.0, 0.0, -FLOOR_HALF_Z), body_index=0)
    drop = SoftDrop(world)

    frames = []
    total = args.steps

    def render(step):
        e = smoothstep(step / max(1, total - 1))
        cx, cy, cz = drop.centroid()
        # A tight hero orbit that TRACKS the live ball centre: side profile easing to
        # a front-three-quarter, pulling in as it settles (cloth-demo arc, ball scale).
        az = -0.45 + 0.70 * e
        elev = math.radians(17.0 - 3.0 * e + 4.0 * math.sin(e * math.pi))
        r = 0.82 - 0.10 * e
        look = (cx, cy, max(RADIUS * 0.8, cz))
        eye = (look[0] + r * math.cos(elev) * math.sin(az),
               look[1] - r * math.cos(elev) * math.cos(az),
               look[2] + r * math.sin(elev))
        img = world.render_beauty(eye=eye, look=look, width=WIDTH, height=HEIGHT, spp=SPP)
        path = os.path.join(args.out, f"frame_{len(frames):04d}.png")
        Image.fromarray(np.ascontiguousarray(img)).save(path)
        frames.append(path)
        nonbg = int((np.asarray(img).reshape(-1, 3).max(axis=1) > 24).sum())
        print(f"frame {len(frames):2d} step {step:4d} centroid_z={cz:+.4f} "
              f"nonbg={nonbg} -> {path}")

    # Airborne hold, then drop + squash + settle. The floor is pinned EVERY step.
    for _ in range(HOLD):
        floor.pin()
        drop.hold()
        world.step()
    drop.release()
    render(0)
    for s in range(1, total):
        floor.pin()
        world.step()
        drop.damp(0.02)
        if s % args.stride == 0 or s + 1 == total:
            render(s)

    world.destroy()
    dev.close()
    if frames and shutil.which("ffmpeg") is not None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i",
                        os.path.join(args.out, "frame_%04d.png"), "-pix_fmt", "yuv420p",
                        os.path.join(args.out, "soft_ball_py.mp4")], check=False)
    print(f"DONE: {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
