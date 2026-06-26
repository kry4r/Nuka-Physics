#!/usr/bin/env python3
"""Drop a soft tet ball onto a ground -- authored ENTIRELY from Python.

Builds a world through the nuka authoring facade: a flat baked ground (the general,
margin-aware particle-ground the cloth also pools onto, ``contact_family=1``) + a
``TetSphere`` soft body (an XPBD tet rest-lattice with distance + volume constraints)
authored above it, cooked through the general ``SceneBuilder`` media path
(``Scene.build``). The DROP is a CONTROL input (``SoftDrop`` holds the ball airborne,
then releases it, then a settle drag) -- never baked onto a morph. The ball falls
under gravity, the volume constraint squashes + recovers it on the ground through the
ONE general body<->particle row solver, and each frame beauty-renders the LIVE world
with ``World.render_beauty``: the deforming tet boundary surface (rebuilt from the
live particle field each frame) over a studio floor, hero-orbited like the cloth demo.

The contact is TIGHT: the authored ``SimOptions`` request a strong contact solve
(extra split-impulse position sweeps + a speculative margin) through the built-scene
SolverConfig overrides, so the ball rests with its bottom AT the floor (z = 0) with no
penetration -- not hidden below the studio floor. (An analytic box-slab floor cannot
give non-penetrating particle contact: a particle whose centre crosses into the box
gets a degenerate sideways push, so the ball sinks. The flat baked ground is the
margin-aware face contact the cloth already rests on -- ONE general path, no box.)

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
from nuka.author.control import SoftDrop
from nuka.author.render import smoothstep

# A ~0.10 m soft sphere dropped from this centre height onto the flat ground (z = 0);
# a gentle drop so the squash + recovery reads and the ball settles round, not warped.
RADIUS, CELLS, DROP_Z = 0.10, 12, 0.118
HOLD, WIDTH, HEIGHT, SPP = 8, 1280, 720, 16

# Tight-contact SolverConfig overrides threaded through the built-scene path: extra
# split-impulse position sweeps + a speculative margin expel penetration, Baumgarte-bounded.
SOLVER = dict(solver_vel_iters=96, solver_pos_iters=44, solver_contact_margin=0.045,
              solver_max_pairs=96, baumgarte_max_velocity=0.8)


def settle_damp(step):
    """Per-step settle drag: light through the fall + first squash (a natural impact),
    ramping heavier across the settle so the soft pulse dies and the ball comes to a
    clean rest on the ground. A control INPUT (uniform velocity drag), not a solver fork."""
    lo, hi, s0, s1 = 0.02, 0.10, 70, 160
    if step < s0:
        return lo
    if step >= s1:
        return hi
    return lo + (hi - lo) * (step - s0) / (s1 - s0)


def main():
    ap = argparse.ArgumentParser(description="soft tet ball drop (python facade)")
    ap.add_argument("--out", default="out/soft_py")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = nuka.Device.create(0)

    # A flat baked ground (contact_family=1, the cloth's margin-aware particle-ground) +
    # a tet-soft sphere above it (XPBD distance + volume) with a smoothing render skin.
    scene = Scene(SimOptions(dt=1.0 / 240.0, env_count=1, contact_family=1,
                             heightfield_terrain_type=0, **SOLVER))
    scene.add_entity(
        morphs.TetSphere((0.0, 0.0, DROP_Z), RADIUS, CELLS),
        materials.Soft.XPBD(mass=0.005, friction=0.6,
                            distance_alpha=3.0e-6, volume_alpha=6.0e-6, iters=44),
        surfaces.Soft(offset=0.002, smooth_iters=6, smooth_lambda=0.5))
    world = scene.build(dev)
    print(f"soft world: links={world.base_link_count} particles={world.particle_count}")

    # CONTROL layer: the drop -- hold the ball airborne, release it, then the settle
    # drag. The ground is baked static (no per-step pose pin needed).
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

    # Airborne hold, then drop + squash + settle.
    for _ in range(HOLD):
        drop.hold()
        world.step()
    drop.release()
    render(0)
    for s in range(1, total):
        world.step()
        drop.damp(settle_damp(s))
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
