#!/usr/bin/env python3
"""Render a declarative ``.nks`` media scene -- a GENERIC loader, no demo code.

A ``.nks`` asset declares the WHOLE media scene (rigid primitives + cloth / soft-tet
/ fluid media + their materials and render skins). This loader reads ANY such asset
through ``nuka.SceneBuilder.create(<scene>.nks)`` -- the SAME load path a file scene
uses, now carrying the declared media -- cooks it through the ONE
``CookSceneToModel`` (``builder.build``), then steps and beauty-renders the live
world each ``--stride`` frames. The camera tracks the live media centroid, so the
loader is scene-agnostic: it draws whatever media the asset declares.

Editing the ``.nks`` (the ball radius, the drop height, the material, the solver
iters) changes the rendered result with NO recompile -- the asset IS the demo.

The floor a soft body / fluid rests on is the create-time flat heightfield
(``contact_family=1``): the margin-aware particle ground, a world-desc knob that is
NOT part of the scene asset. The tight-contact ``solver_*`` overrides are likewise
create-time. Both are CLI flags here, defaulting to the soft-body demo's settings.

Run from the repo root:
    python examples/demo/nks_render_demo.py --scene examples/scenes/soft_ball.nks
"""

import argparse
import math
import os
import shutil
import subprocess

import numpy as np
from PIL import Image

import nuka


def hero_camera(centroid, elapsed, radius):
    """A hero orbit that tracks the media centroid: a side profile easing to a
    front-three-quarter, pulling in as the scene settles. Pure framing, no physics."""
    cx, cy, cz = centroid
    az = -0.45 + 0.70 * elapsed
    elev = math.radians(17.0 - 3.0 * elapsed + 4.0 * math.sin(elapsed * math.pi))
    r = radius * (1.0 - 0.12 * elapsed)
    look = (cx, cy, max(0.04, cz))
    eye = (look[0] + r * math.cos(elev) * math.sin(az),
           look[1] - r * math.cos(elev) * math.cos(az),
           look[2] + r * math.sin(elev))
    return eye, look


def main():
    ap = argparse.ArgumentParser(description="declarative .nks media render loader")
    ap.add_argument("--scene", default="examples/scenes/soft_ball.nks",
                    help="the .nks asset to load + render")
    ap.add_argument("--out", default="out/nks_demo")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--orbit", type=float, default=0.82, help="camera orbit radius (m)")
    # Create-time world knobs (NOT part of the scene asset): the particle-ground
    # heightfield + the tight-contact solver overrides.
    ap.add_argument("--contact_family", type=int, default=1)
    ap.add_argument("--solver_vel_iters", type=int, default=96)
    ap.add_argument("--solver_pos_iters", type=int, default=44)
    ap.add_argument("--solver_contact_margin", type=float, default=0.045)
    ap.add_argument("--solver_max_pairs", type=int, default=96)
    ap.add_argument("--baumgarte_max_velocity", type=float, default=0.8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = nuka.Device.create(0)
    builder = nuka.SceneBuilder.create(args.scene)
    try:
        world = builder.build(
            dev, env_count=1, dt=1.0 / 240.0, contact_family=int(args.contact_family),
            heightfield_terrain_type=0,
            solver_vel_iters=int(args.solver_vel_iters),
            solver_pos_iters=int(args.solver_pos_iters),
            solver_contact_margin=float(args.solver_contact_margin),
            solver_max_pairs=int(args.solver_max_pairs),
            baumgarte_max_velocity=float(args.baumgarte_max_velocity))
    finally:
        builder.destroy()
    print(f"loaded {args.scene}: particles={world.particle_count}")
    if world.particle_count == 0:
        raise SystemExit(f"{args.scene} declares no media particles")

    def centroid():
        p = np.asarray(world.download_field(nuka.Field.PARTICLE_POSITION),
                       dtype=np.float32).reshape(-1, 3)
        return tuple(float(v) for v in p.mean(axis=0))

    frames = []

    def render(step):
        e = step / max(1, args.steps - 1)
        e = e * e * (3.0 - 2.0 * e)  # smoothstep
        cen = centroid()
        eye, look = hero_camera(cen, e, args.orbit)
        img = world.render_beauty(eye=eye, look=look, width=args.width,
                                  height=args.height, spp=args.spp)
        path = os.path.join(args.out, f"frame_{len(frames):04d}.png")
        Image.fromarray(np.ascontiguousarray(img)).save(path)
        frames.append(path)
        nonbg = int((np.asarray(img).reshape(-1, 3).max(axis=1) > 24).sum())
        print(f"frame {len(frames):3d} step {step:4d} centroid_z={cen[2]:+.4f} "
              f"nonbg={nonbg} -> {path}")

    render(0)
    for step in range(1, args.steps):
        world.step()
        if step % args.stride == 0:
            render(step)

    # Stitch an mp4 when ffmpeg is on PATH (optional; the frames are the artifact).
    if shutil.which("ffmpeg") and frames:
        mp4 = os.path.join(args.out, "render.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "24", "-i",
             os.path.join(args.out, "frame_%04d.png"), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", mp4],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"wrote {mp4}")
    print(f"wrote {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
