#!/usr/bin/env python3
"""Drape a free cloth over a standing Go2 -- authored ENTIRELY from Python.

Builds the coupled Go2 + free-cloth world through the nuka authoring facade
(morphs + materials + surfaces -> Scene.build), parks the unpinned square sheet
flat above a PD-standing Go2 (cooked from go2.nks for its full per-link collision
skeleton) while the dog settles, RELEASES it to flutter down under anisotropic
aero drag and conform to the body on the ONE general body<->particle row solver,
and beauty-renders the LIVE world each frame with World.render_beauty. The cloth
params + the stand/drape choreography reproduce examples/demo/go2_cloth_drape_demo.cpp.

Pure Python over the prebuilt nuka module -- NO recompile. Writes
out/cloth_py/frame_####.png (and an mp4 when ffmpeg is on PATH).

Run from the repo root:
    python examples/demo/go2_cloth_drape.py
"""

import argparse
import os
import shutil
import subprocess

import numpy as np
from PIL import Image

import nuka
from nuka.author import Scene, SimOptions, materials, morphs, surfaces
from nuka.author.control import ClothDrape, StandHold
from nuka.author.render import hero_orbit, smoothstep

# Cloth lattice + stance + render the C++ go2_cloth_drape demo uses (reproduced).
SCENE = "examples/scenes/go2.nks"
NX, NY, SPACING = 55, 51, 0.024
STAND_BASE_Z, LIFT, PARK = 0.32, 0.58, 0.45
CLOTH_ORIGIN = (0.0, 0.0, STAND_BASE_Z + LIFT)
SETTLE, WIDTH, HEIGHT, SPP = 120, 1280, 720, 16


def main():
    ap = argparse.ArgumentParser(description="go2 cloth drape (python facade)")
    ap.add_argument("--out", default="out/cloth_py")
    ap.add_argument("--drape", type=int, default=800)
    ap.add_argument("--stride", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = nuka.Device.create(0)

    # Assemble the coupled world: the Go2 scene morph + a free cloth Grid with its
    # XPBD material + drape surface, on a baked flat heightfield (contact_family=1).
    grid = morphs.Grid(NX, NY, SPACING, origin=CLOTH_ORIGIN)
    scene = Scene(SimOptions(
        dt=1.0 / 240.0, env_count=1, contact_family=1,
        solver_vel_iters=80, solver_pos_iters=28, solver_contact_margin=0.044,
        solver_max_pairs=64, baumgarte_max_velocity=0.45))
    scene.add_entity(morphs.NKS(SCENE))
    scene.add_entity(
        grid,
        materials.Cloth.XPBD(mass=0.012, friction=1.8, bend_alpha=0.09, iters=80),
        surfaces.Cloth(free=True, aero=(30.0, 0.12, 0.16)),
        contact_radius=0.022)
    world = scene.build(dev)
    print(f"coupled world: links={world.base_link_count} particles={world.particle_count}")

    # Control choreography (the CONTROL layer, never on a morph): hold the crouch by
    # joint name; park the cloth clear, release it, apply a uniform fabric drag.
    stand = StandHold(
        world, base_z=STAND_BASE_Z, stiffness=60.0,
        targets=dict(thigh=0.80, calf=-1.50,
                     FL_hip=-0.10, FR_hip=0.10, RL_hip=-0.10, RR_hip=0.10))
    cloth = ClothDrape(world, grid, park_lift=PARK, damp=0.10)
    for _ in range(SETTLE):
        stand.hold(); cloth.park(); world.step()
    cloth.release()

    frames = []
    for s in range(args.drape):
        stand.hold(); world.step(); cloth.damp()
        if s % args.stride == 0 or s + 1 == args.drape:
            eye, look = hero_orbit(smoothstep(s / max(1, args.drape - 1)))
            img = world.render_beauty(eye=eye, look=look, width=WIDTH, height=HEIGHT, spp=SPP)
            path = os.path.join(args.out, f"frame_{len(frames):04d}.png")
            Image.fromarray(np.ascontiguousarray(img)).save(path)
            frames.append(path)
            print(f"frame {len(frames):2d} step {s:4d} -> {path}")

    world.destroy()
    dev.close()
    if frames and shutil.which("ffmpeg") is not None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i",
                        os.path.join(args.out, "frame_%04d.png"), "-pix_fmt", "yuv420p",
                        os.path.join(args.out, "go2_cloth_drape.mp4")], check=False)
    print(f"DONE: {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
