#!/usr/bin/env python3
"""Headless smoke for the LIVE-WORLD render bridge (World.render_beauty).

Builds the SAME coupled cloth+Go2 world the go2_cloth_drape demo cooks (a free,
unpinned square cloth dropped over a PD-standing Go2 quadruped, with anisotropic
aerodynamic drag), drives it from Python with NO recompile -- seed the standing
crouch + hold the base, park the cloth flat above the dog while it settles, then
RELEASE it and let gravity drape it -- and beauty-renders the LIVE stepped world
each frame with World.render_beauty (the offline CUDA path-tracer, the shared
studio look). Writes out/render_bridge/frame_####.png; the early frames catch the
cloth mid-fall and the last frame the settled drape.

Run from the repo root (so examples/scenes/go2.nks resolves):
    python examples/demo/render_bridge_smoke.py
"""

import argparse
import math
import os

import numpy as np

import nuka

# ---- cloth + stance parameters (mirror go2_cloth_drape_demo.cpp) -------------
NX, NY = 55, 51                 # cloth lattice (~1.30 m x 1.20 m at 24 mm spacing).
SPACING = 0.024
PARTICLE_MASS = 0.012
FRICTION = 1.8                  # high Coulomb grip so the drape seats, not slides.
BEND_ALPHA = 0.09               # soft folds, no sharp kinks.
ITERS = 80
CONTACT_RADIUS = 0.022          # particle sphere radius (d_min = 44 mm).
STAND_BASE_Z = 0.32             # base height for the crouch.
LIFT = 0.58                     # release height above the crouch base.
PARK_LIFT = 0.45                # settle-time park height above the release lay.
AERO_N, AERO_T, AERO_CLAMP = 30.0, 0.12, 0.16   # anisotropic aero drag -> flutter.
CLOTH_DAMP = 0.10               # per-step uniform fabric drag.
DRIVE_STIFF = 60.0
# The Go2 standing crouch (link 0 = floating base; 1..12 = FL,FR,RL,RR x hip,thigh,calf).
STAND = [0.0, -0.10, 0.80, -1.50, 0.10, 0.80, -1.50,
         -0.10, 0.80, -1.50, 0.10, 0.80, -1.50]


def rest_grid(cz):
    """The flat rest lattice centred at the origin at height cz (row-major j*NX+i,
    matching the engine's cloth cook layout)."""
    xs = (np.arange(NX, dtype=np.float32) - 0.5 * (NX - 1)) * SPACING
    ys = (np.arange(NY, dtype=np.float32) - 0.5 * (NY - 1)) * SPACING
    g = np.zeros((NY, NX, 3), dtype=np.float32)
    g[..., 0] = xs[None, :]
    g[..., 1] = ys[:, None]
    g[..., 2] = cz
    return g.reshape(-1, 3)


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def hero_camera(e):
    """The demo's orbit: a wide side profile of the falling sheet (e=0) arcing to a
    tighter front-3/4 over the draped dog + the hem on the floor (e=1)."""
    az = -0.50 + 1.05 * e
    elev = math.radians(18.0 - 4.0 * e + 4.0 * math.sin(e * math.pi))
    radius = 1.95 - 0.55 * e
    look = (0.05, 0.0, 0.42 - 0.24 * e)
    eye = (look[0] + radius * math.cos(elev) * math.sin(az),
           look[1] - radius * math.cos(elev) * math.cos(az),
           look[2] + radius * math.sin(elev))
    return eye, look


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="examples/scenes/go2.nks")
    ap.add_argument("--out", default="out/render_bridge")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--spp", type=int, default=12)
    ap.add_argument("--settle", type=int, default=120)
    ap.add_argument("--drape", type=int, default=300)
    ap.add_argument("--stride", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from PIL import Image

    dev = nuka.Device.create(0)
    # The coupled cloth+robot world: a free drape (cloth_free) with aero flutter,
    # plus a flat baked heightfield (contact_family=1) the dog stands on and the
    # cloth hem pools on. Identical media to the C++ demo, authored from Python.
    world = nuka.World.create_coupled_from_scene(
        device=dev, scene_path=args.scene, env_count=1,
        contact_family=1, heightfield_terrain_type=0,
        cloth_nx=NX, cloth_ny=NY, cloth_spacing=SPACING,
        cloth_origin_x=0.0, cloth_origin_y=0.0, cloth_origin_z=STAND_BASE_Z + LIFT,
        cloth_particle_mass=PARTICLE_MASS, cloth_friction=FRICTION,
        cloth_bend_alpha=BEND_ALPHA, cloth_iters=ITERS, contact_radius=CONTACT_RADIUS,
        cloth_free=True, aero_normal=AERO_N, aero_tangent=AERO_T, aero_max_dv=AERO_CLAMP)

    L = int(world.base_link_count)
    P = int(world.particle_count)
    F = nuka.Field
    print(f"[render_bridge_smoke] coupled world: links={L} particles={P}")

    # Seed the standing crouch + the PD hold (the cooked motor actuators seed gain 0).
    q = np.zeros(L, dtype=np.float32)
    for i in range(min(L, len(STAND))):
        q[i] = STAND[i]
    stiff = np.zeros(L, dtype=np.float32)
    damp = np.zeros(L, dtype=np.float32)
    for i in range(1, min(L, 13)):
        stiff[i] = DRIVE_STIFF
        damp[i] = 2.0 * math.sqrt(DRIVE_STIFF)
    world.upload_field(F.JOINT_POSITION, q)
    world.upload_field(F.DRIVE_STIFFNESS, stiff)
    world.upload_field(F.DRIVE_DAMPING, damp)
    base = np.array(world.download_field(F.BASE_POSE, 0, 7), dtype=np.float32)
    base[2] = STAND_BASE_Z          # stand the floating base at the crouch height.
    root_vel_zero = np.zeros(6, dtype=np.float32)

    def hold_dog():
        # Holding the stance + base is a control input (the public upload_field), not
        # a solver fork: PD target, re-stamp the base pose, zero the root spatial vel.
        world.upload_field(F.DRIVE_TARGET, q)
        world.upload_field(F.BASE_POSE, base)
        world.upload_field(F.LINK_VELOCITY, root_vel_zero, 0)

    rest = rest_grid(STAND_BASE_Z + LIFT)
    park = rest.copy()
    park[:, 2] += PARK_LIFT
    zero_vel = np.zeros((P, 3), dtype=np.float32)
    # Release at a slight tilt + faint ripple so the anisotropic drag seeds flutter.
    release = rest.copy()
    tilt = math.tan(math.radians(4.0))
    release[:, 2] += (tilt * release[:, 0] +
                      0.018 * np.sin(3.0 * math.pi * release[:, 0]) *
                      np.cos(2.0 * math.pi * release[:, 1]))

    def set_cloth(pos):
        world.upload_field(F.PARTICLE_POSITION, pos.reshape(-1))
        world.upload_field(F.PARTICLE_VELOCITY, zero_vel.reshape(-1))

    def damp_cloth():
        v = np.array(world.download_field(F.PARTICLE_VELOCITY), dtype=np.float32)
        v *= (1.0 - CLOTH_DAMP)
        world.upload_field(F.PARTICLE_VELOCITY, v)

    # Settle the dog off-camera (cloth parked high so it does not touch the trunk).
    for _ in range(args.settle):
        hold_dog()
        set_cloth(park)
        world.step()
    print(f"[render_bridge_smoke] settled {args.settle} steps; releasing cloth")
    set_cloth(release)

    saved = []

    def render(step, name):
        e = smoothstep(step / max(1, args.drape - 1))
        eye, look = hero_camera(e)
        img = world.render_beauty(eye=eye, look=look, up=(0.0, 0.0, 1.0),
                                  fov_deg=40.0, width=args.width, height=args.height,
                                  spp=args.spp)
        path = os.path.join(args.out, name)
        Image.fromarray(np.ascontiguousarray(img)).save(path)
        nonblack = int((img.reshape(-1, 3).astype(np.int32).sum(1) > 16).sum())
        saved.append(path)
        print(f"[render_bridge_smoke] frame {len(saved):2d} step {step:4d} "
              f"-> {path} (nonblack={nonblack})")

    # Drape the released cloth over the standing dog, rendering the LIVE world.
    nframe = 0
    mid = args.drape // 2
    for s in range(args.drape):
        hold_dog()
        world.step()
        damp_cloth()
        if (s % args.stride) == 0 or s == mid or (s + 1) == args.drape:
            render(s, f"frame_{nframe:04d}.png")
            nframe += 1

    print(f"[render_bridge_smoke] DONE: {len(saved)} frames -> {args.out}")
    world.destroy()


if __name__ == "__main__":
    main()
