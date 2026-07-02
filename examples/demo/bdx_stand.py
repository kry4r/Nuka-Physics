#!/usr/bin/env python3
"""PD-hold stand of the BDX biped (open_duck_mini_v2) -- pure Python, no recompile.

Loads examples/scenes/bdx_stand.nks (MJCF-imported duck + capsule sole colliders +
a flat heightfield floor + the home stance baked as the cooked IC) through the ONE
general Scene -> CookToModel -> nk::World path, applies the sts3215 servo gains
over the public field I/O (upload_field / download_field), free-steps 5 s, gates
base height / uprightness / NaN / foot penetration, and beauty-renders a still
with World.render_beauty.

Run from the repo root:
    python examples/demo/bdx_stand.py [--seconds 5] [--out out/bdx_stand.png]
"""

import argparse
import json
import os

import numpy as np

import nuka

SCENE = "examples/scenes/bdx_stand.nks"
KP, KD, FORCE_LIMIT = 13.37, 0.0, 3.23   # sts3215 position servo (kv=0 in MJCF).
SPAWN_Z, Z_TOL, UP_MIN, PEN_MM = 0.21, 0.05, 0.98, 2.0


def foot_capsules(nks_path):
    """The two sole-capsule (local_pos, radius) pairs, tree order = left, right."""
    caps = []
    def walk(node):
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        shape = node.get("collision_shape")
        if node.get("name") == "foot_contact" and shape:
            caps.append((np.array(shape["local"]["pos"], np.float32),
                         float(shape["radius"])))
        walk(node.get("children", []))
    walk(json.load(open(nks_path))["tree"])
    return caps


def quat_rotate(q, v):
    """Rotate v by the w-first quaternion q (both numpy)."""
    w, x, y, z = q
    u = np.array([x, y, z], np.float32)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def main():
    ap = argparse.ArgumentParser(description="BDX PD-hold stand gate (python)")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--out", default="out/bdx_stand.png")
    args = ap.parse_args()

    dev = nuka.Device.create(0)
    world = nuka.World.create_from_scene(dev, SCENE, env_count=1, dt=1.0 / 240.0)
    field = nuka.Field
    blc = int(world.base_link_count)
    print(f"[cook] base_link_count={blc} action_dim={world.action_dim} "
          f"dofs=6+{world.action_dim}={6 + int(world.action_dim)}")
    print(f"[cook] dof_names={world.dof_names()}")

    # sts3215 servo: Kp on every actuated slot, Kd=0 (the passive MJCF joint
    # damping 0.56 is applied by the engine), target = the cooked home stance.
    q0 = np.asarray(world.download_field(field.JOINT_POSITION), np.float32)[:blc]
    gains = np.zeros(blc, np.float32); gains[1:] = KP
    damps = np.zeros(blc, np.float32); damps[1:] = KD
    limits = np.zeros(blc, np.float32); limits[1:] = FORCE_LIMIT
    world.upload_field(field.DRIVE_STIFFNESS, gains)
    world.upload_field(field.DRIVE_DAMPING, damps)
    world.upload_field(field.DRIVE_FORCE_LIMIT, limits)
    world.upload_field(field.DRIVE_TARGET, q0)

    # Foot links by DOF name (the ankle joints' child links), sole capsules from
    # the scene asset; the capsule axis is horizontal so bottom = centre_z - r.
    feet = [int(world.dof_index("^left_ankle$")[0]),
            int(world.dof_index("^right_ankle$")[0])]
    caps = foot_capsules(SCENE)

    def foot_bottom_z():
        lp = np.asarray(world.download_field(field.ARTICULATION_LINK_POSE),
                        np.float32).reshape(blc, 7)
        lows = []
        for (link, (lpos, radius)) in zip(feet, caps):
            p, q = lp[link, :3], lp[link, 3:]
            lows.append(float((p + quat_rotate(q, lpos))[2] - radius))
        return min(lows)

    def base_state():
        b = np.asarray(world.download_field(field.BASE_POSE, 0, 7), np.float32)
        return float(b[2]), float(1.0 - 2.0 * (b[4] ** 2 + b[5] ** 2))  # z, up.z

    z0, up0 = base_state()
    print(f"[spawn] base_z={z0:.4f} up_dot={up0:.4f} foot_bottom_z={foot_bottom_z():.4f}")

    steps = int(round(args.seconds * 240.0))
    min_foot, any_nan = foot_bottom_z(), False
    for i in range(steps):
        world.step()
        min_foot = min(min_foot, foot_bottom_z())
        if (i + 1) % max(1, steps // 10) == 0 or i + 1 == steps:
            z, up = base_state()
            q = np.asarray(world.download_field(field.JOINT_POSITION), np.float32)
            any_nan |= not np.isfinite(q).all()
            print(f"  t={(i + 1) / 240.0:5.3f}s base_z={z:.4f} up_dot={up:.4f} "
                  f"foot_bottom_z={foot_bottom_z():.4f}")

    zf, upf = base_state()
    pen_mm = max(0.0, -min_foot) * 1000.0
    gates = {
        f"height(|{zf:.4f}-{SPAWN_Z}|<={Z_TOL})": abs(zf - SPAWN_Z) <= Z_TOL,
        f"upright({upf:.4f}>{UP_MIN})": upf > UP_MIN,
        "no-nan": not any_nan and np.isfinite(zf),
        f"penetration({pen_mm:.3f}mm<{PEN_MM}mm)": pen_mm < PEN_MM,
    }
    for name, ok in gates.items():
        print(f"[gate b] {name}: {'PASS' if ok else 'FAIL'}")
    print(f"[gate b] STAND {'PASS' if all(gates.values()) else 'FAIL'}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img = world.render_beauty(eye=(0.62, -0.55, 0.38), look=(0.0, 0.0, 0.14),
                              fov_deg=38.0, width=1280, height=720, spp=32)
    try:
        from PIL import Image
        Image.fromarray(np.asarray(img)).save(args.out)
    except ImportError:  # PPM fallback keeps the still verifiable without PIL.
        raw = np.asarray(img)
        with open(args.out + ".ppm", "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (raw.shape[1], raw.shape[0]))
            f.write(raw.tobytes())
    print(f"[render] wrote {args.out}")

    world.destroy()
    dev.close()
    return 0 if all(gates.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
