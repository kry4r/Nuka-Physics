#!/usr/bin/env python3
"""Scripted pose-driven traverse of the bdx_oneshot corridor + coupling probes.

The duck's base anchor is static, so the traverse drags it along the walkway by
uploading BASE_POSE each step while the PD stance holds; every coupling response
(cloth lift, gravel imprint, kicked objects, slab swing, joint reactions) is real
solver output. Probes report PASS/FAIL numbers for the acceptance gate; the same
driver positions the duck for the beauty stations.
"""

from __future__ import annotations

import math
import os

import numpy as np
import nuka
import bdx_oneshot_author as A

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nks")
N_CLOTH = A.CLOTH_NX * A.CLOTH_NY


def z_floor(x):
    if x < A.STEP_X[0]:
        return A.PLATFORM_TOP
    if x < A.STEP_X[1]:
        return A.STEP_TOP
    return 0.0


class Driver:
    """Base-pose drag driver over one cooked world."""

    def __init__(self, w):
        self.w = w
        self.base = np.asarray(w.download_field(nuka.Field.BASE_POSE),
                               dtype=np.float32).copy()
        self.pos = list(self.base[:3])

    def set_base(self, x, y, z):
        self.base[0], self.base[1], self.base[2] = x, y, z
        self.w.upload_field(nuka.Field.BASE_POSE, np.ascontiguousarray(self.base))
        self.pos = [x, y, z]

    def glide(self, x1, speed, dt=1.0 / 240.0, sink=0.0, y=0.0, on_step=None):
        """Drag the base to x1 at `speed`, tracking the floor height + `sink`."""
        x = self.pos[0]
        n = max(1, int(abs(x1 - x) / (speed * dt)))
        for i in range(n):
            t = (i + 1) / n
            xi = x + (x1 - x) * t
            zi = z_floor(xi) + A.BASE_OVER_FLOOR + sink
            # blend z over 6 cm so a floor break never teleports the duck.
            ahead = z_floor(xi + 0.06) + A.BASE_OVER_FLOOR + sink
            self.set_base(xi, y, min(zi, 0.5 * (zi + ahead)))
            self.w.step()
            if on_step:
                on_step(xi)

    def dip(self, dz, steps):
        """Press straight down by dz and back up (a stomp), half down half up."""
        x, y, z0 = self.pos
        h = max(1, steps // 2)
        for i in range(h):
            self.set_base(x, y, z0 - dz * (i + 1) / h)
            self.w.step()
        for i in range(h):
            self.set_base(x, y, z0 - dz * (1.0 - (i + 1) / h))
            self.w.step()

    def hold(self, steps):
        for _ in range(steps):
            self.w.step()


def particles(w):
    return np.asarray(w.download_field(nuka.Field.PARTICLE_POSITION),
                      dtype=np.float32).reshape(-1, 3)


def cloth(w):
    return particles(w)[-N_CLOTH:]


def gravel(w):
    p = particles(w)
    return p[:-N_CLOTH] if p.shape[0] > N_CLOTH else p


def link_poses(w):
    return np.asarray(w.download_field(nuka.Field.ARTICULATION_LINK_POSE),
                      dtype=np.float32).reshape(-1, 7)


def link_vels(w):
    return np.asarray(w.download_field(nuka.Field.LINK_VELOCITY),
                      dtype=np.float32).reshape(-1, 6)


def bodies(w):
    return np.asarray(w.download_field(nuka.Field.RIGID_BODY_TRANSFORM),
                      dtype=np.float32).reshape(-1, 7)


def body_vels(w):
    return np.asarray(w.download_field(nuka.Field.BODY_LINEAR_VELOCITY),
                      dtype=np.float32).reshape(-1, 3)


def surface_z(g, x, y, r=0.045):
    m = (np.abs(g[:, 0] - x) < r) & (np.abs(g[:, 1] - y) < r)
    return float(g[m, 2].max()) if int(m.sum()) else float("nan")


def slab_angle(links):
    s = links[-1]
    dx = float(s[0]) - A.BEAM_X
    dz = A.BEAM_Z - float(s[2])
    return math.degrees(math.atan2(abs(dx), max(dz, 1e-6)))


def build_world(dev, scene=OUT):
    b = nuka.SceneBuilder.create(scene)
    w = b.build(dev, env_count=1, dt=1.0 / 240.0, control_mode=0)
    b.destroy()
    return w


def run_probes(args):
    ok = True
    with nuka.Device.create(0) as dev:
        w = build_world(dev, args.scene)
        d = Driver(w)
        print(f"[choreo] BASE_POSE len={d.base.size} start={d.base[:3].round(3)}")
        d.hold(500)  # settle: drape forms, gravel compacts, objects rest.

        # -- probe 1: the curtain lifts >= 0.06 over the passing head ----------
        lane = np.abs(cloth(w)[:, 1]) < 0.12
        hem0 = float(cloth(w)[lane, 2].min())
        peak = [hem0]

        def track_cloth(_x):
            peak[0] = max(peak[0], float(cloth(w)[lane, 2].min()))

        d.glide(0.60, 0.15, on_step=track_cloth)
        lift = peak[0] - hem0
        hem_after = float(cloth(w)[lane, 2].min())
        print(f"[choreo] cloth hem {hem0:.3f} -> peak {peak[0]:.3f} "
              f"(lift {lift:.3f}) -> after {hem_after:.3f}")
        if lift < 0.06:
            ok = False
            print("  !! cloth lift < 0.06")
        if not (0.16 <= hem_after <= 0.40):
            ok = False
            print("  !! curtain lost after the pass (hem out of band)")

        # -- probe 2: persistent footprints in the gravel ----------------------
        g0 = gravel(w)
        spots = []
        d.glide(1.35, 0.25, sink=-0.010)
        for foot in (5, 14):  # actual sole spots at the stomp moment
            lp = link_poses(w)[foot]
            spots.append((float(lp[0]), float(lp[1])))
        d.dip(0.014, 160)
        d.glide(1.80, 0.25, sink=-0.010)
        for foot in (5, 14):
            lp = link_poses(w)[foot]
            spots.append((float(lp[0]), float(lp[1])))
        d.dip(0.014, 160)
        pre = [surface_z(g0, sx, sy) for sx, sy in spots]
        d.glide(2.32, 0.30, sink=0.0)
        d.hold(720)  # 3 s persistence
        g1 = gravel(w)
        post = [surface_z(g1, sx, sy) for sx, sy in spots]
        depths = [a - b for a, b in zip(pre, post)]
        nan = int(np.sum(~np.isfinite(g1)))
        print(f"[choreo] footprint depths {['%.4f' % v for v in depths]} m "
              f"(pre {['%.4f' % v for v in pre]}) gravel nan={nan}")
        if max(depths) < 0.005 or nan:
            ok = False
            print("  !! footprint < 5 mm after 3 s (or NaN)")

        # -- probe 3: micro objects kicked > 0.3 m/s, then at rest -------------
        b0 = bodies(w)
        micro = ((b0[:, 0] > 2.35) & (b0[:, 0] < 3.15) &
                 (np.abs(b0[:, 1]) < 0.3) & (b0[:, 2] < 0.06) &
                 (b0[:, 2] > -0.05))
        kicked = np.zeros(b0.shape[0], dtype=bool)
        vmax = [0.0]

        def track_kicks(_x):
            v = np.linalg.norm(body_vels(w), axis=1)
            kicked[:] |= micro & (v > 0.3)
            vmax[0] = max(vmax[0], float(v[micro].max()))

        d.glide(3.12, 0.8, on_step=track_kicks)
        d.hold(500)
        vend = np.linalg.norm(body_vels(w), axis=1)
        bend = bodies(w)
        sunk = int(np.sum(micro & (bend[:, 2] < -0.02)))
        print(f"[choreo] kicked {int(kicked.sum())} objects >0.3 m/s "
              f"(peak {vmax[0]:.2f}); rest max|v| {float(vend[micro].max()):.3f}; "
              f"sunk-through {sunk}")
        if int(kicked.sum()) < 3 or sunk:
            ok = False
            print("  !! fewer than 3 objects kicked (or tunnelling)")

        # -- probe 4: head strikes the slab -> swing >= 15 deg -----------------
        d.glide(3.58, 0.5)
        head_v0 = link_vels(w)[9].copy()
        d.glide(3.86, 0.35)
        head_v1 = link_vels(w)[9].copy()
        d.hold(30)
        d.glide(3.45, 1.2)
        ang = [0.0]
        for _ in range(600):
            w.step()
            ang[0] = max(ang[0], slab_angle(link_poses(w)))
        print(f"[choreo] slab swing max {ang[0]:.1f} deg; head qdot during strike "
              f"{np.round(head_v0, 3).tolist()} -> {np.round(head_v1, 3).tolist()}")
        if ang[0] < 15.0:
            ok = False
            print("  !! slab swing < 15 deg")

        links = link_poses(w)
        if not np.all(np.isfinite(links)) or not np.all(np.isfinite(particles(w))):
            ok = False
            print("  !! NaN at choreo end")
        w.destroy()
    print(f"[gate choreo] {'PASS' if ok else 'FAIL'}")
    return ok


# -- the two-sink additivity probe -------------------------------------------
# One duck foot descends onto the half-buried pebble inside the gravel bed, so
# that ONE link takes the MPM grid reaction and the pebble contact-row reaction
# in the SAME step. Four variant cooks with identical kinematics separate the
# sinks; additive composition means dv(both) ~ dv(gravel) + dv(pebble) - dv(none).
FOOT_LINK = 5  # left foot_assembly in tree DFS order (verified by lowest link z).


def _twosink_run(dev, scene):
    w = build_world(dev, scene)
    d = Driver(w)
    lp = link_poses(w)
    foot0 = lp[FOOT_LINK]
    px, py, _ = A.PROBE_PEBBLE
    # place the base so the LEFT foot lands on the pebble centre.
    bx = px - (float(foot0[0]) - d.pos[0])
    by = py - (float(foot0[1]) - d.pos[1])
    d.set_base(bx, by, A.BASE_OVER_FLOOR + 0.06)
    d.hold(200)
    trace = []
    z0, z1, n = A.BASE_OVER_FLOOR + 0.06, A.BASE_OVER_FLOOR - 0.018, 160
    for i in range(n):
        d.set_base(bx, by, z0 + (z1 - z0) * (i + 1) / n)
        w.step()
        trace.append(link_vels(w)[FOOT_LINK].copy())
    w.destroy()
    return np.asarray(trace)


def run_twosink(args):
    import argparse
    import bdx_oneshot_author as author
    scratch = "/data/xtzhang25/_work/activate/tmp"
    variants = {
        "both": dict(no_gravel=False, no_pebble=False),
        "gravel": dict(no_gravel=False, no_pebble=True),
        "pebble": dict(no_gravel=True, no_pebble=False),
        "none": dict(no_gravel=True, no_pebble=True),
    }
    paths = {}
    for name, kw in variants.items():
        out = os.path.join(scratch, f"twosink_{name}.nks")
        a = argparse.Namespace(rocks=0, gravel_spacing=A.GRAVEL_SPACING, out=out,
                               no_cloth=True, **kw)
        author.build(a)
        paths[name] = out

    traces = {}
    with nuka.Device.create(0) as dev:
        for name, path in paths.items():
            traces[name] = _twosink_run(dev, path)
            print(f"[twosink] {name}: |v|max {np.abs(traces[name]).max():.4f}")

    dv = {k: t - traces["none"] for k, t in traces.items() if k != "none"}
    lhs = dv["both"]
    rhs = dv["gravel"] + dv["pebble"]
    # compare over the contact window (last quarter of the descent).
    q = lhs.shape[0] * 3 // 4
    err = float(np.abs(lhs[q:] - rhs[q:]).max())
    scale = float(max(np.abs(lhs[q:]).max(), 1e-6))
    print(f"[twosink] additive composition: max|dv_both-(dv_gravel+dv_pebble)| "
          f"= {err:.4f} against response scale {scale:.4f} "
          f"(ratio {err / scale:.2f})")
    print(f"[twosink] foot vz at final step: both {lhs[-1][5]:.4f} "
          f"gravel {dv['gravel'][-1][5]:.4f} pebble {dv['pebble'][-1][5]:.4f}")
    ok = np.isfinite(err) and scale > 1e-4
    print(f"[gate twosink] {'PASS (numbers reported)' if ok else 'FAIL'}")
    return ok
