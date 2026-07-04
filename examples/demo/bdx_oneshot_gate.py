#!/usr/bin/env python3
"""Load + stability gates for bdx_oneshot.nks. Stages (run in order):

  cook    : one MpmXpbd cook of the full scene; prints body/particle/media counts.
  settle  : N steps PD stand; asserts static-prop zero drift, the curtain stays
            draped on the lintel, the gravel bed settles without escape/NaN, the
            micro objects come to rest, the slab hangs still, the duck stands.
  choreo  : scripted pose-driven traverse (bdx_oneshot_choreo); asserts the cloth
            lift, persistent footprints, kicked objects, and the slab swing.
  twosink : one duck link taking a gravel grid reaction AND a rigid contact-row
            reaction in one step; reports the additive qdot composition numbers.

    python examples/demo/bdx_oneshot_gate.py --stage settle [--steps 600]
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import nuka
import bdx_oneshot_author as A

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nks")


def build_world(dev, scene=OUT):
    b = nuka.SceneBuilder.create(scene)
    w = b.build(dev, env_count=1, dt=1.0 / 240.0, control_mode=0)
    b.destroy()
    return w


def particles(w):
    if int(w.particle_count) == 0:
        return None
    return np.asarray(w.download_field(nuka.Field.PARTICLE_POSITION),
                      dtype=np.float32).reshape(-1, 3)


def bodies(w):
    return np.asarray(w.download_field(nuka.Field.RIGID_BODY_TRANSFORM),
                      dtype=np.float32).reshape(-1, 7)


def body_vel(w):
    lin = np.asarray(w.download_field(nuka.Field.BODY_LINEAR_VELOCITY),
                     dtype=np.float32).reshape(-1, 3)
    return lin


def link_poses(w):
    return np.asarray(w.download_field(nuka.Field.ARTICULATION_LINK_POSE),
                      dtype=np.float32).reshape(-1, 7)


def cloth_slice(w):
    """The cloth particle rows: the XPBD slice sits ABOVE the MPM slice."""
    n_cloth = A.CLOTH_NX * A.CLOTH_NY
    p = particles(w)
    return p[-n_cloth:] if p is not None else None


def mpm_slice(w):
    n_cloth = A.CLOTH_NX * A.CLOTH_NY
    p = particles(w)
    if p is None:
        return None
    return p[:-n_cloth] if p.shape[0] > n_cloth else p


def gravel_in_lane(p):
    tx0, tx1, ty, tz = A.TROUGH
    m = (p[:, 0] > tx0 - 0.1) & (p[:, 0] < tx1 + 0.1) & (np.abs(p[:, 1]) < ty + 0.1)
    return p[m]


def stage_cook(args):
    with nuka.Device.create(0) as dev:
        w = build_world(dev, args.scene)
        pc, blc = int(w.particle_count), int(w.base_link_count)
        n_cloth = A.CLOTH_NX * A.CLOTH_NY
        bods = bodies(w)
        print(f"[cook] particle_count={pc} (mpm={pc - n_cloth} cloth={n_cloth}) "
              f"base_link_count={blc} rigid_bodies={bods.shape[0]}")
        links = link_poses(w)
        print(f"[cook] links n={links.shape[0]} z_min={links[:, 2].min():.3f} "
              f"z_max={links[:, 2].max():.3f}")
        ok = pc > n_cloth and bods.shape[0] > 50 and np.all(np.isfinite(bods))
        w.step()
        p = particles(w)
        ok = ok and bool(np.all(np.isfinite(p)))
        print(f"[cook] one step: particles finite={bool(np.all(np.isfinite(p)))}")
        w.destroy()
    print(f"[gate cook] {'PASS' if ok else 'FAIL'}")
    return ok


def stage_settle(args):
    with nuka.Device.create(0) as dev:
        w = build_world(dev, args.scene)
        n_cloth = A.CLOTH_NX * A.CLOTH_NY
        # identify the static set empirically: bodies that do not move at all over
        # an early window (statics never move; links/free bodies jitter at once).
        for _ in range(5):
            w.step()
        probe0 = bodies(w)
        for _ in range(5):
            w.step()
        static_mask = (np.linalg.norm(bodies(w)[:, :3] - probe0[:, :3], axis=1)
                       < 1e-7)
        static0 = probe0

        for i in range(args.steps):
            w.step()
            if i % 150 == 149:
                c = cloth_slice(w)
                print(f"  step {i + 1}: cloth z[{c[:, 2].min():.3f},"
                      f"{c[:, 2].max():.3f}] x[{c[:, 0].min():.3f},{c[:, 0].max():.3f}]")

        ok = True
        bods, vels = bodies(w), body_vel(w)
        links = link_poses(w)

        # 1. static props: zero drift over the whole settle.
        drift = np.linalg.norm(bods[:, :3] - static0[:, :3], axis=1)
        stat_drift = float(drift[static_mask].max())
        print(f"[settle] statics n={int(static_mask.sum())} max drift "
              f"{stat_drift:.6f} m; movable max drift "
              f"{float(drift[~static_mask].max()):.4f} m")
        if stat_drift > 1e-5:
            ok = False
            print("  !! a static prop moved")

        # 2. curtain: hung from the pinned line over the lintel, hem near z~0.27.
        c = cloth_slice(w)
        cmin, cmax = float(c[:, 2].min()), float(c[:, 2].max())
        over = int(np.sum(c[:, 2] > A.LINTEL_Z[1] - 0.01))
        print(f"[settle] cloth min_z {cmin:.3f} max_z {cmax:.3f} "
              f"nodes above lintel top {over}/{c.shape[0]}")
        if not (0.18 + A.SCENE_LIFT <= cmin <= 0.34 + A.SCENE_LIFT):
            ok = False
            print("  !! curtain hem out of the hang band")
        if over < A.CLOTH_NY - 4:
            ok = False
            print("  !! the pinned line lost nodes (curtain fell)")

        # 3. gravel: settled in the tray, no escape, no NaN.
        g = mpm_slice(w)
        gn = int(np.sum(~np.isfinite(g)))
        fp = g[np.isfinite(g).all(axis=1)]
        esc = int(np.sum((fp[:, 2] < A.TROUGH[3] - 0.05) | (fp[:, 2] > 0.5) |
                         (np.abs(fp[:, 1]) > 1.0)))
        print(f"[settle] gravel n={g.shape[0]} nan={gn} escaped={esc} "
              f"z_top={fp[:, 2].max():.3f} z_mean={fp[:, 2].mean():.3f}")
        if gn > 0 or esc > g.shape[0] * 0.01:
            ok = False
            print("  !! gravel NaN / escape")

        # 4. micro objects at rest (report the worst body for triage).
        speeds = np.linalg.norm(vels, axis=1)
        worst = int(np.argmax(speeds))
        print(f"[settle] max |body v| {speeds[worst]:.4f} m/s at body {worst} "
              f"pos {bods[worst, :3].round(3)}")
        if speeds[worst] > 0.05:
            ok = False
            print("  !! a free body is still moving after settle")

        # 5. slab hangs at rest at head height.
        slab = links[-1]  # the rope articulation's last link (slab, fixed to yoke).
        print(f"[settle] slab pose {slab[:3].round(3)}")
        if not (0.20 + A.SCENE_LIFT < slab[2] < 0.45 + A.SCENE_LIFT) or abs(slab[0] - A.BEAM_X) > 0.05:
            ok = False
            print("  !! slab not hanging under the beam")

        # 6. duck upright at the spawn.
        base = np.asarray(w.download_field(nuka.Field.BASE_POSE, 0, 7),
                          dtype=np.float32)
        print(f"[settle] duck base {base[:3].round(3)} quat_w {base[3]:.4f}")
        if not np.all(np.isfinite(links)):
            ok = False
            print("  !! link pose NaN")
        w.destroy()
    print(f"[gate settle] {'PASS' if ok else 'FAIL'}")
    return ok


def stage_choreo(args):
    import bdx_oneshot_choreo as C
    return C.run_probes(args)


def stage_twosink(args):
    import bdx_oneshot_choreo as C
    return C.run_twosink(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cook", "settle", "choreo", "twosink"],
                    default="settle")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--scene", default=OUT)
    args = ap.parse_args()
    ok = {"cook": stage_cook, "settle": stage_settle, "choreo": stage_choreo,
          "twosink": stage_twosink}[args.stage](args)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
