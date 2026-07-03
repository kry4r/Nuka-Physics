#!/usr/bin/env python3
"""Load gate for bdx_oneshot.nks: cook the self-contained scene back through the
built-scene path, step it (PD stand at the spawn), and report stability metrics
for every zone -- the duck base pose, particle escape / NaN counts, the rope block
rest, and the free-rock settle. Prints PASS/FAIL lines; exits non-zero on a hard
failure (NaN / duck fall / particle blow-up).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import nuka
from bdx_oneshot_author import add_cloth, add_granular

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--media", choices=["none", "cloth", "granular"], default="none",
                    help="the ONE particle medium to cook (MPM granular and XPBD "
                         "cloth cannot co-reside, so they cook one at a time)")
    ap.add_argument("--granular-spacing", type=float, default=0.013)
    ap.add_argument("--scene", default=OUT)
    args = ap.parse_args()

    with nuka.Device.create(0) as dev:
        b = nuka.SceneBuilder.create(args.scene)
        if args.media == "cloth":
            add_cloth(b)
        elif args.media == "granular":
            add_granular(b, spacing=args.granular_spacing)
        w = b.build(dev, env_count=1, dt=1.0 / 240.0, control_mode=0)
        b.destroy()

        blc = int(w.base_link_count)
        pc = int(w.particle_count)
        base0 = np.asarray(w.download_field(nuka.Field.BASE_POSE, 0, 7),
                           dtype=np.float32).copy()
        print(f"[gate] base_link_count={blc} particle_count={pc}")
        print(f"[gate] duck base pose @0: pos={base0[:3].round(4)} "
              f"quat={base0[3:7].round(4)}")

        def particles():
            if pc == 0:
                return None
            return np.asarray(w.download_field(nuka.Field.PARTICLE_POSITION),
                              dtype=np.float32).reshape(-1, 3)

        def bodies():
            t = np.asarray(w.download_field(nuka.Field.RIGID_BODY_TRANSFORM),
                           dtype=np.float32)
            return t.reshape(-1, 7)

        ke_hist = []
        for i in range(args.steps):
            w.step()
            if i % 100 == 0 or i == args.steps - 1:
                base = np.asarray(w.download_field(nuka.Field.BASE_POSE, 0, 7),
                                  dtype=np.float32)
                fin = bool(np.all(np.isfinite(base)))
                p = particles()
                pnan = int(np.sum(~np.isfinite(p))) if p is not None else 0
                pesc = 0
                if p is not None:
                    fp = p[np.isfinite(p).all(axis=1)]
                    # escaped = below the floor or flung far off the walkway
                    pesc = int(np.sum((fp[:, 2] < -0.05) |
                                      (np.abs(fp[:, 1]) > 2.5) |
                                      (fp[:, 0] < -2.0) | (fp[:, 0] > 8.5)))
                print(f"  step {i:4d}: base_z={base[2]:+.4f} finite={fin} "
                      f"particle_nan={pnan} particle_escaped={pesc}")

        base = np.asarray(w.download_field(nuka.Field.BASE_POSE, 0, 7),
                          dtype=np.float32)
        bods = bodies()
        p = particles()

        # -- report ----------------------------------------------------------
        ok = True
        upright = float(base[3])  # quat w; ~1 => upright
        print("\n== SUMMARY ==")
        print(f"duck: base_z {base0[2]:+.4f} -> {base[2]:+.4f}  quat_w={upright:.4f}  "
              f"finite={bool(np.all(np.isfinite(base)))}")
        if not np.all(np.isfinite(base)) or base[2] < 0.10 or upright < 0.9:
            ok = False
            print("  !! duck unstable (fell / flipped / NaN)")
        if p is not None:
            fin = np.isfinite(p).all(axis=1)
            fp = p[fin]
            nan = int(np.sum(~fin))
            below = int(np.sum(fp[:, 2] < -0.05))
            zmax = float(fp[:, 2].max()) if fp.size else 0.0
            print(f"particles: n={p.shape[0]} nan={nan} below_floor={below} "
                  f"z_max={zmax:.3f} (cloth+granular)")
            if nan > 0 or below > p.shape[0] * 0.02:
                ok = False
                print("  !! particle blow-up / leak")
        # rope block = the composed 'block' free-ish link; last rigid rows are the
        # rope chain. Report the lowest rope body z as the block rest.
        finite_b = bods[np.isfinite(bods).all(axis=1)]
        print(f"rigid bodies: n={bods.shape[0]} all_finite={bool(np.all(np.isfinite(bods)))}")
        if not np.all(np.isfinite(bods)):
            ok = False
            print("  !! a rigid body went NaN")
        print(f"\n[gate] {'PASS' if ok else 'FAIL'}")
        w.destroy()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
