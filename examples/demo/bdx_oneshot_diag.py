#!/usr/bin/env python3
"""Diagnostic: settle the scene, print the max|v| decay curve, and at the end
list every rigid body above a velocity floor with its position + size, so the
worst movers can be identified (roll limit-cycle vs spawn overlap vs slope)."""
from __future__ import annotations
import os
import numpy as np
import nuka
import bdx_oneshot_author as A

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nks")


def bodies(w):
    return np.asarray(w.download_field(nuka.Field.RIGID_BODY_TRANSFORM),
                      dtype=np.float32).reshape(-1, 7)


def vels(w):
    return np.asarray(w.download_field(nuka.Field.BODY_LINEAR_VELOCITY),
                      dtype=np.float32).reshape(-1, 3)


def avels(w):
    try:
        return np.asarray(w.download_field(nuka.Field.BODY_ANGULAR_VELOCITY),
                          dtype=np.float32).reshape(-1, 3)
    except Exception:
        return None


def main():
    steps = int(os.environ.get("DIAG_STEPS", "500"))
    with nuka.Device.create(0) as dev:
        b = nuka.SceneBuilder.create(OUT)
        w = b.build(dev, env_count=1, dt=1.0 / 240.0, control_mode=0)
        b.destroy()
        for i in range(steps):
            w.step()
            if (i + 1) % 100 == 0:
                v = np.linalg.norm(vels(w), axis=1)
                k = int(np.sum(v > 0.05))
                print(f"  step {i+1}: max|v|={v.max():.4f} nbody>0.05={k} "
                      f"nbody>0.15={int(np.sum(v>0.15))}", flush=True)
        v = np.linalg.norm(vels(w), axis=1)
        av = avels(w)
        bod = bodies(w)
        order = np.argsort(-v)
        print(f"[diag] {steps} steps done. bodies with |v|>0.05:")
        for idx in order:
            if v[idx] <= 0.05:
                break
            avn = float(np.linalg.norm(av[idx])) if av is not None else float("nan")
            print(f"   body {idx:3d} |v|={v[idx]:.4f} |w|={avn:7.3f} "
                  f"pos={bod[idx,:3].round(4).tolist()}")
        print(f"[diag] total bodies={bod.shape[0]} "
              f"count>0.05={int(np.sum(v>0.05))} count>0.02={int(np.sum(v>0.02))}")
        w.destroy()


if __name__ == "__main__":
    main()
