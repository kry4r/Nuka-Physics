#!/usr/bin/env python3
"""Confirm the co-residence contact gap: does the duck stand on the corridor floor
when the MPM/cloth/cable media are removed (a plain rigid world)? Stands -> the
co-resident media step drops the articulation x static-rigid contact."""
from __future__ import annotations

import json
import os

import numpy as np
import nuka
from nuka.tasks import bdx_obs as B

SCENES = os.path.dirname(
    "/data/xtzhang25/_work/activate/agent-a79c434fcb3e6c705/examples/scenes/")
SRC = os.path.join(SCENES, "bdx_oneshot.nks")


def try_create(tag, path):
    try:
        b = nuka.SceneBuilder.create(path)
        return b
    except Exception as e:
        print(f"[{tag}] create FAILED: {type(e).__name__}: {e}", flush=True)
        return None


def drop(tag, b):
    with nuka.Device.create(0) as dev:
        w = b.build(dev, env_count=1, dt=1.0 / 240.0, control_mode=0)
        b.destroy()
        ankle = [int(w.dof_index("^left_ankle$")[0]), int(w.dof_index("^right_ankle$")[0])]
        obs = B.BdxObsBuilder(w, ankle, [(np.zeros(3, np.float32), 0.01)] * 2)
        obs.apply_pd_gains()
        z0 = float(obs.base_pos()[0, 2])
        for k in range(80):
            for _ in range(5):
                w.step()
        zf = float(obs.base_pos()[0, 2])
        print(f"[{tag}] z {z0:.3f}->{zf:.3f}  STANDS={zf > z0 - 0.15}", flush=True)
        w.destroy()


def main():
    doc = json.load(open(SRC))
    # 1: unmodified re-dump next to the original (isolates the round-trip itself).
    p_same = os.path.join(SCENES, "bdx_oneshot_rt.nks")
    json.dump(doc, open(p_same, "w"))
    b = try_create("REDUMP", p_same)
    if b is not None:
        b.destroy()
        print("[REDUMP] create OK -> round-trip is fine", flush=True)
    # 2: media stripped.
    doc2 = json.load(open(SRC))
    n = len(doc2.get("media", []))
    doc2["media"] = []
    p_nm = os.path.join(SCENES, "bdx_oneshot_nomedia.nks")
    json.dump(doc2, open(p_nm, "w"))
    print(f"[nomedia] stripped {n} media", flush=True)
    b2 = try_create("NOMEDIA", p_nm)
    if b2 is not None:
        drop("NOMEDIA", b2)


if __name__ == "__main__":
    main()
