#!/usr/bin/env python3
"""One-off API/layout introspection for the interaction probe."""
import numpy as np
import torch
import nuka
import bdx_oneshot_choreo as C
import bdx_oneshot_author as A

print("UPPER consts:", sorted(x for x in dir(nuka) if x.isupper()))
try:
    print("Field members:", sorted(x for x in dir(nuka.Field) if not x.startswith("_")))
except Exception as e:
    print("no Field:", e)

N_CLOTH = A.CLOTH_NX * A.CLOTH_NY
print("N_CLOTH", N_CLOTH)

with nuka.Device.create(0) as dev:
    w = C.build_world(dev)
    print("env_count", w.env_count, "base_link_count", w.base_link_count,
          "action_dim", w.action_dim)
    print("dof_names", list(w.dof_names()))

    pp = np.asarray(w.download_field(nuka.Field.PARTICLE_POSITION),
                    dtype=np.float32).reshape(-1, 3)
    print("PARTICLES total", pp.shape[0],
          "x[", round(float(pp[:, 0].min()), 3), round(float(pp[:, 0].max()), 3), "]",
          "z[", round(float(pp[:, 2].min()), 3), round(float(pp[:, 2].max()), 3), "]")
    print("HEAD 5 particles:", np.round(pp[:5], 3).tolist())
    print("TAIL 20 particles (cloth/cable region):", np.round(pp[-20:], 3).tolist())
    print("cloth slice last-728 x[",
          round(float(pp[-N_CLOTH:, 0].min()), 3), round(float(pp[-N_CLOTH:, 0].max()), 3),
          "] z[", round(float(pp[-N_CLOTH:, 2].min()), 3),
          round(float(pp[-N_CLOTH:, 2].max()), 3), "]")

    # buffer_view availability for zero-copy GPU reads.
    for nm in ["PARTICLE_POSITION", "PARTICLE_VELOCITY", "LINK_CONTACT_WRENCH",
               "CONTACT_WRENCH", "ARTICULATION_LINK_POSE", "LINK_VELOCITY"]:
        c = getattr(nuka, nm, None)
        ok = False
        shp = None
        if c is not None:
            try:
                t = torch.from_dlpack(w.buffer_view(c))
                ok, shp = True, tuple(t.shape)
            except Exception as e:
                shp = f"ERR {e}"
        print(f"nuka.{nm} present={c is not None} bufview={ok} shape={shp}")

    lp = np.asarray(w.download_field(nuka.Field.ARTICULATION_LINK_POSE),
                    dtype=np.float32).reshape(-1, 7)
    print("LINK poses", lp.shape[0])
    for i, row in enumerate(lp):
        print("  link", i, "pos", np.round(row[:3], 3).tolist())

    rb = np.asarray(w.download_field(nuka.Field.RIGID_BODY_TRANSFORM),
                    dtype=np.float32).reshape(-1, 7)
    print("RIGID bodies", rb.shape[0])
    w.destroy()
print("DONE")
