#!/usr/bin/env python3
"""Isolate why the duck free-falls in the corridor. Drop-test (PD-hold home, no
policy) the real corridor at default vs bumped solver_max_pairs / contact margin.
Stands with a bigger pair budget -> the 77-body broadphase starved the foot
contacts. Still falls -> the co-resident cook drops articulation x rigid contact."""
from __future__ import annotations

import numpy as np
import nuka
from nuka.tasks import bdx_obs as B

CORRIDOR = "/data/xtzhang25/_work/activate/agent-a79c434fcb3e6c705/examples/scenes/bdx_oneshot.nks"


def drop_test(tag, steps=90, decim=5, **buildkw):
    with nuka.Device.create(0) as dev:
        b = nuka.SceneBuilder.create(CORRIDOR)
        w = b.build(dev, env_count=1, dt=1.0 / 240.0, control_mode=0, **buildkw)
        b.destroy()
        ankle = [int(w.dof_index("^left_ankle$")[0]), int(w.dof_index("^right_ankle$")[0])]
        obs = B.BdxObsBuilder(w, ankle, [(np.zeros(3, np.float32), 0.01)] * 2)
        obs.apply_pd_gains()
        z0 = float(obs.base_pos()[0, 2])
        fz0 = obs.foot_bottom_z()[0].detach().cpu().numpy()
        print(f"[{tag}] cook base_z={z0:.3f} foot_z(L,R)={fz0.round(3)} "
              f"(platform_top~0.15 floor~0.05)", flush=True)
        for k in range(steps):
            for _ in range(decim):
                w.step()
            if k % 15 == 0 or k == steps - 1:
                bp = obs.base_pos()[0].detach().cpu().numpy()
                print(f"[{tag}] step {k*decim:4d} base_z={bp[2]:+.3f} "
                      f"tilt={float(obs.tilt_deg()[0]):.1f}", flush=True)
        zf = float(obs.base_pos()[0, 2])
        print(f"[{tag}] => z {z0:.3f}->{zf:.3f}  STANDS={zf > z0 - 0.15}\n", flush=True)
        w.destroy()


if __name__ == "__main__":
    drop_test("DEFAULT", bake_link_sdf=True)
    drop_test("MAXPAIRS", bake_link_sdf=True, solver_max_pairs=32768)
    drop_test("MARGIN+PAIRS", bake_link_sdf=True, solver_max_pairs=32768,
              solver_contact_margin=0.02, solver_pos_iters=8, solver_vel_iters=12)
