"""Dynamic settle probe: NOSE-DOWN-VERTICAL front-support handstand.

Uses the FK-probe-derived geometry (pitch sign +1 => pgx -> +1, front legs
thigh ~0.6-1.1 / calf -2.2..-1.5, base_z ~0.28) and checks which candidates
settle into a sustained vertical pose supported on BOTH front paws with the
trunk clear of the ground -- the IC/nominal posture for the task.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

import nuka
from nuka.tasks import go2_obs as G

SCENE = "examples/scenes/go2_float.usda"

_World = None
_Obs = None
_Dev = None


def probe(front_thigh, front_calf, rear_thigh, rear_calf, base_z,
          steps=120, tail=40):
    w = _World
    obs = _Obs
    nuka.sync()
    w.reset()
    phi = math.radians(90.0)
    pq = torch.tensor(
        [[math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0]],
        dtype=torch.float32, device=_Dev)
    cand = np.array([0.0, front_thigh, front_calf,
                     0.0, front_thigh, front_calf,
                     0.0, rear_thigh, rear_calf,
                     0.0, rear_thigh, rear_calf], dtype=np.float32)
    cand_t = torch.from_numpy(cand).to(_Dev)
    q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
    q[:, 1:G.GO2_BLC] = cand_t.index_select(0, obs.urdf_from_nuka_slot)
    tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
    tgt[:, 1:G.GO2_BLC] = cand_t.index_select(0, obs.nuka_slot_for_urdf)
    bp = torch.from_dlpack(w.buffer_view(nuka.BASE_POSE))
    bp[:, 2] = base_z
    bp[:, 3:7] = pq
    pgx_series = []
    for s in range(steps):
        w.step_n(4)
        if s >= steps - tail:
            pgx_series.append(float(obs.projected_gravity_auth()[:, 0].mean()))
    pgx = float(np.mean(pgx_series))
    pgx_std = float(np.std(pgx_series))
    pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
    wrench = torch.from_dlpack(w.buffer_view(nuka.LINK_CONTACT_WRENCH))
    f_fz = [float(wrench[0, int(sl), 2]) for sl in (3, 6)]      # FL/FR calf links
    r_knee_z = float(pose[0, 9, 2])                             # RL knee origin
    trunk_z = float(bp[0, 2])
    trunk_fz = float(wrench[0, 0, 2])
    ok = (pgx > 0.7 and pgx_std < 0.06 and min(f_fz) > 5.0
          and abs(trunk_fz) < 3.0 and r_knee_z > 0.20)
    return dict(ok=ok, pgx=pgx, std=pgx_std,
                ffz=tuple(round(v, 1) for v in f_fz),
                rkz=round(r_knee_z, 3), tz=round(trunk_z, 3),
                tfz=round(trunk_fz, 1))


def main():
    global _World, _Obs, _Dev
    _Dev = torch.device("cuda")
    dev = nuka.Device.create(0)
    _World = nuka.World.create_from_scene(
        dev, SCENE, 4,
        contact_family=1, heightfield_terrain_type=1,
        heightfield_nrow=161, heightfield_ncol=161, heightfield_cell=0.25)
    _Obs = G.Go2ObsBuilder(dev, _World)
    _Obs.apply_pd_gains()

    grid = list(itertools.product(
        (0.75, 0.85, 0.95),       # front thigh
        (-1.9, -1.8, -1.7),       # front calf (folded under, paw down)
        (-1.3, -1.2, -1.1),       # rear thigh
        (-1.4, -1.3, -1.2),       # rear calf (shallow fold)
        (0.25, 0.27, 0.29),       # base z
    ))
    print(f"sweeping {len(grid)} candidates ...", flush=True)
    results = []
    for i, (ft, fc, rt, rc, bz) in enumerate(grid):
        r = probe(ft, fc, rt, rc, bz)
        r["cand"] = (ft, fc, rt, rc, bz)
        results.append(r)

    results.sort(key=lambda x: (not x["ok"], -x["pgx"]))
    for r in results[:14]:
        tag = "HIT " if r["ok"] else "    "
        print(f"{tag}pgx={r['pgx']:.3f}+-{r['std']:.3f} ffz={r['ffz']} "
              f"rkz={r['rkz']} tz={r['tz']} tfz={r['tfz']} cand={r['cand']}",
              flush=True)
    n_hits = sum(1 for r in results if r["ok"])
    print(f"{n_hits}/{len(results)} hits")


if __name__ == "__main__":
    main()
