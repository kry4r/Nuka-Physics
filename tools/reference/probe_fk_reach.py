"""Kinematic probe (v2): front-PAW reach under a pitched vertical base.

Paw world position = calf-link origin + R(calf quat) . (0,0,-0.213) -- the
Go2 foot sphere's fixed offset down the shank (same convention as
go2_locomotion._foot_local_offset). Zero gravity, pure FK.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import nuka
from nuka.tasks import go2_obs as G


def paw_world(pose, slot, dev):
    """(N,3) paw world position from the calf link pose at ``slot``."""
    origin = pose[:, int(slot), 0:3]
    quat = pose[:, int(slot), 3:7]
    off = torch.tensor([[0.0, 0.0, -0.213]], device=dev)
    conj = torch.tensor([[1.0, -1.0, -1.0, -1.0]], device=dev)
    return origin + G.quat_rotate_inverse_wxyz(quat * conj, off)


def main():
    dev = torch.device("cuda")
    d = nuka.Device.create(0)
    w = nuka.World.create_from_scene(
        d, "examples/scenes/go2_float.usda", 1,
        contact_family=1, heightfield_terrain_type=1,
        heightfield_nrow=161, heightfield_ncol=161, heightfield_cell=0.25)
    w.set_gravity_z(0.0)
    obs = G.Go2ObsBuilder(d, w)
    nuka.sync()
    slot_fl_calf = 3  # link layout: 0 trunk; FL hip1 thigh2 calf3; FR 4/5/6 ...

    phi = math.radians(90.0)
    for sign in (+1.0, -1.0):
        pq = torch.tensor(
            [[math.cos(phi / 2), 0.0, sign * math.sin(phi / 2), 0.0]],
            dtype=torch.float32, device=dev)
        print(f"--- pitch sign {sign:+.0f} (pgx -> {sign:+.0f}) ---")
        rows = []
        for thigh in np.arange(0.6, 3.7, 0.25):
            for calf in (-1.0, -1.5, -2.2):
                w.reset()
                nuka.sync()
                cand = np.array([0.0, float(thigh), float(calf),
                                 0.0, float(thigh), float(calf),
                                 0.0, -0.9, -1.4,
                                 0.0, -0.9, -1.4], dtype=np.float32)
                ct = torch.from_numpy(cand).to(dev)
                q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
                q[:, 1:G.GO2_BLC] = ct.index_select(0, obs.urdf_from_nuka_slot)
                tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
                tgt[:, 1:G.GO2_BLC] = ct.index_select(0, obs.nuka_slot_for_urdf)
                bp = torch.from_dlpack(w.buffer_view(nuka.BASE_POSE))
                bp[:, 2] = 0.40
                bp[:, 3:7] = pq
                w.step_n(4)
                nuka.sync()
                pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
                paw = paw_world(pose, slot_fl_calf, dev)[0].cpu().numpy()
                hipz = float(pose[0, 0, 2])
                rows.append((float(paw[2]), float(paw[0]), thigh,
                             float(calf), round(float(bp[0, 2]), 3)))
        rows.sort(key=lambda r: r[0])
        print("  deepest-paw configs (paw_z, paw_x, thigh, calf, base_z):")
        seen = set()
        shown = 0
        for pz, px, th, ca, bz in rows:
            key = (round(th, 2), ca)
            if key in seen:
                continue
            seen.add(key)
            print(f"    paw_z={pz:+.3f} paw_x={px:+.3f} "
                  f"thigh={th:.2f} calf={ca:.2f} base_z~{bz}")
            shown += 1
            if shown >= 8:
                break


if __name__ == "__main__":
    main()
