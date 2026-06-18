#!/usr/bin/env python3
"""[diag] Measure per-dog terrain penetration in a GO2T-v1 20-dog dump.

For every recorded frame, for every dog, for every link that carries cooked
collision geometry, reconstruct the collidable's LOWEST world point from the
recorded link pose + the cooked link-local geom offset, and compare it against
the COMPOSITE terrain surface (sampled through the SAME C-ABI sampler the foot
kernel rests feet on). Penetration = surface_z - lowest_point_z (positive ==
the link is BELOW the surface == 穿模). Reports worst-case across the whole clip
per dog and globally. Pure diagnostic; no physics is stepped.
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np
import torch

T_COMPOSITE = 4
GO2_BLC = 13


def _quat_rotate(q_wxyz, v):
    """Rotate (N,3) v by (N,4) wxyz quats (numpy)."""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return np.stack((rx, ry, rz), axis=1)


def _load_go2t(path):
    with open(path, "rb") as f:
        hdr = struct.unpack("<8i2f", f.read(40))
        magic, version, R, N, BLC, _blc2, POSE_F, decim, ctrl_dt, _ = hdr
        # per-dog spawn meta (2f + i + f).
        for _ in range(R):
            f.read(16)
        # frames: per step per dog -> drive (BLC f) + pose (BLC*POSE_F f).
        rec = bytearray(f.read())
    stride_dog = (BLC + BLC * POSE_F) * 4
    poses = np.empty((N, R, BLC, POSE_F), dtype="<f4")
    off = 0
    for s in range(N):
        for r in range(R):
            base = off
            # skip drive (BLC floats)
            p0 = base + BLC * 4
            block = np.frombuffer(rec, dtype="<f4", count=BLC * POSE_F, offset=p0)
            poses[s, r] = block.reshape(BLC, POSE_F)
            off += stride_dog
    return dict(R=R, N=N, BLC=BLC, POSE_F=POSE_F, poses=poses)


# Cooked go2 collision geometry per link (mirror of go2_locomotion.usda). The
# trunk is a box; each thigh + calf is a CAPSULE along the link-local Z shaft.
# We sample each collidable's lowest world point against the terrain. Links with
# NO collision geometry are skipped.
THIGH_LINKS = {2, 5, 8, 11}  # fl/fr/rl/rr thigh capsule.
CALF_LINKS = {3, 6, 9, 12}   # fl/fr/rl/rr calf capsule (reaches the foot).
# Capsule shaft endpoints (link-local) + radius: shaft from origin to (0,0,-0.213).
CAP_A = np.array([0.0, 0.0, 0.0], dtype=np.float32)
CAP_B = np.array([0.0, 0.0, -0.213], dtype=np.float32)
THIGH_R = 0.025
CALF_R = 0.0175
TRUNK_HE = np.array([0.1881, 0.04675, 0.057], dtype=np.float32)  # real go2 body box.


def _sample_surface(world, xs, ys):
    """Surface z at (xs,ys) via the world's cooked heightfield grid (the ONE source
    the foot kernel rests feet on)."""
    xs_t = torch.as_tensor(xs, dtype=torch.float32).contiguous()
    ys_t = torch.as_tensor(ys, dtype=torch.float32).contiguous()
    surf = world.sample_terrain_height(xs_t, ys_t)
    return np.asarray(torch.as_tensor(surf, dtype=torch.float32))


def _link_lowest_points(poses_frame):
    """(R,BLC,7) one frame -> list of (label, world_pts (M,3)) collidable samples
    spanning EVERY dog. Returns concatenated (P,3) world pts + a (P,) dog index +
    a (P,) link index so we can attribute penetration."""
    R = poses_frame.shape[0]
    pts, dog_idx, link_idx = [], [], []
    for r in range(R):
        P = poses_frame[r]                       # (BLC,7)
        # Trunk (link 0): box, 8 corners.
        t0 = P[0, 0:3]
        q0 = P[0, 3:7][None, :]
        corners = np.array([[sx, sy, sz] for sx in (-TRUNK_HE[0], TRUNK_HE[0])
                            for sy in (-TRUNK_HE[1], TRUNK_HE[1])
                            for sz in (-TRUNK_HE[2], TRUNK_HE[2])], dtype=np.float32)
        wc = t0[None, :] + _quat_rotate(np.repeat(q0, 8, axis=0), corners)
        pts.append(wc); dog_idx += [r] * 8; link_idx += [0] * 8
        # Thigh + calf capsules: sample the lowest point along the shaft (densely)
        # minus the radius (the capsule surface). The bottom cap of the calf is the
        # foot. Sampling several points along the segment catches a sprawled shaft.
        ts = np.linspace(0.0, 1.0, 7)[:, None].astype(np.float32)
        for links, rad in ((THIGH_LINKS, THIGH_R), (CALF_LINKS, CALF_R)):
            for lk in links:
                tl = P[lk, 0:3]
                ql = P[lk, 3:7]
                seg_local = CAP_A[None, :] * (1.0 - ts) + CAP_B[None, :] * ts  # (7,3)
                seg = tl[None, :] + _quat_rotate(np.repeat(ql[None, :], 7, axis=0),
                                                 seg_local)
                low = seg.copy()
                low[:, 2] -= rad
                pts.append(low); dog_idx += [r] * 7; link_idx += [lk] * 7
    return (np.concatenate(pts, axis=0),
            np.asarray(dog_idx, dtype=np.int64),
            np.asarray(link_idx, dtype=np.int64))


def measure(path, stride=1):
    import nuka
    d = _load_go2t(path)
    R, N = d["R"], d["N"]
    poses = d["poses"]
    # Build a sampling world whose cooked composite heightfield reproduces the dump's
    # terrain (the SAME grid the foot kernel rested feet on). Sampling offline against
    # the cooked grid avoids any python height formula.
    dev = nuka.Device.create(0)
    go2 = "/root/Nuka-Physics/examples/scenes/go2_locomotion.usda"
    world = nuka.World.create_from_scene(
        dev, go2, 1, 1.0 / 200.0, instance_count=1, instance_spacing=2.5,
        contact_family=1, heightfield_terrain_type=T_COMPOSITE,
        heightfield_nrow=161, heightfield_ncol=161, heightfield_cell=0.25,
        terrain_step_height=0.04, terrain_step_width=0.40,
        terrain_platform_width=2.0, terrain_grid_width=0.45,
        terrain_grid_height_max=0.15)
    worst_dog = np.full(R, -1e9, dtype=np.float64)   # max penetration per dog.
    worst_dog_frame = np.full(R, -1, dtype=np.int64)
    worst_dog_link = np.full(R, -1, dtype=np.int64)
    nan_dogs = set()
    for s in range(0, N, stride):
        frame = poses[s]
        if not np.isfinite(frame).all():
            for r in range(R):
                if not np.isfinite(frame[r]).all():
                    nan_dogs.add(r)
            continue
        wpts, dogs, links = _link_lowest_points(frame)
        surf = _sample_surface(world, wpts[:, 0].tolist(), wpts[:, 1].tolist())
        pen = surf - wpts[:, 2]                 # >0 == below surface (穿模).
        for r in range(R):
            mask = dogs == r
            if not mask.any():
                continue
            pr = pen[mask]
            j = int(np.argmax(pr))
            if pr[j] > worst_dog[r]:
                worst_dog[r] = float(pr[j])
                worst_dog_frame[r] = s
                lr = links[mask]
                worst_dog_link[r] = int(lr[j])
    return dict(R=R, N=N, worst_dog=worst_dog, worst_frame=worst_dog_frame,
                worst_link=worst_dog_link, nan_dogs=sorted(nan_dogs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args(argv)
    res = measure(args.dump, args.stride)
    R = res["R"]
    link_name = {0: "trunk", 2: "fl_thigh", 3: "fl_calf", 5: "fr_thigh",
                 6: "fr_calf", 8: "rl_thigh", 9: "rl_calf", 11: "rr_thigh",
                 12: "rr_calf"}
    print("=" * 70)
    print(f"TERRAIN PENETRATION REPORT  {args.dump}  (R={R}, N={res['N']})")
    print("=" * 70)
    for r in range(R):
        ln = link_name.get(res["worst_link"][r], f"link{res['worst_link'][r]}")
        print(f"  dog[{r:2d}] worst penetration = {res['worst_dog'][r]*1000:+8.1f} mm "
              f"@frame {res['worst_frame'][r]:4d} ({ln})")
    gw = float(np.max(res["worst_dog"]))
    gd = int(np.argmax(res["worst_dog"]))
    print("-" * 70)
    print(f"  WORST-CASE across all {R} dogs = {gw*1000:.1f} mm "
          f"({gw:.4f} m) on dog[{gd}]")
    print(f"  ACCEPTANCE (<20 mm): {'PASS' if gw < 0.020 else 'FAIL'}")
    if res["nan_dogs"]:
        print(f"  NaN/non-finite dogs: {res['nan_dogs']}")
    else:
        print(f"  all {R} dogs finite (no NaN)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
