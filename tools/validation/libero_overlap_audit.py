#!/usr/bin/env python3
"""Measure OBB overlap from recorded body poses and authored collision boxes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def boxes(scene, body_name):
    text = scene.read_text(encoding="utf-8")
    if "nuka:" in text and "xmlns:nuka=" not in text:
        text = text.replace("<mujoco ", '<mujoco xmlns:nuka="https://nuka.physics/mjcf" ', 1)
    root = ET.fromstring(text)
    body = next(b for b in root.iter("body") if b.get("name") == body_name)
    geoms = [g for g in body.findall("geom") if g.get("type") == "box"
             and (int(g.get("contype", "1")) or int(g.get("conaffinity", "1")))]
    if not geoms:
        raise ValueError(f"no authored collision boxes on {body_name}")
    return (np.array([np.fromstring(g.get("pos", "0 0 0"), sep=" ") for g in geoms]),
            rotation([np.fromstring(g.get("quat", "1 0 0 0"), sep=" ") for g in geoms]),
            np.array([np.fromstring(g.get("size"), sep=" ") for g in geoms]),
            [g.get("name", str(i)) for i, g in enumerate(geoms)])


def world_boxes(poses, geometry):
    local_centers, local_rotation, _, _ = geometry
    r = rotation(poses[:, 3:])
    centers = poses[:, None, :3] + np.einsum("tij,bj->tbi", r, local_centers)
    axes = (r[:, None] @ local_rotation[None]).swapaxes(-1, -2)
    return centers, axes


def pair_penetration(center_a, axes_a, half_a, center_b, axes_b, half_b):
    t, a, b = len(center_a), len(half_a), len(half_b)
    aa = np.broadcast_to(axes_a[:, :, None], (t, a, b, 3, 3))
    bb = np.broadcast_to(axes_b[:, None], (t, a, b, 3, 3))
    cross = np.cross(aa[..., :, None, :], bb[..., None, :, :]).reshape(t, a, b, 9, 3)
    directions = np.concatenate((aa, bb, cross), axis=-2)
    length = np.linalg.norm(directions, axis=-1)
    directions = directions / np.maximum(length[..., None], 1e-15)
    ra = np.sum(np.abs(np.einsum("tabkd,taed->tabke", directions, axes_a)) * half_a[None, :, None, None], axis=-1)
    rb = np.sum(np.abs(np.einsum("tabkd,tbed->tabke", directions, axes_b)) * half_b[None, None, :, None], axis=-1)
    distance = np.abs(np.einsum("tabkd,tabd->tabk", directions, center_b[:, None]-center_a[:, :, None]))
    return np.min(np.where(length > 1e-10, ra+rb-distance, np.inf), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--body-a", required=True)
    parser.add_argument("--body-b", required=True)
    parser.add_argument("--poses-a", default="target_poses")
    parser.add_argument("--poses-b", default="plate_poses")
    parser.add_argument("--dt", type=float, default=0.002)
    args = parser.parse_args()
    identity = np.eye(3)[None, None]
    origin = np.zeros((1, 1, 3))
    half = np.ones((1, 3))
    assert abs(pair_penetration(origin, identity, half, np.array([[[1.5, 0, 0]]]), identity, half).item()-0.5) < 1e-12
    assert pair_penetration(origin, identity, half, np.array([[[3.0, 0, 0]]]), identity, half).item() < 0
    recording = args.run / "rollout.npz"
    if not recording.exists():
        recording = args.run / "rollout_partial.npz"
    data = np.load(recording)
    poses_a, poses_b = data[args.poses_a], data[args.poses_b]
    ga, gb = boxes(args.scene, args.body_a), boxes(args.scene, args.body_b)
    depth = np.zeros(len(poses_a))
    pair_at_peak = None
    peak = 0.0
    for start in range(0, len(poses_a), 64):
        ca, aa = world_boxes(poses_a[start:start+64], ga)
        cb, ab = world_boxes(poses_b[start:start+64], gb)
        overlap = pair_penetration(ca, aa, ga[2], cb, ab, gb[2])
        depth[start:start+len(ca)] = np.maximum(overlap.max(axis=(1, 2)), 0)
        index = np.unravel_index(np.argmax(overlap), overlap.shape)
        if overlap[index] > peak:
            peak = float(overlap[index])
            pair_at_peak = [ga[3][index[1]], gb[3][index[2]]]
    dt = float(data["physics_dt"]) if "physics_dt" in data else args.dt
    summary = dict(body_a=args.body_a, body_b=args.body_b, samples=len(depth),
                   max_overlap_m=float(depth.max()), max_overlap_time_s=float(depth.argmax()*dt),
                   final_overlap_m=float(depth[-1]), tail_max_overlap_m=float(depth[-round(0.5/dt):].max()),
                   deepest_box_pair=pair_at_peak, contract="SAT minimum translation distance of authored collision boxes")
    (args.run / "overlap_audit.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    np.savez_compressed(args.run / "overlap_audit.npz", penetration_m=depth, dt=dt)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
