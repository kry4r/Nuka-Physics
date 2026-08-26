#!/usr/bin/env python3
"""Render a SINGLE env from a captured rollout npz as a shaded 3D robot dog.

Draws real geometry from the simulated world link poses: oriented trunk box,
head box, hip->thigh->calf limb chains as tapered cylinders, paw spheres, a
ground grid with world-anchored distance ticks, and a follow camera. This is
the "see the actual robot form" verification view (vs the 4x4 stick grid).

Run:
    python examples/demo/go2_render3d.py --npz out/go2_handstand_walk/go2_hs_walk.npz \
        --env 0 --stride 2 --out-dir out/go2_handstand_walk/render3d
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Go2 geometry (m): trunk box, head box, limb chain lengths, radii.
TRUNK = np.array([0.155, 0.055, 0.055])     # half-extents (l, w, h)
HEAD = np.array([0.055, 0.030, 0.030])
LIMB_R = 0.018
PAW_R = 0.022
LEG_CHAINS = {  # link indices into the 13-link pose array
    "FL": (0, 1, 2, 3), "FR": (0, 4, 5, 6),
    "RL": (0, 7, 8, 9), "RR": (0, 10, 11, 12),
}
FOOT_OFF = np.array([0.0, 0.0, -0.213])
LEG_COLOR = {"FL": "#e8963c", "FR": "#4aa8e8", "RL": "#e86478", "RR": "#5ec87c"}
TRUNK_COLOR = "#d8dae8"
HEAD_COLOR = "#b8bcd0"


def quat_mat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def box_faces(center, R, half):
    """8 rotated box corners -> 6 quad faces (world frame)."""
    ext = np.array([[sx, sy, sz] for sx in (-1, 1)
                    for sy in (-1, 1) for sz in (-1, 1)], dtype=float) * half
    pts = center + ext @ R.T
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
           (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [pts[list(i)] for i in idx]


def capsule(ax, p0, p1, r, color):
    """A limb segment: thick line + joint spheres (reads as a shaded cylinder)."""
    ax.plot(*zip(p0, p1), color=color, lw=r * 350, solid_capstyle="round",
            zorder=5, alpha=0.95)
    for p in (p0, p1):
        ax.scatter(*p, s=(r * 350) ** 2 * 3.0, color=color, depthshade=False,
                   zorder=6, edgecolors="none")


def draw_env(ax, links):
    base_p, base_q = links[0, 0:3], links[0, 3:7]
    R = quat_mat(base_q)

    ax.add_collection3d(Poly3DCollection(
        box_faces(base_p, R, TRUNK), facecolors=TRUNK_COLOR,
        edgecolors="#5a5e78", linewidths=0.6, zorder=4))
    head_c = base_p + R @ np.array([TRUNK[0] + HEAD[0] * 0.6, 0, 0.01])
    ax.add_collection3d(Poly3DCollection(
        box_faces(head_c, R, HEAD), facecolors=HEAD_COLOR,
        edgecolors="#5a5e78", linewidths=0.6, zorder=4))

    for leg, (ib, ih, it, ic) in LEG_CHAINS.items():
        col = LEG_COLOR[leg]
        hip = links[ih, 0:3]
        thigh = links[it, 0:3]
        calf = links[ic, 0:3]
        Rc = quat_mat(links[ic, 3:7])
        paw = calf + Rc @ FOOT_OFF
        capsule(ax, base_p + (hip - base_p) * 0.0 + (hip - base_p), hip, LIMB_R, col)
        capsule(ax, hip, thigh, LIMB_R, col)
        capsule(ax, thigh, calf, LIMB_R * 0.85, col)
        ax.scatter(*paw, s=(PAW_R * 350) ** 2 * 3.0, color="#2a2c38",
                   depthshade=False, zorder=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="out/go2_handstand_walk/go2_hs_walk.npz")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out-dir", default="out/go2_handstand_walk/render3d")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    data = np.load(os.path.abspath(args.npz))
    frames = data["frames"]                    # (T,N,13,7)
    dx = data["dx_world"]                      # (N,)
    T, N, _, _ = frames.shape
    k = args.env % N
    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    for t in range(0, T, args.stride):
        links = frames[t, k]
        ax.clear()
        ax.set_facecolor("#0c0e14")
        fig.patch.set_facecolor("#0c0e14")

        # ground grid + world-anchored meter ticks (follow camera on base x).
        cx = links[0, 0]
        xs = np.arange(math.floor(cx) - 3, math.ceil(cx) + 4, 0.5)
        for gx in xs:
            major = abs(gx - round(gx)) < 1e-6
            ax.plot([gx, gx], [-1.6, 1.6], [0, 0], color="#2c3242",
                    lw=1.0 if major else 0.5)
        for gy in np.arange(-1.5, 1.6, 0.5):
            ax.plot([cx - 3, cx + 3], [gy, gy], [0, 0], color="#232838", lw=0.5)

        draw_env(ax, links)

        ax.set_xlim(cx - 1.6, cx + 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_zlim(-0.1, 1.5)
        ax.set_box_aspect((3.2, 3.2, 1.6))
        ax.view_init(elev=12, azim=-78)
        ax.axis("off")
        ax.set_title(f"t={t * 0.02:.2f}s  env{k}  dx={dx[k]:+.2f}m",
                     color="#aeb4cc", fontsize=10, pad=0)

        out = os.path.join(os.path.abspath(args.out_dir), f"frame_{t:05d}.png")
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        if (t // args.stride) % 50 == 0:
            print(f"  frame {t}/{T}", flush=True)
    print(f"[render3d] wrote {(T + args.stride - 1) // args.stride} PNGs "
          f"to {os.path.abspath(args.out_dir)} (env {k})")


if __name__ == "__main__":
    main()
