#!/usr/bin/env python3
"""[p04-T2 demo] Rasterize the captured Go2 rollout (.npz) to PPM frames.

RENDER-PATH HONESTY
-------------------
This is a SELF-CONTAINED Python rasterizer (numpy -> PPM P6), NOT the documented
offscreen-Vulkan path (src/render/vulkan_renderer.cpp / nuka_scene_demo). We use
it because:
  * The C++ nuka_scene_demo runs its OWN default-drive physics; it has no way to
    load the trained policy or to ingest externally-captured policy poses without
    a C++ rebuild + new pose-injection plumbing.
  * Its debug view is a TOP-DOWN X/Y projection, which hides the walking gait.
The offscreen-Vulkan path DOES work on this box (verified: a 640x360 Go2 frame
and a 16-env batched frame both rendered via lavapipe to PPM) -- but it rendered
the demo's internal default-drive sim, NOT the policy. So for the policy-walk
video we draw the captured world link poses ourselves, side-on (X/Z), as a stick
skeleton, in a 4x4 grid. Same PPM -> ffmpeg encode tail as the documented path.

The skeleton per env is drawn DIRECTLY from ARTICULATION_LINK_POSE world positions
(13 links): a trunk box at the base, and 4 legs as polylines
base -> hip -> thigh -> calf -> foot. Nothing is faked; every joint position is
the simulated world pose of that link.

Run:
  /root/miniconda3/envs/nuka-v03/bin/python examples/demo/go2_demo_render.py \
      --npz out/go2_demo/go2_rollout.npz --frames-dir out/go2_demo/frames \
      --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# scene link order (go2_float.usda): base, then per leg hip/thigh/calf.
# leg chains as link indices into the 13-link pose array.
LEG_CHAINS = {
    "FL": (0, 1, 2, 3),
    "FR": (0, 4, 5, 6),
    "RL": (0, 7, 8, 9),
    "RR": (0, 10, 11, 12),
}
CALF_INDICES = {"FL": 3, "FR": 6, "RL": 9, "RR": 12}
FOOT_OFFSET_LOCAL = np.array([0.0, 0.0, -0.213], np.float32)  # calf-frame foot offset

# leg colors (left legs warm, right legs cool) so the gait reads in the side view.
LEG_COLOR = {
    "FL": (255, 170, 60),   # orange
    "FR": (90, 200, 255),   # cyan
    "RL": (255, 100, 120),  # red
    "RR": (120, 230, 140),  # green
}
TRUNK_COLOR = (235, 235, 245)
GROUND_COLOR = (70, 80, 95)
BG = (10, 12, 18)


def quat_rotate_wxyz(q_wxyz, v):
    """Rotate vector v by quaternion q=[w,x,y,z] (world = R(q) @ v_local)."""
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], np.float64)
    t = 2.0 * np.cross(qv, v)
    return (v + w * t + np.cross(qv, t)).astype(np.float32)


class Canvas:
    """Tiny RGB raster with line + filled-box primitives (numpy)."""

    def __init__(self, w, h, bg=BG):
        self.w, self.h = w, h
        self.img = np.empty((h, w, 3), np.uint8)
        self.img[:] = np.array(bg, np.uint8)

    def _in(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def line(self, p0, p1, color, thick=2):
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        col = np.array(color, np.uint8)
        r = thick // 2
        while True:
            for ox in range(-r, r + 1):
                for oy in range(-r, r + 1):
                    xx, yy = x0 + ox, y0 + oy
                    if self._in(xx, yy):
                        self.img[yy, xx] = col
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def disc(self, p, radius, color):
        cx, cy = int(round(p[0])), int(round(p[1]))
        col = np.array(color, np.uint8)
        r = int(round(radius))
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                if ox * ox + oy * oy <= r * r and self._in(cx + ox, cy + oy):
                    self.img[cy + oy, cx + ox] = col

    def hline(self, y, x0, x1, color):
        y = int(round(y))
        if not (0 <= y < self.h):
            return
        x0 = max(0, int(round(x0)))
        x1 = min(self.w - 1, int(round(x1)))
        self.img[y, x0:x1 + 1] = np.array(color, np.uint8)


def project_xz(p_world, cam):
    """World (x,z) -> pixel (col,row) within a sub-cell. cam: dict with origin
    pixel (ox,oy=bottom), scale (px/m), and x_center (world x mapped to cell mid)."""
    px = cam["ox"] + (p_world[0] - cam["x_center"]) * cam["scale"]
    py = cam["oy"] - (p_world[2] - 0.0) * cam["scale"]   # z up -> row up; ground z=0
    return (px, py)


def draw_env(canvas, frame_links, cell, scale, x_center, ok=True):
    """Draw one env's skeleton in a cell. frame_links: (13,7) world [xyz,wxyz].
    cell: (cx0,cy0,cw,ch) pixel rect. Side view (X right, Z up). A FOLLOW-CAM keeps
    the robot centered; WORLD-ANCHORED ground ticks (vertical marks at integer
    world-x) scroll under the feet so the forward TRAVERSAL reads -- without them a
    re-centered robot looks like it is pedaling in place."""
    cx0, cy0, cw, ch = cell
    ground_row = cy0 + int(ch * 0.82)
    cam = {"ox": cx0 + cw * 0.5, "oy": ground_row, "scale": scale, "x_center": x_center}

    # ground line for this cell.
    canvas.hline(ground_row, cx0 + 4, cx0 + cw - 4, GROUND_COLOR)

    # WORLD-ANCHORED scrolling ground ticks: vertical marks every 0.5 m of world-x,
    # projected with the SAME per-frame follow-cam, so they slide left as the robot
    # advances -- this is what makes locomotion (not in-place gait) unmistakable.
    half_span_m = (cw * 0.5) / scale + 1.0
    x_lo = x_center - half_span_m
    x_hi = x_center + half_span_m
    tick0 = np.ceil(x_lo / 0.5) * 0.5
    xt = tick0
    while xt <= x_hi:
        px = cam["ox"] + (xt - cam["x_center"]) * cam["scale"]
        if cx0 + 2 <= px <= cx0 + cw - 2:
            major = abs(xt - round(xt)) < 1e-3   # taller tick at integer meters
            tlen = 12 if major else 6
            canvas.line((px, ground_row), (px, ground_row + tlen), GROUND_COLOR, thick=1)
        xt += 0.5

    if not ok:
        # mark an env that failed its per-env walk check (kept visible but flagged).
        for ox in range(6):
            canvas.line((cx0 + 6 + ox, cy0 + 6), (cx0 + 18 + ox, cy0 + 18),
                        (220, 60, 60), thick=1)

    base_p = frame_links[0, 0:3]
    base_q = frame_links[0, 3:7]
    # trunk: draw a short oriented bar along body-x through the base (0.20 cube
    # half-len ~ 0.10 each side, scaled visually a bit for readability).
    half = 0.13
    bx0 = base_p + quat_rotate_wxyz(base_q, np.array([-half, 0, 0], np.float32))
    bx1 = base_p + quat_rotate_wxyz(base_q, np.array([+half, 0, 0], np.float32))
    canvas.line(project_xz(bx0, cam), project_xz(bx1, cam), TRUNK_COLOR, thick=5)
    canvas.disc(project_xz(base_p, cam), 4, TRUNK_COLOR)

    # legs: base -> hip -> thigh -> calf -> foot polyline, per leg.
    for leg, chain in LEG_CHAINS.items():
        pts = [frame_links[i, 0:3] for i in chain]
        # foot tip = calf pos + calf-frame foot offset.
        calf_i = CALF_INDICES[leg]
        calf_q = frame_links[calf_i, 3:7]
        foot = frame_links[calf_i, 0:3] + quat_rotate_wxyz(calf_q, FOOT_OFFSET_LOCAL)
        pts.append(foot)
        col = LEG_COLOR[leg]
        for a, b in zip(pts[:-1], pts[1:]):
            canvas.line(project_xz(a, cam), project_xz(b, cam), col, thick=2)
        canvas.disc(project_xz(foot, cam), 3, col)  # foot marker


def write_ppm(path, img):
    h, w, _ = img.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(img).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="out/go2_demo/go2_rollout.npz")
    ap.add_argument("--frames-dir", default="out/go2_demo/frames")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = all captured frames")
    ap.add_argument("--grid", type=int, default=4, help="grid is grid x grid")
    args = ap.parse_args()

    data = np.load(os.path.abspath(args.npz))
    frames = data["frames"]            # (T,N,13,7)
    cmds = data["cmds"]                # (N,3)
    dx_world = data["dx_world"]        # (N,) net world-x advance per env
    T, N, L, _ = frames.shape
    grid = args.grid
    n_show = min(N, grid * grid)
    if args.max_frames > 0:
        T = min(T, args.max_frames)

    # PER-ENV walk check (honor the "no flailing env" claim, which the aggregate
    # capture gate alone does not guarantee): an env is OK iff it advanced forward
    # (dx>0.1 m) AND stayed upright every frame (base z never collapsed, tilt small).
    base_z_all = frames[:, :, 0, 2]                       # (T,N)
    # tilt per frame from base quat z-axis: proj_grav_z = -(1-2(qx^2+qy^2)); tilt
    # via base 'up' world-z component. up_z = 1-2(qx^2+qy^2).
    qx = frames[:, :, 0, 4]
    qy = frames[:, :, 0, 5]
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)                # (T,N) cos(tilt)
    tilt_max_per_env = np.degrees(np.arccos(np.clip(up_z.min(axis=0), -1.0, 1.0)))
    z_min_per_env = base_z_all.min(axis=0)
    env_ok = (dx_world > 0.1) & (tilt_max_per_env < 35.0) & (z_min_per_env > 0.18)
    print(f"[render] per-env walk check: {int(env_ok.sum())}/{N} envs OK "
          f"(dx>0.1m & tilt_max<35deg & z_min>0.18m); "
          f"flagged: {np.where(~env_ok)[0].tolist()}")

    print(f"[render] npz {args.npz}: T={frames.shape[0]} frames, N={N} envs, "
          f"link={L}; drawing {n_show} envs in {grid}x{grid}, {T} frames "
          f"@ {args.width}x{args.height}")

    os.makedirs(os.path.abspath(args.frames_dir), exist_ok=True)

    # Per-cell sizing.
    cw = args.width // grid
    ch = args.height // grid

    # Fixed scale + a per-env CAMERA that PANS with the robot so the walking robot
    # stays centered in its cell (the body moves forward several meters; without a
    # follow-cam it would walk off the cell). Scale chosen so the ~0.7 m tall robot
    # fills the cell vertically.
    scale = ch * 0.42 / 0.7      # px per meter (robot ~0.7 m standing extent)

    # Precompute each env's base-x track for the follow-cam center per frame.
    base_x = frames[:, :, 0, 0]   # (T,N)

    for t in range(T):
        canvas = Canvas(args.width, args.height)
        for cell_idx in range(n_show):
            gx = cell_idx % grid
            gy = cell_idx // grid
            cx0 = gx * cw
            cy0 = gy * ch
            x_center = float(base_x[t, cell_idx])   # follow-cam: keep robot centered
            draw_env(canvas, frames[t, cell_idx], (cx0, cy0, cw, ch), scale,
                     x_center, ok=bool(env_ok[cell_idx]))
        # thin grid separators
        for g in range(1, grid):
            canvas.img[:, g * cw, :] = np.array((40, 46, 58), np.uint8)
            canvas.img[g * ch, :, :] = np.array((40, 46, 58), np.uint8)
        write_ppm(os.path.join(os.path.abspath(args.frames_dir), f"frame_{t:05d}.ppm"),
                  canvas.img)
        if (t % 25) == 0 or t == T - 1:
            print(f"  frame {t+1}/{T}", flush=True)

    print(f"[render] wrote {T} PPM frames to {os.path.abspath(args.frames_dir)}")
    print(f"[render] cmd vx spread across grid: "
          f"{np.round(cmds[:n_show,0],3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
