#!/usr/bin/env python3
"""Textured-PBR + HDRI beauty showcase, authored entirely from Python.

Assembles the BDX duck (bdx_stand.nks) on a gravel plot with wooden steps and a
fabric cushion, lights it with an outdoor HDRI environment, and beauty-renders
a few high-sample frames (plus an optional turntable mp4 when ffmpeg is on
PATH). Every appearance is data on the ONE render path: materials.Render bags
with CC0 texture maps + Scene.set_environment.

Run from the repo root:
    python examples/demo/render_realism_demo.py [--out DIR] [--spp 256] [--video]
"""

import argparse
import math
import os
import subprocess

import numpy as np
from PIL import Image

import nuka
from nuka.author import Scene, SimOptions, materials, morphs

TEX = "examples/assets/textures"


def build_world(dev):
    s = Scene(SimOptions(dt=1.0 / 240.0, env_count=1))
    s.add_entity(morphs.NKS("examples/scenes/bdx_stand.nks"))

    wood = materials.Render(
        "wood_planks",
        albedo_map=f"{TEX}/wood_planks/albedo.jpg",
        roughness_map=f"{TEX}/wood_planks/roughness.jpg",
        normal_map=f"{TEX}/wood_planks/normal.jpg",
        uv_scale=3.0)
    gravel = materials.Render(
        "gravel",
        albedo_map=f"{TEX}/gravel/albedo.jpg",
        roughness_map=f"{TEX}/gravel/roughness.jpg",
        normal_map=f"{TEX}/gravel/normal.jpg",
        uv_scale=1.8)
    fabric = materials.Render(
        "fabric",
        base_color=(0.75, 0.30, 0.28), sheen=0.5,
        albedo_map=f"{TEX}/fabric/albedo.jpg",
        roughness_map=f"{TEX}/fabric/roughness.jpg",
        normal_map=f"{TEX}/fabric/normal.jpg",
        uv_scale=9.0)

    rigid = materials.Rigid(static=True)
    # Gravel field: covers the whole studio floor disc, top 2 mm above it (no
    # coplanar fight); beyond its rim the HDRI horizon takes over.
    s.add_entity(morphs.Box((8.0, 8.0, 0.05), pos=(0.0, 0.0, -0.048)),
                 rigid, render=gravel)
    # Three wooden steps rising away from the duck (+x).
    for i, (hz, x) in enumerate([(0.035, 0.38), (0.070, 0.68), (0.105, 0.98)]):
        s.add_entity(morphs.Box((0.15, 0.42, hz), pos=(x, 0.0, 0.002 + hz)),
                     rigid, render=wood)
    # A fabric cushion beside the steps.
    s.add_entity(morphs.Box((0.11, 0.11, 0.045), pos=(0.28, -0.55, 0.047),
                            quat=(0.966, 0.0, 0.0, 0.259)),
                 rigid, render=fabric)

    s.set_environment(hdri=f"{TEX}/hdri/sky_2k.hdr", yaw_deg=-40.0,
                      intensity=1.0, use_scene_materials=True)
    return s.build(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/render_realism")
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--video", action="store_true",
                    help="also render a 72-frame turntable and mp4 it")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with nuka.Device.create(0) as dev:
        world = build_world(dev)
        for _ in range(60):
            world.step()

        look = (0.28, 0.0, 0.16)
        views = [
            ("01_three_quarter", (1.35, -1.05, 0.62)),
            ("02_steps_low", (1.15, 0.85, 0.30)),
            ("03_wide", (1.9, 1.35, 0.85)),
            ("04_front_low", (-0.95, -0.60, 0.35)),
        ]
        for name, eye in views:
            img = world.render_beauty(eye=eye, look=look, width=args.width,
                                      height=args.height, spp=args.spp)
            path = os.path.join(args.out, f"{name}.png")
            Image.fromarray(np.ascontiguousarray(img)).save(path)
            print(f"[render_realism] {path}")

        if args.video:
            frames = os.path.join(args.out, "turn")
            os.makedirs(frames, exist_ok=True)
            for k in range(72):
                a = 2.0 * math.pi * k / 72.0
                eye = (look[0] + 1.5 * math.cos(a), look[1] + 1.5 * math.sin(a),
                       0.6)
                img = world.render_beauty(eye=eye, look=look, width=args.width,
                                          height=args.height,
                                          spp=max(64, args.spp // 4))
                Image.fromarray(np.ascontiguousarray(img)).save(
                    os.path.join(frames, f"f_{k:04d}.png"))
            mp4 = os.path.join(args.out, "turntable.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-framerate", "24", "-i",
                 os.path.join(frames, "f_%04d.png"), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", mp4], check=False)
            print(f"[render_realism] {mp4}")


if __name__ == "__main__":
    main()
