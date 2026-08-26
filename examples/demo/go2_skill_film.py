#!/usr/bin/env python3
"""Film a skill policy on a SINGLE env with the engine's real 3D beauty render.

Drives the trained policy deterministically on one env, and every --stride
control steps captures the LIVE world through World.render_beauty (the same
path the cloth-drape and terrain demos use -- actual robot meshes, not stick
figures). Camera follows the base with a slow hero orbit. Muxes an mp4 when
ffmpeg is available.

Run from repo root:
    PYTHONPATH=python python examples/demo/go2_skill_film.py \
        --task hs --checkpoint runs/.../ckpt.pth --seconds 10
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, "python")

import nuka  # noqa: F401
from nuka.author.render import hero_orbit

WIDTH, HEIGHT, SPP = 1280, 720, 16


def load_policy(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    sd = ck["model"]
    rms_mean = sd["running_mean_std.running_mean"].float()
    rms_var = sd["running_mean_std.running_var"].clamp(min=1e-6).float()
    layers = []
    for i in (0, 2, 4):
        w = sd[f"a2c_network.actor_mlp.{i}.weight"]
        layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
        layers[-1].weight.data = w.float()
        layers[-1].bias.data = sd[f"a2c_network.actor_mlp.{i}.bias"].float()
        layers.append(torch.nn.ELU())
    mu = torch.nn.Linear(128, 12)
    mu.weight.data = sd["a2c_network.mu.weight"].float()
    mu.bias.data = sd["a2c_network.mu.bias"].float()
    return torch.nn.Sequential(*layers, mu).cuda().eval(), rms_mean, rms_var.sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["hs", "bf"], default="hs")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=5, help="control steps per frame")
    ap.add_argument("--out", default="out/skill_film")
    args = ap.parse_args()

    if args.task == "hs":
        from nuka.tasks.go2_front_handstand import make_env
        env = make_env(1)
    else:
        from nuka.tasks.go2_backflip import make_env
        env = make_env(1, action_mode="pd")

    policy, rm, rv = load_policy(args.checkpoint)
    obs, _ = env.reset()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    frames = []
    steps = int(args.seconds * 50)
    with torch.no_grad():
        for t in range(steps):
            x = ((obs - rm) / rv.sqrt()).float()
            a = policy(x).clamp(-1.0, 1.0)
            obs, r, term, trunc, _ = env.step(a)
            if t % args.stride == 0:
                bp = env._obs.base_pos()[0].detach().cpu().numpy()
                e = min(1.0, t / max(1, steps))
                eye, look = hero_orbit(
                    e, center=(float(bp[0]), float(bp[1]), float(bp[2]) + 0.15),
                    radius=1.6, pull_in=0.35, drop=0.05)
                img = env._world.render_beauty(
                    eye=tuple(eye), look=tuple(look),
                    width=WIDTH, height=HEIGHT, spp=SPP)
                path = os.path.join(out, f"frame_{len(frames):04d}.png")
                img.save(path) if hasattr(img, "save") else _write_png(img, path)
                frames.append(path)
                if len(frames) % 10 == 0:
                    print(f"  frame {len(frames)} @ t={t * 0.02:.1f}s", flush=True)
    env.close()

    ffmpeg = shutil.which("ffmpeg") or (
        "tools/ffmpeg/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe")
    if os.path.exists(ffmpeg) or shutil.which("ffmpeg"):
        mp4 = os.path.join(out, "skill_film.mp4")
        subprocess.run([ffmpeg, "-y", "-framerate", "25",
                        "-i", os.path.join(out, "frame_%04d.png"),
                        "-pix_fmt", "yuv420p", mp4],
                       check=False, capture_output=True)
        print(f"[film] mp4: {mp4}")
    print(f"[film] {len(frames)} frames -> {out}")


def _write_png(arr, path):
    from PIL import Image
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    main()
