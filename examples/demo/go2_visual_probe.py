#!/usr/bin/env python3
"""Minimal proof that go2.nks renders the real visual meshes in Python."""
from __future__ import annotations

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, "python")
import nuka
from nuka.author import Scene, SimOptions, morphs
from nuka.author.render import hero_orbit


def main() -> None:
    out = os.path.abspath("out/go2_visual_probe")
    os.makedirs(out, exist_ok=True)
    dev = nuka.Device.create(0)
    scene = Scene(SimOptions(dt=1.0 / 240.0, env_count=1, contact_family=1))
    scene.add_entity(morphs.NKS("examples/scenes/go2.nks"))
    scene.add_entity(morphs.Ground())
    world = scene.build(dev)
    for _ in range(120):
        world.step()
    for i, az in enumerate((-0.9, -0.35, 0.2)):
        eye = (2.0 * np.cos(az), 2.0 * np.sin(az), 1.1)
        look = (0.0, 0.0, 0.35)
        img = world.render_beauty(eye=eye, look=look, width=1280, height=720, spp=16)
        Image.fromarray(np.ascontiguousarray(img)).save(os.path.join(out, f"frame_{i:02d}.png"))
    world.destroy()
    dev.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
