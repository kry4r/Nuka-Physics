#!/usr/bin/env python3
"""Settle and bound-check the MLS-MPM prefix of an authored BDX scene."""

from __future__ import annotations

import argparse
import json

import numpy as np
import nuka

from bdx_oneshot_perf import scene_geometry


def run(scene, steps):
    geometry = scene_geometry(scene)
    n_mpm = geometry["mpm_particles_per_env"]
    origin = np.asarray(geometry["grid_origin"], dtype=np.float64)
    dims = np.asarray(geometry["grid_dims"], dtype=np.int64)
    domain_max = origin + geometry["dx"] * (dims - 1)
    snapshots = {}
    with nuka.Device.create(0) as device:
        builder = nuka.SceneBuilder.create(scene)
        world = builder.build(device, env_count=1, dt=1.0 / 240.0,
                              control_mode=0, bake_link_sdf=True)
        builder.destroy()
        capture_steps = {150, 300, 450, max(1, steps - 10), steps}
        for step in range(1, steps + 1):
            world.step()
            if step in capture_steps:
                particles = np.asarray(
                    world.download_field(nuka.Field.PARTICLE_POSITION),
                    dtype=np.float32,
                ).reshape(-1, 3)[:n_mpm].copy()
                snapshots[step] = particles
                print(
                    f"step={step} z=[{particles[:, 2].min():.6f},"
                    f"{particles[:, 2].max():.6f}] "
                    f"mean={particles[:, 2].mean():.6f}",
                    flush=True,
                )
        world.destroy()
    particles = snapshots[steps]
    finite = np.isfinite(particles).all(axis=1)
    outside = np.any((particles < origin) | (particles > domain_max), axis=1)
    lane_escape = (
        (particles[:, 0] < 0.87)
        | (particles[:, 0] > 3.13)
        | (np.abs(particles[:, 1]) > 1.0)
        | (particles[:, 2] < -0.08)
        | (particles[:, 2] > 0.5)
    )
    speed_window = max(1, steps - 10)
    speed10 = np.linalg.norm(
        particles - snapshots[speed_window], axis=1
    ) / ((steps - speed_window) / 240.0)
    drift_step = max(k for k in snapshots if k <= steps - 150)
    drift150 = np.linalg.norm(particles - snapshots[drift_step], axis=1)
    result = {
        **geometry,
        "grid_max": domain_max.tolist(),
        "nonfinite": int((~finite).sum()),
        "outside_grid": int(outside.sum()),
        "lane_escape": int(lane_escape.sum()),
        "z_min": float(particles[:, 2].min()),
        "z_mean": float(particles[:, 2].mean()),
        "z_max": float(particles[:, 2].max()),
        "speed10_mean": float(speed10.mean()),
        "speed10_p95": float(np.percentile(speed10, 95)),
        "speed10_max": float(speed10.max()),
        "drift150_mean": float(drift150.mean()),
        "drift150_p95": float(np.percentile(drift150, 95)),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    result = run(args.scene, args.steps)
    print(json.dumps(result, sort_keys=True), flush=True)
    stable = (result["nonfinite"] == 0 and result["outside_grid"] == 0 and
              result["lane_escape"] == 0)
    raise SystemExit(0 if stable else 1)


if __name__ == "__main__":
    main()
