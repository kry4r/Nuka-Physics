#!/usr/bin/env python3
"""Reproducible step-time, VRAM, particle, and MPM-grid probe for BDX scenes."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
import time


MIB = 1024 * 1024


def summarize(samples):
    ordered = sorted(float(x) for x in samples)
    if not ordered:
        raise ValueError("at least one timing sample is required")
    rank = 0.95 * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    weight = rank - lo
    p95 = ordered[lo] * (1.0 - weight) + ordered[hi] * weight
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": p95,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _box_particle_count(box):
    spacing = float(box["spacing"])
    if spacing <= 0.0:
        return 0
    dims = [
        max(0, int(math.floor((float(box["max"][axis]) -
                               float(box["min"][axis])) / spacing + 1.0e-6)))
        for axis in range(3)
    ]
    return math.prod(dims)


def scene_geometry(scene):
    with open(scene, encoding="utf-8") as stream:
        doc = json.load(stream)
    granular = next(m for m in doc["media"] if m["kind"] == "granular")
    boxes = [granular["fluid_box"]] + [fill["box"]
                                               for fill in granular.get("mpm_fills", [])]
    low = [min(float(box["min"][axis]) for box in boxes) for axis in range(3)]
    high = [max(float(box["max"][axis]) for box in boxes) for axis in range(3)]
    mpm = granular["mpm"]
    dx = float(mpm["dx"])
    margin = 4.0 * dx
    base_z = min(low[2], float(mpm["floor_d"])) - margin
    top_z = high[2] + (high[2] - low[2]) + margin + max(
        0.0, float(mpm["loft_headroom"]))
    extents = [
        high[0] - low[0] + 2.0 * margin,
        high[1] - low[1] + 2.0 * margin,
        top_z - base_z,
    ]
    grid_dims = [int(math.ceil(extent / dx)) + 1 for extent in extents]
    counts = [_box_particle_count(box) for box in boxes]
    return {
        "zone_b_particles_per_env": counts[0],
        "zone_c_particles_per_env": sum(counts[1:]),
        "mpm_particles_per_env": sum(counts),
        "grid_dims": grid_dims,
        "grid_nodes_per_env": math.prod(grid_dims),
        "grid_origin": [low[0] - margin, low[1] - margin, base_z],
        "grid_top_z": top_z,
        "dx": dx,
        "substeps": int(mpm["substeps"]),
        "loft_headroom": float(mpm["loft_headroom"]),
    }


def _cudart():
    for name in ("libcudart.so.12", "libcudart.so"):
        try:
            lib = ctypes.CDLL(name)
            lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t)]
            lib.cudaMemGetInfo.restype = ctypes.c_int
            return lib
        except OSError:
            continue
    return None


def _cuda_mem_info(lib):
    if lib is None:
        return None
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    if lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
        return None
    return int(free.value), int(total.value)


def run(args):
    import numpy as np
    import nuka

    scene = os.path.abspath(args.scene)
    geometry = scene_geometry(scene)
    cudart = _cudart()
    result = {
        "scene": scene,
        "env_count": args.envs,
        "warmup_steps": args.warmup,
        "sample_count": args.samples,
        "steps_per_sample": args.steps_per_sample,
        "measured_steps": args.samples * args.steps_per_sample,
        "wrench_demand": bool(args.wrench),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        **geometry,
    }
    with nuka.Device.create(0) as device:
        mem0 = _cuda_mem_info(cudart)
        builder = nuka.SceneBuilder.create(scene)
        world = builder.build(device, env_count=args.envs, dt=1.0 / 240.0,
                              control_mode=0, bake_link_sdf=True)
        builder.destroy()
        if args.wrench:
            _ = world.buffer_view(nuka.LINK_CONTACT_WRENCH)
        positions = np.asarray(
            world.download_field(nuka.Field.PARTICLE_POSITION), dtype=np.float32
        ).reshape(-1, 3)
        mem_build = _cuda_mem_info(cudart)
        for _ in range(args.warmup):
            world.step()
        nuka.sync()
        mem_warm = _cuda_mem_info(cudart)
        samples = []
        for _ in range(args.samples):
            nuka.sync()
            start = time.perf_counter()
            for _ in range(args.steps_per_sample):
                world.step()
            nuka.sync()
            elapsed = time.perf_counter() - start
            samples.append(elapsed * 1.0e3 / args.steps_per_sample)
        mem_measured = _cuda_mem_info(cudart)
        result.update(summarize(samples))
        result["samples_ms"] = samples
        result["particle_rows_total"] = int(positions.shape[0])
        result["particle_rows_per_env"] = int(positions.shape[0] // args.envs)
        valid_mem = [m for m in (mem_build, mem_warm, mem_measured) if m is not None]
        if mem0 is not None and valid_mem:
            min_free = min(m[0] for m in valid_mem)
            delta = max(0, mem0[0] - min_free)
            steady_used = mem0[1] - mem_measured[0]
            result["vram_delta_mib"] = delta / MIB
            result["vram_delta_mib_per_env"] = delta / MIB / args.envs
            result["device_used_mib_after_measure"] = steady_used / MIB
        world.destroy()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--steps-per-sample", type=int, default=10)
    parser.add_argument("--wrench", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), sort_keys=True), flush=True)
    except Exception as exc:
        failure = {
            "scene": os.path.abspath(args.scene),
            "env_count": args.envs,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
