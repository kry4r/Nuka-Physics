#!/usr/bin/env python3
"""Profile the real SceneBuilder world pipeline and optionally enforce gates.

This is intentionally a reusable pipeline/integration probe, not a local unit
test.  One invocation exercises

    scene load -> SceneIR -> rigid/media cook -> model upload -> step -> readback

and reports build time, process peak RSS, CUDA memory growth, particle count and
finite-state status as one JSON record.  It is suitable for cold/warm cache runs
of any ``.nks`` scene; BDX additionally passes ``--bake-link-sdf`` so its visual
link SDFs participate in the MPM coupling path.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import pathlib
import resource
import statistics
import time


MIB = 1024 * 1024


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


def _percentile(values, q):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    rank = q * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    alpha = rank - lo
    return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha


def _sdf_cache_stats(root):
    files = list(pathlib.Path(root).glob("*.nukasdf"))
    return len(files), sum(path.stat().st_size for path in files)


def run(args):
    import numpy as np
    import nuka

    scene = os.path.abspath(args.scene)
    cache_root = os.path.abspath(".nuka_cache")
    cache_before = _sdf_cache_stats(cache_root)
    cudart = _cudart()
    mem_before = _cuda_mem_info(cudart)
    result = {
        "scene": scene,
        "env_count": args.envs,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "bake_link_sdf": bool(args.bake_link_sdf),
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "sdf_cache_root": cache_root,
        "sdf_cache_files_before": cache_before[0],
        "sdf_cache_bytes_before": cache_before[1],
    }

    process_start = time.perf_counter()
    with nuka.Device.create(0) as device:
        create_start = time.perf_counter()
        builder = nuka.SceneBuilder.create(scene)
        result["scene_load_ms"] = (time.perf_counter() - create_start) * 1.0e3

        build_start = time.perf_counter()
        world = builder.build(
            device,
            env_count=args.envs,
            dt=args.dt,
            control_mode=0,
            bake_link_sdf=args.bake_link_sdf,
        )
        result["world_build_ms"] = (time.perf_counter() - build_start) * 1.0e3
        builder.destroy()
        mem_after_build = _cuda_mem_info(cudart)

        for _ in range(args.warmup):
            world.step()
        nuka.sync()

        step_ms = []
        for _ in range(args.steps):
            start = time.perf_counter()
            world.step()
            nuka.sync()
            step_ms.append((time.perf_counter() - start) * 1.0e3)

        particles = np.asarray(
            world.download_field(nuka.Field.PARTICLE_POSITION), dtype=np.float32
        ).reshape(-1, 3)
        links = np.asarray(
            world.download_field(nuka.Field.ARTICULATION_LINK_POSE), dtype=np.float32
        ).reshape(-1, 7)
        result["particle_rows_total"] = int(particles.shape[0])
        result["particle_rows_per_env"] = int(particles.shape[0] // args.envs)
        result["particle_state_finite"] = bool(np.isfinite(particles).all())
        result["link_pose_rows_total"] = int(links.shape[0])
        result["link_pose_state_finite"] = bool(np.isfinite(links).all())
        result["step_mean_ms"] = statistics.fmean(step_ms) if step_ms else 0.0
        result["step_p95_ms"] = _percentile(step_ms, 0.95)
        mem_after_readback = _cuda_mem_info(cudart)
        world.destroy()

    result["pipeline_wall_ms"] = (time.perf_counter() - process_start) * 1.0e3
    cache_after = _sdf_cache_stats(cache_root)
    result["sdf_cache_files_after"] = cache_after[0]
    result["sdf_cache_bytes_after"] = cache_after[1]
    result["sdf_cache_files_added"] = cache_after[0] - cache_before[0]
    result["sdf_cache_bytes_added"] = cache_after[1] - cache_before[1]
    # ru_maxrss is KiB on Linux. This probe is launched in a fresh process, so
    # the value captures transient cook peaks as well as the final live world.
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    valid_mem = [m for m in (mem_after_build, mem_after_readback) if m is not None]
    if mem_before is not None and valid_mem:
        min_free = min(m[0] for m in valid_mem)
        result["vram_delta_mib"] = max(0, mem_before[0] - min_free) / MIB

    failures = []
    if not result["particle_state_finite"]:
        failures.append("particle readback contains NaN/Inf")
    if not result["link_pose_state_finite"]:
        failures.append("articulation link-pose readback contains NaN/Inf")
    if args.max_build_ms is not None and result["world_build_ms"] > args.max_build_ms:
        failures.append(
            f"world_build_ms={result['world_build_ms']:.3f} > {args.max_build_ms:.3f}"
        )
    if args.max_rss_mib is not None and result["peak_rss_mib"] > args.max_rss_mib:
        failures.append(
            f"peak_rss_mib={result['peak_rss_mib']:.3f} > {args.max_rss_mib:.3f}"
        )
    result["gate_failures"] = failures
    print(json.dumps(result, sort_keys=True), flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--bake-link-sdf", action="store_true")
    parser.add_argument("--max-build-ms", type=float)
    parser.add_argument("--max-rss-mib", type=float)
    args = parser.parse_args()
    if args.envs <= 0 or args.warmup < 0 or args.steps < 0:
        parser.error("envs must be positive and step counts non-negative")
    run(args)


if __name__ == "__main__":
    main()
