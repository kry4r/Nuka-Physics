#!/usr/bin/env python3
"""Reusable BDX corridor camera pipeline gate.

Builds the real rigid corridor, renders the legacy full camera tensor and the
DEPTH|PRIM primary-hit profile from the same live world, checks depth/prim byte
identity, verifies omitted channels are inaccessible, and emits one JSON record
with timing + authored scene statistics. Run each env/resolution configuration as
an independent process so peak VRAM and failure attribution stay honest.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO / "examples" / "scenes" / "corridor_nomedia.nks"


def _enum_value(value) -> int:
    return int(value.value) if hasattr(value, "value") else int(value)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = q * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    alpha = rank - lo
    return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _authored_stats(scene: Path) -> dict[str, int]:
    doc = json.loads(scene.read_text(encoding="utf-8"))
    visual = collision = nodes = 0
    stack = [doc]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            visual += int("visual_mesh" in value)
            collision += int("collision_shape" in value)
            nodes += int("name" in value and "children" in value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return {
        "tree_nodes": nodes,
        "visual_mesh_records": visual,
        "collision_shape_records": collision,
        "media_records": len(doc.get("media", [])),
    }


def _time_render(nuka, world, warmup: int, samples: int) -> list[float]:
    for _ in range(warmup):
        world.render_sensors()
    nuka.sync()
    timings: list[float] = []
    for _ in range(samples):
        nuka.sync()
        start = time.perf_counter()
        world.render_sensors()
        nuka.sync()
        timings.append((time.perf_counter() - start) * 1.0e3)
    return timings


def run(args) -> dict:
    import torch
    import nuka

    scene = Path(args.scene).resolve()
    if not scene.is_file():
        raise FileNotFoundError(scene)
    paired = scene.with_suffix(".nka")
    if not paired.is_file():
        raise FileNotFoundError(f"paired scene asset is missing: {paired}")

    result = {
        "scene": str(scene),
        "paired_asset": str(paired),
        "env_count": args.envs,
        "width": args.width,
        "height": args.height,
        "warmup": args.warmup,
        "samples": args.samples,
        **_authored_stats(scene),
    }

    mount_base = _enum_value(nuka.SensorMount.BASE)
    depth_mask = _enum_value(nuka.SensorAov.DEPTH)
    prim_mask = _enum_value(nuka.SensorAov.PRIM)
    fast_mask = depth_mask | prim_mask

    with nuka.Device.create(0) as device:
        builder = nuka.SceneBuilder.create(str(scene))
        world = builder.build(
            device, env_count=args.envs, dt=1.0 / 240.0, control_mode=0
        )
        builder.destroy()
        # Identity camera rotation means local -Z looks down. This preflight pose
        # guarantees scene coverage while isolating renderer cost; the BDX head
        # mount is authored and validated by the downstream sensor-fusion gate.
        world.attach_camera_sensor(
            mount_base,
            0,
            [0.0, 0.0, args.camera_height, 1.0, 0.0, 0.0, 0.0],
            args.vfov,
            args.width,
            args.height,
        )

        compare_envs = min(args.compare_envs, args.envs)
        full_times = None
        full_depth = full_prim = None
        if not args.fast_only:
            world.set_sensor_aov_mask(0)
            full_times = _time_render(nuka, world, args.warmup, args.samples)
            full_depth = torch.from_dlpack(
                world.get_sensor_view(nuka.SensorChannel.DEPTH)
            )[:compare_envs].clone()
            full_prim = torch.from_dlpack(
                world.get_sensor_view(nuka.SensorChannel.PRIM)
            )[:compare_envs].clone()

        world.set_sensor_aov_mask(fast_mask)
        fast_times = _time_render(nuka, world, args.warmup, args.samples)
        fast_depth = torch.from_dlpack(
            world.get_sensor_view(nuka.SensorChannel.DEPTH)
        )
        fast_prim = torch.from_dlpack(
            world.get_sensor_view(nuka.SensorChannel.PRIM)
        )

        depth_equal = (
            None if full_depth is None
            else bool(torch.equal(full_depth, fast_depth[:compare_envs]))
        )
        prim_equal = (
            None if full_prim is None
            else bool(torch.equal(full_prim, fast_prim[:compare_envs]))
        )
        sample_depth = fast_depth[0]
        valid = torch.isfinite(sample_depth) & (sample_depth < 1.0e20)
        omitted_color_rejected = False
        try:
            world.get_sensor_view(nuka.SensorChannel.COLOR)
        except RuntimeError:
            omitted_color_rejected = True

        result.update(
            {
                "sensor_dims": [int(v) for v in world.sensor_dims()],
                "compared_envs": compare_envs,
                "full": None if full_times is None else _summary(full_times),
                "depth_prim": _summary(fast_times),
                "speedup": None if full_times is None else (
                    statistics.fmean(full_times) / statistics.fmean(fast_times)
                ),
                "depth_byte_equal": depth_equal,
                "prim_byte_equal": prim_equal,
                "omitted_color_rejected": omitted_color_rejected,
                "valid_depth_fraction": float(valid.float().mean().item()),
                "depth_min": float(sample_depth[valid].min().item()),
                "depth_max": float(sample_depth[valid].max().item()),
            }
        )
        world.destroy()

    if not args.fast_only and (
        not result["depth_byte_equal"] or not result["prim_byte_equal"]
    ):
        raise RuntimeError("primary-hit depth/prim differs from the legacy full path")
    if not result["omitted_color_rejected"]:
        raise RuntimeError("an omitted AOV exposed stale color data")
    if args.max_fast_ms > 0.0 and result["depth_prim"]["mean_ms"] > args.max_fast_ms:
        raise RuntimeError(
            f"depth/prim mean {result['depth_prim']['mean_ms']:.3f} ms exceeds "
            f"gate {args.max_fast_ms:.3f} ms"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--vfov", type=float, default=60.0)
    parser.add_argument("--camera-height", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--compare-envs", type=int, default=8)
    parser.add_argument(
        "--fast-only", action="store_true",
        help="measure DEPTH|PRIM without allocating the legacy full AOV tensor",
    )
    parser.add_argument("--max-fast-ms", type=float, default=0.0)
    args = parser.parse_args()
    if args.envs <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("--envs/--width/--height must be positive")
    if args.warmup < 0 or args.samples <= 0 or args.compare_envs <= 0:
        parser.error("--warmup must be >=0; --samples/--compare-envs must be >0")
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
