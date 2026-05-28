# Nuka Physics v1.0 – Phase 5: CUDA RT Optimization Pass

> **Master plan reference:** §3 Round 10 (self-written CUDA RT) + §7 v1.0 exit
> **Prerequisites:** v0.7 Phase 12 + 13 (RT pipeline operational); v1.0 Phases 2-4 (sim2real features added load)
> **Blocks:** v1.0 Phases 7-8 (pour / wring demos need throughput; multi-env training in v1.5 needs RT throughput)
> **Exit criteria gate:** v1.0
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

The v0.7 CUDA RT is correct but unoptimized — sim2real features (Phases 2-4) added overhead, and the v1.5 warehouse scene (S3) will need more throughput than the baseline supports. This phase tunes the RT pipeline for production performance.

Target: from baseline ~50 MRays/s (v0.7 Phase 12) to ~150 MRays/s on RTX 4090 for typical sensor workloads.

## Tech Stack

- CUDA 12+
- NVIDIA Nsight Compute (kernel profiling)
- NVIDIA Nsight Systems (timeline tracing)
- Phase 12-13 RT infrastructure

## Files to Create

- `src/rt/lbvh_morton_optimized.cu` — improved BVH quality via finer-grained SAH
- `src/rt/persistent_thread_scheduler_optimized.cuh` — better load balancing
- `src/rt/scene_compilation_cache.hpp` — incremental BVH update
- `src/rt/perf_counters.hpp` — RT-specific perf counters
- `tools/perf/run_rt_perf_sweep.py`
- `tests/perf/test_rt_throughput_sensor_workloads.cpp`
- `docs/architecture/rt-perf-tuning.md`

## Files to Modify

- `src/rt/rt_lbvh.cu` — adopt optimized build path
- `src/rt/traversal_stackless.cu` — register pressure reduction
- `src/rt/intersect_*.cuh` — vectorized intersection where applicable

## Tasks

### Task 10.5.1 — Profile and identify bottlenecks

Use Nsight Compute on existing v0.7 scenes. Capture:
- Memory throughput (achieved vs theoretical max)
- Warp occupancy
- L1/L2 cache hit rate
- Divergence per warp
- Compute / memory bound ratio

Document baseline in `docs/architecture/rt-perf-tuning.md`.

### Task 10.5.2 — BVH quality improvement (SAH-driven build)

v0.7's RT-LBVH used basic Karras + minor SAH refit. Replace with:
- **Top-down SAH BVH** (Wald 2014 style) for static / slow-moving geometry.
- Keep fast Karras refit for very dynamic content.
- Auto-select per geometry's "dynamism score" (rate of position change).

Typical 20-40% traversal speedup from better tree quality.

### Task 10.5.3 — Persistent thread scheduler tuning

- Adjust block size and persistent thread count for the GPU's SM count.
- Use shared memory for the work counter to reduce atomic contention.
- Stride ray distribution to improve cache locality (adjacent threads process adjacent pixels).

### Task 10.5.4 — Intersection kernel SIMD-style optimization

- Vectorize triangle intersection: process 4 triangles per thread (warp-level work).
- Use `__ldg` for read-only loads where confirmed beneficial.
- Reduce register pressure in `traverse_lbvh_stackless` (currently spills to local memory on RTX 4090).

### Task 10.5.5 — Scene compilation caching

For dynamic scenes where most rigid bodies don't move (training environments where only the robot moves):
- Build BVH for static geometry once.
- Refit only the dynamic subset.
- Combine via two-level BVH (instance-BVH containing leaf-BVHs).

This dramatically helps the warehouse scene (v1.5 — 50K rigids, most static).

### Task 10.5.6 — Sensor-specific kernels

Different sensors stress RT differently:
- Lidar: ~64K rays, narrow FOV.
- RGB 640x480: ~300K rays, ~1 ms budget.
- RGB 4096-env × 64×64: ~16M rays.

Build specialized kernel variants per workload pattern; route at runtime.

### Task 10.5.7 — Tests

`tests/perf/test_rt_throughput_sensor_workloads.cpp`:

```cpp
TEST(RtPerf, LidarSingleBeam64k) {
    auto scene = LoadBenchmarkScene("warehouse_lite");
    auto rays = GenerateLidarRays(64000);
    auto t0 = now_us();
    TraceRays(rays);
    auto throughput = 64000.0 / (now_us() - t0) * 1e6;
    EXPECT_GT(throughput, 150e6) << "Below 150 MRays/s target";
}

TEST(RtPerf, RgbCamera640x480) { ... }
TEST(RtPerf, MultiEnvRgb4096x64x64) { ... }
```

### Task 10.5.8 — Determinism preserved

All optimizations must preserve D1. After each optimization commit, re-run determinism test (same ray batch → same output). If broken: revert.

## Validation

- 150 MRays/s on typical sensor workloads.
- Sim2real features (Phases 2-4) overhead reduced.
- D1 determinism preserved.
- All Phase 13 correctness tests still pass.

## Exit Criteria for v1.0 Phase 5

1. Baseline + optimized perf metrics documented.
2. SAH BVH build operational.
3. Two-level BVH for hybrid static/dynamic scenes.
4. Persistent thread scheduler tuned.
5. Intersection kernels optimized.
6. Sensor-workload throughput targets met.
7. Determinism preserved.

## What This Phase Does Not Do

- No new RT features (deferred — focus is perf).
- No diff-rendering (v2.0 Phase 7).
- No RT Cores (master plan §3 Round 10 commits to self-written, no OptiX).
- No multi-GPU RT (S3 fits single GPU).
