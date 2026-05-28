# Nuka Physics v1.0 – Phase 3: Sim-to-Real N3 — Rolling Shutter

> **Master plan reference:** §3 Round 10 (sim2real N3 noise)
> **Prerequisites:** v1.0 Phase 2 (camera intrinsics)
> **Blocks:** v1.0 Phase 7+ (realistic visual policies need accurate motion artifacts)
> **Exit criteria gate:** v1.0
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Add a **rolling shutter** camera model. Most robot cameras (RealSense, MIPI smartphone cameras, mvBlueFOX, etc.) use rolling shutters — pixel rows are read out sequentially over a finite "readout time," not at a single instant. Fast-moving objects appear sheared/skewed.

Without rolling shutter modeling, policies trained on simulated images expecting clean shutters fail when deployed on real cameras with significant readout time (typically 30-100 ms for full frame).

## Tech Stack

- CUDA 12+
- Phase 2 camera intrinsics + ray generation
- v0.5 Phase 2 tape (need per-substep state for rolling shutter "snapshot per row")

## Files to Create

- `src/sensor/rolling_shutter.hpp`
- `src/sensor/rolling_shutter.cu` — per-row time interpolation
- `src/sensor/shutter_config.hpp`
- `tests/sensor/test_rolling_shutter_known_motion.cpp`
- `tests/sensor/test_rolling_shutter_vs_global_shutter.cpp`

## Tasks

### Task 10.3.1 — Shutter config

```cpp
struct ShutterConfig {
    enum class Mode { Global, RollingTopToBottom, RollingLeftToRight };
    Mode mode = Mode::Global;
    float readout_time_s = 0.020f;     // 20 ms typical
    uint32_t row_count_for_interp = 60; // sample N intermediate states; interpolate per row
};
```

Per-camera config.

### Task 10.3.2 — Multi-state buffering

Rolling shutter: each row's exposure happens at a slightly different scene time. To render this faithfully, we need scene state at multiple substeps within the exposure window.

Approach:
- Engine maintains a small ring buffer of recent world states (last 8-16 substeps).
- At render time, for each output row `r`, compute the corresponding scene time `t_r = frame_time + (r / total_rows) * readout_time`.
- Linearly interpolate the world state at `t_r` from the ring buffer; render that row.

Memory cost: ~8 × world state size per env. For Go2: ~10 KB × 8 = 80 KB / env; for 4096 envs = 320 MB. Acceptable.

### Task 10.3.3 — Per-row ray generation + render

```cuda
__global__ void rolling_shutter_render_kernel(
    uint32_t width, uint32_t height,
    CameraIntrinsics K, Transform world_from_cam_frame_base,
    const WorldStateHistory* history,
    float frame_time, float readout_time,
    float3* out_color)
{
    uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;

    // Time at which this row was exposed
    float t_row = frame_time + (float(v) / height) * readout_time;

    // Interpolate world state at t_row
    auto world_at_t = interpolate_world_state(history, t_row);

    // Build ray with world state at this time
    Transform cam_pose = interp_cam_pose_at(world_at_t, ...);
    Ray r = generate_ray_with_intrinsics(K, cam_pose, u, v);

    // Trace + shade (Phase 12 + 13 RT pipeline)
    Intersection isect = traverse_lbvh_stackless(r, scene_bvh_at(world_at_t));
    float3 color = shade(isect, r, materials, lights);
    out_color[v * width + u] = color;
}
```

Key complexity: the BVH itself is per-state. Approach: rebuild BVH for the "midpoint" state of the readout window; treat per-row time as small perturbation (bodies' positions interpolated within the static BVH). This is approximate but works when bodies don't move multiple BVH leaves in 20 ms.

### Task 10.3.4 — Cost mitigation

Rolling shutter rendering is significantly more expensive than global shutter. Mitigations:
- Lower resolution for rolling shutter sensors (320x240 instead of 640x480).
- Per-N-row interpolation (sample world state every 10 rows; interpolate within).
- Disable rolling shutter for training-only mode; enable for sim2real eval.

Configurable via `ShutterConfig::row_count_for_interp`.

### Task 10.3.5 — Verification: known motion

Generate a known scenario:
- Camera mounted on H1 head.
- H1 rotates rapidly.
- A vertical pole in the scene should appear *tilted* in rolling-shutter output, *straight* in global-shutter output.

```cpp
TEST(RollingShutter, TiltedPoleUnderRotation) {
    // ... set up scenario ...
    auto img_rs = RenderWithShutter(ShutterConfig::RollingTopToBottom, ...);
    auto img_gs = RenderWithShutter(ShutterConfig::Global, ...);
    auto tilt_rs = MeasureLineSegmentAngle(img_rs, /*pole_pixels*/);
    auto tilt_gs = MeasureLineSegmentAngle(img_gs, /*pole_pixels*/);
    EXPECT_GT(abs(tilt_rs - tilt_gs), 0.5f);  // rolling differs from global
}
```

### Task 10.3.6 — Per-episode randomization

Add `readout_time` to domain randomization range (e.g., 10-30 ms uniform per episode).

## Validation

- Rotating-rod test: rolling-shutter output shows expected tilt.
- Static scene: rolling-shutter == global-shutter (consistency).
- Rendering cost is roughly 1.5-2× global shutter (acceptable).
- D1 determinism preserved (per-row interpolation is deterministic).

## Exit Criteria for v1.0 Phase 3

1. Rolling shutter mode operational on RGB camera.
2. Per-row time interpolation working.
3. World state ring buffer integrated.
4. Static-scene consistency check passes.
5. Motion-artifact check passes.
6. Per-episode randomization extended.

## What This Phase Does Not Do

- No row-wise per-pixel exposure (assume all pixels in a row sample at one time).
- No lidar rolling shutter (Phase 4 covers lidar specifically).
- No imaging noise post-shutter (already covered by N1).
