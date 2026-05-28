# Nuka Physics v1.0 – Phase 4: Sim-to-Real N3 — Lidar Beam Divergence + Motion Blur

> **Master plan reference:** §3 Round 10 (sim2real N3 noise)
> **Prerequisites:** v1.0 Phases 2 (intrinsics) + 3 (rolling shutter — shares ring buffer infrastructure)
> **Blocks:** v1.0 Phase 9 (exit gate retrospective)
> **Exit criteria gate:** v1.0
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Add two remaining N3 sensor physics models:
1. **Lidar beam divergence** — real lidar beams are cones, not lines. Returns from the cone are mixed; near-edge objects produce noisy returns.
2. **Motion blur** for RGB camera — fast-moving objects in long-exposure frames smear.

Both effects matter for sim-to-real in S3 (warehouse) and S5 (VLA) scenarios where robots move quickly and observe dynamic scenes.

## Tech Stack

- CUDA 12+
- Phase 3 world state ring buffer (for motion blur exposure window)
- Phase 12/13 CUDA RT

## Files to Create

- `src/sensor/lidar_beam_divergence.cu` — sample N rays within beam cone
- `src/sensor/motion_blur.cu` — accumulate frames within exposure window
- `src/sensor/n3_config.hpp` — central config for all N3 effects
- `tests/sensor/test_lidar_beam_divergence_known_geometry.cpp`
- `tests/sensor/test_motion_blur_static_vs_moving.cpp`

## Tasks

### Task 10.4.1 — Lidar beam divergence

Real lidar: each "ray" is a beam with angular divergence (typically 0.05° to 1°). Returns are an aggregate of intersections within the beam cone.

Model:
- For each lidar beam, sample N rays (typically 4-16) uniformly within the divergence cone.
- For each sampled ray, intersect scene.
- Aggregate returns: mean distance if all returns are coherent; if dispersion is high, output a "noisy" return.

```cuda
__global__ void lidar_beam_divergent_trace_kernel(
    uint32_t beam_count,
    const float3* beam_origin, const float3* beam_dir,
    float divergence_rad,
    uint32_t samples_per_beam,
    uint32_t rng_seed,
    const RtLbvhNode* bvh,
    float* out_range,           // aggregated range per beam
    float* out_intensity)       // aggregated intensity
{
    uint32_t beam_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (beam_idx >= beam_count) return;

    float3 axis = beam_dir[beam_idx];
    float3 origin = beam_origin[beam_idx];

    float range_sum = 0.f;
    uint32_t hits = 0;
    for (uint32_t s = 0; s < samples_per_beam; ++s) {
        // Stratified sample within cone
        float2 disk = stratified_disk_sample(beam_idx, s, samples_per_beam, rng_seed);
        float3 perturbed_dir = normalize_perturbation(axis, disk, divergence_rad);
        Ray r{origin, perturbed_dir, 0.f, 100.f};
        Intersection isect = traverse_lbvh_stackless(r, bvh);
        if (!is_miss(isect)) {
            range_sum += isect.t;
            hits++;
        }
    }

    out_range[beam_idx] = hits > 0 ? range_sum / hits : -1.f;
    out_intensity[beam_idx] = float(hits) / samples_per_beam;
}
```

Lower intensity → less confident return → more noise added by N1 Gaussian.

### Task 10.4.2 — Motion blur

Long-exposure camera: integrates light over the exposure window. Fast-moving objects appear smeared.

Model:
- For each pixel, sample N timestamps within exposure window.
- For each timestamp, trace the ray against the world state at that time (using ring buffer from Phase 3).
- Average shaded RGB.

```cuda
__global__ void motion_blur_render_kernel(
    uint32_t width, uint32_t height,
    CameraIntrinsics K, Transform cam_pose_base,
    const WorldStateHistory* history,
    float frame_time, float exposure_time,
    uint32_t samples_per_pixel,
    float3* out_color)
{
    uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;

    float3 accumulated = float3{0,0,0};
    for (uint32_t s = 0; s < samples_per_pixel; ++s) {
        float t_sample = frame_time + (s + 0.5f) / samples_per_pixel * exposure_time;
        auto world_at_t = interpolate_world_state(history, t_sample);
        Ray r = generate_ray_with_intrinsics(K, interp_cam_pose_at(world_at_t, ...), u, v);
        Intersection isect = traverse_lbvh_stackless(r, scene_bvh_at(world_at_t));
        accumulated += shade(isect, r, materials, lights);
    }
    out_color[v*width + u] = accumulated * (1.f / samples_per_pixel);
}
```

`samples_per_pixel` typically 4-8. Heavy cost; enable only when needed.

### Task 10.4.3 — Combined with rolling shutter

Motion blur and rolling shutter compose. The per-row time from rolling shutter is the *exposure window center* for that row. Motion blur samples within that row's exposure window.

```cuda
float t_row_center = frame_time + (float(v) / height) * readout_time;
// Samples within [t_row_center - exposure/2, t_row_center + exposure/2]
```

### Task 10.4.4 — N3 config struct

```cpp
struct N3Config {
    // Lens distortion (Phase 2)
    CameraIntrinsics intrinsics = {};
    // Rolling shutter (Phase 3)
    ShutterConfig shutter = {};
    // Motion blur (this phase)
    bool motion_blur = false;
    float exposure_time_s = 0.005f;
    uint32_t motion_blur_samples = 4;
    // Lidar beam divergence (this phase)
    float lidar_beam_divergence_rad = 0.0035f; // 0.2 deg typical
    uint32_t lidar_samples_per_beam = 8;
};
```

Per-camera / per-lidar; per-episode randomizable.

### Task 10.4.5 — Performance considerations

These effects compound multiplicatively. Combined motion blur + rolling shutter + lens distortion ~ 10× slower than basic RGB.

Mitigation:
- Reduce samples per pixel when training.
- Enable only for sim-to-real evaluation runs.
- Make a separate "fast sensor mode" for training.

### Task 10.4.6 — Tests

`tests/sensor/test_lidar_beam_divergence_known_geometry.cpp`:

```cpp
// Cube edge facing lidar beam
// With zero divergence: clean single distance
// With significant divergence: returns either edge-distance, surface-distance, or a mix
// Verify variance increases with divergence
```

`tests/sensor/test_motion_blur_static_vs_moving.cpp`:

```cpp
// Static scene: motion blur output == single-sample output (no smearing)
// Moving object: blur extent matches expected per object's velocity × exposure
```

## Validation

- Lidar at zero divergence == single-ray result.
- Lidar at typical divergence shows expected variance.
- Motion blur on static scene matches single-shot.
- Motion blur on moving object shows expected smear.
- Per-episode randomization works.

## Exit Criteria for v1.0 Phase 4

1. Lidar beam divergence operational.
2. Motion blur operational.
3. Combined with rolling shutter + lens distortion correctly composed.
4. N3 config struct + per-episode randomization.
5. Static vs moving baseline tests pass.

## What This Phase Does Not Do

- No event camera (v2.0).
- No depth sensor IR-pattern modeling (RealSense / Kinect specific; potential future).
- No global illumination / GI ray tracing (this is shading model fidelity, not sensor physics).
- No HDR sensor saturation modeling (out of scope for v1.0).
