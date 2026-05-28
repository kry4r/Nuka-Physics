# Nuka Physics v1.0 – Phase 2: Sim-to-Real N3 — Lens Distortion + Camera Intrinsics

> **Master plan reference:** §3 Round 10 (sim2real N3 noise)
> **Prerequisites:** v0.7 (RGB + depth sensors via CUDA RT)
> **Blocks:** v1.0 Phase 7+ demos (realistic visual sensors needed for true sim2real evaluation)
> **Exit criteria gate:** v1.0
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Add **realistic camera intrinsics + lens distortion** to the RGB / depth pipeline. v0.5 sim2real N1 added Gaussian/Poisson noise; v0.5 N2 added per-episode domain randomization. v1.0 N3 starts adding physical sensor models — this phase covers the most impactful one for visual policies: lens model.

Specifically:
- Pinhole camera model with `fx, fy, cx, cy` intrinsics.
- Brown-Conrady radial + tangential distortion (`k1, k2, p1, p2, k3`).
- Per-episode randomized intrinsics (within calibration uncertainty).

## Tech Stack

- CUDA 12+
- Phase 13 CUDA RT pipeline
- Phase 4 v0.5 noise infrastructure

## Files to Create

- `src/sensor/camera_intrinsics.hpp`
- `src/sensor/camera_intrinsics.cu`
- `src/sensor/lens_distortion.cuh` — forward + inverse distortion
- `src/sensor/lens_distortion_adjoint.cuh` — for diff-sim through distorted camera
- `tools/calibration/camera_intrinsics_yaml_loader.cpp` — load real calibration files
- `tests/sensor/test_camera_intrinsics_pinhole.cpp`
- `tests/sensor/test_lens_distortion_forward_inverse_consistency.cpp`
- `tests/sensor/test_camera_distortion_real_data.cpp` — verify against OpenCV reference

## Tasks

### Task 10.2.1 — Camera intrinsics structure

```cpp
struct CameraIntrinsics {
    float fx, fy;       // focal lengths in pixels
    float cx, cy;       // principal point
    float k1, k2, k3;   // radial distortion
    float p1, p2;       // tangential distortion
};
```

Replaces the simple FOV-based projection from v0.7 RGB camera.

### Task 10.2.2 — Distortion forward

Brown-Conrady model:
- Compute normalized image coordinates `(x, y)`.
- Apply radial: `r² = x² + y²; x_d = x(1 + k1 r² + k2 r⁴ + k3 r⁶); y_d = y(1 + k1 r² + k2 r⁴ + k3 r⁶)`.
- Apply tangential: `x_d += 2 p1 x y + p2 (r² + 2 x²); y_d += p1 (r² + 2 y²) + 2 p2 x y`.
- Project to pixel: `u = fx x_d + cx; v = fy y_d + cy`.

For each pixel `(u, v)` in the output image, ray tracing needs the **ray direction in camera space**. This requires the **inverse distortion** — given pixel, find the undistorted direction.

### Task 10.2.3 — Inverse distortion (Newton's method)

No closed form; iterate:

```cuda
__device__ float2 inverse_brown_conrady(float2 d, const CameraIntrinsics& K)
{
    float2 x = d;   // initial guess
    for (int i = 0; i < 5; ++i) {
        float2 dp = brown_conrady_forward(x, K) - d;
        // dp/dx ≈ identity → step = -dp
        x -= dp;
    }
    return x;
}
```

Test: forward(inverse(p)) = p within 1e-5 for all pixels (consistency check).

### Task 10.2.4 — Ray generation

`src/sensor/camera_intrinsics.cu`:

```cuda
__global__ void generate_camera_rays_with_intrinsics_kernel(
    CameraIntrinsics K,
    Transform world_from_camera,
    uint32_t width, uint32_t height,
    Ray* out_rays)
{
    uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;

    float2 distorted_normalized = float2{
        (float(u) - K.cx) / K.fx,
        (float(v) - K.cy) / K.fy
    };
    float2 undistorted = inverse_brown_conrady(distorted_normalized, K);
    float3 dir_cam = normalize(float3{undistorted.x, undistorted.y, 1.f});
    float3 dir_world = transform_dir(world_from_camera, dir_cam);

    out_rays[v * width + u] = Ray{world_from_camera.translation, dir_world, 0.f, 1e6f};
}
```

### Task 10.2.5 — Per-episode randomization

Extend `DomainRandomizationConfig`:

```cpp
struct CameraIntrinsicsRange {
    float fx_lo, fx_hi;     // typically ±5%
    float fy_lo, fy_hi;
    float cx_offset_pixels; // ±5 pixels
    float cy_offset_pixels;
    float k1_lo, k1_hi;     // typically ±0.05
    float k2_lo, k2_hi;
    float p1_lo, p1_hi;
    float p2_lo, p2_hi;
};
```

On env reset, sample these per env; store in env's CameraIntrinsics; subsequent renders use sampled values.

### Task 10.2.6 — Real calibration YAML loader

For sim2real benchmarking, allow loading **real robot camera calibration** files (OpenCV / ROS YAML format):

```yaml
image_width: 640
image_height: 480
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distortion_coefficients:
  data: [k1, k2, p1, p2, k3]
```

Engine consumes the file; sets intrinsics on the matching camera.

### Task 10.2.7 — Diff-sim adjoint

Lens distortion enters the forward path; for diff-sim through the rendered image (v2.0 diff-rendering), we need the adjoint of the distortion model.

Implementation: closed-form adjoint of the forward distortion (Jacobian); for inverse distortion, use implicit-function-theorem at convergence (the Newton solve).

For v1.0, ship the analytical forward adjoint; the inverse adjoint can defer to v2.0 when actual diff-rendering goes live.

### Task 10.2.8 — Tests

`tests/sensor/test_lens_distortion_forward_inverse_consistency.cpp`:

```cpp
TEST(LensDistortion, ForwardInverseRoundTrip) {
    auto K = LoadKnownIntrinsics();
    for (auto [u, v] : SampleGridPixels(640, 480)) {
        float2 dist = {(u-K.cx)/K.fx, (v-K.cy)/K.fy};
        float2 undist = inverse_brown_conrady(dist, K);
        float2 redist = brown_conrady_forward(undist, K);
        EXPECT_NEAR(redist.x, dist.x, 1e-5f);
        EXPECT_NEAR(redist.y, dist.y, 1e-5f);
    }
}
```

`tests/sensor/test_camera_distortion_real_data.cpp`:

```cpp
TEST(CameraDistortion, MatchesOpenCvReference) {
    // Use a known calibration (e.g., RealSense D435 calibration)
    // Render a synthetic checkerboard
    // Compare distorted pattern in our output to OpenCV's undistort/distort path
    // Pixel agreement < 0.5 pixel mean error
}
```

## Validation

- Forward + inverse distortion consistent within 1e-5.
- Real RealSense calibration loaded and applied successfully.
- Per-episode randomization produces varied intrinsics within configured ranges.
- Adjoint forward distortion verified via FD < 1e-3 rel error.

## Exit Criteria for v1.0 Phase 2

1. Brown-Conrady forward + inverse implemented.
2. Camera ray generation uses intrinsics + distortion.
3. Real calibration YAML loader operational.
4. Per-episode randomization extended.
5. Forward distortion adjoint passes FD check.
6. Tests pass.

## What This Phase Does Not Do

- No rolling shutter (Phase 3).
- No lidar beam divergence (Phase 4).
- No motion blur (Phase 4).
- No inverse distortion adjoint (deferred to v2.0).
