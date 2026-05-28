# Nuka Physics v0.7 – Phase 14: RGB + Depth + Tactile + Force/Torque Sensors

> **Master plan reference:** §3 Round 10 (sensor matrix) + §7 v0.7 exit criteria
> **Prerequisites:** v0.7 Phase 13 (CUDA RT operational for RGB / depth); v0.1 Phase 5 (row lambda for tactile)
> **Blocks:** v0.7 Phase 16 (H1 grasp demo uses tactile + F/T)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Wire the sensor matrix expansion for S2: RGB camera, depth camera, tactile (per-contact), and force/torque (6-axis at joint or link). These join the existing IMU + joint encoder + lidar from S1.

RGB + depth render through the CUDA RT pipeline (Phase 13). Tactile + F/T read directly from constraint row lambda values (Phase 5 infrastructure).

## Tech Stack

- CUDA 12+
- Phase 13 CUDA RT
- Phase 5 row scheduler (for lambda readback)

## Files to Create

- `src/sensor/rgb_camera.hpp`
- `src/sensor/rgb_camera.cu`
- `src/sensor/depth_camera.hpp`
- `src/sensor/depth_camera.cu`
- `src/sensor/tactile.hpp`
- `src/sensor/tactile.cu` — per-contact lambda readback
- `src/sensor/force_torque.hpp`
- `src/sensor/force_torque.cu`
- `src/sensor/sensor_buffer_layout.hpp` — multi-sensor SoA per env
- `src/include/nuka/nuka_sensors.h` — C ABI extension
- `tests/sensor/test_rgb_camera_render.cpp`
- `tests/sensor/test_depth_camera_consistency.cpp`
- `tests/sensor/test_tactile_per_contact.cpp`
- `tests/sensor/test_force_torque_at_wrist.cpp`

## Files to Modify

- `src/runtime/world_stepper.cpp` — sensor sampling at end of step
- `src/include/nuka/nuka.h` — extend `nuka_state_field_t` for sensor fields
- `tests/python/test_sensor_dlpack.py` — Python access to sensor buffers

## Tasks

### Task 7.14.1 — Camera sensor configuration

```cpp
struct CameraConfig {
    float3 position_local;     // relative to a body / link
    float3 forward, up;        // optical axes
    float  fovy;
    float  near, far;
    uint32_t width, height;
    uint32_t attached_body_id; // sensor moves with this body
};
```

Attached to a body or articulation link; sensor follows the link transform each frame.

### Task 7.14.2 — RGB camera

`src/sensor/rgb_camera.cu`:

```cuda
__global__ void rgb_camera_generate_rays_kernel(
    const CameraConfig cam,
    const Transform world_from_link,
    Ray* out_rays);

void RgbCamera::Sample(const phi::DeviceContext& ctx, const Framebuffer& fb_target) {
    rgb_camera_generate_rays_kernel<<<...>>>(config_, link_pose_, ray_buffer_);
    // Use Phase 12 RT to render into fb_target
    rt_render(ray_buffer_, scene_, fb_target);
}
```

Output: per-pixel RGB stored in sensor buffer.

### Task 7.14.3 — Depth camera

Same as RGB but reads only the `depth` aux buffer (Phase 13). Lighter weight (no shading needed). Often disabled shading path for performance.

```cuda
__global__ void depth_only_traversal_kernel(...);
```

### Task 7.14.4 — Tactile sensor (per-contact reading)

`src/sensor/tactile.cu`:

For each contact on the sensorized link, read normal and friction lambda values from the corresponding `Row` in `RowBuffers`. Aggregate:

```
struct ContactPoint {
    float3 position_world;
    float3 normal_world;
    float  normal_force;     // λ_normal
    float2 tangent_force;    // λ_friction_t1, λ_friction_t2
};

// Per link: list of contacts (CSR layout)
```

This is essentially zero-cost: Phase 5 already computes the lambdas; the sensor just collects + reshapes them.

### Task 7.14.5 — Force/torque sensor (6-axis)

For a "wrist" sensor (between two bodies / links), the 6-axis wrench is the sum of all contact / constraint forces transferred through the sensor frame.

For Featherstone articulations: read joint constraint lambda directly (already computed by ABA).

For free-rigid + maximal joints: integrate contact lambdas at the body the sensor is attached to.

```cpp
struct ForceTorqueReading {
    float3 force;
    float3 torque;
};
```

### Task 7.14.6 — Sensor buffer layout (multi-env SoA)

`src/sensor/sensor_buffer_layout.hpp`:

```cpp
struct SensorBuffers {
    // RGB cameras: [env × camera × H × W × 3]
    float* rgb_buffers;
    uint32_t rgb_camera_count;
    uint32_t rgb_width, rgb_height;
    // Depth: [env × camera × H × W]
    float* depth_buffers;
    // Tactile: variable count per env (CSR)
    uint32_t* tactile_per_env_offset;
    ContactPoint* tactile_contacts;
    // F/T: [env × sensor × 6]
    float* force_torque_buffers;
    uint32_t ft_sensor_count;
    // IMU + lidar (existing from S1)
    // joint encoder (from Featherstone)
};
```

All SoA, all GPU-resident, all DLPack-exportable.

### Task 7.14.7 — Sensor sampling in world step

End of step:

```cpp
void WorldStepper::Step(float dt) {
    // ... physics ...

    // Sensor sampling (only if enabled)
    if (rgb_cameras_enabled_) {
        for (auto& cam : rgb_cameras_) cam.Sample(ctx_, sensor_buffers_.rgb_for(cam.id));
    }
    if (depth_cameras_enabled_) {
        for (auto& cam : depth_cameras_) cam.Sample(ctx_, sensor_buffers_.depth_for(cam.id));
    }
    if (tactile_enabled_) {
        tactile_sampler_.Sample(buffers_, sensor_buffers_.tactile);
    }
    if (ft_enabled_) {
        ft_sampler_.Sample(buffers_, sensor_buffers_.force_torque);
    }
}
```

For training (4096 envs), only enable the sensors needed by the task (avoid unnecessary RT rendering).

### Task 7.14.8 — C ABI surface

```c
typedef enum {
    NUKA_FIELD_IMU = 10,
    NUKA_FIELD_LIDAR = 11,
    NUKA_FIELD_RGB = 12,
    NUKA_FIELD_DEPTH = 13,
    NUKA_FIELD_TACTILE = 14,
    NUKA_FIELD_FORCE_TORQUE = 15,
    NUKA_FIELD_JOINT_ENCODER = 16,
} nuka_sensor_field_t;

nuka_result_t nuka_world_get_sensor_buffer_view(nuka_world_handle w,
                                                 nuka_sensor_field_t field,
                                                 uint32_t sensor_index,
                                                 nuka_buffer_view_t* out);
```

DLPack exports follow.

### Task 7.14.9 — Tests

`tests/sensor/test_rgb_camera_render.cpp`:

```cpp
// 16 envs, 1 camera each at known position
// Render Go2 stand pose
// Verify image is non-uniform (something is being rendered)
// Verify per-env images differ if env states differ
```

`tests/sensor/test_tactile_per_contact.cpp`:

```cpp
// Set up Go2 standing on ground
// Verify tactile sensor on each foot reports normal force ~ body_mass * g / 4
// Verify when robot lifts a foot, that foot's tactile reading drops to 0
```

## Validation

- RGB camera renders multi-env scenes; per-env independent.
- Depth camera matches scene geometry (sanity comparison vs RGB).
- Tactile per-contact readings match expected forces in known scenarios.
- F/T sensor at wrist correctly reports applied wrench in test loads.
- Step time impact: ≤ 30% overhead with all sensors enabled at 64x64 RGB (for many parallel envs).

## Exit Criteria for v0.7 Phase 14

1. Four new sensors (RGB, depth, tactile, F/T) operational.
2. Per-env, per-sensor SoA buffer layout.
3. C ABI + Python DLPack access.
4. Per-sensor on/off configurable.
5. Performance budget met when sensors disabled.
6. Tests pass.

## What This Phase Does Not Do

- No sim-to-real noise yet (v1.0 N3).
- No semantic segmentation (v1.0).
- No event camera (v2.0).
- No camera distortion / rolling shutter (v1.0 N3 phase).
- No texture sampling on RGB (Phase 13 used flat materials; texturing is v1.0).
