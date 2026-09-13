# Sensor observations

Physical state and measured observations have separate storage. `world.buffer_view(field)` and `world.download_field(field)` return physical state. Mounted sensors sample automatically during `world.step()`. Scalar field observations use explicit acquisition and separate observation views.

## Mounted state sensors

Attach sensors before or after stepping; attaching one rebuilds the execution plan while preserving existing views:

```python
imu = world.attach_state_sensor(
    nuka.StateSensorKind.IMU,
    mount=nuka.SensorMount.BASE, mount_index=0,
    local_offset=(0.02, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0),
    update_period=2,
    latency=0.01, latency_jitter=0.001,
    dropout_probability=0.01, seed=42,
)
nuka.MeasurementError(
    bias=0.01, noise_density=0.002, bias_random_walk=0.0001,
    correlated_bias_stddev=0.001, correlation_time=30.0,
).configure_sensor(world, imu, component=0)
world.step_n(10)
values = world.download_state_sensor(imu)  # shape: (environments, values)
stamp = world.state_sensor_stamp(imu, env=0)
```

`local_offset` is position xyz followed by quaternion wxyz. `LINK` selects a cooked link within each environment, `BASE` selects an articulation root, and `BODY` selects an owning rigid body. Articulation body mounts resolve to the corresponding link. These indices belong to the cooked model; imported SceneIR body and joint references are resolved during cooking.

| Kind | Output and units |
| --- | --- |
| `IMU` | Specific force xyz in m/s², then angular velocity xyz in rad/s, both in sensor axes |
| `FRAME_POSE` | World position xyz in meters, then unit quaternion wxyz |
| `JOINT_STATE` | Position and velocity: rad, rad/s for revolute joints; m, m/s for prismatic joints |
| `LINEAR_VELOCITY` | Mounting point linear velocity xyz in m/s, expressed in sensor axes |
| `CONTACT_WRENCH` | Contact force xyz in N, then contact torque xyz in N·m, at the mounting point in sensor axes |
| `FORCE_TORQUE` | Parent-on-subtree force xyz in N and torque xyz in N·m, at the mounting point in sensor axes |

IMU acquisition integrates velocity changes over physical substeps and subtracts gravity. Free fall therefore gives zero specific force at the center of mass; supported rest measures the opposing gravity vector. Off-center mounts include angular acceleration and centripetal acceleration. Solved contact impulses are included. Each substep uses the midpoint sensor orientation; high angular rates still require adequate integration resolution. Position and velocity sensors acquire the endpoint state.

Contact transducers sum normal and friction impulses on all collision shapes belonging to the mounting body or link. Dynamic bodies and static force plates use the same contact rows, including contacts with XPBD particles and MLS-MPM grid endpoints. Torque includes the lever arm to `local_offset`; there is no spatial patch filter. A body mount must select the owning body, not an extra collision-shape proxy.

`FORCE_TORQUE` measures the load transmitted from the parent into the selected link and its descendants. It uses live spatial inertias, changes in linear and spin momentum, gravity, and solved external contacts, then accumulates child loads through the articulation tree. Static weight, actuator effort transmitted to links, joint limits, friction and contact reactions contribute to this balance. Off-center centers of mass and rotated inertia frames are included. A fixed articulation root measures support from the world; free bodies and floating roots have no parent interface and are rejected. Place the sensor on a supported articulation link, such as a wrist or fixed tool attachment.

Both load sensors integrate impulses over physical substeps and report their average over the acquisition interval. Lever arms and sensor axes use substep midpoints. This is a discrete momentum-balance measurement; it converges to continuous inverse-dynamics loads as the integration timestep decreases. Split-impulse position corrections do not count as physical momentum. Torque is expressed about the sensor point, even if the sensor is offset from the actual joint.

```python
wrist_load = world.attach_state_sensor(
    nuka.StateSensorKind.FORCE_TORQUE,
    mount=nuka.SensorMount.LINK, mount_index=wrist_link,
    local_offset=(0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0),
)
nuka.MeasurementError(noise_density=0.01, bias=0.02).configure_sensor(
    world, wrist_load, component=2,
)
```

Error components follow output order except for pose orientation: components 3–5 are local rotation-vector errors in radians, applied on SO(3). Orientation response filtering preserves unit quaternions. Scalar gain errors are not defined for these orientation components and are rejected. Other state sensors and pose translation use the scalar error parameters below.

`update_period` counts outer timesteps. A positive `sample_rate_hz` overrides it and cannot exceed the integration substep rate. Acquisition and delivery occur at integration boundaries; rates that do not divide that cadence are rounded to the next boundary without accumulating schedule drift. The first acquisition follows one sample period after attachment or reconfiguration.

Delivery uses fixed `latency` plus uniform jitter in `[-latency_jitter, +latency_jitter]`, with negative delays clamped to zero. Packets preserve acquisition order. `dropout_probability` drops deliveries while continuing the sensor's acquisition clock and stochastic error processes. The latest delivered sample is held between deliveries. Queue storage is sized from the maximum delay and sample period; unexpected overflow sets `ENV_STATUS_SENSOR_QUEUE_OVERFLOW`.

`state_sensor_stamp` reports `sequence`, `acquisitions`, `dropped`, `sample_time`, `delivery_time` and `valid`. Times use the environment's simulation clock, which restarts on environment reset. Reading a view or stamp does not acquire a sample. `world.get_state_sensor_view(imu)` provides a float32 device view with the same shape as the downloaded array.

Checkpoint replay includes pending packets, clocks, accumulated IMU and load exposure, filter state and random processes. Restoring a checkpoint from before attachment deactivates later sensors and zeroes their existing views without freeing them. IDs remain stable allocation slots: `state_sensor_count()` includes inactive slots, and `state_sensor_active(id)` identifies active sensors. Reconfiguring an allocated sensor reactivates it and clears its complete sampling history.

MJCF accelerometer/gyro, joint position/velocity, frame pose, velocimeter and force/torque records create these sensors automatically. IMU, joint and force/torque records expose the complete paired outputs shown above. Child bodies without a joint retain a fixed connection and a separate link frame, including tool inertia and mounted sensors. MJCF `touch` remains metadata: its scalar, site-volume normal-force integral differs from a whole-body contact wrench and is not automatically attached. Runtime noise and delivery settings are preserved by checkpoints but are not serialized in NKS. Camera and lidar APIs remain separate. Differentiable Tape creation, stepping and backward replay reject active mounted sensors because Tape does not execute this sampling pipeline.

## Explicit scalar observations

`world.get_observation_view(field)` and `world.download_observation(field)` read manually acquired scalar observations.

Configure an observation once, then acquire after advancing the world:

```python
error = nuka.MeasurementError(
    bias=0.001, noise_density=0.002,
    initial_bias_stddev=0.003, bias_random_walk=0.0001,
    correlated_bias_stddev=0.001, correlation_time=30.0,
    quantization=0.0001,
    temperature_coefficient=0.0002, reference_temperature=25.0,
    seed=42,
)
error.configure(world, nuka.JOINT_VELOCITY)
world.step_n(4)
world.sample_observation(nuka.JOINT_VELOCITY, 4 * dt, temperature=35.0)
measured_velocity = world.get_observation_view(nuka.JOINT_VELOCITY)
```

The example's velocity errors are in radians per second for revolute joints. Values are configurable examples, not a calibrated model of a particular encoder. Mixed joint types require units consistent with each observed quantity.

The scalar field interface accepts float32 scalar fields. Integer IDs, status bits, poses and structured vectors are rejected. State acquisition for mounted sensors and image formation require their own physical models; adding noise to a joint field does not construct an IMU or force/torque sensor.

## Error parameters

| Parameter | Meaning |
| --- | --- |
| `bias` | Constant additive calibration offset, in measurement units |
| `scale_error` | Fractional gain error; measured gain is `1 + scale_error` |
| `noise_density` | Square root of two-sided power spectral density, in measurement units × √s |
| `initial_bias_stddev` | Fixed Gaussian offset drawn independently for each element and environment after reset |
| `bias_random_walk` | Bias diffusion amplitude, in measurement units / √s |
| `correlated_bias_stddev`, `correlation_time` | Stationary standard deviation and correlation time in seconds of an Ornstein–Uhlenbeck bias |
| `response_time` | First-order response time in seconds; zero disables filtering |
| `temperature_coefficient` | Additive drift per degree Celsius relative to `reference_temperature` |
| `quantization` | Measurement units per least significant bit; zero disables rounding |
| `minimum`, `maximum` | Optional output saturation bounds; supply both |

For an acquisition interval Δt, white measurement noise has variance `noise_density² / Δt`; random-walk increments have variance `bias_random_walk² × Δt`. A one-sided amplitude density from a datasheet must be divided by √2 for this convention. Correlated bias uses the exact interval decay `exp(-Δt / correlation_time)` and starts in its stationary distribution.

The response filter assumes a constant input over each acquisition interval and initializes from the first input. Calibration, bias and white noise follow the filter; quantization and saturation follow those errors. This is a scalar measurement model, without cross-axis calibration or shared environmental noise.

`nuka.GaussianNoise` specifies a standard deviation per acquisition, independent of Δt. `nuka.PoissonNoise` adds a Poisson count at a fixed rate. The latter is a count perturbation, not signal-dependent photon shot noise for a camera. Supported Poisson rates are finite values from zero to 10⁸.

## Acquisition and replay

`sample_observation` explicitly acquires the current physical field. Its interval must be positive and is measured in seconds; temperature is Celsius. `apply_sensor_noise` acquires using the world's fixed timestep and the configured reference temperature. Reading either observation view does not acquire another sample. Acquisition is manual: these calls do not model a scheduler, latency, jitter or dropped frames.

`observation_stamp(field, env)` returns the acquisition count, accumulated acquisition time and validity. This time is the sum of supplied intervals, not the world's simulation clock. Configuration and reset clear the output and mark it invalid until another acquisition.

Random sequences use the seed, channel, element and that environment's acquisition count. Resetting selected environments restarts only their histories. Checkpoints preserve output, filter state, bias processes and counters. Restoring a checkpoint from before a channel was registered deactivates that channel; existing views retain their allocation but contain invalidated zero data. Subsequent acquisition can reuse it.

Observation views retain their addresses across configuration, acquisition, reset and checkpoint restoration. They are valid while the world owns their storage. All configuration values must be finite; deviations, densities, quantization and time constants must be nonnegative, with a positive correlation time for an enabled correlated bias.

Camera and lidar attachment APIs remain separate. Camera sensor rendering currently omits deformable surfaces that the offline beauty renderer displays.
