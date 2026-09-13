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
| `TOUCH` | Positive contact normal force integrated over a sensing volume, in N |
| `TACTILE` | Taxel force: local tangent x/y and compression-positive normal, in N |

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

MJCF accelerometer/gyro, joint position/velocity, frame pose, velocimeter, force/torque and body-mounted `touch` records create sensors automatically. IMU, joint and force/torque records expose the complete paired outputs shown above. Child bodies without a joint retain a fixed connection and a separate link frame, including tool inertia and mounted sensors. Runtime noise and delivery settings are preserved by checkpoints but are not serialized in NKS. Camera and lidar APIs remain separate. Differentiable Tape creation, stepping and backward replay reject active mounted sensors because Tape does not execute this sampling pipeline.

## Touch volumes and tactile taxels

Touch and tactile sensors read the actual impulses from the common contact solver, including friction and contacts with XPBD and MLS-MPM. Their mounts select an owning rigid body or articulation link. The spatial mask selects contacts on that mount; it does not create contacts or add stiffness to the simulation. A deforming particle surface cannot itself carry these rigidly mounted sensors.

`TOUCH` sums positive normal forces. A contact contributes when the ray from its contact point along the mounting body's outward contact normal intersects the sensing volume. This follows MuJoCo's touch-region convention, including projection of soft contacts into a site. Sphere, box, ellipsoid, capsule and cylinder regions are supported. Box and ellipsoid `size` contains three half-axes; sphere uses `(radius, 0, 0)`; capsule and cylinder use `(radius, half_height, 0)` along local Z.

`TACTILE` represents one rectangular taxel. Its local +Z points out of the sensing surface; it reports local force x/y and the negative local z force, so compression is positive. Only contacts facing that side and lying within the configured Z half-thickness contribute. The X/Y rectangle is half-open, preventing double counting at shared edges. Force divided by `4 * size[0] * size[1]` is the taxel's average traction in Pa.

```python
taxel = world.attach_tactile_sensor(
    size=(0.005, 0.005, 0.003),
    mount=nuka.SensorMount.LINK, mount_index=pad_link,
    local_offset=(0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0),
    spread_fraction=0.3, spread_sigma=0.002,
    hysteresis_strength=0.2, hysteresis_time=0.03,
    sample_rate_hz=100, latency=0.01, seed=42,
)
nuka.MeasurementError(noise_density=0.001, quantization=0.01,
                      minimum=0, maximum=20).configure_sensor(world, taxel, component=2)
world.step_n(10)
forces = world.download_state_sensor(taxel)
```

These parameters are examples, not a calibration of a particular device. Attach adjacent patches with the same sampling settings to form an array; each patch retains its own observation view and noise history. For a scalar volume sensor, pass `kind=nuka.StateSensorKind.TOUCH` and the desired `ContactRegionShape` to the same attachment method. The C API is `nuka_world_attach_tactile_sensor`, followed by the existing state-sensor readout and error APIs.

Mechanical crosstalk uses `spread_fraction` to mix the ideal footprint with a normalized Gaussian of standard deviation `spread_sigma` in meters. Each taxel integrates that distribution over its rectangular area. An infinite planar partition preserves the total force; a finite array loses the part spread beyond its edges. The model uses a spatial transfer function for a sensing layer; it does not simulate elastomer deformation or optical gel markers. Spreading is available for rectangular taxels, not scalar touch volumes.

The optional Maxwell observation branch models viscoelastic relaxation and dynamic hysteresis: `y = x + hysteresis_strength * z`, with `dz/dt = dx/dt - z/hysteresis_time`. Each physical substep treats force as constant, updates the relaxation state exactly and integrates the measured signal over the acquisition interval. Steady input has gain one; loading produces a transient overshoot and unloading can produce negative readings. Configure electronic saturation explicitly when the device clips those readings. This branch affects observations only. Existing response filtering, calibration, noise, temperature drift, quantization and saturation then apply to each output component.

MJCF touch sites preserve their shape, size, default-class inheritance and local orientation, including quaternion, Euler, axis-angle, XY axes, Z axis and `fromto`. Site Euler and axis-angle values follow the compiler's angle settings. NKS serializes authored contact regions and tactile response parameters in the sensor's `tactile` object; `type` is `contact` for scalar touch and `tactile` for taxels. Old `contact` metadata without a sensing region requires a region definition or reimport from MJCF before world creation. Runtime attachments and measurement-error settings remain checkpoint state rather than NKS authoring data.

Acquisition, delayed delivery, dropped packets, selective reset and checkpoint replay use the same state-sensor bank. Checkpoints include pending measurements, integrated force exposure and Maxwell history. Reading a taxel does not sample again or alter the physical state.

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

## Cameras and lidar

Camera and lidar attachment APIs use the same scene geometry and ray traversal. Articulated links, rigid bodies, and fixed-topology particle surfaces are visible: XPBD cloth and tetrahedral soft bodies update from each environment's live particle positions before tracing. A scene containing only one surface or one rigid instance is supported.

Use `nuka.SensorMount.WORLD.value` for a fixed camera or lidar: `local_offset` is its world pose and `mount_index` is ignored. This also works in particle-only scenes without a rigid body or articulation. NKS sensor records use `"mount": "world"`. Mounted state sensors continue to require a body or articulation link.

Particle surfaces retain their authored material and render skin. `skin_smooth_iters` and `skin_smooth_lambda` relax the render vertices before recomputing area-weighted smooth normals; `skin_normal_offset` offsets those vertices in meters along the normals. These settings change the observed surface without changing physical particle positions. Topology and adjacency are uploaded once; deformed vertices and acceleration structures stay on the device. Camera and lidar observations follow selective environment resets and NKS media roundtrips.

Camera depth is distance along the center ray, in meters, rather than optical-axis Z depth. Normal, albedo and primitive ID also use that center ray; color integrates the configured shading samples. Lidar range uses its beam ray, with the configured maximum range for a miss. Equal camera and lidar rays therefore measure the same surface distance. Monte Carlo color samples depend on the global camera index as well as the configured seed; comparing a batched tile with a standalone camera requires matching those sample indices.

Image and range observations are ideal unless their response model is enabled. MLS-MPM density surfaces, individual grains and PBF fluid surfaces are not yet included in camera/lidar rendering.

### Electronic image formation

Configure each camera separately, using its index in the camera tensor:

```python
nuka.CameraResponse(
    exposure_time=0.01, electrons_per_unit_second=1_000_000,
    full_well_electrons=10_000, read_noise_electrons=3,
    row_noise_electrons=1, pixel_gain_stddev=0.01,
    dark_current=25, dark_doubling_temperature=6,
    temperature=35, reference_temperature=25, adc_bits=12, seed=42,
).configure(world, sensor_index=0)
world.render_sensors()
stamp = world.imaging_stamp(nuka.SensorChannel.COLOR.value, sensor_index=0, env=0)
```

Color first accumulates the configured shading samples in linear light. For each channel, expected charge is `exposure_time × (linear_light × electrons_per_unit_second × pixel_gain + dark_current)`. The responsivity parameter calibrates the renderer's linear-light units to electron rate; those lighting units are not an absolute radiometric calibration. Changing exposure changes brightness, shot noise and saturation.

`shot_noise=True` samples a Poisson electron count. Expected counts above 10⁸ use the normal limit. Charge saturates at `full_well_electrons`, then receives independent Gaussian readout noise, a shared row offset and fixed pixel offset. `analog_gain` scales the result and `black_level_electrons` adds an electronic offset. The result is clipped and normalized by full well, then rounded to `2**adc_bits - 1` levels. Zero bits keeps continuous output; supported ADC depths are 1–24 bits. Existing ACES and sRGB settings apply afterward. Disable both with `set_sensor_fidelity` to inspect normalized linear ADC output.

`pixel_gain_stddev` is fractional pixel response nonuniformity; `pixel_offset_stddev_electrons` is fixed electronic offset variation. Their spatial patterns repeat across frames. Read noise varies per channel, pixel and acquisition; row noise is shared across pixels and color channels in the same row. Different environments and cameras have independent patterns. `dead_pixel_probability` removes photoresponse while retaining dark current; `hot_pixel_probability` adds `hot_pixel_current` electrons/s. The two failure classes are disjoint and their probabilities must sum to at most one.

Dark and hot currents use the reference temperature and multiply by `2**((temperature-reference_temperature)/dark_doubling_temperature)`. A zero doubling temperature disables this scaling. Current rates are electrons/s; readout, row, pixel offset, full well and black level are electrons. This model represents three color channels without a Bayer mosaic, demosaicing or charge blooming. Exposure integrates a held scene radiance; motion blur and rolling shutter require temporal scene sampling and are not provided by this response model.

### Depth and lidar returns

```python
response = nuka.RangeResponse(
    bias=0.001, distance_stddev=0.0005, quadratic_stddev=0.0001,
    return_photons=1500, reference_distance=1.0,
    background_photons=10, precision=0.02, minimum_return=1,
    quantization=0.0005, dropout_probability=0.001, seed=42,
)
response.configure(world, nuka.SensorChannel.DEPTH, sensor_index=0)
response.configure(world, nuka.SensorChannel.RANGE, sensor_index=0)
```

Both channels use the same surface response. Expected diffuse return photons are `return_photons × reflectance × abs(normal·ray) × (reference_distance / range)**2`. Reflectance is linear RGB luminance of the textured albedo, multiplied by `1-transmission`. It is a configurable diffuse approximation using visible appearance, not a wavelength-calibrated infrared BRDF; specular returns, multipath and false detections are outside this model.

The return count `N` and ambient count `B` are independent Poisson draws. A return below `minimum_return` is invalid. Distance variance is `distance_stddev**2 + (quadratic_stddev × range**2)**2 + precision**2 × (N+B)/N**2`. The quadratic term can represent disparity-derived depth uncertainty; the photon term represents timing uncertainty limited by return signal and background. Setting `return_photons=0` disables photon-based detection and requires zero background and precision.

Calibration adds `bias + scale_error × range + incidence_bias × (1-abs(normal·ray))`, followed by quantization. An independent dropout probability models missing electronic deliveries. Invalid or out-of-range measurements use the existing channel miss value: positive infinity for camera depth, configured maximum range for lidar. Errors do not change normal, albedo or primitive ID; those remain geometric reference channels. Distances and quantization are meters, `quadratic_stddev` is 1/m, and `precision` is meters × √photon.

### Imaging acquisition and replay

Each `render_sensors()` call explicitly acquires all attached cameras and lidars at the current environment simulation time. Reads do not acquire. Camera color and depth share an acquisition counter; lidar counters are separate. `imaging_stamp` returns `acquisitions`, `sample_time` and `valid`. Noise streams include environment, sensor index, channel, pixel or beam and acquisition count; shading sample seeds remain independent of measurement noise.

Configuring a camera or its depth response restarts that camera's complete acquisition history and clears its views. Configuring a lidar restarts that lidar. Other sensors retain their histories. Selective environment reset clears only selected tiles, stamps and counters. Checkpoints preserve response configuration, appearance settings, image/range tensors and acquisition histories, with stable existing view addresses. Checkpoints require the same attachment topology: adding/replacing sensors or rebuilding camera intrinsics invalidates earlier checkpoints and restoration rejects before changing physics. Reattachment retains existing response settings for retained sensors but starts new acquisition histories.

`enabled=False` restores ideal output for that response; a null C descriptor does the same. Runtime response settings are checkpointed but are not yet serialized in NKS. Image acquisition remains explicit, without automatic frame scheduling, delivery queues, rolling shutter, scanning motion or an image gradient through differentiable Tape.
