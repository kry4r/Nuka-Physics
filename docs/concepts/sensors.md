# Sensor observations

Physical state and measured observations have separate storage. `world.buffer_view(field)` and `world.download_field(field)` return physical state. Noise acquisition writes only to `world.get_observation_view(field)` and `world.download_observation(field)`.

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
