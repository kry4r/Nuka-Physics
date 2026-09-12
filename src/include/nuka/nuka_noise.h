#ifndef NUKA_NUKA_NOISE_H
#define NUKA_NUKA_NOISE_H

#include "nuka/nuka.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nuka_noise_kind_t {
    NUKA_NOISE_NONE = 0,
    NUKA_NOISE_GAUSSIAN = 1,
    NUKA_NOISE_POISSON = 2
} nuka_noise_kind_t;

typedef struct nuka_sensor_noise_desc_t {
    nuka_noise_kind_t kind;
    float param1;   /* Gaussian mean or additive Poisson rate, 0 <= rate <= 1e8. */
    float param2;   /* Gaussian standard deviation, nonnegative. */
    uint64_t seed;
} nuka_sensor_noise_desc_t;

typedef struct nuka_sensor_error_desc_t {
    uint32_t struct_size;
    float bias;                      /* Measurement units. */
    float scale_error;               /* Fractional calibration error. */
    float noise_density;             /* Measurement units * sqrt(seconds), two-sided PSD. */
    float initial_bias_stddev;       /* Per-environment and per-element reset bias. */
    float bias_random_walk;          /* Measurement units / sqrt(seconds). */
    float correlated_bias_stddev;    /* Stationary Ornstein-Uhlenbeck standard deviation. */
    float correlation_time;          /* Seconds; positive when correlated bias is enabled. */
    float quantization;              /* Measurement units per least significant bit; 0 disables. */
    float minimum;
    float maximum;
    float response_time;             /* Seconds; 0 disables first-order response lag. */
    float temperature_coefficient;   /* Measurement units / degree Celsius. */
    float reference_temperature;     /* Degrees Celsius. */
    uint32_t saturation_enabled;
    uint64_t seed;
} nuka_sensor_error_desc_t;

typedef struct nuka_observation_stamp_t {
    uint64_t sequence;      /* Acquisitions since reset; zero before the first sample. */
    double elapsed_time;   /* Sum of explicit acquisition intervals, in seconds. */
    uint32_t valid;         /* Reset or configuration invalidates the previous measurement. */
    uint32_t reserved;
} nuka_observation_stamp_t;

// Configure scalar float32 field observations; registration resets their stochastic history.
// NULL clears all errors. Integer and structured fields are rejected, without changing physics.
nuka_result_t nuka_world_set_sensor_noise(nuka_world_handle world,
    nuka_state_field_t sensor_field, const nuka_sensor_noise_desc_t* desc);
nuka_result_t nuka_world_set_sensor_error(nuka_world_handle world,
    nuka_state_field_t sensor_field, const nuka_sensor_error_desc_t* desc);

// Sample the current physical field into independent observation storage, advancing its sequence.
// apply_sensor_noise uses fixed_dt and reference temperature; sample_observation uses explicit units.
nuka_result_t nuka_world_apply_sensor_noise(nuka_world_handle world, nuka_state_field_t sensor_field);
nuka_result_t nuka_world_sample_observation(nuka_world_handle world,
    nuka_state_field_t sensor_field, double sample_interval, float temperature);

// Views retain their address across sampling, configuration, reset and checkpoint restoration.
// They contain the last sampled value; reading a view does not acquire another measurement.
nuka_result_t nuka_world_get_observation_view(nuka_world_handle world,
    nuka_state_field_t sensor_field, nuka_buffer_view_t* out);
nuka_result_t nuka_world_download_observation(nuka_world_handle world,
    nuka_state_field_t sensor_field, void* bytes, size_t nbytes, size_t byte_offset);
nuka_result_t nuka_world_get_observation_stamp(nuka_world_handle world,
    nuka_state_field_t sensor_field, uint32_t env, nuka_observation_stamp_t* out);

typedef struct nuka_domain_randomization_desc_t {
    float mass_mul_lo;       /* mass multiplier range  [lo, hi] (nominal * mult) */
    float mass_mul_hi;
    float friction_mul_lo;   /* friction multiplier range */
    float friction_mul_hi;
    float restitution_off_lo; /* restitution OFFSET range (nominal + offset) */
    float restitution_off_hi;
    float armature_off_lo;   /* joint armature OFFSET range */
    float armature_off_hi;
    float gravity_off_lo;    /* gravity.z OFFSET range (m/s^2) */
    float gravity_off_hi;
    uint64_t seed;           /* Philox key; the recorded deterministic state */
    int enabled;             /* 0 (default) -> apply is a byte no-op */
} nuka_domain_randomization_desc_t;

// Records the domain-randomization descriptor on the world. A NULL desc (or
// enabled == 0) disables DR (apply becomes a byte no-op). Recording a desc does
// NOT sample or apply -- call nuka_world_apply_domain_randomization at episode
// reset to sample + apply. Returns NUKA_RESULT_NULL_HANDLE for a bad world.
nuka_result_t nuka_world_set_domain_randomization(
    nuka_world_handle world, const nuka_domain_randomization_desc_t* desc);

// Samples + applies the stored randomization for ALL envs (call at episode
// reset). For each env: sample (seed, env_idx, param) -> multiplier/offset, then
// apply to the engine buffers per the mapping above. On the FIRST enabled apply
// it snapshots the nominal baseline (per-link mass, gravity.z, per-DOF armature),
// so the apply is idempotent across resets. DR disabled -> byte no-op returning
// NUKA_RESULT_OK with nothing touched (oracle safe). Returns
// NUKA_RESULT_NOT_SUPPORTED for a non-articulated world (no link inertia to
// scale), NUKA_RESULT_NULL_HANDLE for a bad world. No throw across the boundary.
nuka_result_t nuka_world_apply_domain_randomization(nuka_world_handle world);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_NOISE_H */
