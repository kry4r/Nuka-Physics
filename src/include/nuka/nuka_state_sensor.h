#ifndef NUKA_STATE_SENSOR_H
#define NUKA_STATE_SENSOR_H

#include "nuka/nuka_noise.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nuka_state_sensor_kind_t {
    NUKA_STATE_SENSOR_IMU = 0,
    NUKA_STATE_SENSOR_FRAME_POSE = 1,
    NUKA_STATE_SENSOR_JOINT_STATE = 2,
    NUKA_STATE_SENSOR_LINEAR_VELOCITY = 3,
    NUKA_STATE_SENSOR_CONTACT_WRENCH = 4, /* Contact force xyz (N), torque xyz (N*m) in sensor axes. */
    NUKA_STATE_SENSOR_FORCE_TORQUE = 5,   /* Parent-on-subtree wrench; requires a supported articulation link. */
    NUKA_STATE_SENSOR_TOUCH = 6,          /* Positive normal force integrated over a sensing volume, in N. */
    NUKA_STATE_SENSOR_TACTILE = 7         /* Taxel force: tangent x/y and compression-positive normal, in N. */
} nuka_state_sensor_kind_t;

typedef enum nuka_contact_region_shape_t {
    NUKA_CONTACT_REGION_BOX = 0,
    NUKA_CONTACT_REGION_SPHERE = 1,
    NUKA_CONTACT_REGION_ELLIPSOID = 2,
    NUKA_CONTACT_REGION_CAPSULE = 3,
    NUKA_CONTACT_REGION_CYLINDER = 4
} nuka_contact_region_shape_t;

typedef struct nuka_tactile_desc_t {
    uint32_t struct_size;
    nuka_contact_region_shape_t shape;
    float size[3];              /* Box/ellipsoid half-axes; sphere (r,0,0); local-Z capsule/cylinder (r,h,0). */
    float spread_fraction;     /* TACTILE only: fraction spread by a normalized Gaussian, in [0,1]. */
    float spread_sigma;        /* Gaussian standard deviation in meters. */
    float hysteresis_strength; /* Maxwell observation branch strength; zero disables it. */
    float hysteresis_time;     /* Relaxation time in seconds; positive for an enabled Maxwell branch. */
} nuka_tactile_desc_t;

typedef struct nuka_state_sensor_desc_t {
    uint32_t struct_size;
    nuka_state_sensor_kind_t kind;
    nuka_sensor_mount_t mount;
    uint32_t mount_index;
    float local_offset[7];       /* Position xyz, unit quaternion wxyz. */
    double sample_rate_hz;      /* Zero uses update_period outer steps; positive rates cannot exceed the substep rate. */
    uint32_t update_period;
    double latency;             /* Seconds; delivery occurs on integration boundaries. */
    double latency_jitter;      /* Uniform delivery jitter half-width, in seconds; negative delays clamp to zero. */
    float dropout_probability;
    float temperature;          /* Degrees Celsius. */
    uint64_t seed;
} nuka_state_sensor_desc_t;

typedef struct nuka_state_sensor_stamp_t {
    uint64_t sequence;          /* Most recently delivered acquisition; zero means no delivery. */
    uint64_t acquisitions;      /* Includes dropped and pending acquisitions. */
    uint64_t dropped;
    double sample_time;         /* Environment simulation time at the end of the acquisition interval. */
    double delivery_time;
    uint32_t valid;
    uint32_t reserved;
} nuka_state_sensor_stamp_t;

// Appends an automatically sampled sensor; existing observations keep their device addresses.
// Body mounts use owning body frames; articulation bodies resolve to their link frame.
nuka_result_t nuka_world_attach_state_sensor(nuka_world_handle world,
    const nuka_state_sensor_desc_t* desc, uint32_t* out_sensor);
// TOUCH requires a sensing volume; TACTILE requires a rectangular patch with local +Z pointing outward.
// Taxels compose into arrays; acquisition, error configuration and readout use the state-sensor APIs.
nuka_result_t nuka_world_attach_tactile_sensor(nuka_world_handle world,
    const nuka_state_sensor_desc_t* desc, const nuka_tactile_desc_t* tactile, uint32_t* out_sensor);
// IDs are stable allocation slots, including sensors deactivated by checkpoint restore.
nuka_result_t nuka_world_get_state_sensor_count(nuka_world_handle world, uint32_t* out_count);
nuka_result_t nuka_world_get_state_sensor_active(nuka_world_handle world, uint32_t sensor, uint32_t* out_active);
nuka_result_t nuka_world_get_state_sensor_dims(nuka_world_handle world, uint32_t sensor,
    uint32_t* out_envs, uint32_t* out_values);

// Components: IMU specific force xyz and angular velocity xyz; joint position and velocity.
// Pose errors use world position xyz and local rotation-vector xyz, never quaternion components.
nuka_result_t nuka_world_set_state_sensor_error(nuka_world_handle world, uint32_t sensor,
    uint32_t component, const nuka_sensor_error_desc_t* desc);

// Values are float32, environment-major; pose output uses position xyz then quaternion wxyz.
// Configuration and environment reset invalidate output and restart its noise and delivery history.
nuka_result_t nuka_world_get_state_sensor_view(nuka_world_handle world, uint32_t sensor, nuka_buffer_view_t* out);
nuka_result_t nuka_world_download_state_sensor(nuka_world_handle world, uint32_t sensor,
    void* bytes, size_t nbytes, size_t byte_offset);
nuka_result_t nuka_world_get_state_sensor_stamp(nuka_world_handle world, uint32_t sensor,
    uint32_t env, nuka_state_sensor_stamp_t* out);

#ifdef __cplusplus
}
#endif
#endif
