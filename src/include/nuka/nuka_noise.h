#ifndef NUKA_NUKA_NOISE_H
#define NUKA_NUKA_NOISE_H
// ---------------------------------------------------------------------------
// nuka_noise.h -- C ABI for sim-to-real N1 sensor noise (v0.5 p04 Task 5.4.8)
// ---------------------------------------------------------------------------
//
// Registers per-sensor-field domain-randomization noise and applies it to that
// field's live device buffer. The noise is COUNTER-BASED (Philox4x32-10): a
// sample is a pure function of (seed, element_idx, sequence). This makes it D1
// (DeterminismLevel::Strong) two-run bit-exact AND replay-stable -- the reverse
// pass re-derives the identical noise from the same indices with no RNG state to
// checkpoint (v0.5 exit #6).
//
// DEFAULT IS NONE: a field with no registered noise is byte-unchanged by apply,
// and no existing call site invokes apply -- so V1 oracle scenes stay byte-for-
// byte identical unless noise is explicitly enabled.
//
// Usage:
//   nuka_sensor_noise_desc_t d = { NUKA_NOISE_GAUSSIAN, 0.0f, 0.02f, 1234u };
//   nuka_world_set_sensor_noise(world, NUKA_FIELD_JOINT_VELOCITY, &d);
//   nuka_world_step(world);                 // observation produced
//   nuka_world_apply_sensor_noise(world, NUKA_FIELD_JOINT_VELOCITY);
//   // -> the field's device buffer now carries Gaussian noise; the per-field
//   //    sequence counter advanced (so the NEXT apply is independent noise).
//
// Plain C: no STL, no exceptions cross the boundary.
// ---------------------------------------------------------------------------

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
    float param1;   /* Gaussian: mean   / Poisson: lambda */
    float param2;   /* Gaussian: stddev / (unused for Poisson) */
    uint64_t seed;  /* RNG seed (Philox key) */
} nuka_sensor_noise_desc_t;

// Registers the noise descriptor for `sensor_field`. Pass kind == NUKA_NOISE_NONE
// (or a NULL desc) to clear it. Recording a desc resets that field's sequence
// counter to 0. Returns NUKA_RESULT_INVALID_ARG for an out-of-range field or an
// unknown kind, NUKA_RESULT_NULL_HANDLE for a bad world handle.
nuka_result_t nuka_world_set_sensor_noise(nuka_world_handle world,
                                          nuka_state_field_t sensor_field,
                                          const nuka_sensor_noise_desc_t* desc);

// Applies the registered noise to `sensor_field`'s device buffer ONCE, in place,
// then advances that field's sequence counter (so the next apply is independent
// noise across steps). NONE (or no registered desc) is a byte no-op returning
// NUKA_RESULT_OK with zero writes. Resolves the buffer the SAME way
// nuka_world_get_buffer_view does. Only float-stride fields are supported (the
// noise primitive operates on float32 elements); a non-float-stride field (e.g.
// the 7-float pose / 6-float velocity struct fields) returns
// NUKA_RESULT_NOT_SUPPORTED. Returns NUKA_RESULT_INVALID_ARG for an out-of-range
// field, NUKA_RESULT_NULL_HANDLE for a bad world handle.
nuka_result_t nuka_world_apply_sensor_noise(nuka_world_handle world,
                                            nuka_state_field_t sensor_field);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_NOISE_H */
