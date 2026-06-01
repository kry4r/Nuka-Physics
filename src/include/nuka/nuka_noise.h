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

// ---------------------------------------------------------------------------
// N2 per-episode domain randomization (v0.5 p04 Task 5.4.7).
//
// Mirrors the C++ DomainRandomizationConfig. Each [lo, hi] range is sampled ONCE
// per env per episode-reset to a per-env multiplier / offset that is a PURE
// FUNCTION of (seed, env_idx, param_idx) via the SAME counter-based Philox the
// sensor noise uses. There is no mutable sampled state -- the deterministic seed
// IS the recorded state, so the diff-sim reverse/replay pass re-derives identical
// values and the backward stays D1 byte-exact two-run.
//
// mass / friction are MULTIPLIERS; restitution / armature / gravity are OFFSETS.
// Apply snapshots a NOMINAL baseline ONCE (first enabled apply) and thereafter
// computes value = nominal*mult / nominal+offset, so repeated resets re-randomize
// AROUND nominal (idempotent) rather than compounding a random walk.
//
// DEFAULT IS DISABLED (enabled == 0): a world with DR off is byte-unchanged by
// apply, so V1 oracle scenes stay byte-identical unless DR is explicitly enabled.
//
// ENGINE-BUFFER MAPPING (single-env contact-free diff-sim tape):
//   mass        -> per-link spatial inertia (link_inertia, rebuilt via the SAME
//                  MakeSpatialInertia path nuka_world_set_link_mass uses). TAPE-
//                  VISIBLE: changes the contact-free ABA forward + its gradient.
//   gravity_z   -> step_options.gravity.z (read by nuka_tape_create into
//                  RolloutParams.gravity_z). TAPE-VISIBLE. NOTE: one world scalar
//                  -> the same offset applies to all envs (gravity is global).
//   armature    -> per-DOF joint_armature buffer (host mirror + device). Present
//                  on the articulation state but INERT in the contact-free tape
//                  forward (does not enter the tape's ABA + integrate).
//   friction    -> the batched contact path's scalar friction_coefficient; the
//                  single-env contact-free tape has no contact solve -> INERT.
//   restitution -> no engine buffer in the contact-free path -> sampled for RL
//                  completeness but INERT (applied where a contact buffer exists).
// ---------------------------------------------------------------------------

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
