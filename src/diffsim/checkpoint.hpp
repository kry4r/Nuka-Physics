#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- gradient-checkpoint snapshot/restore (p02-B/C)
// ---------------------------------------------------------------------------
//
// A Checkpoint captures the AUTHORITATIVE per-env state at one forward step --
// exactly the buffers that diverge across a Step and are NOT recomputed from
// scratch before they are read, i.e. the SAME set the engine's Reset snapshot
// owns (the ResetEnvs port in phi/backend_cuda/ops/readout.cu):
//
//   base_pose[articulation_count]    -- floating-base world pose
//   link_velocity[total_link_count]  -- base + per-link spatial velocities
//   q[total_link_count]              -- joint positions
//   qdot[total_link_count]           -- joint velocities
//
// Restoring this set + re-running the forward kernels regenerates EVERY derived
// buffer (link_pose / link_acceleration / the ABA scratch / qddot / tau) bit-
// exactly, so a replay from a checkpoint to a later step reproduces the original
// forward to the bit. That is the property the whole recompute path rides on.
//
// R2 -- LAMBDA WARM-START SLOT: the Checkpoint struct ALSO carries a slot for
// the persistent contact warm-start lambda (the ONLY genuine cross-step
// accumulator in the engine's contact solve, carried for warm-start). The
// contact-free p02-B/C path never writes or reads it, but it is captured /
// restored here NOW so that when p02-D lands the solve, replay-from-a-checkpoint
// stays bit-exact (the solve's first iteration seeds from the warm-start lambda;
// if lambda is not restored, the regenerated forward diverges). Reserving +
// wiring the slot now keeps the Checkpoint layout stable across the p02-D land.
//
// All storage is a phi v2 Buffer* (NO DeviceVector). Capacity (max_checkpoints)
// is sized once from the device-level DEFAULT (stream-0) BufferType; Capture/
// Restore are D2D copies on the context stream (raw cudaMemcpyAsync over the
// base pointers -- D1: a flat contiguous copy has no float arithmetic and a
// fixed byte order).
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>
#include <vector>

namespace nuka::diffsim {

namespace articulation = nuka::runtime::articulation;

struct ContactWarmStartDeviceState {
    uint32_t point_count = 0u;
    const uint64_t* pair = nullptr;
    const uint64_t* feature = nullptr;
    const float* lambda = nullptr;       // 3 floats per point
    const nuka::math::Vec3* normal = nullptr;
    const nuka::math::Vec3* tangent1 = nullptr;
    const nuka::math::Vec3* tangent2 = nullptr;
    const uint64_t* material = nullptr;
    const uint32_t* age = nullptr;
};

// The per-checkpoint device storage (one contiguous block per buffer, sliced by
// slot). Plus the host-side record of which forward step each slot snapshots.
class CheckpointManager {
public:
    // Sizes the snapshot storage for `max_checkpoints` slots of a world with
    // `articulation_count` floating-base poses, `total_link_count` link
    // velocities / q / qdot, and a per-env lambda warm-start of `lambda_width`
    // floats per env (R2 reserve; pass 0 on the contact-free path -- the slot is
    // still present, just zero-width). lambda_width is the engine's
    // slot_count*3 / env (== kMaxFootContactsPerEnv*3) when contacts land.
    CheckpointManager(cudaStream_t stream, int device_id,
                      uint32_t max_checkpoints, uint32_t articulation_count,
                      uint32_t total_link_count, uint32_t lambda_width = 0u,
                      uint32_t contact_point_count = 0u);

    uint32_t Capacity() const { return max_checkpoints_; }
    uint32_t Count() const { return count_; }

    // Forgets all captured checkpoints (keeps the device storage).
    void Reset();

    // Captures the authoritative state of `state` at forward step `step_index`
    // into a fresh slot. `lambda_device` (length lambda_width, R2 warm-start) may
    // be null on the contact-free path. Returns the new slot index. Throws if
    // capacity is exceeded. Slots are filled in increasing step order, so the
    // step->slot map is monotone (NearestCheckpointBefore exploits that).
    uint32_t Capture(const articulation::ArticulationDeviceState& state,
                     uint32_t step_index, const float* lambda_device = nullptr,
                     const ContactWarmStartDeviceState* contact_cache = nullptr);

    // Restores slot `slot`'s snapshot back into `state` (D2D copy of base_pose /
    // link_velocity / q / qdot) and, if `lambda_device` is non-null and the slot
    // has a lambda reservation, into the warm-start lambda. Every other buffer in
    // `state` is left untouched (the caller re-runs the forward to regenerate the
    // derived buffers). D1: flat contiguous D2D copies.
    void Restore(uint32_t slot,
                 const articulation::ArticulationDeviceState& state,
                 float* lambda_device = nullptr,
                 const ContactWarmStartDeviceState* contact_cache = nullptr) const;

    // The forward step a slot snapshots.
    uint32_t StepOf(uint32_t slot) const { return slot_steps_[slot]; }

    // The slot whose step is the LARGEST step <= `target_step`, or returns false
    // via the out-param if no checkpoint at or before target_step exists. Slots
    // are monotone in step, so this is a fixed-order scan (D1, host-side).
    bool NearestCheckpointBefore(uint32_t target_step, uint32_t* out_slot) const;

    ~CheckpointManager();
    CheckpointManager(const CheckpointManager&) = delete;
    CheckpointManager& operator=(const CheckpointManager&) = delete;

private:
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    uint32_t max_checkpoints_ = 0u;
    uint32_t articulation_count_ = 0u;
    uint32_t total_link_count_ = 0u;
    uint32_t lambda_width_ = 0u;
    uint32_t contact_point_count_ = 0u;
    uint32_t count_ = 0u;

    // Contiguous per-buffer storage; slot s lives at offset s*width.
    phi::Buffer* base_pose_ = nullptr;     // Transform[max_checkpoints * articulation_count]
    phi::Buffer* link_velocity_ = nullptr; // LinkSpatialVel[max_checkpoints * total_link_count]
    phi::Buffer* q_ = nullptr;             // float[max_checkpoints * total_link_count]
    phi::Buffer* qdot_ = nullptr;          // float[max_checkpoints * total_link_count]
    phi::Buffer* lambda_ = nullptr;        // float[max_checkpoints * lambda_width] (R2)
    phi::Buffer* contact_pair_ = nullptr;
    phi::Buffer* contact_feature_ = nullptr;
    phi::Buffer* contact_lambda_ = nullptr;
    phi::Buffer* contact_normal_ = nullptr;
    phi::Buffer* contact_tangent1_ = nullptr;
    phi::Buffer* contact_tangent2_ = nullptr;
    phi::Buffer* contact_material_ = nullptr;
    phi::Buffer* contact_age_ = nullptr;

    std::vector<uint32_t> slot_steps_;  // host: slot -> forward step index
};

}  // namespace nuka::diffsim
