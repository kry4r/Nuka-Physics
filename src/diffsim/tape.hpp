#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- multi-step differentiable rollout TAPE (p02-B/C)
// ---------------------------------------------------------------------------
//
// The tape records, per forward step, exactly what the reverse pass needs to
// (a) replay that step's forward deterministically and (b) reconstruct the
// per-step StepBackwardInputs the p02-A single-step adjoint consumes. On the
// CONTACT-FREE PD path (drive -> ABA -> integrate) the only per-step INPUT the
// forward depends on (beyond the carried authoritative state) is the per-link
// ACTION (drive_targets) the policy applied that step. The drive gains
// (stiffness / damping / force_limits) and the dt / gravity_z are rollout-
// constant, so they live once on the TapeDesc, not per step.
//
// Forward-compat for contacts (p02-D): each TapeEntry already carries
// present-but-UNUSED slots for the per-row contact impulses (lambda) and the
// contact event flags. The contact-free path never writes them; when p02-D
// lands the solve, replay-from-a-checkpoint is only bit-exact if the persistent
// warm-start lambda is restored, and the adjoint of the solve needs the
// converged lambda -- both ride here. Reserving the layout now keeps the tape
// ABI stable across the p02-D landing.
//
// Storage model (gradient checkpointing): the tape stores, per step, the CHEAP
// per-env action slice (total_link_count floats) -- NOT the expensive ABA
// intermediates. The authoritative state is snapshotted only at checkpoints
// (see checkpoint.hpp); between checkpoints the backward re-rolls the forward
// to regenerate the intermediates on demand. In the full-tape debug mode
// (recompute_on_backward == 0) every step additionally gets a checkpoint, so
// the reverse is a single pass with no recompute -- the CI oracle the
// checkpoint path is validated against.
//
// All device storage is a phi v2 Buffer* (NO DeviceVector). The action buffer is
// one contiguous device array of [max_tape_entries * total_link_count] floats,
// env-major within each step (the SAME per-link env-major layout the engine's
// drive_targets use), so step k's action slice is a plain pointer offset --
// zero-copy into StepBackwardInputs::drive_targets and grad_actions_out. The
// Buffer* is allocated from the device-level DEFAULT (stream-0) BufferType; the
// per-step action D2D copy runs on context.stream (raw cudaMemcpyAsync over the
// Buffer's base pointer -- unchanged), the kernels read it on the same stream.
// ---------------------------------------------------------------------------

#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace nuka::diffsim {

// Rollout-constant configuration for a tape (mirrors nuka_tape_desc_t).
struct TapeDesc {
    // Gradient-checkpoint interval K: a checkpoint is captured every K steps
    // (step 0 always gets one). Smaller K -> less recompute, more memory.
    uint32_t checkpoint_interval = 16u;
    // Hard caps (the device storage is sized once at construction).
    uint32_t max_tape_entries = 1024u;
    uint32_t max_checkpoints = 128u;
    // Debug mode: when 0, EVERY step gets a checkpoint (full-tape; the reverse
    // is a single pass, no recompute -- the CI oracle). When 1 (default), only
    // every-K-th step gets a checkpoint and the reverse recomputes the window.
    uint32_t recompute_on_backward = 1u;
};

// Per-step record. Scalars only -- the bulk per-link action data lives in the
// Tape's contiguous action buffer at this step's slice (see Tape::ActionSlice).
struct TapeEntry {
    uint32_t step_index = 0u;       // 0-based forward step ordinal
    uint32_t has_checkpoint = 0u;   // 1 if a checkpoint was captured AT this step
    uint32_t checkpoint_slot = 0u;  // index into CheckpointManager (valid iff above)
    // --- contact forward-compat (present-but-unused on the contact-free path) --
    // p02-D: per-step count of active contact rows + the device-buffer offset
    // into the tape's lambda/event reservation. Zeroed on the contact-free path.
    uint32_t contact_row_count = 0u;
    uint32_t lambda_offset = 0u;    // offset into the (reserved) lambda buffer
    uint32_t event_flags = 0u;      // reserved contact event bitmask
};

// Owns the per-step action storage + per-step records for one rollout. Sized
// once from a TapeDesc + the world dimensions; reused across rollouts via
// Reset(). The Tape itself is mode-agnostic: it stores actions + records; the
// CheckpointManager (separate) owns the authoritative-state snapshots.
class Tape {
public:
    // total_link_count: per-step action width (env-major per-link, the engine's
    // drive layout). The action buffer is [max_tape_entries * total_link_count].
    Tape(cudaStream_t stream, int device_id, const TapeDesc& desc,
         uint32_t total_link_count);

    const TapeDesc& Desc() const { return desc_; }
    uint32_t TotalLinkCount() const { return total_link_count_; }
    uint32_t StepCount() const { return step_count_; }
    uint32_t Capacity() const { return desc_.max_tape_entries; }

    // Clears all recorded steps (keeps the device storage). The next RecordStep
    // starts at step 0 again.
    void Reset();

    // Appends one step's action slice (device pointer, length total_link_count)
    // by D2D-copying it into this step's slot, and records the TapeEntry. Returns
    // the new step's index. Throws if capacity is exceeded. `has_checkpoint` /
    // `checkpoint_slot` are filled by the orchestrator (it decides the K-cadence).
    uint32_t RecordStep(const float* device_actions, uint32_t has_checkpoint,
                        uint32_t checkpoint_slot);

    // Mutable / const device pointer to step k's action slice (length
    // total_link_count). Used as StepBackwardInputs::drive_targets (const) at
    // replay and as the grad_actions_out write target offset at backward.
    float* ActionSlice(uint32_t step);
    const float* ActionSlice(uint32_t step) const;

    // The per-step record (host-resident; cheap scalars).
    const TapeEntry& Entry(uint32_t step) const { return entries_[step]; }
    TapeEntry& Entry(uint32_t step) { return entries_[step]; }

    // Mutates an entry's checkpoint bookkeeping after the fact (the orchestrator
    // captures the checkpoint, then back-annotates the slot here).
    void SetCheckpoint(uint32_t step, uint32_t slot);

    // Device base pointer of the contiguous action buffer (all steps).
    const float* ActionBuffer() const {
        return static_cast<const float*>(phi::BufferBase(actions_));
    }

    ~Tape();
    Tape(const Tape&) = delete;
    Tape& operator=(const Tape&) = delete;

private:
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    TapeDesc desc_;
    uint32_t total_link_count_ = 0u;
    uint32_t step_count_ = 0u;

    phi::Buffer* actions_ = nullptr;  // float[max_tape_entries * total_link_count]
    std::vector<TapeEntry> entries_;  // host records, one per step
};

}  // namespace nuka::diffsim
