// ---------------------------------------------------------------------------
// nuka::diffsim -- CheckpointManager implementation (p02-B/C)
// ---------------------------------------------------------------------------

#include "diffsim/checkpoint.hpp"

#include "phi/backend.hpp"  // DeviceBufferType, InitBestDevice

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::diffsim {

namespace {
void CheckCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("nuka::diffsim::CheckpointManager ") +
                                 what + ": " + cudaGetErrorString(status));
    }
}
}  // namespace

CheckpointManager::CheckpointManager(const phi::DeviceContext& context,
                                     uint32_t max_checkpoints,
                                     uint32_t articulation_count,
                                     uint32_t total_link_count,
                                     uint32_t lambda_width)
    : context_(context),
      max_checkpoints_(max_checkpoints),
      articulation_count_(articulation_count),
      total_link_count_(total_link_count),
      lambda_width_(lambda_width) {
    if (max_checkpoints_ == 0u) {
        throw std::invalid_argument(
            "nuka::diffsim::CheckpointManager: max_checkpoints == 0");
    }
    if (total_link_count_ == 0u || articulation_count_ == 0u) {
        throw std::invalid_argument(
            "nuka::diffsim::CheckpointManager: zero world dimensions");
    }
    const size_t mc = max_checkpoints_;
    // Device-level DEFAULT (stream-0) BufferType -- Capture/Restore D2D copies run
    // on context_.stream (raw cudaMemcpyAsync over the base pointers), so the
    // allocation stream is irrelevant; only the device pointer is consumed.
    phi::BufferType* bt = phi::DeviceBufferType(phi::InitBestDevice());
    base_pose_ = phi::BufferAlloc(
        bt, mc * articulation_count_ * sizeof(nuka::math::Transform));
    link_velocity_ = phi::BufferAlloc(
        bt, mc * total_link_count_ * sizeof(articulation::LinkSpatialVel));
    q_ = phi::BufferAlloc(bt, mc * total_link_count_ * sizeof(float));
    qdot_ = phi::BufferAlloc(bt, mc * total_link_count_ * sizeof(float));
    if (lambda_width_ > 0u) {
        lambda_ = phi::BufferAlloc(bt, mc * lambda_width_ * sizeof(float));
    }
    slot_steps_.reserve(max_checkpoints_);
}

CheckpointManager::~CheckpointManager() {
    if (base_pose_ != nullptr) { phi::BufferFree(base_pose_); }
    if (link_velocity_ != nullptr) { phi::BufferFree(link_velocity_); }
    if (q_ != nullptr) { phi::BufferFree(q_); }
    if (qdot_ != nullptr) { phi::BufferFree(qdot_); }
    if (lambda_ != nullptr) { phi::BufferFree(lambda_); }
}

void CheckpointManager::Reset() {
    count_ = 0u;
    slot_steps_.clear();
}

uint32_t CheckpointManager::Capture(
    const articulation::ArticulationDeviceState& state, uint32_t step_index,
    const float* lambda_device) {
    if (count_ >= max_checkpoints_) {
        throw std::runtime_error(
            "nuka::diffsim::CheckpointManager::Capture: max_checkpoints "
            "exceeded (" +
            std::to_string(max_checkpoints_) + ")");
    }
    const uint32_t slot = count_;
    const cudaStream_t stream = context_.stream.Native();
    phi::ScopedDeviceGuard guard(context_.device_id);

    auto* base_dst = static_cast<nuka::math::Transform*>(phi::BufferBase(base_pose_)) +
                     static_cast<size_t>(slot) * articulation_count_;
    CheckCuda(cudaMemcpyAsync(base_dst, state.base_pose,
                              static_cast<size_t>(articulation_count_) *
                                  sizeof(nuka::math::Transform),
                              cudaMemcpyDeviceToDevice, stream),
              "Capture base_pose");

    auto* vel_dst =
        static_cast<articulation::LinkSpatialVel*>(phi::BufferBase(link_velocity_)) +
        static_cast<size_t>(slot) * total_link_count_;
    CheckCuda(cudaMemcpyAsync(vel_dst, state.link_velocity,
                              static_cast<size_t>(total_link_count_) *
                                  sizeof(articulation::LinkSpatialVel),
                              cudaMemcpyDeviceToDevice, stream),
              "Capture link_velocity");

    auto* q_dst = static_cast<float*>(phi::BufferBase(q_)) +
                  static_cast<size_t>(slot) * total_link_count_;
    CheckCuda(cudaMemcpyAsync(q_dst, state.q,
                              static_cast<size_t>(total_link_count_) *
                                  sizeof(float),
                              cudaMemcpyDeviceToDevice, stream),
              "Capture q");

    auto* qdot_dst = static_cast<float*>(phi::BufferBase(qdot_)) +
                     static_cast<size_t>(slot) * total_link_count_;
    CheckCuda(cudaMemcpyAsync(qdot_dst, state.qdot,
                              static_cast<size_t>(total_link_count_) *
                                  sizeof(float),
                              cudaMemcpyDeviceToDevice, stream),
              "Capture qdot");

    // R2: persist the warm-start lambda if both a reservation and a source
    // exist. Contact-free path: lambda_width_ == 0 OR lambda_device == null ->
    // the slot is present but holds nothing (no divergence, since the contact-
    // free replay never reads it).
    if (lambda_width_ > 0u && lambda_device != nullptr) {
        auto* lam_dst = static_cast<float*>(phi::BufferBase(lambda_)) +
                        static_cast<size_t>(slot) * lambda_width_;
        CheckCuda(cudaMemcpyAsync(lam_dst, lambda_device,
                                  static_cast<size_t>(lambda_width_) *
                                      sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream),
                  "Capture lambda (R2 warm-start)");
    }

    slot_steps_.push_back(step_index);
    count_ += 1u;
    return slot;
}

void CheckpointManager::Restore(
    uint32_t slot, const articulation::ArticulationDeviceState& state,
    float* lambda_device) const {
    if (slot >= count_) {
        throw std::out_of_range(
            "nuka::diffsim::CheckpointManager::Restore: slot out of range");
    }
    const cudaStream_t stream = context_.stream.Native();
    phi::ScopedDeviceGuard guard(context_.device_id);

    const auto* base_src =
        static_cast<const nuka::math::Transform*>(phi::BufferBase(base_pose_)) +
        static_cast<size_t>(slot) * articulation_count_;
    CheckCuda(cudaMemcpyAsync(state.base_pose, base_src,
                              static_cast<size_t>(articulation_count_) *
                                  sizeof(nuka::math::Transform),
                              cudaMemcpyDeviceToDevice, stream),
              "Restore base_pose");

    const auto* vel_src =
        static_cast<const articulation::LinkSpatialVel*>(phi::BufferBase(link_velocity_)) +
        static_cast<size_t>(slot) * total_link_count_;
    CheckCuda(cudaMemcpyAsync(state.link_velocity, vel_src,
                              static_cast<size_t>(total_link_count_) *
                                  sizeof(articulation::LinkSpatialVel),
                              cudaMemcpyDeviceToDevice, stream),
              "Restore link_velocity");

    const auto* q_src = static_cast<const float*>(phi::BufferBase(q_)) +
                        static_cast<size_t>(slot) * total_link_count_;
    CheckCuda(cudaMemcpyAsync(state.q, q_src,
                              static_cast<size_t>(total_link_count_) *
                                  sizeof(float),
                              cudaMemcpyDeviceToDevice, stream),
              "Restore q");

    const auto* qdot_src = static_cast<const float*>(phi::BufferBase(qdot_)) +
                           static_cast<size_t>(slot) * total_link_count_;
    CheckCuda(cudaMemcpyAsync(state.qdot, qdot_src,
                              static_cast<size_t>(total_link_count_) *
                                  sizeof(float),
                              cudaMemcpyDeviceToDevice, stream),
              "Restore qdot");

    if (lambda_width_ > 0u && lambda_device != nullptr) {
        const auto* lam_src = static_cast<const float*>(phi::BufferBase(lambda_)) +
                              static_cast<size_t>(slot) * lambda_width_;
        CheckCuda(cudaMemcpyAsync(lambda_device, lam_src,
                                  static_cast<size_t>(lambda_width_) *
                                      sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream),
                  "Restore lambda (R2 warm-start)");
    }
}

bool CheckpointManager::NearestCheckpointBefore(uint32_t target_step,
                                                uint32_t* out_slot) const {
    // Slots are filled in increasing step order, so scan forward and keep the
    // last slot whose step <= target_step. Fixed-order, host-side (D1).
    bool found = false;
    uint32_t best = 0u;
    for (uint32_t s = 0u; s < count_; ++s) {
        if (slot_steps_[s] <= target_step) {
            best = s;
            found = true;
        } else {
            break;  // monotone: nothing later qualifies
        }
    }
    if (found && out_slot != nullptr) {
        *out_slot = best;
    }
    return found;
}

}  // namespace nuka::diffsim
