// ---------------------------------------------------------------------------
// nuka::diffsim -- Tape implementation (p02-B/C)
// ---------------------------------------------------------------------------

#include "diffsim/tape.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::diffsim {

namespace {
void CheckCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("nuka::diffsim::Tape ") + what +
                                 ": " + cudaGetErrorString(status));
    }
}
}  // namespace

Tape::Tape(const phi::DeviceContext& context, const TapeDesc& desc,
           uint32_t total_link_count)
    : context_(context), desc_(desc), total_link_count_(total_link_count) {
    if (total_link_count_ == 0u) {
        throw std::invalid_argument("nuka::diffsim::Tape: total_link_count == 0");
    }
    if (desc_.max_tape_entries == 0u) {
        throw std::invalid_argument("nuka::diffsim::Tape: max_tape_entries == 0");
    }
    if (desc_.checkpoint_interval == 0u) {
        // K == 0 is meaningless (would never re-cadence); treat as "every step".
        desc_.checkpoint_interval = 1u;
    }
    const size_t bytes = static_cast<size_t>(desc_.max_tape_entries) *
                         static_cast<size_t>(total_link_count_) * sizeof(float);
    actions_ = phi::Buffer(bytes, phi::MemoryKind::Device);
    entries_.reserve(desc_.max_tape_entries);
}

void Tape::Reset() {
    step_count_ = 0u;
    entries_.clear();
}

uint32_t Tape::RecordStep(const float* device_actions, uint32_t has_checkpoint,
                          uint32_t checkpoint_slot) {
    if (step_count_ >= desc_.max_tape_entries) {
        throw std::runtime_error(
            "nuka::diffsim::Tape::RecordStep: max_tape_entries exceeded (" +
            std::to_string(desc_.max_tape_entries) + ")");
    }
    const uint32_t step = step_count_;
    float* dst = ActionSlice(step);
    if (device_actions != nullptr) {
        phi::ScopedDeviceGuard guard(context_.device_id);
        CheckCuda(cudaMemcpyAsync(dst, device_actions,
                                  static_cast<size_t>(total_link_count_) *
                                      sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  context_.stream.Native()),
                  "RecordStep action D2D");
    }
    TapeEntry entry;
    entry.step_index = step;
    entry.has_checkpoint = has_checkpoint;
    entry.checkpoint_slot = checkpoint_slot;
    entries_.push_back(entry);
    step_count_ += 1u;
    return step;
}

float* Tape::ActionSlice(uint32_t step) {
    float* base = static_cast<float*>(actions_.Data());
    return base + static_cast<size_t>(step) * total_link_count_;
}

const float* Tape::ActionSlice(uint32_t step) const {
    const float* base = static_cast<const float*>(actions_.Data());
    return base + static_cast<size_t>(step) * total_link_count_;
}

void Tape::SetCheckpoint(uint32_t step, uint32_t slot) {
    entries_[step].has_checkpoint = 1u;
    entries_[step].checkpoint_slot = slot;
}

}  // namespace nuka::diffsim
