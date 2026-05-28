// ---------------------------------------------------------------------------
// nuka::core::diagnostics - invariant trace sinks
// ---------------------------------------------------------------------------

#include "core/diagnostics/trace_sink.hpp"

#include <stdexcept>

namespace nuka::core::diagnostics {

void NullTraceSink::OnSample(const InvariantSample&) {}

void NullTraceSink::Flush() {}

CsvTraceSink::CsvTraceSink(std::string_view path) {
    const std::string owned_path(path);
    file_ = std::fopen(owned_path.c_str(), "w");
    if (file_ == nullptr) {
        throw std::runtime_error("failed to open invariant CSV trace: " + owned_path);
    }
    std::fprintf(file_, "step,env_id,invariant,value,threshold,violation\n");
}

CsvTraceSink::~CsvTraceSink() {
    if (file_ != nullptr) {
        Flush();
        std::fclose(file_);
        file_ = nullptr;
    }
}

void CsvTraceSink::OnSample(const InvariantSample& sample) {
    if (file_ == nullptr) {
        return;
    }
    std::fprintf(file_,
                 "%u,%u,%s,%.9g,%.9g,%s\n",
                 sample.step_index,
                 sample.env_id,
                 InvariantName(sample.which),
                 sample.value,
                 sample.threshold,
                 sample.violation ? "true" : "false");
}

void CsvTraceSink::Flush() {
    if (file_ != nullptr) {
        std::fflush(file_);
    }
}

InMemoryRingTraceSink::InMemoryRingTraceSink(uint32_t capacity)
    : capacity_(capacity) {}

void InMemoryRingTraceSink::OnSample(const InvariantSample& sample) {
    if (capacity_ == 0u) {
        return;
    }
    if (samples_.size() == capacity_) {
        samples_.pop_front();
    }
    samples_.push_back(sample);
}

void InMemoryRingTraceSink::Flush() {}

std::vector<InvariantSample> InMemoryRingTraceSink::Samples() const {
    return {samples_.begin(), samples_.end()};
}

} // namespace nuka::core::diagnostics
