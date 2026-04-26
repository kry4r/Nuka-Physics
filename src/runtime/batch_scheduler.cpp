// ---------------------------------------------------------------------------
// nuka::runtime::BatchScheduler -- Implementation
// ---------------------------------------------------------------------------

#include "runtime/batch_scheduler.hpp"

namespace nuka::runtime {

BatchResult BuildTestBatch(uint32_t instance_count) {
    BatchResult result{};
    result.instance_count = instance_count;
    result.template_count = 1; // template reuse: one template shared
    return result;
}

void BatchScheduler::Configure(const BatchConfig& config) {
    config_ = config;
}

BatchResult BatchScheduler::GetStatus() const {
    BatchResult result{};
    result.instance_count = config_.instance_count;
    result.template_count = config_.template_reuse ? 1 : config_.instance_count;
    return result;
}

uint32_t BatchScheduler::InstanceCount() const {
    return config_.instance_count;
}

} // namespace nuka::runtime
