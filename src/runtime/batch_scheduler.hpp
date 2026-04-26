#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::BatchScheduler -- BatchContext expansion hooks
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::runtime {

struct BatchConfig {
    uint32_t instance_count = 1;
    bool     template_reuse = true;
};

struct BatchResult {
    uint32_t instance_count;
    uint32_t template_count;
};

/// Build a test batch result: if template reuse is on, template_count == 1.
BatchResult BuildTestBatch(uint32_t instance_count);

class BatchScheduler {
public:
    void Configure(const BatchConfig& config);
    BatchResult GetStatus() const;
    uint32_t InstanceCount() const;

private:
    BatchConfig config_;
};

} // namespace nuka::runtime
