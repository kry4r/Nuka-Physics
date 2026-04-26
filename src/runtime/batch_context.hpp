#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::BatchContext – batching metadata
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::runtime {

struct BatchContext {
    uint32_t template_count = 0;
    uint32_t instance_count = 0;
};

} // namespace nuka::runtime
