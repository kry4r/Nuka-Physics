#pragma once
// ---------------------------------------------------------------------------
// PHI – Physics Hardware Interface: Capability query
// ---------------------------------------------------------------------------

namespace nuka::phi {

struct Capabilities {
    int warp_size;
    int max_threads_per_block;
    int max_shared_memory_per_block;
    int multiprocessor_count;
    int compute_capability_major;
    int compute_capability_minor;
    bool supports_async_copy;
    bool supports_graph_launch;
};

/// Query the capabilities of the current CUDA device.
Capabilities QueryCapabilities();

} // namespace nuka::phi
