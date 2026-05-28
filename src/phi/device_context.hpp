#pragma once
// ---------------------------------------------------------------------------
// PHI - per-handle CUDA device context
// ---------------------------------------------------------------------------

#include "phi/platform_contract.hpp"
#include "phi/stream_view.hpp"

#include <cuda_runtime.h>

namespace nuka::phi {

struct DeviceContext {
    int device_id = 0;
    StreamView stream;
    PlatformContract contract = {};
};

DeviceContext MakeDeviceContext(int device_id,
                                cudaStream_t external_stream,
                                const PlatformContract& contract = {});

DeviceContext MakeDefaultDeviceContext();

class ScopedDeviceGuard {
public:
    explicit ScopedDeviceGuard(int device_id);
    ~ScopedDeviceGuard();

    ScopedDeviceGuard(const ScopedDeviceGuard&) = delete;
    ScopedDeviceGuard& operator=(const ScopedDeviceGuard&) = delete;

private:
    int prior_device_ = 0;
    bool restore_ = false;
};

} // namespace nuka::phi
