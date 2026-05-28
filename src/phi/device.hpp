#pragma once
// ---------------------------------------------------------------------------
// PHI – Physics Hardware Interface: Device management
// ---------------------------------------------------------------------------

#include <cstddef>

namespace nuka::phi {

struct DeviceInfo {
    int device_id;
    char name[256];
    size_t total_memory;
};

/// Return the number of CUDA-capable devices.
int GetDeviceCount();

/// Return information about the specified device (default: device 0).
DeviceInfo GetDeviceInfo(int device_id = 0);

/// Set the active CUDA device for the calling thread. Prefer ScopedDeviceGuard
/// from phi/device_context.hpp at engine entry points.
void SetDevice(int device_id);

} // namespace nuka::phi
