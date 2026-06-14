#pragma once
// ---------------------------------------------------------------------------
// PHI - scoped CUDA active-device guard
// ---------------------------------------------------------------------------

namespace nuka::phi {

// Sets the active CUDA device on construction and restores the prior device on
// destruction (no-op when the requested device already matches). Engine kernel
// entry points wrap their launches in this so a caller's active device is never
// clobbered.
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
