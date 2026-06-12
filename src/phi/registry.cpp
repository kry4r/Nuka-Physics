// ---------------------------------------------------------------------------
// PHI v2 — global backend registry (ggml_backend_reg registry analogue).
//
// Holds the set of registered RegistryEntry* providers. RegisterBackend appends;
// BackendCount / GetBackend enumerate. InitBestDevice picks the first device of
// the first registered backend (this period: the first CUDA device).
//
// STATIC-LIB SELF-REGISTRATION DE-RISK
// ------------------------------------
// nuka_phi2 is a STATIC library. A pure static-initializer registration TU in a
// static lib is dropped by the linker unless something references it. Rather
// than rely on that (and rather than a +Wl,--whole-archive hack), InitBestDevice
// EXPLICITLY calls RegisterCudaBackendEntry() (defined in cuda_backend.cu),
// guarded by the NUKA_PHI2_WITH_CUDA compile definition set by CMake. This keeps
// the registration deterministic AND structurally dlopen-compatible later (a
// future shared-object backend would register through the same RegisterBackend
// free function from its own init).
// ---------------------------------------------------------------------------

#include "phi/backend.hpp"

#include <vector>

namespace nuka::phi {

#if defined(NUKA_PHI2_WITH_CUDA)
// Defined in phi/backend_cuda/cuda_backend.cu.
void RegisterCudaBackendEntry();
#endif

namespace {

// The process-global registry. A function-local static gives deterministic,
// thread-safe (C++11 magic static) first-use initialization with no static-init
// order issues.
std::vector<RegistryEntry*>& Registry() {
    static std::vector<RegistryEntry*> registry;
    return registry;
}

// Register the built-in backends exactly once (the self-registration de-risk).
void EnsureBuiltinBackends() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
#if defined(NUKA_PHI2_WITH_CUDA)
    RegisterCudaBackendEntry();
#endif
}

} // namespace

void RegisterBackend(RegistryEntry* entry) {
    if (entry == nullptr) {
        return;
    }
    Registry().push_back(entry);
}

size_t BackendCount() {
    EnsureBuiltinBackends();
    return Registry().size();
}

RegistryEntry* GetBackend(size_t i) {
    EnsureBuiltinBackends();
    std::vector<RegistryEntry*>& reg = Registry();
    if (i >= reg.size()) {
        return nullptr;
    }
    return reg[i];
}

Device* InitBestDevice() {
    EnsureBuiltinBackends();
    std::vector<RegistryEntry*>& reg = Registry();
    for (RegistryEntry* entry : reg) {
        if (entry == nullptr) {
            continue;
        }
        if (RegistryEntryDeviceCount(entry) > 0) {
            return RegistryEntryGetDevice(entry, 0);
        }
    }
    return nullptr;
}

} // namespace nuka::phi
