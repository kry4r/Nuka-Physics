#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "phi/backend.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <exception>
#include <memory>

namespace {

namespace nphi = nuka::phi;

// Select the phi v2 Device for the requested CUDA ordinal. Walks the registry's
// devices counting across backends until `ordinal` is reached; falls back to
// InitBestDevice() (the first available device) when the ordinal is out of range
// or negative. Mirrors recorder.cpp's SelectDeviceByOrdinal (commit fcae2a6) so
// the validated DeviceRecord ordinal is honored instead of unconditionally
// routing every device to device 0. NOTE: BackendCount()/GetBackend()/
// InitBestDevice() each call EnsureBuiltinBackends() internally, which runs
// RegisterCudaBackendEntry() exactly once (phi/registry.cpp) -- this is what
// closes the static-lib backend-drop trap, so no explicit registration call is
// needed here.
nphi::Device* SelectDeviceByOrdinal(int ordinal) {
    if (ordinal >= 0) {
        int seen = 0;
        const size_t backend_count = nphi::BackendCount();
        for (size_t b = 0u; b < backend_count; ++b) {
            nphi::RegistryEntry* entry = nphi::GetBackend(b);
            if (entry == nullptr) continue;
            const size_t dev_count = nphi::RegistryEntryDeviceCount(entry);
            for (size_t d = 0u; d < dev_count; ++d) {
                if (seen == ordinal) {
                    return nphi::RegistryEntryGetDevice(entry, d);
                }
                ++seen;
            }
        }
    }
    return nphi::InitBestDevice();  // fallback: first available device.
}

}  // namespace

extern "C" {

nuka_result_t nuka_device_create(const nuka_device_desc_t* desc,
                                 nuka_device_handle* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    *out = nullptr;
    if (desc == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }

    try {
        auto record = std::make_unique<nuka::c_abi::DeviceRecord>();
        const int device_id = static_cast<int>(desc->gpu_index);
        cudaError_t cuda_result = cudaSetDevice(device_id);
        if (cuda_result != cudaSuccess) {
            return NUKA_RESULT_CUDA_ERROR;
        }
        record->device_id = device_id;

        // BUF-14: the legacy orchestration that used to run on a per-handle
        // OWNED stream now runs on STREAM 0 (the NULL/default stream). desc->
        // cuda_stream is accepted by the public ABI but the legacy paths only
        // ever ran their kernels + explicit syncs on the handle's stream, which
        // is now stream 0; the NULL stream serializes with the v2 backend's
        // blocking `main` stream exactly as the owned stream did before.

        // --- phi v2 device/backend acquisition (M9 T2) ---------------------
        // Acquire the OWNED phi v2 Backend on the requested CUDA ordinal so the
        // C-ABI can later build an nk::World + call cook::Settle (T4/T5). Mirrors
        // recorder.cpp (commit fcae2a6): honor the validated ordinal via
        // SelectDeviceByOrdinal, then take ownership IMMEDIATELY so any failure
        // path frees the backend through the DeviceRecord destructor. The Device
        // is registry-owned (not freed); the Backend is freed in ~DeviceRecord.
        // TODO(M9-T5): feed record->phi_device + record->backend into the
        //              nk::World ctor in world.cpp create.
        // TODO(M9-T4): use them for cook::Settle in nuka_scene.cpp.
        nphi::Device* dev = SelectDeviceByOrdinal(device_id);
        if (dev == nullptr) {
            return NUKA_RESULT_NOT_SUPPORTED;  // no phi v2 backend/device.
        }
        nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
        if (backend == nullptr) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        record->phi_device = dev;     // registry-owned; not freed.
        record->backend = backend;    // OWNED; freed by ~DeviceRecord.
        nphi::SetActiveBackend(backend);  // PHI + RHI share this architecture.

        *out = nuka::c_abi::DeviceTable().Insert(std::move(record));
        return *out == nullptr ? NUKA_RESULT_INTERNAL : NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

void nuka_device_destroy(nuka_device_handle device) {
    (void)nuka::c_abi::DeviceTable().Remove(device);
}

} // extern "C"
