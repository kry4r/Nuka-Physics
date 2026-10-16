#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "nk/pipeline/world.hpp"
#include "phi/backend_cuda/cuda_internal.cuh"

#include <cuda_runtime.h>

namespace {

cudaStream_t WorldStream(const nuka::c_abi::WorldRecord& record) {
    return nuka::phi::CudaBackendMainStream(
        reinterpret_cast<nuka::phi::CudaBackend*>(record.world->Backend()));
}

}  // namespace

extern "C" {

nuka_result_t nuka_world_get_stream(nuka_world_handle world, void** out_stream) {
    if (out_stream == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out_stream = nullptr;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    *out_stream = WorldStream(*record);
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_record_event(nuka_world_handle world, void* cuda_event) {
    if (cuda_event == nullptr) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    return cudaEventRecord(static_cast<cudaEvent_t>(cuda_event), WorldStream(*record)) == cudaSuccess
               ? NUKA_RESULT_OK
               : NUKA_RESULT_CUDA_ERROR;
}

nuka_result_t nuka_get_runtime_info(nuka_runtime_info_t* out) {
    if (!out || out->struct_size != sizeof(*out) || out->schema_version != NUKA_RUNTIME_INFO_VERSION)
        return NUKA_RESULT_INVALID_ARG;
    *out = {};
    out->struct_size = sizeof(*out);
    out->schema_version = NUKA_RUNTIME_INFO_VERSION;
    int runtime_version = 0;
    int driver_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
        cudaDriverGetVersion(&driver_version) != cudaSuccess)
        return NUKA_RESULT_CUDA_ERROR;
    out->cuda_runtime_version = runtime_version;
    out->cuda_driver_version = driver_version;
    return NUKA_RESULT_OK;
}

}  // extern "C"
