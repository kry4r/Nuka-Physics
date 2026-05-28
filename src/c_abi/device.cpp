#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include <cuda_runtime.h>

#include <exception>
#include <memory>

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

        cudaStream_t stream = static_cast<cudaStream_t>(desc->cuda_stream);
        if (stream == nullptr) {
            record->owned_stream = std::make_unique<nuka::phi::OwnedStream>();
            stream = record->owned_stream->Native();
        }
        record->context = nuka::phi::MakeDeviceContext(device_id, stream);
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
