#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "core/version.hpp"

#include <cuda_runtime.h>

#include <exception>
#include <new>
#include <stdexcept>
#include <string>

namespace nuka::c_abi {

HandleTable<nuka_device_t, DeviceRecord>& DeviceTable() {
    static HandleTable<nuka_device_t, DeviceRecord> table;
    return table;
}

HandleTable<nuka_world_t, WorldRecord>& WorldTable() {
    static HandleTable<nuka_world_t, WorldRecord> table;
    return table;
}

HandleTable<nuka_buffer_t, BufferRecord>& BufferTable() {
    static HandleTable<nuka_buffer_t, BufferRecord> table;
    return table;
}

nuka_result_t MapExceptionToResult(const std::exception& error) noexcept {
    const std::string message = error.what();
    if (message.find("cuda") != std::string::npos ||
        message.find("CUDA") != std::string::npos) {
        return NUKA_RESULT_CUDA_ERROR;
    }
    if (message.find("not found") != std::string::npos ||
        message.find("failed to open") != std::string::npos) {
        return NUKA_RESULT_FILE_NOT_FOUND;
    }
    if (message.find("parse") != std::string::npos ||
        message.find("XML") != std::string::npos ||
        message.find("USD") != std::string::npos) {
        return NUKA_RESULT_PARSE_ERROR;
    }
    if (message.find("not supported") != std::string::npos ||
        message.find("requires CUDA") != std::string::npos) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    return NUKA_RESULT_INTERNAL;
}

} // namespace nuka::c_abi

extern "C" {

nuka_version_t nuka_get_version(void) {
    (void)nuka::core::EngineVersion();
    return {0u, 1u, 0u};
}

const char* nuka_result_message(nuka_result_t result) {
    switch (result) {
        case NUKA_RESULT_OK: return "ok";
        case NUKA_RESULT_INVALID_ARG: return "invalid argument";
        case NUKA_RESULT_NULL_HANDLE: return "null or invalid handle";
        case NUKA_RESULT_CUDA_ERROR: return "CUDA error";
        case NUKA_RESULT_FILE_NOT_FOUND: return "file not found";
        case NUKA_RESULT_PARSE_ERROR: return "parse error";
        case NUKA_RESULT_OUT_OF_MEMORY: return "out of memory";
        case NUKA_RESULT_NOT_SUPPORTED: return "not supported";
        case NUKA_RESULT_INTERNAL: return "internal error";
        default: return "unknown result";
    }
}

} // extern "C"
