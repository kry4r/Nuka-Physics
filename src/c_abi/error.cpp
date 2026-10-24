#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "core/version.hpp"

#include <cuda_runtime.h>

#include <exception>
#include <new>
#include <stdexcept>
#include <string>

namespace nuka::c_abi {
namespace {

// The text of the last exception mapped on this thread, kept until nuka_last_error_message takes it.
thread_local std::string g_last_error;
thread_local std::string g_taken_error;

}  // namespace

HandleTable<nuka_device_t, DeviceRecord>& DeviceTable() {
    static HandleTable<nuka_device_t, DeviceRecord> table;
    return table;
}

HandleTable<nuka_world_t, WorldRecord>& WorldTable() {
    static HandleTable<nuka_world_t, WorldRecord> table;
    return table;
}

nuka_result_t MapStatusToResult(phi::Status status) noexcept {
    switch (status) {
        case phi::Status::Ok: return NUKA_RESULT_OK;
        case phi::Status::Unsupported: return NUKA_RESULT_NOT_SUPPORTED;
        case phi::Status::OutOfMemory: return NUKA_RESULT_OUT_OF_MEMORY;
        case phi::Status::InvalidArgument: return NUKA_RESULT_INVALID_ARG;
        default: return NUKA_RESULT_INTERNAL;
    }
}

void SetErrorMessage(const char* message) noexcept {
    try { g_last_error = message ? message : ""; }
    catch (...) { g_last_error.clear(); }
}

nuka_result_t MapExceptionToResult(const std::exception& error) noexcept {
    const std::string message = error.what();
    g_last_error = message;
    if (dynamic_cast<const std::invalid_argument*>(&error)) return NUKA_RESULT_INVALID_ARG;
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

const char* nuka_last_error_message(void) {
    nuka::c_abi::g_taken_error = std::move(nuka::c_abi::g_last_error);
    nuka::c_abi::g_last_error.clear();
    return nuka::c_abi::g_taken_error.c_str();
}

} // extern "C"
