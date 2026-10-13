#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include "core/checked_size.hpp"

#include "phi/backend.hpp"  // BufferType, BackendDeviceBufferType, DeviceBufferType
#include "phi/buffer.hpp"   // Buffer, BufferAlloc / BufferUpload / BufferDownload / BufferFree

namespace nuka::phi {

// Allocate a device Buffer* from `bt` and upload a host vector. An empty vector
// allocates a zero-byte Buffer and skips the copy.
template <typename T>
Buffer* UploadVectorV2(BufferType* bt, const std::vector<T>& values) {
    Buffer* buffer = BufferAlloc(bt, CheckedProduct({values.size(), sizeof(T)}));
    if (buffer == nullptr) throw std::bad_alloc();
    if (buffer != nullptr && !values.empty()) {
        if (BufferUpload(buffer, values.data(), 0, values.size() * sizeof(T)) != Status::Ok) {
            BufferFree(buffer);
            throw std::runtime_error("vector buffer upload failed");
        }
    }
    return buffer;
}

// Download `count` elements from a device Buffer* into a new vector.
template <typename T>
std::vector<T> DownloadVectorV2(Buffer* buffer, std::uint32_t count) {
    if (!buffer && count != 0u) throw std::invalid_argument("null download buffer");
    std::vector<T> values(count);
    if (buffer != nullptr && !values.empty()) {
        if (BufferDownload(buffer, values.data(), 0, values.size() * sizeof(T)) != Status::Ok)
            throw std::runtime_error("vector buffer download failed");
    }
    return values;
}

// Download `count` elements into a caller-provided vector (out-param form; the
// v2 replacement for the Size()-derived legacy overload used by
// articulation_state.cpp — `count` is now explicit because the v2 Buffer is
// opaque, and the caller always knows the element count from its host state).
// Null-safe on `out`.
template <typename T>
void DownloadVectorV2(Buffer* buffer, std::vector<T>* out, std::size_t count) {
    if (out == nullptr) {
        return;
    }
    out->resize(count);
    if (!buffer && count != 0u) throw std::invalid_argument("null download buffer");
    if (buffer != nullptr && !out->empty()) {
        if (BufferDownload(buffer, out->data(), 0, out->size() * sizeof(T)) != Status::Ok)
            throw std::runtime_error("vector buffer download failed");
    }
}

} // namespace nuka::phi
