#pragma once
// ---------------------------------------------------------------------------
// nuka::phi — PHI v2 host-side typed Buffer* <-> std::vector transfer helpers.
//
// The ONE host-side transfer header (its legacy RAII predecessor was deleted in the
// M11 BUF sweep). Operates on the opaque v2 `Buffer*` (phi/buffer.hpp).
//
// Allocation is backend-owned via a BufferType*:
//   * DeviceBufferType(device)        -> the device-level DEFAULT type, bound to
//                                        the NULL/default stream (stream 0). Use
//                                        for consumers whose kernels coordinate
//                                        via NULL-stream implicit sync (the
//                                        legacy oracle path) -> byte+timing
//                                        IDENTICAL to the legacy stream-0 memcpy.
//   * BackendDeviceBufferType(backend)-> the backend's MAIN-stream type. Use for
//                                        consumers whose ops/kernels run on
//                                        backend->main (the diffsim op path).
// Stream choice is the D1 anchor: ALWAYS pick the BufferType whose stream matches
// the consumer's kernels (m11-recon.md R1). The caller OWNS the returned Buffer*
// (BufferFree it).
//
// HOST-ONLY header (no __device__ code); compiles under -Wall -Wextra -Wpedantic
// -Wconversion -Wshadow -Werror and under nvcc -std=c++17. Byte arithmetic uses
// size_t throughout, matching the legacy originals so migration is a pure
// type+alloc swap with no behavioral change.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <vector>

#include "phi/backend.hpp"  // BufferType, BackendDeviceBufferType, DeviceBufferType
#include "phi/buffer.hpp"   // Buffer, BufferAlloc / BufferUpload / BufferDownload / BufferFree

namespace nuka::phi {

// Allocate a device Buffer* from `bt` and upload a host vector. An empty vector
// allocates a zero-byte Buffer and skips the copy (mirrors legacy UploadVector).
template <typename T>
Buffer* UploadVectorV2(BufferType* bt, const std::vector<T>& values) {
    Buffer* buffer = BufferAlloc(bt, values.size() * sizeof(T));
    if (buffer != nullptr && !values.empty()) {
        BufferUpload(buffer, values.data(), 0, values.size() * sizeof(T));
    }
    return buffer;
}

// Download `count` elements from a device Buffer* into a new vector (mirrors the
// by-count legacy DownloadVector overload).
template <typename T>
std::vector<T> DownloadVectorV2(Buffer* buffer, std::uint32_t count) {
    std::vector<T> values(count);
    if (buffer != nullptr && !values.empty()) {
        BufferDownload(buffer, values.data(), 0, values.size() * sizeof(T));
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
    if (buffer != nullptr && !out->empty()) {
        BufferDownload(buffer, out->data(), 0, out->size() * sizeof(T));
    }
}

} // namespace nuka::phi
