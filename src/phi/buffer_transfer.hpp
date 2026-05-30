#pragma once
// ---------------------------------------------------------------------------
// nuka::phi - host-side typed Buffer <-> std::vector transfer helpers
//
// Consolidates the templated UploadVector / DownloadVector helpers that are
// currently re-declared in the anonymous namespace of every host TU that
// touches GPU buffers:
//   src/runtime/articulation/articulation_state.cpp
//   src/runtime/gpu/device_world.cu
//   src/runtime/gpu/cuda_particle_world.cu
//   src/runtime/gpu/batched_device_world.cu
//   src/sensor/gpu/cuda_sensors.cu
//   src/solver/gpu/row_solver.cu
//   src/collision/gpu/broadphase.cu
//   src/constraint/gpu/contact_generation.cu
//
// This is a HOST-ONLY header (no __device__ code); it is included from both
// .cpp and .cu translation units, so it must compile clean under the strict
// host flags (-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror) and
// under nvcc -std=c++17.
//
// Signatures are reproduced VERBATIM from the existing copies (including the
// `uint32_t count` parameter and the `size()*sizeof(T)` byte math) so the
// Phase-2 migration is a pure delete-local-def + qualified-call substitution
// with no behavioral change. -Wconversion is satisfied because the byte
// arithmetic uses size_t throughout, exactly as the originals do.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

#include "phi/buffer.hpp"

namespace nuka::phi {

// Upload a host vector to a freshly-allocated device Buffer.
// Identical body in all 6 copies (articulation_state.cpp, device_world.cu,
// cuda_particle_world.cu, batched_device_world.cu, cuda_sensors.cu,
// row_solver.cu). Empty vectors allocate a zero-byte Buffer and skip the copy.
template <typename T>
Buffer UploadVector(const std::vector<T>& values) {
    Buffer buffer(values.size() * sizeof(T), MemoryKind::Device);
    if (!values.empty()) {
        buffer.CopyFromHost(values.data(), values.size() * sizeof(T));
    }
    return buffer;
}

// Download `count` elements from a device Buffer, returning a new vector.
// Identical body in broadphase.cu, contact_generation.cu, device_world.cu,
// batched_device_world.cu, cuda_particle_world.cu, cuda_sensors.cu.
// NOTE: this is a DISTINCT overload from the out-parameter form below; both are
// in active use and must coexist.
template <typename T>
std::vector<T> DownloadVector(const Buffer& buffer, std::uint32_t count) {
    std::vector<T> values(count);
    if (!values.empty()) {
        buffer.CopyToHost(values.data(), values.size() * sizeof(T));
    }
    return values;
}

// Download an entire Buffer into a caller-provided vector, sizing the output by
// the Buffer's byte size. This is the form used ONLY by articulation_state.cpp.
// It is a genuinely different overload (out-param, null-safe, size derived from
// Buffer::Size()) -- do not conflate it with the by-count form above.
template <typename T>
void DownloadVector(const Buffer& buffer, std::vector<T>* out) {
    if (out == nullptr) {
        return;
    }
    out->resize(buffer.Size() / sizeof(T));
    if (!out->empty()) {
        buffer.CopyToHost(out->data(), out->size() * sizeof(T));
    }
}

} // namespace nuka::phi
