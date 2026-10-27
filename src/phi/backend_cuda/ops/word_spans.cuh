#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace nuka::phi {

// Word fills and device copies gathered into one kernel; a captured step then carries no copy-engine
// nodes, whose engine switches stall the compute queue.
struct WordSpans {
    static constexpr uint32_t kCapacity = 16u;
    uint32_t* target[kCapacity];
    const uint32_t* source[kCapacity];
    uint32_t value[kCapacity];
    uint32_t words[kCapacity];
    uint32_t count = 0u;
    uint32_t largest = 0u;

    void Fill(void* ptr, size_t bytes, uint32_t word = 0u) { Add(ptr, nullptr, bytes, word); }
    void Copy(void* to, const void* from, size_t bytes) { Add(to, from, bytes, 0u); }
    void Add(void* to, const void* from, size_t bytes, uint32_t word) {
        if (bytes == 0u) return;
        target[count] = static_cast<uint32_t*>(to);
        source[count] = static_cast<const uint32_t*>(from);
        value[count] = word;
        words[count] = static_cast<uint32_t>(bytes / sizeof(uint32_t));
        largest = std::max(largest, words[count]);
        ++count;
    }
};

static __global__ void WordSpansKernel(WordSpans spans) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t k = 0u; k < spans.count; ++k) {
        uint32_t* const target = spans.target[k];
        const uint32_t* const source = spans.source[k];
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < spans.words[k]; i += stride)
            target[i] = source != nullptr ? source[i] : spans.value[k];
    }
}

inline cudaError_t LaunchWordSpans(const WordSpans& spans, cudaStream_t stream) {
    if (spans.count == 0u) return cudaSuccess;
    const uint32_t blocks = std::min((spans.largest + 255u) / 256u, 1024u);
    WordSpansKernel<<<blocks, 256u, 0u, stream>>>(spans);
    return cudaGetLastError();
}

inline cudaError_t FillWords(void* ptr, size_t bytes, uint32_t word, cudaStream_t stream) {
    WordSpans spans;
    spans.Fill(ptr, bytes, word);
    return LaunchWordSpans(spans, stream);
}

inline cudaError_t CopyWords(void* to, const void* from, size_t bytes, cudaStream_t stream) {
    WordSpans spans;
    spans.Copy(to, from, bytes);
    return LaunchWordSpans(spans, stream);
}

}  // namespace nuka::phi
