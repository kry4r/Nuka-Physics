#pragma once
// ---------------------------------------------------------------------------
// PHI - non-owning CUDA stream view
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

namespace nuka::phi {

// Non-owning view over an externally-managed CUDA stream. The caller must keep
// the stream alive for every operation that receives this view.
class StreamView {
public:
    StreamView() = default;
    explicit StreamView(cudaStream_t stream) noexcept : handle_(stream) {}

    cudaStream_t Native() const noexcept { return handle_; }
    bool IsNull() const noexcept { return handle_ == nullptr; }

    void Synchronize() const;

private:
    cudaStream_t handle_ = nullptr;
};

} // namespace nuka::phi
