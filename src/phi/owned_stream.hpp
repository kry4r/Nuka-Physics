#pragma once
// ---------------------------------------------------------------------------
// PHI - owning CUDA stream wrapper
// ---------------------------------------------------------------------------

#include "phi/stream_view.hpp"

#include <cuda_runtime.h>

namespace nuka::phi {

// RAII stream owner for engine-created streams. New public integration points
// should accept StreamView so callers can provide their own cudaStream_t.
class OwnedStream {
public:
    OwnedStream();
    ~OwnedStream();

    OwnedStream(OwnedStream&& other) noexcept;
    OwnedStream& operator=(OwnedStream&& other) noexcept;

    OwnedStream(const OwnedStream&) = delete;
    OwnedStream& operator=(const OwnedStream&) = delete;

    StreamView View() const noexcept { return StreamView{handle_}; }
    cudaStream_t Native() const noexcept { return handle_; }
    void Synchronize() const;

private:
    cudaStream_t handle_ = nullptr;
};

} // namespace nuka::phi
