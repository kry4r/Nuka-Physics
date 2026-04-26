#pragma once
// ---------------------------------------------------------------------------
// PHI – Physics Hardware Interface: Async stream
// ---------------------------------------------------------------------------

namespace nuka::phi {

/// RAII wrapper around a CUDA stream.
class Stream {
public:
    Stream();
    ~Stream();

    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    /// Block the host until all previously enqueued work on this stream completes.
    void Synchronize();

    /// Return the underlying cudaStream_t (as void*).
    void* NativeHandle();

private:
    void* handle_ = nullptr;
};

} // namespace nuka::phi
