#pragma once
// ---------------------------------------------------------------------------
// PHI – Physics Hardware Interface: GPU buffer primitives
// ---------------------------------------------------------------------------

#include <cstddef>

namespace nuka::phi {

enum class MemoryKind { Device, Host, Managed };

/// Type-erased GPU buffer with RAII ownership semantics.
class Buffer {
public:
    Buffer() = default;
    Buffer(size_t size_bytes, MemoryKind kind = MemoryKind::Device);
    ~Buffer();

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* Data();
    const void* Data() const;
    size_t Size() const;
    MemoryKind Kind() const;

    void CopyFromHost(const void* src, size_t bytes);
    void CopyToHost(void* dst, size_t bytes) const;

private:
    void* data_ = nullptr;
    size_t size_ = 0;
    MemoryKind kind_ = MemoryKind::Device;
};

/// Typed view over a Buffer – does not own the underlying memory.
template <typename T>
class BufferView {
public:
    explicit BufferView(Buffer& buffer)
        : data_(static_cast<T*>(buffer.Data()))
        , count_(buffer.Size() / sizeof(T)) {}

    T* Data() { return data_; }
    const T* Data() const { return data_; }
    size_t Count() const { return count_; }

private:
    T* data_;
    size_t count_;
};

} // namespace nuka::phi
