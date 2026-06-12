#pragma once
// ---------------------------------------------------------------------------
// PHI v2 — buffer interface (ggml_backend_buffer analogue).
//
// This is the v2 REWRITE of phi/buffer.hpp. The legacy RAII Buffer/BufferView
// (consumed by all current production paths) was renamed to
// phi/buffer_legacy.hpp and is untouched until M9. The v2 contract here is a
// pure-vtable, opaque-handle design: `BufferType` is the allocator factory
// (one per backend memory class — device / pinned-host), `Buffer` is an
// allocation. Both are OPAQUE structs whose real definitions live in the
// backend implementation (phi/backend_cuda/cuda_buffer.cu); each carries its
// vtable pointer as its first member so the free-function wrappers below can
// dispatch without seeing the backend's CUDA types.
//
// PURE C++ — zero CUDA types. No STL in the public surface.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

namespace nuka::phi {

// Opaque handles. Defined in the backend; the FIRST member of each concrete
// definition MUST be `const <Iface>* iface;` so the wrappers below can read the
// vtable through the handle (the ggml pattern).
struct BufferType;
struct Buffer;

// ---------------------------------------------------------------------------
// BufferType — allocator factory for one memory class (ggml_backend_buffer_type_i).
// ---------------------------------------------------------------------------
struct BufferTypeI {
    const char* (*get_name)(BufferType*);
    Buffer*     (*alloc)(BufferType*, size_t bytes);   // backend owns device memory
    size_t      (*alignment)(BufferType*);             // CUDA returns 256
    bool        (*is_host)(BufferType*);
};

// ---------------------------------------------------------------------------
// Buffer — a single allocation (ggml_backend_buffer_i).
//   * upload/download are async on the backend's main stream.
//   * memset fills [off, off+n) with byte value v.
//   * copy_from is a device-to-device copy dst[doff..] <- src[soff..] (n bytes).
// ---------------------------------------------------------------------------
struct BufferI {
    void  (*free)(Buffer*);
    void* (*base)(Buffer*);                                          // device base address
    void  (*upload)  (Buffer*, const void* src, size_t off, size_t n);   // async on backend stream
    void  (*download)(Buffer*, void* dst,       size_t off, size_t n);
    void  (*memset)  (Buffer*, uint8_t v,       size_t off, size_t n);
    void  (*copy_from)(Buffer* dst, Buffer* src, size_t doff, size_t soff, size_t n); // D2D
};

// The opaque-handle layout contract: every concrete BufferType / Buffer begins
// with its interface pointer. The wrappers reinterpret the handle as this head
// to fetch the vtable. Backends MUST keep `iface` first.
struct BufferTypeHead { const BufferTypeI* iface; };
struct BufferHead     { const BufferI*     iface; };

inline const BufferTypeI* IfaceOf(BufferType* t) {
    return reinterpret_cast<const BufferTypeHead*>(t)->iface;
}
inline const BufferI* IfaceOf(Buffer* b) {
    return reinterpret_cast<const BufferHead*>(b)->iface;
}

// --- BufferType convenience wrappers ------------------------------------
inline const char* BufferTypeName(BufferType* t)   { return IfaceOf(t)->get_name(t); }
inline Buffer*     BufferAlloc(BufferType* t, size_t bytes) { return IfaceOf(t)->alloc(t, bytes); }
inline size_t      BufferTypeAlignment(BufferType* t) { return IfaceOf(t)->alignment(t); }
inline bool        BufferTypeIsHost(BufferType* t) { return IfaceOf(t)->is_host(t); }

// --- Buffer convenience wrappers ----------------------------------------
inline void  BufferFree(Buffer* b)  { IfaceOf(b)->free(b); }
inline void* BufferBase(Buffer* b)  { return IfaceOf(b)->base(b); }
inline void  BufferUpload(Buffer* b, const void* src, size_t off, size_t n)  { IfaceOf(b)->upload(b, src, off, n); }
inline void  BufferDownload(Buffer* b, void* dst, size_t off, size_t n)      { IfaceOf(b)->download(b, dst, off, n); }
inline void  BufferMemset(Buffer* b, uint8_t v, size_t off, size_t n)        { IfaceOf(b)->memset(b, v, off, n); }
inline void  BufferCopyFrom(Buffer* dst, Buffer* src, size_t doff, size_t soff, size_t n) {
    IfaceOf(dst)->copy_from(dst, src, doff, soff, n);
}

} // namespace nuka::phi
