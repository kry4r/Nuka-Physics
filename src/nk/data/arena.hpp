#pragma once
// ---------------------------------------------------------------------------
// nk::Arena — the data-owned device memory (plan §3.3).
//
// Arena owns the THREE phi Buffers of an nk::Data (Persistent / Scratch / Tape)
// sized from the Model capacities x env_count via the generated arena_layout
// table. Each data-owned field gets one 256B-aligned segment within its arena's
// buffer; the layout is env-major and DETERMINISTIC (same Model + env_count =>
// byte-identical segment table). Ptr(FieldId) / Ptr<T>(FieldId) return the
// device base of a field's segment. Zero-initialized via BufferI memset.
//
// PURE C++ — zero CUDA tokens. All device memory via phi::BufferType/BufferI.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

#include "phi/backend.hpp"   // BufferType, Buffer
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/generated/arena_layout.hpp"

namespace nuka::nk {

struct ModelCapacities;  // nk/model/model.hpp

class Arena {
public:
    Arena() = default;
    ~Arena();
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) noexcept;
    Arena& operator=(Arena&&) noexcept;

    // Allocate the 3 arena buffers from the capacities and lay out + memset(0)
    // every data-owned field. Returns Ok / OutOfMemory / Failed.
    phi::Status Allocate(phi::BufferType* bt, const ModelCapacities& caps);

    // Device base pointer of a field's segment (null if the field is not data-
    // owned or Arena is unallocated).
    void* Ptr(FieldId id) const;
    template <class T> T* Ptr(FieldId id) const { return static_cast<T*>(Ptr(id)); }

    // Zero every data-owned field (used by Reset / construction). No-op if
    // unallocated.
    void ZeroAll();

    // The per-field segment descriptor (one per data-owned field, in FieldId
    // order). Public so the acceptance test can assert determinism.
    struct Segment {
        FieldId  field;
        uint8_t  arena;    // FieldArena as uint8 (Persistent=0/Scratch=1/Tape=2)
        uint64_t offset;   // byte offset within that arena's buffer
        uint64_t bytes;
    };
    const std::vector<Segment>& Segments() const { return segments_; }

    // Compute the segment table + per-arena totals WITHOUT allocating (the
    // determinism gate compares two of these). Pure function of `caps`.
    static std::vector<Segment> ComputeSegments(const ModelCapacities& caps,
                                                 uint64_t arena_bytes[3]);

    phi::Buffer* PersistentBuffer() const { return persistent_; }
    phi::Buffer* ScratchBuffer() const { return scratch_; }
    phi::Buffer* TapeBuffer() const { return tape_; }

private:
    void FreeAll();

    phi::Buffer* persistent_ = nullptr;
    phi::Buffer* scratch_    = nullptr;
    phi::Buffer* tape_       = nullptr;
    uint64_t     arena_bytes_[3] = {0, 0, 0};
    std::vector<Segment> segments_;  // FieldId order, data-owned only.
};

} // namespace nuka::nk
