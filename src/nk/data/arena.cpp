// ---------------------------------------------------------------------------
// nk::Arena implementation (plan §3.3).
// ---------------------------------------------------------------------------

#include "nk/data/arena.hpp"

#include <utility>

#include "nk/model/model.hpp"   // ModelCapacities

namespace nuka::nk {

namespace {

constexpr uint64_t kAlign = 256;
uint64_t AlignUp(uint64_t v) { return (v + (kAlign - 1)) & ~(kAlign - 1); }

uint8_t ArenaIndex(FieldArena a) {
    switch (a) {
        case FieldArena::Persistent: return 0;
        case FieldArena::Scratch:    return 1;
        case FieldArena::Tape:       return 2;
    }
    return 0;
}

}  // namespace

std::vector<Arena::Segment> Arena::ComputeSegments(const ModelCapacities& caps,
                                                   uint64_t arena_bytes[3]) {
    uint64_t cursor[3] = {0, 0, 0};
    std::vector<Segment> segs;
    for (int i = 0; i < kFieldCount; ++i) {
        const FieldId id = static_cast<FieldId>(i);
        const FieldLayout& lay = LayoutOf(id);
        if (lay.owner != FieldOwner::Data) {
            continue;
        }
        const uint8_t ai = ArenaIndex(lay.arena);
        const uint64_t count = caps.ElementCount(id);
        const uint64_t bytes = count * static_cast<uint64_t>(lay.elem_size);
        const uint64_t off = AlignUp(cursor[ai]);
        segs.push_back(Segment{id, ai, off, bytes});
        cursor[ai] = off + bytes;
    }
    for (int a = 0; a < 3; ++a) {
        arena_bytes[a] = AlignUp(cursor[a]);
    }
    return segs;
}

phi::Status Arena::Allocate(phi::BufferType* bt, const ModelCapacities& caps) {
    if (bt == nullptr) {
        return phi::Status::Failed;
    }
    FreeAll();
    segments_ = ComputeSegments(caps, arena_bytes_);

    phi::Buffer** bufs[3] = {&persistent_, &scratch_, &tape_};
    for (int a = 0; a < 3; ++a) {
        const uint64_t n = arena_bytes_[a] == 0 ? kAlign : arena_bytes_[a];
        phi::Buffer* b = phi::BufferAlloc(bt, n);
        if (b == nullptr) {
            FreeAll();
            return phi::Status::OutOfMemory;
        }
        *bufs[a] = b;
    }
    ZeroAll();
    return phi::Status::Ok;
}

void* Arena::Ptr(FieldId id) const {
    if (persistent_ == nullptr) {
        return nullptr;
    }
    for (const Segment& s : segments_) {
        if (s.field != id) {
            continue;
        }
        phi::Buffer* b = (s.arena == 0) ? persistent_
                       : (s.arena == 1) ? scratch_
                                        : tape_;
        return static_cast<uint8_t*>(phi::BufferBase(b)) + s.offset;
    }
    return nullptr;
}

void Arena::ZeroAll() {
    phi::Buffer* bufs[3] = {persistent_, scratch_, tape_};
    for (int a = 0; a < 3; ++a) {
        if (bufs[a] != nullptr && arena_bytes_[a] > 0) {
            phi::BufferMemset(bufs[a], 0, 0, arena_bytes_[a]);
        }
    }
}

void Arena::FreeAll() {
    if (persistent_ != nullptr) { phi::BufferFree(persistent_); persistent_ = nullptr; }
    if (scratch_    != nullptr) { phi::BufferFree(scratch_);    scratch_    = nullptr; }
    if (tape_       != nullptr) { phi::BufferFree(tape_);       tape_       = nullptr; }
    arena_bytes_[0] = arena_bytes_[1] = arena_bytes_[2] = 0;
    segments_.clear();
}

Arena::~Arena() { FreeAll(); }

Arena::Arena(Arena&& o) noexcept
    : persistent_(o.persistent_), scratch_(o.scratch_), tape_(o.tape_),
      segments_(std::move(o.segments_)) {
    arena_bytes_[0] = o.arena_bytes_[0];
    arena_bytes_[1] = o.arena_bytes_[1];
    arena_bytes_[2] = o.arena_bytes_[2];
    o.persistent_ = o.scratch_ = o.tape_ = nullptr;
    o.arena_bytes_[0] = o.arena_bytes_[1] = o.arena_bytes_[2] = 0;
}

Arena& Arena::operator=(Arena&& o) noexcept {
    if (this != &o) {
        FreeAll();
        persistent_ = o.persistent_;
        scratch_    = o.scratch_;
        tape_       = o.tape_;
        arena_bytes_[0] = o.arena_bytes_[0];
        arena_bytes_[1] = o.arena_bytes_[1];
        arena_bytes_[2] = o.arena_bytes_[2];
        segments_ = std::move(o.segments_);
        o.persistent_ = o.scratch_ = o.tape_ = nullptr;
        o.arena_bytes_[0] = o.arena_bytes_[1] = o.arena_bytes_[2] = 0;
    }
    return *this;
}

} // namespace nuka::nk
