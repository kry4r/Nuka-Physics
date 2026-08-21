#pragma once
// ---------------------------------------------------------------------------
// nk::Data — the mutable per-World state (plan §3.3).
//
// Data wraps an nk::Arena (the 3 device buffers) and fills a phi::DataView of
// typed device pointers, one per data-owned field, that the ops read & write.
// It also carries the host-side snapshot/restore plumbing that backs the
// SnapshotState / RestoreState ops (M3a: a host-side full-buffer copy via
// BufferI download/upload of the Persistent arena, which holds the live state).
//
// PURE C++ — zero CUDA tokens.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

#include "phi/backend.hpp"   // DataView, BufferType
#include "nk/data/arena.hpp"
#include "nk/model/generated/views.hpp"

namespace nuka::nk {

struct ModelCapacities;  // nk/model/model.hpp

class Data {
public:
    Data() = default;

    // Allocate the arena from the capacities and fill `out_view`. Ok / OOM.
    phi::Status Allocate(phi::BufferType* bt, const ModelCapacities& caps,
                         phi::DataView* out_view);

    // Re-fill `out_view` from the current arena (after a move or for a second
    // consumer). No allocation.
    void FillView(phi::DataView* out_view) const;

    // Zero every data-owned field (Reset baseline; the per-env ResetEnvs op does
    // the selective reset in M3b — M3a's Reset just re-zeros for now).
    void ZeroAll() { arena_.ZeroAll(); }

    // -- snapshot / restore plumbing (backs SnapshotState / RestoreState ops) --
    // M3a: a host-side full copy of the PERSISTENT arena buffer (the live state
    // class). The device op forms land in M3b; these give World a working
    // host-mediated path now (BufferI download/upload). Returns false if
    // unallocated.
    // Capture or restore the complete persistent arena. This is the production
    // checkpoint boundary for all cross-step state, including warm-start caches.
    uint64_t PersistentByteSize() const;
    bool DownloadPersistent(std::vector<uint8_t>* out) const;
    bool UploadPersistent(const std::vector<uint8_t>& bytes) const;

    bool Snapshot();
    bool Restore();
    bool HasSnapshot() const { return !snapshot_persistent_.empty(); }

    void* Ptr(FieldId id) const { return arena_.Ptr(id); }
    template <class T> T* Ptr(FieldId id) const { return arena_.Ptr<T>(id); }

    // Host<->field transfer (init-time seeding / test readback; see Arena).
    bool UploadField(FieldId id, const void* src, uint64_t bytes,
                     uint64_t byte_offset = 0) const {
        return arena_.UploadField(id, src, bytes, byte_offset);
    }
    bool DownloadField(FieldId id, void* dst, uint64_t bytes,
                       uint64_t byte_offset = 0) const {
        return arena_.DownloadField(id, dst, bytes, byte_offset);
    }

    const Arena& GetArena() const { return arena_; }
    const std::vector<Arena::Segment>& Segments() const { return arena_.Segments(); }

private:
    Arena arena_;
    std::vector<uint8_t> snapshot_persistent_;  // host mirror of the Persistent buffer.
};

} // namespace nuka::nk
