#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "phi/buffer.hpp"

namespace nuka::phi {

// A reusable copy of backend-owned buffers; capture and restore preserve live addresses.
class BufferSnapshot {
public:
    struct Region { Buffer* buffer; size_t bytes; };

    Status Capture(BufferType* type, const std::vector<Region>& regions) {
        valid_ = false;
        if (entries_.size() != regions.size()) entries_.clear();
        entries_.resize(regions.size());
        for (size_t i = 0; i < regions.size(); ++i) {
            auto& entry = entries_[i];
            const auto& region = regions[i];
            if (!region.buffer) return Status::InvalidArgument;
            if (!entry.copy || entry.bytes != region.bytes) {
                Status status = Status::Ok;
                entry.copy.reset(BufferAlloc(type, region.bytes, &status));
                if (!entry.copy) return status == Status::Ok ? Status::OutOfMemory : status;
                entry.bytes = region.bytes;
            }
            entry.live = region.buffer;
            const auto status = BufferCopyFrom(entry.copy.get(), entry.live, 0, 0, entry.bytes);
            if (status != Status::Ok) return status;
        }
        valid_ = true;
        return Status::Ok;
    }

    Status Restore() const {
        if (!valid_) return Status::InvalidArgument;
        for (const auto& entry : entries_) {
            const auto status = BufferCopyFrom(entry.live, entry.copy.get(), 0, 0, entry.bytes);
            if (status != Status::Ok) return status;
        }
        return Status::Ok;
    }

    size_t ByteSize() const {
        size_t bytes = 0;
        for (const auto& entry : entries_) bytes += entry.bytes;
        return bytes;
    }

private:
    struct Release { void operator()(Buffer* b) const { if (b) BufferFree(b); } };
    struct Entry {
        Buffer* live = nullptr;
        size_t bytes = 0;
        std::unique_ptr<Buffer, Release> copy;
    };
    std::vector<Entry> entries_;
    bool valid_ = false;
};

}  // namespace nuka::phi
