#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "nk/pipeline/world.hpp"

namespace nuka::demo {

inline void FlushCaptureFile(const std::filesystem::path& path) {
#ifdef _WIN32
    const int descriptor = _wopen(path.c_str(), _O_RDWR | _O_BINARY);
    if (descriptor < 0) throw std::runtime_error("Cannot open capture for flush");
    const int result = _commit(descriptor);
    _close(descriptor);
#else
    const int descriptor = open(path.c_str(), O_RDONLY);
    if (descriptor < 0) throw std::runtime_error("Cannot open capture for flush");
    const int result = fsync(descriptor);
    close(descriptor);
#endif
    if (result != 0) throw std::runtime_error("Capture file flush failed");
}

template <class T> void WriteCheckpointValue(std::ostream& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <class T> void ReadCheckpointValue(std::istream& in, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
}

inline uint64_t CaptureFileFingerprint(const std::filesystem::path& path,
                                      uint64_t bytes = std::numeric_limits<uint64_t>::max()) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot fingerprint " + path.string());
    const uint64_t size = std::filesystem::file_size(path);
    if (bytes == std::numeric_limits<uint64_t>::max()) bytes = size;
    if (bytes > size) throw std::runtime_error("Incomplete fingerprint input");
    uint64_t hash = 14695981039346656037ull;
    std::array<char, 65536u> buffer{};
    while (bytes != 0u) {
        const auto count = static_cast<std::streamsize>(std::min<uint64_t>(bytes, buffer.size()));
        in.read(buffer.data(), count);
        if (!in) throw std::runtime_error("Fingerprint read failed");
        for (std::streamsize i = 0; i < count; ++i) {
            hash ^= static_cast<uint8_t>(buffer[i]);
            hash *= 1099511628211ull;
        }
        bytes -= static_cast<uint64_t>(count);
    }
    return hash;
}

struct CaptureCheckpointHeader {
    uint64_t magic = 0x4E554B4143484B50ull;
    uint64_t identity = 0u;
    uint64_t sequence = 0u;
    uint64_t persistent_bytes = 0u;
    uint64_t recorder_bytes = 0u;
    uint64_t capture_bytes = 0u;
    uint64_t metrics_bytes = 0u;
    uint32_t version = 1u;
    uint32_t frame = 0u;
};

struct CaptureCheckpoint {
    CaptureCheckpointHeader header;
    std::vector<uint8_t> persistent, recorder;
};

// Alternating complete files retain a recoverable state if a write is interrupted.
inline void SaveCaptureCheckpoint(const std::filesystem::path& directory, nk::World& world,
                                  CaptureCheckpoint& checkpoint) {
    FlushCaptureFile(directory / "states.bin");
    FlushCaptureFile(directory / "metrics.csv");
    if (world.Synchronize() != phi::Status::Ok ||
        !world.GetData().DownloadPersistent(&checkpoint.persistent))
        throw std::runtime_error("Checkpoint state download failed");
    checkpoint.header.persistent_bytes = checkpoint.persistent.size();
    checkpoint.header.recorder_bytes = checkpoint.recorder.size();
    const auto temporary = directory / "checkpoint.tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        WriteCheckpointValue(out, checkpoint.header);
        out.write(reinterpret_cast<const char*>(checkpoint.persistent.data()), checkpoint.persistent.size());
        out.write(reinterpret_cast<const char*>(checkpoint.recorder.data()), checkpoint.recorder.size());
        out.flush();
        if (!out) throw std::runtime_error("Checkpoint write failed");
    }
    const uint64_t checksum = CaptureFileFingerprint(temporary);
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::app);
        WriteCheckpointValue(out, checksum);
        out.flush();
        if (!out) throw std::runtime_error("Checkpoint checksum write failed");
    }
    FlushCaptureFile(temporary);
    const auto destination = directory /
        ("checkpoint_" + std::to_string(checkpoint.header.sequence % 2u) + ".bin");
    std::filesystem::remove(destination);
    std::filesystem::rename(temporary, destination);
}

inline CaptureCheckpoint LoadCaptureCheckpoint(const std::filesystem::path& directory,
                                               nk::World& world, uint64_t identity,
                                               uint64_t recorder_bytes, uint32_t last_frame) {
    CaptureCheckpoint selected;
    bool found = false;
    for (uint32_t slot = 0u; slot < 2u; ++slot) {
        const auto path = directory / ("checkpoint_" + std::to_string(slot) + ".bin");
        std::ifstream in(path, std::ios::binary);
        if (!in) continue;
        CaptureCheckpoint candidate;
        auto& header = candidate.header;
        ReadCheckpointValue(in, header);
        const uint64_t expected = sizeof(header) + world.GetData().PersistentByteSize() +
                                  recorder_bytes + sizeof(uint64_t);
        if (!in || header.magic != CaptureCheckpointHeader{}.magic || header.version != 1u ||
            header.identity != identity || header.frame > last_frame ||
            header.persistent_bytes != world.GetData().PersistentByteSize() ||
            header.recorder_bytes != recorder_bytes || std::filesystem::file_size(path) != expected ||
            header.capture_bytes > std::filesystem::file_size(directory / "states.bin") ||
            header.metrics_bytes > std::filesystem::file_size(directory / "metrics.csv")) continue;
        in.seekg(static_cast<std::streamoff>(expected - sizeof(uint64_t)));
        uint64_t checksum = 0u;
        ReadCheckpointValue(in, checksum);
        if (!in || checksum != CaptureFileFingerprint(path, expected - sizeof(uint64_t))) continue;
        if (found && header.sequence <= selected.header.sequence) continue;
        in.seekg(sizeof(header));
        candidate.persistent.resize(header.persistent_bytes);
        candidate.recorder.resize(recorder_bytes);
        in.read(reinterpret_cast<char*>(candidate.persistent.data()), candidate.persistent.size());
        in.read(reinterpret_cast<char*>(candidate.recorder.data()), candidate.recorder.size());
        if (!in) continue;
        selected = std::move(candidate);
        found = true;
    }
    if (!found) throw std::runtime_error("No complete checkpoint matches this executable and capture configuration");
    return selected;
}

}  // namespace nuka::demo
