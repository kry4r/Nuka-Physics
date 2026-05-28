#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::tests::oracle {

enum class GoldenKind : uint32_t {
    RandomQacc = 1u,
    JointTrajectory = 2u,
};

struct GoldenTrajectory {
    GoldenKind kind = GoldenKind::RandomQacc;
    uint32_t sample_count = 0u;
    uint32_t qpos_count = 0u;
    uint32_t qvel_count = 0u;
    uint32_t qacc_count = 0u;
    std::string model_name;
    std::vector<float> payload;

    size_t RecordFloatCount() const {
        if (kind == GoldenKind::RandomQacc) {
            return static_cast<size_t>(qpos_count) +
                   static_cast<size_t>(qvel_count) +
                   static_cast<size_t>(qvel_count) +
                   static_cast<size_t>(qacc_count);
        }
        if (kind == GoldenKind::JointTrajectory) {
            return static_cast<size_t>(qpos_count) +
                   static_cast<size_t>(qvel_count) +
                   static_cast<size_t>(qacc_count);
        }
        return 0u;
    }

    const float* Record(uint32_t sample_index) const {
        if (sample_index >= sample_count) {
            throw std::out_of_range("golden sample index out of range");
        }
        return payload.data() + static_cast<size_t>(sample_index) * RecordFloatCount();
    }
};

inline uint32_t ReadU32(std::ifstream& in, const char* field) {
    uint32_t value = 0u;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error(std::string("failed to read golden field: ") + field);
    }
    return value;
}

inline GoldenTrajectory LoadGoldenTrajectory(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open golden trajectory: " + path.string());
    }

    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in || std::string_view(magic.data(), magic.size()) != "NUKAGOLD") {
        throw std::runtime_error("invalid golden trajectory magic: " + path.string());
    }

    const uint32_t version = ReadU32(in, "version");
    if (version != 1u) {
        throw std::runtime_error("unsupported golden trajectory version: " +
                                 std::to_string(version));
    }

    GoldenTrajectory result;
    const uint32_t kind = ReadU32(in, "kind");
    if (kind != static_cast<uint32_t>(GoldenKind::RandomQacc) &&
        kind != static_cast<uint32_t>(GoldenKind::JointTrajectory)) {
        throw std::runtime_error("unsupported golden trajectory kind: " +
                                 std::to_string(kind));
    }
    result.kind = static_cast<GoldenKind>(kind);
    result.sample_count = ReadU32(in, "sample_count");
    result.qpos_count = ReadU32(in, "qpos_count");
    result.qvel_count = ReadU32(in, "qvel_count");
    result.qacc_count = ReadU32(in, "qacc_count");
    const uint32_t name_size = ReadU32(in, "model_name_size");
    if (name_size > 4096u) {
        throw std::runtime_error("golden model name is too large");
    }
    result.model_name.resize(name_size);
    if (name_size > 0u) {
        in.read(result.model_name.data(), static_cast<std::streamsize>(name_size));
        if (!in) {
            throw std::runtime_error("failed to read golden model name");
        }
    }

    const size_t record_floats = result.RecordFloatCount();
    if (result.sample_count == 0u || record_floats == 0u) {
        throw std::runtime_error("golden trajectory has empty shape");
    }
    const size_t expected_floats =
        static_cast<size_t>(result.sample_count) * record_floats;
    result.payload.resize(expected_floats);
    in.read(reinterpret_cast<char*>(result.payload.data()),
            static_cast<std::streamsize>(expected_floats * sizeof(float)));
    if (!in) {
        throw std::runtime_error("golden trajectory payload is truncated");
    }
    char extra = 0;
    if (in.read(&extra, 1)) {
        throw std::runtime_error("golden trajectory payload has trailing bytes");
    }
    return result;
}

} // namespace nuka::tests::oracle
