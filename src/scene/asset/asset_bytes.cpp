#include "scene/asset/asset_bytes.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>

#include "scene/asset/asset_ref.hpp"
#include "scene/asset/nka.hpp"

namespace nuka::scene {
namespace {

// A path ending in "#offset+length" names bytes inside a container file (a .glb, a .usdz).
bool SplitRange(const std::string& path, std::string& file, std::size_t& offset, std::size_t& length) {
    const std::size_t hash = path.rfind('#');
    const std::size_t plus = path.rfind('+');
    if (hash == std::string::npos || plus == std::string::npos || plus < hash + 2 || plus + 1 == path.size())
        return false;
    const auto digits = [&](std::size_t b, std::size_t e) {
        return std::all_of(path.begin() + b, path.begin() + e, [](unsigned char c) { return std::isdigit(c); });
    };
    if (!digits(hash + 1, plus) || !digits(plus + 1, path.size())) return false;
    file = path.substr(0, hash);
    offset = std::stoull(path.substr(hash + 1, plus - hash - 1));
    length = std::stoull(path.substr(plus + 1));
    return true;
}

}  // namespace

bool NamesTextureChunk(const std::string& path) {
    const std::size_t hash = path.rfind('#');
    return hash != std::string::npos && path.compare(hash, 6, "#TEXB/") == 0;
}

std::vector<uint8_t> ReadAssetBytes(const std::string& path) {
    if (NamesTextureChunk(path)) {
        const AssetRef ref = ParseAssetRef(path);
        if (!std::filesystem::exists(ref.nka_path)) return {};
        const NkaFile nka = NkaFile::Open(ref.nka_path);
        if (ref.index >= nka.CountChunks(ref.fourcc)) return {};
        return nka.LoadChunk(ref.fourcc, ref.index);
    }
    std::string file = path;
    std::size_t offset = 0, length = 0;
    const bool range = SplitRange(path, file, offset, length);
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!range) length = static_cast<std::size_t>(std::max<std::streamoff>(in.tellg(), 0));
    std::vector<uint8_t> bytes(length);
    in.seekg(static_cast<std::streamoff>(offset));
    if (!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(length))) bytes.clear();
    return bytes;
}

}  // namespace nuka::scene
