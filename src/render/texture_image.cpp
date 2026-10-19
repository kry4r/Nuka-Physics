// ---------------------------------------------------------------------------
// nuka::render -- host image decode (stb_image) for the beauty texture pipeline.
// The single translation unit that defines STB_IMAGE_IMPLEMENTATION. HOST, no CUDA.
// ---------------------------------------------------------------------------

#include "render/texture_image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_TGA
#define STBI_ONLY_BMP
#define STBI_ONLY_HDR
#include "stb/stb_image.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>

namespace nuka::render {

namespace {

// stb writes RGBE .hdr paths as linear floats; LDR bytes come back in [0,255].
constexpr float kInv255 = 1.0f / 255.0f;

// A path ending in "#offset+length" names image bytes inside a container file (a .glb).
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

unsigned char* DecodeImage(const std::string& path, int& w, int& h, int& comp) {
    std::string file;
    std::size_t offset = 0, length = 0;
    if (!SplitRange(path, file, offset, length)) return stbi_load(path.c_str(), &w, &h, &comp, 0);
    std::ifstream in(file, std::ios::binary);
    std::vector<unsigned char> bytes(length);
    in.seekg(static_cast<std::streamoff>(offset));
    if (!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(length))) return nullptr;
    return stbi_load_from_memory(bytes.data(), static_cast<int>(length), &w, &h, &comp, 0);
}

}  // namespace

rt::Texture LoadTexture(const std::string& path, bool srgb) {
    rt::Texture tex;
    if (path.empty()) return tex;
    int w = 0, h = 0, comp = 0;
    unsigned char* data = DecodeImage(path, w, h, comp);
    if (data == nullptr || w <= 0 || h <= 0 || comp <= 0) {
        if (data != nullptr) stbi_image_free(data);
        return tex;
    }
    tex.width = static_cast<uint32_t>(w);
    tex.height = static_cast<uint32_t>(h);
    tex.channels = static_cast<uint32_t>(comp);
    tex.srgb = srgb ? 1u : 0u;
    const std::size_t n = static_cast<std::size_t>(w) * h * comp;
    tex.texels.resize(n);
    // Store raw normalized [0,1]; the device sampler applies sRGB->linear per the flag.
    for (std::size_t i = 0; i < n; ++i) tex.texels[i] = static_cast<float>(data[i]) * kInv255;
    stbi_image_free(data);
    return tex;
}

rt::EnvironmentMap LoadEnvironment(const std::string& path, float yaw, float intensity) {
    rt::EnvironmentMap env;
    if (path.empty()) return env;
    env.yaw = yaw;
    env.intensity = intensity;
    int w = 0, h = 0, comp = 0;
    if (stbi_is_hdr(path.c_str())) {
        float* data = stbi_loadf(path.c_str(), &w, &h, &comp, 3);
        if (data == nullptr || w <= 0 || h <= 0) {
            if (data != nullptr) stbi_image_free(data);
            return env;
        }
        env.width = static_cast<uint32_t>(w);
        env.height = static_cast<uint32_t>(h);
        env.texels.assign(data, data + static_cast<std::size_t>(w) * h * 3u);
        stbi_image_free(data);
        return env;
    }
    // LDR fallback: decode to 3 channels and linearise from sRGB.
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (data == nullptr || w <= 0 || h <= 0) {
        if (data != nullptr) stbi_image_free(data);
        return env;
    }
    env.width = static_cast<uint32_t>(w);
    env.height = static_cast<uint32_t>(h);
    const std::size_t n = static_cast<std::size_t>(w) * h * 3u;
    env.texels.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float s = static_cast<float>(data[i]) * kInv255;
        env.texels[i] = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
    }
    stbi_image_free(data);
    return env;
}

}  // namespace nuka::render
