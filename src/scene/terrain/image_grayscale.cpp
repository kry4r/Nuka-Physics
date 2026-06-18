// ---------------------------------------------------------------------------
// image_grayscale.cpp -- self-written Netpbm PGM (P5/P2) + raw-16 decoder.
//
// Symmetric to recorder.cpp::WritePpmP6 (the only other Netpbm codec in the tree)
// but for grayscale READ. Supports 8- and 16-bit maxval (the precision a height
// map needs). PNG is rejected loudly -- it requires a self-written DEFLATE, a
// deliberate follow-up, not a stub. No external SDK.
// ---------------------------------------------------------------------------

#include "scene/terrain/image_grayscale.hpp"

#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace nuka::terrain {
namespace {

std::string LowerExt(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return std::string();
    std::string ext = path.substr(dot);
    for (char& ch : ext) ch = static_cast<char>(std::tolower(ch));
    return ext;
}

// Read the whole file into bytes. Returns false (err set) on open/read failure.
bool ReadAllBytes(const std::string& path, std::vector<unsigned char>& bytes,
                  std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open image file: " + path;
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        std::fclose(f);
        err = "empty image file: " + path;
        return false;
    }
    bytes.resize(static_cast<size_t>(size));
    const size_t got = std::fread(bytes.data(), 1u, bytes.size(), f);
    std::fclose(f);
    if (got != bytes.size()) {
        err = "short read on image file: " + path;
        return false;
    }
    return true;
}

// Advance pos past whitespace and '#'-to-end-of-line comments (Netpbm rule).
void SkipWhitespaceAndComments(const std::vector<unsigned char>& b, size_t& pos) {
    while (pos < b.size()) {
        const unsigned char ch = b[pos];
        if (ch == '#') {
            while (pos < b.size() && b[pos] != '\n') ++pos;
        } else if (std::isspace(ch)) {
            ++pos;
        } else {
            break;
        }
    }
}

// Parse a non-negative ASCII integer token. Returns false on no digits.
bool ParseUint(const std::vector<unsigned char>& b, size_t& pos, uint32_t& out) {
    SkipWhitespaceAndComments(b, pos);
    if (pos >= b.size() || !std::isdigit(b[pos])) return false;
    uint32_t v = 0u;
    while (pos < b.size() && std::isdigit(b[pos])) {
        v = v * 10u + static_cast<uint32_t>(b[pos] - '0');
        ++pos;
    }
    out = v;
    return true;
}

}  // namespace

bool LoadGrayscalePgm(const std::string& path, GrayscaleImage& out,
                      std::string& err) {
    out = GrayscaleImage{};
    std::vector<unsigned char> b;
    if (!ReadAllBytes(path, b, err)) return false;
    if (b.size() < 2u || b[0] != 'P' || (b[1] != '5' && b[1] != '2')) {
        err = "not a PGM (P5/P2) file: " + path;
        return false;
    }
    const bool binary = (b[1] == '5');
    size_t pos = 2u;
    uint32_t w = 0u, h = 0u, maxval = 0u;
    if (!ParseUint(b, pos, w) || !ParseUint(b, pos, h) ||
        !ParseUint(b, pos, maxval)) {
        err = "malformed PGM header (need W H maxval): " + path;
        return false;
    }
    if (w < 1u || h < 1u || maxval < 1u) {
        err = "degenerate PGM dimensions/maxval: " + path;
        return false;
    }
    const bool wide = maxval > 255u;
    const float inv = 1.0f / static_cast<float>(maxval);
    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);
    out.values.resize(count);
    out.width = w;
    out.height = h;

    if (binary) {
        // A single whitespace byte separates the maxval and the raster (Netpbm).
        if (pos < b.size() && std::isspace(b[pos])) ++pos;
        const size_t bytes_per = wide ? 2u : 1u;
        if (b.size() - pos < count * bytes_per) {
            err = "truncated PGM raster: " + path;
            return false;
        }
        for (size_t i = 0u; i < count; ++i) {
            uint32_t v;
            if (wide) {
                // Netpbm 16-bit is big-endian (most-significant byte first).
                v = (static_cast<uint32_t>(b[pos]) << 8) | b[pos + 1u];
                pos += 2u;
            } else {
                v = b[pos++];
            }
            out.values[i] = static_cast<float>(v) * inv;
        }
    } else {
        for (size_t i = 0u; i < count; ++i) {
            uint32_t v;
            if (!ParseUint(b, pos, v)) {
                err = "truncated ASCII PGM raster: " + path;
                return false;
            }
            out.values[i] = static_cast<float>(v) * inv;
        }
    }
    return true;
}

bool LoadGrayscaleRaw16(const std::string& path, uint32_t width, uint32_t height,
                        GrayscaleImage& out, std::string& err) {
    out = GrayscaleImage{};
    if (width < 1u || height < 1u) {
        err = "raw-16 needs positive width/height: " + path;
        return false;
    }
    std::vector<unsigned char> b;
    if (!ReadAllBytes(path, b, err)) return false;
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (b.size() < count * 2u) {
        err = "raw-16 file shorter than width*height*2: " + path;
        return false;
    }
    out.values.resize(count);
    out.width = width;
    out.height = height;
    const float inv = 1.0f / 65535.0f;
    for (size_t i = 0u; i < count; ++i) {
        const uint32_t v =
            (static_cast<uint32_t>(b[i * 2u]) << 8) | b[i * 2u + 1u];
        out.values[i] = static_cast<float>(v) * inv;
    }
    return true;
}

bool LoadGrayscaleImage(const std::string& path, GrayscaleImage& out,
                        std::string& err) {
    const std::string ext = LowerExt(path);
    if (ext == ".pgm") {
        return LoadGrayscalePgm(path, out, err);
    }
    if (ext == ".png") {
        err = "PNG height maps are not supported (needs a self-written DEFLATE "
              "decoder); author the height map as PGM (P5/P2) or raw-16: " +
              path;
        return false;
    }
    // .r16/.raw carry no dimensions in-band -> caller must use LoadGrayscaleRaw16.
    err = "unsupported height-map image extension '" + ext +
          "' (supported: .pgm; raw-16 via the dims-explicit loader): " + path;
    return false;
}

}  // namespace nuka::terrain
