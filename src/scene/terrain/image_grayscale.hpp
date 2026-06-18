#pragma once
// ---------------------------------------------------------------------------
// image_grayscale.hpp -- a minimal self-written grayscale image decoder.
//
// Decodes a height-map image into normalized [0,1] samples (row-major, top row
// first). Supported: Netpbm PGM (P5 binary, P2 ASCII, 8- or 16-bit maxval) and a
// raw 16-bit big-endian grid (.r16, dims carried out-of-band by the caller). PNG
// is NOT supported (it needs a self-written DEFLATE) -- a .png path is a LOUD
// failure with a clear message, never a silent flat fallback. No external SDK.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::terrain {

// A decoded grayscale grid: normalized [0,1] samples, row-major, width*height.
struct GrayscaleImage {
    std::vector<float> values;  // length width*height, in [0,1].
    uint32_t width = 0u;        // columns (X).
    uint32_t height = 0u;       // rows (Y).
};

// Decode a PGM (P5/P2) file -> normalized grayscale. Returns false (with err set)
// on any malformed/undecodable input or an unsupported format. err is the loud
// human-readable reason. A .png path fails here with a clear "not supported".
bool LoadGrayscalePgm(const std::string& path, GrayscaleImage& out,
                      std::string& err);

// Decode a raw 16-bit big-endian grid of the given dims -> normalized grayscale
// (value/65535). The caller supplies width/height (the raw format carries none).
bool LoadGrayscaleRaw16(const std::string& path, uint32_t width, uint32_t height,
                        GrayscaleImage& out, std::string& err);

// Dispatch by file extension: .pgm/.ppm-gray -> PGM; .r16/.raw needs explicit
// dims (use LoadGrayscaleRaw16 directly). .png -> a loud "not supported" error.
bool LoadGrayscaleImage(const std::string& path, GrayscaleImage& out,
                        std::string& err);

}  // namespace nuka::terrain
