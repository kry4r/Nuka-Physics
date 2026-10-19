#pragma once
// ---------------------------------------------------------------------------
// nuka::scene -- the bytes an image path names, wherever they live: a file, a
// range inside a container file, or a TEXB chunk of an .nka.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

namespace nuka::scene {

// True when `path` names a TEXB chunk of an .nka ("x.nka#TEXB/2"), the form a packed image takes.
bool NamesTextureChunk(const std::string& path);

// The bytes `path` names: a TEXB chunk, a "file#offset+length" range inside a container (a .glb, a .usdz), or a
// whole file. Empty when the file or the chunk is missing.
std::vector<uint8_t> ReadAssetBytes(const std::string& path);

}  // namespace nuka::scene
