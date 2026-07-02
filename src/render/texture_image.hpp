#pragma once
// ---------------------------------------------------------------------------
// nuka::render -- host image decode for the offline beauty texture pipeline.
//
// Decodes an image file (PNG/JPG/TGA/BMP via the vendored stb_image, HDR/Radiance
// .hdr via stbi_loadf) into an rt::Texture (LDR maps) or rt::EnvironmentMap (HDR
// equirect). HOST-ONLY, ZERO CUDA: lives in nuka_render; the device upload of the
// decoded texels happens later in the CUDA backend (BuildTwoLevelScene).
// ---------------------------------------------------------------------------

#include "rt/material.hpp"   // rt::Texture / rt::EnvironmentMap

#include <string>

namespace nuka::render {

// Decode `path` into an rt::Texture. `srgb` flags an albedo image (the sampler
// linearises it on read; roughness/normal maps pass srgb=false). Returns an empty
// texture (Empty()==true) if the file is missing or fails to decode.
rt::Texture LoadTexture(const std::string& path, bool srgb);

// Decode an equirectangular environment map. A `.hdr` (Radiance RGBE) loads as
// linear HDR radiance directly; an LDR image is linearised from sRGB. `yaw`/
// `intensity` seed the returned map. Empty (Enabled()==false) on failure.
rt::EnvironmentMap LoadEnvironment(const std::string& path, float yaw, float intensity);

}  // namespace nuka::render
