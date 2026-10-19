#pragma once
// ---------------------------------------------------------------------------
// nuka::import - USD importer
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

#include <string>

namespace nuka::import {

// Loads a USD scene into the canonical SceneIR. OpenUSD opens and composes the file (references,
// payloads, variants, instancing), so every format it reads imports the same way.
nuka::scene::SceneIR LoadUsd(const std::string& path);

// True when OpenUSD reads the path's extension: usd, usda, usdc, usdz, plus plugin formats (abc, drc).
bool IsUsdPath(const std::string& path);

} // namespace nuka::import
