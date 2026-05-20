#pragma once
// ---------------------------------------------------------------------------
// nuka::import - USD / USDA importer
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

#include <string>

namespace nuka::import {

// Load a USD scene into the canonical SceneIR. The current adapter supports
// ASCII USDA/text USD and isolates the import boundary for future OpenUSD SDK
// replacement without changing engine-facing callers.
nuka::scene::SceneIR LoadUsd(const std::string& path);

} // namespace nuka::import
