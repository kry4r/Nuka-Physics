#pragma once
// ---------------------------------------------------------------------------
// nuka::import - USD / USDA importer
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

#include <string>

namespace nuka::import {

// Load a USD scene into the canonical SceneIR. The importer consumes stage data
// from usd_stage_adapter, which supports ASCII USDA/text USD today and isolates
// the future OpenUSD SDK backend for USDC/USDZ without changing callers.
nuka::scene::SceneIR LoadUsd(const std::string& path);

} // namespace nuka::import
