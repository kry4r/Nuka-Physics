#pragma once
// ---------------------------------------------------------------------------
// nuka::import - USD stage adapter: OpenUSD opens and composes every USD format
// ---------------------------------------------------------------------------

#include <pxr/usd/usd/stage.h>

#include <string>

namespace nuka::import {

enum class UsdStageFormat {
    Unknown,
    Usd,
    UsdaText,
    UsdcCrate,
    UsdzPackage
};

// The USD flavour a path's extension names, case-insensitively; a ".usd" holds either text or crate.
UsdStageFormat DetectUsdStageFormat(const std::string& path);

// Opens `path` in any format OpenUSD reads (usda, usdc, usdz, plus plugins such as abc and drc) with
// every payload loaded. Throws std::runtime_error with OpenUSD's messages when it fails to open or compose.
pxr::UsdStageRefPtr OpenUsdStage(const std::string& path);

} // namespace nuka::import
