#pragma once
// ---------------------------------------------------------------------------
// nuka::import - isolated USD stage adapter boundary
// ---------------------------------------------------------------------------

#include <string>

namespace nuka::import {

enum class UsdStageFormat {
    Unknown,
    Usd,
    UsdaText,
    UsdcCrate,
    UsdzPackage
};

struct UsdStageData {
    UsdStageFormat format = UsdStageFormat::Unknown;
    std::string path;
    std::string text;
};

UsdStageFormat DetectUsdStageFormat(const std::string& path);

// True iff the file begins with the USDC crate magic ("PXR-USDC"). A ".usd"
// file may be either ASCII or a crate, so callers that must distinguish the
// binary crate use this content sniff rather than the extension alone.
bool IsUsdcCrateFile(const std::string& path);

UsdStageData LoadUsdStageData(const std::string& path);

} // namespace nuka::import
