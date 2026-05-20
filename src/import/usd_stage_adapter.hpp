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

UsdStageData LoadUsdStageData(const std::string& path);

} // namespace nuka::import
