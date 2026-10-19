// ---------------------------------------------------------------------------
// nuka::import - USD stage adapter implementation
// ---------------------------------------------------------------------------

#include "import/usd_stage_adapter.hpp"

#include "import/usd_importer.hpp"

#include <pxr/base/tf/errorMark.h>
#include <pxr/usd/pcp/errors.h>
#include <pxr/usd/sdf/fileFormat.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>

namespace nuka::import {

namespace {

std::string ExtensionOf(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

} // namespace

UsdStageFormat DetectUsdStageFormat(const std::string& path) {
    const std::string ext = ExtensionOf(path);
    if (ext == ".usd") {
        return UsdStageFormat::Usd;
    }
    if (ext == ".usda") {
        return UsdStageFormat::UsdaText;
    }
    if (ext == ".usdc") {
        return UsdStageFormat::UsdcCrate;
    }
    if (ext == ".usdz") {
        return UsdStageFormat::UsdzPackage;
    }
    return UsdStageFormat::Unknown;
}

bool IsUsdPath(const std::string& path) {
    const std::string ext = ExtensionOf(path);
    return ext.size() > 1 && pxr::SdfFileFormat::FindByExtension(ext.substr(1));
}

pxr::UsdStageRefPtr OpenUsdStage(const std::string& path) {
    pxr::TfErrorMark mark;
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(path, pxr::UsdStage::LoadAll);
    std::string errors;
    for (auto error = mark.GetBegin(); error != mark.GetEnd(); ++error) {
        errors += "\n  " + error->GetCommentary();
    }
    mark.Clear();
    if (stage) {
        for (const pxr::PcpErrorBasePtr& error : stage->GetCompositionErrors()) {
            errors += "\n  " + error->ToString();
        }
    }
    if (!stage || !errors.empty()) {
        throw std::runtime_error("USD: cannot open " + path + errors);
    }
    return stage;
}

} // namespace nuka::import
