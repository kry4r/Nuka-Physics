// ---------------------------------------------------------------------------
// nuka::import - isolated USD stage adapter implementation
// ---------------------------------------------------------------------------

#include "import/usd_stage_adapter.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace nuka::import {

namespace {

std::string Lowercase(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

std::string ExtensionOf(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return {};
    }
    return Lowercase(std::string_view(path).substr(dot));
}

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("USD: failed to load file: " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

bool StartsWith(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

bool LooksLikeTextUsd(std::string_view contents) {
    return StartsWith(contents, "#usda")
        || contents.find("\ndef ") != std::string_view::npos
        || contents.find("\r\ndef ") != std::string_view::npos
        || StartsWith(contents, "def ");
}

const char* FormatLabel(UsdStageFormat format) {
    switch (format) {
    case UsdStageFormat::Usd:
        return ".usd";
    case UsdStageFormat::UsdaText:
        return "USDA";
    case UsdStageFormat::UsdcCrate:
        return "USDC";
    case UsdStageFormat::UsdzPackage:
        return "USDZ";
    case UsdStageFormat::Unknown:
        return "unknown";
    }
    return "unknown";
}

[[noreturn]] void ThrowOpenUsdRequired(UsdStageFormat format, const std::string& path) {
    throw std::runtime_error(
        std::string("USD: ") + FormatLabel(format) +
        " import requires the OpenUSD SDK adapter: " + path);
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

UsdStageData LoadUsdStageData(const std::string& path) {
    const UsdStageFormat format = DetectUsdStageFormat(path);
    if (format == UsdStageFormat::Unknown) {
        throw std::runtime_error("USD: unsupported USD extension for " + path);
    }

    if (format == UsdStageFormat::UsdcCrate || format == UsdStageFormat::UsdzPackage) {
        (void)ReadFile(path);
        ThrowOpenUsdRequired(format, path);
    }

    std::string contents = ReadFile(path);
    if (format == UsdStageFormat::Usd && !LooksLikeTextUsd(contents)) {
        ThrowOpenUsdRequired(format, path);
    }

    UsdStageData data;
    data.format = format;
    data.path = path;
    data.text = std::move(contents);
    return data;
}

} // namespace nuka::import
