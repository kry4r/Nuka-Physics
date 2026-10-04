#pragma once

#include "imgui.h"

namespace nuka::render::imgui::theme {
constexpr ImVec4 Color(unsigned rgb) {
    return {static_cast<float>((rgb >> 16u) & 255u) / 255.0f,
            static_cast<float>((rgb >> 8u) & 255u) / 255.0f,
            static_cast<float>(rgb & 255u) / 255.0f, 1.0f};
}
inline constexpr ImVec4 kBgVoid = Color(0xE7EAE5u);
inline constexpr ImVec4 kBgPanel = Color(0xF5F6F2u);
inline constexpr ImVec4 kTopBar = Color(0xFAFBF8u);
inline constexpr ImVec4 kBgRaised = Color(0xE9EDE5u);
inline constexpr ImVec4 kBgHover = Color(0xDDE5D3u);
inline constexpr ImVec4 kBgActive = Color(0xE0E9CFu);
inline constexpr ImVec4 kLine = Color(0xD2D9CBu);
inline constexpr ImVec4 kText = Color(0x273023u);
inline constexpr ImVec4 kTextDim = Color(0x5F6B58u);
inline constexpr ImVec4 kAccent = Color(0x5F7828u);
inline constexpr ImVec4 kAccentDim = Color(0x526D1Eu);
inline constexpr ImVec4 kWarn = Color(0x806018u);
inline constexpr ImVec4 kError = Color(0xA33029u);
}  // namespace nuka::render::imgui::theme
