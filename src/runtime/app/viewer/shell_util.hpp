#pragma once
// Small helpers the editor shell's translation units share.

#include "imgui.h"
#include "scene/format/json.hpp"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

namespace nuka::runtime::app::viewer::shell {

inline ImU32 U32(const ImVec4& c) { return ImGui::GetColorU32(c); }

template <class... T>
std::string Fmt(const char* format, T... values) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), format, values...);
    return buf;
}

inline nuka::scene::json::Value Args(std::initializer_list<std::pair<const char*, nuka::scene::json::Value>> items) {
    nuka::scene::json::Value args = nuka::scene::json::Value::Object();
    for (const auto& [key, value] : items) args.Set(key, value);
    return args;
}

inline std::string Lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

inline std::string Upper(std::string_view text) {
    std::string out(text);
    for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

inline std::string Iconed(const char* glyph, const char* text) { return std::string(glyph) + "  " + text; }

// Dashes `dash` long with gaps as long from `a` to `b`.
inline void DashedLine(ImDrawList* list, const ImVec2& a, const ImVec2& b, ImU32 color, float dash, float width) {
    const float dx = b.x - a.x, dy = b.y - a.y, length = std::sqrt(dx * dx + dy * dy);
    for (float d = 0.0f; d < length; d += 2.0f * dash) {
        const float e = std::min(d + dash, length);
        list->AddLine(ImVec2(a.x + dx * d / length, a.y + dy * d / length), ImVec2(a.x + dx * e / length, a.y + dy * e / length),
                      color, width);
    }
}

// The whole-pixel text top that centres `font`'s capitals at `px` on the height `mid`.
inline float CapTop(ImFont* font, float px, float mid) {
    const ImFontGlyph* cap = font->GetFontBaked(px)->FindGlyph('N');
    return std::round(mid - (cap->Y0 + cap->Y1) * 0.5f);
}

// `text`, or as much of it as fits in `width` behind or ahead of an ellipsis.
inline std::string Elide(ImFont* font, float px, const std::string& text, float width, bool keep_tail) {
    const auto fits = [&](const std::string& t) { return font->CalcTextSizeA(px, FLT_MAX, 0.0f, t.c_str()).x <= width; };
    if (fits(text)) return text;
    for (size_t n = 1; n < text.size(); ++n) {
        const size_t i = keep_tail ? n : text.size() - n;
        if ((static_cast<unsigned char>(text[i]) & 0xC0u) == 0x80u) continue;
        const std::string t = keep_tail ? "\xE2\x80\xA6" + text.substr(i) : text.substr(0, i) + "\xE2\x80\xA6";
        if (fits(t)) return t;
    }
    return "\xE2\x80\xA6";
}

}  // namespace nuka::runtime::app::viewer::shell
