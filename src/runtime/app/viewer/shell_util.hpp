#pragma once
// Small helpers the editor shell's translation units share.

#include "imgui.h"
#include "scene/format/json.hpp"

#include <cctype>
#include <cfloat>
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
