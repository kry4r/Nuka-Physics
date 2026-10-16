#pragma once
// Shared by the nui translation units: prop readers and small look helpers.

#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/nui/nui.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace nuka::nui::detail {

namespace theme = nuka::render::imgui::theme;
namespace icon = nuka::render::imgui::icon;

inline ImU32 U32(const ImVec4& color) { return ImGui::GetColorU32(color); }

inline void Read(const json::Value& props, const char* key, std::string& out) {
    if (const json::Value* v = props.Find(key)) out = v->AsString();
}
inline void Read(const json::Value& props, const char* key, float& out) {
    if (const json::Value* v = props.Find(key)) out = v->AsFloat();
}
inline void Read(const json::Value& props, const char* key, int& out) {
    if (const json::Value* v = props.Find(key)) out = static_cast<int>(v->AsInt());
}
inline void Read(const json::Value& props, const char* key, bool& out) {
    if (const json::Value* v = props.Find(key)) out = v->AsBool();
}
inline void Read(const json::Value& props, const char* key, Tone& out) {
    if (const json::Value* v = props.Find(key)) out = ToneOf(v->AsString());
}
inline void Read(const json::Value& props, const char* key, TextStyle& out) {
    if (const json::Value* v = props.Find(key)) out = TextStyleOf(v->AsString());
}
inline void Read(const json::Value& props, const char* key, std::vector<std::string>& out) {
    if (const json::Value* v = props.Find(key)) {
        out.clear();
        for (const json::Value& e : v->Elements()) out.push_back(e.AsString());
    }
}
inline void ReadIcon(const json::Value& props, const char* key, std::string& out) {
    if (const json::Value* v = props.Find(key)) out = IconOf(v->AsString());
}

// ASCII upper case; other bytes (CJK) pass through.
std::string Upper(std::string_view text);
std::string Lower(std::string_view text);
// "icon  text", or the text alone without an icon.
std::string WithIcon(const std::string& glyph, const std::string& text);
// The action's label and shortcut as a tooltip line.
std::string ActionTip(const std::string& action);
void RunAction(const std::string& action, const json::Value& args);
// The 40% label cell of a field row; inside a PropertyGrid it is the first table column.
void FieldLabel(const std::string& label, Tier tier, bool in_grid);
bool InGrid(const Node* parent);
// The C function table over `host`.
void FillApi(nuka_nui_api_t& api, Host* host);

}  // namespace nuka::nui::detail
