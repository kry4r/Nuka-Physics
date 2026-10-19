// nui look helpers, the node base, layout containers, panels and the host.

#include "runtime/app/viewer/nui/nui_internal.hpp"

#include "render/imgui/nuka_imgui.hpp"

#include "imgui_internal.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace nuka::nui {

using namespace detail;
using nuka::render::imgui::FontRole;
using nuka::render::imgui::GetNukaFont;

// ---- look ---------------------------------------------------------------------------------

ImVec4 ToneColor(Tone tone) {
    switch (tone) {
        case Tone::Muted: return theme::kFgMuted;
        case Tone::Dim: return theme::kFgDim;
        case Tone::Accent: return theme::kAccent;
        case Tone::Ok: return theme::kOk;
        case Tone::Warn: return theme::kWarn;
        case Tone::Err: return theme::kErr;
        case Tone::Info: return theme::kInfo;
        default: return theme::kFg;
    }
}

Tone ToneOf(std::string_view name) {
    if (name == "muted") return Tone::Muted;
    if (name == "dim") return Tone::Dim;
    if (name == "accent") return Tone::Accent;
    if (name == "ok") return Tone::Ok;
    if (name == "warn") return Tone::Warn;
    if (name == "err" || name == "error") return Tone::Err;
    if (name == "info") return Tone::Info;
    return Tone::Normal;
}

TextStyle TextStyleOf(std::string_view name) {
    if (name == "strong") return TextStyle::Strong;
    if (name == "label") return TextStyle::Label;
    if (name == "mono") return TextStyle::Mono;
    if (name == "hero") return TextStyle::Hero;
    return TextStyle::Body;
}

const char* IconOf(const std::string& name) {
    static const std::unordered_map<std::string, const char*> table = {
        {"play", icon::kPlay}, {"pause", icon::kPause}, {"step", icon::kStep}, {"reset", icon::kReset},
        {"undo", icon::kUndo}, {"redo", icon::kRedo}, {"speed", icon::kSpeed}, {"move", icon::kMove},
        {"rotate", icon::kRotate}, {"world", icon::kWorldSpace}, {"local", icon::kLocalSpace},
        {"snap", icon::kSnap}, {"gizmo", icon::kGizmo}, {"zero_copy", icon::kZeroCopy},
        {"host_copy", icon::kHostCopy}, {"warn", icon::kWarn}, {"error", icon::kError}, {"info", icon::kInfo},
        {"ok", icon::kOk}, {"dot", icon::kDot}, {"open", icon::kOpen}, {"save", icon::kSave},
        {"save_as", icon::kSaveAs}, {"new", icon::kNew}, {"close", icon::kClose}, {"file", icon::kFile},
        {"unload", icon::kUnload}, {"stage", icon::kStage}, {"property", icon::kProperty},
        {"stats", icon::kStats}, {"debug", icon::kDebug}, {"camera", icon::kCamera}, {"drive", icon::kDrive},
        {"script", icon::kScript}, {"console", icon::kConsole}, {"inspector", icon::kInspector},
        {"search", icon::kSearch}, {"settings", icon::kSettings}, {"extensions", icon::kExtensions},
        {"layout", icon::kLayout}, {"help", icon::kHelp}, {"robot", icon::kRobot}, {"soft", icon::kSoft},
        {"cloth", icon::kCloth}, {"fluid", icon::kFluid}, {"body", icon::kBody}, {"static", icon::kStatic},
        {"joint", icon::kJoint}, {"collider", icon::kCollider}, {"visual", icon::kVisual}, {"eye", icon::kVisual},
        {"camera_node", icon::kCameraNode}, {"light", icon::kLight}, {"group", icon::kGroup},
        {"scene", icon::kScene}, {"script_node", icon::kScriptNode}, {"chevron_right", icon::kChevronRight},
        {"chevron_down", icon::kChevronDown}, {"add", icon::kAdd}, {"trash", icon::kTrash},
        {"edit", icon::kEdit}, {"live", icon::kLive}, {"rebuild", icon::kRebuild}, {"run", icon::kRun},
        {"clear", icon::kClear}, {"frame", icon::kFrame}, {"home", icon::kHome}, {"target", icon::kTarget},
        {"box", icon::kBox}, {"sphere", icon::kSphere}, {"capsule", icon::kCapsule}, {"plane", icon::kPlane},
        {"cylinder", icon::kCylinder}, {"chip", icon::kChip}, {"timer", icon::kTimer},
        {"keyboard", icon::kKeyboard}, {"hexagon", icon::kHexagon}, {"command", icon::kCommand},
        {"atom", icon::kAtom}, {"palette", icon::kPalette}, {"lock", icon::kLock}, {"filter", icon::kFilter},
        {"plot", icon::kPlot}, {"waves", icon::kWaves}, {"translate", icon::kTranslate}, {"power", icon::kPower},
        {"prohibit", icon::kProhibit}, {"plug", icon::kPlug}, {"package", icon::kPackage}, {"code", icon::kCode},
        {"file_py", icon::kFilePy}, {"file_cpp", icon::kFileCpp}, {"stop", icon::kStop}, {"repeat", icon::kRepeat},
        {"cursor_text", icon::kCursorText}, {"function", icon::kFunction}, {"terminal", icon::kTerminal}};
    const auto it = table.find(name);
    return it != table.end() ? it->second : name.c_str();
}

float Scale() { return ImGui::GetStyle().FontScaleDpi; }

void PushTextStyle(TextStyle style) {
    switch (style) {
        case TextStyle::Strong: ImGui::PushFont(GetNukaFont(FontRole::Strong), theme::kFontBody); break;
        case TextStyle::Label: ImGui::PushFont(GetNukaFont(FontRole::Display), theme::kFontLabel); break;
        case TextStyle::Mono: ImGui::PushFont(GetNukaFont(FontRole::Mono), theme::kFontMono); break;
        case TextStyle::Hero: ImGui::PushFont(GetNukaFont(FontRole::Strong), theme::kFontHero); break;
        default: ImGui::PushFont(GetNukaFont(FontRole::Body), theme::kFontBody); break;
    }
}

void PopTextStyle() { ImGui::PopFont(); }

namespace detail {

std::string Upper(std::string_view text) {
    std::string out(text);
    for (char& c : out)
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    return out;
}

std::string Lower(std::string_view text) {
    std::string out(text);
    for (char& c : out)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return out;
}

std::string WithIcon(const std::string& glyph, const std::string& text) {
    if (glyph.empty()) return text;
    if (text.empty()) return glyph;
    return glyph + "  " + text;
}

std::string ActionTip(const std::string& action) {
    const ActionHooks& hooks = Host::Current()->actions;
    std::string tip = Tr(hooks.label(action));
    const std::string keys = hooks.shortcut(action);
    if (!keys.empty()) tip += "   " + keys;
    return tip;
}

void RunAction(const std::string& action, const json::Value& args) { Host::Current()->actions.run(action, args); }

bool InGrid(const Node* parent) { return parent != nullptr && std::string_view(parent->Kind()) == "property_grid"; }

// The tier glyph, the text and the gap before the editor.
float FieldLabelWidth(const std::string& caption, Tier tier) {
    const float s = Scale();
    return std::ceil(ImGui::CalcTextSize(Tr(caption).c_str()).x + ((tier != Tier::None ? 14.0f : 0.0f) + theme::kSpace) * s);
}

void FieldLabel(const std::string& caption, Tier tier, bool in_grid) {
    const float s = Scale();
    const std::string& label = Tr(caption);
    if (in_grid) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
    }
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetFrameHeight();
    const float w = in_grid ? ImGui::GetContentRegionAvail().x : ImGui::GetContentRegionAvail().x * theme::kLabelColumn;
    ImGui::Dummy(ImVec2(w, h));
    ImDrawList* list = ImGui::GetWindowDrawList();
    float x = pos.x;
    if (tier != Tier::None) {
        // Live parameters apply next tick; rebuild ones re-cook the world.
        const bool live = tier == Tier::Live;
        draw::Glyph(list, ImVec2(x + 6.0f * s, pos.y + h * 0.5f), live ? icon::kLive : icon::kRebuild,
                    U32(live ? theme::kAccentDim : theme::kWarn), live, 12.0f);
        x += 14.0f * s;
    }
    const float right = pos.x + w - theme::kSpace * s;
    ImGui::PushStyleColor(ImGuiCol_Text, theme::kFgMuted);
    ImGui::RenderTextEllipsis(list, ImVec2(x, pos.y + ImGui::GetStyle().FramePadding.y), ImVec2(right, pos.y + h), right, label.c_str(),
                              nullptr, nullptr);
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip) && ImGui::CalcTextSize(label.c_str()).x > w - (x - pos.x))
        ImGui::SetTooltip("%s", label.c_str());
    if (in_grid) ImGui::TableSetColumnIndex(1);
    else ImGui::SameLine(0.0f, 0.0f);
}

}  // namespace detail

// ---- draw ---------------------------------------------------------------------------------

namespace draw {

void Chamfer(ImDrawList* list, const ImVec2& min, const ImVec2& max, float cut, ImU32 fill, ImU32 border) {
    const ImVec2 points[6] = {ImVec2(min.x + cut, min.y), ImVec2(max.x, min.y), ImVec2(max.x, max.y - cut),
                              ImVec2(max.x - cut, max.y), ImVec2(min.x, max.y), ImVec2(min.x, min.y + cut)};
    if ((fill & IM_COL32_A_MASK) != 0u) list->AddConvexPolyFilled(points, 6, fill);
    if ((border & IM_COL32_A_MASK) != 0u) list->AddPolyline(points, 6, border, ImDrawFlags_Closed, 1.0f);
}

void Glyph(ImDrawList* list, const ImVec2& center, const char* glyph, ImU32 color, bool filled, float size) {
    const float s = Scale();
    ImFont* font = filled ? GetNukaFont(FontRole::IconFill) : GetNukaFont(FontRole::Body);
    const float unscaled = size > 0.0f ? size : theme::kFontIcon;
    // The regular icons are merged into the body face at kFontIcon/kFontBody of its size.
    const float px = (filled ? unscaled : unscaled * theme::kFontBody / theme::kFontIcon) * s;
    unsigned int codepoint = 0u;
    ImTextCharFromUtf8(&codepoint, glyph, nullptr);
    const ImFontGlyph* g = font->GetFontBaked(px)->FindGlyph(static_cast<ImWchar>(codepoint));
    const ImVec2 pos(std::round(center.x - (g->X0 + g->X1) * 0.5f), std::round(center.y - (g->Y0 + g->Y1) * 0.5f));
    list->AddText(font, px, pos, color, glyph);
}

void SectionHeader(const char* label, const char* trailing) {
    const float s = Scale();
    ImGui::Dummy(ImVec2(0.0f, 0.5f * theme::kSpace * s));
    const ImVec2 start = ImGui::GetCursorScreenPos();
    const float right = start.x + ImGui::GetContentRegionAvail().x;
    PushTextStyle(TextStyle::Label);
    const std::string text = Upper(label);
    const ImVec2 size = ImGui::CalcTextSize(text.c_str());
    const float tick = 6.0f * s;
    ImGui::Dummy(ImVec2(right - start.x, size.y + theme::kSpace * s));
    ImDrawList* list = ImGui::GetWindowDrawList();
    const float mid = start.y + (size.y + theme::kSpace * s) * 0.5f;
    list->AddRectFilled(ImVec2(start.x, mid - 1.5f * s), ImVec2(start.x + 3.0f * s, mid + 1.5f * s), U32(theme::kAccent));
    list->AddText(ImVec2(start.x + tick + 2.0f * s, mid - size.y * 0.5f), U32(theme::kFgMuted), text.c_str());
    float line_end = right;
    if (trailing != nullptr && *trailing != '\0') {
        const ImVec2 tsize = ImGui::CalcTextSize(trailing);
        list->AddText(ImVec2(right - tsize.x, mid - tsize.y * 0.5f), U32(theme::kFgDim), trailing);
        line_end = right - tsize.x - 2.0f * theme::kSpace * s;
    }
    const float line_start = start.x + tick + size.x + 2.0f * theme::kSpace * s;
    if (line_end > line_start) list->AddLine(ImVec2(line_start, mid), ImVec2(line_end, mid), U32(theme::kLine));
    PopTextStyle();
}

void Badge(const char* text, Tone tone, const char* glyph) {
    const float s = Scale();
    const float h = ImGui::GetFrameHeight();
    PushTextStyle(TextStyle::Label);
    const std::string content = glyph != nullptr && *glyph != '\0' ? std::string(glyph) + " " + Upper(text) : Upper(text);
    const ImVec2 size = ImGui::CalcTextSize(content.c_str());
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float w = size.x + 12.0f * s;
    const ImVec4 c = ToneColor(tone);
    const ImVec2 a(pos.x, pos.y + 3.0f * s), b(pos.x + w, pos.y + h - 3.0f * s);
    draw::Chamfer(ImGui::GetWindowDrawList(), a, b, 4.0f * s, U32(theme::Alpha(c, 0.12f)), U32(theme::Alpha(c, 0.45f)));
    ImGui::GetWindowDrawList()->AddText(ImVec2(pos.x + 6.0f * s, pos.y + (h - size.y) * 0.5f), U32(c), content.c_str());
    ImGui::Dummy(ImVec2(w, h));
    PopTextStyle();
}

void StatusDot(Tone tone, bool pulse) {
    const float s = Scale();
    const float h = ImGui::GetFrameHeight();
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 center(pos.x + 6.0f * s, pos.y + h * 0.5f);
    const ImVec4 c = ToneColor(tone);
    ImDrawList* list = ImGui::GetWindowDrawList();
    if (pulse) {
        const float t = static_cast<float>(std::fmod(ImGui::GetTime(), 1.6) / 1.6);
        list->AddCircleFilled(center, (3.5f + 4.0f * t) * s, U32(theme::Alpha(c, 0.35f * (1.0f - t))));
    }
    list->AddCircleFilled(center, 3.5f * s, U32(c));
    ImGui::Dummy(ImVec2(12.0f * s, h));
}

bool IconButton(const char* id, const char* glyph, bool checked, const char* tooltip, Tone tone) {
    const float s = Scale();
    const float h = ImGui::GetFrameHeight();
    const bool pressed = ImGui::InvisibleButton(id, ImVec2(h + 2.0f * s, h));
    const bool hovered = ImGui::IsItemHovered();
    const bool held = ImGui::IsItemActive();
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    ImDrawList* list = ImGui::GetWindowDrawList();
    const float r = theme::kRadius * s;
    if (checked) {
        list->AddRectFilled(a, b, U32(theme::Alpha(theme::kAccent, held ? 0.26f : 0.14f)), r);
        list->AddLine(ImVec2(a.x + 5.0f * s, b.y - 1.0f * s), ImVec2(b.x - 5.0f * s, b.y - 1.0f * s), U32(theme::kAccent),
                      2.0f * s);
    } else if (held) {
        list->AddRectFilled(a, b, U32(theme::kBg3), r);
    } else if (hovered) {
        list->AddRectFilled(a, b, U32(theme::kBg2), r);
    }
    const ImVec4 color = checked ? theme::kAccent
                         : tone != Tone::Normal ? ToneColor(tone)
                         : hovered ? theme::kFg
                                   : theme::kFgMuted;
    Glyph(list, ImVec2((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f), glyph, U32(color), checked);
    if (tooltip != nullptr && *tooltip != '\0' && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", tooltip);
    return pressed;
}

void ThinProgress(float fraction, float width, Tone tone) {
    const float s = Scale();
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float h = 3.0f * s;
    ImDrawList* list = ImGui::GetWindowDrawList();
    list->AddRectFilled(pos, ImVec2(pos.x + width, pos.y + h), U32(theme::kBg3));
    const ImU32 fill = U32(ToneColor(tone));
    if (fraction < 0.0f) {
        const float t = static_cast<float>(std::fmod(ImGui::GetTime(), 1.2) / 1.2);
        const float x0 = pos.x + (width * 1.3f) * t - width * 0.3f;
        list->AddRectFilled(ImVec2(std::max(pos.x, x0), pos.y), ImVec2(std::min(pos.x + width, x0 + width * 0.3f), pos.y + h), fill);
    } else {
        list->AddRectFilled(pos, ImVec2(pos.x + width * std::clamp(fraction, 0.0f, 1.0f), pos.y + h), fill);
    }
    ImGui::Dummy(ImVec2(width, h));
}

void ViewportFrame(const ImVec2& min, const ImVec2& max, const std::vector<std::string>& hud, float hud_top) {
    const float s = Scale();
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    const float inset = 8.0f * s, arm = 14.0f * s, t = 1.5f * s;
    const ImU32 c = U32(theme::Alpha(theme::kAccent, 0.55f));
    const ImVec2 a(min.x + inset, min.y + inset), b(max.x - inset, max.y - inset);
    list->AddLine(a, ImVec2(a.x + arm, a.y), c, t);
    list->AddLine(a, ImVec2(a.x, a.y + arm), c, t);
    list->AddLine(ImVec2(b.x, a.y), ImVec2(b.x - arm, a.y), c, t);
    list->AddLine(ImVec2(b.x, a.y), ImVec2(b.x, a.y + arm), c, t);
    list->AddLine(ImVec2(a.x, b.y), ImVec2(a.x + arm, b.y), c, t);
    list->AddLine(ImVec2(a.x, b.y), ImVec2(a.x, b.y - arm), c, t);
    list->AddLine(b, ImVec2(b.x - arm, b.y), c, t);
    list->AddLine(b, ImVec2(b.x, b.y - arm), c, t);
    ImFont* font = GetNukaFont(FontRole::Display);
    const float px = theme::kFontLabel * s;
    float x = a.x + 10.0f * s;
    const float y = a.y + 8.0f * s + hud_top;
    for (const std::string& chip : hud) {
        const ImVec2 size = font->CalcTextSizeA(px, FLT_MAX, 0.0f, chip.c_str());
        const ImVec2 c0(x, y), c1(x + size.x + 12.0f * s, y + size.y + 8.0f * s);
        Chamfer(list, c0, c1, 4.0f * s, U32(theme::Alpha(theme::kBg0, 0.78f)), U32(theme::Alpha(theme::kLineHi, 0.9f)));
        list->AddText(font, px, ImVec2(x + 6.0f * s, y + 4.0f * s), U32(theme::kFgMuted), chip.c_str());
        x = c1.x + 6.0f * s;
    }
}

bool Chip(const char* id, const std::string& text, bool on) {
    const float s = Scale();
    const ImVec2 size = ImGui::CalcTextSize(text.c_str());
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const ImVec2 box(size.x + 16.0f * s, ImGui::GetFrameHeight() - 2.0f * s);
    ImGui::PushID(id);
    const bool clicked = ImGui::InvisibleButton("##chip", box);
    ImGui::PopID();
    const bool hot = ImGui::IsItemHovered();
    ImDrawList* list = ImGui::GetWindowDrawList();
    Chamfer(list, p, ImVec2(p.x + box.x, p.y + box.y), 4.0f * s,
            U32(on ? theme::Alpha(theme::kAccent, 0.16f) : hot ? theme::kBg2 : theme::kBg1),
            U32(on ? theme::kAccent : hot ? theme::kLineHi : theme::kLine));
    list->AddText(ImVec2(p.x + 8.0f * s, p.y + (box.y - size.y) * 0.5f), U32(on ? theme::kAccent : hot ? theme::kFg : theme::kFgMuted),
                  text.c_str());
    return clicked;
}

}  // namespace draw

// ---- items --------------------------------------------------------------------------------

const char* ItemModel::Icon(uint64_t) const { return ""; }
ImVec4 ItemModel::IconColor(uint64_t) const { return theme::kFgMuted; }
std::string_view ItemModel::Badge(uint64_t) const { return {}; }

// ---- node ---------------------------------------------------------------------------------

void Node::Draw() {
    if (!Shown()) return;
    const bool disabled = !enabled || (enabled_if && !enabled_if());
    if (disabled) ImGui::BeginDisabled();
    if (id.empty()) ImGui::PushID(this);
    else ImGui::PushID(id.c_str());
    DrawSelf();
    ImGui::PopID();
    rect_min = ImGui::GetItemRectMin();
    rect_max = ImGui::GetItemRectMax();
    if (!tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", Tr(tooltip).c_str());
    if (disabled) ImGui::EndDisabled();
    ++draws;
}

Node& Node::Adopt(std::unique_ptr<Node> child) {
    child->parent_ = this;
    children_.push_back(std::move(child));
    return *children_.back();
}

std::unique_ptr<Node> Node::Detach(Node* child) {
    const auto it = std::find_if(children_.begin(), children_.end(), [child](const auto& c) { return c.get() == child; });
    std::unique_ptr<Node> out = std::move(*it);
    children_.erase(it);
    out->parent_ = nullptr;
    return out;
}

void Node::SetProps(const json::Value& props) {
    Read(props, "id", id);
    Read(props, "width", width);
    Read(props, "tooltip", tooltip);
    Read(props, "visible", visible);
    Read(props, "enabled", enabled);
}

void Node::Emit(const char* type, json::Value value) {
    if (on_event) on_event(type, value);
}

void Node::DrawChildren() {
    for (size_t i = 0; i < children_.size(); ++i) children_[i]->Draw();
}

float Node::ItemWidth(float fallback) const {
    if (assigned_width > 0.0f) return assigned_width;
    if (width < 0.0f) return -FLT_MIN;
    if (width > 0.0f) return width * Scale();
    return fallback;
}

// ---- layout -------------------------------------------------------------------------------

void VStack::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "spacing", spacing);
}

void VStack::DrawSelf() {
    ImGui::BeginGroup();
    if (spacing >= 0.0f)
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, spacing * Scale()));
    DrawChildren();
    if (spacing >= 0.0f) ImGui::PopStyleVar();
    ImGui::EndGroup();
}

void HStack::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "spacing", spacing);
    Read(props, "wrap", wrap);
}

// Fill children share what the others left over last frame.
void HStack::DrawSelf() {
    const float gap = spacing >= 0.0f ? spacing * Scale() : ImGui::GetStyle().ItemSpacing.x;
    float fixed = 0.0f;
    int fills = 0, count = 0;
    for (const auto& child : children_) {
        if (!child->Shown()) continue;
        ++count;
        if (child->width < 0.0f) ++fills;
        else fixed += child->rect_max.x - child->rect_min.x;
    }
    const float share = fills > 0
        ? std::max(1.0f, (ImGui::GetContentRegionAvail().x - fixed - gap * static_cast<float>(count - 1)) / static_cast<float>(fills))
        : 0.0f;
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    ImGui::BeginGroup();
    ImGui::AlignTextToFramePadding();
    bool first = true;
    float end = 0.0f;
    for (const auto& child : children_) {
        if (!child->Shown()) continue;
        const float w = child->rect_max.x - child->rect_min.x;
        if (!first && !(wrap && child->width >= 0.0f && end + gap + w > right)) ImGui::SameLine(0.0f, gap);
        first = false;
        child->assigned_width = child->width < 0.0f ? share : 0.0f;
        child->Draw();
        end = child->rect_max.x;
    }
    ImGui::EndGroup();
}

void ZStack::DrawSelf() {
    const ImVec2 start = ImGui::GetCursorPos();
    float bottom = start.y;
    for (const auto& child : children_) {
        if (!child->Shown()) continue;
        ImGui::SetCursorPos(start);
        child->Draw();
        bottom = std::max(bottom, ImGui::GetCursorPosY());
    }
    ImGui::SetCursorPos(ImVec2(start.x, bottom));
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
}

void Grid::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "columns", columns);
}

void Grid::DrawSelf() {
    if (!ImGui::BeginTable("##grid", std::max(1, columns), ImGuiTableFlags_SizingStretchSame)) return;
    for (const auto& child : children_) {
        if (!child->Shown()) continue;
        ImGui::TableNextColumn();
        child->Draw();
    }
    ImGui::EndTable();
}

void Spacer::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "size", size);
}

void Spacer::DrawSelf() {
    const bool row = parent_ != nullptr && std::string_view(parent_->Kind()) == "hstack";
    if (row) ImGui::Dummy(ImVec2(assigned_width > 0.0f ? assigned_width : size * Scale(), 1.0f));
    else ImGui::Dummy(ImVec2(0.0f, size * Scale()));
}

void Separator::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "label", label);
}

void Separator::DrawSelf() {
    if (label.empty()) ImGui::Separator();
    else draw::SectionHeader(Tr(label).c_str());
}

void Collapsible::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "title", title);
    ReadIcon(props, "icon", icon);
    Read(props, "trailing", trailing);
    Read(props, "open", open);
}

void Collapsible::DrawSelf() {
    const float s = Scale();
    const float h = ImGui::GetFrameHeight();
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    if (ImGui::InvisibleButton("##header", ImVec2(w, h))) {
        open = !open;
        Emit("toggle", json::Value::Bool(open));
    }
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* list = ImGui::GetWindowDrawList();
    const ImVec2 end(pos.x + w, pos.y + h);
    list->AddRectFilled(pos, end, U32(hovered ? theme::kBg2 : theme::Alpha(theme::kBg2, 0.55f)), theme::kRadius * s);
    if (open) list->AddRectFilled(pos, ImVec2(pos.x + 2.0f * s, end.y), U32(theme::kAccent));
    const float mid = pos.y + h * 0.5f;
    draw::Glyph(list, ImVec2(pos.x + 11.0f * s, mid), open ? icon::kChevronDown : icon::kChevronRight,
                U32(theme::kFgDim), false, 12.0f);
    float x = pos.x + 22.0f * s;
    if (!icon.empty()) {
        draw::Glyph(list, ImVec2(x + 7.0f * s, mid), icon.c_str(), U32(open ? theme::kAccent : theme::kFgMuted), false, 14.0f);
        x += 20.0f * s;
    }
    PushTextStyle(TextStyle::Label);
    const std::string text = Upper(Tr(title));
    const ImVec2 size = ImGui::CalcTextSize(text.c_str());
    list->AddText(ImVec2(x, mid - size.y * 0.5f), U32(hovered || open ? theme::kFg : theme::kFgMuted), text.c_str());
    if (!trailing.empty()) {
        const ImVec2 tsize = ImGui::CalcTextSize(trailing.c_str());
        list->AddText(ImVec2(end.x - tsize.x - 8.0f * s, mid - tsize.y * 0.5f), U32(theme::kFgDim), trailing.c_str());
    }
    PopTextStyle();
    if (!open) return;
    ImGui::Dummy(ImVec2(0.0f, 0.5f * theme::kSpace * s));
    ImGui::BeginGroup();
    DrawChildren();
    ImGui::EndGroup();
    ImGui::Dummy(ImVec2(0.0f, theme::kSpace * s));
}

void Tabs::DrawSelf() {
    if (!ImGui::BeginTabBar("##tabs")) return;
    DrawChildren();
    ImGui::EndTabBar();
}

void Tab::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "title", title);
}

void Tab::DrawSelf() {
    const std::string name = Tr(title) + "###" + title;
    if (!ImGui::BeginTabItem(name.c_str())) return;
    DrawChildren();
    ImGui::EndTabItem();
}

void ScrollFrame::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "height", height);
}

void ScrollFrame::DrawSelf() {
    if (ImGui::BeginChild("##scroll", ImVec2(0.0f, height > 0.0f ? height * Scale() : 0.0f))) DrawChildren();
    ImGui::EndChild();
}

// ---- panels -------------------------------------------------------------------------------

Panel::Panel(std::string panel_id, std::string caption, const char* glyph)
    : id(std::move(panel_id)), title(std::move(caption)), icon(glyph != nullptr ? glyph : ""),
      root_(std::make_unique<VStack>()) {}

std::string Panel::WindowName() const { return WithIcon(icon, Tr(title)) + "###" + id; }

void Panel::Draw(Host&) {
    if (!open) return;
    if (focus_request) {
        ImGui::SetNextWindowFocus();
        focus_request = false;
    }
    const std::string name = WindowName();
    if (ImGui::Begin(name.c_str(), &open, flags | ImGuiWindowFlags_NoCollapse)) {
        const uint64_t k = key ? key() : 0u;
        if (build && (!built_ || k != key_)) {
            root_->Clear();
            build(*root_);
            key_ = k;
            built_ = true;
            ++rebuilds;
        }
        root_->Draw();
        ++draws;
    }
    ImGui::End();
}

// ---- host ---------------------------------------------------------------------------------

namespace {

Host* current_host = nullptr;

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char* LevelIcon(Level level) {
    switch (level) {
        case Level::Ok: return icon::kOk;
        case Level::Warn: return icon::kWarn;
        case Level::Err: return icon::kError;
        default: return icon::kInfo;
    }
}

Tone LevelTone(Level level) {
    switch (level) {
        case Level::Ok: return Tone::Ok;
        case Level::Warn: return Tone::Warn;
        case Level::Err: return Tone::Err;
        default: return Tone::Info;
    }
}

}  // namespace

Host* Host::Current() { return current_host; }

Host::Host() {
    current_host = this;
    FillApi(api_, this);
}

Host::~Host() {
    if (current_host == this) current_host = nullptr;
}

Panel& Host::AddPanel(std::string panel_id, std::string panel_title, const char* glyph) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (Panel* existing = FindPanel(panel_id)) return *existing;
    panels_.push_back(std::make_unique<Panel>(std::move(panel_id), std::move(panel_title), glyph));
    return *panels_.back();
}

Panel* Host::FindPanel(std::string_view panel_id) {
    for (const auto& panel : panels_)
        if (panel->id == panel_id) return panel.get();
    return nullptr;
}

void Host::RemovePanel(std::string_view panel_id) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    const auto it = std::find_if(panels_.begin(), panels_.end(), [&](const auto& p) { return p->id == panel_id; });
    if (it == panels_.end()) return;
    Forget((*it)->Root());
    for (auto root = panel_roots_.begin(); root != panel_roots_.end();) {
        if (root->second == it->get()) root = panel_roots_.erase(root);
        else ++root;
    }
    panels_.erase(it);
}

void Host::DrawPanels(const std::function<ImGuiID(const Panel&)>& dock_node) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    for (size_t i = 0; i < panels_.size(); ++i) {
        Panel& panel = *panels_[i];
        if (!panel.open) continue;
        const ImGuiID node = dock_node(panel);
        if (node != 0u) ImGui::SetNextWindowDockID(node, ImGuiCond_FirstUseEver);
        panel.Draw(*this);
    }
}

void Host::Toast(Level level, std::string text) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    toasts_.push_back(ToastEntry{level, std::move(text), Now()});
    while (toasts_.size() > 6u) toasts_.pop_front();
}

void Host::Progress(const std::string& progress_id, const std::string& label, float fraction) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    const auto it = std::find_if(progress_.begin(), progress_.end(), [&](const auto& p) { return p.id == progress_id; });
    if (fraction < 0.0f) {
        if (it != progress_.end()) progress_.erase(it);
    } else if (it != progress_.end()) {
        it->label = label;
        it->fraction = fraction;
    } else {
        progress_.push_back(ProgressEntry{progress_id, label, fraction});
    }
}

// Newest at the bottom.
void Host::DrawNotifications(ImDrawList* list, const ImVec2& min, const ImVec2& max) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    const double now = Now();
    while (!toasts_.empty()) {
        const ToastEntry& front = toasts_.front();
        if (now - front.born < (front.level == Level::Err ? 8.0 : 4.0)) break;
        toasts_.pop_front();
    }
    const float s = Scale();
    ImFont* font = GetNukaFont(FontRole::Body);
    const float px = theme::kFontBody * s;
    const float width = 340.0f * s, pad = 10.0f * s;
    float bottom = max.y - 16.0f * s;
    const float right = max.x - 16.0f * s;
    for (auto it = toasts_.rbegin(); it != toasts_.rend(); ++it) {
        const double age = now - it->born;
        const double life = it->level == Level::Err ? 8.0 : 4.0;
        // Toasts fade in and out in place, so a quick run of them leaves the stack still.
        const float fade = static_cast<float>(std::clamp(std::min(age / 0.15, (life - age) / 0.4), 0.0, 1.0));
        const float text_w = width - 2.0f * pad - 22.0f * s;
        const ImVec2 tsize = font->CalcTextSizeA(px, FLT_MAX, text_w, it->text.c_str());
        const float h = std::max(tsize.y, px) + 2.0f * pad;
        const float x0 = right - width;
        const ImVec2 a(x0, bottom - h), b(x0 + width, bottom);
        if (a.y < min.y) break;
        const ImVec4 tone = ToneColor(LevelTone(it->level));
        draw::Chamfer(list, a, b, theme::kChamfer * s, U32(theme::Alpha(theme::kBg0, 0.94f * fade)),
                      U32(theme::Alpha(theme::kLineHi, fade)));
        list->AddRectFilled(ImVec2(a.x, a.y + theme::kChamfer * s), ImVec2(a.x + 2.0f * s, b.y), U32(theme::Alpha(tone, fade)));
        draw::Glyph(list, ImVec2(a.x + pad + 8.0f * s, a.y + pad + px * 0.5f), LevelIcon(it->level), U32(theme::Alpha(tone, fade)));
        list->AddText(font, px, ImVec2(a.x + pad + 22.0f * s, a.y + pad), U32(theme::Alpha(theme::kFg, fade)), it->text.c_str(),
                      nullptr, text_w);
        bottom = a.y - 8.0f * s;
    }
    for (const ProgressEntry& p : progress_) {
        const float h = px + 2.0f * pad + 6.0f * s;
        const ImVec2 a(right - width, bottom - h), b(right, bottom);
        if (a.y < min.y) break;
        draw::Chamfer(list, a, b, theme::kChamfer * s, U32(theme::Alpha(theme::kBg0, 0.94f)), U32(theme::kLineHi));
        list->AddText(font, px, ImVec2(a.x + pad, a.y + pad), U32(theme::kFg), p.label.c_str());
        char pct[16];
        std::snprintf(pct, sizeof(pct), "%d%%", static_cast<int>(std::round(p.fraction * 100.0f)));
        const ImVec2 psize = font->CalcTextSizeA(px, FLT_MAX, 0.0f, pct);
        list->AddText(font, px, ImVec2(b.x - pad - psize.x, a.y + pad), U32(theme::kFgMuted), pct);
        const float y = b.y - pad;
        list->AddRectFilled(ImVec2(a.x + pad, y), ImVec2(b.x - pad, y + 3.0f * s), U32(theme::kBg3));
        list->AddRectFilled(ImVec2(a.x + pad, y), ImVec2(a.x + pad + (b.x - a.x - 2.0f * pad) * std::clamp(p.fraction, 0.0f, 1.0f), y + 3.0f * s),
                            U32(theme::kAccent));
        bottom = a.y - 8.0f * s;
    }
}

void Host::DispatchEvents() {
    ApplyStrings();
    std::vector<Event> pending;
    {
        std::lock_guard<std::recursive_mutex> lock(mu_);
        pending.swap(events_);
    }
    // A node removed since its event queued, with the extension that listened, gets nothing.
    for (const Event& e : pending) {
        {
            std::lock_guard<std::recursive_mutex> lock(mu_);
            if (Lookup(e.node) == nullptr) continue;
        }
        e.fn(e.user, e.node, e.json.c_str());
    }
}

uint32_t Host::Register(Node& node) {
    node.handle = next_handle_++;
    nodes_[node.handle] = &node;
    return node.handle;
}

void Host::Forget(Node& node) {
    if (node.handle != 0u) nodes_.erase(node.handle);
    for (const auto& child : node.Children()) Forget(*child);
}

Node* Host::Lookup(uint32_t node_handle) {
    const auto it = nodes_.find(node_handle);
    return it != nodes_.end() ? it->second : nullptr;
}

}  // namespace nuka::nui
