// The shell's scene-building UI: the Assets browser and its viewport drop, the import wizard,
// the collision dialog and the schema groups of the Properties panel.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/shell_util.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace nuka::runtime::app::viewer {

namespace json = nuka::scene::json;
namespace theme = nuka::render::imgui::theme;
namespace icon = nuka::render::imgui::icon;
namespace draw = nuka::nui::draw;
using nuka::nui::TextStyle;
using nuka::nui::Tone;
using nuka::nui::Tr;
using nuka::render::imgui::FontRole;
using nuka::render::imgui::GetNukaFont;
using namespace shell;

namespace {

constexpr const char* kAssetPayload = "NUKA_ASSET";

struct KindSpec {
    const char* kind;
    const char* label;
    const char* glyph;
};
// The last kind is a file no importer reads, which the Assets panel never lists.
const KindSpec kKinds[] = {{"scene", "Scene", icon::kScene},  {"urdf", "URDF", icon::kRobot},
                           {"mjcf", "MJCF", icon::kRobot},    {"usd", "USD", icon::kPackage},
                           {"mesh", "Mesh", icon::kMeshShape}, {"unknown", "File", icon::kFile}};

const KindSpec& KindOf(const std::string& kind) {
    return *std::find_if(std::begin(kKinds), std::end(kKinds), [&](const KindSpec& k) { return kind == k.kind; });
}

const char* GroupGlyph(const std::string& kind) {
    if (kind == "body") return icon::kBody;
    if (kind == "joint") return icon::kJoint;
    if (kind == "drive") return icon::kDrive;
    if (kind == "collision") return icon::kCollider;
    return icon::kSettings;
}

std::string Leaf(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1u);
}

// The import op's pose: standing upright at `at`.
json::Value GroundPose(const math::Vec3& at) {
    json::Value pos = json::Value::Array();
    for (const float v : {at.x, at.y, at.z}) pos.PushBack(json::Value::Float(v));
    json::Value quat = json::Value::Array();
    for (const float v : {1.0f, 0.0f, 0.0f, 0.0f}) quat.PushBack(json::Value::Float(v));
    return Args({{"pos", pos}, {"quat", quat}});
}

// A small toggle chip; true when clicked.
bool Chip(const char* id, const std::string& text, bool on) {
    const float s = nui::Scale();
    const ImVec2 size = ImGui::CalcTextSize(text.c_str());
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const ImVec2 box(size.x + 16.0f * s, ImGui::GetFrameHeight() - 2.0f * s);
    ImGui::PushID(id);
    const bool clicked = ImGui::InvisibleButton("##chip", box);
    ImGui::PopID();
    const bool hot = ImGui::IsItemHovered();
    ImDrawList* list = ImGui::GetWindowDrawList();
    draw::Chamfer(list, p, ImVec2(p.x + box.x, p.y + box.y), 4.0f * s,
                  U32(on ? theme::Alpha(theme::kAccent, 0.16f) : hot ? theme::kBg2 : theme::kBg1),
                  U32(on ? theme::kAccent : hot ? theme::kLineHi : theme::kLine));
    list->AddText(ImVec2(p.x + 8.0f * s, p.y + (box.y - size.y) * 0.5f), U32(on ? theme::kAccent : hot ? theme::kFg : theme::kFgMuted),
                  text.c_str());
    return clicked;
}

// A button in the accent fill, the one a dialog's Enter presses; a disabled one drops the fill.
bool PrimaryButton(const char* label, float width, bool enabled) {
    ImGui::BeginDisabled(!enabled);
    ImGui::PushStyleColor(ImGuiCol_Button, enabled ? theme::kAccent : theme::kBg2);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, theme::Alpha(theme::kAccent, 0.85f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, theme::kAccentDim);
    ImGui::PushStyleColor(ImGuiCol_Text, enabled ? theme::kAccentInk : theme::kFgDim);
    const bool pressed = ImGui::Button(label, ImVec2(width, 0.0f)) ||
                         (enabled && (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)));
    ImGui::PopStyleColor(4);
    ImGui::EndDisabled();
    return pressed;
}

// A label column cell, then the value column, of a dialog's option table.
void OptionLabel(const char* label) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(theme::kFgMuted, "%s", Tr(label));
    ImGui::TableSetColumnIndex(1);
}

// A schema field: a bar marks a value that differs from the saved scene or waits for the
// rebuild, and a right click offers revert, copy and paste.
class SchemaField final : public nui::FloatField {
public:
    using nui::FloatField::FloatField;
    ImU32 mark = 0u;
    std::string note;
    std::function<void()> menu;

protected:
    void DrawSelf() override {
        nui::FloatField::DrawSelf();
        const float s = nui::Scale();
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        if (mark != 0u)
            ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(a.x - 4.0f * s, a.y + 3.0f * s), ImVec2(a.x - 2.0f * s, b.y - 3.0f * s), mark);
        if (!note.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) ImGui::SetTooltip("%s", note.c_str());
        if (ImGui::BeginPopupContextItem("##field_menu")) {
            menu();
            ImGui::EndPopup();
        }
    }
};

}  // namespace

// ---- assets -------------------------------------------------------------------------------

void ImGuiLayer::BuildAssets(nui::VStack& root) {
    root.Add<nui::Custom>([this] {
        AssetBrowser& b = frame_.ui->assets;
        b.shown = true;
        const float s = nui::Scale();
        const ImGuiStyle& style = ImGui::GetStyle();
        const float kind_w = 120.0f * s;
        const float buttons = 2.0f * (ImGui::GetFrameHeight() + 2.0f * s + style.ItemSpacing.x);
        ImGui::SetNextItemWidth(std::max(ImGui::GetContentRegionAvail().x - buttons - kind_w - style.ItemSpacing.x, 60.0f * s));
        const std::string hint = Iconed(icon::kSearch, Tr("Search assets"));
        ImGui::InputTextWithHint("##asset_filter", hint.c_str(), b.filter, sizeof(b.filter));
        ImGui::SameLine();
        ImGui::SetNextItemWidth(kind_w);
        const std::string shown_kind = b.kind.empty() ? std::string(Tr("All kinds")) : Iconed(KindOf(b.kind).glyph, Tr(KindOf(b.kind).label));
        if (ImGui::BeginCombo("##asset_kind", shown_kind.c_str())) {
            if (ImGui::Selectable(Tr("All kinds"), b.kind.empty())) b.kind.clear();
            for (auto k = std::begin(kKinds); k + 1 != std::end(kKinds); ++k)
                if (ImGui::Selectable(Iconed(k->glyph, Tr(k->label)).c_str(), b.kind == k->kind)) b.kind = k->kind;
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (draw::IconButton("##add_folder", icon::kFolderPlus, false, Tr("Add Asset Folder..."))) Queue("assets.add_folder");
        ImGui::SameLine();
        if (draw::IconButton("##rescan", icon::kReset, false, Tr("Rescan Assets"))) Queue("assets.rescan");

        // The folders as chips that wrap; a user folder can be dropped from its menu.
        const float row_right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
        if (Chip("all", Tr("All folders"), b.root.empty())) b.root.clear();
        for (size_t i = 0; i < b.roots.size(); ++i) {
            const std::string& folder = b.roots[i];
            const std::string text = Iconed(icon::kFolder, Leaf(folder).c_str());
            ImGui::SameLine(0.0f, 4.0f * s);
            if (ImGui::GetCursorScreenPos().x + ImGui::CalcTextSize(text.c_str()).x + 16.0f * s > row_right) ImGui::NewLine();
            ImGui::PushID(static_cast<int>(i));
            if (Chip("folder", text, b.root == folder)) b.root = folder;
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) ImGui::SetTooltip("%s", folder.c_str());
            const bool user = std::find(b.user_roots.begin(), b.user_roots.end(), folder) != b.user_roots.end();
            if (user && ImGui::BeginPopupContextItem("##folder_menu")) {
                if (ImGui::MenuItemEx(Tr("Remove Folder"), icon::kFolderMinus))
                    Queue("assets.remove_folder", Args({{"path", json::Value::Str(folder)}}));
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }

        std::vector<size_t> shown;
        const std::string needle = Lower(b.filter);
        for (size_t i = 0; i < b.cards.size(); ++i) {
            const AssetCard& c = b.cards[i];
            if ((!b.root.empty() && c.root != b.root) || (!b.kind.empty() && c.kind != b.kind) ||
                (!needle.empty() && Lower(c.name).find(needle) == std::string::npos))
                continue;
            shown.push_back(i);
        }
        if (shown.empty()) {
            ImGui::TextColored(theme::kFgDim, "%s",
                               Tr(b.cards.empty() ? "No importable files yet. Add a folder with the folder button."
                                                  : "No asset matches the filter."));
            return;
        }

        const float gap = 10.0f * s, inset = 3.0f * s;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap, gap));
        ImGui::BeginChild("##asset_grid", ImVec2(0.0f, 0.0f));
        ImDrawList* list = ImGui::GetWindowDrawList();
        ImFont* body = ImGui::GetFont();
        const float body_px = ImGui::GetFontSize();
        ImFont* tag_font = GetNukaFont(FontRole::Display);
        const float tag_px = theme::kFontLabel * s;
        const float line = ImGui::GetTextLineHeight();
        const float avail = ImGui::GetContentRegionAvail().x;
        const size_t per_row = static_cast<size_t>(std::max(1.0f, std::floor((avail + gap) / (128.0f * s + gap))));
        const float card_w = (avail - gap * static_cast<float>(per_row - 1u)) / static_cast<float>(per_row);
        const float thumb_h = (card_w - inset * 2.0f) * 9.0f / 16.0f;
        const float card_h = inset + thumb_h + line + 12.0f * s;
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>((shown.size() + per_row - 1u) / per_row), card_h + gap);
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                for (size_t col = 0; col < per_row; ++col) {
                    const size_t k = static_cast<size_t>(row) * per_row + col;
                    if (k >= shown.size()) break;
                    const AssetCard& c = b.cards[shown[k]];
                    const KindSpec& kind = KindOf(c.kind);
                    if (col != 0u) ImGui::SameLine();
                    ImGui::PushID(c.path.c_str());
                    const ImVec2 p = ImGui::GetCursorScreenPos(), q(p.x + card_w, p.y + card_h);
                    ImGui::InvisibleButton("##card", ImVec2(card_w, card_h));
                    const bool hot = ImGui::IsItemHovered();
                    if (hot && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                        Queue("file.import", Args({{"path", json::Value::Str(c.path)}}));
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) ImGui::SetTooltip("%s", c.path.c_str());
                    if (ImGui::BeginDragDropSource()) {
                        ImGui::SetDragDropPayload(kAssetPayload, c.path.data(), c.path.size());
                        ImGui::TextUnformatted(Iconed(kind.glyph, c.name.c_str()).c_str());
                        ImGui::TextColored(theme::kFgDim, "%s", Tr("Drop on the viewport to add it"));
                        ImGui::EndDragDropSource();
                    }
                    if (ImGui::BeginPopupContextItem("##card_menu")) {
                        const auto run_import = [&](const char* mode) {
                            json::Value args = Args({{"path", json::Value::Str(c.path)}});
                            if (mode != nullptr) args.Set("mode", json::Value::Str(mode));
                            Queue("file.import", std::move(args));
                        };
                        if (ImGui::MenuItemEx(Tr("Import..."), icon::kImport)) run_import(nullptr);
                        if (ImGui::MenuItemEx(Tr("Add to Scene"), icon::kAdd, nullptr, false, frame_.ui->has_scene)) run_import("add");
                        if (ImGui::MenuItemEx(Tr("Open as New Scene"), icon::kOpen)) run_import("open");
                        ImGui::Separator();
                        if (ImGui::MenuItemEx(Tr("Copy Path"), icon::kCopy)) ImGui::SetClipboardText(c.path.c_str());
                        ImGui::EndPopup();
                    }
                    draw::Chamfer(list, p, q, theme::kChamfer * s, U32(hot ? theme::kBg2 : theme::kBg1),
                                  U32(hot ? theme::kAccent : theme::kLine));
                    const ImVec2 t0(p.x + inset, p.y + inset), t1(q.x - inset, p.y + inset + thumb_h);
                    if (ImTextureData* tex = c.thumbnail.empty() ? nullptr : Thumbnail(c.thumbnail)) {
                        list->AddImage(tex->GetTexRef(), t0, t1);
                    } else {
                        list->AddRectFilled(t0, t1, U32(theme::kBg0));
                        draw::Glyph(list, ImVec2((t0.x + t1.x) * 0.5f, (t0.y + t1.y) * 0.5f), kind.glyph, U32(theme::kLineHi), false, 26.0f);
                    }
                    const std::string tag = Upper(Tr(kind.label));
                    const ImVec2 ts = tag_font->CalcTextSizeA(tag_px, FLT_MAX, 0.0f, tag.c_str());
                    const ImVec2 g0(t0.x + 4.0f * s, t1.y - ts.y - 8.0f * s), g1(g0.x + ts.x + 10.0f * s, t1.y - 4.0f * s);
                    list->AddRectFilled(g0, g1, U32(theme::Alpha(theme::kBg0, 0.85f)), theme::kRadius * s);
                    list->AddText(tag_font, tag_px, ImVec2(g0.x + 5.0f * s, g0.y + 2.0f * s), U32(hot ? theme::kAccent : theme::kFgMuted),
                                  tag.c_str());
                    list->AddText(body, body_px, ImVec2(p.x + inset + 4.0f * s, t1.y + 6.0f * s), U32(hot ? theme::kFg : theme::kFgMuted),
                                  Elide(body, body_px, c.name, card_w - inset * 2.0f - 8.0f * s, false).c_str());
                    ImGui::PopID();
                }
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
    });
}

void ImGuiLayer::DrawAssetDrop() {
    const ImGuiPayload* payload = ImGui::GetDragDropPayload();
    if (payload == nullptr || !payload->IsDataType(kAssetPayload) || !viewport_.Valid()) return;
    const float s = nui::Scale();
    const ImVec2 vmin(viewport_.x, viewport_.y), vmax(viewport_.x + viewport_.width, viewport_.y + viewport_.height);
    ImGui::SetNextWindowPos(vmin);
    ImGui::SetNextWindowSize(ImVec2(viewport_.width, viewport_.height));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollWithMouse;
    const bool open = ImGui::Begin("##asset_drop", nullptr, flags);
    ImGui::PopStyleVar(2);
    if (open) {
        ImGui::InvisibleButton("##drop", ImVec2(viewport_.width, viewport_.height));
        ImDrawList* list = ImGui::GetWindowDrawList();
        const bool over = ImGui::BeginDragDropTarget();
        draw::Chamfer(list, ImVec2(vmin.x + 6.0f * s, vmin.y + 6.0f * s), ImVec2(vmax.x - 6.0f * s, vmax.y - 6.0f * s), 10.0f * s,
                      U32(theme::Alpha(theme::kAccent, over ? 0.06f : 0.02f)), U32(theme::Alpha(theme::kAccent, over ? 0.8f : 0.35f)));
        if (over) {
            const ImVec2 m = ImGui::GetMousePos();
            const Ray ray = frame_.camera->ScreenRay(viewport_.LocalX(m.x), viewport_.LocalY(m.y), viewport_.pixels.width,
                                                     viewport_.pixels.height);
            math::Vec3 at;
            const bool ground = frame_.camera->RayPlaneHit(ray, math::Vec3{0.0f, 0.0f, 0.0f}, math::Vec3{0.0f, 0.0f, 1.0f}, &at);
            const float r = 14.0f * s;
            list->AddCircle(m, r, U32(theme::kAccent), 32, 1.5f * s);
            list->AddCircleFilled(m, 2.5f * s, U32(theme::kAccent));
            for (const float d : {-1.0f, 1.0f}) {
                list->AddLine(ImVec2(m.x + d * (r - 4.0f * s), m.y), ImVec2(m.x + d * (r + 6.0f * s), m.y), U32(theme::kAccent), 1.5f * s);
                list->AddLine(ImVec2(m.x, m.y + d * (r - 4.0f * s)), ImVec2(m.x, m.y + d * (r + 6.0f * s)), U32(theme::kAccent), 1.5f * s);
            }
            ImFont* mono = GetNukaFont(FontRole::Mono);
            const float mono_px = theme::kFontMono * s;
            const std::string where = ground ? Fmt("X %.2f  Y %.2f", at.x, at.y) : std::string(Tr("AT THE ORIGIN"));
            list->AddText(mono, mono_px, ImVec2(m.x + r + 10.0f * s, m.y + 4.0f * s), U32(theme::kAccent), where.c_str());
            if (const ImGuiPayload* drop = ImGui::AcceptDragDropPayload(kAssetPayload, ImGuiDragDropFlags_AcceptNoDrawDefaultRect)) {
                const std::string path(static_cast<const char*>(drop->Data), static_cast<size_t>(drop->DataSize));
                json::Value args = Args({{"path", json::Value::Str(path)}, {"mode", json::Value::Str("auto")}});
                if (ground) args.Set("pose", GroundPose(at));
                Queue("file.import", std::move(args));
            }
            ImGui::EndDragDropTarget();
        }
    }
    ImGui::End();
}

// ---- import wizard ------------------------------------------------------------------------

void ImGuiLayer::DrawImportWizard() {
    ImportWizard& w = frame_.ui->import;
    const std::string title = std::string(Tr("Import Asset")) + "###import";
    if (w.appear) {
        ImGui::OpenPopup(title.c_str());
        w.appear = false;
    }
    const float s = nui::Scale();
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(620.0f * s, 0.0f), ImVec2(620.0f * s, FLT_MAX));
    if (!ImGui::BeginPopupModal(title.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize)) return;
    if (w.path.empty()) {
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return;
    }
    const json::Value& a = w.analysis;
    const std::string kind = a.At("kind").AsString();
    const std::string profile = a.At("profile").AsString();
    const bool ok = a.At("ok").AsBool();
    const json::Value& stats = a.At("stats");
    const json::Value& diagnostics = a.At("diagnostics");
    size_t errors = 0u, warnings = 0u;
    for (const json::Value& d : diagnostics.Elements()) (d.At("severity").AsString() == "error" ? errors : warnings) += 1u;
    // Analyzing again under the other profile keeps the wizard's name and choices.
    const auto analyze = [&](const char* next) {
        json::Value args = Args({{"path", json::Value::Str(w.path)}, {"profile", json::Value::Str(next)}});
        if (!w.pose.IsNull()) args.Set("pose", w.pose);
        Queue("file.import", std::move(args));
    };

    ImDrawList* list = ImGui::GetWindowDrawList();
    ImFont* mono = GetNukaFont(FontRole::Mono);
    const float mono_px = theme::kFontMono * s;
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float full = ImGui::GetContentRegionAvail().x;
    draw::Glyph(list, ImVec2(at.x + 16.0f * s, at.y + 18.0f * s), KindOf(kind).glyph, U32(ok ? theme::kAccent : theme::kErr), true, 28.0f);
    ImGui::SetCursorScreenPos(ImVec2(at.x + 44.0f * s, at.y));
    ImGui::BeginGroup();
    nui::PushTextStyle(TextStyle::Strong);
    ImGui::TextUnformatted(a.At("name").AsString().c_str());
    nui::PopTextStyle();
    ImGui::SameLine();
    draw::Badge(Upper(Tr(KindOf(kind).label)).c_str(), Tone::Accent);
    list->AddText(mono, mono_px, ImGui::GetCursorScreenPos(), U32(theme::kFgDim),
                  Elide(mono, mono_px, w.path, full - 44.0f * s, true).c_str());
    ImGui::Dummy(ImVec2(full - 44.0f * s, mono_px));
    ImGui::EndGroup();
    ImGui::Dummy(ImVec2(0.0f, 4.0f * s));

    draw::StatusDot(!ok ? Tone::Err : warnings != 0u ? Tone::Warn : Tone::Ok);
    ImGui::SameLine();
    const std::string verdict = !ok              ? Fmt(Tr("Does not import: %zu errors"), errors)
                                : warnings != 0u ? Fmt(Tr("Imports with %zu warnings"), warnings)
                                                 : std::string(Tr("Imports cleanly"));
    ImGui::TextColored(!ok ? theme::kErr : warnings != 0u ? theme::kWarn : theme::kOk, "%s", verdict.c_str());

    if (a.At("loaded").AsBool() && ImGui::BeginTable("##import_stats", 6, ImGuiTableFlags_SizingStretchProp)) {
        for (int c = 0; c < 6; ++c) ImGui::TableSetupColumn(nullptr, ImGuiTableColumnFlags_WidthStretch, c == 5 ? 2.2f : c == 2 ? 1.4f : 1.0f);
        const auto count = [&](const char* key) { return static_cast<long long>(stats.At(key).AsInt()); };
        const json::Value& lo = stats.At("lo");
        const json::Value& hi = stats.At("hi");
        const auto extent = [&](size_t i) { return hi.Elements()[i].AsDouble() - lo.Elements()[i].AsDouble(); };
        const long long meshes = count("mesh_colliders");
        const std::string cells[6][2] = {
            {"Links", Fmt("%lld", count("links"))},
            {"Joints", Fmt("%lld", count("joints"))},
            {"Colliders", meshes != 0 ? Fmt(Tr("%lld (%lld mesh)"), count("colliders"), meshes) : Fmt("%lld", count("colliders"))},
            {"Visuals", Fmt("%lld", count("visual_meshes"))},
            {"Mass", Fmt("%.3g kg", stats.At("mass").AsDouble())},
            {"Size", Fmt("%.2f \xC3\x97 %.2f \xC3\x97 %.2f m", extent(0), extent(1), extent(2))}};
        ImGui::TableNextRow();
        nui::PushTextStyle(TextStyle::Label);
        for (int c = 0; c < 6; ++c) {
            ImGui::TableSetColumnIndex(c);
            ImGui::TextColored(theme::kFgDim, "%s", Upper(Tr(cells[c][0])).c_str());
        }
        nui::PopTextStyle();
        ImGui::TableNextRow();
        for (int c = 0; c < 6; ++c) {
            ImGui::TableSetColumnIndex(c);
            ImGui::TextUnformatted(cells[c][1].c_str());
        }
        ImGui::EndTable();
    }
    ImGui::Dummy(ImVec2(0.0f, 4.0f * s));

    draw::SectionHeader(Tr("Diagnostics"), Fmt("%zu", diagnostics.Size()).c_str());
    if (diagnostics.Size() == 0u) {
        ImGui::TextColored(theme::kOk, "%s", Iconed(icon::kOk, Tr("Nothing lost in the import.")).c_str());
    } else {
        const float row_h = ImGui::GetTextLineHeightWithSpacing() + 4.0f * s;
        // The table scrolls past a fifth of the window so the wizard's buttons stay on screen.
        const float height = std::min(row_h * static_cast<float>(diagnostics.Size() + 1u), ImGui::GetMainViewport()->Size.y * 0.2f);
        if (ImGui::BeginTable("##diagnostics", 4, ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH,
                              ImVec2(0.0f, height))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 18.0f * s);
            ImGui::TableSetupColumn(Tr("Code"), ImGuiTableColumnFlags_WidthFixed, 130.0f * s);
            ImGui::TableSetupColumn(Tr("Where"), ImGuiTableColumnFlags_WidthFixed, 140.0f * s);
            ImGui::TableSetupColumn(Tr("Message"), ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            for (const json::Value& d : diagnostics.Elements()) {
                const bool error = d.At("severity").AsString() == "error";
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextColored(error ? theme::kErr : theme::kWarn, "%s", error ? icon::kError : icon::kWarn);
                ImGui::TableSetColumnIndex(1);
                nui::PushTextStyle(TextStyle::Mono);
                ImGui::TextUnformatted(d.At("code").AsString().c_str());
                nui::PopTextStyle();
                ImGui::TableSetColumnIndex(2);
                const std::string owner = d.At("owner").AsString();
                const long long line = static_cast<long long>(d.At("line").AsInt());
                const std::string where = line > 0 ? Fmt("%s:%lld", owner.empty() ? "line" : owner.c_str(), line) : owner;
                ImGui::TextColored(theme::kFgMuted, "%s", where.c_str());
                ImGui::TableSetColumnIndex(3);
                ImGui::TextWrapped("%s", d.At("message").AsString().c_str());
            }
            ImGui::EndTable();
        }
    }
    if (!ok && kind == "urdf" && profile == "strict") {
        ImGui::TextColored(theme::kFgMuted, "%s", Tr("The legacy profile accepts the forms older exporters write."));
        ImGui::SameLine();
        if (ImGui::SmallButton(Tr("Retry with Legacy"))) analyze("legacy");
    }
    ImGui::Dummy(ImVec2(0.0f, 6.0f * s));

    int dest = w.open_as_new || !w.add_allowed ? 1 : 0;
    if (ImGui::BeginTable("##import_options", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, 130.0f * s);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
        if (kind == "urdf") {
            OptionLabel("Profile");
            int choice = profile == "legacy" ? 1 : 0;
            nui::Segmented profiles({"Strict", "Legacy"}, nui::Ref(&choice));
            profiles.id = "profile";
            profiles.tips = {"Reject deprecated and ambiguous forms", "Accept the forms older exporters write"};
            profiles.on_change = [&](int v) { analyze(v == 1 ? "legacy" : "strict"); };
            profiles.Draw();
        }
        OptionLabel("Destination");
        nui::Segmented destinations({"Add to Scene", "Open as New Scene"}, nui::Ref(&dest));
        destinations.id = "destination";
        destinations.icons = {icon::kAdd, icon::kOpen};
        destinations.enabled = w.add_allowed;
        destinations.on_change = [&](int v) { w.open_as_new = v == 1; };
        destinations.Draw();
        if (dest == 0) {
            OptionLabel("Name");
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##name", w.name, sizeof(w.name));
            OptionLabel("Placement");
            ImGui::Checkbox(Tr("Rest on the ground"), &w.ground);
            if (!w.pose.IsNull()) {
                ImGui::SameLine();
                const json::Value& pos = w.pose.At("pos");
                ImGui::TextColored(theme::kFgDim, "%s",
                                   Fmt(Tr("at X %.2f  Y %.2f"), pos.Elements()[0].AsDouble(), pos.Elements()[1].AsDouble()).c_str());
            }
        }
        ImGui::EndTable();
    }
    ImGui::Dummy(ImVec2(0.0f, 8.0f * s));

    const float bw = 120.0f * s, spacing = ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - bw * 2.0f - spacing);
    if (ImGui::Button(Tr("Cancel"), ImVec2(bw, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        w.path.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (PrimaryButton(Tr("Import"), bw, ok)) {
        const bool add = dest == 0;
        json::Value args = Args({{"path", json::Value::Str(w.path)},
                                 {"mode", json::Value::Str(add ? "add" : "open")},
                                 {"profile", json::Value::Str(profile)},
                                 {"name", json::Value::Str(w.name)},
                                 {"ground", json::Value::Bool(w.ground)}});
        if (add && !w.pose.IsNull()) args.Set("pose", w.pose);
        Queue("file.import", std::move(args));
        w.path.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

// ---- collision dialog ---------------------------------------------------------------------

void ImGuiLayer::DrawCollisionDialog() {
    CollisionDialog& d = frame_.ui->collision;
    const std::string title = std::string(Tr("Generate Collision")) + "###collision";
    const bool dismiss = d.dismiss;
    d.dismiss = false;
    if (d.appear) {
        ImGui::OpenPopup(title.c_str());
        d.appear = false;
    }
    const float s = nui::Scale();
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(580.0f * s, 0.0f), ImVec2(580.0f * s, FLT_MAX));
    if (!ImGui::BeginPopupModal(title.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize)) return;
    if (dismiss || d.preview.IsNull()) {
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return;
    }

    static const char* const kMethods[] = {"hull", "vhacd", "sdf", "mesh"};
    bool changed = false;
    if (ImGui::BeginTable("##collision_options", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, 130.0f * s);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
        OptionLabel("Method");
        int method = static_cast<int>(std::find(std::begin(kMethods), std::end(kMethods), d.method) - std::begin(kMethods));
        nui::Segmented methods({"Convex Hull", "V-HACD", "SDF", "Triangle Mesh"}, nui::Ref(&method));
        methods.id = "method";
        methods.icons = {icon::kHull, icon::kShapes, icon::kWaves, icon::kMeshShape};
        methods.tips = {"One convex hull around the body's visuals", "Convex pieces from a V-HACD decomposition",
                        "The visual mesh as an SDF contact surface", "The visual triangles as they are"};
        methods.on_change = [&](int v) {
            d.method = kMethods[v];
            changed = true;
        };
        methods.Draw();
        if (d.method == "vhacd") {
            OptionLabel("Max pieces");
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::SliderInt("##pieces", &d.max_pieces, 1, 64);
            changed = ImGui::IsItemDeactivatedAfterEdit() || changed;
        }
        OptionLabel("Bodies");
        changed = ImGui::Checkbox(Tr("Include descendants"), &d.subtree) || changed;
        OptionLabel("");
        changed = ImGui::Checkbox(Tr("Only bodies without colliders"), &d.only_missing) || changed;
        OptionLabel("Existing");
        changed = ImGui::Checkbox(Tr("Replace the body's colliders"), &d.replace) || changed;
        ImGui::EndTable();
    }
    if (changed) Queue("scene.collision_preview", CollisionArgs(d));
    ImGui::Dummy(ImVec2(0.0f, 4.0f * s));

    size_t builds = 0u, total = 0u;
    if (d.preview.IsArray()) {
        for (const json::Value& row : d.preview.Elements()) {
            ++total;
            if (row.At("skipped").AsString().empty()) ++builds;
        }
    }
    draw::SectionHeader(Tr("Preview"), Fmt("%zu / %zu", builds, total).c_str());
    if (total == 0u) {
        ImGui::TextColored(theme::kFgDim, "%s", Tr("The selection holds no body."));
    } else {
        const float row_h = ImGui::GetTextLineHeightWithSpacing() + 4.0f * s;
        const float height = std::min(row_h * static_cast<float>(total + 1u), 240.0f * s);
        if (ImGui::BeginTable("##collision_preview", 5, ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH,
                              ImVec2(0.0f, height))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn(Tr("Body"), ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn(Tr("Triangles"), ImGuiTableColumnFlags_WidthFixed, 70.0f * s);
            ImGui::TableSetupColumn(Tr("Pieces"), ImGuiTableColumnFlags_WidthFixed, 52.0f * s);
            ImGui::TableSetupColumn(Tr("Vertices"), ImGuiTableColumnFlags_WidthFixed, 64.0f * s);
            ImGui::TableSetupColumn(Tr("Result"), ImGuiTableColumnFlags_WidthFixed, 150.0f * s);
            ImGui::TableHeadersRow();
            for (const json::Value& row : d.preview.Elements()) {
                const std::string skipped = row.At("skipped").AsString();
                const ImVec4 tone = skipped.empty() ? theme::kFg : theme::kFgDim;
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextColored(tone, "%s", Leaf(row.At("path").AsString()).c_str());
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) ImGui::SetTooltip("%s", row.At("path").AsString().c_str());
                for (int c = 1; c <= 3; ++c) {
                    static const char* const kCounts[] = {"triangles", "pieces", "vertices"};
                    ImGui::TableSetColumnIndex(c);
                    ImGui::TextColored(tone, "%lld", static_cast<long long>(row.At(kCounts[c - 1]).AsInt()));
                }
                ImGui::TableSetColumnIndex(4);
                if (!skipped.empty()) ImGui::TextColored(theme::kFgDim, "%s", Tr(skipped.c_str()));
                else if (row.At("had_collider").AsBool()) ImGui::TextColored(theme::kWarn, "%s", Tr(d.replace ? "replaces" : "adds to"));
                else ImGui::TextColored(theme::kOk, "%s", Tr("new"));
            }
            ImGui::EndTable();
        }
    }
    ImGui::Dummy(ImVec2(0.0f, 8.0f * s));

    const float bw = 120.0f * s, spacing = ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - bw * 2.0f - spacing);
    if (ImGui::Button(Tr("Cancel"), ImVec2(bw, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    ImGui::SameLine();
    if (PrimaryButton(Tr("Generate"), bw, builds != 0u)) {
        Queue("scene.generate_collision", CollisionArgs(d));
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

// ---- properties ---------------------------------------------------------------------------

nui::Collapsible& ImGuiLayer::Fold(nui::Collapsible& group, const std::string& key) {
    if (const auto it = folds_.find(key); it != folds_.end()) group.open = it->second;
    group.on_event = [this, key](const char* type, const json::Value& value) {
        if (std::string_view(type) == "toggle") folds_[key] = value.AsBool();
    };
    return group;
}

void ImGuiLayer::CommitField(const PropertyGroup& group, const PropertyField& field, float value) {
    if (field.live) {
        json::Value ops = json::Value::Array();
        for (const std::string& target : group.targets)
            ops.PushBack(Args({{"target", json::Value::Str(target)},
                               {"field", json::Value::Str(field.field)},
                               {"after", json::Value::Float(value)}}));
        Queue("edit.apply", Args({{"label", json::Value::Str(field.label)}, {"ops", ops}}));
        return;
    }
    std::vector<StagedEdit>& staged = frame_.ui->staged;
    for (const std::string& target : group.targets) {
        const auto it = std::find_if(staged.begin(), staged.end(),
                                     [&](const StagedEdit& e) { return e.target == target && e.field == field.field; });
        if (it != staged.end()) it->after = value;
        else staged.push_back(StagedEdit{target, field.field, field.label, value});
    }
    ++staged_version_;
}

void ImGuiLayer::BuildFieldGroups(nui::VStack& root) {
    ViewerUiState& ui = *frame_.ui;
    auto& bar = root.Add<nui::Custom>([this] {
        std::vector<StagedEdit>& staged = frame_.ui->staged;
        const float s = nui::Scale();
        ImDrawList* list = ImGui::GetWindowDrawList();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x, h = ImGui::GetFrameHeight() + 10.0f * s;
        draw::Chamfer(list, p, ImVec2(p.x + w, p.y + h), 5.0f * s, U32(theme::Alpha(theme::kWarn, 0.10f)),
                      U32(theme::Alpha(theme::kWarn, 0.55f)));
        draw::Glyph(list, ImVec2(p.x + 15.0f * s, p.y + h * 0.5f), icon::kRebuild, U32(theme::kWarn), true, 14.0f);
        ImGui::SetCursorScreenPos(ImVec2(p.x + 30.0f * s, p.y + 5.0f * s));
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(theme::kWarn, "%s", Fmt(Tr("%zu pending"), staged.size()).c_str());
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
            ImGui::BeginTooltip();
            for (const StagedEdit& e : staged)
                ImGui::Text("%s  %s  %g", Tr(e.label.c_str()), e.target.empty() ? Tr("world") : e.target.c_str(), static_cast<double>(e.after));
            ImGui::EndTooltip();
        }
        const char* discard = Tr("Discard");
        const std::string apply = Iconed(icon::kRebuild, Tr("Apply & Rebuild"));
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float dw = ImGui::CalcTextSize(discard).x + 20.0f * s, aw = ImGui::CalcTextSize(apply.c_str()).x + 20.0f * s;
        ImGui::SameLine();
        ImGui::SetCursorScreenPos(ImVec2(p.x + w - dw - aw - spacing - 5.0f * s, p.y + 5.0f * s));
        if (ImGui::Button(discard, ImVec2(dw, 0.0f))) {
            staged.clear();
            ++staged_version_;
        }
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, theme::kWarn);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, theme::Alpha(theme::kWarn, 0.85f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, theme::Alpha(theme::kWarn, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_Text, theme::kAccentInk);
        if (ImGui::Button(apply.c_str(), ImVec2(aw, 0.0f))) {
            json::Value ops = json::Value::Array();
            for (const StagedEdit& e : staged)
                ops.PushBack(Args({{"target", json::Value::Str(e.target)},
                                   {"field", json::Value::Str(e.field)},
                                   {"after", json::Value::Float(e.after)}}));
            const std::string label = staged.size() == 1u ? staged.front().label : std::string("Physics Edits");
            Queue("edit.apply", Args({{"label", json::Value::Str(label)}, {"ops", ops}}));
            staged.clear();
            ++staged_version_;
        }
        ImGui::PopStyleColor(4);
        ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h));
        ImGui::Dummy(ImVec2(w, 0.0f));
    });
    bar.visible_if = [this] { return !frame_.ui->staged.empty(); };

    for (const PropertyGroup& g : ui.properties) {
        auto& group = Fold(root.Add<nui::Collapsible>(g.title, true, GroupGlyph(g.kind)), g.title + "/" + g.detail);
        group.trailing = g.targets.size() > 1u ? Fmt(Tr("%zu nodes"), g.targets.size()) : g.detail;
        auto& grid = group.Add<nui::PropertyGrid>();
        for (const PropertyField& f : g.fields) {
            const auto staged = std::find_if(ui.staged.begin(), ui.staged.end(), [&](const StagedEdit& e) {
                return e.target == g.targets.front() && e.field == f.field;
            });
            const bool pending = staged != ui.staged.end();
            auto value = std::make_shared<float>(pending ? staged->after : f.value);
            const bool bounded = f.min != f.max;
            auto& field = grid.Add<SchemaField>(f.label, nui::Bind<float>([value] { return *value; }, [value](const float& v) { *value = v; }),
                                                f.speed, bounded ? f.min : 0.0f, bounded ? f.max : 0.0f);
            field.unit = f.unit;
            field.tier = f.live ? nui::Tier::Live : nui::Tier::Rebuild;
            if (f.integer) field.format = "%.0f";
            if (f.mixed && !pending) {
                field.format = "\xE2\x80\x94";
                field.unit.clear();
                field.note = Tr("The selected nodes hold different values");
            }
            const bool dirty = f.has_saved && *value != f.saved;
            if (pending) {
                field.mark = U32(theme::kWarn);
                field.note = Tr("Waits for Apply & Rebuild");
            } else if (dirty) {
                field.mark = U32(theme::kAccentDim);
                field.note = Fmt(Tr("Saved: %g %s"), static_cast<double>(f.saved), f.unit.c_str());
            } else if (f.inherited) {
                field.note = Tr("From the material; an edit gives the collider its own");
            }
            field.on_commit = [this, g, f](const float&, const float& after) { CommitField(g, f, f.integer ? std::round(after) : after); };
            field.menu = [this, g, f, value] {
                if (ImGui::MenuItemEx(Tr("Revert to Saved"), icon::kReset, nullptr, false, f.has_saved && *value != f.saved)) {
                    *value = f.saved;
                    CommitField(g, f, f.saved_raw);
                }
                if (ImGui::MenuItemEx(Tr("Copy Value"), icon::kCopy)) ImGui::SetClipboardText(Fmt("%.9g", static_cast<double>(*value)).c_str());
                const char* clip = ImGui::GetClipboardText();
                char* end = nullptr;
                const float pasted = clip != nullptr ? std::strtof(clip, &end) : 0.0f;
                if (ImGui::MenuItemEx(Tr("Paste Value"), icon::kPaste, nullptr, false, end != clip)) {
                    *value = pasted;
                    CommitField(g, f, pasted);
                }
            };
        }
    }
}

}  // namespace nuka::runtime::app::viewer
