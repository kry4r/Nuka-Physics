// The Properties panel's materials: the selection's parts and the material each draws with, the
// edited material's name and sharing, its maps and the presets a new one starts from. The Assets
// panel's Materials page lists the document's materials to drag onto parts.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/shell_util.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace nuka::runtime::app::viewer {

namespace json = nuka::scene::json;
namespace theme = nuka::render::imgui::theme;
namespace icon = nuka::render::imgui::icon;
namespace draw = nuka::nui::draw;
using nuka::nui::Tr;
using nuka::render::imgui::FontRole;
using nuka::render::imgui::GetNukaFont;
using namespace shell;

namespace {

// The presets material.create knows, as the New menu lists them.
constexpr std::pair<const char*, const char*> kPresets[] = {{"plastic", "Plastic"}, {"metal", "Metal"},
                                                            {"rubber", "Rubber"},   {"glass", "Glass"},
                                                            {"emissive", "Emissive"}, {"textured", "Textured"}};
// The map slots' labels, in kMaterialMaps order.
constexpr const char* kMapLabels[5] = {"Albedo", "Roughness", "Metallic", "Normal", "Emissive"};
constexpr const char* kChannels[3] = {"R", "G", "B"};

// A linear color as the display shows it.
ImU32 Display(const float* color) {
    const auto gamma = [](float c) { return std::pow(std::clamp(c, 0.0f, 1.0f), 1.0f / 2.2f); };
    return ImGui::GetColorU32(ImVec4(gamma(color[0]), gamma(color[1]), gamma(color[2]), 1.0f));
}

// A square of `color` `size` across, centred in a cell one frame high.
void Swatch(const float* color, float size) {
    const float s = nui::Scale(), frame = ImGui::GetFrameHeight();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(size, frame));
    const ImVec2 a(at.x, at.y + (frame - size) * 0.5f), b(a.x + size, a.y + size);
    ImDrawList* list = ImGui::GetWindowDrawList();
    list->AddRectFilled(a, b, Display(color), theme::kRadius * s);
    list->AddRect(a, b, U32(theme::kLineHi), theme::kRadius * s);
}

// A label cell, then the value cell, of a two-column table.
void Row(const char* label) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(theme::kFgMuted, "%s", Tr(label));
    ImGui::TableSetColumnIndex(1);
}

bool BeginRows(const char* id) {
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) return false;
    ImGui::TableSetupColumn("##key", ImGuiTableColumnFlags_WidthFixed, 96.0f * nui::Scale());
    ImGui::TableSetupColumn("##value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    return true;
}

// The name a material card drags.
std::string PayloadText(const ImGuiPayload* payload) {
    return std::string(static_cast<const char*>(payload->Data), static_cast<size_t>(payload->DataSize));
}

// A node path's last name.
std::string Tail(const std::string& path) { return path.substr(path.find_last_of('/') + 1u); }

std::string UsersText(uint32_t users) {
    if (users == 0u) return Tr("Nothing uses it");
    return users == 1u ? std::string(Tr("One shape uses it")) : Fmt(Tr("%u shapes use it"), users);
}

}  // namespace

void ImGuiLayer::MaterialCombo(const char* id, const std::string& current, const std::string& path) {
    const MaterialView& v = frame_.ui->materials;
    const std::string shown = current.empty() ? std::string(Tr("Default")) : current;
    if (!draw::BeginCombo(id, shown.c_str(), ImGuiComboFlags_HeightLarge)) return;
    if (ImGui::IsWindowAppearing()) material_filter_[0] = '\0';
    if (v.materials.size() > 8u) {
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(-FLT_MIN);
        const std::string hint = Iconed(icon::kSearch, Tr("Filter"));
        ImGui::InputTextWithHint("##filter", hint.c_str(), material_filter_, sizeof(material_filter_));
    }
    const auto pick = [&](const std::string& name) {
        if (name != current)
            Queue("material.assign", Args({{"material", json::Value::Str(name)}, {"path", json::Value::Str(path)}}));
    };
    const float box = ImGui::GetTextLineHeight() * 0.8f;
    if (ImGui::Selectable(Tr("Default"), current.empty())) pick(std::string());
    const std::string needle = Lower(material_filter_);
    for (const MaterialItem& m : v.materials) {
        if (!needle.empty() && Lower(m.name).find(needle) == std::string::npos) continue;
        ImGui::PushID(m.name.c_str());
        Swatch(m.base_color, box);
        ImGui::SameLine();
        const std::string label = m.textured ? m.name + "  " + icon::kImage : m.name;
        if (ImGui::Selectable(label.c_str(), m.name == current)) pick(m.name);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
            ImGui::SetTooltip("%s", UsersText(m.users).c_str());
        }
        ImGui::PopID();
    }
    ImGui::EndCombo();
}

void ImGuiLayer::DrawMaterialParts() {
    MaterialView& v = frame_.ui->materials;
    const ImGuiStyle& style = ImGui::GetStyle();
    const float frame = ImGui::GetFrameHeight();
    // Up to ten rows show at once; a longer list scrolls.
    const float row = frame + style.CellPadding.y * 2.0f;
    const float height = row * static_cast<float>(std::min<size_t>(v.parts.size(), 10u)) + 2.0f;
    const ImGuiTableFlags flags = ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY;
    if (!ImGui::BeginTable("##parts", 3, flags, ImVec2(0.0f, height))) return;
    ImGui::TableSetupColumn("##swatch", ImGuiTableColumnFlags_WidthFixed, frame * 0.7f);
    ImGui::TableSetupColumn("##part", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("##material", ImGuiTableColumnFlags_WidthStretch, 1.1f);
    ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.0f, 0.5f));
    for (size_t i = 0; i < v.parts.size(); ++i) {
        const MaterialPartItem& part = v.parts[i];
        const bool active = v.active == static_cast<int>(i);
        ImGui::PushID(static_cast<int>(i));
        ImGui::TableNextRow(ImGuiTableRowFlags_None, frame);
        ImGui::TableSetColumnIndex(0);
        Swatch(part.base_color, frame * 0.6f);
        ImGui::TableSetColumnIndex(1);
        const ImGuiSelectableFlags select = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap;
        if (ImGui::Selectable(part.name.c_str(), active, select, ImVec2(0.0f, frame))) v.active = static_cast<int>(i);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) ImGui::SetTooltip("%s", part.path.c_str());
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* drop = ImGui::AcceptDragDropPayload(kMaterialPayload))
                Queue("material.assign",
                      Args({{"material", json::Value::Str(PayloadText(drop))}, {"path", json::Value::Str(part.path)}}));
            ImGui::EndDragDropTarget();
        }
        ImGui::TableSetColumnIndex(2);
        ImGui::SetNextItemWidth(-FLT_MIN);
        MaterialCombo("##material", part.material, part.path);
        ImGui::PopID();
    }
    ImGui::PopStyleVar();
    ImGui::EndTable();
}

void ImGuiLayer::DrawMaterialHeader() {
    const MaterialView& v = frame_.ui->materials;
    if (v.parts.empty()) return;
    const MaterialPartItem& part = v.parts[static_cast<size_t>(v.active)];
    const float s = nui::Scale();
    if (!BeginRows("##material_head")) return;
    Row("Material");
    const float button = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - button);
    MaterialCombo("##material", part.material, part.path);
    ImGui::SameLine();
    // A new material from a preset or a copy of this one; the edited part takes it.
    if (draw::IconButton("##new", icon::kAdd, false, Tr("New material for this part"))) ImGui::OpenPopup("##new_menu");
    if (ImGui::BeginPopup("##new_menu")) {
        const auto create = [&](json::Value args) {
            args.Set("assign", json::Value::Str(part.path));
            Queue("material.create", std::move(args));
        };
        if (!v.name.empty() &&
            ImGui::MenuItemEx(Fmt(Tr("Copy of %s"), v.name.c_str()).c_str(), icon::kCopy))
            create(Args({{"from", json::Value::Str(v.name)}}));
        if (ImGui::MenuItemEx(Tr("Blank"), icon::kNew)) create(json::Value::Object());
        ImGui::Separator();
        for (const auto& [preset, label] : kPresets)
            if (ImGui::MenuItemEx(Tr(label), icon::kSwatches)) create(Args({{"preset", json::Value::Str(preset)}}));
        ImGui::EndPopup();
    }
    if (v.name.empty()) {
        Row("");
        ImGui::TextColored(theme::kFgDim, "%s", Tr("This part draws with the default material."));
        Row("");
        if (ImGui::Button(Iconed(icon::kAdd, Tr("Create Material")).c_str()))
            Queue("material.make_unique", Args({{"path", json::Value::Str(part.path)}}));
        ImGui::EndTable();
        return;
    }
    Row("Name");
    const ImGuiID name_id = ImGui::GetID("##material_name");
    if (ImGui::GetActiveID() != name_id && material_name_seed_ != v.name) {
        std::snprintf(material_name_, sizeof(material_name_), "%s", v.name.c_str());
        material_name_seed_ = v.name;
    }
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputText("##material_name", material_name_, sizeof(material_name_));
    if (ImGui::IsItemDeactivatedAfterEdit() && v.name != material_name_) {
        Queue("material.rename",
              Args({{"material", json::Value::Str(v.name)}, {"name", json::Value::Str(material_name_)}}));
        material_name_seed_ = "\n";
    }
    Row("Shared");
    ImGui::AlignTextToFramePadding();
    if (v.users > 1u) {
        ImGui::TextColored(theme::kWarn, "%s", Fmt(Tr("%u shapes use it"), v.users).c_str());
        ImGui::SetItemTooltip("%s", Tr("An edit here changes every shape that uses this material."));
        ImGui::SameLine(0.0f, 8.0f * s);
        if (ImGui::SmallButton(Tr("Make Unique")))
            Queue("material.make_unique", Args({{"path", json::Value::Str(part.path)}}));
        ImGui::SetItemTooltip("%s", Tr("Give this part a copy of the material of its own."));
    } else {
        ImGui::TextColored(theme::kFgDim, "%s", Tr("This part only"));
    }
    ImGui::EndTable();
}

void ImGuiLayer::DrawMaterialMaps() {
    MaterialView& v = frame_.ui->materials;
    const float s = nui::Scale();
    const ImGuiStyle& style = ImGui::GetStyle();
    const float thumb = 44.0f * s, frame = ImGui::GetFrameHeight(), channel_w = 46.0f * s;
    const auto set = [this, &v](const char* key, json::Value value) {
        Queue("material.set", Args({{"material", json::Value::Str(v.name)}, {key, std::move(value)}}));
    };
    const ImGuiPayload* dragged = ImGui::GetDragDropPayload();
    const bool image_drag = dragged != nullptr && dragged->IsDataType(kImagePayload);
    ImDrawList* list = ImGui::GetWindowDrawList();
    ImFont* font = ImGui::GetFont();
    const float px = ImGui::GetFontSize(), line = ImGui::GetTextLineHeight();
    for (size_t k = 0; k < 5u; ++k) {
        const MaterialMapItem& map = v.maps[k];
        const bool channel = (k == 1u || k == 2u) && !map.path.empty();
        ImGui::PushID(static_cast<int>(k));
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        ImGui::SetNextItemAllowOverlap();
        ImGui::InvisibleButton("##slot", ImVec2(w, thumb));
        const bool hot = ImGui::IsItemHovered();
        if (hot && !map.path.empty()) ImGui::SetItemTooltip("%s", map.path.c_str());
        bool over = false;
        if (ImGui::BeginDragDropTarget()) {
            over = true;
            if (const ImGuiPayload* drop = ImGui::AcceptDragDropPayload(kImagePayload)) {
                const std::string path(static_cast<const char*>(drop->Data), static_cast<size_t>(drop->DataSize));
                set(kMaterialMaps[k], json::Value::Str(path));
            }
            ImGui::EndDragDropTarget();
        }
        const ImVec2 next = ImGui::GetCursorScreenPos();
        // The picture, else an empty well; an image being dragged lights every slot.
        const ImVec2 t0 = p, t1(p.x + thumb, p.y + thumb);
        if (ImTextureData* tex = map.thumbnail.empty() ? nullptr : Thumbnail(map.thumbnail)) {
            list->AddImage(tex->GetTexRef(), t0, t1);
        } else {
            list->AddRectFilled(t0, t1, U32(theme::kBg0), theme::kRadius * s);
            const ImVec2 centre(p.x + thumb * 0.5f, p.y + thumb * 0.5f);
            draw::Glyph(list, centre, icon::kImage, U32(theme::kLineHi), false, 18.0f);
        }
        const ImVec4 edge = image_drag ? theme::Alpha(theme::kAccent, over ? 1.0f : 0.5f)
                            : hot      ? theme::kLineHi
                                       : theme::kLine;
        list->AddRect(t0, t1, U32(edge), theme::kRadius * s, 0, image_drag ? 1.5f * s : 1.0f);
        // The buttons from the right: clear, choose and, for a packed map, the channel it reads.
        float x = p.x + w - frame;
        const float y = p.y + (thumb - frame) * 0.5f;
        ImGui::SetCursorScreenPos(ImVec2(x, y));
        ImGui::BeginDisabled(map.path.empty());
        if (draw::IconButton("##clear", icon::kClose, false, Tr("Clear"))) set(kMaterialMaps[k], json::Value::Str(""));
        ImGui::EndDisabled();
        x -= frame + style.ItemSpacing.x;
        ImGui::SetCursorScreenPos(ImVec2(x, y));
        if (draw::IconButton("##choose", icon::kOpen, false, Tr("Choose Image...")))
            Queue("material.choose_map",
                  Args({{"material", json::Value::Str(v.name)}, {"map", json::Value::Str(kMaterialMaps[k])}}));
        if (channel) {
            x -= channel_w + style.ItemSpacing.x;
            ImGui::SetCursorScreenPos(ImVec2(x, y));
            ImGui::SetNextItemWidth(channel_w);
            const uint32_t now = k == 1u ? v.roughness_channel : v.metallic_channel;
            if (draw::BeginCombo("##channel", kChannels[now])) {
                for (uint32_t c = 0; c < 3u; ++c)
                    if (ImGui::Selectable(kChannels[c], c == now) && c != now)
                        set(k == 1u ? "roughness_channel" : "metallic_channel", json::Value::Int(c));
                ImGui::EndCombo();
            }
            ImGui::SetItemTooltip("%s", Tr("The image channel this value reads"));
        }
        // The slot's name over the file's.
        const float tx = p.x + thumb + 10.0f * s, room = x - style.ItemSpacing.x - tx;
        const float mid = p.y + thumb * 0.5f;
        list->AddText(font, px, ImVec2(tx, mid - line), U32(theme::kFgMuted), Tr(kMapLabels[k]));
        const std::string file = map.path.empty() ? std::string(Tr("None")) : Elide(font, px, map.shown, room, false);
        const ImU32 ink = U32(map.path.empty() ? theme::kFgDim : theme::kFg);
        list->AddText(font, px, ImVec2(tx, mid + 1.0f * s), ink, file.c_str());
        ImGui::SetCursorScreenPos(next);
        ImGui::Dummy(ImVec2(0.0f, 2.0f * s));
        ImGui::PopID();
    }
    if (!BeginRows("##mapping")) return;
    Row("Mapping");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (draw::BeginCombo("##mapping", Tr(v.triplanar ? "Triplanar" : "UV"))) {
        if (ImGui::Selectable(Tr("UV"), !v.triplanar) && v.triplanar) set("triplanar", json::Value::Bool(false));
        if (ImGui::Selectable(Tr("Triplanar"), v.triplanar) && !v.triplanar) set("triplanar", json::Value::Bool(true));
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("%s", Tr("UV follows the mesh's texture coordinates; triplanar projects along the "
                                   "world axes, for a mesh without them."));
    Row("Scale");
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::DragFloat("##uv_scale", &v.uv_scale, 0.01f, 0.01f, 100.0f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) set("uv_scale", json::Value::Float(v.uv_scale));
    ImGui::SetItemTooltip("%s", Tr("Repeats of the maps per UV unit, or per metre with triplanar"));
    ImGui::EndTable();
}

void ImGuiLayer::DrawMaterialLibrary() {
    const MaterialView& v = frame_.ui->materials;
    const float s = nui::Scale();
    const bool has_scene = frame_.ui->has_scene;
    const float button = ImGui::GetFrameHeight() + 2.0f * s + ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetNextItemWidth(std::max(ImGui::GetContentRegionAvail().x - button, 60.0f * s));
    const std::string hint = Iconed(icon::kSearch, Tr("Search materials"));
    ImGui::InputTextWithHint("##library_filter", hint.c_str(), library_filter_, sizeof(library_filter_));
    ImGui::SameLine();
    ImGui::BeginDisabled(!has_scene);
    if (draw::IconButton("##library_new", icon::kAdd, false, Tr("New Material")))
        ImGui::OpenPopup("##library_new_menu");
    ImGui::EndDisabled();
    if (ImGui::BeginPopup("##library_new_menu")) {
        if (ImGui::MenuItemEx(Tr("Blank"), icon::kNew)) Queue("material.create");
        ImGui::Separator();
        for (const auto& [preset, label] : kPresets)
            if (ImGui::MenuItemEx(Tr(label), icon::kSwatches))
                Queue("material.create", Args({{"preset", json::Value::Str(preset)}}));
        ImGui::EndPopup();
    }

    std::vector<const MaterialItem*> shown;
    const std::string needle = Lower(library_filter_);
    for (const MaterialItem& m : v.materials)
        if (needle.empty() || Lower(m.name).find(needle) != std::string::npos) shown.push_back(&m);
    if (shown.empty()) {
        const char* empty = !has_scene            ? "Open a scene to see its materials."
                            : v.materials.empty() ? "No materials yet. Make one with the + button."
                                                  : "No material matches the search.";
        ImGui::TextColored(theme::kFgDim, "%s", Tr(empty));
    } else {
        // Cards like the files', a ball in a square well; the chip counts the shapes using it.
        const float gap = 10.0f * s, inset = 3.0f * s;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap, gap));
        ImGui::BeginChild("##material_grid", ImVec2(0.0f, 0.0f));
        ImDrawList* list = ImGui::GetWindowDrawList();
        ImFont* body = ImGui::GetFont();
        const float body_px = ImGui::GetFontSize(), line = ImGui::GetTextLineHeight();
        ImFont* tag_font = GetNukaFont(FontRole::Display);
        const float tag_px = theme::kFontLabel * s;
        const float avail = ImGui::GetContentRegionAvail().x;
        const size_t per_row = static_cast<size_t>(std::max(1.0f, std::floor((avail + gap) / (104.0f * s + gap))));
        const float card_w = (avail - gap * static_cast<float>(per_row - 1u)) / static_cast<float>(per_row);
        const float thumb = card_w - inset * 2.0f;
        const float card_h = inset + thumb + line + 12.0f * s;
        const std::vector<std::string>& drop = frame_.ui->material_drop;
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>((shown.size() + per_row - 1u) / per_row), card_h + gap);
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                for (size_t col = 0; col < per_row; ++col) {
                    const size_t k = static_cast<size_t>(row) * per_row + col;
                    if (k >= shown.size()) break;
                    const MaterialItem& m = *shown[k];
                    const json::Value named = Args({{"material", json::Value::Str(m.name)}});
                    if (col != 0u) ImGui::SameLine();
                    ImGui::PushID(m.name.c_str());
                    const ImVec2 p = ImGui::GetCursorScreenPos(), q(p.x + card_w, p.y + card_h);
                    ImGui::InvisibleButton("##card", ImVec2(card_w, card_h));
                    const bool hot = ImGui::IsItemHovered();
                    if (ImGui::IsItemClicked()) library_selected_ = m.name;
                    if (hot && m.users > 0u && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                        Queue("material.select_users", named);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip))
                        ImGui::SetTooltip("%s\n%s", m.name.c_str(), UsersText(m.users).c_str());
                    // The preview names the parts under the pointer while the viewport has some.
                    if (ImGui::BeginDragDropSource()) {
                        ImGui::SetDragDropPayload(kMaterialPayload, m.name.data(), m.name.size());
                        Swatch(m.base_color, ImGui::GetTextLineHeight());
                        ImGui::SameLine();
                        ImGui::AlignTextToFramePadding();
                        ImGui::TextUnformatted(m.name.c_str());
                        if (drop.empty()) {
                            ImGui::TextColored(theme::kFgDim, "%s", Tr("Drop on a part in the viewport or the Stage"));
                        } else {
                            const auto count = static_cast<unsigned>(drop.size());
                            const std::string onto =
                                count == 1u ? Tail(drop.front()) : Fmt(Tr("%u selected nodes"), count);
                            ImGui::TextColored(theme::kAccent, "\xE2\x86\x92  %s", onto.c_str());
                        }
                        ImGui::EndDragDropSource();
                    }
                    if (ImGui::BeginPopupContextItem("##card_menu")) {
                        library_selected_ = m.name;
                        if (ImGui::IsWindowAppearing())
                            std::snprintf(library_rename_, sizeof(library_rename_), "%s", m.name.c_str());
                        ImGui::SetNextItemWidth(170.0f * s);
                        const bool enter = ImGui::InputTextWithHint("##rename", Tr("New name"), library_rename_,
                                                                    sizeof(library_rename_),
                                                                    ImGuiInputTextFlags_EnterReturnsTrue);
                        ImGui::SameLine();
                        if (ImGui::SmallButton(Tr("Rename")) || enter) {
                            if (m.name != library_rename_)
                                Queue("material.rename", Args({{"material", json::Value::Str(m.name)},
                                                               {"name", json::Value::Str(library_rename_)}}));
                            ImGui::CloseCurrentPopup();
                        }
                        const bool selection = !frame_.ui->selection.empty();
                        if (ImGui::MenuItemEx(Tr("Assign to Selection"), icon::kSwatches, nullptr, false, selection))
                            Queue("material.assign", named);
                        if (ImGui::MenuItemEx(Tr("Select Users"), icon::kTarget, nullptr, false, m.users > 0u))
                            Queue("material.select_users", named);
                        if (ImGui::MenuItemEx(Tr("Duplicate"), icon::kCopy))
                            Queue("material.create", Args({{"from", json::Value::Str(m.name)}}));
                        if (ImGui::MenuItemEx(Tr("Copy Name"), icon::kCopy)) ImGui::SetClipboardText(m.name.c_str());
                        ImGui::Separator();
                        if (ImGui::MenuItemEx(Tr("Delete"), icon::kTrash)) {
                            if (m.users == 0u) {
                                Queue("material.delete", named);
                            } else {
                                library_delete_ = m.name;
                                library_replace_.clear();
                                library_delete_open_ = true;
                            }
                        }
                        ImGui::EndPopup();
                    }
                    const bool picked = library_selected_ == m.name;
                    draw::Chamfer(list, p, q, theme::kChamfer * s, U32(hot ? theme::kBg2 : theme::kBg1),
                                  U32(picked ? theme::kAccent : hot ? theme::kLineHi : theme::kLine));
                    const ImVec2 t0(p.x + inset, p.y + inset), t1(q.x - inset, p.y + inset + thumb);
                    if (ImTextureData* tex = m.ball.empty() ? nullptr : Thumbnail(m.ball)) {
                        list->AddImage(tex->GetTexRef(), t0, t1);
                    } else {
                        list->AddRectFilled(t0, t1, U32(theme::kBg0));
                        const ImVec2 centre((t0.x + t1.x) * 0.5f, (t0.y + t1.y) * 0.5f);
                        list->AddCircleFilled(centre, thumb * 0.3f, Display(m.base_color), 48);
                    }
                    if (m.textured) {
                        const ImVec2 corner(t1.x - 11.0f * s, t0.y + 11.0f * s);
                        draw::Glyph(list, corner, icon::kImage, U32(theme::kFgMuted), false, 14.0f);
                    }
                    const std::string tag = m.users == 0u ? Upper(Tr("Unused")) : Fmt("\xC3\x97%u", m.users);
                    const ImVec2 ts = tag_font->CalcTextSizeA(tag_px, FLT_MAX, 0.0f, tag.c_str());
                    const ImVec2 g0(t0.x + 4.0f * s, t1.y - ts.y - 8.0f * s);
                    const ImVec2 g1(g0.x + ts.x + 10.0f * s, t1.y - 4.0f * s);
                    list->AddRectFilled(g0, g1, U32(theme::Alpha(theme::kBg0, 0.85f)), theme::kRadius * s);
                    const ImVec4 tag_ink = m.users == 0u ? theme::kFgDim : hot ? theme::kAccent : theme::kFgMuted;
                    const ImVec2 tag_at(g0.x + 5.0f * s, g0.y + 2.0f * s);
                    list->AddText(tag_font, tag_px, tag_at, U32(tag_ink), tag.c_str());
                    list->AddText(body, body_px, ImVec2(p.x + inset + 4.0f * s, t1.y + 6.0f * s),
                                  U32(hot || picked ? theme::kFg : theme::kFgMuted),
                                  Elide(body, body_px, m.name, card_w - inset * 2.0f - 8.0f * s, false).c_str());
                    ImGui::PopID();
                }
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
    }

    // Deleting a material in use asks what its users take instead.
    if (library_delete_open_) {
        ImGui::OpenPopup("###delete_material");
        library_delete_open_ = false;
    }
    const std::string title = std::string(Tr("Delete Material")) + "###delete_material";
    if (!ImGui::BeginPopupModal(title.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
    ImGui::TextUnformatted(Fmt(Tr("'%s' is in use. What do its users take instead?"), library_delete_.c_str()).c_str());
    ImGui::SetNextItemWidth(280.0f * s);
    const std::string replace = library_replace_.empty() ? std::string(Tr("Default")) : library_replace_;
    if (draw::BeginCombo("##replace", replace.c_str())) {
        if (ImGui::Selectable(Tr("Default"), library_replace_.empty())) library_replace_.clear();
        for (const MaterialItem& m : v.materials)
            if (m.name != library_delete_ && ImGui::Selectable(m.name.c_str(), m.name == library_replace_))
                library_replace_ = m.name;
        ImGui::EndCombo();
    }
    ImGui::Spacing();
    if (ImGui::Button(Iconed(icon::kTrash, Tr("Delete")).c_str())) {
        Queue("material.delete", Args({{"material", json::Value::Str(library_delete_)},
                                       {"replace", json::Value::Str(library_replace_)}}));
        library_delete_.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button(Tr("Cancel"))) {
        library_delete_.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void ImGuiLayer::DrawMaterialDrop() {
    const ImGuiPayload* drop =
        ImGui::AcceptDragDropPayload(kMaterialPayload, ImGuiDragDropFlags_AcceptNoDrawDefaultRect);
    if (drop == nullptr || frame_.ui->material_drop.empty()) return;
    json::Value paths = json::Value::Array();
    for (const std::string& path : frame_.ui->material_drop) paths.PushBack(json::Value::Str(path));
    Queue("material.assign", Args({{"material", json::Value::Str(PayloadText(drop))}, {"paths", std::move(paths)}}));
}

}  // namespace nuka::runtime::app::viewer
