// nui data views: the virtualized tree, tables and the property grid.

#include "runtime/app/viewer/nui/nui_internal.hpp"

#include "render/imgui/nuka_imgui.hpp"

#include "imgui_internal.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace nuka::nui {

using namespace detail;
using nuka::render::imgui::FontRole;
using nuka::render::imgui::GetNukaFont;

namespace {

uint64_t next_items_version = 1u;

// Items from JSON: [{"id", "label", "parent", "icon", "badge"}], parent 0 is the root.
class JsonItems final : public ItemModel {
public:
    explicit JsonItems(const json::Value& items) : version_(next_items_version++) {
        for (const json::Value& e : items.Elements()) {
            const auto item = static_cast<uint64_t>(e.At("id").AsInt());
            const json::Value* parent = e.Find("parent");
            Entry entry;
            Read(e, "label", entry.label);
            ReadIcon(e, "icon", entry.icon);
            Read(e, "badge", entry.badge);
            entries_[item] = std::move(entry);
            children_[parent != nullptr ? static_cast<uint64_t>(parent->AsInt()) : 0u].push_back(item);
        }
    }
    uint64_t Version() override { return version_; }
    void Children(uint64_t item, std::vector<uint64_t>& out) const override {
        out.clear();
        const auto it = children_.find(item);
        if (it != children_.end()) out = it->second;
    }
    std::string_view Label(uint64_t item) const override { return entries_.at(item).label; }
    const char* Icon(uint64_t item) const override { return entries_.at(item).icon.c_str(); }
    std::string_view Badge(uint64_t item) const override { return entries_.at(item).badge; }

private:
    struct Entry {
        std::string label;
        std::string icon;
        std::string badge;
    };
    std::unordered_map<uint64_t, Entry> entries_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> children_;
    uint64_t version_;
};

bool PathTo(const ItemModel& model, uint64_t from, uint64_t target, std::vector<uint64_t>& path) {
    std::vector<uint64_t> kids;
    model.Children(from, kids);
    for (const uint64_t kid : kids) {
        if (kid == target) return true;
        path.push_back(kid);
        if (PathTo(model, kid, target, path)) return true;
        path.pop_back();
    }
    return false;
}

void OpenAll(const ItemModel& model, uint64_t item, std::unordered_set<uint64_t>& open) {
    std::vector<uint64_t> kids;
    model.Children(item, kids);
    if (item != 0u && !kids.empty()) open.insert(item);
    for (const uint64_t kid : kids) OpenAll(model, kid, open);
}

}  // namespace

// ---- tree ---------------------------------------------------------------------------------

void TreeView::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "height", height);
    Read(props, "open_depth", open_depth);
    if (const json::Value* v = props.Find("items")) {
        model = std::make_shared<JsonItems>(*v);
        seeded_ = false;
    }
    if (const json::Value* v = props.Find("selected")) selected_ = static_cast<uint64_t>(v->AsInt());
}

json::Value TreeView::GetValue() const { return json::Value::Int(static_cast<int64_t>(selected_)); }

void TreeView::ExpandAll() {
    OpenAll(*model, 0u, open_);
    ++layout_;
}

void TreeView::CollapseAll() {
    open_.clear();
    ++layout_;
}

void TreeView::Reveal(uint64_t item) { reveal_ = item; }

void TreeView::Select(uint64_t item) {
    selected_ = item;
    anchor_ = item;
    if (on_select) on_select(item);
    Emit("select", json::Value::Int(static_cast<int64_t>(item)));
}

bool TreeView::Matches(uint64_t item, const std::string& needle) const {
    if (Lower(model->Label(item)).find(needle) != std::string::npos) return true;
    std::vector<uint64_t> kids;
    model->Children(item, kids);
    return std::any_of(kids.begin(), kids.end(), [&](uint64_t kid) { return Matches(kid, needle); });
}

// A filter shows every path to a match expanded; otherwise only open items descend.
void TreeView::Walk(uint64_t item, int depth, const std::string& needle) {
    std::vector<uint64_t> kids;
    model->Children(item, kids);
    std::vector<uint64_t> grand;
    for (const uint64_t kid : kids) {
        if (!needle.empty() && !Matches(kid, needle)) continue;
        model->Children(kid, grand);
        const bool has_children = !grand.empty();
        rows_.push_back(Row{kid, depth, has_children});
        if (has_children && (!needle.empty() || open_.count(kid) != 0u)) Walk(kid, depth + 1, needle);
    }
}

void TreeView::Flatten() {
    rows_.clear();
    Walk(0u, 0, filter ? Lower(filter->Get()) : std::string());
}

void TreeView::DrawSelf() {
    if (!model) return;
    const float s = Scale();
    // The version first: a model may refresh its items in it, before seeding or a reveal walks them.
    const uint64_t version = model->Version();
    if (!seeded_) {
        std::vector<std::pair<uint64_t, int>> stack = {{0u, -1}};
        std::vector<uint64_t> kids;
        while (!stack.empty()) {
            const auto [item, depth] = stack.back();
            stack.pop_back();
            model->Children(item, kids);
            if (item != 0u && !kids.empty() && depth < open_depth) open_.insert(item);
            if (depth + 1 < open_depth)
                for (const uint64_t kid : kids) stack.emplace_back(kid, depth + 1);
        }
        seeded_ = true;
        ++layout_;
    }
    if (reveal_ != 0u) {
        std::vector<uint64_t> path;
        if (PathTo(*model, 0u, reveal_, path)) open_.insert(path.begin(), path.end());
        ++layout_;
    }
    const std::string needle = filter ? Lower(filter->Get()) : std::string();
    if (version != version_ || layout_ != flattened_layout_ || needle != flattened_filter_) {
        Flatten();
        version_ = version;
        flattened_layout_ = layout_;
        flattened_filter_ = needle;
    }

    const float row_h = std::floor(ImGui::GetTextLineHeight() + 10.0f * s);
    const float indent = 14.0f * s;
    if (ImGui::BeginChild("##tree", ImVec2(0.0f, height > 0.0f ? height * s : 0.0f))) {
        auto scroll_to = [&](size_t index) {
            const float top = static_cast<float>(index) * row_h;
            const float view = ImGui::GetWindowHeight();
            if (top < ImGui::GetScrollY()) ImGui::SetScrollY(top);
            else if (top + row_h > ImGui::GetScrollY() + view) ImGui::SetScrollY(top + row_h - view);
        };
        const auto selected_row = std::find_if(rows_.begin(), rows_.end(), [&](const Row& r) {
            return is_selected ? is_selected(r.item) : r.item == selected_;
        });
        const size_t current = static_cast<size_t>(selected_row - rows_.begin());
        uint64_t toggle = 0u;
        if (reveal_ != 0u) {
            const auto it = std::find_if(rows_.begin(), rows_.end(), [&](const Row& r) { return r.item == reveal_; });
            if (it != rows_.end()) ImGui::SetScrollY(std::max(0.0f, static_cast<float>(it - rows_.begin()) * row_h - ImGui::GetWindowHeight() * 0.4f));
            reveal_ = 0u;
        }
        // Arrow keys walk the rows; left and right fold and unfold.
        if (ImGui::IsWindowFocused() && current < rows_.size() && !ImGui::GetIO().WantTextInput) {
            const Row& row = rows_[current];
            if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) && current + 1u < rows_.size()) {
                Select(rows_[current + 1u].item);
                scroll_to(current + 1u);
            } else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) && current > 0u) {
                Select(rows_[current - 1u].item);
                scroll_to(current - 1u);
            } else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) && row.has_children && open_.count(row.item) == 0u) {
                toggle = row.item;
            } else if (on_rename && ImGui::IsKeyPressed(ImGuiKey_F2)) {
                const std::string_view label = model->Label(row.item);
                const size_t n = std::min(label.size(), sizeof(rename_buf_) - 1u);
                std::memcpy(rename_buf_, label.data(), n);
                rename_buf_[n] = '\0';
                renaming_ = row.item;
                rename_focus_ = true;
            } else if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
                if (row.has_children && open_.count(row.item) != 0u) {
                    toggle = row.item;
                } else {
                    for (size_t i = current; i-- > 0u;) {
                        if (rows_[i].depth < row.depth) {
                            Select(rows_[i].item);
                            scroll_to(i);
                            break;
                        }
                    }
                }
            }
        }

        ImDrawList* list = ImGui::GetWindowDrawList();
        ImFont* badge_font = GetNukaFont(FontRole::Display);
        const float badge_px = theme::kFontLabel * s;
        const float tool_w = 20.0f * s;
        const float tools_w = static_cast<float>(row_tools.size()) * tool_w;
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(rows_.size()), row_h);
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                const Row row = rows_[static_cast<size_t>(i)];
                ImGui::PushID(reinterpret_cast<const void*>(static_cast<uintptr_t>(row.item)));
                const ImVec2 pos = ImGui::GetCursorScreenPos();
                const float w = std::max(ImGui::GetContentRegionAvail().x, 1.0f);
                // Rows advance by exactly row_h, which the clipper, keyboard scroll and reveal all assume.
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 0.0f));
                if (!row_tools.empty()) ImGui::SetNextItemAllowOverlap();
                const bool clicked = ImGui::InvisibleButton("##row", ImVec2(w, row_h));
                ImGui::PopStyleVar();
                const bool hovered = ImGui::IsItemHovered();
                const bool selected = is_selected ? is_selected(row.item) : row.item == selected_;
                const bool open = !needle.empty() || open_.count(row.item) != 0u;
                const float x = pos.x + static_cast<float>(row.depth) * indent;
                if (clicked) {
                    const ImGuiIO& io = ImGui::GetIO();
                    const auto anchor = std::find_if(rows_.begin(), rows_.end(), [&](const Row& r) { return r.item == anchor_; });
                    if (row.has_children && ImGui::GetIO().MouseClickedPos[0].x < x + 16.0f * s) toggle = row.item;
                    else if (io.KeyShift && on_range && anchor != rows_.end()) {
                        const std::ptrdiff_t a = anchor - rows_.begin(), b = static_cast<std::ptrdiff_t>(i);
                        std::vector<uint64_t> items;
                        for (std::ptrdiff_t k = a;; k += a <= b ? 1 : -1) {
                            items.push_back(rows_[static_cast<size_t>(k)].item);
                            if (k == b) break;
                        }
                        on_range(items);
                    } else if (io.KeyCtrl && on_toggle) {
                        anchor_ = row.item;
                        on_toggle(row.item);
                    } else {
                        Select(row.item);
                    }
                }
                if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    if (row.has_children) toggle = row.item;
                    if (on_activate) on_activate(row.item);
                    Emit("activate", json::Value::Int(static_cast<int64_t>(row.item)));
                }
                if (ImGui::IsItemClicked(ImGuiMouseButton_Right) && !selected) Select(row.item);
                if (context_menu && ImGui::BeginPopupContextItem("##context")) {
                    context_menu(row.item);
                    ImGui::EndPopup();
                }
                if (on_drop) {
                    if (ImGui::BeginDragDropSource()) {
                        ImGui::SetDragDropPayload("NUI_TREE_ITEM", &row.item, sizeof(row.item));
                        const std::string_view text = model->Label(row.item);
                        ImGui::TextUnformatted(text.data(), text.data() + text.size());
                        ImGui::EndDragDropSource();
                    }
                    if (ImGui::BeginDragDropTarget()) {
                        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("NUI_TREE_ITEM")) {
                            uint64_t dragged = 0u;
                            std::memcpy(&dragged, payload->Data, sizeof(dragged));
                            if (dragged != row.item) on_drop(dragged, row.item);
                        }
                        ImGui::EndDragDropTarget();
                    }
                }
                if (drop_target) drop_target(row.item);

                const ImVec2 end(pos.x + w, pos.y + row_h);
                const ImVec2 after = ImGui::GetCursorScreenPos();
                const bool row_hovered = ImGui::IsMouseHoveringRect(pos, end) && ImGui::IsWindowHovered();
                if (selected) {
                    list->AddRectFilled(pos, end, U32(theme::Alpha(theme::kAccent, 0.14f)));
                    list->AddRectFilled(pos, ImVec2(pos.x + 2.0f * s, end.y), U32(theme::kAccent));
                } else if (row_hovered) {
                    list->AddRectFilled(pos, end, U32(theme::kBg2));
                }
                for (int d = 0; d < row.depth; ++d) {
                    const float gx = pos.x + static_cast<float>(d) * indent + 8.0f * s;
                    list->AddLine(ImVec2(gx, pos.y), ImVec2(gx, end.y), U32(theme::kLine));
                }
                const float mid = pos.y + row_h * 0.5f;
                if (row.has_children)
                    draw::Glyph(list, ImVec2(x + 8.0f * s, mid), open ? icon::kChevronDown : icon::kChevronRight,
                                U32(hovered ? theme::kFgMuted : theme::kFgDim), false, 11.0f);
                float tx = x + 18.0f * s;
                const char* glyph = model->Icon(row.item);
                if (glyph != nullptr && *glyph != '\0') {
                    draw::Glyph(list, ImVec2(tx + 7.0f * s, mid), glyph,
                                U32(selected ? theme::kAccent : model->IconColor(row.item)), false, 14.0f);
                    tx += 20.0f * s;
                }
                const float tools_x = end.x - tools_w;
                float text_end = tools_x - 6.0f * s;
                const std::string_view label = model->Label(row.item);
                const float label_w = ImGui::CalcTextSize(label.data(), label.data() + label.size()).x;
                const std::string badge = Tr(std::string(model->Badge(row.item)));
                if (!badge.empty()) {
                    const ImVec2 bsize = badge_font->CalcTextSizeA(badge_px, FLT_MAX, 0.0f, badge.data(), badge.data() + badge.size());
                    // The kind badge yields to the name when the row is too narrow for both.
                    if (tx + label_w + bsize.x + 18.0f * s <= tools_x) {
                        list->AddText(badge_font, badge_px, ImVec2(tools_x - bsize.x - 8.0f * s, mid - bsize.y * 0.5f),
                                      U32(theme::kFgDim), badge.data(), badge.data() + badge.size());
                        text_end -= bsize.x + 10.0f * s;
                    }
                }
                const float clip = std::max(tx, text_end);
                if (row.item == renaming_) {
                    ImGui::SetCursorScreenPos(ImVec2(tx - 4.0f * s, mid - ImGui::GetFrameHeight() * 0.5f));
                    ImGui::SetNextItemWidth(std::max(tools_x - tx, 60.0f * s));
                    if (rename_focus_) ImGui::SetKeyboardFocusHere();
                    const bool done = ImGui::InputText("##rename", rename_buf_, sizeof(rename_buf_),
                                                       ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
                    // The field takes focus a frame late; losing it after that cancels.
                    if (done) {
                        renaming_ = 0u;
                        if (on_rename && model->Label(row.item) != rename_buf_) on_rename(row.item, rename_buf_);
                    } else if (ImGui::IsItemActive()) {
                        rename_focus_ = false;
                    } else if (!rename_focus_) {
                        renaming_ = 0u;
                    }
                } else {
                    const bool dimmed = is_dimmed && is_dimmed(row.item);
                    ImGui::PushStyleColor(ImGuiCol_Text, dimmed     ? theme::kFgDim
                                                         : selected ? theme::kFg
                                                                    : theme::Alpha(theme::kFg, 0.92f));
                    ImGui::RenderTextEllipsis(list, ImVec2(tx, mid - ImGui::GetTextLineHeight() * 0.5f), ImVec2(clip, end.y),
                                              clip, label.data(), label.data() + label.size(), nullptr);
                    ImGui::PopStyleColor();
                }
                for (size_t k = 0; k < row_tools.size(); ++k) {
                    const RowTool& tool = row_tools[k];
                    if (tool.applies && !tool.applies(row.item)) continue;
                    const bool on = tool.on && tool.on(row.item);
                    const ImVec2 at(tools_x + static_cast<float>(k) * tool_w, pos.y);
                    ImGui::SetCursorScreenPos(at);
                    ImGui::PushID(static_cast<int>(k));
                    const bool pressed = ImGui::InvisibleButton("##tool", ImVec2(tool_w, row_h));
                    const bool tool_hovered = ImGui::IsItemHovered();
                    ImGui::PopID();
                    if (tool_hovered && !tool.tooltip.empty()) ImGui::SetTooltip("%s", Tr(tool.tooltip.c_str()));
                    if (pressed && tool.on_click) tool.on_click(row.item);
                    if (on || row_hovered)
                        draw::Glyph(list, ImVec2(at.x + tool_w * 0.5f, mid), on ? tool.icon_on : tool.icon_off,
                                    U32(on ? theme::kAccent : tool_hovered ? theme::kFg : theme::kFgDim), false, 13.0f);
                }
                ImGui::SetCursorScreenPos(after);
                ImGui::PopID();
            }
        }
        // The row tools leave the cursor set by hand; an item validates the clipper's final extent.
        ImGui::Dummy(ImVec2(0.0f, 0.0f));
        if (toggle != 0u) {
            if (open_.erase(toggle) == 0u) open_.insert(toggle);
            ++layout_;
        }
    }
    ImGui::EndChild();
}

// ---- table --------------------------------------------------------------------------------

void Table::SetProps(const json::Value& props) {
    Node::SetProps(props);
    Read(props, "columns", columns);
    Read(props, "height", height);
    if (const json::Value* v = props.Find("rows")) {
        rows.clear();
        for (const json::Value& row : v->Elements()) {
            std::vector<std::string> cells;
            for (const json::Value& c : row.Elements())
                cells.push_back(c.type() == json::Value::Type::String ? c.AsString() : c.Dump(0));
            rows.push_back(std::move(cells));
        }
    }
}

void Table::DrawSelf() {
    const int count = std::max(1, static_cast<int>(columns.size()));
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Resizable;
    if (!ImGui::BeginTable("##table", count, flags, ImVec2(0.0f, height > 0.0f ? height * Scale() : 0.0f))) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    for (const std::string& name : columns) ImGui::TableSetupColumn(Tr(name).c_str());
    if (!columns.empty()) ImGui::TableHeadersRow();
    const size_t total = row_count ? row_count() : rows.size();
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(total));
    while (clipper.Step()) {
        for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
            const auto row = static_cast<size_t>(r);
            ImGui::TableNextRow();
            for (int c = 0; c < count; ++c) {
                ImGui::TableSetColumnIndex(c);
                const auto column = static_cast<size_t>(c);
                if (cell) {
                    const std::string text = cell(row, column);
                    ImGui::TextUnformatted(text.c_str());
                } else if (column < rows[row].size()) {
                    ImGui::TextUnformatted(rows[row][column].c_str());
                }
            }
        }
    }
    ImGui::EndTable();
}

// ---- property grid ------------------------------------------------------------------------

void PropertyGrid::SetProps(const json::Value& props) {
    Node::SetProps(props);
    if (const json::Value* v = props.Find("schema")) {
        Clear();
        BuildFromSchema(*v);
    }
}

json::Value PropertyGrid::GetValue() const {
    json::Value out = json::Value::Object();
    for (const auto& child : children_)
        if (!child->id.empty()) out.Set(child->id, child->GetValue());
    return out;
}

// Each entry is {"key", "type", "label"?, "value"?, ...field props}; the label defaults to the key.
void PropertyGrid::BuildFromSchema(const json::Value& schema) {
    for (const json::Value& entry : schema.Elements()) {
        const std::string key = entry.At("key").AsString();
        const json::Value* type = entry.Find("type");
        std::unique_ptr<Node> node = MakeNode(type != nullptr ? type->AsString() : "float");
        json::Value props = entry;
        if (!props.Has("label")) props.Set("label", json::Value::Str(key));
        node->SetProps(props);
        node->id = key;
        node->on_event = [this, key](const char* event, const json::Value& value) {
            json::Value out = json::Value::Object();
            out.Set("key", json::Value::Str(key));
            out.Set("value", value);
            Emit(event, std::move(out));
        };
        Adopt(std::move(node));
    }
}

void PropertyGrid::DrawSelf() {
    // The label column fits the longest label, between a quarter and 55% of the row.
    float widest = 0.0f;
    for (const auto& child : children_)
        if (child->Shown()) widest = std::max(widest, child->LabelWidth());
    const float avail = ImGui::GetContentRegionAvail().x;
    if (!ImGui::BeginTable("##properties", 2, ImGuiTableFlags_SizingStretchProp)) return;
    ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed,
                            std::min(std::max(widest, avail * 0.25f), avail * 0.55f));
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
    for (const auto& child : children_) {
        if (!child->Shown()) continue;
        if (!child->IsField()) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
        }
        child->Draw();
    }
    ImGui::EndTable();
}

}  // namespace nuka::nui
