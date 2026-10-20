// Semantic labels in Properties and the Stage menu, and the Export Dataset window that steps the
// scene, reads its sensors each frame and writes them out as a dataset.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/editor_dataset.hpp"
#include "runtime/app/viewer/shell_util.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <string>

namespace nuka::runtime::app::viewer {

namespace json = nuka::scene::json;
namespace theme = nuka::render::imgui::theme;
namespace icon = nuka::render::imgui::icon;
namespace draw = nuka::nui::draw;
using nuka::nui::Tr;
using namespace shell;

namespace {

// The window's names for kDatasetChannels, in their order.
constexpr const char* kChannelLabels[] = {"RGB", "Depth", "Instance", "Semantic", "Normal", "Albedo"};

// The last part of a node path.
std::string Leaf(const std::string& path) {
    return path.substr(path.rfind('/') + 1u);
}

}  // namespace

void ImGuiLayer::DrawSemanticField() {
    const LabelView& l = frame_.ui->labels;
    const auto node = frame_.scene->Ecs().NodeOf(frame_.ui->selected_entity);
    const std::string path = frame_.scene->Tree().PathOf(node);
    std::string from;
    const std::string label = LabelOf(l.labels, path, &from);
    const bool own = !label.empty() && from == path;
    if (label_path_ != path || label_version_ != l.version) {
        std::snprintf(label_buf_, sizeof(label_buf_), "%s", own ? label.c_str() : "");
        label_path_ = path;
        label_version_ = l.version;
    }
    // Every selected node takes the label; the copy is taken again once the host has it.
    const auto set = [this](const std::string& text) {
        Queue("label.set", Args({{"label", json::Value::Str(text)}}));
        label_version_ = ~0ull;
    };
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(theme::kFgMuted, "%s", Tr("Semantic"));
    ImGui::SameLine();
    const float button = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - button);
    const std::string hint = label.empty() ? std::string(Tr("unlabeled"))
                                           : label + "  \xC2\xB7  " + Fmt(Tr("from %s"), Leaf(from).c_str());
    ImGui::InputTextWithHint("##semantic", hint.c_str(), label_buf_, sizeof(label_buf_));
    if (ImGui::IsItemDeactivatedAfterEdit()) set(label_buf_);
    ImGui::SetItemTooltip("%s", Tr("The class this node and its children show in semantic images; empty inherits."));
    ImGui::SameLine();
    if (ImGui::BeginCombo("##classes", nullptr, ImGuiComboFlags_NoPreview | ImGuiComboFlags_PopupAlignLeft)) {
        if (l.classes.empty()) ImGui::TextDisabled("%s", Tr("No labels yet"));
        for (const std::string& c : l.classes)
            if (ImGui::Selectable(c.c_str(), own && c == label)) set(c);
        if (own) {
            ImGui::Separator();
            if (ImGui::Selectable(Tr("Clear"))) set(std::string());
        }
        ImGui::EndCombo();
    }
}

void ImGuiLayer::SemanticMenu(const std::string& path) {
    if (!ImGui::BeginMenuEx(Tr("Semantic Label"), icon::kTag)) return;
    const LabelView& l = frame_.ui->labels;
    std::string from;
    const std::string label = LabelOf(l.labels, path, &from);
    const bool own = !label.empty() && from == path;
    const auto set = [this, &path](const std::string& text) {
        Queue("label.set", Args({{"path", json::Value::Str(path)}, {"label", json::Value::Str(text)}}));
    };
    for (const std::string& c : l.classes)
        if (ImGui::MenuItem(c.c_str(), nullptr, own && c == label)) set(c);
    if (!l.classes.empty()) ImGui::Separator();
    ImGui::SetNextItemWidth(150.0f * nui::Scale());
    if (ImGui::InputTextWithHint("##new_label", Tr("New label"), new_label_, sizeof(new_label_),
                                 ImGuiInputTextFlags_EnterReturnsTrue) &&
        new_label_[0] != '\0') {
        set(new_label_);
        new_label_[0] = '\0';
        ImGui::CloseCurrentPopup();
    }
    if (ImGui::MenuItemEx(Tr("Clear"), icon::kClose, nullptr, false, own)) set(std::string());
    if (!own && !label.empty()) ImGui::TextDisabled("%s", Fmt(Tr("inherits %s"), label.c_str()).c_str());
    ImGui::EndMenu();
}

void ImGuiLayer::DrawDatasetExporter() {
    const DatasetView& d = frame_.ui->dataset;
    if (!d.open || frame_.scene == nullptr) {
        dataset_form_open_ = false;
        return;
    }
    const RandomizeView& r = frame_.ui->randomize;
    const std::vector<SensorItem>& items = frame_.ui->sensors.items;
    const LabelView& l = frame_.ui->labels;
    DatasetForm& f = dataset_form_;
    // An opened window starts from the Randomize panel's seed and kinds.
    if (!dataset_form_open_) {
        f.seed = static_cast<int>(r.config.seed);
        f.render = r.config.render && r.camera;
        f.physics = r.config.physics && r.articulated;
        dataset_form_open_ = true;
    }
    const float s = nui::Scale();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImVec2 size(std::min(520.0f * s, vp->WorkSize.x * 0.9f), std::min(640.0f * s, vp->WorkSize.y * 0.9f));
    ImGui::SetNextWindowSize(size, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(vp->GetWorkCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    bool open = true;
    const std::string title = std::string(Tr("Export Dataset")) + "###dataset_export";
    if (ImGui::Begin(title.c_str(), &open, ImGuiWindowFlags_NoCollapse)) {
        const auto row = [](const char* label) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(theme::kFgMuted, "%s", Tr(label));
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
        };
        // The form scrolls; the Export row and its result line stay pinned at the bottom.
        const float footer = ImGui::GetFrameHeightWithSpacing() + 2.0f * ImGui::GetTextLineHeightWithSpacing() +
                             ImGui::GetStyle().ItemSpacing.y;
        ImGui::BeginChild("##dataset_form", ImVec2(0.0f, -footer));
        ImGui::BeginDisabled(d.running);
        if (ImGui::BeginTable("##dataset_options", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, 130.0f * s);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
            row("Folder");
            ImGui::InputTextWithHint("##out", Tr("empty asks for a folder"), f.out, sizeof(f.out));
            row("Frames");
            ImGui::DragInt("##frames", &f.frames, 1.0f, 1, 1000000, "%d", ImGuiSliderFlags_AlwaysClamp);
            row("Ticks per frame");
            ImGui::DragInt("##every", &f.every, 0.1f, 1, 100000, "%d", ImGuiSliderFlags_AlwaysClamp);
            row("Seed");
            ImGui::DragInt("##seed", &f.seed, 0.2f, 0, INT_MAX, "%d", ImGuiSliderFlags_AlwaysClamp);
            ImGui::SetItemTooltip("%s", Tr("Frame k draws its randomization from seed + k."));
            row("Start");
            ImGui::Checkbox(Tr("Reset the scene first"), &f.reset);
            ImGui::EndTable();
        }

        size_t cameras = 0u, lidars = 0u, states = 0u;
        for (const SensorItem& item : items) {
            cameras += item.camera ? 1u : 0u;
            lidars += item.lidar ? 1u : 0u;
            states += !item.camera && !item.lidar ? 1u : 0u;
        }
        draw::SectionHeader(Tr("Cameras"), Fmt("%zu", cameras).c_str());
        if (cameras == 0u) ImGui::TextDisabled("%s", Tr("No camera sensors; add one to write images."));
        const ImGuiStyle& style = ImGui::GetStyle();
        const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
        size_t shown = 0u;
        for (size_t i = 0; i < items.size(); ++i) {
            if (!items[i].camera) continue;
            const auto it = std::find(f.cameras.begin(), f.cameras.end(), items[i].name);
            bool on = it == f.cameras.end();
            const std::string name = Iconed(FindSensorKind(items[i].type).glyph, items[i].name.c_str());
            const float w = ImGui::GetFrameHeight() + style.ItemInnerSpacing.x + ImGui::CalcTextSize(name.c_str()).x;
            if (shown++ > 0u && ImGui::GetItemRectMax().x + style.ItemSpacing.x * 4.0f + w <= right) {
                ImGui::SameLine(0.0f, style.ItemSpacing.x * 4.0f);
            }
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::Checkbox(name.c_str(), &on)) {
                if (on) f.cameras.erase(it);
                else f.cameras.push_back(items[i].name);
            }
            ImGui::PopID();
        }
        ImGui::BeginDisabled(cameras == 0u);
        for (size_t c = 0; c < kDatasetChannelCount; ++c) {
            if (c % 3u != 0u) ImGui::SameLine(static_cast<float>(c % 3u) * 120.0f * s);
            ImGui::Checkbox(Tr(kChannelLabels[c]), &f.channels[c]);
        }
        ImGui::EndDisabled();

        draw::SectionHeader(Tr("Other sensors"));
        ImGui::BeginDisabled(lidars == 0u);
        ImGui::Checkbox(Fmt(Tr("Lidars (%zu)"), lidars).c_str(), &f.lidar);
        ImGui::EndDisabled();
        ImGui::SameLine(240.0f * s);
        ImGui::BeginDisabled(states == 0u);
        ImGui::Checkbox(Fmt(Tr("State sensors (%zu)"), states).c_str(), &f.state);
        ImGui::EndDisabled();

        draw::SectionHeader(Tr("Randomize each frame"));
        ImGui::BeginDisabled(!r.camera);
        ImGui::Checkbox(Tr("Render"), &f.render);
        ImGui::SetItemTooltip("%s", Tr("The ranges are the Randomize panel's."));
        ImGui::EndDisabled();
        ImGui::SameLine(240.0f * s);
        ImGui::BeginDisabled(!r.articulated && r.param_keys.empty());
        ImGui::Checkbox(Tr("Physics"), &f.physics);
        ImGui::SetItemTooltip("%s", Tr("The ranges are the Randomize panel's."));
        ImGui::EndDisabled();

        draw::SectionHeader(Tr("Labels"), Fmt("%zu", l.labels.size()).c_str());
        ImGui::PushTextWrapPos(0.0f);
        if (l.classes.empty()) {
            ImGui::TextColored(theme::kWarn, "%s",
                               Tr("No semantic labels, so semantic images and boxes stay empty. Label nodes from the "
                                  "Stage menu or in Properties."));
        } else {
            std::string classes;
            for (const std::string& c : l.classes) classes += (classes.empty() ? "" : "  \xC2\xB7  ") + c;
            ImGui::TextColored(theme::kFgMuted, "%s", classes.c_str());
        }
        ImGui::PopTextWrapPos();
        ImGui::EndDisabled();
        ImGui::EndChild();

        ImGui::Separator();
        if (d.running) {
            const float cancel = ImGui::CalcTextSize(Tr("Cancel")).x + ImGui::GetStyle().FramePadding.x * 2.0f +
                                 ImGui::GetStyle().ItemSpacing.x;
            const float done = d.frames > 0u ? static_cast<float>(d.frame) / static_cast<float>(d.frames) : 0.0f;
            ImGui::ProgressBar(done, ImVec2(ImGui::GetContentRegionAvail().x - cancel, 0.0f),
                               Fmt("%u / %u", d.frame, d.frames).c_str());
            ImGui::SameLine();
            if (ImGui::Button(Tr("Cancel"))) Queue("dataset.cancel");
        } else {
            json::Value names = json::Value::Array(), channels = json::Value::Array();
            for (const SensorItem& item : items)
                if (item.camera && std::find(f.cameras.begin(), f.cameras.end(), item.name) == f.cameras.end())
                    names.PushBack(json::Value::Str(item.name));
            for (size_t c = 0; c < kDatasetChannelCount; ++c)
                if (f.channels[c]) channels.PushBack(json::Value::Str(kDatasetChannels[c]));
            if (ImGui::Button(Iconed(icon::kDatabase, Tr("Export")).c_str()))
                Queue("dataset.export", Args({{"out", json::Value::Str(f.out)},
                                              {"frames", json::Value::Int(f.frames)},
                                              {"every", json::Value::Int(f.every)},
                                              {"seed", json::Value::Int(f.seed)},
                                              {"reset", json::Value::Bool(f.reset)},
                                              {"cameras", std::move(names)},
                                              {"channels", std::move(channels)},
                                              {"lidar", json::Value::Bool(f.lidar)},
                                              {"state", json::Value::Bool(f.state)},
                                              {"randomize", Args({{"render", json::Value::Bool(f.render)},
                                                                  {"physics", json::Value::Bool(f.physics)}})}}));
        }
        ImGui::PushTextWrapPos(0.0f);
        if (!d.running && !d.error.empty()) {
            ImGui::TextColored(theme::kErr, "%s", d.error.c_str());
        } else if (!d.running && d.frame > 0u) {
            const std::string done = d.cancelled
                                         ? Fmt(Tr("Cancelled after %u frames, written to %s"), d.frame, d.out.c_str())
                                         : Fmt(Tr("Wrote %u frames to %s"), d.frame, d.out.c_str());
            ImGui::TextColored(theme::kOk, "%s", done.c_str());
        }
        ImGui::PopTextWrapPos();
    }
    ImGui::End();
    if (!open) Queue("dataset.exporter", Args({{"on", json::Value::Bool(false)}}));
}

}  // namespace nuka::runtime::app::viewer
