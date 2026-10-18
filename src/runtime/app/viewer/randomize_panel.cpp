// The Randomize panel: render jitters, physics ranges and parameter ranges drawn from one seed,
// each edit sent as dr.set when its widget lets go.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/shell_util.hpp"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <climits>
#include <string>

namespace nuka::runtime::app::viewer {

namespace json = nuka::scene::json;
namespace theme = nuka::render::imgui::theme;
namespace icon = nuka::render::imgui::icon;
namespace draw = nuka::nui::draw;
using nuka::nui::Tr;
using namespace shell;

namespace {

// The panel's names for kRenderJitters and kPhysicsRanges, in their order.
constexpr const char* kJitterLabels[] = {"Base color",      "Roughness",   "Metallic", "Light direction",
                                         "Light intensity", "Light color", "Ambient"};
constexpr const char* kJitterTips[] = {"Added to each base color channel, up to this much either way.",
                                       "Added to roughness, up to this much either way.",
                                       "Added to metallic, up to this much either way.",
                                       "Added to each light direction component, up to this much either way.",
                                       "A light's intensity scales by up to this fraction either way.",
                                       "Each light color channel scales by up to this fraction either way.",
                                       "The ambient light scales by up to this fraction either way."};
constexpr const char* kRangeLabels[] = {"Mass", "Armature", "Gravity"};
constexpr const char* kRangeTips[] = {
    "Each link's mass times a factor drawn in [lo, hi].",
    "Each joint's armature plus an offset in kg*m^2 drawn in [lo, hi].",
    "Gravity's z plus one offset in m/s^2 drawn in [lo, hi], shared by every environment."};
constexpr const char* kRangeFormats[] = {"\xC3\x97%.2f", "%+.4f", "%+.2f"};
constexpr float kRangeSpeed[] = {0.005f, 0.0005f, 0.02f};
constexpr float kRangeMin[] = {0.01f, -1.0f, -20.0f};
constexpr float kRangeMax[] = {10.0f, 1.0f, 20.0f};

json::Value Pair(float lo, float hi) {
    json::Value a = json::Value::Array();
    a.PushBack(json::Value::Float(lo));
    a.PushBack(json::Value::Float(hi));
    return a;
}

// "joint_damping" reads "Joint damping".
std::string KeyLabel(std::string key) {
    std::replace(key.begin(), key.end(), '_', ' ');
    key[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(key[0])));
    return key;
}

}  // namespace

void ImGuiLayer::BuildRandomize(nui::VStack& root) {
    root.Add<nui::Custom>([this] { DrawRandomizePanel(); });
}

void ImGuiLayer::DrawRandomizePanel() {
    const RandomizeView& v = frame_.ui->randomize;
    if (frame_.scene == nullptr) {
        ImGui::TextDisabled("%s", Tr("Open a scene to randomize it."));
        return;
    }
    if (randomize_edit_version_ != v.version) {
        randomize_edit_ = v.config;
        randomize_edit_version_ = v.version;
        // Every parameter gets a row; one never turned on stays out of the settings.
        for (const std::string& key : v.param_keys)
            if (std::none_of(randomize_edit_.params.begin(), randomize_edit_.params.end(),
                             [&](const ParamRange& p) { return p.key == key; }))
                randomize_edit_.params.push_back(ParamRange{key, false});
    }
    RandomizeConfig& e = randomize_edit_;
    // The copy is taken again next frame, so a refused edit shows the settings as they stay.
    const auto commit = [this](json::Value fields) {
        Queue("dr.set", std::move(fields));
        randomize_edit_version_ = ~0ull;
    };
    const auto finished = [] { return ImGui::IsItemDeactivatedAfterEdit(); };
    // Labels take a share of the width, so ranges keep room in a narrow dock.
    const auto begin = [](const char* id) {
        if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) return false;
        ImGui::TableSetupColumn("##key", ImGuiTableColumnFlags_WidthStretch, 0.75f);
        ImGui::TableSetupColumn("##value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        return true;
    };
    const auto row = [](const char* label) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(theme::kFgMuted, "%s", label);
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
    };

    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(theme::kFgMuted, "%s", Tr("Seed"));
    ImGui::SameLine();
    const float button = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - button);
    int seed = static_cast<int>(e.seed);
    if (ImGui::DragInt("##seed", &seed, 0.2f, 0, INT_MAX)) e.seed = static_cast<uint32_t>(seed);
    if (finished()) commit(Args({{"seed", json::Value::Int(e.seed)}}));
    ImGui::SameLine();
    ImGui::BeginDisabled(!v.config.Active());
    if (draw::IconButton("##resample", icon::kShuffle, false, Tr("Resample"))) Queue("dr.sample");
    ImGui::EndDisabled();
    if (v.drawn)
        ImGui::TextColored(theme::kAccent, "%s", Fmt(Tr("drawn from seed %u"), v.config.seed).c_str());
    else
        ImGui::TextDisabled("%s", Tr("nominal values"));
    if (ImGui::Checkbox(Tr("Draw again on reset"), &e.reset_draws))
        commit(Args({{"reset_draws", json::Value::Bool(e.reset_draws)}}));
    ImGui::SetItemTooltip("%s", Tr("Each reset draws with the next seed; off, a reset keeps the draw."));

    draw::SectionHeader(Tr("Render"));
    ImGui::BeginDisabled(!v.camera);
    if (ImGui::Checkbox(Tr("Randomize what cameras see"), &e.render))
        commit(Args({{"render", Args({{"on", json::Value::Bool(e.render)}})}}));
    if (begin("##render")) {
        for (size_t k = 0; k < kJitterCount; ++k) {
            ImGui::PushID(static_cast<int>(k));
            row(Tr(kJitterLabels[k]));
            ImGui::DragFloat("##jitter", &e.jitter[k], 0.002f, 0.0f, 1.0f, "\xC2\xB1%.3f");
            if (finished()) commit(Args({{"render", Args({{kRenderJitters[k], json::Value::Float(e.jitter[k])}})}}));
            ImGui::SetItemTooltip("%s", Tr(kJitterTips[k]));
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::EndDisabled();
    if (!v.camera) ImGui::TextDisabled("%s", Tr("Add a camera sensor to randomize what it sees."));

    draw::SectionHeader(Tr("Physics"));
    ImGui::BeginDisabled(!v.articulated);
    if (ImGui::Checkbox(Tr("Randomize articulations"), &e.physics))
        commit(Args({{"physics", Args({{"on", json::Value::Bool(e.physics)}})}}));
    if (begin("##physics")) {
        for (size_t k = 0; k < kRangeCount; ++k) {
            ImGui::PushID(static_cast<int>(k));
            row(Tr(kRangeLabels[k]));
            float* range = &e.range[2u * k];
            ImGui::DragFloat2("##range", range, kRangeSpeed[k], kRangeMin[k], kRangeMax[k], kRangeFormats[k]);
            if (finished()) commit(Args({{"physics", Args({{kPhysicsRanges[k], Pair(range[0], range[1])}})}}));
            ImGui::SetItemTooltip("%s", Tr(kRangeTips[k]));
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::EndDisabled();
    if (!v.articulated) ImGui::TextDisabled("%s", Tr("Physics ranges act on articulated bodies; this scene has none."));

    draw::SectionHeader(Tr("Parameters"), Fmt("%zu", v.param_keys.size()).c_str());
    if (v.param_keys.empty()) {
        ImGui::TextDisabled("%s", Tr("This world has no per-link parameters to scale."));
    } else if (begin("##params")) {
        for (size_t i = 0; i < v.param_keys.size(); ++i) {
            const std::string& key = v.param_keys[i];
            ParamRange& p =
                *std::find_if(e.params.begin(), e.params.end(), [&](const ParamRange& r) { return r.key == key; });
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            if (ImGui::Checkbox(Tr(KeyLabel(key).c_str()), &p.on))
                commit(Args({{"params", Args({{key.c_str(), Args({{"on", json::Value::Bool(p.on)}})}})}}));
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            float range[2] = {p.lo, p.hi};
            if (ImGui::DragFloat2("##range", range, 0.005f, 0.01f, 10.0f, "\xC3\x97%.2f")) {
                p.lo = range[0];
                p.hi = range[1];
            }
            if (finished())
                commit(Args({{"params", Args({{key.c_str(), Args({{"on", json::Value::Bool(true)},
                                                                   {"range", Pair(p.lo, p.hi)}})}})}}));
            ImGui::SetItemTooltip(Tr("Each link's value, in %s, times its own factor drawn in [lo, hi]."),
                                  v.param_units[i].c_str());
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", Tr("Every environment draws its own values; the draw is part of the timeline."));
    ImGui::PopTextWrapPos();
    if (ImGui::Button(Iconed(icon::kDatabase, Tr("Export Dataset...")).c_str()))
        Queue("dataset.exporter", Args({{"on", json::Value::Bool(true)}}));
}

}  // namespace nuka::runtime::app::viewer
