// The run's progress plate over the scene viewport, under the HUD chips, and the Training panel: the
// task's settings, the runs and their logs, the curves of several runs at once and a run's saves.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/shell_util.hpp"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
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

constexpr const char kDash[] = "\xE2\x80\x93";

// The Dashboard's run colours: the selected run's, then each compared run's.
constexpr ImVec4 kRunColors[] = {theme::kAccent,       theme::kInfo,         theme::kWarn,
                                 theme::Hex(0xB48CF0), theme::Hex(0xF07AB4), theme::Hex(0x4FD1C5)};

struct PageSpec {
    TrainPage page;
    const char* glyph;
    const char* label;
};
constexpr PageSpec kPages[] = {{TrainPage::Task, icon::kListChecks, "Task"},
                               {TrainPage::Runs, icon::kRocket, "Runs"},
                               {TrainPage::Dashboard, icon::kTrendUp, "Dashboard"},
                               {TrainPage::Checkpoints, icon::kFlag, "Checkpoints"}};

// The Dashboard's curves over the epochs, each run's column of one metric.
struct CurveSpec {
    const char* title;
    std::vector<double> TrainSeries::*column;
};
constexpr CurveSpec kCurves[] = {{"Reward", &TrainSeries::reward},
                                 {"Episode length", &TrainSeries::length},
                                 {"Actor loss", &TrainSeries::actor_loss},
                                 {"Critic loss", &TrainSeries::critic_loss},
                                 {"Entropy", &TrainSeries::entropy},
                                 {"KL divergence", &TrainSeries::kl},
                                 {"Steps per second", &TrainSeries::fps},
                                 {"Learning rate", &TrainSeries::lr}};

// "tracking_lin_vel" reads "Tracking lin vel".
std::string KeyLabel(std::string key) {
    std::replace(key.begin(), key.end(), '_', ' ');
    key[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(key[0])));
    return key;
}

Tone StateTone(const std::string& state) {
    if (state == "running") return Tone::Accent;
    if (state == "starting") return Tone::Info;
    if (state == "finished") return Tone::Ok;
    if (state == "stopped") return Tone::Muted;
    return Tone::Err;
}

const TrainRunRow* FindRun(const std::vector<TrainRunRow>& runs, const std::string& name) {
    const auto at = std::find_if(runs.begin(), runs.end(), [&](const TrainRunRow& r) { return r.name == name; });
    return at == runs.end() ? nullptr : &*at;
}

// 950, 12.4K, 3.21M.
std::string Count(double v) {
    if (std::abs(v) >= 1e6) return Fmt("%.2fM", v / 1e6);
    if (std::abs(v) >= 1e3) return Fmt("%.1fK", v / 1e3);
    return Fmt("%.0f", v);
}

std::string Maybe(const std::optional<double>& v, const char* format) { return v ? Fmt(format, *v) : kDash; }

// A wall time on the local clock, with the date when it is not today.
std::string ClockText(double seconds) {
    if (seconds <= 0.0) return kDash;
    const std::time_t at = static_cast<std::time_t>(seconds), now = std::time(nullptr);
    std::tm when{}, today{};
#if defined(_WIN32)
    localtime_s(&when, &at);
    localtime_s(&today, &now);
#else
    localtime_r(&at, &when);
    localtime_r(&now, &today);
#endif
    char text[32];
    const bool same_day = when.tm_year == today.tm_year && when.tm_yday == today.tm_yday;
    std::strftime(text, sizeof(text), same_day ? "%H:%M" : "%m-%d %H:%M", &when);
    return text;
}

// 12s, 3m 05s, 1h 02m.
std::string SpanText(double seconds) {
    const long s = std::lround(std::max(seconds, 0.0));
    if (s < 60) return Fmt("%lds", s);
    if (s < 3600) return Fmt("%ldm %02lds", s / 60, s % 60);
    return Fmt("%ldh %02ldm", s / 3600, (s / 60) % 60);
}

std::string ValueText(const json::Value& v) {
    switch (v.type()) {
        case json::Value::Type::Bool: return Tr(v.AsBool() ? "on" : "off");
        case json::Value::Type::Int: return std::to_string(v.AsInt());
        case json::Value::Type::Float: return Fmt("%.4g", v.AsDouble());
        case json::Value::Type::String: return v.AsString();
        case json::Value::Type::Array: {
            std::string out;
            for (const json::Value& e : v.Elements()) out += (out.empty() ? "" : ", ") + ValueText(e);
            return out;
        }
        default: return kDash;
    }
}

void Wrapped(const ImVec4& color, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(color, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

// A section header whose click folds what follows; the state lives in ImGui's storage, `open` at first.
bool FoldHeader(const char* label, const std::string& trailing, bool open) {
    ImGuiStorage* storage = ImGui::GetStateStorage();
    const ImGuiID id = ImGui::GetID("##fold");
    const bool shown = storage->GetBool(id, open);
    const std::string tail =
        trailing + (trailing.empty() ? "" : "  ") + (shown ? icon::kChevronDown : icon::kChevronRight);
    draw::SectionHeader(label, tail.c_str());
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    ImGui::SetCursorScreenPos(a);
    if (ImGui::InvisibleButton("##fold", ImVec2(b.x - a.x, b.y - a.y))) storage->SetBool(id, !shown);
    return shown;
}

// Labelled figures in chamfered tiles, as many to a row as fit, the rows evened out.
void Tiles(const std::vector<std::pair<std::string, std::string>>& tiles) {
    const float s = nui::Scale();
    const int n = static_cast<int>(tiles.size());
    const int fit = std::clamp(static_cast<int>(ImGui::GetContentRegionAvail().x / (84.0f * s)), 1, n);
    const int rows = (n + fit - 1) / fit, columns = (n + rows - 1) / rows;
    if (!ImGui::BeginTable("##tiles", columns, ImGuiTableFlags_SizingStretchSame)) return;
    ImFont* font = GetNukaFont(FontRole::Display);
    const float label = theme::kFontLabel * s, value = theme::kFontHero * s, pad = 7.0f * s;
    const float h = label + value + pad * 2.5f;
    ImDrawList* list = ImGui::GetWindowDrawList();
    for (const auto& [name, figure] : tiles) {
        ImGui::TableNextColumn();
        const ImVec2 a = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        draw::Chamfer(list, a, ImVec2(a.x + w, a.y + h), 5.0f * s, U32(theme::kBg2), U32(theme::kLine));
        list->AddText(font, label, ImVec2(a.x + pad, a.y + pad), U32(theme::kFgDim), Upper(name).c_str());
        list->AddText(font, value, ImVec2(a.x + pad, a.y + pad * 1.5f + label), U32(theme::kFg), figure.c_str());
        ImGui::Dummy(ImVec2(w, h));
    }
    ImGui::EndTable();
}

// A parameter's help, unit, default and path while its label is hovered.
void ParamTip(const json::Value& param, const std::string& path) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
    const std::string& help = param.At("help").AsString();
    if (!help.empty()) ImGui::TextUnformatted(Tr(help).c_str());
    const std::string& unit = param.At("unit").AsString();
    const std::string fallback = ValueText(param.At("default")) + (unit.empty() ? "" : " " + unit);
    ImGui::TextColored(theme::kFgMuted, "%s", Fmt(Tr("Default %s"), fallback.c_str()).c_str());
    ImGui::TextColored(theme::kFgDim, "%s", path.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

// A button that stays lit while `on`.
bool Toggle(const std::string& label, bool on, float width) {
    if (on) {
        const ImVec4 c = nui::ToneColor(Tone::Accent);
        ImGui::PushStyleColor(ImGuiCol_Button, theme::Alpha(c, 0.18f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, theme::Alpha(c, 0.28f));
        ImGui::PushStyleColor(ImGuiCol_Text, c);
    }
    const bool pressed = ImGui::Button(label.c_str(), ImVec2(width, 0.0f));
    if (on) ImGui::PopStyleColor(3);
    return pressed;
}

}  // namespace

void ImGuiLayer::DrawTrainHud(const ImVec2& vmin, const ImVec2& vmax, float hud_top) {
    const TrainView& t = frame_.ui->train;
    if (!t.observing) return;
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    list->PushClipRect(vmin, vmax, true);
    const float s = nui::Scale();
    ImFont* font = GetNukaFont(FontRole::Display);
    const float px = theme::kFontLabel * s, line = font->CalcTextSizeA(px, FLT_MAX, 0.0f, "M").y;
    const std::string title = Fmt(Tr("TRAINING  %s"), t.run.c_str());
    const std::string reward = t.has_reward ? Fmt("%.2f", t.reward) : std::string("-");
    const std::string stats =
        Fmt(Tr("EPOCH %lld / %lld"), static_cast<long long>(t.epoch), static_cast<long long>(t.max_epochs)) + "   " +
        Fmt(Tr("REWARD %s"), reward.c_str()) + "   " + Fmt(Tr("%.1fK STEPS/S"), t.fps / 1000.0) + "   " +
        Fmt(Tr("%u OF %u ENVS"), t.envs, t.run_envs);
    // The legend reads the current environment's planar speeds off its arrows.
    const auto speed = [&](bool command) {
        if (t.current >= t.arrows.size()) return 0.0f;
        const float* v = command ? t.arrows[t.current].command : t.arrows[t.current].velocity;
        return std::hypot(v[0], v[1]);
    };
    const std::string command = Fmt(Tr("COMMAND %.2f M/S"), speed(true));
    const std::string velocity = Fmt(Tr("VELOCITY %.2f M/S"), speed(false));
    const float swatch = 14.0f * s, gap = 6.0f * s, pad = 10.0f * s;
    const float legend = swatch * 2.0f + gap * 6.0f + font->CalcTextSizeA(px, FLT_MAX, 0.0f, command.c_str()).x +
                         font->CalcTextSizeA(px, FLT_MAX, 0.0f, velocity.c_str()).x;
    const float width = std::max({font->CalcTextSizeA(px, FLT_MAX, 0.0f, title.c_str()).x,
                                  font->CalcTextSizeA(px, FLT_MAX, 0.0f, stats.c_str()).x, legend});
    const ImVec2 p0(vmin.x + 18.0f * s, vmin.y + 30.0f * s + hud_top + line);
    const ImVec2 p1(p0.x + width + pad * 2.0f, p0.y + line * 3.0f + gap * 2.0f + pad * 2.0f);
    draw::Chamfer(list, p0, p1, 4.0f * s, U32(theme::Alpha(theme::kBg0, 0.82f)),
                  U32(theme::Alpha(theme::kOk, 0.55f)));
    list->AddRectFilled(ImVec2(p0.x, p0.y + 6.0f * s), ImVec2(p0.x + 2.0f * s, p1.y - 6.0f * s), U32(theme::kOk));
    float y = p0.y + pad;
    list->AddText(font, px, ImVec2(p0.x + pad, y), U32(theme::kAccent), title.c_str());
    y += line + gap;
    list->AddText(font, px, ImVec2(p0.x + pad, y), U32(theme::kFgMuted), stats.c_str());
    y += line + gap;
    float x = p0.x + pad;
    for (const auto& [label, color] :
         {std::pair{command.c_str(), theme::kOk}, std::pair{velocity.c_str(), theme::kInfo}}) {
        const float mid = y + line * 0.5f;
        list->AddRectFilled(ImVec2(x, mid - 2.0f * s), ImVec2(x + swatch, mid + 2.0f * s), U32(color));
        x += swatch + gap;
        list->AddText(font, px, ImVec2(x, y), U32(theme::kFgMuted), label);
        x += font->CalcTextSizeA(px, FLT_MAX, 0.0f, label).x + gap * 2.0f;
    }
    list->PopClipRect();
}

// ---- training panel -----------------------------------------------------------------------

void ImGuiLayer::BuildTraining(nui::VStack& root) {
    root.Add<nui::Custom>([this] {
        TrainPanel& p = frame_.ui->training;
        p.shown = true;
        const float s = nui::Scale();
        // The page chips drop their labels when the four do not fit on one line.
        float wide = 3.0f * 4.0f * s;
        for (const auto& page : kPages)
            wide += ImGui::CalcTextSize(Iconed(page.glyph, Tr(page.label)).c_str()).x + 16.0f * s;
        const bool labels = wide <= ImGui::GetContentRegionAvail().x;
        for (size_t i = 0; i < std::size(kPages); ++i) {
            const std::string text = labels ? Iconed(kPages[i].glyph, Tr(kPages[i].label)) : kPages[i].glyph;
            if (i > 0) ImGui::SameLine(0.0f, 4.0f * s);
            ImGui::PushID(static_cast<int>(i));
            if (draw::Chip("page", text, p.page == kPages[i].page)) p.page = kPages[i].page;
            if (!labels) ImGui::SetItemTooltip("%s", Tr(kPages[i].label));
            ImGui::PopID();
        }
        if (!labels) {
            ImGui::SameLine(0.0f, 8.0f * s);
            ImGui::AlignTextToFramePadding();
            nui::PushTextStyle(nui::TextStyle::Strong);
            ImGui::TextUnformatted(Tr(kPages[static_cast<size_t>(p.page)].label));
            nui::PopTextStyle();
        }
        // The run this editor trains shows on every page.
        if (const TrainRunRow* live = FindRun(p.runs, p.training)) {
            ImGui::AlignTextToFramePadding();
            draw::StatusDot(Tone::Accent, true);
            ImGui::SameLine();
            ImGui::TextColored(theme::kFgMuted, "%s",
                               Fmt(Tr("%s trains: epoch %lld of %lld"), live->name.c_str(),
                                   static_cast<long long>(live->epoch), static_cast<long long>(live->max_epochs))
                                   .c_str());
        }
        ImGui::Spacing();
        switch (p.page) {
            case TrainPage::Task: DrawTrainTask(); break;
            case TrainPage::Runs: DrawTrainRuns(); break;
            case TrainPage::Dashboard: DrawTrainDashboard(); break;
            case TrainPage::Checkpoints: DrawTrainSaves(); break;
        }
    });
}

void ImGuiLayer::TrainRunPicker() {
    TrainPanel& p = frame_.ui->training;
    const auto label = [](const TrainRunRow& r) { return r.name + "   " + Tr(r.state) + "###" + r.name; };
    const TrainRunRow* run = FindRun(p.runs, p.selected);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(theme::kFgMuted, "%s", Tr("Run"));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    const std::string shown = run != nullptr ? run->name + "   " + Tr(run->state) : std::string(Tr("No runs yet"));
    if (ImGui::BeginCombo("##run_pick", shown.c_str())) {
        for (const TrainRunRow& r : p.runs)
            if (ImGui::Selectable(label(r).c_str(), r.name == p.selected)) p.selected = r.name;
        ImGui::EndCombo();
    }
}

void ImGuiLayer::DrawTrainTask() {
    const TrainPanel& p = frame_.ui->training;
    if (frame_.scene == nullptr) {
        ImGui::TextDisabled("%s", Tr("Open a scene with a robot to train it."));
        return;
    }
    if (!p.config_error.empty()) {
        Wrapped(theme::kErr, p.config_error);
        return;
    }
    if (!p.config.IsObject()) {
        ImGui::TextDisabled("%s", Tr("Reading the task..."));
        return;
    }
    if (train_edit_version_ != p.config_version) {
        train_edit_.clear();
        for (const auto& [group, values] : p.config.At("config").Items())
            for (const auto& [key, value] : values.Items()) train_edit_[group + "." + key] = value;
        train_edit_version_ = p.config_version;
    }
    const float s = nui::Scale();
    const std::string& task = p.config.At("task").AsString();
    const json::Value& changed = p.config.At("changed");
    const json::Value& errors = p.config.At("errors");
    const json::Value* schema = nullptr;
    for (const json::Value& t : p.tasks.Elements())
        if (t.At("name").AsString() == task) schema = &t;
    // A refused edit is answered by a toast; the copy is taken again so the field shows what stays.
    const auto commit = [this](const std::string& path, json::Value value) {
        Queue("train.set", Args({{"path", json::Value::Str(path)}, {"value", std::move(value)}}));
        train_edit_version_ = ~0ull;
    };

    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(theme::kFgMuted, "%s", Tr("Task"));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##task", schema != nullptr ? Tr(schema->At("title").AsString()).c_str() : task.c_str())) {
        for (const json::Value& t : p.tasks.Elements()) {
            const std::string& name = t.At("name").AsString();
            if (ImGui::Selectable(Tr(t.At("title").AsString()).c_str(), name == task) && name != task)
                Queue("train.set", Args({{"task", json::Value::Str(name)}}));
            ImGui::SetItemTooltip("%s  \xC2\xB7  %s", name.c_str(), t.At("env").AsString().c_str());
        }
        ImGui::EndCombo();
    }
    if (schema != nullptr && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
        std::string robot;
        for (const auto& [key, count] : schema->At("robot").Items())
            robot += (robot.empty() ? "" : ", ") + Fmt("%lld ", static_cast<long long>(count.AsInt())) + Tr(key);
        ImGui::SetTooltip("%s", Fmt(Tr("One robot (%s) per environment, stepped at %.0f Hz"), robot.c_str(),
                                    1.0 / schema->At("dt").AsDouble()).c_str());
    }
    const size_t problems = errors.Size();
    if (problems == 0u)
        draw::Badge(Tr("Ready to train"), Tone::Ok, icon::kOk);
    else
        draw::Badge(Fmt(Tr(problems == 1u ? "1 problem" : "%zu problems"), problems).c_str(), Tone::Err, icon::kWarn);
    // Problems of the scene, or of no row, read here; a row's own reads under it.
    for (const json::Value& e : errors.Elements())
        if (train_edit_.count(e.At("path").AsString()) == 0u)
            Wrapped(theme::kErr, Iconed(icon::kWarn, e.At("message").AsString().c_str()));
    if (schema == nullptr) return;

    for (const json::Value& group : schema->At("groups").Elements()) {
        const std::string& gkey = group.At("key").AsString();
        const json::Value& params = group.At("params");
        const json::Value* edited = changed.Find(gkey);
        bool terms = true;
        int64_t dims = 0;
        for (const json::Value& param : params.Elements()) {
            terms = terms && param.At("kind").AsString() == "term";
            dims += param.At("dims").AsInt();
        }
        const std::string trailing = terms ? Fmt(Tr("%lld values"), static_cast<long long>(dims))
                                     : edited != nullptr ? Fmt(Tr("%zu changed"), edited->Items().size())
                                                         : std::string();
        ImGui::PushID(gkey.c_str());
        // The deployment contract's groups start folded.
        const bool open = FoldHeader(Tr(group.At("title").AsString()).c_str(), trailing, !terms && gkey != "action");
        if (open && ImGui::BeginTable("##params", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("##key", ImGuiTableColumnFlags_WidthStretch, 0.85f);
            ImGui::TableSetupColumn("##value", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            for (const json::Value& param : params.Elements()) {
                const std::string& key = param.At("key").AsString();
                const std::string& kind = param.At("kind").AsString();
                const std::string path = gkey + "." + key;
                const auto at = train_edit_.find(path);
                if (at == train_edit_.end()) continue;
                json::Value& v = at->second;
                const bool readonly = param.At("readonly").AsBool();
                const bool differs = edited != nullptr && edited->Find(key) != nullptr;
                const bool faulty = std::any_of(errors.Elements().begin(), errors.Elements().end(),
                                                [&](const json::Value& e) { return e.At("path").AsString() == path; });
                ImGui::PushID(key.c_str());
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                // A bar marks a value off its default, red when it keeps the run from starting.
                const ImVec2 mark = ImGui::GetCursorScreenPos();
                if (differs || faulty)
                    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(mark.x, mark.y + 3.0f * s),
                                                              ImVec2(mark.x + 2.0f * s,
                                                                     mark.y + ImGui::GetFrameHeight() - 3.0f * s),
                                                              U32(faulty ? theme::kErr : theme::kAccent));
                ImGui::SetCursorScreenPos(ImVec2(mark.x + 7.0f * s, mark.y));
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(differs ? theme::kFg : theme::kFgMuted, "%s", Tr(KeyLabel(key)).c_str());
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) ParamTip(param, path);
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                const float lo = param.At("min").IsNull() ? 0.0f : param.At("min").AsFloat();
                const float hi = param.At("max").IsNull() ? 0.0f : param.At("max").AsFloat();
                const std::string& unit = param.At("unit").AsString();
                const std::string suffix = unit.empty() ? std::string() : " " + unit;
                // Drags move a hundredth of the value per pixel.
                const auto speed = [](float x) { return std::max(std::abs(x) * 0.01f, 1e-4f); };
                const auto finished = [] { return ImGui::IsItemDeactivatedAfterEdit(); };
                if (kind == "term") {
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextColored(theme::kFgDim, "%s",
                                       Fmt("%lld \xC3\x97 %.3g", static_cast<long long>(param.At("dims").AsInt()),
                                           param.At("default").AsDouble())
                                           .c_str());
                } else if (readonly) {
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextColored(theme::kFgDim, "%s  %s", icon::kLock, (ValueText(v) + suffix).c_str());
                } else if (kind == "bool") {
                    bool b = v.AsBool();
                    if (ImGui::Checkbox("##v", &b)) commit(path, json::Value::Bool(b));
                } else if (kind == "int") {
                    int x = static_cast<int>(v.AsInt());
                    if (ImGui::DragInt("##v", &x, std::max(0.1f, speed(static_cast<float>(x))), static_cast<int>(lo),
                                       static_cast<int>(hi), ("%d" + suffix).c_str(), ImGuiSliderFlags_AlwaysClamp))
                        v = json::Value::Int(x);
                    if (finished()) commit(path, v);
                } else if (kind == "float") {
                    float x = v.AsFloat();
                    if (ImGui::DragFloat("##v", &x, speed(x), lo, hi, ("%.4g" + suffix).c_str(),
                                         ImGuiSliderFlags_AlwaysClamp))
                        v = json::Value::Float(x);
                    if (finished()) commit(path, v);
                } else if (kind == "range" || kind == "vec3") {
                    float x[3] = {0.0f, 0.0f, 0.0f};
                    const size_t n = std::min<size_t>(v.Size(), 3u);
                    float step = 1e-4f;
                    for (size_t k = 0; k < n; ++k) {
                        x[k] = v.Elements()[k].AsFloat();
                        step = std::max(step, speed(x[k]));
                    }
                    // Too narrow for units; the tooltip has them.
                    const bool moved = n == 2u ? ImGui::DragFloat2("##v", x, step, lo, hi, "%.3g")
                                               : ImGui::DragFloat3("##v", x, step, lo, hi, "%.3g");
                    if (moved) {
                        json::Value list = json::Value::Array();
                        for (size_t k = 0; k < n; ++k) list.PushBack(json::Value::Float(x[k]));
                        v = std::move(list);
                    }
                    if (finished()) commit(path, v);
                } else if (kind == "enum") {
                    if (ImGui::BeginCombo("##v", ValueText(v).c_str())) {
                        for (const json::Value& choice : param.At("choices").Elements())
                            if (ImGui::Selectable(ValueText(choice).c_str(), ValueText(choice) == ValueText(v)))
                                commit(path, choice);
                        ImGui::EndCombo();
                    }
                } else if (kind == "ints") {
                    // Typed as numbers apart; the text stays in the copy until the field lets go.
                    char text[128];
                    std::snprintf(text, sizeof(text), "%s",
                                  v.type() == json::Value::Type::String ? v.AsString().c_str() : ValueText(v).c_str());
                    if (ImGui::InputText("##v", text, sizeof(text))) v = json::Value::Str(text);
                    if (finished()) {
                        json::Value list = json::Value::Array();
                        for (const char* c = text; *c != '\0';) {
                            char* end = nullptr;
                            const long n = std::strtol(c, &end, 10);
                            if (end == c) {
                                ++c;
                                continue;
                            }
                            list.PushBack(json::Value::Int(n));
                            c = end;
                        }
                        commit(path, std::move(list));
                    }
                }
                if (!readonly && ImGui::BeginPopupContextItem("##param_menu")) {
                    if (ImGui::MenuItemEx(Tr("Reset to Default"), icon::kReset, nullptr, false, differs))
                        commit(path, param.At("default"));
                    if (ImGui::MenuItemEx(Tr("Copy Path"), icon::kCopy)) ImGui::SetClipboardText(path.c_str());
                    ImGui::EndPopup();
                }
                for (const json::Value& e : errors.Elements()) {
                    if (e.At("path").AsString() != path) continue;
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(1);
                    Wrapped(theme::kErr, e.At("message").AsString());
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        ImGui::PopID();
    }
}

void ImGuiLayer::DrawTrainRuns() {
    TrainPanel& p = frame_.ui->training;
    const TrainView& watch = frame_.ui->train;
    const float s = nui::Scale();
    const ImGuiStyle& style = ImGui::GetStyle();

    // The interpreter runs start in; Enter checks the one typed.
    draw::SectionHeader(Tr("Trainer"));
    const bool probed = p.python.IsObject();
    const bool ready = probed && p.python.At("ready").AsBool();
    const std::string python = probed ? p.python.At("python").AsString() : std::string();
    if (python != train_python_seed_) {
        train_python_seed_ = python;
        std::snprintf(train_python_, sizeof(train_python_), "%s", python.c_str());
    }
    ImGui::AlignTextToFramePadding();
    draw::StatusDot(ready ? Tone::Ok : probed ? Tone::Err : Tone::Dim);
    ImGui::SameLine();
    const float button = ImGui::GetFrameHeight() + 2.0f * s + style.ItemSpacing.x;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - button);
    if (ImGui::InputTextWithHint("##python", Tr("Python with torch and rl_games"), train_python_, sizeof(train_python_),
                                 ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_ElideLeft))
        Queue("train.python", Args({{"path", json::Value::Str(train_python_)}}));
    ImGui::SetItemTooltip("%s",
                          Tr("The interpreter runs start in; Enter checks it, and empty takes the editor's own."));
    ImGui::SameLine();
    if (draw::IconButton("##probe", icon::kReset, false, Tr("Check Again"))) Queue("train.python");
    if (!probed)
        ImGui::TextDisabled("%s", Tr("Checking..."));
    else if (!ready)
        Wrapped(theme::kErr, p.python.At("error").AsString());

    // A new run of the scene as it stands, with the Task page's settings.
    draw::SectionHeader(Tr("New Run"));
    std::string blocked;
    if (frame_.scene == nullptr)
        blocked = Tr("Open a scene first.");
    else if (!p.training.empty())
        blocked = Fmt(Tr("Run '%s' is training; stop it first."), p.training.c_str());
    else if (!p.config.IsObject())
        blocked = Tr("The task's settings are not read yet.");
    else if (p.config.At("errors").Size() > 0u)
        blocked = Tr("Fix the problems on the Task page first.");
    else if (!ready)
        blocked = Tr("Choose a Python that can train.");
    const std::string start = Iconed(icon::kRocket, Tr("Start"));
    const float start_w = ImGui::CalcTextSize(start.c_str()).x + style.FramePadding.x * 2.0f;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - start_w - style.ItemSpacing.x);
    ImGui::InputTextWithHint("##run_name", Tr("Run name"), train_run_name_, sizeof(train_run_name_));
    ImGui::SetItemTooltip("%s", Tr("Left empty, the run is named by the time it starts."));
    ImGui::SameLine();
    ImGui::BeginDisabled(!blocked.empty());
    if (ImGui::Button(start.c_str())) {
        json::Value args = json::Value::Object();
        if (train_run_name_[0] != '\0') args.Set("name", json::Value::Str(train_run_name_));
        Queue("train.start", std::move(args));
        train_run_name_[0] = '\0';
    }
    ImGui::EndDisabled();
    if (!blocked.empty()) {
        ImGui::TextDisabled("%s", blocked.c_str());
    } else if (const json::Value* run = p.config.At("config").Find("run")) {
        const auto whole = [&](const char* key) {
            const json::Value* v = run->Find(key);
            return v != nullptr ? static_cast<long long>(v->AsInt()) : 0ll;
        };
        ImGui::TextDisabled("%s", Fmt(Tr("%lld robots for %lld epochs, saving every %lld"), whole("num_envs"),
                                      whole("max_epochs"), whole("save_every"))
                                      .c_str());
    }

    if (const TrainRunRow* run = FindRun(p.runs, p.selected)) {
        draw::SectionHeader(Tr("Selected Run"));
        const Tone tone = StateTone(run->state);
        ImGui::AlignTextToFramePadding();
        draw::StatusDot(tone, run->alive);
        ImGui::SameLine();
        nui::PushTextStyle(TextStyle::Strong);
        ImGui::TextUnformatted(run->name.c_str());
        nui::PopTextStyle();
        ImGui::SameLine();
        draw::Badge(Tr(run->state).c_str(), tone);
        const float done =
            run->max_epochs > 0 ? static_cast<float>(run->epoch) / static_cast<float>(run->max_epochs) : 0.0f;
        draw::ThinProgress(run->state == "starting" ? -1.0f : done, ImGui::GetContentRegionAvail().x, tone);
        ImGui::Spacing();
        Tiles({{Tr("Epoch"),
                Fmt("%lld/%lld", static_cast<long long>(run->epoch), static_cast<long long>(run->max_epochs))},
               {Tr("Reward"), Maybe(run->reward, "%.2f")},
               {Tr("Best"), Maybe(run->best, "%.2f")},
               {Tr("Steps/s"), run->fps ? Count(*run->fps) : std::string(kDash)}});
        if (run->alive) {
            // Stop, Watch and Follow share the row.
            const bool watching = watch.observing && watch.run == run->name;
            const float gap = style.ItemSpacing.x;
            const float w = (ImGui::GetContentRegionAvail().x - 2.0f * gap) / 3.0f;
            if (ImGui::Button(Iconed(icon::kStop, Tr("Stop")).c_str(), ImVec2(w, 0.0f)))
                Queue("train.stop", Args({{"name", json::Value::Str(run->name)}}));
            ImGui::SetItemTooltip("%s", Tr("Ends the run after its current epoch, with a final save."));
            ImGui::SameLine(0.0f, gap);
            ImGui::BeginDisabled(!watching && !run->serving);
            if (Toggle(Iconed(icon::kVisual, Tr("Watch")), watching, w))
                Queue("train.observe", watching ? Args({{"on", json::Value::Bool(false)}})
                                                : Args({{"on", json::Value::Bool(true)},
                                                        {"name", json::Value::Str(run->name)},
                                                        {"envs", json::Value::Int(kMaxGridEnvs)}}));
            ImGui::EndDisabled();
            ImGui::SetItemTooltip("%s", Tr("Shows 16 of its robots in the viewport, under arrows of their commanded "
                                           "and actual velocity."));
            ImGui::SameLine(0.0f, gap);
            ImGui::BeginDisabled(!watching);
            if (Toggle(Iconed(icon::kCamera, Tr("Follow")), watching && watch.follow, w))
                Queue("train.observe", Args({{"follow", json::Value::Bool(!watch.follow)}}));
            ImGui::EndDisabled();
            ImGui::SetItemTooltip("%s", Tr("Keeps the camera on the first watched robot."));
        }
        if (!run->error.empty()) Wrapped(theme::kErr, run->error);
    }

    draw::SectionHeader(Tr("Runs"), Fmt("%zu", p.runs.size()).c_str());
    if (p.runs.empty()) {
        ImGui::TextDisabled("%s", Tr("No runs yet; start one above."));
    } else if (ImGui::BeginTable("##runs", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn(Tr("Run"), ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn(Tr("Epoch"), ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn(Tr("Reward"), ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn(Tr("Best"), ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn(Tr("Started"), ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableHeadersRow();
        for (const TrainRunRow& r : p.runs) {
            ImGui::PushID(r.name.c_str());
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const ImVec2 dot(at.x + 4.0f * s, at.y + ImGui::GetTextLineHeight() * 0.5f);
            ImGui::GetWindowDrawList()->AddCircleFilled(dot, 3.5f * s, U32(nui::ToneColor(StateTone(r.state))));
            ImGui::SetCursorScreenPos(ImVec2(at.x + 12.0f * s, at.y));
            if (ImGui::Selectable(r.name.c_str(), r.name == p.selected,
                                  ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap))
                p.selected = r.name;
            const auto compared = std::find(p.compare.begin(), p.compare.end(), r.name);
            const char* compare_item = compared != p.compare.end() ? "Remove from Comparison" : "Compare on Dashboard";
            if (ImGui::BeginPopupContextItem("##run_menu")) {
                if (r.name != p.selected && ImGui::MenuItemEx(Tr(compare_item), icon::kTrendUp)) {
                    if (compared != p.compare.end())
                        p.compare.erase(compared);
                    else
                        p.compare.push_back(r.name);
                }
                if (ImGui::MenuItemEx(Tr("Stop"), icon::kStop, nullptr, false, r.alive))
                    Queue("train.stop", Args({{"name", json::Value::Str(r.name)}}));
                if (ImGui::MenuItemEx(Tr("Show in Folder"), icon::kOpen))
                    Queue("train.reveal", Args({{"name", json::Value::Str(r.name)}}));
                ImGui::EndPopup();
            }
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(theme::kFgMuted, "%lld/%lld", static_cast<long long>(r.epoch),
                               static_cast<long long>(r.max_epochs));
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(Maybe(r.reward, "%.2f").c_str());
            ImGui::TableSetColumnIndex(3);
            ImGui::TextUnformatted(Maybe(r.best, "%.2f").c_str());
            ImGui::TableSetColumnIndex(4);
            ImGui::TextColored(theme::kFgMuted, "%s", ClockText(r.started).c_str());
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    if (p.selected.empty()) return;
    draw::SectionHeader(Tr("Log"), p.selected.c_str());
    const float h = std::max(140.0f * s, ImGui::GetContentRegionAvail().y);
    if (ImGui::BeginChild("##run_log", ImVec2(-FLT_MIN, h), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_HorizontalScrollbar)) {
        const bool bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f;
        nui::PushTextStyle(TextStyle::Mono);
        ImGui::TextUnformatted(p.log.c_str());
        nui::PopTextStyle();
        if (bottom) ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
}

void ImGuiLayer::DrawTrainDashboard() {
    TrainPanel& p = frame_.ui->training;
    const float s = nui::Scale();
    TrainRunPicker();
    const TrainRunRow* run = FindRun(p.runs, p.selected);
    if (run == nullptr || p.series.empty()) {
        ImGui::TextDisabled("%s", Tr("No runs yet; start one on the Runs page."));
        return;
    }
    // The runs drawn in their colours; a compared one leaves by its button, another joins from the combo.
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    ImDrawList* list = ImGui::GetWindowDrawList();
    for (size_t i = 0; i < p.series.size(); ++i) {
        const std::string& name = p.series[i].run;
        const float w = 18.0f * s + ImGui::CalcTextSize(name.c_str()).x + (i > 0 ? ImGui::GetFrameHeight() : 0.0f);
        if (i > 0) {
            ImGui::SameLine(0.0f, 10.0f * s);
            if (ImGui::GetCursorScreenPos().x + w > right) ImGui::NewLine();
        }
        ImGui::PushID(static_cast<int>(i));
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float mid = at.y + ImGui::GetFrameHeight() * 0.5f;
        list->AddRectFilled(ImVec2(at.x, mid - 2.0f * s), ImVec2(at.x + 12.0f * s, mid + 2.0f * s),
                            U32(kRunColors[i % std::size(kRunColors)]));
        ImGui::SetCursorScreenPos(ImVec2(at.x + 16.0f * s, at.y));
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(name.c_str());
        if (i > 0) {
            ImGui::SameLine(0.0f, 2.0f * s);
            if (draw::IconButton("##drop", icon::kClose, false, Tr("Remove from Comparison")))
                p.compare.erase(std::find(p.compare.begin(), p.compare.end(), name));
        }
        ImGui::PopID();
    }
    ImGui::SameLine(0.0f, 10.0f * s);
    const float combo = 150.0f * s;
    if (ImGui::GetCursorScreenPos().x + combo > right) ImGui::NewLine();
    ImGui::SetNextItemWidth(combo);
    if (ImGui::BeginCombo("##compare", Tr("Compare with..."))) {
        for (const TrainRunRow& r : p.runs) {
            if (r.name == p.selected || std::find(p.compare.begin(), p.compare.end(), r.name) != p.compare.end())
                continue;
            if (ImGui::Selectable(r.name.c_str())) p.compare.push_back(r.name);
        }
        ImGui::EndCombo();
    }

    const double elapsed = run->started <= 0.0 ? 0.0
                           : run->ended > 0.0  ? run->ended - run->started
                                               : static_cast<double>(std::time(nullptr)) - run->started;
    Tiles({{Tr("Epoch"), Fmt("%lld/%lld", static_cast<long long>(run->epoch), static_cast<long long>(run->max_epochs))},
           {Tr("Reward"), Maybe(run->reward, "%.2f")},
           {Tr("Best"), Maybe(run->best, "%.2f")},
           {Tr("Steps/s"), run->fps ? Count(*run->fps) : std::string(kDash)},
           {Tr("Frames"), Count(static_cast<double>(run->frames))},
           {Tr("Time"), SpanText(elapsed)}});

    // Two plots abreast once the panel is wide enough; the selected run draws last, on top.
    const int columns = ImGui::GetContentRegionAvail().x > 560.0f * s ? 2 : 1;
    if (!ImGui::BeginTable("##curves", columns, ImGuiTableFlags_SizingStretchSame)) return;
    for (const CurveSpec& curve : kCurves) {
        ImGui::TableNextColumn();
        ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(theme::kSpace * s, theme::kSpace * s));
        if (ImPlot::BeginPlot(Tr(curve.title), ImVec2(-FLT_MIN, 150.0f * s),
                              ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoLegend |
                                  ImPlotFlags_NoMouseText)) {
            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            for (size_t i = p.series.size(); i-- > 0;) {
                const TrainSeries& r = p.series[i];
                const std::vector<double>& ys = r.*curve.column;
                ImPlotSpec spec;
                spec.LineColor = kRunColors[i % std::size(kRunColors)];
                spec.LineWeight = (i == 0u ? 2.0f : 1.5f) * s;
                spec.Flags = ImPlotLineFlags_SkipNaN;
                // Short runs show their points.
                if (ys.size() <= 40u) {
                    spec.Marker = ImPlotMarker_Circle;
                    spec.MarkerSize = 2.5f * s;
                }
                ImPlot::PlotLine(r.run.c_str(), r.epoch.data(), ys.data(), static_cast<int>(ys.size()), spec);
            }
            if (ImPlot::IsPlotHovered()) {
                const double epoch = std::round(ImPlot::GetPlotMousePos().x);
                ImGui::BeginTooltip();
                ImGui::TextColored(theme::kFgMuted, "%s", Fmt(Tr("Epoch %.0f"), epoch).c_str());
                for (size_t i = 0; i < p.series.size(); ++i) {
                    const TrainSeries& r = p.series[i];
                    const auto row = std::find(r.epoch.begin(), r.epoch.end(), epoch);
                    if (row == r.epoch.end()) continue;
                    ImGui::TextColored(kRunColors[i % std::size(kRunColors)], "%s  %.4g", r.run.c_str(),
                                       (r.*curve.column)[static_cast<size_t>(row - r.epoch.begin())]);
                }
                ImGui::EndTooltip();
            }
            ImPlot::EndPlot();
        }
        ImPlot::PopStyleVar();
    }
    ImGui::EndTable();
}

void ImGuiLayer::DrawTrainSaves() {
    const TrainPanel& p = frame_.ui->training;
    const float s = nui::Scale();
    TrainRunPicker();
    if (p.selected.empty()) {
        ImGui::TextDisabled("%s", Tr("No runs yet; start one on the Runs page."));
        return;
    }
    const bool scene = frame_.scene != nullptr;
    const bool best = std::any_of(p.saves.begin(), p.saves.end(), [](const TrainSave& v) { return v.kind == "best"; });
    const auto play = [this, &p](const char* kind) {
        Queue("train.play", Args({{"name", json::Value::Str(p.selected)}, {"kind", json::Value::Str(kind)}}));
    };
    const auto tip = [] {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", Tr("Playing loads a save's policy onto the scene's robot and runs the simulation."));
    };
    ImGui::BeginDisabled(!scene || !best);
    if (ImGui::Button(Iconed(icon::kTrophy, Tr("Play Best")).c_str())) play("best");
    tip();
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!scene || p.saves.empty());
    if (ImGui::Button(Iconed(icon::kPlay, Tr("Play Latest")).c_str())) play("last");
    tip();
    ImGui::EndDisabled();
    draw::SectionHeader(Tr("Saves"), Fmt("%zu", p.saves.size()).c_str());
    if (p.saves.empty()) {
        ImGui::TextDisabled("%s", Tr("No saves yet; a run saves every few epochs and when it ends."));
        return;
    }
    // Badges, epochs and times keep their widths; reward and frames share the rest.
    if (!ImGui::BeginTable("##saves", 6, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) return;
    ImGui::TableSetupColumn(Tr("Kind"), ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn(Tr("Epoch"), ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn(Tr("Reward"), ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn(Tr("Frames"), ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn(Tr("Saved at"), ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("##play", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFrameHeight() + 2.0f * s);
    ImGui::TableHeadersRow();
    int row = 0;
    for (auto save = p.saves.rbegin(); save != p.saves.rend(); ++save, ++row) {
        ImGui::PushID(row);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        if (save->kind == "best")
            draw::Badge(Tr("best"), Tone::Accent);
        else if (save->kind == "final")
            draw::Badge(Tr("final"), Tone::Ok);
        else
            draw::Badge(Tr(save->kind).c_str(), Tone::Muted);
        ImGui::TableSetColumnIndex(1);
        ImGui::AlignTextToFramePadding();
        ImGui::Text("%lld", static_cast<long long>(save->epoch));
        ImGui::TableSetColumnIndex(2);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(Maybe(save->reward, "%.2f").c_str());
        ImGui::TableSetColumnIndex(3);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(theme::kFgMuted, "%s", Count(static_cast<double>(save->frames)).c_str());
        ImGui::TableSetColumnIndex(4);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(theme::kFgMuted, "%s", ClockText(save->time).c_str());
        ImGui::TableSetColumnIndex(5);
        ImGui::BeginDisabled(!scene);
        if (draw::IconButton("##play", icon::kPlay, false, save->policy.c_str()))
            Queue("train.play", Args({{"policy", json::Value::Str(save->policy)}}));
        ImGui::EndDisabled();
        if (ImGui::BeginPopupContextItem("##save_menu")) {
            if (ImGui::MenuItemEx(Tr("Copy Policy Path"), icon::kCopy)) ImGui::SetClipboardText(save->policy.c_str());
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }
    ImGui::EndTable();
}

}  // namespace nuka::runtime::app::viewer
