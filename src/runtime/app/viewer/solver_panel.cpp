// The Solver panel and its viewport HUD: residual, sweep and energy curves of the telemetry
// records, the capped solves with their worst elements, the heatmap scale and the locate marker.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "implot.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/shell_util.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>
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

constexpr uint32_t kBudgets[] = {10u, 32u, 128u, 512u};
constexpr float kAlerts[] = {1.0e-4f, 1.0e-3f, 1.0e-2f};
constexpr float kTimeSteps[] = {0.001f, 0.002f, 1.0f / 240.0f, 0.005f};
// Log axes clamp exact zeros to this floor.
constexpr float kFloor = 1.0e-9f;

bool Capped(const nuka_telemetry_record_t& r) { return (r.flags & NUKA_TELEMETRY_CAP) != 0u; }

std::string WorstText(const nuka_telemetry_record_t& r) {
    if (r.worst_kind == NUKA_TELEMETRY_ELEMENT_PARTICLE) return Fmt(Tr("particle %u"), r.worst_index);
    if (r.worst_kind == NUKA_TELEMETRY_ELEMENT_ROW) return Fmt(Tr("contact row %u"), r.worst_index);
    return Tr("none");
}

int IndexOf(const uint32_t* values, size_t count, uint32_t value) {
    const auto it = std::find(values, values + count, value);
    return it == values + count ? -1 : static_cast<int>(it - values);
}

ImPlotSpec Line(const ImVec4& color, float weight) {
    ImPlotSpec spec;
    spec.LineColor = color;
    spec.LineWeight = weight * nui::Scale();
    return spec;
}

// A plot over the panel's width with the shell's flags; false when ImPlot skipped it.
bool BeginSolverPlot(const char* id, float height, const char* x_label, const char* y_label, bool log_y) {
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(theme::kSpace * nui::Scale(), theme::kSpace * nui::Scale()));
    const bool open = ImPlot::BeginPlot(id, ImVec2(-FLT_MIN, height * nui::Scale()),
                                        ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoFrame);
    if (!open) {
        ImPlot::PopStyleVar();
        return false;
    }
    ImPlot::SetupAxes(x_label, y_label, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
    if (log_y) ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
    ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);
    return true;
}

void EndSolverPlot() {
    ImPlot::EndPlot();
    ImPlot::PopStyleVar();
}

void AlertLine(float alert) {
    ImPlotSpec spec = Line(theme::kErr, 1.0f);
    spec.Flags = ImPlotInfLinesFlags_Horizontal;
    ImPlot::PlotInfLines(Tr("alert"), &alert, 1, spec);
}

}  // namespace

void ImGuiLayer::BuildSolver(nui::VStack& root) {
    const auto has_scene = [this] { return frame_.ui->has_scene; };
    const auto view = [this]() -> const SolverView& { return frame_.ui->solver; };

    auto& hero = root.Add<nui::HStack>();
    auto& state = hero.Add<nui::Label>(std::string(), Tone::Accent, TextStyle::Hero);
    state.text_fn = [view] {
        const SolverView& v = view();
        if (v.mask == 0u) return std::string(Tr("OFF"));
        return v.recent_caps != 0u ? Fmt(Tr("CAP \xC3\x97%u"), v.recent_caps) : std::string(Tr("CONVERGED"));
    };
    state.tone_fn = [view] {
        const SolverView& v = view();
        return v.mask == 0u ? Tone::Dim : v.recent_caps != 0u ? Tone::Err : Tone::Ok;
    };
    auto& rec = hero.Add<nui::Badge>();
    rec.text_fn = [view] { return std::string(Tr(view().mask != 0u ? "recording" : "idle")); };
    rec.tone_fn = [view] { return view().mask != 0u ? Tone::Accent : Tone::Dim; };
    auto& hint = root.Add<nui::Label>("Records each solve's residual, sweeps and energy; the heatmap colours "
                                      "cloth by its Newton correction.",
                                      Tone::Dim);
    hint.wrap = true;

    auto& toggles = root.Add<nui::HStack>();
    toggles.Add<nui::ToggleButton>("Record").Action("solver.telemetry").enabled_if = has_scene;
    toggles.Add<nui::ToggleButton>("Heatmap").Action("solver.heatmap").enabled_if = has_scene;
    auto& locate = root.Add<nui::Button>("Locate worst", icon::kTarget).Action("solver.locate");
    locate.enabled_if = [view, has_scene] { return has_scene() && !view().history.empty(); };

    root.Add<nui::Separator>("Time step");
    auto& step = root.Add<nui::Segmented>(
        std::vector<std::string>{"1 ms", "2 ms", "1/240 s", "5 ms"},
        nui::Bind<int>(
            [view] {
                const float dt = view().dt;
                for (size_t i = 0; i < std::size(kTimeSteps); ++i)
                    if (std::fabs(dt - kTimeSteps[i]) <= kTimeSteps[i] * 1.0e-4f) return static_cast<int>(i);
                return -1;
            },
            [this](const int& i) {
                Queue("sim.dt", Args({{"seconds", json::Value::Float(kTimeSteps[static_cast<size_t>(i)])}}));
            }));
    step.enabled_if = has_scene;
    step.tooltip = "The physics step; a change rebuilds the world and keeps its state.";
    root.Add<nui::Separator>("Sweep budget");
    std::vector<std::string> budgets;
    for (const uint32_t n : kBudgets) budgets.push_back(std::to_string(n));
    auto& budget = root.Add<nui::Segmented>(
        budgets, nui::Bind<int>([view] { return IndexOf(kBudgets, std::size(kBudgets), view().iterations); },
                                [this](const int& i) {
                                    Queue("solver.set_iterations",
                                          Args({{"n", json::Value::Int(kBudgets[static_cast<size_t>(i)])}}));
                                }));
    budget.enabled_if = has_scene;
    budget.tooltip = "Velocity sweeps per substep; the solve stops early at the tolerance.";
    root.Add<nui::Separator>("Alert residual");
    auto& alert = root.Add<nui::Segmented>(
        std::vector<std::string>{"1e-4 m/s", "1e-3 m/s", "1e-2 m/s"},
        nui::Bind<int>(
            [view] {
                const float a = view().alert;
                for (size_t i = 0; i < std::size(kAlerts); ++i)
                    if (std::fabs(a - kAlerts[i]) <= kAlerts[i] * 1.0e-3f) return static_cast<int>(i);
                return -1;
            },
            [this](const int& i) {
                const uint32_t mask = frame_.ui->solver.mask != 0u ? frame_.ui->solver.mask : kSolverTelemetryAll;
                Queue("solver.telemetry", Args({{"mask", json::Value::Int(mask)},
                                                {"alert", json::Value::Float(kAlerts[static_cast<size_t>(i)])}}));
            }));
    alert.enabled_if = has_scene;
    alert.tooltip = "A spent budget above this residual counts as a cap, unless the scene sets a tolerance.";

    root.Add<nui::Separator>("Telemetry");
    const auto stat = [&root](const char* label, std::function<std::string()> fn) -> nui::Stat& {
        return root.Add<nui::Stat>(label, std::move(fn));
    };
    stat("records", [view] { return Fmt("%llu", static_cast<unsigned long long>(view().records)); });
    stat("capped", [view] { return Fmt("%llu", static_cast<unsigned long long>(view().caps)); }).tone_fn = [view] {
        return view().caps != 0u ? Tone::Warn : Tone::Normal;
    };
    stat("recent capped", [view] {
        const SolverView& v = view();
        return Fmt("%u / %u", v.recent_caps, v.recent_records);
    }).tone_fn = [view] { return view().recent_caps != 0u ? Tone::Err : Tone::Ok; };
    stat("dropped", [view] { return Fmt("%llu", static_cast<unsigned long long>(view().dropped)); });
    const auto last = [view]() -> const nuka_telemetry_record_t* {
        const SolverView& v = view();
        return v.history.empty() ? nullptr : &v.history.back();
    };
    stat("sweeps", [last] { return last() ? Fmt("%u / %u", last()->sweeps, last()->sweep_budget) : std::string(Tr("N/A")); });
    stat("vertex residual", [last] {
        return last() ? Fmt("%.3e m/s", static_cast<double>(last()->vertex_residual)) : std::string(Tr("N/A"));
    });
    stat("contact residual", [last] {
        return last() ? Fmt("%.3e m/s", static_cast<double>(last()->contact_residual)) : std::string(Tr("N/A"));
    });
    stat("penetration", [last] {
        return last() ? Fmt("%.3e m", static_cast<double>(last()->max_penetration)) : std::string(Tr("N/A"));
    });
    stat("worst", [last] { return last() ? WorstText(*last()) : std::string(Tr("N/A")); });

    root.Add<nui::Separator>("Residual");
    root.Add<nui::Custom>([view] {
        const SolverView& v = view();
        std::vector<float> x, vertex, contact, cap_x, cap_y;
        x.reserve(v.history.size());
        for (size_t i = 0; i < v.history.size(); ++i) {
            const nuka_telemetry_record_t& r = v.history[i];
            x.push_back(static_cast<float>(v.first + i));
            vertex.push_back(std::max(r.vertex_residual, kFloor));
            contact.push_back(std::max(r.contact_residual, kFloor));
            if (Capped(r)) {
                cap_x.push_back(x.back());
                cap_y.push_back(std::max(r.worst_residual, kFloor));
            }
        }
        if (!BeginSolverPlot("##residual", 150.0f, nullptr, "m/s", true)) return;
        const int n = static_cast<int>(x.size());
        ImPlot::PlotLine(Tr("vertex"), x.data(), vertex.data(), n, Line(theme::kInfo, 1.5f));
        ImPlot::PlotLine(Tr("contact"), x.data(), contact.data(), n, Line(theme::kAccent, 1.5f));
        ImPlotSpec caps = Line(theme::kErr, 1.0f);
        caps.Marker = ImPlotMarker_Diamond;
        caps.MarkerSize = 3.5f * nui::Scale();
        caps.MarkerFillColor = theme::kErr;
        ImPlot::PlotScatter(Tr("cap"), cap_x.data(), cap_y.data(), static_cast<int>(cap_x.size()), caps);
        AlertLine(v.alert);
        EndSolverPlot();
    });
    root.Add<nui::Custom>([view] {
        const SolverView& v = view();
        std::vector<float> x, sweeps;
        for (size_t i = 0; i < v.history.size(); ++i) {
            x.push_back(static_cast<float>(v.first + i));
            sweeps.push_back(static_cast<float>(v.history[i].sweeps));
        }
        if (!BeginSolverPlot("##sweeps", 80.0f, nullptr, Tr("sweeps"), false)) return;
        ImPlotSpec used = Line(theme::kAccent, 1.0f);
        used.Flags = ImPlotLineFlags_Shaded;
        used.FillColor = theme::Alpha(theme::kAccent, 0.16f);
        ImPlot::PlotLine(Tr("sweeps"), x.data(), sweeps.data(), static_cast<int>(x.size()), used);
        const float budget = static_cast<float>(v.iterations);
        ImPlotSpec cap = Line(theme::kWarn, 1.0f);
        cap.Flags = ImPlotInfLinesFlags_Horizontal;
        ImPlot::PlotInfLines(Tr("budget"), &budget, 1, cap);
        EndSolverPlot();
    });

    root.Add<nui::Separator>("Convergence").visible_if = [view] { return (view().mask & NUKA_TELEMETRY_SWEEPS) != 0u; };
    auto& convergence = root.Add<nui::Custom>([view] {
        const SolverView& v = view();
        // The newest solve that sampled its sweeps; a capped one when any is kept.
        const auto it = std::find_if(v.history.rbegin(), v.history.rend(), Capped);
        const nuka_telemetry_record_t* r = it != v.history.rend() ? &*it : v.history.empty() ? nullptr : &v.history.back();
        if (r == nullptr || r->sample_count == 0u) {
            ImGui::TextDisabled("%s", Tr("No sweep samples yet."));
            return;
        }
        std::vector<float> sweep, residual;
        for (uint32_t i = 0; i < std::min<uint32_t>(r->sample_count, NUKA_TELEMETRY_SAMPLES); ++i) {
            sweep.push_back(static_cast<float>(r->sample_sweep[i]));
            residual.push_back(std::max(r->sample_residual[i], kFloor));
        }
        if (!BeginSolverPlot("##convergence", 120.0f, Tr("sweep"), "m/s", true)) return;
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlotSpec spec = Line(Capped(*r) ? theme::kErr : theme::kOk, 1.5f);
        spec.Marker = ImPlotMarker_Circle;
        spec.MarkerSize = 2.5f * nui::Scale();
        ImPlot::PlotLine(Fmt(Tr("solve %u"), r->sequence).c_str(), sweep.data(), residual.data(),
                         static_cast<int>(sweep.size()), spec);
        AlertLine(v.alert);
        EndSolverPlot();
    });
    convergence.visible_if = [view] { return (view().mask & NUKA_TELEMETRY_SWEEPS) != 0u; };

    root.Add<nui::Separator>("Energy").visible_if = [view] { return (view().mask & NUKA_TELEMETRY_ENERGY) != 0u; };
    auto& energy = root.Add<nui::Custom>([view] {
        const SolverView& v = view();
        std::vector<float> x, total, residual;
        for (size_t i = 0; i < v.history.size(); ++i) {
            const nuka_telemetry_record_t& r = v.history[i];
            if ((r.flags & NUKA_TELEMETRY_ENERGY_VALID) == 0u) continue;
            x.push_back(static_cast<float>(v.first + i));
            total.push_back(r.energy);
            residual.push_back(r.energy_residual);
        }
        if (!BeginSolverPlot("##energy", 100.0f, nullptr, "J", false)) return;
        const int n = static_cast<int>(x.size());
        ImPlot::PlotLine(Tr("energy"), x.data(), total.data(), n, Line(theme::kInfo, 1.5f));
        ImPlot::PlotLine(Tr("ledger residual"), x.data(), residual.data(), n, Line(theme::kWarn, 1.0f));
        EndSolverPlot();
    });
    energy.visible_if = [view] { return (view().mask & NUKA_TELEMETRY_ENERGY) != 0u; };

    root.Add<nui::Separator>("Capped solves");
    root.Add<nui::Custom>([this, view] {
        const SolverView& v = view();
        const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp;
        int shown = 0;
        if (ImGui::BeginTable("##caps", 5, flags)) {
            ImGui::TableSetupColumn(Tr("record"));
            ImGui::TableSetupColumn(Tr("sweeps"));
            ImGui::TableSetupColumn(Tr("residual"));
            ImGui::TableSetupColumn(Tr("worst"));
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableHeadersRow();
            for (size_t k = v.history.size(); k > 0u && shown < 8; --k) {
                const nuka_telemetry_record_t& r = v.history[k - 1u];
                if (!Capped(r)) continue;
                ++shown;
                const uint64_t number = v.first + k - 1u;
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%llu", static_cast<unsigned long long>(number));
                ImGui::TableNextColumn();
                ImGui::Text("%u/%u", r.sweeps, r.sweep_budget);
                ImGui::TableNextColumn();
                ImGui::TextColored(theme::kErr, "%.2e", static_cast<double>(r.worst_residual));
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(WorstText(r).c_str());
                ImGui::TableNextColumn();
                ImGui::PushID(static_cast<int>(k));
                if (draw::IconButton("##locate", icon::kTarget, false, Tr("Locate")))
                    Queue("solver.locate", Args({{"record", json::Value::Int(static_cast<int64_t>(number))}}));
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        if (shown == 0) ImGui::TextDisabled("%s", Tr(v.mask == 0u ? "Recording is off." : "No capped solve kept."));
    });
}

void HeatColor(float residual, float alert, float rgb[3]) {
    // Deep blue, cyan, green, amber, red.
    constexpr float kStops[5][3] = {{0.06f, 0.10f, 0.32f}, {0.05f, 0.62f, 0.86f}, {0.18f, 0.80f, 0.44f},
                                    {0.96f, 0.62f, 0.04f}, {0.94f, 0.16f, 0.22f}};
    const float t = std::clamp((std::log10(std::max(residual, 1.0e-30f) / alert) + 3.0f) * 0.25f, 0.0f, 1.0f);
    const int i = std::min(static_cast<int>(t * 4.0f), 3);
    const float f = t * 4.0f - static_cast<float>(i);
    for (int c = 0; c < 3; ++c) rgb[c] = kStops[i][c] + f * (kStops[i + 1][c] - kStops[i][c]);
}

std::string MarkLabel(const SolverView& v, bool translate) {
    const auto tr = [translate](const char* text) { return translate ? Tr(text) : text; };
    const std::string element = Fmt(tr(v.mark_kind), v.mark_index);
    if (v.mark_row < 0) return element;
    return Fmt(tr("contact row %u"), static_cast<uint32_t>(v.mark_row)) + " / " + element;
}

void ImGuiLayer::DrawSolverHud(const ImVec2& vmin, const ImVec2& vmax) {
    const SolverView& v = frame_.ui->solver;
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    const float s = nui::Scale();
    ImFont* font = GetNukaFont(FontRole::Display);
    const float px = theme::kFontLabel * s;
    const auto color = [](const float rgb[3], float alpha) {
        return ImGui::GetColorU32(ImVec4(rgb[0], rgb[1], rgb[2], alpha));
    };
    if (v.heatmap) {
        // Four decades of log scale up the right edge under the mode menu, the alert at three quarters.
        const float h = std::min(200.0f * s, (vmax.y - vmin.y) * 0.45f), w = 9.0f * s;
        const ImVec2 bar1(vmax.x - 70.0f * s, vmin.y + 74.0f * s + h), bar0(bar1.x - w, bar1.y - h);
        draw::Chamfer(list, ImVec2(bar0.x - 12.0f * s, bar0.y - 30.0f * s), ImVec2(vmax.x - 14.0f * s, bar1.y + 14.0f * s),
                      6.0f * s, U32(theme::Alpha(theme::kBg0, 0.78f)), U32(theme::kLineHi));
        list->AddText(font, px, ImVec2(bar0.x - 4.0f * s, bar0.y - 22.0f * s), U32(theme::kFgMuted), Tr("NEWTON  M/S"));
        constexpr int kSteps = 40;
        for (int i = 0; i < kSteps; ++i) {
            const float t0 = static_cast<float>(i) / static_cast<float>(kSteps);
            const float t1 = static_cast<float>(i + 1) / static_cast<float>(kSteps);
            float c0[3], c1[3];
            HeatColor(v.alert * std::pow(10.0f, 4.0f * t0 - 3.0f), v.alert, c0);
            HeatColor(v.alert * std::pow(10.0f, 4.0f * t1 - 3.0f), v.alert, c1);
            list->AddRectFilledMultiColor(ImVec2(bar0.x, bar1.y - t1 * h), ImVec2(bar1.x, bar1.y - t0 * h + 0.5f),
                                          color(c1, 1.0f), color(c1, 1.0f), color(c0, 1.0f), color(c0, 1.0f));
        }
        list->AddRect(bar0, bar1, U32(theme::kLineHi));
        for (int d = 0; d <= 4; ++d) {
            const float y = bar1.y - static_cast<float>(d) * 0.25f * h;
            const bool at_alert = d == 3;
            list->AddLine(ImVec2(bar1.x, y), ImVec2(bar1.x + (at_alert ? 8.0f : 4.0f) * s, y),
                          U32(at_alert ? theme::kErr : theme::kFgMuted), at_alert ? 2.0f * s : 1.0f * s);
            const std::string label = Fmt("%.0e", static_cast<double>(v.alert) * std::pow(10.0, d - 3));
            list->AddText(font, px, ImVec2(bar1.x + 10.0f * s, y - px * 0.55f), U32(at_alert ? theme::kErr : theme::kFg),
                          label.c_str());
        }
    }
    if (!v.marked) return;
    float view_m[16], proj[16];
    CameraMatrices(*frame_.camera, view_m, proj);
    const float p[4] = {v.mark[0], v.mark[1], v.mark[2], 1.0f};
    float e[4], c[4];
    for (int r = 0; r < 4; ++r) e[r] = view_m[r] * p[0] + view_m[4 + r] * p[1] + view_m[8 + r] * p[2] + view_m[12 + r] * p[3];
    for (int r = 0; r < 4; ++r) c[r] = proj[r] * e[0] + proj[4 + r] * e[1] + proj[8 + r] * e[2] + proj[12 + r] * e[3];
    if (c[3] <= 1.0e-6f) return;
    const ImVec2 at(vmin.x + (c[0] / c[3] * 0.5f + 0.5f) * (vmax.x - vmin.x),
                    vmin.y + (0.5f - c[1] / c[3] * 0.5f) * (vmax.y - vmin.y));
    // A pulsing reticle with corner ticks and the element's label.
    const float t = static_cast<float>(std::fmod(ImGui::GetTime(), 1.4) / 1.4);
    const float r0 = 14.0f * s;
    list->AddCircle(at, r0 + 10.0f * t * s, U32(theme::Alpha(theme::kErr, 0.6f * (1.0f - t))), 32, 1.5f * s);
    list->AddCircle(at, r0, U32(theme::kErr), 32, 1.5f * s);
    list->AddCircleFilled(at, 2.5f * s, U32(theme::kErr));
    for (int q = 0; q < 4; ++q) {
        const float a = static_cast<float>(q) * 1.5707963f;
        const ImVec2 d(std::cos(a), std::sin(a));
        list->AddLine(ImVec2(at.x + d.x * (r0 + 3.0f * s), at.y + d.y * (r0 + 3.0f * s)),
                      ImVec2(at.x + d.x * (r0 + 9.0f * s), at.y + d.y * (r0 + 9.0f * s)), U32(theme::kErr), 1.5f * s);
    }
    const std::string label = Upper(MarkLabel(v, true) + Fmt("  %.2e m/s", static_cast<double>(v.mark_residual)));
    const ImVec2 ts = font->CalcTextSizeA(px, FLT_MAX, 0.0f, label.c_str());
    // The label flips left of the reticle rather than run under the heatmap scale or off the view.
    const float width = ts.x + 16.0f * s, gap = r0 + 14.0f * s;
    const float right = v.heatmap ? vmax.x - 96.0f * s : vmax.x;
    const ImVec2 l0(at.x + gap + width <= right ? at.x + gap : at.x - gap - width, at.y - ts.y * 0.5f - 4.0f * s);
    draw::Chamfer(list, l0, ImVec2(l0.x + width, l0.y + ts.y + 8.0f * s), 4.0f * s,
                  U32(theme::Alpha(theme::kBg0, 0.85f)), U32(theme::Alpha(theme::kErr, 0.7f)));
    list->AddText(font, px, ImVec2(l0.x + 8.0f * s, l0.y + 4.0f * s), U32(theme::kErr), label.c_str());
}

}  // namespace nuka::runtime::app::viewer
