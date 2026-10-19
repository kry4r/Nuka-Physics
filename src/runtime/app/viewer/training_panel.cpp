// The run's progress plate over the scene viewport, under the HUD chips; the robots' arrows are scene
// meshes.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/shell_util.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>

namespace nuka::runtime::app::viewer {

namespace theme = nuka::render::imgui::theme;
namespace draw = nuka::nui::draw;
using nuka::nui::Tr;
using nuka::render::imgui::FontRole;
using nuka::render::imgui::GetNukaFont;
using namespace shell;

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

}  // namespace nuka::runtime::app::viewer
