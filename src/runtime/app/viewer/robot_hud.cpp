// The robot overlays over the scene viewport: link frames, joint axes with their limits, centres of
// mass and inertia boxes, the Measure tool's line, and the hovered link's tooltip.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"

#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/robot_tools.hpp"
#include "runtime/app/viewer/shell_util.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

namespace theme = nuka::render::imgui::theme;
namespace draw = nuka::nui::draw;
using nuka::nui::Tr;
using nuka::render::imgui::FontRole;
using nuka::render::imgui::GetNukaFont;
using namespace shell;

namespace {

constexpr float kPi = 3.14159265f;
// Link axes are this share of the link's visual size, held within [kAxisMin, kAxisMax] m.
constexpr float kAxisShare = 0.25f, kAxisMin = 0.03f, kAxisMax = 0.5f;
constexpr int kFanSegments = 48;

float At(const std::vector<float>& v, uint32_t i) { return i < v.size() ? v[i] : 0.0f; }

float AxisLength(const RobotLink& link) { return std::clamp(link.size * kAxisShare, kAxisMin, kAxisMax); }

// A unit vector at right angles to the unit `axis`.
math::Vec3 Across(const math::Vec3& axis) {
    const math::Vec3 e = std::fabs(axis.x) < 0.9f ? math::Vec3::UnitX() : math::Vec3::UnitY();
    return (e - axis * e.Dot(axis)).Normalized();
}

// Millimetres under a metre, else metres.
std::string Length(float m) {
    return std::fabs(m) < 1.0f ? Fmt("%.1f mm", static_cast<double>(m) * 1000.0) : Fmt("%.3f m", static_cast<double>(m));
}

// The quartered disc that marks a centre of mass.
void ComMark(ImDrawList* list, const ImVec2& c, float r, ImU32 ink, ImU32 paper) {
    list->AddCircleFilled(c, r, paper, 20);
    for (const float from : {0.0f, kPi}) {
        list->PathLineTo(c);
        list->PathArcTo(c, r, from, from + kPi * 0.5f, 8);
        list->PathFillConvex(ink);
    }
    list->AddCircle(c, r, ink, 20, 1.0f);
}

// A filled triangle pointing up at `p`, the target caret of the Joints panel.
void Caret(ImDrawList* list, const ImVec2& p, float s, ImU32 color) {
    list->AddTriangleFilled(ImVec2(p.x, p.y - 5.0f * s), ImVec2(p.x + 4.0f * s, p.y + 3.0f * s),
                            ImVec2(p.x - 4.0f * s, p.y + 3.0f * s), color);
}

// Text on the HUD's chamfered plate with its top-left at `at`.
void Plate(ImDrawList* list, const ImVec2& at, const std::string& text, ImU32 ink, float s) {
    ImFont* font = GetNukaFont(FontRole::Display);
    const float px = theme::kFontLabel * s;
    const ImVec2 ts = font->CalcTextSizeA(px, FLT_MAX, 0.0f, text.c_str());
    draw::Chamfer(list, at, ImVec2(at.x + ts.x + 12.0f * s, at.y + ts.y + 6.0f * s), 3.0f * s,
                  U32(theme::Alpha(theme::kBg0, 0.85f)), U32(theme::Alpha(theme::kLineHi, 0.8f)));
    list->AddText(font, px, ImVec2(at.x + 6.0f * s, at.y + 3.0f * s), ink, text.c_str());
}

}  // namespace

void ImGuiLayer::DrawRobotHud(const ImVec2& vmin, const ImVec2& vmax) {
    const ViewerUiState& ui = *frame_.ui;
    const RobotView& v = ui.robot;
    const RobotOverlays& o = v.overlays;
    const size_t n = std::min(v.links.size(), v.link_poses.size());
    if (n == 0u) return;
    const ViewProjector project = Projector(vmin, vmax);
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    list->PushClipRect(vmin, vmax, true);
    const float s = nui::Scale();
    const auto line = [&](const math::Vec3& a, const math::Vec3& b, ImU32 color, float width) {
        ImVec2 pa, pb;
        if (project(a, &pa) && project(b, &pb)) list->AddLine(pa, pb, color, width * s);
    };

    if (o.on && o.frames) {
        const math::Vec3 units[3] = {math::Vec3::UnitX(), math::Vec3::UnitY(), math::Vec3::UnitZ()};
        const ImU32 colors[3] = {U32(theme::kAxisX), U32(theme::kAxisY), U32(theme::kAxisZ)};
        for (size_t l = 0; l < n; ++l) {
            const math::Transform& pose = v.link_poses[l];
            for (int k = 0; k < 3; ++k) line(pose.position, pose.TransformPoint(units[k] * AxisLength(v.links[l])), colors[k], 1.5f);
        }
    }

    // Each joint's axis from its parent's side, with its limits swept about or along it; the joint the
    // panel drags shows while the overlay is off.
    const ImU32 edge = U32(theme::Alpha(theme::kInfo, 0.8f)), fill = U32(theme::Alpha(theme::kInfo, 0.16f));
    const ImU32 accent = U32(theme::kAccent), caret = U32(theme::kInfo);
    for (size_t i = 0; i < v.joints.size(); ++i) {
        const RobotJoint& j = v.joints[i];
        if ((!(o.on && o.axes) && static_cast<int>(i) != v.dragging) || j.link >= n) continue;
        const RobotLink& link = v.links[j.link];
        if (link.parent < 0 || static_cast<size_t>(link.parent) >= n) continue;
        const math::Transform zero = v.link_poses[static_cast<size_t>(link.parent)] * link.zero_frame;
        const math::Vec3 axis = link.axis.Normalized(), world_axis = zero.rotation.Rotate(axis);
        const float length = AxisLength(link), q = At(v.live.q, j.link), target = At(ui.drive_targets, j.link);
        ImVec2 o2, tip;
        if (!project(zero.position, &o2) || !project(zero.position + world_axis * (length * 1.2f), &tip)) continue;
        const float dx = tip.x - o2.x, dy = tip.y - o2.y, d = std::max(std::sqrt(dx * dx + dy * dy), 1.0f);
        const ImVec2 u(dx / d, dy / d), w(-dy / d, dx / d);
        list->AddLine(o2, tip, edge, 2.0f * s);
        list->AddTriangleFilled(tip, ImVec2(tip.x - u.x * 8.0f * s + w.x * 4.0f * s, tip.y - u.y * 8.0f * s + w.y * 4.0f * s),
                                ImVec2(tip.x - u.x * 8.0f * s - w.x * 4.0f * s, tip.y - u.y * 8.0f * s - w.y * 4.0f * s), edge);
        const float lo = j.limited ? j.lower : -length, hi = j.limited ? j.upper : length;
        ImVec2 p;
        if (j.prismatic) {
            const auto at = [&](float x) { return zero.position + world_axis * x; };
            line(at(lo), at(hi), edge, 1.5f);
            for (const float x : {lo, hi}) {
                if (!j.limited || !project(at(x), &p)) continue;
                list->AddLine(ImVec2(p.x - w.x * 6.0f * s, p.y - w.y * 6.0f * s), ImVec2(p.x + w.x * 6.0f * s, p.y + w.y * 6.0f * s),
                              edge, 1.5f * s);
            }
            if (project(at(q), &p)) list->AddCircleFilled(p, 3.5f * s, accent);
            if (project(at(target), &p)) Caret(list, p, s, caret);
            continue;
        }
        const math::Vec3 ref = Across(axis);
        const auto at = [&](float angle) {
            return zero.TransformPoint(math::Quat::FromAxisAngle(axis, angle).Rotate(ref) * (length * 0.8f));
        };
        if (j.limited) {
            ImVec2 prev;
            bool had = project(at(lo), &prev);
            for (int k = 1; k <= kFanSegments; ++k) {
                ImVec2 next;
                const bool has = project(at(lo + (hi - lo) * static_cast<float>(k) / kFanSegments), &next);
                if (had && has) {
                    list->AddTriangleFilled(o2, prev, next, fill);
                    list->AddLine(prev, next, edge, 1.0f * s);
                }
                prev = next;
                had = has;
            }
            line(zero.position, at(lo), edge, 1.0f);
            line(zero.position, at(hi), edge, 1.0f);
        }
        line(zero.position, at(q), accent, 2.0f);
        if (project(at(target), &p)) Caret(list, p, s, caret);
    }

    if (o.on && o.com) {
        const ImU32 ink = U32(theme::kFg), paper = U32(theme::kBg0);
        for (size_t l = 0; l < n; ++l) {
            ImVec2 p;
            if (v.links[l].mass > 0.0f && project(LinkCom(v.links[l], v.link_poses[l]), &p)) ComMark(list, p, 4.0f * s, ink, paper);
        }
        // The robot's centre of mass and its drop to the ground.
        math::Vec3 com;
        float mass = 0.0f;
        ImVec2 pc, pg;
        if (RobotCom(v, &com, &mass) && project(com, &pc) && project(math::Vec3{com.x, com.y, 0.0f}, &pg)) {
            const ImU32 warn = U32(theme::kWarn);
            DashedLine(list, pc, pg, warn, 5.0f * s, 1.5f * s);
            list->AddCircle(pg, 5.0f * s, warn, 16, 1.5f * s);
            list->AddLine(ImVec2(pg.x - 7.0f * s, pg.y), ImVec2(pg.x + 7.0f * s, pg.y), warn, 1.5f * s);
            ComMark(list, pc, 7.0f * s, warn, paper);
            Plate(list, ImVec2(pc.x + 12.0f * s, pc.y - 10.0f * s), Fmt(Tr("COM  %.2f kg"), static_cast<double>(mass)), warn, s);
        }
    }

    if (o.on && o.inertia) {
        for (size_t l = 0; l < n; ++l) {
            math::Vec3 edges;
            if (!InertiaBox(v.links[l], &edges)) continue;
            const math::Transform frame = v.link_poses[l] * v.links[l].inertial;
            const math::Vec3 h = edges * 0.5f;
            math::Vec3 corner[8];
            for (int k = 0; k < 8; ++k)
                corner[k] = frame.TransformPoint(math::Vec3{(k & 1) != 0 ? h.x : -h.x, (k & 2) != 0 ? h.y : -h.y,
                                                            (k & 4) != 0 ? h.z : -h.z});
            for (int k = 0; k < 8; ++k)
                for (const int bit : {1, 2, 4})
                    if ((k & bit) == 0) line(corner[k], corner[k | bit], U32(theme::Alpha(theme::kAccent, 0.75f)), 1.0f);
        }
    }

    // The measurement: a dashed line, its run along each axis, and the distance.
    const Measure& m = v.measure;
    math::Vec3 from, to;
    ImVec2 a, b;
    if (MeasureEnds(v, &from, &to) && project(from, &a) && project(to, &b)) {
        const math::Vec3 x{to.x, from.y, from.z}, xy{to.x, to.y, from.z};
        line(from, x, U32(theme::Alpha(theme::kAxisX, 0.8f)), 1.0f);
        line(x, xy, U32(theme::Alpha(theme::kAxisY, 0.8f)), 1.0f);
        line(xy, to, U32(theme::Alpha(theme::kAxisZ, 0.8f)), 1.0f);
        DashedLine(list, a, b, accent, 6.0f * s, 2.0f * s);
        for (const ImVec2& e : {a, b}) {
            list->AddCircle(e, 5.0f * s, accent, 16, 1.5f * s);
            list->AddCircleFilled(e, 2.0f * s, accent);
        }
        const math::Vec3 delta = to - from;
        const std::string text = Length(delta.Length()) + "\n" +
                                 Fmt("dx %s  dy %s  dz %s", Length(delta.x).c_str(), Length(delta.y).c_str(), Length(delta.z).c_str());
        Plate(list, ImVec2((a.x + b.x) * 0.5f + 10.0f * s, (a.y + b.y) * 0.5f), text, accent, s);
    } else if (MeasurePoint(v, m.a, &from) && project(from, &a)) {
        list->AddCircle(a, 5.0f * s, accent, 16, 1.5f * s);
        list->AddCircleFilled(a, 2.0f * s, accent);
    }
    if (ui.tool == ViewportTool::Measure) {
        const char* hint = m.a.kind != MeasureEnd::Kind::None && m.b.kind == MeasureEnd::Kind::None
                               ? Tr("Click a second link or joint, or empty space for the ground")
                               : Tr("Click a link, or a joint origin while the joint axes show");
        const ImVec2 ts = GetNukaFont(FontRole::Display)->CalcTextSizeA(theme::kFontLabel * s, FLT_MAX, 0.0f, hint);
        Plate(list, ImVec2((vmin.x + vmax.x - ts.x) * 0.5f - 6.0f * s, vmax.y - ts.y - 48.0f * s), hint, U32(theme::kFgMuted), s);
    }
    list->PopClipRect();

    // The link under the pointer: its mass and the joint it hangs from.
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    if (v.hovered < 0 || static_cast<size_t>(v.hovered) >= v.links.size() || mouse.x < vmin.x || mouse.y < vmin.y ||
        mouse.x > vmax.x || mouse.y > vmax.y)
        return;
    const RobotLink& link = v.links[static_cast<size_t>(v.hovered)];
    std::string text = link.path.substr(link.path.rfind('/') + 1u) + "\n" + Fmt(Tr("mass %.3f kg"), static_cast<double>(link.mass));
    if (!link.joint.empty()) {
        text += "\n" + link.joint + "  \xC2\xB7  " + Tr(link.type.c_str());
        const int j = FindJoint(v.joints, link.joint);
        if (j >= 0) text += "  " + JointText(v.joints[static_cast<size_t>(j)], At(v.live.q, v.joints[static_cast<size_t>(j)].link), v.radians);
    }
    math::Vec3 edges;
    if (o.on && o.inertia && link.mass > 0.0f && !InertiaBox(link, &edges))
        text += std::string("\n") + Tr("No inertia box: the principal moments break the triangle inequality.");
    ImGui::SetTooltip("%s", text.c_str());
}

}  // namespace nuka::runtime::app::viewer
