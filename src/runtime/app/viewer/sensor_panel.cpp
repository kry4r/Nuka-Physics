// The sensors in the editor: the types it offers, their overlays over the scene viewport, and the
// selected sensor's fields in Properties.

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
using namespace shell;

namespace {

constexpr float kDeg = 57.2957795f;
// Overlay sizes in metres: the frustum's depth, the fan's reach and a state sensor's axes.
constexpr float kFrustumDepth = 0.3f, kFanReach = 0.5f, kAxis = 0.06f;
constexpr int kFanRays = 24, kFanRows = 4;
// The planes a camera renders, as `aov_mask` bits; a mask of 0 renders every one.
constexpr std::pair<const char*, uint32_t> kAovs[] = {
    {"Color", 1u}, {"Depth", 2u}, {"Normal", 4u}, {"Albedo", 8u}, {"Instance", 16u}};
constexpr uint32_t kAllAovs = 31u;

json::Value Vec3Json(const math::Vec3& v) {
    json::Value a = json::Value::Array();
    for (const float c : {v.x, v.y, v.z}) a.PushBack(json::Value::Float(c));
    return a;
}

}  // namespace

const std::vector<SensorKind>& SensorKinds() {
    static const std::vector<SensorKind> kinds = {
        {"camera", "Camera", icon::kCamera},
        {"depth", "Depth Camera", icon::kCameraNode},
        {"lidar", "Lidar", icon::kWaves},
        {"range_scan", "Range Scan", icon::kRuler},
        {"imu", "IMU", icon::kPlot},
        {"linear_velocity", "Velocity", icon::kSpeed},
        {"frame_pose", "Frame Pose", icon::kTarget},
        {"joint_state", "Joint State", icon::kJoint},
        {"force_torque", "Force/Torque", icon::kAtom},
        {"contact", "Contact", icon::kGrab},
        {"tactile", "Tactile Pad", icon::kHexagon},
    };
    return kinds;
}

const SensorKind& FindSensorKind(const std::string& type) {
    const std::vector<SensorKind>& kinds = SensorKinds();
    return *std::find_if(kinds.begin(), kinds.end(), [&](const SensorKind& k) { return type == k.type; });
}

void ImGuiLayer::DrawSensorHud(const ImVec2& vmin, const ImVec2& vmax) {
    const SensorView& v = frame_.ui->sensors;
    const SensorOverlays& o = v.overlays;
    if (v.items.empty()) return;
    const ViewProjector project = Projector(vmin, vmax);
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    list->PushClipRect(vmin, vmax, true);
    const float s = nui::Scale();
    const auto line = [&](const math::Vec3& a, const math::Vec3& b, ImU32 color, float width) {
        ImVec2 pa, pb;
        if (project(a, &pa) && project(b, &pb)) list->AddLine(pa, pb, color, width * s);
    };
    // The selected sensor shows while the overlays are off.
    for (size_t i = 0; i < v.items.size(); ++i) {
        const SensorItem& item = v.items[i];
        const bool selected = static_cast<int>(i) == v.selected;
        if (!item.posed || (!o.on && !selected)) continue;
        const math::Transform& pose = item.pose;
        const ImU32 ink = U32(selected ? theme::kAccent : theme::Alpha(theme::kInfo, 0.75f));
        const float width = selected ? 2.0f : 1.0f;
        if (item.camera && (o.frustums || selected)) {
            // The view looks down local -Z with +Y up.
            const float depth = std::min(item.far_clip, kFrustumDepth);
            const float hh = depth * std::tan(item.vfov * 0.5f / kDeg);
            const float hw = hh * static_cast<float>(item.width) / static_cast<float>(item.height);
            math::Vec3 c[4];
            for (int k = 0; k < 4; ++k)
                c[k] = pose.TransformPoint(math::Vec3{(k == 1 || k == 2) ? hw : -hw, k < 2 ? hh : -hh, -depth});
            for (int k = 0; k < 4; ++k) {
                line(pose.position, c[k], ink, width);
                line(c[k], c[(k + 1) % 4], ink, width);
            }
            ImVec2 t[3];
            if (project(pose.TransformPoint(math::Vec3{-hw * 0.4f, hh * 1.1f, -depth}), &t[0]) &&
                project(pose.TransformPoint(math::Vec3{hw * 0.4f, hh * 1.1f, -depth}), &t[1]) &&
                project(pose.TransformPoint(math::Vec3{0.0f, hh * 1.5f, -depth}), &t[2]))
                list->AddTriangleFilled(t[0], t[1], t[2], ink);
        } else if (item.lidar && (o.fans || selected)) {
            // Each drawn row sweeps the azimuths; rays run out from the origin to the reach.
            const float reach = std::min(item.max_range, kFanReach);
            const int rows = std::min(static_cast<int>(item.el_count), kFanRows);
            const ImU32 ray = U32(theme::Alpha(selected ? theme::kAccent : theme::kInfo, selected ? 0.45f : 0.25f));
            for (int r = 0; r < rows; ++r) {
                const float el = rows > 1 ? item.el_min + (item.el_max - item.el_min) * static_cast<float>(r) / static_cast<float>(rows - 1)
                                          : item.el_min;
                const auto tip = [&](int k) {
                    const float az = item.az_min + (item.az_max - item.az_min) * static_cast<float>(k) / kFanRays;
                    return pose.TransformPoint(
                        math::Vec3{std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)} * reach);
                };
                for (int k = 0; k <= kFanRays; ++k) {
                    line(pose.position, tip(k), ray, 1.0f);
                    if (k > 0) line(tip(k - 1), tip(k), ink, width);
                }
            }
        } else if (!item.camera && !item.lidar && (o.frames || selected)) {
            const math::Vec3 units[3] = {math::Vec3::UnitX(), math::Vec3::UnitY(), math::Vec3::UnitZ()};
            const ImU32 colors[3] = {U32(theme::kAxisX), U32(theme::kAxisY), U32(theme::kAxisZ)};
            for (int k = 0; k < 3; ++k) line(pose.position, pose.TransformPoint(units[k] * kAxis), colors[k], width + 0.5f);
            ImVec2 p;
            if (project(pose.position, &p)) {
                const float r = (selected ? 5.0f : 4.0f) * s;
                const ImVec2 d[4] = {ImVec2(p.x, p.y - r), ImVec2(p.x + r, p.y), ImVec2(p.x, p.y + r), ImVec2(p.x - r, p.y)};
                list->AddQuadFilled(d[0], d[1], d[2], d[3], U32(theme::kBg0));
                list->AddQuad(d[0], d[1], d[2], d[3], ink, 1.5f * s);
            }
        }
        ImVec2 at;
        if ((o.labels || selected) && project(pose.position, &at)) {
            const std::string text = Iconed(FindSensorKind(item.type).glyph, item.name.c_str());
            const ImVec2 ts = ImGui::CalcTextSize(text.c_str());
            const ImVec2 a(at.x + 8.0f * s, at.y - ts.y - 8.0f * s), b(a.x + ts.x + 10.0f * s, a.y + ts.y + 4.0f * s);
            draw::Chamfer(list, a, b, 3.0f * s, U32(theme::Alpha(theme::kBg0, 0.85f)),
                          U32(selected ? theme::kAccent : theme::Alpha(theme::kLineHi, 0.8f)));
            list->AddText(ImVec2(a.x + 5.0f * s, a.y + 2.0f * s), U32(selected ? theme::kAccent : theme::kFgMuted), text.c_str());
        }
    }
    list->PopClipRect();
}

void ImGuiLayer::SensorContextMenu(int index) {
    ViewerUiState& ui = *frame_.ui;
    const SensorItem& item = ui.sensors.items[static_cast<size_t>(index)];
    if (ImGui::IsWindowAppearing()) std::snprintf(ui.rename_buf, sizeof(ui.rename_buf), "%s", item.name.c_str());
    nui::PushTextStyle(nui::TextStyle::Label);
    ImGui::TextColored(theme::kFgDim, "%s", Upper(Tr(FindSensorKind(item.type).label)).c_str());
    nui::PopTextStyle();
    ImGui::Separator();
    ImGui::SetNextItemWidth(170.0f * nui::Scale());
    const bool enter = ImGui::InputTextWithHint("##rename", Tr("New name"), ui.rename_buf, sizeof(ui.rename_buf),
                                                ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if (ImGui::SmallButton(Tr("Rename")) || enter) {
        Queue("sensor.set", Args({{"index", json::Value::Int(index)}, {"name", json::Value::Str(ui.rename_buf)}}));
        ImGui::CloseCurrentPopup();
    }
    if (ImGui::MenuItemEx(Tr("Remove Sensor"), icon::kTrash, "Delete"))
        Queue("sensor.remove", Args({{"index", json::Value::Int(index)}}));
}

void ImGuiLayer::DrawSensorInspector() {
    const SensorView& v = frame_.ui->sensors;
    if (sensor_edit_index_ != v.selected || sensor_edit_version_ != v.version) {
        sensor_edit_ = v.items[static_cast<size_t>(v.selected)];
        sensor_edit_index_ = v.selected;
        sensor_edit_version_ = v.version;
        std::snprintf(sensor_name_, sizeof(sensor_name_), "%s", sensor_edit_.name.c_str());
    }
    SensorItem& e = sensor_edit_;
    // The copy is taken again next frame, so a refused edit shows the record as it stays.
    const auto commit = [this](json::Value fields) {
        fields.Set("index", json::Value::Int(sensor_edit_index_));
        Queue("sensor.set", std::move(fields));
        sensor_edit_version_ = ~0ull;
    };
    const float s = nui::Scale();
    if (!ImGui::BeginTable("##sensor", 2, ImGuiTableFlags_SizingStretchProp)) return;
    ImGui::TableSetupColumn("##key", ImGuiTableColumnFlags_WidthFixed, 96.0f * s);
    ImGui::TableSetupColumn("##value", ImGuiTableColumnFlags_WidthStretch);
    const auto row = [](const char* label) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(theme::kFgMuted, "%s", Tr(label));
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
    };
    const auto finished = [] { return ImGui::IsItemDeactivatedAfterEdit(); };

    row("Name");
    ImGui::InputText("##name", sensor_name_, sizeof(sensor_name_));
    if (finished()) commit(Args({{"name", json::Value::Str(sensor_name_)}}));
    row("Type");
    const SensorKind& kind = FindSensorKind(e.type);
    if (ImGui::BeginCombo("##type", Iconed(kind.glyph, Tr(kind.label)).c_str())) {
        for (const SensorKind& k : SensorKinds())
            if (ImGui::Selectable(Iconed(k.glyph, Tr(k.label)).c_str(), e.type == k.type))
                commit(Args({{"type", json::Value::Str(k.type)}}));
        ImGui::EndCombo();
    }
    row("Mount");
    const std::string mount = !e.mount.empty() ? e.mount : e.frame == "world" ? std::string(Tr("World")) : e.frame;
    if (ImGui::BeginCombo("##mount", mount.c_str())) {
        if (ImGui::Selectable(Tr("World"), e.frame == "world")) commit(Args({{"mount", json::Value::Str("")}}));
        for (const std::string& path : v.bodies)
            if (ImGui::Selectable(path.c_str(), path == e.mount)) commit(Args({{"mount", json::Value::Str(path)}}));
        ImGui::EndCombo();
    }
    row("Position");
    ImGui::DragFloat3("##position", &e.offset.position.x, 0.005f, 0.0f, 0.0f, "%.3f");
    if (finished()) commit(Args({{"position", Vec3Json(e.offset.position)}}));
    row("Rotation");
    ImGui::DragFloat3("##rotation", &e.rotation_deg.x, 0.5f, 0.0f, 0.0f, "%.1f\xC2\xB0");
    if (finished()) commit(Args({{"rotation_deg", Vec3Json(e.rotation_deg)}}));
    row("Rate");
    ImGui::DragFloat("##rate", &e.rate_hz, 1.0f, 0.0f, 10000.0f, "%.0f Hz");
    if (finished()) commit(Args({{"rate_hz", json::Value::Float(e.rate_hz)}}));
    row("Period");
    int period = static_cast<int>(e.update_period);
    if (ImGui::DragInt("##period", &period, 0.1f, 1, 1000, Tr("every %d steps"))) e.update_period = static_cast<uint32_t>(period);
    if (finished()) commit(Args({{"update_period", json::Value::Int(e.update_period)}}));

    if (e.camera) {
        row("Image");
        int size[2] = {static_cast<int>(e.width), static_cast<int>(e.height)};
        if (ImGui::DragInt2("##image", size, 1.0f, 1, 4096)) {
            e.width = static_cast<uint32_t>(size[0]);
            e.height = static_cast<uint32_t>(size[1]);
        }
        bool camera = finished();
        ImGui::SetItemTooltip("%s", Tr("Every camera renders one image size; a change here resizes them all."));
        row("Vertical FOV");
        ImGui::DragFloat("##vfov", &e.vfov, 0.5f, 1.0f, 179.0f, "%.1f\xC2\xB0");
        camera = finished() || camera;
        row("Near");
        ImGui::DragFloat("##near", &e.near_clip, 0.001f, 0.001f, 100.0f, "%.3f m");
        camera = finished() || camera;
        row("Far");
        ImGui::DragFloat("##far", &e.far_clip, 0.1f, 0.01f, 10000.0f, "%.2f m");
        camera = finished() || camera;
        if (camera)
            commit(Args({{"camera", Args({{"width", json::Value::Int(e.width)},
                                          {"height", json::Value::Int(e.height)},
                                          {"vfov", json::Value::Float(e.vfov)},
                                          {"near", json::Value::Float(e.near_clip)},
                                          {"far", json::Value::Float(e.far_clip)}})}}));
        row("AOV");
        const uint32_t shown = e.aov_mask == 0u ? kAllAovs : e.aov_mask;
        std::string planes;
        for (const auto& [label, bit] : kAovs)
            if ((shown & bit) != 0u) planes += (planes.empty() ? "" : ", ") + std::string(Tr(label));
        if (ImGui::BeginCombo("##aov", planes.c_str())) {
            for (const auto& [label, bit] : kAovs) {
                bool on = (shown & bit) != 0u;
                ImGui::BeginDisabled(shown == bit);
                if (ImGui::Checkbox(Tr(label), &on))
                    commit(Args({{"aov_mask", json::Value::Int(on ? shown | bit : shown & ~bit)}}));
                ImGui::EndDisabled();
            }
            ImGui::EndCombo();
        }
    }
    if (e.lidar) {
        row("Rays");
        int rays[2] = {static_cast<int>(e.az_count), static_cast<int>(e.el_count)};
        if (ImGui::DragInt2("##rays", rays, 0.5f, 1, 4096)) {
            e.az_count = static_cast<uint32_t>(rays[0]);
            e.el_count = static_cast<uint32_t>(rays[1]);
        }
        bool lidar = finished();
        ImGui::SetItemTooltip("%s", Tr("Azimuth by elevation rays; every lidar casts one fan, so a change here sets them all."));
        row("Azimuth");
        float az[2] = {e.az_min * kDeg, e.az_max * kDeg};
        if (ImGui::DragFloat2("##azimuth", az, 0.5f, -180.0f, 180.0f, "%.1f\xC2\xB0")) {
            e.az_min = az[0] / kDeg;
            e.az_max = az[1] / kDeg;
        }
        lidar = finished() || lidar;
        row("Elevation");
        float el[2] = {e.el_min * kDeg, e.el_max * kDeg};
        if (ImGui::DragFloat2("##elevation", el, 0.25f, -90.0f, 90.0f, "%.1f\xC2\xB0")) {
            e.el_min = el[0] / kDeg;
            e.el_max = el[1] / kDeg;
        }
        lidar = finished() || lidar;
        row("Range");
        float range[2] = {e.min_range, e.max_range};
        if (ImGui::DragFloat2("##range", range, 0.05f, 0.0f, 1000.0f, "%.2f m")) {
            e.min_range = range[0];
            e.max_range = range[1];
        }
        lidar = finished() || lidar;
        if (lidar)
            commit(Args({{"lidar", Args({{"az_count", json::Value::Int(e.az_count)},
                                         {"el_count", json::Value::Int(e.el_count)},
                                         {"az_min", json::Value::Float(e.az_min)},
                                         {"az_max", json::Value::Float(e.az_max)},
                                         {"el_min", json::Value::Float(e.el_min)},
                                         {"el_max", json::Value::Float(e.el_max)},
                                         {"min_range", json::Value::Float(e.min_range)},
                                         {"max_range", json::Value::Float(e.max_range)}})}}));
    }
    ImGui::EndTable();
    if (ImGui::Button(Iconed(icon::kTrash, Tr("Remove Sensor")).c_str()))
        Queue("sensor.remove", Args({{"index", json::Value::Int(sensor_edit_index_)}}));
}

}  // namespace nuka::runtime::app::viewer
