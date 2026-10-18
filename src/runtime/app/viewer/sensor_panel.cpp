// The sensors in the editor: the types it offers, their overlays over the scene viewport, the
// selected sensor's fields in Properties and the Sensors panel's live readings.

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"

#include "nuka/nuka.h"
#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"
#include "runtime/app/viewer/editor_dataset.hpp"
#include "runtime/app/viewer/shell_util.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
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
// An instance plane's value where no surface was hit.
constexpr uint32_t kMissId = 0xFFFFFFFFu;
// The state plot draws this many of a sensor's values, in these colors.
constexpr size_t kMaxSeries = 8u;
constexpr ImVec4 kSeries[] = {theme::kAxisX, theme::kAxisY, theme::kAxisZ, theme::kWarn,
                              theme::kInfo,  theme::kAccent, theme::kErr};

json::Value Vec3Json(const math::Vec3& v) {
    json::Value a = json::Value::Array();
    for (const float c : {v.x, v.y, v.z}) a.PushBack(json::Value::Float(c));
    return a;
}

ImPlotSpec Line(const ImVec4& color, float weight) {
    ImPlotSpec spec;
    spec.LineColor = color;
    spec.LineWeight = weight * nui::Scale();
    return spec;
}

// Near surfaces warm, far ones cold.
ImU32 FalseColor(float t) {
    static constexpr float kStops[5][3] = {{1.00f, 0.96f, 0.72f}, {0.98f, 0.70f, 0.22f}, {0.84f, 0.29f, 0.30f},
                                           {0.38f, 0.16f, 0.52f}, {0.06f, 0.07f, 0.18f}};
    const float x = std::clamp(t, 0.0f, 1.0f) * 4.0f;
    const int i = std::min(static_cast<int>(x), 3);
    const float f = x - static_cast<float>(i);
    const auto mix = [&](int c) { return static_cast<int>((kStops[i][c] + (kStops[i + 1][c] - kStops[i][c]) * f) * 255.0f); };
    return IM_COL32(mix(0), mix(1), mix(2), 255);
}

// A steady color per instance.
ImU32 InstanceColor(uint32_t instance) {
    uint32_t h = (instance + 1u) * 2654435761u;
    h ^= h >> 16u;
    return IM_COL32(72u + (h & 0x9Fu), 72u + ((h >> 8u) & 0x9Fu), 72u + ((h >> 20u) & 0x9Fu), 255u);
}

// A steady color per semantic class; surfaces without a label stay dim.
ImU32 ClassColor(uint32_t k) {
    return k == 0u ? U32(theme::kFgDim) : InstanceColor(k * 40503u);
}

// What a state sensor's value `k` is, when its type fixes how many it has; else v0, v1, ...
std::string ValueName(const std::string& type, size_t count, size_t k) {
    static const std::pair<const char*, std::vector<const char*>> kNames[] = {
        {"imu", {"ax", "ay", "az", "wx", "wy", "wz"}},
        {"frame_pose", {"x", "y", "z", "qw", "qx", "qy", "qz"}},
        {"joint_state", {"q", "qd"}},
        {"linear_velocity", {"vx", "vy", "vz"}},
        {"force_torque", {"fx", "fy", "fz", "tx", "ty", "tz"}},
        {"contact", {"f"}}};
    for (const auto& [t, names] : kNames)
        if (type == t && names.size() == count) return names[k];
    return "v" + std::to_string(k);
}

// What pixel (x, y) of a reading holds, as hovering it shows; a range image runs azimuth to the
// right and elevation up.
std::string PixelText(const SensorPreview& p, const SensorItem& item, const LabelView& labels, int x, int y) {
    if (p.plane == kRangeChannel) {
        const uint32_t el = p.height - 1u - static_cast<uint32_t>(y);
        const float az = item.az_min + (item.az_max - item.az_min) * (static_cast<float>(x) + 0.5f) / static_cast<float>(p.width);
        const float elev = item.el_min + (item.el_max - item.el_min) * (static_cast<float>(el) + 0.5f) / static_cast<float>(p.height);
        return Fmt("az %.1f\xC2\xB0  el %.1f\xC2\xB0  %.3f m", az * kDeg, elev * kDeg,
                   p.values[static_cast<size_t>(x) * p.height + el]);
    }
    const float* v = &p.values[(static_cast<size_t>(y) * p.width + static_cast<size_t>(x)) * p.channels];
    const std::string at = Fmt("(%d, %d)  ", x, y);
    if (p.plane == NUKA_SENSOR_CHANNEL_DEPTH) return at + Fmt("%.3f m", v[0]);
    if (p.plane != NUKA_SENSOR_CHANNEL_PRIM) return at + Fmt("%.3f  %.3f  %.3f", v[0], v[1], v[2]);
    uint32_t bits = 0u;
    std::memcpy(&bits, v, sizeof(bits));
    if (bits == kMissId) return at + Tr("no hit");
    const std::string& node = p.instances[bits >> 20u];
    if (p.semantic) {
        const std::string label = LabelOf(labels.labels, node);
        return at + (label.empty() ? std::string(Tr("unlabeled")) : label) + "  " + node;
    }
    return at + Fmt("instance %u  prim %u  ", bits >> 20u, bits & 0xFFFFFu) + node;
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
    // The previewed lidar's returns in the shown environment; misses at its range limit stay out.
    const SensorPreview& p = v.preview;
    if (p.on && p.plane == kRangeChannel && p.read >= 0 && static_cast<size_t>(p.read) < v.items.size() &&
        v.items[static_cast<size_t>(p.read)].lidar && v.items[static_cast<size_t>(p.read)].posed) {
        const SensorItem& item = v.items[static_cast<size_t>(p.read)];
        const ImU32 dot = U32(theme::kWarn);
        const float r = 1.5f * s;
        for (uint32_t a = 0; a < p.width; ++a) {
            const float az = item.az_min + (item.az_max - item.az_min) * (static_cast<float>(a) + 0.5f) / static_cast<float>(p.width);
            for (uint32_t e = 0; e < p.height; ++e) {
                const float range = p.values[static_cast<size_t>(a) * p.height + e];
                if (range >= item.max_range) continue;
                const float el = item.el_min + (item.el_max - item.el_min) * (static_cast<float>(e) + 0.5f) / static_cast<float>(p.height);
                const math::Vec3 dir{std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el)};
                ImVec2 at;
                if (project(item.pose.TransformPoint(dir * range), &at))
                    list->AddRectFilled(ImVec2(at.x - r, at.y - r), ImVec2(at.x + r, at.y + r), dot);
            }
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

void ImGuiLayer::BuildSensors(nui::VStack& root) {
    root.Add<nui::Custom>([this] { DrawSensorsPanel(); });
}

void ImGuiLayer::UpdateSensorTexture() {
    const SensorView& v = frame_.ui->sensors;
    const SensorPreview& p = v.preview;
    const LabelView& labels = frame_.ui->labels;
    if (sensor_texture_version_ == p.version && sensor_texture_labels_ == labels.version &&
        sensor_texture_semantic_ == p.semantic)
        return;
    sensor_texture_version_ = p.version;
    sensor_texture_labels_ = labels.version;
    sensor_texture_semantic_ = p.semantic;
    if (p.plane == kStatePlane || p.read < 0 || static_cast<size_t>(p.read) >= v.items.size()) return;
    const SensorItem& item = v.items[static_cast<size_t>(p.read)];
    const int w = static_cast<int>(p.width), h = static_cast<int>(p.height);
    if (sensor_texture_ && (sensor_texture_->Width != w || sensor_texture_->Height != h)) {
        sensor_texture_->SetStatus(ImTextureStatus_WantDestroy);
        rt_retired_.push_back(std::move(sensor_texture_));
    }
    if (!sensor_texture_) {
        sensor_texture_ = std::make_unique<ImTextureData>();
        sensor_texture_->Create(ImTextureFormat_RGBA32, w, h);
        ImGui::RegisterUserTexture(sensor_texture_.get());
    } else {
        sensor_texture_->UpdateRect = ImTextureRect{0, 0, static_cast<unsigned short>(w), static_cast<unsigned short>(h)};
        if (sensor_texture_->Status == ImTextureStatus_OK) sensor_texture_->SetStatus(ImTextureStatus_WantUpdates);
    }
    // Depth and range spread their hits over the false color; misses sit at the far limit.
    const float limit = p.plane == kRangeChannel ? item.max_range : item.far_clip;
    const auto hit = [&](float d) { return std::isfinite(d) && d > 0.0f && d < limit; };
    float lo = FLT_MAX, hi = 0.0f;
    if (p.plane == NUKA_SENSOR_CHANNEL_DEPTH || p.plane == kRangeChannel)
        for (const float d : p.values)
            if (hit(d)) {
                lo = std::min(lo, d);
                hi = std::max(hi, d);
            }
    const auto shade = [&](float d) { return hit(d) ? FalseColor((d - lo) / std::max(hi - lo, 1.0e-6f)) : U32(theme::kBg0); };
    const auto byte = [](float c) { return static_cast<int>(std::clamp(c, 0.0f, 1.0f) * 255.0f + 0.5f); };
    const bool semantic = p.semantic && p.plane == NUKA_SENSOR_CHANNEL_PRIM;
    const std::vector<uint32_t> classes =
        semantic ? InstanceClasses(labels.labels, labels.classes, p.instances) : std::vector<uint32_t>();
    ImU32* pixels = static_cast<ImU32*>(sensor_texture_->GetPixels());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            ImU32& c = pixels[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];
            if (p.plane == kRangeChannel) {
                c = shade(p.values[static_cast<size_t>(x) * p.height + (p.height - 1u - static_cast<uint32_t>(y))]);
                continue;
            }
            const float* at = &p.values[(static_cast<size_t>(y) * p.width + static_cast<size_t>(x)) * p.channels];
            uint32_t bits = 0u;
            std::memcpy(&bits, at, sizeof(bits));
            if (p.plane == NUKA_SENSOR_CHANNEL_DEPTH) c = shade(at[0]);
            else if (p.plane == NUKA_SENSOR_CHANNEL_PRIM && bits == kMissId) c = U32(theme::kBg0);
            else if (semantic) c = ClassColor(classes[bits >> 20u]);
            else if (p.plane == NUKA_SENSOR_CHANNEL_PRIM) c = InstanceColor(bits >> 20u);
            else if (p.plane == NUKA_SENSOR_CHANNEL_NORMAL)
                c = IM_COL32(byte(at[0] * 0.5f + 0.5f), byte(at[1] * 0.5f + 0.5f), byte(at[2] * 0.5f + 0.5f), 255);
            else c = IM_COL32(byte(at[0]), byte(at[1]), byte(at[2]), 255);
        }
    sensor_lo_ = lo;
    sensor_hi_ = hi;
}

void ImGuiLayer::DrawSensorsPanel() {
    const SensorView& v = frame_.ui->sensors;
    const SensorPreview& p = v.preview;
    const float s = nui::Scale();
    if (v.items.empty()) {
        ImGui::TextDisabled("%s", Tr("No sensors yet; add one from Create > Sensor."));
        return;
    }
    const SensorItem* shown = p.sensor >= 0 ? &v.items[static_cast<size_t>(p.sensor)] : nullptr;
    // A wide, short panel puts the list beside the reading, a tall one above it.
    const ImVec2 room = ImGui::GetContentRegionAvail();
    const bool wide = room.x > 2.0f * room.y;
    if (wide) ImGui::BeginChild("##sensor_side", ImVec2(std::min(360.0f * s, 0.4f * room.x), 0.0f));
    if (ImGui::Button(Iconed(p.on ? icon::kPause : icon::kPlay, Tr(p.on ? "Pause" : "Preview")).c_str()))
        Queue("sensor.preview", Args({{"on", json::Value::Bool(!p.on)}}));
    if (shown != nullptr && shown->camera) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(130.0f * s);
        if (ImGui::BeginCombo("##plane", Tr(p.semantic ? "Semantic" : kAovs[p.channel].first))) {
            for (uint32_t c = 0; c < std::size(kAovs); ++c)
                if (ImGui::Selectable(Tr(kAovs[c].first), c == p.channel && !p.semantic))
                    Queue("sensor.preview", Args({{"channel", json::Value::Str(kSensorChannels[c])}}));
            if (ImGui::Selectable(Tr("Semantic"), p.semantic))
                Queue("sensor.preview", Args({{"channel", json::Value::Str("semantic")}}));
            ImGui::EndCombo();
        }
    }

    // Picking a row selects the sensor and reads it live.
    const float rows = std::min(static_cast<float>(v.items.size()), 6.0f) + 1.0f;
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("##sensor_list", 3, flags, ImVec2(0.0f, wide ? -FLT_MIN : rows * ImGui::GetFrameHeight()))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn(Tr("Sensor"), ImGuiTableColumnFlags_WidthStretch, 1.4f);
        ImGui::TableSetupColumn(Tr("Mount"), ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn(Tr("State"), ImGuiTableColumnFlags_WidthFixed, 56.0f * s);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < v.items.size(); ++i) {
            const SensorItem& item = v.items[i];
            const int index = static_cast<int>(i);
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::PushID(index);
            if (ImGui::Selectable(Iconed(FindSensorKind(item.type).glyph, item.name.c_str()).c_str(), index == p.sensor,
                                  ImGuiSelectableFlags_SpanAllColumns)) {
                Queue("sensor.select", Args({{"index", json::Value::Int(index)}}));
                Queue("sensor.preview", Args({{"index", json::Value::Int(index)}, {"on", json::Value::Bool(true)}}));
            }
            ImGui::PopID();
            ImGui::TableSetColumnIndex(1);
            const std::string mount = item.mount.empty() ? Tr("World") : item.mount.substr(item.mount.rfind('/') + 1u);
            ImGui::TextColored(theme::kFgMuted, "%s", mount.c_str());
            ImGui::TableSetColumnIndex(2);
            if (index == p.sensor && !p.error.empty()) ImGui::TextColored(theme::kErr, "%s", Tr("ERROR"));
            else if (index == p.sensor && p.on) ImGui::TextColored(theme::kAccent, "%s", Tr("LIVE"));
            else ImGui::TextColored(theme::kFgDim, "%s", item.rate_hz > 0.0f ? Fmt("%.0f Hz", item.rate_hz).c_str() : "-");
        }
        ImGui::EndTable();
    }
    if (!wide) {
        DrawSensorReading(shown);
        return;
    }
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##sensor_reading");
    DrawSensorReading(shown);
    ImGui::EndChild();
}

void ImGuiLayer::DrawSensorReading(const SensorItem* shown) {
    const SensorPreview& p = frame_.ui->sensors.preview;
    const float s = nui::Scale();
    if (!p.error.empty()) ImGui::TextColored(theme::kErr, "%s", p.error.c_str());
    if (shown == nullptr) {
        ImGui::TextDisabled("%s", Tr("Pick a sensor to read it."));
        return;
    }
    if (p.read != p.sensor || p.version == 0u) {
        ImGui::TextDisabled("%s", Tr(p.on ? "Reading..." : "Press Preview to read this sensor live."));
        return;
    }

    if (p.plane == kStatePlane) {
        const int n = static_cast<int>(p.times.size());
        const size_t plotted = std::min(p.values.size(), kMaxSeries);
        // The plotted values ride the legend; the rest are listed under the plot.
        std::string rest;
        for (size_t k = plotted; k < p.values.size() && k < 2u * kMaxSeries; ++k)
            rest += ValueName(shown->type, p.values.size(), k) + Fmt(" %.3f   ", p.values[k]);
        if (p.values.size() > 2u * kMaxSeries) rest += "\xE2\x80\xA6";
        const float caption = rest.empty() ? 0.0f : ImGui::GetTextLineHeightWithSpacing() * 2.0f;
        const float height = std::max(80.0f * s, ImGui::GetContentRegionAvail().y - caption);
        ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(theme::kSpace * s, theme::kSpace * s));
        // Steady values would otherwise sit on the plot's edges.
        ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.02f, 0.1f));
        if (ImPlot::BeginPlot("##sensor_plot", ImVec2(-FLT_MIN, height),
                              ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoFrame)) {
            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupLegend(ImPlotLocation_North,
                                ImPlotLegendFlags_NoButtons | ImPlotLegendFlags_Outside | ImPlotLegendFlags_Horizontal);
            for (size_t k = 0; k < plotted; ++k) {
                // The part after ### keeps the line's id while its value changes.
                const std::string label =
                    ValueName(shown->type, p.values.size(), k) + Fmt(" %.3f###%zu", p.values[k], k);
                ImPlot::PlotLine(label.c_str(), p.times.data(), p.series[k].data(), n,
                                 Line(kSeries[k % std::size(kSeries)], 1.5f));
            }
            ImPlot::EndPlot();
        }
        ImPlot::PopStyleVar(2);
        if (!rest.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(theme::kFgMuted, "%s", rest.c_str());
            ImGui::PopTextWrapPos();
        }
        return;
    }

    UpdateSensorTexture();
    const float w = static_cast<float>(p.width), h = static_cast<float>(p.height);
    const ImVec2 room = ImGui::GetContentRegionAvail();
    const float caption = ImGui::GetTextLineHeightWithSpacing();
    ImVec2 size(room.x, room.x * h / w);
    if (size.y > room.y - caption && room.y - caption > 60.0f * s) size = ImVec2((room.y - caption) * w / h, room.y - caption);
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Image(sensor_texture_->GetTexRef(), size);
    if (ImGui::IsItemHovered()) {
        const ImVec2 m = ImGui::GetMousePos();
        const int x = std::clamp(static_cast<int>((m.x - at.x) / size.x * w), 0, static_cast<int>(p.width) - 1);
        const int y = std::clamp(static_cast<int>((m.y - at.y) / size.y * h), 0, static_cast<int>(p.height) - 1);
        ImGui::SetTooltip("%s", PixelText(p, *shown, frame_.ui->labels, x, y).c_str());
    }
    const char* plane = p.plane == kRangeChannel                               ? "Range"
                        : p.semantic && p.plane == NUKA_SENSOR_CHANNEL_PRIM ? "Semantic"
                                                                              : kAovs[p.plane].first;
    std::string line = Fmt("%u \xC3\x97 %u   %s", p.width, p.height, Tr(plane));
    if (p.plane == NUKA_SENSOR_CHANNEL_DEPTH || p.plane == kRangeChannel)
        line += sensor_hi_ > 0.0f ? Fmt("   %.2f\xE2\x80\x93%.2f m", sensor_lo_, sensor_hi_) : std::string("   ") + Tr("no hit");
    line += Fmt("   tick %llu", static_cast<unsigned long long>(p.tick));
    ImGui::TextColored(theme::kFgMuted, "%s", line.c_str());
}

}  // namespace nuka::runtime::app::viewer
