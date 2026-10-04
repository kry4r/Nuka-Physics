// Records the viewer workspace, contextual inspector and optional tools.
// Scene rendering consumes the same viewport rectangle as input and gizmos.

#include "runtime/app/viewer/imgui_layer.hpp"
#include "render/imgui/nuka_imgui.hpp"
#include "render/imgui/nuka_theme.hpp"

#include "imgui.h"
#include "imgui_internal.h"  // DockBuilder* for the one-time default layout
#include "ImGuizmo.h"        // in-viewport transform gizmo (vendored, MIT)

#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/window_input.hpp"
#include "runtime/app/viewer/file_dialog.hpp"
#include "scene/ecs/components.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>  // std::size for array-derived loop bounds
#include <memory>

namespace nuka::runtime::app::viewer {

namespace {

using namespace nuka::render::imgui::theme;

ImU32 Col(const ImVec4& c) { return ImGui::ColorConvertFloat4ToU32(c); }

void PushHeadingFont() { ImGui::PushFont(render::imgui::GetNukaFont(render::imgui::FontRole::Heading)); }
void PopFont() { ImGui::PopFont(); }

void SectionHeader(const char* label) {
    ImGui::Dummy(ImVec2(0.0f, 4.0f));
    PushHeadingFont();
    ImGui::TextUnformatted(label);
    PopFont();
    ImGui::Spacing();
}

// Flexible columns wrap long values without hiding them outside narrow panels.
void StatRow(const char* label, const char* value, const ImVec4& value_col) {
    ImGui::PushID(label);
    if (ImGui::BeginTable("##stat", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthStretch, 0.45f);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 0.55f);
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(kTextDim, "%s", label);
        ImGui::PopTextWrapPos();
        ImGui::TableNextColumn();
        ImGui::PushStyleColor(ImGuiCol_Text, value_col);
        ImGui::PushTextWrapPos(0.0f);
        const bool numeric = (value[0] >= '0' && value[0] <= '9') || value[0] == '-' || value[0] == '+';
        if (numeric) ImGui::PushFont(render::imgui::GetNukaFont(render::imgui::FontRole::Mono));
        ImGui::TextUnformatted(value);
        if (numeric) ImGui::PopFont();
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
        ImGui::EndTable();
    }
    ImGui::PopID();
}

// A small filled rounded "badge" (custom-drawn): accent for good, muted for idle.
void Badge(const char* text, const ImVec4& fill, const ImVec4& fg) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 pad(8.0f, 3.0f);
    const ImVec2 ts = ImGui::CalcTextSize(text);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 p1(p0.x + ts.x + pad.x * 2.0f, p0.y + ts.y + pad.y * 2.0f);
    dl->AddRectFilled(p0, p1, Col(fill), 5.0f);
    dl->AddText(ImVec2(p0.x + pad.x, p0.y + pad.y), Col(fg), text);
    ImGui::Dummy(ImVec2(ts.x + pad.x * 2.0f, ts.y + pad.y * 2.0f));
}

const char* MeshSourceLabel(render::MeshSource s) {
    return s == render::MeshSource::NkaMesh ? "mesh" : "prim";
}

// A node's display kind: a short badge label + its color, derived from the
// entity's components (a path-prefix robot GROUP carries only a name -> "group").
struct NodeKind {
    const char* label;
    ImVec4      color;
};

NodeKind ClassifyNode(const nuka::scene::Registry& reg, nuka::scene::EntityId e) {
    using namespace nuka::scene;
    if (e == kInvalidEntity || !reg.Alive(e)) return {"scene", kTextDim};
    if (const auto* sk = reg.Get<SystemKindComponent>(e)) {
        switch (sk->kind) {
            case SystemKindComponent::Articulated: return {"artic", kAccent};
            case SystemKindComponent::Soft:        return {"soft", kAccent};
            case SystemKindComponent::Cloth:       return {"cloth", kAccent};
            case SystemKindComponent::Fluid:       return {"fluid", kAccent};
            case SystemKindComponent::Rigid:       break;  // refine below
        }
    }
    if (const auto* body = reg.Get<RigidBodyComponent>(e)) {
        return body->kinematic ? NodeKind{"static", kTextDim} : NodeKind{"body", kText};
    }
    if (reg.Has<JointComponent>(e))          return {"joint", kAccentDim};
    if (reg.Has<CollisionShapeComponent>(e)) return {"geom", kTextDim};
    if (reg.Has<VisualMeshComponent>(e))     return {"visual", kTextDim};
    if (reg.Has<CameraComponent>(e))         return {"cam", kTextDim};
    if (reg.Has<LightComponent>(e))          return {"light", kTextDim};
    return {"group", kTextDim};
}

// The spawnable primitives + labels, shared by the Scene-panel Add picker and the
// tree context-menu Add submenu (one list, no per-site drift).
struct SpawnChoice { const char* label; PrimitiveKind kind; };
constexpr SpawnChoice kSpawnChoices[] = {
    {"Box",     PrimitiveKind::Box},
    {"Sphere",  PrimitiveKind::Sphere},
    {"Capsule", PrimitiveKind::Capsule},
    {"Plane",   PrimitiveKind::Plane},
};

// Latch a one-shot spawn request (the viewer applies it between frames, never here).
void RequestSpawn(ViewerUiState& ui, PrimitiveKind kind) {
    ui.spawn_request = true;
    ui.spawn_kind = kind;
}

// Recursively emit one ImGui TreeNode per SceneGraph node (name + kind badge);
// robots collapse as path-prefixed subtrees. Selection is keyed on node->entity.
bool SceneNodeMatches(const std::shared_ptr<nuka::scene::SceneNode>& node, const char* filter) {
    if (!node) return false;
    if (!filter[0] || node->name.find(filter) != std::string::npos) return true;
    for (auto child = node->first_child; child; child = child->next_sibling)
        if (SceneNodeMatches(child, filter)) return true;
    return false;
}

void RecordSceneNode(const nuka::scene::Registry& reg,
                     const std::shared_ptr<nuka::scene::SceneNode>& node,
                     ViewerUiState& ui, int depth, const std::string& path, const char* filter) {
    if (!SceneNodeMatches(node, filter)) return;
    const NodeKind kind = ClassifyNode(reg, node->entity);
    const bool has_children = static_cast<bool>(node->first_child);
    const bool valid_entity = node->entity != nuka::scene::kInvalidEntity;
    const bool selected = valid_entity && node->entity == ui.selected_entity;

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                               ImGuiTreeNodeFlags_OpenOnDoubleClick |
                               ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding;
    if (!has_children) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    if (selected)      flags |= ImGuiTreeNodeFlags_Selected;

    // Open only the root once so the top level (terrain + robot groups) shows;
    // deeper internals stay collapsed until the user expands them.
    ImGui::SetNextItemOpen(depth == 0 || filter[0], filter[0] ? ImGuiCond_Always : ImGuiCond_Once);
    ImGui::PushID(static_cast<int>(node->id));
    const char* name = node->name.empty() ? "(unnamed)" : node->name.c_str();
    const bool open = ImGui::TreeNodeEx("##node", flags, "%s", name);
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen() && valid_entity) {
        ui.selected_entity = node->entity;
    }
    // Drag a node onto another to reparent it under that node (drop on the root
    // "Scene" -> scene root). The payload is the source node's tree path.
    if (valid_entity && ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
        ImGui::SetDragDropPayload("NUKA_NODE", path.c_str(),
                                  path.size() + 1);
        ImGui::TextUnformatted(path.c_str());
        ImGui::EndDragDropSource();
    }
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* pl = ImGui::AcceptDragDropPayload("NUKA_NODE")) {
            ui.reparent_src.assign(static_cast<const char*>(pl->Data));
            ui.reparent_dst = path;  // this node is the new parent
            ui.reparent_request = true;
        }
        ImGui::EndDragDropTarget();
    }
    // Right-click context menu: the structural tree edits, keyed on THIS node (the
    // viewer applies them through the general SceneIR seam + a re-cook).
    if (ImGui::BeginPopupContextItem("##nodectx")) {
        if (valid_entity) ui.selected_entity = node->entity;
        ImGui::TextColored(kTextDim, "%s", name);
        ImGui::Separator();
        ImGui::SetNextItemWidth(150.0f);
        ImGui::InputTextWithHint("##rn", "new name", ui.rename_buf, sizeof(ui.rename_buf));
        ImGui::SameLine();
        if (ImGui::SmallButton("Rename")) { ui.rename_request = true; ImGui::CloseCurrentPopup(); }
        if (ImGui::BeginMenu("Add primitive")) {
            for (const SpawnChoice& c : kSpawnChoices)
                if (ImGui::MenuItem(c.label)) RequestSpawn(ui, c.kind);
            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Delete subtree")) ui.delete_request = true;
        ImGui::EndPopup();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kind.label);

    if (open && has_children) {
        for (auto child = node->first_child; child; child = child->next_sibling) {
            const std::string child_path =
                path.empty() ? child->name : path + "/" + child->name;
            RecordSceneNode(reg, child, ui, depth + 1, child_path, filter);
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
}

// -- gizmo matrix helpers (column-major float[16], glm/ImGuizmo convention) -----
// Mirror the renderer's camera basis + transform layout so the gizmo lands ON the
// rendered object. The projection is GL-style (y-up NDC): ImGuizmo applies its own
// screen y-flip, so the Vulkan y-flip is intentionally absent here.

void ModelMatrix(const math::Transform& t, float* m) {
    const math::Quat q = t.rotation.Normalized();
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    m[0] = 1.0f - 2.0f * (yy + zz); m[1] = 2.0f * (xy + wz);         m[2]  = 2.0f * (xz - wy); m[3] = 0.0f;
    m[4] = 2.0f * (xy - wz);        m[5] = 1.0f - 2.0f * (xx + zz);  m[6]  = 2.0f * (yz + wx); m[7] = 0.0f;
    m[8] = 2.0f * (xz + wy);        m[9] = 2.0f * (yz - wx);         m[10] = 1.0f - 2.0f * (xx + yy); m[11] = 0.0f;
    m[12] = t.position.x; m[13] = t.position.y; m[14] = t.position.z; m[15] = 1.0f;
}

// Extract pos + (orthonormal) rotation from a column-major model matrix. Scale is
// ignored by construction -- Translate/Rotate keep the 3x3 a pure rotation.
math::Transform TransformFromModel(const float* m) {
    math::Transform t;
    t.position = math::Vec3{m[12], m[13], m[14]};
    const float r00 = m[0], r10 = m[1], r20 = m[2];
    const float r01 = m[4], r11 = m[5], r21 = m[6];
    const float r02 = m[8], r12 = m[9], r22 = m[10];
    const float trace = r00 + r11 + r22;
    math::Quat q;
    if (trace > 0.0f) {
        float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * s; q.x = (r21 - r12) / s; q.y = (r02 - r20) / s; q.z = (r10 - r01) / s;
    } else if (r00 > r11 && r00 > r22) {
        float s = std::sqrt(1.0f + r00 - r11 - r22) * 2.0f;
        q.w = (r21 - r12) / s; q.x = 0.25f * s; q.y = (r01 + r10) / s; q.z = (r02 + r20) / s;
    } else if (r11 > r22) {
        float s = std::sqrt(1.0f + r11 - r00 - r22) * 2.0f;
        q.w = (r02 - r20) / s; q.x = (r01 + r10) / s; q.y = 0.25f * s; q.z = (r12 + r21) / s;
    } else {
        float s = std::sqrt(1.0f + r22 - r00 - r11) * 2.0f;
        q.w = (r10 - r01) / s; q.x = (r02 + r20) / s; q.y = (r12 + r21) / s; q.z = 0.25f * s;
    }
    t.rotation = q.Normalized();
    return t;
}

void ViewMatrix(const math::Vec3& eye, const math::Vec3& target,
                const math::Vec3& up, float* m) {
    const math::Vec3 f = (target - eye).Normalized();
    math::Vec3 s = f.Cross(up);
    if (s.LengthSq() < 1e-12f) s = math::Vec3{1.0f, 0.0f, 0.0f};
    s = s.Normalized();
    const math::Vec3 u = s.Cross(f);
    m[0] = s.x; m[4] = s.y; m[8]  = s.z;  m[12] = -s.Dot(eye);
    m[1] = u.x; m[5] = u.y; m[9]  = u.z;  m[13] = -u.Dot(eye);
    m[2] = -f.x; m[6] = -f.y; m[10] = -f.z; m[14] = f.Dot(eye);
    m[3] = 0.0f; m[7] = 0.0f; m[11] = 0.0f; m[15] = 1.0f;
}

void ProjMatrixGL(float fov_y_rad, float aspect, float znear, float zfar, float* m) {
    const float t = std::tan(fov_y_rad * 0.5f);
    for (int i = 0; i < 16; ++i) m[i] = 0.0f;
    m[0] = 1.0f / (aspect * t);
    m[5] = 1.0f / t;  // GL y-up NDC (ImGuizmo flips to screen itself)
    m[10] = (zfar + znear) / (znear - zfar);
    m[11] = -1.0f;
    m[14] = (2.0f * zfar * znear) / (znear - zfar);
}

}  // namespace

static void RecordCameraControls(const render::RenderWorld& world, CameraController& camera,
                                 ViewerUiState& ui_state, float aspect, bool compact = false) {
        SectionHeader(compact ? "View" : "Camera");
        if (!compact) {
        const math::Vec3 eye = camera.ResolvedEye();
        const math::Vec3 tgt = camera.ResolvedTarget();
        char buf[80];
        std::snprintf(buf, sizeof(buf), "%.2f  %.2f  %.2f", eye.x, eye.y, eye.z);
        StatRow("eye", buf, kText);
        std::snprintf(buf, sizeof(buf), "%.2f  %.2f  %.2f", tgt.x, tgt.y, tgt.z);
        StatRow("target", buf, kText);
        std::snprintf(buf, sizeof(buf), "%.1f deg", camera.Yaw() * 57.29578f);
        StatRow("yaw", buf, kTextDim);
        std::snprintf(buf, sizeof(buf), "%.1f deg", camera.Pitch() * 57.29578f);
        StatRow("pitch", buf, kTextDim);
        std::snprintf(buf, sizeof(buf), "%.2f", camera.Distance());
        StatRow("distance", buf, kTextDim);
        }

        ImGui::Dummy(ImVec2(0.0f, 6.0f));
        ImGui::TextColored(kTextDim, "field of view");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::SliderFloat("##fov", &camera.fov_degrees, 20.0f, 90.0f, "%.0f deg");

        ImGui::Dummy(ImVec2(0.0f, 6.0f));

        ImGui::BeginDisabled(world.instances.empty());
        if (ImGui::Button("Frame All (Home)", ImVec2(-1.0f, 0.0f))) camera.FrameAll(world, aspect);
        ImGui::EndDisabled();
        const bool can_frame_selected = ui_state.selected_entity != scene::kInvalidEntity &&
            std::any_of(world.instances.begin(), world.instances.end(), [&](const auto& instance) {
                return instance.entity == ui_state.selected_entity && instance.mesh_id < world.meshes.Count() &&
                       !world.meshes.Geometry(instance.mesh_id).Empty();
            });
        ImGui::BeginDisabled(!can_frame_selected);
        if (ImGui::Button("Frame Selected (F)", ImVec2(-1.0f, 0.0f)))
            camera.FrameSelected(world, ui_state.selected_entity, aspect);
        ImGui::EndDisabled();
        if (ImGui::Button("Reset View", ImVec2(-1.0f, 0.0f))) ui_state.camera_reset = true;

        ImGui::Dummy(ImVec2(0.0f, 10.0f));
        if (!compact) {
        SectionHeader("Controls");
        ImGui::TextColored(kTextDim, "LMB drag  orbit");
        ImGui::TextColored(kTextDim, "MMB / Shift  pan");
        ImGui::TextColored(kTextDim, "wheel  zoom");
        ImGui::TextWrapped("F / Home: frame selection / all while the pointer is over the viewport");
        ImGui::TextColored(kTextDim, "Ctrl+LMB  pick / drag entity");
        }
}

static void BuildDefaultDockLayout(ImGuiID id) {
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::DockBuilderRemoveNode(id);
    ImGui::DockBuilderAddNode(id, static_cast<ImGuiDockNodeFlags>(ImGuiDockNodeFlags_DockSpace) | ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::DockBuilderSetNodePos(id, vp->WorkPos);
    ImGui::DockBuilderSetNodeSize(id, vp->WorkSize);
    ImGuiID center = id;
    const float left_width = vp->WorkSize.x < 1100.0f ? 180.0f : 224.0f;
    const float right_width = vp->WorkSize.x < 1100.0f ? 260.0f : 288.0f;
    const ImGuiID left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, left_width / vp->WorkSize.x, nullptr, &center);
    const ImGuiID right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right,
                                                    right_width / (vp->WorkSize.x - left_width), nullptr, &center);
    const ImGuiID bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.32f, nullptr, &center);
    ImGui::DockBuilderDockWindow("Hierarchy", left);
    ImGui::DockBuilderDockWindow("Inspector", right);
    ImGui::DockBuilderDockWindow("Scene View", center);
    for (const char* name : {"Stats", "Physics Debug", "Drive", "Script", "Camera"})
        ImGui::DockBuilderDockWindow(name, bottom);
    for (const ImGuiID node : {left, right, center, bottom})
        if (auto* n = ImGui::DockBuilderGetNode(node)) n->LocalFlags |= ImGuiDockNodeFlags_AutoHideTabBar;
    ImGui::DockBuilderFinish(id);
}

void ImGuiLayer::EnableDocking(bool persist_layout) {
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = nullptr;
    if (!persist_layout) return;
#ifdef _WIN32
    const char* root = std::getenv("APPDATA");
#else
    const char* root = std::getenv("XDG_CONFIG_HOME");
#endif
    std::filesystem::path directory;
    if (root && root[0]) directory = root;
#ifndef _WIN32
    else if (const char* home = std::getenv("HOME")) directory = std::filesystem::path(home) / ".config";
#endif
    if (directory.empty()) return;
    directory /= "nuka";
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) return;
    layout_path_ = (directory / "viewer-layout-v2.ini").string();
    io.IniFilename = layout_path_.c_str();
}

void ImGuiLayer::RecordUi(const render::RenderWorld& world, const ViewerStats& stats,
                          CameraController& camera, ViewerUiState& ui_state,
                          const nuka::scene::SceneIR* scene,
                          uint32_t framebuffer_width, uint32_t framebuffer_height) {
    viewport_ = {};
    viewport_hovered_ = false;
    bool frame_all = false, frame_selected = false;
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::BeginFrame();
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGuiIO& io = ImGui::GetIO();
    const bool narrow = vp->Size.x < 960.0f;
    if (narrow != narrow_) {
        if (narrow) { hierarchy_wide_open_ = ui_state.show_hierarchy; ui_state.show_hierarchy = false; }
        else ui_state.show_hierarchy = hierarchy_wide_open_;
        narrow_ = narrow;
    }
    const ImGuiWindowFlags bar_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking |
                                      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kTopBar);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.0f, 10.0f));
    if (ImGui::BeginViewportSideBar("##NukaTopBar", vp, ImGuiDir_Up, 48.0f, bar_flags)) {
        render::imgui::DrawNukaLogo(26.0f);
        ImGui::SameLine(0.0f, 8.0f); PushHeadingFont(); ImGui::TextUnformatted("Nuka"); PopFont();
        ImGui::SameLine(0.0f, 24.0f);
        if (ImGui::Button("File")) ImGui::OpenPopup("File menu");
        if (ImGui::BeginPopup("File menu")) {
            if (ImGui::MenuItem("Open scene...")) {
                const auto path = OpenFileDialog("Open Scene", "Nuka scenes", "*.nks");
                if (!path.empty()) ui_state.load_request = path;
                else ui_state.show_file_panel = true;
            }
            if (ImGui::MenuItem("Open by path...")) ui_state.show_file_panel = true;
            ImGui::Separator();
            if (ImGui::MenuItem("Save", nullptr, false, ui_state.has_scene)) {
                if (ui_state.save_path[0]) ui_state.save_request = ui_state.save_path;
                else ui_state.show_file_panel = true;
            }
            if (ImGui::MenuItem("Save as...", nullptr, false, ui_state.has_scene)) {
                const auto path = SaveFileDialog("Save Scene", "Nuka scenes", "*.nks", ui_state.save_path);
                if (!path.empty()) ui_state.save_request = path;
                else ui_state.show_file_panel = true;
            }
            if (ImGui::MenuItem("Unload scene", nullptr, false, ui_state.has_scene)) ui_state.unload_request = true;
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Edit")) ImGui::OpenPopup("Edit menu");
        if (ImGui::BeginPopup("Edit menu")) {
            if (ImGui::MenuItem("Undo", "Ctrl+Z", false, ui_state.can_undo)) ui_state.undo_request = true;
            if (ImGui::MenuItem("Redo", "Ctrl+Y", false, ui_state.can_redo)) ui_state.redo_request = true;
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Window")) ImGui::OpenPopup("Window menu");
        if (ImGui::BeginPopup("Window menu")) {
            ImGui::MenuItem("Hierarchy", nullptr, &ui_state.show_hierarchy);
            ImGui::MenuItem("Inspector", nullptr, &ui_state.show_inspector);
            ImGui::Separator();
            ImGui::MenuItem("Physics Debug", nullptr, &ui_state.show_debug_panel);
            ImGui::MenuItem("Statistics", nullptr, &ui_state.show_stats_panel);
            ImGui::MenuItem("Drive / Teleop", nullptr, &ui_state.show_drive_panel);
            ImGui::MenuItem("Script console", nullptr, &ui_state.show_script_panel);
            ImGui::MenuItem("Camera", nullptr, &ui_state.show_camera_panel);
            ImGui::Separator();
            if (ImGui::MenuItem("Reset workspace layout")) reset_layout_ = true;
            ImGui::EndPopup();
        }
        ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 16.0f, (vp->Size.x - 250.0f) * 0.5f));
        ImGui::BeginDisabled(!ui_state.has_scene);
        ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kAccentDim);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, kAccentDim);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
        if (ImGui::Button(ui_state.playing ? "Pause" : "Play", ImVec2(72.0f, 28.0f))) {
            ui_state.playing = !ui_state.playing;
            ui_state.play_toggled = true;
        }
        ImGui::PopStyleColor(4);
        ImGui::SameLine(); ImGui::BeginDisabled(ui_state.playing);
        if (ImGui::Button("Step")) ui_state.step_requested = true;
        ImGui::EndDisabled(); ImGui::SameLine();
        if (ImGui::Button("Reset")) ui_state.reset_requested = true;
        ImGui::EndDisabled(); ImGui::SameLine();
        char speed[32]; std::snprintf(speed, sizeof(speed), "%.2gx", ui_state.speed);
        if (ImGui::Button(speed)) ImGui::OpenPopup("Speed");
        if (ImGui::BeginPopup("Speed")) {
            for (const float value : {0.25f, 0.5f, 1.0f, 2.0f, 4.0f}) {
                char label[16]; std::snprintf(label, sizeof(label), "%.2gx", value);
                if (ImGui::Selectable(label, ui_state.speed == value)) ui_state.speed = value;
            }
            ImGui::EndPopup();
        }
    }
    ImGui::End(); ImGui::PopStyleVar(); ImGui::PopStyleColor();
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.0f, 3.0f));
    if (ImGui::BeginViewportSideBar("##NukaStatus", vp, ImGuiDir_Down, 24.0f, bar_flags)) {
        ImGui::TextColored(kAccent, "%s", ui_state.has_scene ? (ui_state.playing ? "Running" : "Paused") : "No scene");
        ImGui::SameLine(0.0f, 24.0f);
        if (ImGui::SmallButton("Diagnostics")) ui_state.show_debug_panel = !ui_state.show_debug_panel;
        ImGui::SameLine();
        if (!ui_state.load_error.empty()) ImGui::TextColored(kError, "Scene load failed - File > Open by path");
        else if (!render::imgui::HasNukaCjkFont()) ImGui::TextColored(kWarn, "CJK font unavailable");
        else ImGui::TextColored(kTextDim, "%s", stats.session_note.empty() ? "F  Frame selected    Home  Frame all" : stats.session_note.c_str());
    }
    ImGui::End(); ImGui::PopStyleVar();

    const ImGuiID dockspace_id = ImGui::GetID("NukaWorkspace.v2");
    const bool stored_layout = ImGui::DockBuilderGetNode(dockspace_id) != nullptr;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    ImGui::DockSpaceOverViewport(dockspace_id, vp, ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::PopStyleColor();
    if ((!dock_built_ && !stored_layout) || reset_layout_) {
        BuildDefaultDockLayout(dockspace_id);
        reset_layout_ = false;
    }
    dock_built_ = true;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
    if (ImGui::Begin("Scene View", nullptr, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
                                              ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove)) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, kBgPanel);
        if (ImGui::BeginChild("##viewport_tools", ImVec2(0.0f, 38.0f), ImGuiChildFlags_None,
                             ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
            ImGui::SetCursorPos(ImVec2(8.0f, 4.0f));
            if (ImGui::Button("Orbit")) ui_state.gizmo.enabled = false;
            ImGui::SameLine();
            if (ImGui::Button("Move")) { ui_state.gizmo.enabled = true; ui_state.gizmo.op = GizmoState::Op::Translate; }
            ImGui::SameLine();
            if (ImGui::Button("Rotate")) { ui_state.gizmo.enabled = true; ui_state.gizmo.op = GizmoState::Op::Rotate; }
            ImGui::SameLine();
            if (ImGui::Button(ui_state.gizmo.local ? "Local" : "World")) ui_state.gizmo.local = !ui_state.gizmo.local;
            ImGui::SameLine(0.0f, 12.0f);
            if (ImGui::Button("View")) ImGui::OpenPopup("View menu");
            if (ImGui::BeginPopup("View menu")) {
                if (ImGui::MenuItem("Frame all", "Home")) frame_all = true;
                if (ImGui::MenuItem("Frame selected", "F")) frame_selected = true;
                ImGui::MenuItem("Camera settings", nullptr, &ui_state.show_camera_panel);
                ImGui::EndPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Render")) ImGui::OpenPopup("Render info");
            if (ImGui::BeginPopup("Render info")) {
                ImGui::TextUnformatted("PBR shaded");
                ImGui::TextDisabled("Metallic / roughness, GGX, ACES");
                ImGui::TextWrapped("Additional view modes and post-processing are unavailable.");
                ImGui::EndPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Debug")) ui_state.show_debug_panel = !ui_state.show_debug_panel;
        }
        ImGui::EndChild(); ImGui::PopStyleColor();
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const ImVec2 size = ImGui::GetContentRegionAvail();
        if (framebuffer_width == ~uint32_t(0)) framebuffer_width = static_cast<uint32_t>(std::max(io.DisplaySize.x * io.DisplayFramebufferScale.x, 0.0f));
        if (framebuffer_height == ~uint32_t(0)) framebuffer_height = static_cast<uint32_t>(std::max(io.DisplaySize.y * io.DisplayFramebufferScale.y, 0.0f));
        viewport_ = SceneViewportRect::FromLogical(pos.x, pos.y, size.x, size.y, vp->Pos.x, vp->Pos.y,
                                                  io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y,
                                                  framebuffer_width, framebuffer_height);
        if (viewport_.Valid()) {
            if (frame_all) camera.FrameAll(world, viewport_.Aspect());
            if (frame_selected) camera.FrameSelected(world, ui_state.selected_entity, viewport_.Aspect());
            ImGui::Dummy(size);
            const ImGuiWindow* canvas = ImGui::GetCurrentWindow();
            const bool own_active = GImGui->ActiveId == canvas->MoveId || GImGui->ActiveId == canvas->RootWindowDockTree->MoveId;
            viewport_hovered_ = ui_state.window_focused && viewport_.Contains(io.MousePos.x, io.MousePos.y) &&
                ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) &&
                (!ImGui::IsAnyItemActive() || own_active) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId);
        }
    }
    ImGui::End(); ImGui::PopStyleColor(); ImGui::PopStyleVar();

    if (ui_state.show_file_panel) {
    ImGui::SetNextWindowSize(ImVec2(440.0f, 480.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Open / Save", &ui_state.show_file_panel, ImGuiWindowFlags_NoFocusOnAppearing)) {
        if (!ui_state.load_error.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, kError);
            ImGui::TextWrapped("%s", ui_state.load_error.c_str());
            ImGui::PopStyleColor();
        }
        SectionHeader("Scene");
        if (ui_state.has_scene) {
            ImGui::TextColored(kTextDim, "loaded");
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(kAccent, "%s", ui_state.loaded_path.c_str());
            ImGui::PopTextWrapPos();
            ImGui::Dummy(ImVec2(0.0f, 4.0f));
            if (ImGui::Button("Unload", ImVec2(-1.0f, 0.0f))) ui_state.unload_request = true;
        } else {
            ImGui::TextColored(kTextDim, "no scene loaded");
            ImGui::TextColored(kTextDim, "open a scene to begin");
        }

        ImGui::Dummy(ImVec2(0.0f, 8.0f));
        SectionHeader("Open");
        // Game-engine-style browse: the native OS Open dialog filtered to .nks
        // feeds the SAME load_request seam the runtime Load consumes.
        if (ImGui::Button("Open Scene...", ImVec2(-1.0f, 0.0f))) {
            const std::string picked = OpenFileDialog("Open Scene", "Nuka scenes", "*.nks");
            if (!picked.empty()) {
                std::snprintf(ui_state.load_path, sizeof(ui_state.load_path), "%s",
                              picked.c_str());
                ui_state.load_request = picked;
            }
        }

        ImGui::Dummy(ImVec2(0.0f, 8.0f));
        ImGui::TextColored(kTextDim, "or type a path");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputText("##loadpath", ui_state.load_path, sizeof(ui_state.load_path));
        if (ImGui::Button("Load", ImVec2(-1.0f, 0.0f)) && ui_state.load_path[0] != '\0') {
            ui_state.load_request = ui_state.load_path;
        }

        // -- Save: write the edited scene (full fidelity) back to .nks ----------
        if (ui_state.has_scene) {
            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            SectionHeader("Save");
            if (ui_state.save_dirty) {
                Badge("UNSAVED", ImVec4(kWarn.x, kWarn.y, kWarn.z, 0.22f), kWarn);
                ImGui::SameLine();
            }
            ImGui::TextColored(kTextDim, "edits persist to .nks (+ sibling .nka)");
            ImGui::SetNextItemWidth(-1.0f);
            ImGui::InputText("##savepath", ui_state.save_path, sizeof(ui_state.save_path));
            if (ImGui::Button("Save", ImVec2(-1.0f, 0.0f)) && ui_state.save_path[0] != '\0') {
                ui_state.save_request = ui_state.save_path;
            }
            if (ImGui::Button("Save As...", ImVec2(-1.0f, 0.0f))) {
                const std::string picked =
                    SaveFileDialog("Save Scene", "Nuka scenes", "*.nks", ui_state.save_path);
                if (!picked.empty()) {
                    std::snprintf(ui_state.save_path, sizeof(ui_state.save_path), "%s",
                                  picked.c_str());
                    ui_state.save_request = picked;
                }
            }
        }
    }
    ImGui::End();
    }

    if (ui_state.show_stats_panel) {
    if (ImGui::Begin("Stats", &ui_state.show_stats_panel, ImGuiWindowFlags_NoFocusOnAppearing)) {
        SectionHeader("CPU update");
        ImGui::TextColored(kTextDim, "step + state publish");

        // Hero: big step time in the heading font + a live/idle badge.
        PushHeadingFont();
        char hero[32];
        if (stats.cpu_timing_available) std::snprintf(hero, sizeof(hero), "%.2f ms", stats.step_time_ms);
        else std::snprintf(hero, sizeof(hero), "N/A");
        ImGui::TextColored(stats.step_healthy ? kAccent : kWarn, "%s", hero);
        PopFont();
        ImGui::SameLine();
        if (ui_state.playing)
            Badge("LIVE", ImVec4(kAccent.x, kAccent.y, kAccent.z, 0.22f), kAccent);
        else
            Badge("IDLE", kBgRaised, kTextDim);

        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.0f", stats.fps);
        StatRow("fps", stats.frame_timing_available ? buf : "N/A", kText);
        std::snprintf(buf, sizeof(buf), "%.2f ms", stats.frame_time_ms);
        StatRow("frame", stats.frame_timing_available ? buf : "N/A", kText);
        StatRow("GPU time", "N/A", kTextDim);
        std::snprintf(buf, sizeof(buf), "x%u", stats.sub_steps);
        StatRow("sub-steps", buf, kText);
        std::snprintf(buf, sizeof(buf), "%llu",
                      static_cast<unsigned long long>(stats.frame_index));
        StatRow("frame index", buf, kTextDim);

        ImGui::Dummy(ImVec2(0.0f, 6.0f));
        SectionHeader("Model");
        std::snprintf(buf, sizeof(buf), "%u", stats.dof);
        StatRow("DOF", buf, kAccent);
        std::snprintf(buf, sizeof(buf), "%u", stats.links);
        StatRow("links", buf, kText);
        std::snprintf(buf, sizeof(buf), "%u", stats.bodies);
        StatRow("bodies", buf, kText);
        std::snprintf(buf, sizeof(buf), "%u", stats.contact_cap);
        StatRow("contact cap", buf, kText);
        std::snprintf(buf, sizeof(buf), "%u", ui_state.env_index);
        StatRow("env index", buf, kText);

        ImGui::Dummy(ImVec2(0.0f, 6.0f));
        SectionHeader("Render");
        std::snprintf(buf, sizeof(buf), "%u", stats.draw_calls);
        StatRow("draw calls", stats.draw_calls_available ? buf : "N/A", kText);
        std::snprintf(buf, sizeof(buf), "%u", world.InstanceCount());
        StatRow("instances", buf, kText);
        std::snprintf(buf, sizeof(buf), "%llu",
                      static_cast<unsigned long long>(stats.non_bg_pixels));
        StatRow("lit pixels", stats.pixel_count_available ? buf : "N/A", kText);
        if (!stats.device_name.empty()) {
            ImGui::Dummy(ImVec2(0.0f, 4.0f));
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(kTextDim, "%s", stats.device_name.c_str());
            ImGui::PopTextWrapPos();
        }
    }
    ImGui::End();
    }

    if (ui_state.show_debug_panel) {
    if (ImGui::Begin("Physics Debug", &ui_state.show_debug_panel, ImGuiWindowFlags_NoFocusOnAppearing)) {
        SectionHeader("Physics Debug");
        Badge("READ ONLY", kBgRaised, kAccent);
        ImGui::Spacing();
        ImGui::BeginDisabled(!ui_state.has_scene);
        ImGui::Checkbox("Collider proxies", &ui_state.show_colliders);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("See-through wireframe; translucent fill on devices without wireframe support");
        ImGui::Checkbox("Contact points", &ui_state.show_contacts);
        if (ImGui::Button("Hide all")) {
            ui_state.show_colliders = false;
            ui_state.show_contacts = false;
        }
        ImGui::EndDisabled();
        if (!ui_state.has_scene) ImGui::TextWrapped("Load a scene to inspect its published state.");
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.08f, 0.40f, 0.14f, 1.0f), "Dynamic collider");
        ImGui::TextColored(ImVec4(0.55f, 0.15f, 0.55f, 1.0f), "Static collider");
        ImGui::TextColored(ImVec4(0.65f, 0.25f, 0.04f, 1.0f), "Contact marker");
        ImGui::Spacing();
        ImGui::TextColored(kTextDim, "Previous overlay snapshot");
        char count[48];
        std::snprintf(count, sizeof(count), "%u", stats.debug_colliders);
        StatRow("proxies", stats.debug_colliders_available ? count : "N/A", kText);
        std::snprintf(count, sizeof(count), "%u", stats.debug_contacts);
        StatRow("markers", stats.debug_contacts_available ? count : "N/A", kText);
        if (ui_state.has_scene && stats.debug_skipped_shapes != 0u) {
            ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
            ImGui::TextWrapped("%u unsupported collider shapes skipped", stats.debug_skipped_shapes);
            ImGui::PopStyleColor();
        }
        if (ui_state.has_scene && (stats.debug_invalid_colliders != 0u || stats.debug_invalid_contacts != 0u)) {
            ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
            ImGui::TextWrapped("Invalid data skipped: %u colliders, %u contact points",
                               stats.debug_invalid_colliders, stats.debug_invalid_contacts);
            ImGui::PopStyleColor();
        }
        if (ui_state.has_scene && stats.debug_omitted_instances != 0u) {
            ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
            ImGui::TextWrapped("Overlay limit %u: %llu valid instances omitted", stats.debug_capacity,
                               static_cast<unsigned long long>(stats.debug_omitted_instances));
            ImGui::PopStyleColor();
        }
        if (ui_state.has_scene && stats.debug_contacts_budget_skipped) {
            ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
            ImGui::TextWrapped("Contacts not read: collider proxies used the overlay budget. Omitted count unknown.");
            ImGui::PopStyleColor();
        }
        ImGui::TextWrapped("N/A: disabled, unavailable, or not read. Markers show positions, not forces.");
    }
    ImGui::End();
    }

    if (ui_state.show_hierarchy) {
    if (ImGui::Begin("Hierarchy", &ui_state.show_hierarchy, ImGuiWindowFlags_NoFocusOnAppearing)) {
        if (scene == nullptr) {
            SectionHeader("Scene Tree");
            ImGui::TextWrapped(ui_state.has_scene ? "Scene hierarchy unavailable" : "No scene loaded");
        } else {
            // Add picker: spawn a primitive as a free movable body near the selection
            // (the lightweight analog of a content-browser drag-drop spawn).
            SectionHeader("Hierarchy");
            ImGui::SetNextItemWidth(-40.0f);
            ImGui::InputTextWithHint("##hierarchy_search", "Search scene", hierarchy_filter_, sizeof(hierarchy_filter_));
            ImGui::SameLine();
            if (ImGui::Button("+")) ImGui::OpenPopup("Add object");
            if (ImGui::BeginPopup("Add object")) {
                for (const auto& choice : kSpawnChoices)
                    if (ImGui::MenuItem(choice.label)) RequestSpawn(ui_state, choice.kind);
                ImGui::EndPopup();
            }
            RecordSceneNode(scene->Ecs(), scene->Tree().Root(), ui_state, 0, std::string(), hierarchy_filter_);
        }
    }
    ImGui::End();
    }

    if (ui_state.show_camera_panel) {
    if (ImGui::Begin("Camera", &ui_state.show_camera_panel, ImGuiWindowFlags_NoFocusOnAppearing)) {
        RecordCameraControls(world, camera, ui_state, viewport_.Aspect());
    }
    ImGui::End();
    }

    if (ui_state.show_drive_panel) {
    if (ImGui::Begin("Drive", &ui_state.show_drive_panel, ImGuiWindowFlags_NoFocusOnAppearing)) {
        SectionHeader("Drive Targets");
        const size_t n = ui_state.drive_targets.size();
        if (n == 0u) {
            ImGui::TextColored(kTextDim, "no articulation DOFs");
        } else {
            ImGui::TextColored(kTextDim, "%zu DOF  (env %u)", n, ui_state.env_index);
            ImGui::Dummy(ImVec2(0.0f, 4.0f));
            if (ui_state.drive_dirty.size() != n) ui_state.drive_dirty.assign(n, 0u);
            const bool have_labels = (ui_state.dof_labels.size() == n);
            for (size_t d = 0; d < n; ++d) {
                char id[32];  // "##drive" + up to a 20-digit size_t + null.
                std::snprintf(id, sizeof(id), "##drive%zu", d);
                char label[64];
                if (have_labels && !ui_state.dof_labels[d].empty()) {
                    std::snprintf(label, sizeof(label), "%s", ui_state.dof_labels[d].c_str());
                } else {
                    std::snprintf(label, sizeof(label), "dof %zu", d);
                }
                ImGui::TextColored(kTextDim, "%s", label);
                ImGui::SetNextItemWidth(-1.0f);
                if (ImGui::SliderFloat(id, &ui_state.drive_targets[d], -3.1416f,
                                       3.1416f, "%.3f")) {
                    ui_state.drive_dirty[d] = 1u;  // mark for upload by the viewer
                }
            }
        }

        // -- Teleop: keyboard -> the SAME per-DOF drive seam + a policy command. The
        // viewer reads the keys + applies it BETWEEN frames (never here); this section
        // only edits the plain ui_state so it records deterministically. ------------
        ImGui::Dummy(ImVec2(0.0f, 10.0f));
        SectionHeader("Teleop (keyboard)");
        ImGui::Checkbox("enable##teleop", &ui_state.teleop_enabled);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Keyboard drives the robot; ignored while a field/console has focus");
        if (ui_state.teleop_enabled) {
            const int dof_n = static_cast<int>(ui_state.drive_targets.size());
            ImGui::TextColored(kTextDim, "per-DOF:  [ / ] nudge     - / = select DOF");
            if (dof_n > 0) {
                ImGui::SetNextItemWidth(120.0f);
                ImGui::DragInt("##teleopdof", &ui_state.teleop_dof, 0.1f, 0, dof_n - 1, "dof %d");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120.0f);
                ImGui::DragFloat("##teleopstep", &ui_state.teleop_step, 0.005f, 0.0f, 0.0f,
                                 "step %.3f");
                ImGui::Checkbox("PD-hold##teleop", &ui_state.teleop_hold);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Apply uniform PD gains so drive targets move the joints");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(70.0f);
                ImGui::DragFloat("##teleopkp", &ui_state.teleop_kp, 0.5f, 0.0f, 0.0f, "kp %.1f");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(70.0f);
                ImGui::DragFloat("##teleopkd", &ui_state.teleop_kd, 0.05f, 0.0f, 0.0f, "kd %.2f");
            } else {
                ImGui::TextColored(kTextDim, "no articulation DOFs");
            }
            ImGui::Dummy(ImVec2(0.0f, 4.0f));
            ImGui::TextColored(kTextDim, "locomotion:  WASD move   Q / E yaw  (needs a policy)");
            ImGui::TextColored(kAccent, "cmd  vx %.2f  vy %.2f  wyaw %.2f", ui_state.teleop_cmd[0],
                               ui_state.teleop_cmd[1], ui_state.teleop_cmd[2]);
        }
    }
    ImGui::End();
    }

    if (ui_state.show_inspector) {
    if (ImGui::Begin("Inspector", &ui_state.show_inspector, ImGuiWindowFlags_NoFocusOnAppearing)) {
        SectionHeader("Inspector");
        const bool has_sel = ui_state.selected_entity != nuka::scene::kInvalidEntity;
        const render::RenderInstance* inst = nullptr;
        if (has_sel) {
            for (const render::RenderInstance& ri : world.instances) {
                if (ri.entity == ui_state.selected_entity) { inst = &ri; break; }
            }
        }
        if (!has_sel) {
            PushHeadingFont();
            const std::filesystem::path source(ui_state.loaded_path);
            const std::string title = source.parent_path().filename() == "nuka_lab" && source.filename() == "gripper.nks"
                ? "Nuka Dynamics Lab" : (ui_state.has_scene ? source.stem().string() : "Scene overview");
            ImGui::TextWrapped("%s", title.c_str());
            PopFont();
            ImGui::TextWrapped("Select a hierarchy node, or Ctrl-click a movable body in the scene.");
            ImGui::Spacing();
            RecordCameraControls(world, camera, ui_state, viewport_.Aspect(), true);
            SectionHeader("Rendering");
            StatRow("Shading", "PBR / GGX", kText);
            StatRow("Tone map", "ACES", kTextDim);
            ImGui::TextWrapped("Select an object to edit its authored material. Interactive shadows and post-processing are unavailable.");
        } else {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "%u", ui_state.selected_entity.index);
            StatRow("entity", buf, kAccent);
            if (scene != nullptr) {
                if (const auto n = scene->Ecs().NodeOf(ui_state.selected_entity)) {
                    StatRow("node", n->name.empty() ? "(unnamed)" : n->name.c_str(), kText);
                    const std::string path = scene->Tree().PathOf(n);
                    if (!path.empty()) StatRow("path", path.c_str(), kTextDim);
                }
            }
            if (inst != nullptr) {
                const bool real_mesh = inst->mesh_id != render::kNoId &&
                                       inst->mesh_id < world.meshes.Count();
                StatRow("mesh",
                        real_mesh ? MeshSourceLabel(world.meshes.Source(inst->mesh_id)) : "none",
                        kTextDim);
            }
            ImGui::SameLine();
            if (ui_state.inspector.movable)
                Badge("MOVABLE", ImVec4(kAccent.x, kAccent.y, kAccent.z, 0.22f), kAccent);
            else
                Badge("FIXED", kBgRaised, kTextDim);

            InspectorState& ins = ui_state.inspector;
            if (!ins.valid) {
                ImGui::Dummy(ImVec2(0.0f, 6.0f));
                ImGui::TextColored(kTextDim, "no editable geometry");
            } else {
                // -- Transform ---------------------------------------------------
                ImGui::Dummy(ImVec2(0.0f, 6.0f));
                SectionHeader("Transform");
                bool t_changed = false, t_commit = false;
                ImGui::TextColored(kTextDim, "position");
                ImGui::SetNextItemWidth(-1.0f);
                t_changed |= ImGui::DragFloat3("##pos", ins.pos, 0.01f, 0.0f, 0.0f, "%.3f");
                t_commit  |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::TextColored(kTextDim, "rotation (deg)");
                ImGui::SetNextItemWidth(-1.0f);
                t_changed |= ImGui::DragFloat3("##rot", ins.rot_deg, 0.5f, 0.0f, 0.0f, "%.1f");
                t_commit  |= ImGui::IsItemDeactivatedAfterEdit();
                ins.transform_changed = t_changed;
                ins.transform_commit  = t_commit;

                // -- Gizmo mode (the in-viewport handles share this edit seam) ----
                ImGui::Dummy(ImVec2(0.0f, 4.0f));
                GizmoState& gz = ui_state.gizmo;
                ImGui::Checkbox("gizmo", &gz.enabled);
                ImGui::BeginDisabled(!gz.enabled);
                if (ImGui::RadioButton("move", gz.op == GizmoState::Op::Translate))
                    gz.op = GizmoState::Op::Translate;
                ImGui::SameLine();
                if (ImGui::RadioButton("rotate", gz.op == GizmoState::Op::Rotate))
                    gz.op = GizmoState::Op::Rotate;
                bool local = gz.local;
                if (ImGui::Checkbox("local", &local)) gz.local = local;
                ImGui::SameLine();
                ImGui::Checkbox("snap", &gz.snap_on);
                ImGui::SetNextItemWidth(-1.0f);
                ImGui::BeginDisabled(!gz.snap_on);
                // ONE step field: world units (Translate) or degrees (Rotate).
                if (gz.op == GizmoState::Op::Rotate)
                    ImGui::DragFloat("##snapstep", &gz.snap_rotate, 0.5f, 0.0f, 0.0f, "%.1f deg");
                else
                    ImGui::DragFloat("##snapstep", &gz.snap_translate, 0.01f, 0.0f, 0.0f, "%.3f m");
                ImGui::EndDisabled();
                ImGui::EndDisabled();

                // -- Material ----------------------------------------------------
                if (ins.has_material) {
                    ImGui::Dummy(ImVec2(0.0f, 8.0f));
                    SectionHeader("Material");
                    bool m_changed = false, m_commit = false;
                    ImGui::TextColored(kTextDim, "base color");
                    ImGui::SetNextItemWidth(-1.0f);
                    m_changed |= ImGui::ColorEdit4("##basecol", ins.base_color,
                                                   ImGuiColorEditFlags_AlphaBar);
                    m_commit  |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::TextColored(kTextDim, "roughness");
                    ImGui::SetNextItemWidth(-1.0f);
                    m_changed |= ImGui::SliderFloat("##rough", &ins.roughness, 0.0f, 1.0f, "%.3f");
                    m_commit  |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::TextColored(kTextDim, "metallic");
                    ImGui::SetNextItemWidth(-1.0f);
                    m_changed |= ImGui::SliderFloat("##metal", &ins.metallic, 0.0f, 1.0f, "%.3f");
                    m_commit  |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::TextColored(kTextDim, "emissive");
                    ImGui::SetNextItemWidth(-1.0f);
                    m_changed |= ImGui::ColorEdit3("##emis", ins.emissive,
                                                   ImGuiColorEditFlags_HDR |
                                                   ImGuiColorEditFlags_Float);
                    m_commit  |= ImGui::IsItemDeactivatedAfterEdit();

                    // Beauty-only fields: badged RT (the raster preview ignores them).
                    ImGui::Dummy(ImVec2(0.0f, 4.0f));
                    Badge("RT", ImVec4(kAccentDim.x, kAccentDim.y, kAccentDim.z, 0.30f), kAccent);
                    ImGui::SameLine();
                    ImGui::TextColored(kTextDim, "offline beauty only");
                    ImGui::TextColored(kTextDim, "transmission");
                    ImGui::SetNextItemWidth(-1.0f);
                    m_changed |= ImGui::SliderFloat("##transm", &ins.transmission, 0.0f, 1.0f, "%.3f");
                    m_commit  |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::TextColored(kTextDim, "ior");
                    ImGui::SetNextItemWidth(-1.0f);
                    m_changed |= ImGui::SliderFloat("##ior", &ins.ior, 1.0f, 2.5f, "%.3f");
                    m_commit  |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::TextColored(kTextDim, "sheen");
                    ImGui::SetNextItemWidth(-1.0f);
                    m_changed |= ImGui::SliderFloat("##sheen", &ins.sheen, 0.0f, 1.0f, "%.3f");
                    m_commit  |= ImGui::IsItemDeactivatedAfterEdit();
                    ins.material_changed = m_changed;
                    ins.material_commit  = m_commit;
                }
            }
        }
    }
    ImGui::End();
    }

    if (ui_state.show_script_panel) {
    if (ImGui::Begin("Script", &ui_state.show_script_panel, ImGuiWindowFlags_NoFocusOnAppearing)) {
        SectionHeader("Live Script");
        ImGui::TextColored(kTextDim, ui_state.has_scene
                                         ? "runs against the loaded world (nuka.*)"
                                         : "load a scene, then nuka.* drives it");

        const float avail_y = ImGui::GetContentRegionAvail().y;
        float editor_h = avail_y * 0.5f;
        if (editor_h < 80.0f) editor_h = 80.0f;
        ImGui::InputTextMultiline("##scriptsrc", ui_state.script_buf,
                                  sizeof(ui_state.script_buf), ImVec2(-1.0f, editor_h),
                                  ImGuiInputTextFlags_AllowTabInput);

        bool run_now = false;
        ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.04f, 0.06f, 0.07f, 1.0f));
        if (ImGui::Button("Run")) run_now = true;
        ImGui::PopStyleColor(2);
        ImGui::SameLine();
        if (ImGui::Button("Clear Console")) ui_state.script_clear_request = true;
        ImGui::SameLine();
        ImGui::TextColored(kTextDim, "Ctrl+Enter");

        // Ctrl+Enter runs only when this window holds focus (so typing elsewhere
        // never fires it, and a no-input frame records deterministically).
        const bool ctrl_enter =
            ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
            (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl)) &&
            ImGui::IsKeyPressed(ImGuiKey_Enter, false);
        if (run_now || ctrl_enter) ui_state.script_run_request = true;

        // Persist the buffer as a first-class /script node (Save writes it inline) or
        // load the selected node's source back -- a scene's scripts live in the scene.
        if (ImGui::Button("Attach as Node")) ui_state.script_attach_request = true;
        ImGui::SameLine();
        if (ImGui::Button("Load Node")) ui_state.script_load_request = true;
        ImGui::SameLine();
        ImGui::TextColored(kTextDim, "%d /script node(s)", ui_state.script_node_count);

        ImGui::Dummy(ImVec2(0.0f, 6.0f));
        SectionHeader("Console");
        ImGui::BeginChild("##console", ImVec2(-1.0f, -1.0f), true,
                          ImGuiWindowFlags_HorizontalScrollbar);
        if (ui_state.console_log.empty()) {
            ImGui::TextColored(kTextDim, "(no output yet)");
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, kText);
            ImGui::TextUnformatted(ui_state.console_log.c_str(),
                                   ui_state.console_log.c_str() +
                                       ui_state.console_log.size());
            ImGui::PopStyleColor();
        }
        // Keep the newest output in view when already pinned to the bottom.
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) ImGui::SetScrollHereY(1.0f);
        ImGui::EndChild();
    }
    ImGui::End();
    }

}

void ImGuiLayer::HandleCameraShortcuts(const render::RenderWorld& world, CameraController& camera,
                                      const ViewerUiState& ui_state) {
    ApplyCameraShortcuts(world, camera, ui_state.selected_entity, viewport_hovered_, ui_state.gizmo.active, viewport_.Aspect());
}

// ---------------------------------------------------------------------------
// DrawGizmo -- the in-viewport transform manipulator. Builds the SAME camera
// basis the renderer uses (right-handed LookAt, world +Z up; GL-NDC projection so
// ImGuizmo's own screen flip lands the handles ON the rendered object), runs the
// gizmo over the selected entity's WORLD pose, and on a drag rewrites `world` +
// latches `changed` so the viewer routes the result through the general edit seam.
// gizmo.active latches when the handles are hovered/used (orbit + picking stand
// down). A no-op when disabled / unselected. Must run inside the ImGui frame,
// after RecordUi (which armed ImGuizmo for this frame).
// ---------------------------------------------------------------------------
void ImGuiLayer::DrawGizmo(CameraController& camera, uint32_t vp_w, uint32_t vp_h,
                           ViewerUiState& ui_state, math::Transform& world, bool& changed) {
    changed = false;
    GizmoState& gz = ui_state.gizmo;
    if (!gz.enabled || ui_state.selected_entity == nuka::scene::kInvalidEntity ||
        vp_w == 0u || vp_h == 0u || !viewport_.Valid() || !ui_state.window_focused) {
        gz.active = false;
        gz.using_now = false;
        return;
    }

    float view[16], proj[16], model[16];
    ViewMatrix(camera.ResolvedEye(), camera.ResolvedTarget(),
               math::Vec3{0.0f, 0.0f, 1.0f}, view);
    const float aspect = viewport_.Aspect();
    ProjMatrixGL(camera.fov_degrees * 3.14159265358979323846f / 180.0f, aspect,
                 0.05f, 500.0f, proj);
    ModelMatrix(world, model);

    ImGuizmo::Enable(viewport_hovered_ && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId));
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    draw->PushClipRect(ImVec2(viewport_.x, viewport_.y), ImVec2(viewport_.x + viewport_.width, viewport_.y + viewport_.height), true);
    ImGuizmo::SetDrawlist(draw);
    ImGuizmo::SetAlternativeWindow(ImGui::FindWindowByName("Scene View"));
    ImGuizmo::SetRect(viewport_.x, viewport_.y, viewport_.width, viewport_.height);
    const ImGuizmo::OPERATION op =
        (gz.op == GizmoState::Op::Rotate) ? ImGuizmo::ROTATE : ImGuizmo::TRANSLATE;
    const ImGuizmo::MODE mode = gz.local ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    // Snap step: world units for Translate, degrees for Rotate (ImGuizmo reads snap[0]).
    const float snap_step = (gz.op == GizmoState::Op::Rotate) ? gz.snap_rotate
                                                              : gz.snap_translate;
    const float snap_vec[3] = {snap_step, snap_step, snap_step};
    ImGuizmo::Manipulate(view, proj, op, mode, model, /*deltaMatrix=*/nullptr,
                         gz.snap_on ? snap_vec : nullptr);
    draw->PopClipRect();

    gz.active = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
    gz.using_now = ImGuizmo::IsUsing();
    if (ImGuizmo::IsUsing()) {
        world = TransformFromModel(model);
        changed = true;
    }
}

}  // namespace nuka::runtime::app::viewer
