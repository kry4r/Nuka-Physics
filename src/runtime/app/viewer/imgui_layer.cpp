// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer::ImGuiLayer -- the custom-beautified viewer UI (T3).
//
// Records the docked panel UI each frame. Consumes the vendored Dear ImGui as a
// SYSTEM include (so its warnings stay out of -Werror and it is outside the
// zero-CUDA lint scope). This TU pulls NO Vulkan headers -- the draw-data
// submission lives in the present/offscreen overlay callback.
//
// BEYOND-STOCK polish (the OWNER requirement) is concentrated here:
//   - a docked layout built once via the docking branch (central node passthru so
//     the rendered viewport shows through; panels docked left/right),
//   - section headers + hero stats in the heading font (atlas index 1),
//   - the teal accent used purposefully (active Play, LIVE badge, speed pills),
//   - custom-drawn badges / accent rules rather than raw text dumps.
//
// HOST-ONLY / zero-CUDA-token.
// ---------------------------------------------------------------------------

#include "runtime/app/viewer/imgui_layer.hpp"

#include "imgui.h"
#include "imgui_internal.h"  // DockBuilder* for the one-time default layout
#include "ImGuizmo.h"        // in-viewport transform gizmo (vendored, MIT)

#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/file_dialog.hpp"
#include "scene/ecs/components.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_ir.hpp"

#include <cmath>
#include <cstdio>
#include <iterator>  // std::size for array-derived loop bounds
#include <memory>

namespace nuka::runtime::app::viewer {

namespace {

// The Nuka palette tokens (kept in sync with nuka_imgui.cpp's ApplyNukaTheme; T1
// documents these as the design tokens T3 reuses rather than inventing colors).
const ImVec4 kAccent   = ImVec4(0.176f, 0.831f, 0.749f, 1.00f);  // #2DD4BF teal
const ImVec4 kAccentDim= ImVec4(0.118f, 0.612f, 0.549f, 1.00f);  // #1E9C8C
const ImVec4 kText     = ImVec4(0.776f, 0.816f, 0.855f, 1.00f);  // #C6D0DA
const ImVec4 kTextDim  = ImVec4(0.431f, 0.478f, 0.533f, 1.00f);  // #6E7A88
const ImVec4 kBgRaised = ImVec4(0.110f, 0.137f, 0.173f, 1.00f);  // #1C232C
const ImVec4 kWarn     = ImVec4(0.93f, 0.55f, 0.30f, 1.00f);     // amber (unhealthy)

ImU32 Col(const ImVec4& c) { return ImGui::ColorConvertFloat4ToU32(c); }

// Push the heading font (atlas index 1, JetBrains Mono Bold @21 loaded by
// LoadNukaFonts). Falls back gracefully to the body font if only one was loaded.
void PushHeadingFont() {
    ImGuiIO& io = ImGui::GetIO();
    if (io.Fonts->Fonts.Size > 1) {
        ImGui::PushFont(io.Fonts->Fonts[1]);
    } else {
        ImGui::PushFont(io.Fonts->Fonts.empty() ? nullptr : io.Fonts->Fonts[0]);
    }
}
void PopFont() { ImGui::PopFont(); }

// A section header: heading font + an accent rule beneath it. The signature
// "engineering pro-tool" divider (not a stock CollapsingHeader).
void SectionHeader(const char* label) {
    PushHeadingFont();
    ImGui::TextColored(kText, "%s", label);
    PopFont();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p0 = ImGui::GetItemRectMin();
    const ImVec2 p1 = ImGui::GetItemRectMax();
    const float y = p1.y + 3.0f;
    // Right edge of the content area in screen space (avail width from the header's
    // left edge). GetContentRegionAvail() is the non-obsolete way to size the rule.
    const float right = p0.x + ImGui::GetContentRegionAvail().x;
    // A short bright accent stub + a faint full rule -> a tasteful underline.
    dl->AddLine(ImVec2(p0.x, y), ImVec2(p0.x + 26.0f, y), Col(kAccent), 2.0f);
    dl->AddLine(ImVec2(p0.x + 30.0f, y), ImVec2(right, y),
                Col(ImVec4(kAccent.x, kAccent.y, kAccent.z, 0.18f)), 1.0f);
    ImGui::Dummy(ImVec2(0.0f, 8.0f));
}

// A labeled stat row: dim label left, value in the second column. The label
// column width follows the FONT (not a fixed pixel literal) so it stays aligned
// at non-default font sizes / HiDPI. ~11 glyphs wide covers the longest label.
constexpr float kStatLabelGlyphs = 11.0f;
void StatRow(const char* label, const char* value, const ImVec4& value_col) {
    ImGui::TextColored(kTextDim, "%s", label);
    ImGui::SameLine(ImGui::CalcTextSize("M").x * kStatLabelGlyphs);
    ImGui::TextColored(value_col, "%s", value);
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

// Recursively emit one ImGui TreeNode per SceneGraph node (name + kind badge);
// robots collapse as path-prefixed subtrees. Selection is keyed on node->entity.
void RecordSceneNode(const nuka::scene::Registry& reg,
                     const std::shared_ptr<nuka::scene::SceneNode>& node,
                     ViewerUiState& ui, int depth, const std::string& path) {
    if (!node) return;
    const NodeKind kind = ClassifyNode(reg, node->entity);
    const bool has_children = static_cast<bool>(node->first_child);
    const bool valid_entity = node->entity != nuka::scene::kInvalidEntity;
    const bool selected = valid_entity && node->entity == ui.selected_entity;

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                               ImGuiTreeNodeFlags_OpenOnDoubleClick |
                               ImGuiTreeNodeFlags_SpanAvailWidth;
    if (!has_children) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    if (selected)      flags |= ImGuiTreeNodeFlags_Selected;

    // Open only the root once so the top level (terrain + robot groups) shows;
    // deeper internals stay collapsed until the user expands them.
    ImGui::SetNextItemOpen(depth == 0, ImGuiCond_Once);
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
        if (ImGui::MenuItem("Add box child")) ui.add_box_request = true;
        if (ImGui::MenuItem("Delete subtree")) ui.delete_request = true;
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    ImGui::TextColored(kind.color, "[%s]", kind.label);

    if (open && has_children) {
        for (auto child = node->first_child; child; child = child->next_sibling) {
            const std::string child_path =
                path.empty() ? child->name : path + "/" + child->name;
            RecordSceneNode(reg, child, ui, depth + 1, child_path);
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

// ---------------------------------------------------------------------------
// The one-time default dock layout: a left column (Stats over Scene Tree) and a
// right column (Camera), central node passthru so the rendered viewport shows.
// ---------------------------------------------------------------------------
static void BuildDefaultDockLayout(ImGuiID dockspace_id) {
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace |
                                                ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

    ImGuiID central = dockspace_id;
    ImGuiID left  = ImGui::DockBuilderSplitNode(central, ImGuiDir_Left, 0.24f, nullptr, &central);
    ImGuiID right = ImGui::DockBuilderSplitNode(central, ImGuiDir_Right, 0.26f, nullptr, &central);
    // The left column is three stacked nodes: Load (top), Stats (middle), Scene
    // (bottom) -- the Load panel leads so an empty editor still offers it first.
    ImGuiID left_mid    = ImGui::DockBuilderSplitNode(left, ImGuiDir_Down, 0.68f, nullptr, &left);
    ImGuiID left_bottom = ImGui::DockBuilderSplitNode(left_mid, ImGuiDir_Down, 0.5f, nullptr, &left_mid);

    ImGuiID right_bottom =
        ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.55f, nullptr, &right);

    ImGui::DockBuilderDockWindow("Load", left);
    ImGui::DockBuilderDockWindow("Stats", left_mid);
    ImGui::DockBuilderDockWindow("Scene", left_bottom);
    // The live-world Script console shares the lower-left node (tabbed with Scene).
    ImGui::DockBuilderDockWindow("Script", left_bottom);
    ImGui::DockBuilderDockWindow("Camera", right);
    // The Drive editor + the Entity inspector share the lower-right node.
    ImGui::DockBuilderDockWindow("Drive", right_bottom);
    ImGui::DockBuilderDockWindow("Entity", right_bottom);
    ImGui::DockBuilderFinish(dockspace_id);
}

void ImGuiLayer::EnableDocking() {
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // The dock layout is built programmatically each session (BuildDefaultDockLayout)
    // -> do NOT persist/read imgui.ini (it would otherwise be written to the cwd and
    // its stale dock state would fight the programmatic layout). Disabling it also
    // keeps the GATE-B composite + windowed run side-effect-free.
    io.IniFilename = nullptr;
}

void ImGuiLayer::RecordUi(const render::RenderWorld& world, const ViewerStats& stats,
                          CameraController& camera, ViewerUiState& ui_state,
                          const nuka::scene::SceneIR* scene) {
    // -- the host dockspace (passthru central node -> viewport shows through) ---
    const ImGuiID dockspace_id =
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                     ImGuiDockNodeFlags_PassthruCentralNode);
    if (!dock_built_) {
        BuildDefaultDockLayout(dockspace_id);
        dock_built_ = true;
    }

    // Arm ImGuizmo for this frame (perspective camera). DrawGizmo runs after the
    // panels so its handles overlay the central viewport.
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::BeginFrame();

    // ======================================================================
    // TRANSPORT TOOLBAR -- a slim accent-styled bar pinned to the top.
    // ======================================================================
    {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + 12.0f, vp->WorkPos.y + 8.0f));
        ImGui::SetNextWindowBgAlpha(0.92f);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking |
                                 ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNav |
                                 ImGuiWindowFlags_NoMove;
        if (ImGui::Begin("##transport", nullptr, flags)) {
            PushHeadingFont();
            ImGui::TextColored(kAccent, "NUKA EDITOR");
            PopFont();
            ImGui::SameLine();
            ImGui::TextColored(kTextDim, ui_state.has_scene ? "| live" : "| no scene");
            ImGui::SameLine(0.0f, 18.0f);

            // Transport drives the live world; disabled until a scene is loaded.
            if (!ui_state.has_scene) ImGui::BeginDisabled();

            // Play / Pause -- accent-FILLED when active (the showcase accent use).
            if (ui_state.playing) {
                ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kAccent);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, kAccentDim);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.04f, 0.06f, 0.07f, 1.0f));
            }
            if (ImGui::Button(ui_state.playing ? "Pause" : "Play")) {
                ui_state.playing = !ui_state.playing;
                ui_state.play_toggled = true;
            }
            if (ui_state.playing) ImGui::PopStyleColor(4);

            ImGui::SameLine();
            if (ImGui::Button("Step")) ui_state.step_requested = true;
            ImGui::SameLine();
            if (ImGui::Button("Reset")) ui_state.reset_requested = true;

            // Undo / Redo -- one-shot requests the viewer applies between frames;
            // each disabled when its history is empty (viewer-set can_* hints).
            ImGui::SameLine(0.0f, 12.0f);
            ImGui::BeginDisabled(!ui_state.can_undo);
            if (ImGui::Button("Undo")) ui_state.undo_request = true;
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(!ui_state.can_redo);
            if (ImGui::Button("Redo")) ui_state.redo_request = true;
            ImGui::EndDisabled();

            if (!ui_state.has_scene) ImGui::EndDisabled();

            ImGui::SameLine(0.0f, 18.0f);
            ImGui::TextColored(kTextDim, "speed");
            ImGui::SameLine();
            const float kSpeeds[] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
            const char* kSpeedLbl[] = {"x.25", "x.5", "x1", "x2", "x4"};
            for (int i = 0; i < static_cast<int>(std::size(kSpeeds)); ++i) {
                const bool sel = (ui_state.speed == kSpeeds[i]);
                if (sel) {
                    ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.04f, 0.06f, 0.07f, 1.0f));
                }
                if (i > 0) ImGui::SameLine();
                if (ImGui::Button(kSpeedLbl[i])) ui_state.speed = kSpeeds[i];
                if (sel) ImGui::PopStyleColor(2);
            }
        }
        ImGui::End();
    }

    // ======================================================================
    // LOAD PANEL -- "Open Scene" runs the native OS file dialog (filtered to .nks);
    // a manual path field is the fallback. Both set load_request; Unload drops empty.
    // ======================================================================
    if (ImGui::Begin("Load")) {
        SectionHeader("Scene");
        if (ui_state.has_scene) {
            ImGui::TextColored(kTextDim, "loaded");
            ImGui::TextColored(kAccent, "%s", ui_state.loaded_path.c_str());
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

    // ======================================================================
    // STATS PANEL -- hero step-time + tidy labeled rows + a LIVE/IDLE badge.
    // ======================================================================
    if (ImGui::Begin("Stats")) {
        SectionHeader("Performance");

        // Hero: big step time in the heading font + a live/idle badge.
        PushHeadingFont();
        char hero[32];
        std::snprintf(hero, sizeof(hero), "%.2f ms", stats.step_time_ms);
        ImGui::TextColored(stats.step_healthy ? kAccent : kWarn, "%s", hero);
        PopFont();
        ImGui::SameLine();
        if (ui_state.playing)
            Badge("LIVE", ImVec4(kAccent.x, kAccent.y, kAccent.z, 0.22f), kAccent);
        else
            Badge("IDLE", kBgRaised, kTextDim);

        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.0f", stats.fps);
        StatRow("fps", buf, kText);
        std::snprintf(buf, sizeof(buf), "x%u", stats.sub_steps);
        StatRow("sub-steps", buf, kText);
        std::snprintf(buf, sizeof(buf), "%llu",
                      static_cast<unsigned long long>(stats.frame_index));
        StatRow("frame", buf, kTextDim);

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
        StatRow("draw calls", buf, kText);
        std::snprintf(buf, sizeof(buf), "%llu",
                      static_cast<unsigned long long>(stats.non_bg_pixels));
        StatRow("lit pixels", buf, kText);
        if (!stats.device_name.empty()) {
            ImGui::Dummy(ImVec2(0.0f, 4.0f));
            ImGui::TextColored(kTextDim, "%s", stats.device_name.c_str());
        }
    }
    ImGui::End();

    // ======================================================================
    // SCENE TREE -- the SceneGraph hierarchy (robots as collapsible subtrees,
    // terrain/media at root) with kind badges; selection keyed on node->entity.
    // ======================================================================
    if (ImGui::Begin("Scene")) {
        if (scene == nullptr) {
            SectionHeader("Scene Tree");
            ImGui::TextColored(kTextDim, "no scene loaded");
        } else {
            // Read-only debug-draw toggles (Isaac-style Show-by-Type): collider
            // proxies + live contact points. The viewer rebuilds the overlay between
            // frames from these flags; nothing here mutates the scene.
            SectionHeader("Show");
            ImGui::Checkbox("colliders", &ui_state.show_colliders);
            ImGui::SameLine();
            ImGui::Checkbox("contacts", &ui_state.show_contacts);
            ImGui::Dummy(ImVec2(0.0f, 8.0f));
            SectionHeader("Scene Tree");
            ImGui::TextColored(kTextDim, "%u instances  %u meshes  %u materials",
                               world.InstanceCount(), world.meshes.Count(),
                               world.MaterialCount());
            ImGui::Dummy(ImVec2(0.0f, 4.0f));
            RecordSceneNode(scene->Ecs(), scene->Tree().Root(), ui_state, 0, std::string());
        }
    }
    ImGui::End();

    // ======================================================================
    // CAMERA PANEL -- orbit/pan/zoom readout + reset + fov slider (wires camera).
    // ======================================================================
    if (ImGui::Begin("Camera")) {
        SectionHeader("Camera");
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

        ImGui::Dummy(ImVec2(0.0f, 6.0f));
        ImGui::TextColored(kTextDim, "field of view");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::SliderFloat("##fov", &camera.fov_degrees, 20.0f, 90.0f, "%.0f deg");

        ImGui::Dummy(ImVec2(0.0f, 6.0f));
        if (ImGui::Button("Reset View", ImVec2(-1.0f, 0.0f))) ui_state.camera_reset = true;

        ImGui::Dummy(ImVec2(0.0f, 10.0f));
        SectionHeader("Controls");
        ImGui::TextColored(kTextDim, "LMB drag  orbit");
        ImGui::TextColored(kTextDim, "MMB / Shift  pan");
        ImGui::TextColored(kTextDim, "wheel  zoom");
        ImGui::TextColored(kTextDim, "Ctrl+LMB  pick / drag entity");
    }
    ImGui::End();

    // ======================================================================
    // DRIVE PANEL (VIEW-2) -- a GENERIC per-DOF drive-target editor. One slider
    // per DOF; moving a slider latches drive_dirty[d] so the viewer uploads ONLY
    // that DOF into FieldId::DriveTarget (env-major). NEVER a hardcoded grasp /
    // choreography table (OD-7 / highest directive). With a FIXED ui_state this
    // records deterministically (GATE-B / R12).
    // ======================================================================
    if (ImGui::Begin("Drive")) {
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
    }
    ImGui::End();

    // ======================================================================
    // ENTITY PANEL -- the editable inspector for ui_state.selected_entity. Reads
    // identity (name / path / pose-src) and EDITS Transform + Material; the widget
    // values + dirty/commit latches are applied by the viewer through the general
    // edit seam (the SAME pattern as the Drive panel above).
    // ======================================================================
    if (ImGui::Begin("Entity")) {
        SectionHeader("Selection");
        const bool has_sel = ui_state.selected_entity != nuka::scene::kInvalidEntity;
        const render::RenderInstance* inst = nullptr;
        if (has_sel) {
            for (const render::RenderInstance& ri : world.instances) {
                if (ri.entity == ui_state.selected_entity) { inst = &ri; break; }
            }
        }
        if (!has_sel) {
            ImGui::TextColored(kTextDim, "nothing selected");
            ImGui::Dummy(ImVec2(0.0f, 4.0f));
            ImGui::TextColored(kTextDim, "click a node or Ctrl+LMB a body");
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
                ImGui::SameLine();
                ImGui::BeginDisabled(!gz.enabled);
                if (ImGui::RadioButton("move", gz.op == GizmoState::Op::Translate))
                    gz.op = GizmoState::Op::Translate;
                ImGui::SameLine();
                if (ImGui::RadioButton("rotate", gz.op == GizmoState::Op::Rotate))
                    gz.op = GizmoState::Op::Rotate;
                ImGui::SameLine();
                bool local = gz.local;
                if (ImGui::Checkbox("local", &local)) gz.local = local;
                ImGui::SameLine();
                ImGui::Checkbox("snap", &gz.snap_on);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(90.0f);
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

    // ======================================================================
    // SCRIPT PANEL -- the live-world console: a multiline editor + Run (the viewer
    // execs it) and a read-only Console of the captured output. It touches ONLY plain
    // ui_state (no interpreter / time input), so a fixed state records deterministic.
    // ======================================================================
    if (ImGui::Begin("Script")) {
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
        vp_w == 0u || vp_h == 0u) {
        gz.active = false;
        gz.using_now = false;
        return;
    }

    float view[16], proj[16], model[16];
    ViewMatrix(camera.ResolvedEye(), camera.ResolvedTarget(),
               math::Vec3{0.0f, 0.0f, 1.0f}, view);
    const float aspect = static_cast<float>(vp_w) / static_cast<float>(vp_h);
    ProjMatrixGL(camera.fov_degrees * 3.14159265358979323846f / 180.0f, aspect,
                 0.05f, 500.0f, proj);
    ModelMatrix(world, model);

    ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
    ImGuizmo::SetRect(0.0f, 0.0f, static_cast<float>(vp_w), static_cast<float>(vp_h));
    const ImGuizmo::OPERATION op =
        (gz.op == GizmoState::Op::Rotate) ? ImGuizmo::ROTATE : ImGuizmo::TRANSLATE;
    const ImGuizmo::MODE mode = gz.local ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    // Snap step: world units for Translate, degrees for Rotate (ImGuizmo reads snap[0]).
    const float snap_step = (gz.op == GizmoState::Op::Rotate) ? gz.snap_rotate
                                                              : gz.snap_translate;
    const float snap_vec[3] = {snap_step, snap_step, snap_step};
    ImGuizmo::Manipulate(view, proj, op, mode, model, /*deltaMatrix=*/nullptr,
                         gz.snap_on ? snap_vec : nullptr);

    gz.active = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
    gz.using_now = ImGuizmo::IsUsing();
    if (ImGuizmo::IsUsing()) {
        world = TransformFromModel(model);
        changed = true;
    }
}

}  // namespace nuka::runtime::app::viewer
