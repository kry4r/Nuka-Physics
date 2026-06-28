// ---------------------------------------------------------------------------
// nuka_editor -- the windowed scene editor entry. Opens EMPTY (no world cooked),
// renders the Dear ImGui shell over the swapchain, and loads / swaps scenes at
// runtime through ONE load path.
//
// PIPELINE:
//   1. Stand up the persistent shell FIRST (no scene): phi device + backend,
//      MakeSurface -> PresentRenderer (swapchain) -> NukaImGuiContext over the
//      present pass + the renderer's descriptor pool. The viewport clears to the
//      background until a scene is loaded.
//   2. Runtime load (the ONE path): LoadEditorScene(path) cooks a .nks ->
//      nk::World + RenderWorld + Simulation, swapped into `loaded`. The Load UI
//      button and the --scene flag both go through it -- --scene just seeds a load
//      request the loop honors after the shell is up (no startup-only cook).
//   3. The main loop, until the window closes:
//        process a pending load/unload (WaitIdle -> teardown -> rebuild -> reframe)
//        -> poll window events -> ImGui io + CameraController + CommandQueue
//        -> Simulation::FramePublish (step + publish; skipped while empty)
//        -> PresentRenderer::DrawFrame(active or empty RenderWorld, camera opts,
//           overlay = the imgui_layer record) -> pace to the frame interval.
//
// PRESENT egress is DIRECT-to-swapchain; the offscreen Render() stays the untouched
// D1 oracle. The editor publishes poses via each loaded scene's HostDownloadPublisher
// (the general path that fits any instance count across runtime swaps).
//
// HEADLESS-SAFE: a frame budget (--frames / NUKA_VIEWER_FRAMES) + swapchain readback
// (--capture-frame) make both the empty shell and a runtime load verifiable without
// a human at the window. GATING: built ONLY when NK_BUILD_VULKAN_VALIDATION is ON.
//
// HOST-ONLY / zero-CUDA-token: pure host C++/Vulkan/ImGui. The only device touch
// is nk::World step + DownloadField (through the phi v2 vtable in nuka_nk).
// ---------------------------------------------------------------------------

#include <vulkan/vulkan.h>

#include "imgui.h"
#ifdef _WIN32
// The GLFW platform backend feeds ImGui keyboard/char/mouse on Windows (the xcb
// build self-feeds io). It draws nothing -- imgui_impl_vulkan stays the renderer.
// imgui_impl_glfw.h forward-declares GLFWwindow, so no GLFW header is needed here.
#include "backends/imgui_impl_glfw.h"
#endif

#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "render/imgui/nuka_imgui.hpp"
#include "render/raster/vulkan_present_renderer.hpp"
#include "render/render_world.hpp"
#include "render/window/window_surface.hpp"
#include "runtime/app/command_queue.hpp"        // MoveEntity command (VIEW-4)
#include "runtime/app/pose_publisher.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/editor_edits.hpp"    // the general scene-edit seam
#include "runtime/app/viewer/editor_scene.hpp"   // EditorScene + LoadEditorScene
#include "runtime/app/viewer/imgui_layer.hpp"
#include "runtime/inference/go2_policy_controller.hpp"  // --policy closed-loop drive
#include "scene/format/nks.hpp"                   // nks::Save (persist edits)
#include "scene/scene_ir.hpp"

#include "nk/model/generated/field_ids.hpp"     // FieldId::DriveTarget (VIEW-2)
#include "nk/model/model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace app = nuka::runtime::app;
namespace viewer = nuka::runtime::app::viewer;
namespace render = nuka::render;
namespace window = nuka::render::window;

// Default physics timestep when neither --dt nor a scene override is supplied.
constexpr float kDefaultDt = 1.0f / 240.0f;

// Seed the GENERIC per-DOF drive editor from a freshly loaded world's DriveTarget
// field (per:link, env 0). A flat per-DOF editor, NEVER a choreography table.
void SeedDriveTargets(viewer::ViewerUiState& ui, viewer::EditorScene& es) {
    ui.drive_targets.clear();
    ui.drive_dirty.clear();
    ui.dof_labels.clear();
    if (es.caps.links_per_env > 0u) {
        ui.drive_targets.assign(es.caps.links_per_env, 0.0f);
        ui.drive_dirty.assign(es.caps.links_per_env, 0u);
        es.sim->GetWorld().GetData().DownloadField(
            nk::FieldId::DriveTarget, ui.drive_targets.data(),
            static_cast<uint64_t>(es.caps.links_per_env) * sizeof(float), 0);
    }
}

// Drop all model-bound UI state back to the empty-editor defaults.
void ClearModelUi(viewer::ViewerUiState& ui) {
    ui.drive_targets.clear();
    ui.drive_dirty.clear();
    ui.dof_labels.clear();
    ui.selected_entity = nuka::scene::kInvalidEntity;
    ui.playing = false;
}

// Compute the world AABB of the rendered instances so the camera can frame it.
// When `movable_only` and at least one movable (Link/Body/Base) instance exists,
// fit just that cluster -- so a handful of robots on a large terrain fill the view
// instead of rendering tiny; otherwise the whole scene. General: no scene-specific
// camera, the predicate is "is this instance driven by physics".
void SceneAabb(const render::RenderWorld& world, bool movable_only,
               nuka::math::Vec3* lo, nuka::math::Vec3* hi) {
    bool has_movable = false;
    if (movable_only) {
        for (const render::RenderInstance& inst : world.instances) {
            if (inst.pose_source.kind != render::PoseSource::Kind::Static) {
                has_movable = true;
                break;
            }
        }
    }
    const bool restrict_movable = movable_only && has_movable;
    const float fmax = std::numeric_limits<float>::max();
    *lo = {fmax, fmax, fmax};
    *hi = {-fmax, -fmax, -fmax};
    bool any = false;
    for (const render::RenderInstance& inst : world.instances) {
        if (inst.mesh_id == render::kNoId || inst.mesh_id >= world.meshes.Count()) continue;
        if (restrict_movable && inst.pose_source.kind == render::PoseSource::Kind::Static) {
            continue;
        }
        const render::MeshGeometry& geo = world.meshes.Geometry(inst.mesh_id);
        for (size_t v = 0; v + 2 < geo.positions.size(); v += 3) {
            const nuka::math::Vec3 local{geo.positions[v], geo.positions[v + 1],
                                         geo.positions[v + 2]};
            const nuka::math::Vec3 wp = inst.world_xform.TransformPoint(local);
            lo->x = std::min(lo->x, wp.x); lo->y = std::min(lo->y, wp.y); lo->z = std::min(lo->z, wp.z);
            hi->x = std::max(hi->x, wp.x); hi->y = std::max(hi->y, wp.y); hi->z = std::max(hi->z, wp.z);
            any = true;
        }
    }
    if (!any) { *lo = {-1.0f, -1.0f, 0.0f}; *hi = {1.0f, 1.0f, 1.0f}; }
}

// Map a window MouseButton (0=L,1=M,2=R) to ImGui's button index (0=L,1=R,2=M).
int ToImGuiMouseButton(uint32_t b) {
    if (b == 0u) return 0;       // left
    if (b == 2u) return 1;       // right
    return 2;                    // middle
}

// World-space AABB of one render instance (transform its mesh positions). Empty
// (lo>hi) when the instance has no geometry.
void InstanceAabb(const render::RenderInstance& inst, const render::RenderWorld& w,
                  nuka::math::Vec3* lo, nuka::math::Vec3* hi) {
    const float fmax = std::numeric_limits<float>::max();
    *lo = {fmax, fmax, fmax};
    *hi = {-fmax, -fmax, -fmax};
    if (inst.mesh_id == render::kNoId || inst.mesh_id >= w.meshes.Count()) return;
    const render::MeshGeometry& geo = w.meshes.Geometry(inst.mesh_id);
    for (size_t v = 0; v + 2 < geo.positions.size(); v += 3) {
        const nuka::math::Vec3 local{geo.positions[v], geo.positions[v + 1],
                                     geo.positions[v + 2]};
        const nuka::math::Vec3 p = inst.world_xform.TransformPoint(local);
        lo->x = std::min(lo->x, p.x); lo->y = std::min(lo->y, p.y); lo->z = std::min(lo->z, p.z);
        hi->x = std::max(hi->x, p.x); hi->y = std::max(hi->y, p.y); hi->z = std::max(hi->z, p.z);
    }
}

// Ray vs AABB slab test. Returns the near hit distance t>=0 in *t_out and true on
// a hit in front of the ray.
bool RayAabb(const viewer::Ray& ray, const nuka::math::Vec3& lo,
             const nuka::math::Vec3& hi, float* t_out) {
    float tmin = 0.0f;
    float tmax = std::numeric_limits<float>::max();
    const float o[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    const float d[3] = {ray.dir.x, ray.dir.y, ray.dir.z};
    const float lo3[3] = {lo.x, lo.y, lo.z};
    const float hi3[3] = {hi.x, hi.y, hi.z};
    for (int a = 0; a < 3; ++a) {
        if (std::fabs(d[a]) < 1e-8f) {
            if (o[a] < lo3[a] || o[a] > hi3[a]) return false;  // parallel + outside
        } else {
            const float inv = 1.0f / d[a];
            float t1 = (lo3[a] - o[a]) * inv;
            float t2 = (hi3[a] - o[a]) * inv;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
    if (t_out) *t_out = tmin;
    return true;
}

// VIEW-F2: the u8 value of ArticulationJointType::FloatingBase (==3). The canonical
// enum is in runtime/articulation/articulation_state.hpp, but that header pulls in
// <cuda_runtime.h> and this file is on the zero-CUDA-token red-line, so we mirror the
// stable cooked-Model value here (same constant as systems.cpp::kFloatingBaseJointType).
constexpr uint8_t kFloatingBaseJointType = 3u;

// True iff `src` names the FLOATING articulation root link (joint_type[root]==
// FloatingBase). Such a link is the ONE link whose live pose IS editable (it is
// driven via BasePose; ApplyMoveEntity reroutes a root-link Link source to Base).
bool IsFloatingRootLink(const render::PoseSource& src, const nk::ModelArticulation& art) {
    if (src.kind != render::PoseSource::Kind::Link) return false;
    const uint32_t root = (art.root_link != ~uint32_t(0)) ? art.root_link : 0u;
    if (src.row != root) return false;
    return root < art.joint_type.size() && art.joint_type[root] == kFloatingBaseJointType;
}

// Pick the nearest MOVABLE instance the ray hits. Returns its index or ~0u.
// VIEW-F1: a movable instance is a free body (Body), an articulation floating base
// (Base), OR the FLOATING ROOT LINK (its Link source reroutes to Base on move) --
// so the advertised floating-base teleport is genuinely UI-reachable (the Base
// source is never emitted by ResolvePoseSource, which has no Model; the root link is
// the reachable handle for it).
uint32_t PickInstance(const viewer::Ray& ray, const render::RenderWorld& w,
                      const nk::ModelArticulation& art) {
    uint32_t best = ~0u;
    float best_t = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < w.InstanceCount(); ++i) {
        const render::RenderInstance& inst = w.instances[i];
        const bool movable = inst.pose_source.kind == render::PoseSource::Kind::Body ||
                             inst.pose_source.kind == render::PoseSource::Kind::Base ||
                             IsFloatingRootLink(inst.pose_source, art);
        if (!movable) continue;  // only movable bodies / the floating root are draggable
        nuka::math::Vec3 lo, hi;
        InstanceAabb(inst, w, &lo, &hi);
        if (lo.x > hi.x) continue;  // empty
        float t;
        if (RayAabb(ray, lo, hi, &t) && t < best_t) { best_t = t; best = i; }
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    // ---- args: scene path (OPTIONAL -> empty editor), frame budget, dt ----------
    std::string scene_path;          // empty -> open EMPTY; --scene seeds a load.
    int max_frames = 0;  // 0 -> run until the window closes (interactive).
    float dt = kDefaultDt;           // overridable physics timestep (--dt).
    int capture_frame = -1;          // -1 -> no swapchain readback capture.
    std::string capture_out = "nuka_present_capture.ppm";
    std::string policy_path;         // --policy <bin>: attach the closed-loop drive.
    bool want_play = false;          // --play: start the transport running.
    bool want_planned = false;       // --planned: CUDA-graph step (one launch/frame).
    bool dt_set = false;             // honor an explicit --dt over the policy default.
    bool force_host_policy = false;  // --host-policy: run the host MLP (A/B vs GPU).
    // --cam tx,ty,tz,dist,yaw_deg,pitch_deg : explicit orbit override (else auto-frame).
    bool cam_on = false;
    nuka::math::Vec3 cam_target{};
    float cam_dist = 0.0f, cam_yaw = 0.0f, cam_pitch = 0.0f;
    // Headless editing aids: pre-select a node by path + apply a sample material
    // recolor through the general edit seam (so a capture shows a populated
    // inspector + a viewport change with no interactive mouse).
    std::string select_path;
    bool  edit_color_on = false;
    float edit_color[3] = {0.0f, 0.0f, 0.0f};
    bool  edit_move_on = false;
    float edit_move[3] = {0.0f, 0.0f, 0.0f};   // a position delta on the selection
    std::string save_to;                       // one-shot Save after the load edits
    std::string gizmo_op;                       // headless gizmo mode: move | rotate
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--scene" && i + 1 < argc) scene_path = argv[++i];
        else if (a == "--frames" && i + 1 < argc) max_frames = std::atoi(argv[++i]);
        else if (a == "--dt" && i + 1 < argc) { dt = static_cast<float>(std::atof(argv[++i])); dt_set = true; }
        else if (a == "--capture-frame" && i + 1 < argc) capture_frame = std::atoi(argv[++i]);
        else if (a == "--capture-out" && i + 1 < argc) capture_out = argv[++i];
        else if (a == "--policy" && i + 1 < argc) policy_path = argv[++i];
        else if (a == "--play") want_play = true;
        else if (a == "--planned") want_planned = true;
        else if (a == "--host-policy") force_host_policy = true;
        else if (a == "--cam" && i + 1 < argc) {
            float yd = 0.0f, pd = 0.0f;
            if (std::sscanf(argv[++i], "%f,%f,%f,%f,%f,%f", &cam_target.x, &cam_target.y,
                            &cam_target.z, &cam_dist, &yd, &pd) == 6) {
                cam_yaw = yd * 3.14159265358979323846f / 180.0f;
                cam_pitch = pd * 3.14159265358979323846f / 180.0f;
                cam_on = true;
            } else {
                std::fprintf(stderr, "[nuka_editor] --cam wants tx,ty,tz,dist,yaw_deg,pitch_deg\n");
            }
        }
        else if (a == "--select" && i + 1 < argc) select_path = argv[++i];
        else if (a == "--edit-color" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%f,%f,%f", &edit_color[0], &edit_color[1],
                            &edit_color[2]) == 3) {
                edit_color_on = true;
            }
        }
        else if (a == "--edit-move" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%f,%f,%f", &edit_move[0], &edit_move[1],
                            &edit_move[2]) == 3) {
                edit_move_on = true;
            }
        }
        else if (a == "--save-to" && i + 1 < argc) save_to = argv[++i];
        else if (a == "--gizmo-op" && i + 1 < argc) gizmo_op = argv[++i];
        else if (a == "--help") {
            std::printf("usage: nuka_editor [--scene <.nks>] [--policy <bin>] [--play] "
                        "[--frames N] [--dt SECONDS] "
                        "[--cam tx,ty,tz,dist,yaw_deg,pitch_deg] "
                        "[--capture-frame N --capture-out FILE.ppm]\n"
                        "  no --scene -> opens the empty editor; load scenes from the UI\n"
                        "  --policy attaches a trained locomotion controller to each load\n"
                        "  --host-policy runs the host MLP round-trip instead of the GPU path (A/B)\n"
                        "  --cam overrides the auto-frame (default: fit the moving-instance cluster)\n");
            return 0;
        }
    }
    if (const char* env = std::getenv("NUKA_VIEWER_FRAMES")) max_frames = std::atoi(env);
    // The trained control rate is 0.005s; default to it when a policy is attached and
    // no explicit --dt was given (an explicit --dt always wins).
    if (!dt_set && !policy_path.empty()) dt = 0.005f;
    if (!(dt > 0.0f)) {
        std::fprintf(stderr, "[nuka_editor] invalid --dt %g (must be > 0)\n", dt);
        return 2;
    }

    // ---- 1. physics backend (scene-independent; no world cooked yet) -----------
    nphi::Device* dev = nphi::InitBestDevice();
    if (!dev) { std::fprintf(stderr, "[nuka_editor] no physics device\n"); return 3; }
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    if (!backend) { std::fprintf(stderr, "[nuka_editor] no physics backend\n"); return 3; }

    // The empty viewport draws this 0-instance world (DrawFrame just clears) until a
    // scene is loaded. `loaded` holds the cooked scene; null == empty editor.
    render::RenderWorld empty_world;
    // The closed-loop controller (when --policy is given) must OUTLIVE the Simulation
    // it drives, so it is declared before `loaded` (destroyed after it).
    std::unique_ptr<nuka::runtime::inference::Go2PolicyController> policy_ctrl;
    std::unique_ptr<viewer::EditorScene> loaded;
    // The general edit machinery: entity->record reverse index (rebuilt per load)
    // and the last entity the inspector was seeded from (re-seed on a new pick).
    viewer::EntityRecordIndex record_index;
    nuka::scene::EntityId     last_seeded = nuka::scene::kInvalidEntity;

    // Resolve a node PATH to its entity by matching SceneGraph::PathOf (the headless
    // --select aid + a general path lookup). kInvalidEntity when no node matches.
    auto entity_of_path = [](const viewer::EditorScene& es,
                             const std::string& path) -> nuka::scene::EntityId {
        nuka::scene::EntityId found = nuka::scene::kInvalidEntity;
        const nuka::scene::SceneGraph& tree = es.scene.Tree();
        tree.Traverse(tree.Root(), [&](const std::shared_ptr<nuka::scene::SceneNode>& n) {
            if (found == nuka::scene::kInvalidEntity && tree.PathOf(n) == path) {
                found = n->entity;
            }
        });
        return found;
    };

    // ---- 2. window surface + present renderer ---------------------------------
    window::SurfaceBackendKind kind = window::SurfaceBackendKind::None;
    std::unique_ptr<window::WindowSurface> surface;
    try {
        surface = window::MakeSurface("Nuka Editor", 1280, 720, &kind);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[nuka_editor] MakeSurface threw: %s\n", e.what());
        return 5;
    }
    if (!surface || kind == window::SurfaceBackendKind::None) {
        std::fprintf(stderr, "[nuka_editor] no window surface (no display / no surface ext). "
                             "Run under: xvfb-run -a -s '-screen 0 1280x720x24' nuka_editor\n");
        return 5;
    }

    // Grab the native window handle BEFORE the surface is moved into the present
    // renderer, so the GLFW ImGui backend can bind it (Windows only; nullptr else).
    void* native_window = surface->NativeWindowHandle();
    (void)native_window;

    std::unique_ptr<render::PresentRenderer> present;
    try {
        present = std::make_unique<render::PresentRenderer>(std::move(surface));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[nuka_editor] PresentRenderer ctor failed: %s\n", e.what());
        return 6;
    }
    if (capture_frame >= 0) present->SetCaptureFrame(capture_frame, capture_out);

    // Pose egress is each loaded scene's HostDownloadPublisher -- the general path
    // that fits ANY instance count, so runtime scene swaps need no fixed-size wiring.

    // ---- 3. ImGui context bound to the present pass + the renderer pool --------
    render::RendererVulkanHandles vk = present->VulkanHandles();
    nuka::render::imgui::NukaImGuiInitInfo info;
    info.api_version     = vk.api_version;
    info.instance        = reinterpret_cast<NukaVkInstance>(vk.instance);
    info.physical_device = reinterpret_cast<NukaVkPhysicalDevice>(vk.physical_device);
    info.device          = reinterpret_cast<NukaVkDevice>(vk.device);
    info.queue_family    = vk.graphics_family;
    info.queue           = reinterpret_cast<NukaVkQueue>(vk.graphics_queue);
    info.descriptor_pool = reinterpret_cast<NukaVkDescriptorPool>(vk.imgui_descriptor_pool);
    info.min_image_count = present->MinImageCount();
    info.image_count     = present->SwapchainImageCount();
    info.render_pass     = reinterpret_cast<NukaVkRenderPass>(present->PresentRenderPass());
    info.subpass         = 0u;

    nuka::render::imgui::NukaImGuiContext imgui;
    if (!imgui.Init(info)) {
        std::fprintf(stderr, "[nuka_editor] ImGui init failed\n");
        return 7;
    }
#ifdef _WIN32
    // Bind the GLFW platform backend (install_callbacks=true chains the window
    // backend's callbacks) so ImGui receives keyboard/char/mouse from glfw.
    if (native_window != nullptr) {
        ImGui_ImplGlfw_InitForVulkan(static_cast<GLFWwindow*>(native_window), true);
    }
#endif

    // ---- 4. UI + camera state (empty editor; nothing cooked yet) --------------
    viewer::ImGuiLayer ui;
    ui.EnableDocking();  // MUST precede the first NewFrame (ImGui asserts this).
    viewer::CameraController camera;
    viewer::ViewerUiState ui_state;
    // Frame the world: honor an explicit --cam, else auto-fit the moving-instance
    // cluster (the robots) so they fill the view rather than the whole terrain.
    auto frame_world = [&](const render::RenderWorld& rw) {
        if (cam_on) {
            camera.SetView(cam_target, cam_dist, cam_yaw, cam_pitch);
            return;
        }
        nuka::math::Vec3 lo, hi;
        SceneAabb(rw, /*movable_only=*/true, &lo, &hi);
        camera.FrameAabb(lo, hi);
    };
    {
        // Frame a default box so the empty viewport has a sensible camera.
        const nuka::math::Vec3 lo{-1.0f, -1.0f, 0.0f};
        const nuka::math::Vec3 hi{1.0f, 1.0f, 1.0f};
        camera.FrameAabb(lo, hi);
    }

    // The ONE runtime load: teardown the current scene (if any), cook the new one,
    // swap it in, reframe the camera + reset model-bound UI. WaitIdle first so no
    // in-flight frame references the world / RenderWorld being freed. A failed load
    // leaves the editor exactly as it was.
    auto apply_load = [&](const std::string& path) {
        present->WaitIdle();
        std::unique_ptr<viewer::EditorScene> next =
            viewer::LoadEditorScene(path, dev, backend, dt);
        if (!next) return;
        policy_ctrl.reset();       // drop any controller bound to the old scene
        loaded = std::move(next);  // old scene (if any) destructed here
        loaded->sim->SetPlannedStep(want_planned);  // CUDA-graph replay when --planned
        loaded->sim->FramePublish(nullptr, /*do_step=*/false);  // poses before framing
        frame_world(loaded->sim->GetRenderWorld());
        SeedDriveTargets(ui_state, *loaded);
        // The general edit machinery for this scene: rebuild the entity->record
        // reverse index, reset the inspector, seed the Save path from the source.
        record_index.Build(loaded->scene);
        ui_state.inspector = viewer::InspectorState{};
        last_seeded = nuka::scene::kInvalidEntity;
        std::snprintf(ui_state.save_path, sizeof(ui_state.save_path), "%s", path.c_str());
        ui_state.save_dirty = false;
        ui_state.selected_entity = nuka::scene::kInvalidEntity;
        // Headless aids: pre-select a node + apply a sample recolor via the seam.
        if (!select_path.empty()) {
            const nuka::scene::EntityId e = entity_of_path(*loaded, select_path);
            if (e != nuka::scene::kInvalidEntity) ui_state.selected_entity = e;
            else std::fprintf(stderr, "[nuka_editor] --select: no node at '%s'\n",
                              select_path.c_str());
        }
        if (edit_color_on && ui_state.selected_entity != nuka::scene::kInvalidEntity) {
            const viewer::MaterialBinding mb =
                viewer::ResolveMaterial(*loaded, record_index, ui_state.selected_entity);
            if (mb.valid) {
                nuka::scene::RenderMaterial m =
                    loaded->sim->GetRenderWorld().materials[mb.render_slot];
                m.base_color[0] = edit_color[0];
                m.base_color[1] = edit_color[1];
                m.base_color[2] = edit_color[2];
                viewer::ApplyMaterialEdit(*loaded, mb, m, /*commit=*/true);
                ui_state.save_dirty = true;
            }
        }
        if (edit_move_on && ui_state.selected_entity != nuka::scene::kInvalidEntity) {
            if (const render::RenderInstance* inst = viewer::InstanceOfEntity(
                    loaded->sim->GetRenderWorld(), ui_state.selected_entity)) {
                nuka::math::Transform world = inst->world_xform;
                world.position.x += edit_move[0];
                world.position.y += edit_move[1];
                world.position.z += edit_move[2];
                viewer::ApplyTransformEdit(*loaded, record_index, ui_state.selected_entity,
                                           world, /*commit=*/true);
                ui_state.save_dirty = true;
            }
        }
        if (gizmo_op == "rotate") ui_state.gizmo.op = viewer::GizmoState::Op::Rotate;
        else if (gizmo_op == "move") ui_state.gizmo.op = viewer::GizmoState::Op::Translate;
        // Headless one-shot Save (verifies the full editor save path end-to-end).
        if (!save_to.empty()) ui_state.save_request = save_to;
        ui_state.playing = false;  // open paused on the authored pose
        // Attach the trained locomotion controller to this freshly cooked scene when
        // --policy was given and the scene retained terrain (height-scan obs source).
        if (!policy_path.empty() && nuka::terrain::IsValid(loaded->terrain)) {
            auto pc = std::make_unique<nuka::runtime::inference::Go2PolicyController>();
            if (pc->Init(policy_path, loaded->caps.links_per_env,
                         loaded->caps.articulations_per_env, loaded->terrain)) {
                if (force_host_policy) pc->SetUseGpu(false);  // A/B: host round-trip
                policy_ctrl = std::move(pc);
                loaded->sim->SetController(policy_ctrl.get());
                ui_state.playing = want_play;
                std::printf("[nuka_editor] policy attached: %s  play=%s  path=%s\n",
                            policy_path.c_str(), want_play ? "yes" : "no",
                            policy_ctrl->UsingGpu() ? "GPU" : "host");
            } else {
                std::fprintf(stderr, "[nuka_editor] policy Init failed: %s\n",
                             policy_path.c_str());
            }
        }
    };
    auto apply_unload = [&]() {
        present->WaitIdle();
        policy_ctrl.reset();
        loaded.reset();
        ClearModelUi(ui_state);
        record_index = viewer::EntityRecordIndex{};
        ui_state.inspector = viewer::InspectorState{};
        last_seeded = nuka::scene::kInvalidEntity;
        ui_state.save_dirty = false;
        const nuka::math::Vec3 lo{-1.0f, -1.0f, 0.0f};
        const nuka::math::Vec3 hi{1.0f, 1.0f, 1.0f};
        camera.FrameAabb(lo, hi);
    };

    // Seed the inspector's editable values from the selected instance (world pose +
    // material). Called when the selection changes; the viewer also tracks a
    // movable body's live pose each frame while the user is not editing.
    auto seed_inspector = [&](nuka::scene::EntityId e) {
        viewer::InspectorState& ins = ui_state.inspector;
        ins = viewer::InspectorState{};
        if (!loaded || e == nuka::scene::kInvalidEntity) return;
        const render::RenderInstance* inst =
            viewer::InstanceOfEntity(loaded->sim->GetRenderWorld(), e);
        if (inst == nullptr) return;
        ins.valid   = true;
        ins.movable = viewer::IsMovableInstance(*inst, *loaded);
        const nuka::math::Vec3 p = inst->world_xform.position;
        ins.pos[0] = p.x; ins.pos[1] = p.y; ins.pos[2] = p.z;
        const nuka::math::Vec3 rd = viewer::EulerDegFromQuat(inst->world_xform.rotation);
        ins.rot_deg[0] = rd.x; ins.rot_deg[1] = rd.y; ins.rot_deg[2] = rd.z;
        const viewer::MaterialBinding mb =
            viewer::ResolveMaterial(*loaded, record_index, e);
        if (mb.valid) {
            const nuka::scene::RenderMaterial& m =
                loaded->sim->GetRenderWorld().materials[mb.render_slot];
            ins.has_material = true;
            for (int k = 0; k < 4; ++k) ins.base_color[k] = m.base_color[k];
            ins.roughness = m.roughness; ins.metallic = m.metallic;
            for (int k = 0; k < 3; ++k) ins.emissive[k] = m.emissive[k];
            ins.sheen = m.sheen; ins.transmission = m.transmission; ins.ior = m.ior;
            for (int k = 0; k < 3; ++k) ins.absorption[k] = m.absorption[k];
        }
    };

    // Apply the inspector's pending edits through the general seam: live every
    // frame a widget moves, persist the record on the commit (edit-finished) frame.
    auto apply_inspector_edits = [&]() {
        if (!loaded) return;
        viewer::InspectorState& ins = ui_state.inspector;
        const nuka::scene::EntityId e = ui_state.selected_entity;
        if (!ins.valid || e == nuka::scene::kInvalidEntity) return;
        if (ins.transform_changed || ins.transform_commit) {
            nuka::math::Transform world;
            world.position = nuka::math::Vec3{ins.pos[0], ins.pos[1], ins.pos[2]};
            world.rotation = viewer::QuatFromEulerDeg(
                nuka::math::Vec3{ins.rot_deg[0], ins.rot_deg[1], ins.rot_deg[2]});
            viewer::ApplyTransformEdit(*loaded, record_index, e, world, ins.transform_commit);
            if (ins.transform_commit) ui_state.save_dirty = true;
            ins.transform_changed = false; ins.transform_commit = false;
        }
        if (ins.has_material && (ins.material_changed || ins.material_commit)) {
            const viewer::MaterialBinding mb =
                viewer::ResolveMaterial(*loaded, record_index, e);
            nuka::scene::RenderMaterial m;
            if (mb.render_slot != render::kNoId &&
                mb.render_slot < loaded->sim->GetRenderWorld().MaterialCount()) {
                m = loaded->sim->GetRenderWorld().materials[mb.render_slot];
            }
            for (int k = 0; k < 4; ++k) m.base_color[k] = ins.base_color[k];
            m.opacity = ins.base_color[3];
            m.roughness = ins.roughness; m.metallic = ins.metallic;
            for (int k = 0; k < 3; ++k) m.emissive[k] = ins.emissive[k];
            m.sheen = ins.sheen; m.transmission = ins.transmission; m.ior = ins.ior;
            for (int k = 0; k < 3; ++k) m.absorption[k] = ins.absorption[k];
            viewer::ApplyMaterialEdit(*loaded, mb, m, ins.material_commit);
            if (ins.material_commit) ui_state.save_dirty = true;
            ins.material_changed = false; ins.material_commit = false;
        }
    };

    // --scene goes through the SAME runtime load: seed a request the loop honors
    // once the shell is up (no startup-only cook path).
    if (!scene_path.empty()) ui_state.load_request = scene_path;

    std::printf("[nuka_editor] backend=%s device=%s images=%u  (empty editor)\n",
                present->Report().backend_name.c_str(), present->DeviceName().c_str(),
                present->SwapchainImageCount());

    // ---- the main loop --------------------------------------------------------
    using Clock = std::chrono::steady_clock;
    auto prev = Clock::now();
    double fps_ema = 0.0;
    uint64_t frame_index = 0;
    int presented = 0;
    bool last_step_healthy = true;
    // Headless fps instrumentation: the swapchain readback is the only trusted
    // on-screen proof on Windows (GameViewer/WGC return stale frames), so report the
    // gate numerically. step_ms (physics+policy+publish, no present) is the decisive
    // before/after of the policy round-trip; frame_ms is the end-to-end pace.
    double ft_sum = 0.0, ft_min = 1e30, ft_max = 0.0, st_sum = 0.0;
    uint64_t ft_n = 0;
    constexpr uint64_t kFpsWarmup = 30u;
    // A pace ceiling so an unbounded loop never busy-spins; FIFO present is the real
    // cap at the display refresh, so this sits well above 60.
    const auto frame_budget = std::chrono::duration<double>(1.0 / 240.0);

    // VIEW-3/4: picker + drag state. Ctrl is tracked by RESOLVED keysym (keymap-
    // independent), matching camera_controller's Shift handling. The window backend
    // resolves keycode->keysym; we compare XKB_KEY_*/XK_* protocol values.
    constexpr uint32_t kKeyCtrlL = 0xffe3u;  // XKB_KEY_Control_L / XK_Control_L
    constexpr uint32_t kKeyCtrlR = 0xffe4u;  // XKB_KEY_Control_R / XK_Control_R
    bool     ctrl_down = false;
    uint32_t drag_inst = ~0u;          // the instance being dragged (~0u == none)
    float    last_mouse_x = 0.0f;
    float    last_mouse_y = 0.0f;

    std::vector<window::WindowEvent> events;
    while (!present->ShouldClose()) {
        if (max_frames > 0 && static_cast<int>(frame_index) >= max_frames) break;

        // -- honor a pending Load / Unload (UI or --scene) before anything reads the
        // world this frame. Unload first so a request for both ends empty-then-load.
        if (ui_state.unload_request) { apply_unload(); ui_state.unload_request = false; }
        if (!ui_state.load_request.empty()) {
            const std::string path = ui_state.load_request;
            ui_state.load_request.clear();
            apply_load(path);
        }

        // -- poll + dispatch input ----------------------------------------------
        events.clear();
        present->PollEvents(events);
        ImGuiIO& io = ImGui::GetIO();
        const uint32_t vp_w = present->Report().width;
        const uint32_t vp_h = present->Report().height;
        for (const window::WindowEvent& ev : events) {
            // On Windows the GLFW ImGui backend feeds io mouse/wheel directly, so
            // the manual io feeds are skipped there (else double input); the camera
            // + picker still read the WindowEvents below on both platforms.
            switch (ev.type) {
                case window::WindowEvent::Type::MouseMove:
#ifndef _WIN32
                    io.AddMousePosEvent(static_cast<float>(ev.mouse_x),
                                        static_cast<float>(ev.mouse_y));
#endif
                    last_mouse_x = static_cast<float>(ev.mouse_x);
                    last_mouse_y = static_cast<float>(ev.mouse_y);
                    break;
                case window::WindowEvent::Type::MouseButton:
#ifndef _WIN32
                    io.AddMouseButtonEvent(ToImGuiMouseButton(ev.button), ev.pressed);
#endif
                    break;
                case window::WindowEvent::Type::Scroll:
#ifndef _WIN32
                    io.AddMouseWheelEvent(0.0f, static_cast<float>(ev.scroll_delta));
#endif
                    break;
                case window::WindowEvent::Type::Key:
                    if (ev.keysym == kKeyCtrlL || ev.keysym == kKeyCtrlR) ctrl_down = ev.pressed;
                    break;
                default:
                    break;
            }

            // -- VIEW-3/4: Ctrl+LMB pick + drag (only when a scene is loaded, ImGui
            // isn't capturing the mouse, and Ctrl is held -> never fights orbit). ----
            const bool over_ui = io.WantCaptureMouse;
            if (loaded && ctrl_down && !over_ui) {
                app::Simulation& sim = *loaded->sim;
                if (ev.type == window::WindowEvent::Type::MouseButton && ev.button == 0u) {
                    if (ev.pressed) {
                        const viewer::Ray ray = camera.ScreenRay(
                            static_cast<float>(ev.mouse_x),
                            static_cast<float>(ev.mouse_y), vp_w, vp_h);
                        const uint32_t hit = PickInstance(
                            ray, sim.GetRenderWorld(),
                            sim.GetWorld().GetModel().articulation);
                        drag_inst = hit;
                        if (hit != ~0u) {
                            ui_state.selected_entity =
                                sim.GetRenderWorld().instances[hit].entity;
                        }
                    } else {
                        drag_inst = ~0u;  // release ends the drag
                    }
                } else if (ev.type == window::WindowEvent::Type::MouseMove &&
                           drag_inst != ~0u &&
                           drag_inst < sim.GetRenderWorld().InstanceCount()) {
                    // VIEW-4: unproject the cursor onto a view-facing plane through
                    // the dragged entity's current position, then push a MoveEntity
                    // (the GENERAL path: command_queue -> ApplyMoveEntity -> Data).
                    const render::RenderInstance& inst =
                        sim.GetRenderWorld().instances[drag_inst];
                    const nuka::math::Vec3 anchor = inst.world_xform.position;
                    const viewer::Ray ray = camera.ScreenRay(
                        static_cast<float>(ev.mouse_x),
                        static_cast<float>(ev.mouse_y), vp_w, vp_h);
                    const nuka::math::Vec3 fwd =
                        (camera.ResolvedTarget() - camera.ResolvedEye()).Normalized();
                    nuka::math::Vec3 hit;
                    if (camera.RayPlaneHit(ray, anchor, fwd, &hit)) {
                        nuka::math::Transform xf = inst.world_xform;
                        xf.position = hit;  // keep rotation; teleport position
                        sim.Commands().Push(nuka::runtime::app::Command::MakeMoveEntity(
                            inst.entity, xf));
                    }
                }
            }

            // Camera gets the event only if ImGui is not capturing that input, a
            // Ctrl-drag is not in progress, and the gizmo is not in use (so neither
            // picking nor the transform handles ever spin the camera).
            const bool cam_allow =
                !io.WantCaptureMouse && !ctrl_down && !ui_state.gizmo.active;
            camera.HandleEvent(ev, /*allow_drag=*/cam_allow,
                               /*allow_scroll=*/!io.WantCaptureMouse);
        }
        (void)last_mouse_x; (void)last_mouse_y;

        // -- timing -------------------------------------------------------------
        const auto now = Clock::now();
        const double frame_dt = std::chrono::duration<double>(now - prev).count();
        prev = now;
        if (frame_dt > 0.0) {
            const double inst_fps = 1.0 / frame_dt;
            fps_ema = (fps_ema == 0.0) ? inst_fps : (fps_ema * 0.9 + inst_fps * 0.1);
        }

        // -- step + publish (only when a scene is loaded; the empty editor just
        // clears + draws the shell). The present path draws either way. -----------
        const render::RenderWorld& render_world =
            loaded ? loaded->sim->GetRenderWorld() : empty_world;
        double step_ms = 0.0;
        app::InputIntents intents;
        if (loaded) {
            app::Simulation& sim = *loaded->sim;
            const bool do_step = ui_state.playing || ui_state.step_requested;
            const auto step_t0 = Clock::now();
            last_step_healthy = sim.FramePublish(&intents, do_step);
            step_ms = std::chrono::duration<double, std::milli>(Clock::now() - step_t0).count();

            // Apply queue-dispatched control intents (out-of-band producers) on top
            // of the in-UI transport edits.
            if (intents.toggle_play) ui_state.playing = !ui_state.playing;
            if (intents.set_play) ui_state.playing = intents.play_value;
            if (intents.reset) ui_state.reset_requested = true;
            if (intents.camera_reset) ui_state.camera_reset = true;
            if (intents.set_env) {
                // Honor the PosePublisher wrap contract: clamp the requested env into
                // [0, env_count) so an out-of-band SetEnv never reads out of range.
                const uint32_t env_count =
                    (loaded->caps.env_count > 0u) ? loaded->caps.env_count : 1u;
                const uint32_t env = intents.env_value % env_count;
                ui_state.env_index = env;
                sim.SetEnvIndex(env);
            }
        } else {
            last_step_healthy = true;  // nothing to step in the empty editor
        }
        // The in-UI Step button (step_requested) was consumed by `do_step` this
        // frame -> clear the one-shot edge (a queued StepOnce is honored in-frame).
        ui_state.step_requested = false;

        // Reset / camera-reset re-frame the active world (the empty world frames a
        // default box). Reset re-frames the camera, the cheap honest action.
        if (ui_state.reset_requested) {
            frame_world(render_world);
            ui_state.reset_requested = false;
        }
        if (ui_state.camera_reset) {
            frame_world(render_world);
            ui_state.camera_reset = false;
        }

        // -- stats for the UI (zeroed model facts while empty) ------------------
        viewer::ViewerStats stats;
        stats.step_time_ms  = static_cast<float>(step_ms);
        stats.fps           = static_cast<float>(fps_ema);
        stats.sub_steps     = 1u;
        if (loaded) {
            stats.dof         = loaded->caps.dofs_per_env;
            stats.links       = loaded->caps.links_per_env;
            stats.bodies      = loaded->caps.bodies_per_env;
            stats.contact_cap = loaded->caps.max_contacts_per_env;
        }
        stats.draw_calls    = render_world.InstanceCount();
        stats.frame_index   = frame_index;
        stats.step_healthy  = last_step_healthy;
        stats.device_name   = present->DeviceName();

        // Surface the load state to the UI (panels read "no scene" gracefully).
        ui_state.has_scene   = static_cast<bool>(loaded);
        ui_state.loaded_path = loaded ? loaded->path : std::string();

        // -- ImGui frame: set display size/dt, record panels, render draw data --
        io.DisplaySize = ImVec2(static_cast<float>(present->Report().width),
                                static_cast<float>(present->Report().height));
        io.DeltaTime = (dt > 0.0) ? static_cast<float>(dt) : (1.0f / 60.0f);
#ifdef _WIN32
        // Pull glfw input into io before ImGui::NewFrame (must precede it).
        if (native_window != nullptr) ImGui_ImplGlfw_NewFrame();
#endif
        // Re-seed the inspector on a new selection; otherwise track a movable body's
        // live pose while the user is not actively editing it.
        if (loaded) {
            if (ui_state.selected_entity != last_seeded) {
                seed_inspector(ui_state.selected_entity);
                last_seeded = ui_state.selected_entity;
            } else if (ui_state.inspector.valid && ui_state.inspector.movable &&
                       !ui_state.inspector.transform_changed) {
                if (const render::RenderInstance* inst = viewer::InstanceOfEntity(
                        loaded->sim->GetRenderWorld(), ui_state.selected_entity)) {
                    const nuka::math::Vec3 p = inst->world_xform.position;
                    ui_state.inspector.pos[0] = p.x;
                    ui_state.inspector.pos[1] = p.y;
                    ui_state.inspector.pos[2] = p.z;
                    const nuka::math::Vec3 rd =
                        viewer::EulerDegFromQuat(inst->world_xform.rotation);
                    ui_state.inspector.rot_deg[0] = rd.x;
                    ui_state.inspector.rot_deg[1] = rd.y;
                    ui_state.inspector.rot_deg[2] = rd.z;
                }
            }
        }

        imgui.NewFrame();
        // Thread the loaded scene's SceneIR (its authoritative Tree + ECS) into the
        // UI for the hierarchical scene tree; null while the editor is empty.
        ui.RecordUi(render_world, stats, camera, ui_state,
                    loaded ? &loaded->scene : nullptr);

        // In-viewport transform gizmo over the selected entity's LIVE world pose.
        // A drag rewrites the pose and routes it through the SAME edit seam the
        // inspector uses (ApplyTransformEdit); the inspector's numeric rows re-seed
        // from the live pose next frame, so the two stay in lockstep.
        if (loaded && ui_state.selected_entity != nuka::scene::kInvalidEntity &&
            ui_state.inspector.valid) {
            if (const render::RenderInstance* gi = viewer::InstanceOfEntity(
                    loaded->sim->GetRenderWorld(), ui_state.selected_entity)) {
                nuka::math::Transform gworld = gi->world_xform;
                bool gizmo_changed = false;
                ui.DrawGizmo(camera, vp_w, vp_h, ui_state, gworld, gizmo_changed);
                if (gizmo_changed) {
                    viewer::ApplyTransformEdit(*loaded, record_index,
                                               ui_state.selected_entity, gworld,
                                               /*commit=*/true);
                    ui_state.save_dirty = true;
                }
            }
        } else {
            ui_state.gizmo.active = false;  // nothing to manipulate this frame
        }

        // Apply the inspector's pending edits (live + commit) through the general
        // seam, then honor a pending Save of the full-fidelity scene.
        apply_inspector_edits();
        if (loaded && !ui_state.save_request.empty()) {
            const std::string sp = ui_state.save_request;
            ui_state.save_request.clear();
            try {
                nuka::scene::nks::Save(loaded->scene, sp);
                ui_state.save_dirty = false;
                std::printf("[nuka_editor] saved %s\n", sp.c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[nuka_editor] save failed: %s: %s\n",
                             sp.c_str(), e.what());
            }
        }

        // WASD/QE fly the camera (Q/E = down/up, Shift = sprint), gated on ImGui not
        // owning the keyboard so typing in a panel never moves the view.
        if (!io.WantCaptureKeyboard) {
            float cf = 0.0f, cr = 0.0f, cu = 0.0f;
            if (ImGui::IsKeyDown(ImGuiKey_W)) cf += 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_S)) cf -= 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_D)) cr += 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_A)) cr -= 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_E)) cu += 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_Q)) cu -= 1.0f;
            if (cf != 0.0f || cr != 0.0f || cu != 0.0f) {
                const bool sprint = ImGui::IsKeyDown(ImGuiKey_LeftShift) ||
                                    ImGui::IsKeyDown(ImGuiKey_RightShift);
                camera.Move(cf, cr, cu, static_cast<float>(frame_dt) * (sprint ? 4.0f : 1.0f));
            }
        }
        ImGui::Render();

        // VIEW-2: upload any drive sliders that moved this frame into the LIVE nk
        // Data (FieldId::DriveTarget, env-major: env*links_per_env + dof). Per-DOF
        // upload of ONLY the changed rows; the general write path (UploadField),
        // never a choreography table. R13: runtime Data only, never the .nks.
        if (loaded && loaded->caps.links_per_env > 0u &&
            ui_state.drive_dirty.size() == ui_state.drive_targets.size()) {
            const uint32_t per_env = loaded->caps.links_per_env;
            const uint32_t env =
                (loaded->caps.env_count > 0u) ? (ui_state.env_index % loaded->caps.env_count) : 0u;
            for (uint32_t d = 0; d < ui_state.drive_targets.size(); ++d) {
                if (!ui_state.drive_dirty[d]) continue;
                const uint64_t at =
                    (static_cast<uint64_t>(env) * per_env + d) * sizeof(float);
                loaded->sim->GetWorld().GetData().UploadField(
                    nk::FieldId::DriveTarget, &ui_state.drive_targets[d],
                    sizeof(float), at);
                ui_state.drive_dirty[d] = 0u;
            }
        }

        // -- camera override + present (overlay = the imgui draw data) ----------
        render::RasterOptions opts;
        opts.width  = present->Report().width;
        opts.height = present->Report().height;
        camera.WriteOptions(opts);

        render::PresentFrameResult r = present->DrawFrame(
            render_world, opts,
            [&imgui](void* cmd) { imgui.RenderDrawData(
                reinterpret_cast<NukaVkCommandBuffer>(cmd)); });
        if (r == render::PresentFrameResult::Error) {
            std::fprintf(stderr, "[nuka_editor] present error at frame %llu\n",
                         static_cast<unsigned long long>(frame_index));
            break;
        }
        if (r == render::PresentFrameResult::Recreated) {
            // Swapchain + present pass were rebuilt on resize/maximize. The new pass is
            // render-pass-compatible, so ImGui keeps drawing; just refresh the image count.
            imgui.NotifySwapchainRecreated(present->MinImageCount());
        }
        if (r == render::PresentFrameResult::Presented) ++presented;

        ++frame_index;

        if (frame_dt > 0.0 && frame_index > kFpsWarmup) {
            const double fms = frame_dt * 1000.0;
            ft_sum += fms;
            st_sum += step_ms;
            ++ft_n;
            ft_min = std::min(ft_min, fms);
            ft_max = std::max(ft_max, fms);
            if (frame_index % 120u == 0u) {
                std::printf("[fps] frame=%llu fps_ema=%.1f frame_ms=%.2f step_ms=%.2f\n",
                            static_cast<unsigned long long>(frame_index), fps_ema, fms,
                            step_ms);
                std::fflush(stdout);
            }
        }

        // -- pace (no busy-spin) ------------------------------------------------
        const auto spent = std::chrono::duration<double>(Clock::now() - now);
        if (spent < frame_budget) {
            std::this_thread::sleep_for(
                std::chrono::duration_cast<std::chrono::microseconds>(frame_budget - spent));
        }
    }

    present->WaitIdle();
#ifdef _WIN32
    if (native_window != nullptr) ImGui_ImplGlfw_Shutdown();
#endif
    imgui.Shutdown();

    // Tear the loaded world down while the present renderer + backend are still
    // alive (no in-flight frame after WaitIdle).
    loaded.reset();

    std::printf("[nuka_editor] done: frames=%llu presented=%d last_step_healthy=%s\n",
                static_cast<unsigned long long>(frame_index), presented,
                last_step_healthy ? "yes" : "no");
    if (ft_n > 0) {
        const double mean_fms = ft_sum / static_cast<double>(ft_n);
        const double mean_sms = st_sum / static_cast<double>(ft_n);
        std::printf("[fps] SUMMARY measured=%llu mean_fps=%.1f mean_frame_ms=%.2f "
                    "(min=%.2f max=%.2f) mean_step_ms=%.2f\n",
                    static_cast<unsigned long long>(ft_n), 1000.0 / mean_fms, mean_fms,
                    ft_min, ft_max, mean_sms);
    }
    return 0;
}
