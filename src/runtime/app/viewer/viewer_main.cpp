// ---------------------------------------------------------------------------
// nuka_viewer -- the M8.5 windowed realtime viewer entry (T3, the user-facing
// deliverable). Drives the physics sim and shows it inside the custom-beautified
// Dear ImGui UI over the swapchain present path.
//
// PIPELINE (recon §2 Phase 3 + the decisions in §8):
//   1. Cook a real .nks scene -> nk::World (the render_frame_loop_smoke fixture:
//      Load(.nks) -> CookToModel -> nk::World) + a RenderWorld from the same cook.
//   2. HostDownloadPublisher + Simulation over them (the M8 frame-loop spine).
//   3. MakeSurface (xcb under Xvfb here / headless on a modern loader) ->
//      PresentRenderer (the swapchain renderer) -> NukaImGuiContext (init against
//      the present pass + the renderer's descriptor pool).
//   4. The main loop, until the window closes:
//        poll window events -> feed ImGui io + CameraController + CommandQueue
//        -> Simulation::FramePublish (step + publish, NO offscreen draw)
//        -> PresentRenderer::DrawFrame(RenderWorld, camera opts, overlay = the
//           imgui_layer record) -> pace to the target frame interval.
//
// PRESENT egress is DIRECT-to-swapchain (D-open-Q 7); the offscreen Render() stays
// the untouched D1 oracle. Host-download pose is fine for M8.5 (interop = M11).
//
// HEADLESS-SAFE: the whole loop runs under Xvfb (the present substrate is T2-/
// vkcube-proven there). A frame budget (--frames / NUKA_VIEWER_FRAMES) makes it
// CI-runnable for a few frames without a human at the window.
//
// GATING: this exe is built ONLY when NK_BUILD_VULKAN_VALIDATION is ON (D3); the
// default build-cuda128 config never sees it (opt-in, like the recorder).
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
#include "phi/interop_scatter.hpp"  // ExternalMemoryDesc (CUDA-free seam, INT-8)
#include "render/imgui/nuka_imgui.hpp"
#include "render/raster/interop_transform_ssbo.hpp"  // exportable SSBO (INT-4)
#include "render/raster/vulkan_present_renderer.hpp"
#include "render/render_world.hpp"
#include "render/window/window_surface.hpp"
#include "runtime/app/command_queue.hpp"        // MoveEntity command (VIEW-4)
#include "runtime/app/cuda_vulkan_interop.hpp"  // CudaVulkanInteropPublisher (INT-6)
#include "runtime/app/pose_publisher.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "scene/graph/scene_graph.hpp"          // SceneGraph::SetSelected (VIEW-3)
#include "scene/scene_ir.hpp"

#include "nk/model/generated/field_ids.hpp"     // FieldId::DriveTarget (VIEW-2)
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"

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
#include <thread>
#include <vector>

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace app = nuka::runtime::app;
namespace viewer = nuka::runtime::app::viewer;
namespace render = nuka::render;
namespace window = nuka::render::window;
namespace cook = nuka::scene::cook;

// Default physics timestep when neither --dt nor a scene override is supplied.
constexpr float kDefaultDt = 1.0f / 240.0f;

// Strip shape mesh geometry so CookToModel skips the heavy V-HACD pass (the
// viewer validates the LIVE loop, not collision-hull fidelity -- same trick the
// frame-loop smoke uses). Pose Data fields the publisher reads are unaffected.
nuka::scene::SceneIR LoadLightScene(const std::string& path) {
    nuka::scene::SceneIR scene = nuka::scene::nks::Load(path);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& shape = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        shape.mesh_vertices.clear();
        shape.mesh_indices.clear();
    }
    return scene;
}

nk::Pipeline::SolverConfig DefaultCfg(float dt) {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = -9.81f;
    return cfg;
}

// Compute the scene AABB from the RenderWorld so the camera can frame it.
void SceneAabb(const render::RenderWorld& world, nuka::math::Vec3* lo,
               nuka::math::Vec3* hi) {
    const float fmax = std::numeric_limits<float>::max();
    *lo = {fmax, fmax, fmax};
    *hi = {-fmax, -fmax, -fmax};
    bool any = false;
    for (const render::RenderInstance& inst : world.instances) {
        if (inst.mesh_id == render::kNoId || inst.mesh_id >= world.meshes.Count()) continue;
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
    // ---- args: scene path (REQUIRED), frame budget (0 = run until close), dt ----
    std::string scene_path;          // no default scene -> require --scene.
    int max_frames = 0;  // 0 -> run until the window closes (interactive).
    float dt = kDefaultDt;           // overridable physics timestep (--dt).
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--scene" && i + 1 < argc) scene_path = argv[++i];
        else if (a == "--frames" && i + 1 < argc) max_frames = std::atoi(argv[++i]);
        else if (a == "--dt" && i + 1 < argc) dt = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--help") {
            std::printf("usage: nuka_viewer --scene <.nks> [--frames N] [--dt SECONDS]\n");
            return 0;
        }
    }
    if (const char* env = std::getenv("NUKA_VIEWER_FRAMES")) max_frames = std::atoi(env);

    if (scene_path.empty()) {
        std::fprintf(stderr, "[nuka_viewer] no scene: pass --scene <.nks> "
                             "(see --help)\n");
        return 2;
    }
    if (!(dt > 0.0f)) {
        std::fprintf(stderr, "[nuka_viewer] invalid --dt %g (must be > 0)\n", dt);
        return 2;
    }
    if (!std::filesystem::exists(scene_path)) {
        std::fprintf(stderr, "[nuka_viewer] scene not found: %s\n", scene_path.c_str());
        return 2;
    }

    // ---- 1. physics backend + cook -> nk::World + RenderWorld ------------------
    nphi::Device* dev = nphi::InitBestDevice();
    if (!dev) { std::fprintf(stderr, "[nuka_viewer] no physics device\n"); return 3; }
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    if (!backend) { std::fprintf(stderr, "[nuka_viewer] no physics backend\n"); return 3; }

    const nuka::scene::SceneIR scene = LoadLightScene(scene_path);
    cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
    render::RenderWorld render_world =
        render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);

    const nk::ModelCapacities caps = cooked.model.capacities;  // copy BEFORE move.
    nk::World world(std::move(cooked.model), 1u, dev, backend, DefaultCfg(dt));
    if (!world.Ready()) { std::fprintf(stderr, "[nuka_viewer] world not ready\n"); return 4; }

    app::HostDownloadPublisher publisher;
    app::Simulation sim(world, publisher, std::move(render_world));

    // ---- 2. window surface + present renderer ---------------------------------
    window::SurfaceBackendKind kind = window::SurfaceBackendKind::None;
    std::unique_ptr<window::WindowSurface> surface;
    try {
        surface = window::MakeSurface("Nuka Physics Viewer", 1280, 720, &kind);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[nuka_viewer] MakeSurface threw: %s\n", e.what());
        return 5;
    }
    if (!surface || kind == window::SurfaceBackendKind::None) {
        std::fprintf(stderr, "[nuka_viewer] no window surface (no display / no surface ext). "
                             "Run under: xvfb-run -a -s '-screen 0 1280x720x24' nuka_viewer\n");
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
        std::fprintf(stderr, "[nuka_viewer] PresentRenderer ctor failed: %s\n", e.what());
        return 6;
    }

    // ---- 2b. CUDA<->Vulkan interop publisher (M11 INT-8) -- zero-copy device
    // scatter behind the PosePublisher seam, with GRACEFUL FALLBACK to
    // HostDownloadPublisher. On this lavapipe/CPU-Vulkan box the exportable SSBO is
    // unsupported (vkGetMemoryFdKHR absent) -> TryEnable returns false -> we keep
    // `publisher` (HostDownloadPublisher). On a display+NVIDIA machine the SSBO
    // exports an OPAQUE_FD, the CUDA scatter imports it, and the sim publishes the
    // physics device state with NO D2H copy. The interop publisher + SSBO must
    // outlive `sim` (declared at function scope).
    render::RendererVulkanHandles vk_handles = present->VulkanHandles();
    render::InteropTransformSsbo  interop_ssbo;
    app::CudaVulkanInteropPublisher interop_publisher;
    bool interop_active = false;
    if (nphi::CudaVkScatterAvailable()) {
        const uint32_t cap = std::max<uint32_t>(sim.GetRenderWorld().InstanceCount(), 1u);
        if (interop_ssbo.Create(reinterpret_cast<VkDevice>(vk_handles.device),
                                reinterpret_cast<VkPhysicalDevice>(vk_handles.physical_device),
                                cap)) {
            nphi::ExternalMemoryDesc mem;
            mem.fd = interop_ssbo.TakeExportedFd();
            mem.size_bytes = interop_ssbo.SizeBytes();
            mem.dedicated = interop_ssbo.Dedicated();
            interop_active = interop_publisher.TryEnable(
                sim.GetWorld(), sim.GetRenderWorld(), mem, cap);
        }
    }
    if (interop_active) {
        // INT-F1: close the producer->consumer loop. The publisher SCATTERS the
        // composed transforms into the SSBO; wire that SAME SSBO into the present
        // renderer's instanced pipeline (set 1) so the draw READS the scattered
        // device transforms by gl_InstanceIndex (true zero-copy, no per-draw
        // world_xform). If the present renderer cannot build the instanced pipeline,
        // it is NOT genuine zero-copy -> revert to HostDownloadPublisher so the
        // scatter output is never silently drawn over frozen bind poses.
        present->SetInteropTransforms(
            reinterpret_cast<render::NkVkDescriptorSetLayout>(interop_ssbo.SetLayout()),
            reinterpret_cast<render::NkVkDescriptorSet>(interop_ssbo.DescriptorSet()));
        if (present->InteropDrawActive()) {
            sim.SetPublisher(interop_publisher);  // zero-copy device scatter -> SSBO -> instanced draw
            std::printf("[nuka_viewer] CUDA<->Vulkan interop ACTIVE "
                        "(zero-copy transform SSBO + instanced draw)\n");
        } else {
            interop_active = false;
            std::printf("[nuka_viewer] interop scatter available but the instanced draw "
                        "pipeline did not install -> HostDownloadPublisher (no frozen draw)\n");
        }
    } else {
        std::printf("[nuka_viewer] interop unavailable -> HostDownloadPublisher "
                    "(graceful fallback; expected on lavapipe / no external-memory-fd)\n");
    }

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
        std::fprintf(stderr, "[nuka_viewer] ImGui init failed\n");
        return 7;
    }
#ifdef _WIN32
    // Bind the GLFW platform backend (install_callbacks=true chains the window
    // backend's callbacks) so ImGui receives keyboard/char/mouse from glfw.
    if (native_window != nullptr) {
        ImGui_ImplGlfw_InitForVulkan(static_cast<GLFWwindow*>(native_window), true);
    }
#endif

    // ---- 4. UI + camera state -------------------------------------------------
    viewer::ImGuiLayer ui;
    ui.EnableDocking();  // MUST precede the first NewFrame (ImGui asserts this).
    viewer::CameraController camera;
    viewer::ViewerUiState ui_state;
    {
        nuka::math::Vec3 lo, hi;
        // Publish once so the RenderWorld holds live poses before framing.
        sim.FramePublish(nullptr, /*do_step=*/false);
        SceneAabb(sim.GetRenderWorld(), &lo, &hi);
        camera.FrameAabb(lo, hi);
    }

    // VIEW-2: seed the GENERIC per-DOF drive-target editor from the cooked
    // DriveTarget field for env 0 (DriveTarget is per:link -> links_per_env floats
    // per env). The slider edits feed FieldId::DriveTarget LIVE each frame. This is
    // a flat per-DOF editor, NOT a hardcoded choreography (OD-7).
    if (caps.links_per_env > 0u) {
        ui_state.drive_targets.assign(caps.links_per_env, 0.0f);
        ui_state.drive_dirty.assign(caps.links_per_env, 0u);
        sim.GetWorld().GetData().DownloadField(
            nk::FieldId::DriveTarget, ui_state.drive_targets.data(),
            static_cast<uint64_t>(caps.links_per_env) * sizeof(float), 0);
    }

    // VIEW-3: a viewer-owned editor SceneGraph honoring the SceneGraph::SetSelected
    // seam. The SceneIR (`scene`) is RETAINED at function scope (its Ecs() resolves
    // entity->node); `editor_graph` shares those nodes via the registry backref, so
    // SetSelected(weak) tracks the picked node without copying the tree.
    nuka::scene::SceneGraph editor_graph;

    std::printf("[nuka_viewer] backend=%s device=%s images=%u scene=%s dof=%u instances=%u\n",
                present->Report().backend_name.c_str(), present->DeviceName().c_str(),
                present->SwapchainImageCount(), scene_path.c_str(), caps.dofs_per_env,
                sim.GetRenderWorld().InstanceCount());

    // ---- the main loop --------------------------------------------------------
    using Clock = std::chrono::steady_clock;
    auto prev = Clock::now();
    double fps_ema = 0.0;
    uint64_t frame_index = 0;
    int presented = 0;
    bool last_step_healthy = true;
    const auto frame_budget = std::chrono::duration<double>(1.0 / 60.0);

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

            // -- VIEW-3/4: Ctrl+LMB pick + drag (only when ImGui isn't capturing
            // the mouse and Ctrl is held -> never fights the orbit camera). --------
            const bool over_ui = io.WantCaptureMouse;
            if (ctrl_down && !over_ui) {
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
                            ui_state.selected_inst = hit;
                            const nuka::scene::EntityId e =
                                sim.GetRenderWorld().instances[hit].entity;
                            ui_state.selected_entity = e.index;
                            // VIEW-3: honor the SceneGraph editor-selection seam.
                            editor_graph.SetSelected(scene.Ecs().NodeOf(e));
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

            // Camera gets the event only if ImGui is not capturing that input AND a
            // Ctrl-drag is not in progress (so picking never spins the camera).
            const bool cam_allow = !io.WantCaptureMouse && !(ctrl_down);
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

        // -- step + publish (no offscreen draw; the present path draws) ---------
        app::InputIntents intents;
        const bool do_step = ui_state.playing || ui_state.step_requested;
        const auto step_t0 = Clock::now();
        last_step_healthy = sim.FramePublish(&intents, do_step);
        const double step_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - step_t0).count();

        // Apply queue-dispatched control intents (out-of-band producers) on top of
        // the in-UI transport edits.
        if (intents.toggle_play) ui_state.playing = !ui_state.playing;
        if (intents.set_play) ui_state.playing = intents.play_value;
        if (intents.reset) ui_state.reset_requested = true;
        if (intents.camera_reset) ui_state.camera_reset = true;
        if (intents.set_env) {
            // Honor the documented PosePublisher wrap contract at the call site:
            // clamp the requested env into [0, env_count) so an out-of-band SetEnv
            // never selects past the world's env range (env_count==1 today -> the
            // value is always pinned to 0).
            const uint32_t env_count =
                (caps.env_count > 0u) ? caps.env_count : 1u;
            const uint32_t env = intents.env_value % env_count;
            ui_state.env_index = env;
            sim.SetEnvIndex(env);
        }
        // VIEW-6: the in-UI Step button (ui_state.step_requested) was consumed by
        // `do_step` this frame -> clear the one-shot edge. An OUT-OF-BAND queued
        // StepOnce is now honored SAME-FRAME inside FramePublish (the queue is
        // drained before the step decision), so it must NOT also be folded into the
        // next frame (that would double-step). Just clear the transport edge.
        ui_state.step_requested = false;

        // Reset request: re-frame the camera (a full re-cook is viewer-policy / M11;
        // here Reset re-frames + clears velocity-free, the cheap honest action).
        if (ui_state.reset_requested) {
            nuka::math::Vec3 lo, hi;
            SceneAabb(sim.GetRenderWorld(), &lo, &hi);
            camera.FrameAabb(lo, hi);
            ui_state.reset_requested = false;
        }
        if (ui_state.camera_reset) {
            nuka::math::Vec3 lo, hi;
            SceneAabb(sim.GetRenderWorld(), &lo, &hi);
            camera.FrameAabb(lo, hi);
            ui_state.camera_reset = false;
        }

        // -- stats for the UI ---------------------------------------------------
        viewer::ViewerStats stats;
        stats.step_time_ms  = static_cast<float>(step_ms);
        stats.fps           = static_cast<float>(fps_ema);
        stats.sub_steps     = 1u;
        stats.dof           = caps.dofs_per_env;
        stats.links         = caps.links_per_env;
        stats.bodies        = caps.bodies_per_env;
        stats.contact_cap   = caps.max_contacts_per_env;
        stats.draw_calls    = sim.GetRenderWorld().InstanceCount();
        stats.frame_index   = frame_index;
        stats.step_healthy  = last_step_healthy;
        stats.device_name   = present->DeviceName();

        // -- ImGui frame: set display size/dt, record panels, render draw data --
        io.DisplaySize = ImVec2(static_cast<float>(present->Report().width),
                                static_cast<float>(present->Report().height));
        io.DeltaTime = (dt > 0.0) ? static_cast<float>(dt) : (1.0f / 60.0f);
#ifdef _WIN32
        // Pull glfw input into io before ImGui::NewFrame (must precede it).
        if (native_window != nullptr) ImGui_ImplGlfw_NewFrame();
#endif
        imgui.NewFrame();
        ui.RecordUi(sim.GetRenderWorld(), stats, camera, ui_state);
        ImGui::Render();

        // VIEW-2: upload any drive sliders that moved this frame into the LIVE nk
        // Data (FieldId::DriveTarget, env-major: env*links_per_env + dof). Per-DOF
        // upload of ONLY the changed rows; the general write path (UploadField),
        // never a choreography table. R13: runtime Data only, never the .nks.
        if (caps.links_per_env > 0u &&
            ui_state.drive_dirty.size() == ui_state.drive_targets.size()) {
            const uint32_t per_env = caps.links_per_env;
            const uint32_t env =
                (caps.env_count > 0u) ? (ui_state.env_index % caps.env_count) : 0u;
            for (uint32_t d = 0; d < ui_state.drive_targets.size(); ++d) {
                if (!ui_state.drive_dirty[d]) continue;
                const uint64_t at =
                    (static_cast<uint64_t>(env) * per_env + d) * sizeof(float);
                sim.GetWorld().GetData().UploadField(
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
            sim.GetRenderWorld(), opts,
            [&imgui](void* cmd) { imgui.RenderDrawData(
                reinterpret_cast<NukaVkCommandBuffer>(cmd)); });
        if (r == render::PresentFrameResult::Error) {
            std::fprintf(stderr, "[nuka_viewer] present error at frame %llu\n",
                         static_cast<unsigned long long>(frame_index));
            break;
        }
        if (r == render::PresentFrameResult::Recreated) {
            // VIEW-5: the swapchain (and its present render pass) was rebuilt on
            // OUT_OF_DATE/SUBOPTIMAL (resize). The ImGui Vulkan backend's pipeline
            // is now bound to the STALE pass -> rebind it to the renderer's CURRENT
            // present pass + image counts so the overlay keeps drawing. The ImGui
            // context (docking layout / panels) survives the rebind.
            render::RendererVulkanHandles rh = present->VulkanHandles();
            nuka::render::imgui::NukaImGuiInitInfo rinfo;
            rinfo.api_version     = rh.api_version;
            rinfo.instance        = reinterpret_cast<NukaVkInstance>(rh.instance);
            rinfo.physical_device = reinterpret_cast<NukaVkPhysicalDevice>(rh.physical_device);
            rinfo.device          = reinterpret_cast<NukaVkDevice>(rh.device);
            rinfo.queue_family    = rh.graphics_family;
            rinfo.queue           = reinterpret_cast<NukaVkQueue>(rh.graphics_queue);
            rinfo.descriptor_pool = reinterpret_cast<NukaVkDescriptorPool>(rh.imgui_descriptor_pool);
            rinfo.min_image_count = present->MinImageCount();
            rinfo.image_count     = present->SwapchainImageCount();
            rinfo.render_pass     = reinterpret_cast<NukaVkRenderPass>(present->PresentRenderPass());
            rinfo.subpass         = 0u;
            if (!imgui.RebuildForRenderPass(rinfo)) {
                std::fprintf(stderr, "[nuka_viewer] ImGui rebind after swapchain "
                             "recreate failed at frame %llu\n",
                             static_cast<unsigned long long>(frame_index));
                break;
            }
        }
        if (r == render::PresentFrameResult::Presented) ++presented;

        ++frame_index;

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

    std::printf("[nuka_viewer] done: frames=%llu presented=%d last_step_healthy=%s\n",
                static_cast<unsigned long long>(frame_index), presented,
                last_step_healthy ? "yes" : "no");
    return 0;
}
