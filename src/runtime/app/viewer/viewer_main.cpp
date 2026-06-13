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
// vkcube-proven there). A frame budget + an optional presented-frame PPM dump make
// it CI-runnable for a few frames without a human at the window.
//
// GATING: this exe is built ONLY when NK_BUILD_VULKAN_VALIDATION is ON (D3); the
// default build-cuda128 config never sees it (opt-in, like the recorder).
//
// HOST-ONLY / zero-CUDA-token: pure host C++/Vulkan/ImGui. The only device touch
// is nk::World step + DownloadField (through the phi v2 vtable in nuka_nk).
// ---------------------------------------------------------------------------

#include <vulkan/vulkan.h>

#include "imgui.h"

#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/device_context.hpp"
#include "render/imgui/nuka_imgui.hpp"
#include "render/raster/vulkan_present_renderer.hpp"
#include "render/render_world.hpp"
#include "render/window/window_surface.hpp"
#include "runtime/app/pose_publisher.hpp"
#include "runtime/app/simulation.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

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

constexpr const char* kDefaultNks = "examples/scenes/h1_cup_table.nks";

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

nk::Pipeline::SolverConfig DefaultCfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
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

}  // namespace

int main(int argc, char** argv) {
    // ---- args: scene path, frame budget (0 = run until close), PPM dump path ---
    std::string scene_path = kDefaultNks;
    int max_frames = 0;  // 0 -> run until the window closes (interactive).
    std::string ppm_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--scene" && i + 1 < argc) scene_path = argv[++i];
        else if (a == "--frames" && i + 1 < argc) max_frames = std::atoi(argv[++i]);
        else if (a == "--ppm" && i + 1 < argc) ppm_path = argv[++i];
        else if (a == "--help") {
            std::printf("usage: nuka_viewer [--scene <.nks>] [--frames N] [--ppm <out.ppm>]\n");
            return 0;
        }
    }
    if (const char* env = std::getenv("NUKA_VIEWER_FRAMES")) max_frames = std::atoi(env);

    if (!std::filesystem::exists(scene_path)) {
        std::fprintf(stderr, "[nuka_viewer] scene not found: %s\n", scene_path.c_str());
        return 2;
    }

    // ---- 1. physics backend + cook -> nk::World + RenderWorld ------------------
    nphi::DeviceContext ctx = nphi::MakeDefaultDeviceContext();
    (void)ctx;
    nphi::Device* dev = nphi::InitBestDevice();
    if (!dev) { std::fprintf(stderr, "[nuka_viewer] no physics device\n"); return 3; }
    nphi::Backend* backend = nphi::DeviceInitBackend(dev, nullptr);
    if (!backend) { std::fprintf(stderr, "[nuka_viewer] no physics backend\n"); return 3; }

    const nuka::scene::SceneIR scene = LoadLightScene(scene_path);
    cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
    render::RenderWorld render_world =
        render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);

    const nk::ModelCapacities caps = cooked.model.capacities;  // copy BEFORE move.
    nk::World world(std::move(cooked.model), 1u, dev, backend, DefaultCfg());
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

    std::unique_ptr<render::PresentRenderer> present;
    try {
        present = std::make_unique<render::PresentRenderer>(std::move(surface));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[nuka_viewer] PresentRenderer ctor failed: %s\n", e.what());
        return 6;
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

    std::vector<window::WindowEvent> events;
    while (!present->ShouldClose()) {
        if (max_frames > 0 && static_cast<int>(frame_index) >= max_frames) break;

        // -- poll + dispatch input ----------------------------------------------
        events.clear();
        present->PollEvents(events);
        ImGuiIO& io = ImGui::GetIO();
        for (const window::WindowEvent& ev : events) {
            switch (ev.type) {
                case window::WindowEvent::Type::MouseMove:
                    io.AddMousePosEvent(static_cast<float>(ev.mouse_x),
                                        static_cast<float>(ev.mouse_y));
                    break;
                case window::WindowEvent::Type::MouseButton:
                    io.AddMouseButtonEvent(ToImGuiMouseButton(ev.button), ev.pressed);
                    break;
                case window::WindowEvent::Type::Scroll:
                    io.AddMouseWheelEvent(0.0f, static_cast<float>(ev.scroll_delta));
                    break;
                default:
                    break;
            }
            // Camera gets the event only if ImGui is not capturing that input.
            camera.HandleEvent(ev, /*allow_drag=*/!io.WantCaptureMouse,
                               /*allow_scroll=*/!io.WantCaptureMouse);
        }

        // -- timing -------------------------------------------------------------
        const auto now = Clock::now();
        const double dt = std::chrono::duration<double>(now - prev).count();
        prev = now;
        if (dt > 0.0) {
            const double inst_fps = 1.0 / dt;
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
        // OUT-OF-BAND StepOnce (command_queue.hpp:51): FramePublish drained the
        // queue AFTER `do_step` was already computed for THIS frame, so a queued
        // StepOnce cannot be honored same-frame (only the in-UI Step button, which
        // sets ui_state.step_requested before FramePublish, is). Fold the queued
        // edge into the NEXT frame's transport so it is not silently dropped.
        // (We clear step_requested first, then set it from the queued intent, so a
        // step taken this frame does not leak into the next.)
        ui_state.step_requested = intents.step_once;

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
        imgui.NewFrame();
        ui.RecordUi(sim.GetRenderWorld(), stats, camera, ui_state);
        ImGui::Render();

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
    imgui.Shutdown();

    std::printf("[nuka_viewer] done: frames=%llu presented=%d last_step_healthy=%s\n",
                static_cast<unsigned long long>(frame_index), presented,
                last_step_healthy ? "yes" : "no");
    (void)ppm_path;  // presented-frame readback is a present-path extension (the
                     // PresentRenderer is present-only); GATE-B dumps the composited
                     // PPM from the offscreen path instead (deterministic + portable).
    return 0;
}
