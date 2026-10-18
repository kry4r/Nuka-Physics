#include "runtime/app/viewer/editor_host.hpp"

#include <vulkan/vulkan.h>

#include "imgui.h"
#include "TextEditor.h"
// The SDL3 platform backend feeds ImGui input, IME and cursors; imgui_impl_vulkan draws.
#include "backends/imgui_impl_sdl3.h"
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

#include "nuka/nuka_scene.h"
#include "render/imgui/nuka_imgui.hpp"
#include "render/raster/interop_transform_ssbo.hpp"
#include "render/raster/vulkan_present_renderer.hpp"
#include "render/render_world.hpp"
#include "render/viewer_resources.hpp"
#include "render/window/window_surface.hpp"
#include "runtime/app/viewer/camera_controller.hpp"
#include "runtime/app/viewer/debug_overlay.hpp"
#include "runtime/app/viewer/editor_actions.hpp"
#include "runtime/app/viewer/editor_assets.hpp"
#include "runtime/app/viewer/editor_collision.hpp"
#include "runtime/app/viewer/editor_document.hpp"
#include "runtime/app/viewer/editor_edits.hpp"
#include "runtime/app/viewer/editor_extensions.hpp"
#include "runtime/app/viewer/editor_import.hpp"
#include "runtime/app/viewer/editor_instance.hpp"
#include "runtime/app/viewer/editor_profile.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "runtime/app/viewer/entity_drag.hpp"
#include "runtime/app/viewer/file_dialog.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"
#include "runtime/app/viewer/physics_worker.hpp"
#include "runtime/app/viewer/pose_scatter.hpp"
#include "runtime/app/viewer/property_schema.hpp"
#include "runtime/app/viewer/robot_tools.hpp"
#include "runtime/app/viewer/script_host.hpp"
#include "runtime/app/viewer/teleop.hpp"
#include "runtime/inference/go2_policy_controller.hpp"
#include "scene/format/json.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb/stb_image_write.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace nuka::runtime::app::viewer {
namespace {

namespace window = nuka::render::window;
namespace json = nuka::scene::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr float kDefaultDt = 1.0f / 240.0f;
// ImGui's frame interval clamps to this so a hitch (load, resize, debugger) never jumps time.
constexpr double kMaxFrameDt = 0.25;
constexpr size_t kConsoleLogCap = 256u * 1024u;
constexpr size_t kScriptRunsKept = 64u;
// A spawned primitive starts this far above its reference so a free body drops in.
constexpr float kSpawnDropHeight = 0.75f;
constexpr uint64_t kFpsWarmup = 30u;
// Frame timings keep this many recent samples for the exit percentiles.
constexpr size_t kSampleCap = 65536u;
// Pose slots beyond the frames in flight: the latest, the one being written, the one pinned.
constexpr uint32_t kPoseSlotsBeyondFrames = 3u;
// A pace ceiling against busy-spinning; FIFO present caps at the display refresh.
constexpr double kFrameBudgetSeconds = 1.0 / 240.0;
constexpr float kPi = 3.14159265358979323846f;
constexpr double kAutosaveSeconds = 60.0;
// Logs beyond this many, oldest first, are deleted at startup.
constexpr size_t kLogKeep = 20u;
// A press and release closer than this many pixels is a click, not a drag.
constexpr float kClickSlop = 4.0f;
// A grab spring at unit gain rings at this rate on what it holds, critically damped.
constexpr float kGrabHz = 3.0f;
// The viewport draws a plain held force as an arrow this long [m].
constexpr float kForceArrow = 0.3f;
// The most environments sim.envs builds.
constexpr uint32_t kMaxEnvs = 4096u;
// The RT preview renders the viewport scaled down to this many pixels on its long side.
constexpr float kRtMaxSide = 1920.0f;
// A gizmo move's locked orientation weighs this much against its position, so a short chain
// still reaches the point.
constexpr float kIkLockWeight = 0.1f;
// A gizmo IK solve this far from its goal draws the miss.
constexpr float kIkMissMm = 1.0f;
constexpr float kIkMissDeg = 0.5f;
// A Measure click this many pixels from a drawn joint origin takes the joint.
constexpr float kJointPickPixels = 8.0f;

struct LaunchOptions {
    std::string scene_path;
    bool empty_requested = false;
    bool offscreen = false;
    int max_frames = 0;  // 0 runs until the window closes
    float dt = kDefaultDt;
    bool dt_set = false;
    int capture_frame = -1;
    std::string capture_out = "nuka_present_capture.ppm";
    std::string policy_path;
    bool want_play = false;
    bool want_planned = false;
    bool force_host_policy = false;
    bool cam_on = false;
    math::Vec3 cam_target{};
    float cam_dist = 0.0f, cam_yaw = 0.0f, cam_pitch = 0.0f;
    std::string cams_path;  // camera bookmarks for the first scene opened
    std::string select_path;
    bool edit_color_on = false;
    float edit_color[3] = {0.0f, 0.0f, 0.0f};
    bool edit_move_on = false;
    float edit_move[3] = {0.0f, 0.0f, 0.0f};
    std::string save_to;
    std::string gizmo_op;
    bool gizmo_local = false;
    std::string run_script_path;
    std::string rename_to;
    bool add_box_on = false;
    std::string spawn_kind;
    bool delete_on = false;
    std::string reparent_to;
    bool undo = false;
    bool redo = false;
    int reset_at_frame = -1;
    bool show_colliders = false;
    bool show_contacts = false;
    bool teleop_nudge_on = false;
    int teleop_nudge_dof = 0;
    float teleop_nudge_delta = 0.0f;
    bool teleop_cmd_on = false;
    float teleop_cmd[3] = {0.0f, 0.0f, 0.0f};
    float speed = 1.0f;
    std::string publish = "auto";
    bool perturb_on = false;
    float perturb[3] = {0.0f, 0.0f, 0.0f};
    std::string layout;
    float ui_scale = 1.0f;
    std::string locale;
    std::vector<std::string> ext_ids;
    std::vector<std::string> ext_folders;
    std::string profile;
    std::string log_path;
    std::vector<std::string> relaunch;  // the command that starts another editor of this kind
    std::string new_template;
    std::string script_name;
    bool primary = false;  // listen for the files later launches hand over
};

#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
std::string Format(const char* format, ...) {
    va_list args;
    va_list sizing;
    va_start(args, format);
    va_copy(sizing, args);
    const int n = std::vsnprintf(nullptr, 0, format, sizing);
    va_end(sizing);
    std::string out(static_cast<size_t>(n), '\0');
    std::vsnprintf(out.data(), out.size() + 1u, format, args);
    va_end(args);
    return out;
}

bool ReadEntireFile(const std::string& path, std::string* out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n < 0) { std::fclose(f); return false; }
    std::string buf(static_cast<size_t>(n), '\0');
    const size_t rd = (n > 0) ? std::fread(&buf[0], 1, static_cast<size_t>(n), f) : 0u;
    std::fclose(f);
    buf.resize(rd);
    *out = std::move(buf);
    return true;
}

std::string FileName(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1u);
}

// A file of the editor's resources beside the library, else in the source tree.
std::string Resource(const std::string& relative) {
    return render::ViewerResource(relative.c_str(), (std::string(NUKA_EDITOR_SOURCE_DIR) + "/" + relative).c_str());
}

// The note beside an autosave: the document's own path, the writing process and the time.
fs::path SidecarOf(const std::string& autosave) { return Utf8Path(autosave).replace_extension(".json"); }

// Crops the largest centered 16:9 window out of a captured viewport and box-filters it to
// kThumbWidth x kThumbHeight.
void WriteThumbnail(const std::string& capture, const fs::path& out) {
    std::ifstream in(Utf8Path(capture), std::ios::binary);
    std::string magic;
    int width = 0, height = 0, depth = 0;
    in >> magic >> width >> height >> depth;
    in.get();
    std::vector<unsigned char> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u);
    in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    int cw = width, ch = width * kThumbHeight / kThumbWidth;
    if (ch > height) {
        ch = height;
        cw = height * kThumbWidth / kThumbHeight;
    }
    const int x0 = (width - cw) / 2;
    const int y0 = (height - ch) / 2;
    std::string pixels(static_cast<size_t>(kThumbWidth * kThumbHeight * 3), '\0');
    for (int ty = 0; ty < kThumbHeight; ++ty) {
        const int sy0 = y0 + ty * ch / kThumbHeight, sy1 = std::max(sy0 + 1, y0 + (ty + 1) * ch / kThumbHeight);
        for (int tx = 0; tx < kThumbWidth; ++tx) {
            const int sx0 = x0 + tx * cw / kThumbWidth, sx1 = std::max(sx0 + 1, x0 + (tx + 1) * cw / kThumbWidth);
            unsigned sum[3] = {0u, 0u, 0u};
            for (int sy = sy0; sy < sy1; ++sy)
                for (int sx = sx0; sx < sx1; ++sx)
                    for (int c = 0; c < 3; ++c)
                        sum[c] += rgb[(static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx)) * 3u +
                                      static_cast<size_t>(c)];
            const unsigned n = static_cast<unsigned>((sy1 - sy0) * (sx1 - sx0));
            for (int c = 0; c < 3; ++c)
                pixels[static_cast<size_t>((ty * kThumbWidth + tx) * 3 + c)] = static_cast<char>(sum[c] / n);
        }
    }
    std::ofstream file(out, std::ios::binary | std::ios::trunc);
    file << "P6\n" << kThumbWidth << " " << kThumbHeight << "\n255\n" << pixels;
}

void PrintUsage() {
    std::printf("usage: nuka-editor [scene.nks] [--policy <bin>] [--play] [--frames N] [--dt SECONDS]\n"
                "  [--offscreen] [--cam tx,ty,tz,dist,yaw_deg,pitch_deg]\n"
                "  [--cams FILE.json loads camera bookmarks for the scene it opens]\n"
                "  [--capture-frame N --capture-out FILE.ppm]\n"
                "  [--run-script FILE.py runs a script once against the loaded scene]\n"
                "  [--undo / --redo fire one undo/redo after a headless load edit]\n"
                "  [--spawn box|sphere|capsule|plane spawns a primitive body after load]\n"
                "  [--reset-at N fires transport Reset (re-seed physics) at frame N]\n"
                "  [--show-colliders / --show-contacts enable the read-only debug overlays]\n"
                "  [--teleop-nudge DOF,DELTA PD-holds the robot + nudges one drive target]\n"
                "  [--teleop-cmd VX,VY,WYAW feeds a command to a command-consuming controller]\n"
                "  [--speed X sets the transport speed; --publish auto|host|zero-copy picks the pose path]\n"
                "  [--perturb FX,FY,FZ holds a world force [N] on the --select entity's link or body]\n"
                "  [--layout authoring|debug|robot|data|training picks the panel layout; --ui-scale X]\n"
                "  [--locale en|zh-CN picks the UI language]\n"
                "  [--ext ID enables an installed extension; --ext-folder DIR adds a search folder]\n"
                "  [--profile DIR holds settings, recent scenes, logs and autosaves; --log FILE]\n"
                "  [--template empty|ground starts a new scene; --new-window skips the open editor]\n"
                "  [--console shows a console window for the application's output]\n"
                "  no scene -> the start page; --empty opens a blank workspace\n"
                "  --offscreen renders to a display-less surface\n"
                "  --policy attaches a trained locomotion controller to each load\n"
                "  --host-policy runs the host MLP round-trip instead of the GPU path (A/B)\n"
                "  --cam overrides the auto-frame (default: fit the moving-instance cluster)\n");
}

// False after printing the usage for --help.
bool ParseOptions(const std::vector<std::string>& args, LaunchOptions* opt) {
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const bool has_value = i + 1 < args.size();
        if (a == "--scene" && has_value) opt->scene_path = args[++i];
        else if (a == "--empty") opt->empty_requested = true;
        else if (a == "--offscreen") opt->offscreen = true;
        else if (a == "--frames" && has_value) opt->max_frames = std::atoi(args[++i].c_str());
        else if (a == "--dt" && has_value) {
            opt->dt = static_cast<float>(std::atof(args[++i].c_str()));
            opt->dt_set = true;
        }
        else if (a == "--capture-frame" && has_value) opt->capture_frame = std::atoi(args[++i].c_str());
        else if (a == "--capture-out" && has_value) opt->capture_out = args[++i];
        else if (a == "--policy" && has_value) opt->policy_path = args[++i];
        else if (a == "--play") opt->want_play = true;
        else if (a == "--planned") opt->want_planned = true;
        else if (a == "--host-policy") opt->force_host_policy = true;
        else if (a == "--cam" && has_value) {
            float yaw_deg = 0.0f, pitch_deg = 0.0f;
            if (std::sscanf(args[++i].c_str(), "%f,%f,%f,%f,%f,%f", &opt->cam_target.x,
                            &opt->cam_target.y, &opt->cam_target.z, &opt->cam_dist, &yaw_deg,
                            &pitch_deg) == 6) {
                opt->cam_yaw = yaw_deg * kPi / 180.0f;
                opt->cam_pitch = pitch_deg * kPi / 180.0f;
                opt->cam_on = true;
            } else {
                std::fprintf(stderr, "[nuka_editor] --cam wants tx,ty,tz,dist,yaw_deg,pitch_deg\n");
            }
        }
        else if (a == "--cams" && has_value) opt->cams_path = args[++i];
        else if (a == "--select" && has_value) opt->select_path = args[++i];
        else if (a == "--edit-color" && has_value) {
            opt->edit_color_on = std::sscanf(args[++i].c_str(), "%f,%f,%f", &opt->edit_color[0],
                                             &opt->edit_color[1], &opt->edit_color[2]) == 3;
        }
        else if (a == "--edit-move" && has_value) {
            opt->edit_move_on = std::sscanf(args[++i].c_str(), "%f,%f,%f", &opt->edit_move[0],
                                            &opt->edit_move[1], &opt->edit_move[2]) == 3;
        }
        else if (a == "--save-to" && has_value) opt->save_to = args[++i];
        else if (a == "--gizmo-op" && has_value) opt->gizmo_op = args[++i];
        else if (a == "--gizmo-local") opt->gizmo_local = true;
        else if (a == "--run-script" && has_value) opt->run_script_path = args[++i];
        else if (a == "--rename" && has_value) opt->rename_to = args[++i];
        else if (a == "--add-box") opt->add_box_on = true;
        else if (a == "--spawn" && has_value) opt->spawn_kind = args[++i];
        else if (a == "--delete") opt->delete_on = true;
        else if (a == "--reparent" && has_value) opt->reparent_to = args[++i];
        else if (a == "--undo") opt->undo = true;
        else if (a == "--redo") opt->redo = true;
        else if (a == "--reset-at" && has_value) opt->reset_at_frame = std::atoi(args[++i].c_str());
        else if (a == "--show-colliders") opt->show_colliders = true;
        else if (a == "--show-contacts") opt->show_contacts = true;
        else if (a == "--teleop-nudge" && has_value) {
            opt->teleop_nudge_on = std::sscanf(args[++i].c_str(), "%d,%f", &opt->teleop_nudge_dof,
                                               &opt->teleop_nudge_delta) == 2;
            if (!opt->teleop_nudge_on) std::fprintf(stderr, "[nuka_editor] --teleop-nudge wants DOF,DELTA\n");
        }
        else if (a == "--teleop-cmd" && has_value) {
            opt->teleop_cmd_on = std::sscanf(args[++i].c_str(), "%f,%f,%f", &opt->teleop_cmd[0],
                                             &opt->teleop_cmd[1], &opt->teleop_cmd[2]) == 3;
            if (!opt->teleop_cmd_on) std::fprintf(stderr, "[nuka_editor] --teleop-cmd wants VX,VY,WYAW\n");
        }
        else if (a == "--speed" && has_value) opt->speed = static_cast<float>(std::atof(args[++i].c_str()));
        else if (a == "--publish" && has_value) opt->publish = args[++i];
        else if (a == "--perturb" && has_value) {
            opt->perturb_on = std::sscanf(args[++i].c_str(), "%f,%f,%f", &opt->perturb[0], &opt->perturb[1],
                                          &opt->perturb[2]) == 3;
            if (!opt->perturb_on) std::fprintf(stderr, "[nuka_editor] --perturb wants FX,FY,FZ\n");
        }
        else if (a == "--layout" && has_value) opt->layout = args[++i];
        else if (a == "--ui-scale" && has_value) opt->ui_scale = static_cast<float>(std::atof(args[++i].c_str()));
        else if (a == "--locale" && has_value) opt->locale = args[++i];
        else if (a == "--ext" && has_value) opt->ext_ids.push_back(args[++i]);
        else if (a == "--ext-folder" && has_value) opt->ext_folders.push_back(args[++i]);
        else if (a == "--profile" && has_value) opt->profile = args[++i];
        else if (a == "--log" && has_value) opt->log_path = args[++i];
        else if (a == "--template" && has_value) opt->new_template = args[++i];
        else if (a == "--script-name" && has_value) opt->script_name = args[++i];
        else if (a == "--primary") opt->primary = true;
        else if (a == "--relaunch" && has_value) {
            const json::Value command = json::Value::Parse(args[++i]);
            for (const json::Value& part : command.Elements()) opt->relaunch.push_back(part.AsString());
        }
        else if (a == "--help") {
            PrintUsage();
            return false;
        }
        else if (a.rfind("--", 0) != 0) opt->scene_path = a;
    }
    return true;
}

// An IK goal for the origin of `link`: `goal`'s position, its rotation as well when `orient`.
nuka_ik_target_t IkGoal(uint32_t link, const math::Transform& goal, bool orient, float weight) {
    nuka_ik_target_t t{};
    t.link = link;
    t.flags = orient ? static_cast<uint32_t>(NUKA_IK_ORIENT) : 0u;
    const float p[3] = {goal.position.x, goal.position.y, goal.position.z};
    const float r[4] = {goal.rotation.w, goal.rotation.x, goal.rotation.y, goal.rotation.z};
    std::copy(p, p + 3, t.position);
    std::copy(r, r + 4, t.rotation);
    t.weight = weight;
    return t;
}

// An IK solve of `targets` in environment `env` below link `root` (-1 for each target's
// articulation root) with the core's iteration settings; it writes what the transport does with a drag.
TickCommand IkCommand(std::vector<nuka_ik_target_t> targets, int root, float rotation_weight, uint32_t env) {
    TickCommand cmd;
    cmd.kind = CommandKind::IkSolve;
    cmd.env = env;
    cmd.ik.root = root < 0 ? ~uint32_t{0} : static_cast<uint32_t>(root);
    cmd.ik.iterations = 200u;
    cmd.ik.rotation_weight = rotation_weight;
    cmd.ik.damping = 1.0e-3f;
    cmd.ik.max_step = 0.2f;
    cmd.ik.position_tolerance = 1.0e-5f;
    cmd.ik.rotation_tolerance = 1.0e-4f;
    cmd.ik_targets = std::move(targets);
    return cmd;
}

void ClearModelUi(ViewerUiState& ui) {
    ui.drive_targets.clear();
    ui.drive_dirty.clear();
    ui.robot.joints.clear();
    ui.robot.roots.clear();
    ui.selected_entity = scene::kInvalidEntity;
    ui.playing = false;
}

// The world AABB of the rendered instances; with `movable_only`, of the physics-driven
// cluster when one exists, so robots on a large terrain fill the view.
void SceneAabb(const render::RenderWorld& world, bool movable_only, math::Vec3* lo, math::Vec3* hi) {
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
        if (restrict_movable && inst.pose_source.kind == render::PoseSource::Kind::Static) continue;
        const render::MeshGeometry& geo = world.meshes.Geometry(inst.mesh_id);
        for (size_t v = 0; v + 2 < geo.positions.size(); v += 3) {
            const math::Vec3 local{geo.positions[v], geo.positions[v + 1], geo.positions[v + 2]};
            const math::Vec3 wp = inst.world_xform.TransformPoint(local);
            lo->x = std::min(lo->x, wp.x); lo->y = std::min(lo->y, wp.y); lo->z = std::min(lo->z, wp.z);
            hi->x = std::max(hi->x, wp.x); hi->y = std::max(hi->y, wp.y); hi->z = std::max(hi->z, wp.z);
            any = true;
        }
    }
    if (!any) { *lo = {-1.0f, -1.0f, 0.0f}; *hi = {1.0f, 1.0f, 1.0f}; }
}

// World AABB of one instance; empty (lo > hi) without geometry.
void InstanceAabb(const render::RenderInstance& inst, const render::RenderWorld& w,
                  math::Vec3* lo, math::Vec3* hi) {
    const float fmax = std::numeric_limits<float>::max();
    *lo = {fmax, fmax, fmax};
    *hi = {-fmax, -fmax, -fmax};
    if (inst.mesh_id == render::kNoId || inst.mesh_id >= w.meshes.Count()) return;
    const render::MeshGeometry& geo = w.meshes.Geometry(inst.mesh_id);
    for (size_t v = 0; v + 2 < geo.positions.size(); v += 3) {
        const math::Vec3 local{geo.positions[v], geo.positions[v + 1], geo.positions[v + 2]};
        const math::Vec3 p = inst.world_xform.TransformPoint(local);
        lo->x = std::min(lo->x, p.x); lo->y = std::min(lo->y, p.y); lo->z = std::min(lo->z, p.z);
        hi->x = std::max(hi->x, p.x); hi->y = std::max(hi->y, p.y); hi->z = std::max(hi->z, p.z);
    }
}

// Slab test; the near hit distance t >= 0 lands in *t_out.
bool RayAabb(const Ray& ray, const math::Vec3& lo, const math::Vec3& hi, float* t_out) {
    float tmin = 0.0f;
    float tmax = std::numeric_limits<float>::max();
    const float o[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    const float d[3] = {ray.dir.x, ray.dir.y, ray.dir.z};
    const float lo3[3] = {lo.x, lo.y, lo.z};
    const float hi3[3] = {hi.x, hi.y, hi.z};
    for (int a = 0; a < 3; ++a) {
        if (std::fabs(d[a]) < 1e-8f) {
            if (o[a] < lo3[a] || o[a] > hi3[a]) return false;
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
    *t_out = tmin;
    return true;
}

// Where the ray from `o` along `d` crosses triangle abc from either face: its distance and the
// barycentric weights of b and c.
bool RayTriangle(const math::Vec3& o, const math::Vec3& d, const math::Vec3& a, const math::Vec3& b,
                 const math::Vec3& c, float* t, float* u, float* v) {
    const math::Vec3 e1 = b - a;
    const math::Vec3 e2 = c - a;
    const math::Vec3 p = d.Cross(e2);
    const float det = e1.Dot(p);
    if (std::fabs(det) < 1e-12f) return false;
    const math::Vec3 s = o - a;
    *u = s.Dot(p) / det;
    if (*u < 0.0f || *u > 1.0f) return false;
    const math::Vec3 q = s.Cross(e1);
    *v = d.Dot(q) / det;
    if (*v < 0.0f || *u + *v > 1.0f) return false;
    *t = e2.Dot(q) / det;
    return *t >= 0.0f;
}

// The nearest triangle of the instance's mesh along the ray, both faces; t lands in *t_out.
bool RayMesh(const Ray& ray, const render::RenderInstance& inst, const render::RenderWorld& w, float* t_out) {
    const render::MeshGeometry& geo = w.meshes.Geometry(inst.mesh_id);
    const math::Transform inv = inst.world_xform.Inverse();
    const math::Vec3 o = inv.TransformPoint(ray.origin);
    const math::Vec3 d = inv.TransformDirection(ray.dir);
    const auto at = [&](uint32_t i) {
        return math::Vec3{geo.positions[3u * i], geo.positions[3u * i + 1u], geo.positions[3u * i + 2u]};
    };
    bool hit = false;
    for (size_t k = 0; k + 2u < geo.indices.size(); k += 3u) {
        float t = 0.0f, u = 0.0f, v = 0.0f;
        if (RayTriangle(o, d, at(geo.indices[k]), at(geo.indices[k + 1u]), at(geo.indices[k + 2u]), &t, &u, &v) &&
            (!hit || t < *t_out)) {
            *t_out = t;
            hit = true;
        }
    }
    return hit;
}

// The nearest instance among those `pickable` accepts whose triangles the ray hits, or ~0u;
// `t_out` gets its distance.
uint32_t PickInstance(const Ray& ray, const render::RenderWorld& w,
                      const std::function<bool(const render::RenderInstance&)>& pickable, float* t_out = nullptr) {
    uint32_t best = ~0u;
    float best_t = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < w.InstanceCount(); ++i) {
        const render::RenderInstance& inst = w.instances[i];
        if (!pickable(inst)) continue;
        math::Vec3 lo, hi;
        InstanceAabb(inst, w, &lo, &hi);
        float t = 0.0f;
        if (lo.x > hi.x || !RayAabb(ray, lo, hi, &t) || t >= best_t) continue;
        if (RayMesh(ray, inst, w, &t) && t < best_t) {
            best_t = t;
            best = i;
        }
    }
    if (t_out != nullptr) *t_out = best_t;
    return best;
}

// The corner nearest the ray's first hit on a particle surface, as one environment's particle,
// or kNoParticle; `t_out` gets the hit's distance.
uint32_t PickParticle(const Ray& ray, const WorldLayout& layout, const std::vector<math::Vec3>& x, float* t_out) {
    uint32_t best = kNoParticle;
    *t_out = std::numeric_limits<float>::max();
    for (const ParticleSurfaceBinding& s : layout.surfaces) {
        for (size_t k = 0; k + 2u < s.triangles.size(); k += 3u) {
            const uint32_t i[3] = {s.triangles[k], s.triangles[k + 1u], s.triangles[k + 2u]};
            float t = 0.0f, u = 0.0f, v = 0.0f;
            if (!RayTriangle(ray.origin, ray.dir, x[i[0]], x[i[1]], x[i[2]], &t, &u, &v) || t >= *t_out) continue;
            *t_out = t;
            const float w0 = 1.0f - u - v;
            best = u >= w0 && u >= v ? i[1] : v >= w0 ? i[2] : i[0];
        }
    }
    return best;
}

// True when `path` lies strictly under `root`.
bool Under(const std::string& path, const std::string& root) {
    return path.size() > root.size() && path.compare(0, root.size(), root) == 0 && path[root.size()] == '/';
}

bool UnderAny(const std::string& path, const std::vector<std::string>& roots) {
    return std::any_of(roots.begin(), roots.end(), [&](const std::string& r) { return path == r || Under(path, r); });
}

// The body node carrying `e`, `e` itself when it is one or lies outside every body.
scene::EntityId CarrierOf(const scene::SceneIR& s, scene::EntityId e) {
    std::vector<scene::EntityId> bodies;
    for (scene::BodyId b = 0; b < s.RigidBodyCount(); ++b) bodies.push_back(s.EntityOfBody(b));
    for (auto n = s.Ecs().NodeOf(e); n; n = n->parent.lock())
        if (std::find(bodies.begin(), bodies.end(), n->entity) != bodies.end()) return n->entity;
    return e;
}

scene::EntityId EntityOfPath(const EditorScene& es, const std::string& path) {
    scene::EntityId found = scene::kInvalidEntity;
    const scene::SceneGraph& tree = es.scene.Tree();
    tree.Traverse(tree.Root(), [&](const std::shared_ptr<scene::SceneNode>& n) {
        if (found == scene::kInvalidEntity && tree.PathOf(n) == path) found = n->entity;
    });
    return found;
}

json::Value Object(std::initializer_list<std::pair<const char*, json::Value>> items) {
    json::Value v = json::Value::Object();
    for (const auto& [key, value] : items) v.Set(key, value);
    return v;
}

std::string StrArg(const json::Value& args, const char* key, const std::string& fallback = std::string()) {
    const json::Value* v = args.Find(key);
    return v != nullptr && !v->IsNull() ? v->AsString() : fallback;
}

double NumArg(const json::Value& args, const char* key, double fallback) {
    const json::Value* v = args.Find(key);
    return v != nullptr && !v->IsNull() ? v->AsDouble() : fallback;
}

bool BoolArg(const json::Value& args, const char* key, bool fallback) {
    const json::Value* v = args.Find(key);
    return v != nullptr && !v->IsNull() ? v->AsBool() : fallback;
}

// A three-number argument; false when `key` is absent.
bool Vec3Arg(const json::Value& args, const char* key, math::Vec3* out) {
    const json::Value* v = args.Find(key);
    if (v == nullptr || v->IsNull()) return false;
    if (v->Size() != 3u) throw std::runtime_error(std::string(key) + " wants three numbers");
    *out = math::Vec3{v->Elements()[0].AsFloat(), v->Elements()[1].AsFloat(), v->Elements()[2].AsFloat()};
    return true;
}

json::Value Vec3Json(const math::Vec3& p) {
    json::Value a = json::Value::Array();
    for (const float c : {p.x, p.y, p.z}) a.PushBack(json::Value::Float(c));
    return a;
}

// Where a held load acts in the snapshot's world; false when its row is missing.
bool LoadPoint(const Snapshot& snap, const Perturbation& p, math::Vec3* out) {
    math::Transform pose;
    if (p.particle != kNoParticle) *out = snap.particles[p.particle];
    else if (ResolvePose(p.source, snap.link_poses, snap.body_poses, &pose)) *out = pose.TransformPoint(p.point);
    else return false;
    return true;
}

std::optional<bool> OptBoolArg(const json::Value& args, const char* key) {
    const json::Value* v = args.Find(key);
    return v != nullptr && !v->IsNull() ? std::optional<bool>(v->AsBool()) : std::nullopt;
}

std::vector<std::string> StrListArg(const json::Value& args, const char* key) {
    std::vector<std::string> out;
    if (const json::Value* v = args.Find(key); v != nullptr && !v->IsNull())
        for (const json::Value& s : v->Elements()) out.push_back(s.AsString());
    return out;
}

json::Value StrList(const std::vector<std::string>& values) {
    json::Value list = json::Value::Array();
    for (const std::string& s : values) list.PushBack(json::Value::Str(s));
    return list;
}

ActionResult Ok(json::Value value = json::Value()) {
    ActionResult result;
    result.value = std::move(value);
    return result;
}

ActionResult Outcome(const ApplyResult& r) {
    return r.ok ? Ok(Object({{"focus", json::Value::Str(r.focus)}})) : ActionResult::Fail(r.error);
}

// The frameless window's hit test: a restored window's edges resize it and the title bar's empty
// spans move it; Windows then gives the bar double-click maximize and snapping.
SDL_HitTestResult SDLCALL TitleBarHitTest(SDL_Window* window, const SDL_Point* point, void* data) {
    const TitleBar& bar = *static_cast<const TitleBar*>(data);
    const float x = static_cast<float>(point->x), y = static_cast<float>(point->y);
    if (!bar.maximized) {
        int w = 0, h = 0;
        SDL_GetWindowSize(window, &w, &h);
        const bool left = x < bar.edge, right = x >= static_cast<float>(w) - bar.edge;
        if (y < bar.edge) return left ? SDL_HITTEST_RESIZE_TOPLEFT : right ? SDL_HITTEST_RESIZE_TOPRIGHT : SDL_HITTEST_RESIZE_TOP;
        if (y >= static_cast<float>(h) - bar.edge)
            return left ? SDL_HITTEST_RESIZE_BOTTOMLEFT : right ? SDL_HITTEST_RESIZE_BOTTOMRIGHT : SDL_HITTEST_RESIZE_BOTTOM;
        if (left) return SDL_HITTEST_RESIZE_LEFT;
        if (right) return SDL_HITTEST_RESIZE_RIGHT;
    }
    if (y < bar.height)
        for (const auto& [x0, x1] : bar.drag)
            if (x >= x0 && x < x1) return SDL_HITTEST_DRAGGABLE;
    return SDL_HITTEST_NORMAL;
}

// The Solver view keeps this many of the newest telemetry records.
constexpr size_t kSolverHistory = 2048u;
// The Joints view keeps this many of the newest joint samples.
constexpr size_t kJointHistory = 4096u;

// One telemetry record as the solver actions report it.
json::Value RecordJson(const nuka_telemetry_record_t& r, uint64_t number) {
    const auto kind = [](uint32_t k) {
        return json::Value::Str(k == NUKA_TELEMETRY_ELEMENT_PARTICLE ? "particle"
                                : k == NUKA_TELEMETRY_ELEMENT_ROW    ? "row"
                                                                     : "none");
    };
    json::Value sides = json::Value::Array();
    for (uint32_t s = 0; s < 2u && r.worst_kind == NUKA_TELEMETRY_ELEMENT_ROW; ++s)
        sides.PushBack(Object({{"kind", json::Value::Int(r.worst_side_kind[s])},
                               {"index", json::Value::Int(r.worst_side_index[s])}}));
    json::Value samples = json::Value::Array();
    for (uint32_t i = 0; i < std::min<uint32_t>(r.sample_count, NUKA_TELEMETRY_SAMPLES); ++i)
        samples.PushBack(Object({{"sweep", json::Value::Int(r.sample_sweep[i])},
                                 {"residual", json::Value::Float(r.sample_residual[i])}}));
    const bool energy = (r.flags & NUKA_TELEMETRY_ENERGY_VALID) != 0u;
    return Object({{"record", json::Value::Int(static_cast<int64_t>(number))},
                   {"substep", json::Value::Int(r.substep)},
                   {"sweeps", json::Value::Int(r.sweeps)},
                   {"sweep_budget", json::Value::Int(r.sweep_budget)},
                   {"capped", json::Value::Bool((r.flags & NUKA_TELEMETRY_CAP) != 0u)},
                   {"converged", json::Value::Bool((r.flags & NUKA_TELEMETRY_CONVERGED) != 0u)},
                   {"contacts", json::Value::Int(r.contacts)},
                   {"vertex_residual", json::Value::Float(r.vertex_residual)},
                   {"vertex_force", json::Value::Float(r.vertex_force)},
                   {"contact_residual", json::Value::Float(r.contact_residual)},
                   {"max_penetration", json::Value::Float(r.max_penetration)},
                   {"energy", energy ? json::Value::Float(r.energy) : json::Value::Null()},
                   {"energy_residual", energy ? json::Value::Float(r.energy_residual) : json::Value::Null()},
                   {"worst_kind", kind(r.worst_kind)},
                   {"worst_index", json::Value::Int(r.worst_index)},
                   {"worst_residual", json::Value::Float(r.worst_residual)},
                   {"sides", std::move(sides)},
                   {"samples", std::move(samples)}});
}

double Ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

uint64_t Fnv1a(const std::string& bytes, uint64_t hash = 14695981039346656037ull) {
    for (const char c : bytes) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string Hex(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

// What one RT preview image shows: the view, its pixel size and the snapshot it was asked at.
struct RtKey {
    nuka_beauty_camera_t camera{};
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint64_t sequence = 0u;
    bool SameView(const RtKey& o) const {
        return std::memcmp(&camera, &o.camera, sizeof(camera)) == 0 && width == o.width && height == o.height;
    }
    bool Same(const RtKey& o) const { return SameView(o) && sequence == o.sequence; }
};

// A named orbit view of a document; angles in degrees.
struct CameraBookmark {
    std::string name;
    math::Vec3 target = math::Vec3::Zero();
    float distance = 1.0f;
    float yaw = 0.0f;
    float pitch = 0.0f;
    float fov = 45.0f;
};

// A pane's view; the active pane's camera and mode live in the host's camera and view mode instead.
struct PaneView {
    CameraController camera;
    render::ViewMode mode = render::ViewMode::Lit;
    std::string camera_path;  // the scene camera it looks through; empty is the free camera
};

// The pane layouts in ViewLayout order: view.layout's name and the menu label.
constexpr std::pair<const char*, const char*> kViewLayouts[] = {
    {"single", "Single View"}, {"split", "Split View"}, {"quad", "Quad View"}};
static_assert(kViewPanes == render::kMaxSceneViews);

// {"cameras": [{name, target, distance, yaw, pitch, fov}]}, the file --cams reads.
json::Value BookmarksJson(const std::vector<CameraBookmark>& marks) {
    json::Value list = json::Value::Array();
    for (const CameraBookmark& b : marks)
        list.PushBack(Object({{"name", json::Value::Str(b.name)},
                              {"target", Vec3Json(b.target)},
                              {"distance", json::Value::Float(b.distance)},
                              {"yaw", json::Value::Float(b.yaw)},
                              {"pitch", json::Value::Float(b.pitch)},
                              {"fov", json::Value::Float(b.fov)}}));
    return Object({{"cameras", std::move(list)}});
}

// The bookmarks a file holds; false when it cannot be read.
bool ReadBookmarks(const std::string& path, std::vector<CameraBookmark>* out) {
    std::string text;
    if (!ReadEntireFile(path, &text)) return false;
    const json::Value doc = json::Value::Parse(text);
    std::vector<CameraBookmark> marks;
    for (const json::Value& c : doc.At("cameras").Elements()) {
        CameraBookmark b;
        b.name = c.At("name").AsString();
        Vec3Arg(c, "target", &b.target);
        b.distance = c.At("distance").AsFloat();
        b.yaw = c.At("yaw").AsFloat();
        b.pitch = c.At("pitch").AsFloat();
        b.fov = c.At("fov").AsFloat();
        marks.push_back(std::move(b));
    }
    *out = std::move(marks);
    return true;
}

json::Value PoseJson(const JointPose& p) {
    json::Value joints = json::Value::Object();
    for (const auto& [name, q] : p.joints) joints.Set(name, json::Value::Float(q));
    return Object({{"name", json::Value::Str(p.name)}, {"joints", std::move(joints)}});
}

// {"poses": [{name, joints: {joint: position}}]}, positions in rad or m.
json::Value PosesJson(const std::vector<JointPose>& poses) {
    json::Value list = json::Value::Array();
    for (const JointPose& p : poses) list.PushBack(PoseJson(p));
    return Object({{"poses", std::move(list)}});
}

// The poses a file holds; false when it cannot be read.
bool ReadPoses(const std::string& path, std::vector<JointPose>* out) {
    std::string text;
    if (!ReadEntireFile(path, &text)) return false;
    const json::Value doc = json::Value::Parse(text);
    std::vector<JointPose> poses;
    for (const json::Value& p : doc.At("poses").Elements()) {
        JointPose& pose = poses.emplace_back();
        pose.name = p.At("name").AsString();
        for (const auto& [name, q] : p.At("joints").Items()) pose.joints.emplace_back(name, q.AsFloat());
    }
    *out = std::move(poses);
    return true;
}

// The most recent samples of one frame timing.
struct SampleWindow {
    std::vector<double> values;
    size_t added = 0u;
    void Add(double x) {
        if (values.size() < kSampleCap) values.push_back(x);
        else values[added % kSampleCap] = x;
        ++added;
    }
    double Mean() const {
        double sum = 0.0;
        for (double v : values) sum += v;
        return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
    }
    double Percentile(double q) const {
        if (values.empty()) return 0.0;
        std::vector<double> sorted = values;
        const size_t k = static_cast<size_t>(q * static_cast<double>(sorted.size() - 1u));
        std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(k), sorted.end());
        return sorted[k];
    }
};

}  // namespace

struct EditorHost::State : DocumentHost {
    State(const std::vector<std::string>& args, nuka_world_handle world, float dt);
    ~State() override;

    void OpenLog();
    void StartShell();
    void StartChecks();
    void PushParam(const ParamWrite& write) override;
    bool Rebuild(const std::string& label) override;
    void ScriptsChanged() override { SyncScripts(); }
    ApplyResult ApplyTxn(const EditTxn& txn);
    void Reselect(const std::vector<std::string>& paths);
    ActionResult SubmitTxn(const EditTxn& txn);
    ActionResult StepHistory(bool redo);
    uint64_t DocumentHash();
    json::Value DocumentState();

    void RegisterActions();
    void RegisterFileActions();
    void RegisterAppActions();
    void RegisterEditActions();
    void RegisterSceneActions();
    void RegisterAssetActions();
    void RegisterSimActions();
    void RegisterViewActions();
    void RegisterSolverActions();
    void RegisterTimelineActions();
    void RegisterRobotActions();
    void RegisterPoseActions();
    void RegisterGainActions();
    void RegisterAssemblyActions();
    void RegisterIkActions();
    void RegisterOverlayActions();
    void RegisterTeleopActions();
    void RegisterPolicyActions();
    void RegisterDemoActions();
    json::Value TimelineStatus();
    // What a recording must match to load: the document, the world's counts and step, and the
    // names of the link and body rows its poses fill.
    json::Value RecordingIdentity();
    void StepVideo();
    void ConfigureExtensions();
    void Queue(const std::string& id, json::Value args, const std::string& source);
    void RunQueuedActions();
    ActionResult RunAction(const std::string& id, const json::Value& args, const std::string& source);
    ActionResult SaveScene(const std::string& path);
    ActionResult OpenDocument(const std::string& file, const std::string& doc_path);
    bool Confirmed(const std::string& id, const json::Value& args, ActionResult* asked);
    ActionResult Spawn(const std::vector<std::string>& args);
    std::string DocTitle() const { return doc_path_.empty() ? std::string("untitled") : FileName(doc_path_); }
    // A script session is named after its script; the start page shows while the editor's own
    // session has no scene.
    std::string SessionTitle() const {
        return !borrowed_ ? DocTitle() : opt_.script_name.empty() ? std::string("script") : FileName(opt_.script_name);
    }
    bool StartShown() const { return !loaded_ && !borrowed_ && !opt_.empty_requested; }
    void Autosave();
    void DropAutosave();
    void RefreshStartPage();
    std::string SelectedPath() const { return PathOfEntity(ui_state_.selected_entity); }
    std::vector<std::string> SelectionPaths() const;
    std::vector<std::string> TargetPaths(const json::Value& args) const;
    ActionResult ShowImport(const ImportAnalysis& analysis, const json::Value& pose);
    void ApplyHidden();
    void RefreshAssets();
    void RefreshProperties();
    void PickSelect(const CameraController& camera, float x, float y);
    // The present options for the scene viewport through `camera`: view mode and instance flags.
    render::RasterOptions ViewOptions(const CameraController& camera);
    // The present options for pane `rect` through `camera` in `mode`, without instance flags.
    render::RasterOptions ViewOptions(const CameraController& camera, const SceneViewportRect& rect, render::ViewMode mode);
    // Each shown instance's node path, and its selected, hovered and locked flags.
    const std::vector<std::string>& InstancePaths();
    std::vector<uint8_t> InstanceFlags();
    // Pins the latest zero-copy pose slot into `interop`; false without one.
    bool PinPose(uint32_t* slot, render::PresentInterop* interop);
    // What the id pass shows between two window points of the viewport, through `camera`.
    render::PickResult PickBox(const CameraController& camera, float x0, float y0, float x1, float y1);
    std::string CarrierPath(uint32_t instance) const;
    // scene.select's arguments for the bodies a box pick saw, after the selection when `add`.
    json::Value BoxSelectArgs(const render::PickResult& pick, bool add);
    void SetViewMode(render::ViewMode mode);
    // The panes the layout shows in layout order: the active one alone, the first two or all four.
    std::vector<uint32_t> ShownViews() const;
    CameraController& ViewCamera(uint32_t index) { return index == ui_state_.active_view ? camera_ : views_[index].camera; }
    render::ViewMode PaneMode(uint32_t index) const {
        return index == ui_state_.active_view ? ui_state_.view_mode : views_[index].mode;
    }
    // Moves input, the gizmo and the HUD to pane `index`.
    void ActivateView(uint32_t index);
    void SetViewLayout(ViewLayout layout);
    // Lists the scene cameras and the shown panes for the UI; a pane bound to a camera looks through it.
    void SyncViews(const render::RenderWorld& rw);
    // Residual and Contact turn on what they show while a shown pane uses them, and off after.
    void SyncModeOverlays();
    json::Value ViewJson(uint32_t index);
    // Shows the worker's RT images of the view through `camera` and asks for the next one, each
    // with twice the samples; a changed view or snapshot starts again at one.
    void StepRt(const CameraController& camera);
    // Reports an offline render once its process exits.
    void StepRender();
    // The document's bookmark file in the profile; empty while untitled.
    std::string BookmarkFile() const;
    CameraBookmark* FindBookmark(const std::string& name);
    // Shows the bookmarks' names and, with `save`, writes them to the document's file.
    void SyncBookmarks(bool save);
    // The document's pose file in the profile; empty while untitled.
    std::string PoseFile() const;
    JointPose* FindPose(const std::string& name);
    void SavePoses();
    // A load on the link or free body `inst` draws, acting at the world point `at` with a spring of
    // `gain` times the default toward it; false for an instance no dynamics move.
    bool BodyLoad(const render::RenderInstance& inst, const math::Vec3& at, float gain, Perturbation* out) const;
    void PushPerturb(const Perturbation& p);
    void GrabAt(const CameraController& camera, float x, float y);
    void DragGrab(const CameraController& camera, float x, float y);
    void ReleaseGrab();
    void PinAt(const CameraController& camera, float x, float y);
    // The Measure tool's click: the joint origin nearest the pointer or the link under it becomes the
    // next end; a second click on nothing measures down to the ground.
    void MeasureAt(const CameraController& camera, float x, float y);
    // The measurement between the tool's ends, null while they do not resolve.
    json::Value MeasureJson() const;
    const render::RenderWorld& ShownWorld() const { return loaded_ ? loaded_->sim->GetRenderWorld() : empty_world_; }

    void Frame(bool step_physics);
    // Editor output goes to `stream` and the session log, and with `console` to the Console panel.
    void Emit(std::FILE* stream, const std::string& text, bool console = false);
    void WriteLog(const std::string& text);
    void ConsoleAppend(const std::string& s);
    void DrainConsoleInbox();
    void SetHooks(const nuka_editor_script_hooks_t& hooks);
    // Runs `fn` on the UI thread. Other threads wait for it to run there between frames; the
    // physics thread counts as parked meanwhile, so `fn` may edit the world.
    void OnUi(const std::function<void()>& fn);
    // Runs the calls other threads queued, waiting for more until `until`.
    void ServeRemote(Clock::time_point until);
    void StopWorker();
    void RegisterScriptActions();
    // Queues `code` for the physics side, compiled as `name` ("" for the console); the seq
    // script.result answers for.
    uint64_t RunScript(const std::string& code, const std::string& name, bool echo);
    // The tab `args` names by id, else the active one.
    ScriptTab& TabArg(const json::Value& args);

    void FrameWorld(const render::RenderWorld& rw);
    std::string PathOfEntity(scene::EntityId e) const;
    uint64_t Generation() const;
    void SyncScripts();
    void SeedDriveTargets();
    // The current environment's drive targets into the drive editor; the worker must be parked.
    void LoadDriveTargets();
    // The environments the grid draws beside `current`: the rest of its page of `show` cells.
    std::vector<uint32_t> OtherShownEnvs(uint32_t current) const;
    // Where environment `env` draws relative to `current`, which stays at world coordinates.
    math::Vec3 EnvOffset(uint32_t env, uint32_t current) const;
    // Makes `env` current; on the same page the camera follows its cell, so the view holds still.
    void SwitchEnv(uint32_t env);
    json::Value EnvsJson() const;
    // Each link's longest visual edge, from its meshes' bounds in its frame.
    void SizeLinks();
    void BindWorker(uint64_t tick = 0u, double sim_time = 0.0);
    void DropPublisher();
    void ApplySnapshot(const Snapshot& snap);
    // Takes the telemetry records the worker read since the last frame into the Solver view.
    void SyncSolver();
    // Takes the joint samples the worker took since the last frame into the Joints view, which
    // follows the joint of the selected link.
    void SyncJoints();
    // Adds a finished gain test to the tuner's runs with what it measured.
    void SyncGainTest();
    // The selection and the scene body's offset from its parent link the Robot Assembler shows.
    void SyncAssembler();
    // The link `name` names by node path or incoming joint, by index into the robot's links; -1 when none.
    int FindLink(const std::string& name) const;
    // The selected link the gizmo drags through IK; -1 for a root or while IK is off.
    int IkLink() const;
    // Shows an IK report: the gizmo drag's miss, and the drive targets no edit holds.
    void TakeIkReport(const IkReport& report);
    // Takes the newest IK report and sends the drag's next solve once the last one is done.
    void SyncIk();
    // The gizmo on IK link `link`: a press holds the pins and the link's pose under the gizmo at
    // `live`; each move makes the next solve toward `gizmo`.
    void DragIk(int link, const math::Transform& live, const math::Transform& gizmo, bool changed, bool using_now);
    // The devices teleop can read: the keyboard, the virtual device, the SDL pads and the extensions'.
    void ListTeleopDevices();
    // The active device's reading through the mode's mapping into a controller command, a joint's
    // target, the IK goal or an extension's targets; true while the keyboard device owns W A S D.
    bool StepTeleop(double dt);
    // The ik mode's link: the one teleop.use named, else the selected link below a root; -1 when none.
    int TeleopLink() const;
    json::Value TeleopStatus() const;
    // A policy over the loaded robots, or null with `error` saying why it does not fit them.
    std::unique_ptr<inference::Go2PolicyController> MakeController(const std::string& path, bool host,
                                                                   std::string* error) const;
    // The Policy panel's copy of the loaded policy's contract; the command and the norms follow it.
    void SyncPolicyView();
    // The shown robots' action norm at a new tick, read back from the drive targets the policy wrote.
    void RecordPolicyNorm(const Snapshot& snap);
    json::Value PolicyStatus();
    // Ends the demonstration recording; each environment's consecutive steps become an episode.
    void StopDemo();
    // Ends a replay; the drive targets keep its last row.
    void StopReplay();
    void SyncDemoView();
    json::Value DemoStatus() const;
    // Runs nuka.editor.demos.`fn` between the HDF5 file at `path` and the exchange files in the
    // cache; returns why it failed, empty when it ran.
    std::string RunDemoExchange(const char* fn, const std::string& path);
    void TakeWorkerResults();
    void PushTransport();
    void PushPoseWrites();
    void PushEdits();
    void DrawWorld(const render::RenderWorld& rw, const std::vector<render::RasterOptions>& views);
    void PrintSummary();
    void SetLoadError(const std::string& path);
    void CreateWorld(const std::string& path);
    void ApplyLoad(const std::string& path);
    void Attach(const std::string& path);
    void OnLoaded(const std::string& path);
    void OnRebound(uint64_t tick, double sim_time);
    void AttachController();
    void ApplyUnload();
    void SeedInspector(scene::EntityId e);
    void ApplyInspectorEdits();

    LaunchOptions opt_;
    nuka_world_handle world_ = nullptr;
    nuka_device_handle device_ = nullptr;  // set when the editor created the world
    const bool borrowed_;                  // the host's script owns the world
    std::optional<EditorProfile> profile_;
    std::ofstream log_;
    std::string log_path_;
    std::unique_ptr<InstanceServer> instance_;
    SceneCache cache_;
    render::RenderWorld empty_world_;
    std::unique_ptr<inference::Go2PolicyController> policy_ctrl_;
    bool policy_on_ = true;      // a loaded policy steps; off, its drive targets hold
    uint64_t policy_tick_ = 0u;  // the tick RecordPolicyNorm last read
    DemoSet demos_;              // the demonstrations recorded or loaded
    bool demo_recording_ = false;
    std::vector<float> demo_axes_;  // the teleop axes last sent while recording, by the set's names
    uint64_t demo_play_seq_ = 0u;   // the replay's track command
    std::unique_ptr<EditorScene> loaded_;
    bool attach_pending_ = false;
    EntityDrag entity_drag_;
    scene::EntityId last_seeded_ = scene::kInvalidEntity;

    // Edits preview live; a gesture's commit queues one edit.apply for the next frame top.
    EditorDocument doc_;
    std::string doc_path_;  // the file the document saves to; empty while untitled
    // An action that would drop unsaved changes waits here for the answer.
    struct Prompt {
        std::string action;
        json::Value args;
    };
    std::optional<Prompt> prompt_;
    // A Save answer resumes its action once the frame showing the saved scene is captured.
    std::optional<Prompt> resume_;
    std::string autosave_file_;
    Clock::time_point autosave_due_{};
    uint64_t autosaved_revision_ = 0u;
    double autosave_seconds_ = kAutosaveSeconds;
    std::vector<StartEntry> checks_;     // startup checks reported through app.check
    struct Dropped {
        std::string path;
        float x, y;  // where it landed, in window points
    };
    std::vector<Dropped> dropped_;  // files dropped on the window since the last frame
    std::string start_locale_;          // persisted on exit once the user switched languages
    bool start_stale_ = true;           // the start page lists need a refresh
    // A save's viewport picture, captured without the UI from the next drawn frame.
    struct ThumbRequest {
        std::string scene;
        std::string capture;
        uint64_t frame = 0u;
    };
    std::optional<ThumbRequest> thumb_;
    // An mp4 export: each frame seeks the playhead to its tick and captures the viewport once the
    // seek shows; ffmpeg then encodes the frames.
    struct VideoExport {
        std::string path;
        std::string frames;  // the folder of captured frames
        std::vector<uint64_t> ticks;
        double fps = 30.0;
        uint64_t back = 0u;         // the playhead to return to
        size_t next = 0u;           // the frame in progress
        uint64_t seek = 0u;         // its seek, 0 until sent
        uint64_t capture = ~0ull;   // the drawn frame capturing it
        uint64_t encoder = 0u;      // ffmpeg's process
    };
    std::optional<VideoExport> video_;
    json::Value exported_;  // the last export's path and frames, with an error when it failed
    // The RT preview; one request is in flight at a time and `chain` is the view its samples refine.
    struct RtPreview {
        bool on = false;
        uint32_t max_spp = 64u;
        uint64_t inflight = 0u;  // the Beauty command not yet answered
        RtKey asked;
        RtKey chain;
        RtKey shown;
        uint32_t spp = 0u;           // the most samples shown of `chain`
        std::vector<uint32_t> done;  // the samples of each image shown of `chain`
        std::string error;
    };
    RtPreview rt_;
    // An offline render: another process path-traces a copy of the scene and its state.
    struct RenderJob {
        std::string path;
        std::string result;  // the JSON file it answers in
        uint64_t process = 0u;
        Clock::time_point started{};
    };
    std::optional<RenderJob> render_;
    json::Value rendered_;  // the last render's answer
    std::vector<CameraBookmark> bookmarks_;
    std::unique_ptr<ThumbnailCache> thumbnails_;  // asset and example pictures
    uint64_t start_thumb_version_ = 0u;
    std::vector<AssetFile> asset_files_;
    bool assets_stale_ = true;
    uint64_t cards_version_ = ~0ull;  // the picture version the asset cards show
    bool cards_pictures_ = false;     // the cards ask for pictures once the panel showed
    // Hidden nodes' instances draw without a mesh; each held-aside mesh by instance index.
    std::vector<std::pair<uint32_t, uint32_t>> hidden_meshes_;
    std::vector<std::string> applied_hidden_;
    bool hidden_stale_ = false;  // a rebind made new instances
    std::vector<std::string> props_paths_;  // what the Properties schema was built for
    uint64_t props_scene_version_ = ~0ull;
    uint64_t props_revision_ = ~0ull;
    uint64_t joints_revision_ = ~0ull;
    bool click_armed_ = false;
    float click_x_ = 0.0f, click_y_ = 0.0f;
    std::vector<std::string> instance_paths_;  // by shown instance; cleared when instances change
    uint64_t instance_paths_version_ = ~0ull;
    std::string hovered_path_;  // the body under the pointer, outlined as hovered
    bool hover_due_ = false;    // the pointer moved over the viewport since the last hover pick
    float hover_x_ = 0.0f, hover_y_ = 0.0f;
    bool mode_heatmap_ = false;   // the Residual view turned the heatmap on
    bool mode_contacts_ = false;  // the Contact view turned the contact markers on
    std::array<PaneView, kViewPanes> views_{};
    std::optional<ViewLayout> restore_layout_;  // the layout view.maximize goes back to
    bool views_stale_ = false;  // a new document: the other panes start at its framing
    ActionRegistry actions_;
    uint64_t param_seq_ = 0u;  // the last registry write and the ticks it was queued and applied at
    uint64_t param_pushed_tick_ = 0u;
    uint64_t param_applied_tick_ = 0u;
    bool param_done_ = true;
    std::string last_error_;
    bool xform_gesture_active_ = false;
    bool xform_dirty_ = false;
    std::string xform_gesture_path_;
    math::Transform xform_before_;
    math::Transform xform_after_;
    bool mat_gesture_active_ = false;
    std::string mat_gesture_path_;
    scene::RenderMaterial mat_before_{};
    bool gizmo_using_prev_ = false;
    // An IK drag: the link's pose in the gizmo's frame and at the press, the gizmo while held, the
    // pins' press poses and the solve waiting for the one in flight.
    bool ik_drag_ = false;
    math::Transform ik_offset_;
    math::Transform ik_press_;
    math::Transform ik_held_;
    std::vector<std::pair<uint32_t, math::Transform>> ik_pins_;
    std::optional<TickCommand> ik_next_;
    // The Perturb tool's load while its drag lasts; the target slides on the view plane through the anchor.
    std::optional<Perturbation> grab_;
    math::Vec3 grab_anchor_ = math::Vec3::Zero();

    void* native_window_ = nullptr;
    std::string window_title_;
    std::unique_ptr<render::PresentRenderer> present_;
    render::imgui::NukaImGuiContext imgui_;
    ImGuiLayer ui_;
    ExtensionManager extensions_;
    bool extensions_started_ = false;
    CameraController camera_;
    ViewerUiState ui_state_;
    DebugOverlay debug_overlay_;
    ScriptHost script_host_;
    PhysicsWorker worker_{&script_host_};
    ScriptCompleter completer_{&script_host_};
    std::unique_ptr<render::InteropTransformSsbo> interop_;
    std::unique_ptr<PoseScatter> scatter_;
    std::string publish_reason_;          // why poses publish through host downloads
    std::deque<uint32_t> frame_pins_;     // pose slots of submitted frames still in flight
    uint64_t bound_epoch_ = 0u;
    const Snapshot* shown_ = nullptr;     // the front snapshot, valid until the next acquire
    uint64_t last_sequence_ = 0u;
    uint64_t snapshot_drops_ = 0u;
    WorkerTelemetry last_telemetry_;
    uint64_t last_tick_ = 0u;
    bool last_step_healthy_ = true;
    // What the worker was last told, so changes become commands.
    bool sent_playing_ = false;
    uint32_t sent_env_ = 0u;
    std::vector<uint32_t> sent_shown_;
    uint32_t picked_env_ = 0u;       // the environment PickBox's centre hit lies in
    float env_auto_spacing_ = 1.0f;  // grid cells sized from the moving instances at bind time
    bool sent_colliders_ = false;
    bool sent_contacts_ = false;
    float sent_speed_ = 1.0f;
    uint64_t sent_static_revision_ = 0u;
    uint64_t transport_seq_ = 0u;
    uint64_t reset_seq_ = 0u;
    uint64_t solver_seq_ = 0u;  // the last solver command
    uint64_t timeline_seq_ = 0u;  // the last timeline command
    bool reset_framing_ = false;
    uint64_t run_script_seq_ = 0u;
    bool frame_pending_ = false;
    render::SceneViewport last_framing_rect_;
    std::map<uint64_t, std::string> live_scripts_;  // id -> source last registered

    Clock::time_point prev_ = Clock::now();
    double fps_ema_ = 0.0;
    uint64_t frame_index_ = 0;
    int presented_ = 0;
    uint64_t measured_ = 0u;
    SampleWindow frame_ms_, ui_ms_, apply_ms_, draw_ms_, sleep_ms_, age_ms_, step_ms_;
    bool ctrl_down_ = false;
    bool surface_suspended_ = false;
    TeleopPads pads_;
    // The virtual device teleop.input sets, in the pad layout.
    std::vector<float> virtual_axes_ = std::vector<float>(kPadAxes.size(), 0.0f);
    std::vector<uint8_t> virtual_buttons_ = std::vector<uint8_t>(kPadButtons.size(), 0u);
    bool teleop_hold_prev_ = false;
    // The command last sent to the controller; none makes the next frame send it.
    std::optional<std::array<float, 3>> teleop_command_sent_;
    float teleop_press_[2] = {0.0f, 0.0f};  // last frame's next and prev, so a press steps once
    // The ik mode's link, its pose when teleop took it and the pinned links' poses then.
    int teleop_ik_link_ = -1;
    math::Transform teleop_ik_start_;
    std::vector<std::pair<uint32_t, math::Transform>> teleop_ik_pins_;
    float last_mouse_x_ = 0.0f;
    float last_mouse_y_ = 0.0f;
    bool script_ran_ = false;
    std::vector<window::WindowEvent> events_;
    std::atomic<bool> running_{true};
    std::atomic<bool> playing_mirror_{false};  // read by script callbacks off the UI thread
    std::mutex inbox_mu_;
    std::string console_inbox_;                // host console text, appended from any thread
    struct RemoteCall {
        std::function<void()> fn;
        bool done = false;
        std::exception_ptr error;
    };
    std::atomic<std::thread::id> ui_thread_{};
    std::mutex remote_mu_;
    std::condition_variable remote_cv_;
    std::deque<RemoteCall*> remote_;
    bool closing_ = false;  // queued calls fail instead of running
    struct ScriptRun {
        std::string name;
        bool done = false;
        bool failed = false;
        std::string output;
    };
    std::map<uint64_t, ScriptRun> script_runs_;  // by seq; the oldest go beyond kScriptRunsKept
    Clock::time_point scripts_polled_{};
    size_t console_base_ = 0u;  // console bytes trimmed off the front so far
};

EditorHost::State::State(const std::vector<std::string>& args, nuka_world_handle world, float dt)
    : world_(world), borrowed_(world != nullptr) {
    ui_thread_ = std::this_thread::get_id();
    ui_state_.teleop.maps = DefaultTeleopMaps();
    if (dt > 0.0f) {
        opt_.dt = dt;
        opt_.dt_set = true;
    }
    if (!ParseOptions(args, &opt_)) {
        running_ = false;
        return;
    }
    profile_.emplace(opt_.profile);
    OpenLog();
    thumbnails_ = std::make_unique<ThumbnailCache>(profile_->Dir() / "thumbnails" / "assets");
    if (opt_.locale.empty()) opt_.locale = profile_->Setting("ui", "locale", "");
    autosave_seconds_ = profile_->SettingNumber("editor", "autosave_seconds", kAutosaveSeconds);
    script_host_.SetBudget(profile_->SettingNumber("script", "budget_s", script_host_.Budget()));
    ui_state_.script_rerun = profile_->SettingBool("script", "rerun_on_save", false);
    autosave_file_ = PathText(profile_->Dir() / "autosave" /
                              (std::to_string(UnixNow()) + "-" + std::to_string(ProcessId()) + ".nks"));
    if (const char* env = std::getenv("NUKA_VIEWER_FRAMES")) opt_.max_frames = std::atoi(env);
    // The trained control rate is 0.005 s; an explicit dt always wins.
    if (!opt_.dt_set && !opt_.policy_path.empty()) opt_.dt = 0.005f;
    if (!(opt_.dt > 0.0f)) throw std::runtime_error("invalid --dt (must be > 0)");
    // The failure reaches the user as a dialog or EditorStartupError, so it names the log.
    try {
        StartShell();
    } catch (const std::exception& e) {
        Emit(stderr, std::string("[nuka_editor] startup failed: ") + e.what() + "\n");
        throw std::runtime_error(std::string(e.what()) + "\nlog: " + log_path_);
    }
    if (opt_.primary && !borrowed_) instance_ = std::make_unique<InstanceServer>(profile_->InstanceName());
    ui_state_.script_session = borrowed_;
    ui_state_.script_name = opt_.script_name;

    // Loads run inside a frame, once the shell is up and the host installed its hooks.
    if (!opt_.scene_path.empty()) Queue("file.open", Object({{"path", json::Value::Str(opt_.scene_path)}}), "cli");
    else if (!opt_.new_template.empty())
        Queue("file.new", Object({{"template", json::Value::Str(opt_.new_template)}}), "cli");
    else attach_pending_ = borrowed_;
    Emit(stdout, Format("[nuka_editor] backend=%s device=%s images=%u\n", present_->Report().backend_name.c_str(),
                        present_->DeviceName().c_str(), present_->SwapchainImageCount()));
}

// One log per session in the profile; the oldest beyond kLogKeep are deleted.
void EditorHost::State::OpenLog() {
    const fs::path logs = profile_->Dir() / "logs";
    std::vector<fs::path> old;
    for (const fs::directory_entry& entry : fs::directory_iterator(logs))
        if (entry.path().extension() == ".log") old.push_back(entry.path());
    std::sort(old.begin(), old.end());
    for (size_t i = 0; i + kLogKeep <= old.size(); ++i) fs::remove(old[i]);
    log_path_ = opt_.log_path.empty()
                    ? PathText(logs / ("editor-" + LocalTimeText(UnixNow(), "%Y%m%d-%H%M%S") + "-" +
                                       std::to_string(ProcessId()) + ".log"))
                    : NormalPath(opt_.log_path);
    log_.open(Utf8Path(log_path_), std::ios::binary | std::ios::app);
    nuka_version_t version = nuka_get_version();
    WriteLog(Format("[nuka_editor] nuka %u.%u.%u, profile %s\n", version.major, version.minor, version.patch,
                    PathText(profile_->Dir()).c_str()));
}

void EditorHost::State::StartShell() {
    window::SurfaceBackendKind kind = window::SurfaceBackendKind::Headless;
    std::unique_ptr<window::WindowSurface> surface =
        opt_.offscreen ? window::MakeHeadlessSurface(1280u, 720u)
                       : window::MakeSurface("Nuka Editor", 1280u, 720u, &kind);
    if (!surface || kind == window::SurfaceBackendKind::None)
        throw std::runtime_error("no window surface: no SDL video driver and no VK_EXT_headless_surface");

    // Bind the native window and route its events to ImGui before the renderer takes the surface;
    // dropped files open at the next frame.
    native_window_ = surface->NativeWindowHandle();
    surface->SetNativeEventHook([this](const void* event) {
        const SDL_Event* e = static_cast<const SDL_Event*>(event);
        if (e->type == SDL_EVENT_DROP_FILE && e->drop.data != nullptr) dropped_.push_back({e->drop.data, e->drop.x, e->drop.y});
        if (ImGui::GetCurrentContext() != nullptr) ImGui_ImplSDL3_ProcessEvent(e);
    });
    present_ = std::make_unique<render::PresentRenderer>(std::move(surface));
    if (opt_.capture_frame >= 0) present_->SetCaptureFrame(opt_.capture_frame, opt_.capture_out);

    render::RendererVulkanHandles vk = present_->VulkanHandles();
    render::imgui::NukaImGuiInitInfo info;
    info.api_version     = vk.api_version;
    info.instance        = reinterpret_cast<NukaVkInstance>(vk.instance);
    info.physical_device = reinterpret_cast<NukaVkPhysicalDevice>(vk.physical_device);
    info.device          = reinterpret_cast<NukaVkDevice>(vk.device);
    info.queue_family    = vk.graphics_family;
    info.queue           = reinterpret_cast<NukaVkQueue>(vk.graphics_queue);
    info.descriptor_pool = reinterpret_cast<NukaVkDescriptorPool>(vk.imgui_descriptor_pool);
    info.min_image_count = present_->MinImageCount();
    info.image_count     = present_->SwapchainImageCount();
    info.render_pass     = reinterpret_cast<NukaVkRenderPass>(present_->PresentRenderPass());
    info.subpass         = 0u;
    if (!imgui_.Init(info)) throw std::runtime_error("ImGui init failed");
    if (native_window_ != nullptr) {
        SDL_Window* window = static_cast<SDL_Window*>(native_window_);
        ImGui_ImplSDL3_InitForVulkan(window);
        // The window has no system frame: the UI draws the title bar and the hit test moves and sizes it.
        SDL_SetWindowHitTest(window, TitleBarHitTest, &ui_state_.title_bar);
        uint32_t w = 0u, h = 0u;
        std::vector<unsigned char> logo = render::imgui::NukaLogoPixels(&w, &h);
        if (!logo.empty()) {
            SDL_Surface* icon = SDL_CreateSurfaceFrom(static_cast<int>(w), static_cast<int>(h), SDL_PIXELFORMAT_RGBA32,
                                                      logo.data(), static_cast<int>(w * 4u));
            SDL_SetWindowIcon(window, icon);
            SDL_DestroySurface(icon);
        }
    }

    ui_.EnableDocking();  // must precede the first NewFrame
    RegisterActions();
    ConfigureExtensions();
    if (!opt_.layout.empty() && !ui_.SetLayout(opt_.layout))
        Emit(stderr, "[nuka_editor] unknown --layout '" + opt_.layout + "'\n");
    ui_.SetUiScale(opt_.ui_scale);
    if (!opt_.locale.empty()) nui::SetLocale(opt_.locale);
    start_locale_ = nui::Locale();
    ui_state_.show_colliders = opt_.show_colliders;
    ui_state_.show_contacts = opt_.show_contacts;
    ui_state_.speed = opt_.speed;
    camera_.FrameAabb(math::Vec3{-1.0f, -1.0f, 0.0f}, math::Vec3{1.0f, 1.0f, 1.0f});
    StartChecks();
}

// Facts the start page shows; Python adds its own through app.check.
void EditorHost::State::StartChecks() {
    const nuka_version_t version = nuka_get_version();
    nuka_runtime_info_t info{};
    info.struct_size = sizeof(info);
    info.schema_version = NUKA_RUNTIME_INFO_VERSION;
    const bool cuda = nuka_get_runtime_info(&info) == NUKA_RESULT_OK && info.cuda_driver_version > 0;
    const auto cuda_text = [](int32_t v) { return std::to_string(v / 1000) + "." + std::to_string(v % 1000 / 10); };
    checks_.push_back(StartEntry{"Nuka", Format("%u.%u.%u", version.major, version.minor, version.patch), "", "", true});
    checks_.push_back(StartEntry{"GPU", present_->DeviceName(), "", "", true});
    checks_.push_back(StartEntry{"Vulkan", present_->Report().backend_name, "", "", true});
    checks_.push_back(StartEntry{"CUDA",
                                 cuda ? "runtime " + cuda_text(info.cuda_runtime_version) + ", driver " +
                                            cuda_text(info.cuda_driver_version)
                                      : std::string("no CUDA driver"),
                                 "", "", cuda});
    const bool interop = present_->VulkanHandles().interop_supported;
    checks_.push_back(StartEntry{"Publish", interop ? "zero-copy" : "host copies", "", "", interop});
    for (const StartEntry& c : checks_)
        WriteLog("[nuka_editor] check " + c.title + ": " + c.detail + (c.available ? "\n" : " (failed)\n"));
}

EditorHost::State::~State() {
    {
        std::lock_guard<std::mutex> lock(remote_mu_);
        closing_ = true;
    }
    if (profile_ && present_) {
        DropAutosave();
        if (nui::Locale() != start_locale_) profile_->SetSetting("ui", "locale", nui::Locale());
    }
    extensions_.Shutdown();
    StopWorker();
    if (present_) {
        present_->WaitIdle();
        if (native_window_ != nullptr) {
            SDL_SetWindowHitTest(static_cast<SDL_Window*>(native_window_), nullptr, nullptr);
            ImGui_ImplSDL3_Shutdown();
        }
        imgui_.Shutdown();
        Emit(stdout, Format("[nuka_editor] done: frames=%llu presented=%d last_step_healthy=%s\n",
                            static_cast<unsigned long long>(frame_index_), presented_,
                            last_step_healthy_ ? "yes" : "no"));
        PrintSummary();
        std::fflush(stdout);
        DropPublisher();
    }
    loaded_.reset();
    policy_ctrl_.reset();
    if (device_ != nullptr) {
        nuka_world_destroy(world_);
        nuka_device_destroy(device_);
    }
    cache_.Remove();
}

void EditorHost::State::Emit(std::FILE* stream, const std::string& text, bool console) {
    std::fputs(text.c_str(), stream);
    if (console) ConsoleAppend(text);
    else WriteLog(text);
}

void EditorHost::State::WriteLog(const std::string& text) {
    log_ << text;
    log_.flush();
}

// The Console panel's text also lands in the session log. Past the cap the oldest quarter goes,
// whole lines at a time.
void EditorHost::State::ConsoleAppend(const std::string& s) {
    if (s.empty()) return;
    WriteLog(s);
    std::string& log = ui_state_.console_log;
    log += s;
    if (log.size() <= kConsoleLogCap) return;
    const size_t nl = log.find('\n', log.size() - kConsoleLogCap * 3u / 4u);
    const size_t cut = nl == std::string::npos ? log.size() : nl + 1u;
    log.erase(0, cut);
    console_base_ += cut;
}

void EditorHost::State::DrainConsoleInbox() {
    std::string text;
    {
        std::lock_guard<std::mutex> lock(inbox_mu_);
        text.swap(console_inbox_);
    }
    ConsoleAppend(text);
}

// Search order: --ext-folder, so a working copy wins over an installed one, then next to the
// module, the Python package's folder and the profile's.
void EditorHost::State::ConfigureExtensions() {
    std::vector<std::filesystem::path> folders;
    for (const std::string& folder : opt_.ext_folders) folders.push_back(Utf8Path(folder));
    const std::filesystem::path module = render::ViewerModuleDirectory();
    folders.push_back(module / "extensions");
    folders.push_back(module / "editor" / "extensions");
    folders.push_back(profile_->Dir() / "extensions");
    ExtensionServices services;
    services.actions = &actions_;
    services.ui = &ui_;
    services.run = [this](const std::string& id, const json::Value& args, const std::string& source) {
        ActionResult result;
        OnUi([&] { result = RunAction(id, args, source); });
        return result;
    };
    services.console = [this](const std::string& text) { Emit(stdout, text, /*console=*/true); };
    services.state_file = profile_->Dir() / "extensions.toml";
    extensions_.Configure(std::move(services), std::move(folders), opt_.ext_ids);
}

void EditorHost::State::PrintSummary() {
    if (frame_ms_.values.empty()) return;
    const std::vector<double>& f = frame_ms_.values;
    const double mean = frame_ms_.Mean();
    Emit(stdout, Format("[fps] SUMMARY measured=%llu mean_fps=%.1f mean_frame_ms=%.2f (min=%.2f max=%.2f) "
                        "mean_step_ms=%.2f p50=%.2f p95=%.2f p99=%.2f ui=%.2f apply=%.2f draw=%.2f sleep=%.2f "
                        "age_p95=%.2f drops=%llu publishes=%llu ticks=%llu capped=%llu dropped_sim_s=%.3f "
                        "mode=%s reason=\"%s\"\n",
                        static_cast<unsigned long long>(measured_), 1000.0 / mean, mean,
                        *std::min_element(f.begin(), f.end()), *std::max_element(f.begin(), f.end()),
                        step_ms_.Mean(), frame_ms_.Percentile(0.50), frame_ms_.Percentile(0.95),
                        frame_ms_.Percentile(0.99), ui_ms_.Mean(), apply_ms_.Mean(), draw_ms_.Mean(),
                        sleep_ms_.Mean(), age_ms_.Percentile(0.95), static_cast<unsigned long long>(snapshot_drops_),
                        static_cast<unsigned long long>(last_sequence_), static_cast<unsigned long long>(last_tick_),
                        static_cast<unsigned long long>(last_telemetry_.capped_batches), last_telemetry_.dropped_sim_s,
                        scatter_ ? "zero-copy" : "host", publish_reason_.c_str()));
}

void EditorHost::State::SetHooks(const nuka_editor_script_hooks_t& hooks) {
    script_host_.SetHooks(hooks);
    ui_state_.complete = [this](std::string code, int line, int column) {
        return completer_.Request(std::move(code), line, column);
    };
    ui_state_.take_completion = [this](uint64_t ticket, std::string* json) { return completer_.Take(ticket, json); };
    live_scripts_.clear();
    SyncScripts();
}

void EditorHost::State::OnUi(const std::function<void()>& fn) {
    if (std::this_thread::get_id() == ui_thread_.load()) {
        fn();
        return;
    }
    RemoteCall call;
    call.fn = fn;
    const auto served = [&] {
        std::unique_lock<std::mutex> lock(remote_mu_);
        remote_cv_.wait(lock, [&] { return call.done; });
    };
    {
        std::lock_guard<std::mutex> lock(remote_mu_);
        if (closing_) throw std::runtime_error("the editor is closing");
        remote_.push_back(&call);
    }
    remote_cv_.notify_all();
    if (worker_.OnThread()) worker_.ParkWhile(served);
    else served();
    if (call.error) std::rethrow_exception(call.error);
}

void EditorHost::State::ServeRemote(Clock::time_point until) {
    std::unique_lock<std::mutex> lock(remote_mu_);
    for (;;) {
        while (!remote_.empty()) {
            RemoteCall* call = remote_.front();
            remote_.pop_front();
            const bool closing = closing_;
            lock.unlock();
            try {
                if (closing) throw std::runtime_error("the editor is closing");
                call->fn();
            } catch (...) {
                call->error = std::current_exception();
            }
            lock.lock();
            call->done = true;
            remote_cv_.notify_all();
        }
        if (!remote_cv_.wait_until(lock, until, [&] { return !remote_.empty(); })) return;
    }
}

// A script on the physics thread may be waiting on a UI call; the join serves it.
void EditorHost::State::StopWorker() {
    worker_.StopThread([this] { ServeRemote(Clock::now() + std::chrono::milliseconds(2)); });
}

void EditorHost::State::FrameWorld(const render::RenderWorld& rw) {
    if (opt_.cam_on) {
        camera_.SetView(opt_.cam_target, opt_.cam_dist, opt_.cam_yaw, opt_.cam_pitch);
        return;
    }
    const auto& report = present_->Report();
    const float reference = report.height ? static_cast<float>(report.width) / static_cast<float>(report.height) : 1.0f;
    if (camera_.UseSceneCamera(rw, ui_.Viewport().Aspect(), reference)) return;
    math::Vec3 lo, hi;
    SceneAabb(rw, /*movable_only=*/true, &lo, &hi);
    camera_.FrameAabb(lo, hi, ui_.Viewport().Aspect());
}

std::string EditorHost::State::PathOfEntity(scene::EntityId e) const {
    if (!loaded_ || e == scene::kInvalidEntity) return std::string();
    if (const auto n = loaded_->scene.Ecs().NodeOf(e)) return loaded_->scene.Tree().PathOf(n);
    return std::string();
}

uint64_t EditorHost::State::Generation() const {
    uint64_t generation = 0u;
    nuka_world_get_generation(world_, &generation);
    return generation;
}

// Diffs the scene's /script records against the registered set, so attach, delete
// and undo keep the host in step.
void EditorHost::State::SyncScripts() {
    std::map<uint64_t, std::string> want;
    if (loaded_)
        for (const auto& sr : loaded_->scene.Scripts()) want[sr.stable_id] = sr.source;
    for (auto it = live_scripts_.begin(); it != live_scripts_.end();) {
        if (want.count(it->first) == 0u) {
            script_host_.UnregisterScript(it->first);
            it = live_scripts_.erase(it);
        } else {
            ++it;
        }
    }
    bool registered = false;
    for (const auto& entry : want) {
        const auto it = live_scripts_.find(entry.first);
        if (it != live_scripts_.end() && it->second == entry.second) continue;
        // Re-registering replaces the old namespace, so a replaced on_step stops.
        ConsoleAppend(script_host_.RegisterScript(entry.first, entry.second));
        live_scripts_[entry.first] = entry.second;
        registered = true;
    }
    if (registered) ConsoleAppend(script_host_.DispatchReady());
    ui_state_.script_node_count = loaded_ ? static_cast<int>(loaded_->scene.ScriptCount()) : 0;
}

// The joint table and the per-DOF drive editor, from environment 0's live targets.
void EditorHost::State::SeedDriveTargets() {
    ui_state_.drive_targets.clear();
    ui_state_.drive_dirty.clear();
    const WorldLayout& layout = loaded_->Layout();
    RobotView& robot = ui_state_.robot;
    const auto path_of = [this](scene::EntityId e) { return PathOfEntity(e); };
    robot.joints = RobotJoints(loaded_->scene, layout, path_of);
    robot.links = RobotLinks(loaded_->scene, layout, path_of);
    robot.roots = RobotRoots(loaded_->scene, layout, path_of);
    SizeLinks();
    robot.floating = std::any_of(layout.links.begin(), layout.links.end(),
                                 [](const nuka_link_binding_t& l) { return l.is_floating != 0u; });
    LoadDriveTargets();
}

void EditorHost::State::LoadDriveTargets() {
    const WorldLayout& layout = loaded_->Layout();
    const uint32_t links = layout.links_per_env;
    if (links == 0u) return;
    const size_t env = ui_state_.env_index % layout.env_count;
    ui_state_.drive_targets.assign(links, 0.0f);
    ui_state_.drive_dirty.assign(links, 0u);
    nuka_world_download_field(world_, NUKA_FIELD_DRIVE_TARGET, ui_state_.drive_targets.data(),
                              links * sizeof(float), env * links * sizeof(float));
}

std::vector<uint32_t> EditorHost::State::OtherShownEnvs(uint32_t current) const {
    const EnvView& v = ui_state_.envs;
    const uint32_t first = current / v.show * v.show, last = std::min(first + v.show, v.count);
    std::vector<uint32_t> out;
    for (uint32_t e = first; e < last; ++e)
        if (e != current) out.push_back(e);
    return out;
}

math::Vec3 EditorHost::State::EnvOffset(uint32_t env, uint32_t current) const {
    const EnvView& v = ui_state_.envs;
    const auto cols = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<float>(v.show))));
    const float s = v.spacing > 0.0f ? v.spacing : env_auto_spacing_;
    const auto cell = [&](uint32_t e) {
        const uint32_t r = e % v.show;
        return math::Vec3{static_cast<float>(r % cols) * s, -static_cast<float>(r / cols) * s, 0.0f};
    };
    return cell(env) - cell(current);
}

void EditorHost::State::SwitchEnv(uint32_t env) {
    const EnvView& v = ui_state_.envs;
    const uint32_t current = ui_state_.env_index % v.count;
    env %= v.count;
    if (env == current) return;
    if (env / v.show == current / v.show)
        camera_.SetView(camera_.ResolvedTarget() - EnvOffset(env, current), camera_.Distance(), camera_.Yaw(),
                        camera_.Pitch());
    ui_state_.env_index = env;
    PushTransport();
    worker_.Exclusive([this] { LoadDriveTargets(); });
}

json::Value EditorHost::State::EnvsJson() const {
    const EnvView& v = ui_state_.envs;
    const uint32_t current = ui_state_.env_index % v.count;
    // The current page's cells with the offset each draws at; the current one's is zero.
    json::Value cells = json::Value::Array();
    std::vector<uint32_t> page = OtherShownEnvs(current);
    page.insert(std::lower_bound(page.begin(), page.end(), current), current);
    for (const uint32_t e : page)
        cells.PushBack(Object({{"env", json::Value::Int(e)}, {"offset", Vec3Json(EnvOffset(e, current))}}));
    return Object({{"count", json::Value::Int(v.count)},
                   {"current", json::Value::Int(current)},
                   {"show", json::Value::Int(v.show)},
                   {"spacing", json::Value::Float(v.spacing > 0.0f ? v.spacing : env_auto_spacing_)},
                   {"auto_spacing", json::Value::Bool(v.spacing <= 0.0f)},
                   {"cells", std::move(cells)},
                   {"copies", json::Value::Int(static_cast<int64_t>(loaded_ ? loaded_->sim->EnvCopies().size() : 0u))}});
}

void EditorHost::State::SizeLinks() {
    std::vector<RobotLink>& links = ui_state_.robot.links;
    const render::RenderWorld& w = loaded_->sim->GetRenderWorld();
    const float fmax = std::numeric_limits<float>::max();
    std::vector<math::Vec3> lo(links.size(), math::Vec3{fmax, fmax, fmax}), hi(links.size(), math::Vec3{-fmax, -fmax, -fmax});
    for (const render::RenderInstance& inst : w.instances) {
        const uint32_t l = inst.pose_source.row;
        if (inst.pose_source.kind != render::PoseSource::Kind::Link || l >= links.size() || inst.mesh_id == render::kNoId)
            continue;
        const std::vector<float>& p = w.meshes.Geometry(inst.mesh_id).positions;
        for (size_t v = 0; v + 2u < p.size(); v += 3u) {
            const math::Vec3 x = inst.cached_visual_local.TransformPoint(math::Vec3{p[v], p[v + 1u], p[v + 2u]});
            lo[l] = math::Vec3{std::min(lo[l].x, x.x), std::min(lo[l].y, x.y), std::min(lo[l].z, x.z)};
            hi[l] = math::Vec3{std::max(hi[l].x, x.x), std::max(hi[l].y, x.y), std::max(hi[l].z, x.z)};
        }
    }
    for (size_t l = 0; l < links.size(); ++l)
        links[l].size = lo[l].x > hi[l].x ? 0.0f : std::max({hi[l].x - lo[l].x, hi[l].y - lo[l].y, hi[l].z - lo[l].z});
}

// Drops the zero-copy pose path; the GPU must be idle.
void EditorHost::State::DropPublisher() {
    present_->SetInteropTransforms(nullptr);
    scatter_.reset();  // frees CUDA's imports, synchronizing the device
    interop_.reset();
    frame_pins_.clear();
}

// Binds the worker to the loaded scene from `tick` on. Poses publish zero-copy when the
// renderer and the world's CUDA device share pose slots, otherwise through host downloads.
void EditorHost::State::BindWorker(uint64_t tick, double sim_time) {
    present_->WaitIdle();
    DropPublisher();
    const render::RenderWorld& rw = loaded_->sim->GetRenderWorld();
    const render::RendererVulkanHandles vk = present_->VulkanHandles();
    publish_reason_.clear();
    if (opt_.publish == "host") {
        publish_reason_ = "--publish host";
    } else if (!vk.interop_supported) {
        publish_reason_ = "the Vulkan device cannot export memory or timeline semaphores";
    } else {
        try {
            auto interop = std::make_unique<render::InteropTransformSsbo>();
            interop->Create(reinterpret_cast<VkDevice>(vk.device), reinterpret_cast<VkPhysicalDevice>(vk.physical_device),
                            std::max(rw.InstanceCount(), 1u), present_->FramesInFlight() + kPoseSlotsBeyondFrames);
            const render::InteropExport exported = interop->TakeExport();
            PoseSlotImport slots;
            slots.memory_handle = exported.memory_handle;
            slots.memory_fd = exported.memory_fd;
            slots.memory_bytes = exported.memory_bytes;
            slots.semaphore_handle = exported.semaphore_handle;
            slots.semaphore_fd = exported.semaphore_fd;
            slots.slot_bytes = exported.slot_bytes;
            std::memcpy(slots.device_uuid, vk.device_uuid, sizeof(slots.device_uuid));
            auto scatter = std::make_unique<PoseScatter>(world_, slots);
            scatter->SetRows(world_, BuildScatterRows(rw));
            present_->SetInteropTransforms(reinterpret_cast<render::NkVkDescriptorSetLayout>(interop->SetLayout()));
            worker_.Slots().Reset(interop->SlotCount());
            interop_ = std::move(interop);
            scatter_ = std::move(scatter);
        } catch (const std::exception& e) {
            publish_reason_ = e.what();
        }
    }
    Emit(stdout, Format("[nuka_editor] publish=%s%s%s\n", scatter_ ? "zero-copy" : "host", scatter_ ? "" : " reason=",
                        publish_reason_.c_str()));

    // The document classifies edits against this world's parameter registry.
    uint32_t count = 0u;
    CheckResult(nuka_world_param_count(world_, &count), "nuka_world_param_count");
    std::vector<nuka_param_info_t> infos(count);
    for (nuka_param_info_t& info : infos) {
        info.struct_size = sizeof(info);
        info.schema_version = NUKA_PARAM_INFO_VERSION;
    }
    CheckResult(nuka_world_get_param_infos(world_, infos.data(), count), "nuka_world_get_param_infos");
    std::vector<double> values(count, 0.0);
    for (uint32_t i = 0; i < count; ++i)
        if (infos[i].scene_field[0] == '\0')
            CheckResult(nuka_world_get_param(world_, i, 0u, &values[i]), "nuka_world_get_param");
    doc_.SetParams(std::move(infos), std::move(values));

    const WorldLayout& layout = loaded_->Layout();
    WorkerBinding binding;
    binding.world = world_;
    binding.layout = layout;
    binding.rest_targets = loaded_->rest_targets;
    binding.controller = policy_on_ ? policy_ctrl_.get() : nullptr;
    binding.scatter = scatter_.get();
    EnvView& envs = ui_state_.envs;
    envs.count = layout.env_count;
    envs.show = std::min(envs.show, envs.count);
    math::Vec3 lo, hi;
    SceneAabb(rw, /*movable_only=*/true, &lo, &hi);
    env_auto_spacing_ = std::max(1.0f, 1.5f * std::max(hi.x - lo.x, hi.y - lo.y));
    binding.env = ui_state_.env_index % layout.env_count;
    binding.shown = OtherShownEnvs(binding.env);
    binding.dt = opt_.dt;
    binding.speed = ui_state_.speed;
    binding.colliders = ui_state_.show_colliders;
    binding.contacts = ui_state_.show_contacts;
    binding.tick = tick;
    binding.sim_time = sim_time;
    sent_playing_ = false;
    sent_env_ = binding.env;
    sent_shown_ = binding.shown;
    sent_speed_ = binding.speed;
    sent_colliders_ = binding.colliders;
    sent_contacts_ = binding.contacts;
    sent_static_revision_ = loaded_->sim->StaticRevision();
    // A rebuild ends a demonstration recording and replay; what it recorded stays.
    if (demo_recording_) {
        CutEpisodes(worker_.TakeDemo(), &demos_);
        demo_recording_ = false;
        SyncDemoView();
    }
    ui_state_.demos.playing = -1;
    ui_state_.demos.left = 0u;
    teleop_command_sent_.reset();
    bound_epoch_ = worker_.Bind(std::move(binding));
    ui_state_.robot.ik.pending = 0u;
    ui_state_.robot.ik.link = -1;
    ik_drag_ = false;
    ik_next_.reset();
    shown_ = nullptr;
    // A new binding starts paused; the transport re-sends the UI's play state.
    if (const Snapshot* snap = worker_.AcquireSnapshot()) ApplySnapshot(*snap);
}

// Shows a snapshot of the current binding; one the UI already superseded only counts.
void EditorHost::State::ApplySnapshot(const Snapshot& snap) {
    if (last_sequence_ != 0u && snap.sequence > last_sequence_ + 1u)
        snapshot_drops_ += snap.sequence - last_sequence_ - 1u;
    last_sequence_ = snap.sequence;
    shown_ = loaded_ && snap.epoch == bound_epoch_ ? &snap : nullptr;
    ui_state_.held.clear();
    if (shown_ == nullptr) return;
    loaded_->sim->ApplyPoses(snap.link_poses, snap.body_poses);
    loaded_->sim->ApplyGhost(snap.ghost_links, snap.ghost_bodies);
    loaded_->sim->ClearEnvs();
    for (const PoseSample& e : snap.envs) loaded_->sim->AddEnv(e.env, e.links, e.bodies, EnvOffset(e.env, snap.env));
    if (!snap.particles.empty()) loaded_->sim->ApplyParticles(snap.particles, snap.heat, ui_state_.solver.alert);
    ui_state_.robot.live = snap.joints;
    RecordPolicyNorm(snap);
    DemoView& demo = ui_state_.demos;
    demo.samples = worker_.DemoCount();
    if (snap.acked >= demo_play_seq_) {
        demo.left = snap.track_left;
        if (demo.left == 0u) demo.playing = -1;
    }
    ui_state_.robot.link_poses = snap.link_poses;
    // A spring draws as a band to its target, a plain force as an arrow along it.
    for (const Perturbation& p : snap.perturbs) {
        math::Vec3 at = p.target, target = p.target;
        if (!LoadPoint(snap, p, &at)) continue;
        HeldLoad h;
        if (p.particle == kNoParticle) {
            const math::Vec3 f = math::Vec3{p.force[0], p.force[1], p.force[2]} + (p.target - at) * p.stiffness;
            h.body = true;
            h.force = f.Length();
            if (p.stiffness == 0.0f) target = at + f * (kForceArrow / std::max(h.force, 1e-6f));
        }
        const float a[3] = {at.x, at.y, at.z}, t[3] = {target.x, target.y, target.z};
        std::copy(a, a + 3, h.at);
        std::copy(t, t + 3, h.target);
        ui_state_.held.push_back(h);
    }
    last_telemetry_ = snap.telemetry;
    last_tick_ = snap.tick;
    last_step_healthy_ = snap.healthy;
    // Once the worker saw the last transport command and none is pending, its state is the
    // truth (it pauses itself on a failed tick).
    if (snap.acked >= transport_seq_ && ui_state_.playing == sent_playing_)
        ui_state_.playing = sent_playing_ = snap.playing;
    if (reset_framing_ && snap.acked >= reset_seq_) {
        reset_framing_ = false;
        FrameWorld(loaded_->sim->GetRenderWorld());
    }
}

void EditorHost::State::SyncSolver() {
    SolverView& v = ui_state_.solver;
    std::vector<nuka_telemetry_record_t> fresh;
    SolverStats stats = worker_.ReadSolver(v.first + v.history.size(), &fresh);
    if (stats.session != v.session) {
        v.history.clear();
        v.first = 0u;
        v.session = stats.session;
        stats = worker_.ReadSolver(0u, &fresh);
    }
    // Records the worker's log dropped before this frame read them leave a gap.
    const uint64_t from = stats.records - fresh.size();
    if (from != v.first + v.history.size()) {
        v.history.clear();
        v.first = from;
    }
    v.history.insert(v.history.end(), fresh.begin(), fresh.end());
    if (v.history.size() > kSolverHistory) {
        const size_t drop = v.history.size() - kSolverHistory;
        v.history.erase(v.history.begin(), v.history.begin() + static_cast<std::ptrdiff_t>(drop));
        v.first += drop;
    }
    v.mask = stats.mask;
    v.alert = stats.alert;
    v.iterations = stats.iterations;
    v.records = stats.records;
    v.caps = stats.caps;
    v.recent_caps = stats.recent_caps;
    v.recent_records = static_cast<uint32_t>(std::min<uint64_t>(stats.records, kRecentRecords));
    v.dropped = stats.dropped;
}

void EditorHost::State::SyncJoints() {
    RobotView& v = ui_state_.robot;
    // Live edits change gains without a rebuild.
    if (doc_.Revision() != joints_revision_) {
        joints_revision_ = doc_.Revision();
        v.joints = RobotJoints(loaded_->scene, loaded_->Layout(), [this](scene::EntityId e) { return PathOfEntity(e); });
        v.links = RobotLinks(loaded_->scene, loaded_->Layout(), [this](scene::EntityId e) { return PathOfEntity(e); });
        SizeLinks();
    }
    std::vector<JointSample> fresh;
    uint64_t session = 0u;
    uint64_t count = worker_.ReadJoints(v.first + v.history.size(), &fresh, &session);
    if (session != v.session) {
        v.history.clear();
        v.first = 0u;
        v.session = session;
        count = worker_.ReadJoints(0u, &fresh, &session);
    }
    // Samples the worker's log dropped before this frame read them leave a gap.
    const uint64_t from = count - fresh.size();
    if (from != v.first + v.history.size()) {
        v.history.clear();
        v.first = from;
    }
    for (JointSample& s : fresh) v.history.push_back(std::move(s));
    while (v.history.size() > kJointHistory) {
        v.history.pop_front();
        ++v.first;
    }
    const std::string selected =
        ui_state_.selected_entity == scene::kInvalidEntity ? std::string() : PathOfEntity(ui_state_.selected_entity);
    v.focus = -1;
    for (size_t i = 0; i < v.joints.size(); ++i)
        if (v.joints[i].path == selected) v.focus = static_cast<int>(i);
}

void EditorHost::State::SyncGainTest() {
    GainTuner& g = ui_state_.robot.tuner;
    GainTrace trace;
    if (!worker_.TakeGainTrace(&trace) || trace.request != g.running) return;
    GainRun run = std::move(g.next);
    run.metrics = FitGain(trace);
    run.trace = std::move(trace);
    g.runs.push_back(std::move(run));
    while (g.runs.size() > kGainRuns) g.runs.pop_front();
    g.running = 0u;
}

void EditorHost::State::SyncAssembler() {
    Assembler& a = ui_state_.robot.assembler;
    if (!a.open) return;
    a.selected = ui_state_.selected_entity == scene::kInvalidEntity ? std::string() : PathOfEntity(ui_state_.selected_entity);
    const auto body = [this](const std::string& path) {
        const auto node = loaded_->scene.Tree().NodeOf(path);
        const auto& bodies = doc_.Records().body;
        const auto it = node ? bodies.find(node->entity) : bodies.end();
        return it == bodies.end() ? scene::kInvalidBody : it->second;
    };
    const scene::BodyId child = body(a.child), parent = body(a.parent);
    a.posed = !a.from_asset && child != scene::kInvalidBody && parent != scene::kInvalidBody;
    if (!a.posed) return;
    const math::Transform offset = BodyWorld(loaded_->scene, parent).Inverse() * BodyWorld(loaded_->scene, child);
    const math::Vec3 rot = EulerDegFromQuat(offset.rotation);
    const float values[6] = {offset.position.x, offset.position.y, offset.position.z, rot.x, rot.y, rot.z};
    std::copy(values, values + 3, a.offset_pos);
    std::copy(values + 3, values + 6, a.offset_rot);
}

int EditorHost::State::FindLink(const std::string& name) const {
    const std::vector<RobotLink>& links = ui_state_.robot.links;
    for (size_t i = 0; i < links.size(); ++i)
        if (links[i].path == name || (!links[i].joint.empty() && links[i].joint == name)) return static_cast<int>(i);
    return -1;
}

int EditorHost::State::IkLink() const {
    const RobotView& v = ui_state_.robot;
    if (!v.ik.on || shown_ == nullptr || ui_state_.selected_entity == scene::kInvalidEntity) return -1;
    const int link = FindLink(PathOfEntity(ui_state_.selected_entity));
    return link >= 0 && v.links[static_cast<size_t>(link)].parent >= 0 ? link : -1;
}

void EditorHost::State::TakeIkReport(const IkReport& report) {
    IkState& ik = ui_state_.robot.ik;
    if (report.request >= ik.pending) ik.pending = 0u;
    for (size_t l = 0; l < std::min(report.targets.size(), ui_state_.drive_targets.size()); ++l)
        if (ui_state_.drive_dirty[l] == 0u) ui_state_.drive_targets[l] = report.targets[l];
    ik.error = report.error;
    if (ik.link < 0 || report.position_error.empty()) return;
    ik.error_mm = report.position_error[0] * 1000.0f;
    ik.error_deg = report.rotation_error[0] * 180.0f / kPi;
    ik.missed = !report.error.empty() || ik.error_mm > kIkMissMm || ik.error_deg > kIkMissDeg;
}

void EditorHost::State::SyncIk() {
    IkState& ik = ui_state_.robot.ik;
    IkReport report;
    if (worker_.TakeIk(&report)) TakeIkReport(report);
    if (ik.pending == 0u && ik_next_) {
        ik.pending = worker_.Push(std::move(*ik_next_));
        ik_next_.reset();
    }
    if (ik.link < 0 || shown_ == nullptr) return;
    const math::Vec3 at = shown_->link_poses[static_cast<size_t>(ik.link)].position;
    const float p[3] = {at.x, at.y, at.z};
    std::copy(p, p + 3, ik.at);
}

void EditorHost::State::DragIk(int link, const math::Transform& live, const math::Transform& gizmo, bool changed,
                               bool using_now) {
    IkState& ik = ui_state_.robot.ik;
    if (using_now && !ik_drag_) {
        const std::vector<math::Transform>& poses = shown_->link_poses;
        ik_drag_ = true;
        ik.link = link;
        ik.missed = false;
        ik_press_ = poses[static_cast<size_t>(link)];
        ik_offset_ = live.Inverse() * ik_press_;
        ik_held_ = live;
        ik_pins_.clear();
        for (const std::string& pin : ik.pins)
            if (const int p = FindLink(pin); p >= 0 && p != link)
                ik_pins_.emplace_back(static_cast<uint32_t>(p), poses[static_cast<size_t>(p)]);
    }
    if (ik_drag_ && changed) {
        // A move takes the link's origin with the gizmo, a turn its orientation about where it is.
        ik_held_ = gizmo;
        const math::Transform goal = gizmo * ik_offset_;
        const bool turn = ui_state_.gizmo.op == GizmoState::Op::Rotate;
        math::Transform aim;
        aim.position = turn ? ik_press_.position : goal.position;
        aim.rotation = turn ? goal.rotation : ik_press_.rotation;
        const auto l = static_cast<uint32_t>(link);
        std::vector<nuka_ik_target_t> targets = {IkGoal(l, aim, turn, 1.0f)};
        if (!turn && ik.lock_orientation) targets.push_back(IkGoal(l, aim, true, kIkLockWeight));
        for (const auto& [p, pose] : ik_pins_) targets.push_back(IkGoal(p, pose, true, 1.0f));
        ik_next_ = IkCommand(std::move(targets), FindLink(ik.root), ik.rotation_weight, shown_->env);
        const float t[3] = {aim.position.x, aim.position.y, aim.position.z};
        std::copy(t, t + 3, ik.target);
    }
    if (using_now) return;
    ik_drag_ = false;
    ik.link = -1;
    ik.missed = false;
}

void EditorHost::State::ListTeleopDevices() {
    TeleopView& t = ui_state_.teleop;
    t.devices.clear();
    t.devices.push_back({"keyboard", "Keyboard", "keyboard", "", kPadAxes, kPadButtons});
    t.devices.push_back({"virtual", "Virtual device", "virtual", "", kPadAxes, kPadButtons});
    pads_.Update(&t.devices);
    extensions_.TeleopEntries(&t.devices, &t.retargets);
}

int EditorHost::State::TeleopLink() const {
    const TeleopView& t = ui_state_.teleop;
    const int link = FindLink(t.link.empty() ? PathOfEntity(ui_state_.selected_entity) : t.link);
    return link >= 0 && ui_state_.robot.links[static_cast<size_t>(link)].parent >= 0 ? link : -1;
}

bool EditorHost::State::StepTeleop(double dt) {
    TeleopView& t = ui_state_.teleop;
    const ImGuiIO& io = ImGui::GetIO();
    ListTeleopDevices();
    const auto found =
        std::find_if(t.devices.begin(), t.devices.end(), [&](const TeleopDeviceInfo& d) { return d.id == t.device; });
    t.connected = found != t.devices.end();
    const TeleopDeviceInfo device = t.connected ? *found : TeleopDeviceInfo{t.device, t.device, "", "", {}, {}};
    t.axes.assign(device.axes.size(), 0.0f);
    t.buttons.assign(device.buttons.size(), 0u);
    // Ctrl chords belong to shortcuts (Ctrl+S, Ctrl+=), so they never drive the robot.
    const bool keys = t.on && t.device == "keyboard" && !io.WantCaptureKeyboard && !io.KeyCtrl && !io.WantTextInput &&
                      ui_.OwnsScenePointer() && ui_state_.window_focused;
    if (t.device == "keyboard") {
        const auto key = [&](ImGuiKey k) { return keys && ImGui::IsKeyDown(k) ? 1.0f : 0.0f; };
        t.axes[kLeftX] = key(ImGuiKey_D) - key(ImGuiKey_A);
        t.axes[kLeftY] = key(ImGuiKey_S) - key(ImGuiKey_W);
        t.axes[kRightX] = key(ImGuiKey_E) - key(ImGuiKey_Q);
        t.axes[kRightY] = key(ImGuiKey_PageDown) - key(ImGuiKey_PageUp);
        t.buttons[kLeftShoulder] = static_cast<uint8_t>(key(ImGuiKey_Minus) != 0.0f);
        t.buttons[kRightShoulder] = static_cast<uint8_t>(key(ImGuiKey_Equal) != 0.0f);
    } else if (t.device == "virtual") {
        t.axes = virtual_axes_;
        t.buttons = virtual_buttons_;
    } else if (t.connected) {
        t.connected = device.owner.empty() ? pads_.Read(t.device, &t.axes, &t.buttons)
                                           : extensions_.ReadTeleopDevice(t.device, &t.axes, &t.buttons);
    }
    t.quantities = TeleopQuantities(device, t.axes, t.buttons, t.maps[t.mode], ModeQuantities(t.mode), t.deadzone);
    const auto q = [&](const char* name) {
        const auto it = t.quantities.find(name);
        return it != t.quantities.end() ? it->second : 0.0f;
    };
    const bool live = t.on && loaded_ != nullptr;
    const float step = static_cast<float>(dt);
    t.error.clear();

    // A command consumer gets the command when it changes, so an unchanged one logs nothing; off, it is zero.
    const bool cmd_ctrl = loaded_ && policy_ctrl_ && policy_on_ && policy_ctrl_->AcceptsCommand();
    t.controller = cmd_ctrl;
    const bool command = live && t.mode == "command";
    const float drive[3] = {q("vx"), q("vy"), q("wyaw")};
    for (int k = 0; k < 3; ++k) t.command[k] = command ? t.command_base[k] + drive[k] : 0.0f;
    const std::array<float, 3> sent = {t.command[0], t.command[1], t.command[2]};
    if (cmd_ctrl && teleop_command_sent_ != sent) {
        worker_.Push(TickCommand::Param(ParamKind::ControllerCommand, {t.command[0], t.command[1], t.command[2]}));
        std::copy(t.command, t.command + 3, ui_state_.policy.command);
        teleop_command_sent_ = sent;
    }
    // A demonstration records the device's axes under the names its set keeps.
    if (demo_recording_) {
        std::vector<float> axes(demos_.axes.size(), 0.0f);
        for (size_t a = 0; a < axes.size(); ++a) {
            const auto at = std::find(device.axes.begin(), device.axes.end(), demos_.axes[a]);
            if (at != device.axes.end()) axes[a] = t.axes[static_cast<size_t>(at - device.axes.begin())];
        }
        if (axes != demo_axes_) worker_.Push(TickCommand::Param(ParamKind::TeleopAxes, axes));
        demo_axes_ = std::move(axes);
    }
    // PD hold's rising edge makes the drive targets act without a policy.
    const bool hold = live && t.hold && !(command && cmd_ctrl);
    if (hold && !teleop_hold_prev_) worker_.Push(TickCommand::Param(ParamKind::HoldGains, {t.kp, t.kd}));
    teleop_hold_prev_ = hold;

    const std::vector<RobotJoint>& joints = ui_state_.robot.joints;
    const bool editable = !joints.empty() && ui_state_.drive_targets.size() == ui_state_.drive_dirty.size();
    const float next = q("next"), prev = q("prev");
    if (live && t.mode == "joint" && editable) {
        const int n = static_cast<int>(joints.size());
        if (next > 0.5f && teleop_press_[0] <= 0.5f) ++t.joint;
        if (prev > 0.5f && teleop_press_[1] <= 0.5f) --t.joint;
        t.joint = (t.joint % n + n) % n;
        const RobotJoint& j = joints[static_cast<size_t>(t.joint)];
        if (const float rate = q("rate"); rate != 0.0f) {
            const bool limited = j.limited && j.upper > j.lower;
            float& target = ui_state_.drive_targets[j.link];
            target = std::clamp(target + rate * step, limited ? j.lower : -FLT_MAX, limited ? j.upper : FLT_MAX);
            ui_state_.drive_dirty[j.link] = 1u;
        }
    }
    teleop_press_[0] = next;
    teleop_press_[1] = prev;

    // The ik mode moves the goal at the mapped velocity and solves toward it while it moves.
    const int link = live && t.mode == "ik" && shown_ != nullptr ? TeleopLink() : -1;
    if (live && t.mode == "ik" && link < 0) t.error = "select a robot link below its root";
    if (link < 0) t.goal_set = false;
    if (link >= 0) {
        const std::vector<math::Transform>& poses = shown_->link_poses;
        const IkState& ik = ui_state_.robot.ik;
        if (!t.goal_set || link != teleop_ik_link_) {
            teleop_ik_link_ = link;
            teleop_ik_start_ = poses[static_cast<size_t>(link)];
            const math::Vec3 at = teleop_ik_start_.position;
            const float g[3] = {at.x, at.y, at.z};
            std::copy(g, g + 3, t.goal);
            t.goal_set = true;
            teleop_ik_pins_.clear();
            for (const std::string& pin : ik.pins)
                if (const int p = FindLink(pin); p >= 0 && p != link)
                    teleop_ik_pins_.emplace_back(static_cast<uint32_t>(p), poses[static_cast<size_t>(p)]);
        }
        const float v[3] = {q("x"), q("y"), q("z")};
        if (v[0] != 0.0f || v[1] != 0.0f || v[2] != 0.0f) {
            for (int k = 0; k < 3; ++k) t.goal[k] += v[k] * step;
            math::Transform aim = teleop_ik_start_;
            aim.position = math::Vec3{t.goal[0], t.goal[1], t.goal[2]};
            const auto l = static_cast<uint32_t>(link);
            std::vector<nuka_ik_target_t> targets = {IkGoal(l, aim, false, 1.0f)};
            if (ik.lock_orientation) targets.push_back(IkGoal(l, aim, true, kIkLockWeight));
            for (const auto& [p, pose] : teleop_ik_pins_) targets.push_back(IkGoal(p, pose, true, 1.0f));
            ik_next_ = IkCommand(std::move(targets), FindLink(ik.root), ik.rotation_weight, shown_->env);
        }
    }

    // An extension's mode maps the device and the joints to joint targets.
    const bool builtin = std::find(kTeleopModes.begin(), kTeleopModes.end(), t.mode) != kTeleopModes.end();
    if (live && !builtin && editable) {
        json::Value axes = json::Value::Object(), buttons = json::Value::Object(), quantities = json::Value::Object();
        for (size_t i = 0; i < device.axes.size(); ++i) axes.Set(device.axes[i], json::Value::Float(t.axes[i]));
        for (size_t i = 0; i < device.buttons.size(); ++i) buttons.Set(device.buttons[i], json::Value::Bool(t.buttons[i] != 0u));
        for (const auto& [name, value] : t.quantities) quantities.Set(name, json::Value::Float(value));
        const std::vector<float>& qs = ui_state_.robot.live.q;
        json::Value list = json::Value::Array();
        for (const RobotJoint& j : joints)
            list.PushBack(Object({{"name", json::Value::Str(j.name)},
                                  {"q", json::Value::Float(j.link < qs.size() ? qs[j.link] : 0.0f)},
                                  {"target", json::Value::Float(ui_state_.drive_targets[j.link])}}));
        const json::Value state =
            Object({{"device", Object({{"id", json::Value::Str(t.device)}, {"axes", axes}, {"buttons", buttons}})},
                    {"quantities", quantities},
                    {"joints", list},
                    {"dt", json::Value::Float(dt)}});
        std::string out;
        if (!extensions_.Retarget(t.mode, state.Dump(0), &out)) {
            t.error = out;
        } else {
            try {
                const json::Value targets = json::Value::Parse(out);
                for (const auto& [name, value] : targets.Items()) {
                    const auto j = std::find_if(joints.begin(), joints.end(), [&](const RobotJoint& r) { return r.name == name; });
                    if (j == joints.end()) {
                        t.error = "teleop mode '" + t.mode + "' set no joint '" + name + "'";
                        continue;
                    }
                    ui_state_.drive_targets[j->link] = value.AsFloat();
                    ui_state_.drive_dirty[j->link] = 1u;
                }
            } catch (const std::exception& e) {
                t.error = "teleop mode '" + t.mode + "': " + e.what();
            }
        }
    }
    return keys;
}

json::Value EditorHost::State::TeleopStatus() const {
    const TeleopView& t = ui_state_.teleop;
    const auto found =
        std::find_if(t.devices.begin(), t.devices.end(), [&](const TeleopDeviceInfo& d) { return d.id == t.device; });
    json::Value axes = json::Value::Object(), buttons = json::Value::Object(), quantities = json::Value::Object();
    if (found != t.devices.end()) {
        for (size_t i = 0; i < std::min(found->axes.size(), t.axes.size()); ++i)
            axes.Set(found->axes[i], json::Value::Float(t.axes[i]));
        for (size_t i = 0; i < std::min(found->buttons.size(), t.buttons.size()); ++i)
            buttons.Set(found->buttons[i], json::Value::Bool(t.buttons[i] != 0u));
    }
    for (const auto& [name, value] : t.quantities) quantities.Set(name, json::Value::Float(value));
    const std::vector<RobotJoint>& joints = ui_state_.robot.joints;
    const int link = TeleopLink();
    const int n = static_cast<int>(joints.size());
    return Object({{"on", json::Value::Bool(t.on)},
                   {"device", json::Value::Str(t.device)},
                   {"connected", json::Value::Bool(t.connected)},
                   {"mode", json::Value::Str(t.mode)},
                   {"axes", std::move(axes)},
                   {"buttons", std::move(buttons)},
                   {"quantities", std::move(quantities)},
                   {"command", Vec3Json(math::Vec3{t.command[0], t.command[1], t.command[2]})},
                   {"command_base", Vec3Json(math::Vec3{t.command_base[0], t.command_base[1], t.command_base[2]})},
                   {"controller", json::Value::Bool(t.controller)},
                   {"joint", n == 0 ? json::Value() : json::Value::Str(joints[static_cast<size_t>(std::clamp(t.joint, 0, n - 1))].name)},
                   {"link", link < 0 ? json::Value() : json::Value::Str(ui_state_.robot.links[static_cast<size_t>(link)].path)},
                   {"goal", t.goal_set ? Vec3Json(math::Vec3{t.goal[0], t.goal[1], t.goal[2]}) : json::Value()},
                   {"hold", json::Value::Bool(t.hold)},
                   {"kp", json::Value::Float(t.kp)},
                   {"kd", json::Value::Float(t.kd)},
                   {"deadzone", json::Value::Float(t.deadzone)},
                   {"error", json::Value::Str(t.error)}});
}

void EditorHost::State::TakeWorkerResults() {
    for (const TickResult& r : worker_.TakeResults()) {
        if (r.error) std::fputs(r.text.c_str(), stderr);
        if (r.seq != 0u && r.seq == param_seq_) {
            param_pushed_tick_ = r.pushed_tick;
            param_applied_tick_ = r.applied_tick;
            param_done_ = true;
            if (r.error) last_error_ = r.text;
        }
        if (r.seq != 0u && r.seq == run_script_seq_) {
            std::fputs(r.text.c_str(), stderr);
            std::fflush(stderr);
            Emit(stdout, Format("[nuka_editor] ran --run-script %s (%zu bytes captured)\n", opt_.run_script_path.c_str(),
                                r.text.size()));
            std::fflush(stdout);
        }
        const auto run = script_runs_.find(r.seq);
        if (run != script_runs_.end()) {
            run->second.done = true;
            run->second.failed = r.error;
            run->second.output = r.text;
            if (!run->second.name.empty()) ui_state_.scripts.Mark(r.text, r.error);
        }
        if (!r.streamed) ConsoleAppend(r.text);
    }
}

uint64_t EditorHost::State::RunScript(const std::string& code, const std::string& name, bool echo) {
    const uint64_t seq = worker_.Push(TickCommand::Script(code, name, echo));
    ScriptRun& run = script_runs_[seq];
    run.name = name;
    while (script_runs_.size() > kScriptRunsKept) script_runs_.erase(script_runs_.begin());
    return seq;
}

ScriptTab& EditorHost::State::TabArg(const json::Value& args) {
    if (args.Find("tab") == nullptr) return ui_state_.scripts.Active();
    const uint64_t id = static_cast<uint64_t>(args.At("tab").AsInt());
    ScriptTab* tab = ui_state_.scripts.Find(id);
    if (tab == nullptr) throw std::runtime_error("no script tab " + std::to_string(id));
    return *tab;
}

// Transport and display changes become worker commands, after the moves made before them.
void EditorHost::State::PushTransport() {
    if (!loaded_) return;
    PushPoseWrites();
    if (ui_state_.speed != sent_speed_) {
        TickCommand cmd = TickCommand::Transport(TransportOp::Speed);
        cmd.speed = ui_state_.speed;
        worker_.Push(std::move(cmd));
        sent_speed_ = ui_state_.speed;
    }
    if (ui_state_.playing != sent_playing_) {
        transport_seq_ = worker_.Push(TickCommand::Transport(ui_state_.playing ? TransportOp::Play : TransportOp::Pause));
        sent_playing_ = ui_state_.playing;
    }
    const uint32_t env = ui_state_.env_index % loaded_->Layout().env_count;
    std::vector<uint32_t> shown = OtherShownEnvs(env);
    if (env != sent_env_ || shown != sent_shown_) {
        std::vector<float> values;
        for (const uint32_t e : shown) values.push_back(static_cast<float>(e));
        TickCommand cmd = TickCommand::Param(ParamKind::DisplayEnv, std::move(values));
        cmd.env = env;
        worker_.Push(std::move(cmd));
        sent_env_ = env;
        sent_shown_ = std::move(shown);
    }
    if (ui_state_.show_colliders != sent_colliders_ || ui_state_.show_contacts != sent_contacts_) {
        worker_.Push(TickCommand::Param(ParamKind::Overlay, {ui_state_.show_colliders ? 1.0f : 0.0f,
                                                             ui_state_.show_contacts ? 1.0f : 0.0f}));
        sent_colliders_ = ui_state_.show_colliders;
        sent_contacts_ = ui_state_.show_contacts;
    }
}

void EditorHost::State::PushPoseWrites() {
    for (const PoseWrite& write : loaded_->sim->TakePoseWrites()) {
        TickCommand cmd;
        cmd.kind = CommandKind::EditApply;
        cmd.pose = write;
        worker_.Push(std::move(cmd));
    }
}

// Live edits of this frame become worker commands: teleports, static rows, drive targets.
void EditorHost::State::PushEdits() {
    if (!loaded_) return;
    PushPoseWrites();
    if (scatter_ && loaded_->sim->StaticRevision() != sent_static_revision_) {
        TickCommand cmd = TickCommand::Param(ParamKind::ScatterRows, {});
        cmd.rows = BuildScatterRows(loaded_->sim->GetRenderWorld());
        worker_.Push(std::move(cmd));
        sent_static_revision_ = loaded_->sim->StaticRevision();
    }
    // Moved drive sliders write only their own rows of the selected environment.
    if (ui_state_.drive_dirty.size() == ui_state_.drive_targets.size()) {
        const WorldLayout& layout = loaded_->Layout();
        const size_t env = ui_state_.env_index % layout.env_count;
        for (size_t d = 0; d < ui_state_.drive_targets.size(); ++d) {
            if (!ui_state_.drive_dirty[d]) continue;
            TickCommand cmd = TickCommand::Param(ParamKind::Field, {ui_state_.drive_targets[d]});
            cmd.field = NUKA_FIELD_DRIVE_TARGET;
            cmd.offset = (env * layout.links_per_env + d) * sizeof(float);
            worker_.Push(std::move(cmd));
            ui_state_.drive_dirty[d] = 0u;
        }
    }
}

// Draws with the latest zero-copy pose slot when there is one; a slot stays pinned until
// the frames in flight after its frame were submitted.
void EditorHost::State::DrawWorld(const render::RenderWorld& rw, const std::vector<render::RasterOptions>& views) {
    render::PresentInterop interop;
    uint32_t slot = 0u;
    const bool pinned = PinPose(&slot, &interop);
    const uint64_t submitted = present_->SubmittedFrames();
    const render::PresentFrameResult r = present_->DrawViews(rw, views, [this](void* cmd) {
        imgui_.RenderDrawData(reinterpret_cast<NukaVkCommandBuffer>(cmd));
    }, pinned ? &interop : nullptr);
    if (present_->SubmittedFrames() != submitted) {
        frame_pins_.push_back(pinned ? slot : ~0u);
        while (frame_pins_.size() > present_->FramesInFlight()) {
            if (frame_pins_.front() != ~0u) worker_.Slots().Release(frame_pins_.front());
            frame_pins_.pop_front();
        }
    } else if (pinned) {
        worker_.Slots().Release(slot);
    }
    if (r == render::PresentFrameResult::Error) {
        Emit(stderr, Format("[nuka_editor] present error at frame %llu\n", static_cast<unsigned long long>(frame_index_)));
        running_ = false;
        return;
    }
    // A recreated swapchain keeps a compatible pass; ImGui only needs the image count.
    if (r == render::PresentFrameResult::Recreated) imgui_.NotifySwapchainRecreated(present_->MinImageCount());
    if (r == render::PresentFrameResult::Presented) ++presented_;
}

bool EditorHost::State::PinPose(uint32_t* slot, render::PresentInterop* interop) {
    if (scatter_ == nullptr || !worker_.Slots().AcquireRead(slot, &interop->ready_value)) return false;
    interop->set = reinterpret_cast<render::NkVkDescriptorSet>(interop_->Slot(*slot));
    interop->timeline = reinterpret_cast<render::NkVkSemaphore>(interop_->Timeline());
    return true;
}

render::RasterOptions EditorHost::State::ViewOptions(const CameraController& camera) {
    render::RasterOptions opts = ViewOptions(camera, ui_.Viewport(), ui_state_.view_mode);
    opts.instance_flags = InstanceFlags();
    return opts;
}

render::RasterOptions EditorHost::State::ViewOptions(const CameraController& camera, const SceneViewportRect& rect,
                                                     render::ViewMode mode) {
    render::RasterOptions opts;
    opts.width = present_->Report().width;
    opts.height = present_->Report().height;
    camera.WriteOptions(opts);
    opts.scene_viewport = rect.pixels;
    if (loaded_ && loaded_->scene.Environment().sky.enabled) {
        const auto background = loaded_->scene.Environment().sky.background;
        const auto byte = [](float value) {
            return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
        };
        if (std::isfinite(background.x) && std::isfinite(background.y) && std::isfinite(background.z))
            opts.background = {byte(background.x), byte(background.y), byte(background.z), 255u};
    }
    opts.view_mode = mode;
    return opts;
}

void EditorHost::State::StepRt(const CameraController& camera) {
    RtPreview& rt = rt_;
    RtView& view = ui_state_.rt;
    const SceneViewportRect& rect = ui_.Viewport();
    RtKey key;
    if (rect.Valid()) {
        const float w = static_cast<float>(rect.pixels.width), h = static_cast<float>(rect.pixels.height);
        const float scale = std::min(1.0f, kRtMaxSide / std::max(w, h));
        key.width = std::max(1u, static_cast<uint32_t>(std::lround(w * scale)));
        key.height = std::max(1u, static_cast<uint32_t>(std::lround(h * scale)));
    }
    const math::Vec3 eye = camera.ResolvedEye(), look = camera.ResolvedTarget();
    key.camera = nuka_beauty_camera_t{{eye.x, eye.y, eye.z}, {look.x, look.y, look.z}, {0.0f, 0.0f, 1.0f}, camera.fov_degrees};
    key.sequence = shown_ != nullptr ? shown_->sequence : 0u;
    // An unbound worker drops what it is asked.
    if (!loaded_) rt.inflight = 0u;
    BeautyImage image;
    if (worker_.TakeBeauty(&image) && image.request == rt.inflight) {
        rt.inflight = 0u;
        if (!image.error.empty()) {
            rt.error = image.error;
        } else if (rt.on && rt.asked.SameView(key)) {
            view.rgb = std::move(image.rgb);
            view.width = image.width;
            view.height = image.height;
            view.spp = image.spp;
            ++view.version;
            rt.shown = rt.asked;
            if (rt.asked.Same(rt.chain)) {
                rt.spp = image.spp;
                rt.done.push_back(image.spp);
            }
        }
    }
    if (rt.on && loaded_ && rt.error.empty() && key.width > 0u) {
        if (!key.Same(rt.chain)) {
            rt.chain = key;
            rt.spp = 0u;
            rt.done.clear();
        }
        if (rt.inflight == 0u && rt.spp < rt.max_spp) {
            TickCommand cmd;
            cmd.kind = CommandKind::Beauty;
            cmd.camera = key.camera;
            cmd.width = key.width;
            cmd.height = key.height;
            cmd.spp = rt.spp == 0u ? 1u : std::min(2u * rt.spp, rt.max_spp);
            rt.asked = key;
            rt.inflight = worker_.Push(std::move(cmd));
        }
    }
    view.on = rt.on;
    view.max_spp = rt.max_spp;
    view.shown = rt.on && !view.rgb.empty() && rt.shown.SameView(key);
    view.converged = view.shown && rt.shown.Same(rt.chain) && rt.spp >= rt.max_spp;
}

void EditorHost::State::StepRender() {
    if (!render_ || ProcessAlive(static_cast<uint32_t>(render_->process))) return;
    std::string text;
    rendered_ = ReadEntireFile(render_->result, &text)
                    ? json::Value::Parse(text)
                    : Object({{"ok", json::Value::Bool(false)},
                              {"path", json::Value::Str(render_->path)},
                              {"error", json::Value::Str("the render process exited without an answer")}});
    if (rendered_.At("ok").AsBool()) {
        Emit(stdout, "[nuka_editor] rendered " + render_->path + "\n");
        ui_.Nui().Toast(nui::Level::Ok, std::string(nui::Tr("Rendered")) + " " + FileName(render_->path));
    } else {
        const std::string error = rendered_.At("error").AsString();
        Emit(stderr, "[nuka_editor] render failed: " + error + "\n");
        ui_.Nui().Toast(nui::Level::Err, std::string(nui::Tr("Render failed")) + ": " + error);
    }
    render_.reset();
    ui_state_.rendering = false;
}

std::string EditorHost::State::BookmarkFile() const {
    return doc_path_.empty() ? std::string() : PathText(profile_->Dir() / "bookmarks" / (Hex(Fnv1a(doc_path_)) + ".json"));
}

CameraBookmark* EditorHost::State::FindBookmark(const std::string& name) {
    const auto it = std::find_if(bookmarks_.begin(), bookmarks_.end(), [&](const CameraBookmark& b) { return b.name == name; });
    return it == bookmarks_.end() ? nullptr : &*it;
}

void EditorHost::State::SyncBookmarks(bool save) {
    ui_state_.bookmarks.clear();
    for (const CameraBookmark& b : bookmarks_) ui_state_.bookmarks.push_back(b.name);
    const std::string file = BookmarkFile();
    if (!save || file.empty()) return;
    if (bookmarks_.empty()) {
        fs::remove(Utf8Path(file));
        return;
    }
    fs::create_directories(Utf8Path(file).parent_path());
    std::ofstream(Utf8Path(file), std::ios::binary | std::ios::trunc) << BookmarksJson(bookmarks_).Dump(2);
}

std::string EditorHost::State::PoseFile() const {
    return doc_path_.empty() ? std::string() : PathText(profile_->Dir() / "poses" / (Hex(Fnv1a(doc_path_)) + ".json"));
}

JointPose* EditorHost::State::FindPose(const std::string& name) {
    std::vector<JointPose>& poses = ui_state_.robot.poses;
    const auto it = std::find_if(poses.begin(), poses.end(), [&](const JointPose& p) { return p.name == name; });
    return it == poses.end() ? nullptr : &*it;
}

void EditorHost::State::SavePoses() {
    const std::string file = PoseFile();
    const std::vector<JointPose>& poses = ui_state_.robot.poses;
    if (file.empty()) return;
    if (poses.empty()) {
        fs::remove(Utf8Path(file));
        return;
    }
    fs::create_directories(Utf8Path(file).parent_path());
    std::ofstream(Utf8Path(file), std::ios::binary | std::ios::trunc) << PosesJson(poses).Dump(2);
}

const std::vector<std::string>& EditorHost::State::InstancePaths() {
    const render::RenderWorld& w = ShownWorld();
    if (instance_paths_.size() != w.InstanceCount() || instance_paths_version_ != ui_state_.scene_version) {
        instance_paths_.clear();
        for (const render::RenderInstance& inst : w.instances) instance_paths_.push_back(PathOfEntity(inst.entity));
        instance_paths_version_ = ui_state_.scene_version;
    }
    return instance_paths_;
}

std::vector<uint8_t> EditorHost::State::InstanceFlags() {
    const std::vector<std::string>& paths = InstancePaths();
    const std::vector<std::string> selected = SelectionPaths();
    const std::vector<std::string> hovered =
        hovered_path_.empty() ? std::vector<std::string>{} : std::vector<std::string>{hovered_path_};
    // The overlays see through the robot's meshes.
    const RobotOverlays& o = ui_state_.robot.overlays;
    const std::vector<render::RenderInstance>& instances = ShownWorld().instances;
    std::vector<uint8_t> flags(paths.size(), 0u);
    for (size_t i = 0; i < paths.size(); ++i) {
        if (UnderAny(paths[i], selected)) flags[i] |= render::kInstanceSelected;
        if (UnderAny(paths[i], hovered)) flags[i] |= render::kInstanceHovered;
        if (UnderAny(paths[i], ui_state_.locked)) flags[i] |= render::kInstanceUnpickable;
        if (o.on && o.translucent && instances[i].pose_source.kind == render::PoseSource::Kind::Link)
            flags[i] |= render::kInstanceSeeThrough;
    }
    return flags;
}

render::PickResult EditorHost::State::PickBox(const CameraController& camera, float x0, float y0, float x1, float y1) {
    const SceneViewportRect& rect = ui_.Viewport();
    const auto edge = [](float local, uint32_t size) {
        return static_cast<uint32_t>(std::clamp(local, 0.0f, static_cast<float>(size - 1u)));
    };
    const uint32_t l = edge(rect.LocalX(std::min(x0, x1)), rect.pixels.width);
    const uint32_t r = edge(rect.LocalX(std::max(x0, x1)), rect.pixels.width);
    const uint32_t t = edge(rect.LocalY(std::min(y0, y1)), rect.pixels.height);
    const uint32_t b = edge(rect.LocalY(std::max(y0, y1)), rect.pixels.height);
    const render::SceneViewport region{true, rect.pixels.x + l, rect.pixels.y + t, r - l + 1u, b - t + 1u};
    render::PresentInterop interop;
    uint32_t slot = 0u;
    const bool pinned = PinPose(&slot, &interop);
    render::PickResult pick = present_->Pick(ShownWorld(), ViewOptions(camera), region, pinned ? &interop : nullptr);
    if (pinned) worker_.Slots().Release(slot);
    // Another environment's copy picks its source instance there; boxes keep the current one's.
    const auto rows = static_cast<uint32_t>(ShownWorld().instances.size());
    picked_env_ = ui_state_.env_index % ui_state_.envs.count;
    if (pick.center != ~0u && pick.center >= rows) {
        const EditorSim::EnvCopy& copy = loaded_->sim->EnvCopies()[pick.center - rows];
        picked_env_ = copy.env;
        pick.center = copy.instance;
    }
    pick.instances.erase(std::remove_if(pick.instances.begin(), pick.instances.end(),
                                        [rows](uint32_t i) { return i >= rows; }),
                         pick.instances.end());
    return pick;
}

std::string EditorHost::State::CarrierPath(uint32_t instance) const {
    return PathOfEntity(CarrierOf(loaded_->scene, ShownWorld().instances[instance].entity));
}

json::Value EditorHost::State::BoxSelectArgs(const render::PickResult& pick, bool add) {
    std::vector<std::string> paths = add ? SelectionPaths() : std::vector<std::string>{};
    for (const uint32_t instance : pick.instances) {
        const std::string path = CarrierPath(instance);
        if (std::find(paths.begin(), paths.end(), path) == paths.end()) paths.push_back(path);
    }
    // The primary stays when adding; otherwise the body at the box's centre leads.
    const std::string lead = add && !SelectedPath().empty() ? SelectedPath()
                             : pick.center != ~0u           ? CarrierPath(pick.center)
                             : paths.empty()                ? std::string()
                                                            : paths.front();
    return Object({{"path", json::Value::Str(lead)}, {"paths", StrList(paths)}, {"mode", json::Value::Str("range")}});
}

void EditorHost::State::SetViewMode(render::ViewMode mode) {
    ui_state_.view_mode = mode;
    SyncModeOverlays();
}

// What the user turned on themselves stays on.
void EditorHost::State::SyncModeOverlays() {
    bool residual = false, contact = false;
    for (const uint32_t i : ShownViews()) {
        residual = residual || PaneMode(i) == render::ViewMode::Residual;
        contact = contact || PaneMode(i) == render::ViewMode::Contact;
    }
    if (loaded_ && residual != mode_heatmap_ && (mode_heatmap_ || !ui_state_.solver.heatmap)) {
        mode_heatmap_ = residual;
        RunAction("solver.heatmap", Object({{"on", json::Value::Bool(residual)}}), "view");
    }
    if (contact != mode_contacts_ && (mode_contacts_ || !ui_state_.show_contacts)) {
        mode_contacts_ = contact;
        ui_state_.show_contacts = contact;
    }
}

std::vector<uint32_t> EditorHost::State::ShownViews() const {
    if (ui_state_.view_layout == ViewLayout::Quad) return {0u, 1u, 2u, 3u};
    if (ui_state_.view_layout == ViewLayout::Split) return {0u, 1u};
    return {ui_state_.active_view};
}

void EditorHost::State::ActivateView(uint32_t index) {
    const uint32_t was = ui_state_.active_view;
    if (index == was) return;
    camera_.CancelInteraction();
    views_[was].camera = camera_;
    views_[was].mode = ui_state_.view_mode;
    camera_ = views_[index].camera;
    ui_state_.view_mode = views_[index].mode;
    ui_state_.active_view = index;
    ui_.SetActiveView(index);
}

// A layout without the active pane hands input to its first pane.
void EditorHost::State::SetViewLayout(ViewLayout layout) {
    ui_state_.view_layout = layout;
    const std::vector<uint32_t> shown = ShownViews();
    if (std::find(shown.begin(), shown.end(), ui_state_.active_view) == shown.end()) ActivateView(shown.front());
    SyncModeOverlays();
}

void EditorHost::State::SyncViews(const render::RenderWorld& rw) {
    std::vector<std::string>& cameras = ui_state_.scene_cameras;
    cameras.clear();
    for (const render::RenderCamera& c : rw.cameras) cameras.push_back(PathOfEntity(c.entity));
    ui_state_.panes.clear();
    for (const uint32_t i : ShownViews()) {
        std::string& path = views_[i].camera_path;
        const auto it = std::find(cameras.begin(), cameras.end(), path);
        if (it == cameras.end()) path.clear();
        const SceneViewportRect& rect = ui_.PaneRect(i);
        if (!path.empty() && rect.Valid())
            ViewCamera(i).UseSceneCamera(rw, rect.Aspect(), rect.Aspect(), static_cast<size_t>(it - cameras.begin()));
        ui_state_.panes.push_back(ViewPane{i, PaneMode(i), path});
    }
}

json::Value EditorHost::State::ViewJson(uint32_t index) {
    const CameraController& camera = ViewCamera(index);
    const SceneViewportRect& rect = ui_.PaneRect(index);
    const std::vector<uint32_t> shown = ShownViews();
    return Object({{"index", json::Value::Int(index)},
                   {"mode", json::Value::Str(kViewModes[static_cast<size_t>(PaneMode(index))].name)},
                   {"camera", json::Value::Str(views_[index].camera_path)},
                   {"active", json::Value::Bool(index == ui_state_.active_view)},
                   {"shown", json::Value::Bool(std::find(shown.begin(), shown.end(), index) != shown.end())},
                   {"rect", rect.Valid() ? Object({{"x", json::Value::Int(rect.pixels.x)},
                                                   {"y", json::Value::Int(rect.pixels.y)},
                                                   {"width", json::Value::Int(rect.pixels.width)},
                                                   {"height", json::Value::Int(rect.pixels.height)}})
                                         : json::Value::Null()},
                   {"eye", Vec3Json(camera.ResolvedEye())},
                   {"target", Vec3Json(camera.ResolvedTarget())},
                   {"fov", json::Value::Float(camera.fov_degrees)}});
}

void EditorHost::State::SetLoadError(const std::string& path) {
    ui_state_.load_error = "Could not load scene: " + path + ". Check the application log for details.";
    std::snprintf(ui_state_.load_path, sizeof(ui_state_.load_path), "%s", path.c_str());
}

void EditorHost::State::CreateWorld(const std::string& path) {
    if (device_ == nullptr) {
        nuka_device_desc_t device_desc{};
        CheckResult(nuka_device_create(&device_desc, &device_), "nuka_device_create");
    }
    nuka_scene_handle scene = nullptr;
    CheckResult(nuka_scene_load(path.c_str(), &scene), "nuka_scene_load");
    nuka_world_desc_t desc{};
    desc.env_count = 1u;
    desc.fixed_dt = opt_.dt;
    nuka_world_handle created = nullptr;
    const nuka_result_t result = nuka_world_create_from_built_scene(device_, &desc, scene, nullptr, &created);
    nuka_scene_destroy(scene);
    CheckResult(result, "nuka_world_create_from_built_scene");
    world_ = created;
    if (opt_.want_planned)
        CheckResult(nuka_world_set_execution_mode(world_, NUKA_EXECUTION_GRAPH), "nuka_world_set_execution_mode");
}

// Loads replace the world's scene in place, so the handle a host holds stays valid.
void EditorHost::State::ApplyLoad(const std::string& path) {
    present_->WaitIdle();
    try {
        if (world_ == nullptr) CreateWorld(path);
        else CheckResult(nuka_world_rebuild_from_scene(world_, path.c_str(), NUKA_REBUILD_INITIAL_STATE),
                         "nuka_world_rebuild_from_scene");
    } catch (const std::exception& e) {
        Emit(stderr, std::string("[nuka_editor] load failed: ") + e.what() + "\n");
        SetLoadError(path);
        return;
    }
    ApplyUnload();
    Attach(path);
}

void EditorHost::State::Attach(const std::string& path) {
    try {
        loaded_ = AttachEditorScene(world_, path, cache_);
    } catch (const std::exception& e) {
        Emit(stderr, std::string("[nuka_editor] attach failed: ") + e.what() + "\n");
        SetLoadError(path);
        return;
    }
    OnLoaded(path);
}

void EditorHost::State::OnLoaded(const std::string& path) {
    ++ui_state_.scene_version;
    ui_state_.load_error.clear();
    frame_pending_ = true;
    views_stale_ = true;
    for (PaneView& v : views_) v.camera_path.clear();
    entity_drag_.Cancel();
    debug_overlay_.Reset();
    doc_.Attach(loaded_.get(), this);  // no undo across loads
    DropAutosave();                    // the previous document's
    autosaved_revision_ = doc_.Revision();
    autosave_due_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(autosave_seconds_));
    AttachController();
    BindWorker();  // poses before framing
    FrameWorld(loaded_->sim->GetRenderWorld());
    SeedDriveTargets();
    script_host_.ClearScripts();
    live_scripts_.clear();
    SyncScripts();
    ui_state_.robot.tuner.runs.clear();
    ui_state_.robot.tuner.running = 0u;
    ui_state_.inspector = InspectorState{};
    last_seeded_ = scene::kInvalidEntity;
    xform_gesture_active_ = mat_gesture_active_ = gizmo_using_prev_ = false;
    std::snprintf(ui_state_.save_path, sizeof(ui_state_.save_path), "%s", path.c_str());
    ui_state_.selected_entity = scene::kInvalidEntity;
    ui_state_.selection.clear();
    ui_state_.hidden.clear();
    ui_state_.locked.clear();
    ui_state_.staged.clear();
    ui_state_.collision.preview = json::Value();
    ui_state_.collision.dismiss = true;
    hidden_meshes_.clear();
    hidden_stale_ = true;

    // Headless aids queue the same actions the UI runs, on the node --select names.
    const std::string sel_path = opt_.select_path;
    const scene::EntityId sel = sel_path.empty() ? scene::kInvalidEntity : EntityOfPath(*loaded_, sel_path);
    const auto queue_edit = [&](const char* label, const char* field, json::Value before, json::Value after) {
        Queue("edit.apply", TxnToJson(EditTxn{label, {EditOp{sel_path, field, std::move(before), std::move(after)}}}),
              "cli");
    };
    if (!sel_path.empty()) Queue("scene.select", Object({{"path", json::Value::Str(sel_path)}}), "cli");
    if (opt_.edit_color_on && sel != scene::kInvalidEntity) {
        const MaterialBinding mb = ResolveMaterial(*loaded_, doc_.Records(), sel);
        if (mb.valid) {
            scene::RenderMaterial m = loaded_->sim->GetRenderWorld().materials[mb.render_slot];
            for (int k = 0; k < 3; ++k) m.base_color[k] = opt_.edit_color[k];
            queue_edit("Color", "material", json::Value(), MaterialToJson(m));
        }
    }
    if (opt_.edit_move_on && sel != scene::kInvalidEntity) {
        if (const render::RenderInstance* inst = InstanceOfEntity(loaded_->sim->GetRenderWorld(), sel)) {
            math::Transform moved = inst->world_xform;
            moved.position.x += opt_.edit_move[0];
            moved.position.y += opt_.edit_move[1];
            moved.position.z += opt_.edit_move[2];
            queue_edit("Move", "transform", TransformToJson(inst->world_xform), TransformToJson(moved));
        }
    }
    if (opt_.gizmo_op == "rotate") ui_state_.gizmo.op = GizmoState::Op::Rotate;
    else if (opt_.gizmo_op == "move") ui_state_.gizmo.op = GizmoState::Op::Translate;
    ui_state_.gizmo.local = opt_.gizmo_local;
    if (!opt_.rename_to.empty())
        Queue("edit.rename", Object({{"path", json::Value::Str(sel_path)}, {"name", json::Value::Str(opt_.rename_to)}}),
              "cli");
    if (opt_.add_box_on) Queue("scene.spawn", Object({{"kind", json::Value::Str("box")}}), "cli");
    if (!opt_.spawn_kind.empty()) Queue("scene.spawn", Object({{"kind", json::Value::Str(opt_.spawn_kind)}}), "cli");
    if (opt_.delete_on) Queue("edit.delete", Object({{"path", json::Value::Str(sel_path)}}), "cli");
    if (!opt_.reparent_to.empty())
        Queue("edit.reparent",
              Object({{"path", json::Value::Str(sel_path)}, {"parent", json::Value::Str(opt_.reparent_to)}}), "cli");
    if (!opt_.save_to.empty()) Queue("file.save", Object({{"path", json::Value::Str(opt_.save_to)}}), "cli");
    if (opt_.want_play) Queue("sim.play", json::Value::Object(), "cli");
    if (opt_.perturb_on && sel != scene::kInvalidEntity) {
        const render::RenderInstance* inst = InstanceOfEntity(loaded_->sim->GetRenderWorld(), sel);
        const render::PoseSource::Kind kind = inst != nullptr ? inst->pose_source.kind : render::PoseSource::Kind::Static;
        if (kind == render::PoseSource::Kind::Link || kind == render::PoseSource::Kind::Body) {
            TickCommand cmd;
            cmd.kind = CommandKind::Perturb;
            cmd.perturb.source = inst->pose_source;
            for (int k = 0; k < 3; ++k) cmd.perturb.force[k] = opt_.perturb[k];
            worker_.Push(std::move(cmd));
            Emit(stdout, Format("[nuka_editor] perturb: %s holds (%g, %g, %g) N\n", opt_.select_path.c_str(),
                                static_cast<double>(opt_.perturb[0]), static_cast<double>(opt_.perturb[1]),
                                static_cast<double>(opt_.perturb[2])));
        } else {
            Emit(stderr, "[nuka_editor] --perturb: '" + opt_.select_path + "' has no link or body\n");
        }
    }

    // Headless teleop: PD-hold and nudge one drive target, or feed a command.
    const bool cmd_ctrl = policy_ctrl_ && policy_ctrl_->AcceptsCommand();
    TeleopView& teleop = ui_state_.teleop;
    if (opt_.teleop_nudge_on && !cmd_ctrl && opt_.teleop_nudge_dof >= 0 &&
        static_cast<size_t>(opt_.teleop_nudge_dof) < ui_state_.drive_targets.size()) {
        ApplyHoldGains(world_, loaded_->Layout(), teleop.kp, teleop.kd);
        teleop.on = true;
        teleop.hold = true;
        const size_t dof = static_cast<size_t>(opt_.teleop_nudge_dof);
        ui_state_.drive_targets[dof] += opt_.teleop_nudge_delta;
        ui_state_.drive_dirty[dof] = 1u;
        Emit(stdout, Format("[nuka_editor] teleop nudge: dof %d += %g (PD-hold kp=%g kd=%g)\n", opt_.teleop_nudge_dof,
                            static_cast<double>(opt_.teleop_nudge_delta), static_cast<double>(teleop.kp),
                            static_cast<double>(teleop.kd)));
    }
    if (opt_.teleop_cmd_on && cmd_ctrl) {
        // The CLI command is the baseline the every-frame producer re-sends.
        policy_ctrl_->SetCommandVector(opt_.teleop_cmd, 3);
        std::copy(opt_.teleop_cmd, opt_.teleop_cmd + 3, ui_state_.policy.command);
        teleop.on = true;
        teleop.mode = "command";
        for (int k = 0; k < 3; ++k) teleop.command_base[k] = opt_.teleop_cmd[k];
        Emit(stdout, Format("[nuka_editor] teleop cmd: vx=%g vy=%g wyaw=%g\n", static_cast<double>(opt_.teleop_cmd[0]),
                            static_cast<double>(opt_.teleop_cmd[1]), static_cast<double>(opt_.teleop_cmd[2])));
    }
}

// A rebuild replaced the simulation binding: re-attach the controller and drive editor.
void EditorHost::State::OnRebound(uint64_t tick, double sim_time) {
    AttachController();
    SeedDriveTargets();
    BindWorker(tick, sim_time);
    grab_.reset();
    video_.reset();
    hidden_meshes_.clear();
    hidden_stale_ = true;
}

void EditorHost::State::PushParam(const ParamWrite& write) {
    TickCommand cmd = TickCommand::Param(ParamKind::Registry, {static_cast<float>(write.value)});
    cmd.index = write.param;
    cmd.element = write.element;
    param_seq_ = worker_.Push(std::move(cmd));
    param_done_ = false;
}

// Runs with the worker parked. The rebuilt world continues the timeline from this tick, and
// the transport resumes because the binding starts paused while the UI still plays.
bool EditorHost::State::Rebuild(const std::string& label) {
    entity_drag_.Cancel();
    present_->WaitIdle();
    const uint64_t tick = worker_.Tick();
    const double sim_time = worker_.SimTime();
    // Moves queued this frame land before the state migrates by body name.
    for (const PoseWrite& write : loaded_->sim->TakePoseWrites())
        ApplyPoseWrite(world_, loaded_->Layout(), sent_env_, write);
    if (!RecookEditorScene(*loaded_, cache_)) return false;
    OnRebound(tick, sim_time);
    doc_.AddBranch(TimelineBranch{tick, Generation(), label});
    return true;
}

// Runs with the worker parked. Selection follows the node the edit created or moved, else
// keeps the nodes that are still there.
ApplyResult EditorHost::State::ApplyTxn(const EditTxn& txn) {
    const std::vector<std::string> selected = SelectionPaths();
    const ApplyResult result = doc_.Apply(txn);
    Reselect(result.focus.empty() ? selected : std::vector<std::string>{result.focus});
    return result;
}

// The first of `paths` that exists becomes the primary selection.
void EditorHost::State::Reselect(const std::vector<std::string>& paths) {
    ++ui_state_.scene_version;
    ui_state_.selected_entity = scene::kInvalidEntity;
    ui_state_.selection.clear();
    for (const std::string& path : paths)
        if (const auto n = loaded_->scene.Tree().NodeOf(path)) ui_state_.selection.push_back(n->entity);
    if (!ui_state_.selection.empty()) ui_state_.selected_entity = ui_state_.selection.front();
    ui_state_.inspector = InspectorState{};
    last_seeded_ = scene::kInvalidEntity;
}

// The selected nodes' paths, the primary first.
std::vector<std::string> EditorHost::State::SelectionPaths() const {
    std::vector<std::string> paths;
    const auto add = [&](scene::EntityId e) {
        std::string path = PathOfEntity(e);
        if (!path.empty() && std::find(paths.begin(), paths.end(), path) == paths.end()) paths.push_back(std::move(path));
    };
    add(ui_state_.selected_entity);
    for (const scene::EntityId e : ui_state_.selection) add(e);
    return paths;
}

ActionResult EditorHost::State::SubmitTxn(const EditTxn& txn) {
    ApplyResult result;
    worker_.Exclusive([&] { result = ApplyTxn(txn); });
    return Outcome(result);
}

// Selection stays put unless the restored step names a node to follow.
ActionResult EditorHost::State::StepHistory(bool redo) {
    const std::vector<std::string> selected = SelectionPaths();
    ApplyResult result;
    worker_.Exclusive([&] { result = redo ? doc_.Redo() : doc_.Undo(); });
    entity_drag_.Cancel();
    xform_gesture_active_ = mat_gesture_active_ = gizmo_using_prev_ = false;
    Reselect(result.focus.empty() ? selected : std::vector<std::string>{result.focus});
    return Outcome(result);
}

void EditorHost::State::Queue(const std::string& id, json::Value args, const std::string& source) {
    ui_state_.actions.push_back(ActionCall{id, std::move(args), source});
}

// Runs what the last frame queued; what these runs queue waits for the next frame top.
void EditorHost::State::RunQueuedActions() {
    std::vector<ActionCall> calls;
    calls.swap(ui_state_.actions);
    for (const ActionCall& call : calls) RunAction(call.id, call.args, call.source);
}

ActionResult EditorHost::State::RunAction(const std::string& id, const json::Value& args, const std::string& source) {
    const Action* action = actions_.Find(id);
    const std::string owner = action != nullptr ? action->owner : std::string();
    ActionResult result = actions_.Run(id, args, source, frame_index_);
    if (!result.ok) {
        last_error_ = result.error;
        const std::string msg = "[nuka_editor] " + id + " failed: " + result.error + "\n";
        std::fputs(msg.c_str(), stderr);
        ConsoleAppend(msg);
        ui_.Nui().Toast(nui::Level::Err, id + ": " + result.error);
    }
    if (!owner.empty()) extensions_.Report(owner, id, result.ok);
    return result;
}

ActionResult EditorHost::State::SaveScene(const std::string& path) {
    scene::nks::Save(loaded_->scene, path);
    doc_.MarkSaved();
    DropAutosave();
    doc_path_ = NormalPath(path);
    std::snprintf(ui_state_.save_path, sizeof(ui_state_.save_path), "%s", doc_path_.c_str());
    profile_->TouchRecent(doc_path_);
    SyncBookmarks(true);
    SavePoses();
    start_stale_ = true;
    assets_stale_ = true;
    // The recent list shows the viewport as it was saved.
    const uint64_t next = present_->SubmittedFrames();
    if (ui_.Viewport().Valid()) {
        thumb_ = ThumbRequest{doc_path_, cache_.File("thumbnail.ppm"), next};
        present_->SetSceneCaptureFrame(static_cast<int>(next), thumb_->capture);
    }
    Emit(stdout, "[nuka_editor] saved " + path + "\n");
    ui_.Nui().Toast(nui::Level::Ok, std::string(nui::Tr("Saved")) + " " + FileName(path));
    return Ok(Object({{"path", json::Value::Str(path)}}));
}

// Loads `file` as the document that saves to `doc_path`; an empty one stays untitled.
ActionResult EditorHost::State::OpenDocument(const std::string& file, const std::string& doc_path) {
    std::snprintf(ui_state_.load_path, sizeof(ui_state_.load_path), "%s", file.c_str());
    worker_.Exclusive([&] { ApplyLoad(file); });
    if (!ui_state_.load_error.empty()) return ActionResult::Fail(ui_state_.load_error);
    doc_path_ = doc_path.empty() ? std::string() : NormalPath(doc_path);
    std::snprintf(ui_state_.save_path, sizeof(ui_state_.save_path), "%s", doc_path_.c_str());
    if (!doc_path_.empty()) profile_->TouchRecent(doc_path_);
    bookmarks_.clear();
    if (!doc_path_.empty()) ReadBookmarks(BookmarkFile(), &bookmarks_);
    if (!opt_.cams_path.empty()) {
        if (!ReadBookmarks(opt_.cams_path, &bookmarks_)) Emit(stderr, "[nuka_editor] --cams cannot read " + opt_.cams_path + "\n");
        opt_.cams_path.clear();
    }
    SyncBookmarks(false);
    ui_state_.robot.poses.clear();
    if (!doc_path_.empty()) ReadPoses(PoseFile(), &ui_state_.robot.poses);
    start_stale_ = true;
    assets_stale_ = true;
    Emit(stdout, "[nuka_editor] opened " + file + "\n");
    ui_.Nui().Toast(nui::Level::Ok, std::string(nui::Tr("Opened")) + " " + DocTitle());
    return Ok(Object({{"path", json::Value::Str(doc_path_)}, {"title", json::Value::Str(DocTitle())}}));
}

// True when `id` may drop the document: it is clean or `discard` says so. Otherwise the
// question opens and `asked` reports it; app.answer_prompt re-runs `id` with the answer.
bool EditorHost::State::Confirmed(const std::string& id, const json::Value& args, ActionResult* asked) {
    if (!loaded_ || !doc_.Dirty() || BoolArg(args, "discard", false)) return true;
    prompt_ = Prompt{id, args};
    ui_state_.prompt = id;
    *asked = Ok(Object({{"prompt", json::Value::Str(id)}}));
    return false;
}

// Starts another editor that owns its own world, with this one's profile.
ActionResult EditorHost::State::Spawn(const std::vector<std::string>& args) {
    if (opt_.relaunch.empty()) return ActionResult::Fail("this session has no command to start another editor");
    std::vector<std::string> argv = opt_.relaunch;
    argv.insert(argv.end(), args.begin(), args.end());
    argv.insert(argv.end(), {"--new-window", "--profile", PathText(profile_->Dir())});
    std::string error;
    const uint64_t pid = SpawnDetached(argv, &error);
    if (pid == 0u) return ActionResult::Fail(error);
    Emit(stdout, Format("[nuka_editor] started editor process %llu\n", static_cast<unsigned long long>(pid)));
    ui_.Nui().Toast(nui::Level::Ok, nui::Tr("Opened in a new window"));
    return Ok(Object({{"spawned", json::Value::Int(static_cast<int64_t>(pid))}}));
}

// Writes the dirty document beside a note of where it belongs, so the next start can recover it.
void EditorHost::State::Autosave() {
    scene::nks::Save(loaded_->scene, autosave_file_);
    const json::Value note = Object({{"path", json::Value::Str(doc_path_)},
                                     {"pid", json::Value::Int(static_cast<int64_t>(ProcessId()))},
                                     {"time", json::Value::Int(UnixNow())}});
    std::ofstream(SidecarOf(autosave_file_), std::ios::binary | std::ios::trunc) << note.Dump(2);
    autosaved_revision_ = doc_.Revision();
    autosave_due_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(autosave_seconds_));
    WriteLog("[nuka_editor] autosaved " + autosave_file_ + "\n");
}

void EditorHost::State::DropAutosave() {
    const fs::path file = Utf8Path(autosave_file_);
    for (const char* ext : {".nks", ".nka", ".json"}) fs::remove(fs::path(file).replace_extension(ext));
}

// Templates and examples from start.json, the recent list, autosaves of editors that are gone,
// and the startup checks.
void EditorHost::State::RefreshStartPage() {
    StartPage& page = ui_state_.start;
    page.templates.clear();
    page.recent.clear();
    page.recovered.clear();
    page.examples.clear();
    std::string text;
    ReadEntireFile(Resource("assets/editor/start.json"), &text);
    const json::Value start = json::Value::Parse(text);
    for (const json::Value& t : start.At("templates").Elements())
        page.templates.push_back(StartEntry{t.At("title").AsString(), t.At("detail").AsString(), t.At("id").AsString(), "", true});
    for (const json::Value& e : start.At("examples").Elements()) {
        const std::string path = Resource(e.At("path").AsString());
        const bool present = fs::exists(Utf8Path(path));
        page.examples.push_back(StartEntry{e.At("title").AsString(), e.At("detail").AsString(), path,
                                           present ? thumbnails_->Thumbnail(AssetAt(path, "")) : std::string(), present});
    }
    for (const RecentScene& r : profile_->Recent()) {
        const fs::path thumb = profile_->Thumbnail(r.path);
        page.recent.push_back(StartEntry{FileName(r.path), LocalTimeText(r.time), r.path,
                                         fs::exists(thumb) ? PathText(thumb) : std::string(), fs::exists(Utf8Path(r.path))});
    }
    for (const fs::directory_entry& entry : fs::directory_iterator(profile_->Dir() / "autosave")) {
        if (entry.path().extension() != ".json") continue;
        std::string note_text;
        ReadEntireFile(PathText(entry.path()), &note_text);
        const json::Value note = json::Value::Parse(note_text);
        if (ProcessAlive(static_cast<uint32_t>(note.At("pid").AsInt()))) continue;
        const std::string doc = note.At("path").AsString();
        page.recovered.push_back(StartEntry{doc.empty() ? std::string("untitled") : FileName(doc),
                                            LocalTimeText(note.At("time").AsInt()),
                                            PathText(fs::path(entry.path()).replace_extension(".nks")), "", true});
    }
    page.checks = checks_;
}

void EditorHost::State::RegisterActions() {
    RegisterFileActions();
    RegisterAppActions();
    RegisterEditActions();
    RegisterSceneActions();
    RegisterAssetActions();
    RegisterSimActions();
    RegisterViewActions();
    RegisterSolverActions();
    RegisterTimelineActions();
    RegisterRobotActions();
    RegisterScriptActions();
    ui_.RegisterActions(actions_);
}

void EditorHost::State::RegisterRobotActions() {
    const auto has_joints = [this] { return loaded_ != nullptr && !ui_state_.robot.joints.empty(); };
    // The moving joints with their limits and drives, and the shown environment's live values.
    actions_.Add({"robot.joints", "Robot Joints", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      const RobotView& v = ui_state_.robot;
                      json::Value joints = json::Value::Array();
                      for (const RobotJoint& j : v.joints) joints.PushBack(JointJson(j, v.live));
                      return Ok(Object({{"joints", std::move(joints)},
                                        {"roots", StrList(v.roots)},
                                        {"floating", json::Value::Bool(v.floating)},
                                        {"env", json::Value::Int(ui_state_.env_index)},
                                        {"watch", json::Value::Bool(v.watch)}}));
                  }});
    // {targets: {joint: value}} in rad or m, for the shown environment; the next step reads them.
    actions_.Add({"robot.set_targets", "Set Joint Targets", "robot", "", ShortcutScope::Global, has_joints, {},
                  [this](const json::Value& args) {
                      const RobotView& v = ui_state_.robot;
                      std::vector<std::pair<uint32_t, float>> writes;
                      for (const auto& [name, value] : args.At("targets").Items()) {
                          const int j = FindJoint(v.joints, name);
                          if (j < 0) return ActionResult::Fail("no joint '" + name + "'");
                          writes.emplace_back(v.joints[static_cast<size_t>(j)].link, value.AsFloat());
                      }
                      for (const auto& [link, value] : writes) {
                          ui_state_.drive_targets[link] = value;
                          ui_state_.drive_dirty[link] = 1u;
                      }
                      PushEdits();
                      return Ok(Object({{"set", json::Value::Int(static_cast<int64_t>(writes.size()))}}));
                  }});
    // Puts every joint's target back at its initial position.
    actions_.Add({"robot.reset_targets", "Reset Joint Targets", "robot", "", ShortcutScope::Global, has_joints, {},
                  [this](const json::Value&) {
                      const std::vector<RobotJoint>& joints = ui_state_.robot.joints;
                      for (const RobotJoint& j : joints) {
                          ui_state_.drive_targets[j.link] = j.initial;
                          ui_state_.drive_dirty[j.link] = 1u;
                      }
                      PushEdits();
                      return Ok(Object({{"set", json::Value::Int(static_cast<int64_t>(joints.size()))}}));
                  }});
    // Revolute joints read in radians while on, else degrees; without `on` it toggles.
    actions_.Add({"robot.radians", "Radians", "robot", "", ShortcutScope::Global, {},
                  [this] { return ui_state_.robot.radians; },
                  [this](const json::Value& args) {
                      bool& radians = ui_state_.robot.radians;
                      radians = BoolArg(args, "on", !radians);
                      return Ok(Object({{"on", json::Value::Bool(radians)}}));
                  }});
    // While on, every step samples the shown environment's joints for robot.samples and the curves.
    actions_.Add({"robot.watch", "Watch Joints", "robot", "", ShortcutScope::Global, has_joints,
                  [this] { return ui_state_.robot.watch; },
                  [this](const json::Value& args) {
                      RobotView& v = ui_state_.robot;
                      v.watch = BoolArg(args, "on", !v.watch);
                      worker_.Push(TickCommand::Param(ParamKind::JointWatch, {v.watch ? 1.0f : 0.0f}));
                      return Ok(Object({{"on", json::Value::Bool(v.watch)}}));
                  }});
    // The newest `count` samples of one joint, oldest first; `total` counts the watch's samples.
    actions_.Add({"robot.samples", "Joint Samples", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const RobotView& v = ui_state_.robot;
                      const std::string name = StrArg(args, "joint");
                      const int j = FindJoint(v.joints, name);
                      if (j < 0) return ActionResult::Fail("no joint '" + name + "'");
                      const uint32_t link = v.joints[static_cast<size_t>(j)].link;
                      const size_t count = std::min(
                          static_cast<size_t>(NumArg(args, "count", static_cast<double>(v.history.size()))), v.history.size());
                      json::Value tick = json::Value::Array(), t = json::Value::Array(), q = json::Value::Array(),
                                  target = json::Value::Array(), effort = json::Value::Array();
                      for (size_t i = v.history.size() - count; i < v.history.size(); ++i) {
                          const JointSample& s = v.history[i];
                          tick.PushBack(json::Value::Int(static_cast<int64_t>(s.tick)));
                          t.PushBack(json::Value::Float(s.time));
                          q.PushBack(json::Value::Float(s.q[link]));
                          target.PushBack(json::Value::Float(s.target[link]));
                          effort.PushBack(json::Value::Float(s.effort[link]));
                      }
                      return Ok(Object({{"joint", json::Value::Str(name)},
                                        {"count", json::Value::Int(static_cast<int64_t>(count))},
                                        {"total", json::Value::Int(static_cast<int64_t>(v.first + v.history.size()))},
                                        {"watch", json::Value::Bool(v.watch)},
                                        {"tick", std::move(tick)},
                                        {"t", std::move(t)},
                                        {"q", std::move(q)},
                                        {"target", std::move(target)},
                                        {"effort", std::move(effort)}}));
                  }});
    // One undoable edit gives `joints` (all by default) a drive of `type`; a position drive takes kp.
    actions_.Add({"robot.drive", "Position Drive", "robot", "", ShortcutScope::Global, has_joints, {},
                  [this](const json::Value& args) {
                      const RobotView& v = ui_state_.robot;
                      const std::string type = StrArg(args, "type", "position");
                      json::Value after = Object({{"type", json::Value::Str(type)}});
                      if (type == "position") after.Set("kp", json::Value::Float(NumArg(args, "kp", v.drive_kp)));
                      std::vector<std::string> names = StrListArg(args, "joints");
                      if (names.empty())
                          for (const RobotJoint& j : v.joints) names.push_back(j.name);
                      EditTxn txn{"Drive " + type, {}};
                      for (const std::string& name : names) {
                          const int j = FindJoint(v.joints, name);
                          if (j < 0) return ActionResult::Fail("no joint '" + name + "'");
                          txn.ops.push_back(EditOp{v.joints[static_cast<size_t>(j)].path, "drive", json::Value(), after});
                      }
                      return SubmitTxn(txn);
                  }});
    // Static roots are fixed bases; without `on` it toggles.
    actions_.Add({"robot.fix_base", "Fixed Base", "robot", "", ShortcutScope::Global,
                  [this] { return loaded_ != nullptr && !ui_state_.robot.roots.empty(); },
                  [this] { return !ui_state_.robot.roots.empty() && !ui_state_.robot.floating; },
                  [this](const json::Value& args) {
                      const RobotView& v = ui_state_.robot;
                      const bool on = BoolArg(args, "on", v.floating);
                      EditTxn txn{on ? "Fix Base" : "Free Base", {}};
                      for (const std::string& root : v.roots)
                          txn.ops.push_back(EditOp{root, "static", json::Value(), json::Value::Bool(on)});
                      return SubmitTxn(txn);
                  }});
    RegisterPoseActions();
    RegisterGainActions();
    RegisterAssemblyActions();
    RegisterIkActions();
    RegisterOverlayActions();
    RegisterTeleopActions();
    RegisterPolicyActions();
    RegisterDemoActions();
}

void EditorHost::State::RegisterIkActions() {
    const auto has_links = [this] { return loaded_ != nullptr && shown_ != nullptr && !ui_state_.robot.joints.empty(); };
    // Moves `link` (node path or joint name) to world `pos`, and `quat` or XYZ-degree `rot`, keeping `pins` put.
    // `apply` (default) writes the solution as a gizmo drag does, else it only reports it.
    actions_.Add({"robot.ik", "Solve IK", "robot", "", ShortcutScope::Global, has_links, {},
                  [this](const json::Value& args) {
                      const IkState& ik = ui_state_.robot.ik;
                      const std::vector<math::Transform>& poses = shown_->link_poses;
                      const std::string name = StrArg(args, "link");
                      const int link = FindLink(name);
                      if (link < 0) return ActionResult::Fail("no link '" + name + "'");
                      math::Transform goal = poses[static_cast<size_t>(link)];
                      const auto& pos = args.At("pos").Elements();
                      goal.position = math::Vec3{pos[0].AsFloat(), pos[1].AsFloat(), pos[2].AsFloat()};
                      const json::Value* quat = args.Find("quat");
                      const json::Value* rot = args.Find("rot");
                      if (quat != nullptr) {
                          const auto& e = quat->Elements();
                          goal.rotation = math::Quat(e[0].AsFloat(), e[1].AsFloat(), e[2].AsFloat(), e[3].AsFloat()).Normalized();
                      } else if (rot != nullptr) {
                          const auto& e = rot->Elements();
                          goal.rotation = QuatFromEulerDeg(math::Vec3{e[0].AsFloat(), e[1].AsFloat(), e[2].AsFloat()});
                      }
                      const bool orient = (quat != nullptr || rot != nullptr) && !BoolArg(args, "position_only", false);
                      std::vector<nuka_ik_target_t> targets = {IkGoal(static_cast<uint32_t>(link), goal, orient, 1.0f)};
                      for (const std::string& pin : args.Find("pins") != nullptr ? StrListArg(args, "pins") : ik.pins) {
                          const int p = FindLink(pin);
                          if (p < 0) return ActionResult::Fail("no link '" + pin + "'");
                          targets.push_back(IkGoal(static_cast<uint32_t>(p), poses[static_cast<size_t>(p)], true, 1.0f));
                      }
                      const std::string root_name = StrArg(args, "root", ik.root);
                      const int root = FindLink(root_name);
                      if (!root_name.empty() && root < 0) return ActionResult::Fail("no link '" + root_name + "'");
                      TickCommand cmd = IkCommand(std::move(targets), root,
                                                  static_cast<float>(NumArg(args, "rotation_weight", ik.rotation_weight)),
                                                  shown_->env);
                      const bool apply = BoolArg(args, "apply", true);
                      IkReport report;
                      if (apply) {
                          worker_.Push(std::move(cmd));
                          worker_.Exclusive([] {});
                          worker_.TakeIk(&report);
                          TakeIkReport(report);
                      } else {
                          worker_.Exclusive([&] { report = RunIk(world_, loaded_->Layout().links_per_env, cmd); });
                      }
                      if (!report.error.empty()) return ActionResult::Fail(report.error);
                      json::Value q = json::Value::Object();
                      for (const RobotJoint& j : ui_state_.robot.joints) q.Set(j.name, json::Value::Float(report.q[j.link]));
                      json::Value position_error = json::Value::Array(), rotation_error = json::Value::Array();
                      for (float e : report.position_error) position_error.PushBack(json::Value::Float(e));
                      for (float e : report.rotation_error) rotation_error.PushBack(json::Value::Float(e));
                      return Ok(Object({{"q", std::move(q)},
                                        {"position_error", std::move(position_error)},
                                        {"rotation_error", std::move(rotation_error)},
                                        {"iterations", json::Value::Int(report.iterations)},
                                        {"joints", json::Value::Int(report.joints)},
                                        {"at_limit", json::Value::Int(report.at_limit)},
                                        {"applied", json::Value::Bool(apply)}}));
                  }});
    // The gizmo's IK: the chain root (empty for the articulation's), the links a drag pins, whether a
    // move locks the orientation and the rotation weight [m/rad]; an argument given replaces its setting.
    actions_.Add({"robot.ik_settings", "IK Settings", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      IkState& ik = ui_state_.robot.ik;
                      const std::string root = StrArg(args, "root", ik.root);
                      const std::vector<std::string> pins = args.Find("pins") != nullptr ? StrListArg(args, "pins") : ik.pins;
                      for (const std::string& link : pins)
                          if (FindLink(link) < 0) return ActionResult::Fail("no link '" + link + "'");
                      if (!root.empty() && FindLink(root) < 0) return ActionResult::Fail("no link '" + root + "'");
                      ik.root = root;
                      ik.pins = pins;
                      ik.lock_orientation = BoolArg(args, "lock_orientation", ik.lock_orientation);
                      ik.rotation_weight = static_cast<float>(NumArg(args, "rotation_weight", ik.rotation_weight));
                      return Ok(Object({{"on", json::Value::Bool(ik.on)},
                                        {"root", json::Value::Str(ik.root)},
                                        {"pins", StrList(ik.pins)},
                                        {"lock_orientation", json::Value::Bool(ik.lock_orientation)},
                                        {"rotation_weight", json::Value::Float(ik.rotation_weight)}}));
                  }});
    // While on, the gizmo on a jointed link poses the robot through IK; without `on` it toggles.
    actions_.Add({"tool.ik", "IK Drag", "sim", "", ShortcutScope::Global, {}, [this] { return ui_state_.robot.ik.on; },
                  [this](const json::Value& args) {
                      IkState& ik = ui_state_.robot.ik;
                      ik.on = BoolArg(args, "on", !ik.on);
                      return Ok(Object({{"on", json::Value::Bool(ik.on)}}));
                  }});
}

void EditorHost::State::RegisterTeleopActions() {
    const auto mode_known = [this](const std::string& mode) {
        const std::vector<TeleopRetarget>& r = ui_state_.teleop.retargets;
        return std::find(kTeleopModes.begin(), kTeleopModes.end(), mode) != kTeleopModes.end() ||
               std::any_of(r.begin(), r.end(), [&](const TeleopRetarget& t) { return t.id == mode; });
    };
    // A button reads true, false or a number (nonzero is down).
    const auto pressed = [](const json::Value& v) {
        return v.type() == json::Value::Type::Bool ? v.AsBool() : v.AsDouble() != 0.0;
    };
    // The devices teleop reads and the modes it maps them through, with the active ones.
    actions_.Add({"teleop.devices", "Teleop Devices", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      ListTeleopDevices();
                      const TeleopView& t = ui_state_.teleop;
                      json::Value devices = json::Value::Array();
                      for (const TeleopDeviceInfo& d : t.devices)
                          devices.PushBack(Object({{"id", json::Value::Str(d.id)},
                                                   {"label", json::Value::Str(d.label)},
                                                   {"kind", json::Value::Str(d.kind)},
                                                   {"owner", json::Value::Str(d.owner)},
                                                   {"axes", StrList(d.axes)},
                                                   {"buttons", StrList(d.buttons)}}));
                      json::Value modes = json::Value::Array();
                      for (const std::string& m : kTeleopModes)
                          modes.PushBack(Object({{"id", json::Value::Str(m)},
                                                 {"label", json::Value::Str(m)},
                                                 {"owner", json::Value::Str(std::string())},
                                                 {"quantities", StrList(ModeQuantities(m))}}));
                      for (const TeleopRetarget& r : t.retargets)
                          modes.PushBack(Object({{"id", json::Value::Str(r.id)},
                                                 {"label", json::Value::Str(r.label)},
                                                 {"owner", json::Value::Str(r.owner)},
                                                 {"quantities", json::Value::Array()}}));
                      return Ok(Object({{"devices", std::move(devices)},
                                        {"modes", std::move(modes)},
                                        {"device", json::Value::Str(t.device)},
                                        {"mode", json::Value::Str(t.mode)}}));
                  }});
    // Switches teleop on or off and picks the device, the mode, the joint mode's joint, the ik
    // mode's link ("" follows the selection), PD hold and the command baseline; bare, it toggles.
    actions_.Add({"teleop.use", "Teleop", "robot", "", ShortcutScope::Global, {}, [this] { return ui_state_.teleop.on; },
                  [this, mode_known](const json::Value& args) {
                      ListTeleopDevices();
                      TeleopView& t = ui_state_.teleop;
                      const std::string device = StrArg(args, "device", t.device);
                      const std::string mode = StrArg(args, "mode", t.mode);
                      // Only a change is checked, so teleop still switches off after its device is gone.
                      if (device != t.device &&
                          std::none_of(t.devices.begin(), t.devices.end(), [&](const TeleopDeviceInfo& d) { return d.id == device; }))
                          return ActionResult::Fail("no teleop device '" + device + "'");
                      if (mode != t.mode && !mode_known(mode)) return ActionResult::Fail("no teleop mode '" + mode + "'");
                      int joint = t.joint;
                      if (const json::Value* j = args.Find("joint"); j != nullptr && !j->IsNull()) {
                          const std::vector<RobotJoint>& joints = ui_state_.robot.joints;
                          const auto it = std::find_if(joints.begin(), joints.end(),
                                                       [&](const RobotJoint& r) { return r.name == j->AsString(); });
                          if (it == joints.end()) return ActionResult::Fail("no joint '" + j->AsString() + "'");
                          joint = static_cast<int>(it - joints.begin());
                      }
                      std::string link = StrArg(args, "link", t.link);
                      if (!link.empty()) {
                          const int l = FindLink(link);
                          if (l < 0 || ui_state_.robot.links[static_cast<size_t>(l)].parent < 0)
                              return ActionResult::Fail("no link '" + link + "' below a root");
                          link = ui_state_.robot.links[static_cast<size_t>(l)].path;
                      }
                      const float kp = static_cast<float>(NumArg(args, "kp", t.kp));
                      const float kd = static_cast<float>(NumArg(args, "kd", t.kd));
                      if (kp < 0.0f || kd < 0.0f) return ActionResult::Fail("kp and kd must not be negative");
                      math::Vec3 base{t.command_base[0], t.command_base[1], t.command_base[2]};
                      Vec3Arg(args, "command_base", &base);
                      if (device != t.device || mode != t.mode || link != t.link) t.goal_set = false;
                      t.on = BoolArg(args, "on", args.Items().empty() ? !t.on : t.on);
                      t.device = device;
                      t.mode = mode;
                      t.joint = joint;
                      t.link = link;
                      t.hold = BoolArg(args, "hold", t.hold);
                      t.kp = kp;
                      t.kd = kd;
                      const float b[3] = {base.x, base.y, base.z};
                      std::copy(b, b + 3, t.command_base);
                      return Ok(TeleopStatus());
                  }});
    // Sets the virtual device's axes ([-1, 1]) and buttons by name; the others keep their values.
    actions_.Add({"teleop.input", "Teleop Input", "robot", "", ShortcutScope::Global, {}, {},
                  [this, pressed](const json::Value& args) {
                      std::vector<std::pair<size_t, float>> axes;
                      std::vector<std::pair<size_t, uint8_t>> buttons;
                      if (const json::Value* a = args.Find("axes"))
                          for (const auto& [name, v] : a->Items()) {
                              const size_t i = static_cast<size_t>(std::find(kPadAxes.begin(), kPadAxes.end(), name) - kPadAxes.begin());
                              if (i == kPadAxes.size()) return ActionResult::Fail("no axis '" + name + "'");
                              axes.emplace_back(i, std::clamp(v.AsFloat(), -1.0f, 1.0f));
                          }
                      if (const json::Value* b = args.Find("buttons"))
                          for (const auto& [name, v] : b->Items()) {
                              const size_t i =
                                  static_cast<size_t>(std::find(kPadButtons.begin(), kPadButtons.end(), name) - kPadButtons.begin());
                              if (i == kPadButtons.size()) return ActionResult::Fail("no button '" + name + "'");
                              buttons.emplace_back(i, static_cast<uint8_t>(pressed(v)));
                          }
                      for (const auto& [i, v] : axes) virtual_axes_[i] = v;
                      for (const auto& [i, v] : buttons) virtual_buttons_[i] = v;
                      json::Value out_axes = json::Value::Object(), out_buttons = json::Value::Object();
                      for (size_t i = 0; i < kPadAxes.size(); ++i) out_axes.Set(kPadAxes[i], json::Value::Float(virtual_axes_[i]));
                      for (size_t i = 0; i < kPadButtons.size(); ++i)
                          out_buttons.Set(kPadButtons[i], json::Value::Bool(virtual_buttons_[i] != 0u));
                      return Ok(Object({{"axes", std::move(out_axes)}, {"buttons", std::move(out_buttons)}}));
                  }});
    // A mode's bindings [{input, quantity, gain, invert}]: given ones replace them and `reset`
    // restores the defaults; `deadzone` in [0, 1) is shared by every axis.
    actions_.Add({"teleop.map", "Teleop Mapping", "robot", "", ShortcutScope::Global, {}, {},
                  [this, mode_known](const json::Value& args) {
                      TeleopView& t = ui_state_.teleop;
                      const std::string mode = StrArg(args, "mode", t.mode);
                      if (!mode_known(mode)) return ActionResult::Fail("no teleop mode '" + mode + "'");
                      const float deadzone = static_cast<float>(NumArg(args, "deadzone", t.deadzone));
                      if (deadzone < 0.0f || deadzone >= 1.0f) return ActionResult::Fail("deadzone wants [0, 1)");
                      const std::vector<std::string>& names = ModeQuantities(mode);
                      std::vector<TeleopBinding> bindings = t.maps[mode];
                      if (BoolArg(args, "reset", false)) {
                          const auto defaults = DefaultTeleopMaps();
                          const auto it = defaults.find(mode);
                          bindings = it != defaults.end() ? it->second : std::vector<TeleopBinding>();
                      }
                      if (const json::Value* list = args.Find("bindings"); list != nullptr && !list->IsNull()) {
                          bindings.clear();
                          for (const json::Value& b : list->Elements()) {
                              TeleopBinding binding{StrArg(b, "input"), StrArg(b, "quantity"),
                                                    static_cast<float>(NumArg(b, "gain", 1.0)), BoolArg(b, "invert", false)};
                              if (binding.input.empty() || binding.quantity.empty())
                                  return ActionResult::Fail("a binding wants an input and a quantity");
                              if (!names.empty() && std::find(names.begin(), names.end(), binding.quantity) == names.end())
                                  return ActionResult::Fail("mode '" + mode + "' has no quantity '" + binding.quantity + "'");
                              bindings.push_back(std::move(binding));
                          }
                      }
                      t.maps[mode] = bindings;
                      t.deadzone = deadzone;
                      json::Value list = json::Value::Array();
                      for (const TeleopBinding& b : bindings)
                          list.PushBack(Object({{"input", json::Value::Str(b.input)},
                                                {"quantity", json::Value::Str(b.quantity)},
                                                {"gain", json::Value::Float(b.gain)},
                                                {"invert", json::Value::Bool(b.invert)}}));
                      return Ok(Object({{"mode", json::Value::Str(mode)},
                                        {"bindings", std::move(list)},
                                        {"quantities", StrList(names)},
                                        {"deadzone", json::Value::Float(t.deadzone)}}));
                  }});
    // A virtual gamepad on SDL's joystick path: `attach` names a new one; `id` with `axes` and
    // `buttons` sets its inputs, and with `detach` removes it.
    actions_.Add({"teleop.pad", "Virtual Gamepad", "robot", "", ShortcutScope::Global, {}, {},
                  [this, pressed](const json::Value& args) {
                      std::string error;
                      if (const json::Value* name = args.Find("attach"); name != nullptr && !name->IsNull()) {
                          const std::string id = pads_.Attach(name->AsString(), &error);
                          if (id.empty()) return ActionResult::Fail(error);
                          return Ok(Object({{"id", json::Value::Str(id)}}));
                      }
                      const std::string id = StrArg(args, "id");
                      if (BoolArg(args, "detach", false)) {
                          if (!pads_.Detach(id)) return ActionResult::Fail("'" + id + "' is not a virtual gamepad");
                          return Ok(Object({{"id", json::Value::Str(id)}, {"detached", json::Value::Bool(true)}}));
                      }
                      std::map<std::string, float> axes, buttons;
                      if (const json::Value* a = args.Find("axes"))
                          for (const auto& [axis, v] : a->Items()) axes[axis] = v.AsFloat();
                      if (const json::Value* b = args.Find("buttons"))
                          for (const auto& [button, v] : b->Items()) buttons[button] = pressed(v) ? 1.0f : 0.0f;
                      if (!pads_.Set(id, axes, buttons, &error)) return ActionResult::Fail(error);
                      return Ok(Object({{"id", json::Value::Str(id)}}));
                  }});
    // The active device's reading, the mode's quantities and what they drove last frame.
    actions_.Add({"teleop.status", "Teleop Status", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) { return Ok(TeleopStatus()); }});
}

void EditorHost::State::RegisterPolicyActions() {
    const auto has_policy = [this] { return policy_ctrl_ != nullptr; };
    // Attaches an NKML locomotion policy to the loaded robots in place of the one before, and
    // sends it teleop's command; the dialog asks without `path`, `host` runs the host MLP.
    actions_.Add({"policy.load", "Load Policy...", "robot", "", ShortcutScope::Global, [this] { return loaded_ != nullptr; },
                  {},
                  [this](const json::Value& args) {
                      if (loaded_ == nullptr) return ActionResult::Fail("no scene is loaded");
                      std::string path = StrArg(args, "path");
                      if (path.empty()) path = OpenFileDialog(nui::Tr("Load Policy"), nui::Tr("NKML policies"), "*.bin;*.nkml");
                      if (path.empty()) return ActionResult::Fail("no policy file chosen");
                      const bool host = BoolArg(args, "host", false);
                      std::string error;
                      std::unique_ptr<inference::Go2PolicyController> pc = MakeController(path, host, &error);
                      if (pc == nullptr) {
                          ui_state_.policy.error = error;
                          return ActionResult::Fail(error);
                      }
                      opt_.policy_path = path;
                      opt_.force_host_policy = host;
                      policy_on_ = true;
                      worker_.Exclusive([&] {
                          policy_ctrl_ = std::move(pc);
                          worker_.SetController(policy_ctrl_.get());
                          SyncPolicyView();
                      });
                      ui_state_.policy.error.clear();
                      teleop_command_sent_.reset();  // the next frame sends teleop's command, zero while it is off
                      Emit(stdout, Format("[nuka_editor] policy attached: %s  path=%s\n", path.c_str(),
                                          policy_ctrl_->UsingGpu() ? "GPU" : "host"));
                      return Ok(PolicyStatus());
                  }});
    actions_.Add({"policy.unload", "Unload Policy", "robot", "", ShortcutScope::Global, has_policy, {},
                  [this](const json::Value&) {
                      if (policy_ctrl_ == nullptr) return ActionResult::Fail("no policy is loaded");
                      worker_.Exclusive([&] {
                          worker_.SetController(nullptr);
                          policy_ctrl_.reset();
                      });
                      opt_.policy_path.clear();
                      SyncPolicyView();
                      return Ok(PolicyStatus());
                  }});
    // Runs or stops the loaded policy; stopped, the drive targets hold where it left them. Bare, it toggles.
    actions_.Add({"policy.enable", "Run Policy", "robot", "", ShortcutScope::Global, has_policy,
                  [this] { return policy_ctrl_ != nullptr && policy_on_; },
                  [this](const json::Value& args) {
                      if (policy_ctrl_ == nullptr) return ActionResult::Fail("no policy is loaded");
                      policy_on_ = BoolArg(args, "on", args.Items().empty() ? !policy_on_ : policy_on_);
                      worker_.Exclusive([&] { worker_.SetController(policy_on_ ? policy_ctrl_.get() : nullptr); });
                      ui_state_.policy.on = policy_on_;
                      teleop_command_sent_.reset();
                      return Ok(PolicyStatus());
                  }});
    // The loaded policy, its contract, the joints it drives and the command it reads.
    actions_.Add({"policy.status", "Policy Status", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) { return Ok(PolicyStatus()); }});
}

void EditorHost::State::RegisterDemoActions() {
    const auto has_robot = [this] { return loaded_ != nullptr && !ui_state_.robot.joints.empty(); };
    const auto has_demos = [this] { return !demos_.episodes.empty(); };
    const auto episode_arg = [this](const json::Value& args, size_t* k) {
        const double i = NumArg(args, "episode", static_cast<double>(ui_state_.demos.selected));
        if (i < 0.0 || i >= static_cast<double>(demos_.episodes.size())) return false;
        *k = static_cast<size_t>(i);
        return true;
    };
    // Records the shown environment, with `all` every displayed one, each step at the head; the
    // stop cuts each environment's consecutive steps into an episode. Bare, it toggles.
    actions_.Add({"demo.record", "Record Demonstration", "robot", "", ShortcutScope::Global, has_robot,
                  [this] { return demo_recording_; },
                  [this](const json::Value& args) {
                      const bool on = BoolArg(args, "on", !demo_recording_);
                      if (on == demo_recording_) return Ok(DemoStatus());
                      if (!on) {
                          StopDemo();
                          return Ok(DemoStatus());
                      }
                      if (loaded_ == nullptr || ui_state_.robot.joints.empty())
                          return ActionResult::Fail("the scene has no robot joints to record");
                      DemoView& d = ui_state_.demos;
                      d.all = BoolArg(args, "all", d.all);
                      const uint32_t env = ui_state_.env_index % loaded_->Layout().env_count;
                      std::vector<uint32_t> envs = {env};
                      if (d.all)
                          for (const uint32_t e : OtherShownEnvs(env)) envs.push_back(e);
                      // A new set takes the scene's joints and the device's axes; later episodes keep them.
                      if (demos_.episodes.empty()) {
                          const TeleopView& t = ui_state_.teleop;
                          const auto device = std::find_if(t.devices.begin(), t.devices.end(),
                                                           [&](const TeleopDeviceInfo& i) { return i.id == t.device; });
                          demos_ = DemoSet{};
                          demos_.scene = doc_path_;
                          demos_.dt = opt_.dt;
                          for (const RobotJoint& j : ui_state_.robot.joints) {
                              demos_.joints.push_back(j.name);
                              demos_.links.push_back(j.link);
                          }
                          demos_.axes = device != t.devices.end() ? device->axes : kPadAxes;
                      }
                      demo_axes_.clear();
                      worker_.Exclusive([&] { worker_.RecordDemo(std::move(envs)); });
                      demo_recording_ = true;
                      SyncDemoView();
                      return Ok(DemoStatus());
                  }});
    // The episodes, the recording and the replay in progress.
    actions_.Add({"demo.episodes", "Demonstrations", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) { return Ok(DemoStatus()); }});
    // Writes an episode's recorded drive targets into its environment a row a step from the state
    // now, the controller waiting meanwhile; `play` (the default) starts the transport.
    actions_.Add({"demo.play", "Replay Demonstration", "robot", "", ShortcutScope::Global, has_demos, {},
                  [this, episode_arg](const json::Value& args) {
                      size_t k = 0u;
                      if (!episode_arg(args, &k))
                          return ActionResult::Fail(Format("episode: 0 to %zu", demos_.episodes.size() - 1u));
                      const DemoEpisode& e = demos_.episodes[k];
                      if (e.env >= loaded_->Layout().env_count)
                          return ActionResult::Fail(Format("episode %zu drove environment %u; the scene has %u", k, e.env,
                                                           loaded_->Layout().env_count));
                      std::vector<float> values;
                      for (const uint32_t link : demos_.links) values.push_back(static_cast<float>(link));
                      values.insert(values.end(), e.action.begin(), e.action.end());
                      TickCommand cmd = TickCommand::Param(ParamKind::TargetTrack, std::move(values));
                      cmd.index = static_cast<uint32_t>(e.Steps());
                      cmd.element = static_cast<uint32_t>(demos_.links.size());
                      cmd.env = e.env;
                      demo_play_seq_ = worker_.Push(std::move(cmd));
                      ui_state_.demos.playing = static_cast<int>(k);
                      ui_state_.demos.left = static_cast<uint32_t>(e.Steps());
                      if (BoolArg(args, "play", true)) ui_state_.playing = true;
                      return Ok(Object({{"episode", json::Value::Int(static_cast<int64_t>(k))},
                                        {"env", json::Value::Int(e.env)},
                                        {"steps", json::Value::Int(static_cast<int64_t>(e.Steps()))}}));
                  }});
    actions_.Add({"demo.stop", "Stop Replay", "robot", "", ShortcutScope::Global,
                  [this] { return ui_state_.demos.playing >= 0; }, {},
                  [this](const json::Value&) {
                      StopReplay();
                      return Ok(DemoStatus());
                  }});
    // The episodes as robomimic-style HDF5 through nuka.editor.demos; the dialog asks without `path`.
    actions_.Add({"demo.save", "Save Demonstrations...", "robot", "", ShortcutScope::Global, has_demos, {},
                  [this](const json::Value& args) {
                      if (demos_.episodes.empty()) return ActionResult::Fail("no demonstrations are recorded");
                      std::string path = StrArg(args, "path");
                      if (path.empty())
                          path = SaveFileDialog(nui::Tr("Save Demonstrations"), nui::Tr("HDF5 demonstrations"), "*.hdf5;*.h5",
                                                "demos.hdf5");
                      if (path.empty()) return ActionResult::Fail("no demonstration file chosen");
                      WriteDemoExchange(demos_, cache_.File("demos.json"), cache_.File("demos.bin"));
                      ui_state_.demos.error = RunDemoExchange("_save", path);
                      if (!ui_state_.demos.error.empty()) return ActionResult::Fail(ui_state_.demos.error);
                      size_t steps = 0u;
                      for (const DemoEpisode& e : demos_.episodes) steps += e.Steps();
                      ui_.Nui().Toast(nui::Level::Ok, std::string(nui::Tr("Demonstrations saved")) + " " + FileName(path));
                      return Ok(Object({{"path", json::Value::Str(path)},
                                        {"episodes", json::Value::Int(static_cast<int64_t>(demos_.episodes.size()))},
                                        {"steps", json::Value::Int(static_cast<int64_t>(steps))}}));
                  }});
    // Replaces the episodes with a file's, its joints matched by name to the scene's.
    actions_.Add({"demo.load", "Load Demonstrations...", "robot", "", ShortcutScope::Global, has_robot, {},
                  [this](const json::Value& args) {
                      if (demo_recording_) return ActionResult::Fail("stop the recording first");
                      std::string path = StrArg(args, "path");
                      if (path.empty())
                          path = OpenFileDialog(nui::Tr("Load Demonstrations"), nui::Tr("HDF5 demonstrations"), "*.hdf5;*.h5");
                      if (path.empty()) return ActionResult::Fail("no demonstration file chosen");
                      DemoView& d = ui_state_.demos;
                      d.error = RunDemoExchange("_load", path);
                      if (!d.error.empty()) return ActionResult::Fail(d.error);
                      DemoSet set = ReadDemoExchange(cache_.File("demos.json"), cache_.File("demos.bin"));
                      std::string missing;
                      for (size_t i = 0; i < set.joints.size(); ++i) {
                          const int j = FindJoint(ui_state_.robot.joints, set.joints[i]);
                          if (j >= 0)
                              set.links[i] = ui_state_.robot.joints[static_cast<size_t>(j)].link;
                          else
                              missing += (missing.empty() ? "" : ", ") + set.joints[i];
                      }
                      if (!missing.empty()) {
                          d.error = "'" + FileName(path) + "' drives joints the scene lacks: " + missing;
                          return ActionResult::Fail(d.error);
                      }
                      demos_ = std::move(set);
                      if (d.playing >= 0) StopReplay();
                      d.selected = demos_.episodes.empty() ? -1 : 0;
                      SyncDemoView();
                      ui_.Nui().Toast(nui::Level::Ok, std::string(nui::Tr("Demonstrations loaded")) + " " + FileName(path));
                      json::Value v = DemoStatus();
                      v.Set("path", json::Value::Str(path));
                      return Ok(std::move(v));
                  }});
    // Drops episode `episode`, without it all of them.
    actions_.Add({"demo.discard", "Discard Demonstrations", "robot", "", ShortcutScope::Global, has_demos, {},
                  [this, episode_arg](const json::Value& args) {
                      size_t k = 0u;
                      if (args.Find("episode") == nullptr) {
                          demos_.episodes.clear();
                      } else if (episode_arg(args, &k)) {
                          demos_.episodes.erase(demos_.episodes.begin() + static_cast<std::ptrdiff_t>(k));
                      } else {
                          return ActionResult::Fail(Format("episode: 0 to %zu", demos_.episodes.size() - 1u));
                      }
                      if (ui_state_.demos.playing >= 0) StopReplay();
                      SyncDemoView();
                      return Ok(DemoStatus());
                  }});
}

void EditorHost::State::RegisterOverlayActions() {
    // Each switch given sets it; the result holds them all.
    actions_.Add({"view.robot_overlays", "Robot Overlays", "view", "", ShortcutScope::Global, {},
                  [this] { return ui_state_.robot.overlays.on; },
                  [this](const json::Value& args) {
                      RobotOverlays& o = ui_state_.robot.overlays;
                      json::Value v = json::Value::Object();
                      for (const auto& [key, flag] : std::initializer_list<std::pair<const char*, bool*>>{
                               {"on", &o.on}, {"frames", &o.frames}, {"axes", &o.axes}, {"com", &o.com},
                               {"inertia", &o.inertia}, {"translucent", &o.translucent}}) {
                          *flag = BoolArg(args, key, *flag);
                          v.Set(key, json::Value::Bool(*flag));
                      }
                      return Ok(std::move(v));
                  }});
}

void EditorHost::State::RegisterGainActions() {
    const auto has_joints = [this] { return loaded_ != nullptr && !ui_state_.robot.joints.empty(); };
    // Drives one joint's target through a step or sine from where it is, in a copy of the world the
    // test throws away; defaults come from the tuner, its joint else the selected or first one, and
    // the tuner keeps what ran.
    actions_.Add({"robot.gain_test", "Run Gain Test", "robot", "", ShortcutScope::Global, has_joints, {},
                  [this](const json::Value& args) {
                      RobotView& v = ui_state_.robot;
                      GainTuner& g = v.tuner;
                      if (g.running != 0u) return ActionResult::Fail("a gain test is running");
                      std::string fallback = g.joint;
                      if (FindJoint(v.joints, fallback) < 0)
                          fallback = v.joints[static_cast<size_t>(std::max(v.focus, 0))].name;
                      const std::string name = StrArg(args, "joint", fallback);
                      const int j = FindJoint(v.joints, name);
                      if (j < 0) return ActionResult::Fail("no joint '" + name + "'");
                      const std::string kind = StrArg(args, "kind", g.sine ? "sine" : "step");
                      if (kind != "step" && kind != "sine") return ActionResult::Fail("the kind is step or sine");
                      const double amplitude = NumArg(args, "amplitude", g.amplitude);
                      const double frequency = NumArg(args, "frequency", g.frequency);
                      const double duration = NumArg(args, "duration", g.duration);
                      if (!std::isfinite(amplitude) || amplitude == 0.0)
                          return ActionResult::Fail("the amplitude must be a nonzero number");
                      if (!(duration > 0.0 && duration <= 10.0)) return ActionResult::Fail("the duration must be in (0, 10] s");
                      if (kind == "sine" && !(frequency > 0.0 && std::isfinite(frequency)))
                          return ActionResult::Fail("a sine's frequency must be above 0");
                      const RobotJoint& joint = v.joints[static_cast<size_t>(j)];
                      if (joint.drive != "position") return ActionResult::Fail("'" + name + "' has no position drive");
                      g.joint = name;
                      g.sine = kind == "sine";
                      g.amplitude = static_cast<float>(amplitude);
                      g.frequency = static_cast<float>(frequency);
                      g.duration = static_cast<float>(duration);
                      g.next = GainRun{name, joint.prismatic, joint.kp, joint.damping, joint.force_limit,
                                       static_cast<float>(duration), {}, {}};
                      TickCommand cmd;
                      cmd.kind = CommandKind::GainTest;
                      cmd.index = joint.link;
                      cmd.values = {kind == "sine" ? 1.0f : 0.0f, static_cast<float>(amplitude), static_cast<float>(frequency),
                                    static_cast<float>(duration)};
                      g.running = worker_.Push(std::move(cmd));
                      return Ok(Object({{"joint", json::Value::Str(name)},
                                        {"kind", json::Value::Str(kind)},
                                        {"steps", json::Value::Int(std::lround(duration / static_cast<double>(opt_.dt)))}}));
                  }});
    // Whether a test runs, how many runs the tuner keeps and the newest one's figures.
    actions_.Add({"robot.gain_status", "Gain Test Status", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      const GainTuner& g = ui_state_.robot.tuner;
                      return Ok(Object({{"running", json::Value::Bool(g.running != 0u)},
                                        {"runs", json::Value::Int(static_cast<int64_t>(g.runs.size()))},
                                        {"last", g.runs.empty() ? json::Value::Null() : GainRunJson(g.runs.back(), false)}}));
                  }});
    // The kept runs, oldest first; `trace` adds their samples.
    actions_.Add({"robot.gain_runs", "Gain Test Runs", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const bool trace = BoolArg(args, "trace", false);
                      json::Value runs = json::Value::Array();
                      for (const GainRun& run : ui_state_.robot.tuner.runs) runs.PushBack(GainRunJson(run, trace));
                      return Ok(std::move(runs));
                  }});
    actions_.Add({"robot.gain_clear", "Clear Gain Runs", "robot", "", ShortcutScope::Global,
                  [this] { return !ui_state_.robot.tuner.runs.empty(); }, {},
                  [this](const json::Value&) {
                      ui_state_.robot.tuner.runs.clear();
                      return Ok(Object({{"count", json::Value::Int(0)}}));
                  }});
    actions_.Add({"robot.gain_tuner", "Gain Tuner", "robot", "", ShortcutScope::Global, has_joints,
                  [this] { return ui_state_.robot.tuner.open; },
                  [this](const json::Value& args) {
                      GainTuner& g = ui_state_.robot.tuner;
                      g.open = BoolArg(args, "on", !g.open);
                      return Ok(Object({{"on", json::Value::Bool(g.open)}}));
                  }});
}

void EditorHost::State::RegisterAssemblyActions() {
    const auto has_scene = [this] { return loaded_ != nullptr; };
    // The shown environment's links in link order, each with its parent link and incoming joint.
    actions_.Add({"robot.links", "Robot Links", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      const RobotView& v = ui_state_.robot;
                      const bool posed = v.link_poses.size() == v.links.size();
                      json::Value links = json::Value::Array();
                      for (size_t l = 0; l < v.links.size(); ++l)
                          links.PushBack(LinkJson(v.links[l], v.links, posed ? &v.link_poses[l] : nullptr));
                      float mass = 0.0f, weighed = 0.0f;
                      for (const RobotLink& l : v.links) mass += l.mass;
                      math::Vec3 com;
                      const bool massive = posed && RobotCom(v, &com, &weighed);
                      return Ok(Object({{"links", std::move(links)},
                                        {"mass", json::Value::Float(mass)},
                                        {"com", massive ? Vec3Json(com) : json::Value::Null()},
                                        {"dof", json::Value::Int(static_cast<int64_t>(v.joints.size()))}}));
                  }});
    // An asset's links, its roots first, which the Assembler offers as mounts; without `path` it asks.
    actions_.Add({"robot.asset_links", "Choose Asset to Attach...", "robot", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      Assembler& a = ui_state_.robot.assembler;
                      std::string path = StrArg(args, "path");
                      if (path.empty())
                          path = OpenFileDialog(nui::Tr("Attach Asset"), nui::Tr("Assets"),
                                                "*.urdf;*.xml;*.usd;*.usda;*.stl;*.obj;*.nks");
                      if (path.empty()) return ActionResult::Fail("no asset chosen");
                      const ImportAnalysis analysis = AnalyzeImport(path, StrArg(args, "profile", a.profile));
                      a.path = path;
                      a.profile = analysis.profile.empty() ? a.profile : analysis.profile;
                      a.mount.clear();
                      a.asset_links.clear();
                      a.asset_error = Importable(analysis) ? std::string() : ImportError(analysis);
                      if (!a.asset_error.empty()) return ActionResult::Fail(a.asset_error);
                      const scene::SceneIR& s = *analysis.scene;
                      std::vector<uint8_t> jointed(s.RigidBodyCount(), 0u);
                      for (const scene::JointRecord& j : s.Joints()) jointed[j.child_body] = 1u;
                      std::vector<std::string> roots, rest;
                      for (scene::BodyId b = 0; b < s.RigidBodyCount(); ++b)
                          (jointed[b] != 0u ? rest : roots).push_back(s.GetBody(b).name);
                      a.asset_links = roots;
                      a.asset_links.insert(a.asset_links.end(), rest.begin(), rest.end());
                      return Ok(Object({{"path", json::Value::Str(path)},
                                        {"links", StrList(a.asset_links)},
                                        {"roots", StrList(roots)}}));
                  }});
    // The attach op's fields from an action's arguments: `frame` takes `pos` with a `quat` (w, x, y, z)
    // or XYZ degrees `rot`, and `joint_name` names the new joint.
    const auto attach_op = [](const json::Value& args) {
        json::Value after = Object({{"parent", json::Value::Str(StrArg(args, "parent"))},
                                    {"joint", json::Value::Str(StrArg(args, "joint", "fixed"))}});
        if (const json::Value* axis = args.Find("axis"); axis != nullptr && !axis->IsNull()) after.Set("axis", *axis);
        if (const json::Value* frame = args.Find("frame"); frame != nullptr && !frame->IsNull()) {
            float p[3] = {0.0f, 0.0f, 0.0f}, r[3] = {0.0f, 0.0f, 0.0f};
            for (const auto& [key, out] : {std::pair{"pos", p}, std::pair{"rot", r}})
                if (const json::Value* v = frame->Find(key))
                    for (size_t i = 0; i < 3u; ++i) out[i] = v->Elements()[i].AsFloat();
            math::Transform t;
            t.position = math::Vec3{p[0], p[1], p[2]};
            t.rotation = QuatFromEulerDeg(math::Vec3{r[0], r[1], r[2]});
            if (const json::Value* q = frame->Find("quat")) {
                const auto& e = q->Elements();
                t.rotation = math::Quat(e[0].AsFloat(), e[1].AsFloat(), e[2].AsFloat(), e[3].AsFloat()).Normalized();
            }
            after.Set("frame", TransformToJson(t));
        }
        if (const std::string mount = StrArg(args, "mount"); !mount.empty()) after.Set("mount", json::Value::Str(mount));
        if (const std::string name = StrArg(args, "joint_name"); !name.empty()) after.Set("name", json::Value::Str(name));
        return after;
    };
    // Names the joint the attached root now hangs off and closes the Assembler.
    const auto attached = [this](ActionResult r) {
        if (!r.ok) return r;
        const std::string focus = r.value.At("focus").AsString();
        for (const scene::JointRecord& j : loaded_->scene.Joints())
            if (PathOfEntity(loaded_->scene.EntityOfBody(j.child_body)) == focus) r.value.Set("joint", json::Value::Str(j.name));
        ui_state_.robot.assembler.open = false;
        return r;
    };
    // Hangs the body `child` and what its joints carry off the link `parent` through a new joint; `frame`
    // puts `mount` (default the child) in the parent's frame, else it stays where it is.
    actions_.Add({"robot.attach", "Attach to Robot", "robot", "", ShortcutScope::Global, has_scene, {},
                  [this, attach_op, attached](const json::Value& args) {
                      const std::string child = StrArg(args, "child");
                      const std::string label = "Attach " + child.substr(child.rfind('/') + 1u);
                      return attached(SubmitTxn(EditTxn{label, {EditOp{child, "attach", json::Value(), attach_op(args)}}}));
                  }});
    // Adds the asset at `path` and hangs its link `mount` (default its root) off `parent` in one edit;
    // without `frame` the mount takes the parent's frame.
    actions_.Add({"robot.attach_asset", "Attach Asset to Robot", "robot", "", ShortcutScope::Global, has_scene, {},
                  [this, attach_op, attached](const json::Value& args) {
                      const std::string path = StrArg(args, "path");
                      if (!fs::exists(Utf8Path(path))) return ActionResult::Fail("no file at " + path);
                      const std::string name = StrArg(args, "name", PathText(Utf8Path(path).stem()));
                      json::Value after = attach_op(args);
                      if (after.Find("frame") == nullptr) after.Set("frame", TransformToJson(math::Transform::Identity()));
                      after.Set("asset", Object({{"path", json::Value::Str(path)},
                                                 {"profile", json::Value::Str(StrArg(args, "profile", "strict"))},
                                                 {"name", json::Value::Str(name)}}));
                      return attached(SubmitTxn(EditTxn{"Attach " + name, {EditOp{"", "attach", json::Value(), std::move(after)}}}));
                  }});
    // Opening it on a selected link makes that link the parent.
    actions_.Add({"robot.assembler", "Robot Assembler", "robot", "", ShortcutScope::Global, has_scene,
                  [this] { return ui_state_.robot.assembler.open; },
                  [this](const json::Value& args) {
                      Assembler& a = ui_state_.robot.assembler;
                      a.open = BoolArg(args, "on", !a.open);
                      const std::string selected = ui_state_.selected_entity == scene::kInvalidEntity
                                                       ? std::string()
                                                       : PathOfEntity(ui_state_.selected_entity);
                      for (const RobotLink& link : ui_state_.robot.links)
                          if (a.open && link.path == selected) a.parent = selected;
                      return Ok(Object({{"on", json::Value::Bool(a.open)}}));
                  }});
}

void EditorHost::State::RegisterPoseActions() {
    const auto has_joints = [this] { return loaded_ != nullptr && !ui_state_.robot.joints.empty(); };
    const auto has_poses = [this] { return loaded_ != nullptr && !ui_state_.robot.poses.empty(); };
    actions_.Add({"robot.poses", "Joint Poses", "robot", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) { return Ok(PosesJson(ui_state_.robot.poses).At("poses")); }});
    // Keeps the shown environment's joint positions as `name`, replacing a pose of that name;
    // without one it is "Pose N".
    actions_.Add({"robot.pose_capture", "Capture Pose", "robot", "", ShortcutScope::Global, has_joints, {},
                  [this](const json::Value& args) {
                      RobotView& v = ui_state_.robot;
                      std::string name = StrArg(args, "name");
                      for (size_t n = v.poses.size() + 1u; name.empty(); ++n)
                          if (FindPose(Format("Pose %zu", n)) == nullptr) name = Format("Pose %zu", n);
                      JointPose pose{name, {}};
                      for (const RobotJoint& j : v.joints)
                          pose.joints.emplace_back(j.name, j.link < v.live.q.size() ? v.live.q[j.link] : 0.0f);
                      JointPose* p = FindPose(name);
                      if (p == nullptr) p = &v.poses.emplace_back();
                      *p = std::move(pose);
                      SavePoses();
                      return Ok(PoseJson(*p));
                  }});
    // Blends the shown environment's targets to `name` over `duration` simulated seconds, 0 at
    // once; the robot's joints the pose lacks keep their targets.
    actions_.Add({"robot.pose_apply", "Apply Pose", "robot", "", ShortcutScope::Global, has_joints, {},
                  [this](const json::Value& args) {
                      const RobotView& v = ui_state_.robot;
                      const std::string name = StrArg(args, "name");
                      const JointPose* p = FindPose(name);
                      if (p == nullptr) return ActionResult::Fail("no pose '" + name + "'");
                      const double duration = NumArg(args, "duration", 0.0);
                      if (!(duration >= 0.0)) return ActionResult::Fail("duration must be at least 0");
                      std::vector<float> ramp;
                      std::vector<std::string> missing, unknown;
                      for (const RobotJoint& j : v.joints) {
                          const auto it = std::find_if(p->joints.begin(), p->joints.end(),
                                                       [&](const auto& e) { return e.first == j.name; });
                          if (it == p->joints.end()) {
                              missing.push_back(j.name);
                              continue;
                          }
                          ramp.insert(ramp.end(), {static_cast<float>(j.link), 0.0f, it->second});
                          ui_state_.drive_targets[j.link] = it->second;
                      }
                      for (const auto& [joint, q] : p->joints)
                          if (FindJoint(v.joints, joint) < 0) unknown.push_back(joint);
                      if (ramp.empty()) return ActionResult::Fail("pose '" + name + "' names none of the robot's joints");
                      const size_t set = ramp.size() / 3u;
                      TickCommand cmd = TickCommand::Param(ParamKind::TargetRamp, std::move(ramp));
                      cmd.index = static_cast<uint32_t>(std::lround(duration / static_cast<double>(opt_.dt)));
                      worker_.Push(cmd);
                      return Ok(Object({{"name", json::Value::Str(name)},
                                        {"set", json::Value::Int(static_cast<int64_t>(set))},
                                        {"missing", StrList(missing)},
                                        {"unknown", StrList(unknown)},
                                        {"steps", json::Value::Int(cmd.index)}}));
                  }});
    // Writes `name` into its joints' initial positions as one undoable edit; sim.reset starts there.
    actions_.Add({"robot.pose_initial", "Set Pose as Initial", "robot", "", ShortcutScope::Global, has_joints, {},
                  [this](const json::Value& args) {
                      const RobotView& v = ui_state_.robot;
                      const std::string name = StrArg(args, "name");
                      const JointPose* p = FindPose(name);
                      if (p == nullptr) return ActionResult::Fail("no pose '" + name + "'");
                      EditTxn txn{"Initial Pose " + name, {}};
                      for (const auto& [joint, q] : p->joints) {
                          const int j = FindJoint(v.joints, joint);
                          if (j >= 0)
                              txn.ops.push_back(EditOp{v.joints[static_cast<size_t>(j)].path, "initial_position", json::Value(),
                                                       json::Value::Float(q)});
                      }
                      if (txn.ops.empty()) return ActionResult::Fail("pose '" + name + "' names none of the robot's joints");
                      return SubmitTxn(txn);
                  }});
    actions_.Add({"robot.pose_rename", "Rename Pose", "robot", "", ShortcutScope::Global, has_poses, {},
                  [this](const json::Value& args) {
                      const std::string name = StrArg(args, "name"), to = StrArg(args, "to");
                      JointPose* p = FindPose(name);
                      if (p == nullptr) return ActionResult::Fail("no pose '" + name + "'");
                      if (to.empty() || (to != name && FindPose(to) != nullptr))
                          return ActionResult::Fail("the new name is empty or taken");
                      p->name = to;
                      SavePoses();
                      return Ok(Object({{"name", json::Value::Str(to)}}));
                  }});
    actions_.Add({"robot.pose_remove", "Remove Pose", "robot", "", ShortcutScope::Global, has_poses, {},
                  [this](const json::Value& args) {
                      std::vector<JointPose>& poses = ui_state_.robot.poses;
                      const std::string name = StrArg(args, "name");
                      const JointPose* p = FindPose(name);
                      if (p == nullptr) return ActionResult::Fail("no pose '" + name + "'");
                      poses.erase(poses.begin() + (p - poses.data()));
                      SavePoses();
                      return Ok(Object({{"count", json::Value::Int(static_cast<int64_t>(poses.size()))}}));
                  }});
    // Writes the poses as {"poses": [...]}; the dialog asks without `path`.
    actions_.Add({"robot.poses_save", "Export Poses...", "robot", "", ShortcutScope::Global, has_poses, {},
                  [this](const json::Value& args) {
                      std::string path = StrArg(args, "path");
                      if (path.empty()) path = SaveFileDialog(nui::Tr("Export Poses"), nui::Tr("Joint poses"), "*.json", "poses.json");
                      if (path.empty()) return ActionResult::Fail("no pose file chosen");
                      std::ofstream out(Utf8Path(path), std::ios::binary | std::ios::trunc);
                      out << PosesJson(ui_state_.robot.poses).Dump(2);
                      if (!out) return ActionResult::Fail("cannot write " + path);
                      return Ok(Object({{"path", json::Value::Str(path)},
                                        {"count", json::Value::Int(static_cast<int64_t>(ui_state_.robot.poses.size()))}}));
                  }});
    // Adds a file's poses, replacing those of the same name.
    actions_.Add({"robot.poses_load", "Import Poses...", "robot", "", ShortcutScope::Global,
                  [this] { return loaded_ != nullptr; }, {},
                  [this](const json::Value& args) {
                      std::string path = StrArg(args, "path");
                      if (path.empty()) path = OpenFileDialog(nui::Tr("Import Poses"), nui::Tr("Joint poses"), "*.json");
                      if (path.empty()) return ActionResult::Fail("no pose file chosen");
                      std::vector<JointPose> read;
                      if (!ReadPoses(path, &read)) return ActionResult::Fail("cannot read " + path);
                      const size_t added = read.size();
                      for (JointPose& p : read) {
                          JointPose* have = FindPose(p.name);
                          if (have != nullptr) *have = std::move(p);
                          else ui_state_.robot.poses.push_back(std::move(p));
                      }
                      SavePoses();
                      return Ok(Object({{"path", json::Value::Str(path)},
                                        {"read", json::Value::Int(static_cast<int64_t>(added))},
                                        {"count", json::Value::Int(static_cast<int64_t>(ui_state_.robot.poses.size()))}}));
                  }});
}

void EditorHost::State::RegisterFileActions() {
    const auto has_scene = [this] { return loaded_ != nullptr; };
    const auto owns_world = [this] { return !borrowed_; };
    // Opening replaces the editor's own world; a script's world stays and a new editor opens.
    const auto open = [this](const std::string& id, const json::Value& args) {
        std::string path = StrArg(args, "path");
        if (path.empty())
            path = OpenFileDialog(nui::Tr("Open Scene"), nui::Tr("Scenes"), "*.nks;*.urdf;*.xml;*.usd;*.usda");
        if (path.empty()) return ActionResult::Fail("no scene chosen");
        if (!fs::exists(Utf8Path(path))) return ActionResult::Fail("no file at " + path);
        if (borrowed_) return Spawn({path});
        json::Value resolved = args;
        resolved.Set("path", json::Value::Str(path));
        ActionResult asked;
        if (!Confirmed(id, resolved, &asked)) return asked;
        // Formats other than NKS import into an untitled document.
        return OpenDocument(path, Utf8Path(path).extension() == ".nks" ? path : std::string());
    };
    actions_.Add({"file.new", "New Scene", "file", "Ctrl+N", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const std::string name = StrArg(args, "template", "empty");
                      if (borrowed_) return Spawn({"--template", name});
                      const std::string file = Resource("assets/editor/templates/" + name + ".nks");
                      if (!fs::exists(Utf8Path(file))) return ActionResult::Fail("no scene template '" + name + "'");
                      ActionResult asked;
                      if (!Confirmed("file.new", args, &asked)) return asked;
                      return OpenDocument(file, std::string());
                  }});
    actions_.Add({"file.open", "Open Scene...", "file", "", ShortcutScope::Global, {}, {},
                  [open](const json::Value& args) { return open("file.open", args); }});
    actions_.Add({"file.open_recent", "Open Recent", "file", "", ShortcutScope::Global, {}, {},
                  [open](const json::Value& args) { return open("file.open_recent", args); }});
    actions_.Add({"file.forget_recent", "Remove from Recent", "file", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      profile_->ForgetRecent(StrArg(args, "path"));
                      start_stale_ = true;
                      return Ok();
                  }});
    actions_.Add({"file.open_window", "Open in New Window...", "file", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      std::string path = StrArg(args, "path");
                      if (path.empty() && !BoolArg(args, "empty", false))
                          path = OpenFileDialog(nui::Tr("Open Scene"), nui::Tr("Scenes"), "*.nks;*.urdf;*.xml;*.usd;*.usda");
                      return Spawn(path.empty() ? std::vector<std::string>{} : std::vector<std::string>{path});
                  }});
    actions_.Add({"file.revert", "Revert", "file", "", ShortcutScope::Global,
                  [this] { return loaded_ != nullptr && !borrowed_ && !doc_path_.empty() && doc_.Dirty(); }, {},
                  [this](const json::Value&) {
                      const std::string path = doc_path_;
                      return OpenDocument(path, path);
                  }});
    actions_.Add({"file.close", "Close Scene", "file", "", ShortcutScope::Global,
                  [this] { return loaded_ != nullptr && !borrowed_; }, {},
                  [this](const json::Value& args) {
                      ActionResult asked;
                      if (!Confirmed("file.close", args, &asked)) return asked;
                      worker_.Exclusive([&] { ApplyUnload(); });
                      DropAutosave();
                      doc_path_.clear();
                      ui_state_.save_path[0] = '\0';
                      return Ok();
                  }});
    actions_.Add({"file.recover", "Recover Autosave", "file", "", ShortcutScope::Global, owns_world, {},
                  [this](const json::Value& args) {
                      const std::string file = StrArg(args, "path");
                      std::string note_text;
                      if (!ReadEntireFile(PathText(SidecarOf(file)), &note_text)) return ActionResult::Fail("no autosave at " + file);
                      ActionResult asked;
                      if (!Confirmed("file.recover", args, &asked)) return asked;
                      const std::string doc = json::Value::Parse(note_text).At("path").AsString();
                      ActionResult result = OpenDocument(file, doc);
                      if (!result.ok) return result;
                      // The recovered file becomes this session's autosave until the document is saved.
                      autosave_file_ = file;
                      doc_.MarkUnsaved();
                      Autosave();
                      start_stale_ = true;
                      return result;
                  }});
    actions_.Add({"file.autosave", "Autosave", "file", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value&) {
                      Autosave();
                      return Ok(Object({{"path", json::Value::Str(autosave_file_)}}));
                  },
                  "", true});
    actions_.Add({"file.unload", "Unload", "file", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value&) {
                      worker_.Exclusive([&] { ApplyUnload(); });
                      return Ok();
                  },
                  "", true});
    // An untitled document asks where to save.
    actions_.Add({"file.save", "Save", "file", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      std::string path = StrArg(args, "path", doc_path_);
                      if (path.empty()) path = SaveFileDialog(nui::Tr("Save Scene"), nui::Tr("Nuka scenes"), "*.nks", "untitled.nks");
                      return path.empty() ? ActionResult::Fail("no save path chosen") : SaveScene(path);
                  }});
    actions_.Add({"file.save_as", "Save As...", "file", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      std::string path = StrArg(args, "path");
                      if (path.empty())
                          path = SaveFileDialog(nui::Tr("Save Scene"), nui::Tr("Nuka scenes"), "*.nks",
                                                (doc_path_.empty() ? std::string("untitled.nks") : doc_path_).c_str());
                      return path.empty() ? ActionResult::Fail("no save path chosen") : SaveScene(path);
                  }});
}

void EditorHost::State::RegisterAppActions() {
    // A script's world outlives the window: closing only hides it and ends app.run().
    actions_.Add({"app.exit", "Exit", "file", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      ActionResult asked;
                      if (!Confirmed("app.exit", args, &asked)) return asked;
                      running_ = false;
                      if (native_window_ != nullptr) SDL_HideWindow(static_cast<SDL_Window*>(native_window_));
                      return Ok();
                  }});
    // The title bar's window buttons, its close button being app.exit, and the window's state.
    const auto windowed = [this] { return native_window_ != nullptr; };
    actions_.Add({"window.minimize", "Minimize", "window", "", ShortcutScope::Global, windowed, {},
                  [this](const json::Value&) {
                      SDL_MinimizeWindow(static_cast<SDL_Window*>(native_window_));
                      return Ok();
                  }});
    actions_.Add({"window.maximize", "Maximize", "window", "", ShortcutScope::Global, windowed,
                  [this] { return ui_state_.title_bar.maximized; },
                  [this](const json::Value&) {
                      SDL_Window* window = static_cast<SDL_Window*>(native_window_);
                      if (SDL_GetWindowFlags(window) & SDL_WINDOW_MAXIMIZED) SDL_RestoreWindow(window);
                      else SDL_MaximizeWindow(window);
                      return Ok();
                  }});
    actions_.Add({"window.restore", "Restore", "window", "", ShortcutScope::Global, windowed, {},
                  [this](const json::Value&) {
                      SDL_RestoreWindow(static_cast<SDL_Window*>(native_window_));
                      return Ok();
                  }});
    actions_.Add({"window.state", "Window State", "window", "", ShortcutScope::Global, windowed, {},
                  [this](const json::Value&) {
                      SDL_Window* window = static_cast<SDL_Window*>(native_window_);
                      const SDL_WindowFlags flags = SDL_GetWindowFlags(window);
                      int x = 0, y = 0, w = 0, h = 0;
                      SDL_GetWindowPosition(window, &x, &y);
                      SDL_GetWindowSize(window, &w, &h);
                      json::Value drag = json::Value::Array();
                      for (const auto& [x0, x1] : ui_state_.title_bar.drag) {
                          json::Value span = json::Value::Array();
                          span.PushBack(json::Value::Float(x0));
                          span.PushBack(json::Value::Float(x1));
                          drag.PushBack(std::move(span));
                      }
                      return Ok(Object({{"maximized", json::Value::Bool((flags & SDL_WINDOW_MAXIMIZED) != 0)},
                                        {"minimized", json::Value::Bool((flags & SDL_WINDOW_MINIMIZED) != 0)},
                                        {"focused", json::Value::Bool((flags & SDL_WINDOW_INPUT_FOCUS) != 0)},
                                        {"x", json::Value::Int(x)}, {"y", json::Value::Int(y)},
                                        {"width", json::Value::Int(w)}, {"height", json::Value::Int(h)},
                                        {"title_bar", Object({{"height", json::Value::Float(ui_state_.title_bar.height)},
                                                              {"drag", std::move(drag)}})}}));
                  }});
    actions_.Add({"app.answer_prompt", "Answer", "app", "", ShortcutScope::Global, [this] { return prompt_.has_value(); }, {},
                  [this](const json::Value& args) {
                      const Prompt asked = *prompt_;
                      const std::string choice = StrArg(args, "choice");
                      if (choice != "save" && choice != "discard" && choice != "cancel")
                          return ActionResult::Fail("the choice is save, discard or cancel");
                      prompt_.reset();
                      ui_state_.prompt.clear();
                      if (choice == "cancel") return Ok(Object({{"choice", json::Value::Str(choice)}}));
                      json::Value again = asked.args;
                      again.Set("discard", json::Value::Bool(true));
                      if (choice == "save") {
                          json::Value save = json::Value::Object();
                          if (const json::Value* path = args.Find("path")) save.Set("path", *path);
                          const ActionResult saved = RunAction("file.save", save, "prompt");
                          if (!saved.ok) return saved;
                          resume_ = Prompt{asked.action, std::move(again)};
                          return Ok(Object({{"choice", json::Value::Str(choice)}}));
                      }
                      return RunAction(asked.action, again, "prompt");
                  },
                  "", true});
    actions_.Add({"app.check", "Report Check", "app", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      StartEntry check{StrArg(args, "name"), StrArg(args, "value"), "", "", BoolArg(args, "ok", true)};
                      WriteLog("[nuka_editor] check " + check.title + ": " + check.detail + (check.available ? "\n" : " (failed)\n"));
                      checks_.push_back(std::move(check));
                      start_stale_ = true;
                      return Ok();
                  },
                  "", true});
}

void EditorHost::State::RegisterEditActions() {
    const auto has_scene = [this] { return loaded_ != nullptr; };
    // A node edit targets `path`, or the selection without one.
    const auto node_edit = [this](const char* label, const char* field, const json::Value& args, json::Value after) {
        const std::string path = StrArg(args, "path", SelectedPath());
        if (path.empty()) return ActionResult::Fail("no node selected");
        return SubmitTxn(EditTxn{label, {EditOp{path, field, json::Value(), std::move(after)}}});
    };
    actions_.Add({"edit.undo", "Undo", "edit", "Ctrl+Z", ShortcutScope::Global,
                  [this] { return loaded_ != nullptr && doc_.CanUndo(); }, {},
                  [this](const json::Value&) { return StepHistory(/*redo=*/false); }});
    actions_.Add({"edit.redo", "Redo", "edit", "Ctrl+Y, Ctrl+Shift+Z", ShortcutScope::Global,
                  [this] { return loaded_ != nullptr && doc_.CanRedo(); }, {},
                  [this](const json::Value&) { return StepHistory(/*redo=*/true); }});
    actions_.Add({"edit.apply", "Apply Edit", "edit", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) { return SubmitTxn(TxnFromJson(args)); }});
    actions_.Add({"edit.rename", "Rename", "edit", "", ShortcutScope::Global, has_scene, {},
                  [node_edit](const json::Value& args) {
                      return node_edit("Rename", "name", args, json::Value::Str(StrArg(args, "name")));
                  }});
    // Deletes `paths`, `path` or the selection in one step; a node under another goes with it.
    actions_.Add({"edit.delete", "Delete Subtree", "edit", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const std::vector<std::string> paths = TargetPaths(args);
                      if (paths.empty()) return ActionResult::Fail("no node selected");
                      EditTxn txn{"Delete", {}};
                      for (const std::string& p : paths)
                          if (std::none_of(paths.begin(), paths.end(), [&](const std::string& q) { return Under(p, q); }))
                              txn.ops.push_back(EditOp{p, "delete", json::Value(), json::Value()});
                      return SubmitTxn(txn);
                  }});
    actions_.Add({"edit.reparent", "Reparent", "edit", "", ShortcutScope::Global, has_scene, {},
                  [node_edit](const json::Value& args) {
                      return node_edit("Reparent", "reparent", args, json::Value::Str(StrArg(args, "parent")));
                  }});
    // Without a pose, a spawn drops in above the selection.
    actions_.Add({"scene.spawn", "Add Primitive", "scene", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const std::string kind = StrArg(args, "kind", "box");
                      json::Value pose;
                      if (const json::Value* p = args.Find("pose")) {
                          pose = *p;
                      } else {
                          math::Transform placement;
                          placement.position = math::Vec3{0.0f, 0.0f, kSpawnDropHeight};
                          if (ui_state_.selected_entity != scene::kInvalidEntity) {
                              placement.position = AuthoredWorldTransform(loaded_->scene, ui_state_.selected_entity).position;
                              placement.position.z += kSpawnDropHeight;
                          }
                          pose = TransformToJson(placement);
                      }
                      const json::Value spawn = Object({{"kind", json::Value::Str(kind)}, {"pose", pose}});
                      return SubmitTxn(EditTxn{"Spawn " + kind, {EditOp{"", "spawn", json::Value(), spawn}}});
                  }});
    // `mode` add and toggle grow or shrink the selection; range takes `paths`, `path` leading.
    actions_.Add({"scene.select", "Select", "scene", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const std::string path = StrArg(args, "path");
                      const std::string mode = StrArg(args, "mode", "replace");
                      const scene::EntityId e = path.empty() ? scene::kInvalidEntity : EntityOfPath(*loaded_, path);
                      if (!path.empty() && e == scene::kInvalidEntity) return ActionResult::Fail("no node at '" + path + "'");
                      std::vector<scene::EntityId>& sel = ui_state_.selection;
                      const auto at = std::find(sel.begin(), sel.end(), e);
                      if (mode == "replace") {
                          sel.clear();
                          if (e != scene::kInvalidEntity) sel.push_back(e);
                      } else if (mode == "add") {
                          if (at == sel.end() && e != scene::kInvalidEntity) sel.push_back(e);
                      } else if (mode == "toggle") {
                          if (at != sel.end()) sel.erase(at);
                          else if (e != scene::kInvalidEntity) sel.push_back(e);
                      } else if (mode == "range") {
                          sel.clear();
                          for (const std::string& p : StrListArg(args, "paths"))
                              if (const auto n = loaded_->scene.Tree().NodeOf(p)) sel.push_back(n->entity);
                      } else {
                          return ActionResult::Fail("the mode is replace, add, toggle or range");
                      }
                      // The named node leads unless a toggle just dropped it.
                      const bool kept = std::find(sel.begin(), sel.end(), e) != sel.end();
                      ui_state_.selected_entity = kept ? e : sel.empty() ? scene::kInvalidEntity : sel.back();
                      return Ok(Object({{"path", json::Value::Str(SelectedPath())}, {"selection", StrList(SelectionPaths())}}));
                  }});
}

void EditorHost::State::RegisterScriptActions() {
    const auto has_scene = [this] { return loaded_ != nullptr; };
    const auto tab_json = [](const ScriptTab& tab) {
        return Object({{"tab", json::Value::Int(static_cast<int64_t>(tab.id))},
                       {"title", json::Value::Str(tab.title)},
                       {"path", json::Value::Str(tab.path)}});
    };
    actions_.Add({"script.new", "New Script", "script", "", ShortcutScope::Global, {}, {},
                  [this, tab_json](const json::Value& args) { return Ok(tab_json(ui_state_.scripts.Add(StrArg(args, "text")))); }});
    actions_.Add({"script.open", "Open Script...", "script", "", ShortcutScope::Global, {}, {},
                  [this, tab_json](const json::Value& args) {
                      std::string path = StrArg(args, "path");
                      if (path.empty()) path = OpenFileDialog(nui::Tr("Open Script"), nui::Tr("Python scripts"), "*.py");
                      if (path.empty()) return ActionResult::Fail("no file chosen");
                      ScriptTab& tab = ui_state_.scripts.Open(path);
                      if (const json::Value* line = args.Find("line")) ui_state_.scripts.Goto(tab, static_cast<int>(line->AsInt()));
                      return Ok(tab_json(tab));
                  }});
    // A buffer saves to the path given, else where the dialog says; saving runs it when asked to.
    actions_.Add({"script.save", "Save Script", "script", "", ShortcutScope::Global, {}, {},
                  [this, tab_json](const json::Value& args) {
                      ScriptTab& tab = TabArg(args);
                      std::string path = StrArg(args, "path");
                      if (path.empty() && tab.path.empty())
                          path = SaveFileDialog(nui::Tr("Save Script"), nui::Tr("Python scripts"), "*.py", "script.py");
                      if (path.empty() && tab.path.empty()) return ActionResult::Fail("no file chosen");
                      ui_state_.scripts.Save(tab, path);
                      if (ui_state_.script_rerun) RunScript(tab.Text(), tab.Name(), false);
                      return Ok(tab_json(tab));
                  }});
    actions_.Add({"script.close", "Close Script", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      ScriptTab& tab = TabArg(args);
                      if (tab.Dirty() && !BoolArg(args, "discard", false))
                          return ActionResult::Fail(tab.title + " has unsaved edits; pass discard to drop them");
                      ui_state_.scripts.Close(tab.id);
                      return Ok();
                  }});
    // One of `code`, `path` (a file on disk) or a tab; `selection` runs the tab's selection, or
    // its caret line, and prints the value of a final expression. Answers at once with the seq.
    actions_.Add({"script.run", "Run Script", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      std::string code, name;
                      bool echo = false;
                      if (const json::Value* c = args.Find("code")) {
                          code = c->AsString();
                          name = StrArg(args, "name", "<run>");
                      } else if (const json::Value* p = args.Find("path")) {
                          name = NormalPath(p->AsString());
                          if (!ReadEntireFile(name, &code)) return ActionResult::Fail("cannot read " + name);
                      } else {
                          ScriptTab& tab = TabArg(args);
                          name = tab.Name();
                          code = tab.Text();
                          if (BoolArg(args, "selection", false)) {
                              TextEditor::DocSelection sel = tab.editor->GetMainCursorSelection();
                              if (sel.start == sel.end) {
                                  sel.start.index = 0u;
                                  sel.end = TextEditor::DocPos(sel.start.line + 1u, 0u);
                              }
                              // Leading newlines keep tracebacks on the tab's own line numbers.
                              code = std::string(sel.start.line, '\n') + tab.editor->GetSectionText(sel);
                              echo = true;
                          }
                      }
                      const uint64_t seq = RunScript(code, name, echo);
                      return Ok(Object({{"seq", json::Value::Int(static_cast<int64_t>(seq))}, {"name", json::Value::Str(name)}}));
                  }});
    // A console line: echoed, then run so the value of a final expression prints.
    actions_.Add({"script.eval", "Evaluate in Console", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const std::string code = StrArg(args, "code");
                      ConsoleAppend(">>> " + code + "\n");
                      const uint64_t seq = RunScript(code, std::string(), true);
                      return Ok(Object({{"seq", json::Value::Int(static_cast<int64_t>(seq))}}));
                  }});
    actions_.Add({"script.stop", "Stop Script", "script", "", ShortcutScope::Global,
                  [this] { return script_host_.BusySeconds() > 0.0; }, {},
                  [this](const json::Value&) {
                      return Ok(Object({{"stopped", json::Value::Bool(script_host_.Interrupt("stopped by the user"))}}));
                  }});
    actions_.Add({"script.result", "Script Result", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const uint64_t seq = static_cast<uint64_t>(args.At("seq").AsInt());
                      const auto run = script_runs_.find(seq);
                      if (run == script_runs_.end()) return ActionResult::Fail("no script run " + std::to_string(seq));
                      return Ok(Object({{"done", json::Value::Bool(run->second.done)},
                                        {"failed", json::Value::Bool(run->second.failed)},
                                        {"output", json::Value::Str(run->second.output)},
                                        {"name", json::Value::Str(run->second.name)}}));
                  }});
    // Completions at a 1-based line and 0-based column, as {name, type, signature}.
    actions_.Add({"script.complete", "Complete", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      return Ok(json::Value::Parse(script_host_.Complete(StrArg(args, "code"),
                                                                         static_cast<int>(NumArg(args, "line", 1.0)),
                                                                         static_cast<int>(NumArg(args, "column", 0.0)))));
                  }});
    actions_.Add({"script.tabs", "Script Tabs", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) { return Ok(ui_state_.scripts.List()); }});
    // Shows a traceback's file at its line: an open tab by name, else the file.
    actions_.Add({"script.goto", "Go to Script Line", "script", "", ShortcutScope::Global, {}, {},
                  [this, tab_json](const json::Value& args) {
                      const std::string path = StrArg(args, "path");
                      ScriptTab* tab = ui_state_.scripts.Named(path);
                      if (tab == nullptr) tab = &ui_state_.scripts.Open(path);
                      ui_state_.scripts.Activate(tab->id);
                      ui_state_.scripts.Goto(*tab, static_cast<int>(NumArg(args, "line", 1.0)));
                      actions_.Run("window.focus", Object({{"panel", json::Value::Str("script")}}), "script.goto", frame_index_);
                      return Ok(tab_json(*tab));
                  }});
    // Console text from byte `since` on (0: all that is kept); `next` continues from here.
    actions_.Add({"script.console", "Console Text", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const size_t since = static_cast<size_t>(NumArg(args, "since", 0.0));
                      const std::string& log = ui_state_.console_log;
                      const size_t from = since > console_base_ ? std::min(since - console_base_, log.size()) : 0u;
                      return Ok(Object({{"text", json::Value::Str(log.substr(from))},
                                        {"next", json::Value::Int(static_cast<int64_t>(console_base_ + log.size()))},
                                        {"dropped", json::Value::Bool(since < console_base_)}}));
                  }});
    actions_.Add({"script.clear_console", "Clear Console", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      console_base_ += ui_state_.console_log.size();
                      ui_state_.console_log.clear();
                      return Ok();
                  }});
    // A tab whose file changed under its edits reloads, or keeps them with `keep`.
    actions_.Add({"script.reload", "Reload Script", "script", "", ShortcutScope::Global, {}, {},
                  [this, tab_json](const json::Value& args) {
                      ScriptTab& tab = TabArg(args);
                      if (BoolArg(args, "keep", false)) tab.disk_changed = false;
                      else ui_state_.scripts.Reload(tab);
                      return Ok(tab_json(tab));
                  }});
    actions_.Add({"script.toggle_rerun", "Rerun on Save", "script", "", ShortcutScope::Global, {},
                  [this] { return ui_state_.script_rerun; },
                  [this](const json::Value& args) {
                      ui_state_.script_rerun = OptBoolArg(args, "on").value_or(!ui_state_.script_rerun);
                      profile_->SetSettingBool("script", "rerun_on_save", ui_state_.script_rerun);
                      return Ok(json::Value::Bool(ui_state_.script_rerun));
                  }});
    // Seconds a script may run before it stops; 0 lets it run.
    actions_.Add({"script.set_budget", "Script Time Budget", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      script_host_.SetBudget(NumArg(args, "seconds", script_host_.Budget()));
                      profile_->SetSettingNumber("script", "budget_s", script_host_.Budget());
                      return Ok(json::Value::Float(script_host_.Budget()));
                  }});
    // Attach persists a tab as a /script node: the node it came from, else a selected script
    // node, else a new one.
    actions_.Add({"script.attach", "Attach as Node", "script", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      ScriptTab& tab = TabArg(args);
                      std::string target = tab.node;
                      const scene::EntityId sel = ui_state_.selected_entity;
                      if (target.empty() && sel != scene::kInvalidEntity &&
                          loaded_->scene.Ecs().Get<scene::ScriptComponent>(sel) != nullptr)
                          target = SelectedPath();
                      const json::Value source = json::Value::Str(StrArg(args, "source", tab.Text()));
                      return SubmitTxn(EditTxn{"Attach script", {EditOp{target, "script", json::Value(), source}}});
                  }});
    // Opens a /script node's source in a tab: `path`, else the selection.
    actions_.Add({"script.load", "Open Node Source", "script", "", ShortcutScope::Global, has_scene, {},
                  [this, tab_json](const json::Value& args) {
                      const std::string path = StrArg(args, "path", SelectedPath());
                      const auto node = loaded_->scene.Tree().NodeOf(path);
                      const auto* sc = node ? loaded_->scene.Ecs().Get<scene::ScriptComponent>(node->entity) : nullptr;
                      if (sc == nullptr) return ActionResult::Fail("'" + path + "' is not a /script node");
                      return Ok(tab_json(ui_state_.scripts.AddNode(path, sc->source)));
                  }});
    // The scene's nodes as {path, name, kind}: all, those under `under`, or of one `kind`.
    actions_.Add({"scene.nodes", "List Nodes", "scene", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const std::string under = StrArg(args, "under");
                      const std::string kind = StrArg(args, "kind");
                      const scene::SceneGraph& tree = loaded_->scene.Tree();
                      const auto start = under.empty() ? tree.Root() : tree.NodeOf(under);
                      if (!start) return ActionResult::Fail("no node '" + under + "'");
                      json::Value out = json::Value::Array();
                      tree.Traverse(start, [&](const std::shared_ptr<scene::SceneNode>& n) {
                          if (n == tree.Root()) return;
                          const std::string k = NodeKind(loaded_->scene.Ecs(), n->entity);
                          if (!kind.empty() && k != kind) return;
                          out.PushBack(Object({{"path", json::Value::Str(tree.PathOf(n))},
                                               {"name", json::Value::Str(n->name)},
                                               {"kind", json::Value::Str(k)}}));
                      });
                      return Ok(std::move(out));
                  }});
}

void EditorHost::State::RegisterSceneActions() {
    const auto has_scene = [this] { return loaded_ != nullptr; };
    // Puts the nodes into `list` or takes them out; unset, the first one's state flips.
    const auto flag = [](std::vector<std::string>& list, const std::vector<std::string>& paths, std::optional<bool> in) {
        const bool put = in.value_or(std::find(list.begin(), list.end(), paths.front()) == list.end());
        for (const std::string& p : paths) {
            const auto at = std::find(list.begin(), list.end(), p);
            if (put && at == list.end()) list.push_back(p);
            if (!put && at != list.end()) list.erase(at);
        }
        return put;
    };
    actions_.Add({"scene.set_visible", "Show/Hide", "edit", "H", ShortcutScope::Viewport, has_scene, {},
                  [this, flag](const json::Value& args) {
                      const std::vector<std::string> paths = TargetPaths(args);
                      if (paths.empty()) return ActionResult::Fail("no node selected");
                      const std::optional<bool> visible = OptBoolArg(args, "value");
                      const bool hidden = flag(ui_state_.hidden, paths, visible ? std::optional<bool>(!*visible) : std::nullopt);
                      return Ok(Object({{"paths", StrList(paths)}, {"visible", json::Value::Bool(!hidden)}}));
                  }});
    actions_.Add({"scene.set_locked", "Lock/Unlock", "edit", "", ShortcutScope::Global, has_scene, {},
                  [this, flag](const json::Value& args) {
                      const std::vector<std::string> paths = TargetPaths(args);
                      if (paths.empty()) return ActionResult::Fail("no node selected");
                      const bool locked = flag(ui_state_.locked, paths, OptBoolArg(args, "value"));
                      return Ok(Object({{"paths", StrList(paths)}, {"locked", json::Value::Bool(locked)}}));
                  }});
    // The collider options of `args` and the nodes they apply to, as one edit op's value.
    const auto collision_op = [this](const json::Value& args) {
        json::Value op = CollisionOptionsToJson(CollisionOptionsFromJson(args));
        op.Set("paths", StrList(TargetPaths(args)));
        return op;
    };
    actions_.Add({"scene.collision_preview", "Preview Collision", "scene", "", ShortcutScope::Global, has_scene, {},
                  [this, collision_op](const json::Value& args) {
                      const json::Value op = collision_op(args);
                      ui_state_.collision.preview =
                          PreviewToJson(PreviewCollision(loaded_->scene, StrListArg(op, "paths"), CollisionOptionsFromJson(op)));
                      return Ok(ui_state_.collision.preview);
                  },
                  "", true});
    actions_.Add({"scene.generate_collision", "Generate Collision", "scene", "", ShortcutScope::Global, has_scene, {},
                  [this, collision_op](const json::Value& args) {
                      const json::Value op = collision_op(args);
                      if (StrListArg(op, "paths").empty()) return ActionResult::Fail("no node selected");
                      const ActionResult r = SubmitTxn(EditTxn{"Generate Collision", {EditOp{"", "collision", json::Value(), op}}});
                      if (r.ok) {
                          ui_state_.collision.preview = json::Value();
                          ui_state_.collision.dismiss = true;
                      }
                      return r;
                  },
                  "", true});
    actions_.Add({"scene.collision_dialog", "Generate Collision...", "scene", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value&) {
                      ui_state_.collision.appear = true;
                      return RunAction("scene.collision_preview", CollisionArgs(ui_state_.collision), "ui");
                  }});
}

void EditorHost::State::RegisterAssetActions() {
    actions_.Add({"file.import_analyze", "Analyze Asset", "file", "", ShortcutScope::Global, {}, {},
                  [](const json::Value& args) {
                      return Ok(AnalysisToJson(AnalyzeImport(StrArg(args, "path"), StrArg(args, "profile", "strict"))));
                  },
                  "", true});
    // Without `mode` the wizard opens on the asset; "auto" adds a clean asset to the open
    // document at once, "add" adds it, "open" opens it as a new document.
    actions_.Add({"file.import", "Import Asset...", "file", "Ctrl+I", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      std::string path = StrArg(args, "path");
                      if (path.empty())
                          path = OpenFileDialog(nui::Tr("Import Asset"), nui::Tr("Assets"),
                                                "*.urdf;*.xml;*.usd;*.usda;*.stl;*.obj;*.nks");
                      if (path.empty()) return ActionResult::Fail("no asset chosen");
                      if (!fs::exists(Utf8Path(path))) return ActionResult::Fail("no file at " + path);
                      const std::string mode = StrArg(args, "mode");
                      if (mode != "" && mode != "auto" && mode != "add" && mode != "open")
                          return ActionResult::Fail("the mode is auto, add or open");
                      const std::string profile = StrArg(args, "profile", "strict");
                      const json::Value* pose = args.Find("pose");
                      if (mode.empty() || mode == "auto") {
                          const ImportAnalysis a = AnalyzeImport(path, profile);
                          if (mode.empty() || !loaded_ || !Importable(a) || !a.diagnostics.empty())
                              return ShowImport(a, pose != nullptr ? *pose : json::Value());
                      }
                      if (mode == "open") {
                          if (borrowed_) return Spawn({path});
                          const json::Value resolved = Object({{"path", json::Value::Str(path)}, {"mode", json::Value::Str(mode)},
                                                               {"profile", json::Value::Str(profile)},
                                                               {"discard", json::Value::Bool(BoolArg(args, "discard", false))}});
                          ActionResult asked;
                          if (!Confirmed("file.import", resolved, &asked)) return asked;
                          ui_state_.import.path.clear();
                          if (AssetKindOf(path) == AssetKind::Scene) return OpenDocument(path, path);
                          const ImportAnalysis a = AnalyzeImport(path, profile);
                          if (!Importable(a)) return ActionResult::Fail(ImportError(a));
                          const std::string file = cache_.File("import.nks");
                          scene::nks::Save(*a.scene, file);
                          return OpenDocument(file, std::string());
                      }
                      if (!loaded_) return ActionResult::Fail("no document is open to add to");
                      const std::string name = StrArg(args, "name", PathText(Utf8Path(path).stem()));
                      json::Value after = Object({{"path", json::Value::Str(path)}, {"profile", json::Value::Str(profile)},
                                                  {"name", json::Value::Str(name)},
                                                  {"ground", json::Value::Bool(BoolArg(args, "ground", true))}});
                      if (pose != nullptr && !pose->IsNull()) after.Set("pose", *pose);
                      const ActionResult r = SubmitTxn(EditTxn{"Import " + name, {EditOp{"", "import", json::Value(), after}}});
                      if (r.ok) ui_state_.import.path.clear();
                      return r;
                  }});
    actions_.Add({"assets.add_folder", "Add Asset Folder...", "assets", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      std::string folder = StrArg(args, "path");
                      if (folder.empty()) folder = OpenFolderDialog(nui::Tr("Add Asset Folder"));
                      if (folder.empty()) return ActionResult::Fail("no folder chosen");
                      if (!fs::is_directory(Utf8Path(folder))) return ActionResult::Fail("no folder at " + folder);
                      folder = NormalPath(folder);
                      std::vector<std::string> folders = profile_->SettingList("assets", "folders");
                      if (std::find(folders.begin(), folders.end(), folder) == folders.end()) folders.push_back(folder);
                      profile_->SetSettingList("assets", "folders", folders);
                      ui_state_.assets.root = folder;
                      assets_stale_ = true;
                      return Ok(Object({{"path", json::Value::Str(folder)}}));
                  }});
    actions_.Add({"assets.remove_folder", "Remove Asset Folder", "assets", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const std::string folder = NormalPath(StrArg(args, "path"));
                      std::vector<std::string> folders = profile_->SettingList("assets", "folders");
                      folders.erase(std::remove(folders.begin(), folders.end(), folder), folders.end());
                      profile_->SetSettingList("assets", "folders", folders);
                      if (ui_state_.assets.root == folder) ui_state_.assets.root.clear();
                      assets_stale_ = true;
                      return Ok();
                  },
                  "", true});
    actions_.Add({"assets.rescan", "Rescan Assets", "assets", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      assets_stale_ = true;
                      return Ok();
                  }});
    // The cards as the panel lists them; `pictures` asks for their pictures as the panel does.
    actions_.Add({"assets.list", "List Assets", "assets", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      if (BoolArg(args, "pictures", false) && !cards_pictures_) {
                          cards_pictures_ = true;
                          cards_version_ = ~0ull;
                      }
                      RefreshAssets();
                      json::Value cards = json::Value::Array();
                      for (const AssetCard& c : ui_state_.assets.cards)
                          cards.PushBack(Object({{"path", json::Value::Str(c.path)}, {"name", json::Value::Str(c.name)},
                                                 {"root", json::Value::Str(c.root)}, {"kind", json::Value::Str(c.kind)},
                                                 {"thumbnail", json::Value::Str(c.thumbnail)}}));
                      return Ok(Object({{"roots", StrList(ui_state_.assets.roots)}, {"cards", std::move(cards)}}));
                  },
                  "", true});
}

// `paths`, else `path`, else the selection.
std::vector<std::string> EditorHost::State::TargetPaths(const json::Value& args) const {
    std::vector<std::string> paths = StrListArg(args, "paths");
    if (paths.empty() && !StrArg(args, "path").empty()) paths.push_back(StrArg(args, "path"));
    return paths.empty() ? SelectionPaths() : paths;
}

// Opens the import wizard on `a`; a file new to it starts from the file's own name.
ActionResult EditorHost::State::ShowImport(const ImportAnalysis& a, const json::Value& pose) {
    ImportWizard& w = ui_state_.import;
    json::Value analysis = AnalysisToJson(a);
    if (w.path != a.path) {
        std::snprintf(w.name, sizeof(w.name), "%s", analysis.At("name").AsString().c_str());
        w.open_as_new = !loaded_;
    }
    w.path = a.path;
    w.analysis = analysis;
    w.pose = pose;
    w.add_allowed = loaded_ != nullptr;
    w.appear = true;
    return Ok(std::move(analysis));
}

// Hidden nodes' instances lose their mesh until shown again or the world rebinds.
void EditorHost::State::ApplyHidden() {
    render::RenderWorld& rw = loaded_->sim->GetRenderWorld();
    instance_paths_.clear();
    hovered_path_.clear();
    for (const auto& [index, mesh] : hidden_meshes_) rw.instances[index].mesh_id = mesh;
    hidden_meshes_.clear();
    if (!ui_state_.hidden.empty()) {
        for (uint32_t i = 0; i < rw.InstanceCount(); ++i) {
            render::RenderInstance& inst = rw.instances[i];
            if (inst.mesh_id == render::kNoId || !UnderAny(PathOfEntity(inst.entity), ui_state_.hidden)) continue;
            hidden_meshes_.emplace_back(i, inst.mesh_id);
            inst.mesh_id = render::kNoId;
        }
    }
    applied_hidden_ = ui_state_.hidden;
    hidden_stale_ = false;
}

// The Assets panel's folders: the bundled examples, the document's own, then the user's.
void EditorHost::State::RefreshAssets() {
    AssetBrowser& b = ui_state_.assets;
    if (assets_stale_) {
        assets_stale_ = false;
        b.roots.clear();
        const std::string examples = Resource("examples");
        if (fs::is_directory(Utf8Path(examples))) b.roots.push_back(NormalPath(examples));
        if (!doc_path_.empty()) b.roots.push_back(PathText(Utf8Path(doc_path_).parent_path()));
        b.user_roots = profile_->SettingList("assets", "folders");
        for (const std::string& r : b.user_roots)
            if (std::find(b.roots.begin(), b.roots.end(), r) == b.roots.end()) b.roots.push_back(r);
        asset_files_ = ScanAssets(b.roots);
        cards_version_ = ~0ull;
    }
    // Pictures are drawn on demand, so the cards ask for them once the panel has shown.
    const bool pictures = cards_pictures_ || b.shown;
    const uint64_t version = thumbnails_->Version();
    if (version == cards_version_ && pictures == cards_pictures_) return;
    cards_version_ = version;
    cards_pictures_ = pictures;
    b.cards.clear();
    for (const AssetFile& f : asset_files_)
        b.cards.push_back(AssetCard{f.path, f.name, f.root, AssetKindName(f.kind),
                                    pictures ? thumbnails_->Thumbnail(f) : std::string()});
    ++b.version;
}

// The Properties schema follows the selection and every document change.
void EditorHost::State::RefreshProperties() {
    if (!loaded_) {
        if (!ui_state_.properties.empty()) ++ui_state_.properties_version;
        ui_state_.properties.clear();
        props_scene_version_ = ~0ull;
        return;
    }
    std::vector<std::string> paths = SelectionPaths();
    if (paths == props_paths_ && ui_state_.scene_version == props_scene_version_ && doc_.Revision() == props_revision_)
        return;
    ui_state_.properties = PropertySchema(loaded_->scene, doc_, paths);
    ++ui_state_.properties_version;
    props_paths_ = std::move(paths);
    props_scene_version_ = ui_state_.scene_version;
    props_revision_ = doc_.Revision();
}

// A click picks the body the id pass shows under the pointer, seeing through locked nodes;
// Shift adds or drops it. One on another environment's copy makes that environment current.
void EditorHost::State::PickSelect(const CameraController& camera, float x, float y) {
    const render::PickResult pick = PickBox(camera, x, y, x, y);
    const bool shift = ImGui::GetIO().KeyShift;
    if (pick.center == ~0u) {
        if (!shift) Queue("scene.select", json::Value::Object(), "viewport");
        return;
    }
    SwitchEnv(picked_env_);
    Queue("scene.select",
          Object({{"path", json::Value::Str(CarrierPath(pick.center))}, {"mode", json::Value::Str(shift ? "toggle" : "replace")}}),
          "viewport");
}

bool EditorHost::State::BodyLoad(const render::RenderInstance& inst, const math::Vec3& at, float gain,
                                 Perturbation* out) const {
    math::Transform pose;
    if (shown_ == nullptr || !ResolvePose(inst.pose_source, shown_->link_poses, shown_->body_poses, &pose)) return false;
    const std::vector<nuka_link_binding_t>& links = loaded_->Layout().links;
    const bool link = inst.pose_source.kind == render::PoseSource::Kind::Link;
    const uint32_t body = link ? links[inst.pose_source.row].body_index : inst.pose_source.row;
    // A link drags its whole articulation, so its spring is sized for the articulation's mass.
    float mass = loaded_->scene.GetBody(body).mass;
    if (link) {
        mass = 0.0f;
        for (const nuka_link_binding_t& l : links)
            if (l.articulation_index == links[inst.pose_source.row].articulation_index)
                mass += loaded_->scene.GetBody(l.body_index).mass;
    }
    const float w = 2.0f * kPi * kGrabHz;
    Perturbation p;
    p.source = inst.pose_source;
    p.point = pose.Inverse().TransformPoint(at);
    p.com = loaded_->scene.GetBody(body).inertial_transform.position;
    p.target = at;
    p.stiffness = gain * mass * w * w;
    p.damping = 2.0f * mass * w * std::sqrt(gain);
    *out = p;
    return true;
}

void EditorHost::State::PushPerturb(const Perturbation& p) {
    TickCommand cmd;
    cmd.kind = CommandKind::Perturb;
    cmd.perturb = p;
    worker_.Push(std::move(cmd));
}

// The Perturb tool's press: the nearest moving body or particle under the pointer starts an
// unstretched pull.
void EditorHost::State::GrabAt(const CameraController& camera, float x, float y) {
    if (shown_ == nullptr) return;
    const SceneViewportRect& rect = ui_.Viewport();
    const Ray ray = camera.ScreenRay(rect.LocalX(x), rect.LocalY(y), rect.pixels.width, rect.pixels.height);
    const render::RenderWorld& w = loaded_->sim->GetRenderWorld();
    float body_t = 0.0f, particle_t = 0.0f;
    const uint32_t hit = PickInstance(
        ray, w,
        [&](const render::RenderInstance& inst) {
            return (inst.pose_source.kind == render::PoseSource::Kind::Link ||
                    inst.pose_source.kind == render::PoseSource::Kind::Body) &&
                   (ui_state_.locked.empty() || !UnderAny(PathOfEntity(inst.entity), ui_state_.locked));
        },
        &body_t);
    const uint32_t particle = PickParticle(ray, loaded_->Layout(), shown_->particles, &particle_t);
    Perturbation p;
    if (particle != kNoParticle && particle_t < body_t) {
        p.particle = particle;
        p.target = shown_->particles[particle];
    } else if (hit == ~0u || !BodyLoad(w.instances[hit], ray.origin + ray.dir * body_t, ui_state_.perturb_gain, &p)) {
        return;
    }
    grab_ = p;
    grab_anchor_ = p.target;
    PushPerturb(p);
}

void EditorHost::State::DragGrab(const CameraController& camera, float x, float y) {
    const SceneViewportRect& rect = ui_.Viewport();
    const Ray ray = camera.ScreenRay(rect.LocalX(x), rect.LocalY(y), rect.pixels.width, rect.pixels.height);
    const math::Vec3 fwd = (camera.ResolvedTarget() - camera.ResolvedEye()).Normalized();
    if (camera.RayPlaneHit(ray, grab_anchor_, fwd, &grab_->target)) PushPerturb(*grab_);
}

void EditorHost::State::ReleaseGrab() {
    grab_->hold = false;
    PushPerturb(*grab_);
    grab_.reset();
}

// The Pin tool's click: a cloth vertex toggles its pin and a free body turns static where it
// stands, or moves again; each is one undoable edit that rebuilds the world.
void EditorHost::State::PinAt(const CameraController& camera, float x, float y) {
    if (shown_ == nullptr) return;
    const SceneViewportRect& rect = ui_.Viewport();
    const Ray ray = camera.ScreenRay(rect.LocalX(x), rect.LocalY(y), rect.pixels.width, rect.pixels.height);
    const render::RenderWorld& w = loaded_->sim->GetRenderWorld();
    float body_t = 0.0f, particle_t = 0.0f;
    const uint32_t hit = PickInstance(
        ray, w,
        [&](const render::RenderInstance& inst) {
            return doc_.Records().IsBody(CarrierOf(loaded_->scene, inst.entity)) &&
                   (ui_state_.locked.empty() || !UnderAny(PathOfEntity(inst.entity), ui_state_.locked));
        },
        &body_t);
    const uint32_t particle = PickParticle(ray, loaded_->Layout(), shown_->particles, &particle_t);
    if (particle != kNoParticle && particle_t < body_t) {
        for (const ParticleSurfaceBinding& s : loaded_->Layout().surfaces) {
            if (particle < s.info.particle_first || particle - s.info.particle_first >= s.info.particle_count) continue;
            const uint32_t vertex = particle - s.info.particle_first;
            const std::vector<uint32_t>& pins = loaded_->scene.GetMedia(s.info.medium).cloth_mesh.pinned_vertices;
            const bool on = std::find(pins.begin(), pins.end(), vertex) == pins.end();
            const json::Value after = Object({{"vertex", json::Value::Int(vertex)}, {"on", json::Value::Bool(on)}});
            Queue("edit.apply",
                  TxnToJson(EditTxn{on ? "Pin" : "Unpin",
                                    {EditOp{PathOfEntity(loaded_->scene.EntityOfMedia(s.info.medium)), "pin",
                                            json::Value::Null(), after}}}),
                  "viewport");
        }
        return;
    }
    if (hit == ~0u) return;
    const scene::EntityId e = CarrierOf(loaded_->scene, w.instances[hit].entity);
    const std::string path = PathOfEntity(e);
    const bool pinned = loaded_->scene.GetBody(doc_.Records().body.at(e)).is_static;
    EditTxn txn{pinned ? "Unpin" : "Pin", {}};
    if (const render::RenderInstance* inst = PoseInstance(*loaded_, doc_.Records(), e); inst != nullptr && !pinned)
        txn.ops.push_back(EditOp{path, "transform", json::Value::Null(),
                                 TransformToJson(LiveWorldTransform(loaded_->scene, e, *inst))});
    txn.ops.push_back(EditOp{path, "static", json::Value::Null(), json::Value::Bool(!pinned)});
    Queue("edit.apply", TxnToJson(txn), "viewport");
}

void EditorHost::State::MeasureAt(const CameraController& camera, float x, float y) {
    RobotView& v = ui_state_.robot;
    Measure& m = v.measure;
    const SceneViewportRect& rect = ui_.Viewport();
    const float lx = rect.LocalX(x), ly = rect.LocalY(y);
    const Ray ray = camera.ScreenRay(lx, ly, rect.pixels.width, rect.pixels.height);
    // A joint origin takes the click within a few pixels of it, as the ray's slope off the pointer.
    float best = (camera.ScreenRay(lx + kJointPickPixels, ly, rect.pixels.width, rect.pixels.height).dir - ray.dir).Length();
    MeasureEnd end;
    const size_t joints = v.overlays.on && v.overlays.axes ? std::min(v.links.size(), v.link_poses.size()) : 0u;
    for (size_t l = 0; l < joints; ++l) {
        if (v.links[l].parent < 0) continue;
        const math::Vec3 p = (v.link_poses[l] * v.links[l].frame).position - ray.origin;
        const float t = p.Dot(ray.dir);
        if (t <= 0.0f || (p - ray.dir * t).Length() / t >= best) continue;
        best = (p - ray.dir * t).Length() / t;
        end = MeasureEnd{MeasureEnd::Kind::Joint, static_cast<int>(l)};
    }
    const render::RenderWorld& w = loaded_->sim->GetRenderWorld();
    if (end.kind == MeasureEnd::Kind::None) {
        const uint32_t hit = PickInstance(ray, w, [&](const render::RenderInstance& inst) {
            return inst.pose_source.kind == render::PoseSource::Kind::Link && inst.pose_source.row < v.links.size();
        });
        if (hit != ~0u) end = MeasureEnd{MeasureEnd::Kind::Link, static_cast<int>(w.instances[hit].pose_source.row)};
    }
    if (m.a.kind == MeasureEnd::Kind::None || m.b.kind != MeasureEnd::Kind::None) {
        m.a = end;
        m.b = MeasureEnd{};
        return;
    }
    m.b = end.kind == MeasureEnd::Kind::None ? MeasureEnd{MeasureEnd::Kind::Ground, -1} : end;
}

json::Value EditorHost::State::MeasureJson() const {
    const RobotView& v = ui_state_.robot;
    math::Vec3 from, to;
    if (!MeasureEnds(v, &from, &to)) return Object({{"distance", json::Value::Null()}});
    const auto name = [&](const MeasureEnd& e) {
        const RobotLink& link = v.links[static_cast<size_t>(std::max(e.link, 0))];
        return e.kind == MeasureEnd::Kind::Ground ? std::string("ground")
               : e.kind == MeasureEnd::Kind::Joint ? link.joint
                                                   : link.path;
    };
    return Object({{"distance", json::Value::Float((to - from).Length())},
                   {"delta", Vec3Json(to - from)},
                   {"from", Vec3Json(from)},
                   {"to", Vec3Json(to)},
                   {"a", json::Value::Str(name(v.measure.a))},
                   {"b", json::Value::Str(name(v.measure.b))},
                   {"at", json::Value::Str(v.measure.com ? "com" : "origin")}});
}

void EditorHost::State::RegisterSimActions() {
    const auto has_scene = [this] { return loaded_ != nullptr; };
    const auto set_playing = [this](bool playing) {
        ui_state_.playing = playing;
        PushTransport();
        return Ok(Object({{"playing", json::Value::Bool(playing)}}));
    };
    actions_.Add({"sim.play", "Play", "sim", "", ShortcutScope::Global, has_scene, {},
                  [set_playing](const json::Value&) { return set_playing(true); }});
    actions_.Add({"sim.pause", "Pause", "sim", "", ShortcutScope::Global, has_scene, {},
                  [set_playing](const json::Value&) { return set_playing(false); }});
    actions_.Add({"sim.toggle_play", "Play/Pause", "sim", "", ShortcutScope::Global, has_scene,
                  [this] { return ui_state_.playing; },
                  [this, set_playing](const json::Value&) { return set_playing(!ui_state_.playing); }});
    actions_.Add({"sim.step", "Step", "sim", "", ShortcutScope::Global,
                  [this] { return loaded_ != nullptr && !ui_state_.playing; }, {},
                  [this](const json::Value& args) {
                      TickCommand cmd = TickCommand::Transport(TransportOp::Step);
                      cmd.steps = static_cast<uint32_t>(std::max(NumArg(args, "steps", 1.0), 1.0));
                      PushTransport();
                      worker_.Push(std::move(cmd));
                      return Ok();
                  }});
    actions_.Add({"sim.reset", "Reset", "sim", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      if (!loaded_) {
                          FrameWorld(empty_world_);
                          return Ok();
                      }
                      reset_seq_ = transport_seq_ = worker_.Push(TickCommand::Transport(TransportOp::Reset));
                      reset_framing_ = true;
                      // The new history logs the command from its first tick.
                      teleop_command_sent_.reset();
                      ui_state_.playing = sent_playing_ = false;
                      return Ok();
                  }});
    actions_.Add({"sim.speed", "Speed", "sim", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      ui_state_.speed = static_cast<float>(NumArg(args, "value", 1.0));
                      PushTransport();
                      return Ok(Object({{"speed", json::Value::Float(ui_state_.speed)}}));
                  }});
    // {count} rebuilds the world with that many environments, all from the scene's initial state at
    // tick 0; loads and edits keep the count. Without it, the grid as view.envs reports it.
    actions_.Add({"sim.envs", "Environments", "sim", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const double count = NumArg(args, "count", ui_state_.envs.count);
                      if (count < 1.0 || count > kMaxEnvs)
                          return ActionResult::Fail("count: 1 to " + std::to_string(kMaxEnvs) + " environments");
                      const auto n = static_cast<uint32_t>(count);
                      if (n == ui_state_.envs.count) return Ok(EnvsJson());
                      bool ok = false;
                      worker_.Exclusive([&] {
                          entity_drag_.Cancel();
                          present_->WaitIdle();
                          ok = ResizeEditorEnvs(*loaded_, cache_, n);
                          if (!ok) return;
                          ui_state_.env_index %= n;
                          OnRebound(0u, 0.0);
                          doc_.AddBranch(TimelineBranch{0u, Generation(), std::to_string(n) + " environments"});
                      });
                      if (!ok) return ActionResult::Fail("the world refused " + std::to_string(n) + " environments");
                      return Ok(EnvsJson());
                  }});
    // A viewport tool toggles against Select; `on` sets it.
    const auto set_tool = [this](ViewportTool tool, const json::Value& args) {
        ui_state_.tool = BoolArg(args, "on", ui_state_.tool != tool) ? tool : ViewportTool::Select;
        if (grab_) ReleaseGrab();
        const char* names[] = {"select", "perturb", "pin", "measure"};
        return Ok(Object({{"tool", json::Value::Str(names[static_cast<int>(ui_state_.tool)])}}));
    };
    actions_.Add({"tool.perturb", "Perturb", "sim", "", ShortcutScope::Global, {},
                  [this] { return ui_state_.tool == ViewportTool::Perturb; },
                  [set_tool](const json::Value& args) { return set_tool(ViewportTool::Perturb, args); }});
    actions_.Add({"tool.pin", "Pin", "sim", "", ShortcutScope::Global, {},
                  [this] { return ui_state_.tool == ViewportTool::Pin; },
                  [set_tool](const json::Value& args) { return set_tool(ViewportTool::Pin, args); }});
    // `a` and `b` set the Measure tool's ends (a link's node path, a joint's name, or "ground" for b)
    // and return the measurement; `at` takes link ends at their origin or com. Else it is the tool.
    actions_.Add({"tool.measure", "Measure", "sim", "", ShortcutScope::Global, {},
                  [this] { return ui_state_.tool == ViewportTool::Measure; },
                  [this, set_tool](const json::Value& args) {
                      Measure& m = ui_state_.robot.measure;
                      const std::string at = StrArg(args, "at", m.com ? "com" : "origin");
                      if (at != "origin" && at != "com") return ActionResult::Fail("at is origin or com");
                      m.com = at == "com";
                      if (args.Find("a") == nullptr) return set_tool(ViewportTool::Measure, args);
                      const std::vector<RobotLink>& links = ui_state_.robot.links;
                      const auto end = [&](const std::string& name, MeasureEnd* out) {
                          *out = name == "ground" ? MeasureEnd{MeasureEnd::Kind::Ground, -1} : MeasureEnd{};
                          for (size_t l = 0; l < links.size(); ++l) {
                              if (links[l].path == name) *out = MeasureEnd{MeasureEnd::Kind::Link, static_cast<int>(l)};
                              if (!links[l].joint.empty() && links[l].joint == name)
                                  *out = MeasureEnd{MeasureEnd::Kind::Joint, static_cast<int>(l)};
                          }
                          return out->kind != MeasureEnd::Kind::None;
                      };
                      MeasureEnd a, b;
                      const std::string an = StrArg(args, "a"), bn = StrArg(args, "b", "ground");
                      if (an == "ground" || !end(an, &a)) return ActionResult::Fail("no link or joint '" + an + "'");
                      if (!end(bn, &b)) return ActionResult::Fail("no link or joint '" + bn + "'");
                      m.a = a;
                      m.b = b;
                      return Ok(MeasureJson());
                  }});
    actions_.Add({"tool.measure_clear", "Clear Measurement", "sim", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      ui_state_.robot.measure.a = ui_state_.robot.measure.b = MeasureEnd{};
                      return Ok();
                  }});
    // A world force and torque at a world point (the COM by default), plus a spring toward `target`;
    // no load and no target releases the body.
    actions_.Add({"sim.perturb", "Perturb Body", "sim", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const std::string path = StrArg(args, "path");
                      const scene::EntityId e = EntityOfPath(*loaded_, path);
                      const render::RenderInstance* inst =
                          e != scene::kInvalidEntity ? PoseInstance(*loaded_, doc_.Records(), e) : nullptr;
                      const float gain = static_cast<float>(NumArg(args, "gain", 1.0));
                      if (!(gain > 0.0f)) return ActionResult::Fail("gain must be positive");
                      math::Vec3 point = math::Vec3::Zero();
                      const bool at_point = Vec3Arg(args, "point", &point);
                      Perturbation p;
                      if (inst == nullptr || !BodyLoad(*inst, point, gain, &p))
                          return ActionResult::Fail("'" + path + "' is not a moving link or body");
                      if (!at_point) p.point = p.com;
                      if (!Vec3Arg(args, "target", &p.target)) p.stiffness = p.damping = 0.0f;
                      math::Vec3 force = math::Vec3::Zero(), torque = math::Vec3::Zero();
                      Vec3Arg(args, "force", &force);
                      Vec3Arg(args, "torque", &torque);
                      const float f[3] = {force.x, force.y, force.z}, t[3] = {torque.x, torque.y, torque.z};
                      std::copy(f, f + 3, p.force);
                      std::copy(t, t + 3, p.torque);
                      p.hold = p.stiffness > 0.0f || force.LengthSq() > 0.0f || torque.LengthSq() > 0.0f;
                      PushPerturb(p);
                      return Ok(Object({{"held", json::Value::Bool(p.hold)},
                                        {"stiffness", json::Value::Float(p.stiffness)},
                                        {"damping", json::Value::Float(p.damping)}}));
                  }});
    // Holds one environment particle kinematic at `target`, where it is by default.
    actions_.Add({"sim.grab_particle", "Grab Particle", "sim", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const double index = NumArg(args, "particle", -1.0);
                      if (shown_ == nullptr || index < 0.0 || index >= static_cast<double>(shown_->particles.size()))
                          return ActionResult::Fail("no particle " + Format("%g", index));
                      Perturbation p;
                      p.particle = static_cast<uint32_t>(index);
                      p.target = shown_->particles[p.particle];
                      Vec3Arg(args, "target", &p.target);
                      PushPerturb(p);
                      return Ok(Object({{"particle", json::Value::Int(p.particle)}, {"target", Vec3Json(p.target)}}));
                  }});
    // Lets go of the load on `path`, the held `particle`, or with neither every load.
    actions_.Add({"sim.release", "Release", "sim", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      Perturbation p;
                      p.hold = false;
                      p.particle = static_cast<uint32_t>(NumArg(args, "particle", kNoParticle));
                      if (const std::string path = StrArg(args, "path"); !path.empty()) {
                          const scene::EntityId e = EntityOfPath(*loaded_, path);
                          const render::RenderInstance* inst =
                              e != scene::kInvalidEntity ? PoseInstance(*loaded_, doc_.Records(), e) : nullptr;
                          if (inst == nullptr || inst->pose_source.kind == render::PoseSource::Kind::Static)
                              return ActionResult::Fail("'" + path + "' is not a moving link or body");
                          p.source = inst->pose_source;
                      }
                      grab_.reset();
                      PushPerturb(p);
                      return Ok();
                  }});
    // The loads held on the shown environment as its last frame has them.
    actions_.Add({"sim.loads", "Held Loads", "sim", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value&) {
                      json::Value out = json::Value::Array();
                      if (shown_ == nullptr) return Ok(std::move(out));
                      for (const Perturbation& p : shown_->perturbs) {
                          math::Vec3 at = p.target;
                          LoadPoint(*shown_, p, &at);
                          json::Value v = Object({{"at", Vec3Json(at)}, {"target", Vec3Json(p.target)}});
                          if (p.particle != kNoParticle) {
                              v.Set("particle", json::Value::Int(p.particle));
                          } else {
                              const bool link = p.source.kind == render::PoseSource::Kind::Link;
                              const uint32_t body = link ? loaded_->Layout().links[p.source.row].body_index : p.source.row;
                              v.Set("path", json::Value::Str(PathOfEntity(loaded_->scene.EntityOfBody(body))));
                              v.Set("force", Vec3Json(math::Vec3{p.force[0], p.force[1], p.force[2]}));
                              v.Set("torque", Vec3Json(math::Vec3{p.torque[0], p.torque[1], p.torque[2]}));
                              v.Set("stiffness", json::Value::Float(p.stiffness));
                              v.Set("damping", json::Value::Float(p.damping));
                          }
                          out.PushBack(std::move(v));
                      }
                      return Ok(std::move(out));
                  }});
}

void EditorHost::State::RegisterViewActions() {
    const auto has_scene = [this] { return loaded_ != nullptr; };
    actions_.Add({"view.frame_all", "Frame All", "view", "Home", ShortcutScope::Viewport, {}, {},
                  [this](const json::Value&) {
                      return camera_.FrameAll(ShownWorld(), ui_.Viewport().Aspect()) ? Ok()
                                                                                     : ActionResult::Fail("nothing to frame");
                  }});
    actions_.Add({"view.frame_selected", "Frame Selected", "view", "F", ShortcutScope::Viewport, {}, {},
                  [this](const json::Value&) {
                      // A selected node frames with everything under it, as a robot's root holds no geometry.
                      const std::vector<std::string> roots = SelectionPaths();
                      std::vector<scene::EntityId> framed;
                      for (const render::RenderInstance& inst : ShownWorld().instances)
                          if (UnderAny(PathOfEntity(inst.entity), roots)) framed.push_back(inst.entity);
                      return camera_.FrameEntities(ShownWorld(), framed, ui_.Viewport().Aspect())
                                 ? Ok()
                                 : ActionResult::Fail("the selection has no geometry to frame");
                  }});
    actions_.Add({"view.reset_camera", "Reset View", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      FrameWorld(ShownWorld());
                      frame_pending_ = true;
                      return Ok();
                  }});
    actions_.Add({"view.colliders", "Collider Proxies", "view", "", ShortcutScope::Global, {},
                  [this] { return ui_state_.show_colliders; },
                  [this](const json::Value& args) {
                      ui_state_.show_colliders = BoolArg(args, "on", !ui_state_.show_colliders);
                      return Ok();
                  }});
    actions_.Add({"view.contacts", "Contact Points", "view", "", ShortcutScope::Global, {},
                  [this] { return ui_state_.show_contacts; },
                  [this](const json::Value& args) {
                      ui_state_.show_contacts = BoolArg(args, "on", !ui_state_.show_contacts);
                      return Ok();
                  }});
    actions_.Add({"view.hide_overlays", "Hide Overlays", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      ui_state_.show_colliders = ui_state_.show_contacts = false;
                      return Ok();
                  }});
    // The next drawn frame is written as a PPM.
    actions_.Add({"view.screenshot", "Screenshot", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const std::string path = StrArg(args, "path", "nuka_screenshot.ppm");
                      present_->SetCaptureFrame(static_cast<int>(present_->SubmittedFrames()), path);
                      return Ok(Object({{"path", json::Value::Str(path)}}));
                  }});
    actions_.Add({"view.stats", "Frame Stats", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      return Ok(Object({{"frame", json::Value::Int(static_cast<int64_t>(frame_index_))},
                                        {"measured", json::Value::Int(static_cast<int64_t>(measured_))},
                                        {"fps", json::Value::Float(fps_ema_)},
                                        {"frame_ms_p50", json::Value::Float(frame_ms_.Percentile(0.50))},
                                        {"frame_ms_p95", json::Value::Float(frame_ms_.Percentile(0.95))},
                                        {"ui_ms_p50", json::Value::Float(ui_ms_.Percentile(0.50))},
                                        {"ui_ms_p95", json::Value::Float(ui_ms_.Percentile(0.95))},
                                        {"age_ms_p95", json::Value::Float(age_ms_.Percentile(0.95))},
                                        {"tick", json::Value::Int(static_cast<int64_t>(worker_.Tick()))},
                                        {"snapshot_drops", json::Value::Int(static_cast<int64_t>(snapshot_drops_))},
                                        {"publish", json::Value::Str(scatter_ ? "zero-copy" : "host")},
                                        {"playing", json::Value::Bool(ui_state_.playing)},
                                        {"view_mode", json::Value::Str(kViewModes[static_cast<size_t>(ui_state_.view_mode)].name)},
                                        {"viewport", Object({{"x", json::Value::Int(ui_.Viewport().pixels.x)},
                                                             {"y", json::Value::Int(ui_.Viewport().pixels.y)},
                                                             {"width", json::Value::Int(ui_.Viewport().pixels.width)},
                                                             {"height", json::Value::Int(ui_.Viewport().pixels.height)}})}}));
                  }});
    // The shading the viewport draws with; without a mode it reports the current one.
    actions_.Add({"view.mode", "View Mode", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const std::string name =
                          StrArg(args, "mode", kViewModes[static_cast<size_t>(ui_state_.view_mode)].name);
                      const auto it = std::find_if(std::begin(kViewModes), std::end(kViewModes),
                                                   [&](const ViewModeInfo& m) { return name == m.name; });
                      if (it == std::end(kViewModes))
                          return ActionResult::Fail("the mode is lit, unlit, normals, material, residual, contact or wireframe");
                      SetViewMode(it->mode);
                      return Ok(Object({{"mode", json::Value::Str(it->name)}}));
                  }});
    // The panes: one, two side by side or four; without a layout it reports the current one.
    actions_.Add({"view.layout", "View Layout", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const std::string name = StrArg(args, "layout", kViewLayouts[static_cast<size_t>(ui_state_.view_layout)].first);
                      const auto it = std::find_if(std::begin(kViewLayouts), std::end(kViewLayouts),
                                                   [&](const auto& l) { return name == l.first; });
                      if (it == std::end(kViewLayouts)) return ActionResult::Fail("the layout is single, split or quad");
                      const auto layout = static_cast<ViewLayout>(it - std::begin(kViewLayouts));
                      if (layout != ui_state_.view_layout) restore_layout_.reset();
                      SetViewLayout(layout);
                      return Ok(Object({{"layout", json::Value::Str(name)}, {"active", json::Value::Int(ui_state_.active_view)}}));
                  }});
    for (const ViewLayout layout : {ViewLayout::Single, ViewLayout::Split, ViewLayout::Quad}) {
        const auto& [name, label] = kViewLayouts[static_cast<size_t>(layout)];
        const std::string id = name;
        actions_.Add({"view.layout_" + id, label, "view", "", ShortcutScope::Global, {},
                      [this, layout] { return ui_state_.view_layout == layout; },
                      [this, id](const json::Value&) {
                          return RunAction("view.layout", Object({{"layout", json::Value::Str(id)}}), "view");
                      }});
    }
    // Shows one pane alone, the active one without an index, and goes back to the layout it left.
    actions_.Add({"view.maximize", "Maximize View", "view", "Shift+Space", ShortcutScope::Viewport,
                  [this] { return ui_state_.view_layout != ViewLayout::Single || restore_layout_.has_value(); },
                  [this] { return restore_layout_.has_value(); },
                  [this](const json::Value& args) {
                      if (ui_state_.view_layout == ViewLayout::Single) {
                          if (!restore_layout_) return ActionResult::Fail("no layout to go back to");
                          const ViewLayout back = *restore_layout_;
                          restore_layout_.reset();
                          SetViewLayout(back);
                      } else {
                          const double index = NumArg(args, "index", ui_state_.active_view);
                          const std::vector<uint32_t> shown = ShownViews();
                          if (std::find(shown.begin(), shown.end(), index) == shown.end())
                              return ActionResult::Fail("index: a pane the layout shows");
                          restore_layout_ = ui_state_.view_layout;
                          ActivateView(static_cast<uint32_t>(index));
                          SetViewLayout(ViewLayout::Single);
                      }
                      return Ok(Object({{"layout", json::Value::Str(kViewLayouts[static_cast<size_t>(ui_state_.view_layout)].first)},
                                        {"active", json::Value::Int(ui_state_.active_view)}}));
                  }});
    // A pane's mode and scene camera ("" is free), each kept when left out; activate makes it the active pane.
    actions_.Add({"view.view", "View Pane", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const double at = NumArg(args, "index", ui_state_.active_view);
                      if (!(at >= 0.0 && at < kViewPanes) || at != std::floor(at)) return ActionResult::Fail("index: a pane from 0 to 3");
                      const auto index = static_cast<uint32_t>(at);
                      const std::string mode = StrArg(args, "mode", kViewModes[static_cast<size_t>(PaneMode(index))].name);
                      const auto it = std::find_if(std::begin(kViewModes), std::end(kViewModes),
                                                   [&](const ViewModeInfo& m) { return mode == m.name; });
                      if (it == std::end(kViewModes))
                          return ActionResult::Fail("the mode is lit, unlit, normals, material, residual, contact or wireframe");
                      const std::string camera = StrArg(args, "camera", views_[index].camera_path);
                      const std::vector<std::string>& cameras = ui_state_.scene_cameras;
                      if (!camera.empty() && std::find(cameras.begin(), cameras.end(), camera) == cameras.end())
                          return ActionResult::Fail("no scene camera '" + camera + "'");
                      const std::vector<uint32_t> shown = ShownViews();
                      const bool activate = BoolArg(args, "activate", false);
                      if (activate && std::find(shown.begin(), shown.end(), index) == shown.end())
                          return ActionResult::Fail("the layout hides pane " + std::to_string(index));
                      if (index == ui_state_.active_view) ui_state_.view_mode = it->mode;
                      else views_[index].mode = it->mode;
                      views_[index].camera_path = camera;
                      if (activate) ActivateView(index);
                      SyncModeOverlays();
                      return Ok(ViewJson(index));
                  }});
    actions_.Add({"view.views", "View Panes", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      json::Value panes = json::Value::Array();
                      for (uint32_t i = 0; i < kViewPanes; ++i) panes.PushBack(ViewJson(i));
                      return Ok(Object({{"layout", json::Value::Str(kViewLayouts[static_cast<size_t>(ui_state_.view_layout)].first)},
                                        {"active", json::Value::Int(ui_state_.active_view)},
                                        {"maximized", json::Value::Bool(restore_layout_.has_value())},
                                        {"cameras", StrList(ui_state_.scene_cameras)},
                                        {"panes", std::move(panes)}}));
                  }});
    // What the viewport shows at x, y: fractions of its width and height from the top left; `select`
    // clicks there, making the environment it hits current and selecting the body.
    actions_.Add({"view.pick", "Pick", "view", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const SceneViewportRect& rect = ui_.Viewport();
                      if (!rect.Valid()) return ActionResult::Fail("the viewport is hidden");
                      const float x = rect.x + static_cast<float>(NumArg(args, "x", 0.5)) * rect.width;
                      const float y = rect.y + static_cast<float>(NumArg(args, "y", 0.5)) * rect.height;
                      const uint32_t hit = PickBox(camera_, x, y, x, y).center;
                      if (hit == ~0u)
                          return Ok(Object({{"path", json::Value::Null()}, {"node", json::Value::Null()},
                                            {"instance", json::Value::Int(-1)}, {"env", json::Value::Null()}}));
                      const uint32_t env = picked_env_;
                      if (BoolArg(args, "select", false)) {
                          SwitchEnv(env);
                          const ActionResult selected =
                              RunAction("scene.select", Object({{"path", json::Value::Str(CarrierPath(hit))}}), "view");
                          if (!selected.ok) return selected;
                      }
                      return Ok(Object({{"path", json::Value::Str(CarrierPath(hit))},
                                        {"node", json::Value::Str(PathOfEntity(ShownWorld().instances[hit].entity))},
                                        {"instance", json::Value::Int(hit)},
                                        {"env", json::Value::Int(env)}}));
                  }});
    // The environment grid: `show` cells of the current environment's page (up to 16), `current`
    // makes one current, `spacing` sets the metres between cells (0 sizes them from the robot).
    actions_.Add({"view.envs", "Environment Grid", "view", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      EnvView& v = ui_state_.envs;
                      const double show = NumArg(args, "show", v.show);
                      if (show < 1.0 || show > kMaxGridEnvs)
                          return ActionResult::Fail("show: 1 to " + std::to_string(kMaxGridEnvs) + " environments");
                      const double current = NumArg(args, "current", ui_state_.env_index % v.count);
                      if (current < 0.0 || current >= v.count)
                          return ActionResult::Fail("current: 0 to " + std::to_string(v.count - 1u));
                      const double spacing = NumArg(args, "spacing", v.spacing);
                      if (spacing < 0.0) return ActionResult::Fail("spacing: 0 or more metres");
                      v.show = std::min(static_cast<uint32_t>(show), v.count);
                      v.spacing = static_cast<float>(spacing);
                      SwitchEnv(static_cast<uint32_t>(current));
                      PushTransport();
                      return Ok(EnvsJson());
                  }});
    // The active pane's orbit camera: `target`, `distance` and `yaw`, `pitch` in degrees, each kept when left out.
    actions_.Add({"view.camera", "Camera", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      math::Vec3 target = camera_.ResolvedTarget();
                      Vec3Arg(args, "target", &target);
                      const double distance = NumArg(args, "distance", camera_.Distance());
                      if (distance <= 0.0) return ActionResult::Fail("distance: more than 0 metres");
                      const double yaw = NumArg(args, "yaw", camera_.Yaw() * 180.0f / kPi);
                      const double pitch = NumArg(args, "pitch", camera_.Pitch() * 180.0f / kPi);
                      camera_.SetView(target, static_cast<float>(distance), static_cast<float>(yaw) * kPi / 180.0f,
                                      static_cast<float>(pitch) * kPi / 180.0f);
                      return Ok(Object({{"target", Vec3Json(camera_.ResolvedTarget())},
                                        {"eye", Vec3Json(camera_.ResolvedEye())},
                                        {"distance", json::Value::Float(camera_.Distance())},
                                        {"yaw", json::Value::Float(camera_.Yaw() * 180.0f / kPi)},
                                        {"pitch", json::Value::Float(camera_.Pitch() * 180.0f / kPi)}}));
                  }});
    // Selects what the viewport shows inside x0, y0 to x1, y1 in the same fractions; add keeps the selection.
    actions_.Add({"view.select_rect", "Box Select", "view", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const SceneViewportRect& rect = ui_.Viewport();
                      if (!rect.Valid()) return ActionResult::Fail("the viewport is hidden");
                      const std::string mode = StrArg(args, "mode", "replace");
                      if (mode != "replace" && mode != "add") return ActionResult::Fail("the mode is replace or add");
                      const auto at = [&](const char* key, float origin, float size, double fallback) {
                          return origin + static_cast<float>(NumArg(args, key, fallback)) * size;
                      };
                      const render::PickResult pick =
                          PickBox(camera_, at("x0", rect.x, rect.width, 0.0), at("y0", rect.y, rect.height, 0.0),
                                  at("x1", rect.x, rect.width, 1.0), at("y1", rect.y, rect.height, 1.0));
                      return RunAction("scene.select", BoxSelectArgs(pick, mode == "add"), "view");
                  }});
    // Path-traced frames of the view over the raster one, refined up to max_spp samples while nothing moves.
    actions_.Add({"view.rt_preview", "RT Preview", "view", "", ShortcutScope::Global, has_scene,
                  [this] { return rt_.on; },
                  [this](const json::Value& args) {
                      const double max_spp = NumArg(args, "max_spp", rt_.max_spp);
                      if (max_spp < 1.0) return ActionResult::Fail("max_spp: at least one sample");
                      rt_.on = BoolArg(args, "on", !rt_.on);
                      rt_.max_spp = static_cast<uint32_t>(max_spp);
                      rt_.chain = RtKey{};
                      rt_.error.clear();
                      return Ok(Object({{"on", json::Value::Bool(rt_.on)}, {"max_spp", json::Value::Int(rt_.max_spp)}}));
                  }});
    actions_.Add({"view.rt_status", "RT Preview Status", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      json::Value done = json::Value::Array();
                      for (const uint32_t s : rt_.done) done.PushBack(json::Value::Int(s));
                      return Ok(Object({{"on", json::Value::Bool(rt_.on)},
                                        {"spp", json::Value::Int(rt_.spp)},
                                        {"max_spp", json::Value::Int(rt_.max_spp)},
                                        {"converged", json::Value::Bool(ui_state_.rt.converged)},
                                        {"shown", json::Value::Bool(ui_state_.rt.shown)},
                                        {"pending", json::Value::Bool(rt_.inflight != 0u)},
                                        {"done", std::move(done)},
                                        {"width", json::Value::Int(rt_.chain.width)},
                                        {"height", json::Value::Int(rt_.chain.height)},
                                        {"error", rt_.error.empty() ? json::Value::Null() : json::Value::Str(rt_.error)}}));
                  }});
    // The RT image the viewport shows, as a PNG; the dialog asks without `path`.
    actions_.Add({"view.rt_save", "Save RT Image...", "view", "", ShortcutScope::Global,
                  [this] { return ui_state_.rt.shown; }, {},
                  [this](const json::Value& args) {
                      const RtView& v = ui_state_.rt;
                      std::string path = StrArg(args, "path");
                      if (path.empty()) path = SaveFileDialog(nui::Tr("Save RT Image"), nui::Tr("PNG images"), "*.png", "preview.png");
                      if (path.empty()) return ActionResult::Fail("no image path chosen");
                      const int w = static_cast<int>(v.width), h = static_cast<int>(v.height);
                      if (stbi_write_png(path.c_str(), w, h, 3, v.rgb.data(), w * 3) == 0)
                          return ActionResult::Fail("cannot write " + path);
                      return Ok(Object({{"path", json::Value::Str(path)}, {"width", json::Value::Int(w)},
                                        {"height", json::Value::Int(h)}, {"spp", json::Value::Int(v.spp)}}));
                  }});
    // A path-traced PNG of the view at {width, height, spp}, rendered by another process from a copy of
    // the scene and its state; view.render_status reports it.
    actions_.Add({"view.render", "Render This View...", "view", "", ShortcutScope::Global,
                  [this] { return loaded_ != nullptr && !render_; }, {},
                  [this](const json::Value& args) {
                      const double width = NumArg(args, "width", 1920.0), height = NumArg(args, "height", 1080.0);
                      const double spp = NumArg(args, "spp", 64.0);
                      if (width < 1.0 || height < 1.0 || spp < 1.0) return ActionResult::Fail("width, height, spp: at least one");
                      if (opt_.relaunch.empty()) return ActionResult::Fail("this session has no command to start a render");
                      std::string path = StrArg(args, "path");
                      if (path.empty()) path = SaveFileDialog(nui::Tr("Render This View"), nui::Tr("PNG images"), "*.png", "render.png");
                      if (path.empty()) return ActionResult::Fail("no image path chosen");
                      const fs::path out = fs::absolute(Utf8Path(path));
                      if (!fs::is_directory(out.parent_path())) return ActionResult::Fail("no folder " + PathText(out.parent_path()));
                      const fs::path dir = Utf8Path(cache_.File("render"));
                      fs::remove_all(dir);
                      fs::create_directories(dir);
                      const std::string scene = PathText(dir / "scene.nks"), state = PathText(dir / "state.ckpt");
                      const std::string job = PathText(dir / "job.json"), result = PathText(dir / "result.json");
                      scene::nks::Save(loaded_->scene, scene);
                      worker_.Exclusive([&] {
                          nuka_checkpoint_handle cp = nullptr;
                          CheckResult(nuka_world_checkpoint_capture(world_, &cp), "nuka_world_checkpoint_capture");
                          const nuka_result_t r = nuka_checkpoint_write(cp, state.c_str());
                          nuka_checkpoint_destroy(cp);
                          CheckResult(r, "nuka_checkpoint_write");
                      });
                      const json::Value camera = Object({{"eye", Vec3Json(camera_.ResolvedEye())},
                                                         {"look", Vec3Json(camera_.ResolvedTarget())},
                                                         {"up", Vec3Json(math::Vec3{0.0f, 0.0f, 1.0f})},
                                                         {"fov", json::Value::Float(camera_.fov_degrees)}});
                      const json::Value spec = Object({{"scene", json::Value::Str(scene)},
                                                       {"checkpoint", json::Value::Str(state)},
                                                       {"output", json::Value::Str(PathText(out))},
                                                       {"result", json::Value::Str(result)},
                                                       {"width", json::Value::Int(static_cast<int64_t>(width))},
                                                       {"height", json::Value::Int(static_cast<int64_t>(height))},
                                                       {"spp", json::Value::Int(static_cast<int64_t>(spp))},
                                                       {"dt", json::Value::Float(opt_.dt)},
                                                       {"envs", json::Value::Int(loaded_->Layout().env_count)},
                                                       {"camera", camera}});
                      std::ofstream(Utf8Path(job), std::ios::binary | std::ios::trunc) << spec.Dump(2);
                      std::vector<std::string> argv = opt_.relaunch;
                      argv.insert(argv.end(), {"--render", job});
                      std::string error;
                      const uint64_t pid = SpawnDetached(argv, &error);
                      if (pid == 0u) return ActionResult::Fail(error);
                      render_ = RenderJob{PathText(out), result, pid, Clock::now()};
                      ui_state_.rendering = true;
                      return Ok(Object({{"path", json::Value::Str(PathText(out))},
                                        {"process", json::Value::Int(static_cast<int64_t>(pid))}}));
                  }});
    actions_.Add({"view.render_status", "Render Status", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      return Ok(Object({{"running", json::Value::Bool(render_.has_value())},
                                        {"path", render_ ? json::Value::Str(render_->path) : json::Value::Null()},
                                        {"seconds", render_ ? json::Value::Float(std::chrono::duration<double>(
                                                                                     Clock::now() - render_->started).count())
                                                            : json::Value::Null()},
                                        {"last", rendered_}}));
                  }});
    actions_.Add({"view.bookmarks", "Camera Bookmarks", "view", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) { return Ok(BookmarksJson(bookmarks_).At("cameras")); }});
    // Keeps the view as `name`, replacing a bookmark of that name; without one it is "View N".
    actions_.Add({"view.bookmark_add", "Add Camera Bookmark", "view", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      std::string name = StrArg(args, "name");
                      for (size_t n = bookmarks_.size() + 1u; name.empty(); ++n)
                          if (FindBookmark(Format("View %zu", n)) == nullptr) name = Format("View %zu", n);
                      CameraBookmark* b = FindBookmark(name);
                      if (b == nullptr) b = &bookmarks_.emplace_back();
                      *b = CameraBookmark{name, camera_.ResolvedTarget(), camera_.Distance(), camera_.Yaw() * 180.0f / kPi,
                                          camera_.Pitch() * 180.0f / kPi, camera_.fov_degrees};
                      SyncBookmarks(true);
                      return Ok(Object({{"name", json::Value::Str(name)},
                                        {"count", json::Value::Int(static_cast<int64_t>(bookmarks_.size()))}}));
                  }});
    actions_.Add({"view.bookmark_goto", "Go to Camera Bookmark", "view", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const std::string name = StrArg(args, "name");
                      const CameraBookmark* b = FindBookmark(name);
                      if (b == nullptr) return ActionResult::Fail("no camera bookmark '" + name + "'");
                      camera_.SetView(b->target, b->distance, b->yaw * kPi / 180.0f, b->pitch * kPi / 180.0f);
                      camera_.fov_degrees = b->fov;
                      return Ok(Object({{"name", json::Value::Str(name)}}));
                  }});
    actions_.Add({"view.bookmark_rename", "Rename Camera Bookmark", "view", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const std::string name = StrArg(args, "name"), to = StrArg(args, "to");
                      CameraBookmark* b = FindBookmark(name);
                      if (b == nullptr) return ActionResult::Fail("no camera bookmark '" + name + "'");
                      if (to.empty() || (to != name && FindBookmark(to) != nullptr))
                          return ActionResult::Fail("the new name is empty or taken");
                      b->name = to;
                      SyncBookmarks(true);
                      return Ok(Object({{"name", json::Value::Str(to)}}));
                  }});
    actions_.Add({"view.bookmark_remove", "Remove Camera Bookmark", "view", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const std::string name = StrArg(args, "name");
                      const CameraBookmark* b = FindBookmark(name);
                      if (b == nullptr) return ActionResult::Fail("no camera bookmark '" + name + "'");
                      bookmarks_.erase(bookmarks_.begin() + (b - bookmarks_.data()));
                      SyncBookmarks(true);
                      return Ok(Object({{"count", json::Value::Int(static_cast<int64_t>(bookmarks_.size()))}}));
                  }});
    // Writes the bookmarks in the format --cams reads; the dialog asks without `path`.
    actions_.Add({"view.bookmarks_save", "Export Camera Bookmarks...", "view", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      std::string path = StrArg(args, "path");
                      if (path.empty())
                          path = SaveFileDialog(nui::Tr("Export Camera Bookmarks"), nui::Tr("Camera bookmarks"), "*.json", "cameras.json");
                      if (path.empty()) return ActionResult::Fail("no bookmark file chosen");
                      std::ofstream out(Utf8Path(path), std::ios::binary | std::ios::trunc);
                      out << BookmarksJson(bookmarks_).Dump(2);
                      if (!out) return ActionResult::Fail("cannot write " + path);
                      return Ok(Object({{"path", json::Value::Str(path)},
                                        {"count", json::Value::Int(static_cast<int64_t>(bookmarks_.size()))}}));
                  }});
    // Replaces the bookmarks with a file's.
    actions_.Add({"view.bookmarks_load", "Import Camera Bookmarks...", "view", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      std::string path = StrArg(args, "path");
                      if (path.empty()) path = OpenFileDialog(nui::Tr("Import Camera Bookmarks"), nui::Tr("Camera bookmarks"), "*.json");
                      if (path.empty()) return ActionResult::Fail("no bookmark file chosen");
                      if (!ReadBookmarks(path, &bookmarks_)) return ActionResult::Fail("cannot read " + path);
                      SyncBookmarks(true);
                      return Ok(Object({{"path", json::Value::Str(path)},
                                        {"count", json::Value::Int(static_cast<int64_t>(bookmarks_.size()))}}));
                  }});
}

void EditorHost::State::RegisterSolverActions() {
    const auto has_scene = [this] { return loaded_ != nullptr; };
    const auto telemetry = [this](uint32_t mask, float alert) {
        solver_seq_ = worker_.Push(TickCommand::Param(ParamKind::Telemetry, {static_cast<float>(mask), alert}));
    };
    // Without a mask the action toggles every channel; the alert defaults to the current one.
    actions_.Add({"solver.telemetry", "Record Solver Telemetry", "solver", "", ShortcutScope::Global, has_scene,
                  [this] { return ui_state_.solver.mask != 0u; },
                  [this, telemetry](const json::Value& args) {
                      const SolverView& v = ui_state_.solver;
                      const auto mask = static_cast<uint32_t>(
                          NumArg(args, "mask", v.mask != 0u ? 0.0 : static_cast<double>(kSolverTelemetryAll)));
                      const auto alert = static_cast<float>(NumArg(args, "alert", v.alert));
                      telemetry(mask, alert);
                      return Ok(Object({{"mask", json::Value::Int(mask)}, {"alert", json::Value::Float(alert)}}));
                  }});
    // The heatmap needs the per-vertex channel and turns it on.
    actions_.Add({"solver.heatmap", "Residual Heatmap", "solver", "", ShortcutScope::Global, has_scene,
                  [this] { return ui_state_.solver.heatmap; },
                  [this, telemetry](const json::Value& args) {
                      SolverView& v = ui_state_.solver;
                      v.heatmap = BoolArg(args, "on", !v.heatmap);
                      if (v.heatmap && (v.mask & NUKA_TELEMETRY_VERTICES) == 0u)
                          telemetry(v.mask != 0u ? v.mask | NUKA_TELEMETRY_VERTICES : kSolverTelemetryAll, v.alert);
                      solver_seq_ = worker_.Push(TickCommand::Param(ParamKind::Heatmap, {v.heatmap ? 1.0f : 0.0f}));
                      return Ok(Object({{"on", json::Value::Bool(v.heatmap)}}));
                  }});
    actions_.Add({"solver.set_iterations", "Set Sweep Budget", "solver", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const auto n = static_cast<uint32_t>(NumArg(args, "n", 0.0));
                      if (n == 0u) return ActionResult::Fail("n: a positive sweep count");
                      solver_seq_ = worker_.Push(TickCommand::Param(ParamKind::Iterations, {static_cast<float>(n)}));
                      return Ok(Object({{"n", json::Value::Int(n)}}));
                  }});
    actions_.Add({"solver.status", "Solver Status", "solver", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      const SolverView& v = ui_state_.solver;
                      json::Value mark = json::Value::Null();
                      if (v.marked) {
                          json::Value position = json::Value::Array();
                          for (const float x : v.mark) position.PushBack(json::Value::Float(x));
                          mark = Object({{"position", std::move(position)},
                                         {"label", json::Value::Str(MarkLabel(v, false))}});
                      }
                      return Ok(Object(
                          {{"mask", json::Value::Int(v.mask)},
                           {"alert", json::Value::Float(v.alert)},
                           {"iterations", json::Value::Int(v.iterations)},
                           {"heatmap", json::Value::Bool(v.heatmap)},
                           {"records", json::Value::Int(static_cast<int64_t>(v.records))},
                           {"caps", json::Value::Int(static_cast<int64_t>(v.caps))},
                           {"recent_caps", json::Value::Int(v.recent_caps)},
                           {"recent_records", json::Value::Int(v.recent_records)},
                           {"dropped", json::Value::Int(static_cast<int64_t>(v.dropped))},
                           {"first", json::Value::Int(static_cast<int64_t>(v.first))},
                           {"next", json::Value::Int(static_cast<int64_t>(v.first + v.history.size()))},
                           {"applied", json::Value::Bool(shown_ != nullptr && shown_->acked >= solver_seq_)},
                           {"last", v.history.empty() ? json::Value::Null()
                                                      : RecordJson(v.history.back(), v.first + v.history.size() - 1u)},
                           {"marker", std::move(mark)}}));
                  }});
    // Pages forward through the kept records: `since` on, at most `limit`, oldest first.
    actions_.Add({"solver.records", "Solver Records", "solver", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      const SolverView& v = ui_state_.solver;
                      const uint64_t end = v.first + v.history.size();
                      const auto limit = static_cast<uint64_t>(NumArg(args, "limit", 64.0));
                      uint64_t n = std::max(static_cast<uint64_t>(NumArg(args, "since", 0.0)), v.first);
                      json::Value records = json::Value::Array();
                      const bool capped_only = BoolArg(args, "capped", false);
                      for (; n < end && records.Size() < limit; ++n) {
                          const nuka_telemetry_record_t& r = v.history[n - v.first];
                          if (!capped_only || (r.flags & NUKA_TELEMETRY_CAP) != 0u) records.PushBack(RecordJson(r, n));
                      }
                      return Ok(Object({{"records", std::move(records)},
                                        {"first", json::Value::Int(static_cast<int64_t>(v.first))},
                                        {"next", json::Value::Int(static_cast<int64_t>(n))}}));
                  }});
    // Frames and selects the worst element of a kept record: the given one, else the newest capped
    // one, else the newest; {"clear": true} drops the marker.
    actions_.Add({"solver.locate", "Locate Worst Element", "solver", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      SolverView& v = ui_state_.solver;
                      if (BoolArg(args, "clear", false)) {
                          v.marked = false;
                          return Ok();
                      }
                      if (v.history.empty()) return ActionResult::Fail("no telemetry records; record and run first");
                      size_t at = v.history.size() - 1u;
                      if (const json::Value* record = args.Find("record"); record != nullptr && !record->IsNull()) {
                          const auto n = static_cast<uint64_t>(record->AsDouble());
                          if (n < v.first || n >= v.first + v.history.size())
                              return ActionResult::Fail("record " + std::to_string(n) + " is no longer kept");
                          at = static_cast<size_t>(n - v.first);
                      } else {
                          const auto it = std::find_if(v.history.rbegin(), v.history.rend(), [](const nuka_telemetry_record_t& r) {
                              return (r.flags & NUKA_TELEMETRY_CAP) != 0u;
                          });
                          if (it != v.history.rend()) at = static_cast<size_t>(v.history.rend() - it) - 1u;
                      }
                      const nuka_telemetry_record_t record = v.history[at];
                      LocatedElement e;
                      bool found = false;
                      worker_.Exclusive([&] { found = LocateTelemetry(world_, loaded_->Layout(), loaded_->scene, record, &e); });
                      if (!found) return ActionResult::Fail("the record names no element in the world");
                      ui_state_.env_index = e.env;
                      PushTransport();
                      if (e.entity != scene::kInvalidEntity) {
                          ui_state_.selected_entity = e.entity;
                          ui_state_.selection = {e.entity};
                      }
                      const math::Vec3 half{0.2f, 0.2f, 0.2f};
                      camera_.FrameAabb(e.position - half, e.position + half, ui_.Viewport().Aspect());
                      v.marked = true;
                      v.mark[0] = e.position.x;
                      v.mark[1] = e.position.y;
                      v.mark[2] = e.position.z;
                      v.mark_kind = e.kind;
                      v.mark_index = e.index;
                      v.mark_row = e.row;
                      v.mark_residual = record.worst_residual;
                      json::Value position = json::Value::Array();
                      for (const float x : v.mark) position.PushBack(json::Value::Float(x));
                      return Ok(Object({{"record", json::Value::Int(static_cast<int64_t>(v.first + at))},
                                        {"label", json::Value::Str(MarkLabel(v, false))},
                                        {"env", json::Value::Int(e.env)},
                                        {"position", std::move(position)},
                                        {"entity", json::Value::Str(PathOfEntity(e.entity))},
                                        {"residual", json::Value::Float(record.worst_residual)},
                                        {"capped", json::Value::Bool((record.flags & NUKA_TELEMETRY_CAP) != 0u)}}));
                  }});
}

json::Value EditorHost::State::TimelineStatus() {
    const TimelineInfo& t = ui_state_.timeline;
    const auto ticks = [](const std::vector<uint64_t>& values) {
        json::Value list = json::Value::Array();
        for (const uint64_t v : values) list.PushBack(json::Value::Int(static_cast<int64_t>(v)));
        return list;
    };
    const auto id = [](uint32_t branch) {
        return branch == kNoBranch ? json::Value::Null() : json::Value::Int(branch);
    };
    json::Value branches = json::Value::Array();
    for (const TimelineBranchInfo& b : t.branches)
        branches.PushBack(Object({{"id", json::Value::Int(b.id)},
                                  {"parent", id(b.parent)},
                                  {"fork_tick", json::Value::Int(static_cast<int64_t>(b.fork_tick))},
                                  {"head", json::Value::Int(static_cast<int64_t>(b.head))},
                                  {"label", json::Value::Str(b.label)},
                                  {"exact", json::Value::Bool(b.exact)}}));
    const bool ghost_shown = shown_ != nullptr && (!shown_->ghost_links.empty() || !shown_->ghost_bodies.empty());
    return Object({{"tick", json::Value::Int(static_cast<int64_t>(t.tick))},
                   {"start", json::Value::Int(static_cast<int64_t>(t.start))},
                   {"head", json::Value::Int(static_cast<int64_t>(t.head))},
                   {"replaying", json::Value::Bool(t.tick < t.head)},
                   {"branch", json::Value::Int(t.branch)},
                   {"branches", std::move(branches)},
                   {"keys", ticks(t.keys)},
                   {"pinned", ticks(t.pinned)},
                   {"ghost", id(t.ghost)},
                   {"ghost_on", json::Value::Bool(t.ghost_on)},
                   {"ghost_shown", json::Value::Bool(ghost_shown)},
                   {"interval", json::Value::Int(t.interval)},
                   {"keyframes", json::Value::Int(t.keyframes)},
                   {"bytes", json::Value::Int(static_cast<int64_t>(t.bytes))},
                   {"budget_mb", json::Value::Float(static_cast<double>(t.budget) / 1048576.0)},
                   {"applied", json::Value::Bool(shown_ != nullptr && shown_->acked >= timeline_seq_)},
                   {"export", !video_ ? json::Value::Null()
                                      : Object({{"path", json::Value::Str(video_->path)},
                                                {"frame", json::Value::Int(static_cast<int64_t>(video_->next))},
                                                {"frames", json::Value::Int(static_cast<int64_t>(video_->ticks.size()))},
                                                {"encoding", json::Value::Bool(video_->encoder != 0u)}})},
                   {"exported", exported_}});
}

json::Value EditorHost::State::RecordingIdentity() {
    const WorldLayout& l = loaded_->Layout();
    const std::string counts = Format("%u %u %u %u %u %u %.9g", l.env_count, l.links_per_env, l.bodies_per_env,
                                      l.articulations_per_env, l.dofs_per_env, l.particles_per_env,
                                      static_cast<double>(opt_.dt));
    std::vector<std::string> links(l.links_per_env), bodies(l.bodies_per_env);
    for (const render::RenderInstance& inst : loaded_->sim->GetRenderWorld().instances) {
        const bool link = inst.pose_source.kind == render::PoseSource::Kind::Link;
        std::vector<std::string>& rows = link ? links : bodies;
        if ((link || inst.pose_source.kind == render::PoseSource::Kind::Body) && inst.pose_source.row < rows.size())
            rows[inst.pose_source.row] = PathOfEntity(CarrierOf(loaded_->scene, inst.entity));
    }
    const auto names = [](const std::vector<std::string>& rows) {
        json::Value list = json::Value::Array();
        for (const std::string& name : rows) list.PushBack(json::Value::Str(name));
        return list;
    };
    return Object({{"scene", json::Value::Str(Hex(DocumentHash()))},
                   {"layout", json::Value::Str(Hex(Fnv1a(counts)))},
                   {"links", names(links)},
                   {"bodies", names(bodies)}});
}

// One step of an mp4 export a frame: seek the next frame's tick, capture the viewport once the
// seek shows, then hand the frames to ffmpeg and wait for it to finish.
void EditorHost::State::StepVideo() {
    VideoExport& v = *video_;
    const auto frame = [&v](size_t i) { return Format("%s/f%05zu.ppm", v.frames.c_str(), i); };
    const auto finish = [this, &v](const std::string& error) {
        exported_ = Object({{"path", json::Value::Str(v.path)},
                            {"frames", json::Value::Int(static_cast<int64_t>(v.ticks.size()))},
                            {"error", json::Value::Str(error)}});
        if (error.empty()) {
            fs::remove_all(Utf8Path(v.frames));
            ui_.Nui().Toast(nui::Level::Ok, std::string(nui::Tr("Video exported")) + " " + FileName(v.path));
        } else {
            Emit(stderr, "[nuka_editor] video export failed: " + error + "\n");
            ui_.Nui().Toast(nui::Level::Err, std::string(nui::Tr("Video export failed")) + ": " + error);
        }
        video_.reset();
    };
    if (v.encoder != 0u) {
        if (ProcessAlive(static_cast<uint32_t>(v.encoder))) return;
        std::error_code ec;
        const uintmax_t size = fs::file_size(Utf8Path(v.path), ec);
        finish(!ec && size > 0u ? std::string() : "ffmpeg made no video of the frames in " + v.frames);
        return;
    }
    if (v.capture != ~0ull) {
        if (present_->SubmittedFrames() <= v.capture) return;
        if (!fs::exists(Utf8Path(frame(v.next)))) return finish("the viewport was hidden");
        v.capture = ~0ull;
        if (++v.next == v.ticks.size()) {
            RunAction("timeline.seek", Object({{"tick", json::Value::Int(static_cast<int64_t>(v.back))}}), "export");
            std::string error;
            v.encoder = SpawnDetached({"ffmpeg", "-y", "-loglevel", "error", "-framerate", Format("%g", v.fps), "-i",
                                       v.frames + "/f%05d.ppm", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v",
                                       "libx264", "-pix_fmt", "yuv420p", v.path},
                                      &error);
            if (v.encoder == 0u) finish(error + "; mp4 export runs ffmpeg from the PATH");
            return;
        }
    }
    if (v.seek == 0u) {
        RunAction("timeline.seek", Object({{"tick", json::Value::Int(static_cast<int64_t>(v.ticks[v.next]))}}), "export");
        v.seek = timeline_seq_;
        return;
    }
    if (shown_ == nullptr || shown_->acked < v.seek) return;
    v.seek = 0u;
    v.capture = present_->SubmittedFrames();
    present_->SetSceneCaptureFrame(static_cast<int>(v.capture), frame(v.next));
}

void EditorHost::State::RegisterTimelineActions() {
    const auto has_scene = [this] { return loaded_ != nullptr; };
    const auto push = [this](TickCommand cmd) {
        PushTransport();
        timeline_seq_ = worker_.Push(std::move(cmd));
        return Ok();
    };
    // Seeking and switching branches pause first, so the playhead stays where it lands.
    const auto pause = [this] {
        ui_state_.playing = false;
        PushTransport();
    };
    actions_.Add({"timeline.status", "Timeline Status", "timeline", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) { return Ok(TimelineStatus()); }});
    // {tick}, {delta} ticks from the playhead, or {to: "start" | "head"}.
    actions_.Add({"timeline.seek", "Seek", "timeline", "", ShortcutScope::Global, has_scene, {},
                  [push, pause](const json::Value& args) {
                      TickCommand cmd = TickCommand::Timeline(TimelineOp::Seek);
                      const std::string to = StrArg(args, "to");
                      if (to == "start" || to == "head") {
                          cmd.tick = to == "head" ? ~uint64_t(0) : 0u;
                      } else if (args.Find("delta") != nullptr) {
                          cmd.relative = true;
                          cmd.delta = static_cast<int64_t>(NumArg(args, "delta", 0.0));
                      } else if (args.Find("tick") != nullptr) {
                          cmd.tick = static_cast<uint64_t>(std::max(NumArg(args, "tick", 0.0), 0.0));
                      } else {
                          return ActionResult::Fail("timeline.seek takes tick, delta, or to: start or head");
                      }
                      pause();
                      return push(std::move(cmd));
                  }});
    actions_.Add({"timeline.fork", "Fork Branch", "timeline", "", ShortcutScope::Global, has_scene, {},
                  [push](const json::Value& args) {
                      TickCommand cmd = TickCommand::Timeline(TimelineOp::Fork);
                      cmd.label = StrArg(args, "label");
                      return push(std::move(cmd));
                  }});
    actions_.Add({"timeline.branch", "Switch Branch", "timeline", "", ShortcutScope::Global, has_scene, {},
                  [push, pause](const json::Value& args) {
                      if (args.Find("id") == nullptr) return ActionResult::Fail("timeline.branch takes id");
                      TickCommand cmd = TickCommand::Timeline(TimelineOp::Branch);
                      cmd.branch = static_cast<uint32_t>(NumArg(args, "id", 0.0));
                      pause();
                      return push(std::move(cmd));
                  }});
    actions_.Add({"timeline.rename", "Rename Branch", "timeline", "", ShortcutScope::Global, has_scene, {},
                  [this, push](const json::Value& args) {
                      TickCommand cmd = TickCommand::Timeline(TimelineOp::Rename);
                      cmd.branch = static_cast<uint32_t>(NumArg(args, "id", ui_state_.timeline.branch));
                      cmd.label = StrArg(args, "label");
                      return push(std::move(cmd));
                  }});
    // {on, branch}: without `on` it toggles; `branch` picks what it shows, else the last parent.
    actions_.Add({"timeline.ghost", "Ghost of Other Branch", "timeline", "", ShortcutScope::Global, has_scene,
                  [this] { return ui_state_.timeline.ghost_on; },
                  [this, push](const json::Value& args) {
                      TickCommand cmd = TickCommand::Timeline(TimelineOp::Ghost);
                      const bool on = BoolArg(args, "on", !ui_state_.timeline.ghost_on);
                      cmd.values = {on ? 1.0f : 0.0f};
                      if (args.Find("branch") != nullptr) cmd.branch = static_cast<uint32_t>(NumArg(args, "branch", 0.0));
                      ui_state_.timeline.ghost_on = on;
                      return push(std::move(cmd));
                  }});
    actions_.Add({"timeline.configure", "Keyframe Settings", "timeline", "", ShortcutScope::Global, has_scene, {},
                  [this, push](const json::Value& args) {
                      const TimelineInfo& t = ui_state_.timeline;
                      const double interval = NumArg(args, "interval", t.interval);
                      const double budget = NumArg(args, "budget_mb", static_cast<double>(t.budget) / 1048576.0);
                      if (interval < 1.0 || budget <= 0.0)
                          return ActionResult::Fail("interval: ticks >= 1; budget_mb: a positive size");
                      TickCommand cmd = TickCommand::Timeline(TimelineOp::Configure);
                      cmd.values = {static_cast<float>(interval), static_cast<float>(budget)};
                      return push(std::move(cmd));
                  }});
    actions_.Add({"timeline.clear", "Clear History", "timeline", "", ShortcutScope::Global, has_scene, {},
                  [push](const json::Value&) { return push(TickCommand::Timeline(TimelineOp::Clear)); }});
    // The world's state hash once every command queued before it applied.
    actions_.Add({"timeline.hash", "State Hash", "timeline", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value&) {
                      uint64_t hash = 0u;
                      worker_.Exclusive([&] { CheckResult(nuka_world_state_hash(world_, &hash), "nuka_world_state_hash"); });
                      char text[17];
                      std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
                      return Ok(Object({{"hash", json::Value::Str(text)},
                                        {"tick", json::Value::Int(static_cast<int64_t>(worker_.Tick()))}}));
                  }});
    // The current branch's line of history with what it plays in; the dialog asks without `path`.
    actions_.Add({"timeline.save", "Save Recording...", "timeline", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      std::string path = StrArg(args, "path");
                      if (path.empty())
                          path = SaveFileDialog(nui::Tr("Save Recording"), nui::Tr("Nuka recordings"), "*.nkrec",
                                                "recording.nkrec");
                      if (path.empty()) return ActionResult::Fail("no recording path chosen");
                      json::Value manifest = RecordingIdentity();
                      worker_.Exclusive([&] {
                          manifest = worker_.SaveHistory(path, std::move(manifest), cache_.File("keyframe.ckpt"));
                      });
                      ui_.Nui().Toast(nui::Level::Ok, std::string(nui::Tr("Recording saved")) + " " + FileName(path));
                      return Ok(Object({{"path", json::Value::Str(path)},
                                        {"start", manifest.At("start")},
                                        {"head", manifest.At("head")},
                                        {"keyframes", manifest.At("keyframes")},
                                        {"inputs", manifest.At("inputs")}}));
                  }});
    // A recording of this document and world layout becomes the history, back where it was saved.
    actions_.Add({"timeline.load", "Load Recording...", "timeline", "", ShortcutScope::Global, has_scene, {},
                  [this, pause](const json::Value& args) {
                      std::string path = StrArg(args, "path");
                      if (path.empty())
                          path = OpenFileDialog(nui::Tr("Load Recording"), nui::Tr("Nuka recordings"), "*.nkrec");
                      if (path.empty()) return ActionResult::Fail("no recording chosen");
                      const json::Value manifest = ReadRecordingManifest(path);
                      const json::Value here = RecordingIdentity();
                      if (manifest.At("scene").AsString() != here.At("scene").AsString())
                          return ActionResult::Fail("'" + FileName(path) + "' records another scene");
                      if (manifest.At("layout").AsString() != here.At("layout").AsString())
                          return ActionResult::Fail("'" + FileName(path) + "' records another world layout");
                      pause();
                      worker_.Exclusive([&] { worker_.LoadHistory(path, cache_.File("keyframe.ckpt")); });
                      ui_.Nui().Toast(nui::Level::Ok, std::string(nui::Tr("Recording loaded")) + " " + FileName(path));
                      return Ok(Object({{"path", json::Value::Str(path)},
                                        {"start", manifest.At("start")},
                                        {"head", manifest.At("head")},
                                        {"tick", manifest.At("tick")}}));
                  }});
    // An mp4 of ticks `from` to `to` (the start and head) at `fps` frames a simulated second, drawn
    // by the viewport and encoded by ffmpeg from the PATH; timeline.status reports its progress.
    actions_.Add({"timeline.export", "Export Video...", "timeline", "", ShortcutScope::Global,
                  [this] { return loaded_ != nullptr && !video_; }, {},
                  [this](const json::Value& args) {
                      const TimelineInfo& t = ui_state_.timeline;
                      const double fps = NumArg(args, "fps", 30.0);
                      const double from = NumArg(args, "from", static_cast<double>(t.start));
                      const double to = NumArg(args, "to", static_cast<double>(t.head));
                      if (!(fps > 0.0) || from < static_cast<double>(t.start) || to > static_cast<double>(t.head) || from > to)
                          return ActionResult::Fail("fps: positive; from, to: ticks between the start and the head");
                      if (!ui_.Viewport().Valid()) return ActionResult::Fail("the viewport is hidden");
                      std::string path = StrArg(args, "path");
                      if (path.empty()) path = SaveFileDialog(nui::Tr("Export Video"), nui::Tr("Videos"), "*.mp4", "recording.mp4");
                      if (path.empty()) return ActionResult::Fail("no video path chosen");
                      VideoExport v;
                      v.path = path;
                      v.fps = fps;
                      v.back = t.tick;
                      v.frames = cache_.File("video");
                      fs::remove_all(Utf8Path(v.frames));
                      fs::create_directories(Utf8Path(v.frames));
                      const double stride = std::max(1.0, 1.0 / (fps * static_cast<double>(opt_.dt)));
                      for (double at = from; at <= to; at += stride) v.ticks.push_back(static_cast<uint64_t>(std::llround(at)));
                      const size_t frames = v.ticks.size();
                      video_ = std::move(v);
                      exported_ = json::Value::Null();
                      return Ok(Object({{"path", json::Value::Str(path)}, {"frames", json::Value::Int(static_cast<int64_t>(frames))}}));
                  }});
}

// FNV-1a of the document saved as NKS: equal documents hash equal.
uint64_t EditorHost::State::DocumentHash() {
    if (!loaded_) return 0u;
    const std::string file = cache_.File("document.nks");
    scene::nks::Save(loaded_->scene, file);
    std::string nks;
    std::string archive;
    ReadEntireFile(file, &nks);
    ReadEntireFile(file.substr(0, file.size() - 4u) + ".nka", &archive);
    return Fnv1a(archive, Fnv1a(nks));
}

json::Value EditorHost::State::DocumentState() {
    char hash[17];
    std::snprintf(hash, sizeof(hash), "%016llx", static_cast<unsigned long long>(DocumentHash()));
    json::Value branches = json::Value::Array();
    for (const TimelineBranch& b : doc_.Branches()) {
        json::Value v = json::Value::Object();
        v.Set("tick", json::Value::Int(static_cast<int64_t>(b.tick)));
        v.Set("generation", json::Value::Int(static_cast<int64_t>(b.generation)));
        v.Set("label", json::Value::Str(b.label));
        branches.PushBack(std::move(v));
    }
    json::Value v = json::Value::Object();
    v.Set("loaded", json::Value::Bool(loaded_ != nullptr));
    v.Set("revision", json::Value::Int(static_cast<int64_t>(doc_.Revision())));
    v.Set("dirty", json::Value::Bool(loaded_ != nullptr && doc_.Dirty()));
    v.Set("hash", json::Value::Str(hash));
    v.Set("undo", json::Value::Int(static_cast<int64_t>(doc_.UndoDepth())));
    v.Set("redo", json::Value::Int(static_cast<int64_t>(doc_.RedoDepth())));
    v.Set("tick", json::Value::Int(static_cast<int64_t>(worker_.Tick())));
    v.Set("generation", json::Value::Int(static_cast<int64_t>(world_ != nullptr ? Generation() : 0u)));
    v.Set("branches", std::move(branches));
    v.Set("param_done", json::Value::Bool(param_done_));
    v.Set("param_pushed_tick", json::Value::Int(static_cast<int64_t>(param_pushed_tick_)));
    v.Set("param_applied_tick", json::Value::Int(static_cast<int64_t>(param_applied_tick_)));
    v.Set("last_error", json::Value::Str(last_error_));
    v.Set("selection", json::Value::Str(SelectedPath()));
    v.Set("playing", json::Value::Bool(ui_state_.playing));
    v.Set("script_busy_s", json::Value::Float(script_host_.BusySeconds()));
    v.Set("script_budget_s", json::Value::Float(script_host_.Budget()));
    v.Set("path", json::Value::Str(doc_path_));
    v.Set("title", json::Value::Str(SessionTitle()));
    v.Set("prompt", json::Value::Str(ui_state_.prompt));
    v.Set("session", json::Value::Str(borrowed_ ? "script" : "editor"));
    v.Set("start_page", json::Value::Bool(StartShown()));
    v.Set("primary", json::Value::Bool(instance_ != nullptr && instance_->Listening()));
    v.Set("autosave", json::Value::Str(autosave_file_));
    v.Set("log", json::Value::Str(log_path_));
    v.Set("profile", json::Value::Str(PathText(profile_->Dir())));
    return v;
}

std::unique_ptr<inference::Go2PolicyController> EditorHost::State::MakeController(const std::string& path, bool host,
                                                                                  std::string* error) const {
    // Without a heightfield the height scan reads the z = 0 ground plane.
    static const terrain::HeightField kPlane = [] {
        terrain::HeightField plane;
        plane.values.assign(4u, 0.0f);
        plane.nrow = plane.ncol = 2u;
        plane.cell_x = plane.cell_y = 1.0f;
        return plane;
    }();
    const WorldLayout& layout = loaded_->Layout();
    auto pc = std::make_unique<inference::Go2PolicyController>();
    if (!pc->Init(path, layout.links_per_env, layout.articulations_per_env,
                  terrain::IsValid(loaded_->terrain) ? loaded_->terrain : kPlane, error))
        return nullptr;
    if (host) pc->SetUseGpu(false);
    return pc;
}

// The --policy or policy.load policy, attached again at every load and rebuild.
void EditorHost::State::AttachController() {
    policy_ctrl_.reset();
    ui_state_.policy.error.clear();
    if (!opt_.policy_path.empty()) {
        policy_ctrl_ = MakeController(opt_.policy_path, opt_.force_host_policy, &ui_state_.policy.error);
        if (policy_ctrl_ != nullptr)
            Emit(stdout, Format("[nuka_editor] policy attached: %s  play=%s  path=%s\n", opt_.policy_path.c_str(),
                                opt_.want_play ? "yes" : "no", policy_ctrl_->UsingGpu() ? "GPU" : "host"));
        else
            Emit(stderr, "[nuka_editor] policy Init failed: " + opt_.policy_path + ": " + ui_state_.policy.error + "\n");
    }
    SyncPolicyView();
}

void EditorHost::State::SyncPolicyView() {
    PolicyView& p = ui_state_.policy;
    p.loaded = policy_ctrl_ != nullptr;
    p.on = policy_on_;
    p.path = opt_.policy_path;
    p.links.clear();
    p.defaults.clear();
    p.norm = 0.0f;
    p.norm_time.clear();
    p.norm_value.clear();
    if (policy_ctrl_ == nullptr) return;
    const inference::Go2PolicyController::Config& c = policy_ctrl_->Contract();
    p.gpu = policy_ctrl_->UsingGpu();
    p.robots = policy_ctrl_->Robots();
    p.obs_dim = policy_ctrl_->ObsDim();
    p.act_dim = policy_ctrl_->ActDim();
    p.decimation = c.decimation;
    p.rate_hz = 1.0f / (opt_.dt * static_cast<float>(c.decimation));
    p.action_scale = c.action_scale;
    for (uint32_t k = 0; k < p.robots; ++k)
        for (size_t u = 0; u < c.default_angles.size(); ++u) {
            p.links.push_back(k * policy_ctrl_->LinksPerRobot() + 1u + c.slot_of_urdf[u]);
            p.defaults.push_back(c.default_angles[u]);
        }
    std::copy(c.command.begin(), c.command.end(), p.command);
    std::snprintf(p.path_edit, sizeof(p.path_edit), "%s", opt_.policy_path.c_str());
}

void EditorHost::State::RecordPolicyNorm(const Snapshot& snap) {
    constexpr size_t kNorms = 2000u;
    PolicyView& p = ui_state_.policy;
    const std::vector<float>& targets = snap.joints.target;
    if (policy_ctrl_ == nullptr || !policy_on_ || !snap.playing || snap.tick == policy_tick_ || targets.empty()) return;
    policy_tick_ = snap.tick;
    // An action is (target - default) / scale, which the targets keep between decisions.
    const size_t per = p.links.size() / p.robots;
    float sum = 0.0f;
    for (uint32_t k = 0; k < p.robots; ++k) {
        float squares = 0.0f;
        for (size_t i = k * per; i < (k + 1u) * per; ++i) {
            const float a = (targets[p.links[i]] - p.defaults[i]) / p.action_scale;
            squares += a * a;
        }
        sum += std::sqrt(squares);
    }
    p.norm = sum / static_cast<float>(p.robots);
    p.norm_time.push_back(static_cast<float>(snap.sim_time));
    p.norm_value.push_back(p.norm);
    if (p.norm_time.size() > kNorms) {
        p.norm_time.erase(p.norm_time.begin());
        p.norm_value.erase(p.norm_value.begin());
    }
}

json::Value EditorHost::State::PolicyStatus() {
    const PolicyView& p = ui_state_.policy;
    json::Value v = Object({{"loaded", json::Value::Bool(p.loaded)},
                            {"on", json::Value::Bool(policy_on_)},
                            {"path", json::Value::Str(opt_.policy_path)},
                            {"error", json::Value::Str(p.error)}});
    if (policy_ctrl_ == nullptr) return v;
    // The command the policy reads now, which the stepping thread writes.
    std::array<float, 3> command{};
    worker_.Exclusive([&] { command = policy_ctrl_->Contract().command; });
    const std::vector<RobotJoint>& rj = ui_state_.robot.joints;
    const size_t per = p.links.size() / p.robots;
    json::Value joints = json::Value::Array();
    for (size_t i = 0; i < p.links.size(); ++i) {
        const auto j = std::find_if(rj.begin(), rj.end(), [&](const RobotJoint& r) { return r.link == p.links[i]; });
        joints.PushBack(Object({{"robot", json::Value::Int(static_cast<int64_t>(i / per))},
                                {"name", j != rj.end() ? json::Value::Str(j->name) : json::Value()},
                                {"link", json::Value::Int(p.links[i])},
                                {"default", json::Value::Float(p.defaults[i])}}));
    }
    v.Set("device", json::Value::Str(p.gpu ? "gpu" : "host"));
    v.Set("robots", json::Value::Int(p.robots));
    v.Set("obs_dim", json::Value::Int(p.obs_dim));
    v.Set("act_dim", json::Value::Int(p.act_dim));
    v.Set("decimation", json::Value::Int(p.decimation));
    v.Set("rate_hz", json::Value::Float(p.rate_hz));
    v.Set("action_scale", json::Value::Float(p.action_scale));
    v.Set("joints", std::move(joints));
    v.Set("command", Vec3Json(math::Vec3{command[0], command[1], command[2]}));
    v.Set("action_norm", json::Value::Float(p.norm));
    return v;
}

void EditorHost::State::StopDemo() {
    std::vector<DemoSample> samples;
    worker_.Exclusive([&] {
        worker_.RecordDemo({});
        samples = worker_.TakeDemo();
    });
    CutEpisodes(samples, &demos_);
    demo_recording_ = false;
    ui_state_.demos.selected = static_cast<int>(demos_.episodes.size()) - 1;
    SyncDemoView();
}

void EditorHost::State::StopReplay() {
    demo_play_seq_ = worker_.Push(TickCommand::Param(ParamKind::TargetTrack, {}));
    ui_state_.demos.playing = -1;
    ui_state_.demos.left = 0u;
}

void EditorHost::State::SyncDemoView() {
    DemoView& d = ui_state_.demos;
    d.recording = demo_recording_;
    d.episodes.clear();
    for (const DemoEpisode& e : demos_.episodes)
        d.episodes.push_back(DemoView::Episode{e.env, e.Steps(), static_cast<double>(e.Steps()) * demos_.dt});
    d.selected = std::min(d.selected, static_cast<int>(d.episodes.size()) - 1);
}

json::Value EditorHost::State::DemoStatus() const {
    const DemoView& d = ui_state_.demos;
    json::Value episodes = json::Value::Array();
    for (const DemoEpisode& e : demos_.episodes)
        episodes.PushBack(Object({{"env", json::Value::Int(e.env)},
                                  {"start", json::Value::Int(static_cast<int64_t>(e.start))},
                                  {"steps", json::Value::Int(static_cast<int64_t>(e.Steps()))},
                                  {"seconds", json::Value::Float(static_cast<double>(e.Steps()) * demos_.dt)}}));
    json::Value joints = json::Value::Array(), axes = json::Value::Array();
    for (const std::string& j : demos_.joints) joints.PushBack(json::Value::Str(j));
    for (const std::string& a : demos_.axes) axes.PushBack(json::Value::Str(a));
    return Object({{"recording", json::Value::Bool(demo_recording_)},
                   {"samples", json::Value::Int(static_cast<int64_t>(worker_.DemoCount()))},
                   {"playing", json::Value::Int(d.playing)},
                   {"left", json::Value::Int(d.left)},
                   {"joints", std::move(joints)},
                   {"axes", std::move(axes)},
                   {"episodes", std::move(episodes)}});
}

std::string EditorHost::State::RunDemoExchange(const char* fn, const std::string& path) {
    if (!script_host_.Valid()) return "demonstration files need the editor's Python";
    const auto literal = [](const std::string& s) { return json::Value::Str(s).Dump(0); };
    const std::string code = std::string("import nuka.editor.demos\nnuka.editor.demos.") + fn + "(" + literal(path) +
                             ", " + literal(cache_.File("demos.json")) + ", " + literal(cache_.File("demos.bin")) + ")\n";
    bool failed = false;
    const std::string text = script_host_.RunString(code, "<demos>", false, &failed);
    if (!failed) return std::string();
    // A traceback ends with the exception.
    const size_t end = text.find_last_not_of("\r\n");
    if (end == std::string::npos) return "nuka.editor.demos failed";
    const size_t nl = text.find_last_of('\n', end);
    const size_t begin = nl == std::string::npos ? 0u : nl + 1u;
    return text.substr(begin, end + 1u - begin);
}

// Drops the scene binding; the world stays until the editor is destroyed.
void EditorHost::State::ApplyUnload() {
    present_->WaitIdle();
    worker_.Unbind();
    DropPublisher();
    shown_ = nullptr;
    bound_epoch_ = 0u;
    reset_framing_ = false;
    policy_ctrl_.reset();
    SyncPolicyView();
    demos_ = DemoSet{};
    demo_recording_ = false;
    worker_.TakeDemo();
    SyncDemoView();
    entity_drag_.Cancel();
    loaded_.reset();
    hidden_meshes_.clear();
    grab_.reset();
    video_.reset();
    ui_state_.held.clear();
    ui_state_.solver.marked = false;
    debug_overlay_.Reset();
    ClearModelUi(ui_state_);
    ++ui_state_.scene_version;
    doc_.Detach();
    script_host_.ClearScripts();
    live_scripts_.clear();
    ui_state_.script_node_count = 0;
    ui_state_.inspector = InspectorState{};
    last_seeded_ = scene::kInvalidEntity;
    xform_gesture_active_ = mat_gesture_active_ = gizmo_using_prev_ = false;
    camera_.FrameAabb(math::Vec3{-1.0f, -1.0f, 0.0f}, math::Vec3{1.0f, 1.0f, 1.0f});
    start_stale_ = true;
}

// Seeds the inspector from the selection's live world pose and material.
void EditorHost::State::SeedInspector(scene::EntityId e) {
    InspectorState& ins = ui_state_.inspector;
    ins = InspectorState{};
    if (!loaded_ || e == scene::kInvalidEntity) return;
    const render::RenderInstance* inst = PoseInstance(*loaded_, doc_.Records(), e);
    if (inst == nullptr) return;
    ins.valid   = true;
    ins.movable = IsMovableInstance(*inst, *loaded_);
    ins.deformable = doc_.Records().media.count(e) != 0u;
    const math::Transform world = LiveWorldTransform(loaded_->scene, e, *inst);
    const math::Vec3 p = world.position;
    ins.pos[0] = p.x; ins.pos[1] = p.y; ins.pos[2] = p.z;
    const math::Vec3 rd = EulerDegFromQuat(world.rotation);
    ins.rot_deg[0] = rd.x; ins.rot_deg[1] = rd.y; ins.rot_deg[2] = rd.z;
    const MaterialBinding mb = ResolveMaterial(*loaded_, doc_.Records(), e);
    if (mb.valid) {
        const scene::RenderMaterial& m = loaded_->sim->GetRenderWorld().materials[mb.render_slot];
        ins.has_material = true;
        for (int k = 0; k < 4; ++k) ins.base_color[k] = m.base_color[k];
        ins.roughness = m.roughness; ins.metallic = m.metallic;
        for (int k = 0; k < 3; ++k) ins.emissive[k] = m.emissive[k];
        ins.sheen = m.sheen; ins.transmission = m.transmission; ins.ior = m.ior;
        for (int k = 0; k < 3; ++k) ins.absorption[k] = m.absorption[k];
    }
}

// Inspector edits preview live each changed frame; the commit frame submits one transaction.
void EditorHost::State::ApplyInspectorEdits() {
    if (!loaded_) return;
    InspectorState& ins = ui_state_.inspector;
    const scene::EntityId e = ui_state_.selected_entity;
    if (!ins.valid || e == scene::kInvalidEntity) return;
    if (ins.transform_changed || ins.transform_commit) {
        math::Transform edited;
        edited.position = math::Vec3{ins.pos[0], ins.pos[1], ins.pos[2]};
        edited.rotation = QuatFromEulerDeg(math::Vec3{ins.rot_deg[0], ins.rot_deg[1], ins.rot_deg[2]});
        if (!xform_gesture_active_) {
            xform_gesture_active_ = true;
            xform_gesture_path_ = PathOfEntity(e);
            if (const render::RenderInstance* gi = PoseInstance(*loaded_, doc_.Records(), e))
                xform_before_ = LiveWorldTransform(loaded_->scene, e, *gi);
        }
        ApplyTransformEdit(*loaded_, doc_.Records(), e, edited, /*commit=*/false);
        if (ins.transform_commit) {
            Queue("edit.apply", TxnToJson(EditTxn{"Transform", {EditOp{xform_gesture_path_, "transform",
                                                                        TransformToJson(xform_before_),
                                                                        TransformToJson(edited)}}}),
                  "ui");
            xform_gesture_active_ = false;
        }
        ins.transform_changed = false;
        ins.transform_commit = false;
    }
    if (ins.has_material && (ins.material_changed || ins.material_commit)) {
        const MaterialBinding mb = ResolveMaterial(*loaded_, doc_.Records(), e);
        scene::RenderMaterial m;
        if (mb.render_slot != render::kNoId && mb.render_slot < loaded_->sim->GetRenderWorld().MaterialCount())
            m = loaded_->sim->GetRenderWorld().materials[mb.render_slot];
        if (!mat_gesture_active_) {
            mat_gesture_active_ = true;
            mat_gesture_path_ = PathOfEntity(e);
            mat_before_ = m;
        }
        for (int k = 0; k < 4; ++k) m.base_color[k] = ins.base_color[k];
        m.opacity = ins.base_color[3];
        m.roughness = ins.roughness; m.metallic = ins.metallic;
        for (int k = 0; k < 3; ++k) m.emissive[k] = ins.emissive[k];
        m.sheen = ins.sheen; m.transmission = ins.transmission; m.ior = ins.ior;
        for (int k = 0; k < 3; ++k) m.absorption[k] = ins.absorption[k];
        ApplyMaterialEdit(*loaded_, mb, m, /*commit=*/false);
        if (ins.material_commit) {
            Queue("edit.apply", TxnToJson(EditTxn{"Material", {EditOp{mat_gesture_path_, "material",
                                                                       MaterialToJson(mat_before_), MaterialToJson(m)}}}),
                  "ui");
            mat_gesture_active_ = false;
        }
        ins.material_changed = false;
        ins.material_commit = false;
    }
}

void EditorHost::State::Frame(bool step_physics) {
    if (!running_) return;
    const auto now = Clock::now();
    ui_thread_ = std::this_thread::get_id();
    if (step_physics) worker_.StartThread();
    else StopWorker();
    DrainConsoleInbox();
    ServeRemote(now);
    ui_state_.script_driven = borrowed_ && !step_physics;
    // Open script files follow the disk: clean tabs reload, and run again when asked to.
    if (now - scripts_polled_ >= std::chrono::seconds(1)) {
        scripts_polled_ = now;
        for (const uint64_t id : ui_state_.scripts.Poll()) {
            const ScriptTab& tab = *ui_state_.scripts.Find(id);
            ui_.Nui().Toast(nui::Level::Info, std::string(nui::Tr("Reloaded")) + " " + tab.title);
            if (ui_state_.script_rerun) RunScript(tab.Text(), tab.Name(), false);
        }
    }
    ui_state_.script_busy_s = script_host_.BusySeconds();

    // Files that later launches hand over or that land on the window open like File > Open.
    if (instance_) {
        for (const std::string& request : instance_->Take()) {
            const json::Value message = json::Value::Parse(request);
            if (const json::Value* path = message.Find("open"))
                Queue("file.open", Object({{"path", *path}}), "instance");
            SDL_Window* window = static_cast<SDL_Window*>(native_window_);
            SDL_RestoreWindow(window);
            SDL_RaiseWindow(window);
        }
    }
    // A dropped scene opens; any other asset joins the document on the ground where it landed.
    for (const Dropped& d : dropped_) {
        if (AssetKindOf(d.path) == AssetKind::Scene) {
            Queue("file.open", Object({{"path", json::Value::Str(d.path)}}), "drop");
            continue;
        }
        json::Value args = Object({{"path", json::Value::Str(d.path)}, {"mode", json::Value::Str("auto")}});
        const SceneViewportRect& rect = ui_.Viewport();
        math::Transform at;
        if (rect.Contains(d.x, d.y) &&
            camera_.RayPlaneHit(camera_.ScreenRay(rect.LocalX(d.x), rect.LocalY(d.y), rect.pixels.width, rect.pixels.height),
                                math::Vec3{0.0f, 0.0f, 0.0f}, math::Vec3{0.0f, 0.0f, 1.0f}, &at.position))
            args.Set("pose", TransformToJson(at));
        Queue("file.import", std::move(args), "drop");
    }
    dropped_.clear();

    if (attach_pending_) {
        attach_pending_ = false;
        worker_.Exclusive([&] { Attach(""); });
    }
    // Headless triggers: undo at frame 1, redo at frame 2, reset at frame N.
    if (opt_.undo && frame_index_ == 1u) Queue("edit.undo", json::Value::Object(), "cli");
    if (opt_.redo && frame_index_ == 2u) Queue("edit.redo", json::Value::Object(), "cli");
    if (opt_.reset_at_frame >= 0 && static_cast<int>(frame_index_) == opt_.reset_at_frame)
        Queue("sim.reset", json::Value::Object(), "cli");
    // Extensions start once the host has installed its hooks, so Python ones can load.
    if (!extensions_started_) {
        extensions_started_ = true;
        extensions_.Start();
    }
    extensions_.Poll();
    ui_.Nui().DispatchEvents();
    // Queued loads and edits land at a tick boundary, before this frame takes a render-world reference.
    RunQueuedActions();

    // The script runs on the physics side at its next tick boundary.
    if (!opt_.run_script_path.empty() && !script_ran_) {
        script_ran_ = true;
        std::string code;
        if (ReadEntireFile(opt_.run_script_path, &code)) run_script_seq_ = worker_.Push(TickCommand::Script(code));
        else Emit(stderr, "[nuka_editor] --run-script: cannot read " + opt_.run_script_path + "\n");
    }

    events_.clear();
    present_->PollEvents(events_);
    ImGuiIO& io = ImGui::GetIO();
    const uint32_t vp_w = present_->Report().width;
    const uint32_t vp_h = present_->Report().height;
    const bool ctrl_before_events = ctrl_down_;
    bool close_asked = false;
    for (const window::WindowEvent& ev : events_) {
        ui_state_.layout_modifier_down = ev.Held(window::modifier::kLayout);
        switch (ev.type) {
            case window::WindowEvent::Type::Close:
                // Closing asks about unsaved changes like File > Exit.
                if (!close_asked) Queue("app.exit", json::Value::Object(), "window");
                close_asked = true;
                break;
            case window::WindowEvent::Type::MouseMove:
                last_mouse_x_ = static_cast<float>(ev.mouse_x);
                last_mouse_y_ = static_cast<float>(ev.mouse_y);
                break;
            case window::WindowEvent::Type::Key:
                ctrl_down_ = ev.Held(window::modifier::kCtrl);
                break;
            case window::WindowEvent::Type::Resize:
                surface_suspended_ = ev.width == 0u || ev.height == 0u;
                break;
            case window::WindowEvent::Type::FocusGained:
                ui_state_.window_focused = true;
                break;
            case window::WindowEvent::Type::FocusLost:
                // Releases go to the newly focused window: drop latched Ctrl and drags.
                ui_state_.window_focused = false;
                ctrl_down_ = false;
                entity_drag_.Cancel();
                break;
            default:
                break;
        }
    }

    const double frame_dt = std::chrono::duration<double>(now - prev_).count();
    prev_ = now;
    if (frame_dt > 0.0) {
        const double inst_fps = 1.0 / frame_dt;
        fps_ema_ = (fps_ema_ == 0.0) ? inst_fps : (fps_ema_ * 0.9 + inst_fps * 0.1);
    }

    // Without a physics thread this update is the tick boundary; then the newest snapshot
    // re-poses the render world.
    if (!worker_.Threaded()) worker_.Pump();
    TakeWorkerResults();
    const auto apply_t0 = Clock::now();
    if (const Snapshot* snap = worker_.AcquireSnapshot()) ApplySnapshot(*snap);
    if (loaded_) SyncSolver();
    if (loaded_) SyncJoints();
    SyncGainTest();
    if (loaded_) SyncAssembler();
    if (loaded_) worker_.ReadTimeline(&ui_state_.timeline);
    double apply_ms = Ms(Clock::now() - apply_t0);

    const render::RenderWorld& render_world = ShownWorld();

    ViewerStats stats;
    const bool live = loaded_ && shown_ != nullptr;
    stats.cpu_timing_available = live;
    if (live) {
        stats.step_time_ms = static_cast<float>(shown_->telemetry.step_ms);
        stats.sub_steps = shown_->telemetry.batch_steps;
        stats.sim_ticks = shown_->tick;
        stats.capped_batches = shown_->telemetry.capped_batches;
        stats.snapshot_age_ms = static_cast<float>(Ms(Clock::now() - shown_->published));
    }
    stats.snapshot_drops = snapshot_drops_;
    stats.zero_copy = scatter_ != nullptr;
    stats.publish_reason = publish_reason_;
    stats.frame_timing_available = frame_dt > 0.0;
    stats.fps = static_cast<float>(fps_ema_);
    if (loaded_) {
        const WorldLayout& layout = loaded_->Layout();
        stats.dof = layout.dofs_per_env;
        stats.links = layout.links_per_env;
        stats.bodies = layout.bodies_per_env;
        stats.contact_cap = layout.contacts_per_env;
    }
    stats.frame_time_ms = static_cast<float>(frame_dt * 1000.0);
    stats.debug_colliders = debug_overlay_.LastColliderCount();
    stats.debug_contacts = debug_overlay_.LastContactCount();
    stats.debug_skipped_shapes = debug_overlay_.LastSkippedShapes();
    stats.debug_capacity = kMaxDebugOverlayInstances;
    const auto& debug_report = debug_overlay_.DrawReport();
    stats.debug_invalid_colliders = debug_report.invalid_colliders;
    stats.debug_invalid_contacts = debug_report.invalid_contacts;
    stats.debug_omitted_instances = debug_report.omitted_instances;
    stats.debug_contacts_budget_skipped = loaded_ && debug_report.contacts_budget_skipped;
    stats.debug_colliders_available = loaded_ && debug_overlay_.CollidersAvailable();
    stats.debug_contacts_available = loaded_ && debug_overlay_.ContactsAvailable();
    stats.frame_index = frame_index_;
    stats.step_healthy = last_step_healthy_;
    stats.device_name = present_->DeviceName();

    ui_state_.has_scene = static_cast<bool>(loaded_);
    ui_state_.loaded_path = loaded_ ? (doc_path_.empty() ? DocTitle() : doc_path_) : std::string();
    ui_state_.can_undo = doc_.CanUndo();
    ui_state_.can_redo = doc_.CanRedo();
    ui_state_.save_dirty = loaded_ && doc_.Dirty();
    ui_state_.doc_title = SessionTitle();
    const std::string title = !loaded_ ? std::string("Nuka Editor")
                                       : ui_state_.doc_title + (ui_state_.save_dirty ? "*" : "") + " \xE2\x80\x94 Nuka Editor";
    if (native_window_ != nullptr && title != window_title_) {
        window_title_ = title;
        SDL_SetWindowTitle(static_cast<SDL_Window*>(native_window_), title.c_str());
    }
    if (native_window_ != nullptr)
        ui_state_.title_bar.maximized = (SDL_GetWindowFlags(static_cast<SDL_Window*>(native_window_)) & SDL_WINDOW_MAXIMIZED) != 0;
    if (ui_state_.save_dirty && doc_.Revision() != autosaved_revision_ && Clock::now() >= autosave_due_) Autosave();
    // The lists also feed File > Open Recent, so they stay current while a scene is open.
    ui_state_.start.shown = StartShown();
    if (ui_state_.start.shown && thumbnails_->Version() != start_thumb_version_) {
        start_thumb_version_ = thumbnails_->Version();
        start_stale_ = true;
    }
    if (start_stale_) {
        RefreshStartPage();
        start_stale_ = false;
    }
    // The primary selection always leads a selection that holds it.
    std::vector<scene::EntityId>& selection = ui_state_.selection;
    if (ui_state_.selected_entity == scene::kInvalidEntity) selection.clear();
    else if (std::find(selection.begin(), selection.end(), ui_state_.selected_entity) == selection.end())
        selection = {ui_state_.selected_entity};
    if (loaded_ && (hidden_stale_ || ui_state_.hidden != applied_hidden_)) ApplyHidden();
    RefreshAssets();
    RefreshProperties();

    io.DisplaySize = ImVec2(static_cast<float>(present_->Report().width),
                            static_cast<float>(present_->Report().height));
    // ImGui runs on wall-clock time, never the physics dt.
    io.DeltaTime = (frame_dt > 0.0) ? static_cast<float>(std::min(frame_dt, kMaxFrameDt)) : (1.0f / 60.0f);
    // SDL3 replaces the display size with logical points and sets the framebuffer scale.
    if (native_window_ != nullptr) ImGui_ImplSDL3_NewFrame();
    // Re-seed on a new selection; otherwise track a movable body's live pose.
    if (loaded_) {
        if (ui_state_.selected_entity != last_seeded_) {
            SeedInspector(ui_state_.selected_entity);
            last_seeded_ = ui_state_.selected_entity;
        } else if (ui_state_.inspector.valid && ui_state_.inspector.movable &&
                   !ui_state_.inspector.transform_changed) {
            if (const render::RenderInstance* inst =
                    PoseInstance(*loaded_, doc_.Records(), ui_state_.selected_entity)) {
                const math::Transform world = LiveWorldTransform(loaded_->scene, ui_state_.selected_entity, *inst);
                const math::Vec3 p = world.position;
                ui_state_.inspector.pos[0] = p.x;
                ui_state_.inspector.pos[1] = p.y;
                ui_state_.inspector.pos[2] = p.z;
                const math::Vec3 rd = EulerDegFromQuat(world.rotation);
                ui_state_.inspector.rot_deg[0] = rd.x;
                ui_state_.inspector.rot_deg[1] = rd.y;
                ui_state_.inspector.rot_deg[2] = rd.z;
            }
        }
    }

    // The monitor's content scale; ImGui applies the pixel density itself.
    SDL_Window* sdl_window = static_cast<SDL_Window*>(native_window_);
    ui_.PrepareFrame(sdl_window != nullptr ? SDL_GetWindowDisplayScale(sdl_window) / SDL_GetWindowPixelDensity(sdl_window)
                                           : 1.0f);
    imgui_.NewFrame();
    SyncViews(render_world);
    ui_.RecordUi(render_world, stats, camera_, ui_state_, loaded_ ? &loaded_->scene : nullptr,
                 surface_suspended_ ? 0u : vp_w, surface_suspended_ ? 0u : vp_h);
    // A press in another pane makes it the active one before anything takes the press.
    for (const window::WindowEvent& ev : events_)
        if (ev.type == window::WindowEvent::Type::MouseButton && ev.pressed && ui_.HoveredPane() >= 0) {
            ActivateView(static_cast<uint32_t>(ui_.HoveredPane()));
            break;
        }
    if (frame_pending_ && ui_.Viewport().Valid()) {
        const auto rect = ui_.Viewport().pixels;
        if (rect.x == last_framing_rect_.x && rect.y == last_framing_rect_.y &&
            rect.width == last_framing_rect_.width && rect.height == last_framing_rect_.height) {
            FrameWorld(render_world);
            frame_pending_ = false;
            if (views_stale_) {
                for (uint32_t i = 0; i < kViewPanes; ++i)
                    if (i != ui_state_.active_view) views_[i].camera = camera_;
                views_stale_ = false;
            }
        }
        last_framing_rect_ = rect;
    }
    extensions_.DrawOverlays(camera_, last_tick_);

    CameraController render_camera = camera_;

    // The gizmo edits the selection's live pose through the inspector's edit seam; one drag
    // (press to release) records one op. On a jointed link with IK on it poses the robot instead.
    if (loaded_ && ui_state_.selected_entity != scene::kInvalidEntity && ui_state_.inspector.valid) {
        if (const render::RenderInstance* gi = PoseInstance(*loaded_, doc_.Records(), ui_state_.selected_entity)) {
            const math::Transform pre_drag = LiveWorldTransform(loaded_->scene, ui_state_.selected_entity, *gi);
            const int ik_link = ik_drag_ ? ui_state_.robot.ik.link : IkLink();
            math::Transform gworld = ik_drag_ ? ik_held_ : pre_drag;
            bool gizmo_changed = false;
            ui_.DrawGizmo(render_camera, vp_w, vp_h, ui_state_, gworld, gizmo_changed);
            const bool using_now = ui_state_.gizmo.using_now;
            if (ik_link >= 0) {
                DragIk(ik_link, pre_drag, gworld, gizmo_changed, using_now);
            } else {
                if (using_now && !gizmo_using_prev_ && !xform_gesture_active_) {
                    xform_gesture_active_ = true;
                    xform_dirty_ = false;
                    xform_gesture_path_ = PathOfEntity(ui_state_.selected_entity);
                    xform_before_ = pre_drag;
                    xform_after_ = pre_drag;
                }
                if (gizmo_changed) {
                    ApplyTransformEdit(*loaded_, doc_.Records(), ui_state_.selected_entity, gworld, /*commit=*/false);
                    xform_after_ = gworld;
                    xform_dirty_ = true;
                }
            }
            if (!using_now && gizmo_using_prev_ && xform_gesture_active_) {
                if (xform_dirty_)
                    Queue("edit.apply", TxnToJson(EditTxn{"Gizmo", {EditOp{xform_gesture_path_, "transform",
                                                                            TransformToJson(xform_before_),
                                                                            TransformToJson(xform_after_)}}}),
                          "ui");
                xform_gesture_active_ = false;
            }
            gizmo_using_prev_ = using_now;
        }
    } else {
        ui_state_.gizmo.active = false;
        ui_state_.gizmo.using_now = false;
        gizmo_using_prev_ = false;
        ik_drag_ = false;
        ui_state_.robot.ik.link = -1;
    }
    SyncIk();

    const auto& scene_rect = ui_.Viewport();
    if (!scene_rect.Valid()) {
        entity_drag_.Cancel();
        camera_.CancelInteraction();
        if (grab_) ReleaseGrab();
    }
    bool scene_ctrl = ctrl_before_events;
    for (const window::WindowEvent& ev : events_) {
        entity_drag_.Observe(ev);
        if (ev.type == window::WindowEvent::Type::Key) scene_ctrl = ev.Held(window::modifier::kCtrl);
        if (ev.type == window::WindowEvent::Type::FocusLost) scene_ctrl = ui_state_.select_box = false;
        const bool scroll = ev.type == window::WindowEvent::Type::Scroll;
        const float mouse_x = scroll ? last_mouse_x_ : static_cast<float>(ev.mouse_x);
        const float mouse_y = scroll ? last_mouse_y_ : static_cast<float>(ev.mouse_y);
        // The wheel over another pane zooms that pane where it is.
        if (scroll && ui_.HoveredPane() >= 0 && static_cast<uint32_t>(ui_.HoveredPane()) != ui_state_.active_view) {
            views_[static_cast<size_t>(ui_.HoveredPane())].camera.HandleEvent(ev, false, true);
            continue;
        }
        const bool over_ui = !ui_.OwnsScenePointer() || !scene_rect.Contains(mouse_x, mouse_y) ||
                             !ui_state_.window_focused || ui_state_.layout_modifier_down || frame_pending_;
        if (loaded_ && scene_ctrl && !over_ui && !ui_state_.gizmo.active) {
            EditorSim& sim = *loaded_->sim;
            if (ev.type == window::WindowEvent::Type::MouseButton && ev.button == 0u) {
                if (ev.pressed) {
                    const Ray ray = render_camera.ScreenRay(scene_rect.LocalX(mouse_x), scene_rect.LocalY(mouse_y),
                                                            scene_rect.pixels.width, scene_rect.pixels.height);
                    const WorldLayout& layout = loaded_->Layout();
                    const uint32_t hit = PickInstance(ray, sim.GetRenderWorld(), [&](const render::RenderInstance& inst) {
                        return IsMovableSource(inst.pose_source, layout) &&
                               (ui_state_.locked.empty() || !UnderAny(PathOfEntity(inst.entity), ui_state_.locked));
                    });
                    entity_drag_.instance = hit;
                    if (hit != ~0u) ui_state_.selected_entity = sim.GetRenderWorld().instances[hit].entity;
                } else {
                    entity_drag_.Cancel();
                }
            } else if (ev.type == window::WindowEvent::Type::MouseMove && entity_drag_.instance != ~0u &&
                       entity_drag_.instance < sim.GetRenderWorld().InstanceCount()) {
                // Drag on a view-facing plane through the entity, as a queued teleport.
                const render::RenderInstance& inst = sim.GetRenderWorld().instances[entity_drag_.instance];
                const math::Vec3 anchor = inst.world_xform.position;
                const Ray ray = render_camera.ScreenRay(scene_rect.LocalX(mouse_x), scene_rect.LocalY(mouse_y),
                                                        scene_rect.pixels.width, scene_rect.pixels.height);
                const math::Vec3 fwd = (render_camera.ResolvedTarget() - render_camera.ResolvedEye()).Normalized();
                math::Vec3 hit;
                if (render_camera.RayPlaneHit(ray, anchor, fwd, &hit)) {
                    math::Transform xf = inst.world_xform;
                    xf.position = hit;
                    TickCommand cmd;
                    cmd.kind = CommandKind::Teleport;
                    if (ResolvePoseWrite(sim.GetRenderWorld(), loaded_->Layout(), inst.entity, xf, &cmd.pose))
                        worker_.Push(std::move(cmd));
                }
            }
        }
        // The Perturb tool's press grabs and its release lets go; between them the pointer pulls.
        if (loaded_ && ui_state_.tool == ViewportTool::Perturb && ev.type == window::WindowEvent::Type::MouseButton &&
            ev.button == 0u) {
            if (ev.pressed && !over_ui && !scene_ctrl) GrabAt(render_camera, mouse_x, mouse_y);
            else if (!ev.pressed && grab_) ReleaseGrab();
        } else if (grab_ && ev.type == window::WindowEvent::Type::MouseMove) {
            DragGrab(render_camera, mouse_x, mouse_y);
        }
        // An Alt-drag with the Select tool boxes; its release selects what the box shows, Shift adding.
        if (loaded_ && ui_state_.tool == ViewportTool::Select && ev.type == window::WindowEvent::Type::MouseButton &&
            ev.button == 0u) {
            if (ev.pressed && ev.Held(window::modifier::kAlt) && !over_ui && !scene_ctrl && !ui_state_.gizmo.active) {
                ui_state_.select_box = true;
                ui_state_.select_box_from[0] = ui_state_.select_box_to[0] = mouse_x;
                ui_state_.select_box_from[1] = ui_state_.select_box_to[1] = mouse_y;
            } else if (!ev.pressed && ui_state_.select_box) {
                ui_state_.select_box = false;
                const render::PickResult pick = PickBox(render_camera, ui_state_.select_box_from[0],
                                                        ui_state_.select_box_from[1], mouse_x, mouse_y);
                Queue("scene.select", BoxSelectArgs(pick, ev.Held(window::modifier::kShift)), "viewport");
            }
        } else if (ui_state_.select_box && ev.type == window::WindowEvent::Type::MouseMove) {
            ui_state_.select_box_to[0] = mouse_x;
            ui_state_.select_box_to[1] = mouse_y;
        }
        // A free pointer over the viewport hovers the body under it with the Select and Measure tools.
        if (ev.type == window::WindowEvent::Type::MouseMove) {
            hover_due_ = loaded_ && (ui_state_.tool == ViewportTool::Select || ui_state_.tool == ViewportTool::Measure) &&
                         !over_ui && !scene_ctrl && !ui_state_.select_box && !ImGui::IsAnyMouseDown();
            hover_x_ = mouse_x;
            hover_y_ = mouse_y;
            if (!hover_due_) hovered_path_.clear();
        }
        // The camera never takes input ImGui, a Ctrl-drag, the gizmo, a grab or a box owns.
        const bool cam_allow = !over_ui && !scene_ctrl && !ui_state_.gizmo.active && !grab_ && !ui_state_.select_box;
        // A left press and release without a drag in between picks, or pins with the Pin tool.
        if (loaded_ && ev.type == window::WindowEvent::Type::MouseButton && ev.button == 0u) {
            if (ev.pressed) {
                click_armed_ = !over_ui && !scene_ctrl && !ui_state_.gizmo.active && !ui_state_.select_box &&
                               ui_state_.tool != ViewportTool::Perturb;
                click_x_ = mouse_x;
                click_y_ = mouse_y;
            } else if (click_armed_) {
                click_armed_ = false;
                if (ui_state_.tool == ViewportTool::Pin) PinAt(render_camera, mouse_x, mouse_y);
                else if (ui_state_.tool == ViewportTool::Measure) MeasureAt(render_camera, mouse_x, mouse_y);
                else PickSelect(render_camera, mouse_x, mouse_y);
            }
        } else if (click_armed_ && ev.type == window::WindowEvent::Type::MouseMove &&
                   std::fabs(mouse_x - click_x_) + std::fabs(mouse_y - click_y_) > kClickSlop) {
            click_armed_ = false;
        }
        camera_.HandleEvent(ev, /*allow_drag=*/cam_allow, /*allow_scroll=*/!over_ui && !ui_state_.gizmo.active);
    }
    if (hover_due_) {
        hover_due_ = false;
        const uint32_t hit = PickBox(render_camera, hover_x_, hover_y_, hover_x_, hover_y_).center;
        const bool here = picked_env_ == ui_state_.env_index % ui_state_.envs.count;
        hovered_path_ = hit == ~0u || !here ? std::string() : CarrierPath(hit);
    }
    ui_state_.robot.hovered = hovered_path_.empty() ? -1 : FindLink(hovered_path_);

    ApplyInspectorEdits();

    // Teleop maps the active device every frame, so a released stick or key stops the robot.
    const bool teleop_owns_wasd = StepTeleop(frame_dt);

    // Shortcuts queue their actions; viewport ones need the pointer over a free viewport.
    const bool viewport_keys = !io.WantCaptureKeyboard && !io.WantTextInput && !io.AppFocusLost &&
                               ui_.OwnsScenePointer() && !ui_state_.gizmo.active;
    for (const std::string& id : actions_.Pressed(!io.WantCaptureKeyboard, viewport_keys))
        Queue(id, json::Value::Object(), "shortcut");

    // The WASD/QE camera fly, unless ImGui owns the keyboard.
    if (!io.WantCaptureKeyboard) {
        const bool shift_held = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);
        float cf = 0.0f, cr = 0.0f, cu = 0.0f;
        if (!teleop_owns_wasd && ui_.OwnsScenePointer() && ui_state_.window_focused &&
            !ui_state_.layout_modifier_down) {
            if (ImGui::IsKeyDown(ImGuiKey_W)) cf += 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_S)) cf -= 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_D)) cr += 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_A)) cr -= 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_E)) cu += 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_Q)) cu -= 1.0f;
        }
        if (cf != 0.0f || cr != 0.0f || cu != 0.0f)
            camera_.Move(cf, cr, cu, static_cast<float>(frame_dt) * (shift_held ? 4.0f : 1.0f));
    }
    ImGui::Render();
    PushTransport();
    PushEdits();

    StepRt(render_camera);
    // The active pane leads: its instance flags outline the selection in every pane.
    std::vector<render::RasterOptions> pane_views{ViewOptions(render_camera)};
    for (const uint32_t i : ShownViews())
        if (i != ui_state_.active_view) pane_views.push_back(ViewOptions(views_[i].camera, ui_.PaneRect(i), views_[i].mode));

    // Debug overlays rebuild from the shown snapshot; off leaves the real instance set untouched.
    const auto overlay_t0 = Clock::now();
    if (loaded_ && shown_ != nullptr)
        debug_overlay_.Rebuild(shown_->overlay, loaded_->sim->GetRenderWorld(), ui_state_.show_colliders,
                               ui_state_.show_contacts);
    apply_ms += Ms(Clock::now() - overlay_t0);

    if (video_) StepVideo();
    StepRender();
    ui_state_.exporting = video_.has_value();
    ui_state_.export_done = !video_ ? 0.0f
                            : video_->encoder != 0u
                                ? -1.0f
                                : static_cast<float>(video_->next) / static_cast<float>(video_->ticks.size());
    const auto draw_t0 = Clock::now();
    DrawWorld(render_world, pane_views);
    const double draw_ms = Ms(Clock::now() - draw_t0);
    if (!running_) return;
    if (thumb_ && present_->SubmittedFrames() > thumb_->frame) {
        // A scene closed since the save, or a hidden viewport, leaves no picture for it.
        if (fs::exists(Utf8Path(thumb_->capture))) {
            if (doc_path_ == thumb_->scene) {
                WriteThumbnail(thumb_->capture, profile_->Thumbnail(thumb_->scene));
                start_stale_ = true;
            }
            fs::remove(Utf8Path(thumb_->capture));
        }
        thumb_.reset();
    }
    if (resume_ && !thumb_) {
        const Prompt resume = *resume_;
        resume_.reset();
        RunAction(resume.action, resume.args, "prompt");
    }
    ++frame_index_;
    playing_mirror_ = ui_state_.playing;

    const auto work_end = Clock::now();
    // The rest of the frame budget serves calls from other threads instead of sleeping.
    ServeRemote(now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(kFrameBudgetSeconds)));
    if (frame_dt > 0.0 && frame_index_ > kFpsWarmup) {
        const double fms = frame_dt * 1000.0;
        ++measured_;
        frame_ms_.Add(fms);
        apply_ms_.Add(apply_ms);
        draw_ms_.Add(draw_ms);
        ui_ms_.Add(Ms(work_end - now) - apply_ms - draw_ms);
        sleep_ms_.Add(Ms(Clock::now() - work_end));
        if (live) {
            age_ms_.Add(stats.snapshot_age_ms);
            step_ms_.Add(stats.step_time_ms);
        }
        if (frame_index_ % 120u == 0u) {
            Emit(stdout, Format("[fps] frame=%llu fps_ema=%.1f frame_ms=%.2f step_ms=%.2f age_ms=%.1f ticks=%llu\n",
                                static_cast<unsigned long long>(frame_index_), fps_ema_, fms,
                                static_cast<double>(stats.step_time_ms), static_cast<double>(stats.snapshot_age_ms),
                                static_cast<unsigned long long>(stats.sim_ticks)));
            std::fflush(stdout);
        }
    }
    // A Close() from a script callback during this frame stays in force; the window's close
    // button runs app.exit.
    running_ = running_ && !(opt_.max_frames > 0 && frame_index_ >= static_cast<uint64_t>(opt_.max_frames));
}

EditorHost::EditorHost(const std::vector<std::string>& args, nuka_world_handle world, float dt)
    : state_(std::make_unique<State>(args, world, dt)) {}

EditorHost::~EditorHost() = default;

bool EditorHost::Forward(const std::vector<std::string>& args) {
    LaunchOptions opt;
    if (!ParseOptions(args, &opt) || opt.scene_path.empty()) return false;
    const EditorProfile profile(opt.profile);
    return SendToPrimary(profile.InstanceName(), Object({{"open", json::Value::Str(NormalPath(opt.scene_path))}}).Dump(0));
}

void EditorHost::SetScriptHooks(const nuka_editor_script_hooks_t& hooks) { state_->SetHooks(hooks); }
void EditorHost::SetPythonHooks(const nuka_editor_python_hooks_t& hooks) { state_->extensions_.SetPythonHooks(hooks); }
void EditorHost::Update(bool step_physics) { state_->Frame(step_physics); }
void EditorHost::StopPhysics() { state_->StopWorker(); }
bool EditorHost::Running() const { return state_->running_; }
bool EditorHost::Playing() const { return state_->playing_mirror_; }
void EditorHost::Close() { state_->running_ = false; }
nuka_world_handle EditorHost::World() const { return state_->world_; }
float EditorHost::Dt() const { return state_->opt_.dt; }
void EditorHost::ConsoleAppend(const std::string& text) {
    std::lock_guard<std::mutex> lock(state_->inbox_mu_);
    state_->console_inbox_ += text;
}
// Arguments arrive as JSON text; text that does not parse fails without running.
std::string EditorHost::RunAction(const std::string& id, const std::string& args_json, const std::string& source,
                                  bool* ok) {
    ActionResult result;
    json::Value args = json::Value::Object();
    try {
        if (!args_json.empty()) args = json::Value::Parse(args_json);
        state_->OnUi([&] { result = state_->RunAction(id, args, source); });
    } catch (const json::ParseError& e) {
        result = ActionResult::Fail(std::string("bad action arguments: ") + e.what());
        state_->last_error_ = result.error;
    } catch (const std::runtime_error& e) {
        result = ActionResult::Fail(e.what());
    }
    *ok = result.ok;
    return Object({{"ok", json::Value::Bool(result.ok)}, {"value", result.value}, {"error", json::Value::Str(result.error)}})
        .Dump(0);
}
// Queries read UI-thread state, so they run there like actions.
std::string EditorHost::Actions() const {
    std::string text;
    state_->OnUi([&] { text = state_->actions_.List().Dump(0); });
    return text;
}
std::string EditorHost::ActionLog(uint64_t since) const {
    std::string text;
    state_->OnUi([&] { text = state_->actions_.LogSince(since).Dump(0); });
    return text;
}
std::string EditorHost::DocumentState() {
    std::string text;
    state_->OnUi([&] { text = state_->DocumentState().Dump(0); });
    return text;
}
const nuka_nui_api_t* EditorHost::Nui() { return state_->ui_.Nui().Api(); }
const nuka_editor_host_api_t* EditorHost::ExtensionHost(const std::string& id) const {
    return state_->extensions_.HostApi(id);
}

}  // namespace nuka::runtime::app::viewer
