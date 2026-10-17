#include "runtime/app/viewer/editor_host.hpp"

#include <vulkan/vulkan.h>

#include "imgui.h"
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
#include "runtime/app/viewer/editor_document.hpp"
#include "runtime/app/viewer/editor_edits.hpp"
#include "runtime/app/viewer/editor_extensions.hpp"
#include "runtime/app/viewer/editor_instance.hpp"
#include "runtime/app/viewer/editor_profile.hpp"
#include "runtime/app/viewer/editor_scene.hpp"
#include "runtime/app/viewer/entity_drag.hpp"
#include "runtime/app/viewer/file_dialog.hpp"
#include "runtime/app/viewer/imgui_layer.hpp"
#include "runtime/app/viewer/physics_worker.hpp"
#include "runtime/app/viewer/pose_scatter.hpp"
#include "runtime/app/viewer/script_host.hpp"
#include "runtime/app/viewer/teleop.hpp"
#include "runtime/inference/go2_policy_controller.hpp"
#include "scene/format/json.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
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

constexpr int kThumbWidth = 320;
constexpr int kThumbHeight = 180;

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

void ClearModelUi(ViewerUiState& ui) {
    ui.drive_targets.clear();
    ui.drive_dirty.clear();
    ui.dof_labels.clear();
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

// The nearest movable instance (free body or floating root link) the ray hits, or ~0u.
uint32_t PickInstance(const Ray& ray, const render::RenderWorld& w, const WorldLayout& layout) {
    uint32_t best = ~0u;
    float best_t = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < w.InstanceCount(); ++i) {
        const render::RenderInstance& inst = w.instances[i];
        if (!IsMovableSource(inst.pose_source, layout)) continue;
        math::Vec3 lo, hi;
        InstanceAabb(inst, w, &lo, &hi);
        if (lo.x > hi.x) continue;
        float t = 0.0f;
        if (RayAabb(ray, lo, hi, &t) && t < best_t) { best_t = t; best = i; }
    }
    return best;
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

ActionResult Ok(json::Value value = json::Value()) {
    ActionResult result;
    result.value = std::move(value);
    return result;
}

ActionResult Outcome(const ApplyResult& r) {
    return r.ok ? Ok(Object({{"focus", json::Value::Str(r.focus)}})) : ActionResult::Fail(r.error);
}

double Ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

uint64_t Fnv1a(const std::string& bytes, uint64_t hash = 14695981039346656037ull) {
    for (const char c : bytes) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ull;
    }
    return hash;
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
    void Reselect(const std::string& path);
    ActionResult SubmitTxn(const EditTxn& txn);
    ActionResult StepHistory(bool redo);
    uint64_t DocumentHash();
    json::Value DocumentState();

    void RegisterActions();
    void RegisterFileActions();
    void RegisterAppActions();
    void RegisterEditActions();
    void RegisterSimActions();
    void RegisterViewActions();
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
    const render::RenderWorld& ShownWorld() const { return loaded_ ? loaded_->sim->GetRenderWorld() : empty_world_; }

    void Frame(bool step_physics);
    // Editor output goes to `stream` and the session log, and with `console` to the Console panel.
    void Emit(std::FILE* stream, const std::string& text, bool console = false);
    void WriteLog(const std::string& text);
    void ConsoleAppend(const std::string& s);
    void DrainConsoleInbox();
    void SetHooks(const nuka_editor_script_hooks_t& hooks);

    void FrameWorld(const render::RenderWorld& rw);
    std::string PathOfEntity(scene::EntityId e) const;
    uint64_t Generation() const;
    void SyncScripts();
    void SeedDriveTargets();
    void BindWorker(uint64_t tick = 0u, double sim_time = 0.0);
    void DropPublisher();
    void ApplySnapshot(const Snapshot& snap);
    void TakeWorkerResults();
    void PushTransport();
    void PushEdits();
    void DrawWorld(const render::RenderWorld& rw, const render::RasterOptions& opts);
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
    std::vector<std::string> dropped_;  // files dropped on the window since the last frame
    std::string start_locale_;          // persisted on exit once the user switched languages
    bool start_stale_ = true;           // the start page lists need a refresh
    // A save's viewport picture, captured without the UI from the next drawn frame.
    struct ThumbRequest {
        std::string scene;
        std::string capture;
        uint64_t frame = 0u;
    };
    std::optional<ThumbRequest> thumb_;
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
    bool sent_colliders_ = false;
    bool sent_contacts_ = false;
    float sent_speed_ = 1.0f;
    uint64_t sent_static_revision_ = 0u;
    uint64_t transport_seq_ = 0u;
    uint64_t reset_seq_ = 0u;
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
    bool teleop_hold_prev_ = false;
    bool teleop_prev_enabled_ = false;
    float last_mouse_x_ = 0.0f;
    float last_mouse_y_ = 0.0f;
    bool script_ran_ = false;
    std::vector<window::WindowEvent> events_;
    std::atomic<bool> running_{true};
    std::atomic<bool> playing_mirror_{false};  // read by script callbacks off the UI thread
    std::mutex inbox_mu_;
    std::string console_inbox_;                // host console text, appended from any thread
};

EditorHost::State::State(const std::vector<std::string>& args, nuka_world_handle world, float dt)
    : world_(world), borrowed_(world != nullptr) {
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
    if (opt_.locale.empty()) opt_.locale = profile_->Setting("ui", "locale", "");
    autosave_seconds_ = profile_->SettingNumber("editor", "autosave_seconds", kAutosaveSeconds);
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
        if (e->type == SDL_EVENT_DROP_FILE && e->drop.data != nullptr) dropped_.emplace_back(e->drop.data);
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
    if (native_window_ != nullptr) ImGui_ImplSDL3_InitForVulkan(static_cast<SDL_Window*>(native_window_));

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
    if (profile_ && present_) {
        DropAutosave();
        if (nui::Locale() != start_locale_) profile_->SetSetting("ui", "locale", nui::Locale());
    }
    extensions_.Shutdown();
    worker_.StopThread();
    if (present_) {
        present_->WaitIdle();
        if (native_window_ != nullptr) ImGui_ImplSDL3_Shutdown();
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

// The Console panel's text also lands in the session log.
void EditorHost::State::ConsoleAppend(const std::string& s) {
    if (s.empty()) return;
    WriteLog(s);
    ui_state_.console_log += s;
    if (ui_state_.console_log.size() > kConsoleLogCap)
        ui_state_.console_log.erase(0, ui_state_.console_log.size() - kConsoleLogCap);
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
        return RunAction(id, args, source);
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
    live_scripts_.clear();
    SyncScripts();
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

// The per-DOF drive editor starts from environment 0's live targets.
void EditorHost::State::SeedDriveTargets() {
    ui_state_.drive_targets.clear();
    ui_state_.drive_dirty.clear();
    ui_state_.dof_labels.clear();
    const uint32_t links = loaded_->Layout().links_per_env;
    if (links == 0u) return;
    ui_state_.drive_targets.assign(links, 0.0f);
    ui_state_.drive_dirty.assign(links, 0u);
    nuka_world_download_field(world_, NUKA_FIELD_DRIVE_TARGET, ui_state_.drive_targets.data(),
                              links * sizeof(float), 0u);
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
    binding.controller = policy_ctrl_.get();
    binding.scatter = scatter_.get();
    binding.env = ui_state_.env_index % layout.env_count;
    binding.dt = opt_.dt;
    binding.speed = ui_state_.speed;
    binding.colliders = ui_state_.show_colliders;
    binding.contacts = ui_state_.show_contacts;
    binding.tick = tick;
    binding.sim_time = sim_time;
    sent_playing_ = false;
    sent_env_ = binding.env;
    sent_speed_ = binding.speed;
    sent_colliders_ = binding.colliders;
    sent_contacts_ = binding.contacts;
    sent_static_revision_ = loaded_->sim->StaticRevision();
    bound_epoch_ = worker_.Bind(std::move(binding));
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
    if (shown_ == nullptr) return;
    loaded_->sim->ApplyPoses(snap.link_poses, snap.body_poses);
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
        ConsoleAppend(r.text);
    }
}

// Transport and display changes become worker commands.
void EditorHost::State::PushTransport() {
    if (!loaded_) return;
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
    if (env != sent_env_) {
        TickCommand cmd = TickCommand::Param(ParamKind::DisplayEnv, {});
        cmd.env = env;
        worker_.Push(std::move(cmd));
        sent_env_ = env;
    }
    if (ui_state_.show_colliders != sent_colliders_ || ui_state_.show_contacts != sent_contacts_) {
        worker_.Push(TickCommand::Param(ParamKind::Overlay, {ui_state_.show_colliders ? 1.0f : 0.0f,
                                                             ui_state_.show_contacts ? 1.0f : 0.0f}));
        sent_colliders_ = ui_state_.show_colliders;
        sent_contacts_ = ui_state_.show_contacts;
    }
}

// Live edits of this frame become worker commands: teleports, static rows, drive targets.
void EditorHost::State::PushEdits() {
    if (!loaded_) return;
    for (const PoseWrite& write : loaded_->sim->TakePoseWrites()) {
        TickCommand cmd;
        cmd.kind = CommandKind::EditApply;
        cmd.pose = write;
        worker_.Push(std::move(cmd));
    }
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
void EditorHost::State::DrawWorld(const render::RenderWorld& rw, const render::RasterOptions& opts) {
    render::PresentInterop interop;
    uint32_t slot = 0u;
    const bool pinned = scatter_ != nullptr && worker_.Slots().AcquireRead(&slot, &interop.ready_value);
    if (pinned) {
        interop.set = reinterpret_cast<render::NkVkDescriptorSet>(interop_->Slot(slot));
        interop.timeline = reinterpret_cast<render::NkVkSemaphore>(interop_->Timeline());
    }
    const uint64_t submitted = present_->SubmittedFrames();
    const render::PresentFrameResult r = present_->DrawFrame(rw, opts, [this](void* cmd) {
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
    ui_state_.inspector = InspectorState{};
    last_seeded_ = scene::kInvalidEntity;
    xform_gesture_active_ = mat_gesture_active_ = gizmo_using_prev_ = false;
    std::snprintf(ui_state_.save_path, sizeof(ui_state_.save_path), "%s", path.c_str());
    ui_state_.selected_entity = scene::kInvalidEntity;

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
    if (opt_.teleop_nudge_on && !cmd_ctrl && opt_.teleop_nudge_dof >= 0 &&
        static_cast<size_t>(opt_.teleop_nudge_dof) < ui_state_.drive_targets.size()) {
        ApplyHoldGains(world_, loaded_->Layout(), ui_state_.teleop_kp, ui_state_.teleop_kd);
        ui_state_.teleop_enabled = true;
        ui_state_.teleop_hold = true;
        ui_state_.teleop_dof = opt_.teleop_nudge_dof;
        const size_t dof = static_cast<size_t>(opt_.teleop_nudge_dof);
        ui_state_.drive_targets[dof] += opt_.teleop_nudge_delta;
        ui_state_.drive_dirty[dof] = 1u;
        Emit(stdout, Format("[nuka_editor] teleop nudge: dof %d += %g (PD-hold kp=%g kd=%g)\n", opt_.teleop_nudge_dof,
                            static_cast<double>(opt_.teleop_nudge_delta), static_cast<double>(ui_state_.teleop_kp),
                            static_cast<double>(ui_state_.teleop_kd)));
    }
    if (opt_.teleop_cmd_on && cmd_ctrl) {
        // The CLI command is the baseline the every-frame producer re-sends.
        policy_ctrl_->SetCommandVector(opt_.teleop_cmd, 3);
        ui_state_.teleop_enabled = true;
        for (int k = 0; k < 3; ++k) ui_state_.teleop_cmd_base[k] = opt_.teleop_cmd[k];
        Emit(stdout, Format("[nuka_editor] teleop cmd: vx=%g vy=%g wyaw=%g\n", static_cast<double>(opt_.teleop_cmd[0]),
                            static_cast<double>(opt_.teleop_cmd[1]), static_cast<double>(opt_.teleop_cmd[2])));
    }
}

// A rebuild replaced the simulation binding: re-attach the controller and drive editor.
void EditorHost::State::OnRebound(uint64_t tick, double sim_time) {
    AttachController();
    SeedDriveTargets();
    BindWorker(tick, sim_time);
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

// Runs with the worker parked. Selection follows the node the edit created or moved.
ApplyResult EditorHost::State::ApplyTxn(const EditTxn& txn) {
    const std::string selected = SelectedPath();
    const ApplyResult result = doc_.Apply(txn);
    Reselect(result.focus.empty() ? selected : result.focus);
    return result;
}

void EditorHost::State::Reselect(const std::string& path) {
    ++ui_state_.scene_version;
    ui_state_.selected_entity = scene::kInvalidEntity;
    if (!path.empty())
        if (const auto n = loaded_->scene.Tree().NodeOf(path)) ui_state_.selected_entity = n->entity;
    ui_state_.inspector = InspectorState{};
    last_seeded_ = scene::kInvalidEntity;
}

ActionResult EditorHost::State::SubmitTxn(const EditTxn& txn) {
    ApplyResult result;
    worker_.Exclusive([&] { result = ApplyTxn(txn); });
    return Outcome(result);
}

// Selection stays put unless the restored step names a node to follow.
ActionResult EditorHost::State::StepHistory(bool redo) {
    const std::string selected = SelectedPath();
    ApplyResult result;
    worker_.Exclusive([&] { result = redo ? doc_.Redo() : doc_.Undo(); });
    entity_drag_.Cancel();
    xform_gesture_active_ = mat_gesture_active_ = gizmo_using_prev_ = false;
    Reselect(result.focus.empty() ? selected : result.focus);
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
    start_stale_ = true;
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
    start_stale_ = true;
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
        page.examples.push_back(StartEntry{e.At("title").AsString(), e.At("detail").AsString(), path, "",
                                           fs::exists(Utf8Path(path))});
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
    RegisterSimActions();
    RegisterViewActions();
    ui_.RegisterActions(actions_);
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
    actions_.Add({"edit.delete", "Delete Subtree", "edit", "", ShortcutScope::Global, has_scene, {},
                  [node_edit](const json::Value& args) { return node_edit("Delete", "delete", args, json::Value()); }});
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
    actions_.Add({"scene.select", "Select", "scene", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const std::string path = StrArg(args, "path");
                      const scene::EntityId e = path.empty() ? scene::kInvalidEntity : EntityOfPath(*loaded_, path);
                      if (!path.empty() && e == scene::kInvalidEntity) return ActionResult::Fail("no node at '" + path + "'");
                      ui_state_.selected_entity = e;
                      return Ok(Object({{"path", json::Value::Str(path)}}));
                  }});
    actions_.Add({"script.run", "Run Script", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      worker_.Push(TickCommand::Script(StrArg(args, "code", ui_state_.script_buf)));
                      return Ok();
                  }});
    actions_.Add({"script.clear_console", "Clear Console", "script", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value&) {
                      ui_state_.console_log.clear();
                      return Ok();
                  }});
    // Attach persists the source as a /script node; a selected one updates in place.
    actions_.Add({"script.attach", "Attach as Node", "script", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value& args) {
                      const scene::EntityId sel = ui_state_.selected_entity;
                      std::string target;
                      for (size_t i = 0; i < loaded_->scene.ScriptCount(); ++i)
                          if (sel != scene::kInvalidEntity &&
                              loaded_->scene.EntityOfScript(static_cast<scene::ScriptId>(i)) == sel)
                              target = SelectedPath();
                      const json::Value source = json::Value::Str(StrArg(args, "source", ui_state_.script_buf));
                      return SubmitTxn(EditTxn{"Attach script", {EditOp{target, "script", json::Value(), source}}});
                  }});
    // An oversize source is refused, never truncated into the buffer.
    actions_.Add({"script.load", "Load Node", "script", "", ShortcutScope::Global, has_scene, {},
                  [this](const json::Value&) {
                      const auto* sc = ui_state_.selected_entity == scene::kInvalidEntity
                                           ? nullptr
                                           : loaded_->scene.Ecs().Get<scene::ScriptComponent>(ui_state_.selected_entity);
                      if (sc == nullptr) return ActionResult::Fail("the selection is not a /script node");
                      if (sc->source.size() >= sizeof(ui_state_.script_buf))
                          return ActionResult::Fail("script source (" + std::to_string(sc->source.size()) +
                                                    " bytes) exceeds the editor buffer");
                      std::memcpy(ui_state_.script_buf, sc->source.c_str(), sc->source.size() + 1u);
                      return Ok();
                  }});
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
                      ui_state_.playing = sent_playing_ = false;
                      return Ok();
                  }});
    actions_.Add({"sim.speed", "Speed", "sim", "", ShortcutScope::Global, {}, {},
                  [this](const json::Value& args) {
                      ui_state_.speed = static_cast<float>(NumArg(args, "value", 1.0));
                      PushTransport();
                      return Ok(Object({{"speed", json::Value::Float(ui_state_.speed)}}));
                  }});
}

void EditorHost::State::RegisterViewActions() {
    actions_.Add({"view.frame_all", "Frame All", "view", "Home", ShortcutScope::Viewport, {}, {},
                  [this](const json::Value&) {
                      return camera_.FrameAll(ShownWorld(), ui_.Viewport().Aspect()) ? Ok()
                                                                                     : ActionResult::Fail("nothing to frame");
                  }});
    actions_.Add({"view.frame_selected", "Frame Selected", "view", "F", ShortcutScope::Viewport, {}, {},
                  [this](const json::Value&) {
                      return camera_.FrameSelected(ShownWorld(), ui_state_.selected_entity, ui_.Viewport().Aspect())
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
                                        {"playing", json::Value::Bool(ui_state_.playing)}}));
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

void EditorHost::State::AttachController() {
    policy_ctrl_.reset();
    const WorldLayout& layout = loaded_->Layout();
    if (!opt_.policy_path.empty() && terrain::IsValid(loaded_->terrain)) {
        auto pc = std::make_unique<inference::Go2PolicyController>();
        if (pc->Init(opt_.policy_path, layout.links_per_env, layout.articulations_per_env, loaded_->terrain)) {
            if (opt_.force_host_policy) pc->SetUseGpu(false);
            policy_ctrl_ = std::move(pc);
            Emit(stdout, Format("[nuka_editor] policy attached: %s  play=%s  path=%s\n", opt_.policy_path.c_str(),
                                opt_.want_play ? "yes" : "no", policy_ctrl_->UsingGpu() ? "GPU" : "host"));
        } else {
            Emit(stderr, "[nuka_editor] policy Init failed: " + opt_.policy_path + "\n");
        }
    }
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
    entity_drag_.Cancel();
    loaded_.reset();
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

// Seeds the inspector from the selected instance's world pose and material.
void EditorHost::State::SeedInspector(scene::EntityId e) {
    InspectorState& ins = ui_state_.inspector;
    ins = InspectorState{};
    if (!loaded_ || e == scene::kInvalidEntity) return;
    const render::RenderInstance* inst = InstanceOfEntity(loaded_->sim->GetRenderWorld(), e);
    if (inst == nullptr) return;
    ins.valid   = true;
    ins.movable = IsMovableInstance(*inst, *loaded_);
    const math::Vec3 p = inst->world_xform.position;
    ins.pos[0] = p.x; ins.pos[1] = p.y; ins.pos[2] = p.z;
    const math::Vec3 rd = EulerDegFromQuat(inst->world_xform.rotation);
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
            if (const render::RenderInstance* gi = InstanceOfEntity(loaded_->sim->GetRenderWorld(), e))
                xform_before_ = gi->world_xform;
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
    if (step_physics) worker_.StartThread();
    else worker_.StopThread();
    DrainConsoleInbox();
    ui_state_.script_driven = borrowed_ && !step_physics;

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
    for (const std::string& path : dropped_) Queue("file.open", Object({{"path", json::Value::Str(path)}}), "drop");
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
    if (ui_state_.save_dirty && doc_.Revision() != autosaved_revision_ && Clock::now() >= autosave_due_) Autosave();
    // The lists also feed File > Open Recent, so they stay current while a scene is open.
    ui_state_.start.shown = StartShown();
    if (start_stale_) {
        RefreshStartPage();
        start_stale_ = false;
    }

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
                    InstanceOfEntity(loaded_->sim->GetRenderWorld(), ui_state_.selected_entity)) {
                const math::Vec3 p = inst->world_xform.position;
                ui_state_.inspector.pos[0] = p.x;
                ui_state_.inspector.pos[1] = p.y;
                ui_state_.inspector.pos[2] = p.z;
                const math::Vec3 rd = EulerDegFromQuat(inst->world_xform.rotation);
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
    ui_.RecordUi(render_world, stats, camera_, ui_state_, loaded_ ? &loaded_->scene : nullptr,
                 surface_suspended_ ? 0u : vp_w, surface_suspended_ ? 0u : vp_h);
    if (frame_pending_ && ui_.Viewport().Valid()) {
        const auto rect = ui_.Viewport().pixels;
        if (rect.x == last_framing_rect_.x && rect.y == last_framing_rect_.y &&
            rect.width == last_framing_rect_.width && rect.height == last_framing_rect_.height) {
            FrameWorld(render_world);
            frame_pending_ = false;
        }
        last_framing_rect_ = rect;
    }
    extensions_.DrawOverlays(camera_, last_tick_);

    CameraController render_camera = camera_;

    // The gizmo edits the selection's live pose through the inspector's edit seam;
    // one drag (press to release) records one op.
    if (loaded_ && ui_state_.selected_entity != scene::kInvalidEntity && ui_state_.inspector.valid) {
        if (const render::RenderInstance* gi =
                InstanceOfEntity(loaded_->sim->GetRenderWorld(), ui_state_.selected_entity)) {
            const math::Transform pre_drag = gi->world_xform;
            math::Transform gworld = pre_drag;
            bool gizmo_changed = false;
            ui_.DrawGizmo(render_camera, vp_w, vp_h, ui_state_, gworld, gizmo_changed);
            const bool using_now = ui_state_.gizmo.using_now;
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
    }

    const auto& scene_rect = ui_.Viewport();
    if (!scene_rect.Valid()) {
        entity_drag_.Cancel();
        camera_.CancelInteraction();
    }
    bool scene_ctrl = ctrl_before_events;
    for (const window::WindowEvent& ev : events_) {
        entity_drag_.Observe(ev);
        if (ev.type == window::WindowEvent::Type::Key) scene_ctrl = ev.Held(window::modifier::kCtrl);
        if (ev.type == window::WindowEvent::Type::FocusLost) scene_ctrl = false;
        const bool scroll = ev.type == window::WindowEvent::Type::Scroll;
        const float mouse_x = scroll ? last_mouse_x_ : static_cast<float>(ev.mouse_x);
        const float mouse_y = scroll ? last_mouse_y_ : static_cast<float>(ev.mouse_y);
        const bool over_ui = !ui_.OwnsScenePointer() || !scene_rect.Contains(mouse_x, mouse_y) ||
                             !ui_state_.window_focused || ui_state_.layout_modifier_down || frame_pending_;
        if (loaded_ && scene_ctrl && !over_ui && !ui_state_.gizmo.active) {
            EditorSim& sim = *loaded_->sim;
            if (ev.type == window::WindowEvent::Type::MouseButton && ev.button == 0u) {
                if (ev.pressed) {
                    const Ray ray = render_camera.ScreenRay(scene_rect.LocalX(mouse_x), scene_rect.LocalY(mouse_y),
                                                            scene_rect.pixels.width, scene_rect.pixels.height);
                    const uint32_t hit = PickInstance(ray, sim.GetRenderWorld(), loaded_->Layout());
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
        // The camera never takes input ImGui, a Ctrl-drag or the gizmo owns.
        const bool cam_allow = !over_ui && !scene_ctrl && !ui_state_.gizmo.active;
        camera_.HandleEvent(ev, /*allow_drag=*/cam_allow, /*allow_scroll=*/!over_ui && !ui_state_.gizmo.active);
    }

    ApplyInspectorEdits();

    // Teleop sends its command every frame; gating zeroes the keys, so a walking
    // robot stops instead of latching.
    bool teleop_owns_wasd = false;
    {
        const bool cmd_ctrl = loaded_ && policy_ctrl_ && policy_ctrl_->AcceptsCommand();
        // Ctrl chords belong to shortcuts (Ctrl+S, Ctrl+=), so they never drive the robot.
        const bool active = loaded_ && ui_state_.teleop_enabled && !io.WantCaptureKeyboard && !io.KeyCtrl &&
                            !io.WantTextInput && ui_.OwnsScenePointer() && ui_state_.window_focused;
        TeleopKeys tk;
        if (active) {
            tk.dof_minus = ImGui::IsKeyPressed(ImGuiKey_LeftBracket, false) || ImGui::IsKeyPressed(ImGuiKey_Comma, false);
            tk.dof_plus  = ImGui::IsKeyPressed(ImGuiKey_RightBracket, false) || ImGui::IsKeyPressed(ImGuiKey_Period, false);
            tk.dof_prev  = ImGui::IsKeyPressed(ImGuiKey_Minus, false);
            tk.dof_next  = ImGui::IsKeyPressed(ImGuiKey_Equal, false);
            tk.fwd = ImGui::IsKeyDown(ImGuiKey_W); tk.back = ImGui::IsKeyDown(ImGuiKey_S);
            tk.strafe_l = ImGui::IsKeyDown(ImGuiKey_A); tk.strafe_r = ImGui::IsKeyDown(ImGuiKey_D);
            tk.yaw_l = ImGui::IsKeyDown(ImGuiKey_Q); tk.yaw_r = ImGui::IsKeyDown(ImGuiKey_E);
        }
        // PD-hold rising edge makes drive targets effective for direct control.
        const bool hold_now = active && ui_state_.teleop_hold && !cmd_ctrl;
        if (hold_now && !teleop_hold_prev_)
            worker_.Push(TickCommand::Param(ParamKind::HoldGains, {ui_state_.teleop_kp, ui_state_.teleop_kd}));
        teleop_hold_prev_ = hold_now;

        const uint32_t dof_count = static_cast<uint32_t>(ui_state_.drive_targets.size());
        const TeleopResult tr = ApplyTeleop(active, tk, ui_state_, dof_count);
        if (cmd_ctrl) {
            if (ui_state_.teleop_enabled) {
                worker_.Push(TickCommand::Param(ParamKind::ControllerCommand,
                                                {tr.command[0], tr.command[1], tr.command[2]}));
            } else if (teleop_prev_enabled_) {
                // Toggle-off edge: one zero stops the robot, then scripts may own the command.
                worker_.Push(TickCommand::Param(ParamKind::ControllerCommand, {0.0f, 0.0f, 0.0f}));
            }
            teleop_owns_wasd = active;
        }
        teleop_prev_enabled_ = ui_state_.teleop_enabled;
    }

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

    render::RasterOptions opts;
    opts.width = present_->Report().width;
    opts.height = present_->Report().height;
    render_camera.WriteOptions(opts);
    opts.scene_viewport = ui_.Viewport().pixels;
    if (loaded_ && loaded_->scene.Environment().sky.enabled) {
        const auto background = loaded_->scene.Environment().sky.background;
        const auto byte = [](float value) {
            return static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
        };
        if (std::isfinite(background.x) && std::isfinite(background.y) && std::isfinite(background.z))
            opts.background = {byte(background.x), byte(background.y), byte(background.z), 255u};
    }

    // Debug overlays rebuild from the shown snapshot; off leaves the real instance set untouched.
    const auto overlay_t0 = Clock::now();
    if (loaded_ && shown_ != nullptr)
        debug_overlay_.Rebuild(shown_->overlay, loaded_->sim->GetRenderWorld(), ui_state_.show_colliders,
                               ui_state_.show_contacts);
    apply_ms += Ms(Clock::now() - overlay_t0);

    const auto draw_t0 = Clock::now();
    DrawWorld(render_world, opts);
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
    const auto spent = std::chrono::duration<double>(work_end - now);
    const auto budget = std::chrono::duration<double>(kFrameBudgetSeconds);
    if (spent < budget)
        std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::microseconds>(budget - spent));
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
void EditorHost::StopPhysics() { state_->worker_.StopThread(); }
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
        result = state_->RunAction(id, args, source);
    } catch (const json::ParseError& e) {
        result = ActionResult::Fail(std::string("bad action arguments: ") + e.what());
        state_->last_error_ = result.error;
    }
    *ok = result.ok;
    return Object({{"ok", json::Value::Bool(result.ok)}, {"value", result.value}, {"error", json::Value::Str(result.error)}})
        .Dump(0);
}
std::string EditorHost::Actions() const { return state_->actions_.List().Dump(0); }
std::string EditorHost::ActionLog(uint64_t since) const { return state_->actions_.LogSince(since).Dump(0); }
std::string EditorHost::DocumentState() { return state_->DocumentState().Dump(0); }
const nuka_nui_api_t* EditorHost::Nui() { return state_->ui_.Nui().Api(); }
const nuka_editor_host_api_t* EditorHost::ExtensionHost(const std::string& id) const {
    return state_->extensions_.HostApi(id);
}

}  // namespace nuka::runtime::app::viewer
