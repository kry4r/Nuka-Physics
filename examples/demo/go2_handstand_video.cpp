// ---------------------------------------------------------------------------
// go2_walk_video.cpp -- M10 Phase A: THE SECOND HEADLINE DEMO.
//
// A unified, offscreen C++ video tool that POSE-REPLAYS the PROVEN, golden-
// validated Go2 quadruped WALK (the policy rolls +3.91 m forward, deterministic/
// D1) to a PBR-beautiful mp4 on lavapipe-CPU, with the M8.5 VISUAL meshes (16
// MESH chunks -> 33 visual_mesh instances, PBR materials) -- NOT a collision-
// primitive "stick figure". No physics, no policy, no torch in C++: the walk was
// already simulated + dumped by examples/demo/go2_dump_walk_trajectory.py.
//
// PIPELINE:
//   nks::Load(examples/scenes/go2.nks)          -- FROZEN beauty scene (read-only)
//     -> CookToModel(scene, 1)                  -- host-only; gives the SceneMap
//        (EntityId -> cooked link_index) so each visual instance knows which
//        articulation LINK row drives it
//     -> BuildRenderWorld(Ecs, scene_map)        -- 33 visual-mesh RenderInstances,
//        each with pose_source.kind == Link, pose_source.row == cook link_index,
//        and a cached_visual_local (geom-local in its link frame).
//   PER FRAME (no device, no nk::World): read the dumped link_pose[row] for this
//   instance's link, compose world_xform = link_pose[row] o cached_visual_local
//   (quat w-first), follow-cam tracks the base, render offscreen (PBR + studio
//   ground at z=0 + hero lighting), WritePpmP6 frame_%06d.ppm.
//
// The walk KINEMATICS are byte-for-byte the dumped trajectory (out/
// go2_walk_trajectory.bin, header GO2W v1; see go2_dump_walk_trajectory.py for
// the layout). The visual triangles come from the FROZEN go2.nks/.nka (M8.5
// cook). NOTHING in examples/scenes/ or out/ is edited.
//
// Built behind NK_BUILD_VULKAN_VALIDATION (build-viewer / lavapipe).
//
// USAGE:
//   go2_walk_video [--frames N] [--stride S] [--out-dir DIR] [--width W]
//                  [--height H] [--bin PATH] [--probe]
//     --stride S : render every S-th dumped sub-step (default 1 -> all 1600).
//     --frames N : cap the number of rendered frames (default = all available).
//     --probe    : load+cook+bind, print the instance->link map + a few sampled
//                  base poses, render 1 mid-walk frame, exit.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"

namespace {

namespace render = nuka::render;
namespace cook = nuka::scene::cook;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr const char* kNksPath = "examples/scenes/go2.nks";
constexpr const char* kDefaultBin = "out/go2_handstand_trajectory.bin";
constexpr uint32_t kGo2WMagic = 0x474F3257u;  // 'GO2W'

// ---- CLI ------------------------------------------------------------------
struct Args {
    uint32_t frames = 0u;       // 0 => all available (after stride)
    uint32_t stride = 1u;       // render every stride-th dumped sub-step
    std::string out_dir = "/tmp/go2_walk_frames";
    std::string bin = kDefaultBin;
    uint32_t width = 1920u;
    uint32_t height = 1080u;
    bool probe = false;
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next_u = [&](uint32_t def) -> uint32_t {
            return (i + 1 < argc) ? static_cast<uint32_t>(std::atoi(argv[++i])) : def;
        };
        if (s == "--frames") a.frames = next_u(a.frames);
        else if (s == "--stride") a.stride = std::max(1u, next_u(a.stride));
        else if (s == "--width") a.width = next_u(a.width);
        else if (s == "--height") a.height = next_u(a.height);
        else if (s == "--probe") a.probe = true;
        else if (s == "--out-dir" && i + 1 < argc) a.out_dir = argv[++i];
        else if (s == "--bin" && i + 1 < argc) a.bin = argv[++i];
    }
    return a;
}

// ---- the dumped GO2W trajectory (header + per-sub-step records) ------------
// header = 8 int32 [magic, version, num_steps, dof_count, link_count,
//          pose_floats, decimation, control_hz_x1000] then 2 float32 [dt,ctrl_dt]
// then num_steps records of [drive_target(dof_count) f32 , link_pose(link_count*7) f32]
struct Trajectory {
    uint32_t num_steps = 0u;
    uint32_t dof_count = 0u;
    uint32_t link_count = 0u;
    uint32_t pose_floats = 0u;
    float dt = 0.0f;
    float ctrl_dt = 0.0f;
    // pose[step] is a flat (link_count * 7) array: [px,py,pz,qw,qx,qy,qz] per link.
    std::vector<std::vector<float>> pose;

    // World transform of link `l` at recorded sub-step `step` (quat w-first).
    Transform LinkPose(uint32_t step, uint32_t l) const {
        const float* p = &pose[step][static_cast<size_t>(l) * pose_floats];
        return Transform{Vec3{p[0], p[1], p[2]},
                         Quat{p[3], p[4], p[5], p[6]}};
    }
};

bool LoadTrajectory(const std::string& path, Trajectory* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "[go2_walk_video] cannot open %s\n", path.c_str()); return false; }
    int32_t hdr_i[8];
    float hdr_f[2];
    f.read(reinterpret_cast<char*>(hdr_i), sizeof(hdr_i));
    f.read(reinterpret_cast<char*>(hdr_f), sizeof(hdr_f));
    if (!f) { std::fprintf(stderr, "[go2_walk_video] short header in %s\n", path.c_str()); return false; }
    if (static_cast<uint32_t>(hdr_i[0]) != kGo2WMagic) {
        std::fprintf(stderr, "[go2_walk_video] bad magic 0x%08X (want 0x%08X) in %s\n",
                     static_cast<uint32_t>(hdr_i[0]), kGo2WMagic, path.c_str());
        return false;
    }
    out->num_steps  = static_cast<uint32_t>(hdr_i[2]);
    out->dof_count  = static_cast<uint32_t>(hdr_i[3]);
    out->link_count = static_cast<uint32_t>(hdr_i[4]);
    out->pose_floats = static_cast<uint32_t>(hdr_i[5]);
    out->dt = hdr_f[0];
    out->ctrl_dt = hdr_f[1];
    const size_t pose_n = static_cast<size_t>(out->link_count) * out->pose_floats;
    const size_t rec_floats = static_cast<size_t>(out->dof_count) + pose_n;
    out->pose.resize(out->num_steps);
    std::vector<float> rec(rec_floats);
    for (uint32_t s = 0; s < out->num_steps; ++s) {
        f.read(reinterpret_cast<char*>(rec.data()),
               static_cast<std::streamsize>(rec_floats * sizeof(float)));
        if (!f) {
            std::fprintf(stderr, "[go2_walk_video] short record %u/%u in %s\n",
                         s, out->num_steps, path.c_str());
            return false;
        }
        // keep only the link_pose slice (skip the leading drive_target[dof_count]).
        out->pose[s].assign(rec.begin() + out->dof_count, rec.end());
    }
    return true;
}

bool WritePpm(const render::VulkanOffscreenReport& rep, const std::string& path) {
    if (rep.pixels.empty() || rep.width == 0u || rep.height == 0u) return false;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    char hdr[64];
    const int hn = std::snprintf(hdr, sizeof(hdr), "P6\n%u %u\n255\n", rep.width, rep.height);
    bool ok = (hn > 0) &&
              (std::fwrite(hdr, 1, static_cast<size_t>(hn), f) == static_cast<size_t>(hn));
    std::vector<unsigned char> rgb(static_cast<size_t>(rep.width) * 3u);
    for (uint32_t y = 0; ok && y < rep.height; ++y) {
        for (uint32_t x = 0; x < rep.width; ++x) {
            const auto& p = rep.pixels[static_cast<size_t>(y) * rep.width + x];
            rgb[x * 3 + 0] = p.r; rgb[x * 3 + 1] = p.g; rgb[x * 3 + 2] = p.b;
        }
        if (std::fwrite(rgb.data(), 1, rgb.size(), f) != rgb.size()) ok = false;
    }
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);

    if (!std::filesystem::exists(kNksPath)) {
        std::fprintf(stderr, "[go2_walk_video] missing scene %s\n", kNksPath);
        return 2;
    }
    if (!std::filesystem::exists(args.bin)) {
        std::fprintf(stderr, "[go2_walk_video] missing trajectory %s\n", args.bin.c_str());
        return 2;
    }

    // ---- the dumped, validated WALK kinematics ------------------------------
    Trajectory traj;
    if (!LoadTrajectory(args.bin, &traj)) return 3;
    std::printf("[go2_walk_video] trajectory: %u sub-steps dof=%u links=%u "
                "pose_floats=%u dt=%.4f ctrl_dt=%.4f (sim %.2fs)\n",
                traj.num_steps, traj.dof_count, traj.link_count, traj.pose_floats,
                traj.dt, traj.ctrl_dt, traj.num_steps * traj.dt);

    // ---- FROZEN beauty scene -> host cook (SceneMap) -> RenderWorld ---------
    // CookToModel is PURE C++ (zero CUDA): we use it ONLY for the EntityId ->
    // cooked link_index SceneMap; we DISCARD model (no nk::World, no device --
    // this is a kinematic pose-replay, the poses come from the .bin).
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
    render::RenderWorld rw =
        render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);
    if (rw.instances.empty()) {
        std::fprintf(stderr, "[go2_walk_video] FATAL: BuildRenderWorld produced 0 instances\n");
        return 4;
    }

    // ---- RE-BIND each instance to its TRUE articulation LINK (the cook bug) ----
    // The go2 cook collapses every visual GEOM's pose source onto BODY 0 (the
    // base): BuildRenderWorld resolves all 33 visual instances to body_row 0 with
    // a cached_visual_local composed relative to the base frame, so naively
    // replaying would weld the whole robot to the trunk. We RECOVER the correct
    // per-instance link from the SceneIR: each visual instance carries a CookedRef
    // whose shape_row IS the SceneIR shape id; that shape's body_id is its true
    // owning body, and the SceneMap link binding is identity (link_index i ==
    // body_row i for go2), so trajectory_link = shape.body_id. The visual geom's
    // own local offset INSIDE that link frame is the SceneIR shape local_transform
    // (geom-in-body-frame), so world_xform = link_pose[body_id] o shape.local.
    // This mirrors the H1 video tool's hand-built link<-geom binding.
    std::vector<uint32_t>  inst_link(rw.instances.size(), 0u);   // trajectory link row
    std::vector<Transform> inst_vlocal(rw.instances.size(), Transform::Identity());
    std::vector<uint8_t>   inst_bound(rw.instances.size(), 0u);
    uint32_t n_rebound = 0u, n_unbound = 0u, n_oob = 0u;
    uint32_t max_link_row = 0u;
    for (size_t i = 0; i < rw.instances.size(); ++i) {
        const render::RenderInstance& inst = rw.instances[i];
        const nuka::scene::CookedRef* ref = cooked.scene_map.RefOf(inst.entity);
        if (ref && ref->shape_row != nuka::scene::SceneMap::kNoRow &&
            ref->shape_row < scene.Shapes().size()) {
            const auto& sh = scene.GetShape(ref->shape_row);
            const uint32_t link = static_cast<uint32_t>(sh.body_id);
            if (link < traj.link_count) {
                inst_link[i]   = link;
                inst_vlocal[i] = sh.local_transform;   // geom local IN its link frame
                inst_bound[i]  = 1u;
                max_link_row = std::max(max_link_row, link);
                ++n_rebound;
            } else { ++n_oob; }
        } else { ++n_unbound; }
    }
    std::printf("[go2_walk_video] RenderWorld: instances=%u meshes=%u materials=%u | "
                "rebound_to_link=%u unbound=%u oob=%u max_link_row=%u dumped_links=%u\n",
                rw.InstanceCount(), rw.meshes.Count(), rw.MaterialCount(),
                n_rebound, n_unbound, n_oob, max_link_row, traj.link_count);
    if (n_oob > 0u) {
        std::fprintf(stderr, "[go2_walk_video] FATAL: %u instances reference a link "
                     "row >= dumped link_count %u\n", n_oob, traj.link_count);
        return 5;
    }
    if (n_rebound == 0u && !args.probe) {
        std::fprintf(stderr, "[go2_walk_video] FATAL: no instance re-bound to a link -- "
                     "the dumped link_pose cannot drive this RenderWorld\n");
        return 5;
    }
    if (n_unbound > 0u) {
        std::fprintf(stderr, "[go2_walk_video] WARN: %u instances had no shape->link "
                     "binding (left at link 0)\n", n_unbound);
    }

    // ---- PREMIUM MATERIAL OVERRIDE (owner: make the go2 look better) ----------
    // The frozen go2.nks ships a flat near-white palette (the M8.5 cook default).
    // The real Unitree Go2 is a DARK robot, so we REPLACE the render palette in
    // OUR tool (no edit to go2.nks) with a tasteful studio look: a deep-charcoal
    // composite-shell body, brushed dark-metal actuators on the hip/calf joints,
    // and near-black rubber feet -- so the robot reads premium and POPS against
    // the dark studio floor. Assigned by ARTICULATION LINK semantics (link 0 =
    // trunk; per leg the cook order is hip(abad) -> thigh -> calf, links 1..12 =
    // FL,FR,RL,RR x (hip,thigh,calf)); the calf's lower geom (the foot, vlocal
    // z<-0.15) gets the rubber-foot material.
    auto mk = [](float r, float g, float b, float metallic, float rough) {
        nuka::scene::RenderMaterial m;
        m.base_color[0] = r; m.base_color[1] = g; m.base_color[2] = b;
        m.base_color[3] = 1.0f;
        m.metallic = metallic; m.roughness = rough;
        return m;
    };
    rw.materials.clear();
    const uint32_t kMatShell = 0u;   // deep charcoal composite shell (trunk + thigh shrouds)
    const uint32_t kMatMetal = 1u;   // brushed dark gunmetal actuator (hip/abad + calf links)
    const uint32_t kMatFoot  = 2u;   // near-black matte rubber foot
    const uint32_t kMatAccent = 3u;  // subtle cool-steel accent (face plate / sensor housing)
    rw.materials.push_back(mk(0.016f, 0.017f, 0.020f, 0.02f, 0.62f));  // 0 shell: deep near-black charcoal (matte dielectric)
    rw.materials.push_back(mk(0.040f, 0.043f, 0.052f, 0.90f, 0.32f));  // 1 metal: dark gunmetal actuator, brushed
    rw.materials.push_back(mk(0.010f, 0.010f, 0.012f, 0.02f, 0.85f));  // 2 foot: matte black rubber
    rw.materials.push_back(mk(0.10f,  0.115f, 0.15f,  0.70f, 0.28f));  // 3 accent: cool brushed steel (face/sensor)
    rw.default_material_id = kMatShell;

    // Per-link material role. Cook order links 1..12 = FL,FR,RL,RR x (hip,thigh,calf):
    //   link%3==1 -> hip/abduction actuator (metal)
    //   link%3==2 -> thigh shroud           (shell)
    //   link%3==0 -> calf + foot            (metal upper, rubber foot geom)
    auto link_role_material = [&](uint32_t link, const Transform& vlocal) -> uint32_t {
        if (link == 0u) return kMatShell;            // trunk body shell
        const uint32_t k = ((link - 1u) % 3u);       // 0 hip, 1 thigh, 2 calf
        if (k == 0u) return kMatMetal;               // hip/abduction actuator
        if (k == 1u) return kMatShell;               // thigh shroud
        // calf: the lower geom (the foot capsule, offset down its link) is rubber.
        return (vlocal.position.z < -0.15f) ? kMatFoot : kMatMetal;
    };
    for (size_t i = 0; i < rw.instances.size(); ++i) {
        rw.instances[i].render_material_id = link_role_material(inst_link[i], inst_vlocal[i]);
    }
    // A tasteful cool-steel accent on the trunk's front sensor/face geom (link 0,
    // the LAST base mesh -- the head module sits forward of the body shells).
    // Identify it as the base-link instance whose mesh differs from the big shell;
    // keep it subtle (one accent piece only).
    {
        int last_base = -1;
        for (size_t i = 0; i < rw.instances.size(); ++i)
            if (inst_link[i] == 0u) last_base = static_cast<int>(i);
        if (last_base >= 0)
            rw.instances[static_cast<size_t>(last_base)].render_material_id = kMatAccent;
    }

    // ---- the per-frame publish: world_xform = link_pose[link] o geom_local ----
    auto publish = [&](uint32_t step) {
        for (size_t i = 0; i < rw.instances.size(); ++i) {
            const Transform fk = traj.LinkPose(step, inst_link[i]);
            rw.instances[i].world_xform = fk * inst_vlocal[i];
        }
    };

    // ---- the 4 FOOT instances (rubber-foot material) -> per-foot contact shadow.
    // Each frame we find each foot's LOWEST world vertex (its contact patch xy) and
    // fade its shadow with the foot's height above the z=0 floor (planted = full,
    // lifted = faint) so the swing foot doesn't drag a hard shadow. This is the
    // realistic quadruped look that REPLACES the single belly-blob.
    std::vector<size_t> foot_instances;
    for (size_t i = 0; i < rw.instances.size(); ++i)
        if (rw.instances[i].render_material_id == kMatFoot) foot_instances.push_back(i);
    std::printf("[go2_walk_video] foot instances (contact-shadow casters): %zu\n",
                foot_instances.size());
    auto set_foot_shadows = [&](render::RasterOptions& o) {
        o.contact_points.clear();
        for (size_t fi : foot_instances) {
            const auto& geo = rw.meshes.Geometry(rw.instances[fi].mesh_id);
            const Transform& wx = rw.instances[fi].world_xform;
            float min_z = 1e9f, fx = 0.0f, fy = 0.0f;
            for (size_t v = 0; v + 2 < geo.positions.size(); v += 3) {
                const Vec3 wp = wx.TransformPoint(
                    Vec3{geo.positions[v], geo.positions[v + 1], geo.positions[v + 2]});
                if (wp.z < min_z) { min_z = wp.z; fx = wp.x; fy = wp.y; }
            }
            // planted fraction: 1 at/below z_full, 0 at/above z_zero.
            const float z_full = 0.015f, z_zero = 0.10f;
            float frac = (z_zero - min_z) / (z_zero - z_full);
            frac = frac < 0.0f ? 0.0f : (frac > 1.0f ? 1.0f : frac);
            render::RasterOptions::ContactPoint cp;
            cp.x = fx; cp.y = fy;
            cp.radius   = 0.11f + 0.04f * frac;            // tighter when lifted
            cp.strength = 0.74f * (0.14f + 0.86f * frac);  // faint when lifted, never a hard pop
            o.contact_points.push_back(cp);
        }
    };

    // ---- the renderer (offscreen lavapipe; throws if no Vulkan) -------------
    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    try {
        renderer = std::make_unique<render::VulkanRasterRenderer>();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[go2_walk_video] no Vulkan device: %s\n", e.what());
        return 6;
    }
    std::printf("[go2_walk_video] renderer ICD: %s\n", renderer->DeviceName().c_str());

    // ---- PBR beauty options (the M8.5 go2 hero look) ------------------------
    render::RasterOptions opts;
    opts.width = args.width;
    opts.height = args.height;
    opts.draw_ground = true;       // the go2 walks on the implicit z=0 ground.
    opts.hero_framing = false;     // we drive an explicit FOLLOW camera below.
    opts.use_camera_override = true;
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 40.0f;
    opts.background = {14, 17, 23, 255};
    // GROUNDED LOOK (M10): a mid-grey 18%-card studio sweep (so the near-black
    // rubber feet read AGAINST the floor instead of dissolving into a near-black
    // plate -- the old floating cue) + a soft procedural contact shadow that
    // tracks the dog's footprint and anchors it to the floor every frame.
    opts.ground_color[0] = 0.180f; opts.ground_color[1] = 0.190f; opts.ground_color[2] = 0.210f;
    opts.contact_shadow_strength = 0.0f;  // per-foot contact_points (set_foot_shadows) replace the blob

    // ---- CINEMATIC ORBITING FOLLOW-CAM (owner: vary the angle so the motion
    //      reads against the flat studio floor; sweep SIDE -> FRONT) -------------
    // The TARGET always tracks the base (link 0) + a small body-core lift, so the
    // robot stays centered+framed across the whole +3.9 m walk. The EYE orbits the
    // base in (azimuth phi, elevation, radius), all eased on the normalized walk
    // progress t in [0,1]. The azimuth swing is the key: as the camera arcs around
    // the tracked base, the ground (a flat disc) sweeps + parallaxes underneath,
    // so the forward motion is unmistakable even on a single-color floor.
    //
    // ARC (the robot walks +x; phi measured in the xy-plane, 0 rad = camera on the
    // -y SIDE looking across the flank, +pi/2 = camera AHEAD on +x looking back):
    //   t=0.00  SIDE profile  (phi ~ -8 deg off the -y flank): the gait + body bob
    //           read cleanest in profile -> open on the classic side trot.
    //   t->1.0  FRONT 3/4     (phi ~ +95 deg, camera ahead+right): the dog walks
    //           toward+across camera, a confident hero close.
    // Elevation gently dips then lifts (a soft crane); radius eases in a touch so
    // the dog grows slightly as it arrives front-on. Smoothstep easing throughout
    // so nothing snaps. (Env NK_CAM_FOV still overrides the FOV for probe work.)
    if (const char* fv = std::getenv("NK_CAM_FOV"))
        opts.camera_fov_degrees = static_cast<float>(std::atof(fv));

    auto smoothstep = [](float e0, float e1, float x) {
        float t = (x - e0) / (e1 - e0);
        t = std::max(0.0f, std::min(1.0f, t));
        return t * t * (3.0f - 2.0f * t);
    };
    auto lerp = [](float a, float b, float u) { return a + (b - a) * u; };
    constexpr float kPi = 3.14159265358979323846f;

    // HANDSTAND HERO CAM (owner: LOW angle -- not high/bird's-eye -- with a slow
    // cinematic move: a low slow orbit + gentle dolly-in framing the INVERTED dog
    // dramatically from below). The dog is near-stationary balancing on its rear
    // feet (nose-down, front paws lifted high), so we orbit the FIXED base in a
    // small arc at a LOW elevation and ease the radius in. The target is lifted
    // toward the body core so the lifted front of the dog reads against the sky.
    //
    // The base of an inverted Go2 sits low (z ~ 0.25-0.35) with the trunk pitched
    // up; the lifted front paws reach z ~ 0.6-0.8. We aim a touch ABOVE the base so
    // the whole reared-up body is framed, and keep the EYE LOW (elevation ~7-12 deg)
    // so the camera looks slightly UP at the dramatic inverted silhouette.
    auto aim_camera = [&](uint32_t step) {
        const Transform base = traj.LinkPose(step, 0u);
        const Vec3 b = base.position;
        const float t = (traj.num_steps > 1u)
            ? static_cast<float>(step) / static_cast<float>(traj.num_steps - 1u)
            : 0.0f;
        const float e = smoothstep(0.0f, 1.0f, t);  // eased progress

        // Azimuth: a slow LOW orbit sweeping from a front-3/4 on one flank toward
        // the other (the dog tips nose-DOWN toward -x, front paws up; we keep the
        // camera roughly on the front/side so the lifted paws + chest read).
        const float phi = lerp(-0.55f, 0.55f, e);        // ~ -31deg -> +31deg arc
        // Elevation: LOW and near-constant (a hair of lift mid-move), so the camera
        // looks slightly UP at the inverted dog -- never a top-down/bird's-eye.
        const float elev_deg = lerp(8.0f, 12.0f, e) + 2.0f * std::sin(e * kPi);
        const float elev = elev_deg * kPi / 180.0f;
        // Radius: a slow DOLLY-IN (the dog grows as the move settles).
        const float radius = lerp(2.55f, 1.95f, e);

        const float ch = std::cos(elev), sh = std::sin(elev);
        // The dog pitches nose-down toward -x; put the camera on the -x/front side
        // (phi=0 -> looking from -x toward +x) so we see the lifted front paws.
        const float dx = -std::cos(phi);
        const float dy = std::sin(phi);
        opts.camera_eye = {b.x + radius * ch * dx,
                           b.y + radius * ch * dy,
                           0.05f + radius * sh};            // LOW eye, near the floor
        // Aim at the reared-up body core (above the low base, toward the lifted
        // trunk) so the inverted silhouette is centered and reads against the sky.
        opts.camera_target = {b.x - 0.05f, b.y + 0.0f, b.z + 0.28f};
    };

    // ---- PROBE: print the instance->link map + a few base poses, render 1 ----
    if (args.probe) {
        std::printf("[go2_walk_video] --- instance -> link map (recovered) ---\n");
        for (size_t i = 0; i < rw.instances.size() && i < 40; ++i) {
            const auto& inst = rw.instances[i];
            const nuka::scene::CookedRef* ref = cooked.scene_map.RefOf(inst.entity);
            std::printf("  inst[%2zu] -> traj_link=%u (bound=%d) mesh=%u mat=%u "
                        "entity=%u | ref{body=%u shape=%u} vlocal_t=(%.3f,%.3f,%.3f)\n",
                        i, inst_link[i], inst_bound[i], inst.mesh_id,
                        inst.render_material_id, inst.entity.index,
                        ref ? ref->body_row : 0xffffffffu,
                        ref ? ref->shape_row : 0xffffffffu,
                        inst_vlocal[i].position.x, inst_vlocal[i].position.y,
                        inst_vlocal[i].position.z);
        }
        // Dump SceneIR visual shapes' body_id (the real link owner) so we can
        // confirm the shape->body->link recovery path.
        std::printf("[go2_walk_video] --- SceneIR visual shapes (id, body_id) ---\n");
        for (const auto& s : scene.Shapes()) {
            const bool vis = (s.contype == 0u && s.conaffinity == 0u);
            if (!vis) continue;
            std::printf("  shape id=%u body_id=%u hasMeshRef=%d\n", s.id,
                        static_cast<uint32_t>(s.body_id),
                        s.visual_mesh_ref.empty() ? 0 : 1);
        }
        // Dump the SceneMap link bindings (link_index -> body_row) so we can map
        // each instance's body_row to a trajectory LINK index.
        std::printf("[go2_walk_video] --- SceneMap link bindings (link->body) ---\n");
        for (uint32_t li = 0; li < traj.link_count; ++li) {
            const nuka::scene::EntityId e = cooked.scene_map.EntityOfLink(li);
            const nuka::scene::CookedRef* r =
                (e != nuka::scene::kInvalidEntity) ? cooked.scene_map.RefOf(e) : nullptr;
            std::printf("  link[%u] -> entity=%u body_row=%u\n", li,
                        e.index,
                        r ? r->body_row : 0xffffffffu);
        }
        const uint32_t probes[] = {0u, traj.num_steps / 2u, traj.num_steps - 1u};
        for (uint32_t s : probes) {
            const Transform b = traj.LinkPose(s, 0u);
            std::printf("[go2_walk_video] base @step%u: pos=(%.3f,%.3f,%.3f) "
                        "quat=(%.3f,%.3f,%.3f,%.3f)\n", s, b.position.x, b.position.y,
                        b.position.z, b.rotation.w, b.rotation.x, b.rotation.y, b.rotation.z);
        }
        // Feet-on-ground audit: the lowest WORLD vertex z over a few walk phases.
        // The implicit ground is z=0; a clean walk keeps the lowest foot ~0 (a
        // hair above as the swing foot lifts; the stance foot rests near 0).
        for (uint32_t s : {0u, traj.num_steps / 4u, traj.num_steps / 2u,
                           3u * traj.num_steps / 4u, traj.num_steps - 1u}) {
            publish(s);
            float min_z = 1e9f;
            for (size_t i = 0; i < rw.instances.size(); ++i) {
                const auto& geo = rw.meshes.Geometry(rw.instances[i].mesh_id);
                const Transform& wx = rw.instances[i].world_xform;
                for (size_t v = 0; v + 2 < geo.positions.size(); v += 3) {
                    const Vec3 lp{geo.positions[v], geo.positions[v + 1], geo.positions[v + 2]};
                    const Vec3 wp = wx.TransformPoint(lp);
                    if (wp.z < min_z) min_z = wp.z;
                }
            }
            std::printf("[go2_walk_video] FOOT-Z @step%u: lowest_world_z=%.4f "
                        "(ground=0; +=above, -=sunk)\n", s, min_z);
        }
        const uint32_t mid = traj.num_steps / 2u;
        publish(mid);
        set_foot_shadows(opts);
        aim_camera(mid);
        render::VulkanOffscreenReport rep = renderer->Render(rw, opts);
        std::filesystem::create_directories(args.out_dir);
        const std::string p = args.out_dir + "/probe_frame.ppm";
        const bool ok = WritePpm(rep, p);
        std::printf("[go2_walk_video] PROBE: instances=%u meshes=%u non_bg=%zu "
                    "(%ux%u) eye=(%.2f,%.2f,%.2f) tgt=(%.2f,%.2f,%.2f) -> %s (%s)\n",
                    rw.InstanceCount(), rw.meshes.Count(), rep.non_background_pixel_count,
                    rep.width, rep.height, opts.camera_eye.x, opts.camera_eye.y,
                    opts.camera_eye.z, opts.camera_target.x, opts.camera_target.y,
                    opts.camera_target.z, p.c_str(), ok ? "OK" : "WRITE-FAIL");
        return ok ? 0 : 7;
    }

    // ---- THE REPLAY + RENDER LOOP ------------------------------------------
    std::filesystem::create_directories(args.out_dir);
    uint32_t written = 0u;
    size_t first_nonbg = 0u, last_nonbg = 0u;
    for (uint32_t s = 0; s < traj.num_steps; s += args.stride) {
        if (args.frames != 0u && written >= args.frames) break;
        publish(s);
        set_foot_shadows(opts);
        aim_camera(s);
        render::VulkanOffscreenReport rep = renderer->Render(rw, opts);
        char name[40];
        std::snprintf(name, sizeof(name), "frame_%06u.ppm", written);
        const std::string path = args.out_dir + "/" + name;
        if (!WritePpm(rep, path)) {
            std::fprintf(stderr, "[go2_walk_video] write fail @ sub-step %u\n", s);
            return 7;
        }
        if (written == 0u) first_nonbg = rep.non_background_pixel_count;
        last_nonbg = rep.non_background_pixel_count;
        ++written;
        if ((written % 60u) == 1u)
            std::printf("[go2_walk_video] frame %u (sub-step %u/%u) non_bg=%zu "
                        "base_x=%.3f\n", written - 1u, s, traj.num_steps,
                        rep.non_background_pixel_count, traj.LinkPose(s, 0u).position.x);
    }
    std::printf("[go2_walk_video] DONE: wrote %u frames to %s (first non_bg=%zu, "
                "last non_bg=%zu)\n", written, args.out_dir.c_str(), first_nonbg, last_nonbg);
    return 0;
}
