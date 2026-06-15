// ---------------------------------------------------------------------------
// scene_still -- render ONE offscreen still of any .nks scene (M10 demo tooling).
//
// Loads a scene (.nks), host-cooks it ONLY for the EntityId->link SceneMap
// (CookToModel, pure C++ / zero CUDA -- no nk::World, no device), builds a
// RenderWorld, and rasterises a single frame on the offscreen lavapipe path with
// the M10 grounded-look options (studio floor + contact shadow). PPM out.
//
// This is the kitchen-scene verification + the H1-kitchen-grasp STILL renderer:
// it renders composed .nks scenes (kitchen + H1 + cup) so the owner can approve
// the layout/look BEFORE the RL grasp effort. Pure kinematic bind-pose render --
// no physics, no policy. Built behind NK_BUILD_VULKAN_VALIDATION (real Vulkan).
//
// Usage (from repo root):
//   scene_still --nks <path> --out <ppm> [--w 1920 --h 1080]
//     [--eye X Y Z --target X Y Z --fov 40]   (else auto-frame the scene AABB)
//     [--ground] [--gcolor R G B] [--shadow S] [--bg R G B]   (0..1 colours)
// ---------------------------------------------------------------------------
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
using nuka::math::Vec3;

// A hollow open-top ceramic MUG (the grasp object). Centred on the local origin,
// base at z=0, +z up. Outer wall + inner wall + top rim annulus + inner floor +
// outer bottom -> a believable cup silhouette (no handle; reads as a tumbler/mug).
// The USD cup asset cooks collision-only (no visual mesh path), so we render a
// clean parametric mug here; the real cup hull stays available for grasp physics.
render::MeshGeometry MakeMug(float r, float h, float wall = 0.006f, uint32_t seg = 40) {
    render::MeshGeometry m;
    const float ri = std::max(r - wall, 0.001f);
    const float fz = std::min(wall, h * 0.5f);  // inner floor height
    const float kPi = 3.14159265358979323846f;
    auto push_v = [&](float x, float y, float z, float nx, float ny, float nz) {
        m.positions.push_back(x); m.positions.push_back(y); m.positions.push_back(z);
        m.normals.push_back(nx); m.normals.push_back(ny); m.normals.push_back(nz);
        return (uint32_t)(m.positions.size() / 3 - 1);
    };
    auto quad = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        m.indices.push_back(a); m.indices.push_back(b); m.indices.push_back(c);
        m.indices.push_back(a); m.indices.push_back(c); m.indices.push_back(d);
    };
    for (uint32_t i = 0; i < seg; ++i) {
        const float a0 = 2.0f * kPi * i / seg, a1 = 2.0f * kPi * (i + 1) / seg;
        const float c0 = std::cos(a0), s0 = std::sin(a0), c1 = std::cos(a1), s1 = std::sin(a1);
        // outer wall (normal outward)
        uint32_t o0b = push_v(r * c0, r * s0, 0, c0, s0, 0), o0t = push_v(r * c0, r * s0, h, c0, s0, 0);
        uint32_t o1b = push_v(r * c1, r * s1, 0, c1, s1, 0), o1t = push_v(r * c1, r * s1, h, c1, s1, 0);
        quad(o0b, o1b, o1t, o0t);
        // inner wall (normal inward), from inner floor up to rim
        uint32_t i0b = push_v(ri * c0, ri * s0, fz, -c0, -s0, 0), i0t = push_v(ri * c0, ri * s0, h, -c0, -s0, 0);
        uint32_t i1b = push_v(ri * c1, ri * s1, fz, -c1, -s1, 0), i1t = push_v(ri * c1, ri * s1, h, -c1, -s1, 0);
        quad(i0t, i1t, i1b, i0b);
        // top rim annulus (normal up)
        uint32_t r0o = push_v(r * c0, r * s0, h, 0, 0, 1), r1o = push_v(r * c1, r * s1, h, 0, 0, 1);
        uint32_t r0i = push_v(ri * c0, ri * s0, h, 0, 0, 1), r1i = push_v(ri * c1, ri * s1, h, 0, 0, 1);
        quad(r0i, r1i, r1o, r0o);
        // inner floor (normal up) + outer bottom (normal down) as triangle fans
        uint32_t fic = push_v(0, 0, fz, 0, 0, 1);
        uint32_t fi0 = push_v(ri * c0, ri * s0, fz, 0, 0, 1), fi1 = push_v(ri * c1, ri * s1, fz, 0, 0, 1);
        m.indices.push_back(fic); m.indices.push_back(fi0); m.indices.push_back(fi1);
        uint32_t boc = push_v(0, 0, 0, 0, 0, -1);
        uint32_t bo0 = push_v(r * c0, r * s0, 0, 0, 0, -1), bo1 = push_v(r * c1, r * s1, 0, 0, 0, -1);
        m.indices.push_back(boc); m.indices.push_back(bo1); m.indices.push_back(bo0);
    }
    return m;
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

struct Args {
    std::string nks;
    std::string out = "/tmp/scene_still.ppm";
    uint32_t width = 1920u, height = 1080u;
    bool has_cam = false;
    Vec3 eye{0, 0, 0}, target{0, 0, 0};
    float fov = 40.0f;
    bool ground = false;
    float gcolor[3] = {0.180f, 0.190f, 0.210f};
    float shadow = 0.0f;
    float bg[3] = {0.055f, 0.067f, 0.090f};
    bool cook = false;  // --cook: run CookToModel for an articulated SceneMap (slow on
                        // huge scenes). DEFAULT off: a STILL is a bind-pose render, so
                        // an empty SceneMap -> every instance resolves Static (node-tree
                        // world transform), which is all a static scene needs.
    bool mug = false;   // --mug x y z r h: add a parametric ceramic mug (grasp object)
    float mug_p[5] = {0, 0, 0, 0.045f, 0.10f};  // x,y,z(base), radius, height
    // --overlay <nks> x y z yawdeg: merge another cooked scene (e.g. h1_visual.nks)
    // at a placement (translate + yaw about z). Composes COOKED assets so each keeps
    // its own materials (avoids the fresh-import path's washed-out look).
    std::string overlay;
    float ov_p[4] = {0, 0, 0, 0};  // x, y, z, yaw(deg)
};

float Atof(char* s) { return static_cast<float>(std::atof(s)); }

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto u = [&](uint32_t d) { return (i + 1 < argc) ? (uint32_t)std::atoi(argv[++i]) : d; };
        if (s == "--nks" && i + 1 < argc) a.nks = argv[++i];
        else if (s == "--out" && i + 1 < argc) a.out = argv[++i];
        else if (s == "--w") a.width = u(a.width);
        else if (s == "--h") a.height = u(a.height);
        else if (s == "--fov" && i + 1 < argc) a.fov = Atof(argv[++i]);
        else if (s == "--shadow" && i + 1 < argc) a.shadow = Atof(argv[++i]);
        else if (s == "--ground") a.ground = true;
        else if (s == "--cook") a.cook = true;
        else if (s == "--mug" && i + 5 < argc) {
            for (int k = 0; k < 5; ++k) a.mug_p[k] = Atof(argv[i + 1 + k]);
            i += 5; a.mug = true;
        }
        else if (s == "--overlay" && i + 5 < argc) {
            a.overlay = argv[i + 1];
            for (int k = 0; k < 4; ++k) a.ov_p[k] = Atof(argv[i + 2 + k]);
            i += 5;
        }
        else if (s == "--eye" && i + 3 < argc) {
            a.eye = {Atof(argv[i + 1]), Atof(argv[i + 2]), Atof(argv[i + 3])}; i += 3; a.has_cam = true;
        } else if (s == "--target" && i + 3 < argc) {
            a.target = {Atof(argv[i + 1]), Atof(argv[i + 2]), Atof(argv[i + 3])}; i += 3;
        } else if (s == "--gcolor" && i + 3 < argc) {
            a.gcolor[0] = Atof(argv[i + 1]); a.gcolor[1] = Atof(argv[i + 2]); a.gcolor[2] = Atof(argv[i + 3]); i += 3;
        } else if (s == "--bg" && i + 3 < argc) {
            a.bg[0] = Atof(argv[i + 1]); a.bg[1] = Atof(argv[i + 2]); a.bg[2] = Atof(argv[i + 3]); i += 3;
        }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = ParseArgs(argc, argv);
    if (a.nks.empty()) {
        std::fprintf(stderr, "scene_still: --nks <path> required\n");
        return 2;
    }
    if (!std::filesystem::exists(a.nks)) {
        std::fprintf(stderr, "scene_still: missing scene %s\n", a.nks.c_str());
        return 2;
    }

    auto clk = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };
    const auto t0 = clk();
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(a.nks);
    const auto t1 = clk();
    std::printf("[scene_still] nks::Load %lld ms (shapes=%zu)\n", (long long)ms(t0, t1),
                scene.Shapes().size());
    std::fflush(stdout);

    nuka::scene::SceneMap scene_map;  // empty => all-Static (bind pose), correct for a still
    if (a.cook) {
        cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
        scene_map = std::move(cooked.scene_map);
        std::printf("[scene_still] CookToModel %lld ms\n", (long long)ms(t1, clk()));
        std::fflush(stdout);
    }
    const auto t2 = clk();
    render::RenderWorld rw = render::BuildRenderWorld(scene.Ecs(), scene_map);
    std::printf("[scene_still] BuildRenderWorld %lld ms -> instances=%u meshes=%u materials=%zu\n",
                (long long)ms(t2, clk()), rw.InstanceCount(), rw.meshes.Count(), rw.materials.size());
    std::fflush(stdout);

    if (!a.overlay.empty()) {
        const nuka::scene::SceneIR ov_scene = nuka::scene::nks::Load(a.overlay);
        nuka::scene::SceneMap ov_map;
        render::RenderWorld ov = render::BuildRenderWorld(ov_scene.Ecs(), ov_map);
        // placement: translate to (x,y,z) and yaw about +z
        const float yaw = a.ov_p[3] * 3.14159265358979323846f / 180.0f;
        nuka::math::Transform place = nuka::math::Transform::Identity();
        place.position = {a.ov_p[0], a.ov_p[1], a.ov_p[2]};
        place.rotation = nuka::math::Quat::FromAxisAngle({0, 0, 1}, yaw);
        const uint32_t mat_off = (uint32_t)rw.materials.size();
        for (const auto& m : ov.materials) rw.materials.push_back(m);
        uint32_t added = 0;
        for (const auto& oi : ov.instances) {
            const render::MeshGeometry geo = ov.meshes.Geometry(oi.mesh_id);
            char key[48];
            std::snprintf(key, sizeof(key), "ovl/%u", oi.mesh_id);
            const uint32_t mid = rw.meshes.InternPrimitive(key, [&] { return geo; });
            render::RenderInstance ni;
            ni.mesh_id = mid;
            ni.render_material_id =
                (oi.render_material_id == render::kNoId) ? rw.default_material_id
                                                         : oi.render_material_id + mat_off;
            ni.world_xform = place * oi.world_xform;
            rw.instances.push_back(ni);
            ++added;
        }
        std::printf("[scene_still] + overlay %s: %u instances at (%.2f,%.2f,%.2f) yaw=%.0f\n",
                    a.overlay.c_str(), added, a.ov_p[0], a.ov_p[1], a.ov_p[2], a.ov_p[3]);
    }

    if (a.mug) {
        const uint32_t mid = rw.meshes.InternPrimitive("mug", [&] {
            return MakeMug(a.mug_p[3], a.mug_p[4]);
        });
        nuka::scene::RenderMaterial cer;  // cream ceramic
        cer.base_color[0] = 0.86f; cer.base_color[1] = 0.83f; cer.base_color[2] = 0.78f;
        cer.base_color[3] = 1.0f; cer.metallic = 0.0f; cer.roughness = 0.45f;
        const uint32_t matid = (uint32_t)rw.materials.size();
        rw.materials.push_back(cer);
        render::RenderInstance inst;
        inst.mesh_id = mid;
        inst.render_material_id = matid;
        inst.world_xform = nuka::math::Transform::Identity();
        inst.world_xform.position = {a.mug_p[0], a.mug_p[1], a.mug_p[2]};
        rw.instances.push_back(inst);
        std::printf("[scene_still] + mug at (%.2f,%.2f,%.2f) r=%.3f h=%.3f\n",
                    a.mug_p[0], a.mug_p[1], a.mug_p[2], a.mug_p[3], a.mug_p[4]);
    }
    if (rw.InstanceCount() == 0u) {
        std::fprintf(stderr, "[scene_still] FATAL: 0 render instances\n");
        return 3;
    }

    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    try {
        renderer = std::make_unique<render::VulkanRasterRenderer>();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[scene_still] no Vulkan device: %s\n", e.what());
        return 6;
    }
    std::printf("[scene_still] renderer ICD: %s\n", renderer->DeviceName().c_str());

    render::RasterOptions opts;
    opts.width = a.width;
    opts.height = a.height;
    opts.background = {(uint8_t)(a.bg[0] * 255), (uint8_t)(a.bg[1] * 255), (uint8_t)(a.bg[2] * 255), 255};
    opts.draw_ground = a.ground;
    opts.ground_color[0] = a.gcolor[0]; opts.ground_color[1] = a.gcolor[1]; opts.ground_color[2] = a.gcolor[2];
    opts.contact_shadow_strength = a.shadow;  // single-blob (no per-foot list here)
    if (a.has_cam) {
        opts.use_camera_override = true;
        opts.camera_eye = a.eye;
        opts.camera_target = a.target;
        opts.camera_up = {0.0f, 0.0f, 1.0f};
        opts.camera_fov_degrees = a.fov;
    } else {
        opts.hero_framing = true;  // auto-frame the scene AABB as a 3/4 hero shot
    }

    render::VulkanOffscreenReport rep = renderer->Render(rw, opts);
    std::filesystem::create_directories(std::filesystem::path(a.out).parent_path());
    if (!WritePpm(rep, a.out)) {
        std::fprintf(stderr, "[scene_still] write failed -> %s\n", a.out.c_str());
        return 7;
    }
    std::printf("[scene_still] wrote %ux%u non_bg=%zu -> %s\n",
                rep.width, rep.height, rep.non_background_pixel_count, a.out.c_str());
    return 0;
}
