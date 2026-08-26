#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "render/studio_beauty.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"
#include "scene/ecs/components.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
namespace render = nuka::render;

constexpr uint32_t MAGIC = 0x474F3257u;

constexpr const char* LINK_NAMES[13] = {
    "base", "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh",
    "FR_calf", "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh",
    "RR_calf"};

nuka::scene::SceneMap make_visual_map(const nuka::scene::Registry& registry) {
    nuka::scene::SceneMap map;
    registry.ForEach<nuka::scene::NameComponent>([&](nuka::scene::EntityId e,
                                                     const nuka::scene::NameComponent& name) {
        if (!registry.Has<nuka::scene::RigidBodyComponent>(e)) return;
        for (uint32_t link = 0; link < 13; ++link) {
            if (name.name == LINK_NAMES[link]) {
                nuka::scene::CookedRef ref;
                ref.link_index = link;
                map.Bind(e, ref);
            }
        }
    });
    return map;
}

nuka::scene::RenderMaterial make_material(float r, float g, float b,
                                          float metallic, float roughness) {
    nuka::scene::RenderMaterial material;
    material.base_color[0] = r;
    material.base_color[1] = g;
    material.base_color[2] = b;
    material.base_color[3] = 1.0f;
    material.metallic = metallic;
    material.roughness = roughness;
    return material;
}

nuka::render::MeshGeometry make_floor_tile_geometry() {
    constexpr float half = 0.325f;
    nuka::render::MeshGeometry geometry;
    geometry.positions = {
        -half, -half, 0.0f,
         half, -half, 0.0f,
         half,  half, 0.0f,
        -half,  half, 0.0f,
    };
    geometry.normals = {
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
    };
    geometry.uvs = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,
    };
    geometry.indices = {0u, 1u, 2u, 0u, 2u, 3u};
    return geometry;
}

void add_showcase_floor(nuka::render::StudioScene& studio) {
    const uint32_t tile_mesh = studio.world.meshes.InternPrimitive(
        "showcase_checker_tile", make_floor_tile_geometry);
    const uint32_t tile_dark = static_cast<uint32_t>(studio.world.materials.size());
    studio.world.materials.push_back(make_material(0.055f, 0.070f, 0.095f, 0.02f, 0.84f));
    const uint32_t tile_light = static_cast<uint32_t>(studio.world.materials.size());
    studio.world.materials.push_back(make_material(0.145f, 0.175f, 0.215f, 0.02f, 0.78f));
    const uint32_t tile_edge = static_cast<uint32_t>(studio.world.materials.size());
    studio.world.materials.push_back(make_material(0.42f, 0.16f, 0.045f, 0.12f, 0.52f));

    constexpr int grid = 12;
    constexpr float spacing = 0.65f;
    constexpr float origin = -3.575f;
    for (int y = 0; y < grid; ++y) {
        for (int x = 0; x < grid; ++x) {
            nuka::render::RenderInstance tile;
            tile.mesh_id = tile_mesh;
            const bool edge = x == 0 || y == 0 || x == grid - 1 || y == grid - 1;
            tile.render_material_id = edge ? tile_edge
                                           : (((x + y) & 1) ? tile_light : tile_dark);
            tile.world_xform.position = {
                origin + spacing * static_cast<float>(x),
                origin + spacing * static_cast<float>(y),
                0.004f};
            tile.pose_source.kind = nuka::render::PoseSource::Kind::Static;
            studio.world.instances.push_back(std::move(tile));
        }
    }
}

void update_showcase_contacts(nuka::render::StudioScene& studio,
                              const std::vector<nuka::math::Transform>& links) {
    studio.options.contact_points.clear();
    // The calf origins are the closest stable physics poses to the four Go2 feet;
    // the renderer projects these patches onto the flat display floor.
    constexpr uint32_t calf_links[4] = {3u, 6u, 9u, 12u};
    for (uint32_t link : calf_links) {
        if (link >= links.size()) continue;
        const auto& p = links[link].position;
        const float strength = std::clamp(1.0f - p.z / 0.26f, 0.0f, 1.0f);
        if (strength <= 0.04f) continue;
        nuka::render::RasterOptions::ContactPoint contact;
        contact.x = p.x;
        contact.y = p.y;
        contact.radius = 0.10f;
        contact.strength = 0.35f + 0.55f * strength;
        studio.options.contact_points.push_back(contact);
    }
}

struct Args {
    std::string bin = "out/go2_skill_trajectory.bin";
    std::string out = "out/go2_cuda_frames";
    uint32_t frames = 0;
    uint32_t stride = 1;
    uint32_t width = 1280;
    uint32_t height = 720;
    uint32_t samples = 2;
    bool probe = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&](std::string* dst) {
            if (i + 1 < argc) *dst = argv[++i];
        };
        if (s == "--bin") next(&a.bin);
        else if (s == "--out-dir") next(&a.out);
        else if (s == "--frames" && i + 1 < argc) a.frames = std::atoi(argv[++i]);
        else if (s == "--stride" && i + 1 < argc) a.stride = std::max(1, std::atoi(argv[++i]));
        else if (s == "--width" && i + 1 < argc) a.width = std::atoi(argv[++i]);
        else if (s == "--height" && i + 1 < argc) a.height = std::atoi(argv[++i]);
        else if (s == "--samples" && i + 1 < argc) a.samples = std::atoi(argv[++i]);
        else if (s == "--probe") a.probe = true;
    }
    return a;
}

struct Trajectory {
    uint32_t steps = 0;
    uint32_t dofs = 0;
    uint32_t links = 0;
    uint32_t pose_floats = 0;
    float dt = 0;
    std::vector<std::vector<float>> poses;
};

bool load_trajectory(const std::string& path, Trajectory* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hi[8]{};
    float hf[2]{};
    f.read(reinterpret_cast<char*>(hi), sizeof(hi));
    f.read(reinterpret_cast<char*>(hf), sizeof(hf));
    if (!f || static_cast<uint32_t>(hi[0]) != MAGIC) return false;
    out->steps = static_cast<uint32_t>(hi[2]);
    out->dofs = static_cast<uint32_t>(hi[3]);
    out->links = static_cast<uint32_t>(hi[4]);
    out->pose_floats = static_cast<uint32_t>(hi[5]);
    out->dt = hf[0];
    const size_t pose_count = static_cast<size_t>(out->links) * out->pose_floats;
    const size_t record_count = out->dofs + pose_count;
    std::vector<float> record(record_count);
    out->poses.resize(out->steps);
    for (uint32_t i = 0; i < out->steps; ++i) {
        f.read(reinterpret_cast<char*>(record.data()), record_count * sizeof(float));
        if (!f) return false;
        out->poses[i].assign(record.begin() + out->dofs, record.end());
    }
    return true;
}

bool write_ppm(const render::VulkanOffscreenReport& rep, const std::string& path) {
    if (rep.pixels.empty()) return false;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%u %u\n255\n", rep.width, rep.height);
    for (const auto& p : rep.pixels) {
        const unsigned char rgb[3] = {p.r, p.g, p.b};
        std::fwrite(rgb, 1, 3, f);
    }
    return std::fclose(f) == 0;
}

Transform pose_at(const std::vector<float>& flat, uint32_t link, uint32_t stride) {
    const float* p = flat.data() + static_cast<size_t>(link) * stride;
    return Transform{Vec3{p[0], p[1], p[2]}, Quat{p[3], p[4], p[5], p[6]}};
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    const Args args = parse_args(argc, argv);
    Trajectory traj;
    if (!load_trajectory(args.bin, &traj)) {
        std::fprintf(stderr, "cannot load GO2W trajectory: %s\n", args.bin.c_str());
        return 2;
    }
    std::fprintf(stderr, "[cuda-video] trajectory loaded steps=%u links=%u\n", traj.steps, traj.links);
    std::fprintf(stderr, "[cuda-video] loading go2.nks\n");
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load("examples/scenes/go2.nks");
    std::fprintf(stderr, "[cuda-video] nks loaded\n");
    const nuka::scene::SceneMap map = make_visual_map(scene.Ecs());
    std::fprintf(stderr, "[cuda-video] visual map entries=%zu\n", map.Size());
    render::StudioScene studio = render::BuildStudioScene(
        scene.Ecs(), map, std::vector<nuka::runtime::soft::SurfaceTopology>{},
        args.width, args.height);
    add_showcase_floor(studio);
    studio.options.draw_ground = true;
    studio.options.ground_color[0] = 0.105f;
    studio.options.ground_color[1] = 0.125f;
    studio.options.ground_color[2] = 0.155f;
    studio.options.beauty_sky_fill = 0.42f;
    studio.options.beauty_exposure_ev = 0.25f;
    studio.options.beauty_grade = 0.10f;
    studio.options.beauty_specular_env = true;
    studio.options.contact_shadow_strength = 0.0f;
    std::fprintf(stderr, "[cuda-video] studio built instances=%zu\n", studio.world.instances.size());
    if (studio.world.instances.empty()) {
        std::fprintf(stderr, "go2.nks produced no visual instances\n");
        return 3;
    }
    std::fprintf(stderr, "[cuda-video] creating CUDA RT\n");
    render::StudioRtRenderer renderer;
    if (!renderer.ok()) {
        std::fprintf(stderr, "CUDA RT backend unavailable\n");
        return 4;
    }
    renderer.SetBeauty(true, args.samples);
    std::filesystem::create_directories(args.out);
    const uint32_t limit = args.frames ? std::min(args.frames, traj.steps) : traj.steps;
    std::vector<Transform> links(traj.links, Transform::Identity());
    for (uint32_t i = 0, written = 0; i < limit; i += args.stride, ++written) {
        for (uint32_t link = 0; link < traj.links; ++link)
            links[link] = pose_at(traj.poses[i], link, traj.pose_floats);
        const Vec3 base = links[0].position;
        const float az = -0.65f + 0.35f * std::sin(static_cast<float>(i) * 0.01f);
        const Vec3 look{base.x, base.y, 0.38f};
        const float radius = 1.55f;
        studio.options.camera_target = look;
        studio.options.camera_eye = {
            look.x + radius * std::cos(az),
            look.y + radius * std::sin(az),
            look.z + 0.55f};
        update_showcase_contacts(studio, links);
        render::PublishStudioScene(studio, links, {});
        const auto report = renderer.Render(studio.world, studio.options);
        char name[64];
        std::snprintf(name, sizeof(name), "frame_%06u.ppm", written);
        if (!write_ppm(report, (std::filesystem::path(args.out) / name).string())) return 5;
        if (args.probe) {
            std::printf("CUDA RT probe: instances=%u meshes=%u frame=%u\n",
                        static_cast<unsigned>(studio.world.instances.size()),
                        studio.world.meshes.Count(), i);
            return 0;
        }
    }
    std::printf("CUDA RT rendered %u frames to %s\n",
                (limit + args.stride - 1) / args.stride, args.out.c_str());
    return 0;
}
