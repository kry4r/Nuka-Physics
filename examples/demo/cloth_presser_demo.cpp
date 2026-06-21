// ---------------------------------------------------------------------------
// cloth_presser_demo.cpp -- THE FIRST body<->particle coupling headline clip.
//
// A real XPBD cloth (24x24 = 576 particles) drapes across two low ridges (a
// hammock) above a studio floor, then a rigid sphere "foot" descends into the
// sagging center, presses, and lifts -- the cloth dips under it and springs back.
// The cloth particles are SPHERE collidables flowing through the SAME body<->
// particle row solver a foot uses on the ground (detection -> body<->particle
// narrowphase -> SolveRowsBlockIsland -> finalize compose): there is NO bespoke
// coupler, the medium difference is data (the soft mu), never code.
//
// PIPELINE (physics on CUDA GPU 0, render offscreen on lavapipe):
//   build ONE nk::World (ground plane + two ridges + a presser sphere + the cloth
//   cooked via CookXpbdParticles, which grows the disjoint body<->particle contact
//   budget) -> settle the drape -> SCRIPT the kinematic presser down/hold/up.
//   PER FRAME: download the live particle positions + presser pose, rebuild the
//   cloth as a DEFORMING triangle mesh (the MakeGrid triangulation over the live
//   positions), render the cloth + presser + ridges with a studio floor + an
//   orbiting hero camera + per-contact shadow decals, write a PPM. At three key
//   frames (settled / deepest dip / recovered) also write a hero PNG.
//
// Built behind NK_BUILD_VULKAN_VALIDATION (build-viewer / lavapipe). Usage:
//   cloth_presser_demo [--width W] [--height H] [--out-dir DIR] [--png-dir DIR]
//                      [--stride S] [--probe]
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "collision/shape_kind.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"
#include "runtime/soft/cloth_topology.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
namespace render = nuka::render;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox = nuka::collision::kShapeBox;
constexpr uint32_t kKindSphere = nuka::collision::kShapeSphere;
constexpr uint32_t kKindPlane = nuka::collision::kShapePlane;

// Cloth + scene geometry (mirrors the cloth_presser_two_way scenario gate). An
// unpinned 24x24 sheet drapes across two low ridges; its center sags into the gap.
constexpr uint32_t kGridN = 24u;
constexpr float kSpacing = 0.025f;
constexpr float kClothZ = 0.16f;
constexpr float kParticleMass = 0.01f;
constexpr float kContactDMin = 0.030f;
constexpr float kRidgeHalfX = 0.12f;
constexpr float kRidgeHalfY = 0.30f;
constexpr float kRidgeHalfZ = 0.05f;
constexpr float kRidgeCenterX = 0.16f;
constexpr float kRidgeTopZ = kRidgeHalfZ * 2.0f;
constexpr float kPresserRadius = 0.06f;
constexpr uint16_t kXpbdIters = 20u;
constexpr uint32_t kPresserBody = 3u;
constexpr uint64_t kPoseOff = static_cast<uint64_t>(kPresserBody) * sizeof(Transform);
constexpr uint64_t kVec3Off = static_cast<uint64_t>(kPresserBody) * sizeof(Vec3);

// ---- CLI ------------------------------------------------------------------
struct Args {
    uint32_t width = 1920u;
    uint32_t height = 1080u;
    uint32_t stride = 2u;  // render every stride-th physics step.
    std::string out_dir = "/tmp/cloth_presser_frames";
    std::string png_dir = "out/cloth_demo";
    bool probe = false;
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next_u = [&](uint32_t def) -> uint32_t {
            return (i + 1 < argc) ? static_cast<uint32_t>(std::atoi(argv[++i])) : def;
        };
        if (s == "--width") a.width = next_u(a.width);
        else if (s == "--height") a.height = next_u(a.height);
        else if (s == "--stride") a.stride = std::max(1u, next_u(a.stride));
        else if (s == "--probe") a.probe = true;
        else if (s == "--out-dir" && i + 1 < argc) a.out_dir = argv[++i];
        else if (s == "--png-dir" && i + 1 < argc) a.png_dir = argv[++i];
    }
    return a;
}

// ---- scene construction (mirrors the scenario gate) -----------------------
struct ClothMesh {
    std::vector<Vec3> rest;
    std::vector<uint32_t> tri;  // 3 indices per triangle.
};

ClothMesh MakeClothMesh() {
    ClothMesh c;
    const float c0 = -0.5f * static_cast<float>(kGridN - 1u) * kSpacing;
    for (uint32_t j = 0; j < kGridN; ++j)
        for (uint32_t i = 0; i < kGridN; ++i)
            c.rest.push_back(Vec3{c0 + static_cast<float>(i) * kSpacing,
                                  c0 + static_cast<float>(j) * kSpacing, kClothZ});
    auto idx = [](uint32_t i, uint32_t j) { return j * kGridN + i; };
    for (uint32_t j = 0; j + 1 < kGridN; ++j) {
        for (uint32_t i = 0; i + 1 < kGridN; ++i) {
            const uint32_t v00 = idx(i, j), v10 = idx(i + 1, j);
            const uint32_t v01 = idx(i, j + 1), v11 = idx(i + 1, j + 1);
            c.tri.insert(c.tri.end(), {v00, v10, v11, v00, v11, v01});
        }
    }
    return c;
}

cook::XpbdCookInput BuildClothInput(const ClothMesh& cm) {
    std::vector<soft::ClothTriangle> tris;
    for (size_t t = 0; t + 2 < cm.tri.size(); t += 3)
        tris.push_back(soft::ClothTriangle{{cm.tri[t], cm.tri[t + 1], cm.tri[t + 2]}});
    soft::ClothTopologyOptions opts;
    opts.distance_compliance_alpha = 0.0f;
    opts.bend_compliance_alpha = 1.0e-4f;
    soft::XpbdConstraintSet cs;
    soft::BuildClothConstraints(cm.rest, tris, opts, cs);

    cook::XpbdCookInput in;
    in.positions = cm.rest;
    in.velocities.assign(cm.rest.size(), Vec3{0.0f, 0.0f, -0.05f});
    in.inv_mass.assign(cm.rest.size(), 1.0f / kParticleMass);
    for (const auto& dc : cs.distance) {
        cook::CookDistanceCon c;
        c.a = dc.particle_a; c.b = dc.particle_b;
        c.rest_length = dc.rest_length; c.compliance_alpha = dc.compliance_alpha;
        in.distance.push_back(c);
    }
    for (const auto& bc : cs.bend) {
        cook::CookBendCon c;
        for (uint32_t k = 0; k < 4u; ++k) { c.p[k] = bc.particle[k]; c.k[k] = bc.k[k]; }
        c.compliance_alpha = bc.compliance_alpha;
        in.bend.push_back(c);
    }
    in.solver_iterations = kXpbdIters;
    in.friction = 0.6f;
    return in;
}

void AddStaticBox(nk::Model& m, const Vec3& pos, const Vec3& half, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = 0.0f; bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half.x; sh.params[1] = half.y; sh.params[2] = half.z;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

void AddGroundPlane(nk::Model& m, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.rotation = Quat::FromAxisAngle(Vec3{1, 0, 0}, 1.57079632679f);
    bi.inv_mass = 0.0f; bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindPlane;
    sh.params[0] = 0.0f; sh.params[1] = 0.0f; sh.params[2] = 0.0f;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

void AddSphere(nk::Model& m, const Vec3& pos, float radius, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = 0.0f; bi.inv_inertia = Vec3{0, 0, 0};  // kinematic (scripted).
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindSphere;
    sh.params[0] = radius; sh.params[1] = radius; sh.params[2] = radius;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

nk::Model BuildScene(const ClothMesh& cm, const Vec3& presser_pos) {
    nk::Model m;
    const Vec3 ridge_half{kRidgeHalfX, kRidgeHalfY, kRidgeHalfZ};
    AddGroundPlane(m, 0);
    AddStaticBox(m, Vec3{-kRidgeCenterX, 0.0f, kRidgeHalfZ}, ridge_half, 1);
    AddStaticBox(m, Vec3{+kRidgeCenterX, 0.0f, kRidgeHalfZ}, ridge_half, 2);
    AddSphere(m, presser_pos, kPresserRadius, 3);

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = bodies * 4u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;
    cook::CookXpbdParticles(m, 1u, BuildClothInput(cm));
    m.particles.pp_contact_d_min = kContactDMin;
    return m;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u; cfg.pos_iters = 0u;
    cfg.max_pairs = 64u;
    return cfg;
}

void DrivePresser(nk::World& w, const Vec3& target) {
    Transform tf = Transform::Identity();
    tf.position = target;
    const Vec3 zero = Vec3::Zero();
    w.GetData().UploadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), kPoseOff);
    w.GetData().UploadField(nk::FieldId::BodyLinearVelocity, &zero, sizeof(Vec3), kVec3Off);
    w.GetData().UploadField(nk::FieldId::BodyAngularVelocity, &zero, sizeof(Vec3), kVec3Off);
}

// ---- render helpers -------------------------------------------------------
nuka::scene::RenderMaterial MakeMat(float r, float g, float b, float metallic,
                                    float rough) {
    nuka::scene::RenderMaterial m;
    m.base_color[0] = r; m.base_color[1] = g; m.base_color[2] = b;
    m.base_color[3] = 1.0f;
    m.metallic = metallic; m.roughness = rough;
    return m;
}

// A box mesh with the given half-extents centered at the origin (the renderer
// applies only a rigid Transform, so the extents are baked into the geometry).
render::MeshGeometry MakeBoxGeo(const Vec3& half) {
    render::MeshGeometry g;
    const float xs[2] = {-half.x, half.x};
    const float ys[2] = {-half.y, half.y};
    const float zs[2] = {-half.z, half.z};
    for (int xi = 0; xi < 2; ++xi)
        for (int yi = 0; yi < 2; ++yi)
            for (int zi = 0; zi < 2; ++zi) {
                g.positions.insert(g.positions.end(), {xs[xi], ys[yi], zs[zi]});
            }
    auto v = [](int x, int y, int z) -> uint32_t {
        return static_cast<uint32_t>(x * 4 + y * 2 + z);
    };
    const uint32_t faces[6][4] = {
        {v(0,0,0), v(0,1,0), v(0,1,1), v(0,0,1)},  // -x
        {v(1,0,0), v(1,0,1), v(1,1,1), v(1,1,0)},  // +x
        {v(0,0,0), v(0,0,1), v(1,0,1), v(1,0,0)},  // -y
        {v(0,1,0), v(1,1,0), v(1,1,1), v(0,1,1)},  // +y
        {v(0,0,0), v(1,0,0), v(1,1,0), v(0,1,0)},  // -z
        {v(0,0,1), v(0,1,1), v(1,1,1), v(1,0,1)},  // +z
    };
    for (auto& f : faces) {
        g.indices.insert(g.indices.end(), {f[0], f[1], f[2], f[0], f[2], f[3]});
    }
    return g;
}

// A UV sphere of `radius` centered at the origin.
render::MeshGeometry MakeSphereGeo(float radius, uint32_t rings = 18u,
                                   uint32_t sectors = 28u) {
    render::MeshGeometry g;
    constexpr float kPi = 3.14159265358979323846f;
    for (uint32_t r = 0; r <= rings; ++r) {
        const float phi = kPi * static_cast<float>(r) / static_cast<float>(rings);
        for (uint32_t s = 0; s <= sectors; ++s) {
            const float th = 2.0f * kPi * static_cast<float>(s) /
                             static_cast<float>(sectors);
            g.positions.insert(g.positions.end(),
                               {radius * std::sin(phi) * std::cos(th),
                                radius * std::sin(phi) * std::sin(th),
                                radius * std::cos(phi)});
        }
    }
    const uint32_t stride = sectors + 1u;
    for (uint32_t r = 0; r < rings; ++r)
        for (uint32_t s = 0; s < sectors; ++s) {
            const uint32_t a = r * stride + s, b = a + stride;
            g.indices.insert(g.indices.end(), {a, a + 1u, b, b, a + 1u, b + 1u});
        }
    return g;
}

// The cloth as a double-sided deforming triangle mesh from the live positions.
render::MeshGeometry MakeClothGeo(const std::vector<Vec3>& pos,
                                  const std::vector<uint32_t>& tri) {
    render::MeshGeometry g;
    g.positions.reserve(pos.size() * 3u);
    for (const Vec3& p : pos)
        g.positions.insert(g.positions.end(), {p.x, p.y, p.z});
    g.indices.reserve(tri.size() * 2u);
    for (size_t t = 0; t + 2 < tri.size(); t += 3) {
        g.indices.insert(g.indices.end(), {tri[t], tri[t + 1], tri[t + 2]});
        g.indices.insert(g.indices.end(), {tri[t], tri[t + 2], tri[t + 1]});  // back face
    }
    return g;
}

bool WritePpm(const render::VulkanOffscreenReport& rep, const std::string& path) {
    if (rep.pixels.empty() || rep.width == 0u || rep.height == 0u) return false;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    char hdr[64];
    const int hn = std::snprintf(hdr, sizeof(hdr), "P6\n%u %u\n255\n", rep.width, rep.height);
    bool ok = hn > 0 &&
              std::fwrite(hdr, 1, static_cast<size_t>(hn), f) == static_cast<size_t>(hn);
    std::vector<unsigned char> row(static_cast<size_t>(rep.width) * 3u);
    for (uint32_t y = 0; ok && y < rep.height; ++y) {
        for (uint32_t x = 0; x < rep.width; ++x) {
            const auto& p = rep.pixels[static_cast<size_t>(y) * rep.width + x];
            row[x * 3 + 0] = p.r; row[x * 3 + 1] = p.g; row[x * 3 + 2] = p.b;
        }
        if (std::fwrite(row.data(), 1, row.size(), f) != row.size()) ok = false;
    }
    if (std::fclose(f) != 0) ok = false;
    return ok;
}

bool WritePng(const render::VulkanOffscreenReport& rep, const std::string& path) {
    // Encode via ffmpeg from a temp PPM (lavapipe has no PNG writer); the .sh also
    // uses ffmpeg, so it is on PATH. Falls back to leaving the PPM if ffmpeg fails.
    const std::string tmp = path + ".ppm";
    if (!WritePpm(rep, tmp)) return false;
    const std::string cmd = "/usr/bin/ffmpeg -y -loglevel error -i \"" + tmp +
                            "\" \"" + path + "\" 2>/dev/null";
    const int rc = std::system(cmd.c_str());
    std::error_code ec;
    if (rc == 0) std::filesystem::remove(tmp, ec);
    return rc == 0;
}

// One renderable frame: cloth (deformed) + presser sphere + two ridges, on a
// studio floor, with a soft contact-shadow decal under the presser footprint.
struct FrameScene {
    render::RenderWorld rw;
    uint32_t cloth_inst = 0u;
};

FrameScene BuildFrame(const ClothMesh& cm, const std::vector<Vec3>& cloth_pos,
                      const Transform& presser) {
    FrameScene fs;
    render::RenderWorld& rw = fs.rw;
    rw.materials.push_back(MakeMat(0.55f, 0.13f, 0.16f, 0.0f, 0.72f));  // 0 cloth crimson
    rw.materials.push_back(MakeMat(0.62f, 0.64f, 0.70f, 0.85f, 0.30f));  // 1 presser steel
    rw.materials.push_back(MakeMat(0.12f, 0.13f, 0.16f, 0.10f, 0.65f));  // 2 ridge charcoal
    rw.default_material_id = 0u;

    const uint32_t cloth_mesh = rw.meshes.InternPrimitive(
        "cloth:live", [&] { return MakeClothGeo(cloth_pos, cm.tri); });
    const uint32_t sphere_mesh = rw.meshes.InternPrimitive(
        "presser:sphere", [&] { return MakeSphereGeo(kPresserRadius); });
    const Vec3 rh{kRidgeHalfX, kRidgeHalfY, kRidgeHalfZ};
    const uint32_t box_mesh =
        rw.meshes.InternPrimitive("ridge:box", [&] { return MakeBoxGeo(rh); });

    auto add = [&](uint32_t mesh, uint32_t mat, const Transform& xf) {
        render::RenderInstance inst;
        inst.mesh_id = mesh; inst.render_material_id = mat; inst.world_xform = xf;
        inst.pose_source.kind = render::PoseSource::Kind::Static;
        rw.instances.push_back(inst);
    };
    add(cloth_mesh, 0u, Transform::Identity());  // cloth verts are world-space.
    fs.cloth_inst = 0u;
    add(sphere_mesh, 1u, presser);
    Transform ra = Transform::Identity(); ra.position = Vec3{-kRidgeCenterX, 0, kRidgeHalfZ};
    Transform rb = Transform::Identity(); rb.position = Vec3{+kRidgeCenterX, 0, kRidgeHalfZ};
    add(box_mesh, 2u, ra);
    add(box_mesh, 2u, rb);
    return fs;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);

    nphi::Device* dev = nphi::InitBestDevice();
    nphi::Backend* backend = dev ? nphi::DeviceInitBackend(dev, nullptr) : nullptr;
    if (backend == nullptr) {
        std::fprintf(stderr, "[cloth_presser_demo] no CUDA backend\n");
        return 2;
    }

    const ClothMesh cm = MakeClothMesh();
    const float top = 0.40f;       // presser parked / lift height.
    const float bottom = kRidgeTopZ - 0.02f + kPresserRadius;  // deepest press.
    nk::Model model = BuildScene(cm, Vec3{0.0f, 0.0f, top});
    const uint32_t P = model.capacities.particles_per_env;
    nk::World world(std::move(model), 1u, dev, backend, Cfg());
    if (!world.Ready()) {
        std::fprintf(stderr, "[cloth_presser_demo] world not ready\n");
        return 3;
    }

    // The choreography: a list of (presser z, render?) per physics step. Settle
    // (no render) -> a short hold (render begins) -> down -> hold -> up -> relax.
    struct Segment { uint32_t steps; float z0, z1; bool render; };
    const std::vector<Segment> segments = {
        {600u, top, top, false},  // settle the drape (off camera).
        { 90u, top, top, true},   // a beat on the settled drape.
        {220u, top, bottom, true},   // descend into the sag.
        {150u, bottom, bottom, true},  // press + hold.
        {220u, bottom, top, true},   // lift away.
        {240u, top, top, true},   // the cloth springs back.
    };

    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    try {
        renderer = std::make_unique<render::VulkanRasterRenderer>();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[cloth_presser_demo] no Vulkan device: %s\n", e.what());
        return 4;
    }
    std::printf("[cloth_presser_demo] renderer ICD: %s | particles=%u\n",
                renderer->DeviceName().c_str(), P);

    render::RasterOptions opts;
    opts.width = args.width; opts.height = args.height;
    opts.draw_ground = true;
    opts.hero_framing = false;
    opts.use_camera_override = true;
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 38.0f;
    opts.background = {16, 19, 26, 255};
    opts.ground_color[0] = 0.185f; opts.ground_color[1] = 0.195f; opts.ground_color[2] = 0.215f;
    opts.contact_shadow_strength = 0.0f;  // explicit contact_points drive the shadow.

    constexpr float kPi = 3.14159265358979323846f;
    auto smoothstep = [](float t) {
        t = std::max(0.0f, std::min(1.0f, t));
        return t * t * (3.0f - 2.0f * t);
    };

    std::filesystem::create_directories(args.out_dir);
    std::filesystem::create_directories(args.png_dir);

    // Per-frame: orbit the hero camera around the rig + place a contact-shadow decal
    // under the presser footprint (faded by its height above the cloth).
    uint32_t total_render_steps = 0u;
    for (const Segment& ph : segments) if (ph.render) total_render_steps += ph.steps;

    uint32_t written = 0u, render_step = 0u;
    size_t first_nonbg = 0u, last_nonbg = 0u;
    std::vector<Vec3> pos(P, Vec3::Zero());

    auto download_pos = [&]() {
        world.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(),
                                      P * sizeof(Vec3));
    };
    auto render_frame = [&](const Vec3& presser_pos, const std::string* png) {
        download_pos();
        Transform pt = Transform::Identity(); pt.position = presser_pos;
        FrameScene fs = BuildFrame(cm, pos, pt);

        // Hero camera: a slow orbit framing the rig, dipping low as the press lands.
        const float t = total_render_steps > 1u
            ? static_cast<float>(render_step) / static_cast<float>(total_render_steps - 1u)
            : 0.0f;
        const float e = smoothstep(t);
        const float phi = -0.55f + 0.9f * e;            // arc around the rig.
        const float elev = (15.0f + 6.0f * std::sin(e * kPi)) * kPi / 180.0f;
        const float radius = 0.95f - 0.08f * e;
        const Vec3 look{0.0f, 0.0f, 0.11f};             // the drape center.
        opts.camera_eye = {look.x + radius * std::cos(elev) * std::sin(phi),
                           look.y - radius * std::cos(elev) * std::cos(phi),
                           look.z + radius * std::sin(elev)};
        opts.camera_target = look;

        // Contact-shadow decal under the presser, faded as it lifts off the cloth.
        opts.contact_points.clear();
        const float lift = presser_pos.z - bottom;      // 0 at deepest press.
        const float frac = std::max(0.0f, std::min(1.0f, 1.0f - lift / (top - bottom)));
        render::RasterOptions::ContactPoint cp;
        cp.x = presser_pos.x; cp.y = presser_pos.y;
        cp.radius = 0.10f + 0.05f * frac;
        cp.strength = 0.70f * (0.10f + 0.90f * frac);
        opts.contact_points.push_back(cp);

        render::VulkanOffscreenReport rep = renderer->Render(fs.rw, opts);
        char name[40];
        std::snprintf(name, sizeof(name), "frame_%06u.ppm", written);
        WritePpm(rep, args.out_dir + "/" + name);
        if (written == 0u) first_nonbg = rep.non_background_pixel_count;
        last_nonbg = rep.non_background_pixel_count;
        if (png) {
            if (WritePng(rep, *png))
                std::printf("[cloth_presser_demo] hero PNG -> %s\n", png->c_str());
            else
                std::fprintf(stderr, "[cloth_presser_demo] PNG encode failed: %s\n",
                             png->c_str());
        }
        ++written;
    };

    // Drive the segments; the three hero PNGs are captured inline at the moments they
    // depict (settled / deepest press-hold / recovered).
    const size_t kBeatSegment = 1u, kHoldSegment = 3u, kRelaxSegment = segments.size() - 1u;
    for (size_t pi = 0; pi < segments.size(); ++pi) {
        const Segment& ph = segments[pi];
        for (uint32_t s = 0; s < ph.steps; ++s) {
            const float frac = ph.steps > 1u
                ? static_cast<float>(s) / static_cast<float>(ph.steps - 1u) : 1.0f;
            const float z = ph.z0 + (ph.z1 - ph.z0) * frac;
            DrivePresser(world, Vec3{0.0f, 0.0f, z});
            world.Step();
            if (!ph.render) continue;
            const bool last = (s + 1u == ph.steps);
            std::string png_path;
            const std::string* png = nullptr;
            if (pi == kBeatSegment && s == 0u) png_path = args.png_dir + "/01_settled.png";
            else if (pi == kHoldSegment && last) png_path = args.png_dir + "/02_pressed.png";
            else if (pi == kRelaxSegment && last) png_path = args.png_dir + "/03_recovered.png";
            if (!png_path.empty()) png = &png_path;
            const bool do_frame = (render_step % args.stride) == 0u;
            if (do_frame || png) render_frame(Vec3{0, 0, z}, png);
            if (args.probe && png) {  // probe: one hero still, then exit.
                std::printf("[cloth_presser_demo] PROBE %s first_nonbg=%zu\n",
                            png_path.c_str(), first_nonbg);
                return first_nonbg > 0u ? 0 : 5;
            }
            ++render_step;
        }
    }

    std::printf("[cloth_presser_demo] DONE: %u frames -> %s (first_nonbg=%zu "
                "last_nonbg=%zu)\n",
                written, args.out_dir.c_str(), first_nonbg, last_nonbg);
    return written > 0u ? 0 : 6;
}
