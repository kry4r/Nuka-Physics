// ---------------------------------------------------------------------------
// cloth_presser_demo.cpp -- THE FIRST body<->particle coupling headline clip.
//
// A real XPBD cloth (24x24 = 576 particles) is pinned along its two far edges to
// two posts held closer than its natural width, so the slack hangs as a deep
// catenary hammock above a studio floor. A rigid sphere is released above the
// belly and falls under GRAVITY into the sag: the cloth CATCHES it and cradles it
// in a fabric pocket (friction + the two-way non-penetration rows), no scripted
// descent and no clip-through. The cloth particles are SPHERE collidables flowing
// through the SAME body<->particle row solver a foot uses on the ground
// (detection -> body<->particle narrowphase -> SolveRowsBlockIsland -> finalize
// compose): there is NO bespoke coupler, the medium difference is data, never code.
//
// PIPELINE (physics on CUDA GPU 0, render offscreen on lavapipe):
//   build ONE nk::World (ground plane + two posts + a DYNAMIC presser sphere +
//   the cloth cooked via CookXpbdParticles) -> settle the hammock with the presser
//   parked away -> teleport the presser above center at rest -> let gravity drop
//   it into the pocket. PER FRAME: download the live particle positions + presser
//   pose, build the cloth as a DEFORMING surface mesh via the general
//   runtime::soft::BuildSurfaceMesh over the cooked lattice topology (smooth
//   recomputed normals), render cloth + presser + posts with a studio floor + an
//   orbiting hero camera + per-contact shadow decals, write a PPM. At three key
//   frames (settled / cradled / rest) also write a hero PNG.
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
#include "runtime/soft/particle_surface.hpp"
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

// Cloth + scene geometry. A 24x24 sheet is pinned along its two far x-edges to
// the tops of two posts held CLOSER than the cloth's natural width, so the slack
// in between hangs as a deep catenary hammock. A sphere dropped into the belly is
// cradled by the fabric pocket (the cloth is continuous below it -> no fall-
// through), the two-way reaction holding it against gravity.
constexpr uint32_t kGridN = 24u;
constexpr float kSpacing = 0.025f;     // patch ~0.575 m square (flat rest size).
constexpr float kClothZ = 0.30f;       // initial flat lay height of the free interior.
constexpr float kParticleMass = 0.02f; // 20 g per particle (holds the presser).
constexpr float kContactDMin = 0.020f; // particle sphere radius = d_min/2 = 0.010 (< spacing -> no self puff).
constexpr float kAnchorX = 0.18f;      // pinned edges held at x = +/-0.18 (cloth half-width 0.2875 -> slack).
constexpr float kAnchorZ = 0.30f;      // edges pinned at this height -> the belly hangs below.
constexpr float kPostHalfX = 0.03f;
constexpr float kPostHalfY = 0.32f;
constexpr float kPostHalfZ = 0.15f;    // post tops at z = 0.30 (the anchor height).
constexpr float kPostTopZ = kPostHalfZ * 2.0f;
constexpr float kPresserRadius = 0.06f;
constexpr float kPresserMass = 0.15f;             // dynamic presser the cloth catches.
constexpr float kSurfaceOffset = 0.5f * kContactDMin;  // render inflation = particle radius.
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

// ---- cloth cook topology (built once, reused by physics + the render mesh) --
// The cloth lattice triangle list: two triangles per quad cell. This SAME index
// list cooks the XPBD constraints and drives runtime::soft::BuildSurfaceMesh.
soft::SurfaceTopology MakeClothTopology() {
    soft::SurfaceTopology topo;
    topo.normal_offset = kSurfaceOffset;
    auto idx = [](uint32_t i, uint32_t j) { return j * kGridN + i; };
    for (uint32_t j = 0; j + 1 < kGridN; ++j)
        for (uint32_t i = 0; i + 1 < kGridN; ++i) {
            const uint32_t v00 = idx(i, j), v10 = idx(i + 1, j);
            const uint32_t v01 = idx(i, j + 1), v11 = idx(i + 1, j + 1);
            topo.triangles.insert(topo.triangles.end(),
                                  {v00, v10, v11, v00, v11, v01});
        }
    return topo;
}

// The FLAT rest grid: the cloth's natural size. Distance-constraint rest lengths
// come from this so the cloth is inextensible at its true dimensions; the belly
// slack is created by pinning the edges CLOSER than this width, not by stretching.
std::vector<Vec3> MakeClothRest() {
    std::vector<Vec3> rest;
    rest.reserve(kGridN * kGridN);
    const float c0 = -0.5f * static_cast<float>(kGridN - 1u) * kSpacing;
    for (uint32_t j = 0; j < kGridN; ++j)
        for (uint32_t i = 0; i < kGridN; ++i)
            rest.push_back(Vec3{c0 + static_cast<float>(i) * kSpacing,
                                c0 + static_cast<float>(j) * kSpacing, kClothZ});
    return rest;
}

cook::XpbdCookInput BuildClothInput(const std::vector<Vec3>& rest,
                                    const soft::SurfaceTopology& topo) {
    std::vector<soft::ClothTriangle> tris;
    for (size_t t = 0; t + 2 < topo.triangles.size(); t += 3)
        tris.push_back(soft::ClothTriangle{
            {topo.triangles[t], topo.triangles[t + 1], topo.triangles[t + 2]}});
    soft::ClothTopologyOptions opts;
    opts.distance_compliance_alpha = 0.0f;
    opts.bend_compliance_alpha = 5.0e-4f;  // soft bend so it drapes into a pocket.
    soft::XpbdConstraintSet cs;
    soft::BuildClothConstraints(rest, tris, opts, cs);

    // Initial positions: pin the two far x-columns to the post tops, pulled inward
    // to +/-kAnchorX (closer than the flat width) and up to kAnchorZ, so the slack
    // interior bellies into a deep hammock as it settles under gravity.
    std::vector<Vec3> init = rest;
    auto idx = [](uint32_t i, uint32_t j) { return j * kGridN + i; };
    const uint32_t last = kGridN - 1u;
    for (uint32_t j = 0; j < kGridN; ++j) {
        init[idx(0, j)] = Vec3{-kAnchorX, rest[idx(0, j)].y, kAnchorZ};
        init[idx(last, j)] = Vec3{+kAnchorX, rest[idx(last, j)].y, kAnchorZ};
    }

    cook::XpbdCookInput in;
    in.positions = init;
    in.velocities.assign(rest.size(), Vec3{0.0f, 0.0f, -0.05f});
    in.inv_mass.assign(rest.size(), 1.0f / kParticleMass);
    for (uint32_t j = 0; j < kGridN; ++j) {  // pin the two anchored edge columns.
        in.inv_mass[idx(0, j)] = 0.0f;
        in.inv_mass[idx(last, j)] = 0.0f;
    }
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

// A DYNAMIC sphere (inv_mass > 0) the contact reaction can move: gravity pulls
// it into the sag and the cloth catches it. inv_inertia = the solid-sphere diag.
void AddDynamicSphere(nk::Model& m, const Vec3& pos, float radius, float mass,
                      int32_t body_id) {
    const float inv_mass = 1.0f / mass;
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = inv_mass;
    const float ii = inv_mass / (0.4f * radius * radius);
    bi.inv_inertia = Vec3{ii, ii, ii};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindSphere;
    sh.params[0] = radius; sh.params[1] = radius; sh.params[2] = radius;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

nk::Model BuildScene(const std::vector<Vec3>& rest,
                     const soft::SurfaceTopology& topo, const Vec3& presser_pos) {
    nk::Model m;
    const Vec3 post_half{kPostHalfX, kPostHalfY, kPostHalfZ};
    AddGroundPlane(m, 0);
    AddStaticBox(m, Vec3{-kAnchorX, 0.0f, kPostHalfZ}, post_half, 1);
    AddStaticBox(m, Vec3{+kAnchorX, 0.0f, kPostHalfZ}, post_half, 2);
    AddDynamicSphere(m, presser_pos, kPresserRadius, kPresserMass, 3);

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = bodies * 4u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;
    cook::CookXpbdParticles(m, 1u, BuildClothInput(rest, topo));
    m.particles.pp_contact_d_min = kContactDMin;
    return m;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 48u; cfg.pos_iters = 0u;  // stiff enough for the mass ratio.
    cfg.max_pairs = 64u;
    return cfg;
}

// Teleport the dynamic presser to `pos` at rest (zero velocity) -- used once to
// drop it above the settled drape center, after which gravity drives it.
void PlacePresser(nk::World& w, const Vec3& pos) {
    Transform tf = Transform::Identity();
    tf.position = pos;
    const Vec3 zero = Vec3::Zero();
    w.GetData().UploadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), kPoseOff);
    w.GetData().UploadField(nk::FieldId::BodyLinearVelocity, &zero, sizeof(Vec3), kVec3Off);
    w.GetData().UploadField(nk::FieldId::BodyAngularVelocity, &zero, sizeof(Vec3), kVec3Off);
}

Vec3 ReadPresser(nk::World& w) {
    Transform tf = Transform::Identity();
    w.GetData().DownloadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), kPoseOff);
    return tf.position;
}

// Min particle z within reach of (px,py) -- the pressed pocket depth metric.
float MinZUnder(const std::vector<Vec3>& pos, float px, float py) {
    float min_z = 1.0e9f;
    const float reach = kPresserRadius + 4.0f * kSpacing;
    for (const Vec3& p : pos) {
        const float dx = p.x - px, dy = p.y - py;
        if (dx * dx + dy * dy <= reach * reach) min_z = std::min(min_z, p.z);
    }
    return min_z;
}

// Cloth surface z directly beneath the presser axis (nearest particle in xy),
// for the clip check: the presser bottom must stay above this.
float ClothZUnder(const std::vector<Vec3>& pos, float px, float py) {
    float best_d2 = 1.0e9f, best_z = kClothZ;
    for (const Vec3& p : pos) {
        const float dx = p.x - px, dy = p.y - py, d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; best_z = p.z; }
    }
    return best_z;
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

// One renderable frame: cloth (deformed) + presser sphere + two posts, on a
// studio floor, with a soft contact-shadow decal under the presser footprint.
struct FrameScene {
    render::RenderWorld rw;
};

FrameScene BuildFrame(const soft::SurfaceTopology& topo,
                      const std::vector<Vec3>& cloth_pos, const Transform& presser) {
    FrameScene fs;
    render::RenderWorld& rw = fs.rw;
    rw.materials.push_back(MakeMat(0.55f, 0.13f, 0.16f, 0.0f, 0.72f));  // 0 cloth crimson
    rw.materials.push_back(MakeMat(0.62f, 0.64f, 0.70f, 0.85f, 0.30f));  // 1 presser steel
    rw.materials.push_back(MakeMat(0.12f, 0.13f, 0.16f, 0.10f, 0.65f));  // 2 post charcoal
    rw.default_material_id = 0u;

    // The cloth: a single-sided deformed surface mesh from the live particles via
    // the ONE general builder (smooth recomputed normals, no inline triangulation).
    render::MeshGeometry cloth_geo;
    soft::BuildSurfaceMesh(cloth_pos, topo, cloth_geo);
    const uint32_t cloth_mesh =
        rw.meshes.InternPrimitive("cloth:live", [&] { return cloth_geo; });
    const uint32_t sphere_mesh = rw.meshes.InternPrimitive(
        "presser:sphere", [&] { return MakeSphereGeo(kPresserRadius); });
    const Vec3 ph{kPostHalfX, kPostHalfY, kPostHalfZ};
    const uint32_t box_mesh =
        rw.meshes.InternPrimitive("post:box", [&] { return MakeBoxGeo(ph); });

    auto add = [&](uint32_t mesh, uint32_t mat, const Transform& xf) {
        render::RenderInstance inst;
        inst.mesh_id = mesh; inst.render_material_id = mat; inst.world_xform = xf;
        inst.pose_source.kind = render::PoseSource::Kind::Static;
        rw.instances.push_back(inst);
    };
    add(cloth_mesh, 0u, Transform::Identity());  // cloth verts are world-space.
    add(sphere_mesh, 1u, presser);
    Transform pa = Transform::Identity(); pa.position = Vec3{-kAnchorX, 0, kPostHalfZ};
    Transform pb = Transform::Identity(); pb.position = Vec3{+kAnchorX, 0, kPostHalfZ};
    add(box_mesh, 2u, pa);
    add(box_mesh, 2u, pb);
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

    const soft::SurfaceTopology topo = MakeClothTopology();
    const std::vector<Vec3> rest = MakeClothRest();
    // Park the dynamic presser far in xy during settle so it falls to the ground
    // away from the cloth; it is teleported above the center after the drape sets.
    nk::Model model = BuildScene(rest, topo, Vec3{2.0f, 2.0f, kPresserRadius});
    const uint32_t P = model.capacities.particles_per_env;
    nk::World world(std::move(model), 1u, dev, backend, Cfg());
    if (!world.Ready()) {
        std::fprintf(stderr, "[cloth_presser_demo] world not ready\n");
        return 3;
    }

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

    std::vector<Vec3> pos(P, Vec3::Zero());
    auto download_pos = [&]() {
        world.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(),
                                      P * sizeof(Vec3));
    };

    // The choreography: SETTLE the drape (presser parked away) -> a beat on the
    // settled sag -> RELEASE the dynamic presser above center and let gravity drop
    // it into the pocket (CATCH) -> a long hold while it cradles + cloth springs.
    constexpr uint32_t kSettle = 700u, kBeat = 90u, kCatch = 360u, kRest = 320u;
    uint32_t total_render_steps = kBeat + kCatch + kRest;

    uint32_t written = 0u, render_step = 0u;
    size_t first_nonbg = 0u, last_nonbg = 0u;
    Vec3 presser_pos = ReadPresser(world);

    auto render_frame = [&](const std::string* png) {
        download_pos();
        Transform pt = Transform::Identity(); pt.position = presser_pos;
        FrameScene fs = BuildFrame(topo, pos, pt);

        // Hero camera: a slow orbit framing the rig, dipping low as the press lands.
        const float t = total_render_steps > 1u
            ? static_cast<float>(render_step) / static_cast<float>(total_render_steps - 1u)
            : 0.0f;
        const float e = smoothstep(t);
        const float az = -0.55f + 0.9f * e;            // arc around the rig.
        const float elev = (16.0f + 7.0f * std::sin(e * kPi)) * kPi / 180.0f;
        const float radius = 1.05f - 0.10f * e;
        const Vec3 look{0.0f, 0.0f, 0.19f};            // the hammock belly + posts.
        opts.camera_eye = {look.x + radius * std::cos(elev) * std::sin(az),
                           look.y - radius * std::cos(elev) * std::cos(az),
                           look.z + radius * std::sin(elev)};
        opts.camera_target = look;

        // Contact-shadow decal under the presser, faded by its height above the cloth.
        opts.contact_points.clear();
        const float cloth_z = ClothZUnder(pos, presser_pos.x, presser_pos.y);
        const float gap = std::max(0.0f, presser_pos.z - kPresserRadius - cloth_z);
        const float frac = std::max(0.0f, std::min(1.0f, 1.0f - gap / 0.20f));
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

    // SETTLE: cloth drapes into the gap while the presser falls to rest far away.
    for (uint32_t s = 0; s < kSettle; ++s) world.Step();
    download_pos();
    const float rest_min_z = MinZUnder(pos, 0.0f, 0.0f);
    const float sag = kAnchorZ - rest_min_z;  // how far the belly hangs below the pinned edges.
    std::printf("[cloth_presser_demo] settled: anchor_z=%.4f post_top=%.4f "
                "belly_min_z=%.4f sag_depth=%.4f\n",
                kAnchorZ, kPostTopZ, rest_min_z, sag);

    // RELEASE: drop the dynamic presser from just above the sagging center.
    const float release_z = rest_min_z + kPresserRadius + 0.04f;
    PlacePresser(world, Vec3{0.0f, 0.0f, release_z});

    // A beat on the settled drape (presser hovering at release height, pre-fall).
    presser_pos = ReadPresser(world);
    if (!args.probe) {
        for (uint32_t s = 0; s < kBeat; ++s) {
            world.Step();
            presser_pos = ReadPresser(world);
            const std::string* png = nullptr;
            std::string png_path;
            if (s == 0u) { png_path = args.png_dir + "/01_settled.png"; png = &png_path; }
            if ((render_step % args.stride) == 0u || png) render_frame(png);
            ++render_step;
        }
    }

    // CATCH: gravity drops the presser into the sag; the cloth cradles it. The
    // deepest-cradle PNG is captured at the end of this phase.
    float deepest_min_z = rest_min_z, cradle_clearance = 1e9f, cradle_presser_z = release_z;
    const uint32_t catch_steps = args.probe ? (kCatch + kRest) : kCatch;
    for (uint32_t s = 0; s < catch_steps; ++s) {
        world.Step();
        presser_pos = ReadPresser(world);
        download_pos();
        const float mz = MinZUnder(pos, presser_pos.x, presser_pos.y);
        if (mz < deepest_min_z) {
            deepest_min_z = mz;
            cradle_presser_z = presser_pos.z;
            cradle_clearance = (presser_pos.z - kPresserRadius) - mz;
        }
        if (args.probe) continue;
        const bool last = (s + 1u == catch_steps);
        const std::string* png = nullptr;
        std::string png_path;
        if (last) { png_path = args.png_dir + "/02_pressed.png"; png = &png_path; }
        if ((render_step % args.stride) == 0u || png) render_frame(png);
        ++render_step;
    }

    // REST: the presser cradled in the pocket; render the settled hero, then exit.
    if (!args.probe) {
        for (uint32_t s = 0; s < kRest; ++s) {
            world.Step();
            presser_pos = ReadPresser(world);
            const bool last = (s + 1u == kRest);
            const std::string* png = nullptr;
            std::string png_path;
            if (last) { png_path = args.png_dir + "/03_recovered.png"; png = &png_path; }
            if ((render_step % args.stride) == 0u || png) render_frame(png);
            ++render_step;
        }
    }

    // Final clip/cradle measurement: the presser bottom vs the cloth surface
    // beneath it (positive => cradled in a pocket, no clip-through).
    download_pos();
    presser_pos = ReadPresser(world);
    const float final_cloth_z = ClothZUnder(pos, presser_pos.x, presser_pos.y);
    const float final_clearance = (presser_pos.z - kPresserRadius) - final_cloth_z;
    std::printf("[cloth_presser_demo] caught: deepest_min_z=%.4f "
                "presser_z=%.4f cradle_clearance=%.4f final_clearance=%.4f "
                "(sag_depth=%.4f)\n",
                deepest_min_z, presser_pos.z, cradle_clearance, final_clearance, sag);

    if (args.probe) {
        // PASS = visible droop below the flat lay, the presser actually descended,
        // it did NOT punch through to the ground, and its bottom is cradled at the
        // cloth surface (clearance within the render inflation band, no clip).
        const bool caught_above_ground = presser_pos.z > kPresserRadius + 0.02f;
        const bool descended = presser_pos.z < release_z - 0.01f;
        const bool no_clip = final_clearance > -kSurfaceOffset;
        const bool ok = sag > 0.03f && descended && caught_above_ground && no_clip;
        std::printf("[cloth_presser_demo] PROBE %s sag=%.4f drop=%.4f "
                    "presser_z=%.4f clear=%.4f\n",
                    ok ? "PASS" : "FAIL", sag, release_z - presser_pos.z,
                    presser_pos.z, final_clearance);
        return ok ? 0 : 5;
    }

    std::printf("[cloth_presser_demo] DONE: %u frames -> %s (first_nonbg=%zu "
                "last_nonbg=%zu)\n",
                written, args.out_dir.c_str(), first_nonbg, last_nonbg);
    return written > 0u ? 0 : 6;
}
