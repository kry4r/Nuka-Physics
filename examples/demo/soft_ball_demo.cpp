// ---------------------------------------------------------------------------
// soft_ball_demo.cpp -- soft tet ball slams onto the ground (two-way coupling).
//
// A real XPBD tetrahedral soft body (a sphere rest-lattice tessellated into tets
// with per-edge distance + per-tet VOLUME constraints) is dropped onto a static
// ground plane. Its particles are SPHERE collidables flowing through the SAME
// body<->particle row solver a foot uses on the ground and a cloth particle uses:
// detection -> body<->particle narrowphase -> emit -> SolveRowsBlockIsland ->
// Xpbd finalize compose. There is NO bespoke "soft solid" coupler -- a tet body
// is the SAME particle array as a cloth, only its INTERNAL XPBD constraints differ
// (volume vs bend). The ball squashes on impact, the floor pushes back (body-side
// lambda > 0), the volume constraint recovers the shape, and it rests above 0.
//
// PIPELINE (physics on CUDA GPU 0, render on the CUDA RT backend or lavapipe):
//   build ONE nk::World (ground plane + the sphere-tet body cooked via
//   CookXpbdParticles) -> drop it under gravity. The coarse tet cage is invisible:
//   a high-res smooth icosphere skin is barycentrically embedded in the rest cage
//   ONCE (runtime::soft::EmbedSurfaceInTetCage) and deforms with it. PER FRAME:
//   download the live particle positions, DeformEmbeddedSkin -> smooth skin verts,
//   build the ball as a DEFORMING surface mesh via runtime::soft::BuildSurfaceMesh
//   over the SKIN triangles (smooth recomputed normals), render the ball + a studio
//   floor with a crisp key light and a contracting contact shadow, write a PPM. At
//   three key frames (airborne / max-squash / recovered) also write a hero PNG.
//
// Built behind NK_BUILD_VULKAN_VALIDATION. Usage:
//   soft_ball_demo [--width W] [--height H] [--out-dir DIR] [--png-dir DIR]
//                  [--stride S] [--gpu] [--beauty] [--flat-gpu] [--samples N]
//                  [--shadow-map PX] [--probe]
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
#include "render/rt_adapter.hpp"
#include "render/rt_backend.hpp"
#include "render/rt_framebuffer_to_report.hpp"
#include "runtime/soft/particle_surface.hpp"
#include "runtime/soft/tetmesh_topology.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
namespace render = nuka::render;
namespace rt = nuka::rt;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kKindPlane = nuka::collision::kShapePlane;

// Soft ball geometry: a sphere rest-lattice (cells clipped to a radius) split into
// the parity-alternating 5 tets per cell. ~0.10 m radius, light, dropped onto z=0.
constexpr float kBallRadius = 0.10f;
constexpr uint32_t kCells = 12u;            // grid cells per axis spanning the sphere (coarse cage; smooth skin pass).
constexpr float kCellLen = 2.0f * kBallRadius / static_cast<float>(kCells);
constexpr float kDropHeight = 0.16f;        // ball center starts this far above 0 (gentle drop, calm recovery).
constexpr float kParticleMass = 0.01f;      // 10 g per vertex (light soft body).
constexpr float kContactDMin = 0.020f;      // particle sphere radius = d_min/2 = 0.010; seats the surface on the floor.
constexpr float kRadius = kContactDMin * 0.5f;   // body<->particle sphere radius.
constexpr uint16_t kXpbdIters = 40u;
constexpr float kVolAlpha = 1.5e-6f;        // bouncy-but-stable volume: squash, lift off, settle round.
constexpr uint32_t kFloorBody = 0u;         // only collidable is the ground plane.

// ---- CLI ------------------------------------------------------------------
struct Args {
    uint32_t width = 1920u;
    uint32_t height = 1080u;
    uint32_t stride = 2u;          // render every stride-th physics step.
    bool gpu = false;              // CUDA RT backend (else lavapipe raster).
    bool beauty = true;            // GPU beauty path (--flat-gpu turns it off).
    uint32_t samples = 16u;        // beauty MSAA samples per pixel.
    uint32_t shadow_map = 2560u;
    std::string out_dir = "/tmp/soft_ball_frames";
    std::string png_dir = "out/soft_ball_demo";
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
        else if (s == "--gpu") a.gpu = true;
        else if (s == "--flat-gpu") { a.gpu = true; a.beauty = false; }
        else if (s == "--samples") a.samples = std::max(1u, next_u(a.samples));
        else if (s == "--shadow-map") a.shadow_map = std::max(256u, next_u(a.shadow_map));
        else if (s == "--probe") a.probe = true;
        else if (s == "--out-dir" && i + 1 < argc) a.out_dir = argv[++i];
        else if (s == "--png-dir" && i + 1 < argc) a.png_dir = argv[++i];
    }
    return a;
}

// ---- ball cook: a sphere rest-lattice -> tet distance + volume constraints --
// The ball center starts at z = center_z; the boundary triangles drive the render.
cook::XpbdCookInput BuildBallInput(const soft::TetLattice& lat, float center_z) {
    std::vector<Vec3> init = lat.rest;
    for (Vec3& p : init) p.z += center_z;

    soft::TetMeshTopologyOptions opts;
    opts.distance_compliance_alpha = 3.0e-6f;  // soft, dissipative edges -> stable jelly that rounds back.
    opts.volume_compliance_alpha = kVolAlpha;  // bouncy-but-stable volume -> squash, lift, settle round.
    opts.emit_distance_constraints = true;
    soft::XpbdConstraintSet cs;
    soft::BuildTetMeshConstraints(lat.rest, lat.tets, opts, cs);

    cook::XpbdCookInput in;
    in.positions = init;
    in.velocities.assign(init.size(), Vec3::Zero());
    in.inv_mass.assign(init.size(), 1.0f / kParticleMass);  // free (unpinned) solid.
    for (const auto& dc : cs.distance) {
        cook::CookDistanceCon c;
        c.a = dc.particle_a; c.b = dc.particle_b;
        c.rest_length = dc.rest_length; c.compliance_alpha = dc.compliance_alpha;
        in.distance.push_back(c);
    }
    for (const auto& vc : cs.volume) {
        cook::CookVolumeCon c;
        for (uint32_t j = 0; j < 4u; ++j) c.p[j] = vc.particle[j];
        c.rest_volume_times6 = vc.rest_volume_times6;
        c.compliance_alpha = vc.compliance_alpha;
        in.volume.push_back(c);
    }
    in.solver_iterations = kXpbdIters;
    in.friction = 0.6f;
    return in;
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

// Cook the ground plane + the soft ball into ONE Model; CookXpbdParticles grows the
// rigid budget. floor_present=false sinks the plane far below (free-fall control).
nk::Model BuildScene(const soft::TetLattice& lat, float center_z, bool floor_present) {
    nk::Model m;
    AddGroundPlane(m, static_cast<int32_t>(kFloorBody));

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = bodies * 4u;   // rigid budget (cook sizing).
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    cook::CookXpbdParticles(m, 1u, BuildBallInput(lat, center_z));
    m.particles.pp_contact_d_min = kContactDMin;
    if (!floor_present) m.body_init[kFloorBody].pose.position.z = -1000.0f;
    return m;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u; cfg.pos_iters = 8u;
    cfg.max_pairs = 64u;
    return cfg;
}

// NkRow packs to 32 f32: [0]=flags [7]=upper, a.kind [16] a.index [17],
// b.kind [24] b.index [25].
struct RowSides { uint32_t a_kind, b_kind; bool active; float upper; };
RowSides DecodeRow(const std::vector<float>& urows, uint32_t row) {
    const float* r = urows.data() + static_cast<size_t>(row) * 32u;
    auto u = [&](int i) { uint32_t v; std::memcpy(&v, &r[i], 4); return v; };
    RowSides s;
    s.active = (u(0) & 1u) != 0u;
    s.upper = r[7];
    s.a_kind = u(16); s.b_kind = u(24);
    return s;
}

// Vertical extent (max_z - min_z) of the soft body -- compresses when squashed.
float Extent(const std::vector<Vec3>& pos, uint32_t n) {
    float lo = 1.0e9f, hi = -1.0e9f;
    for (uint32_t i = 0; i < n; ++i) { lo = std::min(lo, pos[i].z); hi = std::max(hi, pos[i].z); }
    return hi - lo;
}

float MinZ(const std::vector<Vec3>& pos, uint32_t n) {
    float lo = 1.0e9f;
    for (uint32_t i = 0; i < n; ++i) lo = std::min(lo, pos[i].z);
    return lo;
}

// Vertical-to-horizontal extent ratio (1 == round; <1 squashed; >1 stretched).
float Sphericity(const std::vector<Vec3>& pos, uint32_t n) {
    float zlo = 1e9f, zhi = -1e9f, xlo = 1e9f, xhi = -1e9f, ylo = 1e9f, yhi = -1e9f;
    for (uint32_t i = 0; i < n; ++i) {
        zlo = std::min(zlo, pos[i].z); zhi = std::max(zhi, pos[i].z);
        xlo = std::min(xlo, pos[i].x); xhi = std::max(xhi, pos[i].x);
        ylo = std::min(ylo, pos[i].y); yhi = std::max(yhi, pos[i].y);
    }
    const float vert = zhi - zlo;
    const float horiz = 0.5f * ((xhi - xlo) + (yhi - ylo));
    return horiz > 1e-6f ? vert / horiz : 1.0f;
}

// Mean per-tet signed volume over the live positions (rest -> live ratio = squash).
float MeanTetVolume(const std::vector<Vec3>& pos,
                    const std::vector<soft::TetMeshTet>& tets) {
    double acc = 0.0;
    for (const soft::TetMeshTet& t : tets)
        acc += std::fabs(soft::TetSignedVolumeTimes6(
            pos[t.v[0]], pos[t.v[1]], pos[t.v[2]], pos[t.v[3]]));
    return static_cast<float>(acc / static_cast<double>(tets.size()));
}

// ---- render helpers -------------------------------------------------------
nuka::scene::RenderMaterial MakeMat(float r, float g, float b, float metallic,
                                    float rough, float transmission = 0.0f,
                                    float ior = 1.0f, float ax = 0.0f,
                                    float ay = 0.0f, float az = 0.0f) {
    nuka::scene::RenderMaterial m;
    m.base_color[0] = r; m.base_color[1] = g; m.base_color[2] = b;
    m.base_color[3] = 1.0f;
    m.metallic = metallic; m.roughness = rough;
    m.transmission = transmission; m.ior = ior;
    m.absorption[0] = ax; m.absorption[1] = ay; m.absorption[2] = az;
    return m;
}

render::MeshGeometry MakeFloorGeo(float half, float thickness) {
    render::MeshGeometry g;
    const float xs[2] = {-half, half};
    const float ys[2] = {-half, half};
    const float zs[2] = {-thickness, 0.0f};  // top face at z = 0.
    for (int xi = 0; xi < 2; ++xi)
        for (int yi = 0; yi < 2; ++yi)
            for (int zi = 0; zi < 2; ++zi)
                g.positions.insert(g.positions.end(), {xs[xi], ys[yi], zs[zi]});
    auto v = [](int x, int y, int z) -> uint32_t {
        return static_cast<uint32_t>(x * 4 + y * 2 + z);
    };
    const uint32_t faces[6][4] = {
        {v(0,0,0), v(0,1,0), v(0,1,1), v(0,0,1)}, {v(1,0,0), v(1,0,1), v(1,1,1), v(1,1,0)},
        {v(0,0,0), v(0,0,1), v(1,0,1), v(1,0,0)}, {v(0,1,0), v(1,1,0), v(1,1,1), v(0,1,1)},
        {v(0,0,0), v(1,0,0), v(1,1,0), v(0,1,0)}, {v(0,0,1), v(0,1,1), v(1,1,1), v(1,0,1)},
    };
    for (auto& f : faces)
        g.indices.insert(g.indices.end(), {f[0], f[1], f[2], f[0], f[2], f[3]});
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
    // Encode via ffmpeg from a temp PPM (the renderer has no PNG writer); ffmpeg is
    // on PATH from the .sh. Falls back to leaving the PPM if ffmpeg fails.
    const std::string tmp = path + ".ppm";
    if (!WritePpm(rep, tmp)) return false;
    const std::string cmd = "/usr/bin/ffmpeg -y -loglevel error -i \"" + tmp +
                            "\" \"" + path + "\" 2>/dev/null";
    const int rc = std::system(cmd.c_str());
    std::error_code ec;
    if (rc == 0) std::filesystem::remove(tmp, ec);
    return rc == 0;
}

// ---- CUDA ray-traced backend (the GPU beauty path) ------------------------
// The ball mesh DEFORMS every frame, so the per-mesh BLAS is rebuilt each frame
// (a rigid scene could reuse it); smooth normals keep the deforming surface smooth.
class GpuRenderer {
public:
    explicit GpuRenderer(const render::RenderWorld& rw) {
        backend_ = render::CreateCudaRtBackend();
        if (!backend_) return;
        scene_ = render::RenderWorldToTwoLevelScene(rw);
        handle_ = backend_->BuildScene(scene_);
    }
    ~GpuRenderer() { if (backend_ && handle_) backend_->FreeScene(handle_); }
    bool ok() const { return backend_ != nullptr && handle_ != nullptr; }
    void SetBeauty(bool on, uint32_t samples) { beauty_ = on; samples_ = samples; }

    render::VulkanOffscreenReport Render(const render::RenderWorld& rw,
                                         const render::RasterOptions& opts) {
        // Rebuild the scene (per-mesh BLAS over the LIVE deformed vertices) each
        // frame; BuildScene bakes geometry, so a deforming mesh needs a fresh handle.
        scene_ = render::RenderWorldToTwoLevelScene(rw);
        if (handle_) backend_->FreeScene(handle_);
        handle_ = backend_->BuildScene(scene_);
        ApplyLighting(opts);
        const rt::PinholeCamera cam = CameraFromOptions(opts);
        rt::Framebuffer fb;
        if (beauty_)
            fb = backend_->TraceBeautyToHost(handle_, scene_, cam, BeautyFromOptions(opts));
        else
            fb = backend_->TraceToHost(handle_, scene_, cam);
        return render::FramebufferToReport(fb, opts.background);
    }

private:
    void ApplyLighting(const render::RasterOptions& opts) {
        // RasterOptions.sun_direction points TOWARD the sun; rt::Light.direction is
        // the travel direction (away from the sun) -> negate.
        rt::Light& l = scene_.light;
        l.directional = true;
        Vec3 to_sun{opts.sun_direction[0], opts.sun_direction[1], opts.sun_direction[2]};
        if (to_sun.Length() > 1e-6f) to_sun = to_sun.Normalized();
        l.direction = -to_sun;
        l.color = {opts.sun_color[0], opts.sun_color[1], opts.sun_color[2]};
        l.intensity = 1.0f;
        scene_.ambient.color = {
            0.5f * (opts.sun_ambient_sky[0] + opts.sun_ambient_ground[0]),
            0.5f * (opts.sun_ambient_sky[1] + opts.sun_ambient_ground[1]),
            0.5f * (opts.sun_ambient_sky[2] + opts.sun_ambient_ground[2])};
    }

    rt::PinholeCamera CameraFromOptions(const render::RasterOptions& opts) {
        const float vfov = opts.camera_fov_degrees * (kPi / 180.0f);
        return rt::BuildPinhole(opts.camera_eye, opts.camera_target, opts.camera_up,
                                vfov, opts.width, opts.height);
    }

    rt::BeautyOptions BeautyFromOptions(const render::RasterOptions& opts) {
        rt::BeautyOptions b;
        b.samples = samples_;
        b.shadow_rays = (opts.shadow_strength > 0.0f) ? 6u : 1u;
        b.sun_angular_radius = 0.045f;
        b.gi_bounces = 1u;
        b.ao_samples = 4u;
        b.ao_radius = 0.7f;
        b.seed = 0x9e3779b9u;
        b.smooth_normals = true;  // the deforming ball shades smoothly, not faceted.
        b.sky_top = {opts.sky_top[0], opts.sky_top[1], opts.sky_top[2]};
        b.sky_bottom = {opts.sky_bottom[0], opts.sky_bottom[1], opts.sky_bottom[2]};
        b.sky_ground = {opts.ground_color[0], opts.ground_color[1], opts.ground_color[2]};
        b.fog_color = {opts.fog_color[0], opts.fog_color[1], opts.fog_color[2]};
        b.fog_density = opts.fog_density;
        b.sky_intensity = 0.30f;
        b.download = rt::AovDownloadMask{};
        b.download.depth = false; b.download.normal = false;
        b.download.albedo = false; b.download.uv = false;
        return b;
    }

    std::unique_ptr<render::RtBackendI> backend_;
    render::RtSceneHandle* handle_ = nullptr;
    rt::TwoLevelScene scene_;
    bool beauty_ = false;
    uint32_t samples_ = 16u;
};

// One renderable frame: the deformed ball (smooth embedded skin from the live cage
// via the ONE general builder) + a glossy studio floor whose top is z=0. skin_pos
// are the skin vertices already deformed by the cage this frame.
render::RenderWorld BuildFrame(const soft::SurfaceTopology& topo,
                               const std::vector<Vec3>& skin_pos) {
    render::RenderWorld rw;
    // Glossy opaque rubber ball (crisp silhouette + sharp key highlight reading
    // against the grey floor) on a glossy studio floor that softly mirrors it.
    rw.materials.push_back(MakeMat(0.78f, 0.10f, 0.13f, 0.0f, 0.28f));  // 0 rubber
    rw.materials.push_back(MakeMat(0.16f, 0.17f, 0.20f, 0.1f, 0.22f));  // 1 floor
    rw.default_material_id = 0u;

    render::MeshGeometry ball_geo;
    soft::BuildSurfaceMesh(skin_pos, topo, ball_geo);
    const uint32_t ball_mesh =
        rw.meshes.InternPrimitive("ball:live", [&] { return ball_geo; });
    const uint32_t floor_mesh =
        rw.meshes.InternPrimitive("floor:box", [&] { return MakeFloorGeo(2.5f, 0.10f); });

    auto add = [&](uint32_t mesh, uint32_t mat, const Transform& xf) {
        render::RenderInstance inst;
        inst.mesh_id = mesh; inst.render_material_id = mat; inst.world_xform = xf;
        inst.pose_source.kind = render::PoseSource::Kind::Static;
        rw.instances.push_back(inst);
    };
    add(ball_mesh, 0u, Transform::Identity());   // ball verts are world-space.
    // Seat the visible floor a hair below the contact plane so the bottom skin
    // reads as a clean resting silhouette, never a serrated floor-clip fringe.
    Transform floor_xf = Transform::Identity();
    floor_xf.position.z = -0.010f;
    add(floor_mesh, 1u, floor_xf);
    return rw;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);

    nphi::Device* dev = nphi::InitBestDevice();
    nphi::Backend* backend = dev ? nphi::DeviceInitBackend(dev, nullptr) : nullptr;
    if (backend == nullptr) {
        std::fprintf(stderr, "[soft_ball_demo] no CUDA backend\n");
        return 2;
    }

    // The sphere rest-lattice drives physics; a high-res icosphere SKIN embedded in
    // the rest cage ONCE renders instead, so the coarse voxel cage stays invisible.
    const soft::TetLattice lat =
        soft::BuildSphereTetLattice(Vec3{0, 0, 0}, kBallRadius, kCells, kCellLen);
    const soft::TriMesh skin_rest =
        soft::BuildIcosphere(Vec3{0, 0, kDropHeight}, kBallRadius, 3u);
    std::vector<Vec3> cage_rest = lat.rest;          // cage in the live particle frame.
    for (Vec3& p : cage_rest) p.z += kDropHeight;
    uint32_t skin_extrapolated = 0u;
    const soft::EmbeddedSkin skin = soft::EmbedSurfaceInTetCage(
        skin_rest.positions, skin_rest.triangles, cage_rest, lat.tets, &skin_extrapolated);
    soft::SurfaceTopology topo;
    topo.normal_offset = 0.0f;   // the skin already sits at the full ball radius.
    topo.triangles = skin.triangles;

    nk::Model model = BuildScene(lat, kDropHeight, /*floor_present=*/true);
    const uint32_t P = model.capacities.particles_per_env;
    const uint32_t n_soft = static_cast<uint32_t>(lat.rest.size());
    nk::World world(std::move(model), 1u, dev, backend, Cfg());
    if (!world.Ready()) {
        std::fprintf(stderr, "[soft_ball_demo] world not ready\n");
        return 3;
    }
    std::printf("[soft_ball_demo] sphere tets=%zu cage_verts=%u skin_verts=%zu skin_tris=%zu "
                "extrapolated=%u particles=%u\n",
                lat.tets.size(), n_soft, skin_rest.positions.size(),
                skin.triangles.size() / 3u, skin_extrapolated, P);

    std::vector<Vec3> pos(P, Vec3::Zero());
    auto download_pos = [&]() {
        world.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(),
                                      P * sizeof(Vec3));
    };
    // The cage particle indices ARE the embedded skin's global vertex indices;
    // deform the smooth skin from the live cage each frame, then a light Laplacian
    // pass removes coarse-cage corrugation from the contact rim (smooth jelly skin).
    std::vector<Vec3> skin_pos;
    auto deform_skin = [&]() {
        soft::DeformEmbeddedSkin(skin, pos, skin_pos);
        soft::SmoothSurface(topo.triangles, 4u, 0.40f, skin_pos);
    };

    const cook::XpbdCookInput rest_in = BuildBallInput(lat, kDropHeight);
    const float rest_extent = Extent(rest_in.positions, n_soft);
    const float rest_volume = MeanTetVolume(rest_in.positions, lat.tets);

    // ---- PROBE: the proof-test invariants at demo scale, no render ----------
    if (args.probe) {
        const uint32_t rows = world.GetModel().capacities.max_rows_per_env;
        std::vector<float> urows(static_cast<size_t>(rows) * 32u, 0.0f);
        std::vector<float> lambda(rows, 0.0f);
        nk::Data& d = world.GetData();
        float max_body_lambda = 0.0f, min_extent_during = 1.0e9f;
        float min_volume_during = 1.0e9f;  // the deepest compression (the squash).
        uint32_t particle_rows_seen = 0u, max_rows_active = 0u;
        for (uint32_t s = 0; s < 360u; ++s) {
            world.Step();
            d.DownloadField(nk::FieldId::Urows, urows.data(), urows.size() * sizeof(float));
            d.DownloadField(nk::FieldId::Lambda, lambda.data(), lambda.size() * sizeof(float));
            uint32_t active_now = 0u;
            for (uint32_t row = 0u; row < rows; ++row) {
                const RowSides rs = DecodeRow(urows, row);
                if (!rs.active) continue;
                const bool a_part = rs.a_kind == nk::kNkSideParticle;
                const bool b_part = rs.b_kind == nk::kNkSideParticle;
                if (!(a_part || b_part)) continue;
                ++particle_rows_seen; ++active_now;
                if (rs.upper > 1.0e30f)  // a normal row (friction spokes cap upper to 0).
                    max_body_lambda = std::max(max_body_lambda, lambda[row]);
            }
            max_rows_active = std::max(max_rows_active, active_now);
            download_pos();
            min_extent_during = std::min(min_extent_during, Extent(pos, n_soft));
            min_volume_during = std::min(min_volume_during, MeanTetVolume(pos, lat.tets));
            if (s % 10u == 0u)
                std::printf("  [arc] s=%3u extent=%.4f min_z=%.4f vol=%.6f rows=%u\n",
                            s, Extent(pos, n_soft), MinZ(pos, n_soft),
                            MeanTetVolume(pos, lat.tets), active_now);
        }
        download_pos();
        const float landed_min_z = MinZ(pos, n_soft);
        const float landed_volume = min_volume_during;  // compressed at deepest squash.
        for (uint32_t s = 0; s < 240u; ++s) world.Step();
        download_pos();
        const float recovered_volume = MeanTetVolume(pos, lat.tets);
        const float recovered_extent = Extent(pos, n_soft);
        const float recovered_min_z = MinZ(pos, n_soft);

        nk::Model mc = BuildScene(lat, kDropHeight, /*floor_present=*/false);
        nk::World wc(std::move(mc), 1u, dev, backend, Cfg());
        for (uint32_t s = 0; s < 360u; ++s) wc.Step();
        std::vector<Vec3> ctrl(P, Vec3::Zero());
        wc.GetData().DownloadField(nk::FieldId::ParticlePos, ctrl.data(), P * sizeof(Vec3));
        const float control_min_z = MinZ(ctrl, n_soft);

        const float squash = rest_extent - min_extent_during;
        const bool deform = squash > 0.005f;
        const bool two_way = particle_rows_seen > 0u && max_body_lambda > 0.0f;
        const bool rests = std::fabs(landed_min_z) < kRadius + kContactDMin;
        const bool arrested = control_min_z < landed_min_z - 0.5f;
        const bool recovers = recovered_volume > landed_volume + 1.0e-6f &&
            std::fabs(recovered_volume - rest_volume) / rest_volume < 0.10f;
        const bool ok = deform && two_way && rests && arrested && recovers;
        std::printf("[soft_ball_demo] PROBE %s\n", ok ? "PASS" : "FAIL");
        std::printf("  deform: rest_extent=%.4f min_extent=%.4f squash=%.4f (>0.005=%d)\n",
                    rest_extent, min_extent_during, squash, deform);
        std::printf("  two-way: body_lambda=%.6f particle_rows_seen=%u max_rows_active=%u (>0=%d)\n",
                    max_body_lambda, particle_rows_seen, max_rows_active, two_way);
        std::printf("  rest-z: landed_min_z=%.4f radius=%.4f (|z|<%.4f=%d)\n",
                    landed_min_z, kRadius, kRadius + kContactDMin, rests);
        std::printf("  no-tunnel: control_min_z=%.4f (control<landed-0.5=%d)\n",
                    control_min_z, arrested);
        std::printf("  recover: rest_vol=%.5f landed_vol=%.5f recovered_vol=%.5f (=%d)\n",
                    rest_volume, landed_volume, recovered_volume, recovers);
        std::printf("  recovered shape: extent=%.4f (rest=%.4f) min_z=%.4f\n",
                    recovered_extent, rest_extent, recovered_min_z);
        return ok ? 0 : 5;
    }

    // ---- RENDER: dual backend on the SAME RenderWorld -----------------------
    download_pos();
    deform_skin();
    render::RenderWorld rw0 = BuildFrame(topo, skin_pos);
    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    std::unique_ptr<GpuRenderer> gpu;
    if (args.gpu) {
        gpu = std::make_unique<GpuRenderer>(rw0);
        if (!gpu->ok()) {
            std::fprintf(stderr, "[soft_ball_demo] --gpu: no CUDA RT backend\n");
            return 6;
        }
        gpu->SetBeauty(args.beauty, args.samples);
        std::printf("[soft_ball_demo] renderer: CUDA two-level ray tracer (GPU, %s)\n",
                    args.beauty ? "BEAUTY MSAA/AO/GI smooth-normals" : "flat sensor");
    } else {
        try {
            renderer = std::make_unique<render::VulkanRasterRenderer>();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[soft_ball_demo] no Vulkan device: %s\n", e.what());
            return 6;
        }
        std::printf("[soft_ball_demo] renderer ICD: %s\n", renderer->DeviceName().c_str());
    }

    render::RasterOptions opts;
    opts.width = args.width; opts.height = args.height;
    opts.draw_ground = false;  // an explicit floor mesh is in the scene (beauty has no floor).
    opts.hero_framing = false;
    opts.use_camera_override = true;
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 40.0f;
    opts.background = {200, 205, 214, 255};
    opts.ground_color[0] = 0.16f; opts.ground_color[1] = 0.17f; opts.ground_color[2] = 0.20f;
    opts.contact_shadow_strength = 0.0f;  // explicit contact point drives the shadow.

    // Crisp high white key + neutral studio sky; the contracting contact shadow is
    // the money beat (free in beauty, a raster decal tracking the lowest vertex).
    opts.use_sun_light = true;
    opts.sun_direction[0] = 0.30f; opts.sun_direction[1] = -0.50f; opts.sun_direction[2] = 0.60f;
    opts.sun_color[0] = 3.0f; opts.sun_color[1] = 2.9f; opts.sun_color[2] = 2.7f;
    opts.sun_ambient_sky[0] = 0.16f; opts.sun_ambient_sky[1] = 0.18f; opts.sun_ambient_sky[2] = 0.22f;
    opts.sun_ambient_ground[0] = 0.10f; opts.sun_ambient_ground[1] = 0.10f; opts.sun_ambient_ground[2] = 0.11f;
    opts.shadow_strength = 0.95f;
    opts.shadow_map_size = args.shadow_map;
    opts.shadow_bias = 0.0020f;
    opts.sky_gradient = true;
    opts.sky_top[0] = 0.50f; opts.sky_top[1] = 0.55f; opts.sky_top[2] = 0.62f;
    opts.sky_bottom[0] = 0.78f; opts.sky_bottom[1] = 0.80f; opts.sky_bottom[2] = 0.84f;

    auto smoothstep = [](float t) {
        t = std::max(0.0f, std::min(1.0f, t));
        return t * t * (3.0f - 2.0f * t);
    };

    std::filesystem::create_directories(args.out_dir);
    std::filesystem::create_directories(args.png_dir);

    // The drop, the slam, the settle: ~260 steps (1.1 s) covers airborne ->
    // max-squash -> rebound -> settle, the clean window the cage stays coherent in.
    constexpr uint32_t kSteps = 260u;
    uint32_t written = 0u, render_step = 0u;
    size_t first_nonbg = 0u, last_nonbg = 0u;
    int squash_frame = -1, recovered_frame = static_cast<int>(kSteps) - 1;
    int airborne_frame = 0;

    auto render_frame = [&](const std::string* png) {
        download_pos();
        deform_skin();
        render::RenderWorld rw = BuildFrame(topo, skin_pos);

        // Hero camera: the look point TRACKS the ball center (framed through the
        // drop/slam/recover) with a slow orbit, push-in, and an impact elevation dip.
        const float t = kSteps > 1u
            ? static_cast<float>(render_step) / static_cast<float>(kSteps - 1u) : 0.0f;
        const float e = smoothstep(t);
        float bx = 0.0f, by = 0.0f, bz = 0.0f;
        for (uint32_t i = 0; i < n_soft; ++i) { bx += pos[i].x; by += pos[i].y; bz += pos[i].z; }
        bx /= n_soft; by /= n_soft; bz /= n_soft;
        const float az = -0.40f + 0.6f * e;
        const float elev = (19.0f - 4.0f * std::sin(e * kPi)) * kPi / 180.0f;
        const float radius = 0.82f - 0.08f * e;
        const Vec3 look{bx, by, std::max(kBallRadius * 0.8f, bz)};
        opts.camera_eye = {look.x + radius * std::cos(elev) * std::sin(az),
                           look.y - radius * std::cos(elev) * std::cos(az),
                           look.z + radius * std::sin(elev)};
        opts.camera_target = look;

        // The contracting contact shadow (raster path): footprint grows as the ball
        // pancakes, fades as it lifts, and tracks the ball's xy centroid.
        opts.contact_points.clear();
        float min_z = 1.0e9f, cxy_x = 0.0f, cxy_y = 0.0f;
        for (uint32_t i = 0; i < n_soft; ++i) {
            min_z = std::min(min_z, pos[i].z);
            cxy_x += pos[i].x; cxy_y += pos[i].y;
        }
        cxy_x /= static_cast<float>(n_soft); cxy_y /= static_cast<float>(n_soft);
        const float nearness = std::max(0.0f, std::min(1.0f, 1.0f - min_z / 0.20f));
        render::RasterOptions::ContactPoint cp;
        cp.x = cxy_x; cp.y = cxy_y;
        cp.radius = kBallRadius * (0.9f + 1.1f * nearness);
        cp.strength = 0.85f * (0.10f + 0.90f * nearness);
        opts.contact_points.push_back(cp);

        render::VulkanOffscreenReport rep =
            gpu ? gpu->Render(rw, opts) : renderer->Render(rw, opts);
        char name[40];
        std::snprintf(name, sizeof(name), "frame_%06u.ppm", written);
        WritePpm(rep, args.out_dir + "/" + name);
        if (written == 0u) first_nonbg = rep.non_background_pixel_count;
        last_nonbg = rep.non_background_pixel_count;
        if (png) {
            if (WritePng(rep, *png))
                std::printf("[soft_ball_demo] hero PNG -> %s\n", png->c_str());
            else
                std::fprintf(stderr, "[soft_ball_demo] PNG encode failed: %s\n", png->c_str());
        }
        ++written;
    };

    // A cheap physics-only pre-pass picks the hero frames from the live extent/min_z
    // arc: max-squash = the deepest extent, recovered = the cleanest rebound after it.
    {
        nk::Model mp = BuildScene(lat, kDropHeight, /*floor_present=*/true);
        nk::World wp(std::move(mp), 1u, dev, backend, Cfg());
        std::vector<Vec3> pp(P, Vec3::Zero());
        std::vector<float> ext_of(kSteps, 0.0f), minz_of(kSteps, 0.0f), sph_of(kSteps, 1.0f);
        for (uint32_t s = 0; s < kSteps; ++s) {
            wp.Step();
            wp.GetData().DownloadField(nk::FieldId::ParticlePos, pp.data(), P * sizeof(Vec3));
            ext_of[s] = Extent(pp, n_soft);
            minz_of[s] = MinZ(pp, n_soft);
            sph_of[s] = Sphericity(pp, n_soft);
        }
        // First contact = the first frame the ball reaches the floor; the dramatic
        // hero squash is the deepest compression of that FIRST impact (not a late
        // jiggle), seated (min_z >= 0) so no bottom pokes below the floor.
        uint32_t first_contact = 0u;
        while (first_contact < kSteps && minz_of[first_contact] > 0.02f) ++first_contact;
        const uint32_t sq_hi = std::min(kSteps, first_contact + 60u);
        float min_extent = 1.0e9f;
        for (uint32_t s = first_contact; s < sq_hi; ++s)
            if (minz_of[s] >= -0.001f && ext_of[s] < min_extent) {
                min_extent = ext_of[s]; squash_frame = static_cast<int>(s);
            }
        if (squash_frame < 0) {  // fallback: global deepest seated frame.
            for (uint32_t s = 0; s < kSteps; ++s)
                if (ext_of[s] < min_extent) { min_extent = ext_of[s]; squash_frame = static_cast<int>(s); }
        }
        // Airborne = the highest still-falling frame before the squash (clearly in
        // the air), so the drop reads even when the drop height is small.
        float best_air = -1.0e9f;
        for (int s = 0; s < squash_frame; ++s)
            if (minz_of[s] > best_air) { best_air = minz_of[s]; airborne_frame = s; }
        // Recovered = a SETTLED frame in the back half: round (sphericity ~1, not a
        // stretched vertical wobble), seated cleanly (min_z >= 0), and CALM (low
        // frame-to-frame sphericity change), so it reads as a resting jelly blob.
        const uint32_t lo = std::max(static_cast<uint32_t>(squash_frame) + 1u, kSteps / 2u);
        float best_settle = -1.0e9f;
        for (uint32_t s = lo + 1u; s < kSteps; ++s) {
            if (minz_of[s] < -0.001f) continue;                  // skip penetrating frames
            const float round = 1.0f - std::fabs(sph_of[s] - 1.0f);   // 1 == perfectly round
            const float seated = 1.0f - std::min(1.0f, std::fabs(minz_of[s]) / 0.03f);
            const float calm = 1.0f - std::min(1.0f, std::fabs(sph_of[s] - sph_of[s - 1u]) / 0.05f);
            const float score = round + 0.4f * seated + 0.6f * calm;
            if (score > best_settle) { best_settle = score; recovered_frame = static_cast<int>(s); }
        }
        std::printf("[soft_ball_demo] hero frames: airborne=%d squash=%d (extent=%.4f) "
                    "recovered=%d (extent=%.4f sph=%.3f min_z=%.4f) rest=%.4f\n",
                    airborne_frame, squash_frame, min_extent, recovered_frame,
                    ext_of[recovered_frame], sph_of[recovered_frame],
                    minz_of[recovered_frame], rest_extent);
    }

    for (uint32_t s = 0; s < kSteps; ++s) {
        world.Step();
        const std::string* png = nullptr;
        std::string png_path;
        if (static_cast<int>(s) == airborne_frame) { png_path = args.png_dir + "/01_airborne.png"; png = &png_path; }
        else if (static_cast<int>(s) == squash_frame) { png_path = args.png_dir + "/02_squash.png"; png = &png_path; }
        else if (static_cast<int>(s) == recovered_frame) { png_path = args.png_dir + "/03_recovered.png"; png = &png_path; }
        if ((render_step % args.stride) == 0u || png) render_frame(png);
        ++render_step;
    }

    download_pos();
    std::printf("[soft_ball_demo] DONE: %u frames -> %s (first_nonbg=%zu last_nonbg=%zu) "
                "rest_z=%.4f\n",
                written, args.out_dir.c_str(), first_nonbg, last_nonbg, MinZ(pos, n_soft));
    return written > 0u ? 0 : 7;
}
