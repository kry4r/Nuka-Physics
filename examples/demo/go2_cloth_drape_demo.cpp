// ---------------------------------------------------------------------------
// go2_cloth_drape_demo.cpp -- cloth<->rigid coupling headline clip.
//
// A free, UNPINNED square cloth sheet is released just above a PD-standing Go2
// quadruped and DRAPES over the whole body: it conforms to the trunk, thighs,
// calves and feet (the full per-link collision skeleton cooked from go2.nks),
// caught by the two-way non-penetration rows + friction, no clip-through. The
// cloth particles are SPHERE collidables flowing through the SAME body<->particle
// row solver a foot uses on the ground (detection -> body<->particle narrowphase
// -> SolveRowsBlockIsland -> finalize compose). The robot LINKS are ordinary
// collidable body rows posed by FK each step: there is NO bespoke coupler and no
// per-scene path, the medium difference is data, never code.
//
// PIPELINE (physics on CUDA GPU 0, render offscreen on lavapipe / CUDA RT):
//   nks::Load(go2.nks) -> CookToModel(scene,1,{PairDriven}) (full collision
//   skeleton) -> append the free cloth via CookSoftFluidParticles (soft-only) ->
//   build ONE nk::World. Seed the standing crouch (Q + base height) + a PD hold
//   on the leg actuators, HOLD the cloth flat above the trunk while the dog
//   settles off-camera, then RELEASE the cloth and let gravity drape it. PER
//   FRAME: download the live particle positions (cloth) + the link poses (Go2
//   FK-rebind visuals), build the cloth as a DEFORMING surface mesh via the
//   general runtime::soft::BuildSurfaceMesh (smooth recomputed normals), render
//   the cloth + the Go2 visual meshes on a studio floor with a hero orbit camera,
//   write a PPM. Three key frames (falling / mid-drape / settled) also write a
//   hero PNG.
//
// The Go2 cooks from go2.nks for BOTH physics and visuals, so the physics link
// rows == the visual instance link rows by construction (no FK-rebind drift).
//
// Built behind NK_BUILD_VULKAN_VALIDATION (build-viewer / lavapipe). Usage:
//   go2_cloth_drape_demo [--width W] [--height H] [--out-dir DIR] [--png-dir DIR]
//                        [--stride S] [--gpu] [--beauty] [--samples N] [--probe]
// ---------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
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
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "runtime/soft/cloth_topology.hpp"
#include "runtime/soft/particle_surface.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"
#include "scene/terrain/heightfield.hpp"
#include "scene/terrain/heightfield_loaders.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
namespace render = nuka::render;
namespace rt = nuka::rt;
namespace terrain = nuka::terrain;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr const char* kNksPath = "examples/scenes/go2.nks";
constexpr float kPi = 3.14159265358979323846f;

// ---- free cloth dropped over the standing Go2. A LARGE fine sheet that fully
// covers the dog and drapes to the floor on all four sides: it is released at a
// slight tilt just ABOVE the dog and FLUTTERS DOWN under gravity + anisotropic air
// drag (waves, edge-curl, sway), then conforms to the trunk, thighs, calves and
// feet, hem pooling on the ground (which anchors it against sliding).
constexpr uint32_t kGridNx = 55u;                // along the body (x): ~1.30 m, hem reaches the floor front + rear.
constexpr uint32_t kGridNy = 51u;                // across the body (y): ~1.20 m, wings hang to the floor both sides.
constexpr float kSpacing = 0.024f;               // 1.296 m (x) x 1.200 m (y); fine enough for smooth folds.
constexpr float kClothShiftX = 0.0f;             // centred on the trunk (world x=0, verified from the link poses).
constexpr float kClothShiftY = 0.0f;             // centred across the trunk (world y=0); a bias offset if needed.
constexpr float kParticleMass = 0.012f;          // per-particle mass; enough inertia to settle symmetrically.
// Contact radius set above half-spacing so neighbouring particle spheres overlap and
// bridge the dog's narrow raised joints -- the cloth rides OVER them, no poke-through.
constexpr float kContactDMin = 0.044f;           // particle sphere r=22 mm.
constexpr float kBendAlpha = 9.0e-2f;            // soft folds drape DOWN over the edges; no sharp kinks, no tent.
constexpr uint16_t kXpbdIters = 80u;
// Render inflation is DECOUPLED from the contact radius and kept small so proud
// particles do not amplify into sharp visual spikes (the surface is also Laplacian
// relaxed before normals/inflation, see kSurfaceSmoothIters).
constexpr float kSurfaceOffset = 0.004f;         // 4 mm render skin.
constexpr uint32_t kSurfaceSmoothIters = 3u;     // Laplacian relax passes (render-only).
constexpr float kSurfaceSmoothLambda = 0.55f;    // per-pass blend weight.
// Release height above the standing crouch base: the sheet's lowest released point
// clears the dog top (z~0.470) by >30 cm so the fall has room to develop flutter.
constexpr float kClothLift = 0.58f;
// Per-step contact push-out velocity cap (velocity aref bias + split-impulse position
// projection): bounded so a deep particle climbs out gradually, no one-step fling.
constexpr float kDepenetrationVel = 0.45f;       // m/s (~1.9 mm/step push-out).

// The Go2 standing crouch the leg actuators hold (per LINK; index 0 = the floating
// base, indices 1..12 = FL,FR,RL,RR x (hip,thigh,calf)). These are the nominal
// stance the trained walk policy starts from; a symmetric crouch whose feet span a
// stable support polygon, so the floating base rests statically on the ground.
constexpr float kStand[13] = {0.0f,
                              -0.10f, 0.80f, -1.50f,   // FL hip,thigh,calf
                               0.10f, 0.80f, -1.50f,   // FR
                              -0.10f, 0.80f, -1.50f,   // RL
                               0.10f, 0.80f, -1.50f};  // RR
constexpr float kStandBaseZ = 0.32f;             // base height for the crouch (feet ~ ground).
constexpr float kDriveStiff = 60.0f;             // PD position gain (motor actuators seed 0).

// ---- CLI ------------------------------------------------------------------
struct Args {
    uint32_t width = 1920u;
    uint32_t height = 1080u;
    uint32_t stride = 2u;
    std::string out_dir = "/tmp/go2_cloth_frames";
    std::string png_dir = "out/go2_cloth_demo";
    bool gpu = false;
    bool beauty = false;
    uint32_t samples = 16u;
    uint32_t drape = 800u;  // drape sim/render steps: full fall -> drape -> settle.
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
        else if (s == "--samples") a.samples = next_u(a.samples);
        else if (s == "--drape") a.drape = std::max(60u, next_u(a.drape));
        else if (s == "--gpu") a.gpu = true;
        else if (s == "--beauty") { a.beauty = true; a.gpu = true; }
        else if (s == "--probe") a.probe = true;
        else if (s == "--out-dir" && i + 1 < argc) a.out_dir = argv[++i];
        else if (s == "--png-dir" && i + 1 < argc) a.png_dir = argv[++i];
    }
    if (const char* g = std::getenv("NK_GPU_RENDER"); g && std::atoi(g) != 0) a.gpu = true;
    return a;
}

// ---- cloth cook topology (built once, reused by physics + the render mesh) --
// The cloth lattice triangle list: two triangles per quad cell, single-winding.
// The SAME index list cooks the XPBD constraints and drives BuildSurfaceMesh.
// (Single-sided: the beauty path faceforwards the smooth normal so the drape
// shades from both sides; doubling the winding over the SHARED particle vertices
// would cancel the area-weighted normals.)
soft::SurfaceTopology MakeClothTopology() {
    soft::SurfaceTopology topo;
    topo.normal_offset = kSurfaceOffset;
    topo.smooth_iters = kSurfaceSmoothIters;
    topo.smooth_lambda = kSurfaceSmoothLambda;
    if (const char* o = std::getenv("NK_SURF_OFFSET")) topo.normal_offset = std::atof(o);
    if (const char* si = std::getenv("NK_SURF_SMOOTH")) topo.smooth_iters = std::atoi(si);
    auto idx = [](uint32_t i, uint32_t j) { return j * kGridNx + i; };
    for (uint32_t j = 0; j + 1 < kGridNy; ++j)
        for (uint32_t i = 0; i + 1 < kGridNx; ++i) {
            const uint32_t v00 = idx(i, j), v10 = idx(i + 1, j);
            const uint32_t v01 = idx(i, j + 1), v11 = idx(i + 1, j + 1);
            topo.triangles.insert(topo.triangles.end(),
                                  {v00, v10, v11, v00, v11, v01});
        }
    return topo;
}

// The XPBD constraint triangles (same single winding as the render topology).
std::vector<soft::ClothTriangle> ClothConstraintTris() {
    std::vector<soft::ClothTriangle> tris;
    auto idx = [](uint32_t i, uint32_t j) { return j * kGridNx + i; };
    for (uint32_t j = 0; j + 1 < kGridNy; ++j)
        for (uint32_t i = 0; i + 1 < kGridNx; ++i) {
            const uint32_t v00 = idx(i, j), v10 = idx(i + 1, j);
            const uint32_t v01 = idx(i, j + 1), v11 = idx(i + 1, j + 1);
            tris.push_back(soft::ClothTriangle{{v00, v10, v11}});
            tris.push_back(soft::ClothTriangle{{v00, v11, v01}});
        }
    return tris;
}

// The FLAT rest grid centred at (cx,cy) at height z: the cloth's natural size.
std::vector<Vec3> MakeClothRest(float cx, float cy, float z) {
    std::vector<Vec3> rest;
    rest.reserve(kGridNx * kGridNy);
    const float cx0 = -0.5f * static_cast<float>(kGridNx - 1u) * kSpacing;
    const float cy0 = -0.5f * static_cast<float>(kGridNy - 1u) * kSpacing;
    for (uint32_t j = 0; j < kGridNy; ++j)
        for (uint32_t i = 0; i < kGridNx; ++i)
            rest.push_back(Vec3{cx + cx0 + static_cast<float>(i) * kSpacing,
                                cy + cy0 + static_cast<float>(j) * kSpacing, z});
    return rest;
}

cook::XpbdCookInput BuildClothInput(const std::vector<Vec3>& rest) {
    soft::ClothTopologyOptions opts;
    opts.distance_compliance_alpha = 0.0f;  // inextensible.
    float bend_alpha = kBendAlpha;  // bend compliance (higher = softer fold).
    if (const char* b = std::getenv("NK_BEND_ALPHA")) bend_alpha = std::atof(b);
    opts.bend_compliance_alpha = bend_alpha;
    soft::XpbdConstraintSet cs;
    soft::BuildClothConstraints(rest, ClothConstraintTris(), opts, cs);

    cook::XpbdCookInput in;
    in.positions = rest;
    in.velocities.assign(rest.size(), Vec3::Zero());
    in.inv_mass.assign(rest.size(), 1.0f / kParticleMass);  // uniform, NO pins.
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
    // High Coulomb grip so the drape seats and does not slide; the cone bound is
    // mu*normal-lambda (no mu<=1 clamp), so mu>1 is honoured. NK_CLOTH_MU overrides.
    in.friction = 1.8f;
    if (const char* mu = std::getenv("NK_CLOTH_MU")) in.friction = std::atof(mu);

    // ANISOTROPIC AERODYNAMIC DRAG: the cloth surface triangles + normal-dominant
    // coeffs. A flat sheet released in vacuum descends rigidly (every particle sees
    // the same g); the orientation-dependent normal drag breaks that symmetry into
    // flutter (waves, edge-curl, sway) while the tangent drag stays small so the
    // sheet still slides through the air. Coeffs are lumped 0.5*rho*C; env-tunable.
    for (const soft::ClothTriangle& t : ClothConstraintTris())
        in.aero_triangles.push_back({t.v[0], t.v[1], t.v[2]});
    in.aero_drag_normal  = 30.0f;   // 0.5*rho*Cn (normal-dominant -> floaty + flutter).
    in.aero_drag_tangent = 0.12f;   // 0.5*rho*Ct (small -> air still slips by).
    in.aero_drag_max_dv  = 0.16f;   // per-step impulse clamp (quadratic stability).
    if (const char* an = std::getenv("NK_AERO_N")) in.aero_drag_normal = std::atof(an);
    if (const char* at = std::getenv("NK_AERO_T")) in.aero_drag_tangent = std::atof(at);
    if (const char* ac = std::getenv("NK_AERO_CLAMP")) in.aero_drag_max_dv = std::atof(ac);
    return in;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    // A margin forms contact rows during APPROACH (within the SDF narrow band ~0.048)
    // so a fast-falling particle is arrested BEFORE it penetrates -- anti-tunnelling.
    cfg.contact_margin = 0.044f;
    if (const char* m = std::getenv("NK_CONTACT_MARGIN")) cfg.contact_margin = std::atof(m);
    // Split-impulse position projection (pos_iters>0) pushes penetrating cloth
    // particles back OUT of the dog; with 0 the sheet sinks through the links.
    cfg.vel_iters = 80u; cfg.pos_iters = 28u;
    cfg.max_pairs = 64u;  // rigid body<->body broadphase only; cloth uses per-particle slots.
    return cfg;
}

// ---- render helpers -------------------------------------------------------
nuka::scene::RenderMaterial Mk(float r, float g, float b, float metallic, float rough,
                               float sheen = 0.0f) {
    nuka::scene::RenderMaterial m;
    m.base_color[0] = r; m.base_color[1] = g; m.base_color[2] = b; m.base_color[3] = 1.0f;
    m.metallic = metallic; m.roughness = rough; m.sheen = sheen;
    return m;
}

render::MeshGeometry MakeFloorGeo(float half, float z) {
    render::MeshGeometry g;
    const float xs[2] = {-half, half};
    const float ys[2] = {-half, half};
    for (int yi = 0; yi < 2; ++yi)
        for (int xi = 0; xi < 2; ++xi)
            g.positions.insert(g.positions.end(), {xs[xi], ys[yi], z});
    g.indices.insert(g.indices.end(), {0u, 1u, 3u, 0u, 3u, 2u});
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
    // Encode via ffmpeg from a temp PPM (lavapipe has no PNG writer; the .sh also
    // uses ffmpeg, so it is on PATH). Leaves the PPM if ffmpeg fails.
    const std::string tmp = path + ".ppm";
    if (!WritePpm(rep, tmp)) return false;
    const std::string cmd = "/usr/bin/ffmpeg -y -loglevel error -i \"" + tmp +
                            "\" \"" + path + "\" 2>/dev/null";
    const int rc = std::system(cmd.c_str());
    std::error_code ec;
    if (rc == 0) std::filesystem::remove(tmp, ec);
    return rc == 0;
}

// ---- CUDA ray-traced backend (the GPU render path; mirrors go2_terrain_demo's
// GpuRenderer but enables smooth_normals so the fabric wrinkles + the Go2 shell
// shade smoothly, not faceted). ---------------------------------------------
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
        // The cloth surface deforms every frame, so its BLAS is rebuilt: free the
        // prior handle and rebuild over the CURRENT meshes (offline by design).
        if (backend_ && handle_) backend_->FreeScene(handle_);
        scene_ = render::RenderWorldToTwoLevelScene(rw);
        handle_ = backend_->BuildScene(scene_);
        ApplyLighting(opts);
        const rt::PinholeCamera cam = CameraFromOptions(opts);
        rt::Framebuffer fb;
        if (beauty_) fb = backend_->TraceBeautyToHost(handle_, scene_, cam, BeautyFromOptions(opts));
        else fb = backend_->TraceToHost(handle_, scene_, cam);
        return render::FramebufferToReport(fb, opts.background);
    }

private:
    void ApplyLighting(const render::RasterOptions& opts) {
        rt::Light& l = scene_.light;
        l.directional = true;
        Vec3 to_sun{opts.sun_direction[0], opts.sun_direction[1], opts.sun_direction[2]};
        if (to_sun.Length() > 1e-6f) to_sun = to_sun.Normalized();
        l.direction = -to_sun;  // RasterOptions points TOWARD the sun; rt travels AWAY.
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
        b.smooth_normals = true;  // fabric wrinkles + shell shade smoothly, not faceted.
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

// NkRow packs to 32 f32: [0]=flags [7]=upper, a.kind [16] a.index [17],
// b.kind [24] b.index [25]. (Mirrors articulated_link_particle_coupling.)
struct RowSides {
    uint32_t a_kind, b_kind, a_index, b_index;
    bool active; float upper;
};
RowSides DecodeRow(const std::vector<float>& urows, uint32_t row) {
    const float* r = urows.data() + static_cast<size_t>(row) * 32u;
    auto u = [&](int i) { uint32_t v; std::memcpy(&v, &r[i], 4); return v; };
    RowSides s;
    s.active = (u(0) & 1u) != 0u;
    s.upper = r[7];
    s.a_kind = u(16); s.a_index = u(17);
    s.b_kind = u(24); s.b_index = u(25);
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered progress.
    const Args args = ParseArgs(argc, argv);

    if (!std::filesystem::exists(kNksPath)) {
        std::fprintf(stderr, "[go2_cloth_drape] missing scene %s\n", kNksPath);
        return 2;
    }

    nphi::Device* dev = nphi::InitBestDevice();
    nphi::Backend* backend = dev ? nphi::DeviceInitBackend(dev, nullptr) : nullptr;
    if (backend == nullptr) {
        std::fprintf(stderr, "[go2_cloth_drape] no CUDA backend\n");
        return 2;
    }

    // ---- cook the Go2 PHYSICS from go2.nks (full per-link collision skeleton) --
    const nuka::scene::SceneIR scene = nuka::scene::nks::Load(kNksPath);
    cook::CookToModelOptions copt;
    copt.contact_family = cook::CookContactFamily::PairDriven;
    // Bake a per-link SDF from each link's VISUAL mesh so the cloth particles ride
    // the TRUE silhouette (no clip-through, a rounded grippable ridge), not the
    // inset collision primitives. A no-op for any link without a visual trimesh.
    copt.bake_link_sdf = true;
    nk::Model model = cook::CookToModel(scene, 1, copt).model;

    const uint32_t L = model.capacities.links_per_env;
    const uint32_t bodies = model.capacities.bodies_per_env;
    const std::vector<uint32_t> body_to_link = model.body_to_link;
    const std::vector<nk::ModelFootShape> feet = model.feet;
    // Copied before the model is moved into the World; the probe measures the
    // particle->body penetration from these collision primitives + the link poses.
    const std::vector<nk::Model::PairDrivenShape> shapes = model.shape_table_rows;
    const std::vector<Transform> geom_local = model.articulation.link_geom_local;
    // The cooked per-body silhouette SDFs (host copy): the probe samples them with
    // the SAME sparse_sdf_sample the kernel uses, so its penetration metric measures
    // the TRUE surface the cloth rides (an SdfMesh row has no box/capsule params).
    const std::vector<nk::Model::SdfGrid> sdf_grids = model.sdf_grids;
    const std::vector<uint64_t> sdf_keys = model.sdf_cell_keys;
    const std::vector<float> sdf_values = model.sdf_cell_values;
    const std::vector<Vec3> sdf_grads = model.sdf_cell_gradients;
    uint32_t n_sdf_links = 0u;
    for (const auto& r : shapes)
        if (r.kind == static_cast<uint8_t>(nuka::collision::kShapeSdfMesh) &&
            r.sdf_grid != ~0u)
            ++n_sdf_links;
    std::printf("[go2_cloth_drape] cooked %u link visual-mesh SDFs (grids=%zu cells=%zu)\n",
                n_sdf_links, sdf_grids.size(), sdf_keys.size());
    if (std::getenv("NK_SDF_PROBE")) {
        namespace rsdf = nuka::runtime::sdf;
        // SIGN-CONSISTENCY check: the narrow-band sign is "closest-triangle plane"
        // (not winding number). For a MERGED non-watertight multi-shell mesh, an
        // interior cell can be closest to an internal/back-facing triangle and get
        // the WRONG sign. Report the band's negative/positive split AND the deepest
        // (most-negative) value -- a healthy solid SDF has a clear negative interior.
        for (size_t b = 0; b < shapes.size(); ++b) {
            const auto& r = shapes[b];
            if (r.kind != static_cast<uint8_t>(nuka::collision::kShapeSdfMesh) ||
                r.sdf_grid == ~0u || r.sdf_grid >= sdf_grids.size()) continue;
            const auto& g = sdf_grids[r.sdf_grid];
            uint32_t neg = 0u, pos = 0u; float vmin = 1e9f, vmax = -1e9f;
            for (uint32_t c = 0; c < g.cell_count; ++c) {
                const float v = sdf_values[g.cell_offset + c];
                if (v < 0.0f) ++neg; else ++pos;
                vmin = std::min(vmin, v); vmax = std::max(vmax, v);
            }
            std::printf("[sdf_probe] body %zu grid %u vox=%.4f dims=(%u,%u,%u) cells=%u "
                        "neg=%u pos=%u min=%.4f max=%.4f r=%.3f\n",
                        b, r.sdf_grid, g.voxel_size, g.dims[0], g.dims[1], g.dims[2],
                        g.cell_count, neg, pos, vmin, vmax, r.params[0]);
        }
    }
    if (const char* sd = std::getenv("NK_SHAPE_DUMP")) {
        std::printf("[shape_dump] %zu shape_table rows (one per body; kind 0=sph 1=cap 2=box 4=hull):\n",
                    model.shape_table_rows.size());
        for (size_t b = 0; b < model.shape_table_rows.size(); ++b) {
            const auto& r = model.shape_table_rows[b];
            std::printf("  body %zu kind=%u params=(%.3f,%.3f,%.3f) hull_v=%u\n",
                        b, r.kind, r.params[0], r.params[1], r.params[2], r.hull_vert_count);
        }
        // Write the per-body shape_table rows + the hull-vert pool for offline analysis.
        std::FILE* f = std::fopen(sd, "wb");
        if (f != nullptr) {
            const uint32_t nb = static_cast<uint32_t>(model.shape_table_rows.size());
            const uint32_t nhv = static_cast<uint32_t>(model.hull_verts.size());
            std::fwrite(&nb, sizeof(nb), 1, f);
            for (const auto& r : model.shape_table_rows) {
                std::fwrite(&r.kind, sizeof(r.kind), 1, f);
                std::fwrite(r.params, sizeof(float), 4, f);
                std::fwrite(&r.hull_vert_offset, sizeof(r.hull_vert_offset), 1, f);
                std::fwrite(&r.hull_vert_count, sizeof(r.hull_vert_count), 1, f);
            }
            std::fwrite(&nhv, sizeof(nhv), 1, f);
            std::fwrite(model.hull_verts.data(), sizeof(float), nhv, f);
            std::fclose(f);
        }
    }
    std::printf("[go2_cloth_drape] cooked go2.nks: links=%u bodies=%u feet=%zu\n",
                L, bodies, feet.size());

    // The trunk world xy (the cloth is centred over it); base seeded at the crouch.
    const Vec3 base_xy{model.articulation.base_pose.position.x,
                       model.articulation.base_pose.position.y, 0.0f};

    // ---- a flat ground the feet rest on (the cook's auto static plane is inert in
    // the general path; a flat heightfield collidable is the proven foot-on-ground
    // path, vproof_go2_ground). Seated at z=0; the dog is held at standing height. --
    model.body_init.resize(bodies);  // link body rows; the heightfield lands at `bodies`.
    {
        terrain::TerrainGenConfig tcfg;  // FLAT: every feature amplitude 0.
        tcfg.nrow = 21u; tcfg.ncol = 21u; tcfg.cell_x = 0.25f; tcfg.cell_y = 0.25f;
        tcfg.origin = Vec3{base_xy.x - 0.5f * 20.0f * 0.25f,
                           base_xy.y - 0.5f * 20.0f * 0.25f, 0.0f};
        tcfg.base_z = 0.0f;
        terrain::HeightField hf;
        terrain::GenerateHeightField(tcfg, hf);
        cook::CookHeightfieldGrid(model, hf);
    }

    // ---- seed the contact budget BEFORE the cloth cook so rigid_base != 0 (else
    // GrowContactBudgetForParticles early-returns and the cloth gets no rows) ----
    nk::ModelCapacities& cap = model.capacities;
    cap.bodies_per_env = static_cast<uint32_t>(model.body_init.size());
    cap.max_bodies_total = static_cast<uint32_t>(model.shape_table_rows.size());
    cap.max_contacts_per_env = cap.bodies_per_env * 4u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;

    // ---- append the free cloth (soft-only == byte-identical to CookXpbdParticles) --
    // The release y-offset (default 0 = centred). NK_CLOTH_SHIFT_Y biases the start
    // so the probe can run a directional-asymmetry test (start off-centre, measure
    // whether the drape recentres or keeps drifting on the rounded SDF ridge).
    float cloth_shift_y = kClothShiftY;
    if (const char* sy = std::getenv("NK_CLOTH_SHIFT_Y")) cloth_shift_y = std::atof(sy);
    float cloth_lift = kClothLift;  // release drop height; NK_CLOTH_LIFT sweeps it.
    if (const char* cl = std::getenv("NK_CLOTH_LIFT")) cloth_lift = std::atof(cl);
    const std::vector<Vec3> rest =
        MakeClothRest(base_xy.x + kClothShiftX, base_xy.y + cloth_shift_y, kStandBaseZ + cloth_lift);
    cook::CookSoftFluidParticles(model, 1u, BuildClothInput(rest), /*fluid*/ {});
    model.particles.pp_contact_d_min = kContactDMin;  // particle radius (0 post-cook tunnels).
    if (const char* dm = std::getenv("NK_CONTACT_DMIN")) model.particles.pp_contact_d_min = std::atof(dm);
    model.baumgarte_max_velocity = kDepenetrationVel;  // bounded contact recovery.
    if (const char* bv = std::getenv("NK_DEPEN_VEL")) model.baumgarte_max_velocity = std::atof(bv);
    const uint32_t P = model.capacities.particles_per_env;
    std::printf("[go2_cloth_drape] cloth particles=%u (grid %ux%u) d_min=%.4f "
                "rows/env=%u\n", P, kGridNx, kGridNy, model.particles.pp_contact_d_min,
                cap.max_rows_per_env);

    nk::World world(std::move(model), 1u, dev, backend, Cfg());
    if (!world.Ready()) {
        std::fprintf(stderr, "[go2_cloth_drape] world not ready\n");
        return 3;
    }
    nk::Data& data = world.GetData();

    // ---- PD-stand the Go2: seed the crouch q + base height, hold the leg joints
    // with a finite position gain (the cooked motor actuators seed stiffness 0) and
    // hold the floating base at standing height (re-upload each step). The dog reads
    // as a stable standing statue carrying its FULL per-link collision skeleton, the
    // surface the cloth drapes over. Holding the base is a control input (the public
    // UploadField, like the presser's PlacePresser), not a solver fork. ------------
    std::vector<float> q_stand(L, 0.0f), tgt(L, 0.0f), stiff(L, 0.0f), damp(L, 0.0f);
    for (uint32_t l = 0; l < L && l < 13u; ++l) { q_stand[l] = kStand[l]; tgt[l] = kStand[l]; }
    for (uint32_t l = 1u; l < L && l < 13u; ++l) { stiff[l] = kDriveStiff; damp[l] = 2.0f * std::sqrt(kDriveStiff); }
    Transform base = world.GetModel().articulation.base_pose;
    base.position.z = kStandBaseZ;
    const Vec3 base_zero = Vec3::Zero();
    data.UploadField(nk::FieldId::Q, q_stand.data(), L * sizeof(float));
    data.UploadField(nk::FieldId::DriveStiffness, stiff.data(), L * sizeof(float));
    data.UploadField(nk::FieldId::DriveDamping, damp.data(), L * sizeof(float));
    // Hold the stance + base every step (called inside both the settle and drape).
    // The floating-base velocity is the root link's spatial velocity (LinkVelocity
    // [0], 6 floats); zeroing it + re-stamping the base pose pins the trunk so the
    // dog stands as a stable statue carrying its full collision skeleton.
    const float root_vel_zero[6] = {0, 0, 0, 0, 0, 0};
    auto hold_dog = [&]() {
        data.UploadField(nk::FieldId::DriveTarget, tgt.data(), L * sizeof(float));
        data.UploadField(nk::FieldId::BasePose, &base, sizeof(Transform));
        data.UploadField(nk::FieldId::LinkVelocity, root_vel_zero, sizeof(root_vel_zero), 0u);
    };
    (void)base_zero;

    // Park the cloth HIGH (clear of the dog) while it settles so the held sheet
    // does not contact the trunk during settle (which would make every settle step
    // pay the contact solve); at release we drop it to the low release lay `rest`.
    // The re-upload (positions + zero velocity) keeps the sheet kinematically parked.
    constexpr float kParkLift = 0.45f;  // settle park height above the release lay.
    std::vector<Vec3> cloth_park = rest;
    for (Vec3& p : cloth_park) p.z += kParkLift;
    const std::vector<Vec3> cloth_zero(P, Vec3::Zero());
    auto hold_cloth = [&]() {
        data.UploadField(nk::FieldId::ParticlePos, cloth_park.data(), P * sizeof(Vec3));
        data.UploadField(nk::FieldId::ParticleVel, cloth_zero.data(), P * sizeof(Vec3));
    };
    // Release the sheet at a slight natural TILT + a faint sinusoidal height ripple
    // (nobody drops a cloth perfectly level/flat). This seeds the flutter the instant
    // it is released; the anisotropic normal drag then amplifies it into evolving
    // waves during the fall. Minimal + physical, not a baked wrinkle pattern. Zero vel.
    std::vector<Vec3> cloth_release = rest;
    {
        constexpr float kTiltRad = 4.0f * (kPi / 180.0f);  // ~4 deg about y.
        constexpr float kRippleAmp = 0.018f;               // faint ripple seed.
        const float tilt = std::tan(kTiltRad);
        float tilt_amt = tilt, ripple_amt = kRippleAmp;
        if (const char* t = std::getenv("NK_RELEASE_TILT")) tilt_amt = std::tan(std::atof(t) * (kPi / 180.0f));
        if (const char* r = std::getenv("NK_RELEASE_RIPPLE")) ripple_amt = std::atof(r);
        for (Vec3& p : cloth_release) {
            const float dx = p.x - base_xy.x;
            const float dy = p.y - base_xy.y;
            p.z += tilt_amt * dx + ripple_amt * std::sin(3.0f * kPi * dx) *
                                       std::cos(2.0f * kPi * dy);
        }
    }
    auto release_cloth = [&]() {
        data.UploadField(nk::FieldId::ParticlePos, cloth_release.data(), P * sizeof(Vec3));
        data.UploadField(nk::FieldId::ParticleVel, cloth_zero.data(), P * sizeof(Vec3));
    };

    std::vector<Vec3> pos(P, Vec3::Zero());
    auto download_pos = [&]() {
        data.DownloadField(nk::FieldId::ParticlePos, pos.data(), P * sizeof(Vec3));
    };
    // Per-step uniform velocity damping on the cloth (XPBD velocity is position-derived
    // with no internal drag): a uniform fabric-drag control input, not a solver fork.
    std::vector<Vec3> cloth_vel(P, Vec3::Zero());
    // Dissipates the drop's impact energy so the drape settles instead of sloshing; the
    // quadratic aero still dominates the fast-descent flutter, which this leaves intact.
    float kClothDamp = 0.10f;
    if (const char* d = std::getenv("NK_CLOTH_DAMP")) kClothDamp = std::atof(d);
    auto damp_cloth = [&]() {
        if (kClothDamp <= 0.0f) return;
        data.DownloadField(nk::FieldId::ParticleVel, cloth_vel.data(), P * sizeof(Vec3));
        const float s = 1.0f - kClothDamp;
        for (Vec3& v : cloth_vel) { v.x *= s; v.y *= s; v.z *= s; }
        data.UploadField(nk::FieldId::ParticleVel, cloth_vel.data(), P * sizeof(Vec3));
    };
    std::vector<Transform> link_pose(L, Transform::Identity());
    auto download_links = [&]() {
        data.DownloadField(nk::FieldId::LinkPose, link_pose.data(), L * sizeof(Transform));
    };

    // SETTLE the dog (cloth held flat above the trunk) off-camera. The base is
    // pinned and the legs PD-hold, so the stance is reached quickly.
    constexpr uint32_t kSettle = 120u;
    const auto settle_t0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0; s < kSettle; ++s) {
        hold_dog();
        hold_cloth();
        world.Step();
    }
    const double settle_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - settle_t0).count();
    std::printf("[go2_cloth_drape] settle %u steps in %.0f ms (%.1f ms/step incl JIT)\n",
                kSettle, settle_ms, settle_ms / kSettle);
    download_links();
    {
        // Report the settled stance: base height + the lowest foot z.
        float foot_lo = 1e9f;
        for (const auto& f : feet)
            if (f.calf_local_link < L) {
                const Vec3 c = link_pose[f.calf_local_link].TransformPoint(f.local_offset);
                foot_lo = std::min(foot_lo, c.z - f.radius);
            }
        Transform b; data.DownloadField(nk::FieldId::BasePose, &b, sizeof(Transform));
        std::printf("[go2_cloth_drape] settled: base_z=%.4f lowest_foot_z=%.4f\n",
                    b.position.z, foot_lo);
        // The dog's world AABB from each body's collision-shape extents posed by FK
        // (SDF grid corners for an SdfMesh, box/capsule extents otherwise): the
        // cloth release height is set clear above this highest point.
        float body_hi = -1e9f, body_lo = 1e9f;
        for (size_t bi = 0; bi < shapes.size(); ++bi) {
            const auto& s = shapes[bi];
            if (bi >= body_to_link.size() || body_to_link[bi] >= L) continue;
            const uint32_t lk = body_to_link[bi];
            const Transform wx = link_pose[lk] *
                (lk < geom_local.size() ? geom_local[lk] : Transform::Identity());
            Vec3 lo{0, 0, 0}, hi{0, 0, 0};
            if (s.kind == static_cast<uint8_t>(nuka::collision::kShapeSdfMesh) &&
                s.sdf_grid < sdf_grids.size()) {
                const auto& g = sdf_grids[s.sdf_grid];
                lo = g.origin;
                hi = Vec3{g.origin.x + g.dims[0] * g.voxel_size,
                          g.origin.y + g.dims[1] * g.voxel_size,
                          g.origin.z + g.dims[2] * g.voxel_size};
            } else {
                const float e = std::max(s.params[0], std::max(s.params[1], s.params[2]));
                lo = Vec3{-e, -e, -e}; hi = Vec3{e, e, e};
            }
            for (int c = 0; c < 8; ++c) {
                const Vec3 corner{(c & 1) ? hi.x : lo.x, (c & 2) ? hi.y : lo.y,
                                  (c & 4) ? hi.z : lo.z};
                const float wz = wx.TransformPoint(corner).z;
                body_hi = std::max(body_hi, wz); body_lo = std::min(body_lo, wz);
            }
        }
        std::printf("[go2_cloth_drape] dog world-z extent: lo=%.4f hi=%.4f\n",
                    body_lo, body_hi);
    }

    // Drop the parked cloth to the low release lay just above the trunk so it
    // settles onto the dog without a slamming impulse, then let gravity drape it.
    release_cloth();

    // ---- PROBE: drape, then report the body<->particle coupling per link --------
    if (args.probe) {
        const uint32_t rows = world.GetModel().capacities.max_rows_per_env;
        std::vector<float> urows(static_cast<size_t>(rows) * 32u, 0.0f);
        std::vector<float> lambda(rows, 0.0f);
        // The artic-side row index is the ARTICULATION id, not a body row; the
        // owning link is carried in the chain-J gather (row_cj_link[_b]). Resolve
        // the contacted link from there so an artic-link contact maps to its link.
        std::vector<uint32_t> cj_link_a(rows, ~0u), cj_link_b(rows, ~0u);
        std::vector<uint8_t> link_has_row(L, 0u);
        std::vector<float> link_max_lambda(L, 0.0f);
        uint32_t total_part_rows = 0u, body_lambda_pos = 0u;
        float max_nonfoot_lambda = 0.0f;
        // The calf link carries the foot geom (the foot box is a fixed child of the
        // calf body == same link); links 1..12 are FL,FR,RL,RR x (hip,thigh,calf), so
        // (link-1)%3==2 is a calf/foot link. A NON-foot link = the trunk (0) or a
        // hip/thigh -- a row there proves the cloth conforms over the body, not just
        // the feet.
        auto is_foot_link = [&](uint32_t l) {
            return l != 0u && l <= 12u && ((l - 1u) % 3u) == 2u;
        };

        // The contact rows persist across steps, so the heavy Urows/Lambda download
        // is sampled every few steps (not every step) to keep the probe quick. The
        // sheet contacts the trunk within ~30 steps of release, then folds down the
        // sides over the next few hundred to reach the thighs/calves (multi-link drape),
        // so the probe drapes the full clip length (--drape) and samples periodically.
        const uint32_t kDrape = args.drape;
        constexpr uint32_t kSample = 20u;
        // COM drift over the back half of the clip quantifies migration/collapse:
        // capture the cloth (x,y) centroid at mid-clip then compare to the end.
        Vec3 com_mid{0, 0, 0};
        bool com_mid_set = false;
        // Residual cloth motion (settled => near-zero particle speed; shaking keeps it
        // high): sampled at the periodic row download, back-half max == settled signal.
        std::vector<Vec3> vsamp(P, Vec3::Zero());
        float backhalf_speed_max = 0.0f, end_mean_speed = 0.0f, end_max_speed = 0.0f;
        const auto drape_t0 = std::chrono::steady_clock::now();
        for (uint32_t s = 0; s < kDrape; ++s) {
            hold_dog();
            world.Step();
            damp_cloth();
            if (!com_mid_set && s >= kDrape / 2u) {
                download_pos();
                for (const Vec3& p : pos) { com_mid.x += p.x; com_mid.y += p.y; com_mid.z += p.z; }
                com_mid = Vec3{com_mid.x / P, com_mid.y / P, com_mid.z / P};
                com_mid_set = true;
            }
            if ((s % kSample) != 0u && s + 1u != kDrape) continue;
            data.DownloadField(nk::FieldId::Urows, urows.data(), urows.size() * sizeof(float));
            data.DownloadField(nk::FieldId::Lambda, lambda.data(), lambda.size() * sizeof(float));
            data.DownloadField(nk::FieldId::RowCjLink, cj_link_a.data(), rows * sizeof(uint32_t));
            data.DownloadField(nk::FieldId::RowCjLinkB, cj_link_b.data(), rows * sizeof(uint32_t));
            data.DownloadField(nk::FieldId::ParticleVel, vsamp.data(), P * sizeof(Vec3));
            float msp = 0.0f, xsp = 0.0f;
            for (const Vec3& v : vsamp) {
                const float sp = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
                msp += sp; xsp = std::max(xsp, sp);
            }
            msp /= static_cast<float>(P);
            if (s >= kDrape / 2u) backhalf_speed_max = std::max(backhalf_speed_max, msp);
            end_mean_speed = msp; end_max_speed = xsp;
            for (uint32_t row = 0u; row < rows; ++row) {
                const RowSides rs = DecodeRow(urows, row);
                if (!rs.active) continue;
                const bool a_part = rs.a_kind == nk::kNkSideParticle;
                const bool b_part = rs.b_kind == nk::kNkSideParticle;
                if (!(a_part || b_part)) continue;
                ++total_part_rows;
                // The body side: an artic row carries its link in the chain-J gather
                // (its row index is the articulation id); a rigid row carries a body
                // row resolvable through body_to_link. Pick the body side of the pair.
                const bool a_body = !a_part;
                const uint32_t body_kind = a_body ? rs.a_kind : rs.b_kind;
                const uint32_t body_idx = a_body ? rs.a_index : rs.b_index;
                const uint32_t cj_link = a_body ? cj_link_a[row] : cj_link_b[row];
                uint32_t link = ~uint32_t(0);
                if (body_kind == nk::kNkSideArtic) {
                    link = cj_link;  // global link == template link at one env.
                } else if (body_kind == nk::kNkSideRigid &&
                           body_idx < body_to_link.size() &&
                           body_to_link[body_idx] != ~uint32_t(0)) {
                    link = body_to_link[body_idx];
                }
                const bool normal_row = rs.upper > 1.0e30f;  // friction spokes cap upper.
                if (normal_row && lambda[row] > 0.0f) ++body_lambda_pos;
                if (link < L) {
                    link_has_row[link] = 1u;
                    if (normal_row) {
                        link_max_lambda[link] = std::max(link_max_lambda[link], lambda[row]);
                        if (!is_foot_link(link))
                            max_nonfoot_lambda = std::max(max_nonfoot_lambda, lambda[row]);
                    }
                }
            }
        }
        const double drape_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - drape_t0).count();
        std::printf("[go2_cloth_drape] drape %u steps in %.0f ms (%.0f ms/step active)\n",
                    kDrape, drape_ms, drape_ms / kDrape);
        std::printf("[go2_cloth_drape] PROBE motion: end_mean_speed=%.4f end_max_speed=%.4f "
                    "backhalf_mean_speed_max=%.4f m/s (settled => ~0)\n",
                    end_mean_speed, end_max_speed, backhalf_speed_max);

        // Opt-in body<->particle manifold dump (NK_NPBP_DUMP=<path>): the raw
        // ucontact point/normal/depth + a/b/kind slot stream after the fixed drape,
        // for a byte-for-byte serial-vs-warp narrowphase comparison.
        if (const char* dump = std::getenv("NK_NPBP_DUMP"); dump != nullptr) {
            const uint32_t slots = world.GetModel().capacities.max_contacts_per_env;
            const size_t cells = static_cast<size_t>(slots) * 4u;
            std::vector<Vec3> dp(cells), dn(cells);
            std::vector<float> dd(cells);
            std::vector<uint32_t> da(cells), db(cells), dak(cells), dbk(cells), dgen(cells);
            std::vector<uint32_t> dcnt(slots);
            data.DownloadField(nk::FieldId::UcontactPoint, dp.data(), cells * sizeof(Vec3));
            data.DownloadField(nk::FieldId::UcontactNormal, dn.data(), cells * sizeof(Vec3));
            data.DownloadField(nk::FieldId::UcontactDepth, dd.data(), cells * sizeof(float));
            data.DownloadField(nk::FieldId::UcontactA, da.data(), cells * sizeof(uint32_t));
            data.DownloadField(nk::FieldId::UcontactB, db.data(), cells * sizeof(uint32_t));
            data.DownloadField(nk::FieldId::UcontactAKind, dak.data(), cells * sizeof(uint32_t));
            data.DownloadField(nk::FieldId::UcontactBKind, dbk.data(), cells * sizeof(uint32_t));
            data.DownloadField(nk::FieldId::UcontactGen, dgen.data(), cells * sizeof(uint32_t));
            data.DownloadField(nk::FieldId::UcontactCount, dcnt.data(), slots * sizeof(uint32_t));
            std::FILE* f = std::fopen(dump, "wb");
            if (f != nullptr) {
                auto wr = [&](const void* p, size_t n) { std::fwrite(p, 1, n, f); };
                wr(&slots, sizeof(slots));
                wr(dp.data(), cells * sizeof(Vec3)); wr(dn.data(), cells * sizeof(Vec3));
                wr(dd.data(), cells * sizeof(float));
                wr(da.data(), cells * sizeof(uint32_t)); wr(db.data(), cells * sizeof(uint32_t));
                wr(dak.data(), cells * sizeof(uint32_t)); wr(dbk.data(), cells * sizeof(uint32_t));
                wr(dgen.data(), cells * sizeof(uint32_t)); wr(dcnt.data(), slots * sizeof(uint32_t));
                std::fclose(f);
                std::printf("[go2_cloth_drape] NPBP dump -> %s (%u slots)\n", dump, slots);
            }
        }
        download_pos();
        float cloth_lo = 1e9f, cloth_hi = -1e9f;
        for (const Vec3& p : pos) { cloth_lo = std::min(cloth_lo, p.z); cloth_hi = std::max(cloth_hi, p.z); }
        // Final-shape histogram: bucket particles by where they ended up so the drape
        // shape reads at a glance -- on the back/shoulders (high z), draping the sides
        // (mid z, outboard y), or pooled in the central gap / floor (low z, inboard y).
        uint32_t on_back = 0u, side_drape = 0u, in_gap = 0u, on_floor = 0u;
        for (const Vec3& p : pos) {
            const float ay = std::fabs(p.y - base_xy.y);
            if (p.z > 0.24f) ++on_back;
            else if (p.z < 0.04f) ++on_floor;
            else if (ay > 0.10f) ++side_drape;
            else ++in_gap;
        }
        std::printf("[go2_cloth_drape] PROBE shape: on_back(z>0.24)=%u side_drape(0.04<z<0.24,|y|>0.10)=%u "
                    "central_gap=%u on_floor(z<0.04)=%u of %u\n",
                    on_back, side_drape, in_gap, on_floor, P);
        download_links();

        // COM drift (x,y) over the back half: end centroid minus the mid-clip
        // centroid; a stable symmetric drape stays put (drift << 1 cm).
        Vec3 com_end{0, 0, 0};
        for (const Vec3& p : pos) { com_end.x += p.x; com_end.y += p.y; com_end.z += p.z; }
        com_end = Vec3{com_end.x / P, com_end.y / P, com_end.z / P};
        const float drift_x = com_end.x - com_mid.x, drift_y = com_end.y - com_mid.y;
        const float drift_xy = std::sqrt(drift_x * drift_x + drift_y * drift_y);
        std::printf("[go2_cloth_drape] PROBE COM: mid=(%.4f,%.4f) end=(%.4f,%.4f,%.4f) "
                    "drift_xy=%.4f (dx=%.4f dy=%.4f)\n",
                    com_mid.x, com_mid.y, com_end.x, com_end.y, com_end.z,
                    drift_xy, drift_x, drift_y);

        // Penetration metric: the deepest a particle sits INSIDE any body collision
        // primitive (box/capsule), posed by FK (link_pose o link_geom_local). The
        // particle is a sphere of radius d_min/2; a NEGATIVE signed gap = penetration.
        const float prad = 0.5f * kContactDMin;
        auto box_sd = [](const Vec3& q, const Vec3& he) {
            const Vec3 d{std::fabs(q.x) - he.x, std::fabs(q.y) - he.y, std::fabs(q.z) - he.z};
            const Vec3 dp{std::max(d.x, 0.0f), std::max(d.y, 0.0f), std::max(d.z, 0.0f)};
            const float outside = std::sqrt(dp.x * dp.x + dp.y * dp.y + dp.z * dp.z);
            const float inside = std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f);
            return outside + inside;
        };
        auto cap_sd = [](const Vec3& q, float r, float hh) {
            const float t = std::max(-hh, std::min(hh, q.z));
            const Vec3 c{q.x, q.y, q.z - t};
            return std::sqrt(c.x * c.x + c.y * c.y + c.z * c.z) - r;
        };
        // Build a host SDF view for body b (the cooked silhouette grid), or empty.
        namespace rsdf = nuka::runtime::sdf;
        auto sdf_view = [&](uint32_t grid) {
            rsdf::SparseSdfDevice s;
            if (grid >= sdf_grids.size()) return s;
            const auto& g = sdf_grids[grid];
            s.origin = g.origin; s.voxel_size = g.voxel_size;
            s.dims[0] = g.dims[0]; s.dims[1] = g.dims[1]; s.dims[2] = g.dims[2];
            s.cell_keys = sdf_keys.data() + g.cell_offset;
            s.cell_values = sdf_values.data() + g.cell_offset;
            s.cell_gradients = sdf_grads.data() + g.cell_offset;
            s.cell_count = g.cell_count;
            return s;
        };
        const uint8_t kSdfKind = static_cast<uint8_t>(nuka::collision::kShapeSdfMesh);
        float deepest_pen = 0.0f;     // most-negative signed gap (>= 0 == no penetration).
        uint32_t penetrating = 0u;
        const bool pen_diag = std::getenv("NK_PEN_DIAG") != nullptr;
        Vec3 deepest_wp{0, 0, 0}; uint32_t deepest_lk = ~0u;
        uint32_t deep5 = 0u, deep1 = 0u;  // particles deeper than 5cm / 1cm into a body.
        for (const Vec3& wp : pos) {
            float best = 1e9f;        // nearest surface signed distance for this particle.
            uint32_t best_lk = ~0u;
            for (size_t b = 0; b < shapes.size(); ++b) {
                const auto& s = shapes[b];
                if (s.kind != 1u && s.kind != 2u && s.kind != kSdfKind) continue;
                if (b >= body_to_link.size() || body_to_link[b] >= L) continue;
                const uint32_t lk = body_to_link[b];
                const Transform world = link_pose[lk] *
                                        (lk < geom_local.size() ? geom_local[lk] : Transform::Identity());
                const Vec3 q = world.Inverse().TransformPoint(wp);
                float sd;
                if (s.kind == kSdfKind) {
                    rsdf::SparseSdfDevice sg = sdf_view(s.sdf_grid);
                    Vec3 g{0, 0, 0};
                    const float phi = rsdf::sparse_sdf_sample(sg, q, g);
                    // Outside the narrow band == far from this surface (not closer).
                    sd = (phi < rsdf::SparseSdfDevice::kOutsideBand) ? phi : 1e9f;
                } else {
                    sd = (s.kind == 2u)
                        ? box_sd(q, Vec3{s.params[0], s.params[1], s.params[2]})
                        : cap_sd(q, s.params[0], s.params[1]);
                }
                if (sd < best) { best = sd; best_lk = lk; }
            }
            const float gap = best - prad;  // sphere surface vs body surface.
            if (gap < deepest_pen) { deepest_pen = gap; deepest_wp = wp; deepest_lk = best_lk; }
            if (gap < -1.0e-4f) ++penetrating;
            if (gap < -0.05f) ++deep5;
            if (gap < -0.01f) ++deep1;
        }
        if (pen_diag) {
            std::printf("[pen_diag] deepest particle wp=(%.3f,%.3f,%.3f) in link %u; "
                        "deeper_than_1cm=%u deeper_than_5cm=%u\n",
                        deepest_wp.x, deepest_wp.y, deepest_wp.z, deepest_lk,
                        deep1, deep5);
        }
        std::printf("[go2_cloth_drape] PROBE penetration: deepest=%.4f m (>=0 clean) "
                    "particles_penetrating=%u of %u\n", deepest_pen, penetrating, P);
        // Opt-in final-positions dump (NK_POS_DUMP=<path>): raw P*Vec3 cloth points +
        // L*Transform link poses, for an offline proximity check against the geometry.
        if (const char* pd = std::getenv("NK_POS_DUMP"); pd != nullptr) {
            std::FILE* f = std::fopen(pd, "wb");
            if (f != nullptr) {
                std::fwrite(&P, sizeof(P), 1, f);
                std::fwrite(&L, sizeof(L), 1, f);
                std::fwrite(pos.data(), sizeof(Vec3), P, f);
                std::fwrite(link_pose.data(), sizeof(Transform), L, f);
                std::vector<Vec3> vel(P, Vec3::Zero());
                data.DownloadField(nk::FieldId::ParticleVel, vel.data(), P * sizeof(Vec3));
                std::fwrite(vel.data(), sizeof(Vec3), P, f);
                std::fclose(f);
                std::printf("[go2_cloth_drape] POS dump -> %s (P=%u L=%u)\n", pd, P, L);
            }
        }
        Transform b; data.DownloadField(nk::FieldId::BasePose, &b, sizeof(Transform));

        std::printf("[go2_cloth_drape] PROBE coupling: total_body<->particle_rows=%u "
                    "(per-step accum) body_normal_lambda>0=%u\n",
                    total_part_rows, body_lambda_pos);
        std::printf("[go2_cloth_drape] PROBE links WITH contact rows:");
        uint32_t distinct = 0u, nonfoot_links = 0u;
        for (uint32_t l = 0; l < L; ++l) if (link_has_row[l]) {
            std::printf(" L%u%s(lam=%.4f)", l, is_foot_link(l) ? "[foot]" : "", link_max_lambda[l]);
            ++distinct;
            if (!is_foot_link(l)) ++nonfoot_links;
        }
        std::printf("\n[go2_cloth_drape] PROBE distinct_links=%u (non-foot=%u) "
                    "max_nonfoot_lambda=%.4f\n", distinct, nonfoot_links, max_nonfoot_lambda);
        std::printf("[go2_cloth_drape] PROBE drape: base_z=%.4f cloth_z=[%.4f,%.4f] "
                    "trunk_lay_z=%.4f\n", b.position.z, cloth_lo, cloth_hi,
                    kStandBaseZ + cloth_lift);

        // PASS = the cloth conforms to MULTIPLE links incl. non-foot (trunk/thigh/
        // calf), the body produces a positive normal reaction, and the sheet
        // settled onto the dog (dropped below its lay) without sliding fully off
        // (its lowest point stays near/above the ground, not tunneled).
        const bool multi_link = distinct >= 3u && nonfoot_links >= 1u;
        const bool coupled = body_lambda_pos > 0u && max_nonfoot_lambda > 0.0f;
        const bool draped = cloth_lo < (kStandBaseZ + cloth_lift) - 0.05f && cloth_lo > -0.05f;
        const bool ok = multi_link && coupled && draped;
        std::printf("[go2_cloth_drape] PROBE %s (multi_link=%d coupled=%d draped=%d)\n",
                    ok ? "PASS" : "FAIL", multi_link, coupled, draped);
        return ok ? 0 : 5;
    }

    // ---- RENDER: FK-rebind the Go2 visuals from the SAME go2.nks ---------------
    cook::CookToModelResult cooked = cook::CookToModel(scene, 1);
    render::RenderWorld rw = render::BuildRenderWorld(scene.Ecs(), cooked.scene_map);
    if (rw.instances.empty()) {
        std::fprintf(stderr, "[go2_cloth_drape] FATAL: BuildRenderWorld produced 0 instances\n");
        return 4;
    }
    // Per-instance link + geom-local (the proven shape->link rebind).
    std::vector<uint32_t> inst_link(rw.instances.size(), 0u);
    std::vector<Transform> inst_vlocal(rw.instances.size(), Transform::Identity());
    uint32_t n_rebound = 0u, n_static = 0u, n_oob = 0u;
    for (size_t i = 0; i < rw.instances.size(); ++i) {
        const render::PoseSource& ps = rw.instances[i].pose_source;
        if (ps.kind == render::PoseSource::Kind::Static || ps.row == render::kNoId) { ++n_static; continue; }
        if (ps.row < L) { inst_link[i] = ps.row; inst_vlocal[i] = rw.instances[i].cached_visual_local; ++n_rebound; }
        else ++n_oob;
    }
    if (n_oob > 0u) {
        std::fprintf(stderr, "[go2_cloth_drape] FATAL: %u instances reference link >= %u\n", n_oob, L);
        return 5;
    }
    std::printf("[go2_cloth_drape] RenderWorld: instances=%u meshes=%u rebound=%u static=%u\n",
                rw.InstanceCount(), rw.meshes.Count(), n_rebound, n_static);

    // Premium studio material override (the go2_walk_video hero look).
    rw.materials.clear();
    const uint32_t kMatShell = 0u, kMatMetal = 1u, kMatFoot = 2u, kMatAccent = 3u;
    const uint32_t kMatCloth = 4u, kMatFloor = 5u;
    rw.materials.push_back(Mk(0.072f, 0.076f, 0.085f, 0.05f, 0.55f));  // 0 shell charcoal
    rw.materials.push_back(Mk(0.090f, 0.096f, 0.110f, 0.85f, 0.34f));  // 1 dark gunmetal
    rw.materials.push_back(Mk(0.045f, 0.045f, 0.050f, 0.04f, 0.72f));  // 2 dark rubber foot
    rw.materials.push_back(Mk(0.16f,  0.180f, 0.225f, 0.65f, 0.30f));  // 3 cool steel accent
    // Deep crimson matte fabric: a saturated dielectric at fabric roughness with only a
    // faint grazing sheen (mat.sheen) for velvet quality, not a glossy glint. Env-tunable.
    {
        float cr = 0.400f, cg = 0.035f, cb = 0.060f, crough = 0.58f, csheen = 0.6f;
        if (const char* v = std::getenv("NK_CLOTH_ROUGH")) crough = std::atof(v);
        if (const char* v = std::getenv("NK_CLOTH_SHEEN")) csheen = std::atof(v);
        if (const char* v = std::getenv("NK_CLOTH_R")) cr = std::atof(v);
        if (const char* v = std::getenv("NK_CLOTH_G")) cg = std::atof(v);
        if (const char* v = std::getenv("NK_CLOTH_B")) cb = std::atof(v);
        rw.materials.push_back(Mk(cr, cg, cb, 0.00f, crough, csheen));  // 4 crimson silk
    }
    rw.materials.push_back(Mk(0.190f, 0.196f, 0.210f, 0.02f, 0.82f));  // 5 studio floor (cool neutral)
    rw.default_material_id = kMatShell;
    auto link_role_material = [&](uint32_t link, const Transform& vlocal) -> uint32_t {
        if (link == 0u) return kMatShell;
        const uint32_t k = ((link - 1u) % 3u);
        if (k == 0u) return kMatMetal;
        if (k == 1u) return kMatShell;
        return (vlocal.position.z < -0.15f) ? kMatFoot : kMatMetal;
    };
    int accent_inst = -1;
    for (size_t i = 0; i < rw.instances.size(); ++i) {
        rw.instances[i].render_material_id = link_role_material(inst_link[i], inst_vlocal[i]);
        if (inst_link[i] == 0u) accent_inst = static_cast<int>(i);
    }
    if (accent_inst >= 0) rw.instances[static_cast<size_t>(accent_inst)].render_material_id = kMatAccent;
    const uint32_t go2_inst_end = rw.InstanceCount();

    // The floor + cloth instances are appended after the Go2 instances; their
    // world_xform is driven directly (the cloth verts are world-space). The cloth
    // mesh is re-interned each frame as the sheet deforms.
    const soft::SurfaceTopology topo = MakeClothTopology();
    {
        const uint32_t floor_mesh = rw.meshes.InternPrimitive(
            "floor", [&] { return MakeFloorGeo(8.0f, 0.0f); });
        render::RenderInstance fi;
        fi.mesh_id = floor_mesh; fi.render_material_id = kMatFloor;
        fi.world_xform = Transform::Identity();
        fi.pose_source.kind = render::PoseSource::Kind::Static;
        rw.instances.push_back(fi);
    }

    // Publish the Go2 visuals from the live link poses (world_xform = link o local).
    auto publish_go2 = [&]() {
        for (size_t i = 0; i < go2_inst_end; ++i)
            rw.instances[i].world_xform = link_pose[inst_link[i]] * inst_vlocal[i];
    };
    // Rebuild the cloth surface mesh from the live particles each frame and bind it
    // as ONE instance. The mesh library caches by key, so a per-frame key forces a
    // fresh build of the deforming surface (both backends re-read the world).
    constexpr size_t kNoCloth = ~size_t(0);
    size_t cloth_inst_ = kNoCloth;
    auto publish_cloth = [&](uint32_t frame) {
        render::MeshGeometry cloth_geo;
        soft::BuildSurfaceMesh(pos, topo, cloth_geo);
        char key[32];
        std::snprintf(key, sizeof(key), "cloth:live:%u", frame);
        const uint32_t cloth_mesh = rw.meshes.InternPrimitive(key, [&] { return cloth_geo; });
        if (cloth_inst_ == kNoCloth) {
            render::RenderInstance ci;
            ci.mesh_id = cloth_mesh; ci.render_material_id = kMatCloth;
            ci.world_xform = Transform::Identity();
            ci.pose_source.kind = render::PoseSource::Kind::Static;
            rw.instances.push_back(ci);
            cloth_inst_ = rw.instances.size() - 1u;
        } else {
            rw.instances[cloth_inst_].mesh_id = cloth_mesh;
        }
    };

    // ---- renderer (lavapipe raster preview OR CUDA RT beauty hero) -------------
    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    std::unique_ptr<GpuRenderer> gpu;
    if (args.gpu) {
        gpu = std::make_unique<GpuRenderer>(rw);
        if (!gpu->ok()) { std::fprintf(stderr, "[go2_cloth_drape] --gpu: no CUDA RT backend\n"); return 6; }
        gpu->SetBeauty(args.beauty, args.samples);
        std::printf("[go2_cloth_drape] renderer: CUDA RT (%s)\n", args.beauty ? "BEAUTY" : "flat");
    } else {
        try { renderer = std::make_unique<render::VulkanRasterRenderer>(); }
        catch (const std::exception& e) { std::fprintf(stderr, "[go2_cloth_drape] no Vulkan: %s\n", e.what()); return 6; }
        std::printf("[go2_cloth_drape] renderer ICD: %s | particles=%u\n", renderer->DeviceName().c_str(), P);
    }
    auto render_frame = [&](const render::RasterOptions& o) {
        return gpu ? gpu->Render(rw, o) : renderer->Render(rw, o);
    };

    render::RasterOptions opts;
    opts.width = args.width; opts.height = args.height;
    opts.draw_ground = false;  // explicit floor mesh (the beauty path has no implicit floor).
    opts.hero_framing = false;
    opts.use_camera_override = true;
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 40.0f;
    opts.background = {14, 17, 23, 255};
    opts.ground_color[0] = 0.180f; opts.ground_color[1] = 0.190f; opts.ground_color[2] = 0.210f;
    opts.contact_shadow_strength = 0.0f;
    opts.use_sun_light = true;
    opts.sun_direction[0] = 0.42f; opts.sun_direction[1] = -0.34f; opts.sun_direction[2] = 0.50f;
    opts.sun_color[0] = 2.80f; opts.sun_color[1] = 2.70f; opts.sun_color[2] = 2.55f;
    opts.sun_ambient_sky[0] = 0.16f; opts.sun_ambient_sky[1] = 0.19f; opts.sun_ambient_sky[2] = 0.26f;
    opts.sun_ambient_ground[0] = 0.10f; opts.sun_ambient_ground[1] = 0.10f; opts.sun_ambient_ground[2] = 0.11f;
    opts.shadow_strength = 0.9f;
    opts.shadow_map_size = 2048u;
    opts.shadow_bias = 0.0020f;
    opts.sky_gradient = true;
    opts.sky_top[0] = 0.22f; opts.sky_top[1] = 0.32f; opts.sky_top[2] = 0.52f;
    opts.sky_bottom[0] = 0.58f; opts.sky_bottom[1] = 0.66f; opts.sky_bottom[2] = 0.74f;

    auto smoothstep = [](float t) { t = std::max(0.0f, std::min(1.0f, t)); return t * t * (3.0f - 2.0f * t); };

    std::filesystem::create_directories(args.out_dir);
    std::filesystem::create_directories(args.png_dir);

    // The drape clip: RELEASE the held cloth and let gravity drape it over the
    // standing dog. Side-profile (cloth falling) arcs to front-3/4 as it settles.
    const uint32_t kDrape = args.drape;
    uint32_t written = 0u;
    size_t first_nonbg = 0u, last_nonbg = 0u;

    auto render_one = [&](uint32_t step, const std::string* png) {
        download_links();
        download_pos();
        publish_go2();
        publish_cloth(written);
        const float t = kDrape > 1u ? static_cast<float>(step) / static_cast<float>(kDrape - 1u) : 0.0f;
        const float e = smoothstep(t);
        const float az = -0.50f + 1.05f * e;                 // side -> front-3/4 arc.
        const float elev = (18.0f - 4.0f * e + 4.0f * std::sin(e * kPi)) * kPi / 180.0f;
        // Pull IN and DOWN as it settles: frame the high falling sheet first, then
        // tighten onto the draped dog + the hem pooling on the floor.
        const float radius = 1.95f - 0.55f * e;
        const Vec3 look{base_xy.x + 0.05f, base_xy.y, 0.42f - 0.24f * e};
        opts.camera_eye = {look.x + radius * std::cos(elev) * std::sin(az),
                           look.y - radius * std::cos(elev) * std::cos(az),
                           look.z + radius * std::sin(elev)};
        opts.camera_target = look;

        render::VulkanOffscreenReport rep = render_frame(opts);
        char name[40];
        std::snprintf(name, sizeof(name), "frame_%06u.ppm", written);
        WritePpm(rep, args.out_dir + "/" + name);
        if (written == 0u) first_nonbg = rep.non_background_pixel_count;
        last_nonbg = rep.non_background_pixel_count;
        if (png) {
            if (WritePng(rep, *png)) std::printf("[go2_cloth_drape] hero PNG -> %s\n", png->c_str());
            else std::fprintf(stderr, "[go2_cloth_drape] PNG encode failed: %s\n", png->c_str());
        }
        ++written;
    };

    for (uint32_t s = 0; s < kDrape; ++s) {
        hold_dog();
        world.Step();  // cloth no longer held -> it drapes.
        damp_cloth();
        const std::string* png = nullptr;
        std::string png_path;
        if (s == 6u) { png_path = args.png_dir + "/01_falling.png"; png = &png_path; }
        else if (s == kDrape / 3u) { png_path = args.png_dir + "/02_draping.png"; png = &png_path; }
        else if (s + 1u == kDrape) { png_path = args.png_dir + "/03_settled.png"; png = &png_path; }
        if ((s % args.stride) == 0u || png) render_one(s, png);
    }

    std::printf("[go2_cloth_drape] DONE: %u frames -> %s (first_nonbg=%zu last_nonbg=%zu)\n",
                written, args.out_dir.c_str(), first_nonbg, last_nonbg);
    return written > 0u ? 0 : 7;
}
