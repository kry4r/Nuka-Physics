// ---------------------------------------------------------------------------
// mpm_jelly_demo.cpp -- an MLS-MPM elastic jelly ball drops onto the floor.
//
// A fixed-corotated MLS-MPM continuum ball (densely sampled on a lattice, selected
// via the config sim_method=mlsmpm selector -- NOT a hardcoded ParticleMode) is
// dropped onto the STATIC floor plane. The medium couples to the floor entirely
// through the grid boundary condition (no dynamic body); the ball squashes on
// impact, the grid BC pushes back, the constitutive stress recovers the shape, and
// it bounces. This REPLACES the failing XPBD soft-ball on the showcase.
//
// RENDER: a high-res icosphere skin is bound ONCE to its NEAREST MPM particles
// (runtime::soft::EmbedSurfaceInParticleSet -- a dense particle fill has no
// out-of-set vertex, sidestepping the negative-bary spike bug). PER FRAME: download
// the live particle positions, DeformEmbeddedSkin -> smooth skin verts, build the
// deforming surface mesh, render the ball + a studio floor + a contracting contact
// shadow on the CUDA RT backend (--gpu --beauty) or lavapipe raster, write a PPM.
// At airborne / max-squash / recovered key frames also write a hero PNG.
//
// Built behind NK_BUILD_VULKAN_VALIDATION. Usage:
//   mpm_jelly_demo [--width W] [--height H] [--out-dir DIR] [--png-dir DIR]
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

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
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

// Ball + grid geometry. The lattice fill (dx/2 spacing) gives a dense continuum;
// the soft elastic modulus + the substep count keep the explicit step stable.
constexpr float kBallRadius = 0.10f;
constexpr float kFloorZ     = 0.0f;
constexpr float kDropGap    = 0.05f;      // bottom of the ball above the floor.
constexpr float kDx         = 0.025f;     // grid node spacing.
constexpr float kYoungs     = 3.0e4f;     // soft jelly.
constexpr float kPoisson    = 0.3f;
constexpr float kDensity    = 1000.0f;
constexpr uint32_t kSubsteps = 20u;       // CFL headroom for the explicit step.

// ---- CLI ------------------------------------------------------------------
struct Args {
    uint32_t width = 1920u;
    uint32_t height = 1080u;
    uint32_t stride = 2u;
    bool gpu = false;
    bool beauty = true;
    uint32_t samples = 16u;
    uint32_t shadow_map = 2560u;
    std::string out_dir = "/tmp/mpm_jelly_frames";
    std::string png_dir = "out/mpm_jelly";
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

// ---- jelly cook: a dense lattice fill, selected via sim_method=mlsmpm ------
cook::MpmCookInput BuildJellyInput() {
    cook::MpmCookInput in;
    const float center_z = kFloorZ + kDropGap + kBallRadius;
    const float pdx = kDx * 0.5f;
    const int r_cells = static_cast<int>(kBallRadius / pdx) + 1;
    for (int i = -r_cells; i <= r_cells; ++i)
        for (int j = -r_cells; j <= r_cells; ++j)
            for (int k = -r_cells; k <= r_cells; ++k) {
                const Vec3 local{i * pdx, j * pdx, k * pdx};
                if (local.LengthSq() <= kBallRadius * kBallRadius)
                    in.positions.push_back(Vec3{local.x, local.y, center_z + local.z});
            }
    const size_t n = in.positions.size();
    in.velocities.assign(n, Vec3::Zero());
    const float vol0 = pdx * pdx * pdx;
    in.inv_mass.assign(n, 1.0f / (kDensity * vol0));
    in.vol0.assign(n, vol0);
    in.material.youngs = kYoungs; in.material.poisson = kPoisson;
    in.material.density = kDensity; in.material.model_kind = 0.0f;
    const float span = kBallRadius + kDropGap + 4.0f * kDx;
    in.grid_origin = Vec3{-span, -span, kFloorZ - 4.0f * kDx};
    const float ht = (center_z + kBallRadius) - in.grid_origin.z + 4.0f * kDx;
    in.grid_dims[0] = static_cast<uint32_t>(2.0f * span / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>(ht / kDx) + 1u;
    in.dx = kDx;
    in.substeps = kSubsteps;
    in.floor_normal = Vec3{0.0f, 0.0f, 1.0f};
    in.floor_d = kFloorZ;
    in.floor_friction = 0.5f;
    return in;
}

nk::Model BuildJellyModel() {
    nk::Model m;
    m.capacities.env_count = 1u;
    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;  // the SELECTOR (not a hardcoded mode).
    cook::CookSoftBodyParticles(m, 1u, soft, BuildJellyInput());
    return m;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    return cfg;
}

// Vertical extent (compresses on squash) + horizontal span (collapse guard).
struct Extent { float min_z, height, span_xy; };
Extent Measure(const std::vector<Vec3>& pos, uint32_t n) {
    float zlo = 1e30f, zhi = -1e30f, xlo = 1e30f, xhi = -1e30f, ylo = 1e30f, yhi = -1e30f;
    for (uint32_t i = 0; i < n; ++i) {
        zlo = std::min(zlo, pos[i].z); zhi = std::max(zhi, pos[i].z);
        xlo = std::min(xlo, pos[i].x); xhi = std::max(xhi, pos[i].x);
        ylo = std::min(ylo, pos[i].y); yhi = std::max(yhi, pos[i].y);
    }
    return Extent{zlo, zhi - zlo, 0.5f * ((xhi - xlo) + (yhi - ylo))};
}

// ---- render helpers (mirror the soft-ball studio look) --------------------
nuka::scene::RenderMaterial MakeMat(float r, float g, float b, float metallic,
                                    float rough) {
    nuka::scene::RenderMaterial m;
    m.base_color[0] = r; m.base_color[1] = g; m.base_color[2] = b; m.base_color[3] = 1.0f;
    m.metallic = metallic; m.roughness = rough;
    return m;
}

render::MeshGeometry MakeFloorGeo(float half, float thickness) {
    render::MeshGeometry g;
    const float xs[2] = {-half, half}, ys[2] = {-half, half}, zs[2] = {-thickness, 0.0f};
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
        scene_ = render::RenderWorldToTwoLevelScene(rw);  // rebuild over the live verts.
        if (handle_) backend_->FreeScene(handle_);
        handle_ = backend_->BuildScene(scene_);
        ApplyLighting(opts);
        const rt::PinholeCamera cam = CameraFromOptions(opts);
        rt::Framebuffer fb = beauty_
            ? backend_->TraceBeautyToHost(handle_, scene_, cam, BeautyFromOptions(opts))
            : backend_->TraceToHost(handle_, scene_, cam);
        return render::FramebufferToReport(fb, opts.background);
    }

private:
    void ApplyLighting(const render::RasterOptions& opts) {
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
        b.gi_bounces = 1u; b.ao_samples = 4u; b.ao_radius = 0.7f;
        b.seed = 0x9e3779b9u;
        b.smooth_normals = true;
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

render::RenderWorld BuildFrame(const soft::SurfaceTopology& topo,
                               const std::vector<Vec3>& skin_pos) {
    render::RenderWorld rw;
    rw.materials.push_back(MakeMat(0.20f, 0.62f, 0.45f, 0.0f, 0.18f));  // 0 glossy jelly
    rw.materials.push_back(MakeMat(0.16f, 0.17f, 0.20f, 0.1f, 0.22f));  // 1 floor
    rw.default_material_id = 0u;
    render::MeshGeometry ball_geo;
    soft::BuildSurfaceMesh(skin_pos, topo, ball_geo);
    const uint32_t ball_mesh = rw.meshes.InternPrimitive("ball:live", [&] { return ball_geo; });
    const uint32_t floor_mesh =
        rw.meshes.InternPrimitive("floor:box", [&] { return MakeFloorGeo(2.5f, 0.10f); });
    auto add = [&](uint32_t mesh, uint32_t mat, const Transform& xf) {
        render::RenderInstance inst;
        inst.mesh_id = mesh; inst.render_material_id = mat; inst.world_xform = xf;
        inst.pose_source.kind = render::PoseSource::Kind::Static;
        rw.instances.push_back(inst);
    };
    add(ball_mesh, 0u, Transform::Identity());
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
        std::fprintf(stderr, "[mpm_jelly_demo] no CUDA backend\n");
        return 2;
    }

    nk::Model model = BuildJellyModel();
    const uint32_t P = model.capacities.particles_per_env;
    // The MPM particle REST positions drive the skin bind. A high-res icosphere skin
    // is bound to its NEAREST MPM particles ONCE (dense fill -> no out-of-set vert).
    const std::vector<Vec3> particle_rest = model.particles.initial_pos;
    const float center_z = kFloorZ + kDropGap + kBallRadius;
    const soft::TriMesh skin_rest =
        soft::BuildIcosphere(Vec3{0, 0, center_z}, kBallRadius, 3u);
    const soft::EmbeddedSkin skin = soft::EmbedSurfaceInParticleSet(
        skin_rest.positions, skin_rest.triangles, particle_rest);
    soft::SurfaceTopology topo;
    topo.normal_offset = 0.0f;
    topo.triangles = skin.triangles;

    nk::World world(std::move(model), 1u, dev, backend, Cfg());
    if (!world.Ready()) {
        std::fprintf(stderr, "[mpm_jelly_demo] world not ready\n");
        return 3;
    }
    std::printf("[mpm_jelly_demo] mpm particles=%u skin_verts=%zu skin_tris=%zu substeps=%u\n",
                P, skin_rest.positions.size(), skin.triangles.size() / 3u, kSubsteps);

    std::vector<Vec3> pos(P, Vec3::Zero());
    auto download_pos = [&] {
        world.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(), P * sizeof(Vec3));
    };
    std::vector<Vec3> skin_pos;
    auto deform_skin = [&] {
        soft::DeformEmbeddedSkin(skin, pos, skin_pos);
        soft::SmoothSurface(topo.triangles, 4u, 0.40f, skin_pos);
    };

    download_pos();
    const Extent rest = Measure(pos, P);

    // ---- PROBE: the gate-(a) invariants at demo scale, no render ------------
    if (args.probe) {
        constexpr uint32_t kSteps = 1200u;  // 5 s at dt = 1/240.
        float min_h = 1e30f, min_z = 1e30f, min_span = 1e30f, rebound = 0.0f;
        int sq = -1; bool nonfinite = false; uint32_t escape = 0u;
        std::vector<Vec3> vel(P, Vec3::Zero());
        for (uint32_t s = 0; s < kSteps; ++s) {
            world.Step();
            download_pos();
            world.GetData().DownloadField(nk::FieldId::ParticleVel, vel.data(), P * sizeof(Vec3));
            const Extent e = Measure(pos, P);
            if (e.height < min_h) { min_h = e.height; sq = static_cast<int>(s); }
            min_z = std::min(min_z, e.min_z);
            min_span = std::min(min_span, e.span_xy);
            if (sq >= 0 && static_cast<int>(s) > sq) rebound = std::max(rebound, e.height);
            for (uint32_t i = 0; i < P && !nonfinite; ++i)
                nonfinite = !std::isfinite(pos[i].z) || !std::isfinite(vel[i].z);
            uint32_t st = 0u;
            world.GetData().DownloadField(nk::FieldId::EnvStatus, &st, sizeof(uint32_t));
            escape |= st & nphi::kEnvStatusMpmGridEscape;
            if (s % 120u == 0u)
                std::printf("  [arc] s=%4u h=%.4f min_z=%.4f span=%.4f\n",
                            s, e.height, e.min_z, e.span_xy);
        }
        const float comp = 1.0f - min_h / rest.height;
        const bool ok = !nonfinite && escape == 0u && min_z >= kFloorZ - kDx &&
                        comp <= 0.30f && comp > 0.02f &&
                        rebound > min_h + 0.1f * rest.height &&
                        min_span > 0.5f * rest.span_xy;
        std::printf("[mpm_jelly_demo] PROBE %s\n", ok ? "PASS" : "FAIL");
        std::printf("  peak_compression=%.1f%% (rest_h=%.4f min_h=%.4f) rebound_h=%.4f\n",
                    100.0f * comp, rest.height, min_h, rebound);
        std::printf("  min_z_ever=%.4f (floor=%.4f) min_span=%.4f (rest=%.4f) nonfinite=%d escape=%u\n",
                    min_z, kFloorZ, min_span, rest.span_xy, nonfinite ? 1 : 0, escape);
        return ok ? 0 : 5;
    }

    // ---- RENDER -------------------------------------------------------------
    download_pos();
    deform_skin();
    render::RenderWorld rw0 = BuildFrame(topo, skin_pos);
    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    std::unique_ptr<GpuRenderer> gpu;
    if (args.gpu) {
        gpu = std::make_unique<GpuRenderer>(rw0);
        if (!gpu->ok()) {
            std::fprintf(stderr, "[mpm_jelly_demo] --gpu: no CUDA RT backend\n");
            return 6;
        }
        gpu->SetBeauty(args.beauty, args.samples);
        std::printf("[mpm_jelly_demo] renderer: CUDA two-level ray tracer (GPU, %s)\n",
                    args.beauty ? "BEAUTY MSAA/AO/GI smooth-normals" : "flat sensor");
    } else {
        try {
            renderer = std::make_unique<render::VulkanRasterRenderer>();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[mpm_jelly_demo] no Vulkan device: %s\n", e.what());
            return 6;
        }
        std::printf("[mpm_jelly_demo] renderer ICD: %s\n", renderer->DeviceName().c_str());
    }

    render::RasterOptions opts;
    opts.width = args.width; opts.height = args.height;
    opts.draw_ground = false;
    opts.hero_framing = false;
    opts.use_camera_override = true;
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 40.0f;
    opts.background = {200, 205, 214, 255};
    opts.ground_color[0] = 0.16f; opts.ground_color[1] = 0.17f; opts.ground_color[2] = 0.20f;
    opts.contact_shadow_strength = 0.0f;
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

    // ~3.3 s (800 steps) covers airborne -> impact -> squash -> rebound -> settle.
    constexpr uint32_t kSteps = 800u;
    uint32_t written = 0u, render_step = 0u;
    size_t first_nonbg = 0u, last_nonbg = 0u;
    int squash_frame = -1, recovered_frame = static_cast<int>(kSteps) - 1, airborne_frame = 0;

    auto render_frame = [&](const std::string* png) {
        download_pos();
        deform_skin();
        render::RenderWorld rw = BuildFrame(topo, skin_pos);
        const float t = kSteps > 1u
            ? static_cast<float>(render_step) / static_cast<float>(kSteps - 1u) : 0.0f;
        const float e = smoothstep(t);
        float bx = 0.0f, by = 0.0f, bz = 0.0f;
        for (uint32_t i = 0; i < P; ++i) { bx += pos[i].x; by += pos[i].y; bz += pos[i].z; }
        bx /= P; by /= P; bz /= P;
        const float az = -0.40f + 0.6f * e;
        const float elev = (19.0f - 4.0f * std::sin(e * kPi)) * kPi / 180.0f;
        const float radius = 0.82f - 0.08f * e;
        const Vec3 look{bx, by, std::max(kBallRadius * 0.8f, bz)};
        opts.camera_eye = {look.x + radius * std::cos(elev) * std::sin(az),
                           look.y - radius * std::cos(elev) * std::cos(az),
                           look.z + radius * std::sin(elev)};
        opts.camera_target = look;
        // The contracting contact shadow: footprint grows as the ball pancakes.
        opts.contact_points.clear();
        float min_z = 1e30f, cx = 0.0f, cy = 0.0f;
        for (uint32_t i = 0; i < P; ++i) { min_z = std::min(min_z, pos[i].z); cx += pos[i].x; cy += pos[i].y; }
        cx /= P; cy /= P;
        const float nearness = std::max(0.0f, std::min(1.0f, 1.0f - min_z / 0.20f));
        render::RasterOptions::ContactPoint cp;
        cp.x = cx; cp.y = cy;
        cp.radius = kBallRadius * (0.9f + 1.1f * nearness);
        cp.strength = 0.85f * (0.10f + 0.90f * nearness);
        opts.contact_points.push_back(cp);

        render::VulkanOffscreenReport rep = gpu ? gpu->Render(rw, opts) : renderer->Render(rw, opts);
        char name[40];
        std::snprintf(name, sizeof(name), "frame_%06u.ppm", written);
        WritePpm(rep, args.out_dir + "/" + name);
        if (written == 0u) first_nonbg = rep.non_background_pixel_count;
        last_nonbg = rep.non_background_pixel_count;
        if (png) {
            if (WritePng(rep, *png)) std::printf("[mpm_jelly_demo] hero PNG -> %s\n", png->c_str());
            else std::fprintf(stderr, "[mpm_jelly_demo] PNG encode failed: %s\n", png->c_str());
        }
        ++written;
    };

    // A physics-only pre-pass picks the hero frames from the live extent/min_z arc.
    {
        nk::World wp(BuildJellyModel(), 1u, dev, backend, Cfg());
        std::vector<Vec3> pp(P, Vec3::Zero());
        std::vector<float> h_of(kSteps, 0.0f), minz_of(kSteps, 0.0f);
        for (uint32_t s = 0; s < kSteps; ++s) {
            wp.Step();
            wp.GetData().DownloadField(nk::FieldId::ParticlePos, pp.data(), P * sizeof(Vec3));
            const Extent e = Measure(pp, P);
            h_of[s] = e.height; minz_of[s] = e.min_z;
        }
        uint32_t first_contact = 0u;
        while (first_contact < kSteps && minz_of[first_contact] > kFloorZ + 0.02f) ++first_contact;
        const uint32_t sq_hi = std::min(kSteps, first_contact + 120u);
        float min_h = 1e30f;
        for (uint32_t s = first_contact; s < sq_hi; ++s)
            if (h_of[s] < min_h) { min_h = h_of[s]; squash_frame = static_cast<int>(s); }
        if (squash_frame < 0)
            for (uint32_t s = 0; s < kSteps; ++s)
                if (h_of[s] < min_h) { min_h = h_of[s]; squash_frame = static_cast<int>(s); }
        float best_air = -1e30f;
        for (int s = 0; s < squash_frame; ++s)
            if (minz_of[s] > best_air) { best_air = minz_of[s]; airborne_frame = s; }
        // Recovered = the tallest rebound frame after the squash (the lift-back).
        float best_h = -1e30f;
        for (uint32_t s = static_cast<uint32_t>(squash_frame) + 1u; s < kSteps; ++s)
            if (h_of[s] > best_h) { best_h = h_of[s]; recovered_frame = static_cast<int>(s); }
        std::printf("[mpm_jelly_demo] hero frames: airborne=%d squash=%d (h=%.4f) recovered=%d (h=%.4f) rest=%.4f\n",
                    airborne_frame, squash_frame, min_h, recovered_frame, best_h, rest.height);
    }

    // 8-12 hero PNGs spanning the arc + every stride-th PPM for the mp4.
    const uint32_t hero[8] = {
        static_cast<uint32_t>(airborne_frame),
        static_cast<uint32_t>(std::max(0, (airborne_frame + squash_frame) / 2)),
        static_cast<uint32_t>(std::max(0, squash_frame - 8)),
        static_cast<uint32_t>(squash_frame),
        static_cast<uint32_t>(std::min<int>(kSteps - 1, squash_frame + 16)),
        static_cast<uint32_t>(std::min<int>(kSteps - 1, (squash_frame + recovered_frame) / 2)),
        static_cast<uint32_t>(recovered_frame),
        static_cast<uint32_t>(std::min<int>(kSteps - 1, recovered_frame + 80))};
    const char* hero_name[8] = {
        "01_airborne.png", "02_falling.png", "03_pre_impact.png", "04_squash.png",
        "05_rebound.png",  "06_rising.png",  "07_recovered.png",  "08_settled.png"};

    for (uint32_t s = 0; s < kSteps; ++s) {
        world.Step();
        const std::string* png = nullptr;
        std::string png_path;
        for (int h = 0; h < 8; ++h)
            if (s == hero[h]) { png_path = args.png_dir + "/" + hero_name[h]; png = &png_path; break; }
        if ((render_step % args.stride) == 0u || png) render_frame(png);
        ++render_step;
    }

    download_pos();
    const Extent fin = Measure(pos, P);
    std::printf("[mpm_jelly_demo] DONE: %u frames -> %s (first_nonbg=%zu last_nonbg=%zu) final_h=%.4f min_z=%.4f\n",
                written, args.out_dir.c_str(), first_nonbg, last_nonbg, fin.height, fin.min_z);
    return written > 0u ? 0 : 7;
}
