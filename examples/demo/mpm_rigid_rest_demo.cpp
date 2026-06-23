// ---------------------------------------------------------------------------
// mpm_rigid_rest_demo.cpp -- a free rigid box presses into an MLS-MPM bed.
//
// A free rigid box is dropped onto a firm MLS-MPM continuum bed (sim_method=mlsmpm
// through the config selector). The box couples to the medium TWO-WAY through the
// env-private grid: the box SDF is rasterized onto the grid, the node velocity is
// projected onto the box surface velocity, and the grid->box reaction holds the box
// up. The bed deforms under the box (the press is visible) and the box settles.
//
// RENDER: PER FRAME download the live MPM particle positions, march the bed
// isosurface (the SAME marching-cubes mesher the fluid uses), pose the box cube from
// its live body_pose, render box + bed + studio floor on the CUDA RT backend
// (--gpu --beauty) or lavapipe raster, write a PPM + a handful of hero PNGs.
//
// Built behind NK_BUILD_VULKAN_VALIDATION. Usage:
//   mpm_rigid_rest_demo [--width W] [--height H] [--out-dir DIR] [--png-dir DIR]
//                       [--stride S] [--gpu] [--beauty] [--flat-gpu] [--samples N]
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
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"
#include "render/rt_adapter.hpp"
#include "render/rt_backend.hpp"
#include "render/rt_framebuffer_to_report.hpp"
#include "runtime/fluid/surface_mesher.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace fluid = nuka::runtime::fluid;
namespace render = nuka::render;
namespace rt = nuka::rt;
namespace sdfq = nuka::runtime::sdf;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kPi = 3.14159265358979323846f;

// Scene geometry (mirrors the gate-c rest probe: a firm bed + a free box).
constexpr float kFloorZ    = 0.0f;
constexpr float kDx        = 0.02f;
constexpr float kBedHalfXY = 0.10f;
constexpr float kBedTopZ   = 0.08f;
constexpr float kBoxHalf   = 0.05f;
constexpr float kBoxMass   = 0.5f;
constexpr float kBoxDropZ  = kBedTopZ + kBoxHalf + 0.06f;  // start a bit above the bed.
constexpr uint32_t kSubsteps = 25u;
constexpr uint32_t kKindBox  = 2u;

struct Args {
    uint32_t width = 1600u;
    uint32_t height = 1000u;
    uint32_t stride = 3u;
    bool gpu = false;
    bool beauty = true;
    uint32_t samples = 16u;
    std::string out_dir = "/tmp/mpm_rigid_rest_frames";
    std::string png_dir = "out/mpm_rigid_rest";
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
        else if (s == "--out-dir" && i + 1 < argc) a.out_dir = argv[++i];
        else if (s == "--png-dir" && i + 1 < argc) a.png_dir = argv[++i];
    }
    return a;
}

// A dense analytic box SDF over a narrow band (no mesh dependency): each band voxel
// stores the exact box signed distance + gradient (the body BC samples it).
void AddBoxSdf(nk::Model& m, int32_t body_id, float half) {
    const float vh = kDx;
    const float band = 3.0f * vh;
    const float ext = half + band;
    const int n = static_cast<int>(std::ceil(ext / vh)) + 1;
    const Vec3 origin{-static_cast<float>(n) * vh, -static_cast<float>(n) * vh,
                      -static_cast<float>(n) * vh};
    const uint32_t base = static_cast<uint32_t>(m.sdf_cell_values.size());
    auto box_phi = [&](const Vec3& p, Vec3& grad) -> float {
        const Vec3 d{std::fabs(p.x) - half, std::fabs(p.y) - half, std::fabs(p.z) - half};
        const Vec3 dpos{std::max(d.x, 0.0f), std::max(d.y, 0.0f), std::max(d.z, 0.0f)};
        const float outside = std::sqrt(dpos.LengthSq());
        const float inside = std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f);
        if (outside > 1e-6f) {
            const Vec3 g{dpos.x * (p.x < 0 ? -1.f : 1.f), dpos.y * (p.y < 0 ? -1.f : 1.f),
                         dpos.z * (p.z < 0 ? -1.f : 1.f)};
            const float gl = std::sqrt(g.LengthSq());
            grad = gl > 1e-8f ? g * (1.0f / gl) : Vec3{0, 0, 1};
        } else {
            Vec3 g{0, 0, 0};
            if (d.x >= d.y && d.x >= d.z) g.x = p.x < 0 ? -1.f : 1.f;
            else if (d.y >= d.z) g.y = p.y < 0 ? -1.f : 1.f;
            else g.z = p.z < 0 ? -1.f : 1.f;
            grad = g;
        }
        return outside + inside;
    };
    uint32_t count = 0u;
    for (int i = 0; i <= 2 * n; ++i)
        for (int j = 0; j <= 2 * n; ++j)
            for (int k = 0; k <= 2 * n; ++k) {
                const Vec3 p{origin.x + i * vh, origin.y + j * vh, origin.z + k * vh};
                Vec3 grad{0, 0, 0};
                const float phi = box_phi(p, grad);
                if (std::fabs(phi) > band) continue;
                m.sdf_cell_keys.push_back(sdfq::PackSdfCellKey(
                    static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                    static_cast<uint32_t>(k)));
                m.sdf_cell_values.push_back(phi);
                m.sdf_cell_gradients.push_back(grad);
                ++count;
            }
    nk::Model::SdfGrid sg;
    sg.origin = origin; sg.voxel_size = vh;
    sg.dims[0] = sg.dims[1] = sg.dims[2] = static_cast<uint32_t>(2 * n + 1);
    sg.cell_offset = base; sg.cell_count = count;
    const uint32_t grid_idx = static_cast<uint32_t>(m.sdf_grids.size());
    m.sdf_grids.push_back(sg);

    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
    sh.contype = 1u; sh.conaffinity = 1u;
    sh.sdf_grid = grid_idx; sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

cook::MpmCookInput BuildBedInput() {
    cook::MpmCookInput in;
    const float pdx = kDx * 0.5f;
    const float bed_lo_z = kFloorZ + pdx;
    for (float x = -kBedHalfXY; x <= kBedHalfXY + 1e-4f; x += pdx)
        for (float y = -kBedHalfXY; y <= kBedHalfXY + 1e-4f; y += pdx)
            for (float z = bed_lo_z; z <= kBedTopZ + 1e-4f; z += pdx)
                in.positions.push_back(Vec3{x, y, z});
    const size_t n = in.positions.size();
    in.velocities.assign(n, Vec3::Zero());
    const float vol0 = pdx * pdx * pdx;
    const float density = 1000.0f;
    in.inv_mass.assign(n, 1.0f / (density * vol0));
    in.vol0.assign(n, vol0);
    in.material.youngs = 5.0e5f; in.material.poisson = 0.3f;
    in.material.density = density; in.material.model_kind = 0.0f;
    const float span = kBedHalfXY + 4.0f * kDx;
    in.grid_origin = Vec3{-span, -span, kFloorZ - 4.0f * kDx};
    const float top = kBoxDropZ + kBoxHalf + 4.0f * kDx;
    in.grid_dims[0] = static_cast<uint32_t>(2.0f * span / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>((top - in.grid_origin.z) / kDx) + 1u;
    in.dx = kDx; in.substeps = kSubsteps;
    in.floor_normal = Vec3{0.0f, 0.0f, 1.0f}; in.floor_d = kFloorZ; in.floor_friction = 0.5f;
    return in;
}

nk::Model BuildModel() {
    nk::Model m;
    m.capacities.env_count = 1u;
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = Vec3{0.0f, 0.0f, kBoxDropZ};
    bi.inv_mass = 1.0f / kBoxMass;
    const float I = (1.0f / 6.0f) * kBoxMass * (2.0f * kBoxHalf) * (2.0f * kBoxHalf);
    bi.inv_inertia = Vec3{1.0f / I, 1.0f / I, 1.0f / I};
    m.body_init.push_back(bi);
    AddBoxSdf(m, 0, kBoxHalf);
    nk::Model::BodyInit bf;
    bf.pose = Transform::Identity(); bf.pose.position = Vec3{5.0f, 0.0f, 0.0f};
    bf.inv_mass = 0.0f; bf.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bf);
    AddBoxSdf(m, 1, 0.05f);

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.bodies_per_env = bodies; cap.max_bodies_total = bodies;
    cap.max_sdf_grids = static_cast<uint32_t>(m.sdf_grids.size());
    cap.max_sdf_cells = static_cast<uint32_t>(m.sdf_cell_values.size());
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = 16u * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, 1u, soft, BuildBedInput());
    m.particles.mpm_body_friction = 0.5f;
    return m;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    return cfg;
}

nuka::scene::RenderMaterial MakeMat(float r, float g, float b, float metallic, float rough) {
    nuka::scene::RenderMaterial m;
    m.base_color[0] = r; m.base_color[1] = g; m.base_color[2] = b; m.base_color[3] = 1.0f;
    m.metallic = metallic; m.roughness = rough;
    return m;
}

render::MeshGeometry MakeBoxGeo(float half) {
    render::MeshGeometry g;
    const float xs[2] = {-half, half}, ys[2] = {-half, half}, zs[2] = {-half, half};
    for (int xi = 0; xi < 2; ++xi)
        for (int yi = 0; yi < 2; ++yi)
            for (int zi = 0; zi < 2; ++zi)
                g.positions.insert(g.positions.end(), {xs[xi], ys[yi], zs[zi]});
    auto v = [](int x, int y, int z) -> uint32_t { return static_cast<uint32_t>(x * 4 + y * 2 + z); };
    const uint32_t faces[6][4] = {
        {v(0,0,0), v(0,1,0), v(0,1,1), v(0,0,1)}, {v(1,0,0), v(1,0,1), v(1,1,1), v(1,1,0)},
        {v(0,0,0), v(0,0,1), v(1,0,1), v(1,0,0)}, {v(0,1,0), v(1,1,0), v(1,1,1), v(0,1,1)},
        {v(0,0,0), v(1,0,0), v(1,1,0), v(0,1,0)}, {v(0,0,1), v(0,1,1), v(1,1,1), v(1,0,1)},
    };
    for (auto& f : faces)
        g.indices.insert(g.indices.end(), {f[0], f[1], f[2], f[0], f[2], f[3]});
    return g;
}

render::MeshGeometry MakeFloorGeo(float half, float thickness) {
    render::MeshGeometry g;
    const float xs[2] = {-half, half}, ys[2] = {-half, half}, zs[2] = {-thickness, 0.0f};
    for (int xi = 0; xi < 2; ++xi)
        for (int yi = 0; yi < 2; ++yi)
            for (int zi = 0; zi < 2; ++zi)
                g.positions.insert(g.positions.end(), {xs[xi], ys[yi], zs[zi]});
    auto v = [](int x, int y, int z) -> uint32_t { return static_cast<uint32_t>(x * 4 + y * 2 + z); };
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
    bool ok = hn > 0 && std::fwrite(hdr, 1, static_cast<size_t>(hn), f) == static_cast<size_t>(hn);
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
        scene_ = render::RenderWorldToTwoLevelScene(rw);
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
        b.seed = 0x9e3779b9u; b.smooth_normals = true;
        b.sky_top = {opts.sky_top[0], opts.sky_top[1], opts.sky_top[2]};
        b.sky_bottom = {opts.sky_bottom[0], opts.sky_bottom[1], opts.sky_bottom[2]};
        b.sky_ground = {opts.ground_color[0], opts.ground_color[1], opts.ground_color[2]};
        b.fog_color = {opts.fog_color[0], opts.fog_color[1], opts.fog_color[2]};
        b.fog_density = opts.fog_density; b.sky_intensity = 0.30f;
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

// Bed isosurface params: the marching-cubes mesher over the MPM particle set (the
// same fluid mesher; the medium is data, not a render fork).
fluid::FluidSurfaceParams BedSurfaceParams() {
    fluid::FluidSurfaceParams p;
    p.h = 2.0f * kDx;
    p.rest_density_rho0 = 1.0f;
    p.iso_fraction = 0.20f;
    p.particle_mass = 1.0f;
    p.cell_size = 0.6f * kDx;
    return p;
}

render::RenderWorld BuildFrame(const std::vector<Vec3>& bed_pos, const Transform& box_xf) {
    render::RenderWorld rw;
    rw.materials.push_back(MakeMat(0.78f, 0.62f, 0.36f, 0.0f, 0.85f));  // 0 sandy bed
    rw.materials.push_back(MakeMat(0.20f, 0.45f, 0.62f, 0.2f, 0.30f));  // 1 box
    rw.materials.push_back(MakeMat(0.16f, 0.17f, 0.20f, 0.1f, 0.22f));  // 2 floor
    rw.default_material_id = 0u;
    render::MeshGeometry bed_geo = fluid::MarchFluidSurface(bed_pos, BedSurfaceParams());
    const uint32_t bed_mesh = rw.meshes.InternPrimitive("bed:live", [&] { return bed_geo; });
    const uint32_t box_mesh = rw.meshes.InternPrimitive("box:cube",
                                                        [&] { return MakeBoxGeo(kBoxHalf); });
    const uint32_t floor_mesh = rw.meshes.InternPrimitive("floor:box",
                                                          [&] { return MakeFloorGeo(2.5f, 0.10f); });
    auto add = [&](uint32_t mesh, uint32_t mat, const Transform& xf) {
        render::RenderInstance inst;
        inst.mesh_id = mesh; inst.render_material_id = mat; inst.world_xform = xf;
        inst.pose_source.kind = render::PoseSource::Kind::Static;
        rw.instances.push_back(inst);
    };
    add(bed_mesh, 0u, Transform::Identity());
    add(box_mesh, 1u, box_xf);
    Transform floor_xf = Transform::Identity();
    floor_xf.position.z = -0.010f;
    add(floor_mesh, 2u, floor_xf);
    return rw;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);

    nphi::Device* dev = nphi::InitBestDevice();
    nphi::Backend* backend = dev ? nphi::DeviceInitBackend(dev, nullptr) : nullptr;
    if (backend == nullptr) {
        std::fprintf(stderr, "[mpm_rigid_rest_demo] no CUDA backend\n");
        return 2;
    }

    nk::Model model = BuildModel();
    const uint32_t P = model.capacities.particles_per_env;
    const uint32_t B = model.capacities.bodies_per_env;
    nk::World world(std::move(model), 1u, dev, backend, Cfg());
    if (!world.Ready()) {
        std::fprintf(stderr, "[mpm_rigid_rest_demo] world not ready\n");
        return 3;
    }
    std::printf("[mpm_rigid_rest_demo] mpm bed particles=%u bodies=%u substeps=%u\n",
                P, B, kSubsteps);

    std::vector<Vec3> bed(P, Vec3::Zero());
    std::vector<Transform> body(B, Transform::Identity());
    auto download = [&] {
        world.GetData().DownloadField(nk::FieldId::ParticlePos, bed.data(), P * sizeof(Vec3));
        world.GetData().DownloadField(nk::FieldId::BodyPose, body.data(), B * sizeof(Transform));
    };

    render::RasterOptions opts;
    opts.width = args.width; opts.height = args.height;
    opts.draw_ground = false; opts.hero_framing = false;
    opts.use_camera_override = true;
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 40.0f;
    opts.background = {200, 205, 214, 255};
    opts.ground_color[0] = 0.16f; opts.ground_color[1] = 0.17f; opts.ground_color[2] = 0.20f;
    opts.contact_shadow_strength = 0.0f;
    opts.use_sun_light = true;
    opts.sun_direction[0] = 0.32f; opts.sun_direction[1] = -0.52f; opts.sun_direction[2] = 0.58f;
    opts.sun_color[0] = 3.0f; opts.sun_color[1] = 2.9f; opts.sun_color[2] = 2.7f;
    opts.sun_ambient_sky[0] = 0.16f; opts.sun_ambient_sky[1] = 0.18f; opts.sun_ambient_sky[2] = 0.22f;
    opts.sun_ambient_ground[0] = 0.10f; opts.sun_ambient_ground[1] = 0.10f; opts.sun_ambient_ground[2] = 0.11f;
    opts.shadow_strength = 0.95f; opts.shadow_map_size = 2560u; opts.shadow_bias = 0.0020f;
    opts.sky_gradient = true;
    opts.sky_top[0] = 0.50f; opts.sky_top[1] = 0.55f; opts.sky_top[2] = 0.62f;
    opts.sky_bottom[0] = 0.78f; opts.sky_bottom[1] = 0.80f; opts.sky_bottom[2] = 0.84f;
    // A fixed 3/4 hero camera looking at the box-on-bed.
    const Vec3 look{0.0f, 0.0f, kBedTopZ};
    opts.camera_eye = {0.30f, -0.34f, kBedTopZ + 0.18f};
    opts.camera_target = look;

    download();
    render::RenderWorld rw0 = BuildFrame(bed, body[0]);
    std::unique_ptr<render::VulkanRasterRenderer> renderer;
    std::unique_ptr<GpuRenderer> gpu;
    if (args.gpu) {
        gpu = std::make_unique<GpuRenderer>(rw0);
        if (!gpu->ok()) {
            std::fprintf(stderr, "[mpm_rigid_rest_demo] --gpu: no CUDA RT backend\n");
            return 6;
        }
        gpu->SetBeauty(args.beauty, args.samples);
        std::printf("[mpm_rigid_rest_demo] renderer: CUDA two-level ray tracer (GPU, %s)\n",
                    args.beauty ? "BEAUTY" : "flat");
    } else {
        try {
            renderer = std::make_unique<render::VulkanRasterRenderer>();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[mpm_rigid_rest_demo] no Vulkan device: %s\n", e.what());
            return 6;
        }
        std::printf("[mpm_rigid_rest_demo] renderer ICD: %s\n", renderer->DeviceName().c_str());
    }

    std::filesystem::create_directories(args.out_dir);
    std::filesystem::create_directories(args.png_dir);

    // ~3.75 s (900 steps): drop -> press -> settle. Hero PNGs at key moments.
    constexpr uint32_t kSteps = 900u;
    const uint32_t hero[6] = {0u, 90u, 180u, 320u, 560u, 880u};
    const char* hero_name[6] = {
        "01_drop.png", "02_approach.png", "03_impact.png",
        "04_press.png", "05_settling.png", "06_rest.png"};
    uint32_t written = 0u;
    size_t first_nonbg = 0u, last_nonbg = 0u;

    auto render_frame = [&](const std::string* png) {
        download();
        render::RenderWorld rw = BuildFrame(bed, body[0]);
        render::VulkanOffscreenReport rep = gpu ? gpu->Render(rw, opts) : renderer->Render(rw, opts);
        char name[40];
        std::snprintf(name, sizeof(name), "frame_%06u.ppm", written);
        WritePpm(rep, args.out_dir + "/" + name);
        if (written == 0u) first_nonbg = rep.non_background_pixel_count;
        last_nonbg = rep.non_background_pixel_count;
        if (png) {
            if (WritePng(rep, *png)) std::printf("[mpm_rigid_rest_demo] hero PNG -> %s\n", png->c_str());
            else std::fprintf(stderr, "[mpm_rigid_rest_demo] PNG encode failed: %s\n", png->c_str());
        }
        ++written;
    };

    render_frame(nullptr);  // frame 0 (the drop start).
    for (uint32_t s = 1; s < kSteps; ++s) {
        world.Step();
        const std::string* png = nullptr;
        std::string png_path;
        for (int h = 0; h < 6; ++h)
            if (s == hero[h]) { png_path = args.png_dir + "/" + hero_name[h]; png = &png_path; break; }
        if ((s % args.stride) == 0u || png) render_frame(png);
    }

    download();
    std::printf("[mpm_rigid_rest_demo] DONE: %u frames -> %s (first_nonbg=%zu last_nonbg=%zu) box_z=%.4f\n",
                written, args.out_dir.c_str(), first_nonbg, last_nonbg, body[0].position.z);
    return written > 0u ? 0 : 7;
}
