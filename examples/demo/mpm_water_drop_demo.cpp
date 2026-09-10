// A rigid bunny drops into MLS-MPM water through the production World pipeline.
// CUDA ray tracing, headless quality checks, and completion timing share this input.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "collision/shape_kind.hpp"
#include "import/cooker/sparse_sdf_cooker.hpp"
#include "import/cooker/sparse_sdf_storage.hpp"
#include "import/mesh_file_loader.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "phi/backend_cuda/cuda_internal.cuh"
#include "phi/op_schema.hpp"
#include "render/mesh_normals.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"
#include "render/rt_adapter.hpp"
#include "render/rt_backend.hpp"
#include "render/rt_framebuffer_to_report.hpp"
#include "runtime/fluid/surface_mesher.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "runtime/soft/tetmesh_topology.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/json.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
namespace fluid = nuka::runtime::fluid;
namespace render = nuka::render;
namespace rt = nuka::rt;
namespace sdfq = nuka::runtime::sdf;
namespace nimport = nuka::import;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using Json = nuka::scene::json::Value;
using Clock = std::chrono::steady_clock;

void Require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

void CheckCuda(cudaError_t status) {
    Require(status == cudaSuccess, cudaGetErrorString(status));
}

double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void WriteJson(const std::string& path, const Json& report) {
    if (path.empty()) return;
    std::ofstream file(path);
    file << report.Dump() << '\n';
    Require(file.good(), "cannot write " + path);
}

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kKindPlane = nuka::collision::kShapePlane;

// The grid x/y walls and floor boundary contain the fluid pool.
constexpr float kFloorZ      = 0.0f;
constexpr float kDx          = 0.011f;
constexpr float kTankHalfXY  = 0.215f;  // tank (domain box) half-extent in x/y.
constexpr float kFluidHalfXY = 0.205f;  // fluid fills the tank cross-section ~flush
                                        // to the wall (no rim gap to dilate into).
constexpr float kPoolTopZ    = 0.18f;   // fluid pool rest height (deep enough to swallow the bunny).
constexpr float kDensity     = 1000.0f; // rho0 (water).
constexpr float kBulk        = 2.0e5f;  // Tait coefficient B; c = sqrt(gamma*B/rho0) at J=1.
constexpr float kTaitGamma   = 7.0f;    // water EOS exponent.
constexpr float kViscosity   = 0.4f;    // Dynamic viscosity in the Newtonian stress.
constexpr uint32_t kSubsteps = 40u;

// The heavy bunny couples through its cooked sparse SDF.
constexpr float kBunnyExtent = 0.14f;   // longest AABB extent (< pool depth so it submerges); big
                                        // enough that the plunge displaces a visible splash.
constexpr float kBunnyMass   = 2.5f;    // heavy (stone-dense over its hull) -> a fast, deep plunge.
constexpr float kDropAbove   = 0.60f;   // release this far above the rest surface.

// Sample the pool at dx/2, giving eight particles per grid cell.
cook::MpmCookInput BuildPoolInput() {
    cook::MpmCookInput in;
    const float pdx = kDx * 0.5f;
    const float lo_z = kFloorZ + pdx;
    for (float x = -kFluidHalfXY; x <= kFluidHalfXY + 1e-4f; x += pdx)
        for (float y = -kFluidHalfXY; y <= kFluidHalfXY + 1e-4f; y += pdx)
            for (float z = lo_z; z <= kPoolTopZ + 1e-4f; z += pdx)
                in.positions.push_back(Vec3{x, y, z});
    const size_t n = in.positions.size();
    in.velocities.assign(n, Vec3::Zero());
    const float vol0 = pdx * pdx * pdx;
    in.inv_mass.assign(n, 1.0f / (kDensity * vol0));
    in.vol0.assign(n, vol0);
    in.material.density = kDensity;
    in.material.model_kind = 3.0f;     // weakly-compressible fluid.
    in.material.bulk_modulus = kBulk;
    in.material.tait_gamma = kTaitGamma;
    in.material.viscosity = kViscosity;
    in.grid_origin = Vec3{-kTankHalfXY, -kTankHalfXY, kFloorZ - 3.0f * kDx};
    // The grid covers the fluid pool and splash; its top boundary is open.
    const float top = kPoolTopZ + 0.54f;
    in.grid_dims[0] = static_cast<uint32_t>(2.0f * kTankHalfXY / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>((top - in.grid_origin.z) / kDx) + 1u;
    in.dx = kDx;
    in.substeps = kSubsteps;
    in.floor_normal = Vec3{0.0f, 0.0f, 1.0f};
    in.floor_d = kFloorZ;
    in.floor_friction = 0.0f;          // free slip on the floor (a fluid).
    return in;
}

// Center the mesh AABB, scale its longest extent, and rotate from Y-up to Z-up.
soft::TriMesh LoadBunnyMesh(const std::string& path) {
    nimport::MeshGeometry g = nimport::LoadObj(path);
    soft::TriMesh m;
    const size_t nv = g.VertexCount();
    m.positions.resize(nv);
    Vec3 lo{1e9f, 1e9f, 1e9f}, hi{-1e9f, -1e9f, -1e9f};
    for (size_t i = 0; i < nv; ++i) {
        Vec3 p{g.vertices[i * 3 + 0], g.vertices[i * 3 + 1], g.vertices[i * 3 + 2]};
        m.positions[i] = p;
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    const Vec3 center{0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z)};
    const float longest = std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z));
    const float scale = longest > 1e-6f ? kBunnyExtent / longest : 1.0f;
    const Quat rx = Quat::FromAxisAngle(Vec3{1, 0, 0}, -kPi * 0.5f);  // Y-up -> Z-up.
    for (Vec3& p : m.positions) p = rx.Rotate((p - center) * scale);
    Vec3 lo2{1e9f, 1e9f, 1e9f}, hi2{-1e9f, -1e9f, -1e9f};
    for (const Vec3& p : m.positions) {
        lo2.x = std::min(lo2.x, p.x); lo2.y = std::min(lo2.y, p.y); lo2.z = std::min(lo2.z, p.z);
        hi2.x = std::max(hi2.x, p.x); hi2.y = std::max(hi2.y, p.y); hi2.z = std::max(hi2.z, p.z);
    }
    const Vec3 c2{0.5f * (lo2.x + hi2.x), 0.5f * (lo2.y + hi2.y), 0.5f * (lo2.z + hi2.z)};
    for (Vec3& p : m.positions) p = p - c2;
    m.triangles.assign(g.indices.begin(), g.indices.end());
    return m;
}

// AABB half-extents of the centered bunny (broadphase bound + inertia source).
Vec3 BunnyHalfExtents(const soft::TriMesh& bunny) {
    Vec3 lo{1e9f, 1e9f, 1e9f}, hi{-1e9f, -1e9f, -1e9f};
    for (const Vec3& p : bunny.positions) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    return Vec3{0.5f * (hi.x - lo.x), 0.5f * (hi.y - lo.y), 0.5f * (hi.z - lo.z)};
}

float SurfaceMinZ(const soft::TriMesh& mesh, const Transform& pose) {
    const Vec3 up = pose.rotation.Conjugate().Rotate(Vec3{0.0f, 0.0f, 1.0f});
    float local_min = std::numeric_limits<float>::infinity();
    for (const Vec3& point : mesh.positions) local_min = std::min(local_min, point.Dot(up));
    return pose.position.z + local_min;
}

// The mesh supplies the SDF and local surface samples; its bounds only cull pairs.
void AddBunnySdf(nk::Model& m, int32_t body_id, const soft::TriMesh& bunny, float voxel) {
    std::vector<float> verts; verts.reserve(bunny.positions.size() * 3);
    for (const Vec3& p : bunny.positions) { verts.push_back(p.x); verts.push_back(p.y); verts.push_back(p.z); }
    nuka::import::cooker::SparseSdfParams sp;
    sp.voxel_size = voxel; sp.band_voxels = 4u; sp.padding_voxels = 2.0f;
    nuka::import::cooker::SparseSdfData sdf = nuka::import::cooker::CookSparseSdf(
        verts.data(), static_cast<uint32_t>(bunny.positions.size()),
        bunny.triangles.data(), static_cast<uint32_t>(bunny.triangles.size() / 3u), sp);
    const uint32_t base = static_cast<uint32_t>(m.sdf_cell_values.size());
    for (uint64_t k : sdf.cell_keys) m.sdf_cell_keys.push_back(k);
    for (float v : sdf.cell_values) m.sdf_cell_values.push_back(v);
    for (uint32_t i = 0; i < sdf.CellCount(); ++i)
        m.sdf_cell_gradients.push_back(Vec3{sdf.cell_gradients[i * 3 + 0],
                                            sdf.cell_gradients[i * 3 + 1],
                                            sdf.cell_gradients[i * 3 + 2]});
    nk::Model::SdfGrid sg;
    sg.origin = Vec3{sdf.origin[0], sdf.origin[1], sdf.origin[2]};
    sg.voxel_size = sdf.voxel_size;
    sg.dims[0] = sdf.dims[0]; sg.dims[1] = sdf.dims[1]; sg.dims[2] = sdf.dims[2];
    sg.cell_offset = base; sg.cell_count = sdf.CellCount();
    const uint32_t grid_idx = static_cast<uint32_t>(m.sdf_grids.size());
    m.sdf_grids.push_back(sg);

    const Vec3 half = BunnyHalfExtents(bunny);
    nk::Model::PairDrivenShape sh;
    sh.kind = nuka::collision::kShapeSdfMesh;
    sh.params[0] = std::sqrt(half.Dot(half));
    sh.params[1] = half.x; sh.params[2] = half.y; sh.params[3] = half.z;
    sh.contype = 1u; sh.conaffinity = 1u;
    sh.sdf_grid = grid_idx; sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
    m.samp_ranges.resize(m.body_init.size() * 2u, 0u);
    m.samp_ranges[static_cast<size_t>(body_id) * 2u] =
        static_cast<uint32_t>(m.samp_points.size() / 3u);
    m.samp_ranges[static_cast<size_t>(body_id) * 2u + 1u] =
        static_cast<uint32_t>(bunny.positions.size());
    m.samp_points.insert(m.samp_points.end(), verts.begin(), verts.end());
}

// The analytic plane supports the rigid mesh and fluid at the pool floor.
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

// Heavy free bunny released above the pool + a static pool-floor plane (the heavy
// bunny rests on it), cooked on top of the fluid via the sim_method=mlsmpm selector.
nk::Model BuildModel(float release_z, const soft::TriMesh& bunny) {
    nk::Model m;
    m.capacities.env_count = 1u;

    const Vec3 half = BunnyHalfExtents(bunny);
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = Vec3{0.0f, 0.0f, release_z};
    bi.inv_mass = 1.0f / kBunnyMass;
    // Solid-box inertia from the bunny AABB (diagonal, principal axes).
    const float ex = 2.0f * half.x, ey = 2.0f * half.y, ez = 2.0f * half.z;
    const float c = (1.0f / 12.0f) * kBunnyMass;
    const Vec3 I{c * (ey * ey + ez * ez), c * (ex * ex + ez * ez), c * (ex * ex + ey * ey)};
    bi.inv_inertia = Vec3{1.0f / I.x, 1.0f / I.y, 1.0f / I.z};
    m.body_init.push_back(bi);
    AddBunnySdf(m, 0, bunny, kDx * 0.6f);

    AddGroundPlane(m, 1);

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.bodies_per_env = bodies; cap.max_bodies_total = bodies;
    cap.max_sdf_grids = static_cast<uint32_t>(m.sdf_grids.size());
    cap.max_sdf_cells = static_cast<uint32_t>(m.sdf_cell_values.size());
    cap.max_samp_points = static_cast<uint32_t>(m.samp_points.size() / 3u);
    m.samp_ranges.resize(static_cast<size_t>(bodies) * 2u, 0u);
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = 16u * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, 1u, soft, BuildPoolInput());
    m.particles.mpm_body_friction = 0.0f;       // fluid: free slip on the body too.
    m.particles.mpm_bite_disable_dynamic_bc = 0u;
    return m;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    return cfg;
}

// det of a row-major 3x3 (the per-particle F packing).
float Det3(const float* F) {
    return F[0] * (F[4] * F[8] - F[5] * F[7]) -
           F[1] * (F[3] * F[8] - F[5] * F[6]) +
           F[2] * (F[3] * F[7] - F[4] * F[6]);
}

// Resting free-surface height: ~97th percentile z of the fluid (robust to a few
// escapee droplets vs the raw max).
float RestSurfaceZ(const std::vector<Vec3>& pos, uint32_t n) {
    if (n == 0u) return 0.0f;
    std::vector<float> zs(n);
    for (uint32_t i = 0; i < n; ++i) zs[i] = pos[i].z;
    std::sort(zs.begin(), zs.end());
    return zs[(static_cast<size_t>(n) * 97u) / 100u];
}

float SurfaceMaxZ(const std::vector<Vec3>& pos, uint32_t n) {
    float hi = -1.0e9f;
    for (uint32_t i = 0; i < n; ++i) hi = std::max(hi, pos[i].z);
    return hi;
}

// Count particles risen above (ref + margin) and the max particle z (the splash).
void SplashStats(const std::vector<Vec3>& pos, uint32_t n, float thresh,
                 uint32_t* count, float* max_z) {
    uint32_t c = 0u; float hi = -1e9f;
    for (uint32_t i = 0; i < n; ++i) {
        hi = std::max(hi, pos[i].z);
        if (pos[i].z > thresh) ++c;
    }
    *count = c; *max_z = hi;
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

// A flat basin slab the size of the tank footprint, top face at z = top.
render::MeshGeometry MakeSlabGeo(float half, float thickness, float top) {
    render::MeshGeometry g;
    const float xs[2] = {-half, half}, ys[2] = {-half, half};
    const float zs[2] = {top - thickness, top};
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
    g.normals = render::SmoothNormals(g.positions, g.indices);
    return g;
}

// Two material parities form checker tiles at the pool floor.
render::MeshGeometry MakeCheckerTiles(float half, uint32_t cells, float top, bool odd) {
    render::MeshGeometry g;
    const float step = (2.0f * half) / static_cast<float>(cells);
    for (uint32_t iy = 0; iy < cells; ++iy)
        for (uint32_t ix = 0; ix < cells; ++ix) {
            if (((ix + iy) & 1u) != (odd ? 1u : 0u)) continue;
            const float x0 = -half + ix * step, x1 = x0 + step;
            const float y0 = -half + iy * step, y1 = y0 + step;
            const uint32_t b = static_cast<uint32_t>(g.positions.size() / 3u);
            g.positions.insert(g.positions.end(), {x0, y0, top, x1, y0, top, x1, y1, top, x0, y1, top});
            g.indices.insert(g.indices.end(), {b, b + 1u, b + 2u, b, b + 2u, b + 3u});
        }
    g.normals = render::SmoothNormals(g.positions, g.indices);
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

// ---- CUDA ray-traced backend (GPU beauty path; the scene changes each frame,
// so it is freed + rebuilt per frame). ----
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
    void SetSamples(uint32_t s) { samples_ = s; }

    render::VulkanOffscreenReport Render(const render::RenderWorld& rw,
                                         const render::RasterOptions& opts) {
        scene_ = render::RenderWorldToTwoLevelScene(rw);
        if (handle_) backend_->FreeScene(handle_);
        handle_ = backend_->BuildScene(scene_);
        ApplyLighting(opts);
        const rt::PinholeCamera cam = CameraFromOptions(opts);
        rt::Framebuffer fb =
            backend_->TraceBeautyToHost(handle_, scene_, cam, BeautyFromOptions(opts));
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
        b.sun_angular_radius = 0.025f;  // crisp sun -> sharp specular glints on the water.
        b.gi_bounces = 1u; b.ao_samples = 4u; b.ao_radius = 0.7f;
        b.seed = 0x9e3779b9u; b.smooth_normals = true;
        b.transmit_bounces = 6u;  // let water rays enter, traverse, and reach the sky.
        b.sky_top = {opts.sky_top[0], opts.sky_top[1], opts.sky_top[2]};
        b.sky_bottom = {opts.sky_bottom[0], opts.sky_bottom[1], opts.sky_bottom[2]};
        b.sky_ground = {opts.ground_color[0], opts.ground_color[1], opts.ground_color[2]};
        b.fog_color = {opts.fog_color[0], opts.fog_color[1], opts.fog_color[2]};
        b.fog_density = opts.fog_density; b.sky_intensity = 0.85f;
        b.sun_disc_radiance = {opts.sun_color[0], opts.sun_color[1], opts.sun_color[2]};
        b.download = rt::AovDownloadMask{};
        b.download.depth = false; b.download.normal = false;
        b.download.albedo = false; b.download.uv = false;
        return b;
    }
    std::unique_ptr<render::RtBackendI> backend_;
    render::RtSceneHandle* handle_ = nullptr;
    rt::TwoLevelScene scene_;
    uint32_t samples_ = 24u;
};

// Isotropic kernels reconstruct the water surface at a calibrated iso fraction.
fluid::FluidSurfaceParams WaterSurfaceParams() {
    fluid::FluidSurfaceParams p;
    const float pdx = kDx * 0.5f;            // MPM particle spacing (8/cell).
    p.h = 2.6f * pdx;                        // SPH support radius: full enough to merge the bulk,
                                             // tight enough that the crown sheet survives.
    p.rest_density_rho0 = kDensity;
    p.iso_fraction = 0.5f;
    p.particle_mass = kDensity * pdx * pdx * pdx;
    p.cell_size = 0.30f * p.h;
    p.anisotropic = false;
    return p;
}

// Laplacian-smooth the marched water mesh in place + recompute smooth normals.
void SmoothWaterMesh(render::MeshGeometry& g, uint32_t iters, float lambda, float mu) {
    if (g.positions.empty() || g.indices.empty()) return;
    std::vector<Vec3> pos(g.positions.size() / 3u);
    for (size_t v = 0; v < pos.size(); ++v)
        pos[v] = Vec3{g.positions[v * 3 + 0], g.positions[v * 3 + 1], g.positions[v * 3 + 2]};
    // Taubin lambda|mu: a shrink pass then an inflate pass per iteration removes
    // marching-cubes bumpiness into a flat water surface without volume loss.
    for (uint32_t it = 0; it < iters; ++it) {
        soft::SmoothSurface(g.indices, 1u, lambda, pos);
        soft::SmoothSurface(g.indices, 1u, mu, pos);
    }
    for (size_t v = 0; v < pos.size(); ++v) {
        g.positions[v * 3 + 0] = pos[v].x;
        g.positions[v * 3 + 1] = pos[v].y;
        g.positions[v * 3 + 2] = pos[v].z;
    }
    g.normals = render::SmoothNormals(g.positions, g.indices);
}

// Render the live bunny pose and water surface with the pool and studio floor.
render::RenderWorld BuildFrame(const std::vector<Vec3>& fluid_pos, uint32_t n,
                               const soft::TriMesh& bunny, const Transform& box_xf,
                               float rest_surface) {
    render::RenderWorld rw;
    rw.materials.push_back(MakeMat(0.66f, 0.34f, 0.22f, 0.0f, 0.45f));            // 0 bunny terracotta
    rw.materials.push_back(MakeMat(0.80f, 0.86f, 0.92f, 0.0f, 0.55f));            // 1 light pool tile
    // Water uses refraction and wavelength-dependent depth absorption.
    rw.materials.push_back(MakeMat(0.82f, 0.90f, 0.95f, 0.0f, 0.04f,
                                   /*transmission=*/0.96f, /*ior=*/1.33f,
                                   /*ax=*/3.5f, /*ay=*/1.3f, /*az=*/0.5f));        // 2 water (teal depth)
    rw.materials.push_back(MakeMat(0.60f, 0.64f, 0.70f, 0.0f, 0.55f));            // 3 studio floor
    rw.materials.push_back(MakeMat(0.28f, 0.52f, 0.70f, 0.0f, 0.45f));            // 4 dark pool tile
    rw.materials.push_back(MakeMat(0.16f, 0.18f, 0.21f, 0.0f, 0.70f));            // 5 dark slate wall (opaque)
    rw.default_material_id = 0u;

    render::MeshGeometry bunny_geo;
    bunny_geo.positions.reserve(bunny.positions.size() * 3);
    for (const Vec3& p : bunny.positions) {
        bunny_geo.positions.push_back(p.x); bunny_geo.positions.push_back(p.y); bunny_geo.positions.push_back(p.z);
    }
    bunny_geo.indices = bunny.triangles;
    bunny_geo.normals = render::SmoothNormals(bunny_geo.positions, bunny_geo.indices);

    const uint32_t bunny_mesh = rw.meshes.InternPrimitive(
        "bunny:rest", [&] { return bunny_geo; });
    const uint32_t basin_mesh = rw.meshes.InternPrimitive(
        "basin:slab", [&] { return MakeSlabGeo(kFluidHalfXY, 0.04f, kFloorZ); });
    const uint32_t tiles_even = rw.meshes.InternPrimitive(
        "basin:tiles_even", [&] { return MakeCheckerTiles(kFluidHalfXY, 8u, kFloorZ + 0.001f, false); });
    const uint32_t tiles_odd = rw.meshes.InternPrimitive(
        "basin:tiles_odd", [&] { return MakeCheckerTiles(kFluidHalfXY, 8u, kFloorZ + 0.001f, true); });
    const uint32_t floor_mesh = rw.meshes.InternPrimitive(
        "floor:slab", [&] { return MakeSlabGeo(3.0f, 0.10f, kFloorZ - 0.06f); });

    auto add = [&](uint32_t mesh, uint32_t mat, const Transform& xf) {
        render::RenderInstance inst;
        inst.mesh_id = mesh; inst.render_material_id = mat; inst.world_xform = xf;
        inst.pose_source.kind = render::PoseSource::Kind::Static;
        rw.instances.push_back(inst);
    };

    add(bunny_mesh, 0u, box_xf);
    add(basin_mesh, 1u, Transform::Identity());
    add(tiles_even, 1u, Transform::Identity());
    add(tiles_odd, 4u, Transform::Identity());
    add(floor_mesh, 3u, Transform::Identity());

    // Open-top opaque pool container: four low slate wall slabs just outside the
    // fluid footprint, from the floor up past the rest surface (BC at kTankHalfXY).
    const float wt = 0.012f;                         // wall thickness.
    const float wall_top = kPoolTopZ + 0.01f;        // rim ~ at the rest surface so a low camera sees the water.
    const float wo = kTankHalfXY;                    // wall inner face at the BC.
    const float wlen = kTankHalfXY + wt;
    // Build each wall as an explicit slab in world space (x-walls span y, y-walls span x).
    auto add_wall = [&](const Vec3& center, const Vec3& half) {
        render::MeshGeometry g;
        const float xs[2] = {center.x - half.x, center.x + half.x};
        const float ys[2] = {center.y - half.y, center.y + half.y};
        const float zs[2] = {center.z - half.z, center.z + half.z};
        for (int xi = 0; xi < 2; ++xi)
            for (int yi = 0; yi < 2; ++yi)
                for (int zi = 0; zi < 2; ++zi)
                    g.positions.insert(g.positions.end(), {xs[xi], ys[yi], zs[zi]});
        auto vv = [](int x, int y, int z) -> uint32_t { return static_cast<uint32_t>(x*4 + y*2 + z); };
        const uint32_t faces[6][4] = {
            {vv(0,0,0), vv(0,1,0), vv(0,1,1), vv(0,0,1)}, {vv(1,0,0), vv(1,0,1), vv(1,1,1), vv(1,1,0)},
            {vv(0,0,0), vv(0,0,1), vv(1,0,1), vv(1,0,0)}, {vv(0,1,0), vv(1,1,0), vv(1,1,1), vv(0,1,1)},
            {vv(0,0,0), vv(1,0,0), vv(1,1,0), vv(0,1,0)}, {vv(0,0,1), vv(0,1,1), vv(1,1,1), vv(1,0,1)},
        };
        for (auto& f : faces) g.indices.insert(g.indices.end(), {f[0],f[1],f[2], f[0],f[2],f[3]});
        g.normals = render::SmoothNormals(g.positions, g.indices);
        char key[48]; std::snprintf(key, sizeof(key), "pool:w_%.3f_%.3f", center.x, center.y);
        const uint32_t mesh = rw.meshes.InternPrimitive(key, [&] { return g; });
        add(mesh, 5u, Transform::Identity());
    };
    const float wzc = kFloorZ + 0.5f * wall_top, wzh = 0.5f * wall_top;
    add_wall(Vec3{-(wo + 0.5f * wt), 0.0f, wzc}, Vec3{0.5f * wt, wlen, wzh});  // -x wall
    add_wall(Vec3{+(wo + 0.5f * wt), 0.0f, wzc}, Vec3{0.5f * wt, wlen, wzh});  // +x wall
    add_wall(Vec3{0.0f, -(wo + 0.5f * wt), wzc}, Vec3{wlen, 0.5f * wt, wzh});  // -y wall
    add_wall(Vec3{0.0f, +(wo + 0.5f * wt), wzc}, Vec3{wlen, 0.5f * wt, wzh});  // +y wall

    render::MeshGeometry water_geo = fluid::MarchFluidSurface(fluid_pos, WaterSurfaceParams());
    SmoothWaterMesh(water_geo, 2u, 0.5f, -0.53f);
    const uint32_t water_mesh = rw.meshes.InternPrimitive("water:live", [&] { return water_geo; });
    add(water_mesh, 2u, Transform::Identity());
    return rw;
}

// ---- CLI ------------------------------------------------------------------
struct Args {
    uint32_t width = 1920u;
    uint32_t height = 1080u;
    uint32_t samples = 24u;
    std::string png_dir = "out/demo1_mpm_water";
    bool probe = false;
    bool video = false;
    uint32_t video_stride = 1u;
    uint32_t settle_steps = 240u;
    uint32_t drop_steps = 420u;
    uint32_t start_step = 0u;   // video: skip free-fall steps before the box enters frame.
    std::string dump_path;      // write per-step snapshots here after the sim, then exit.
    std::string from_path;      // load snapshots from here and render WITHOUT simulating.
    std::string execution = "eager";
    std::string perf_json;
    std::string state_output;
};
Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next_value = [&]() -> std::string {
            Require(i + 1 < argc, "missing value for " + s);
            return argv[++i];
        };
        auto next_u = [&](uint32_t) -> uint32_t {
            const auto value = next_value();
            size_t consumed = 0u;
            Require(!value.empty() && value.front() != '-', "invalid unsigned integer");
            const auto parsed = std::stoull(value, &consumed);
            Require(consumed == value.size() && parsed <= std::numeric_limits<uint32_t>::max(),
                    "unsigned integer out of range");
            return static_cast<uint32_t>(parsed);
        };
        if (s == "--width") a.width = next_u(a.width);
        else if (s == "--height") a.height = next_u(a.height);
        else if (s == "--samples") a.samples = std::max(1u, next_u(a.samples));
        else if (s == "--probe") a.probe = true;
        else if (s == "--video") a.video = true;
        else if (s == "--video-stride") a.video_stride = std::max(1u, next_u(a.video_stride));
        else if (s == "--settle") a.settle_steps = next_u(a.settle_steps);
        else if (s == "--drop") a.drop_steps = next_u(a.drop_steps);
        else if (s == "--start") a.start_step = next_u(a.start_step);
        else if (s == "--png-dir") a.png_dir = next_value();
        else if (s == "--dump") a.dump_path = next_value();
        else if (s == "--from") a.from_path = next_value();
        else if (s == "--execution") a.execution = next_value();
        else if (s == "--perf-json") a.perf_json = next_value();
        else if (s == "--state-output") a.state_output = next_value();
        else throw std::invalid_argument("unknown option " + s);
    }
    Require(a.width > 0u && a.height > 0u && a.drop_steps > 0u &&
            (a.execution == "eager" || a.execution == "graph"), "invalid demo configuration");
    Require(a.from_path.empty() || (a.perf_json.empty() && a.state_output.empty()),
            "simulation measurements cannot be requested with --from");
    Require(!a.probe || a.dump_path.empty(), "--probe does not collect render snapshots for --dump");
    return a;
}

Json Distribution(const std::vector<double>& values, size_t begin, size_t end) {
    std::vector<double> sorted(values.begin() + begin, values.begin() + end);
    Json result = Json::Object();
    result.Set("count", Json::Int(sorted.size()));
    if (sorted.empty()) return result;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) { return sorted[static_cast<size_t>(p * (sorted.size() - 1u))]; };
    result.Set("mean_ms", Json::Float(std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size()));
    result.Set("p50_ms", Json::Float(percentile(0.5)));
    result.Set("p95_ms", Json::Float(percentile(0.95)));
    result.Set("p99_ms", Json::Float(percentile(0.99)));
    result.Set("min_ms", Json::Float(sorted.front()));
    result.Set("max_ms", Json::Float(sorted.back()));
    return result;
}

class StepMeasurements {
public:
    StepMeasurements(nk::World& world, bool enabled) : world_(world), enabled_(enabled) {
        if (!enabled_) return;
        Require(std::strcmp(nphi::BackendName(world.Backend()), "cuda") == 0,
                "completion timing requires the CUDA backend");
        auto* backend = reinterpret_cast<nphi::CudaBackend*>(world.Backend());
        CheckCuda(cudaSetDevice(backend->device_id));
        stream_ = nphi::CudaBackendMainStream(backend);
        CheckCuda(cudaEventCreate(&begin_));
        const auto status = cudaEventCreate(&end_);
        if (status != cudaSuccess) { cudaEventDestroy(begin_); CheckCuda(status); }
    }
    ~StepMeasurements() {
        if (begin_) cudaEventDestroy(begin_);
        if (end_) cudaEventDestroy(end_);
    }
    void Step() {
        if (enabled_) CheckCuda(cudaEventRecord(begin_, stream_));
        const auto start = Clock::now();
        const auto status = world_.StepConfigured();
        const double host_ms = Milliseconds(start);
        if (status != nphi::Status::Ok) {
            const auto& error = world_.LastExecutionError();
            throw std::runtime_error("step failed, op " +
                std::to_string(static_cast<unsigned>(error.failed_op)) + ": " + error.message);
        }
        if (!enabled_) return;
        CheckCuda(cudaEventRecord(end_, stream_));
        CheckCuda(cudaEventSynchronize(end_));
        wall_ms_.push_back(Milliseconds(start));
        float gpu_ms = 0.0f;
        CheckCuda(cudaEventElapsedTime(&gpu_ms, begin_, end_));
        gpu_ms_.push_back(gpu_ms);
        host_ms_.push_back(host_ms);
    }
    Json Report(uint32_t settle_steps) const {
        Json result = Json::Object();
        result.Set("boundary", Json::Str("World step completion; uploads, downloads, quality scans and rendering excluded"));
        for (unsigned phase = 0; phase < 2u; ++phase) {
            const size_t begin = phase == 0u ? 0u : settle_steps;
            const size_t end = phase == 0u ? settle_steps : gpu_ms_.size();
            Json item = Json::Object();
            item.Set("gpu_completion", Distribution(gpu_ms_, begin, end));
            item.Set("host_call", Distribution(host_ms_, begin, end));
            item.Set("synchronized_wall", Distribution(wall_ms_, begin, end));
            result.Set(phase == 0u ? "settle" : "drop", std::move(item));
        }
        return result;
    }
private:
    nk::World& world_;
    bool enabled_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t begin_ = nullptr, end_ = nullptr;
    std::vector<double> gpu_ms_, host_ms_, wall_ms_;
};

void HashBytes(uint64_t& hash, const void* data, size_t bytes) {
    const auto* source = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) hash = (hash ^ source[i]) * 1099511628211ull;
}

std::string FormatHash(uint64_t hash) {
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << hash;
    return text.str();
}

void SaveState(nk::World& world, const std::string& path) {
    if (path.empty()) return;
    std::ofstream file(path, std::ios::binary);
    const auto write = [&](const void* data, size_t bytes) {
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        Require(file.good(), "cannot write " + path);
    };
    const std::array<nk::FieldId, 10> fields = {
        nk::FieldId::ParticlePos, nk::FieldId::ParticleVel, nk::FieldId::ParticleF,
        nk::FieldId::ParticleC, nk::FieldId::BodyPose, nk::FieldId::BodyLinearVelocity,
        nk::FieldId::BodyAngularVelocity, nk::FieldId::MpmBodyReaction,
        nk::FieldId::MpmBodyAngReaction, nk::FieldId::EnvStatus};
    for (auto field : fields) {
        const uint64_t size = world.GetModel().capacities.ElementCount(field) * nk::LayoutOf(field).elem_size;
        Require(size > 0u, "missing state field");
        std::vector<uint8_t> bytes(size);
        Require(world.GetData().DownloadField(field, bytes.data(), bytes.size()), "state download failed");
        const uint32_t id = static_cast<uint32_t>(field);
        const uint64_t count = bytes.size();
        write(&id, sizeof(id)); write(&count, sizeof(count)); write(bytes.data(), bytes.size());
    }
}

// Particle positions and bunny poses can be saved and rendered independently.
struct RenderData {
    std::vector<std::vector<Vec3>> fluid_snap;
    std::vector<Transform> pose_snap;
    float rest_surface = 0.0f;
    uint32_t peak_splash_step = 0u;
    uint32_t particle_count = 0u;
};

bool DumpSnap(const std::string& path, const RenderData& rd) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    bool ok = true;
    const char magic[4] = {'N', 'W', 'D', '1'};
    const uint32_t nf = static_cast<uint32_t>(rd.fluid_snap.size());
    ok = ok && std::fwrite(magic, 1, 4, f) == 4;
    ok = ok && std::fwrite(&nf, sizeof(uint32_t), 1, f) == 1;
    ok = ok && std::fwrite(&rd.particle_count, sizeof(uint32_t), 1, f) == 1;
    ok = ok && std::fwrite(&rd.rest_surface, sizeof(float), 1, f) == 1;
    ok = ok && std::fwrite(&rd.peak_splash_step, sizeof(uint32_t), 1, f) == 1;
    for (uint32_t i = 0; i < nf && ok; ++i) {
        ok = ok && rd.fluid_snap[i].size() == rd.particle_count;
        ok = ok && std::fwrite(rd.fluid_snap[i].data(), sizeof(Vec3),
                               rd.particle_count, f) == rd.particle_count;
        ok = ok && std::fwrite(&rd.pose_snap[i], sizeof(Transform), 1, f) == 1;
    }
    ok = (std::fclose(f) == 0) && ok;
    return ok;
}

bool LoadSnap(const std::string& path, RenderData& rd) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    bool ok = true;
    char magic[4] = {0};
    uint32_t nf = 0u;
    ok = ok && std::fread(magic, 1, 4, f) == 4 && magic[0] == 'N' &&
         magic[1] == 'W' && magic[2] == 'D' && magic[3] == '1';
    ok = ok && std::fread(&nf, sizeof(uint32_t), 1, f) == 1;
    ok = ok && std::fread(&rd.particle_count, sizeof(uint32_t), 1, f) == 1;
    ok = ok && std::fread(&rd.rest_surface, sizeof(float), 1, f) == 1;
    ok = ok && std::fread(&rd.peak_splash_step, sizeof(uint32_t), 1, f) == 1;
    if (ok) { rd.fluid_snap.resize(nf); rd.pose_snap.resize(nf); }
    for (uint32_t i = 0; i < nf && ok; ++i) {
        rd.fluid_snap[i].resize(rd.particle_count);
        ok = ok && std::fread(rd.fluid_snap[i].data(), sizeof(Vec3),
                              rd.particle_count, f) == rd.particle_count;
        ok = ok && std::fread(&rd.pose_snap[i], sizeof(Transform), 1, f) == 1;
    }
    std::fclose(f);
    return ok;
}

}  // namespace

// Run settling and release; return a process status or -1 to render snapshots.
int RunSim(const Args& args, const soft::TriMesh& bunny, const Vec3& bunny_half,
           RenderData& out) {
    nphi::Device* dev = nphi::InitBestDevice();
    nphi::Backend* backend = dev ? nphi::DeviceInitBackend(dev, nullptr) : nullptr;
    if (backend == nullptr) {
        std::fprintf(stderr, "[mpm_water_drop] no CUDA backend\n");
        return 2;
    }
    std::unique_ptr<nphi::Backend, decltype(&nphi::BackendFree)> owner(backend, &nphi::BackendFree);

    // Place the bunny bottom above the initial pool surface by kDropAbove.
    const float release_z = kPoolTopZ + bunny_half.z + kDropAbove;
    const auto cook_start = Clock::now();
    nk::Model model = BuildModel(release_z, bunny);
    const double cook_ms = Milliseconds(cook_start);
    const uint32_t P = model.capacities.particles_per_env;
    const uint32_t B = model.capacities.bodies_per_env;
    const auto create_start = Clock::now();
    nk::World world(std::move(model), 1u, dev, backend, Cfg());
    if (!world.Ready()) {
        std::fprintf(stderr, "[mpm_water_drop] world not ready\n");
        return 3;
    }
    Require(world.Synchronize() == nphi::Status::Ok, "world creation did not complete");
    const double create_ms = Milliseconds(create_start);
    const auto capture_start = Clock::now();
    Require(world.SetExecutionMode(args.execution == "graph" ? nk::World::ExecutionMode::Graph
                                                            : nk::World::ExecutionMode::Eager) == nphi::Status::Ok,
            "execution mode is unavailable");
    if (args.execution == "graph")
        Require(world.PrepareGraph() == nphi::Status::Ok, world.LastExecutionError().message);
    const double capture_ms = Milliseconds(capture_start);
    StepMeasurements measurements(world, !args.perf_json.empty());
    std::fprintf(stderr,
                 "[mpm_water_drop] fluid particles=%u bodies=%u substeps=%u K=%.1e visc=%.1f "
                 "dx=%.3f release_z=%.4f bunny_mass=%.2f\n",
                 P, B, kSubsteps, static_cast<double>(kBulk), static_cast<double>(kViscosity),
                 static_cast<double>(kDx), release_z, static_cast<double>(kBunnyMass));

    nk::Data& d = world.GetData();
    std::vector<Vec3> fpos(P, Vec3::Zero());
    std::vector<Transform> body(B, Transform::Identity());
    std::vector<float> F(static_cast<size_t>(P) * 9u, 0.0f);
    std::vector<Vec3> fvel(P), body_vel(B), body_omega(B), body_reaction(B);
    uint32_t status_union = 0u;
    uint64_t trajectory_hash = 14695981039346656037ull;
    float min_J = 1e30f, max_J = -1e30f;
    double current_volume_ratio = 1.0, max_volume_ratio_error = 0.0;
    double fluid_kinetic_j = 0.0;
    bool nonfinite = false;
    Json samples = Json::Array();
    auto download = [&] {
        const auto read = [&](nk::FieldId field, void* target, size_t bytes) {
            Require(d.DownloadField(field, target, bytes), std::string("download failed: ") + nk::FieldName(field));
        };
        read(nk::FieldId::ParticlePos, fpos.data(), fpos.size() * sizeof(Vec3));
        read(nk::FieldId::ParticleVel, fvel.data(), fvel.size() * sizeof(Vec3));
        read(nk::FieldId::BodyPose, body.data(), body.size() * sizeof(Transform));
        read(nk::FieldId::BodyLinearVelocity, body_vel.data(), body_vel.size() * sizeof(Vec3));
        read(nk::FieldId::BodyAngularVelocity, body_omega.data(), body_omega.size() * sizeof(Vec3));
        read(nk::FieldId::MpmBodyReaction, body_reaction.data(), body_reaction.size() * sizeof(Vec3));
        read(nk::FieldId::ParticleF, F.data(), F.size() * sizeof(float));
        uint32_t status = 0u;
        read(nk::FieldId::EnvStatus, &status, sizeof(status));
        status_union |= status;
        double sum_J = 0.0, sum_v2 = 0.0;
        const auto finite_vec = [](const Vec3& v) {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        };
        for (uint32_t i = 0; i < P; ++i) {
            const float J = Det3(&F[static_cast<size_t>(i) * 9u]);
            nonfinite |= !finite_vec(fpos[i]) || !finite_vec(fvel[i]) || !std::isfinite(J);
            min_J = std::min(min_J, J); max_J = std::max(max_J, J);
            sum_J += J;
            sum_v2 += static_cast<double>(fvel[i].LengthSq());
        }
        for (uint32_t i = 0; i < B; ++i)
            nonfinite |= !finite_vec(body[i].position) || !finite_vec(body_vel[i]) ||
                         !finite_vec(body_omega[i]) || !finite_vec(body_reaction[i]);
        current_volume_ratio = sum_J / P;
        max_volume_ratio_error = std::max(max_volume_ratio_error, std::abs(current_volume_ratio - 1.0));
        const double pdx = kDx * 0.5;
        fluid_kinetic_j = 0.5 * kDensity * pdx * pdx * pdx * sum_v2;
        if (!args.perf_json.empty()) {
            HashBytes(trajectory_hash, fpos.data(), fpos.size() * sizeof(Vec3));
            HashBytes(trajectory_hash, fvel.data(), fvel.size() * sizeof(Vec3));
            HashBytes(trajectory_hash, F.data(), F.size() * sizeof(float));
            HashBytes(trajectory_hash, body.data(), body.size() * sizeof(Transform));
            HashBytes(trajectory_hash, body_vel.data(), body_vel.size() * sizeof(Vec3));
            HashBytes(trajectory_hash, body_omega.data(), body_omega.size() * sizeof(Vec3));
            HashBytes(trajectory_hash, body_reaction.data(), body_reaction.size() * sizeof(Vec3));
            HashBytes(trajectory_hash, &status, sizeof(status));
        }
    };
    // Pin the bunny (body 0) at a pose with zero velocity each step: its SDF stays out
    // of the pool while the fluid settles, then it is released into free fall.
    const uint64_t pose_off = 0u;          // body 0 is the dropped bunny.
    const uint64_t vec3_off = 0u;
    auto hold_box = [&](const Vec3& at) {
        Transform tf = Transform::Identity(); tf.position = at;
        const Vec3 zero = Vec3::Zero();
        Require(d.UploadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), pose_off), "pose upload failed");
        Require(d.UploadField(nk::FieldId::BodyLinearVelocity, &zero, sizeof(Vec3), vec3_off), "velocity upload failed");
        Require(d.UploadField(nk::FieldId::BodyAngularVelocity, &zero, sizeof(Vec3), vec3_off), "angular velocity upload failed");
    };

    // SETTLE: hold the bunny parked high so its SDF stays clear of the pool while the
    // fluid relaxes to a flat hydrostatic free surface.
    const uint32_t kSettleSteps = args.settle_steps;
    for (uint32_t s = 0; s < kSettleSteps; ++s) {
        hold_box(Vec3{0.0f, 0.0f, release_z});
        measurements.Step();
        download();
        if (s % 50u == 0u) {
            std::fprintf(stderr, "[mpm_water_drop] settle s=%u surf_max=%.4f box_z=%.4f\n",
                         s, SurfaceMaxZ(fpos, P), body[0].position.z);
        }
    }
    if (kSettleSteps == 0u) download();
    const double settled_volume_ratio = current_volume_ratio;
    const float rest_surface = RestSurfaceZ(fpos, P);
    std::fprintf(stderr, "[mpm_water_drop] settled rest_surface=%.4f (max_z=%.4f) box_z=%.4f\n",
                 rest_surface, SurfaceMaxZ(fpos, P), body[0].position.z);
    // RELEASE: drop the bunny from a real height above the settled surface.
    const float drop_z = rest_surface + bunny_half.z + kDropAbove;
    hold_box(Vec3{0.0f, 0.0f, drop_z});
    std::fprintf(stderr, "[mpm_water_drop] release bunny at z=%.4f (%.3f m above surface)\n",
                 drop_z, drop_z - rest_surface);

    // The drop window: impact -> crown -> submerge -> settle.
    const uint32_t kDropSteps = args.drop_steps;
    const float splash_thresh = rest_surface + 0.03f;

    // Per-step snapshots (fluid + box pose) for the render loop, captured into `out`.
    std::vector<std::vector<Vec3>>& fluid_snap = out.fluid_snap;
    std::vector<Transform>& pose_snap = out.pose_snap;
    if (!args.probe) { fluid_snap.reserve(kDropSteps); pose_snap.reserve(kDropSteps); }

    float peak_surface = rest_surface, box_min_z = 1.0e9f, min_vz = 0.0f;
    float max_splash_z = -1e9f, max_react = 0.0f;
    uint32_t max_splash_count = 0u, peak_splash_step = 0u, escape = 0u;
    bool decel = false;
    float prev_vz = 0.0f;

    Vec3 lin_vel{0, 0, 0}, reaction{0, 0, 0};

    for (uint32_t s = 0; s < kDropSteps; ++s) {
        measurements.Step();
        download();
        lin_vel = body_vel[0];
        reaction = body_reaction[0];
        const float box_z = body[0].position.z;
        const float bottom_z = SurfaceMinZ(bunny, body[0]);
        const float surf = SurfaceMaxZ(fpos, P);
        uint32_t sc; float smz;
        SplashStats(fpos, P, splash_thresh, &sc, &smz);
        peak_surface = std::max(peak_surface, surf);
        max_splash_z = std::max(max_splash_z, smz);
        if (sc > max_splash_count) { max_splash_count = sc; peak_splash_step = s; }
        box_min_z = std::min(box_min_z, bottom_z);
        min_vz = std::min(min_vz, lin_vel.z);
        max_react = std::max(max_react, reaction.z);
        if (lin_vel.z > prev_vz + 1e-4f && min_vz < -0.2f) decel = true;
        prev_vz = lin_vel.z;
        nonfinite = nonfinite || !std::isfinite(box_z) || !std::isfinite(lin_vel.z) ||
                    !std::isfinite(surf);
        escape = status_union & nphi::kEnvStatusMpmGridEscape;
        if (s % 15u == 0u || s + 1u == kDropSteps) {
            Json sample = Json::Object();
            sample.Set("drop_step", Json::Int(s));
            sample.Set("bunny_z_m", Json::Float(box_z));
            sample.Set("bunny_surface_min_z_m", Json::Float(bottom_z));
            sample.Set("bunny_vz_m_s", Json::Float(lin_vel.z));
            sample.Set("reaction_z_Ns", Json::Float(reaction.z));
            sample.Set("volume_ratio", Json::Float(current_volume_ratio));
            sample.Set("fluid_kinetic_J", Json::Float(fluid_kinetic_j));
            sample.Set("surface_max_z_m", Json::Float(surf));
            samples.PushBack(std::move(sample));
            std::fprintf(stderr,
                         "[mpm_water_drop] drop s=%u box_z=%.4f vz=%.3f surf=%.4f splash=%u "
                         "react=%.3e minJ=%.3f maxJ=%.3f escape=%u\n",
                         s, box_z, lin_vel.z, surf, sc, static_cast<double>(reaction.z),
                         min_J, max_J, escape);
        }
        if (!args.probe) {
            fluid_snap.emplace_back(fpos.begin(), fpos.begin() + P);
            pose_snap.push_back(body[0]);
        }
    }

    std::fprintf(stderr,
                 "[mpm_water_drop] PROBE rest_surface=%.4f peak_surface=%.4f (rise=%.4f) "
                 "max_splash_count=%u@step%u max_splash_z=%.4f\n",
                 rest_surface, peak_surface, peak_surface - rest_surface,
                 max_splash_count, peak_splash_step, max_splash_z);
    std::fprintf(stderr,
                 "[mpm_water_drop] PROBE box_min_z=%.4f floor_z=%.4f min_vz=%.4f max_react=%.6e "
                 "min_J=%.4f max_J=%.4f decel=%d escape=%u nonfinite=%d final_box_z=%.4f\n",
                 box_min_z, kFloorZ, min_vz, static_cast<double>(max_react), min_J, max_J,
                 decel ? 1 : 0, escape, nonfinite ? 1 : 0, body[0].position.z);

    out.rest_surface = rest_surface;
    out.peak_splash_step = peak_splash_step;
    out.particle_count = P;
    const bool splash = max_splash_count >= 20u && (max_splash_z - rest_surface) > 0.04f;
    const bool two_way = max_react > 1e-4f && decel;
    const bool stable = !nonfinite && escape == 0u && min_J > 0.0f;
    const bool submerged = box_min_z < rest_surface - 0.5f * bunny_half.z;
    const bool volume_ok = max_volume_ratio_error <= 0.05;
    const bool status_ok = status_union == 0u;
    const bool ok = splash && two_way && stable && submerged && volume_ok && status_ok;
    SaveState(world, args.state_output);
    if (!args.perf_json.empty()) {
        Json report = Json::Object(), configuration = Json::Object(), quality = Json::Object();
        report.Set("schema_version", Json::Int(1));
        report.Set("scene", Json::Str("mpm-bunny-water"));
        configuration.Set("execution", Json::Str(args.execution));
        configuration.Set("envs", Json::Int(world.EnvCount()));
        configuration.Set("particles", Json::Int(P));
        configuration.Set("grid_nodes", Json::Int(world.GetModel().capacities.mpm_grid_nodes_per_env));
        configuration.Set("substeps", Json::Int(kSubsteps));
        configuration.Set("dt", Json::Float(Cfg().dt));
        configuration.Set("dx", Json::Float(kDx));
        configuration.Set("bulk_modulus", Json::Float(kBulk));
        configuration.Set("viscosity", Json::Float(kViscosity));
        configuration.Set("collision_geometry", Json::Str("SdfMesh"));
        configuration.Set("collision_surface_samples", Json::Int(world.GetModel().capacities.max_samp_points));
        configuration.Set("sdf_voxel_size", Json::Float(kDx * 0.6f));
        configuration.Set("bunny_mass_kg", Json::Float(kBunnyMass));
        configuration.Set("inertia_model", Json::Str("uniform box with the mesh AABB extents"));
        configuration.Set("settle_steps", Json::Int(kSettleSteps));
        configuration.Set("drop_steps", Json::Int(kDropSteps));
        report.Set("config", std::move(configuration));
        report.Set("cook_ms", Json::Float(cook_ms));
        report.Set("create_ms", Json::Float(create_ms));
        report.Set("capture_ms", Json::Float(capture_ms));
        report.Set("timing", measurements.Report(kSettleSteps));
        uint64_t model_bytes = 0u, arena_bytes[3]{};
        world.GetModel().ComputeModelSegments(&model_bytes);
        nk::Arena::ComputeSegments(world.GetModel().capacities, arena_bytes);
        Json memory = Json::Object();
        memory.Set("data_bytes", Json::Int(arena_bytes[0] + arena_bytes[1] + arena_bytes[2]));
        memory.Set("model_bytes", Json::Int(model_bytes));
        report.Set("memory", std::move(memory));
        quality.Set("finite", Json::Bool(!nonfinite));
        quality.Set("env_status_union", Json::Int(status_union));
        quality.Set("trajectory_fnv1a64", Json::Str(FormatHash(trajectory_hash)));
        quality.Set("trajectory_scope", Json::Str("all settle/drop particle position, velocity, F; body pose, velocities and reaction; status"));
        quality.Set("min_J", Json::Float(min_J));
        quality.Set("max_J", Json::Float(max_J));
        quality.Set("minimum_bunny_surface_z_m", Json::Float(box_min_z));
        quality.Set("final_bunny_surface_z_m", Json::Float(SurfaceMinZ(bunny, body[0])));
        quality.Set("settled_volume_ratio", Json::Float(settled_volume_ratio));
        quality.Set("max_volume_ratio_error", Json::Float(max_volume_ratio_error));
        quality.Set("volume_ratio_error_limit", Json::Float(0.05));
        quality.Set("samples", std::move(samples));
        quality.Set("splash", Json::Bool(splash));
        quality.Set("two_way_response", Json::Bool(two_way));
        quality.Set("submerged", Json::Bool(submerged));
        quality.Set("coupling_complete", Json::Bool(false));
        quality.Set("coupling_scope", Json::Str("SDF bunny; static plane duplicates the MPM grid floor; single owner; finite-mass and articulation feedback remain incomplete"));
        report.Set("quality", std::move(quality));
        Json status = Json::Object();
        status.Set("valid", Json::Bool(ok));
        status.Set("scope", Json::Str("this configured SDF-bunny/grid-boundary workload; not general coupling acceptance"));
        report.Set("status", std::move(status));
        WriteJson(args.perf_json, report);
    }
    if (!args.dump_path.empty()) {
        const bool ok = DumpSnap(args.dump_path, out);
        std::fprintf(stderr, "[mpm_water_drop] DUMP %s %zu frames (P=%u) -> %s\n",
                     ok ? "OK" : "FAIL", out.fluid_snap.size(), P, args.dump_path.c_str());
        return ok ? 0 : 8;
    }

    if (args.probe) {
        std::fprintf(stderr,
                     "[mpm_water_drop] PROBE %s splash=%d two_way=%d stable=%d submerged=%d volume=%d status=%u\n",
                     ok ? "PASS" : "FAIL", splash, two_way, stable, submerged, volume_ok, status_union);
        return ok ? 0 : 5;
    }
    return -1;
}

int RunDemo(const Args& args) {
    const std::string bunny_path =
        std::string(NUKA_SOURCE_DIR) + "/.nuka-assets/stanford/bunny.obj";
    const soft::TriMesh bunny = LoadBunnyMesh(bunny_path);
    Require(!bunny.positions.empty() && !bunny.triangles.empty(), "bunny mesh is unavailable: " + bunny_path);
    const Vec3 bunny_half = BunnyHalfExtents(bunny);
    std::fprintf(stderr, "[mpm_water_drop] bunny verts=%zu tris=%zu half=(%.4f,%.4f,%.4f)\n",
                 bunny.positions.size(), bunny.triangles.size() / 3u,
                 bunny_half.x, bunny_half.y, bunny_half.z);

    RenderData rd;
    if (args.from_path.empty()) {
        const int rc = RunSim(args, bunny, bunny_half, rd);
        if (rc >= 0) return rc;   // probe verdict, dump-only success, or init failure.
    } else {
        if (!LoadSnap(args.from_path, rd)) {
            std::fprintf(stderr, "[mpm_water_drop] failed to load snapshots from %s\n",
                         args.from_path.c_str());
            return 7;
        }
        std::fprintf(stderr, "[mpm_water_drop] loaded %zu frames (P=%u rest_surface=%.4f) from %s\n",
                     rd.fluid_snap.size(), rd.particle_count, rd.rest_surface,
                     args.from_path.c_str());
    }

    // The render loop reads the snapshots regardless of their source (live sim or
    // a --from dump); alias them so the camera/material code below is unchanged.
    auto& fluid_snap = rd.fluid_snap;
    auto& pose_snap = rd.pose_snap;
    const uint32_t P = rd.particle_count;
    const float rest_surface = rd.rest_surface;
    const uint32_t peak_splash_step = rd.peak_splash_step;
    if (fluid_snap.empty()) {
        std::fprintf(stderr, "[mpm_water_drop] no frames to render\n");
        return 9;
    }

    std::filesystem::create_directories(args.png_dir);

    // Studio rig: bright sun, luminous blue sky-dome (the water mirrors it via
    // Fresnel so the top reads as luminous water, not dark grey).
    render::RasterOptions opts;
    opts.width = args.width; opts.height = args.height;
    opts.draw_ground = false; opts.hero_framing = false;
    opts.use_camera_override = true;
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 40.0f;
    opts.background = {214, 226, 240, 255};
    opts.ground_color[0] = 0.16f; opts.ground_color[1] = 0.18f; opts.ground_color[2] = 0.22f;
    opts.contact_shadow_strength = 0.0f;
    opts.use_sun_light = true;
    opts.sun_direction[0] = 0.32f; opts.sun_direction[1] = -0.46f; opts.sun_direction[2] = 0.48f;
    opts.sun_color[0] = 3.0f; opts.sun_color[1] = 2.95f; opts.sun_color[2] = 2.8f;
    opts.sun_ambient_sky[0] = 0.18f; opts.sun_ambient_sky[1] = 0.20f; opts.sun_ambient_sky[2] = 0.24f;
    opts.sun_ambient_ground[0] = 0.10f; opts.sun_ambient_ground[1] = 0.11f; opts.sun_ambient_ground[2] = 0.12f;
    opts.shadow_strength = 0.92f; opts.shadow_map_size = 2560u; opts.shadow_bias = 0.0020f;
    opts.sky_gradient = true;
    opts.sky_top[0] = 0.22f; opts.sky_top[1] = 0.44f; opts.sky_top[2] = 0.82f;
    opts.sky_bottom[0] = 0.50f; opts.sky_bottom[1] = 0.66f; opts.sky_bottom[2] = 0.86f;
    render::RasterOptions::ContactPoint pool_cp;
    pool_cp.x = 0.0f; pool_cp.y = 0.0f; pool_cp.radius = kFluidHalfXY * 1.05f; pool_cp.strength = 0.40f;
    opts.contact_points.push_back(pool_cp);

    // A low camera angle shows the water reflection and splash with a gentle orbit.
    const Vec3 look{0.0f, 0.0f, rest_surface + 0.05f};
    const float cam_r = kFluidHalfXY * 3.7f;
    const float cam_elev = 20.0f * kPi / 180.0f;
    const float az0 = -0.34f, az_sweep = 0.52f;

    if (args.video) {
        const uint32_t total = static_cast<uint32_t>(fluid_snap.size());
        const uint32_t s0 = std::min(args.start_step, total > 0u ? total - 1u : 0u);
        GpuRenderer gpu(BuildFrame(fluid_snap[s0], P, bunny, pose_snap[s0], rest_surface));
        if (!gpu.ok()) { std::fprintf(stderr, "[mpm_water_drop] no CUDA RT backend\n"); return 6; }
        gpu.SetSamples(args.samples);
        uint32_t fi = 0u;
        for (uint32_t s = s0; s < total; s += args.video_stride) {
            const float t = total > s0 + 1u
                ? static_cast<float>(s - s0) / static_cast<float>(total - 1u - s0) : 0.0f;
            const float az = az0 + az_sweep * t;
            opts.camera_eye = {look.x + cam_r * std::cos(cam_elev) * std::sin(az),
                               look.y - cam_r * std::cos(cam_elev) * std::cos(az),
                               look.z + cam_r * std::sin(cam_elev)};
            opts.camera_target = look;
            render::RenderWorld rw = BuildFrame(fluid_snap[s], P, bunny, pose_snap[s], rest_surface);
            render::VulkanOffscreenReport rep = gpu.Render(rw, opts);
            char name[160];
            std::snprintf(name, sizeof(name), "%s/v%04u.png", args.png_dir.c_str(), fi);
            if (!WritePng(rep, name)) std::fprintf(stderr, "[mpm_water_drop] PNG fail %s\n", name);
            if (fi % 30u == 0u)
                std::fprintf(stderr, "[mpm_water_drop] video frame %u (step %u) nonbg=%zu\n",
                             fi, s, rep.non_background_pixel_count);
            ++fi;
        }
        std::fprintf(stderr, "[mpm_water_drop] VIDEO: %u frames -> %s\n",
                     fi, args.png_dir.c_str());
        return 0;
    }

    // Hero frames (when not --video): settled, impact, peak splash, settling.
    const uint32_t total = static_cast<uint32_t>(fluid_snap.size());
    const uint32_t hero[4] = {0u, std::min(total - 1u, peak_splash_step / 2u),
                              std::min(total - 1u, peak_splash_step), total - 1u};
    GpuRenderer gpu(BuildFrame(fluid_snap[0], P, bunny, pose_snap[0], rest_surface));
    if (!gpu.ok()) { std::fprintf(stderr, "[mpm_water_drop] no CUDA RT backend\n"); return 6; }
    gpu.SetSamples(args.samples);
    opts.camera_eye = {look.x + cam_r * std::cos(cam_elev) * std::sin(az0),
                       look.y - cam_r * std::cos(cam_elev) * std::cos(az0),
                       look.z + cam_r * std::sin(cam_elev)};
    opts.camera_target = look;
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t s = hero[i];
        render::RenderWorld rw = BuildFrame(fluid_snap[s], P, bunny, pose_snap[s], rest_surface);
        render::VulkanOffscreenReport rep = gpu.Render(rw, opts);
        char name[160];
        std::snprintf(name, sizeof(name), "%s/frame%02u.png", args.png_dir.c_str(), i);
        if (WritePng(rep, name))
            std::fprintf(stderr, "[mpm_water_drop] hero frame step=%u -> %s (nonbg=%zu)\n",
                         s, name, rep.non_background_pixel_count);
    }
    std::fprintf(stderr, "[mpm_water_drop] DONE -> %s\n", args.png_dir.c_str());
    return 0;
}

int main(int argc, char** argv) {
    Args args;
    try {
        args = ParseArgs(argc, argv);
        return RunDemo(args);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[mpm_water_drop] %s\n", error.what());
        Json report = Json::Object(), status = Json::Object();
        status.Set("valid", Json::Bool(false));
        status.Set("error", Json::Str(error.what()));
        report.Set("status", std::move(status));
        try { WriteJson(args.perf_json, report); } catch (...) {}
        return 2;
    }
}
