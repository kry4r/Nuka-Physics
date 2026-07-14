// ---------------------------------------------------------------------------
// water_pool_demo.cpp -- a rigid Stanford bunny dropped into a PBF water pool,
// TWO-WAY coupled, with a visible splash.
//
// A PBF fluid pool (a lattice over a z-up boundary floor, confined by four hidden
// static walls) settles to a flat free surface, then a heavy dynamic rigid bunny is
// released well above the surface and plunges in. The fluid particles collide with
// the bunny through the SAME body<->particle row solver a foot uses on the ground:
// detection -> body<->particle narrowphase -> emit -> SolveRowsBlockIsland -> PBF
// finalize compose. There is NO bespoke fluid coupler -- the bunny displaces the
// fluid (splash) and the displaced fluid brakes/sinks the bunny (two-way), both
// EMERGING from the contact rows.
//
// COLLISION = a single dynamic CONVEX HULL of the bunny verts (the support function
// over the hull point set IS the convex hull). RENDER = the full bunny mesh, smooth
// normals, warm ceramic, posed by the live body transform each frame.
//
// Two render modes (--mode):
//   particles : every fluid particle as an instanced translucent-blue icosphere;
//               flying splash particles read as visible water droplets.
//   surface   : MarchFluidSurface bulk water as a CLEAR dielectric, minimal
//               smoothing, plus the above-rest splash particles as droplet spheres.
//
// Built behind NK_BUILD_VULKAN_VALIDATION. Usage:
//   water_pool_demo [--mode particles|surface] [--width W] [--height H]
//                   [--png-dir DIR] [--samples N] [--probe]
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
#include "import/mesh_file_loader.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "render/mesh_normals.hpp"
#include "render/raster/vulkan_raster_renderer.hpp"
#include "render/render_world.hpp"
#include "render/rt_adapter.hpp"
#include "render/rt_backend.hpp"
#include "render/rt_framebuffer_to_report.hpp"
#include "runtime/fluid/surface_mesher.hpp"
#include "runtime/soft/particle_surface.hpp"
#include "runtime/soft/tetmesh_topology.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
namespace fluid = nuka::runtime::fluid;
namespace render = nuka::render;
namespace rt = nuka::rt;
namespace nimport = nuka::import;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kPi = 3.14159265358979323846f;
constexpr uint32_t kKindBox = nuka::collision::kShapeBox;
constexpr uint32_t kKindPlane = nuka::collision::kShapePlane;
constexpr uint32_t kKindConvexHull = nuka::collision::kShapeConvexHull;

// Fluid pool geometry. A wide lattice on a z-up boundary floor at z=0, confined by
// four static walls so it settles coherent. Fine spacing -> a real crown on impact.
constexpr float kSpacing = 0.025f;
constexpr float kSupportRadius = kSpacing * 1.5f;
constexpr float kRestDensity = 1000.0f;
constexpr uint32_t kPoolNx = 28u;              // ~0.70 m wide (coarser, PBF-stable).target
constexpr uint32_t kPoolNz = 10u;              // ~0.25 m deep, 10 layers (fewer -> less hydrostatic load).target
constexpr uint32_t kFluidCount = kPoolNx * kPoolNx * kPoolNz;  // free particles; boundary slab is appended after.
constexpr float kFloorZ = 0.0f;                // ground plane + PBF boundary (bottom).
constexpr float kPoolBottomZ = kFloorZ + 0.5f * kSpacing;
constexpr float kHalfFoot = 0.5f * static_cast<float>(kPoolNx - 1u) * kSpacing;
constexpr float kWallGap = 0.6f * kSpacing;
constexpr float kWallInner = kHalfFoot + kWallGap;
constexpr float kWallHalfThick = 0.02f;
constexpr float kWallHalfHeight = 0.30f;       // walls clear the deeper rest fill + crown headroom.
constexpr float kContactDMin = kSpacing;       // body<->fluid sphere radius = d_min/2.
constexpr uint32_t kSettleSteps = 120u;        // coarse pool stabilizes by ~step 32; settle fully.

// The bunny: a heavy rigid body (clearly denser than the displaced water) released
// above the rest surface so it arrives at speed and plunges, throwing a crown.
constexpr float kBunnyExtent = 0.20f;          // longest AABB extent; < pool depth so it fully submerges.
constexpr float kBunnyMass = 9.0f;             // heavy -> small inv_mass -> it plunges, not floats.
constexpr float kDropAbove = 0.60f;            // release this far above the rest surface (faster impact -> bigger splash).
constexpr uint32_t kBunnyBody = 5u;            // ground(0)+4 walls(1..4)+bunny(5).

struct Backend { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };

// ---- bunny load + transform ------------------------------------------------
// Load the OBJ, center the AABB, scale longest extent to kBunnyExtent, rotate -90deg
// about X (dataset is Y-up; engine is Z-up) so it sits upright, ears up. Returns the
// full render mesh (positions + flat triangle index list).
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
    // Re-center the rotated, scaled body on its own AABB so the body origin is its
    // centroid (so the convex hull / inertia / pose are consistent).
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

// Extract a convex-hull point set: keep, per direction over a dense icosphere of
// probe directions, the farthest vertex. The union is the convex-hull vertices the
// support function actually visits -- support over this subset matches support over
// the full set for every probed direction (a true hull-vertex extraction, not a
// decimation). Caps the collision pool so the per-pair support scan stays cheap.
std::vector<Vec3> BunnyHullPoints(const std::vector<Vec3>& verts) {
    const soft::TriMesh dirs = soft::BuildIcosphere(Vec3{0, 0, 0}, 1.0f, 3u);  // ~642 dirs.
    std::vector<uint8_t> keep(verts.size(), 0u);
    for (const Vec3& d : dirs.positions) {
        float best = -1e30f; size_t bi = 0;
        for (size_t i = 0; i < verts.size(); ++i) {
            const float s = verts[i].Dot(d);
            if (s > best) { best = s; bi = i; }
        }
        keep[bi] = 1u;
    }
    std::vector<Vec3> out;
    for (size_t i = 0; i < verts.size(); ++i) if (keep[i]) out.push_back(verts[i]);
    return out;
}

// Max |vertex| over the hull (the broadphase bound radius packed into params[0]).
float HullBoundRadius(const std::vector<Vec3>& verts) {
    float hi = 0.0f;
    for (const Vec3& v : verts) hi = std::max(hi, v.Length());
    return hi;
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

// A dynamic convex-hull body: mesh-local hull verts go into the hull pool, the shape
// row records the slice + bound radius. inv_mass > 0 -> the reaction moves it.
void AddBunnyHull(nk::Model& m, const Vec3& pos, const std::vector<Vec3>& verts,
                  float inv_mass, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = inv_mass;
    const float r = HullBoundRadius(verts);
    const float ii = (inv_mass > 0.0f) ? inv_mass / (0.4f * r * r) : 0.0f;
    bi.inv_inertia = Vec3{ii, ii, ii};
    m.body_init.push_back(bi);

    nk::Model::PairDrivenShape sh;
    sh.kind = kKindConvexHull;
    const uint32_t base = static_cast<uint32_t>(m.hull_verts.size() / 3u);
    for (const Vec3& v : verts) {
        m.hull_verts.push_back(v.x); m.hull_verts.push_back(v.y); m.hull_verts.push_back(v.z);
    }
    sh.params[0] = r;  // bound radius (broadphase AABB).
    sh.hull_vert_offset = base;
    sh.hull_vert_count = static_cast<uint32_t>(verts.size());
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

cook::PbfCookInput BuildPool() {
    cook::PbfCookInput in;
    const float s = kSpacing, h = kSupportRadius, rho0 = kRestDensity;
    const float c0 = -kHalfFoot;
    for (uint32_t iz = 0; iz < kPoolNz; ++iz)
        for (uint32_t iy = 0; iy < kPoolNx; ++iy)
            for (uint32_t ix = 0; ix < kPoolNx; ++ix)
                in.positions.push_back(Vec3{c0 + ix * s, c0 + iy * s,
                                            kPoolBottomZ + iz * s});
    // Calibrate particle mass so an interior rest-lattice particle reads EXACTLY rho0
    // (mass = rho0 / sum_j Poly6(r_ij)). The naive rho0*s^3 under-reads (~0.95 rho0 at
    // h=1.5s) -> with cohesion clamped off, gravity then crushes the deep column.
    auto poly6 = [](float r2, float hh) -> float {
        const float h2 = hh * hh; if (r2 >= h2) return 0.0f;
        const float h3 = hh * hh * hh, h9 = h3 * h3 * h3, d = h2 - r2;
        return 315.0f / (64.0f * kPi * h9) * d * d * d;
    };
    float ksum = 0.0f; const int R = static_cast<int>(std::ceil(h / s));
    for (int dz = -R; dz <= R; ++dz) for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx)
            ksum += poly6(static_cast<float>(dx*dx + dy*dy + dz*dz) * s * s, h);
    in.particle_mass = rho0 / ksum;
    const float im = 1.0f / in.particle_mass;
    // Pinned boundary FLOOR (inv_mass 0): two lattice layers below the fluid so the
    // bottom layer reads rho0 and the column gets hydrostatic support. Without solid
    // boundary density an unsupported PBF column free-falls into a thin dense film.
    const int nb = static_cast<int>(std::ceil(kWallInner / s));
    for (int layer = 1; layer <= 2; ++layer) {
        const float bz = kPoolBottomZ - static_cast<float>(layer) * s;
        for (int iy = -nb; iy <= nb; ++iy)
            for (int ix = -nb; ix <= nb; ++ix)
                in.positions.push_back(Vec3{ix * s, iy * s, bz});
    }
    in.velocities.assign(in.positions.size(), Vec3::Zero());
    in.inv_mass.assign(in.positions.size(), 0.0f);          // boundary slab pinned.
    for (uint32_t i = 0; i < kFluidCount; ++i) in.inv_mass[i] = im;  // fluid is free.
    in.rest_density = rho0;
    in.support_radius = h;
    in.cfm_epsilon = 1.0e-6f;
    in.iters = 30u;  // deep column needs many density sweeps.
    in.clamp_overdensity = false;  // TEST: two-sided density holds the surface (cohesion).
    in.boundary_enabled = true;
    in.floor_z = kFloorZ;
    // Side-wall box containment so 39k particles stack into a DEEP pool instead of
    // pancaking across the floor (the thin rigid-wall contact skin leaks under PBF).
    in.box_walls_enabled = true;
    in.box_min = Vec3{-kWallInner, -kWallInner, kFloorZ};
    in.box_max = Vec3{+kWallInner, +kWallInner, 1.0e9f};
    in.friction = 0.0f;  // fluid mu ~= 0: splash stays free, normal-driven.
    // Grid AABB spans the walled footprint + generous vertical headroom for the crown.
    const float span_xy = 2.0f * (kWallInner + kWallHalfThick) + 2.0f * h;
    const float top_z = kWallHalfHeight * 2.0f + 0.70f + 4.0f * h;
    const float lo = -(kWallInner + kWallHalfThick) - h;
    const float grid_zlo = kFloorZ - 3.0f * s - h;  // include the 2-layer boundary floor.
    in.grid_min = Vec3{lo, lo, grid_zlo};
    auto cells = [&](float extent) {
        return static_cast<uint32_t>(std::ceil(extent / h)) + 1u;
    };
    in.grid_dims[0] = cells(span_xy);
    in.grid_dims[1] = cells(span_xy);
    in.grid_dims[2] = cells(top_z - grid_zlo);
    return in;
}

nk::Model BuildScene(const Vec3& bunny_pos, const std::vector<Vec3>& hull_verts,
                     float bunny_inv_mass) {
    nk::Model m;
    const Vec3 wx_half{kWallHalfThick, kWallInner + kWallHalfThick, kWallHalfHeight};
    const Vec3 wy_half{kWallInner + kWallHalfThick, kWallHalfThick, kWallHalfHeight};
    const float wz = kFloorZ + kWallHalfHeight;
    AddGroundPlane(m, 0);
    AddStaticBox(m, Vec3{-(kWallInner + kWallHalfThick), 0.0f, wz}, wx_half, 1);
    AddStaticBox(m, Vec3{+(kWallInner + kWallHalfThick), 0.0f, wz}, wx_half, 2);
    AddStaticBox(m, Vec3{0.0f, -(kWallInner + kWallHalfThick), wz}, wy_half, 3);
    AddStaticBox(m, Vec3{0.0f, +(kWallInner + kWallHalfThick), wz}, wy_half, 4);
    AddBunnyHull(m, bunny_pos, hull_verts, bunny_inv_mass, 5);

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = bodies * 4u;
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    cap.max_hull_verts = static_cast<uint32_t>(m.hull_verts.size() / 3u);
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    cook::CookPbfParticles(m, 1u, BuildPool());
    m.particles.pp_contact_d_min = kContactDMin;  // body<->fluid sphere radius (CRITICAL).
    return m;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;
    // The heavy bunny vs fluid mass gap needs the big velocity budget or it tunnels.
    cfg.vel_iters = 48u;
    cfg.pos_iters = 0u;
    cfg.max_pairs = 64u;
    return cfg;
}

// ---- probes ---------------------------------------------------------------
float SurfaceMaxZ(const std::vector<Vec3>& pos, uint32_t n) {
    float hi = -1.0e9f;
    for (uint32_t i = 0; i < n; ++i) hi = std::max(hi, pos[i].z);
    return hi;
}

// Resting free-surface height: the median z of the top layer of particles, robust
// to a few escapees vs the raw max (used as the splash reference).
float RestSurfaceZ(const std::vector<Vec3>& pos, uint32_t n) {
    std::vector<float> zs(n);
    for (uint32_t i = 0; i < n; ++i) zs[i] = pos[i].z;
    std::sort(zs.begin(), zs.end());
    return zs[(n * 97u) / 100u];  // ~97th percentile = the calm free surface.
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

// NkRow packs to 32 f32: [0]=flags [7]=upper, a.kind [16] a.index [17].
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

// A flat basin slab the size of the pool footprint, top face at z = kFloorZ.
render::MeshGeometry MakeBasinGeo(float half, float thickness) {
    render::MeshGeometry g;
    const float xs[2] = {-half, half};
    const float ys[2] = {-half, half};
    const float zs[2] = {-thickness, 0.0f};
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

// A large studio floor under the basin, top face at z = floor_top.
render::MeshGeometry MakeFloorGeo(float half, float thickness, float floor_top) {
    render::MeshGeometry g = MakeBasinGeo(half, thickness);
    for (size_t i = 0; i < g.positions.size(); i += 3) g.positions[i + 2] += floor_top;
    return g;
}

// The pool-bottom tiles for one checker parity (even or odd), each a flat quad at
// z=top. Two parities rendered with two materials read as a tiled pool floor, so
// light refracted through the clear water reveals structure instead of a flat color.
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

// ---- CUDA ray-traced backend (GPU beauty path; the scene changes each frame, so
// it is freed + rebuilt per frame, like the soft-ball demo). ----
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
        b.fog_density = opts.fog_density; b.sky_intensity = 0.7f;
        // Bright sun disc in the sky so the rippled water surface throws crisp glints.
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

// Water isosurface params over the PBF particle set. Anisotropic kernels (Yu &
// Turk) flatten the free surface into a real water sheet and keep splash droplets
// round; the iso level auto-calibrates so iso_fraction stays meaningful.
fluid::FluidSurfaceParams WaterSurfaceParams() {
    fluid::FluidSurfaceParams p;
    p.h = kSupportRadius;
    p.rest_density_rho0 = kRestDensity;
    p.iso_fraction = 0.44f;
    p.particle_mass = kRestDensity * kSpacing * kSpacing * kSpacing;
    p.cell_size = 0.38f * kSupportRadius;
    p.anisotropic = true;
    p.aniso_lambda = 0.95f;
    p.aniso_kr = 4.0f;
    p.aniso_kn = 0.5f;
    p.aniso_ks = 8000.0f;  // recalibrated to our units (sigma ~ spacing^2).
    p.aniso_n_eps = 25u;
    return p;
}

// Hero-frame snapshot cache: the drop sim is identical across render modes, so the
// 4 hero frames (rest surface + fluid positions + bunny pose) persist to disk and
// render-material tuning reloads them instead of re-running the multi-minute drop.
constexpr uint32_t kHeroCount = 4u;
constexpr uint32_t kCacheMagic = 0x4e574332u;

bool SaveHeroCache(const std::string& path, uint32_t P, float rest_surface,
                   const std::vector<uint32_t>& step, const std::vector<uint32_t>& splash,
                   const std::vector<Transform>& pose,
                   const std::vector<std::vector<Vec3>>& fluid) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    auto wu = [&](uint32_t v) { std::fwrite(&v, sizeof(v), 1, f); };
    auto wf = [&](float v) { std::fwrite(&v, sizeof(v), 1, f); };
    wu(kCacheMagic); wu(P); wu(kHeroCount); wf(rest_surface);
    for (uint32_t h = 0; h < kHeroCount; ++h) { wu(step[h]); wu(splash[h]); }
    for (uint32_t h = 0; h < kHeroCount; ++h) {
        const Transform& t = pose[h];
        wf(t.position.x); wf(t.position.y); wf(t.position.z);
        wf(t.rotation.w); wf(t.rotation.x); wf(t.rotation.y); wf(t.rotation.z);
    }
    for (uint32_t h = 0; h < kHeroCount; ++h)
        for (uint32_t i = 0; i < P; ++i)
            { wf(fluid[h][i].x); wf(fluid[h][i].y); wf(fluid[h][i].z); }
    return std::fclose(f) == 0;
}

bool LoadHeroCache(const std::string& path, uint32_t* P, float* rest_surface,
                   std::vector<uint32_t>& step, std::vector<uint32_t>& splash,
                   std::vector<Transform>& pose, std::vector<std::vector<Vec3>>& fluid) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    auto ru = [&](uint32_t* v) { return std::fread(v, sizeof(*v), 1, f) == 1; };
    auto rf = [&](float* v) { return std::fread(v, sizeof(*v), 1, f) == 1; };
    uint32_t magic = 0, nh = 0;
    bool ok = ru(&magic) && ru(P) && ru(&nh) && rf(rest_surface);
    if (!ok || magic != kCacheMagic || nh != kHeroCount) { std::fclose(f); return false; }
    step.resize(kHeroCount); splash.resize(kHeroCount);
    pose.resize(kHeroCount); fluid.assign(kHeroCount, std::vector<Vec3>(*P));
    for (uint32_t h = 0; h < kHeroCount && ok; ++h) ok = ru(&step[h]) && ru(&splash[h]);
    for (uint32_t h = 0; h < kHeroCount && ok; ++h) {
        Vec3 p; Quat q;
        ok = rf(&p.x) && rf(&p.y) && rf(&p.z) && rf(&q.w) && rf(&q.x) && rf(&q.y) && rf(&q.z);
        pose[h].position = p; pose[h].rotation = q;
    }
    for (uint32_t h = 0; h < kHeroCount && ok; ++h)
        for (uint32_t i = 0; i < *P && ok; ++i)
            ok = rf(&fluid[h][i].x) && rf(&fluid[h][i].y) && rf(&fluid[h][i].z);
    std::fclose(f);
    return ok;
}

// Laplacian-smooth the marched water mesh in place + recompute smooth normals.
void SmoothWaterMesh(render::MeshGeometry& g, uint32_t iters, float lambda, float mu) {
    if (g.positions.empty() || g.indices.empty()) return;
    std::vector<Vec3> pos(g.positions.size() / 3u);
    for (size_t v = 0; v < pos.size(); ++v)
        pos[v] = Vec3{g.positions[v * 3 + 0], g.positions[v * 3 + 1], g.positions[v * 3 + 2]};
    // Taubin lambda|mu: a shrink pass then an inflate pass per iteration removes the
    // marching-cubes bumpiness into a flat water surface without Laplacian volume loss.
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

enum class Mode { Particles, Surface };

// Bake every fluid particle as a translated droplet sphere into ONE combined mesh
// (single BLAS/instance) so the whole cloud renders without hitting the per-scene
// instance cap; smooth sphere normals are translation-invariant so reuse the template.
render::MeshGeometry MakeDropletCloud(const std::vector<Vec3>& pos, uint32_t n,
                                      float radius, float zlo = -1e30f) {
    soft::TriMesh ico = soft::BuildIcosphere(Vec3{0, 0, 0}, radius, 1u);
    render::MeshGeometry tmpl;
    soft::SurfaceTopology t; t.triangles = ico.triangles;
    soft::BuildSurfaceMesh(ico.positions, t, tmpl);
    const uint32_t vstride = tmpl.VertexCount();
    render::MeshGeometry g;
    for (uint32_t i = 0; i < n; ++i) {
        if (pos[i].z < zlo) continue;
        const uint32_t base = static_cast<uint32_t>(g.positions.size() / 3u);
        for (uint32_t v = 0; v < vstride; ++v) {
            g.positions.push_back(tmpl.positions[v * 3 + 0] + pos[i].x);
            g.positions.push_back(tmpl.positions[v * 3 + 1] + pos[i].y);
            g.positions.push_back(tmpl.positions[v * 3 + 2] + pos[i].z);
        }
        g.normals.insert(g.normals.end(), tmpl.normals.begin(), tmpl.normals.end());
        for (uint32_t idx : tmpl.indices) g.indices.push_back(base + idx);
    }
    return g;
}

// One renderable frame: the rigid bunny (full mesh, ceramic) posed this frame + the
// water (mode-dependent) + a bright teal pool-bottom slab + a studio floor.
render::RenderWorld BuildFrame(Mode mode, const std::vector<Vec3>& fluid_pos,
                               uint32_t n_particles, const soft::TriMesh& bunny_rest,
                               const soft::SurfaceTopology& bunny_topo,
                               const Transform& body_pose, float rest_surface) {
    render::RenderWorld rw;
    rw.materials.push_back(MakeMat(0.85f, 0.82f, 0.78f, 0.0f, 0.35f));            // 0 bunny ceramic
    // Pool bottom: a light tiled floor (two checker parities) so light refracted
    // through the clear water reveals structure, reading as water over a real pool.
    rw.materials.push_back(MakeMat(0.78f, 0.86f, 0.92f, 0.0f, 0.55f));            // 1 light pool tile
    // Bulk water dielectric (surface mode): near-clear glass-water, ior 1.33, very
    // light Beer-Lambert tint (red absorbs fastest -> a faint blue-green deepening).
    rw.materials.push_back(MakeMat(0.85f, 0.92f, 0.97f, 0.0f, 0.02f,
                                   /*transmission=*/0.96f, /*ior=*/1.33f,
                                   /*ax=*/0.10f, /*ay=*/0.05f, /*az=*/0.03f));     // 2 clear water
    rw.materials.push_back(MakeMat(0.62f, 0.66f, 0.72f, 0.0f, 0.55f));            // 3 studio floor
    // Particle-mode blue translucent water droplets.
    rw.materials.push_back(MakeMat(0.20f, 0.45f, 0.72f, 0.0f, 0.12f,
                                   /*transmission=*/0.35f, /*ior=*/1.33f,
                                   /*ax=*/0.20f, /*ay=*/0.10f, /*az=*/0.05f));      // 4 droplet water
    rw.materials.push_back(MakeMat(0.30f, 0.55f, 0.72f, 0.0f, 0.45f));            // 5 dark pool tile
    rw.default_material_id = 0u;

    // Bunny: pose the rest verts into world space; smooth normals recomputed in build.
    std::vector<Vec3> bunny_world(bunny_rest.positions.size());
    for (size_t i = 0; i < bunny_rest.positions.size(); ++i)
        bunny_world[i] = body_pose.TransformPoint(bunny_rest.positions[i]);
    render::MeshGeometry bunny_geo;
    soft::BuildSurfaceMesh(bunny_world, bunny_topo, bunny_geo);

    const uint32_t bunny_mesh =
        rw.meshes.InternPrimitive("bunny:live", [&] { return bunny_geo; });
    const uint32_t basin_mesh =
        rw.meshes.InternPrimitive("basin:box", [&] { return MakeBasinGeo(kWallInner, 0.04f); });
    const uint32_t tiles_even_mesh = rw.meshes.InternPrimitive(
        "basin:tiles_even", [&] { return MakeCheckerTiles(kWallInner, 8u, 0.001f, false); });
    const uint32_t tiles_odd_mesh = rw.meshes.InternPrimitive(
        "basin:tiles_odd", [&] { return MakeCheckerTiles(kWallInner, 8u, 0.001f, true); });
    const uint32_t floor_mesh =
        rw.meshes.InternPrimitive("floor:box", [&] { return MakeFloorGeo(3.0f, 0.10f, -0.06f); });
    const float drop_r = 0.6f * kSpacing;  // particle-mode droplet radius (overlap into a body).

    auto add = [&](uint32_t mesh, uint32_t mat, const Transform& xf) {
        render::RenderInstance inst;
        inst.mesh_id = mesh; inst.render_material_id = mat; inst.world_xform = xf;
        inst.pose_source.kind = render::PoseSource::Kind::Static;
        rw.instances.push_back(inst);
    };
    add(bunny_mesh, 0u, Transform::Identity());  // verts already world-space.

    // Basin slab top at z = kFloorZ, then a checker tile layer just above it for
    // refracted structure; studio floor below it.
    Transform basin_xf = Transform::Identity(); basin_xf.position.z = 0.0f;
    add(basin_mesh, 1u, basin_xf);
    add(tiles_even_mesh, 1u, basin_xf);   // light tiles
    add(tiles_odd_mesh, 5u, basin_xf);    // dark tiles
    add(floor_mesh, 3u, Transform::Identity());

    if (mode == Mode::Particles) {
        const uint32_t cloud_mesh = rw.meshes.InternPrimitive(
            "cloud:live", [&] { return MakeDropletCloud(fluid_pos, n_particles, drop_r); });
        add(cloud_mesh, 4u, Transform::Identity());
    } else {
        render::MeshGeometry water_geo =
            fluid::MarchFluidSurface(fluid_pos, WaterSurfaceParams());
        // The anisotropic kernel already flattens the sheet; a minimal Taubin pass only
        // removes residual marching-cubes stair-steps while preserving ripples/splash.
        SmoothWaterMesh(water_geo, 2u, 0.5f, -0.53f);
        const uint32_t water_mesh =
            rw.meshes.InternPrimitive("water:live", [&] { return water_geo; });
        add(water_mesh, 2u, Transform::Identity());
    }
    (void)rest_surface;
    return rw;
}

// ---- CLI ------------------------------------------------------------------
struct Args {
    Mode mode = Mode::Particles;
    uint32_t width = 1280u;
    uint32_t height = 720u;
    uint32_t samples = 24u;
    std::string png_dir;          // default chosen from mode below.
    std::string cache;            // hero-frame snapshot cache (skip the drop sim).
    bool probe = false;
    bool video = false;           // render EVERY drop frame to a PNG sequence (for mp4).
    uint32_t video_stride = 1u;
};
Args ParseArgs(int argc, char** argv) {
    Args a;
    bool png_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next_u = [&](uint32_t def) -> uint32_t {
            return (i + 1 < argc) ? static_cast<uint32_t>(std::atoi(argv[++i])) : def;
        };
        if (s == "--mode" && i + 1 < argc) {
            const std::string v = argv[++i];
            a.mode = (v == "surface") ? Mode::Surface : Mode::Particles;
        } else if (s == "--width") a.width = next_u(a.width);
        else if (s == "--height") a.height = next_u(a.height);
        else if (s == "--samples") a.samples = std::max(1u, next_u(a.samples));
        else if (s == "--probe") a.probe = true;
        else if (s == "--cache" && i + 1 < argc) a.cache = argv[++i];
        else if (s == "--video") a.video = true;
        else if (s == "--video-stride") a.video_stride = std::max(1u, next_u(a.video_stride));
        else if (s == "--png-dir" && i + 1 < argc) { a.png_dir = argv[++i]; png_set = true; }
    }
    if (!png_set)
        a.png_dir = a.mode == Mode::Surface ? "out/bunny_water/surface"
                                            : "out/bunny_water/particles";
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = ParseArgs(argc, argv);

    nphi::Device* dev = nphi::InitBestDevice();
    nphi::Backend* backend = dev ? nphi::DeviceInitBackend(dev, nullptr) : nullptr;
    if (backend == nullptr) {
        std::fprintf(stderr, "[bunny_water] no CUDA backend\n");
        return 2;
    }

    const std::string bunny_path =
        std::string(NUKA_SOURCE_DIR) + "/.nuka-assets/stanford/bunny.obj";
    const soft::TriMesh bunny = LoadBunnyMesh(bunny_path);
    soft::SurfaceTopology bunny_topo;
    bunny_topo.triangles = bunny.triangles;
    const std::vector<Vec3> hull_pts = BunnyHullPoints(bunny.positions);
    const float inv_mass = 1.0f / kBunnyMass;
    std::fprintf(stderr, "[bunny_water] bunny verts=%zu tris=%zu hull_pts=%zu bound_r=%.4f\n",
                 bunny.positions.size(), bunny.triangles.size() / 3u, hull_pts.size(),
                 HullBoundRadius(hull_pts));

    // Hero-frame render inputs come from the snapshot cache, or a fresh drop sim.
    uint32_t P = 0u;
    float rest_surface = 0.0f;
    std::vector<uint32_t> hero_step, hero_splash;
    std::vector<Transform> hero_pose;
    std::vector<std::vector<Vec3>> hero_fluid;
    // Full per-step snapshots (fluid + bunny pose) for the --video sequence render;
    // survives the fresh-sim block so the post-sim render loop can walk every frame.
    std::vector<std::vector<Vec3>> fluid_snap;
    std::vector<Transform> pose_snap;
    const bool loaded = !args.cache.empty() && !args.probe &&
        LoadHeroCache(args.cache, &P, &rest_surface, hero_step, hero_splash,
                      hero_pose, hero_fluid);
    if (loaded) {
        std::fprintf(stderr, "[bunny_water] cache HIT %s P=%u rest_surface=%.4f heroes=",
                     args.cache.c_str(), P, rest_surface);
        for (uint32_t h = 0; h < kHeroCount; ++h)
            std::fprintf(stderr, " %u(splash=%u)", hero_step[h], hero_splash[h]);
        std::fprintf(stderr, "\n");
    } else {
    // Settle the pool flat with the bunny parked high (re-held each step), then
    // release it well above the surface. ONE world drives both stages.
    const float park_z = kWallHalfHeight * 2.0f + 0.30f;
    nk::Model model = BuildScene(Vec3{0, 0, park_z}, hull_pts, inv_mass);
    P = model.capacities.particles_per_env;
    const uint32_t rows = model.capacities.max_rows_per_env;
    nk::World world(std::move(model), 1u, dev, backend, Cfg());
    if (!world.Ready()) {
        std::fprintf(stderr, "[bunny_water] world not ready\n");
        return 3;
    }
    std::fprintf(stderr, "[bunny_water] particles=%u rows=%u mode=%s\n",
                 P, rows, args.mode == Mode::Surface ? "surface" : "particles");

    nk::Data& d = world.GetData();
    const uint64_t pose_off = static_cast<uint64_t>(kBunnyBody) * sizeof(Transform);
    const uint64_t vec3_off = static_cast<uint64_t>(kBunnyBody) * sizeof(Vec3);

    std::vector<Vec3> fpos(P, Vec3::Zero());
    auto download_fluid = [&]() {
        d.DownloadField(nk::FieldId::ParticlePos, fpos.data(), P * sizeof(Vec3));
    };
    auto download_pose = [&]() -> Transform {
        Transform t = Transform::Identity();
        d.DownloadField(nk::FieldId::BodyPose, &t, sizeof(Transform), pose_off);
        return t;
    };
    auto hold_bunny = [&](const Vec3& at) {
        Transform tf = Transform::Identity(); tf.position = at;
        const Vec3 zero = Vec3::Zero();
        d.UploadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), pose_off);
        d.UploadField(nk::FieldId::BodyLinearVelocity, &zero, sizeof(Vec3), vec3_off);
        d.UploadField(nk::FieldId::BodyAngularVelocity, &zero, sizeof(Vec3), vec3_off);
    };

    // SETTLE: bunny parked high, the pool relaxes to a flat free surface.
    for (uint32_t s = 0; s < kSettleSteps; ++s) {
        hold_bunny(Vec3{0, 0, park_z}); world.Step();
        if (s % 30u == 0u) { download_fluid();
            std::fprintf(stderr, "[bunny_water] settle s=%u surf=%.4f\n", s, SurfaceMaxZ(fpos, kFluidCount)); }
    }
    download_fluid();
    rest_surface = RestSurfaceZ(fpos, kFluidCount);
    std::fprintf(stderr, "[bunny_water] settled rest_surface=%.4f (max_z=%.4f)\n",
                 rest_surface, SurfaceMaxZ(fpos, kFluidCount));

    // RELEASE: drop the heavy bunny well above the surface so it arrives fast.
    const float release_z = rest_surface + HullBoundRadius(hull_pts) + kDropAbove;
    hold_bunny(Vec3{0, 0, release_z});
    std::fprintf(stderr, "[bunny_water] release_z=%.4f\n", release_z);

    // The drop -> splash -> ripple -> settle window.
    constexpr uint32_t kDropSteps = 240u;
    std::vector<float> surf_of(kDropSteps, 0.0f), cz_of(kDropSteps, 0.0f);
    std::vector<float> cminz_of(kDropSteps, 0.0f);
    std::vector<uint32_t> splash_of(kDropSteps, 0u);
    float peak_surface = rest_surface, bunny_min_z = 1.0e9f, max_splash_z = -1e9f;
    uint32_t max_splash_count = 0u, peak_splash_step = 0u;
    float max_body_lambda = 0.0f;
    uint32_t particle_rows_seen = 0u;
    bool nan_seen = false;
    const float splash_thresh = rest_surface + 0.03f;

    std::vector<float> urows(static_cast<size_t>(rows) * 32u, 0.0f);
    std::vector<float> lambda(rows, 0.0f);

    if (!args.probe) { fluid_snap.resize(kDropSteps); pose_snap.resize(kDropSteps); }

    for (uint32_t s = 0; s < kDropSteps; ++s) {
        world.Step();
        download_fluid();
        const Transform pose = download_pose();
        surf_of[s] = SurfaceMaxZ(fpos, kFluidCount);
        cz_of[s] = pose.position.z;
        float clo = pose.position.z;
        for (const Vec3& v : bunny.positions) clo = std::min(clo, pose.TransformPoint(v).z);
        cminz_of[s] = clo;
        uint32_t sc; float smz;
        SplashStats(fpos, kFluidCount, splash_thresh, &sc, &smz);
        splash_of[s] = sc;
        if (sc > max_splash_count) { max_splash_count = sc; peak_splash_step = s; }
        max_splash_z = std::max(max_splash_z, smz);
        peak_surface = std::max(peak_surface, surf_of[s]);
        bunny_min_z = std::min(bunny_min_z, clo);
        // Snapshot the FLUID particles only (the pinned boundary slab is excluded).
        if (!args.probe) { fluid_snap[s].assign(fpos.begin(), fpos.begin() + kFluidCount); pose_snap[s] = pose; }

        for (const Vec3& p : fpos)
            if (!(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))) nan_seen = true;
        if (!std::isfinite(pose.position.z)) nan_seen = true;
        // The two-way lambda proof samples on a strided cadence: the per-step host
        // row decode is the dominant cost, and a strided sample still catches lambda>0.
        if (args.probe && (s % 24u == 0u)) {
            d.DownloadField(nk::FieldId::Urows, urows.data(), urows.size() * sizeof(float));
            d.DownloadField(nk::FieldId::Lambda, lambda.data(), lambda.size() * sizeof(float));
            for (uint32_t row = 0u; row < rows; ++row) {
                const RowSides rs = DecodeRow(urows, row);
                if (!rs.active) continue;
                const bool a_part = rs.a_kind == nk::kNkSideParticle;
                const bool b_part = rs.b_kind == nk::kNkSideParticle;
                if (!(a_part || b_part)) continue;
                ++particle_rows_seen;
                if (rs.upper > 1.0e30f) max_body_lambda = std::max(max_body_lambda, lambda[row]);
            }
        }
    }

    std::fprintf(stderr,
                 "[bunny_water] PROBE rest_surface=%.4f peak_surface=%.4f (rise=%.4f) "
                 "max_splash_count=%u@step%u max_splash_z=%.4f thresh=%.4f\n",
                 rest_surface, peak_surface, peak_surface - rest_surface,
                 max_splash_count, peak_splash_step, max_splash_z, splash_thresh);
    std::fprintf(stderr,
                 "[bunny_water] PROBE bunny_min_z=%.4f floor_z=%.4f nan_or_escape=%d "
                 "final_bunny_z=%.4f\n",
                 bunny_min_z, kFloorZ, nan_seen ? 1 : 0, cz_of[kDropSteps - 1u]);

    if (args.probe) {
        const bool splash = max_splash_count >= 20u && (max_splash_z - rest_surface) > 0.05f;
        const bool two_way = particle_rows_seen > 0u && max_body_lambda > 0.0f;
        const bool no_tunnel = bunny_min_z > kFloorZ - 0.5f * kSpacing && !nan_seen;
        const bool ok = splash && two_way && no_tunnel;
        std::fprintf(stderr,
                     "[bunny_water] PROBE %s splash=%d two_way=%d (lambda=%.5f rows=%u) "
                     "no_tunnel=%d\n",
                     ok ? "PASS" : "FAIL", splash, two_way, max_body_lambda,
                     particle_rows_seen, no_tunnel);
        return ok ? 0 : 5;
    }

    // Hero frames: settled pool, first contact, peak splash, settling.
    uint32_t first_contact = 0u;
    while (first_contact < kDropSteps && cminz_of[first_contact] > rest_surface) ++first_contact;
    std::vector<uint32_t> hero;
    hero.push_back(0u);                                              // settled pool.
    hero.push_back(std::min(kDropSteps - 1u, first_contact + 3u));   // first contact.
    hero.push_back(std::min(kDropSteps - 1u, peak_splash_step));     // peak splash.
    hero.push_back(kDropSteps - 1u);                                 // settling.
    std::sort(hero.begin(), hero.end());
    hero.erase(std::unique(hero.begin(), hero.end()), hero.end());
    while (hero.size() < 4u) hero.push_back(kDropSteps - 1u);
    if (hero.size() > 4u) hero.resize(4u);

    std::fprintf(stderr, "[bunny_water] hero frames: first_contact=%u peak=%u ->",
                 first_contact, peak_splash_step);
    for (uint32_t h : hero) std::fprintf(stderr, " %u(splash=%u)", h, splash_of[h]);
    std::fprintf(stderr, "\n");

    hero_step.assign(hero.begin(), hero.end());
    hero_splash.resize(kHeroCount); hero_pose.resize(kHeroCount); hero_fluid.resize(kHeroCount);
    for (uint32_t i = 0; i < kHeroCount; ++i) {
        hero_splash[i] = splash_of[hero[i]];
        hero_pose[i] = pose_snap[hero[i]];
        hero_fluid[i] = fluid_snap[hero[i]];
    }
    if (!args.cache.empty())
        SaveHeroCache(args.cache, kFluidCount, rest_surface, hero_step, hero_splash, hero_pose, hero_fluid);
    }  // end fresh-drop simulation

    std::filesystem::create_directories(args.png_dir);

    // Studio rig (from soft_ball): bright sun, grey sky gradient, light background.
    render::RasterOptions opts;
    opts.width = args.width; opts.height = args.height;
    opts.draw_ground = false;
    opts.hero_framing = false;
    opts.use_camera_override = true;
    opts.camera_up = {0.0f, 0.0f, 1.0f};
    opts.camera_fov_degrees = 40.0f;
    opts.background = {214, 226, 240, 255};
    opts.ground_color[0] = 0.16f; opts.ground_color[1] = 0.18f; opts.ground_color[2] = 0.22f;
    opts.contact_shadow_strength = 0.0f;
    opts.use_sun_light = true;
    opts.sun_direction[0] = 0.35f; opts.sun_direction[1] = -0.55f; opts.sun_direction[2] = 0.62f;
    opts.sun_color[0] = 3.0f; opts.sun_color[1] = 2.95f; opts.sun_color[2] = 2.8f;
    opts.sun_ambient_sky[0] = 0.18f; opts.sun_ambient_sky[1] = 0.20f; opts.sun_ambient_sky[2] = 0.24f;
    opts.sun_ambient_ground[0] = 0.10f; opts.sun_ambient_ground[1] = 0.11f; opts.sun_ambient_ground[2] = 0.12f;
    opts.shadow_strength = 0.92f;
    opts.shadow_map_size = 2560u;
    opts.shadow_bias = 0.0020f;
    opts.sky_gradient = true;
    // Bright blue sky-dome: the water surface mirrors it (Fresnel) so the top reads
    // as luminous water, not dark grey, and lights the refracted interior.
    opts.sky_top[0] = 0.40f; opts.sky_top[1] = 0.62f; opts.sky_top[2] = 0.95f;
    opts.sky_bottom[0] = 0.85f; opts.sky_bottom[1] = 0.92f; opts.sky_bottom[2] = 1.00f;
    render::RasterOptions::ContactPoint pool_cp;
    pool_cp.x = 0.0f; pool_cp.y = 0.0f; pool_cp.radius = kWallInner * 1.05f; pool_cp.strength = 0.40f;
    opts.contact_points.push_back(pool_cp);

    // Camera: 3/4 view from above-front, looking DOWN ~30deg, framing the whole pool.
    const Vec3 look{0.0f, 0.0f, rest_surface * 0.5f};
    const float cam_r = kWallInner * 3.2f;
    const float cam_az = -0.55f;
    const float cam_elev = 30.0f * kPi / 180.0f;
    opts.camera_eye = {look.x + cam_r * std::cos(cam_elev) * std::sin(cam_az),
                       look.y - cam_r * std::cos(cam_elev) * std::cos(cam_az),
                       look.z + cam_r * std::sin(cam_elev)};
    opts.camera_target = look;

    // VIDEO: render EVERY stored drop frame to a numbered PNG sequence (ffmpeg -> mp4).
    // Requires a fresh sim (fluid_snap full); --cache only holds the 4 hero frames.
    if (args.video && !fluid_snap.empty()) {
        GpuRenderer gpu(BuildFrame(args.mode, fluid_snap[0], kFluidCount,
                                   bunny, bunny_topo, pose_snap[0], rest_surface));
        if (!gpu.ok()) { std::fprintf(stderr, "[bunny_water] no CUDA RT backend\n"); return 6; }
        gpu.SetSamples(args.samples);
        uint32_t fi = 0u;
        for (uint32_t s = 0; s < fluid_snap.size(); s += args.video_stride) {
            render::RenderWorld rw = BuildFrame(args.mode, fluid_snap[s], kFluidCount,
                                                bunny, bunny_topo, pose_snap[s], rest_surface);
            render::VulkanOffscreenReport rep = gpu.Render(rw, opts);
            char name[96];
            std::snprintf(name, sizeof(name), "%s/v%04u.png", args.png_dir.c_str(), fi);
            if (!WritePng(rep, name)) { std::fprintf(stderr, "[bunny_water] PNG fail %s\n", name); }
            if (fi % 30u == 0u)
                std::fprintf(stderr, "[bunny_water] video frame %u (step %u)\n", fi, s);
            ++fi;
        }
        std::fprintf(stderr, "[bunny_water] VIDEO: %u frames -> %s\n", fi, args.png_dir.c_str());
        return 0;
    }

    render::RenderWorld rw0 =
        BuildFrame(args.mode, hero_fluid[0], static_cast<uint32_t>(hero_fluid[0].size()),
                   bunny, bunny_topo, hero_pose[0], rest_surface);
    GpuRenderer gpu(rw0);
    if (!gpu.ok()) { std::fprintf(stderr, "[bunny_water] no CUDA RT backend\n"); return 6; }
    gpu.SetSamples(args.samples);

    uint32_t written = 0u;
    for (uint32_t i = 0; i < kHeroCount; ++i) {
        render::RenderWorld rw =
            BuildFrame(args.mode, hero_fluid[i], static_cast<uint32_t>(hero_fluid[i].size()),
                       bunny, bunny_topo, hero_pose[i], rest_surface);
        render::VulkanOffscreenReport rep = gpu.Render(rw, opts);
        char name[96];
        std::snprintf(name, sizeof(name), "%s/frame%02u.png", args.png_dir.c_str(), written);
        if (WritePng(rep, name))
            std::fprintf(stderr, "[bunny_water] frame step=%u splash=%u -> %s (nonbg=%zu)\n",
                         hero_step[i], hero_splash[i], name, rep.non_background_pixel_count);
        else
            std::fprintf(stderr, "[bunny_water] PNG encode failed: %s\n", name);
        ++written;
    }

    std::fprintf(stderr, "[bunny_water] DONE: %u frames -> %s\n", written, args.png_dir.c_str());
    return written > 0u ? 0 : 7;
}
