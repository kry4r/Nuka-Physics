// ---------------------------------------------------------------------------
// MediaRecord -> cook-input equivalence (host cook assertions, no GPU).
//
// Proves the media-record path produces the SAME XpbdCookInput / PbfCookInput the
// flat coupled descriptor produced before the cook orchestrator existed. The
// REFERENCE here is a verbatim copy of the legacy c_abi cloth/fluid input builders
// (the logic that lived in c_abi/world.cpp); the SUBJECT is cook::BuildClothXpbdInput
// / cook::BuildFluidPbfInput fed a MediaRecord translated from the same flat slots.
// Every field is compared bit-for-bit, on the go2_cloth_drape demo cloth params and
// a representative fluid box. A divergence here is a translation / move bug.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "import/cooker/fluid_cooker.hpp"   // CookFluidBox (legacy fluid reference)
#include "math/vec3.hpp"
#include "runtime/soft/cloth_topology.hpp"  // BuildClothConstraints (legacy reference)
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"

namespace {

namespace cook = nuka::scene::cook;
using nuka::math::Vec3;
using MediaRecord = nuka::scene::MediaRecord;

// The compact coupled flat slots (a local mirror of nuka_coupled_particles_desc_t's
// cloth/fluid fields, so this cook-layer test carries no C-ABI dependency).
struct FlatDesc {
    uint32_t cloth_nx = 0u, cloth_ny = 0u;
    float cloth_spacing = 0.0f;
    float cloth_origin_x = 0.0f, cloth_origin_y = 0.0f, cloth_origin_z = 0.0f;
    float cloth_particle_mass = 0.0f, cloth_friction = 0.0f, cloth_bend_alpha = 0.0f;
    uint32_t cloth_iters = 0u, cloth_free = 0u;
    float aero_normal = 0.0f, aero_tangent = 0.0f, aero_max_dv = 0.0f;
    float fluid_min_x = 0.0f, fluid_min_y = 0.0f, fluid_min_z = 0.0f;
    float fluid_max_x = 0.0f, fluid_max_y = 0.0f, fluid_max_z = 0.0f;
    float fluid_spacing = 0.0f, fluid_rest_density = 0.0f;
    float fluid_floor_z = 0.0f, fluid_friction = 0.0f;
    uint32_t fluid_iters = 0u;
    uint32_t soft_sim_method = 0u;
    float contact_radius = 0.0f;
};

// ---- LEGACY references: verbatim copies of the pre-orchestrator c_abi builders ----

cook::XpbdCookInput LegacyClothCookInput(const FlatDesc& p) {
    cook::XpbdCookInput in;
    if (p.cloth_nx < 2u || p.cloth_ny < 2u || p.cloth_spacing <= 0.0f) return in;
    const uint32_t nx = p.cloth_nx, ny = p.cloth_ny;
    const float s = p.cloth_spacing;
    const float x0 = p.cloth_origin_x - 0.5f * static_cast<float>(nx - 1u) * s;
    const float y0 = p.cloth_origin_y - 0.5f * static_cast<float>(ny - 1u) * s;
    std::vector<Vec3> rest;
    rest.reserve(static_cast<size_t>(nx) * ny);
    for (uint32_t j = 0u; j < ny; ++j)
        for (uint32_t i = 0u; i < nx; ++i)
            rest.push_back(Vec3{x0 + static_cast<float>(i) * s,
                                y0 + static_cast<float>(j) * s, p.cloth_origin_z});
    auto idx = [nx](uint32_t i, uint32_t j) { return j * nx + i; };
    std::vector<nuka::runtime::soft::ClothTriangle> tris;
    for (uint32_t j = 0u; j + 1u < ny; ++j)
        for (uint32_t i = 0u; i + 1u < nx; ++i) {
            tris.push_back({{idx(i, j), idx(i + 1u, j), idx(i + 1u, j + 1u)}});
            tris.push_back({{idx(i, j), idx(i + 1u, j + 1u), idx(i, j + 1u)}});
        }
    nuka::runtime::soft::ClothTopologyOptions opts;
    opts.distance_compliance_alpha = 0.0f;
    opts.bend_compliance_alpha = p.cloth_bend_alpha;
    nuka::runtime::soft::XpbdConstraintSet cs;
    nuka::runtime::soft::BuildClothConstraints(rest, tris, opts, cs);
    in.positions = rest;
    in.velocities.assign(rest.size(), Vec3::Zero());
    const float mass = p.cloth_particle_mass > 0.0f ? p.cloth_particle_mass : 0.01f;
    in.inv_mass.assign(rest.size(), 1.0f / mass);
    if (p.cloth_free == 0u) {
        const uint32_t last_i = nx - 1u, last_j = ny - 1u;
        for (uint32_t k = 0u; k < nx; ++k) {
            in.inv_mass[idx(k, 0u)] = 0.0f;
            in.inv_mass[idx(k, last_j)] = 0.0f;
        }
        for (uint32_t k = 0u; k < ny; ++k) {
            in.inv_mass[idx(0u, k)] = 0.0f;
            in.inv_mass[idx(last_i, k)] = 0.0f;
        }
    }
    for (const auto& dc : cs.distance)
        in.distance.push_back(
            {dc.particle_a, dc.particle_b, dc.rest_length, dc.compliance_alpha});
    for (const auto& bc : cs.bend) {
        cook::CookBendCon c;
        for (uint32_t k = 0u; k < 4u; ++k) { c.p[k] = bc.particle[k]; c.k[k] = bc.k[k]; }
        c.compliance_alpha = bc.compliance_alpha;
        in.bend.push_back(c);
    }
    in.solver_iterations =
        static_cast<uint16_t>(p.cloth_iters != 0u ? p.cloth_iters : 1u);
    in.friction = p.cloth_friction;
    in.aero_drag_normal = p.aero_normal;
    in.aero_drag_tangent = p.aero_tangent;
    in.aero_drag_max_dv = p.aero_max_dv;
    if (p.aero_normal > 0.0f || p.aero_tangent > 0.0f) {
        in.aero_triangles.reserve(tris.size());
        for (const auto& t : tris) in.aero_triangles.push_back({t.v[0], t.v[1], t.v[2]});
    }
    return in;
}

cook::PbfCookInput LegacyFluidCookInput(const FlatDesc& p) {
    cook::PbfCookInput in;
    if (p.fluid_spacing <= 0.0f || p.fluid_max_x <= p.fluid_min_x ||
        p.fluid_max_y <= p.fluid_min_y || p.fluid_max_z <= p.fluid_min_z)
        return in;
    nuka::import::cooker::FluidBoxSpec spec;
    spec.min_corner = {p.fluid_min_x, p.fluid_min_y, p.fluid_min_z};
    spec.max_corner = {p.fluid_max_x, p.fluid_max_y, p.fluid_max_z};
    spec.spacing = p.fluid_spacing;
    spec.rest_density = p.fluid_rest_density > 0.0f ? p.fluid_rest_density : 1000.0f;
    const nuka::runtime::fluid::PbfParticleSet box =
        nuka::import::cooker::CookFluidBox(spec);
    in.positions = box.positions;
    in.velocities.assign(box.positions.size(), Vec3::Zero());
    in.particle_mass = box.particle_mass;
    in.rest_density = spec.rest_density;
    const float h = 1.5f * p.fluid_spacing;
    in.support_radius = h;
    in.cfm_epsilon = 1.0e-6f;
    in.iters = static_cast<uint16_t>(p.fluid_iters != 0u ? p.fluid_iters : 4u);
    in.clamp_overdensity = true;
    in.boundary_enabled = true;
    in.floor_z = p.fluid_floor_z;
    in.friction = p.fluid_friction;
    const Vec3 lo = spec.min_corner, hi = spec.max_corner;
    in.grid_min = Vec3{lo.x - 3.0f * h, lo.y - 3.0f * h,
                       std::min(lo.z, p.fluid_floor_z) - h};
    auto cells = [h](float extent) {
        return static_cast<uint32_t>(std::ceil(extent / h)) + 1u;
    };
    in.grid_dims[0] = cells((hi.x - lo.x) + 6.0f * h);
    in.grid_dims[1] = cells((hi.y - lo.y) + 6.0f * h);
    in.grid_dims[2] = cells((hi.z - in.grid_min.z) + 4.0f * h);
    return in;
}

// ---- the flat slots -> MediaRecord translation (mirrors c_abi/world.cpp) -----------

MediaRecord ClothMedia(const FlatDesc& p) {
    MediaRecord m;
    m.kind = MediaRecord::Kind::Cloth;
    m.method = (p.soft_sim_method == 1u) ? MediaRecord::Method::MlsMpm
                                         : MediaRecord::Method::Xpbd;
    m.cloth_grid.nx = p.cloth_nx;
    m.cloth_grid.ny = p.cloth_ny;
    m.cloth_grid.spacing = p.cloth_spacing;
    m.cloth_grid.origin = {p.cloth_origin_x, p.cloth_origin_y, p.cloth_origin_z};
    m.cloth_grid.free = (p.cloth_free != 0u);
    m.xpbd.particle_mass = p.cloth_particle_mass;
    m.xpbd.friction = p.cloth_friction;
    m.xpbd.distance_alpha = 0.0f;
    m.xpbd.bend_alpha = p.cloth_bend_alpha;
    m.xpbd.iters = static_cast<uint16_t>(p.cloth_iters);
    m.xpbd.aero_drag_normal = p.aero_normal;
    m.xpbd.aero_drag_tangent = p.aero_tangent;
    m.xpbd.aero_drag_max_dv = p.aero_max_dv;
    return m;
}

MediaRecord FluidMedia(const FlatDesc& p) {
    MediaRecord m;
    m.kind = MediaRecord::Kind::Fluid;
    m.method = (p.soft_sim_method == 1u) ? MediaRecord::Method::MlsMpm
                                         : MediaRecord::Method::Pbf;
    m.fluid_box.min = {p.fluid_min_x, p.fluid_min_y, p.fluid_min_z};
    m.fluid_box.max = {p.fluid_max_x, p.fluid_max_y, p.fluid_max_z};
    m.fluid_box.spacing = p.fluid_spacing;
    m.pbf.rest_density = p.fluid_rest_density;
    m.pbf.support_scale = 1.5f;
    m.pbf.iters = static_cast<uint16_t>(p.fluid_iters);
    m.pbf.friction = p.fluid_friction;
    m.pbf.floor_z = p.fluid_floor_z;
    m.pbf.clamp_overdensity = true;
    return m;
}

// ---- field-by-field comparators (bit-exact for floats) -----------------------------

void ExpectVec3Eq(const std::vector<Vec3>& a, const std::vector<Vec3>& b,
                  const char* what) {
    ASSERT_EQ(a.size(), b.size()) << what << " size";
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].x, b[i].x) << what << "[" << i << "].x";
        EXPECT_EQ(a[i].y, b[i].y) << what << "[" << i << "].y";
        EXPECT_EQ(a[i].z, b[i].z) << what << "[" << i << "].z";
    }
}

void ExpectXpbdEqual(const cook::XpbdCookInput& a, const cook::XpbdCookInput& b) {
    ExpectVec3Eq(a.positions, b.positions, "positions");
    ExpectVec3Eq(a.velocities, b.velocities, "velocities");
    ASSERT_EQ(a.inv_mass.size(), b.inv_mass.size()) << "inv_mass size";
    for (size_t i = 0; i < a.inv_mass.size(); ++i)
        EXPECT_EQ(a.inv_mass[i], b.inv_mass[i]) << "inv_mass[" << i << "]";
    ASSERT_EQ(a.distance.size(), b.distance.size()) << "distance size";
    for (size_t i = 0; i < a.distance.size(); ++i) {
        EXPECT_EQ(a.distance[i].a, b.distance[i].a);
        EXPECT_EQ(a.distance[i].b, b.distance[i].b);
        EXPECT_EQ(a.distance[i].rest_length, b.distance[i].rest_length);
        EXPECT_EQ(a.distance[i].compliance_alpha, b.distance[i].compliance_alpha);
    }
    ASSERT_EQ(a.bend.size(), b.bend.size()) << "bend size";
    for (size_t i = 0; i < a.bend.size(); ++i) {
        for (uint32_t k = 0; k < 4u; ++k) {
            EXPECT_EQ(a.bend[i].p[k], b.bend[i].p[k]);
            EXPECT_EQ(a.bend[i].k[k].x, b.bend[i].k[k].x);
            EXPECT_EQ(a.bend[i].k[k].y, b.bend[i].k[k].y);
            EXPECT_EQ(a.bend[i].k[k].z, b.bend[i].k[k].z);
        }
        EXPECT_EQ(a.bend[i].compliance_alpha, b.bend[i].compliance_alpha);
    }
    EXPECT_EQ(a.volume.size(), b.volume.size());
    EXPECT_EQ(a.shape_match.size(), b.shape_match.size());
    ASSERT_EQ(a.aero_triangles.size(), b.aero_triangles.size()) << "aero size";
    for (size_t i = 0; i < a.aero_triangles.size(); ++i)
        for (uint32_t k = 0; k < 3u; ++k)
            EXPECT_EQ(a.aero_triangles[i][k], b.aero_triangles[i][k]);
    EXPECT_EQ(a.aero_drag_normal, b.aero_drag_normal);
    EXPECT_EQ(a.aero_drag_tangent, b.aero_drag_tangent);
    EXPECT_EQ(a.aero_drag_max_dv, b.aero_drag_max_dv);
    EXPECT_EQ(a.solver_iterations, b.solver_iterations);
    EXPECT_EQ(a.friction, b.friction);
    EXPECT_EQ(a.solver, b.solver);
}

void ExpectPbfEqual(const cook::PbfCookInput& a, const cook::PbfCookInput& b) {
    ExpectVec3Eq(a.positions, b.positions, "fluid positions");
    ExpectVec3Eq(a.velocities, b.velocities, "fluid velocities");
    ASSERT_EQ(a.inv_mass.size(), b.inv_mass.size());
    for (size_t i = 0; i < a.inv_mass.size(); ++i) EXPECT_EQ(a.inv_mass[i], b.inv_mass[i]);
    EXPECT_EQ(a.particle_mass, b.particle_mass);
    EXPECT_EQ(a.rest_density, b.rest_density);
    EXPECT_EQ(a.support_radius, b.support_radius);
    EXPECT_EQ(a.cfm_epsilon, b.cfm_epsilon);
    EXPECT_EQ(a.iters, b.iters);
    EXPECT_EQ(a.clamp_overdensity, b.clamp_overdensity);
    EXPECT_EQ(a.xsph_viscosity, b.xsph_viscosity);
    EXPECT_EQ(a.surface_tension, b.surface_tension);
    EXPECT_EQ(a.grid_min.x, b.grid_min.x);
    EXPECT_EQ(a.grid_min.y, b.grid_min.y);
    EXPECT_EQ(a.grid_min.z, b.grid_min.z);
    for (uint32_t k = 0; k < 3u; ++k) EXPECT_EQ(a.grid_dims[k], b.grid_dims[k]);
    EXPECT_EQ(a.boundary_enabled, b.boundary_enabled);
    EXPECT_EQ(a.floor_z, b.floor_z);
    EXPECT_EQ(a.friction, b.friction);
}

// The go2_cloth_drape demo cloth params (the byte-identity reference scene).
FlatDesc DemoClothDesc() {
    FlatDesc p{};
    p.cloth_nx = 55u; p.cloth_ny = 51u; p.cloth_spacing = 0.024f;
    p.cloth_origin_x = 0.0f; p.cloth_origin_y = 0.0f; p.cloth_origin_z = 0.90f;
    p.cloth_particle_mass = 0.012f; p.cloth_friction = 1.8f;
    p.cloth_bend_alpha = 0.09f; p.cloth_iters = 80u;
    p.cloth_free = 1u;
    p.aero_normal = 30.0f; p.aero_tangent = 0.12f; p.aero_max_dv = 0.16f;
    p.contact_radius = 0.022f;
    return p;
}

}  // namespace

// Cloth: BuildClothXpbdInput(ClothMedia(demo)) == the legacy cloth builder, free drape.
TEST(MediaRecordCookEquivalence, ClothMatchesLegacyDrape) {
    const FlatDesc p = DemoClothDesc();
    ExpectXpbdEqual(cook::BuildClothXpbdInput(ClothMedia(p)), LegacyClothCookInput(p));
}

// Cloth: the perimeter-pinned membrane (cloth_free == 0) matches too.
TEST(MediaRecordCookEquivalence, ClothMatchesLegacyPinned) {
    FlatDesc p = DemoClothDesc();
    p.cloth_free = 0u;
    ExpectXpbdEqual(cook::BuildClothXpbdInput(ClothMedia(p)), LegacyClothCookInput(p));
}

// The render-surface triangle list matches the legacy winding (record vs flat).
TEST(MediaRecordCookEquivalence, ClothSurfaceTrianglesMatch) {
    const FlatDesc p = DemoClothDesc();
    const uint32_t nx = p.cloth_nx, ny = p.cloth_ny;
    std::vector<uint32_t> legacy;
    auto idx = [nx](uint32_t i, uint32_t j) { return j * nx + i; };
    for (uint32_t j = 0u; j + 1u < ny; ++j)
        for (uint32_t i = 0u; i + 1u < nx; ++i)
            legacy.insert(legacy.end(),
                          {idx(i, j), idx(i + 1u, j), idx(i + 1u, j + 1u),
                           idx(i, j), idx(i + 1u, j + 1u), idx(i, j + 1u)});
    EXPECT_EQ(cook::BuildClothSurfaceTriangles(ClothMedia(p)), legacy);
}

// Fluid: BuildFluidPbfInput(FluidMedia(box)) == the legacy fluid builder.
TEST(MediaRecordCookEquivalence, FluidMatchesLegacy) {
    FlatDesc p{};
    p.fluid_min_x = -0.39f; p.fluid_min_y = -0.06f; p.fluid_min_z = 0.10f;
    p.fluid_max_x = -0.26f; p.fluid_max_y = 0.09f; p.fluid_max_z = 0.30f;
    p.fluid_spacing = 0.022f; p.fluid_rest_density = 1000.0f;
    p.fluid_floor_z = 0.12f; p.fluid_friction = 0.05f; p.fluid_iters = 4u;
    ExpectPbfEqual(cook::BuildFluidPbfInput(FluidMedia(p)), LegacyFluidCookInput(p));
}
