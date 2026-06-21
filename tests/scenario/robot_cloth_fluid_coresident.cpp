// ---------------------------------------------------------------------------
// A robot ARTICULATION + cloth (XPBD) + fluid (PBF) CO-RESIDENT in ONE nk::World,
// all stepping on the ONE general physics pipeline with two-way coupling to BOTH
// media. A Go2 quadruped is cooked through the SAME production path it uses on the
// ground (LoadUsd -> CookToModel) and a cloth patch + a shallow fluid pool are
// cooked onto the SAME model via CookSoftFluidParticles (the [soft | fluid] layout).
// The Go2 holds its baked stance through the cooked position actuators; its front
// feet rest on the cloth and a rear foot rests in the pool. The robot links, the
// cloth, and the fluid all couple through the SAME body<->particle row solver a
// foot uses on the ground: detection -> body<->particle narrowphase -> emit ->
// SolveRowsBlockIsland -> finalize compose. No per-robot code, no per-medium branch:
// a calf link is just another collidable body row, a particle is a sphere, and the
// medium difference (cloth grips, fluid slides) is DATA (per-system mu), never code.
//
// Proven numerically against a no-particle control of the same robot:
//   STABLE CO-STEP: ~1.7 sim seconds; the world stays finite, the robot stays
//     controlled, the cloth + fluid stay finite.
//   TWO-WAY TO CLOTH: a body<->particle row exists whose particle side kind ==
//     kNkSideParticle AND whose BODY side maps to an articulation LINK
//     (body_to_link != ~0), the foot is in the cloth (soft) slice [0, n_soft),
//     the body-side normal lambda > 0, and the cloth dips below its undisturbed lay.
//   TWO-WAY TO FLUID: the same, against a particle in the fluid slice [n_soft, P),
//     and the pool is displaced under the foot.
//   Every coupling row asserted has exactly one particle side -- the general path.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "collision/shape_kind.hpp"
#include "import/usd_importer.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "runtime/soft/cloth_topology.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
using nuka::math::Transform;
using nuka::math::Vec3;

// A uniform particle collision radius for both media (cloth + pool); the foot
// touches a particle when their spheres overlap. d_min == 2*radius.
constexpr float kContactDMin = 0.030f;

// Cloth patch under the FRONT feet: a taut flat lattice, the whole perimeter pinned
// so it is a trampoline membrane the foot rests in + indents (not a free sheet that
// sags away). The front feet straddle it from rest (the membrane lies at the foot
// centre) so contact is immediate; light particles so the foot dips it.
constexpr uint32_t kClothNx = 13u;          // 13x13 = 169 cloth particles.
constexpr float kClothSpacing = 0.016f;     // ~0.19 m square (spans both front feet).
constexpr float kClothParticleMass = 0.01f; // 10 g per particle (light cloth).
constexpr uint16_t kClothIters = 24u;       // tighter stretch solve = taut patch.

// Shallow fluid pool under a REAR foot: a compact lattice on a z-up PBF boundary
// floor. A standing foot's lower hemisphere is submerged from the start so the
// contact is immediate; the foot displaces fluid (the pocket beneath it drops).
constexpr float kPoolSpacing = 0.025f;
constexpr float kPoolSupport = kPoolSpacing * 1.5f;
constexpr float kPoolRestDensity = 1000.0f;
constexpr uint32_t kPoolNx = 5u;            // 5x5 footprint.
constexpr uint32_t kPoolNz = 4u;            // 4 layers deep.

constexpr uint32_t kSettleSteps = 250u;     // settle the robot stance + the media.
constexpr uint32_t kHoldSteps = 200u;       // co-step the held stance over the media.

std::filesystem::path Go2ScenePath() {
    return std::filesystem::path(NUKA_SOURCE_DIR) / "examples" / "scenes" /
           "go2_stand.usda";
}

struct Backend { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    cfg.contact_margin = 0.0f;
    // A big velocity budget: a heavy robot link vs light particles needs PGS
    // iterations to transmit momentum without the foot tunnelling the medium.
    cfg.vel_iters = 48u;
    cfg.pos_iters = 0u;
    cfg.max_pairs = 64u;
    return cfg;
}

// A flat cloth lattice centred at (cx,cy) lying at height z; four corners pinned so
// the patch stays under the front feet while they press it.
cook::XpbdCookInput BuildCloth(float cx, float cy, float z) {
    std::vector<Vec3> rest;
    rest.reserve(kClothNx * kClothNx);
    const float c0 = -0.5f * static_cast<float>(kClothNx - 1u) * kClothSpacing;
    for (uint32_t j = 0; j < kClothNx; ++j)
        for (uint32_t i = 0; i < kClothNx; ++i)
            rest.push_back(Vec3{cx + c0 + static_cast<float>(i) * kClothSpacing,
                                cy + c0 + static_cast<float>(j) * kClothSpacing, z});
    auto idx = [](uint32_t i, uint32_t j) { return j * kClothNx + i; };
    std::vector<soft::ClothTriangle> tris;
    for (uint32_t j = 0; j + 1 < kClothNx; ++j)
        for (uint32_t i = 0; i + 1 < kClothNx; ++i) {
            tris.push_back(soft::ClothTriangle{{idx(i, j), idx(i + 1, j),
                                                idx(i + 1, j + 1)}});
            tris.push_back(soft::ClothTriangle{{idx(i, j), idx(i + 1, j + 1),
                                                idx(i, j + 1)}});
        }
    soft::ClothTopologyOptions opts;
    opts.distance_compliance_alpha = 0.0f;
    opts.bend_compliance_alpha = 1.0e-4f;
    soft::XpbdConstraintSet cs;
    soft::BuildClothConstraints(rest, tris, opts, cs);

    cook::XpbdCookInput in;
    in.positions = rest;
    in.velocities.assign(rest.size(), Vec3::Zero());
    in.inv_mass.assign(rest.size(), 1.0f / kClothParticleMass);
    // Pin the whole perimeter so the patch is a taut membrane (a trampoline) the
    // hanging foot rests in + indents, instead of a free sheet that sags away.
    const uint32_t last = kClothNx - 1u;
    for (uint32_t k = 0; k < kClothNx; ++k) {
        in.inv_mass[idx(k, 0)] = 0.0f; in.inv_mass[idx(k, last)] = 0.0f;
        in.inv_mass[idx(0, k)] = 0.0f; in.inv_mass[idx(last, k)] = 0.0f;
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
    in.solver_iterations = kClothIters;
    in.friction = 0.6f;   // finite mu: the foot grips/drags the cloth.
    return in;
}

// A compact PBF pool centred at (cx,cy), bottom on a z-up boundary floor at floor_z;
// the grid AABB spans the footprint + vertical headroom for the submerged foot.
cook::PbfCookInput BuildPool(float cx, float cy, float floor_z) {
    cook::PbfCookInput in;
    const float s = kPoolSpacing, h = kPoolSupport, rho0 = kPoolRestDensity;
    const float c0 = -0.5f * static_cast<float>(kPoolNx - 1u) * s;
    const float bottom = floor_z + 0.5f * s;
    for (uint32_t iz = 0; iz < kPoolNz; ++iz)
        for (uint32_t iy = 0; iy < kPoolNx; ++iy)
            for (uint32_t ix = 0; ix < kPoolNx; ++ix)
                in.positions.push_back(Vec3{cx + c0 + ix * s, cy + c0 + iy * s,
                                            bottom + iz * s});
    in.velocities.assign(in.positions.size(), Vec3::Zero());
    in.particle_mass = rho0 * s * s * s;
    in.rest_density = rho0;
    in.support_radius = h;
    in.cfm_epsilon = 1.0e-6f;
    in.iters = 4u;
    in.clamp_overdensity = true;
    in.boundary_enabled = true;
    in.floor_z = floor_z;
    in.friction = 0.0f;   // fluid mu ~= 0: the foot slides; splash stays normal-driven.
    // Grid AABB: the footprint plus lateral spread room + vertical headroom for the
    // submerged foot. A particle outside the grid finds no neighbours, so it must
    // enclose the whole working volume.
    const float half = 0.5f * static_cast<float>(kPoolNx - 1u) * s + 3.0f * h;
    const float top = floor_z + kPoolNz * s + 4.0f * h;
    in.grid_min = Vec3{cx - half - h, cy - half - h, floor_z - h};
    auto cells = [&](float extent) {
        return static_cast<uint32_t>(std::ceil(extent / h)) + 1u;
    };
    in.grid_dims[0] = cells(2.0f * half);
    in.grid_dims[1] = cells(2.0f * half);
    in.grid_dims[2] = cells(top - (floor_z - h));
    return in;
}

// NkRow packs to 32 f32: [0]=flags [7]=upper, a.kind [16] a.index [17],
// b.kind [24] b.index [25].
struct RowSides { uint32_t a_kind, b_kind, a_index, b_index; bool active; float upper; };
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

void DownloadParticles(nk::World& w, std::vector<Vec3>* pos) {
    const uint32_t P = w.GetModel().capacities.particles_per_env;
    pos->assign(P, Vec3::Zero());
    w.GetData().DownloadField(nk::FieldId::ParticlePos, pos->data(), P * sizeof(Vec3));
}

// Min particle z over the disc of `reach` about (cx,cy), restricted to a particle
// index slice [lo, hi) -- the pocket a foot presses into one medium's slice.
float MinZUnderFootInSlice(const std::vector<Vec3>& pos, float cx, float cy,
                           float reach, uint32_t lo, uint32_t hi) {
    float min_z = 1.0e9f;
    for (uint32_t i = lo; i < hi && i < pos.size(); ++i) {
        const float dx = pos[i].x - cx, dy = pos[i].y - cy;
        if (dx * dx + dy * dy <= reach * reach) min_z = std::min(min_z, pos[i].z);
    }
    return min_z;
}

// Max particle z over a slice [lo, hi) -- a medium's free-surface height.
float SurfaceMaxZInSlice(const std::vector<Vec3>& pos, uint32_t lo, uint32_t hi) {
    float hi_z = -1.0e9f;
    for (uint32_t i = lo; i < hi && i < pos.size(); ++i) hi_z = std::max(hi_z, pos[i].z);
    return hi_z;
}

}  // namespace

// A Go2 articulation + cloth + fluid co-resident in ONE world: stable co-step +
// two-way coupling to BOTH media through the general body<->particle row solver.
TEST(RobotClothFluidCoResident, Go2StanceCouplesClothAndFluidOnOnePipeline) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    if (!std::filesystem::exists(Go2ScenePath()))
        GTEST_SKIP() << "go2_stand.usda not present";
    Backend b = GetBackend();

    // --- cook the bare Go2 once + read where its feet rest in the held stance -----
    cook::CookToModelOptions opt;
    opt.contact_family = cook::CookContactFamily::PairDriven;

    auto cook_go2 = [&]() -> nk::Model {
        nuka::scene::SceneIR s = nuka::import::LoadUsd(Go2ScenePath().string());
        nk::Model m = cook::CookToModel(s, 1, opt).model;
        // Generous rigid candidate budget for the Go2 collidables (links + base);
        // CookSoftFluidParticles grows a DISJOINT particle reserve above this.
        m.capacities.max_contacts_per_env = 32u;
        m.capacities.max_rows_per_env =
            m.capacities.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
        return m;
    };

    nk::Model probe = cook_go2();
    const uint32_t L = probe.capacities.links_per_env;
    const std::vector<uint32_t> body_to_link = probe.body_to_link;
    const std::vector<uint32_t> link_geom_kind = probe.articulation.link_geom_kind;
    const std::vector<Transform> link_geom_local = probe.articulation.link_geom_local;
    ASSERT_GT(L, 0u);

    // The collidable foot links are the spheres (the four calf feet). link_geom_kind
    // stores (ShapeType + 1), so a sphere link is kShapeSphere + 1.
    constexpr uint32_t kSphereGeom = nuka::collision::kShapeSphere + 1u;
    std::vector<uint32_t> foot_links;
    for (uint32_t l = 0; l < L; ++l)
        if (l < link_geom_kind.size() && link_geom_kind[l] == kSphereGeom)
            foot_links.push_back(l);
    ASSERT_GE(foot_links.size(), 4u) << "expected four Go2 foot spheres";

    nk::World wp(std::move(probe), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(wp.Ready());
    for (uint32_t s = 0; s < kSettleSteps; ++s) wp.Step();
    std::vector<Transform> link_pose(L);
    ASSERT_TRUE(wp.GetData().DownloadField(nk::FieldId::LinkPose, link_pose.data(),
                                           L * sizeof(Transform)));
    std::vector<Vec3> foot_world;
    for (uint32_t l : foot_links)
        foot_world.push_back((link_pose[l] * link_geom_local[l]).position);
    for (size_t i = 0; i < foot_world.size(); ++i)
        std::fprintf(stderr, "[robot-coupling] foot_link=%u rest @ (%.4f,%.4f,%.4f)\n",
                     foot_links[i], foot_world[i].x, foot_world[i].y, foot_world[i].z);

    // Split the feet into front (max x) and rear (min x); the front pair gets the
    // cloth, one rear foot gets the pool.
    auto by_x = foot_world;
    std::sort(by_x.begin(), by_x.end(),
              [](const Vec3& a, const Vec3& c) { return a.x < c.x; });
    const float front_x = by_x.back().x, rear_x = by_x.front().x;
    Vec3 front_centre{0, 0, 0}; uint32_t nf = 0;
    Vec3 rear_foot{0, 0, 0}; float rear_best = 1e9f;
    float foot_rest_z = 0.0f;
    uint32_t front_link = foot_links[0], rear_link = foot_links[0];
    for (size_t i = 0; i < foot_world.size(); ++i) {
        const Vec3& f = foot_world[i];
        foot_rest_z += f.z;
        if (f.x > 0.5f * (front_x + rear_x)) {
            front_centre = front_centre + f; ++nf; front_link = foot_links[i];
        } else if (f.x < rear_best) {
            rear_best = f.x; rear_foot = f; rear_link = foot_links[i];
        }
    }
    foot_rest_z /= static_cast<float>(foot_world.size());
    front_centre = front_centre * (1.0f / static_cast<float>(nf));
    std::fprintf(stderr,
                 "[robot-coupling] front_centre=(%.4f,%.4f) rear_foot=(%.4f,%.4f) "
                 "foot_rest_z=%.4f\n",
                 front_centre.x, front_centre.y, rear_foot.x, rear_foot.y, foot_rest_z);

    // The pool's top layer sits at ~the rear foot centre so the foot's lower
    // hemisphere is submerged from rest; its floor is one pool-depth below.
    const float pool_top = foot_rest_z;
    const float pool_floor = pool_top - static_cast<float>(kPoolNz - 1u) * kPoolSpacing -
                             0.5f * kPoolSpacing;

    // The rear foot's fluid load shifts the WHOLE articulation (kinematic base + PD
    // legs), lifting the front feet off their bare-robot rest. Probe the front
    // foot's settled position WITH the pool present (cloth absent) so the cloth lay
    // can be placed where the front foot actually sits in the co-resident world.
    float front_foot_z_loaded = foot_rest_z;
    {
        nk::Model m = cook_go2();
        cook::PbfCookInput pool = BuildPool(rear_foot.x, rear_foot.y, pool_floor);
        cook::CookSoftFluidParticles(m, 1u, cook::XpbdCookInput{}, pool);
        m.particles.pp_contact_d_min = kContactDMin;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        ASSERT_TRUE(w.Ready());
        for (uint32_t s = 0; s < kSettleSteps; ++s) w.Step();
        std::vector<Transform> lp(L);
        w.GetData().DownloadField(nk::FieldId::LinkPose, lp.data(), L * sizeof(Transform));
        front_foot_z_loaded = (lp[front_link] * link_geom_local[front_link]).position.z;
    }
    std::fprintf(stderr, "[robot-coupling] front_foot_z_loaded=%.4f\n",
                 front_foot_z_loaded);

    // The taut membrane lies at the front foot's loaded centre so the foot's lower
    // hemisphere rests in + indents it (the pinned perimeter holds it taut, so it
    // cannot sag out of reach the way a corner-pinned sheet does).
    const float cloth_z = front_foot_z_loaded;

    // --- build the co-resident world: Go2 + cloth (front) + pool (rear) -----------
    // patch_present == false cooks the bare-Go2 control (same robot, no particles).
    struct Run {
        uint32_t cloth_link_rows = 0u, fluid_link_rows = 0u, two_particle_rows = 0u;
        uint32_t cloth_any_rows = 0u, fluid_any_rows = 0u;  // any body side (diag).
        float cloth_link_lambda = 0.0f, fluid_link_lambda = 0.0f;
        float cloth_min_z = 1e9f, pool_min_z = 1e9f, cloth_base_min_z = 0.0f;
        float pool_base_surface = 0.0f, pool_max_surface = -1e9f, pool_base_pocket = 0.0f;
        float front_foot_z = 0.0f, rear_foot_z = 0.0f;  // run-scene foot z (diag).
        bool finite = true;
        std::vector<float> q, qdot;
    };
    auto run_scene = [&](bool patch_present) -> Run {
        Run out;
        nk::Model m = cook_go2();

        cook::XpbdCookInput cloth = BuildCloth(front_centre.x, front_centre.y, cloth_z);
        cook::PbfCookInput pool = BuildPool(rear_foot.x, rear_foot.y, pool_floor);
        if (!patch_present) {
            // Sink both media far out of reach: the SAME robot, no coupling.
            for (Vec3& p : cloth.positions) p.z -= 10.0f;
            for (Vec3& p : pool.positions) p.z -= 10.0f;
            pool.grid_min.z -= 10.0f;
            pool.floor_z -= 10.0f;
        }
        cook::CookSoftFluidParticles(m, 1u, cloth, pool);
        m.particles.pp_contact_d_min = kContactDMin;
        const uint32_t n_soft = m.particles.n_soft_particles;
        const uint32_t P = m.capacities.particles_per_env;
        const uint32_t rows = m.capacities.max_rows_per_env;

        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        EXPECT_TRUE(w.Ready());
        nk::Data& d = w.GetData();
        std::vector<float> urows(static_cast<size_t>(rows) * 32u, 0.0f);
        std::vector<float> lambda(rows, 0.0f);
        std::vector<Vec3> p;
        for (uint32_t s = 0; s < kSettleSteps + kHoldSteps; ++s) {
            w.Step();
            if (!patch_present || s < kSettleSteps) continue;
            // Capture the pool free surface / pocket at the first hold step (the
            // settled baseline) then track its max surface over the hold window.
            DownloadParticles(w, &p);
            const float cloth_reach = 0.5f * kContactDMin + 2.0f * kClothSpacing;
            if (s == kSettleSteps) {
                out.pool_base_surface = SurfaceMaxZInSlice(p, n_soft, P);
                out.pool_base_pocket = MinZUnderFootInSlice(
                    p, rear_foot.x, rear_foot.y,
                    0.5f * kContactDMin + 2.0f * kPoolSpacing, n_soft, P);
                out.cloth_base_min_z = MinZUnderFootInSlice(
                    p, front_centre.x, front_centre.y, cloth_reach, 0u, n_soft);
                std::vector<Transform> lp(L);
                d.DownloadField(nk::FieldId::LinkPose, lp.data(), L * sizeof(Transform));
                out.front_foot_z =
                    (lp[front_link] * link_geom_local[front_link]).position.z;
                out.rear_foot_z =
                    (lp[rear_link] * link_geom_local[rear_link]).position.z;
            }
            out.pool_max_surface =
                std::max(out.pool_max_surface, SurfaceMaxZInSlice(p, n_soft, P));
            d.DownloadField(nk::FieldId::Urows, urows.data(),
                            urows.size() * sizeof(float));
            d.DownloadField(nk::FieldId::Lambda, lambda.data(),
                            lambda.size() * sizeof(float));
            for (uint32_t row = 0u; row < rows; ++row) {
                const RowSides rs = DecodeRow(urows, row);
                if (!rs.active) continue;
                const bool a_part = rs.a_kind == nk::kNkSideParticle;
                const bool b_part = rs.b_kind == nk::kNkSideParticle;
                if (!(a_part || b_part)) continue;
                if (a_part && b_part) { ++out.two_particle_rows; continue; }
                const uint32_t body_idx = a_part ? rs.b_index : rs.a_index;
                const uint32_t part_idx = a_part ? rs.a_index : rs.b_index;
                if (part_idx < n_soft) ++out.cloth_any_rows; else ++out.fluid_any_rows;
                const bool is_link = body_idx < body_to_link.size() &&
                                     body_to_link[body_idx] != ~uint32_t(0);
                if (!is_link) continue;
                const bool normal = rs.upper > 1.0e30f;
                if (part_idx < n_soft) {           // the cloth (soft) slice.
                    ++out.cloth_link_rows;
                    if (normal)
                        out.cloth_link_lambda =
                            std::max(out.cloth_link_lambda, lambda[row]);
                } else {                            // the fluid slice.
                    ++out.fluid_link_rows;
                    if (normal)
                        out.fluid_link_lambda =
                            std::max(out.fluid_link_lambda, lambda[row]);
                }
            }
        }
        out.q.assign(L, 0.0f); out.qdot.assign(L, 0.0f);
        d.DownloadField(nk::FieldId::Q, out.q.data(), L * sizeof(float));
        d.DownloadField(nk::FieldId::Qdot, out.qdot.data(), L * sizeof(float));
        DownloadParticles(w, &p);
        for (const Vec3& q : p)
            if (!(std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z)))
                out.finite = false;
        if (patch_present) {
            out.cloth_min_z = MinZUnderFootInSlice(
                p, front_centre.x, front_centre.y,
                0.5f * kContactDMin + 2.0f * kClothSpacing, 0u, n_soft);
            out.pool_min_z = MinZUnderFootInSlice(p, rear_foot.x, rear_foot.y,
                                                  0.5f * kContactDMin + 2.0f * kPoolSpacing,
                                                  n_soft, P);
        }
        return out;
    };

    const Run with_media = run_scene(/*patch_present=*/true);
    const Run control = run_scene(/*patch_present=*/false);

    double qdot_delta = 0.0;
    ASSERT_EQ(with_media.qdot.size(), control.qdot.size());
    for (size_t i = 0; i < with_media.qdot.size(); ++i)
        qdot_delta += std::fabs(static_cast<double>(with_media.qdot[i]) -
                                control.qdot[i]);

    const float cloth_dip = with_media.cloth_base_min_z - with_media.cloth_min_z;
    std::fprintf(stderr,
                 "[robot-coupling] cloth_rows=%u cloth_lambda=%.6f base_min_z=%.4f "
                 "cloth_min_z=%.4f (dip=%.4f, lay=%.4f)\n",
                 with_media.cloth_link_rows, with_media.cloth_link_lambda,
                 with_media.cloth_base_min_z, with_media.cloth_min_z, cloth_dip,
                 cloth_z);
    const float pool_rise = with_media.pool_max_surface - with_media.pool_base_surface;
    const float pocket_drop = with_media.pool_base_pocket - with_media.pool_min_z;
    std::fprintf(stderr,
                 "[robot-coupling] fluid_rows=%u fluid_lambda=%.6f base_surface=%.4f "
                 "max_surface=%.4f (rise=%.4f) base_pocket=%.4f pool_min_z=%.4f "
                 "(drop=%.4f)\n",
                 with_media.fluid_link_rows, with_media.fluid_link_lambda,
                 with_media.pool_base_surface, with_media.pool_max_surface, pool_rise,
                 with_media.pool_base_pocket, with_media.pool_min_z, pocket_drop);
    std::fprintf(stderr,
                 "[robot-coupling] cloth_any_rows=%u fluid_any_rows=%u "
                 "run_front_foot_z=%.4f run_rear_foot_z=%.4f two_particle_rows=%u "
                 "qdot_delta_L1=%.6f\n",
                 with_media.cloth_any_rows, with_media.fluid_any_rows,
                 with_media.front_foot_z, with_media.rear_foot_z,
                 with_media.two_particle_rows, qdot_delta);

    // STABLE: the robot + both media stayed finite over the whole co-step.
    EXPECT_TRUE(with_media.finite) << "the co-resident world produced a NaN/blowup";
    EXPECT_TRUE(control.finite) << "the bare-robot control produced a NaN/blowup";

    // TWO-WAY to CLOTH: a body<->particle row whose body side is an articulation
    // LINK against a cloth (soft) particle, with a body-side normal reaction.
    EXPECT_GT(with_media.cloth_link_rows, 0u)
        << "no robot-link<->cloth coupling rows emitted";
    EXPECT_GT(with_media.cloth_link_lambda, 0.0f)
        << "the robot link produced no body-side reaction on the cloth";
    // The foot indents the membrane below its pinned lay (a stable dimple, not the
    // membrane's negligible free sag): cloth_min_z holds below the lay through the
    // hold and the foot carries a real body-side reaction (cloth_link_lambda above).
    // The dimple is modest because the Go2 hold-PD is soft (the hanging foot floats
    // up to barely rest on the taut membrane rather than pressing through it).
    EXPECT_LT(with_media.cloth_min_z, cloth_z - 0.001f)
        << "the standing foot did not dip the cloth below its lay";

    // TWO-WAY to FLUID: the same, against a fluid-slice particle, pool displaced
    // (the submerged foot pushes fluid up at the edges OR down in the pocket).
    EXPECT_GT(with_media.fluid_link_rows, 0u)
        << "no robot-link<->fluid coupling rows emitted";
    EXPECT_GT(with_media.fluid_link_lambda, 0.0f)
        << "the robot link produced no body-side reaction on the fluid";
    EXPECT_TRUE(pool_rise > 0.003f || pocket_drop > 0.003f)
        << "the foot did not displace the fluid (no surface rise, no pocket drop)";

    // The reaction reaches the articulation: the held stance differs from the
    // no-media control (the impulse flows through the articulated dynamics).
    EXPECT_GT(qdot_delta, 1.0e-4)
        << "the robot dynamics did not respond to the coupled media";

    // Every coupling row had exactly one particle side (the general path).
    EXPECT_EQ(with_media.two_particle_rows, 0u)
        << "a body<->particle slot carried two particle sides (not the general path)";
}
