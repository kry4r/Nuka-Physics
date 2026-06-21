// ---------------------------------------------------------------------------
// 3D SOFT TET body <-> rigid TWO-WAY coupling on the ONE general row solver.
//
// A real XPBD tetrahedral soft body (a cube tessellated into tets with per-edge
// distance + per-tet VOLUME constraints) is dropped onto a static ground plane.
// Its particles are SPHERE collidables flowing through the SAME body<->particle
// row solver a foot uses on the ground, and the SAME row a cloth particle uses:
// detection -> body<->particle narrowphase -> emit -> SolveRowsBlockIsland ->
// Xpbd finalize compose. There is NO bespoke "soft solid" coupler -- a tet body
// is the SAME particle array as a cloth, only its INTERNAL XPBD constraints
// differ (volume vs bend). The contact row is identical: a particle is a particle.
//
// The two-way effect, proven numerically against a no-floor control:
//   (a) DEFORM: the cube's vertical extent (max_z - min_z) compresses measurably
//       below its rest extent at impact (it squashes on the floor).
//   (b) TWO-WAY: with the floor present the cube RESTS in a shallow band at the
//       floor (well above the no-floor control that free-falls past it) AND a
//       body<->particle normal row carries a body-side lambda > 0 -- the reaction
//       the floor pushes back into the soft body.
//   (c) RECOVER: after the load settles the cube's mean tet volume returns toward
//       its rest volume (the VOLUME constraint restores it).
//   Every coupling row asserted has exactly one side kind == kNkSideParticle, so
//   the contact is the general body<->particle path, not a special-cased coupler.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "collision/shape_kind.hpp"
#include "import/cooker/xpbd_cooker_types.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "runtime/soft/tetmesh_topology.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace soft = nuka::runtime::soft;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindPlane = nuka::collision::kShapePlane;

// Soft cube geometry. A 3x3x3 grid of cells -> 4x4x4 = 64 vertices, each cell
// split into 5 tets (320 tets). A modest cell count keeps the tet body well
// conditioned; the cube is ~0.18 m on a side, light, dropped onto the floor.
constexpr uint32_t kCells = 3u;             // cells per axis (4 vertices per axis).
constexpr uint32_t kVertsPerAxis = kCells + 1u;
constexpr float kCubeSide = 0.18f;          // total cube edge length.
constexpr float kCellLen = kCubeSide / static_cast<float>(kCells);
constexpr float kDropHeight = 0.30f;        // cube bottom starts well above the floor.
constexpr float kParticleMass = 0.01f;      // 10 g per vertex (light soft body).
constexpr float kContactDMin = 0.030f;      // particle sphere radius = d_min/2 = 0.015.
constexpr float kRadius = kContactDMin * 0.5f;
constexpr uint16_t kXpbdIters = 24u;
constexpr float kVolAlpha = 4.0e-5f;        // modest volume compliance: squash + recover.
constexpr uint32_t kFloorBody = 0u;         // only collidable is the ground plane.

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
    cfg.vel_iters = 32u;
    cfg.pos_iters = 0u;
    cfg.max_pairs = 64u;
    return cfg;
}

// A static z-up ground plane at z=0 (the amf plane normal is local +Y, rotated to
// world +Z). One-way by mass; the soft body lands on it.
void AddGroundPlane(nk::Model& m, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.rotation = Quat::FromAxisAngle(Vec3{1, 0, 0}, 1.57079632679f);
    bi.inv_mass = 0.0f;
    bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindPlane;
    sh.params[0] = 0.0f; sh.params[1] = 0.0f; sh.params[2] = 0.0f;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// 5-tet decomposition of one grid cell (a hexahedron with 8 corners). The split
// alternates by cell parity so the diagonals on a shared face MATCH between
// neighbouring cells (a consistent tessellation, no cracks). cn[] = the 8 global
// vertex indices in the (x,y,z) corner order 000,100,010,110,001,101,011,111.
void EmitCellTets(const uint32_t cn[8], bool even,
                  std::vector<soft::TetMeshTet>& out) {
    auto tet = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        out.push_back(soft::TetMeshTet{{cn[a], cn[b], cn[c], cn[d]}});
    };
    if (even) {
        tet(0, 1, 2, 4);
        tet(1, 2, 3, 7);
        tet(1, 4, 5, 7);
        tet(2, 4, 6, 7);
        tet(1, 2, 4, 7);  // central tet.
    } else {
        tet(0, 1, 3, 5);
        tet(0, 2, 3, 6);
        tet(0, 4, 5, 6);
        tet(3, 5, 6, 7);
        tet(0, 3, 5, 6);  // central tet.
    }
}

// A soft cube: 4x4x4 lattice rest positions + a consistent 5-tets-per-cell mesh,
// fed through BuildTetMeshConstraints for distance + volume constraints, then
// transcribed into the XpbdCookInput (volume rides the same cook the cloth bend
// rides). The cube bottom rests at z = bottom_z so the whole solid sits above 0.
cook::XpbdCookInput BuildSoftCube(float bottom_z) {
    const uint32_t n = kVertsPerAxis;
    auto vid = [n](uint32_t i, uint32_t j, uint32_t k) {
        return (k * n + j) * n + i;
    };
    std::vector<Vec3> rest;
    rest.reserve(n * n * n);
    const float c0 = -0.5f * kCubeSide;  // center the cube on x/y.
    for (uint32_t k = 0; k < n; ++k) {
        for (uint32_t j = 0; j < n; ++j) {
            for (uint32_t i = 0; i < n; ++i) {
                rest.push_back(Vec3{c0 + static_cast<float>(i) * kCellLen,
                                    c0 + static_cast<float>(j) * kCellLen,
                                    bottom_z + static_cast<float>(k) * kCellLen});
            }
        }
    }
    std::vector<soft::TetMeshTet> tets;
    for (uint32_t k = 0; k + 1 < n; ++k) {
        for (uint32_t j = 0; j + 1 < n; ++j) {
            for (uint32_t i = 0; i + 1 < n; ++i) {
                const uint32_t cn[8] = {
                    vid(i, j, k),         vid(i + 1, j, k),
                    vid(i, j + 1, k),     vid(i + 1, j + 1, k),
                    vid(i, j, k + 1),     vid(i + 1, j, k + 1),
                    vid(i, j + 1, k + 1), vid(i + 1, j + 1, k + 1)};
                EmitCellTets(cn, ((i + j + k) & 1u) == 0u, tets);
            }
        }
    }

    soft::TetMeshTopologyOptions opts;
    opts.distance_compliance_alpha = 0.0f;     // stiff edges (shape skeleton).
    opts.volume_compliance_alpha = kVolAlpha;  // modest volume -> visible squash.
    opts.emit_distance_constraints = true;
    soft::XpbdConstraintSet cs;
    soft::BuildTetMeshConstraints(rest, tets, opts, cs);

    cook::XpbdCookInput in;
    in.positions = rest;
    in.velocities.assign(rest.size(), Vec3::Zero());
    in.inv_mass.assign(rest.size(), 1.0f / kParticleMass);  // free (unpinned) solid.
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

// Tet list rebuilt host-side to measure per-tet rest/live volume (the same
// tessellation BuildSoftCube emits, so volume tracks the cooked constraints).
std::vector<soft::TetMeshTet> CubeTets() {
    const uint32_t n = kVertsPerAxis;
    auto vid = [n](uint32_t i, uint32_t j, uint32_t k) {
        return (k * n + j) * n + i;
    };
    std::vector<soft::TetMeshTet> tets;
    for (uint32_t k = 0; k + 1 < n; ++k) {
        for (uint32_t j = 0; j + 1 < n; ++j) {
            for (uint32_t i = 0; i + 1 < n; ++i) {
                const uint32_t cn[8] = {
                    vid(i, j, k),         vid(i + 1, j, k),
                    vid(i, j + 1, k),     vid(i + 1, j + 1, k),
                    vid(i, j, k + 1),     vid(i + 1, j, k + 1),
                    vid(i, j + 1, k + 1), vid(i + 1, j + 1, k + 1)};
                EmitCellTets(cn, ((i + j + k) & 1u) == 0u, tets);
            }
        }
    }
    return tets;
}

// Cook the ground plane + the soft cube into ONE Model. CookXpbdParticles sets
// mode = Xpbd (the tet path) and grows the rigid budget additively for the
// body<->particle reserve, exactly as the cloth scene does.
nk::Model BuildScene(float bottom_z, bool floor_present) {
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

    const cook::XpbdCookInput cube = BuildSoftCube(bottom_z);
    cook::CookXpbdParticles(m, 1u, cube);   // appends the tet body + grows the budget.
    m.particles.pp_contact_d_min = kContactDMin;
    // The no-floor control sinks the plane far below so only free-fall acts.
    if (!floor_present) m.body_init[kFloorBody].pose.position.z = -1000.0f;
    return m;
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

void DownloadParticles(nk::World& w, std::vector<Vec3>* pos) {
    const uint32_t P = w.GetModel().capacities.particles_per_env;
    pos->assign(P, Vec3::Zero());
    w.GetData().DownloadField(nk::FieldId::ParticlePos, pos->data(),
                              P * sizeof(Vec3));
}

// Vertical extent (max_z - min_z) of the soft body -- compresses when squashed.
float Extent(const std::vector<Vec3>& pos) {
    float lo = 1.0e9f, hi = -1.0e9f;
    for (const Vec3& p : pos) { lo = std::min(lo, p.z); hi = std::max(hi, p.z); }
    return hi - lo;
}

float MinZ(const std::vector<Vec3>& pos) {
    float lo = 1.0e9f;
    for (const Vec3& p : pos) lo = std::min(lo, p.z);
    return lo;
}

// Mean per-tet signed volume over the live positions (rest -> live ratio = squash).
float MeanTetVolume(const std::vector<Vec3>& pos,
                    const std::vector<soft::TetMeshTet>& tets) {
    double acc = 0.0;
    for (const soft::TetMeshTet& t : tets) {
        acc += std::fabs(soft::TetSignedVolumeTimes6(
            pos[t.v[0]], pos[t.v[1]], pos[t.v[2]], pos[t.v[3]]));
    }
    return static_cast<float>(acc / static_cast<double>(tets.size()));
}

}  // namespace

// (a)+(b)+(c): a soft tet cube dropped onto a static floor DEFORMS (extent
// compresses), is TWO-WAY (rests at its radius vs a no-floor control + body-side
// normal lambda > 0 on a body<->particle row), and RECOVERS its volume after.
TEST(SoftTetPresserTwoWay, SoftCubeLandsDeformsAndRecovers) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    const std::vector<soft::TetMeshTet> tets = CubeTets();

    // Rest geometry (the cube as built): rest extent + rest mean tet volume.
    const cook::XpbdCookInput rest_in = BuildSoftCube(kDropHeight);
    const float rest_extent = Extent(rest_in.positions);
    const float rest_volume = MeanTetVolume(rest_in.positions, tets);

    // WITH the floor: drop the cube, watch it land, deform, and settle.
    nk::Model m = BuildScene(kDropHeight, /*floor_present=*/true);
    const uint32_t rows = m.capacities.max_rows_per_env;
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    std::vector<float> urows(static_cast<size_t>(rows) * 32u, 0.0f);
    std::vector<float> lambda(rows, 0.0f);
    float max_body_lambda = 0.0f;
    uint32_t particle_rows_seen = 0u;
    float min_extent_during = 1.0e9f;  // deepest squash seen across the drop.
    nk::Data& d = w.GetData();

    // Fall + impact: ~360 steps (1.5 s) covers the drop, the squash, and rebound.
    for (uint32_t s = 0; s < 360u; ++s) {
        w.Step();
        d.DownloadField(nk::FieldId::Urows, urows.data(), urows.size() * sizeof(float));
        d.DownloadField(nk::FieldId::Lambda, lambda.data(), lambda.size() * sizeof(float));
        for (uint32_t row = 0u; row < rows; ++row) {
            const RowSides rs = DecodeRow(urows, row);
            if (!rs.active) continue;
            const bool a_part = rs.a_kind == nk::kNkSideParticle;
            const bool b_part = rs.b_kind == nk::kNkSideParticle;
            if (!(a_part || b_part)) continue;
            EXPECT_FALSE(a_part && b_part) << "row has two particle sides";
            ++particle_rows_seen;
            if (rs.upper > 1.0e30f)  // a normal row (friction spokes cap upper to 0).
                max_body_lambda = std::max(max_body_lambda, lambda[row]);
        }
        std::vector<Vec3> p; DownloadParticles(w, &p);
        min_extent_during = std::min(min_extent_during, Extent(p));
    }
    std::vector<Vec3> landed; DownloadParticles(w, &landed);
    const float landed_min_z = MinZ(landed);
    const float landed_volume = MeanTetVolume(landed, tets);

    // RECOVER: let it relax in place ~240 steps; the volume constraint restores it.
    for (uint32_t s = 0; s < 240u; ++s) w.Step();
    std::vector<Vec3> recovered; DownloadParticles(w, &recovered);
    const float recovered_volume = MeanTetVolume(recovered, tets);

    // CONTROL: identical cube, floor sunk far below -> only free-fall, no contact.
    nk::Model mc = BuildScene(kDropHeight, /*floor_present=*/false);
    nk::World wc(std::move(mc), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(wc.Ready());
    for (uint32_t s = 0; s < 360u; ++s) wc.Step();
    std::vector<Vec3> control; DownloadParticles(wc, &control);
    const float control_min_z = MinZ(control);

    std::fprintf(stderr,
                 "[soft-tet] rest_extent=%.4f min_extent_during=%.4f (squash=%.4f) "
                 "rest_vol=%.5f landed_vol=%.5f recovered_vol=%.5f\n",
                 rest_extent, min_extent_during, rest_extent - min_extent_during,
                 rest_volume, landed_volume, recovered_volume);
    std::fprintf(stderr,
                 "[soft-tet] landed_min_z=%.4f control_min_z=%.4f radius=%.4f "
                 "body_lambda=%.6f particle_rows_seen=%u\n",
                 landed_min_z, control_min_z, kRadius, max_body_lambda,
                 particle_rows_seen);

    for (const Vec3& p : landed)
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));

    // (a) DEFORM: the cube squashed measurably below its rest extent at impact.
    EXPECT_LT(min_extent_during, rest_extent - 0.005f)
        << "the soft cube did not deform on impact (no squash)";
    // (b) TWO-WAY, direction 1: a body<->particle normal row carried a push-out.
    EXPECT_GT(particle_rows_seen, 0u) << "no body<->particle coupling rows emitted";
    EXPECT_GT(max_body_lambda, 0.0f)
        << "the floor produced no body-side reaction impulse on the soft body";
    // (b) TWO-WAY, direction 2: the cube RESTS in a shallow band at the floor (the
    // velocity row holds it up to a small steady-state penetration, pos_iters 0),
    // while the no-floor control free-falls far past it (the floor stopped the fall).
    EXPECT_LT(std::fabs(landed_min_z), kRadius + kContactDMin)
        << "the soft body did not rest near the floor (it tunneled or floated)";
    EXPECT_LT(control_min_z, landed_min_z - 0.5f)
        << "the no-floor control did not fall past the landed cube (no two-way arrest)";
    // (c) RECOVER: the squashed cube's mean tet volume returns toward rest.
    EXPECT_GT(recovered_volume, landed_volume + 1.0e-6f)
        << "the soft cube did not recover volume after the load (volume not restored)";
    EXPECT_LT(std::fabs(recovered_volume - rest_volume) / rest_volume, 0.10f)
        << "the recovered volume is not within 10% of rest";
}
