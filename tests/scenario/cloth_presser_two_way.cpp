// ---------------------------------------------------------------------------
// Cloth <-> presser TWO-WAY coupling on the ONE general row solver.
//
// A real XPBD cloth grid (24x24 = 576 particles) drapes over a low static ridge
// above a ground plane, then a simple rigid body presses it. The cloth particles
// are SPHERE collidables flowing through the SAME body<->particle row solver a
// foot uses on the ground: detection -> body<->particle narrowphase -> emit ->
// SolveRowsBlockIsland -> SoftFluid finalize compose. No bespoke coupler: the
// presser is a general rigid body, the cloth is particles, both on one path.
//
// The two directions of the two-way effect, proven numerically:
//   (a) CLOTH DIPS: the min particle z under a kinematically lowered presser
//       drops measurably below the settled drape rest, then RECOVERS toward rest
//       when the presser lifts (the cloth springs back).
//   (b) PRESSER IS RESISTED: a DYNAMIC presser released onto the drape is arrested
//       (its z stays well above a no-cloth control that keeps falling to the ridge)
//       AND a body<->particle row carries a body-side normal lambda > 0 -- the
//       reaction the cloth pushes back into the body. The control proves the arrest
//       is the cloth, not gravity alone.
//   Every coupling row asserted has exactly one side kind == kNkSideParticle, so the
//   contact is the general body<->particle path, not a special-cased coupler.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "collision/shape_kind.hpp"
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

constexpr uint32_t kKindBox = nuka::collision::kShapeBox;
constexpr uint32_t kKindSphere = nuka::collision::kShapeSphere;
constexpr uint32_t kKindPlane = nuka::collision::kShapePlane;

// Cloth + scene geometry. An unpinned sheet across two parallel low ridges settles
// like a hammock: it rests on the ridge tops, its center sagging into the gap.
constexpr uint32_t kGridN = 24u;          // 24x24 = 576 particles.
constexpr float kSpacing = 0.025f;        // patch ~0.575 m square.
constexpr float kClothZ = 0.16f;          // flat lay height just above the ridge tops.
constexpr float kParticleMass = 0.01f;    // 10 g per particle (light cloth).
constexpr float kContactDMin = 0.030f;    // particle sphere radius = d_min/2 = 0.015.
constexpr float kRidgeHalfX = 0.12f;      // each ridge half-extents (tops at z=0.10).
constexpr float kRidgeHalfY = 0.30f;
constexpr float kRidgeHalfZ = 0.05f;
constexpr float kRidgeCenterX = 0.16f;    // ridges at x = +/-0.16 (gap ~0.08 wide).
constexpr float kRidgeTopZ = kRidgeHalfZ * 2.0f;  // ridges sit on the ground -> 0.10.
constexpr float kPresserRadius = 0.06f;
constexpr uint16_t kXpbdIters = 20u;
constexpr uint32_t kSettleSteps = 800u;
constexpr uint32_t kPresserBody = 3u;     // ground(0) + ridge x2(1,2) + presser(3).

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

// A static body collidable (immovable, one-way by mass).
void AddStaticBox(nk::Model& m, const Vec3& pos, const Vec3& half, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = 0.0f;
    bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half.x; sh.params[1] = half.y; sh.params[2] = half.z;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// A static z-up ground plane at z=0 (the amf plane normal is local +Y, rotated to
// world +Z). A plane has robust analytic contact for the hanging cloth edges.
void AddGroundPlane(nk::Model& m, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.rotation = nuka::math::Quat::FromAxisAngle(Vec3{1, 0, 0}, 1.57079632679f);
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

// A sphere body collidable. inv_mass == 0 makes it kinematic (driven by upload),
// inv_mass > 0 makes it a free dynamic body the contact reaction can move.
void AddSphere(nk::Model& m, const Vec3& pos, float radius, float inv_mass,
               int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = inv_mass;
    if (inv_mass > 0.0f) {
        const float ii = inv_mass / (0.4f * radius * radius);  // solid-sphere diag.
        bi.inv_inertia = Vec3{ii, ii, ii};
    } else {
        bi.inv_inertia = Vec3{0, 0, 0};
    }
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindSphere;
    sh.params[0] = radius; sh.params[1] = radius; sh.params[2] = radius;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// The cloth XPBD input: a flat WxH lattice with stretch + bend constraints. Two
// far corners of the top row are pinned so the drape hangs without sliding off.
cook::XpbdCookInput BuildCloth() {
    std::vector<Vec3> rest;
    rest.reserve(kGridN * kGridN);
    const float c0 = -0.5f * static_cast<float>(kGridN - 1u) * kSpacing;
    for (uint32_t j = 0; j < kGridN; ++j) {
        for (uint32_t i = 0; i < kGridN; ++i) {
            rest.push_back(Vec3{c0 + static_cast<float>(i) * kSpacing,
                                c0 + static_cast<float>(j) * kSpacing, kClothZ});
        }
    }
    auto idx = [](uint32_t i, uint32_t j) { return j * kGridN + i; };
    std::vector<soft::ClothTriangle> tris;
    for (uint32_t j = 0; j + 1 < kGridN; ++j) {
        for (uint32_t i = 0; i + 1 < kGridN; ++i) {
            tris.push_back(soft::ClothTriangle{{idx(i, j), idx(i + 1, j),
                                                idx(i + 1, j + 1)}});
            tris.push_back(soft::ClothTriangle{{idx(i, j), idx(i + 1, j + 1),
                                                idx(i, j + 1)}});
        }
    }
    soft::ClothTopologyOptions opts;
    opts.distance_compliance_alpha = 0.0f;     // inextensible stretch.
    opts.bend_compliance_alpha = 1.0e-4f;      // soft bend so it drapes.
    soft::XpbdConstraintSet cs;
    soft::BuildClothConstraints(rest, tris, opts, cs);

    cook::XpbdCookInput in;
    in.positions = rest;
    // A tiny downward seed so the flat sheet commits to the ridge symmetrically.
    in.velocities.assign(rest.size(), Vec3{0.0f, 0.0f, -0.05f});
    in.inv_mass.assign(rest.size(), 1.0f / kParticleMass);  // UNPINNED free drape.
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

// Cook ground + two ridges + a presser sphere + the cloth into ONE model;
// CookXpbdParticles grows the rigid budget additively for the body<->particle reserve.
// presser_inv_mass picks kinematic(0)/dynamic(>0); the resist test passes a gap radius.
nk::Model BuildScene(const Vec3& presser_pos, float presser_inv_mass,
                     float presser_radius = kPresserRadius) {
    nk::Model m;
    const Vec3 ridge_half{kRidgeHalfX, kRidgeHalfY, kRidgeHalfZ};
    AddGroundPlane(m, 0);                                                 // ground.
    AddStaticBox(m, Vec3{-kRidgeCenterX, 0.0f, kRidgeHalfZ}, ridge_half, 1);  // ridge -x.
    AddStaticBox(m, Vec3{+kRidgeCenterX, 0.0f, kRidgeHalfZ}, ridge_half, 2);  // ridge +x.
    AddSphere(m, presser_pos, presser_radius, presser_inv_mass, 3);       // presser.

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = bodies * 4u;   // rigid budget (cook sizing).
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    const cook::XpbdCookInput cloth = BuildCloth();
    cook::CookXpbdParticles(m, 1u, cloth);    // appends cloth + grows the budget.
    m.particles.pp_contact_d_min = kContactDMin;
    return m;
}

// NkRow packs to 32 f32: [0]=flags [7]=upper, a.kind [16] a.index [17],
// b.kind [24] b.index [25].
struct RowSides { uint32_t a_kind, b_kind; bool active; float upper; uint32_t flags; };
RowSides DecodeRow(const std::vector<float>& urows, uint32_t row) {
    const float* r = urows.data() + static_cast<size_t>(row) * 32u;
    auto u = [&](int i) { uint32_t v; std::memcpy(&v, &r[i], 4); return v; };
    RowSides s;
    s.active = (u(0) & 1u) != 0u;
    s.flags = u(0);
    s.upper = r[7];
    s.a_kind = u(16); s.b_kind = u(24);
    return s;
}

// Min particle z over the cloth whose xy lies within reach of the presser sphere
// (the pressed pocket), so the metric tracks the dip, not the wider drape.
float MinZUnderPresser(const std::vector<Vec3>& pos, const Vec3& presser_xy) {
    float min_z = 1.0e9f;
    const float reach = kPresserRadius + 4.0f * kSpacing;
    for (const Vec3& p : pos) {
        const float dx = p.x - presser_xy.x, dy = p.y - presser_xy.y;
        if (dx * dx + dy * dy <= reach * reach) min_z = std::min(min_z, p.z);
    }
    return min_z;
}

void DownloadParticles(nk::World& w, std::vector<Vec3>* pos) {
    const uint32_t P = w.GetModel().capacities.particles_per_env;
    pos->assign(P, Vec3::Zero());
    w.GetData().DownloadField(nk::FieldId::ParticlePos, pos->data(),
                              P * sizeof(Vec3));
}

// Per-body byte offsets into the env-major Data fields (DownloadField/UploadField
// take a BYTE offset; the presser is body row kPresserBody in env 0).
constexpr uint64_t kPoseOff = static_cast<uint64_t>(kPresserBody) * sizeof(Transform);
constexpr uint64_t kVec3Off = static_cast<uint64_t>(kPresserBody) * sizeof(Vec3);
constexpr uint64_t kFloatOff = static_cast<uint64_t>(kPresserBody) * sizeof(float);

// Drive the kinematic presser to `target` this step: set its pose and zero its
// velocity (a scripted kinematic body, the viewer_move_entity primitive).
void DrivePresser(nk::World& w, const Vec3& target) {
    Transform tf = Transform::Identity();
    tf.position = target;
    const Vec3 zero = Vec3::Zero();
    w.GetData().UploadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), kPoseOff);
    w.GetData().UploadField(nk::FieldId::BodyLinearVelocity, &zero, sizeof(Vec3), kVec3Off);
    w.GetData().UploadField(nk::FieldId::BodyAngularVelocity, &zero, sizeof(Vec3), kVec3Off);
}

}  // namespace

// (a) the cloth dips under a lowered presser then recovers when it lifts. The
// presser is kinematic (its motion is scripted); the cloth's two-way response (dip
// + spring-back) flows entirely through the body<->particle rows.
TEST(ClothPresserTwoWay, ClothDipsUnderPresserThenRecovers) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    const Vec3 presser_xy{0.0f, 0.0f, 0.0f};
    const float top = 0.40f;                 // parked well above the cloth.
    nk::Model m = BuildScene(Vec3{presser_xy.x, presser_xy.y, top}, 0.0f);
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    // SETTLE: presser parked high, cloth drapes over the ridge.
    for (uint32_t s = 0; s < kSettleSteps; ++s) { DrivePresser(w, Vec3{0, 0, top}); w.Step(); }
    std::vector<Vec3> settled;
    DownloadParticles(w, &settled);
    const float rest_min_z = MinZUnderPresser(settled, presser_xy);

    // PRESS: lower the presser into the sagging center over ~200 steps, hold ~150.
    // The center spans the gap (no obstacle under it), so the presser pushes the
    // cloth well below its sag rest without hitting rigid geometry.
    const float bottom = kRidgeTopZ - 0.02f + kPresserRadius;  // sphere center ~0.14.
    for (uint32_t s = 0; s < 200u; ++s) {
        const float t = static_cast<float>(s) / 199.0f;
        DrivePresser(w, Vec3{0, 0, top + (bottom - top) * t});
        w.Step();
    }
    for (uint32_t s = 0; s < 150u; ++s) { DrivePresser(w, Vec3{0, 0, bottom}); w.Step(); }
    std::vector<Vec3> pressed;
    DownloadParticles(w, &pressed);
    const float pressed_min_z = MinZUnderPresser(pressed, presser_xy);

    // RECOVER: lift the presser away over ~200 steps, let the cloth relax ~250.
    for (uint32_t s = 0; s < 200u; ++s) {
        const float t = static_cast<float>(s) / 199.0f;
        DrivePresser(w, Vec3{0, 0, bottom + (top - bottom) * t});
        w.Step();
    }
    for (uint32_t s = 0; s < 250u; ++s) { DrivePresser(w, Vec3{0, 0, top}); w.Step(); }
    std::vector<Vec3> recovered;
    DownloadParticles(w, &recovered);
    const float recovered_min_z = MinZUnderPresser(recovered, presser_xy);

    std::fprintf(stderr,
                 "[cloth-presser] rest_min_z=%.4f pressed_min_z=%.4f "
                 "recovered_min_z=%.4f (dip=%.4f recovery=%.4f)\n",
                 rest_min_z, pressed_min_z, recovered_min_z,
                 rest_min_z - pressed_min_z, recovered_min_z - pressed_min_z);

    ASSERT_TRUE(std::isfinite(rest_min_z) && std::isfinite(pressed_min_z) &&
                std::isfinite(recovered_min_z));
    // (a) the cloth dipped measurably under the presser.
    EXPECT_LT(pressed_min_z, rest_min_z - 0.01f)
        << "the presser did not push the cloth down (no dip)";
    // (c) the cloth sprang back toward rest after the presser lifted.
    EXPECT_GT(recovered_min_z, pressed_min_z + 0.01f)
        << "the cloth did not recover after the presser lifted";
}

// (b) a DYNAMIC presser released onto the drape is ARRESTED by the cloth (vs a
// no-cloth control that falls through the gap to the ground) AND a body<->particle
// row carries a body-side normal lambda > 0 -- the reaction direction of the two-way.
TEST(ClothPresserTwoWay, DynamicPresserIsResistedByCloth) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    const Vec3 center{0.0f, 0.0f, 0.0f};
    // A lighter, wider probe: the condim=3 elliptic cone carries less peak
    // tangential grip than the legacy pyramid, so capture must come from the
    // drape's tension arc (ball wider than a mesh cell) rather than edge
    // friction; the reduced mass keeps the impact inside that regime.
    const float inv_mass = 1.0f / 0.35f;
    // A gap-fitting presser (radius < half the ridge gap): WITHOUT the cloth it falls
    // clean through the gap to the ground; WITH the cloth it is caught by the drape.
    const float resist_radius = 0.038f;

    // Drops a DYNAMIC presser (built dynamic from the start, parked far in xy during
    // settle so it lands away from the cloth) onto the center from `release_z`, runs
    // it, and returns its final z + the deepest body<->particle normal reaction. With
    // `cloth_present` false the cloth is sunk far below so only the ground stops it.
    auto drop_presser = [&](bool cloth_present, float* out_lambda,
                            uint32_t* out_particle_rows) -> float {
        nk::Model m = BuildScene(Vec3{2.0f, 2.0f, resist_radius}, inv_mass,
                                 resist_radius);
        if (!cloth_present)
            for (Vec3& p : m.particles.initial_pos) p.z -= 10.0f;  // cloth out of reach.
        const uint32_t rows = m.capacities.max_rows_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready()) return 1e9f;
        // Settle the cloth while the presser falls to rest far away (x=2).
        for (uint32_t s = 0; s < kSettleSteps; ++s) w.Step();
        const float drape_top = [&] {
            std::vector<Vec3> s; DownloadParticles(w, &s);
            return MinZUnderPresser(s, center);
        }();
        const float release_z = (cloth_present ? drape_top : kRidgeTopZ) +
                                resist_radius + 0.02f;
        // Teleport the (still dynamic) presser above the center, at rest, and drop it.
        Transform tf = Transform::Identity();
        tf.position = Vec3{center.x, center.y, release_z};
        const Vec3 zero = Vec3::Zero();
        nk::Data& d = w.GetData();
        d.UploadField(nk::FieldId::BodyPose, &tf, sizeof(Transform), kPoseOff);
        d.UploadField(nk::FieldId::BodyLinearVelocity, &zero, sizeof(Vec3), kVec3Off);
        d.UploadField(nk::FieldId::BodyAngularVelocity, &zero, sizeof(Vec3), kVec3Off);

        std::vector<float> urows(static_cast<size_t>(rows) * 32u, 0.0f);
        std::vector<float> lambda(rows, 0.0f);
        float max_lambda = 0.0f;
        uint32_t particle_rows = 0u;
        for (uint32_t s = 0; s < 300u; ++s) {
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
                ++particle_rows;
                if (!(rs.flags & nk::nk_row_flags::kBlockNormal)) continue;
                max_lambda = std::max(max_lambda, lambda[row]);
            }
        }
        Transform got = Transform::Identity();
        d.DownloadField(nk::FieldId::BodyPose, &got, sizeof(Transform), kPoseOff);
        if (cloth_present) {
            std::vector<Vec3> ppos;
            DownloadParticles(w, &ppos);
            float min_under = 1e9f;
            for (const Vec3& p : ppos) {
                const float dx = p.x - center.x, dy = p.y - center.y;
                if (dx * dx + dy * dy < resist_radius * resist_radius)
                    min_under = std::min(min_under, p.z);
            }
            std::fprintf(stderr,
                         "[cloth-presser] final presser z=%.4f min particle z under "
                         "presser=%.4f particles=%zu\n",
                         got.position.z, min_under, ppos.size());
        }
        if (out_lambda) *out_lambda = max_lambda;
        if (out_particle_rows) *out_particle_rows = particle_rows;
        return got.position.z;
    };

    float body_particle_lambda = 0.0f;
    uint32_t particle_rows_seen = 0u;
    const float presser_z_with_cloth =
        drop_presser(true, &body_particle_lambda, &particle_rows_seen);
    const float presser_z_no_cloth = drop_presser(false, nullptr, nullptr);

    std::fprintf(stderr,
                 "[cloth-presser] presser_z_with_cloth=%.4f presser_z_no_cloth=%.4f "
                 "body_particle_lambda=%.6f particle_rows_seen=%u\n",
                 presser_z_with_cloth, presser_z_no_cloth, body_particle_lambda,
                 particle_rows_seen);

    // The reaction direction: a body<->particle normal row carried a real push-out.
    EXPECT_GT(particle_rows_seen, 0u) << "no body<->particle coupling rows emitted";
    EXPECT_GT(body_particle_lambda, 0.0f)
        << "the cloth produced no body-side reaction impulse on the presser";
    // The presser was ARRESTED by the cloth: it came to rest well above the no-cloth
    // control, which falls through the gap to the ground (rest near the radius).
    EXPECT_GT(presser_z_with_cloth, presser_z_no_cloth + 0.02f)
        << "the cloth did not resist the presser (it fell as far as the no-cloth "
           "control)";
}
