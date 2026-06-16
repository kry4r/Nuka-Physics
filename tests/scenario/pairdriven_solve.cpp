// ---------------------------------------------------------------------------
// General contact pipeline — Phase 1B SOLVER + INTEGRATION gate.
//
// Proves the GENERAL mixed-island solver: a detected PairDriven manifold
// (broadphase LBVH -> cvx narrowphase -> the unified contact buffer) becomes a
// COUPLED two-sided NkRow assembled by OpAssembleRowsPairDriven (S1/S2/S5) and
// solved by the generalized SolveRowsBlockIsland (S3 per-articulation qdot tiles
// + S4 two-key schedule), scattering reaction into BOTH sides. NO special-casing:
// a free rigid body, an articulation link, and a static collidable are all just
// physics bodies fed into the EXISTING coupling engine (SolveUnionRowWarp).
//
// Cases (the Phase 1 exit gate, §9):
//   (b1) TWO free rigid boxes collide head-on -> push apart + SYMMETRIC momentum
//        exchange (free x free, no articulation).
//   (b2) a free box resting on a STATIC ground plane stays (free x static; the
//        static side scatters no reaction, the box is held up).
//   (e)  TWO-RUN byte-identity for the PairDriven solve path (it does NOT inherit
//        single-dog D1 -- proven here directly on q/qdot/body velocity/lambda).
//
// The dog cases (artic x rigid box-on-dog mixed island; artic x artic two dogs)
// live in pairdriven_dog_solve.cpp (they need the go2 USD asset).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox   = 2u;  // collision::ShapeKind Box
constexpr uint32_t kKindPlane = 3u;  // collision::ShapeKind Plane

struct Backend {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
};
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig Cfg(float gz) {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = gz;
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.max_pairs = 32u;
    return cfg;
}

void AddBox(nk::Model& m, const Vec3& pos, const Vec3& vel, float half,
            float mass, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.linear_velocity = vel;
    bi.inv_mass = mass > 0.0f ? 1.0f / mass : 0.0f;
    if (mass > 0.0f) {
        const float ii = 1.0f / (mass * half * half);  // solid-ish box diag.
        bi.inv_inertia = Vec3{ii, ii, ii};
    } else {
        bi.inv_inertia = Vec3{0, 0, 0};
    }
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id;
    sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

void AddGroundPlane(nk::Model& m, int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();      // plane at z=0.
    // The amf:: plane convention puts the plane NORMAL along the body's LOCAL +Y
    // (frame.cy). For a +Z-up ground we rotate local +Y -> world +Z (90 deg about X).
    bi.pose.rotation =
        nuka::math::Quat::FromAxisAngle(Vec3{1, 0, 0}, 1.57079632679f);
    bi.inv_mass = 0.0f;                   // static (immovable; rigid arm im==0 no-op).
    bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindPlane;
    sh.params[0] = 0.0f; sh.params[1] = 0.0f; sh.params[2] = 0.0f;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id;                 // a REGULAR (im==0) body row, in the LBVH.
    sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

void FinishBodyModel(nk::Model& m) {
    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = 16u;       // candidate slots / unified buffer.
    cap.max_rows_per_env = 16u * nk::kPairDrivenRowsPerSlot;  // general row budget.
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;            // env-local pairs.
}

struct Snapshot {
    bool ok = false;
    std::vector<Vec3> lin;   // body linear velocity [bodies]
    std::vector<Transform> pose;  // body pose [bodies]
    std::vector<float> lambda;    // [rows]
};

Snapshot Download(nk::World& w, uint32_t bodies, uint32_t rows) {
    Snapshot s;
    s.lin.assign(bodies, Vec3::Zero());
    s.pose.assign(bodies, Transform::Identity());
    s.lambda.assign(rows, 0.0f);
    nk::Data& d = w.GetData();
    s.ok = d.DownloadField(nk::FieldId::BodyLinearVelocity, s.lin.data(),
                           s.lin.size() * sizeof(Vec3)) &&
           d.DownloadField(nk::FieldId::BodyPose, s.pose.data(),
                           s.pose.size() * sizeof(Transform)) &&
           d.DownloadField(nk::FieldId::Lambda, s.lambda.data(),
                           s.lambda.size() * sizeof(float));
    return s;
}

}  // namespace

// --- (b1) two free boxes collide + symmetric momentum exchange ---------------
TEST(PairDrivenSolve, TwoFreeBoxesCollideExchangeMomentum) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    // Box A at x=-0.09 moving +X (toward B); box B at x=+0.09 moving -X. Half 0.10
    // each -> centers 0.18 apart, faces touch at 0.20 reach -> 0.02 overlap. Equal
    // mass, head-on, no gravity: a symmetric elastic-ish exchange (PGS) MUST push
    // them apart and reverse the sign of each box's x-velocity contribution.
    nk::Model m;
    AddBox(m, Vec3{-0.09f, 0, 0}, Vec3{+0.5f, 0, 0}, 0.10f, 1.0f, 0);
    AddBox(m, Vec3{+0.09f, 0, 0}, Vec3{-0.5f, 0, 0}, 0.10f, 1.0f, 1);
    FinishBodyModel(m);
    const uint32_t bodies = 2u, rows = m.capacities.max_rows_per_env;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg(0.0f));
    ASSERT_TRUE(w.Ready());
    for (uint32_t s = 0; s < 60u; ++s) ASSERT_TRUE(w.Step().AllOk()) << s;

    const Snapshot snap = Download(w, bodies, rows);
    ASSERT_TRUE(snap.ok);
    const float xa = snap.pose[0].position.x;
    const float xb = snap.pose[1].position.x;
    const float sep = std::abs(xb - xa);
    std::fprintf(stderr,
                 "[pd free-boxes] xA=%.4f xB=%.4f sep=%.4f vA.x=%.4f vB.x=%.4f\n",
                 xa, xb, sep, snap.lin[0].x, snap.lin[1].x);

    // (1) PUSH-APART: separation grew past the 0.18 spawn gap toward ~0.20 (2*half).
    EXPECT_GT(sep, 0.18f + 1.0e-3f) << "overlapping boxes must push apart";
    // (2) TWO-WAY: box A (was +X) ends moving -X, box B (was -X) ends moving +X
    // (the contact reversed BOTH, the symmetric momentum exchange).
    EXPECT_LT(snap.lin[0].x, 0.0f) << "box A reaction must reverse its +X motion";
    EXPECT_GT(snap.lin[1].x, 0.0f) << "box B reaction must reverse its -X motion";
    // (3) SYMMETRY: equal mass, head-on -> the two post-contact x-speeds match.
    EXPECT_NEAR(std::abs(snap.lin[0].x), std::abs(snap.lin[1].x), 0.05f)
        << "equal-mass head-on collision must exchange momentum symmetrically";
    // (4) momentum: total x-momentum ~0 (started +0.5 + -0.5 == 0, no external X).
    EXPECT_NEAR(snap.lin[0].x + snap.lin[1].x, 0.0f, 0.05f)
        << "x-momentum must be conserved (sum ~ 0)";
}

// --- (b2) a free box rests on a static ground plane (free x static) ----------
TEST(PairDrivenSolve, FreeBoxRestsOnStaticGround) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    // Ground plane at z=0 (body 0, static im==0) + a box just above it (body 1)
    // dropped under gravity. The general box x plane contact must catch the box so
    // it RESTS on the surface (does not fall through), with the ground (static)
    // scattering no reaction (it is immovable).
    nk::Model m;
    AddGroundPlane(m, 0);
    AddBox(m, Vec3{0, 0, 0.105f}, Vec3{0, 0, 0}, 0.10f, 1.0f, 1);  // box base ~0.005 above ground
    FinishBodyModel(m);
    const uint32_t bodies = 2u, rows = m.capacities.max_rows_per_env;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg(-9.81f));
    ASSERT_TRUE(w.Ready());
    for (uint32_t s = 0; s < 240u; ++s) ASSERT_TRUE(w.Step().AllOk()) << s;

    const Snapshot snap = Download(w, bodies, rows);
    ASSERT_TRUE(snap.ok);
    const float box_z = snap.pose[1].position.z;
    const float ground_z = snap.pose[0].position.z;
    std::fprintf(stderr, "[pd box-on-ground] box_z=%.4f ground_z=%.4f vbox.z=%.5f\n",
                 box_z, ground_z, snap.lin[1].z);

    // (1) the box did NOT fall through (its center stays near the resting height
    // half == 0.10 above the plane; a fall-through would be deeply negative).
    EXPECT_GT(box_z, 0.10f - 0.03f) << "box sank through the ground (穿模)";
    EXPECT_LT(box_z, 0.10f + 0.03f) << "box floats above its resting height";
    // (2) the static ground never moved (im==0 -> no reaction scattered into it).
    EXPECT_NEAR(ground_z, 0.0f, 1.0e-5f) << "static ground must not move";
    EXPECT_NEAR(snap.lin[0].x, 0.0f, 1.0e-5f);
    EXPECT_NEAR(snap.lin[0].z, 0.0f, 1.0e-5f) << "static ground velocity must stay 0";
    // (3) the box is at rest (its vertical velocity damped to ~0 by the contact).
    EXPECT_LT(std::abs(snap.lin[1].z), 0.2f) << "box must come to rest on the ground";
}

// --- (e) two-run byte-identity for the PairDriven solve path ------------------
TEST(PairDrivenSolve, TwoRunsByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    auto run = [&]() -> Snapshot {
        nk::Model m;
        AddGroundPlane(m, 0);
        AddBox(m, Vec3{-0.05f, 0, 0.12f}, Vec3{+0.3f, 0, 0}, 0.10f, 1.0f, 1);
        AddBox(m, Vec3{+0.05f, 0, 0.40f}, Vec3{-0.3f, 0, 0}, 0.10f, 1.0f, 2);
        FinishBodyModel(m);
        const uint32_t bodies = 3u, rows = m.capacities.max_rows_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg(-9.81f));
        if (!w.Ready()) return {};
        for (uint32_t s = 0; s < 120u; ++s)
            if (!w.Step().AllOk()) return {};
        return Download(w, bodies, rows);
    };

    const Snapshot a = run();
    const Snapshot c = run();
    ASSERT_TRUE(a.ok && c.ok);

    auto same = [](const auto& x, const auto& y) {
        return x.size() == y.size() &&
               std::memcmp(x.data(), y.data(),
                           x.size() * sizeof(typename std::decay_t<decltype(x)>::value_type)) == 0;
    };
    EXPECT_TRUE(same(a.lin, c.lin)) << "body velocity differs across runs";
    EXPECT_TRUE(same(a.pose, c.pose)) << "body pose differs across runs";
    EXPECT_TRUE(same(a.lambda, c.lambda)) << "contact lambda differs across runs";
}
