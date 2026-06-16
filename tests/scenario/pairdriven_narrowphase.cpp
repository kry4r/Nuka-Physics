// ---------------------------------------------------------------------------
// General contact pipeline — Phase 1A NARROWPHASE DETECTION gate.
//
// Proves the cvx:: GJK/EPA/face-clip convex narrowphase is WIRED end-to-end into
// the live PairDriven pipeline at the DETECTION layer (the solver consumes it in
// Phase 1B). A small body-only world with contact_family == PairDriven and two
// OVERLAPPING collidables runs broadphase (LBVH) -> narrowphase
// (DispatchPair -> cvx::ConvexNarrowphase) -> the unified contact buffer
// (ucontact_*). The pairs chosen (capsule x box, box x box-via-cvx) EXERCISE the
// cvx GJK/EPA path (capsule x box is NOT in the analytic ladder, so it falls to
// the G5 fallback), not just the inline analytic handlers.
//
// Asserts: contact_count > 0; each manifold <= 4 pts; every point/normal finite
// (no NaN); ucontact_a/b carry the candidate collidable ids; the normal is a
// plausible unit separation dir for side A with positive depth; the tangent basis
// (C4) is orthonormal to the normal; and TWO RUNS produce a byte-identical
// contact stream (the new path does NOT inherit single-dog D1 — proven here).
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

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Transform;
using nuka::math::Vec3;

// Shape kinds as stored in the shape_table (collision::ShapeKind; the cook packs
// CollisionShapeComponent::Kind: Sphere=0, Capsule=1, Box=2).
constexpr uint32_t kKindCapsule = 1u;
constexpr uint32_t kKindBox     = 2u;

// Candidate slot capacity per env (== the PairDriven candidate stride). Small but
// > the handful of pairs two bodies can produce.
constexpr uint32_t kSlots = 16u;

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

nk::Pipeline::SolverConfig PairDrivenConfig() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f;
    cfg.gravity[1] = 0.0f;
    cfg.gravity[2] = 0.0f;       // no gravity: isolate the detection (no drift in 1 step)
    cfg.contact_margin = 0.0f;
    cfg.max_pairs = kSlots;
    return cfg;
}

// A body-only PairDriven Model with two OVERLAPPING collidables:
//   body 0 = a BOX (half-extents 0.10) at the origin (static: inv_mass 0).
//   body 1 = a CAPSULE (radius 0.05, half-height 0.10, axis = local Y) whose
//            center sits 0.12 m from the box center in X -> the capsule's radius
//            (0.05) + the box face (0.10) = 0.15 reach > 0.12 center gap, so they
//            OVERLAP by ~0.03 m. capsule x box is NOT in the analytic ladder, so
//            DispatchPair routes it to cvx::ConvexNarrowphase (the G5 fallback).
// No rows are cooked (max_rows_per_env == 0) so NO solver runs — this is a pure
// detection gate (Phase 1A). The narrowphase/broadphase/tangent ops run under
// has_collidables regardless.
nk::Model BuildTwoBodyPairDrivenModel() {
    nk::Model model;
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1u;
    cap.bodies_per_env = 2u;
    cap.max_bodies_total = 2u;            // shape_table stride == bodies_per_env.
    cap.max_contacts_per_env = kSlots;    // candidate slot capacity / unified buffer.
    cap.max_rows_per_env = 0u;            // NO solver: has_contacts == false so
                                          // AssembleRows/SolveRows are NOT added.
                                          // The general PairDriven assembly is S5
                                          // (Phase 1B); this is a DETECTION-only gate.
                                          // (The FUSED assembly fallback would corrupt
                                          // state if run on the PairDriven buffer.)

    model.contact_family = nk::ContactFamily::PairDriven;

    // body 0: the static box at the origin.
    {
        nk::Model::BodyInit bi;
        bi.pose = Transform::Identity();
        bi.inv_mass = 0.0f;               // static.
        bi.inv_inertia = Vec3{0, 0, 0};
        model.body_init.push_back(bi);
        nk::Model::PairDrivenShape sh;
        sh.kind = kKindBox;
        sh.params[0] = 0.10f; sh.params[1] = 0.10f; sh.params[2] = 0.10f;  // half-extents
        sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
        sh.body_id = 0;                   // owning body row.
        sh.group = 0u;                    // ungrouped == collide-all (R4 default).
        model.shape_table_rows.push_back(sh);
    }
    // body 1: the movable capsule overlapping the box in +X.
    {
        nk::Model::BodyInit bi;
        bi.pose = Transform::Identity();
        bi.pose.position = Vec3{0.12f, 0.0f, 0.0f};  // overlaps the box.
        bi.inv_mass = 1.0f / 0.2f;        // 0.2 kg (movable, but no solver this gate).
        const float r = 0.05f;
        const float ii = 1.0f / (0.2f * r * r);
        bi.inv_inertia = Vec3{ii, ii, ii};
        model.body_init.push_back(bi);
        nk::Model::PairDrivenShape sh;
        sh.kind = kKindCapsule;
        sh.params[0] = 0.05f;             // radius
        sh.params[1] = 0.10f;             // half-height (axis = local Y)
        sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
        sh.body_id = 1;
        sh.group = 0u;
        model.shape_table_rows.push_back(sh);
    }
    return model;
}

// One narrowphase pass: build the world, step once, download the unified contact
// buffer. Returns the raw bytes of (count, point, normal, depth, a, b, gen,
// tangent1, tangent2) for the slot range so the caller can both inspect AND
// byte-compare two runs.
struct ContactStream {
    bool ok = false;
    uint32_t total_points = 0u;
    std::vector<uint32_t> count;       // [kSlots]
    std::vector<Vec3>     point;       // [kSlots*4]
    std::vector<Vec3>     normal;      // [kSlots*4]
    std::vector<float>    depth;       // [kSlots*4]
    std::vector<uint32_t> a;           // [kSlots*4]
    std::vector<uint32_t> b;           // [kSlots*4]
    std::vector<uint32_t> gen;         // [kSlots*4]
    std::vector<Vec3>     tangent1;    // [kSlots*4]
    std::vector<Vec3>     tangent2;    // [kSlots*4]
    std::vector<uint32_t> contact_count;  // [env]
};

ContactStream RunOnce() {
    ContactStream s;
    Backend b = GetBackend();
    if (b.backend == nullptr) return s;
    nk::World world(BuildTwoBodyPairDrivenModel(), 1u, b.dev, b.backend,
                    PairDrivenConfig());
    if (!world.Ready()) return s;
    if (!world.Step().AllOk()) return s;

    const uint32_t pts = kSlots * 4u;
    s.count.assign(kSlots, 0u);
    s.point.assign(pts, Vec3::Zero());
    s.normal.assign(pts, Vec3::Zero());
    s.depth.assign(pts, 0.0f);
    s.a.assign(pts, 0u);
    s.b.assign(pts, 0u);
    s.gen.assign(pts, 0u);
    s.tangent1.assign(pts, Vec3::Zero());
    s.tangent2.assign(pts, Vec3::Zero());
    s.contact_count.assign(1u, 0u);

    nk::Data& d = world.GetData();
    const bool dl =
        d.DownloadField(nk::FieldId::UcontactCount, s.count.data(),
                        s.count.size() * sizeof(uint32_t)) &&
        d.DownloadField(nk::FieldId::UcontactPoint, s.point.data(),
                        s.point.size() * sizeof(Vec3)) &&
        d.DownloadField(nk::FieldId::UcontactNormal, s.normal.data(),
                        s.normal.size() * sizeof(Vec3)) &&
        d.DownloadField(nk::FieldId::UcontactDepth, s.depth.data(),
                        s.depth.size() * sizeof(float)) &&
        d.DownloadField(nk::FieldId::UcontactA, s.a.data(),
                        s.a.size() * sizeof(uint32_t)) &&
        d.DownloadField(nk::FieldId::UcontactB, s.b.data(),
                        s.b.size() * sizeof(uint32_t)) &&
        d.DownloadField(nk::FieldId::UcontactGen, s.gen.data(),
                        s.gen.size() * sizeof(uint32_t)) &&
        d.DownloadField(nk::FieldId::UcontactTangent1, s.tangent1.data(),
                        s.tangent1.size() * sizeof(Vec3)) &&
        d.DownloadField(nk::FieldId::UcontactTangent2, s.tangent2.data(),
                        s.tangent2.size() * sizeof(Vec3)) &&
        d.DownloadField(nk::FieldId::ContactCount, s.contact_count.data(),
                        s.contact_count.size() * sizeof(uint32_t));
    if (!dl) return s;
    for (uint32_t i = 0; i < kSlots; ++i) s.total_points += s.count[i];
    s.ok = true;
    return s;
}

bool Finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

}  // namespace

// --- The detection gate -----------------------------------------------------
TEST(PairDrivenNarrowphase, CvxDetectsOverlapWithIdsAndTangents) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    const ContactStream s = RunOnce();
    ASSERT_TRUE(s.ok) << "the PairDriven narrowphase world failed to build/step/download";

    // (1) CONTACTS DETECTED: the overlapping capsule x box produced >0 contacts
    // through the cvx GJK/EPA fallback (the analytic ladder does NOT handle
    // capsule x box, so this is dispositive that cvx is wired).
    EXPECT_GT(s.total_points, 0u)
        << "no contacts emitted — the cvx narrowphase wiring is not firing";
    EXPECT_GT(s.contact_count[0], 0u)
        << "contact_count watermark not incremented";
    EXPECT_EQ(s.contact_count[0], s.total_points)
        << "the env contact_count must equal the sum of per-slot manifold points";

    // Walk every active slot and validate each manifold.
    uint32_t inspected = 0u;
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
        const uint32_t n = s.count[slot];
        if (n == 0u) continue;
        // (2) <= 4 points per manifold (the cvx face-clip cap).
        EXPECT_LE(n, 4u) << "slot " << slot << " manifold exceeds 4 points";
        for (uint32_t i = 0; i < n; ++i) {
            const size_t at = static_cast<size_t>(slot) * 4u + i;
            ++inspected;
            // (3) FINITE: no NaN/Inf in point/normal/depth.
            EXPECT_TRUE(Finite(s.point[at])) << "non-finite contact point";
            EXPECT_TRUE(Finite(s.normal[at])) << "non-finite contact normal";
            EXPECT_TRUE(std::isfinite(s.depth[at])) << "non-finite depth";

            // (4) IDS: the candidate (a,b) are the two collidable rows {0,1}.
            const uint32_t ca = s.a[at], cb = s.b[at];
            EXPECT_TRUE((ca == 0u && cb == 1u) || (ca == 1u && cb == 0u))
                << "ucontact_a/b must carry the (0,1) collidable ids, got ("
                << ca << "," << cb << ")";
            // gen is the constant ACTIVE marker (1) for emitted points.
            EXPECT_EQ(s.gen[at], 1u) << "active point must carry gen==1";

            // (5) NORMAL is a unit separation dir for side A; depth is a plausible
            // shallow positive penetration (the bodies overlap ~0.03 m).
            const float nlen = std::sqrt(s.normal[at].Dot(s.normal[at]));
            EXPECT_NEAR(nlen, 1.0f, 1.0e-3f) << "normal must be unit length";
            EXPECT_GT(s.depth[at], 0.0f) << "penetration must be positive";
            EXPECT_LT(s.depth[at], 0.5f) << "penetration implausibly deep";

            // (6) TANGENT BASIS (C4): (t1,t2) orthonormal + orthogonal to n.
            const Vec3 t1 = s.tangent1[at], t2 = s.tangent2[at];
            EXPECT_TRUE(Finite(t1)); EXPECT_TRUE(Finite(t2));
            EXPECT_NEAR(std::sqrt(t1.Dot(t1)), 1.0f, 1.0e-3f) << "t1 not unit";
            EXPECT_NEAR(std::sqrt(t2.Dot(t2)), 1.0f, 1.0e-3f) << "t2 not unit";
            EXPECT_NEAR(t1.Dot(s.normal[at]), 0.0f, 1.0e-3f) << "t1 not perp to n";
            EXPECT_NEAR(t2.Dot(s.normal[at]), 0.0f, 1.0e-3f) << "t2 not perp to n";
            EXPECT_NEAR(t1.Dot(t2), 0.0f, 1.0e-3f) << "t1,t2 not orthogonal";
        }
    }
    EXPECT_GT(inspected, 0u) << "no active manifold points were inspected";
}

// --- Two-run byte-identity (the new path proves its OWN D1) ------------------
TEST(PairDrivenNarrowphase, TwoRunsByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    const ContactStream a = RunOnce();
    const ContactStream b = RunOnce();
    ASSERT_TRUE(a.ok && b.ok);
    ASSERT_GT(a.total_points, 0u);

    auto SameBytes = [](const auto& x, const auto& y) -> bool {
        if (x.size() != y.size()) return false;
        if (x.empty()) return true;
        return std::memcmp(x.data(), y.data(),
                           x.size() * sizeof(typename std::decay_t<decltype(x)>::value_type)) == 0;
    };

    EXPECT_TRUE(SameBytes(a.count, b.count)) << "ucontact_count differs across runs";
    EXPECT_TRUE(SameBytes(a.point, b.point)) << "ucontact_point differs across runs";
    EXPECT_TRUE(SameBytes(a.normal, b.normal)) << "ucontact_normal differs across runs";
    EXPECT_TRUE(SameBytes(a.depth, b.depth)) << "ucontact_depth differs across runs";
    EXPECT_TRUE(SameBytes(a.a, b.a)) << "ucontact_a differs across runs";
    EXPECT_TRUE(SameBytes(a.b, b.b)) << "ucontact_b differs across runs";
    EXPECT_TRUE(SameBytes(a.gen, b.gen)) << "ucontact_gen differs across runs";
    EXPECT_TRUE(SameBytes(a.tangent1, b.tangent1)) << "ucontact_tangent1 differs across runs";
    EXPECT_TRUE(SameBytes(a.tangent2, b.tangent2)) << "ucontact_tangent2 differs across runs";
    EXPECT_EQ(a.total_points, b.total_points);
}
