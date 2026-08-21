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

#include <algorithm>
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
    cap.max_rows_per_env = kSlots * nk::kPairDrivenRowsPerSlot;

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
    uint32_t failure_stage = 0u;
    uint32_t total_points = 0u;
    std::vector<uint32_t> count;       // [kSlots]
    std::vector<Vec3>     point;       // [kSlots*4]
    std::vector<Vec3>     normal;      // [kSlots*4]
    std::vector<float>    depth;       // [kSlots*4]
    std::vector<uint32_t> a;           // [kSlots*4]
    std::vector<uint32_t> b;           // [kSlots*4]
    std::vector<uint32_t> gen;         // [kSlots*4]
    std::vector<uint64_t> id_pair;     // [kSlots*4]
    std::vector<uint64_t> id_feature;  // [kSlots*4]
    std::vector<uint64_t> cache_pair;  // persistent cache [kSlots*4]
    std::vector<uint64_t> cache_feature;
    std::vector<float> cache_lambda;   // [kSlots*4*3]
    std::vector<float> lambda;         // [rows]
    std::vector<Vec3>     tangent1;    // [kSlots*4]
    std::vector<Vec3>     tangent2;    // [kSlots*4]
    std::vector<uint32_t> contact_count;  // [env]
};

ContactStream RunOnce() {
    ContactStream s;
    Backend b = GetBackend();
    if (b.backend == nullptr) { s.failure_stage = 1u; return s; }
    nk::World world(BuildTwoBodyPairDrivenModel(), 1u, b.dev, b.backend,
                    PairDrivenConfig());
    if (!world.Ready()) { s.failure_stage = 2u; return s; }
    if (!world.Step().AllOk()) { s.failure_stage = 3u; return s; }

    const uint32_t pts = kSlots * 4u;
    s.count.assign(kSlots, 0u);
    s.point.assign(pts, Vec3::Zero());
    s.normal.assign(pts, Vec3::Zero());
    s.depth.assign(pts, 0.0f);
    s.a.assign(pts, 0u);
    s.b.assign(pts, 0u);
    s.gen.assign(pts, 0u);
    s.id_pair.assign(pts, 0u);
    s.id_feature.assign(pts, 0u);
    s.cache_pair.assign(pts, 0u);
    s.cache_feature.assign(pts, 0u);
    s.cache_lambda.assign(pts * 3u, 0.0f);
    s.lambda.assign(kSlots * nk::kPairDrivenRowsPerSlot, 0.0f);
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
        d.DownloadField(nk::FieldId::UcontactIdPair, s.id_pair.data(),
                        s.id_pair.size() * sizeof(uint64_t)) &&
        d.DownloadField(nk::FieldId::UcontactIdFeature, s.id_feature.data(),
                        s.id_feature.size() * sizeof(uint64_t)) &&
        d.DownloadField(nk::FieldId::ContactCachePair, s.cache_pair.data(),
                        s.cache_pair.size() * sizeof(uint64_t)) &&
        d.DownloadField(nk::FieldId::ContactCacheFeature, s.cache_feature.data(),
                        s.cache_feature.size() * sizeof(uint64_t)) &&
        d.DownloadField(nk::FieldId::ContactCacheLambda, s.cache_lambda.data(),
                        s.cache_lambda.size() * sizeof(float)) &&
        d.DownloadField(nk::FieldId::Lambda, s.lambda.data(),
                        s.lambda.size() * sizeof(float)) &&
        d.DownloadField(nk::FieldId::UcontactTangent1, s.tangent1.data(),
                        s.tangent1.size() * sizeof(Vec3)) &&
        d.DownloadField(nk::FieldId::UcontactTangent2, s.tangent2.data(),
                        s.tangent2.size() * sizeof(Vec3)) &&
        d.DownloadField(nk::FieldId::ContactCount, s.contact_count.data(),
                        s.contact_count.size() * sizeof(uint32_t));
    if (!dl) { s.failure_stage = 4u; return s; }
    for (uint32_t i = 0; i < kSlots; ++i) s.total_points += s.count[i];
    s.ok = true;
    return s;
}

bool Finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

uint32_t FirstCachedPoint(const std::vector<uint64_t>& pair,
                          const std::vector<uint64_t>& feature) {
    for (uint32_t i = 0u; i < pair.size(); ++i) {
        if (pair[i] != 0u && feature[i] != 0u) return i;
    }
    return ~0u;
}

}  // namespace

// --- The detection gate -----------------------------------------------------
TEST(PairDrivenNarrowphase, CvxDetectsOverlapWithIdsAndTangents) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    const ContactStream s = RunOnce();
    ASSERT_TRUE(s.ok) << "the PairDriven narrowphase world failed at stage "
                      << s.failure_stage
                      << " (1=backend, 2=ready, 3=step, 4=download)";

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
            EXPECT_NE(s.id_pair[at], 0u) << "active point must carry a pair id";
            EXPECT_NE(s.id_feature[at], 0u) << "active point must carry a feature id";
            EXPECT_EQ(s.cache_pair[at], s.id_pair[at]);
            EXPECT_EQ(s.cache_feature[at], s.id_feature[at]);

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
        for (uint32_t i = n; i < 4u; ++i) {
            const size_t at = static_cast<size_t>(slot) * 4u + i;
            EXPECT_EQ(s.id_pair[at], 0u) << "inactive point must clear pair id";
            EXPECT_EQ(s.id_feature[at], 0u) << "inactive point must clear feature id";
        }
    }
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
        for (uint32_t i = s.count[slot]; i < 4u; ++i) {
            const size_t at = static_cast<size_t>(slot) * 4u + i;
            EXPECT_EQ(s.id_pair[at], 0u) << "inactive point must clear pair id";
            EXPECT_EQ(s.id_feature[at], 0u) << "inactive point must clear feature id";
        }
    }
    EXPECT_GT(inspected, 0u) << "no active manifold points were inspected";
}

// --- Two-run byte-identity (the new path proves its OWN D1) ------------------
TEST(PairDrivenNarrowphase, TwoRunsByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    const ContactStream a = RunOnce();
    const ContactStream b = RunOnce();
    ASSERT_TRUE(a.ok && b.ok)
        << "PairDriven runs failed at stages " << a.failure_stage << " and "
        << b.failure_stage;
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
    EXPECT_TRUE(SameBytes(a.id_pair, b.id_pair)) << "ucontact_id_pair differs across runs";
    EXPECT_TRUE(SameBytes(a.id_feature, b.id_feature))
        << "ucontact_id_feature differs across runs";
    EXPECT_TRUE(SameBytes(a.cache_pair, b.cache_pair))
        << "contact_cache_pair differs across runs";
    EXPECT_TRUE(SameBytes(a.cache_feature, b.cache_feature))
        << "contact_cache_feature differs across runs";
    EXPECT_TRUE(SameBytes(a.cache_lambda, b.cache_lambda))
        << "contact_cache_lambda differs across runs";
    EXPECT_TRUE(SameBytes(a.tangent1, b.tangent1)) << "ucontact_tangent1 differs across runs";
    EXPECT_TRUE(SameBytes(a.tangent2, b.tangent2)) << "ucontact_tangent2 differs across runs";
    EXPECT_EQ(a.total_points, b.total_points);
}

TEST(PairDrivenNarrowphase, ContactBlock3LambdasStayInsideCone) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const ContactStream s = RunOnce();
    ASSERT_TRUE(s.ok);
    uint32_t checked = 0u;
    for (uint32_t slot = 0u; slot < kSlots; ++slot) {
        for (uint32_t point = 0u; point < s.count[slot]; ++point) {
            const uint32_t base = slot * nk::kPairDrivenRowsPerSlot;
            const float n = std::max(s.lambda[base + point], 0.0f);
            const float t1 = s.lambda[base + nk::kPairDrivenPtsPerSlot + point];
            const float t2 = s.lambda[base + 2u * nk::kPairDrivenPtsPerSlot + point];
            const float radial = std::sqrt(t1 * t1 + t2 * t2);
            EXPECT_GE(n, 0.0f);
            EXPECT_LE(radial, 0.8f * n + 1.0e-5f)
                << "contact point escaped the isotropic friction cone";
            ++checked;
        }
    }
    EXPECT_GT(checked, 0u);
}

TEST(PairDrivenNarrowphase, WarmStartCacheSeedsRowsAndVelocity) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Pipeline::SolverConfig cfg = PairDrivenConfig();
    cfg.vel_iters = 0u;
    nk::World world(BuildTwoBodyPairDrivenModel(), 1u, b.dev, b.backend, cfg);
    ASSERT_TRUE(world.Ready());
    ASSERT_TRUE(world.Step().AllOk());

    constexpr uint32_t kPoints = kSlots * 4u;
    std::vector<uint64_t> pair(kPoints, 0u);
    std::vector<uint64_t> feature(kPoints, 0u);
    std::vector<float> cache_lambda(kPoints * 3u, 0.0f);
    nk::Data& data = world.GetData();
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ContactCachePair, pair.data(),
                                   pair.size() * sizeof(uint64_t)));
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ContactCacheFeature, feature.data(),
                                   feature.size() * sizeof(uint64_t)));
    uint32_t cached_point = FirstCachedPoint(pair, feature);
    ASSERT_NE(cached_point, ~0u);
    constexpr float kSeedImpulse = 0.125f;
    cache_lambda[cached_point * 3u] = kSeedImpulse;
    ASSERT_TRUE(data.UploadField(nk::FieldId::ContactCacheLambda,
                                 cache_lambda.data(),
                                 cache_lambda.size() * sizeof(float)));

    ASSERT_TRUE(world.Step().AllOk());
    std::vector<uint64_t> current_pair(kPoints, 0u);
    std::vector<uint64_t> current_feature(kPoints, 0u);
    std::vector<float> rows(kSlots * nk::kPairDrivenRowsPerSlot, 0.0f);
    std::vector<Vec3> velocity(2u, Vec3::Zero());
    ASSERT_TRUE(data.DownloadField(nk::FieldId::UcontactIdPair,
                                   current_pair.data(),
                                   current_pair.size() * sizeof(uint64_t)));
    ASSERT_TRUE(data.DownloadField(nk::FieldId::UcontactIdFeature,
                                   current_feature.data(),
                                   current_feature.size() * sizeof(uint64_t)));
    ASSERT_TRUE(data.DownloadField(nk::FieldId::Lambda, rows.data(),
                                   rows.size() * sizeof(float)));
    ASSERT_TRUE(data.DownloadField(nk::FieldId::BodyLinearVelocity,
                                   velocity.data(),
                                   velocity.size() * sizeof(Vec3)));

    uint32_t current_point = ~0u;
    for (uint32_t i = 0u; i < kPoints; ++i) {
        if (current_pair[i] == pair[cached_point] &&
            current_feature[i] == feature[cached_point]) {
            current_point = i;
            break;
        }
    }
    ASSERT_NE(current_point, ~0u);
    const uint32_t slot = current_point / 4u;
    const uint32_t point = current_point & 3u;
    const uint32_t normal_row = slot * nk::kPairDrivenRowsPerSlot + point;
    EXPECT_FLOAT_EQ(rows[normal_row], kSeedImpulse);
    EXPECT_GT(velocity[1].Dot(velocity[1]), 1.0e-6f)
        << "cached impulse was not applied to the movable body";
}

TEST(PairDrivenNarrowphase, WarmStartInvalidatesAndResetsDeterministically) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Pipeline::SolverConfig cfg = PairDrivenConfig();
    cfg.vel_iters = 0u;
    nk::World world(BuildTwoBodyPairDrivenModel(), 1u, b.dev, b.backend, cfg);
    ASSERT_TRUE(world.Ready());
    ASSERT_TRUE(world.Step().AllOk());

    constexpr uint32_t kPoints = kSlots * 4u;
    constexpr uint32_t kRows = kSlots * nk::kPairDrivenRowsPerSlot;
    nk::Data& data = world.GetData();
    std::vector<uint64_t> pair(kPoints, 0u), feature(kPoints, 0u);
    std::vector<Vec3> normal(kPoints, Vec3::Zero());
    std::vector<uint32_t> material(kPoints, 0u), age(kPoints, 0u);
    std::vector<float> cache_lambda(kPoints * 3u, 0.0f), rows(kRows, 0.0f);
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ContactCachePair, pair.data(),
                                   pair.size() * sizeof(uint64_t)));
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ContactCacheFeature, feature.data(),
                                   feature.size() * sizeof(uint64_t)));
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ContactCacheNormal, normal.data(),
                                   normal.size() * sizeof(Vec3)));
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ContactCacheMaterial,
                                   material.data(),
                                   material.size() * sizeof(uint32_t)));
    const uint32_t point_index = FirstCachedPoint(pair, feature);
    ASSERT_NE(point_index, ~0u);
    const uint32_t normal_row = (point_index / 4u) *
                                    nk::kPairDrivenRowsPerSlot +
                                (point_index & 3u);

    cache_lambda[point_index * 3u] = 0.125f;
    normal[point_index] = normal[point_index] * -1.0f;
    ASSERT_TRUE(data.UploadField(nk::FieldId::ContactCacheLambda,
                                 cache_lambda.data(),
                                 cache_lambda.size() * sizeof(float)));
    ASSERT_TRUE(data.UploadField(nk::FieldId::ContactCacheNormal, normal.data(),
                                 normal.size() * sizeof(Vec3)));
    ASSERT_TRUE(world.Step().AllOk());
    ASSERT_TRUE(data.DownloadField(nk::FieldId::Lambda, rows.data(),
                                   rows.size() * sizeof(float)));
    EXPECT_FLOAT_EQ(rows[normal_row], 0.0f) << "flipped normal reused cache";

    cache_lambda.assign(kPoints * 3u, 0.0f);
    cache_lambda[point_index * 3u] = 0.125f;
    material[point_index] ^= 0x1u;
    ASSERT_TRUE(data.UploadField(nk::FieldId::ContactCacheLambda,
                                 cache_lambda.data(),
                                 cache_lambda.size() * sizeof(float)));
    ASSERT_TRUE(data.UploadField(nk::FieldId::ContactCacheMaterial,
                                 material.data(),
                                 material.size() * sizeof(uint32_t)));
    ASSERT_TRUE(world.Step().AllOk());
    ASSERT_TRUE(data.DownloadField(nk::FieldId::Lambda, rows.data(),
                                   rows.size() * sizeof(float)));
    EXPECT_FLOAT_EQ(rows[normal_row], 0.0f) << "changed material reused cache";

    std::vector<Transform> pose(2u, Transform::Identity());
    ASSERT_TRUE(data.DownloadField(nk::FieldId::BodyPose, pose.data(),
                                   pose.size() * sizeof(Transform)));
    pose[1].position.x = 2.0f;
    ASSERT_TRUE(data.UploadField(nk::FieldId::BodyPose, pose.data(),
                                 pose.size() * sizeof(Transform)));
    ASSERT_TRUE(world.Step().AllOk());
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ContactCachePair, pair.data(),
                                   pair.size() * sizeof(uint64_t)));
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ContactCacheAge, age.data(),
                                   age.size() * sizeof(uint32_t)));
    EXPECT_NE(pair[point_index], 0u);
    EXPECT_EQ(age[point_index], 1u);
    ASSERT_TRUE(world.Step().AllOk());
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ContactCachePair, pair.data(),
                                   pair.size() * sizeof(uint64_t)));
    EXPECT_EQ(pair[point_index], 0u) << "expired contact cache was not cleared";

    ASSERT_EQ(world.Reset({0u}), nphi::Status::Ok);
    ASSERT_TRUE(data.DownloadField(nk::FieldId::ContactCachePair, pair.data(),
                                   pair.size() * sizeof(uint64_t)));
    for (uint64_t id : pair) EXPECT_EQ(id, 0u);
}
