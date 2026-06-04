// ---------------------------------------------------------------------------
// nuka v0.8 C2b VALIDATION GATE: BuildRigidCandidatePairs (the RIGID<->RIGID
// filtered candidate-pair stream) is correct vs an INDEPENDENT host oracle.
//
// The builder is: LBVH over per-shape world AABBs -> per-shape-pair -> filter
// (contype/conaffinity bitmask + body-level exclude + parent-child auto-exclude)
// -> emit BODY-pair CandidatePairs -> host explicit-<pair> force-include merge ->
// sorted D1 stream (each body-pair once).
//
// We drive REAL inputs through SceneIR -> CookScene so body_ids / contypes /
// conaffinities / filter_policy (excludes + joints + explicit pairs) are the
// cooker's actual output. We hand-build the per-shape collision::AABB array in
// COOKED-shape order (== source AddCollisionShape order; 1:1 for non-mesh shapes)
// so "which shapes are spatially near" is test-controlled, and the host oracle
// reuses the SAME AABBs. The oracle reads the cooked policy DIRECTLY
// (PassesContactBitmask on contact_params, IsBodyPairExcluded on filter_policy,
// iterate explicit_pairs) -- this cross-checks the builder's on-device exclude
// binary search + on-device bitmask + host explicit merge against the host
// helpers + the cooker, so agreement is a genuine independent check, not a mirror.
//
// Gates:
//   1. UNFILTERED EQUIVALENCE: all contype=conaff=1, no excludes/joints/explicit
//      -> builder body-pair SET == brute-force AABB-overlap body-pair set.
//   2. FILTERED CORRECTNESS: bitmask-drop + exclude + parent-child joint
//      (auto-exclude) + explicit force-include -> builder SET == oracle, with
//      per-case non-vacuity membership asserts.
//   3. D1 + scale: >=32 overlapping bodies -> two runs byte-exact; Count()>0.
//   4. SANITY: shape_count==0 and empty-overlap -> empty stream, no crash.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/candidate_pair.hpp"
#include "collision/rigid_candidate_pairs.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"
#include "scene/contact_filter.hpp"
#include "scene/cooked_blob.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <utility>
#include <vector>

using namespace nuka;

namespace {

using CanonPair = std::pair<uint32_t, uint32_t>;

CanonPair Canon(uint32_t a, uint32_t b) {
    return (a < b) ? CanonPair{a, b} : CanonPair{b, a};
}

// Reduce the builder's emitted CandidatePairs to a canonical (min,max) body-pair
// set keyed on the side `handle`s (which the builder stamps with the body id).
std::set<CanonPair> ToBodyPairSet(const std::vector<collision::CandidatePair>& pairs) {
    std::set<CanonPair> out;
    for (const auto& p : pairs) {
        // Every emitted side is a RigidBody whose handle is the body id.
        EXPECT_EQ(p.a.type, constraint::CollidableType::RigidBody);
        EXPECT_EQ(p.b.type, constraint::CollidableType::RigidBody);
        out.insert(Canon(p.a.handle, p.b.handle));
    }
    return out;
}

// One cooked shape's AABB in cooked-shape (== source) order. We build an
// axis-aligned box of half-extent `h` centered at `c` directly (the {min,max}
// route the API doc sanctions), so overlaps are exactly under our control.
collision::AABB BoxAt(math::Vec3 c, float h) {
    collision::AABB box;
    box.min = c - math::Vec3{h, h, h};
    box.max = c + math::Vec3{h, h, h};
    return box;
}

// Upload an AABB array to device and run the builder against a cooked blob.
std::vector<collision::CandidatePair> RunBuilder(
    const std::vector<collision::AABB>& aabbs, const scene::CookedBlob& blob) {
    const uint32_t shape_count = static_cast<uint32_t>(aabbs.size());
    const collision::AABB* dev = nullptr;
    phi::Buffer d_aabbs;
    if (shape_count > 0u) {
        d_aabbs = phi::UploadVector(aabbs);
        dev = static_cast<const collision::AABB*>(d_aabbs.Data());
    }
    auto stream = collision::BuildRigidCandidatePairs(
        dev, shape_count,
        blob.shapes.body_ids.data(),
        blob.contact_params.contypes.data(),
        blob.contact_params.conaffinities.data(),
        blob.filter_policy);
    EXPECT_EQ(stream.Count(), stream.DownloadPairs().size());
    return stream.DownloadPairs();
}

// INDEPENDENT host oracle. Reads the cooked policy/tables directly:
//   keep i<j iff aabb[i] overlaps aabb[j] AND body!=body AND PassesContactBitmask
//             AND NOT IsBodyPairExcluded(policy, body_i, body_j)
//   UNION (applied last, unconditional -- mirrors the .cu host merge which does
//   NOT consult overlap or exclude): every explicit pair's body-pair whose source
//   shapes are in range and whose bodies differ.
std::set<CanonPair> Oracle(const std::vector<collision::AABB>& aabbs,
                           const scene::CookedBlob& blob) {
    std::set<CanonPair> out;
    const auto& body_ids = blob.shapes.body_ids;
    const auto& contypes = blob.contact_params.contypes;
    const auto& conaff = blob.contact_params.conaffinities;
    const uint32_t n = static_cast<uint32_t>(aabbs.size());

    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = i + 1u; j < n; ++j) {
            if (!aabbs[i].Overlaps(aabbs[j])) continue;
            const uint32_t bi = body_ids[i];
            const uint32_t bj = body_ids[j];
            if (bi == bj) continue;
            if (!scene::PassesContactBitmask(contypes[i], conaff[i],
                                             contypes[j], conaff[j])) {
                continue;
            }
            if (scene::IsBodyPairExcluded(blob.filter_policy, bi, bj)) continue;
            out.insert(Canon(bi, bj));
        }
    }
    // Unconditional explicit force-include (overrides bitmask AND exclude).
    for (const auto& ep : blob.filter_policy.explicit_pairs) {
        if (ep.geom1 >= n || ep.geom2 >= n) continue;
        const uint32_t ba = body_ids[ep.geom1];
        const uint32_t bb = body_ids[ep.geom2];
        if (ba == bb) continue;
        out.insert(Canon(ba, bb));
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Gate 1: UNFILTERED EQUIVALENCE -- no-regression proof. All shapes default
// contype=conaff=1, no excludes, no joints, no explicit pairs. The builder set
// must equal the brute-force AABB-overlap body-pair set (1 shape per body, so
// body_i != body_j is automatic).
// ---------------------------------------------------------------------------
TEST(LbvhFilteredPairs, UnfilteredEqualsBruteForceOverlap) {
    scene::SceneIR scene;
    // 6 unit-ish boxes; a deliberate mix of clear overlaps and clear gaps.
    // Centers (half-extent 0.5): boxes 0,1,2 form an overlapping chain along x
    // (gap 0.5 < 1.0 so consecutive overlap, 0&2 do NOT: |1.5|>1.0). 3 overlaps
    // 1. 4 and 5 are far away and overlap only each other.
    const math::Vec3 centers[] = {
        {0.0f, 0.0f, 0.0f},   // 0
        {0.7f, 0.0f, 0.0f},   // 1  overlaps 0
        {1.4f, 0.0f, 0.0f},   // 2  overlaps 1, not 0
        {0.7f, 0.6f, 0.0f},   // 3  overlaps 1 (dx0.0,dy0.6<1.0), overlaps 2? dx0.7 dy0.6 -> yes
        {20.0f, 0.0f, 0.0f},  // 4  isolated cluster with 5
        {20.6f, 0.0f, 0.0f},  // 5  overlaps 4
    };
    // Anchor body with NO shape so cooked shape-id != body-id everywhere below
    // (this DE-ALIASES the two so the builder's handle==BODY-id contract -- emit
    // shape_body_ids[shape], NOT the shape index -- is actually exercised; a
    // handle=shape-id bug would otherwise pass since the two are numerically
    // equal when shapes are added 1:1 from body 0).
    scene.AddRigidBody("anchor");
    std::vector<collision::AABB> aabbs;
    for (const auto& c : centers) {
        const auto body = scene.AddRigidBody("b");
        scene::CollisionShapeRecord shape;
        shape.body_id = body;
        shape.type = scene::ShapeType::Box;
        shape.half_extents = {0.5f, 0.5f, 0.5f};
        scene.AddCollisionShape(shape);
        aabbs.push_back(BoxAt(c, 0.5f));
    }
    const auto blob = scene::CookScene(scene);
    ASSERT_EQ(blob.shape_count, aabbs.size());

    const auto got = ToBodyPairSet(RunBuilder(aabbs, blob));
    const auto oracle = Oracle(aabbs, blob);

    EXPECT_EQ(got.size(), oracle.size());
    EXPECT_EQ(got, oracle);
    EXPECT_GT(oracle.size(), 0u) << "oracle vacuous -- scene has no overlaps";
    // Documented expectation: overlaps {0-1, 0-3, 1-2, 1-3, 2-3, 4-5} = 6.
    EXPECT_EQ(oracle.size(), 6u);
}

// ---------------------------------------------------------------------------
// Gate 2: FILTERED CORRECTNESS. A scene exercising every filter level, each
// case constructed to OVERLAP (so it is in the baseline) and individually
// asserted for non-vacuity:
//   (a) bitmask-mismatch pair  -> overlaps, DROPPED.
//   (b) AddExcludePair pair    -> overlaps, DROPPED.
//   (c) parent-child joint pair-> overlaps, DROPPED (cooker auto-excludes).
//   (d) explicit <pair> on a bitmask-rejected pair -> FORCE-INCLUDED.
//   (e) (optional) explicit <pair> on NON-overlapping shapes -> still INCLUDED
//       (proves the merge is unconditional / independent of the LBVH).
// Plus a plain overlapping pair that should survive (control).
// ---------------------------------------------------------------------------
TEST(LbvhFilteredPairs, FilteredMatchesOracle) {
    scene::SceneIR scene;
    // Shapeless anchor: de-aliases cooked shape-id from body-id (see gate 1) so
    // the handle==BODY-id mapping is independently exercised. Shapes become 0..11
    // while bodies become 1..12; the explicit-pair geom ids below stay in
    // SHAPE space (8,9,10,11), and every oracle/assert reads body_ids[...].
    scene.AddRigidBody("anchor");
    std::vector<collision::AABB> aabbs;

    auto add_box = [&](math::Vec3 c, uint32_t contype, uint32_t conaff) -> scene::BodyId {
        const auto body = scene.AddRigidBody("b");
        scene::CollisionShapeRecord shape;
        shape.body_id = body;
        shape.type = scene::ShapeType::Box;
        shape.half_extents = {0.5f, 0.5f, 0.5f};
        shape.contype = contype;
        shape.conaffinity = conaff;
        scene.AddCollisionShape(shape);
        aabbs.push_back(BoxAt(c, 0.5f));
        return body;
    };

    // Control: 0 & 1 overlap, both default mask -> KEEP.
    const auto b0 = add_box({0.0f, 0.0f, 0.0f}, 1u, 1u);   // shape 0
    const auto b1 = add_box({0.6f, 0.0f, 0.0f}, 1u, 1u);   // shape 1

    // (a) bitmask mismatch: 2 & 3 overlap but disjoint contype/conaff bits.
    //     contype=1,conaff=1 vs contype=2,conaff=2: (1&2)|(2&1) == 0 -> DROP.
    const auto b2 = add_box({5.0f, 0.0f, 0.0f}, 1u, 1u);   // shape 2
    const auto b3 = add_box({5.6f, 0.0f, 0.0f}, 2u, 2u);   // shape 3

    // (b) exclude: 4 & 5 overlap, default mask, but AddExcludePair -> DROP.
    const auto b4 = add_box({10.0f, 0.0f, 0.0f}, 1u, 1u);  // shape 4
    const auto b5 = add_box({10.6f, 0.0f, 0.0f}, 1u, 1u);  // shape 5

    // (c) joint auto-exclude: 6 & 7 overlap, default mask, parent-child joint.
    const auto b6 = add_box({15.0f, 0.0f, 0.0f}, 1u, 1u);  // shape 6
    const auto b7 = add_box({15.6f, 0.0f, 0.0f}, 1u, 1u);  // shape 7

    // (d) explicit force-include on a bitmask-rejected OVERLAPPING pair: 8 & 9.
    const auto b8 = add_box({20.0f, 0.0f, 0.0f}, 1u, 1u);  // shape 8
    const auto b9 = add_box({20.6f, 0.0f, 0.0f}, 4u, 4u);  // shape 9 (disjoint bits vs 8)

    // (e) explicit force-include on NON-overlapping shapes: 10 & 11 (far apart),
    //     default mask but no spatial overlap -> only the explicit merge can add.
    const auto b10 = add_box({30.0f, 0.0f, 0.0f}, 1u, 1u); // shape 10
    const auto b11 = add_box({40.0f, 0.0f, 0.0f}, 1u, 1u); // shape 11

    scene.AddExcludePair(b4, b5);                 // (b)
    scene.AddJoint("j", b6, b7);                  // (c) parent=b6, child=b7
    scene::ContactPairOverride ov_d;
    ov_d.geom1 = 8u; ov_d.geom2 = 9u;
    scene.AddContactPair(ov_d);                   // (d)
    scene::ContactPairOverride ov_e;
    ov_e.geom1 = 10u; ov_e.geom2 = 11u;
    scene.AddContactPair(ov_e);                   // (e)

    const auto blob = scene::CookScene(scene);
    ASSERT_EQ(blob.shape_count, aabbs.size());

    // --- Non-vacuity preconditions: each filtered pair MUST overlap (so it is
    //     in the baseline and the filter is what removes/adds it). ----------
    ASSERT_TRUE(aabbs[0].Overlaps(aabbs[1])) << "control not overlapping";
    ASSERT_TRUE(aabbs[2].Overlaps(aabbs[3])) << "(a) bitmask case not overlapping";
    ASSERT_TRUE(aabbs[4].Overlaps(aabbs[5])) << "(b) exclude case not overlapping";
    ASSERT_TRUE(aabbs[6].Overlaps(aabbs[7])) << "(c) joint case not overlapping";
    ASSERT_TRUE(aabbs[8].Overlaps(aabbs[9])) << "(d) explicit case not overlapping";
    ASSERT_FALSE(aabbs[10].Overlaps(aabbs[11])) << "(e) must be NON-overlapping";
    // Confirm the cooker actually unioned the joint into the exclude list (so (c)
    // is a real auto-exclude, not a silent no-op).
    ASSERT_TRUE(scene::IsBodyPairExcluded(blob.filter_policy, b6, b7))
        << "cooker did not auto-exclude the parent-child joint pair";

    const auto got = ToBodyPairSet(RunBuilder(aabbs, blob));
    const auto oracle = Oracle(aabbs, blob);

    // --- Full-set equality (Count == set size proven inside RunBuilder). -----
    EXPECT_EQ(got.size(), oracle.size());
    EXPECT_EQ(got, oracle);
    // Documented expectation: survivors {b0-b1 (control), b8-b9 (explicit),
    // b10-b11 (explicit non-overlap)} = 3; (a)(b)(c) all dropped.
    EXPECT_EQ(oracle.size(), 3u);

    // --- Per-case membership (independent of the set equality). --------------
    EXPECT_TRUE(got.count(Canon(b0, b1)))  << "control survivor missing";
    EXPECT_FALSE(got.count(Canon(b2, b3))) << "(a) bitmask-mismatch pair leaked";
    EXPECT_FALSE(got.count(Canon(b4, b5))) << "(b) excluded pair leaked";
    EXPECT_FALSE(got.count(Canon(b6, b7))) << "(c) joint auto-excluded pair leaked";
    EXPECT_TRUE(got.count(Canon(b8, b9)))  << "(d) explicit force-include missing";
    EXPECT_TRUE(got.count(Canon(b10, b11)))
        << "(e) explicit force-include on non-overlapping pair missing";
}

// ---------------------------------------------------------------------------
// Gate 3: D1 + scale. >=32 bodies in an overlapping cluster -> two builds are
// byte-exact (memcmp), and the result is non-empty.
// ---------------------------------------------------------------------------
TEST(LbvhFilteredPairs, D1TwoRunByteExactAndScale) {
    constexpr int kBodies = 40;  // >= 32
    scene::SceneIR scene;
    scene.AddRigidBody("anchor");  // de-alias shape-id from body-id (see gate 1)
    std::vector<collision::AABB> aabbs;
    // A dense overlapping cluster: step 0.3 along x with half-extent 0.5 so each
    // box overlaps several neighbors (window ~3 each side) -> a rich pair set.
    for (int i = 0; i < kBodies; ++i) {
        const auto body = scene.AddRigidBody("b");
        scene::CollisionShapeRecord shape;
        shape.body_id = body;
        shape.type = scene::ShapeType::Box;
        shape.half_extents = {0.5f, 0.5f, 0.5f};
        scene.AddCollisionShape(shape);
        aabbs.push_back(BoxAt({0.3f * static_cast<float>(i), 0.0f, 0.0f}, 0.5f));
    }
    const auto blob = scene::CookScene(scene);
    ASSERT_EQ(blob.shape_count, aabbs.size());

    const auto run1 = RunBuilder(aabbs, blob);
    const auto run2 = RunBuilder(aabbs, blob);

    ASSERT_GT(run1.size(), 0u) << "scale scene produced no pairs";
    ASSERT_EQ(run1.size(), run2.size());
    EXPECT_EQ(std::memcmp(run1.data(), run2.data(),
                          run1.size() * sizeof(collision::CandidatePair)), 0)
        << "BuildRigidCandidatePairs is not two-run byte-exact (D1)";

    // Cross-check against the oracle so the scale set is also CORRECT, not just
    // deterministic.
    const auto got = ToBodyPairSet(run1);
    const auto oracle = Oracle(aabbs, blob);
    EXPECT_EQ(got.size(), oracle.size());
    EXPECT_EQ(got, oracle);
    // Documented expectation: step 0.3, half-extent 0.5 -> overlap iff |i-j|<=3
    // (0.3*3=0.9 < 1.0, 0.3*4=1.2 >= 1.0). Over 40 boxes: 39+38+37 = 114.
    EXPECT_EQ(oracle.size(), 114u);
}

// ---------------------------------------------------------------------------
// Gate 4: SANITY. shape_count==0 and an all-disjoint scene both yield an empty
// stream with no crash.
// ---------------------------------------------------------------------------
TEST(LbvhFilteredPairs, EmptyAndNoOverlapAreEmpty) {
    // (a) zero shapes.
    {
        scene::SceneIR scene;
        const auto blob = scene::CookScene(scene);
        ASSERT_EQ(blob.shape_count, 0u);
        std::vector<collision::AABB> empty;
        auto stream = collision::BuildRigidCandidatePairs(
            nullptr, 0u,
            blob.shapes.body_ids.data(),
            blob.contact_params.contypes.data(),
            blob.contact_params.conaffinities.data(),
            blob.filter_policy);
        EXPECT_EQ(stream.Count(), 0u);
        EXPECT_TRUE(stream.DownloadPairs().empty());
    }
    // (b) shapes that never overlap -> empty stream.
    {
        scene::SceneIR scene;
        std::vector<collision::AABB> aabbs;
        for (int i = 0; i < 5; ++i) {
            const auto body = scene.AddRigidBody("b");
            scene::CollisionShapeRecord shape;
            shape.body_id = body;
            shape.type = scene::ShapeType::Box;
            shape.half_extents = {0.5f, 0.5f, 0.5f};
            scene.AddCollisionShape(shape);
            aabbs.push_back(BoxAt({100.0f * static_cast<float>(i), 0.0f, 0.0f}, 0.5f));
        }
        const auto blob = scene::CookScene(scene);
        const auto got = RunBuilder(aabbs, blob);
        EXPECT_TRUE(got.empty());
        EXPECT_TRUE(Oracle(aabbs, blob).empty());
    }
}
